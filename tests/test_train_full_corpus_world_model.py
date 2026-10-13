"""Finite coverage, microbatch weighting and durable resume using real compact replay."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tools.train_full_corpus_world_model as runner
from test_offline_data import setup
from test_pretrain_cache import make_cache
from test_train_component_world_model import small_model
from urban_dreamer.learning.offline_data import _identity
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.models.experimental_dynamics import apply_variant


def model(variant):
    result = apply_variant(small_model(5), variant, seed=5)
    result.flow_projector.eval()
    result.active_vehicle_fraction_readout.eval()
    return result


def args(output, **changes):
    value = dict(output_dir=output, data_seed=19, epochs=2, batch_size=2,
        microbatch_size=1, device='cpu', precision='fp32', checkpoint_every=2,
        validation_every=3, resume=False, stop_after_updates=None)
    return SimpleNamespace(**(value | changes))


def restored_source(source):
    return CompactPretrainDataset(source.config, source.prepared, source.topology, 1,
        cache_manifest_path=source.cache_manifest_path)


def test_manifest_counts_uses_episode_tails_and_rejects_overlapping_groups():
    manifest = dict(status='complete', entries=[
        dict(split='train', windows=65, demand_group='a'),
        dict(split='train', windows=65, demand_group='b'),
        dict(split='validation', windows=1, demand_group='c')])
    manifest['identity'] = _identity(manifest)
    assert runner.manifest_counts(manifest)['train']['updates_per_epoch'] == 4
    manifest['entries'][-1]['demand_group'] = 'a'
    manifest['identity'] = _identity({k: v for k, v in manifest.items() if k != 'identity'})
    with pytest.raises(ValueError, match='overlap'):
        runner.manifest_counts(manifest)


def test_scalar_buffer_tensor_identity_is_supported_and_sensitive():
    state = dict(counter=torch.tensor(5), values=torch.ones(2))
    before = runner.tensor_identity(state)
    assert before == runner.tensor_identity(deepcopy(state))
    state['counter'].add_(1)
    assert runner.tensor_identity(state) != before


def test_microbatch_slice_retains_full_events_and_hybrid_labels(setup, tmp_path, monkeypatch):
    source, _ = make_cache(setup, tmp_path, monkeypatch)
    source.sequence_length = 1
    batch = source.batch_from_starts(0, [0, 1, 2], fm_only=True)
    part = runner.slice_batch(batch, 1, 3)
    selected = batch.action_events.batch_index >= 1
    assert part.action_events.vehicle_ids == tuple(v for v, keep in zip(
        batch.action_events.vehicle_ids, selected.tolist()) if keep)
    torch.testing.assert_close(part.action_events.batch_index, batch.action_events.batch_index[selected]-1)
    assert part.action_events.batch_size == 2
    torch.testing.assert_close(part.transition_target_flow, batch.transition_target_flow[1:])
    torch.testing.assert_close(part.action, batch.action[1:])


@pytest.mark.parametrize('variant', ['regression', 'gaussian'])
def test_microbatch_weighting_matches_logical_batch_without_codec_updates(setup, tmp_path, monkeypatch, variant):
    source, _ = make_cache(setup, tmp_path, monkeypatch)
    source.sequence_length = 1
    batch = source.batch_from_starts(0, [0, 1, 2], fm_only=True)
    left, right = model(variant), model(variant)
    frozen = runner.tensor_identity(left.flow_projector.state_dict())
    metrics = []
    for current, chunk in [(left, 3), (right, 2)]:
        optimizer = torch.optim.AdamW(current.dynamics.parameters(), lr=1e-4, weight_decay=.01)
        metrics.append(runner.update(current, optimizer, batch, torch.ones(3),
            device='cpu', precision='fp32', microbatch_size=chunk))
    assert metrics[0]['loss'] == pytest.approx(metrics[1]['loss'], rel=1e-6)
    for key, value in left.dynamics.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, right.dynamics.state_dict()[key], atol=2e-5, rtol=2e-4)
    assert runner.tensor_identity(left.flow_projector.state_dict()) == frozen
    assert runner.tensor_identity(right.flow_projector.state_dict()) == frozen


def test_encode_stays_fp32_even_under_outer_autocast(setup, tmp_path, monkeypatch):
    source, _ = make_cache(setup, tmp_path, monkeypatch)
    source.sequence_length = 1
    batch = source.batch_from_starts(0, [0], fm_only=True)
    current = model('regression')
    original = current.encode
    flags = []
    def observe(value):
        flags.append((torch.is_autocast_enabled('cpu'), value.dtype))
        return original(value)
    monkeypatch.setattr(current, 'encode', observe)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        initial, target, _ = runner.encoded_inputs(current, batch, torch.ones(3), 'cpu')
    assert flags == [(False, torch.float32), (False, torch.float32)]
    assert initial.dtype == target.dtype == torch.float32


@pytest.mark.parametrize('stop', [1, 4, 5])
@pytest.mark.parametrize('variant', ['regression', 'gaussian'])
def test_interrupted_training_restores_unique_data_optimizer_and_weights(setup, tmp_path, monkeypatch, stop, variant):
    source, _ = make_cache(setup, tmp_path, monkeypatch)
    source.sequence_length = 1
    protocol = dict(fixture='finite_epoch', variant=variant)
    trained = []
    for label in ('whole', 'resumed'):
        current = model(variant)
        optimizer = torch.optim.AdamW(current.dynamics.parameters(), lr=1e-4, weight_decay=.01)
        data = restored_source(source)
        options = args(tmp_path/label, stop_after_updates=stop if label == 'resumed' else None)
        runner.fit(current, data, optimizer, options, protocol, capacities=torch.ones(3))
        if label == 'resumed':
            current = model(variant)
            optimizer = torch.optim.AdamW(current.dynamics.parameters(), lr=1e-4, weight_decay=.01)
            data = restored_source(source)
            runner.fit(current, data, optimizer, args(tmp_path/label, resume=True), protocol,
                capacities=torch.ones(3))
        trained.append(torch.load(tmp_path/label/'latest.pt', weights_only=True))
    assert trained[0]['update'] == trained[1]['update'] == 8
    assert trained[0]['exposures'] == trained[1]['exposures'] == 12
    assert trained[0]['epochs_completed'] == trained[1]['epochs_completed'] == 2
    for key, value in trained[0]['model'].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, trained[1]['model'][key], atol=0, rtol=0)
    for key, value in trained[0]['optimizer']['state'].items():
        for name, tensor in value.items():
            torch.testing.assert_close(tensor, trained[1]['optimizer']['state'][key][name], atol=0, rtol=0)
    whole = [json.loads(row) for row in (tmp_path/'whole/metrics.jsonl').read_text().splitlines()]
    resumed = [json.loads(row) for row in (tmp_path/'resumed/metrics.jsonl').read_text().splitlines()]
    assert whole == resumed
    for epoch in range(2):
        visited = [(row['episode'], start) for row in whole if row['epoch'] == epoch for start in row['starts']]
        assert sorted(visited) == [(episode, start) for episode in range(2) for start in range(3)]


def test_failed_attempt_cursor_and_journal_are_replayed_from_last_commit(setup, tmp_path, monkeypatch):
    source, _ = make_cache(setup, tmp_path, monkeypatch)
    source = restored_source(source)
    current = model('regression')
    optimizer = torch.optim.AdamW(current.dynamics.parameters(), lr=1e-4, weight_decay=.01)
    real = runner.update
    calls = 0
    def fail(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError('forced post-yield interruption')
        return real(*a, **kw)
    monkeypatch.setattr(runner, 'update', fail)
    options = args(tmp_path/'failed')
    with pytest.raises(RuntimeError, match='forced'):
        runner.fit(current, source, optimizer, options, dict(fixture=True), capacities=torch.ones(3))
    monkeypatch.setattr(runner, 'update', real)
    runner.fit(current, restored_source(source), optimizer, args(options.output_dir, resume=True),
        dict(fixture=True), capacities=torch.ones(3))
    assert list(options.output_dir.glob('metrics_uncommitted_*.jsonl'))
    rows = [json.loads(row) for row in (options.output_dir/'metrics.jsonl').read_text().splitlines()]
    assert [row['update'] for row in rows] == list(range(1, 9))


@pytest.mark.parametrize('variant', ['regression', 'gaussian'])
def test_validation_preserves_rng_cursor_modes_and_weights(setup, tmp_path, monkeypatch, variant):
    source, _ = make_cache(setup, tmp_path, monkeypatch)
    source.sequence_length = 1
    next(source.iter_epoch(2, np.random.default_rng(1)))
    current = model(variant)
    cursor, weights, rng = deepcopy(source.state_dict()), runner.tensor_identity(current.flow_projector.state_dict()), torch.get_rng_state()
    modes = [module.training for module in current.modules()]
    conditions = [dict(episode_index=0, start=0, demand_group='heldout')]
    result = runner.validate(current, source, restored_source(source), conditions, torch.ones(3),
        torch.full((3,), 10.), np.ones(2), device='cpu', seed=20, samples=3)
    assert result['states'] == 1 and np.isfinite(result['native']['loss'])
    assert source.state_dict() == cursor
    assert runner.tensor_identity(current.flow_projector.state_dict()) == weights
    assert torch.equal(torch.get_rng_state(), rng)
    assert [module.training for module in current.modules()] == modes
