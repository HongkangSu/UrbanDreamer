"""Behavioral tests of the real experimental runner, with small physical replay."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import tools.train_component_world_model as runner
from test_cohort_targets import cohort_replay
from urban_dreamer.models.experimental_dynamics import VARIANTS, apply_variant
from urban_dreamer.models.world_model import UrbanDreamerWorldModel


CAPACITIES = np.array([10., 20., 30.], dtype=np.float32)


def small_model(seed=5):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = UrbanDreamerWorldModel(adjacency=torch.ones(3, 3)-torch.eye(3),
        static_features=torch.ones(3, 4), low_frequency_rank=1, event_protocol=True)
    model.dynamics.transformer.layers = nn.ModuleList(model.dynamics.transformer.layers[:1])
    model.dynamics.transformer.activation_checkpointing = False
    nn.init.normal_(model.dynamics.flow_head.weight, std=.01)
    model.flow_projector.requires_grad_(False)
    model.active_vehicle_fraction_readout.requires_grad_(False)
    return model


@pytest.mark.parametrize('variant', VARIANTS)
def test_all_predictor_variants_receive_real_gradients_without_codec_updates(variant):
    model = apply_variant(small_model(), variant, seed=12)
    replay = cohort_replay()
    replay.sequence_length = 1
    batch = replay._batch_from_picks([(0, 0), (0, 1)])
    loss, metrics, exposures = runner.supervised_loss(model, batch, CAPACITIES,
        mode='single_step', device='cpu', steps=1)
    assert exposures == 2
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in (model.dynamics.flow_head.weight,
                      model.dynamics.event_encoder.event_mlp[0].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert all(p.grad is None for p in model.flow_projector.parameters())
    if variant == 'no_prior_fm':
        assert metrics['prior_loss'] == 0.


@pytest.mark.parametrize('variant', ['graph_fm', 'no_prior_fm', 'random_prior_fm'])
@pytest.mark.parametrize('mode', ['independent_cohort', 'recursive_fm', 'sample_ms', 'mean_ms'])
def test_cohort_modes_propagate_finite_gradients_through_actual_fm(variant, mode):
    model = apply_variant(small_model(), variant, seed=12)
    batch = cohort_replay()._batch_from_picks([(0, 0)])
    loss, metrics, exposures = runner.supervised_loss(model, batch, CAPACITIES,
        mode=mode, device='cpu', steps=1, ms_samples=2)
    assert exposures == 3
    assert torch.isfinite(loss)
    loss.backward()
    assert model.dynamics.flow_head.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.dynamics.flow_head.weight.grad).all()
    assert all(p.grad is None for p in model.flow_projector.parameters())
    if variant == 'no_prior_fm':
        assert metrics['prior_loss'] == 0


def test_cohort_modes_use_identical_physical_labels_and_exclude_newborn_events(monkeypatch):
    model = small_model()
    batch = cohort_replay()._batch_from_picks([(0, 0)])
    captures = []
    real_independent, real_recursive = runner.experimental_loss, runner.recursive_fm_loss

    def independent(model, **kwargs):
        captures.append((kwargs['z_inputs'], kwargs['z_targets'], kwargs['action'], kwargs['action_events']))
        return real_independent(model, **kwargs)

    def recursive(model, **kwargs):
        sequence = kwargs['z_sequence']
        captures.append((sequence[:, :-1], sequence[:, 1:], kwargs['action'], kwargs['action_events']))
        return real_recursive(model, **kwargs)

    monkeypatch.setattr(runner, 'experimental_loss', independent)
    monkeypatch.setattr(runner, 'recursive_fm_loss', recursive)
    for mode in ('independent_cohort', 'recursive_fm', 'sample_ms', 'mean_ms'):
        runner.supervised_loss(model, batch, CAPACITIES, mode=mode, device='cpu', steps=1, ms_samples=2)
    inputs, targets, actions, events = captures[0]
    # Independently constructed from observed membership: car a arrives at t20,
    # internal enters edge 1 at t30, is in junction at t60, arrives at t75.
    physical = batch.flow.clone()
    physical[0, :, :, 2] = torch.tensor([[.1, 0., 0.], [0., .05, 0.], [0., 0., 0.], [0., 0., 0.]])
    with torch.no_grad():
        expected = model.encode(physical.reshape(-1, 3, 8)).reshape(1, 4, 3, 64)
    torch.testing.assert_close(inputs, expected[:, :-1])
    torch.testing.assert_close(targets, expected[:, 1:])
    assert events.vehicle_ids == ('a', 'internal', 'internal')
    torch.testing.assert_close(actions, torch.tensor([[[0., .5, 0.], [0., 0., .25], [0., 0., 0.]]]))
    for other_inputs, other_targets, other_actions, other_events in captures[1:]:
        torch.testing.assert_close(other_inputs, inputs)
        torch.testing.assert_close(other_targets, targets)
        torch.testing.assert_close(other_actions, actions)
        assert other_events.vehicle_ids == events.vehicle_ids
        assert not other_inputs.requires_grad and not other_targets.requires_grad


def test_single_step_is_declared_production_reference_not_fixed_cohort_labels(monkeypatch):
    model = small_model()
    replay = cohort_replay()
    replay.sequence_length = 1
    # At t30, cars newly born during the previous window are now eligible.
    batch = replay._batch_from_picks([(0, 1)])
    original = runner.experimental_loss
    seen = []

    def capture(model, **kwargs):
        seen.append(kwargs)
        return original(model, **kwargs)

    monkeypatch.setattr(runner, 'experimental_loss', capture)
    runner.supervised_loss(model, batch, CAPACITIES, mode='single_step', device='cpu', steps=1)
    physical_target = batch.flow[:, 1:2].clone()
    physical_target[0, 0, :, 2] = torch.tensor([0., 0., 2/30])
    with torch.no_grad():
        expected = model.encode(physical_target.reshape(-1, 3, 8)).reshape(1, 1, 3, 64)
    torch.testing.assert_close(seen[0]['z_targets'], expected)
    assert seen[0]['action_events'].vehicle_ids == ('internal', 'queued')


def test_sample_and_mean_ms_have_hand_computable_spread_gradient(monkeypatch):
    model = small_model()
    # A zero codec makes the target exactly zero, isolating spread penalties.
    monkeypatch.setattr(model, 'encode', lambda x: x.new_zeros((*x.shape[:-1], 64)))
    spread = model.dynamics.flow_head.bias
    with torch.no_grad():
        spread.fill_(2.)
    replay = cohort_replay()
    replay.sequence_length = 1
    batch = replay._batch_from_picks([(0, 0)])

    def known_recursive(dynamics, *, z_sequence, **kwargs):
        positive = spread.expand_as(z_sequence[:, 1:])
        return spread.sum()*0, {}, positive

    monkeypatch.setattr(runner, 'recursive_fm_loss', known_recursive)
    monkeypatch.setattr(model, 'transition', lambda current, *a, **kw: -spread.expand_as(current))
    sample, sample_metrics, _ = runner.supervised_loss(model, batch, CAPACITIES,
        mode='sample_ms', device='cpu', ms_samples=2)
    mean, mean_metrics, _ = runner.supervised_loss(model, batch, CAPACITIES,
        mode='mean_ms', device='cpu', ms_samples=2)
    assert sample_metrics['endpoint_ms_loss'] == 4.
    assert mean_metrics['endpoint_ms_loss'] == 0.
    sample_grad = torch.autograd.grad(sample, spread)[0]
    mean_grad = torch.autograd.grad(mean, spread)[0]
    torch.testing.assert_close(sample_grad, torch.full_like(spread, 4/64))
    torch.testing.assert_close(mean_grad, torch.zeros_like(spread))


def test_reconcile_archives_uncommitted_rows_and_preserves_unique_history(tmp_path):
    path = tmp_path/'metrics.jsonl'
    rows = [dict(update=i, loss=float(i)) for i in (1, 2, 3)]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    runner.reconcile_training_journal(path, 2)
    assert [json.loads(row)['update'] for row in path.read_text().splitlines()] == [1, 2]
    archived = list(tmp_path.glob('metrics_uncommitted_*.jsonl'))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text()) == rows[-1]
    runner.reconcile_training_journal(path, 2)
    assert len(list(tmp_path.glob('metrics_uncommitted_*.jsonl'))) == 1
    path.write_text('{"update":1}\n{"update":1}\n')
    with pytest.raises(ValueError, match='duplicated'):
        runner.reconcile_training_journal(path, 2)


def train_args(output, variant='graph_fm', resume=False):
    return SimpleNamespace(experiment_dir=output.parent/'experiment', output_dir=output,
        variant=variant, seed=7, data_seed=99, updates=4, batch_size=1,
        learning_rate=1e-4, mode='single_step', horizon=3, steps=1, ms_samples=2,
        ms_weight=1., precision='fp32', device='cpu', threads=1, gpu_memory_fraction=.08,
        checkpoint_every=2, milestone_every=2, resume=resume)


def fake_inputs(monkeypatch, *, fail_at=None, torch_draws=0):
    calls = []
    split_calls = []
    replay = cohort_replay()
    source = SimpleNamespace(manifest=dict(identity='frozen-train-only', entries=[
        dict(windows=3, demand_group='train-a'), dict(windows=3, demand_group='train-b')]))

    def batch_from_starts(episode, starts, **kwargs):
        calls.append((episode, tuple(starts)))
        if fail_at is not None and len(calls) == fail_at:
            raise RuntimeError('simulated data interruption')
        torch.rand(torch_draws)
        # Both synthetic episodes have identical valid physical replay.
        batch = replay._batch_from_picks([(0, start) for start in starts])
        return replace(batch, episode=torch.full_like(batch.episode, episode))

    source.batch_from_starts = batch_from_starts

    def data(*args, split, horizon):
        assert split == 'train'  # Any attempted development/test access fails.
        split_calls.append(split)
        replay.sequence_length = horizon
        return source

    def frozen(*args, device, fresh_dynamics_seed):
        config = SimpleNamespace(method=SimpleNamespace(prior_loss_weight=.1))
        topology = SimpleNamespace(network=SimpleNamespace(capacities=CAPACITIES))
        return config, None, topology, small_model(fresh_dynamics_seed).to(device), {'source': 'fixed'}

    monkeypatch.setattr(runner, 'dataset', data)
    monkeypatch.setattr(runner, 'load_frozen', frozen)
    return calls, split_calls


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0., atol=0.)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_nested_equal(a, b)
    else:
        assert first == second


@pytest.mark.parametrize('variant', ['graph_fm', 'gaussian'])
def test_actual_optimizer_resume_is_bit_exact_and_numpy_state_weights_only_safe(monkeypatch, tmp_path, variant):
    calls, split_calls = fake_inputs(monkeypatch)
    continuous_args = train_args(tmp_path/'continuous', variant)
    runner.train(continuous_args)
    schedule = list(calls)
    assert split_calls == ['train']
    continuous = torch.load(continuous_args.output_dir/'latest.pt', weights_only=True)
    interrupted_args = train_args(tmp_path/'interrupted', variant)
    fake_inputs(monkeypatch, fail_at=4)
    with pytest.raises(RuntimeError, match='simulated'):
        runner.train(interrupted_args)
    partial = torch.load(interrupted_args.output_dir/'latest.pt', weights_only=True)
    assert partial['update'] == 2
    assert json.loads((interrupted_args.output_dir/'status.json').read_text())['phase'] == 'failed'
    recovered_calls, _ = fake_inputs(monkeypatch)
    interrupted_args.resume = True
    runner.train(interrupted_args)
    recovered = torch.load(interrupted_args.output_dir/'latest.pt', weights_only=True)
    for key in ('model', 'optimizer', 'data_rng', 'torch_rng', 'cuda_rng', 'update', 'exposures'):
        assert_nested_equal(continuous[key], recovered[key])
    assert recovered_calls == schedule[2:]
    rows = [json.loads(row) for row in (interrupted_args.output_dir/'metrics.jsonl').read_text().splitlines()]
    assert [row['update'] for row in rows] == [1, 2, 3, 4]
    assert len(list(interrupted_args.output_dir.glob('metrics_uncommitted_*.jsonl'))) == 1


def test_replay_schedule_does_not_depend_on_model_or_global_torch_random_consumption(monkeypatch, tmp_path):
    calls, _ = fake_inputs(monkeypatch)
    runner.train(train_args(tmp_path/'fm', 'graph_fm'))
    expected = list(calls)
    calls, _ = fake_inputs(monkeypatch, torch_draws=1000)
    runner.train(train_args(tmp_path/'gaussian', 'gaussian'))
    assert calls == expected


def test_training_refuses_silent_overwrite_and_protocol_change_on_resume(monkeypatch, tmp_path):
    fake_inputs(monkeypatch)
    args = train_args(tmp_path/'run')
    runner.train(args)
    with pytest.raises(ValueError, match='explicit resume'):
        runner.train(args)
    args.resume = True
    args.data_seed += 1
    with pytest.raises(ValueError, match='protocol differs'):
        runner.train(args)


@pytest.mark.parametrize('variant', ['graph_fm', 'regression', 'gaussian'])
def test_declared_budget_extension_preserves_learning_state_and_parent(monkeypatch, tmp_path, variant):
    fake_inputs(monkeypatch)
    continuous = train_args(tmp_path/'continuous', variant)
    runner.train(continuous)
    expected = torch.load(continuous.output_dir/'latest.pt', weights_only=True)

    parent = train_args(tmp_path/'parent', variant)
    parent.updates = 2
    runner.train(parent)
    parent_file = parent.output_dir/'latest.pt'
    parent_hash = runner.sha256(parent_file)
    parent_metrics = (parent.output_dir/'metrics.jsonl').read_bytes()
    extension = train_args(tmp_path/'extension', variant)
    extension.continue_from = parent_file
    extension.continuation_reason = 'Measure development convergence at a larger predeclared budget'
    runner.train(extension)
    actual = torch.load(extension.output_dir/'latest.pt', weights_only=True)
    for key in ('model','optimizer','data_rng','torch_rng','cuda_rng','update','exposures'):
        assert_nested_equal(expected[key], actual[key])
    assert runner.sha256(parent_file) == parent_hash
    assert (parent.output_dir/'metrics.jsonl').read_bytes() == parent_metrics
    assert actual['protocol']['continuation']['parent_checkpoint_sha256'] == parent_hash
    assert actual['protocol']['continuation']['parent_update'] == 2
    assert [json.loads(row)['update'] for row in (extension.output_dir/'metrics.jsonl').read_text().splitlines()] == [3,4]
    extension.resume = True
    extension.continue_from = None
    runner.train(extension)
    assert torch.load(extension.output_dir/'latest.pt', weights_only=True)['update'] == 4


@pytest.mark.parametrize('change', ['no_reason','data_seed','variant','no_optimizer','unfinished_parent'])
def test_budget_extension_rejects_hidden_treatment_changes(monkeypatch,tmp_path,change):
    fake_inputs(monkeypatch)
    parent = train_args(tmp_path/'parent')
    parent.updates = 2
    runner.train(parent)
    extension = train_args(tmp_path/'extension')
    extension.continue_from = parent.output_dir/'latest.pt'
    extension.continuation_reason = 'Declared budget extension'
    if change == 'no_reason':
        extension.continuation_reason = ''
    elif change == 'data_seed':
        extension.data_seed += 1
    elif change == 'variant':
        extension.variant = 'regression'
    else:
        checkpoint = torch.load(extension.continue_from,weights_only=True)
        if change == 'no_optimizer':
            checkpoint.pop('optimizer')
        else:
            checkpoint['protocol']['planned_updates'] = 3
        torch.save(checkpoint,extension.continue_from)
    with pytest.raises(ValueError):
        runner.train(extension)
