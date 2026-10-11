"""Mainline behavior prior: checkpoint identity, global schedule, collection and evaluation."""
import copy
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_urban_dreamer_config import _write_config
from test_urban_dreamer_training import (make_trainer, fixed_demand,
                                         _complete_demand_session)
from urban_dreamer.models.policy import set_shortest_path_prior, shortest_path_prior_weight
from urban_dreamer.workflow.config import ConfigError, load_config


def configured_trainer(tmp_path, completed=599):
    trainer = make_trainer(tmp_path)
    trainer.config = replace(trainer.config, method=replace(trainer.config.method, sp_prior_weight=1.))
    trainer.metrics['online_episodes_completed'] = completed
    trainer._set_online_prior()
    return trainer


def test_disabled_config_preserves_legacy_fingerprint_bytes(tmp_path, monkeypatch):
    import urban_dreamer.workflow.config as module
    path = _write_config(tmp_path / 'config.toml')
    default = load_config(path)
    original_asdict = module.asdict
    def legacy_asdict(value):
        result = original_asdict(value)
        if isinstance(value, module.MethodConfig):
            result = {key: item for key, item in result.items() if not key.startswith('sp_prior_')}
        return result
    with monkeypatch.context() as patch:
        patch.setattr(module, 'asdict', legacy_asdict)
        legacy = module._semantic_fingerprint(**{key: getattr(default, key) for key in
            ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')})
    assert default.semantic_fingerprint == legacy
    path.write_text(path.read_text() + '\n[method]\nsp_prior_weight = 0.0\nsp_prior_temperature = 30.0\n'
                    'sp_prior_hold_episodes = 200\nsp_prior_off_episode = 1000\n')
    assert load_config(path).semantic_fingerprint == legacy
    path.write_text(path.read_text().replace('sp_prior_weight = 0.0', 'sp_prior_weight = 1.0'))
    enabled = load_config(path).semantic_fingerprint
    assert enabled != legacy
    path.write_text(path.read_text().replace('sp_prior_temperature = 30.0', 'sp_prior_temperature = 31.0'))
    assert load_config(path).semantic_fingerprint != enabled


@pytest.mark.parametrize('setting', ['sp_prior_weight = -1', 'sp_prior_weight = true',
    'sp_prior_evaluation_weight = -1', 'sp_prior_evaluation_weight = true',
    'sp_prior_evaluation_weight = nan', 'sp_prior_evaluation_weight = inf',
    'sp_prior_weight = nan', 'sp_prior_temperature = 0', 'sp_prior_temperature = inf',
    'sp_prior_hold_episodes = 1.5', 'sp_prior_hold_episodes = true',
    'sp_prior_off_episode = 200', 'sp_prior_hold_episodes = -1'])
def test_prior_fields_are_typed_and_validated(tmp_path, setting):
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path / 'config.toml', extra='\n[method]\n' + setting))


@pytest.mark.parametrize('training_weight', [0., 1.])
def test_evaluation_prior_is_opt_in_and_changes_fingerprint(tmp_path, training_weight):
    path = _write_config(tmp_path / 'config.toml',
                         extra=f'\n[method]\nsp_prior_weight = {training_weight}\n')
    default = load_config(path)
    path.write_text(path.read_text() + 'sp_prior_evaluation_weight = 0.0\n')
    assert load_config(path).semantic_fingerprint == default.semantic_fingerprint
    path.write_text(path.read_text().replace('sp_prior_evaluation_weight = 0.0',
                                            'sp_prior_evaluation_weight = 0.75'))
    enabled = load_config(path)
    assert enabled.method.sp_prior_evaluation_weight == .75
    assert enabled.semantic_fingerprint != default.semantic_fingerprint


def test_schedule_uses_global_completed_counter_through_all_passes(tmp_path, monkeypatch):
    trainer = configured_trainer(tmp_path, completed=599)
    seen = []
    monkeypatch.setattr(trainer, 'validate_codec_ready', lambda: None)
    def collect(**kwargs):
        seen.append(('real', trainer.actor.prior_weight))
        return SimpleNamespace(transitions=(), total_travel_minutes=0., total_vehicle_count=0,
                               arrival_times={}, metrics={})
    monkeypatch.setattr(trainer, '_run_episode', collect)
    monkeypatch.setattr(trainer.online_replay, 'append_episode', lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, '_update_world', lambda *args, **kwargs: {})
    def imagine():
        seen.append(('imagined', trainer.actor.prior_weight))
        return {}
    monkeypatch.setattr(trainer, '_train_imagination_pass', imagine)
    trainer.train_online_episode(0)  # A migration-local index cannot restart the hold period.
    assert seen == [('real', .5)] + [('imagined', .5)] * 16
    assert trainer.metrics['online_episodes_completed'] == 600
    trainer.train_online_episode(1)
    assert seen[17][1] == shortest_path_prior_weight(601)
    assert seen[17][1] < .5


def test_checkpoint_restores_global_prior_without_trainable_state_changes(tmp_path):
    trainer = configured_trainer(tmp_path, completed=599)
    state = copy.deepcopy(trainer.state_dict())
    assert all('prior' not in name for name in state['models']['actor'])
    set_shortest_path_prior(trainer.actor, 1., 70.)
    trainer.metrics['online_episodes_completed'] = 0
    trainer.load_state_dict(state)
    assert trainer.actor.prior_weight == pytest.approx(.5)
    assert trainer.actor.prior_temperature_seconds == 30.
    restored = trainer.state_dict()
    assert restored['next_environment_episode'] == state['next_environment_episode']
    assert restored['metrics'] == state['metrics']
    assert torch.equal(restored['rng']['torch_cpu'], state['rng']['torch_cpu'])
    assert restored['replay'] == state['replay']


def test_mainline_actor_uses_same_detached_cost_bias_and_gradient(tmp_path):
    trainer = configured_trainer(tmp_path)
    features = torch.zeros(1, 3, 10)
    features[0, :, 4] = torch.log1p(torch.tensor([30., 60., 120.])) / math.log1p(3600.)
    mask = torch.tensor([[True, True, False]])
    set_shortest_path_prior(trainer.actor, 0., 30.)
    bare = trainer.actor.forward_logits(features, mask)
    trainer._set_online_prior()
    biased = trainer.actor.forward_logits(features, mask)
    assert torch.allclose(biased[0, :2], bare[0, :2] - torch.tensor([0., .5]), atol=1e-6)
    assert torch.isneginf(biased[0, 2])
    loss = -torch.log_softmax(biased, -1)[0, 1]
    loss.backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in trainer.actor.parameters())


@pytest.mark.parametrize('mode,weight,evaluation_weight', [
    ('online', .5, 0.), ('evaluation', 0., 0.),
    ('online', .5, .75), ('evaluation', .75, .75)])
def test_collection_archive_labels_actual_prior_and_preserves_actor(
        tmp_path, monkeypatch, mode, weight, evaluation_weight):
    trainer = configured_trainer(tmp_path)
    trainer.evaluation_temperature = None  # Explicit legacy argmax protocol.
    trainer.config = replace(trainer.config, method=replace(
        trainer.config.method, sp_prior_evaluation_weight=evaluation_weight))
    trainer.demand = fixed_demand(trainer, lambda **kwargs: trainer.topology.vehicles[:1])
    factory = _complete_demand_session(trainer, [])
    seen = []
    original_select = trainer._select
    def select(*args, **kwargs):
        seen.append((kwargs['mode'], kwargs['actor'].prior_weight))
        return original_select(*args, **kwargs)
    monkeypatch.setattr(trainer, '_select', select)
    def session_factory(**kwargs):
        session = factory(**kwargs)
        advance = session.advance_window
        def advance_with_decision(callback):
            state = session.state()
            callback(state['vehicles'], state['flow'], state['time'])
            return advance(callback)
        session.advance_window = advance_with_decision
        return session
    trainer._session_factory = session_factory
    collected = trainer._run_episode(mode=mode, episode_index=0)
    assert collected.metrics[f'{mode}_sp_prior_weight'] == weight
    assert seen == [(mode, weight)]
    assert trainer.actor.prior_weight == .5
    with np.load(tmp_path / 'episodes' / mode / 'episode_000000.npz', allow_pickle=False) as archive:
        metadata = json.loads(archive['summary_json'].tobytes().decode())['metadata']
    assert metadata['sp_prior_weight'] == weight
    assert metadata['sp_prior_schedule_episode'] == 600
    if mode == 'evaluation':
        assert metadata['evaluation_policy'] == ('shortest_path_prior_argmax' if evaluation_weight
                                                 else 'zero_prior_argmax')


@pytest.mark.parametrize('evaluation_weight', [0., .75])
def test_isolated_evaluation_restores_prior_on_exception(tmp_path, evaluation_weight):
    from tools.run_online_pilot import isolated_evaluation
    trainer = configured_trainer(tmp_path)
    trainer.config = replace(trainer.config, method=replace(
        trainer.config.method, sp_prior_evaluation_weight=evaluation_weight))
    with pytest.raises(RuntimeError, match='failed'):
        with isolated_evaluation(trainer, tmp_path / 'eval'):
            set_shortest_path_prior(trainer.actor, evaluation_weight, 60.)
            raise RuntimeError('failed')
    assert trainer.actor.prior_weight == .5
    assert trainer.actor.prior_temperature_seconds == 30.


def test_enabled_prior_parallel_matches_serial_sampling_and_actor_gradients(monkeypatch):
    from test_parallel_imagination import trainer_case
    from urban_dreamer.learning.parallel_imagination import (collect_parallel_imagination,
                                                            close_parallel_imagination)
    trainer = trainer_case(deadline=5., rollouts=1)
    set_shortest_path_prior(trainer.actor, .5, 30.)
    torch.manual_seed(434)
    actor_seed = torch.randint(0, 2**63 - 1, (1, 2))[0, 0].item()
    batch = trainer.online_replay.sample_active_seeds(batch_size=1, rng=np.random.default_rng(124), initial_only=True)
    torch.manual_seed(actor_seed)
    expected, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0, batched_events=True)
    expected_loss = -torch.cat([group['log_prob'] for group in expected]).mean()
    expected_gradients = torch.autograd.grad(expected_loss, tuple(trainer.actor.parameters()))
    try:
        torch.manual_seed(434)
        actual, _ = collect_parallel_imagination(trainer, 1)
        assert [group['event_index'] for group in actual] == [group['event_index'] for group in expected]
        for key in ('log_prob', 'entropy', 'global_return', 'global_advantage', 'counterfactual_baseline'):
            torch.testing.assert_close(torch.cat([group[key] for group in actual]),
                                       torch.cat([group[key] for group in expected]), atol=1e-6, rtol=1e-5)
        actual_loss = -torch.cat([group['log_prob'] for group in actual]).mean()
        actual_gradients = torch.autograd.grad(actual_loss, tuple(trainer.actor.parameters()))
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-5)
    finally:
        close_parallel_imagination(trainer)
