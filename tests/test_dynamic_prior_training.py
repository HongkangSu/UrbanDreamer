"""Real and imagined actor sampling retain identical dynamic prior conditions."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from urban_dreamer.models.policy import set_shortest_path_prior


@pytest.mark.parametrize('mode', ['online', 'evaluation'])
def test_real_sampler_matches_imagination_builder_and_archives_dynamic_mode(tmp_path, monkeypatch, mode):
    from test_mainline_sp_prior import configured_trainer
    from test_urban_dreamer_training import fixed_demand, _complete_demand_session
    trainer = configured_trainer(tmp_path)
    trainer.config = replace(trainer.config, method=replace(trainer.config.method,
        sp_prior_cost_mode='dynamic', sp_prior_dynamic_speed_floor_mps=.5,
        sp_prior_evaluation_weight=.5))
    trainer.demand = fixed_demand(trainer, lambda **kwargs: trainer.topology.vehicles[:1])
    factory = _complete_demand_session(trainer, [])
    calls = []
    original_select = trainer._select
    def select(states, flow, timestamp, **kwargs):
        observations = kwargs['observations']
        reference = trainer._observations(states, flow, kwargs['context'], timestamp)
        assert observations.prior_costs_seconds is not None
        np.testing.assert_array_equal(observations.features, reference.features)
        np.testing.assert_array_equal(observations.prior_costs_seconds, reference.prior_costs_seconds)
        actor = kwargs['actor']
        features = torch.as_tensor(observations.features)
        mask = torch.as_tensor(observations.mask)
        costs = torch.as_tensor(observations.prior_costs_seconds)
        expected = actor.forward_logits(features, mask, prior_costs_seconds=costs)
        seen = []
        sample = actor.sample_action
        def checked_sample(features, mask, **sample_kwargs):
            torch.testing.assert_close(sample_kwargs['prior_costs_seconds'], costs)
            actual = actor.forward_logits(features, mask, prior_costs_seconds=sample_kwargs['prior_costs_seconds'])
            torch.testing.assert_close(actual, expected)
            seen.append(True)
            return sample(features, mask, **sample_kwargs)
        monkeypatch.setattr(actor, 'sample_action', checked_sample)
        try:
            result = original_select(states, flow, timestamp, **kwargs)
        finally:
            monkeypatch.setattr(actor, 'sample_action', sample)
        assert seen
        calls.append(True)
        return result
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
    trainer._run_episode(mode=mode, episode_index=0)
    assert calls
    with np.load(tmp_path / 'episodes' / mode / 'episode_000000.npz', allow_pickle=False) as archive:
        metadata = json.loads(archive['summary_json'].tobytes().decode())['metadata']
    assert metadata['sp_prior_cost_mode'] == 'dynamic'
    assert metadata['sp_prior_dynamic_speed_floor_mps'] == .5


def test_dynamic_serial_parallel_saved_probabilities_losses_and_gradients_match(monkeypatch):
    import test_parallel_imagination as cases
    original = cases.trainer_case
    def dynamic_case(*args, **kwargs):
        trainer = original(*args, **kwargs)
        trainer.config.method.sp_prior_cost_mode = 'dynamic'
        trainer.config.method.sp_prior_dynamic_speed_floor_mps = .5
        trainer.config.simulation.vehicle_max_speed_mps = 5.
        set_shortest_path_prior(trainer.actor, .7, 30.)
        decode = trainer.world_model.decode
        def congested(latent):
            result = decode(latent)
            result[..., 1] = 2.
            result[:, 2, 1] = .1
            return result
        trainer.world_model.decode = congested
        return trainer
    monkeypatch.setattr(cases, 'trainer_case', dynamic_case)
    cases.test_multiwindow_parallel_matches_serial_returns_bootstrap_and_gradients(monkeypatch, 65., 3)


def test_serial_recomputed_logprob_uses_sampling_dynamic_costs(monkeypatch):
    from test_parallel_imagination import trainer_case
    trainer = trainer_case(deadline=25., rollouts=1)
    trainer.config.method.sp_prior_cost_mode = 'dynamic'
    trainer.config.method.sp_prior_dynamic_speed_floor_mps = .5
    set_shortest_path_prior(trainer.actor, .7, 30.)
    batch = trainer.online_replay.sample_active_seeds(batch_size=1, rng=trainer.rng, initial_only=True)
    calls = []
    forward = trainer.actor.forward_logits
    def capture(features, mask, **kwargs):
        costs = kwargs['prior_costs_seconds']
        logits = forward(features, mask, **kwargs)
        calls.append((costs.detach().clone(), logits.detach().clone(), torch.is_grad_enabled()))
        return logits
    monkeypatch.setattr(trainer.actor, 'forward_logits', capture)
    trainer._imagine_from_seed(batch, 0, rollout_index=0, batched_events=True)
    assert len(calls) >= 2
    sampled = [row for row in calls if not row[2]]
    recomputed = [row for row in calls if row[2]]
    assert len(recomputed) == 1
    torch.testing.assert_close(torch.cat([row[0] for row in sampled]), recomputed[0][0])
    torch.testing.assert_close(torch.cat([row[1] for row in sampled]), recomputed[0][1])


def test_speed_cap_changes_only_dynamic_prior_not_neural_observations():
    from test_parallel_imagination import trainer_case
    trainer = trainer_case(deadline=25., rollouts=1)
    trainer.config.method.sp_prior_cost_mode = 'dynamic'
    trainer.config.method.sp_prior_dynamic_speed_floor_mps = .5
    batch = trainer.online_replay.sample_active_seeds(batch_size=1, rng=trainer.rng, initial_only=True)
    flow = np.zeros((4, 4), dtype=np.float32)
    flow[:, 1] = 2.
    context = trainer._context(batch.vehicles[0], 0., started_count=4)
    trainer.config.simulation.vehicle_max_speed_mps = 2.
    slow = trainer._observations(batch.vehicles[0], flow, context, 0.)
    trainer.config.simulation.vehicle_max_speed_mps = 40.
    fast = trainer._observations(batch.vehicles[0], flow, context, 0.)
    np.testing.assert_array_equal(slow.features, fast.features)
    np.testing.assert_array_equal(slow.mask, fast.mask)
    legal = slow.mask
    assert np.all(slow.prior_costs_seconds[legal] >= fast.prior_costs_seconds[legal])
    assert np.any(slow.prior_costs_seconds[legal] > fast.prior_costs_seconds[legal])
