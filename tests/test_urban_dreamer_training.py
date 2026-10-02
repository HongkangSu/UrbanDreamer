from __future__ import annotations

import copy
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology

ROOT = Path(__file__).resolve().parents[1]


def make_trainer(tmp_path):
    config = load_config(ROOT / 'configs/toy_cross_bottleneck.toml')
    config = replace(config, run=replace(config.run, output_dir=tmp_path, device='cpu',
        warmup_episodes=1, world_model_initialization_epochs=1,
        online_training_episodes=1, evaluation_episodes=1),
        simulation=replace(config.simulation, world_model_intervals_per_episode=6,
            training_completion_buffer_seconds=0, evaluation_horizon_seconds=180,
            demand_duration_seconds=30),
        resources=replace(config.resources, world_model_initialization_batch_size=1,
            world_model_initialization_sequence_length=1, online_world_model_batch_size=1))
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    topology = replace(topology, vehicles=topology.vehicles[:2])
    return MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology)


def test_trainer_state_contains_only_global_value_critic(tmp_path):
    trainer = make_trainer(tmp_path)
    state = trainer.state_dict()
    assert set(state['models']) == {'world_model', 'actor', 'latent_critic'}
    assert set(state['optimizers']) == {'world_model_initial', 'world_model_online',
                                       'active_fraction', 'actor', 'latent_critic'}
    assert not hasattr(trainer, 'vehicle_critic')
    assert not any('vehicle_critic' in key for key in trainer.metrics)


def test_checkpoint_roundtrip_restores_all_rng_and_rejects_old(tmp_path):
    trainer = make_trainer(tmp_path)
    state = copy.deepcopy(trainer.state_dict())
    features, mask = torch.zeros(16, 2, 10), torch.ones(16, 2, dtype=torch.bool)
    expected_selection = trainer.actor.sample_action(features, mask).candidate_index
    expected = (trainer.rng.random(), torch.rand(3), np.random.random(), random.random())
    trainer.load_state_dict(state)
    assert torch.equal(trainer.actor.sample_action(features, mask).candidate_index, expected_selection)
    assert trainer.rng.random() == expected[0]
    assert torch.equal(torch.rand(3), expected[1])
    assert np.random.random() == expected[2]
    assert random.random() == expected[3]
    with pytest.raises(ValueError, match='schema'):
        trainer.load_state_dict({**state, 'legacy': {}})
    legacy = dict(state)
    del legacy['reward_contract']
    with pytest.raises(ValueError, match='schema'):
        trainer.load_state_dict(legacy)
    with pytest.raises(ValueError, match='reward contract'):
        trainer.load_state_dict({**state, 'reward_contract': 'shared_timeout'})
    before = copy.deepcopy(trainer.actor.state_dict())
    for section in ('models', 'optimizers'):
        invalid = copy.deepcopy(state)
        invalid[section]['vehicle_critic'] = {}
        invalid['models']['actor']['policy_head.bias'].fill_(99.)
        with pytest.raises(ValueError, match='schema'):
            trainer.load_state_dict(invalid)
        assert all(torch.equal(value, trainer.actor.state_dict()[key])
                   for key, value in before.items())
    assert all('hdv' not in key.lower() for key in state['models']['world_model'])


def test_empty_windows_and_future_only_seeds_skip_imagination(tmp_path, monkeypatch):
    import urban_dreamer.learning.training as module
    from urban_dreamer.learning.replay import MacroTransition
    trainer = make_trainer(tmp_path)
    _set_deadline(trainer, 30)
    monkeypatch.setattr(module, 'IMAGINED_ROLLOUTS_PER_PASS', 1)
    network = trainer.topology.network
    flow = network.extract_flow({})
    demand = trainer.topology.vehicles[0]
    current = network.edge_to_index[trainer.topology.origins[demand.origin_index]]
    destination = network.edge_to_index[trainer.topology.destinations[demand.destination_index]]
    snapshot = dict(vehicle_id=demand.vehicle_id, current_edge=current,
        destination_edge=destination, departure_time=1000., departed=False, arrived=False,
        remaining_distance=float(network.lengths_m[current]), committed_next_edge=None,
        pending_decision=False, junction_remaining_distance=0., junction_speed=1.)
    trainer.online_replay.append_episode([MacroTransition(flow=flow,
        action=np.zeros(network.edge_count), departure=np.zeros(network.edge_count),
        context=np.zeros(2), next_flow=flow, active_vehicle_fraction=0.,
        vehicles=(snapshot,), time=0.)], total_travel_minutes=1., total_vehicle_count=2)
    before = [parameter.detach().clone() for parameter in trainer.latent_critic.parameters()]
    metrics = trainer._train_imagination_pass()
    assert metrics['imagined_events'] == 0
    assert metrics['latent_critic_loss'] == 0
    assert all(torch.equal(a,b) for a,b in zip(before, trainer.latent_critic.parameters()))


def test_readout_does_not_override_exact_cohort_reward_or_event_advantage(tmp_path, monkeypatch):
    import urban_dreamer.learning.training as module
    from urban_dreamer.learning.replay import MacroTransition
    trainer = make_trainer(tmp_path)
    _set_deadline(trainer, 30)
    network = trainer.topology.network
    demand = trainer.topology.vehicles[0]
    current = network.edge_to_index[trainer.topology.origins[demand.origin_index]]
    destination = network.edge_to_index[trainer.topology.destinations[demand.destination_index]]
    snapshot = dict(vehicle_id=demand.vehicle_id, current_edge=current,
        destination_edge=destination, departure_time=0., departed=True, arrived=False,
        remaining_distance=float(network.lengths_m[current]), committed_next_edge=None,
        pending_decision=True, junction_remaining_distance=0., junction_speed=1.)
    flow = network.extract_flow({})
    trainer.online_replay.append_episode([MacroTransition(flow=flow,
        action=np.zeros(network.edge_count), departure=np.zeros(network.edge_count),
        context=np.array([0., .5]), next_flow=flow, active_vehicle_fraction=.5,
        vehicles=(snapshot,), time=0.)], total_travel_minutes=1., total_vehicle_count=2)
    batch = trainer.online_replay.sample_batch(batch_size=1, rng=trainer.rng)
    torch.manual_seed(10)
    rows_a, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    with torch.no_grad():
        trainer.world_model.active_vehicle_fraction_readout.global_logit_bias.add_(4.)
    torch.manual_seed(10)
    rows_b, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    assert rows_a and rows_b
    torch.testing.assert_close(rows_b[0]['global_reward'], rows_a[0]['global_reward'])
    torch.testing.assert_close(rows_b[0]['global_advantage'], rows_a[0]['global_advantage'])


def test_real_toy_lifecycle_resumes_after_initialization(tmp_path, monkeypatch):
    import shutil
    import sys
    import urban_dreamer.learning.training as module
    from urban_dreamer.workflow.pipeline import MainlinePipeline
    from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint
    if shutil.which('sumo') is None and not (Path(sys.executable).parent / 'sumo').exists():
        pytest.skip('SUMO executable unavailable')
    monkeypatch.setattr(module, 'ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE', 1)
    monkeypatch.setattr(module, 'IMAGINED_PASSES_PER_EPISODE', 1)
    monkeypatch.setattr(module, 'IMAGINED_ROLLOUTS_PER_PASS', 1)
    first = make_trainer(tmp_path)
    from types import SimpleNamespace
    first.demand = SimpleNamespace(sample=lambda **kwargs: first.topology.vehicles)
    def interrupt(_):
        raise RuntimeError('simulated phase interruption')
    first.train_online_episode = interrupt
    pipeline = MainlinePipeline(config=first.config, input_hash='toy-two-vehicle-smoke', trainer_factory=lambda: first)
    with pytest.raises(RuntimeError, match='simulated phase interruption'):
        pipeline.run()
    checkpoint = load_checkpoint(output_dir=tmp_path,
        expected_config_fingerprint=first.config.semantic_fingerprint,
        expected_input_hash='toy-two-vehicle-smoke')
    assert checkpoint.phase is CheckpointPhase.ONLINE_TRAINING
    assert len(first.warmup_replay) > 0
    resumed = make_trainer(tmp_path)
    resumed.demand = SimpleNamespace(sample=lambda **kwargs: resumed.topology.vehicles)
    result = MainlinePipeline(config=resumed.config, input_hash='toy-two-vehicle-smoke',
                             trainer_factory=lambda: resumed).run()
    assert result['status'] == 'completed'
    assert result['warmup_episodes_completed'] == 1
    assert result['world_model_initialization_epochs_completed'] == 1
    assert result['online_episodes_completed'] == 1
    assert result['evaluation_episodes_completed'] == 1
    assert result['online_events'] > 2
    assert result['evaluation_completion_rate'] == 1
    assert result['world_model_loss_last'] > 0
    assert result['active_fraction_loss_last'] >= 0
    assert result['latent_critic_loss_last'] >= 0
    assert MainlinePipeline(config=resumed.config, input_hash='toy-two-vehicle-smoke',
        trainer_factory=lambda: pytest.fail('completed pipeline must not rebuild trainer')).run() == result


def _in_transit_seed(trainer, *, junction=False, destination=False):
    from urban_dreamer.learning.replay import MacroTransition
    network = trainer.network
    demand = trainer.topology.vehicles[0]
    current = network.edge_to_index[trainer.topology.origins[demand.origin_index]]
    target = network.edge_to_index[trainer.topology.destinations[demand.destination_index]]
    next_edge = trainer.candidates.candidates(current, target)[0]
    snapshot = dict(vehicle_id=demand.vehicle_id, current_edge=target if destination else current,
        destination_edge=target, departure_time=0., departed=True, arrived=False,
        remaining_distance=10. if destination else (0. if junction else 10000.),
        committed_next_edge=None if destination else next_edge, pending_decision=False,
        in_junction=junction, junction_remaining_distance=60. if junction else 0., junction_speed=1.)
    flow = network.extract_flow({})
    trainer.online_replay.append_episode([MacroTransition(flow=flow,
        action=np.zeros(network.edge_count), departure=np.zeros(network.edge_count),
        context=np.array([0., .5]), next_flow=flow, active_vehicle_fraction=.5,
        vehicles=(snapshot,), time=0.)], total_travel_minutes=.5, total_vehicle_count=2)
    return trainer.online_replay.sample_batch(batch_size=1, rng=trainer.rng)


@pytest.mark.parametrize('junction', [False, True])
def test_macro_critic_trains_without_policy_events(tmp_path, monkeypatch, junction):
    import urban_dreamer.learning.training as module
    trainer = make_trainer(tmp_path)
    _set_deadline(trainer, 30)
    monkeypatch.setattr(module, 'IMAGINED_ROLLOUTS_PER_PASS', 1)
    _in_transit_seed(trainer, junction=junction)
    before = [parameter.detach().clone() for parameter in trainer.latent_critic.parameters()]
    metrics = trainer._train_imagination_pass()
    assert metrics['imagined_events'] == 0
    assert metrics['latent_critic_loss'] > 0
    assert metrics['actor_loss'] == 0
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.latent_critic.parameters()))
    assert not any('vehicle_critic' in key or 'critic_only' in key for key in metrics)


@pytest.mark.parametrize('timeout_returns,expected_loss', [([0., 0.], -2.5), ([-5000., 0.], -.75)])
def test_actor_uses_scale_only_global_advantages(tmp_path, monkeypatch, timeout_returns, expected_loss):
    import urban_dreamer.learning.training as module
    trainer = make_trainer(tmp_path)
    _in_transit_seed(trainer)
    monkeypatch.setattr(module, 'IMAGINED_ROLLOUTS_PER_PASS', 1)
    parameter = trainer.actor.policy_head.bias.reshape(())
    def imagine(*args, **kwargs):
        rows = [dict(log_prob=parameter + value, entropy=parameter * 0,
                     global_advantage=torch.tensor(advantage), timeout_return=timeout)
                for (value, advantage), timeout in zip([(1., 1.), (2., 2.)], timeout_returns)]
        return rows, sum(p.square().sum() for p in trainer.latent_critic.parameters())
    monkeypatch.setattr(trainer, '_imagine_from_seed', imagine)
    metrics = trainer._train_imagination_pass()
    # std=.5 is floored at one; positive advantages retain their sign.
    # The failed vehicle receives -2.5 after clipping; the arrived vehicle keeps
    # its original +2 advantage, unaffected by the other vehicle's -5000 return.
    assert metrics['actor_loss'] == pytest.approx(expected_loss, abs=1e-6)


def test_collection_samples_in_training_and_uses_argmax_in_evaluation(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    snapshot = _in_transit_seed(trainer).vehicles[0][0]
    snapshot['pending_decision'] = True
    modes = []
    original = trainer.actor.sample_action
    def select(*args, **kwargs):
        modes.append(kwargs['deterministic'])
        return original(*args, **kwargs)
    monkeypatch.setattr(trainer.actor, 'sample_action', select)
    flow = trainer.network.extract_flow({})
    for mode in ('online', 'evaluation'):
        decisions = trainer._select([snapshot], flow, 0., context=np.array([0., .5]), mode=mode)
        assert len(decisions) == 1
    assert modes == [False, True]


@pytest.mark.parametrize('frames', [2, 4])
def test_decoder_labels_are_forward_window_means_without_mutating_physical_states(tmp_path, monkeypatch, frames):
    from types import SimpleNamespace
    import urban_dreamer.learning.training as module
    trainer = make_trainer(tmp_path)
    flow = torch.tensor(trainer.network.extract_flow({})).view(1, 1, trainer.edge_count, 8).repeat(1, frames, 1, 1)
    flow[..., 1] = torch.linspace(.2, 1., frames).view(1, frames, 1)
    original = flow.clone()
    seen = {}
    encode = trainer.world_model.encode
    def capture_encode(value):
        seen['encoder'] = value.detach().clone()
        return encode(value)
    monkeypatch.setattr(trainer.world_model, 'encode', capture_encode)
    for name in ('decoder_reconstruction_loss',):
        original_loss = getattr(module, name)
        def capture(*, _name=name, _loss=original_loss, **kwargs):
            seen[_name] = kwargs
            return _loss(**kwargs)
        monkeypatch.setattr(module, name, capture)
    batch = SimpleNamespace(flow=flow, action=torch.zeros(1, frames-1, trainer.edge_count),
        departure=torch.zeros(1, frames-1, trainer.edge_count), context=torch.zeros(1, frames-1, 2),
        total_travel_minutes=torch.ones(1), episode_complete=torch.tensor([False]),
        active_vehicle_fraction=torch.zeros(1, frames-1))
    loss, readout = trainer._world_loss(batch)
    (loss + readout).backward()
    torch.testing.assert_close(flow, original)
    torch.testing.assert_close(seen['encoder'], original.reshape(-1, trainer.edge_count, 8))
    recon = seen['decoder_reconstruction_loss']
    targets = recon['flow_targets']
    expected_speed = (original[:, :-1, :, 1] + original[:, 1:, :, 1]) / 2
    torch.testing.assert_close(targets[:, :-1, :, 1], expected_speed)
    torch.testing.assert_close(targets[..., [0, 2, 3, 4, 5, 6, 7]], original[..., [0, 2, 3, 4, 5, 6, 7]])
    assert recon['speed_target_mask'][:, :-1].all()
    assert not recon['speed_target_mask'][:, -1].any()
    assert torch.isfinite(loss) and torch.isfinite(readout)
    assert trainer.world_model.flow_projector.decoder_speed.weight.grad.abs().sum() > 0


def _set_deadline(trainer, seconds):
    assert seconds % 30 == 0
    trainer.config = replace(trainer.config, simulation=replace(trainer.config.simulation,
        world_model_intervals_per_episode=seconds // 30, training_completion_buffer_seconds=0))


def test_eight_window_truncation_bootstraps_but_real_deadline_penalizes(tmp_path, monkeypatch):
    import urban_dreamer.learning.training as module
    trainer = make_trainer(tmp_path)
    trainer.config = replace(trainer.config, method=replace(trainer.config.method, disagreement_weight=0.))
    _set_deadline(trainer, 330)
    batch = _in_transit_seed(trainer)
    batch.vehicles[0][0].update(remaining_distance=1e6, committed_next_edge=None, pending_decision=True)
    captured = []
    original = module.lambda_return
    def capture(rewards, next_values, *, terminal):
        targets = original(rewards, next_values, terminal=terminal)
        captured.append((rewards.clone(), next_values.clone(), terminal.clone(), targets.clone()))
        return targets
    monkeypatch.setattr(module, 'lambda_return', capture)
    sampling = []
    original_sample = trainer.actor.sample_action
    def sample(*args, **kwargs):
        sampling.append(kwargs.get('deterministic', False))
        return original_sample(*args, **kwargs)
    monkeypatch.setattr(trainer.actor, 'sample_action', sample)
    rows, loss = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    rewards, next_values, terminal, targets = captured[-1]
    assert len(rewards) == 8 and len(rows) == 1
    assert not terminal.any()
    torch.testing.assert_close(targets[-1], rewards[-1] + next_values[-1])
    assert (rewards >= -.5).all()
    assert torch.isfinite(loss) and sampling == [False]
    assert 'reward' not in rows[0] and 'global_reward' in rows[0]
    assert rows[0]['timeout_return'] == rows[0]['timeout_penalty'] == 0.
    batch.time[0] = 300.
    rows, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    rewards, _, terminal, targets = captured[-1]
    assert len(rewards) == 1 and terminal.item()
    assert -.5 <= rewards.item() <= 0.
    assert rows[0]['timeout_return'] == rows[0]['timeout_penalty'] == -5000.
    torch.testing.assert_close(targets, rewards)
    assert rows[0]['global_reward'] == rewards[0]


def test_arrival_at_deadline_has_no_penalty_and_future_schedule_is_unused(tmp_path, monkeypatch):
    import urban_dreamer.learning.training as module
    trainer = make_trainer(tmp_path)
    _set_deadline(trainer, 30)
    batch = _in_transit_seed(trainer, destination=True)
    state = batch.vehicles[0][0]
    edge = state['current_edge']
    speed = min(float(trainer.network.speed_limits_mps[edge]),
                trainer.config.simulation.vehicle_max_speed_mps)
    state['remaining_distance'] = speed * 30.
    def decode(z):
        result = torch.zeros(*z.shape[:-1], 4, device=z.device)
        result[..., 1] = 1.
        return result
    monkeypatch.setattr(trainer.world_model, 'decode', decode)
    future = dict(state, vehicle_id='future', departed=False, departure_time=1., pending_decision=False)
    batch = replace(batch, vehicles=((state, future),))
    captured = []
    original = module.lambda_return
    def capture(rewards, next_values, *, terminal):
        captured.append((rewards.clone(), terminal.clone()))
        return original(rewards, next_values, terminal=terminal)
    monkeypatch.setattr(module, 'lambda_return', capture)
    rows, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    rewards, terminal = captured[-1]
    assert not rows and terminal.item()
    assert -.5 <= rewards.item() <= 0.


@pytest.mark.parametrize('mode', ['warmup', 'online', 'evaluation'])
def test_real_episode_deadline_records_penalty_and_preserves_nonarrival(tmp_path, mode):
    from types import SimpleNamespace
    trainer = make_trainer(tmp_path)
    _set_deadline(trainer, 60)
    trainer.demand = SimpleNamespace(sample=lambda **kwargs: trainer.topology.vehicles)
    trainer.config = replace(trainer.config, simulation=replace(trainer.config.simulation,
        warmup_horizon_seconds=60, warmup_completion_drain_seconds=0,
        evaluation_horizon_seconds=60))
    seed = _in_transit_seed(trainer).vehicles[0][0]
    future = dict(seed, vehicle_id=trainer.topology.vehicles[1].vehicle_id,
                  departure_time=1000., departed=False, pending_decision=False)
    flow = trainer.network.extract_flow({})
    class Session:
        done = False
        travel_times = {}
        microstep_seconds = 1.
        metrics = {'microsteps': 60}
        closed = False
        time = 0.
        def state(self):
            return {'flow': flow.copy(), 'vehicles': [seed.copy(), future.copy()], 'time': self.time}
        def reset(self):
            return self.state()
        def advance_window(self, callback):
            self.time += 30.
            return SimpleNamespace(next_state=self.state(), active_vehicle_seconds=30.,
                decisions=(), action_counts=np.zeros(trainer.edge_count), departure_counts=np.zeros(trainer.edge_count))
        def close(self):
            self.closed = True
    session = Session()
    trainer._session_factory = lambda **kwargs: session
    collected = trainer._run_episode(mode=mode, episode_index=0)
    assert session.closed and session.time == 60.
    assert collected.metrics[f'{mode}_nonarrival_count'] == 2
    assert collected.metrics[f'{mode}_nonarrival_penalty_sum'] == -10000.
    assert collected.metrics[f'{mode}_vehicle_timeout_penalty_sum'] == -10000.
    assert collected.metrics[f'{mode}_global_reward_sum'] == -.5
    assert collected.metrics[f'{mode}_selection_score'] == -10000.5
    assert not any('local_reward' in key for key in collected.metrics)
    assert not collected.transitions[-1].terminal  # No fabricated arrivals in replay.
    assert all(len(row.vehicles) == 1 and row.vehicles[0]['departed'] for row in collected.transitions)
    assert collected.total_travel_minutes == .5  # Physical travel statistics exclude penalties.


def test_episode_context_uses_full_simulation_budget_including_drain(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.config = replace(trainer.config, simulation=replace(trainer.config.simulation,
        world_model_intervals_per_episode=2, training_completion_buffer_seconds=60))
    assert trainer._episode_end_time() == 120.
    assert trainer._context([], 60.)[0] == .5


def _complete_demand_session(trainer, captured_topologies):
    from types import SimpleNamespace
    def factory(*, topology, **kwargs):
        captured_topologies.append(topology)
        flow = trainer.network.extract_flow({})
        snapshots = []
        for vehicle in topology.vehicles:
            current = trainer.network.edge_to_index[topology.origins[vehicle.origin_index]]
            destination = trainer.network.edge_to_index[topology.destinations[vehicle.destination_index]]
            snapshots.append(dict(vehicle_id=vehicle.vehicle_id, current_edge=current,
                destination_edge=destination, departure_time=vehicle.departure_second,
                departed=True, arrived=False, remaining_distance=1000., committed_next_edge=None,
                pending_decision=True, junction_remaining_distance=0., junction_speed=1.))
        class Session:
            done = not bool(snapshots)
            travel_times = {}
            microstep_seconds = 1.
            metrics = {'microsteps': 0}
            time = 0.
            def state(self):
                return dict(flow=flow.copy(), time=self.time,
                            vehicles=copy.deepcopy(snapshots) if self.time == 0. else [])
            def reset(self):
                return self.state()
            def advance_window(self, callback):
                self.time = 30.
                self.done = True
                self.travel_times = {vehicle.vehicle_id: 30. for vehicle in topology.vehicles}
                counts = np.zeros(trainer.edge_count)
                departures = np.zeros(trainer.edge_count)
                if snapshots:
                    counts[0] = 1.
                    departures[0] = len(snapshots)
                return SimpleNamespace(next_state=self.state(), active_vehicle_seconds=30. * len(snapshots),
                    decisions=(), action_counts=counts, departure_counts=departures)
            def close(self):
                pass
        return Session()
    return factory


def test_real_collection_uses_realized_population_and_keeps_zero_demand(tmp_path):
    from types import SimpleNamespace
    trainer = make_trainer(tmp_path)
    template = trainer.topology.vehicles[0]
    samples = [tuple(replace(template, vehicle_id=str(i), departure_second=0.) for i in range(n))
               for n in (3, 1, 0)]
    calls = []
    def sample(*, seed, episode_index):
        calls.append((seed, episode_index))
        return samples[episode_index]
    trainer.demand = SimpleNamespace(sample=sample)
    captured = []
    trainer._session_factory = _complete_demand_session(trainer, captured)
    for index, n in enumerate((3, 1, 0)):
        collected = trainer._run_episode(mode='online', episode_index=index)
        assert collected.total_vehicle_count == n
        assert collected.metrics['online_vehicle_count'] == n
        assert collected.metrics['online_completion_rate'] == 1.
        assert collected.metrics['online_global_reward_sum'] == (-.5 if n else 0.)
        assert collected.total_travel_minutes == (.5 if n else 0.)
        assert len(collected.transitions) == 1 and collected.transitions[0].terminal
        transition = collected.transitions[0]
        assert transition.context[1] == (1. if n else 0.)
        assert transition.action[0] == pytest.approx(1. / n if n else 0.)
        assert transition.departure[0] == (1. if n else 0.)
        assert transition.active_vehicle_fraction == (1. if n else 0.)
    assert calls == [(trainer.config.run.seed, i) for i in range(3)]
    assert len(trainer.topology.vehicles) == trainer.total_vehicle_count == 2
    assert [len(topology.vehicles) for topology in captured] == [3, 1, 0]


def test_imagination_uses_seed_episode_population_not_reference_or_latest(tmp_path, monkeypatch):
    import urban_dreamer.learning.training as module
    trainer = make_trainer(tmp_path)
    batch = _in_transit_seed(trainer)
    batch = replace(batch, total_vehicle_count=torch.tensor([7]))
    batch.vehicles[0][0].update(pending_decision=True, committed_next_edge=None,
                              remaining_distance=1e6)
    trainer.total_vehicle_count = 99
    contexts, counts, actions = [], [], []
    advance, critic = module.advance_imagined_window, trainer.latent_critic.forward
    def capture_advance(*args, **kwargs):
        counts.append(kwargs['total_vehicle_count'])
        result = advance(*args, **kwargs)
        actions.append(result.action)
        return result
    def capture_critic(latent, context):
        contexts.append(context.detach().cpu().numpy())
        return critic(latent, context)
    monkeypatch.setattr(module, 'advance_imagined_window', capture_advance)
    monkeypatch.setattr(trainer.latent_critic, 'forward', capture_critic)
    trainer._imagine_from_seed(batch, 0, rollout_index=0)
    assert counts and all(count == 7 for count in counts)
    assert contexts and all(context[0, 1] == pytest.approx(1. / 7) for context in contexts)
    assert actions[0].sum() == pytest.approx(1. / 7)


def test_empty_poisson_episode_is_stored_for_world_model_but_not_active_imagination(tmp_path):
    from types import SimpleNamespace
    trainer = make_trainer(tmp_path)
    trainer.demand = SimpleNamespace(sample=lambda **kwargs: ())
    trainer._session_factory = _complete_demand_session(trainer, [])
    result = trainer.collect_warmup_episode(0)
    assert result['complete'] and result['transitions'] == 1
    assert result['warmup_vehicle_count'] == 0
    assert result['warmup_global_reward_sum'] == 0.
    assert len(trainer.warmup_replay.episodes) == 1
    assert trainer.warmup_replay.has_samples
    assert not trainer.warmup_replay.has_active_seeds
    assert trainer.warmup_replay.state_dict()['episodes'][0]['total_vehicle_count'] == 0


def test_active_imagination_rejects_zero_population(tmp_path):
    trainer = make_trainer(tmp_path)
    batch = replace(_in_transit_seed(trainer), total_vehicle_count=torch.tensor([0]))
    with pytest.raises(ValueError, match='positive episode population'):
        trainer._imagine_from_seed(batch, 0, rollout_index=0)


def test_poisson_demand_and_resumed_next_episode_are_independent_of_policy_rng(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer._next_environment_episode = 4
    state = copy.deepcopy(trainer.state_dict())
    sample = lambda: trainer.demand.sample(seed=trainer.config.run.seed,
                                          episode_index=trainer._next_environment_episode)
    expected = sample()
    trainer.rng.random(100)
    np.random.random(100)
    torch.rand(100)
    random.random()
    assert sample() == expected
    trainer._next_environment_episode = 17
    trainer.load_state_dict(state)
    assert trainer._next_environment_episode == 4 and sample() == expected


def test_physical_poisson_collection_resumes_exact_next_realization(tmp_path):
    trainer = make_trainer(tmp_path)
    captured = []
    trainer._session_factory = _complete_demand_session(trainer, captured)
    for episode in range(2):
        expected = trainer.demand.sample(seed=trainer.config.run.seed, episode_index=episode)
        result = trainer._run_episode(mode='online', episode_index=episode)
        assert captured[-1].vehicles == expected
        assert result.total_vehicle_count == len(expected)
    checkpoint = copy.deepcopy(trainer.state_dict())
    original = trainer._run_episode(mode='evaluation', episode_index=0)
    expected_topology = captured[-1]
    trainer.load_state_dict(checkpoint)
    trainer.rng.random(100)
    torch.rand(100)
    resumed = trainer._run_episode(mode='evaluation', episode_index=0)
    assert captured[-1].vehicles == expected_topology.vehicles
    assert resumed.total_vehicle_count == original.total_vehicle_count
    assert trainer._next_environment_episode == 3


def test_initialization_epoch_coverage_accumulation_and_exact_resume(tmp_path, monkeypatch):
    from urban_dreamer.learning.replay import MacroTransition
    trainer = make_trainer(tmp_path)
    trainer.config = replace(trainer.config,
        run=replace(trainer.config.run, world_model_initialization_epochs=2),
        resources=replace(trainer.config.resources,
                          world_model_initialization_batch_size=3,
                          world_model_gradient_accumulation_steps=2))
    flow = trainer.network.extract_flow({})
    trainer.warmup_replay.append_episode([
        MacroTransition(flow=flow, next_flow=flow, action=np.zeros(trainer.edge_count),
                        departure=np.zeros(trainer.edge_count), context=np.zeros(2),
                        active_vehicle_fraction=0., time=30. * i, step_index=i,
                        terminal=i == 6) for i in range(7)],
        total_travel_minutes=0., total_vehicle_count=0)
    visits, sizes = [], []
    def tiny_loss(batch):
        visits.extend(batch.start_index.tolist())
        sizes.append(batch.flow.shape[0])
        # Exercise real optimizers and RNG, with inexpensive differentiable losses.
        p = trainer._world_parameters[0].flatten()[0]
        q = next(trainer.world_model.active_vehicle_fraction_readout.parameters()).flatten()[0]
        target = batch.start_index.float().mean() / 100 + torch.rand(()) / 100
        return (p - target).square(), (q - target).square()
    monkeypatch.setattr(trainer, '_world_loss', tiny_loss)
    first = trainer.initialize_world_model_epoch(0)
    assert first['samples'] == 7 and first['microbatches'] == 3 and first['optimizer_updates'] == 2
    assert sorted(visits) == list(range(7)) and sizes == [3, 3, 1]
    state = copy.deepcopy(trainer.state_dict())
    visits.clear()
    expected = trainer.initialize_world_model_epoch(1)
    expected_order = list(visits)
    expected_weights = [p.detach().clone() for p in trainer._world_parameters]
    trainer.load_state_dict(state)
    visits.clear()
    actual = trainer.initialize_world_model_epoch(1)
    assert actual == expected and visits == expected_order
    assert all(torch.equal(left, right) for left, right in zip(expected_weights, trainer._world_parameters))
    assert trainer.metrics['world_model_initialization_epochs_completed'] == 2
    assert trainer.metrics['world_model_initialization_batches_completed'] == 6
    assert trainer.metrics['world_model_initialization_samples_seen'] == 14
    assert trainer.metrics['world_model_initialization_updates'] == 4
    assert trainer.metrics['world_model_updates'] == 4
    with pytest.raises(ValueError, match='next epoch boundary'):
        trainer.initialize_world_model_epoch(1)


def test_partial_accumulation_weights_samples_not_microbatches(tmp_path, monkeypatch):
    from types import SimpleNamespace
    trainer = make_trainer(tmp_path)
    p = torch.nn.Parameter(torch.tensor(0.))
    readout = torch.nn.Module()
    readout.register_parameter('value', torch.nn.Parameter(torch.tensor(0.)))
    readout.register_buffer('calibration_updates', torch.tensor(0))
    trainer.world_model.active_vehicle_fraction_readout = readout
    trainer._world_parameters = [p]
    trainer.optimizers['world_model_initial'] = torch.optim.SGD([p], lr=.1)
    trainer.optimizers['active_fraction'] = torch.optim.SGD(readout.parameters(), lr=.1)
    def loss(batch):
        target = batch.flow.mean()
        return (p - target).square(), (readout.value - target).square()
    monkeypatch.setattr(trainer, '_world_loss', loss)
    batches = [SimpleNamespace(flow=torch.tensor([1., 1.])),
               SimpleNamespace(flow=torch.tensor([4.]))]
    metrics = trainer._update_world(SimpleNamespace(has_samples=True), initial=True, batches=batches)
    # The true sample mean is 2, so the gradient is -4 and SGD moves to 0.4.
    assert p.item() == pytest.approx(.4)
    assert readout.value.item() == pytest.approx(.4)
    assert metrics['world_model_loss'] == pytest.approx(6.)
