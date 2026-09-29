from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from urban_dreamer.learning.training import MainlineTrainer, _close_event
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology

ROOT = Path(__file__).resolve().parents[1]


def make_trainer(tmp_path):
    config = load_config(ROOT / 'configs/toy_cross_bottleneck.toml')
    config = replace(config, run=replace(config.run, output_dir=tmp_path, device='cpu',
        warmup_episodes=1, world_model_initialization_steps=1,
        online_training_episodes=1, evaluation_episodes=1),
        simulation=replace(config.simulation, world_model_intervals_per_episode=6,
            training_completion_buffer_seconds=0, evaluation_horizon_seconds=180),
        resources=replace(config.resources, world_model_initialization_batch_size=1,
            world_model_initialization_sequence_length=1, online_world_model_batch_size=1))
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    topology = replace(topology, vehicles=topology.vehicles[:2])
    return MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology)


def test_event_rewards_keep_destination_elapsed_and_truncation_bootstrap():
    row = {'time': 12.5}
    next_obs = torch.tensor([1., 2.])
    _close_event(row, end_time=77.5, next_obs=next_obs, terminal=True)
    assert row['reward'] == pytest.approx(-65 / 30)
    assert row['terminal'] is True
    row = {'time': 60.}
    _close_event(row, end_time=90., next_obs=next_obs, terminal=False)
    assert row['reward'] == -1.
    assert not row['terminal']
    assert torch.equal(row['next_obs'], next_obs)


def test_checkpoint_roundtrip_restores_all_rng_and_rejects_old(tmp_path):
    trainer = make_trainer(tmp_path)
    state = copy.deepcopy(trainer.state_dict())
    expected = (trainer.rng.random(), torch.rand(3), np.random.random())
    trainer.load_state_dict(state)
    assert trainer.rng.random() == expected[0]
    assert torch.equal(torch.rand(3), expected[1])
    assert np.random.random() == expected[2]
    with pytest.raises(ValueError, match='schema'):
        trainer.load_state_dict({**state, 'legacy': {}})
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
        vehicles=(snapshot,), time=0.)], total_travel_minutes=1.)
    before = [parameter.detach().clone() for parameter in trainer.latent_critic.parameters()]
    metrics = trainer._train_imagination_pass()
    assert metrics['imagined_events'] == 0
    assert metrics['latent_critic_loss'] == 0
    assert all(torch.equal(a,b) for a,b in zip(before, trainer.latent_critic.parameters()))


def test_readout_changes_macro_reward_and_event_advantage(tmp_path, monkeypatch):
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
        vehicles=(snapshot,), time=0.)], total_travel_minutes=1.)
    batch = trainer.online_replay.sample_batch(batch_size=1, rng=trainer.rng)
    torch.manual_seed(10)
    rows_a, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    with torch.no_grad():
        trainer.world_model.active_vehicle_fraction_readout.global_logit_bias.add_(4.)
    torch.manual_seed(10)
    rows_b, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    assert rows_a and rows_b
    assert rows_b[0]['global_reward'] < rows_a[0]['global_reward']
    assert rows_b[0]['global_advantage'] < rows_a[0]['global_advantage']


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
    result = MainlinePipeline(config=resumed.config, input_hash='toy-two-vehicle-smoke',
                             trainer_factory=lambda: resumed).run()
    assert result['status'] == 'completed'
    assert result['warmup_episodes_completed'] == 1
    assert result['world_model_initialization_steps_completed'] == 1
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
        vehicles=(snapshot,), time=0.)], total_travel_minutes=.5)
    return trainer.online_replay.sample_batch(batch_size=1, rng=trainer.rng)


@pytest.mark.parametrize('junction,columns', [(False, [6]), (True, [6, 7, 8])])
def test_in_transit_critic_features_train_without_policy_events(tmp_path, monkeypatch, junction, columns):
    import urban_dreamer.learning.training as module
    trainer = make_trainer(tmp_path)
    _set_deadline(trainer, 30)
    monkeypatch.setattr(module, 'IMAGINED_ROLLOUTS_PER_PASS', 1)
    _in_transit_seed(trainer, junction=junction)
    before = trainer.vehicle_critic.encoder[0].weight.detach().clone()
    metrics = trainer._train_imagination_pass()
    assert metrics['imagined_events'] == 0
    assert metrics['critic_only_intervals'] == 1
    assert metrics['vehicle_critic_loss'] > 0
    for column in columns:
        assert not torch.equal(before[:, column], trainer.vehicle_critic.encoder[0].weight[:, column])


def test_destination_critic_anchor_ends_at_actual_arrival(tmp_path, monkeypatch):
    import urban_dreamer.learning.training as module
    trainer = make_trainer(tmp_path)
    _set_deadline(trainer, 30)
    batch = _in_transit_seed(trainer, destination=True)
    with torch.no_grad():
        decoded = trainer.world_model.decode(trainer.world_model.encode(batch.flow[:, 0]))[0]
    edge = batch.vehicles[0][0]['current_edge']
    speed = min(float(decoded[edge, 1].clamp(.05, 1.5)) * trainer.network.speed_limits_mps[edge],
                trainer.config.simulation.vehicle_max_speed_mps)
    anchors = []
    events, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0, critic_rows=anchors)
    assert events == []
    assert len(anchors) == 1
    assert anchors[0]['terminal']
    assert anchors[0]['reward'] == pytest.approx(-10. / speed / 30., rel=1e-5)
    assert torch.count_nonzero(anchors[0]['next_obs']) == 0


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
    for name in ('decoder_reconstruction_loss', 'transition_decoded_supervision_loss'):
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
    transition = seen['transition_decoded_supervision_loss']
    torch.testing.assert_close(transition['flow_target_next'], targets[:, 1:])
    torch.testing.assert_close(transition['speed_target_mask'], recon['speed_target_mask'][:, 1:])
    assert torch.isfinite(loss) and torch.isfinite(readout)
    assert trainer.world_model.flow_projector.decoder_speed.weight.grad.abs().sum() > 0


def _set_deadline(trainer, seconds):
    assert seconds % 30 == 0
    trainer.config = replace(trainer.config, simulation=replace(trainer.config.simulation,
        world_model_intervals_per_episode=seconds // 30, training_completion_buffer_seconds=0))


def test_timeout_reward_is_additive_unscaled_and_terminal():
    row = {'time': 0.}
    _close_event(row, end_time=60., next_obs=torch.zeros(14), terminal=True, timed_out=True)
    assert row['reward'] == -5002.
    assert row['terminal'] and row['timed_out']
    # Closing the same interval is idempotent; no repeated -5000 accumulation.
    _close_event(row, end_time=60., next_obs=torch.zeros(14), terminal=True, timed_out=True)
    assert row['reward'] == -5002.


def test_eight_window_truncation_bootstraps_but_episode_deadline_penalizes(tmp_path, monkeypatch):
    import urban_dreamer.learning.training as module
    trainer = make_trainer(tmp_path)
    _set_deadline(trainer, 330)
    batch = _in_transit_seed(trainer)
    batch.vehicles[0][0]['remaining_distance'] = 1e6
    batch.vehicles[0][0]['committed_next_edge'] = None
    batch.vehicles[0][0]['pending_decision'] = True
    predict = module.cfm_predict_one_timestep
    calls = []
    def count(*args, **kwargs):
        calls.append(kwargs)
        return predict(*args, **kwargs)
    monkeypatch.setattr(module, 'cfm_predict_one_timestep', count)
    anchors = []
    rows, loss = trainer._imagine_from_seed(batch, 0, rollout_index=0, critic_rows=anchors)
    assert len(calls) == 8
    assert len(rows) == 1
    assert rows[0]['reward'] == -8.
    assert not rows[0]['terminal'] and not rows[0]['timed_out']
    assert sum(a['timed_out'] for a in anchors) == 0
    assert anchors[-1]['reward'] == -1.
    assert torch.isfinite(loss)
    assert all('depart' not in call for call in calls)
    # A later seed gets only the remaining absolute episode time, not a new horizon.
    batch.time[0] = 300.
    calls.clear()
    rows, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    assert len(calls) == 1
    assert rows[0]['reward'] == -5001.


def test_arrival_at_deadline_is_not_penalized_and_future_schedule_is_unused(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    _set_deadline(trainer, 30)
    batch = _in_transit_seed(trainer, destination=True)
    state = batch.vehicles[0][0]
    edge = state['current_edge']
    speed = float(trainer.network.speed_limits_mps[edge])
    state['remaining_distance'] = float(speed * 30.)
    def decode(z):
        result = torch.zeros(*z.shape[:-1], 4, device=z.device)
        result[..., 1] = 1.
        return result
    monkeypatch.setattr(trainer.world_model, 'decode', decode)
    future = dict(state, vehicle_id='future', departed=False, departure_time=1., pending_decision=False)
    from dataclasses import replace as replace_batch
    batch = replace_batch(batch, vehicles=((state, future),))
    anchors = []
    rows, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0, critic_rows=anchors)
    assert not rows
    assert len(anchors) == 1
    assert anchors[0]['terminal'] and not anchors[0]['timed_out']
    assert anchors[0]['reward'] == pytest.approx(-1.)


@pytest.mark.parametrize('mode', ['warmup', 'online', 'evaluation'])
def test_real_episode_deadline_records_penalty_and_preserves_nonarrival(tmp_path, mode):
    from types import SimpleNamespace
    trainer = make_trainer(tmp_path)
    _set_deadline(trainer, 60)
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
    assert collected.metrics[f'{mode}_local_reward_sum'] == -10002.
    assert not collected.transitions[-1].terminal  # No fabricated arrivals in replay.
    assert all(len(row.vehicles) == 1 and row.vehicles[0]['departed'] for row in collected.transitions)
    assert collected.total_travel_minutes == .5  # Physical travel statistics exclude penalties.


def test_episode_context_uses_full_simulation_budget_including_drain(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.config = replace(trainer.config, simulation=replace(trainer.config.simulation,
        world_model_intervals_per_episode=2, training_completion_buffer_seconds=60))
    assert trainer._episode_end_time() == 120.
    assert trainer._context([], 60.)[0] == .5
