from __future__ import annotations

import copy
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.learning.action_events import pack_action_events
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology

ROOT = Path(__file__).resolve().parents[1]


def fixed_demand(trainer, sample):
    """Change the realization while retaining the collection demand provenance."""
    from types import SimpleNamespace
    return SimpleNamespace(sample=sample, duration_seconds=trainer.demand.duration_seconds,
        expected_vehicle_count=trainer.demand.expected_vehicle_count, od_rates=trainer.demand.od_rates)


def branch_edge(trainer, destination):
    return next(edge for edge in range(trainer.edge_count)
                if edge != destination and len(trainer.candidates.candidates(edge, destination)) > 1)


def make_trainer(tmp_path, *, archive_roots=(), traffic_model=None):
    config = load_config(ROOT / 'configs/toy_cross_bottleneck.toml')
    config = replace(config, run=replace(config.run, output_dir=tmp_path, device='cpu',
        warmup_episodes=1, world_model_initialization_epochs=1,
        online_training_episodes=1, evaluation_episodes=1),
        simulation=replace(config.simulation, world_model_intervals_per_episode=6,
            training_completion_buffer_seconds=0, evaluation_horizon_seconds=180,
            demand_duration_seconds=30, routing_mode='next_hop',
            traffic_model=traffic_model or config.simulation.traffic_model),
        resources=replace(config.resources, world_model_initialization_batch_size=1,
            world_model_initialization_sequence_length=1, online_world_model_batch_size=1,
            imagination_passes_per_episode=16, imagination_time_budget_seconds=0.),
        method=replace(config.method, fm_steps=2, fm_solver='euler'),
        codec=replace(config.codec, decoder_pretrain_enabled=False, decoder_online_enabled=False),
        data=replace(config.data, archive_roots=tuple(archive_roots)))
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    topology = replace(topology, vehicles=topology.vehicles[:2])
    return MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology)


def write_empty_offline_archive(trainer, path, *, windows=1):
    """Persist compatible event data independently of the mainline collector."""
    from types import SimpleNamespace
    from urban_dreamer.learning.episode_archive import EpisodeArchive
    flow = trainer.network.extract_flow({})
    archive = EpisodeArchive(dict(city=trainer.config.inputs.network_name,
        method='fixture_collector', seed=trainer.config.run.seed, episode=0, phase='train',
        total_vehicle_count=0, session_protocol='event', step_seconds=1.,
        realized_demand=[], input_hashes=trainer.prepared_inputs.input_hashes,
        edge_ids=list(trainer.network.edge_ids), config=trainer.config.resolved_snapshot(),
        demand=dict(law='independent_poisson_per_od', duration_seconds=trainer.demand.duration_seconds,
            od_rates=[dict(origin_index=o, destination_index=d, rate_vehicles_per_second=r)
                      for o, d, r in trainer.demand.od_rates])))
    archive.start(dict(flow=flow, time=0., done=False, vehicles=[]))
    for index in range(windows):
        archive.append(SimpleNamespace(start_time=30. * index, end_time=30. * (index + 1),
            next_state=dict(flow=flow, time=30. * (index + 1), done=index == windows-1, vehicles=[]),
            active_vehicle_seconds=0., action_counts=np.zeros(trainer.edge_count),
            departure_counts=np.zeros(trainer.edge_count), decisions=()))
    archive.save(path, metrics={}, travel_times={}, completed=True,
                 deadline_seconds=30. * windows)
    return path


def attach_offline_data(trainer, archive_root):
    from urban_dreamer.learning.offline_data import OfflineEpisodeDataset
    trainer.config = replace(trainer.config, data=replace(trainer.config.data,
        archive_roots=(archive_root,)))
    trainer.offline_data = OfflineEpisodeDataset(config=trainer.config,
        prepared_inputs=trainer.prepared_inputs, topology=trainer.topology,
        sequence_length=trainer.config.resources.world_model_initialization_sequence_length)


def mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch):
    """Exercise the real gate with exact oracle metrics for isolated FM tests.

    This mock validates gate plumbing only; it never claims the randomly
    initialized trainer codec is usable or authorizes a physical lifecycle test.
    """
    from types import SimpleNamespace
    import urban_dreamer.learning.training as module
    from urban_dreamer.learning.codec_training import codec_decoded_metrics
    flow = torch.tensor(trainer.network.extract_flow({})).unsqueeze(0)
    flow[0, 0, 2] = 1. / max(float(trainer.network.capacities[0]), 1.)
    next_flow = flow.clone()
    perfect_decoded = flow[..., :4].clone()
    oracle_metrics = codec_decoded_metrics(perfect_decoded, flow, next_flow, trainer.network)
    validation = SimpleNamespace(has_samples=True, manifest={'identity': 'isolated-oracle-gate'},
                                 release_cache=lambda: None)
    validation.state_dict = lambda: {'manifest_identity': validation.manifest['identity']}
    def restore_validation(state):
        if state != validation.state_dict():
            raise ValueError('isolated validation manifest changed')
    validation.load_state_dict = restore_validation
    # The gate proof must retain its independent source identity through the
    # entire isolated test, including any trainer checkpoint roundtrip.
    monkeypatch.setattr(trainer, 'codec_validation_data', validation)
    monkeypatch.setattr(trainer, '_codec_batches', lambda *args, **kwargs: iter([(flow, next_flow)]))
    monkeypatch.setattr(trainer, '_transition_codec_batches', lambda *args, **kwargs: iter([(flow, next_flow)]), raising=False)
    monkeypatch.setattr(module, 'codec_metrics', lambda *args, **kwargs: oracle_metrics)
    result = trainer._validate_codec()
    assert result['codec_ready'] and trainer.codec_ready()
    trainer.world_model.flow_projector.requires_grad_(False)


def test_trainer_state_contains_macro_value_and_event_q_critics(tmp_path):
    trainer = make_trainer(tmp_path)
    state = trainer.state_dict()
    assert set(state['models']) == {'world_model', 'actor', 'latent_critic', 'event_critic'}
    assert set(state['optimizers']) == {'world_model_initial', 'world_model_online',
                                       'codec', 'active_fraction', 'actor', 'latent_critic', 'event_critic'}
    assert not hasattr(trainer, 'vehicle_critic')
    assert not any('vehicle_critic' in key for key in trainer.metrics)
    assert set(state['replay']) == {'online', 'offline', 'codec_validation'}
    assert not hasattr(trainer, 'warmup_replay')
    assert not hasattr(trainer, 'warmup_data')
    assert not hasattr(trainer, 'collect_warmup_episode')
    assert trainer.microstep_seconds == 1.


def test_codec_ready_flag_alone_cannot_authorize_fm(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.metrics['codec_ready'] = 1.
    assert not trainer.codec_ready()
    with pytest.raises(ValueError, match='codec quality gate'):
        trainer.validate_codec_ready()


def test_codec_gate_requires_every_raw_and_hybrid_metric(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
    valid = dict(trainer.metrics)
    metric_names = ('codec_count_mae_occupied', 'codec_count_mae_all', 'codec_count_mae_empty',
                    'codec_total_relative_error', 'codec_speed_mae_mps')
    for prefix in ('codec_validation_', 'codec_hybrid_validation_'):
        for metric in metric_names:
            key = prefix + metric
            assert key in valid
            trainer.metrics = dict(valid)
            del trainer.metrics[key]
            assert not trainer.codec_ready(), key
    trainer.metrics = valid
    assert trainer.codec_ready()


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf'), -.01])
def test_codec_gate_rejects_nonfinite_or_negative_error_proofs(tmp_path, monkeypatch, invalid):
    trainer = make_trainer(tmp_path)
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
    valid = dict(trainer.metrics)
    metrics = ('codec_count_mae_occupied', 'codec_count_mae_all', 'codec_count_mae_empty',
               'codec_total_relative_error', 'codec_speed_mae_mps')
    for prefix in ('codec_validation_', 'codec_hybrid_validation_'):
        for metric in metrics:
            trainer.metrics = dict(valid)
            trainer.metrics[prefix + metric] = invalid
            assert not trainer.codec_ready(), prefix + metric


def test_codec_gate_rechecks_current_thresholds_for_both_scopes(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
    valid = dict(trainer.metrics)
    cases = (('codec_count_mae_occupied', 'occupied_vehicle_mae_max'),
             ('codec_count_mae_all', 'vehicle_mae_max'),
             ('codec_count_mae_empty', 'empty_vehicle_mae_max'),
             ('codec_total_relative_error', 'total_relative_mae_max'),
             ('codec_speed_mae_mps', 'speed_mae_mps_max'))
    original = trainer.config.codec
    for prefix in ('codec_validation_', 'codec_hybrid_validation_'):
        for metric, threshold in cases:
            trainer.metrics = dict(valid)
            trainer.config = replace(trainer.config, codec=original)
            # A previously passing positive measurement must fail after the
            # configured tolerance tightens, even though ready remains true.
            measured = getattr(original, threshold) * .75
            trainer.metrics[prefix + metric] = measured
            assert trainer.codec_ready()
            trainer.config = replace(trainer.config,
                codec=replace(original, **{threshold: measured / 2}))
            assert not trainer.codec_ready(), prefix + metric


def test_codec_gate_is_bound_to_independent_validation_identity_and_update_version(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
    assert trainer.metrics['codec_validation_updates'] == trainer.metrics['codec_updates']
    source = trainer.codec_validation_data
    identity = source.manifest['identity']
    source.manifest['identity'] = 'different-independent-validation'
    assert not trainer.codec_ready()
    source.manifest['identity'] = identity
    assert trainer.codec_ready()
    monkeypatch.setattr(trainer, 'codec_validation_data', None)
    assert not trainer.codec_ready()
    monkeypatch.setattr(trainer, 'codec_validation_data', source)
    trainer.metrics['codec_updates'] += 1
    assert not trainer.codec_ready()
    trainer._validate_codec()
    assert trainer.codec_ready()
    assert trainer.metrics['codec_validation_updates'] == trainer.metrics['codec_updates']


def test_codec_proof_and_source_identity_survive_isolated_checkpoint_roundtrip(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
    state = copy.deepcopy(trainer.state_dict())
    assert state['replay']['codec_validation'] == {'manifest_identity': 'isolated-oracle-gate'}
    trainer.metrics['codec_updates'] += 1
    assert not trainer.codec_ready()
    trainer.load_state_dict(state)
    assert trainer.codec_ready()
    assert all(not parameter.requires_grad for parameter in trainer.world_model.flow_projector.parameters())


def test_raw_validation_pass_cannot_hide_failed_hybrid_validation(tmp_path, monkeypatch):
    import urban_dreamer.learning.training as module
    trainer = make_trainer(tmp_path)
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
    oracle = module.codec_metrics(trainer.world_model, None, None, trainer.network)
    calls = []
    def differing_scope_metrics(*args, **kwargs):
        calls.append(True)
        result = dict(oracle)
        if len(calls) == 2:
            result['codec_count_mae_occupied'] = torch.tensor(trainer.config.codec.occupied_vehicle_mae_max + 1.)
        return result
    monkeypatch.setattr(module, 'codec_metrics', differing_scope_metrics)
    result = trainer._validate_codec()
    assert len(calls) == 2
    assert not result['codec_ready'] and not trainer.codec_ready()
    assert trainer.metrics['codec_validation_codec_count_mae_occupied'] == 0.
    assert trainer.metrics['codec_hybrid_validation_codec_count_mae_occupied'] > trainer.config.codec.occupied_vehicle_mae_max


def test_raw_archive_transition_codec_source_supplies_observed_following_speed_frames(tmp_path):
    from dataclasses import asdict
    from types import SimpleNamespace
    from urban_dreamer.learning.episode_archive import EpisodeArchive
    from urban_dreamer.learning.codec_training import codec_metrics
    trainer = make_trainer(tmp_path / 'run')
    demands = [replace(demand, departure_second=float(index * 15))
               for index, demand in enumerate(trainer.topology.vehicles)]
    network = trainer.network
    def state(time, speed):
        snapshots = []
        for index, demand in enumerate(demands):
            origin = network.edge_to_index[trainer.topology.origins[demand.origin_index]]
            destination = network.edge_to_index[trainer.topology.destinations[demand.destination_index]]
            departed, arrival = time > 0, float(45 + index * 30)
            arrived = time >= arrival
            row = dict(vehicle_id=demand.vehicle_id, current_edge=destination if arrived else origin,
                destination_edge=destination, departure_time=demand.departure_second,
                actual_departure_time=demand.departure_second if departed else None,
                departed=departed, arrived=arrived, in_junction=False, remaining_distance=10.)
            if arrived:
                row['arrival_time'] = arrival
            snapshots.append(row)
        flow = network.extract_flow({})
        flow[:, 1] = speed
        for row in snapshots:
            if row['departed'] and not row['arrived']:
                flow[row['current_edge'], 2] += 1. / max(float(network.capacities[row['current_edge']]), 1.)
        return dict(time=float(time), flow=flow, done=time == 90, vehicles=snapshots)
    archive = EpisodeArchive(dict(city=trainer.config.inputs.network_name, method='fixture_collector',
        seed=42, episode=0, phase='train', session_protocol='event', step_seconds=1.,
        total_vehicle_count=2, realized_demand=[asdict(demand) for demand in demands],
        input_hashes=trainer.prepared_inputs.input_hashes, edge_ids=list(network.edge_ids),
        config=trainer.config.resolved_snapshot(), demand=dict(law='independent_poisson_per_od',
            duration_seconds=trainer.demand.duration_seconds,
            od_rates=[dict(origin_index=o, destination_index=d, rate_vehicles_per_second=r)
                      for o, d, r in trainer.demand.od_rates])))
    archive.start(state(0, 1.))
    for step, (speed, exposure) in enumerate(zip((.8, .6, 1.), (45., 45., 15.))):
        departures = np.zeros(network.edge_count)
        if step == 0:
            for demand in demands:
                departures[network.edge_to_index[trainer.topology.origins[demand.origin_index]]] += 1
        archive.append(SimpleNamespace(start_time=step * 30., end_time=(step + 1) * 30.,
            next_state=state((step + 1) * 30, speed), active_vehicle_seconds=exposure,
            action_counts=np.zeros(network.edge_count), departure_counts=departures, decisions=()))
    source = tmp_path / 'offline'
    archive.save(source / 'train/episode_000000.npz', metrics={},
        travel_times={demand.vehicle_id: float(45 + index * 30) - demand.departure_second
                      for index, demand in enumerate(demands)}, completed=True, deadline_seconds=90.)
    attach_offline_data(trainer, source)
    flow, next_flow = next(trainer._transition_codec_batches(trainer.offline_data, shuffle=False))
    assert flow.shape == next_flow.shape == (2, trainer.edge_count, 8)
    assert flow[..., 2].ge(0).all() and next_flow[..., 2].ge(0).all()
    torch.testing.assert_close(flow[:, 0, 1], torch.tensor([.8, .6]))
    torch.testing.assert_close(next_flow[:, 0, 1], torch.tensor([.6, 1.]))
    capacity = torch.tensor(network.capacities)
    torch.testing.assert_close((flow[..., 2] * capacity).sum(-1), torch.tensor([0., 1.]))
    torch.testing.assert_close((next_flow[..., 2] * capacity).sum(-1), torch.tensor([1., 0.]))
    # This exercises the real codec on archive-derived tensors; no oracle or
    # readiness flag is involved, and there is no assertion of model usability.
    metrics = codec_metrics(trainer.world_model, flow, next_flow, network)
    assert torch.isfinite(metrics['codec_count_mae_all']) and metrics['codec_count_mae_all'] >= 0
    assert not trainer.codec_ready()


def test_pretraining_requires_offline_data_without_collecting_fallback(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer._session_factory = lambda **kwargs: pytest.fail('missing offline data must never start collection')
    before = [parameter.detach().clone() for parameter in trainer._world_parameters]
    with pytest.raises(ValueError, match='offline pretraining'):
        trainer.validate_pretraining_data()
    with pytest.raises(ValueError, match='offline pretraining|codec'):
        trainer.initialize_world_model_epoch(0)
    assert trainer._next_environment_episode == 0
    assert not trainer.online_replay.episodes
    assert trainer.metrics['world_model_updates'] == 0
    assert all(torch.equal(a, b) for a, b in zip(before, trainer._world_parameters))
    assert not (tmp_path / 'episodes').exists()


def test_checkpoint_restores_cpu_shadow_multinomial_stream(tmp_path):
    trainer = make_trainer(tmp_path)
    features = torch.randn(256, 3, 10)
    # The default policy prior reads normalized nonnegative free-flow costs.
    features[..., 4] = torch.rand(256, 3)
    mask = torch.ones(256, 3, dtype=torch.bool)
    mask[::2, -1] = False
    checkpoint = copy.deepcopy(trainer.state_dict())
    def draw():
        shadow = copy.deepcopy(trainer.actor).cpu().requires_grad_(False)
        with torch.no_grad():
            probabilities = shadow.forward_logits(features, mask).softmax(-1)
            return torch.multinomial(probabilities, 1).squeeze(-1)
    expected = draw()
    expected_rng = torch.get_rng_state().clone()
    torch.rand(100)
    trainer.load_state_dict(checkpoint)
    torch.testing.assert_close(draw(), expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert not (expected[::2] == 2).any()


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
    with pytest.raises(ValueError, match='replay schema'):
        trainer.load_state_dict({**state, 'replay': dict(state['replay'], warmup={})})
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


def test_checkpoint_roundtrip_preserves_multiple_same_vehicle_events(tmp_path):
    from urban_dreamer.learning.action_events import ActionEvent
    from urban_dreamer.learning.replay import MacroTransition
    from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint, save_checkpoint
    trainer = make_trainer(tmp_path)
    snapshot = _in_transit_seed(trainer).vehicles[0][0]
    events = tuple(ActionEvent(snapshot['vehicle_id'], index, timestamp,
        snapshot['current_edge'], snapshot['committed_next_edge'], snapshot['destination_edge'])
        for index, timestamp in enumerate((1., 21.)))
    action = np.zeros(trainer.edge_count)
    action[snapshot['committed_next_edge']] = 1.  # Two events / population two.
    flow = trainer.network.extract_flow({})
    trainer.online_replay.append_episode([MacroTransition(flow=flow, next_flow=flow,
        action=action, departure=np.zeros(trainer.edge_count), context=np.array([0., .5]),
        active_vehicle_fraction=.5, vehicles=(snapshot,), time=0., action_events=events)],
        total_travel_minutes=.5, total_vehicle_count=2)
    save_checkpoint(output_dir=tmp_path, config_fingerprint=trainer.config.semantic_fingerprint,
        input_hash='event-replay', phase=CheckpointPhase.ONLINE_TRAINING, next_unit=0,
        training_state=trainer.state_dict())
    state = load_checkpoint(output_dir=tmp_path,
        expected_config_fingerprint=trainer.config.semantic_fingerprint,
        expected_input_hash='event-replay').training_state
    trainer.load_state_dict(state)
    assert trainer.online_replay.episodes[-1][0].action_events == events
    packed = pack_action_events([trainer.online_replay.episodes[-1]])
    assert packed.vehicle_ids == (snapshot['vehicle_id'],) * 2
    torch.testing.assert_close(packed.phase, torch.tensor([1 / 30, 21 / 30], dtype=torch.float64))


def test_empty_windows_and_future_only_seeds_skip_imagination(tmp_path, monkeypatch):
    import urban_dreamer.learning.training as module
    from urban_dreamer.learning.replay import MacroTransition
    trainer = make_trainer(tmp_path)
    _set_deadline(trainer, 30)
    trainer.config = replace(trainer.config, resources=replace(
        trainer.config.resources, imagination_rollouts_per_pass=1))
    network = trainer.topology.network
    flow = network.extract_flow({})
    demand = trainer.topology.vehicles[0]
    current = network.edge_to_index[trainer.topology.origins[demand.origin_index]]
    destination = network.edge_to_index[trainer.topology.destinations[demand.destination_index]]
    current = branch_edge(trainer, destination)
    snapshot = dict(vehicle_id=demand.vehicle_id, current_edge=current,
        destination_edge=destination, departure_time=1000., departed=False, arrived=False,
        remaining_distance=float(network.lengths_m[current]), committed_next_edge=None,
        pending_decision=False, junction_remaining_distance=0., junction_speed=1.)
    trainer.online_replay.append_episode([MacroTransition(flow=flow,
        action=np.zeros(network.edge_count), departure=np.zeros(network.edge_count),
        context=np.zeros(2), next_flow=flow, active_vehicle_fraction=0.,
        vehicles=(snapshot,), time=0., action_events=())], total_travel_minutes=1., total_vehicle_count=2)
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
    current = branch_edge(trainer, destination)
    snapshot = dict(vehicle_id=demand.vehicle_id, current_edge=current,
        destination_edge=destination, departure_time=0., departed=True, arrived=False,
        remaining_distance=float(network.lengths_m[current]), committed_next_edge=None,
        pending_decision=True, junction_remaining_distance=0., junction_speed=1.)
    flow = network.extract_flow({})
    trainer.online_replay.append_episode([MacroTransition(flow=flow,
        action=np.zeros(network.edge_count), departure=np.zeros(network.edge_count),
        context=np.array([0., .5]), next_flow=flow, active_vehicle_fraction=.5,
        started_count=2, next_started_count=2,
        vehicles=(snapshot,), time=0., action_events=())], total_travel_minutes=1., total_vehicle_count=2)
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


def test_single_empty_offline_trajectory_cannot_bypass_codec_validation_and_start_online(tmp_path):
    from urban_dreamer.workflow.pipeline import MainlinePipeline
    output = tmp_path / 'run'
    archive_root = tmp_path / 'offline'
    template = make_trainer(output)
    write_empty_offline_archive(template, archive_root / 'train/episode_000000.npz')
    config = replace(template.config, data=replace(template.config.data, archive_roots=(archive_root,)))
    def factory():
        return MainlineTrainer(config=config, prepared_inputs=template.prepared_inputs, topology=template.topology,
            _session_factory=lambda **kwargs: pytest.fail('unvalidated codec must never create SUMO'))
    with pytest.raises(ValueError, match='codec.*independent demand'):
        MainlinePipeline(config=config, input_hash='toy-two-vehicle-smoke', trainer_factory=factory).run()
    assert not (output / 'episodes').exists()
    assert not (output / 'checkpoints/latest.pt').exists()


def _in_transit_seed(trainer, *, junction=False, destination=False):
    from urban_dreamer.learning.replay import MacroTransition
    network = trainer.network
    demand = trainer.topology.vehicles[0]
    current = network.edge_to_index[trainer.topology.origins[demand.origin_index]]
    target = network.edge_to_index[trainer.topology.destinations[demand.destination_index]]
    current = branch_edge(trainer, target)
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
        started_count=2, next_started_count=2,
        vehicles=(snapshot,), next_vehicles=(snapshot,), time=0., action_events=())],
        total_travel_minutes=.5, total_vehicle_count=2, arrival_times={})
    return trainer.online_replay.sample_batch(batch_size=1, rng=trainer.rng)


@pytest.mark.parametrize('junction', [False, True])
def test_macro_critic_trains_without_policy_events(tmp_path, monkeypatch, junction):
    import urban_dreamer.learning.training as module
    trainer = make_trainer(tmp_path, traffic_model='micro' if junction else 'meso')
    _set_deadline(trainer, 30)
    trainer.config = replace(trainer.config, resources=replace(
        trainer.config.resources, imagination_rollouts_per_pass=1))
    _in_transit_seed(trainer, junction=junction)
    before = [parameter.detach().clone() for parameter in trainer.latent_critic.parameters()]
    metrics = trainer._train_imagination_pass()
    assert metrics['imagined_events'] == 0
    assert metrics['latent_critic_loss'] > 0
    assert metrics['actor_loss'] == 0
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.latent_critic.parameters()))
    assert not any('vehicle_critic' in key or 'critic_only' in key for key in metrics)


@pytest.mark.parametrize('fm_steps', [2, 100])
def test_policy_pass_updates_event_q_with_saved_detached_baseline(tmp_path, monkeypatch, fm_steps):
    import urban_dreamer.learning.training as module
    trainer = make_trainer(tmp_path)
    trainer.config = replace(trainer.config, method=replace(trainer.config.method, fm_steps=fm_steps))
    _set_deadline(trainer, 30)
    batch = _in_transit_seed(trainer)
    batch.vehicles[0][0].update(pending_decision=True, committed_next_edge=None,
                              remaining_distance=1e6)
    monkeypatch.setattr(trainer.online_replay, 'sample_active_seeds', lambda **kwargs: batch)
    trainer.config = replace(trainer.config, resources=replace(
        trainer.config.resources, imagination_rollouts_per_pass=1))
    original = trainer._imagine_from_seed
    seen = {}
    def capture(*args, **kwargs):
        rows, loss = original(*args, **kwargs)
        seen['rows'] = rows
        seen['baselines'] = torch.stack([row['counterfactual_baseline'].clone() for row in rows])
        return rows, loss
    monkeypatch.setattr(trainer, '_imagine_from_seed', capture)
    before = [parameter.detach().clone() for parameter in trainer.event_critic.parameters()]
    metrics = trainer._train_imagination_pass()
    assert metrics['imagined_events'] == 1
    assert np.isfinite(metrics['event_critic_loss']) and metrics['event_critic_loss'] >= 0
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.event_critic.parameters()))
    rows = seen['rows']
    torch.testing.assert_close(torch.stack([row['counterfactual_baseline'] for row in rows]),
                               seen['baselines'], rtol=0, atol=0)
    for row in rows:
        assert not row['counterfactual_baseline'].requires_grad
        assert not row['global_return'].requires_grad
        torch.testing.assert_close(row['global_advantage'],
                                   row['global_return'] - row['counterfactual_baseline'])
        assert row['timeout_return'] == 0.
        assert row['global_return'] > -5000.
    assert all(parameter.grad is None for parameter in trainer.world_model.parameters())
    assert np.isfinite(metrics['counterfactual_baseline_mean'])
    assert np.isfinite(metrics['actor_advantage_mean'])
    assert trainer.metrics['world_model_imagination_execution_profile']['precision'] == 'fp32'
    assert trainer.metrics['world_model_imagination_execution_profile']['compile'] is False  # CPU fallback.


@pytest.mark.parametrize('timeout_returns,expected_loss', [([0., 0.], -2.5), ([-5000., 0.], -2.5)])
def test_actor_uses_scale_only_global_advantages(tmp_path, monkeypatch, timeout_returns, expected_loss):
    import urban_dreamer.learning.training as module
    trainer = make_trainer(tmp_path)
    _in_transit_seed(trainer)
    trainer.config = replace(trainer.config, resources=replace(
        trainer.config.resources, imagination_rollouts_per_pass=1))
    parameter = trainer.actor.policy_head.bias.reshape(())
    def imagine(*args, **kwargs):
        assert kwargs['batched_events']
        rows = [dict(vehicle_id=['a', 'b'], population=2, log_prob=parameter + torch.tensor([1., 2.]),
                     entropy=parameter.expand(2) * 0,
                     global_advantage=torch.tensor([1., 2.]), timeout_return=torch.tensor(timeout_returns),
                     global_return=torch.tensor([1., 2.]),
                     counterfactual_baseline=torch.zeros(2),
                     executed_q=next(trainer.event_critic.parameters()).flatten()[0].expand(2))]
        return rows, sum(p.square().sum() for p in trainer.latent_critic.parameters())
    monkeypatch.setattr(trainer, '_imagine_from_seed', imagine)
    metrics = trainer._train_imagination_pass()
    # std=.5 is floored at one; positive advantages retain their sign.
    # Legacy per-vehicle shaping rows cannot change the paper global-only objective.
    assert metrics['actor_loss'] == pytest.approx(expected_loss, abs=1e-6)


def test_collection_samples_in_training_and_uses_argmax_in_evaluation(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    snapshot = _in_transit_seed(trainer).vehicles[0][0]
    snapshot['pending_decision'] = True
    modes = []
    original = trainer.actor.sample_action
    def select(*args, **kwargs):
        modes.append((kwargs['deterministic'], kwargs['temperature']))
        return original(*args, **kwargs)
    monkeypatch.setattr(trainer.actor, 'sample_action', select)
    flow = trainer.network.extract_flow({})
    for mode in ('online', 'evaluation'):
        decisions = trainer._select([snapshot], flow, 0., context=np.array([0., .5]), mode=mode)
        assert len(decisions) == 1
    assert modes == [(False, 1.), (True, 1.)]


@pytest.mark.parametrize('frames', [2, 4])
def test_codec_labels_are_same_frame_counts_and_forward_speed_means(tmp_path, monkeypatch, frames):
    import urban_dreamer.learning.codec_training as module
    trainer = make_trainer(tmp_path)
    flow = torch.tensor(trainer.network.extract_flow({})).view(1, 1, trainer.edge_count, 8).repeat(1, frames, 1, 1)
    flow[..., 1] = torch.linspace(.2, 1., frames).view(1, frames, 1)
    original = flow.clone()
    seen = {}
    encode = trainer.world_model.flow_projector.forward
    def capture_encode(value):
        seen['encoder'] = value.detach().clone()
        return encode(value)
    monkeypatch.setattr(trainer.world_model.flow_projector, 'forward', capture_encode)
    smooth_l1 = module.F.smooth_l1_loss
    def capture_speed(input, target, **kwargs):
        seen['speed_target'] = target.detach().clone()
        return smooth_l1(input, target, **kwargs)
    monkeypatch.setattr(module.F, 'smooth_l1_loss', capture_speed)
    next_flow = flow[:, 1:].clone()
    # Deliberately different future volume must never become a codec label.
    next_flow[..., 2] = .9
    loss, metrics = module.codec_objective(trainer.world_model, flow[:, :-1], next_flow, trainer.network)
    loss.backward()
    torch.testing.assert_close(flow, original)
    torch.testing.assert_close(seen['encoder'], original[:, :-1])
    expected_speed = (original[:, :-1, :, 1] + original[:, 1:, :, 1]) / 2
    torch.testing.assert_close(seen['speed_target'], expected_speed)
    assert metrics['codec_target_count_total'] == 0.
    assert torch.isfinite(loss)
    assert trainer.world_model.flow_projector.decoder_speed.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in trainer.world_model.dynamics.parameters())


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
    original_logits = trainer.actor.forward_logits
    def sample(*args, **kwargs):
        sampling.append(torch.is_grad_enabled())
        return original_logits(*args, **kwargs)
    monkeypatch.setattr(trainer.actor, 'forward_logits', sample)
    rows, loss = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    rewards, next_values, terminal, targets = captured[-1]
    assert len(rewards) == 8 and len(rows) == 1
    assert not terminal.any()
    torch.testing.assert_close(targets[-1], rewards[-1] + next_values[-1])
    assert (rewards >= -.5).all()
    # Sampling is graph-free; the fixed actor is recomputed once for the
    # window's differentiable log probabilities before its eventual update.
    assert torch.isfinite(loss) and sampling == [False, True]
    assert 'reward' not in rows[0] and 'global_reward' in rows[0]
    assert rows[0]['timeout_return'] == rows[0]['timeout_penalty'] == 0.
    batch.time[0] = 300.
    rows, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    rewards, _, terminal, targets = captured[-1]
    assert len(rewards) == 1 and terminal.item()
    assert -.5 <= rewards.item() <= 0.
    assert rows[0]['timeout_return'] == rows[0]['timeout_penalty'] == 0.
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


@pytest.mark.parametrize('mode', ['online', 'evaluation'])
def test_real_episode_deadline_records_penalty_and_preserves_nonarrival(tmp_path, mode):
    from types import SimpleNamespace
    trainer = make_trainer(tmp_path)
    _set_deadline(trainer, 60)
    trainer.demand = fixed_demand(trainer, lambda **kwargs: trainer.topology.vehicles)
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
            return {'flow': flow.copy(), 'vehicles': [seed.copy(), future.copy()],
                    'time': self.time, 'done': self.done}
        def reset(self):
            return self.state()
        def advance_window(self, callback):
            start = self.time
            self.time += 30.
            return SimpleNamespace(next_state=self.state(), active_vehicle_seconds=30.,
                start_time=start, end_time=self.time,
                decisions=(), action_counts=np.zeros(trainer.edge_count), departure_counts=np.zeros(trainer.edge_count))
        def close(self):
            self.closed = True
    session = Session()
    trainer._session_factory = lambda **kwargs: session
    collected = trainer._run_episode(mode=mode, episode_index=0)
    assert session.closed and session.time == 60.
    assert collected.metrics[f'{mode}_nonarrival_count'] == 2
    assert collected.metrics[f'{mode}_nonarrival_penalty_sum'] == 0.
    assert collected.metrics[f'{mode}_vehicle_timeout_penalty_sum'] == 0.
    assert collected.metrics[f'{mode}_global_reward_sum'] == -1.
    assert collected.metrics[f'{mode}_selection_score'] == -1.
    assert not any('local_reward' in key for key in collected.metrics)
    assert collected.transitions[-1].terminal  # Fixed-horizon terminal bookkeeping.
    assert all(len(row.vehicles) == 1 and row.vehicles[0]['departed'] for row in collected.transitions)
    assert collected.total_travel_minutes == 1.  # One actual trip start; no terminal penalty.


def test_episode_context_uses_full_simulation_budget_including_drain(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.config = replace(trainer.config, simulation=replace(trainer.config.simulation,
        world_model_intervals_per_episode=2, training_completion_buffer_seconds=60))
    assert trainer._episode_end_time() == 120.
    assert trainer._context([], 60.)[0] == .5


def _complete_demand_session(trainer, captured_topologies):
    from types import SimpleNamespace
    from urban_dreamer.traffic.environment import NextHopDecision
    def factory(*, topology, **kwargs):
        captured_topologies.append(topology)
        flow = trainer.network.extract_flow({})
        snapshots = []
        for vehicle in topology.vehicles:
            current = trainer.network.edge_to_index[topology.origins[vehicle.origin_index]]
            destination = trainer.network.edge_to_index[topology.destinations[vehicle.destination_index]]
            current = branch_edge(trainer, destination)
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
                vehicles = copy.deepcopy(snapshots)
                if self.time:
                    for row in vehicles:
                        row.update(arrived=True, current_edge=row['destination_edge'],
                                   pending_decision=False, committed_next_edge=None, remaining_distance=0.)
                return dict(flow=flow.copy(), time=self.time, done=self.done, vehicles=vehicles)
            def reset(self):
                return self.state()
            def advance_window(self, callback):
                self.time = 30.
                self.done = True
                self.travel_times = {vehicle.vehicle_id: 30. for vehicle in topology.vehicles}
                counts = np.zeros(trainer.edge_count)
                departures = np.zeros(trainer.edge_count)
                decisions = ()
                if snapshots:
                    row = snapshots[0]
                    edge = trainer.candidates.candidates(row['current_edge'], row['destination_edge'])[0]
                    counts[edge] = 1.
                    decisions = (NextHopDecision(row['vehicle_id'], row['current_edge'], edge, 0.),)
                    for state in snapshots:
                        departures[state['current_edge']] += 1.
                return SimpleNamespace(next_state=self.state(), active_vehicle_seconds=30. * len(snapshots),
                    start_time=0., end_time=30., decisions=decisions,
                    action_counts=counts, departure_counts=departures)
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
    trainer.demand = fixed_demand(trainer, sample)
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
        assert transition.action.sum() == pytest.approx(1. / n if n else 0.)
        assert transition.departure.sum() == (1. if n else 0.)
        assert len(transition.action_events) == bool(n)
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
    assert contexts and all(context[0, 1] == pytest.approx(1. / 2) for context in contexts)
    assert actions[0].sum() == pytest.approx(1. / 7)


def test_empty_poisson_episode_is_stored_for_world_model_but_not_active_imagination(tmp_path):
    from types import SimpleNamespace
    trainer = make_trainer(tmp_path)
    trainer.demand = fixed_demand(trainer, lambda **kwargs: ())
    trainer._session_factory = _complete_demand_session(trainer, [])
    collected = trainer._run_episode(mode='online', episode_index=0)
    trainer.online_replay.append_episode(collected.transitions,
        total_travel_minutes=collected.total_travel_minutes, total_vehicle_count=collected.total_vehicle_count)
    assert len(collected.transitions) == 1
    assert collected.metrics['online_vehicle_count'] == 0
    assert collected.metrics['online_global_reward_sum'] == 0.
    assert len(trainer.online_replay.episodes) == 1
    assert trainer.online_replay.has_samples
    assert not trainer.online_replay.has_active_seeds
    assert trainer.online_replay.state_dict()['episodes'][0]['total_vehicle_count'] == 0
    assert (tmp_path / 'episodes/online/episode_000000.npz').is_file()


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
    trainer = make_trainer(tmp_path)
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
    trainer.config = replace(trainer.config,
        run=replace(trainer.config.run, world_model_initialization_epochs=2),
        resources=replace(trainer.config.resources,
                          world_model_initialization_batch_size=3,
                          world_model_gradient_accumulation_steps=2))
    archive_root = tmp_path / 'offline'
    write_empty_offline_archive(trainer, archive_root / 'train/episode_000000.npz', windows=7)
    attach_offline_data(trainer, archive_root)
    trainer._session_factory = lambda **kwargs: pytest.fail('offline pretraining must not collect episodes')
    assert not trainer.online_replay.episodes
    policy_models = {name: copy.deepcopy(getattr(trainer, name).state_dict())
                     for name in ('actor', 'latent_critic', 'event_critic')}
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
    assert state['replay']['offline']['manifest_identity'] == trainer.offline_data.manifest['identity']
    assert 'episodes' not in state['replay']['offline']
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
    assert trainer._next_environment_episode == 0 and not trainer.online_replay.episodes
    for name, before in policy_models.items():
        assert all(torch.equal(value, getattr(trainer, name).state_dict()[key])
                   for key, value in before.items())
    with pytest.raises(ValueError, match='next epoch boundary'):
        trainer.initialize_world_model_epoch(1)


def test_partial_accumulation_weights_samples_not_microbatches(tmp_path, monkeypatch):
    from types import SimpleNamespace
    trainer = make_trainer(tmp_path)
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
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


def test_actual_fm_mid_epoch_resume_preserves_remaining_starts_weights_optimizer_and_rng(tmp_path, monkeypatch):
    source = tmp_path / 'offline'
    template = make_trainer(tmp_path / 'template')
    write_empty_offline_archive(template, source / 'train/episode_000000.npz', windows=7)
    visits = {}
    def setup(name):
        trainer = make_trainer(tmp_path / name)
        trainer.config = replace(trainer.config, resources=replace(trainer.config.resources,
            world_model_initialization_batch_size=3, world_model_gradient_accumulation_steps=2))
        attach_offline_data(trainer, source)
        mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
        visits[name] = []
        actual_loss = trainer._world_loss
        def observed_loss(batch):
            visits[name].extend(batch.start_index.tolist())
            return actual_loss(batch)
        monkeypatch.setattr(trainer, '_world_loss', observed_loss)
        return trainer
    reference = setup('reference')
    expected_metrics = reference.initialize_world_model_epoch(0)
    expected_state = copy.deepcopy(reference.state_dict())
    interrupted = setup('interrupted')
    saved = {}
    def interrupt_after_committed_group(epoch):
        assert epoch == 0
        saved['state'] = copy.deepcopy(interrupted.state_dict())
        raise RuntimeError('interrupt after completed optimizer group')
    monkeypatch.setattr(interrupted, '_pretrain_progress_hook', interrupt_after_committed_group, raising=False)
    with pytest.raises(RuntimeError, match='completed optimizer group'):
        interrupted.initialize_world_model_epoch(0)
    assert saved['state']['metrics']['world_model_initialization_samples_seen'] == 6
    assert saved['state']['metrics']['world_model_epoch_samples'] == 6
    assert saved['state']['metrics']['world_model_initialization_updates'] == 1
    assert saved['state']['replay']['offline']['start_cursor'] == 6
    assert saved['state']['metrics']['world_model_initialization_epochs_completed'] == 0
    resumed = setup('resumed')
    resumed.load_state_dict(saved['state'])
    actual_metrics = resumed.initialize_world_model_epoch(0)
    assert actual_metrics == expected_metrics
    assert visits['interrupted'] + visits['resumed'] == visits['reference']
    assert len(visits['resumed']) == 1 and sorted(visits['reference']) == list(range(7))
    def exact(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0., atol=0.)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                exact(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                exact(a, b)
        else:
            assert left == right
    actual_state = resumed.state_dict()
    for section in ('models', 'optimizers', 'rng', 'metrics'):
        exact(expected_state[section], actual_state[section])
    assert actual_state['metrics']['world_model_initialization_samples_seen'] == 7
    assert actual_state['metrics']['world_model_initialization_batches_completed'] == 3
    assert actual_state['metrics']['world_model_initialization_updates'] == 2
    assert all(actual_state['metrics'][key] == 0 for key in ('world_model_epoch_samples', 'world_model_epoch_batches',
        'world_model_epoch_updates', 'world_model_epoch_loss_sum', 'world_model_epoch_readout_loss_sum'))
