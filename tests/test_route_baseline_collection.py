"""Baselines collect full routes on the common physical 30-second policy clock."""
from dataclasses import replace

import numpy as np
import pytest

from test_meso_baselines import arguments, meso_config, tool
from urban_dreamer.learning.baselines.common import EpisodeRunner, resolve_collection_settings
from urban_dreamer.learning.episode_archive import read_episode_archive, to_replay_episode
from urban_dreamer.learning.observations import build_actor_observations
from urban_dreamer.traffic.demand import PoissonDemand
from urban_dreamer.traffic.route_candidates import RouteCandidates
from urban_dreamer.traffic.topology import NextHopCandidates
from urban_dreamer.workflow.config import load_config


@pytest.fixture
def route_config(meso_config, tmp_path):
    path = tmp_path / 'route.toml'
    path.write_text(meso_config.read_text().replace('routing_mode = "next_hop"',
                    'routing_mode = "current_to_dest"'))
    return path


def fixed_demand(runner, monkeypatch):
    vehicles = runner.topology.vehicles[:4]
    monkeypatch.setattr(PoissonDemand, 'sample', lambda self, **kwargs: vehicles)
    runner.deadline = 600.


def test_candidate_factory_and_physical_provenance_keep_next_hop_default(meso_config, route_config, tmp_path):
    default = EpisodeRunner(arguments(meso_config, tmp_path))
    route = EpisodeRunner(arguments(route_config, tmp_path))
    assert type(default.candidates) is NextHopCandidates
    assert isinstance(route.candidates, RouteCandidates)
    assert 'routing_mode' not in default.simulation_settings
    assert route.simulation_settings['routing_mode'] == 'current_to_dest'
    assert route.simulation_settings['policy_decision_interval_seconds'] == 30.
    for key, value in default.simulation_settings.items():
        assert route.simulation_settings[key] == value


def test_route_microstep_rejected_before_collecting(route_config, tmp_path):
    config = load_config(route_config)
    config = replace(config, simulation=replace(config.simulation, traffic_model='micro'))
    args = arguments(route_config, tmp_path, session='microstep')
    with pytest.raises(ValueError, match='current_to_dest routing requires event'):
        resolve_collection_settings(config, args)


@pytest.mark.parametrize('policy', ['random', 'shortest', 'dynamic_shortest'])
def test_real_meso_baseline_collects_route_actions_and_observed_turn_replay(
        route_config, tmp_path, monkeypatch, policy):
    pytest.importorskip('libsumo')
    runner = EpisodeRunner(arguments(route_config, tmp_path, vehicle_limit=4))
    fixed_demand(runner, monkeypatch)
    path = tmp_path / f'{policy}.npz'
    rows, windows, metrics = runner.run_episode(policy, seed=10042, collect=True, record_windows=True,
        archive_path=path, archive_metadata={'episode': 0, 'phase': 'test'})
    data = read_episode_archive(path)
    assert rows and metrics['completion_rate'] == 1.
    assert metrics['p95_travel_seconds'] >= metrics['mean_travel_seconds']
    assert metrics['policy_decision_interval_seconds'] == 30.
    assert metrics['event_count'] == len(rows) == len(data['policy_decisions'])
    assert all(row['time'] % 30 == 0 and row['time'] >= 30 for row in rows)
    for row, decision in zip(rows, data['policy_decisions'], strict=True):
        route = tuple(decision['route'])
        assert row['candidate_routes'][row['action']] == route
        assert route[0] == decision['current_edge'] and route[-1] == decision['destination_edge']
        assert decision['next_edge'] == route[1]
        assert all(target in runner.network.adjacency_list[source] for source, target in zip(route, route[1:]))
    turns = [event for window in windows for event in window['action_events']]
    assert len(turns) == len(data['decisions']) == data['action_counts'].sum()
    assert any(event.decision_time % 30 != 0 for event in turns)
    for window in windows:
        counts = np.zeros(runner.network.edge_count)
        for event in window['action_events']:
            counts[event.next_edge] += 1
        np.testing.assert_allclose(window['action'] * runner.n, counts)
        for vehicle in window['vehicles']:
            assert vehicle['committed_route'][0] == vehicle['current_edge']
            assert vehicle['committed_route'][-1] == vehicle['destination_edge']
    replay = to_replay_episode(path, expected_session_protocol='event')
    assert sum(len(transition.action_events) for transition in replay['transitions']) == len(turns)


def test_shortest_compares_complete_routes_that_share_first_hop(route_config, tmp_path):
    runner = EpisodeRunner(arguments(route_config, tmp_path))
    vehicle = runner.topology.vehicles[0]
    source = runner.network.edge_to_index[runner.topology.origins[vehicle.origin_index]]
    destination = runner.network.edge_to_index[runner.topology.destinations[vehicle.destination_index]]
    row = {'vehicle_id': 'test', 'current_edge': source, 'destination_edge': destination,
           'remaining_distance': runner.network.lengths_m[source]}
    flow = runner.network.extract_flow({})
    observations = build_actor_observations(policy_edge_features=flow[:, [2, 1]], vehicle_states=[row],
        candidates=runner.candidates, episode_progress=0., active_vehicle_fraction=1.)
    paths = observations.routes[0]
    assert paths[0] != paths[1] and paths[0][1] == paths[1][1]
    # Put the more expensive suffix first: a scorer using only first-hop cost
    # would tie and incorrectly select column zero.
    observations = replace(observations, routes=((paths[1], paths[0], paths[2], paths[3]),))
    indices, _ = runner.make_shortest_select()([row], observations, flow, 30.)
    expected = min(range(4), key=lambda i: runner.network.free_flow_travel_seconds[
        list(observations.routes[0][i])].sum())
    assert indices == [expected] and expected != 0


def test_dynamic_shortest_uses_observed_full_route_costs(route_config, tmp_path):
    runner = EpisodeRunner(arguments(route_config, tmp_path))
    vehicle = runner.topology.vehicles[0]
    source = runner.network.edge_to_index[runner.topology.origins[vehicle.origin_index]]
    destination = runner.network.edge_to_index[runner.topology.destinations[vehicle.destination_index]]
    row = {'vehicle_id': 'test', 'current_edge': source, 'destination_edge': destination,
           'remaining_distance': runner.network.lengths_m[source]}
    flow = runner.network.extract_flow({})
    flow[runner.network.edge_to_index['X'], 1] = .01
    flow[runner.network.edge_to_index['X'], 2] = .2
    observations = build_actor_observations(policy_edge_features=flow[:, [2, 1]], vehicle_states=[row],
        candidates=runner.candidates, episode_progress=0., active_vehicle_fraction=1., prior_cost_mode='dynamic',
        prior_speed_floor_mps=runner.config.method.sp_prior_dynamic_speed_floor_mps,
        prior_speed_cap_mps=runner.config.simulation.vehicle_max_speed_mps)
    indices, _ = runner.make_dynamic_shortest_select()([row], observations, flow, 30.)
    assert indices == observations.prior_costs_seconds.argmin(axis=1).tolist()
    assert indices != [0], 'downstream congestion should change the chosen complete suffix'


def test_online_ac_routes_real_collection_learning_and_resume_provenance(route_config, tmp_path, monkeypatch):
    pytest.importorskip('libsumo')
    implementation = tool('train_toy_actor_critic')
    args = arguments(route_config, tmp_path, vehicle_limit=4, global_reward_only=True,
                     reward_objective='observed_mean', episodes=1000)
    runner = implementation.RealActorCritic(args)
    assert isinstance(runner.candidates, RouteCandidates)
    fixed_demand(runner, monkeypatch)
    archive_path = tmp_path / 'online_ac.npz'
    rows, metrics = runner.run_episode(policy='stochastic', seed=10042, collect=True,
        archive_path=archive_path, archive_metadata={'episode': 0, 'phase': 'test'})
    assert rows and metrics['completion_rate'] == 1.
    assert all(row['time'] % 30 == 0 and 'candidate_routes' in row for row in rows)
    actor_before = [parameter.detach().clone() for parameter in runner.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in runner.global_critic.parameters()]
    learned = runner.update(rows)
    assert np.isfinite(learned['actor_loss']) and np.isfinite(learned['global_critic_loss'])
    assert any(not np.array_equal(before.numpy(), after.detach().numpy())
               for before, after in zip(actor_before, runner.actor.parameters(), strict=True))
    assert any(not np.array_equal(before.numpy(), after.detach().numpy())
               for before, after in zip(critic_before, runner.global_critic.parameters(), strict=True))
    data = read_episode_archive(archive_path)
    assert data['metadata']['config']['simulation']['routing_mode'] == 'current_to_dest'
    assert data['metadata']['routing_mode'] == 'current_to_dest'
    assert data['metadata']['policy_decision_interval_seconds'] == 30.
    assert data['metadata']['config_fingerprint'] == runner.config.semantic_fingerprint
    checkpoint = tmp_path / 'route_ac.pt'
    runner.save(checkpoint, 1, 0.)
    restored = implementation.RealActorCritic(arguments(route_config, tmp_path, vehicle_limit=4,
        global_reward_only=True, reward_objective='observed_mean', episodes=1000))
    assert restored.load(checkpoint) == (2, 0.)


def test_evaluation_metrics_use_all_demand_and_realized_turns(route_config, tmp_path, monkeypatch):
    pytest.importorskip('libsumo')
    runner = EpisodeRunner(arguments(route_config, tmp_path, vehicle_limit=4))
    fixed_demand(runner, monkeypatch)
    path = tmp_path / 'evaluation_metrics.npz'
    _, _, metrics = runner.run_episode('shortest', seed=10042, archive_path=path,
        archive_metadata={'episode': 0, 'phase': 'evaluation'})
    data = read_episode_archive(path)
    scheduled = {str(v.vehicle_id): v.departure_second for v in runner.episode_topology.vehicles}
    durations = [min(v['arrival_time'], data['states'][-1]['time']) - scheduled[str(v['vehicle_id'])]
                 for v in data['states'][-1]['vehicles']]
    assert metrics['all_demand_vehicle_count'] == 4
    assert metrics['all_demand_censored_att_seconds'] == pytest.approx(np.mean(durations))
    assert metrics['all_demand_censored_p95_seconds'] == pytest.approx(np.percentile(durations, 95))
    assert metrics['diversion_decision_count'] == len(data['decisions'])
    assert metrics['branch_freeflow_excess_seconds_per_started_vehicle'] == pytest.approx(0., abs=1e-6)
    # VehicleDemand.destination_index indexes OD destinations, not network edge IDs.
    assert any(v.destination_index != runner.network.edge_to_index[runner.topology.destinations[v.destination_index]]
               for v in runner.episode_topology.vehicles)
