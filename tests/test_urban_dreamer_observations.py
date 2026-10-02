from types import SimpleNamespace
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

from urban_dreamer.learning.observations import build_actor_observations, aggregate_next_hop_actions
from urban_dreamer.traffic.topology import NextHopCandidates, load_traffic_topology

ROOT = Path(__file__).resolve().parents[1]


class Candidates:
    max_candidates = 3
    network = SimpleNamespace(edge_count=4, lengths_m=np.array([10.,20.,30.,40.]), free_flow_travel_seconds=np.array([1.,2.,3.,4.]))
    def candidates(self, current, destination):
        return (1,2) if destination == 3 else (2,)
    def distances(self, destination):
        return np.array([70.,40.,40.,0.]) if destination == 3 else np.array([30.,30.,0.,np.inf])
    def costs(self, destination):
        return self.distances(destination)/10
    def junction(self, current, successor):
        return 0., 1.


def test_candidate_features_use_destination_reachability_and_padding():
    states = [dict(current_edge=0,destination_edge=3,remaining_distance=5),dict(current_edge=0,destination_edge=2,remaining_distance=5)]
    result = build_actor_observations(policy_edge_features=np.ones((4,2)),vehicle_states=states,candidates=Candidates(),episode_progress=.2,active_vehicle_fraction=.5)
    assert result.features.shape == (2,3,10)
    np.testing.assert_array_equal(result.mask, [[True,True,False],[True,False,False]])
    np.testing.assert_array_equal(result.edge_indices, [[1,2,-1],[2,-1,-1]])
    np.testing.assert_allclose(result.features[0,:2,-2:], [[.2,.5],[.2,.5]])
    assert np.isfinite(result.features).all()


def test_aggregate_counts_each_event_with_episode_denominator():
    decisions=[SimpleNamespace(next_edge=1),SimpleNamespace(next_edge=2),SimpleNamespace(next_edge=1)]
    np.testing.assert_array_equal(aggregate_next_hop_actions(decisions,edge_count=3,total_vehicle_count=2), [0,1,.5])
    np.testing.assert_array_equal(aggregate_next_hop_actions([],edge_count=3,total_vehicle_count=2), [0,0,0])


def test_actor_static_cost_and_distance_include_initial_junction():
    class LongerJunction(Candidates):
        def junction(self, current, successor):
            return (30., 3.) if successor == 1 else (0., 1.)
    result = build_actor_observations(policy_edge_features=np.ones((4,2)),
        vehicle_states=[dict(current_edge=0,destination_edge=3,remaining_distance=5)],
        candidates=LongerJunction(),episode_progress=0.,active_vehicle_fraction=1.)
    assert result.features[0,0,4] == np.float32(np.log1p(10.+2.+4.)/np.log1p(3600.))
    assert result.features[0,0,5] == np.float32(np.log1p(30.+20.+40.)/np.log1p(10000.))


def _reference_observations_module():
    source = subprocess.run(['git', 'show', 'HEAD:urban_dreamer/learning/observations.py'],
                            check=True, capture_output=True, text=True, cwd=ROOT).stdout
    module = types.ModuleType('reference_observations')
    exec(compile(source, 'observations.py', 'exec'), module.__dict__)
    return module


def _real_candidates(network_name):
    directory = ROOT / 'networks' / network_name
    prepared = SimpleNamespace(network_xml=directory / f'{network_name}.net.xml',
                               network_agents_csv=directory / 'agents.csv',
                               edge_registry=directory / 'edge_registry.json',
                               od_file=directory / f'od_{network_name}.txt')
    topology = load_traffic_topology(prepared)
    return topology, NextHopCandidates(topology.network)


def _mixed_vehicle_states(topology, candidates, count):
    network = topology.network
    actor_states, vehicle_states = [], []
    for index in range(count):
        demand = topology.vehicles[index % len(topology.vehicles)]
        origin = network.edge_to_index[topology.origins[demand.origin_index]]
        destination = network.edge_to_index[topology.destinations[demand.destination_index]]
        choices = candidates.candidates(origin, destination)
        state = dict(vehicle_id=demand.vehicle_id, current_edge=origin, destination_edge=destination)
        variant = index % 5
        if variant == 0:
            state.update(departed=True, remaining_distance=0.37 * float(network.lengths_m[origin]))
        elif variant == 1 and choices:
            state.update(departed=True, in_junction=True, junction_remaining_distance=12.5,
                         junction_speed=3.5, committed_next_edge=choices[0])
        elif variant == 2:
            state.update(departed=False, departure_time=900.0)
        elif variant == 3 and choices:
            state.update(departed=True, committed_next_edge=choices[-1], remaining_distance=1.0)
        if choices:
            actor_states.append(state)
        vehicle_states.append(state)
        if index % 7 == 6:
            vehicle_states.append(dict(vehicle_id=f'{demand.vehicle_id}_arrived', current_edge=destination,
                                       destination_edge=destination, arrived=True, departed=True,
                                       remaining_distance=0.0))
    return actor_states, vehicle_states


def test_vectorized_builders_match_scalar_reference_on_real_topologies():
    reference = _reference_observations_module()
    for network_name in ('toy_cross_bottleneck', 'ingolstadt_custom'):
        topology, candidates = _real_candidates(network_name)
        actor_states, _ = _mixed_vehicle_states(topology, candidates, 400)
        policy_edge_features = np.random.default_rng(0).random((topology.network.edge_count, 2), dtype=np.float32)
        kwargs = dict(policy_edge_features=policy_edge_features, candidates=candidates,
                      episode_progress=0.42, active_vehicle_fraction=0.63)
        expected = reference.build_actor_observations(vehicle_states=actor_states, **kwargs)
        actual = build_actor_observations(vehicle_states=actor_states, **kwargs)
        assert actual.features.dtype == expected.features.dtype
        np.testing.assert_array_equal(actual.mask, expected.mask)
        np.testing.assert_array_equal(actual.edge_indices, expected.edge_indices)
        assert np.allclose(actual.features, expected.features, atol=1e-6)


def test_vectorized_actor_still_rejects_destination_without_candidates():
    topology, candidates = _real_candidates('toy_cross_bottleneck')
    edge = topology.network.edge_to_index[topology.origins[0]]
    state = dict(vehicle_id='stuck', current_edge=edge, destination_edge=edge)
    with pytest.raises(ValueError, match='no legal next-hop candidate'):
        build_actor_observations(policy_edge_features=np.ones((topology.network.edge_count, 2)),
                                 vehicle_states=[state], candidates=candidates,
                                 episode_progress=0., active_vehicle_fraction=1.)
