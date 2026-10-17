from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from urban_dreamer.learning.baselines.dta import DtaMethod
from urban_dreamer.learning.observations import CandidateObservations
from urban_dreamer.traffic.route_candidates import RouteCandidates
from urban_dreamer.traffic.topology import EdgeNetwork, TrafficTopology, VehicleDemand


def make_method():
    # All six OD paths share their first hop, with more paths than actor slots.
    adjacency_list = ((1,), tuple(range(2, 8)), *((8,),) * 6, (), (1,))
    adjacency = np.zeros((10, 10), dtype=np.float32)
    for source, targets in enumerate(adjacency_list):
        adjacency[source, list(targets)] = 1.
    seconds = np.asarray([1., 1., 1., 2., 3., 4., 5., 6., 1., 1.])
    ids = tuple(map(str, range(10)))
    network = EdgeNetwork(ids, dict(zip(ids, range(10))), np.zeros((10, 4)),
        np.ones(10), np.full(10, 10.), seconds, adjacency, adjacency_list,
        np.ones(10, dtype=bool), seconds)
    topology = TrafficTopology(network, ('0',), ('8',),
        tuple(VehicleDemand(f'v{i}', 0, 0, float(i)) for i in range(20)))
    runner = SimpleNamespace(network=network, topology=topology,
        candidates=RouteCandidates(network, traffic_model='meso'))
    return DtaMethod(SimpleNamespace(seed=42), runner), runner


def observation(routes, *, mask=None):
    return CandidateObservations(np.zeros((1, len(routes), 10)),
        np.ones((1, len(routes)), dtype=bool) if mask is None else np.asarray([mask]),
        np.asarray([[path[1] for path in routes]]), routes=(tuple(routes),))


def vehicle(current=0):
    return dict(vehicle_id='v0', current_edge=current, destination_edge=8)


def test_initial_and_aon_paths_stay_in_exact_actor_pool():
    method, runner = make_method()
    pool = runner.candidates.generate_paths(0, 8)
    assert len(pool) == 4
    group = method.od_groups[(0, 8)]
    assert set(group['path_fractions']) <= set(pool)
    assert next(iter(group['path_fractions'])) == min(pool,
        key=lambda path: method._path_cost(path, method.edge_cost))
    branch = next(edge for edge in range(2, 8) if all(edge not in path for path in pool))
    method.realized_seconds[2:8] = 10000.
    method.realized_seconds[branch] = runner.network.free_flow_travel_seconds[branch]
    method.on_episode([], [], episode=1)
    unconstrained, _ = method._shortest_path(0, 8, method.edge_cost)
    assert branch in unconstrained and unconstrained not in pool
    assert set(group['path_fractions']) <= set(pool)
    expected = min(pool, key=lambda path: method._path_cost(path, method.edge_cost))
    assert group['path_fractions'][expected] == pytest.approx(1.)


def test_selection_matches_full_suffix_when_first_hops_are_shared():
    method, runner = make_method()
    first, second, *_ = runner.candidates.generate_paths(0, 8)
    assert first != second and first[1] == second[1]
    method.assignment['v0'] = second
    obs = observation((first, second))
    assert method.select([vehicle()], obs, None, 30.) == [1]
    suffixes = (first[1:], second[1:])
    assert method.select([vehicle(1)], observation(suffixes), None, 60.) == [1]
    assert method.route_assignment_follow_count == 2
    assert method.route_assignment_fallback_count == 0


@pytest.mark.parametrize('current', [0, 9])
def test_unavailable_or_off_path_suffix_uses_full_route_marginal_cost(current):
    method, runner = make_method()
    routes = runner.candidates.generate_paths(current, 8)
    method.assignment['v0'] = (0, 1, 7, 8)  # Outside origin action pool.
    assert method.assignment['v0'] not in routes
    method.edge_cost[2:8] = 1000.
    target = routes[1]
    method.edge_cost[target[-2]] = 1.
    obs = observation(routes)
    expected = min(range(4), key=lambda column: method._path_cost(routes[column], method.edge_cost))
    assert method.select([vehicle(current)], obs, None, 30.) == [expected]
    assert expected == 1
    assert method.route_assignment_follow_count == 0
    assert method.route_assignment_fallback_count == 1


def test_fallback_respects_legal_mask():
    method, runner = make_method()
    routes = runner.candidates.generate_paths(0, 8)
    method.assignment.clear()
    assert method.select([vehicle()], observation(routes, mask=[False, False, True, False]),
        None, 30.) == [2]


def test_route_msa_fraction_mass_and_state_roundtrip():
    method, runner = make_method()
    pool = set(runner.candidates.generate_paths(0, 8))
    method.begin_episode(runner.topology, seed=77)
    for episode in range(1, 7):
        method.realized_seconds[2:8] = 100.
        winner = sorted(pool)[episode % len(pool)]
        method.realized_seconds[winner[-2]] = runner.network.free_flow_travel_seconds[winner[-2]]
        method.on_episode([], [], episode=episode)
        fractions = method.od_groups[(0, 8)]['path_fractions']
        assert set(fractions) <= pool
        assert sum(fractions.values()) == pytest.approx(1.)
        assert all(value >= 0. for value in fractions.values())
    state = method.state_dict()
    restored, _ = make_method()
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    routes = runner.candidates.generate_paths(0, 8)
    assert method.select([vehicle()], observation(routes), None, 30.) == restored.select(
        [vehicle()], observation(routes), None, 30.)
    # Evaluation keeps learned costs/fractions frozen, and resets diagnostics.
    restored.begin_episode(runner.topology, seed=999)
    assert restored.route_assignment_follow_count == restored.route_assignment_fallback_count == 0
    assert restored.state_dict()['path_sets'] == state['path_sets']
    np.testing.assert_array_equal(restored.edge_cost, state['edge_cost'])


def test_route_fraction_draws_ignore_future_arrivals():
    method, runner = make_method()
    first, second, *_ = runner.candidates.generate_paths(0, 8)
    method.od_groups[(0, 8)]['path_fractions'] = {first: .25, second: .75}
    alone = replace(runner.topology, vehicles=runner.topology.vehicles[:1])
    first_count = 0
    for seed in range(400):
        method.begin_episode(alone, seed=seed)
        path = method.assignment['v0']
        first_count += path == first
        method.begin_episode(runner.topology, seed=seed)
        assert method.assignment['v0'] == path
    assert first_count / 400 == pytest.approx(.25, abs=.06)
