from types import SimpleNamespace

import numpy as np
import pytest

from urban_dreamer.learning.observations import prepare_actor_window
from urban_dreamer.traffic.route_candidates import RouteCandidates, build_route_observations
from urban_dreamer.traffic.topology import EdgeNetwork, TopologyError


def network(adjacency=((1,), (2, 3), (4,), (4,), ())):
    count = len(adjacency)
    ids = tuple(map(str, range(count)))
    matrix = np.zeros((count, count), dtype=np.float32)
    for source, targets in enumerate(adjacency):
        matrix[source, list(targets)] = 1.
    costs = np.arange(1, count + 1, dtype=np.float64)
    return EdgeNetwork(ids, dict(zip(ids, range(count))), np.ones((count, 4)),
                       np.full(count, 10.), np.ones(count), costs * 10., matrix,
                       tuple(adjacency), np.ones(count, dtype=bool), costs)


def historical_paths(graph, source, destination):
    """Independent exhaustive reference for the former penalty-Dijkstra rule."""
    paths = []

    def visit(path):
        if path[-1] == destination:
            paths.append(tuple(path))
        else:
            for successor in graph.adjacency_list[path[-1]]:
                if successor not in path:
                    visit(path + [successor])

    visit([source])
    costs = graph.free_flow_travel_seconds.copy()
    selected = []
    for _ in range(4):
        shortest = min(paths, key=lambda path: sum(costs[edge] for edge in path))
        if shortest in selected:
            costs[list(shortest)] *= 3.
            shortest = min(paths, key=lambda path: sum(costs[edge] for edge in path))
            if shortest in selected:
                break
        selected.append(shortest)
        costs[list(shortest)] *= 3.
    return tuple(selected + [selected[0]] * (4 - len(selected)))


@pytest.mark.parametrize('source,destination', [(0, 4), (1, 4), (2, 4), (4, 4)])
def test_historical_penalty_rule_and_padding(source, destination):
    graph = network()
    candidates = RouteCandidates(graph, traffic_model='meso')
    assert candidates.generate_paths(source, destination) == historical_paths(graph, source, destination)
    assert len(candidates.generate_paths(source, destination)) == 4


def test_shared_first_hop_retains_distinct_actor_actions():
    candidates = RouteCandidates(network())
    paths = candidates.generate_paths(0, 4)
    assert paths[:2] == ((0, 1, 2, 4), (0, 1, 3, 4))
    assert paths[2:] == (paths[0], paths[0])
    assert candidates.candidates(0, 4) == (1,)
    observations = build_route_observations(
        policy_edge_features=np.array([[0., 1.], [.1, 1.], [.2, .3], [.9, .1], [0., 1.]]),
        vehicle_states=[dict(current_edge=0, destination_edge=4, remaining_distance=5.)],
        candidates=candidates, episode_progress=.2, active_vehicle_fraction=.7)
    assert observations.features.shape == (1, 4, 10)
    assert observations.mask.all()
    np.testing.assert_array_equal(observations.edge_indices, [[1, 1, 1, 1]])
    assert observations.routes == (paths,)
    assert observations.features[0, 0, 0] != observations.features[0, 1, 0]
    assert observations.features[0, 0, 4] == pytest.approx(np.log1p(11) / np.log1p(3600.))


def test_ties_preserve_original_adjacency_order():
    graph = network(((2, 1), (3,), (3,), ()))
    graph.free_flow_travel_seconds[:] = 1.
    assert RouteCandidates(graph).generate_paths(0, 3)[0] == (0, 2, 3)


def test_only_destination_may_have_singleton_candidate():
    candidates = RouteCandidates(network())
    with pytest.raises(TopologyError, match='no destination-reachable route'):
        candidates.generate_paths(4, 0)
    with pytest.raises(TopologyError, match='unknown current'):
        candidates.generate_paths(-1, 4)
    candidates.network.passenger_allowed[2] = False
    assert candidates.generate_paths(0, 4) == ((0, 1, 3, 4),) * 4
    assert candidates.generate_paths(4, 4) == ((4,),) * 4
    with pytest.raises(TopologyError, match='forbid_revisit'):
        RouteCandidates(network(), forbid_revisit=True)


def test_route_decisions_validate_full_suffix():
    candidates = RouteCandidates(network())
    paths = candidates.generate_paths(0, 4)
    decision = SimpleNamespace(current_edge=0, next_edge=1, route=paths[1])
    assert candidates.decision_route(decision, 4) == paths[1]
    for route, next_edge in [((0, 1, 4), 1), (paths[0], 2), ((1, 2, 4), 2), ((0,), 0)]:
        with pytest.raises(TopologyError):
            candidates.decision_route(SimpleNamespace(current_edge=0, next_edge=next_edge, route=route), 4)


def test_route_cache_is_bounded_lru():
    candidates = RouteCandidates(network())
    candidates._route_candidate_cache_limit = 2
    first = candidates.generate_paths(0, 4)
    candidates.generate_paths(1, 4)
    assert candidates.generate_paths(0, 4) is first
    candidates.generate_paths(2, 4)
    assert tuple(candidates._route_candidate_cache) == ((0, 4), (2, 4))


def test_dynamic_cost_uses_selected_route_and_current_remaining_distance():
    graph = network()
    candidates = RouteCandidates(graph, traffic_model='meso')
    edges = np.array([[.1, .5], [.2, .5], [.1, .2], [.2, 1.], [.1, .5]], dtype=np.float32)
    kwargs = dict(policy_edge_features=edges,
                  vehicle_states=[dict(current_edge=0, destination_edge=4, remaining_distance=5.)],
                  candidates=candidates, episode_progress=0., active_vehicle_fraction=1.,
                  prior_cost_mode='dynamic')
    observations = build_route_observations(**kwargs)
    np.testing.assert_allclose(observations.prior_costs_seconds[0], [30., 19., 30., 30.])
    prepared = build_route_observations(**{**kwargs, 'policy_edge_features': prepare_actor_window(edges, graph.edge_count)})
    np.testing.assert_array_equal(prepared.features, observations.features)
    np.testing.assert_array_equal(prepared.prior_costs_seconds, observations.prior_costs_seconds)
    capped = build_route_observations(**kwargs, prior_speed_cap_mps=4.)
    np.testing.assert_allclose(capped.prior_costs_seconds[0], [33.75, 28.75, 33.75, 33.75])


def test_dynamic_empty_and_stopped_edges_and_invalid_speed_bounds():
    candidates = RouteCandidates(network(((1,), ())))
    kwargs = dict(policy_edge_features=np.array([[0., 0.], [.1, 0.]]),
                  vehicle_states=[dict(current_edge=0, destination_edge=1, remaining_distance=5.)],
                  candidates=candidates, episode_progress=0., active_vehicle_fraction=1., prior_cost_mode='dynamic')
    observations = build_route_observations(**kwargs, prior_speed_floor_mps=2.)
    np.testing.assert_allclose(observations.prior_costs_seconds, [[10.5] * 4])
    with pytest.raises(ValueError, match='speed floor'):
        build_route_observations(**kwargs, prior_speed_floor_mps=0.)
    with pytest.raises(ValueError, match='must not exceed'):
        build_route_observations(**kwargs, prior_speed_floor_mps=2., prior_speed_cap_mps=1.)


def test_empty_route_batch_preserves_shapes():
    observations = build_route_observations(policy_edge_features=np.ones((5, 2)), vehicle_states=[],
        candidates=RouteCandidates(network()), episode_progress=0., active_vehicle_fraction=0.)
    assert observations.features.shape == (0, 4, 10)
    assert observations.edge_indices.shape == observations.mask.shape == (0, 4)
    assert observations.routes == ()
