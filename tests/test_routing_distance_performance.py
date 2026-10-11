"""Exact behavior lock for the routing distance hot loop."""
import heapq
from types import SimpleNamespace

import numpy as np
import pytest

from tools import causal_routing_policies as policies


def network(adjacency, costs, allowed=None):
    size = len(adjacency)
    return SimpleNamespace(
        edge_ids=tuple(str(i) for i in range(size)),
        adjacency_list=tuple(tuple(row) for row in adjacency),
        passenger_allowed=np.ones(size, dtype=bool) if allowed is None else allowed,
        free_flow_travel_seconds=np.asarray(costs, dtype=np.float64))


def old_distance(router, destination):
    """Frozen pre-optimization NumPy loop; also record heap tie order."""
    distance = np.full(len(router.successors), np.inf)
    distance[destination] = router.costs[destination]
    queue = [(float(distance[destination]), destination)]
    popped = []
    while queue:
        value, target = heapq.heappop(queue)
        popped.append((float(value), target))
        if value != distance[target]:
            continue
        for source in router.predecessors[target]:
            candidate = value + router.costs[source]
            if candidate < distance[source]:
                distance[source] = candidate
                heapq.heappush(queue, (candidate, source))
    return distance, popped


@pytest.mark.parametrize('seed', range(12))
def test_randomized_distances_and_heap_order_match_old_bytes(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    size = 2 + seed * 3
    adjacency = [np.flatnonzero(rng.random(size) < .18) for _ in range(size)]
    costs = rng.choice([0., -0., 1., 1e-300, 1e150], size=size)
    costs += rng.choice([0., 0., .1], size=size)
    router = policies.RealTimeShortestRouter(network(adjacency, costs, rng.random(size) > .15))
    original_pop = heapq.heappop
    for destination in range(size):
        expected, expected_pops = old_distance(router, destination)
        popped = []

        def pop(queue):
            value, target = original_pop(queue)
            popped.append((float(value), target))
            return value, target

        with monkeypatch.context() as patch:
            patch.setattr(policies.heapq, 'heappop', pop)
            actual = router._distance(destination)
        assert actual.dtype == np.float64
        assert actual.tobytes() == expected.tobytes()
        assert popped == expected_pops


def test_zero_cost_cycles_ties_disconnected_and_passenger_exclusion():
    router = policies.RealTimeShortestRouter(network(
        ((2, 1, 4), (2, 3), (1, 3), (), (3,), ()),
        [1., 0., 0., 0., 0., 7.], [True, True, True, True, False, True]))
    for destination in range(6):
        expected, _ = old_distance(router, destination)
        assert router._distance(destination).tobytes() == expected.tobytes()
    assert router.next_edge(0, 3, [2, 1]) == 1
    with pytest.raises(ValueError, match='legal passenger'):
        router.next_edge(0, 3, [4])
    with pytest.raises(ValueError, match='unreachable'):
        router.next_edge(0, 5, [2, 1])
    assert router.stats['failed_searches'] == 1
    assert router.stats['decision_count'] == 2


def test_cache_identity_invalidation_stats_and_reset(monkeypatch):
    router = policies.RealTimeShortestRouter(network(((1,), (2,), ()), [1., 2., 3.]))
    clock = iter(range(20))
    monkeypatch.setattr(policies, 'perf_counter', lambda: next(clock))
    router.update_costs([1., 2., 3.], 0.)
    first = router._distance(2)
    assert router._distance(2) is first
    router.update_costs([1., 2., 3.], 0.)
    assert router._distance(2) is first
    assert router.stats['cache_hits'] == 2
    assert router.stats['optimize_calls'] == 1
    assert router.stats['optimize_seconds'] == 1.
    router.update_costs([1., 2., 3.], 1.)
    assert not router._distances
    second = router._distance(2)
    assert second is not first and second.tobytes() == first.tobytes()
    costs = np.array([9., 8., 7.])
    router.update_costs(costs, 1.)
    costs[:] = 100.
    expected, _ = old_distance(router, 2)
    assert router._distance(2).tobytes() == expected.tobytes()
    assert router.stats['optimize_calls'] == 3
    assert router.stats['optimize_seconds'] == 3.
    router.reset()
    assert router.timestamp is None and not router._distances
    assert not any(router.stats.values())
    assert router._distance(2).tobytes() == first.tobytes()


def test_author_ga_inherits_compatible_distance_cache_and_reset():
    from test_causal_routing_policies import AUTHOR, graph

    if not AUTHOR.exists():
        pytest.skip('Downloaded author source not present')
    router = policies.AuthorGeneticRouter(graph(), AUTHOR, population_size=6, generations=2)
    router.update_costs([1., 9., 2., 1., 1.], 30.)
    expected, _ = old_distance(router, 3)
    actual = router._distance(3)
    assert actual.tobytes() == expected.tobytes()
    assert router._distance(3) is actual
    assert router.author._current_step == 30
    assert router.env.edge_states['a'].travel_time == 9.
    router.reset()
    assert not router._distances and not router.author._route_cache
    assert router.author._current_step == 0
    assert not any(router.stats.values())
    expected, _ = old_distance(router, 3)
    assert router._distance(3).tobytes() == expected.tobytes()
