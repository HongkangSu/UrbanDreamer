import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('causal_routing_policies', ROOT / 'tools/causal_routing_policies.py')
policies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policies)
AUTHOR = ROOT / 'outputs/ingo_causal_routing_20261010/references/author_repo/src/baselines/genetic_algorithm.py'


def graph():
    return SimpleNamespace(edge_ids=('o', 'a', 'b', 'd', 'forbidden'),
        edge_to_index=dict(o=0, a=1, b=2, d=3, forbidden=4),
        adjacency_list=((1, 2, 4), (3,), (3,), (), (3,)),
        passenger_allowed=np.array([True, True, True, True, False]),
        lengths_m=np.ones(5) * 10, speed_limits_mps=np.ones(5) * 10,
        free_flow_travel_seconds=np.ones(5))


def test_realtime_observed_costs_change_path_and_invalidate_cache():
    router = policies.RealTimeShortestRouter(graph())
    router.update_costs([1, 1, 9, 1, 1], 0)
    assert router.next_edge(0, 3, [2, 1]) == 1
    assert router.next_edge(0, 3, [2, 1]) == 1
    assert router.stats['cache_hits'] == 1
    router.update_costs([1, 10, 1, 1, 1], 1)
    assert router.next_edge(0, 3, [1, 2]) == 2
    assert router.stats['optimize_calls'] == 2
    assert router.next_edge(0, 3, [1]) == 1
    router.update_costs([1, 1, 1, 1, 1], 2)
    assert router.next_edge(0, 3, [2, 1]) == 1


def test_realtime_destination_and_legal_turns():
    router = policies.RealTimeShortestRouter(graph())
    assert router.next_edge(0, 2, [1, 2]) == 2
    with pytest.raises(ValueError, match='unreachable'):
        router.next_edge(0, 2, [1])
    with pytest.raises(ValueError, match='legal passenger'):
        router.next_edge(0, 3, [4])
    with pytest.raises(TypeError):
        router.update_costs(np.ones(5), 0, future_vehicles=[])
    with pytest.raises(ValueError, match='finite'):
        router.update_costs([1, np.nan, 1, 1, 1], 0)


@pytest.mark.skipif(not AUTHOR.exists(), reason='Downloaded author source not present')
def test_author_route_cache_uses_simulation_seconds_and_resets():
    random.seed(42)
    router = policies.AuthorGeneticRouter(graph(), AUTHOR, population_size=6, generations=2)
    router.update_costs([1, 1, 100, 1, 1], 10)
    assert router.next_edge(0, 3, [1, 2]) == 1
    router.update_costs([1, 100, 1, 1, 1], 309)
    assert router.next_edge(0, 3, [1, 2]) == 1
    assert router.stats['cache_hits'] == 1
    router.update_costs([1, 100, 1, 1, 1], 310)
    assert router.next_edge(0, 3, [1, 2]) == 2
    assert router.stats['optimize_calls'] == 2
    assert router.stats['fallback_events'] == 0
    assert not router.env.agents
    assert not hasattr(router.env, 'demands')
    router.reset()
    assert not router.author._route_cache
    assert router.author._current_step == 0
    assert router.stats['optimize_calls'] == 0


@pytest.mark.skipif(not AUTHOR.exists(), reason='Downloaded author source not present')
def test_author_failed_search_keeps_first_legal_fallback():
    random.seed(42)
    network = graph()
    network.adjacency_list = ((1, 2, 4), (), (), (), ())
    router = policies.AuthorGeneticRouter(network, AUTHOR, population_size=6, generations=2)
    router.update_costs(np.ones(5), 1)
    assert router.next_edge(0, 3, [2, 1]) == 2
    assert router.stats['failed_searches'] == 1
    assert router.stats['fallback_events'] == 1
    assert router.next_edge(0, 3, [1, 2]) == 1
    assert router.stats['failed_searches'] == 1
    assert router.stats['fallback_events'] == 2
    assert router.stats['cache_hits'] == 1


@pytest.mark.skipif(not AUTHOR.exists(), reason='Downloaded author source not present')
def test_author_complete_route_keeps_optimized_suffix(monkeypatch):
    network = graph()
    network.adjacency_list = ((1, 2, 4), (2, 3), (3,), (), (3,))
    router = policies.AuthorGeneticRouter(network, AUTHOR, population_size=6, generations=2)
    monkeypatch.setattr(router.author, '_optimize_route', lambda origin, destination: ['o', 'a', 'b', 'd'])
    candidates = SimpleNamespace(candidates=lambda current, destination: (1, 2),
        full_route=lambda *args: pytest.fail('Successful GA route must retain its optimized suffix'))
    router.update_costs(np.ones(5), 0)
    # A shortest completion through a would skip b; the GA suffix must survive.
    assert router.complete_route(0, 3, candidates) == (0, 1, 2, 3)
    assert router.stats['decision_count'] == 1
    assert router.stats['fallback_events'] == 0
    assert router.stats['full_route_fallbacks'] == 0


@pytest.mark.skipif(not AUTHOR.exists(), reason='Downloaded author source not present')
def test_author_complete_route_preserves_cache_ttl():
    random.seed(42)
    router = policies.AuthorGeneticRouter(graph(), AUTHOR, population_size=6, generations=2)
    candidates = SimpleNamespace(candidates=lambda current, destination: (1, 2))
    router.update_costs([1, 1, 100, 1, 1], 10)
    assert router.complete_route(0, 3, candidates) == (0, 1, 3)
    router.update_costs([1, 100, 1, 1, 1], 309)
    assert router.complete_route(0, 3, candidates) == (0, 1, 3)
    assert router.stats['cache_hits'] == 1
    router.update_costs([1, 100, 1, 1, 1], 310)
    assert router.complete_route(0, 3, candidates) == (0, 2, 3)
    assert router.stats['optimize_calls'] == 2


@pytest.mark.skipif(not AUTHOR.exists(), reason='Downloaded author source not present')
def test_author_complete_route_failed_search_completes_first_legal_suffix(monkeypatch):
    router = policies.AuthorGeneticRouter(graph(), AUTHOR, population_size=6, generations=2)
    # Returning no population exercises the original optimizer's [origin] failure.
    monkeypatch.setattr(router.author, '_initialize_population', lambda *args: [])
    suffix_calls = []
    def suffix(current, selected, destination):
        suffix_calls.append((current, selected, destination))
        return (current, selected, destination)
    candidates = SimpleNamespace(candidates=lambda current, destination: (2, 1), full_route=suffix)
    router.update_costs(np.ones(5), 10)
    assert router.complete_route(0, 3, candidates) == (0, 2, 3)
    assert router.complete_route(0, 3, candidates) == (0, 2, 3)
    assert suffix_calls == [(0, 2, 3), (0, 2, 3)]
    assert router.stats['failed_searches'] == 1
    assert router.stats['fallback_events'] == 2
    assert router.stats['full_route_fallbacks'] == 2
    assert router.stats['cache_hits'] == 1


@pytest.mark.skipif(not AUTHOR.exists(), reason='Downloaded author source not present')
@pytest.mark.parametrize('route', [
    ['o', 'a'], ['a', 'd'], ['o', 'unknown', 'd'],
    ['o', 'a', 'b', 'd'], ['o', 'forbidden', 'd'], [],
])
def test_author_complete_route_rejects_invalid_cached_paths(route):
    router = policies.AuthorGeneticRouter(graph(), AUTHOR, population_size=6, generations=2)
    router.update_costs(np.ones(5), 10)
    router.author._route_cache[('o', 'd')] = route
    router.author._cache_timestamps[('o', 'd')] = 10
    candidates = SimpleNamespace(candidates=lambda current, destination: (1, 2),
        full_route=lambda *args: pytest.fail('Invalid routes must not be hidden by fallback'))
    with pytest.raises(ValueError, match='invalid route'):
        router.complete_route(0, 3, candidates)


@pytest.mark.skipif(not AUTHOR.exists(), reason='Downloaded author source not present')
@pytest.mark.parametrize(('raw_route', 'expected'), [
    (['o', 'a', 'o', 'b', 'd'], (0, 2, 3)),
    (['o', 'a', 'b', 'a', 'd'], (0, 1, 3)),
])
def test_author_complete_route_erases_loops_without_mutating_ga_cache(raw_route, expected):
    network = graph()
    network.adjacency_list = ((1, 2, 4), (0, 2, 3), (1, 3), (), (3,))
    router = policies.AuthorGeneticRouter(network, AUTHOR, population_size=6, generations=2)
    router.update_costs(np.ones(5), 10)
    router.author._route_cache[('o', 'd')] = raw_route.copy()
    router.author._cache_timestamps[('o', 'd')] = 10
    candidates = SimpleNamespace(candidates=lambda current, destination: (1, 2))
    assert router.complete_route(0, 3, candidates) == expected
    assert router.author._route_cache[('o', 'd')] == raw_route
    assert router.stats['loop_erased_routes'] == 1
    assert router.stats['loop_edges_removed'] == 2
    assert router.stats['fallback_loop_erased_routes'] == 0
    assert router.stats['fallback_events'] == 0


@pytest.mark.skipif(not AUTHOR.exists(), reason='Downloaded author source not present')
def test_author_complete_route_reports_loop_erased_failure_fallback(monkeypatch):
    network = graph()
    network.adjacency_list = ((1, 2, 4), (0, 3), (3,), (), (3,))
    router = policies.AuthorGeneticRouter(network, AUTHOR, population_size=6, generations=2)
    monkeypatch.setattr(router.author, '_initialize_population', lambda *args: [])
    candidates = SimpleNamespace(candidates=lambda current, destination: (1, 2),
        full_route=lambda *args: (0, 1, 0, 2, 3))
    router.update_costs(np.ones(5), 10)
    assert router.complete_route(0, 3, candidates) == (0, 2, 3)
    assert router.author._route_cache[('o', 'd')] == ['o']
    assert router.stats['failed_searches'] == 1
    assert router.stats['fallback_events'] == 1
    assert router.stats['full_route_fallbacks'] == 1
    assert router.stats['fallback_loop_erased_routes'] == 1
    assert router.stats['loop_erased_routes'] == 1
    assert router.stats['loop_edges_removed'] == 2
