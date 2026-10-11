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
