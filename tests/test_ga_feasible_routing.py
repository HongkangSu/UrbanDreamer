from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from tools.ga_feasible_routing import FeasibleGeneticRouter


ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / 'outputs/ingo_causal_routing_20261010/references/author_repo/src/baselines/genetic_algorithm.py'
pytestmark = pytest.mark.skipif(not AUTHOR.exists(), reason='Downloaded author source not present')


def network(adjacency, costs=None):
    size = len(adjacency)
    edges = tuple(str(i) for i in range(size))
    return SimpleNamespace(edge_ids=edges, edge_to_index=dict(zip(edges, range(size))),
        adjacency_list=adjacency, passenger_allowed=np.ones(size, dtype=bool),
        lengths_m=np.ones(size) * 10, speed_limits_mps=np.ones(size) * 10,
        free_flow_travel_seconds=np.ones(size) if costs is None else np.asarray(costs))


def router(graph):
    return FeasibleGeneticRouter(graph, AUTHOR, population_size=12, generations=2)


def test_constructive_route_has_no_author_hop_cutoff():
    policy = router(network(tuple((i + 1,) for i in range(149)) + ((),)))
    route = policy.author._generate_random_route('0', '149', max_hops=1)
    assert route == [str(i) for i in range(150)]
    assert policy.stats['constructive_routes'] == 1


def test_routes_are_legal_acyclic_and_diverse_despite_cycles_and_dead_ends():
    graph = network(((1, 2, 4), (0, 3), (1, 3), (), (4,)))
    policy = router(graph)
    routes = set()
    for seed in range(30):
        random.seed(seed)
        route = policy.author._generate_random_route('0', '3')
        routes.add(tuple(route))
        assert route[0] == '0' and route[-1] == '3'
        assert len(route) == len(set(route))
        assert all(int(b) in graph.adjacency_list[int(a)] for a, b in zip(route, route[1:]))
    assert len(routes) >= 2
    assert policy.author._generate_random_route('4', '3') is None
    assert policy.stats['unreachable_attempts'] == 1


def test_proposals_ignore_observed_cost_but_fitness_uses_it():
    policy = router(network(((1, 2), (3,), (3,), ())))
    random.seed(73)
    before = [policy.author._generate_random_route('0', '3') for _ in range(20)]
    policy.update_costs([1, 100, 1, 1], 10)
    random.seed(73)
    after = [policy.author._generate_random_route('0', '3') for _ in range(20)]
    assert before == after
    assert policy.author._evaluate_fitness(['0', '2', '3']) > policy.author._evaluate_fitness(['0', '1', '3'])
    random.seed(73)
    assert policy.next_edge(0, 3, [1, 2]) == 2
    assert policy.stats['failed_searches'] == policy.stats['fallback_events'] == 0
    assert policy.stats['evolved_generations'] == 2
    assert policy.stats['tournament_calls'] == 24


def test_cache_clock_reset_and_bound_generator_survive_reset():
    policy = router(network(((1, 2), (3,), (3,), ())))
    random.seed(42)
    policy.update_costs([1, 1, 100, 1], 10)
    assert policy.next_edge(0, 3, [1, 2]) == 1
    policy.update_costs([1, 100, 1, 1], 309)
    assert policy.next_edge(0, 3, [1, 2]) == 1
    policy.update_costs([1, 100, 1, 1], 310)
    assert policy.next_edge(0, 3, [1, 2]) == 2
    assert policy.stats['cache_hits'] == 1
    policy.reset()
    assert policy.author._current_step == 0
    assert not policy.author._route_cache
    assert not policy._static_potentials
    assert policy.stats['constructive_routes'] == 0
    assert policy.author._generate_random_route('3', '3') == ['3']


def test_positive_static_cost_required_and_passenger_filter_preserved():
    graph = network(((1, 2), (3,), (3,), ()))
    graph.passenger_allowed[1] = False
    policy = router(graph)
    assert policy.author._generate_random_route('0', '3') == ['0', '2', '3']
    with pytest.raises(ValueError, match='strictly positive'):
        router(network(((1,), ()), [0, 1]))
