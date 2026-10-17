from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from tools.evaluate_ga_routes import GeneticRouteCandidates, run_evaluation
from urban_dreamer.traffic.topology import TopologyError
from urban_dreamer.learning.episode_archive import read_episode_archive
from test_meso_baselines import meso_config
from test_route_baseline_collection import route_config

AUTHOR = Path(__file__).resolve().parents[1] / 'outputs/ingo_causal_routing_20261010/references/author_repo/src/baselines/genetic_algorithm.py'


def candidates():
    c = GeneticRouteCandidates.__new__(GeneticRouteCandidates)
    c.network = SimpleNamespace(edge_count=4, passenger_allowed=np.ones(4, dtype=bool),
        adjacency_list=((1, 2), (2, 3), (3,), ()))
    # Whole-network GA routes are not restricted to the actor's four-route pool.
    c.generate_paths = lambda *args: (_ for _ in ()).throw(AssertionError('actor pool consulted'))
    return c


def test_complete_ga_route_is_validated_without_actor_pool_projection():
    d = SimpleNamespace(current_edge=0, next_edge=1, route=(0, 1, 2, 3))
    assert candidates().decision_route(d, 3) == (0, 1, 2, 3)


@pytest.mark.parametrize('route,next_edge', [((0, 3), 3), ((1, 2, 3), 2),
                                           ((0, 1, 2), 1), ((0, 1, 2, 3), 2)])
def test_invalid_complete_ga_routes_rejected(route, next_edge):
    with pytest.raises(TopologyError):
        candidates().decision_route(SimpleNamespace(current_edge=0, next_edge=next_edge, route=route), 3)


@pytest.mark.skipif(not AUTHOR.exists(), reason='Author GA source not present')
def test_real_sumo_ga_decisions_stay_on30s_clock_and_preserve_full_routes(route_config, tmp_path):
    pytest.importorskip('libsumo')
    rows = run_evaluation(route_config, tmp_path/'ga', AUTHOR, seeds=(10042,),
        population_size=6, generations=2, vehicle_limit=4, deadline=600.)
    row = rows[0]
    assert row['policy_decision_times']
    assert all(t % 30 == 0 for t in row['policy_decision_times'])
    archive = read_episode_archive(row['archive'])
    assert archive['policy_decisions']
    assert all(event['decision_time'] % 30 == 0 for event in archive['policy_decisions'])
    assert all(event['route'][0] == event['current_edge'] and event['route'][-1] == event['destination_edge']
               for event in archive['policy_decisions'])
    assert row['diversion_decision_count'] == len(archive['decisions'])
    assert np.isfinite(row['all_demand_censored_att_seconds'])
    assert row['policy_stats']['full_route_fallbacks'] <= row['policy_stats']['fallback_events']


def test_cyclic_submissions_rejected_before_sumo_route_index_ambiguity():
    c = candidates()
    c.network.adjacency_list = ((1,), (0, 2), (3,), ())
    route = SimpleNamespace(current_edge=0, next_edge=1, route=(0, 1, 0, 1, 2, 3))
    with pytest.raises(TopologyError, match='loop-free'):
        c.decision_route(route, 3)
