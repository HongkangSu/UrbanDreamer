from dataclasses import replace

import numpy as np
import pytest

from urban_dreamer.learning.observations import build_actor_observations
from urban_dreamer.traffic.topology import EdgeNetwork, NextHopCandidates, TopologyError


def graph():
    # The unvisited successor 1 is trapped once 0 or previously used 4 is removed.
    adjacency = ((1, 2, 3), (0, 4), (5,), (5,), (5,), ())
    return EdgeNetwork(tuple(map(str, range(6))), {str(i): i for i in range(6)},
                       np.ones((6, 4)), np.ones(6), np.ones(6), np.ones(6),
                       np.zeros((6, 6)), adjacency, np.ones(6, dtype=bool), np.ones(6))


def test_residual_reachability_prunes_unvisited_trapped_successor():
    candidates = NextHopCandidates(graph(), forbid_revisit=True)
    assert candidates.candidates(0, 5) == (1, 2, 3)
    assert candidates.legal_candidates(0, 5, (4,)) == (2, 3)
    assert candidates.legal_candidates(0, 5, (4, 2)) == (3,)
    assert candidates.legal_candidates(0, 5, ()) == (1, 2, 3)
    with pytest.raises(TopologyError, match='no-revisit'):
        candidates.legal_candidates(0, 5, (2, 3, 4))


def test_full_route_avoids_visited_and_current_and_rejects_traps():
    candidates = NextHopCandidates(graph(), forbid_revisit=True)
    assert candidates.full_route(0, 1, 5) == (0, 1, 4, 5)
    assert candidates.full_route(0, 2, 5, visited_edges=(4,)) == (0, 2, 5)
    for history, choice in [((4,), 1), ((2,), 2), ((5,), 3)]:
        with pytest.raises(TopologyError, match='no-revisit'):
            candidates.full_route(0, choice, 5, visited_edges=history)


def test_zero_cost_cycles_still_have_simple_route():
    network = replace(graph(), adjacency_list=((1,), (2, 3), (1, 3), (5,), (), ()),
                      free_flow_travel_seconds=np.zeros(6))
    candidates = NextHopCandidates(network, forbid_revisit=True)
    route = candidates.full_route(0, 1, 5)
    assert route[0] == 0 and route[-1] == 5
    assert len(set(route)) == len(route)


def observations(candidates, histories):
    return build_actor_observations(
        policy_edge_features=np.ones((6, 2), dtype=np.float32), candidates=candidates,
        vehicle_states=[dict(current_edge=0, destination_edge=5, visited_edges=history)
                        for history in histories], episode_progress=.2, active_vehicle_fraction=.8)


def test_same_static_pair_has_isolated_history_masks_and_cache():
    candidates = NextHopCandidates(graph(), forbid_revisit=True)
    obs = observations(candidates, [(4, 2), (4, 3), ()])
    np.testing.assert_array_equal(obs.edge_indices, [[-1, -1, 3], [-1, 2, -1], [1, 2, 3]])
    np.testing.assert_array_equal(obs.mask, obs.edge_indices != -1)
    assert not np.any(obs.features[~obs.mask])
    cached = candidates._actor_static_cache[(0, 5)]
    np.testing.assert_array_equal(cached.edge_indices, [1, 2, 3])
    assert cached.mask.all()
    repeat = observations(candidates, [()])
    np.testing.assert_array_equal(repeat.features[0], obs.features[2])
    with pytest.raises(TopologyError, match='no-revisit'):
        observations(candidates, [(2, 3, 4)])


def test_legacy_default_ignores_history_exactly():
    candidates = NextHopCandidates(graph())
    assert candidates.legal_candidates(0, 5, (1, 2, 3, 4)) == candidates.candidates(0, 5)
    assert candidates.full_route(0, 1, 5, (1, 2, 3, 4)) == candidates.full_route(0, 1, 5)
    obs = observations(candidates, [(), (1, 2, 3, 4)])
    np.testing.assert_array_equal(obs.features[0], obs.features[1])
    np.testing.assert_array_equal(obs.mask[0], obs.mask[1])
    np.testing.assert_array_equal(obs.edge_indices[0], obs.edge_indices[1])


def test_no_revisit_observations_require_explicit_history():
    kwargs = dict(policy_edge_features=np.ones((6, 2), dtype=np.float32),
                  vehicle_states=[dict(current_edge=0, destination_edge=5)],
                  episode_progress=.2, active_vehicle_fraction=.8)
    with pytest.raises(TopologyError, match='missing visited_edges'):
        build_actor_observations(candidates=NextHopCandidates(graph(), forbid_revisit=True), **kwargs)
    assert build_actor_observations(candidates=NextHopCandidates(graph()), **kwargs).mask.all()


def test_certified_static_suffix_matches_legacy_ties_and_falls_back():
    network = replace(graph(), adjacency_list=((1,), (2, 3), (5,), (5,), (), ()))
    candidates = NextHopCandidates(network, forbid_revisit=True)
    assert candidates.full_route(0, 1, 5) == NextHopCandidates(network).full_route(0, 1, 5) == (0, 1, 2, 5)
    assert candidates.full_route(0, 1, 5, (2,)) == (0, 1, 3, 5)


def test_legal_history_cache_is_bounded_and_history_specific():
    candidates = NextHopCandidates(graph(), forbid_revisit=True)
    candidates._legal_candidate_cache_limit = 2
    assert candidates.legal_candidates(0, 5, (4,)) == (2, 3)
    assert candidates.legal_candidates(0, 5, (4, 2)) == (3,)
    assert candidates.legal_candidates(0, 5, (4, 3)) == (2,)
    assert len(candidates._legal_candidate_cache) == 2
    assert (0, 5, (4,)) not in candidates._legal_candidate_cache
    assert candidates.legal_candidates(0, 5, (4,)) == (2, 3)
