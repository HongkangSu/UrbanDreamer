"""Route actions retain complete suffixes while conditioning the existing turn model."""
from copy import deepcopy

import numpy as np
import pytest

from test_event_imagination_protocol import seed as original_seed
from urban_dreamer.learning.action_events import pack_action_events
from urban_dreamer.learning.imagination import advance_imagined_window
from urban_dreamer.traffic.environment import RouteDecision
from urban_dreamer.traffic.route_candidates import RouteCandidates
from urban_dreamer.traffic.topology import EdgeNetwork


def topology(lengths=(10., 10., 10., 20., 1000.), junction_lengths=None):
    lengths = np.asarray(lengths)
    return EdgeNetwork(tuple('abcde'), dict(zip('abcde', range(5))), np.ones((5, 4)),
        np.full(5, 10.), np.ones(5), lengths, np.zeros((5, 5)),
        ((1,), (2, 3), (4,), (4,), ()), np.ones(5, dtype=bool), lengths / 10.,
        connection_lengths_m=junction_lengths or {})


def seed(vehicle_id='v', **updates):
    return original_seed(vehicle_id, destination_edge=4, **updates)


def route_policy(candidates, column=0, calls=None):
    def decide(rows, flow, now):
        if calls is not None:
            calls.append((now, tuple(row['vehicle_id'] for row in rows)))
        decisions = []
        for row in rows:
            route = candidates.generate_paths(row['current_edge'], row['destination_edge'])[column]
            decisions.append(RouteDecision(row['vehicle_id'], row['current_edge'], route[1], now, route))
        return decisions
    return decide


def advance(rows, candidates, *, start=0., seconds=30., decide=None, **kwargs):
    return advance_imagined_window(rows, candidates=candidates,
        edge_speeds_mps=np.full(candidates.network.edge_count, 10.),
        frozen_flow=np.zeros((candidates.network.edge_count, 8)), start_time=start,
        total_vehicle_count=max(1, len(rows)), interval_seconds=seconds,
        decide=decide or route_policy(candidates), session_protocol='event',
        traffic_model=candidates.traffic_model, **kwargs)


def test_routes_sharing_first_hop_follow_the_selected_complete_suffix():
    candidates = RouteCandidates(topology(), traffic_model='meso')
    paths = candidates.generate_paths(0, 4)
    assert paths[:2] == ((0, 1, 2, 4), (0, 1, 3, 4))
    direct, detour = [seed()], [seed()]
    first = advance(direct, candidates, seconds=3.)
    second = advance(detour, candidates, seconds=3., decide=route_policy(candidates, 1))
    assert direct[0]['current_edge'] == 4
    assert detour[0]['current_edge'] == 3
    assert detour[0]['committed_route'] == (3, 4)
    assert first.policy_events[0].next_edge == second.policy_events[0].next_edge == 1
    assert [(event.current_edge, event.next_edge, event.decision_time)
            for event in first.action_events] == [(1, 2, 1.)]
    assert [(event.current_edge, event.next_edge, event.decision_time)
            for event in second.action_events] == [(1, 3, 1.)]
    np.testing.assert_array_equal(second.action, [0., 0., 0., 1., 0.])


def test_policy_is_called_once_per_window_for_all_external_active_vehicles():
    candidates = RouteCandidates(topology((1000., 10., 10., 20., 1000.)), traffic_model='meso')
    rows = [seed('active', remaining_distance=1000.),
            seed('new', departed=False, departure_time=5., committed_next_edge=None,
                 remaining_distance=1000.)]
    calls = []
    first = advance(rows, candidates, decide=route_policy(candidates, calls=calls))
    assert calls == [(0., ('active',))]
    assert [event.vehicle_id for event in first.policy_events] == ['active']
    assert not first.action_events  # Both observations remain on a singleton road.
    restored = deepcopy(rows)
    second = advance(restored, candidates, start=30., decide=route_policy(candidates, calls=calls))
    assert calls == [(0., ('active',)), (30., ('active', 'new'))]
    assert len(second.policy_events) == 2
    assert restored[1]['committed_route'] == (0, 1, 2, 4)


def test_observed_boundary_turn_is_deferred_and_uses_the_new_route():
    candidates = RouteCandidates(topology((300., 1000., 10., 20., 1000.)), traffic_model='meso')
    rows = [seed(remaining_distance=300.)]
    first = advance(rows, candidates, event_index_start=7)
    assert not first.action_events
    assert rows[0]['current_edge'] == 1 and rows[0]['pending_turn']
    second = advance(deepcopy(rows), candidates, start=30., seconds=1.,
        decide=route_policy(candidates, 1), event_index_start=8)
    assert [(event.decision_time, event.event_index, event.current_edge, event.next_edge)
            for event in second.action_events] == [(30., 8, 1, 3)]
    assert second.policy_events[0].decision_time == 30.
    packed = pack_action_events([[first.action_events, second.action_events]], window_starts=[[0., 30.]])
    assert packed.phase.tolist() == [0.]
    assert packed.time_index.tolist() == [1]


def test_route_macro_decision_on_existing_fork_conditions_a_turn_at_phase_zero():
    candidates = RouteCandidates(topology((10., 1000., 10., 20., 1000.)), traffic_model='meso')
    rows = [seed(current_edge=1, remaining_distance=1000., committed_next_edge=3)]
    result = advance(rows, candidates, seconds=1., decide=route_policy(candidates, 1))
    assert result.policy_events[0].next_edge == 3
    assert result.action_events[0].next_edge == 3
    assert result.action_events[0].decision_time == 0.


def test_nonboundary_continuation_keeps_selected_suffix_and_only_publishes_pending_turn():
    candidates = RouteCandidates(topology(), traffic_model='meso')
    rows = [seed()]
    first = advance(rows, candidates, seconds=1., decide=route_policy(candidates, 1))
    assert not first.action_events
    assert rows[0]['committed_route'] == (1, 3, 4) and rows[0]['pending_turn']
    def forbidden(rows, flow, now):
        pytest.fail('a route policy callback occurred away from a 30-second boundary')
    result = advance(deepcopy(rows), candidates, start=1., seconds=2., decide=forbidden,
                     event_index_start=10)
    assert result.policy_events == ()
    assert [(event.decision_time, event.next_edge, event.event_index)
            for event in result.action_events] == [(1., 3, 10)]
    packed = pack_action_events([[result.action_events]], window_starts=[[0.]])
    assert packed.phase.tolist() == pytest.approx([1. / 30.])


def test_multiple_observed_turns_have_unique_indices_and_matching_dense_counts():
    candidates = RouteCandidates(topology((10., 1000., 10., 20., 1000.)), traffic_model='meso')
    rows = [seed('active', current_edge=1, remaining_distance=1000., committed_next_edge=2),
            seed('new', current_edge=1, remaining_distance=1000., departed=False,
                 departure_time=.25, committed_next_edge=None)]
    result = advance(rows, candidates, seconds=2., event_index_start=21)
    assert [(event.vehicle_id, event.decision_time, event.event_index)
            for event in result.action_events] == [('active', 0., 21), ('new', 1., 22)]
    assert [event.vehicle_id for event in result.policy_events] == ['active']
    np.testing.assert_array_equal(result.action, [0., 0., 1., 0., 0.])


def test_departure_on_fork_conditions_existing_route_without_policy_callback():
    candidates = RouteCandidates(topology((10., 1000., 10., 20., 1000.)), traffic_model='meso')
    rows = [seed(current_edge=1, remaining_distance=1000., departed=False,
                 departure_time=.25, committed_next_edge=None)]
    calls = []
    result = advance(rows, candidates, seconds=3., decide=route_policy(candidates, calls=calls),
                     event_index_start=11)
    assert not calls and result.policy_events == ()
    assert [(event.decision_time, event.event_index, event.current_edge, event.next_edge)
            for event in result.action_events] == [(1., 11, 1, 2)]
    assert result.departure[1] == 1.
    assert result.active_vehicle_seconds == pytest.approx(2.75)


def test_unobserved_fork_has_no_world_model_event_or_extra_policy_callback():
    candidates = RouteCandidates(topology((1., 1., 1., 1., 1000.)), traffic_model='meso')
    calls = []
    result = advance([seed(remaining_distance=1.)], candidates, seconds=2.,
        decide=route_policy(candidates, 1, calls))
    assert calls == [(0., ('v',))]
    assert len(result.policy_events) == 1
    assert not result.action_events
    np.testing.assert_array_equal(result.action, np.zeros(5))


def test_legacy_first_hop_seeds_recover_full_route_while_internal_vehicle_keeps_suffix():
    candidates = RouteCandidates(topology(junction_lengths={(0, 1): 1000.}))
    rows = [seed(in_junction=True, junction_remaining_distance=400., junction_speed=10.,
                 committed_route=(0, 1, 3, 4))]
    calls = []
    first = advance(rows, candidates, decide=route_policy(candidates, calls=calls))
    assert not calls and not first.policy_events
    assert rows[0]['committed_route'] == (0, 1, 3, 4)
    restored = deepcopy(rows)
    second = advance(restored, candidates, start=30., seconds=13.,
                     decide=route_policy(candidates, calls=calls))
    assert not calls and not second.policy_events
    assert [(event.decision_time, event.next_edge) for event in second.action_events] == [(40., 3)]
    assert restored[0]['current_edge'] == 4
    assert restored[0]['committed_route'] == (4,)
    legacy = [seed(in_junction=True, junction_remaining_distance=400., junction_speed=10.)]
    advance(legacy, candidates, seconds=1., decide=route_policy(candidates, calls=calls))
    assert legacy[0]['committed_route'] == (0, 1, 2, 4)


@pytest.mark.parametrize('route', [(1, 4), (0, 1, 2), (0, 2, 4), (0, 1, 99, 4)])
def test_invalid_committed_route_seed_is_rejected(route):
    candidates = RouteCandidates(topology(), traffic_model='meso')
    with pytest.raises(ValueError, match='committed route'):
        advance([seed(committed_route=route)], candidates, seconds=1.)


def test_inconsistent_seed_next_hop_is_rejected():
    candidates = RouteCandidates(topology(), traffic_model='meso')
    with pytest.raises(ValueError, match='seed next hop'):
        advance([seed(current_edge=1, committed_next_edge=3, committed_route=(1, 2, 4))],
                candidates, seconds=1.)


def test_meso_route_motion_skips_all_microscopic_junction_lengths():
    candidates = RouteCandidates(topology(junction_lengths={(0, 1): 5000., (1, 2): 5000.}),
                                 traffic_model='meso')
    rows = [seed()]
    result = advance(rows, candidates, seconds=2., decide=route_policy(candidates, 1))
    assert rows[0]['current_edge'] == 3
    assert not rows[0]['in_junction']
    assert rows[0]['junction_remaining_distance'] == rows[0]['junction_speed'] == 0.
    assert result.active_vehicle_seconds == 2.


def test_policy_route_must_be_an_exact_candidate_even_when_first_hop_is_legal():
    candidates = RouteCandidates(topology(), traffic_model='meso')
    def invalid(rows, flow, now):
        return [RouteDecision('v', 0, 1, now, (0, 1, 2))]
    with pytest.raises(ValueError, match='endpoints'):
        advance([seed()], candidates, seconds=1., decide=invalid)
