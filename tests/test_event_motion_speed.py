"""Keep routing and observed exposure exact when motion bookkeeping is fused."""
import numpy as np

from test_event_imagination_protocol import network, seed
from urban_dreamer.learning.imagination import ActiveCostSegment, advance_imagined_window
from urban_dreamer.traffic.environment import NextHopDecision
from urban_dreamer.traffic.topology import NextHopCandidates


def test_cached_route_keeps_initial_and_suffix_rules_separate():
    topology = network((10., 10., 1., 10.), graph=((1,), (2, 3), (3,), ()),
                       junction_lengths={(1, 3): 100.})
    rows = [seed('active', current_edge=1, remaining_distance=100.,
                 committed_next_edge=None, pending_decision=True),
            seed('new', current_edge=1, remaining_distance=100., departed=False,
                 departure_time=.25, committed_next_edge=None)]

    def decide(states, flow, now):
        # The suffix rule includes junction time and therefore prefers edge 2.
        assert states[0]['committed_next_edge'] == 2
        return [NextHopDecision(row['vehicle_id'], row['current_edge'], 2, now)
                for row in states]

    window = advance_imagined_window(rows, candidates=NextHopCandidates(topology),
        edge_speeds_mps=np.ones(4) * 10., frozen_flow=np.zeros((4, 8)),
        start_time=0., total_vehicle_count=2, decide=decide,
        session_protocol='event', interval_seconds=1.)
    assert rows[0]['committed_next_edge'] == 2
    # Initial insertion omits the first junction cost, even for the cached pair.
    assert rows[1]['committed_next_edge'] == 3
    assert rows[1]['pending_decision']
    assert len(window.action_events) == 1
    assert window.active_vehicle_seconds == 1.75


def test_fused_observation_charges_old_arrivals_and_fractional_departures():
    rows = [seed('old', current_edge=3, remaining_distance=1., committed_next_edge=None),
            seed('new_arrival', current_edge=3, remaining_distance=1., departed=False,
                 departure_time=.25, committed_next_edge=None),
            seed('new_active', remaining_distance=100., departed=False,
                 departure_time=.5, committed_next_edge=None),
            seed('boundary', current_edge=1, remaining_distance=100., departed=False,
                 departure_time=1., committed_next_edge=None),
            seed('future', departed=False, departure_time=2., committed_next_edge=None)]
    window = advance_imagined_window(rows, candidates=NextHopCandidates(network()),
        edge_speeds_mps=np.ones(4) * 10., frozen_flow=np.zeros((4, 8)),
        start_time=0., total_vehicle_count=5, decide=lambda *_: (),
        session_protocol='event', interval_seconds=1.)
    assert window.arrivals == {'new_arrival': 1., 'old': 1.}
    assert window.cost_segments == (ActiveCostSegment(0., .25, 1),
                                   ActiveCostSegment(.25, .5, 2),
                                   ActiveCostSegment(.5, 1., 3))
    assert window.active_vehicle_seconds == 2.25
    assert not window.action_events
    assert rows[3]['pending_decision']
    assert not rows[4]['departed']
    np.testing.assert_array_equal(window.departure, np.array([1., 1., 0., 1.], dtype=np.float32) / 5.)


def test_vector_motion_preserves_repeated_rounding_and_callback_visible_distances():
    rows = [seed('actor', current_edge=1, remaining_distance=10.,
                 committed_next_edge=2, pending_decision=True),
            seed('external', current_edge=3, remaining_distance=100.1,
                 committed_next_edge=None),
            seed('junction', remaining_distance=0., in_junction=True,
                 junction_remaining_distance=100.1, junction_speed=.1)]
    callback_times = []

    def repeated_distance(seconds):
        distance = 100.1
        for _ in range(seconds):
            distance -= .1
        return distance

    def decide(states, flow, now):
        callback_times.append(now)
        assert rows[1]['remaining_distance'] == repeated_distance(int(now))
        assert rows[2]['junction_remaining_distance'] == repeated_distance(int(now))
        return [NextHopDecision(row['vehicle_id'], row['current_edge'], 2, now)
                for row in states]

    result = advance_imagined_window(rows, candidates=NextHopCandidates(network()),
        edge_speeds_mps=np.array([10., 10., 10., .1]), frozen_flow=np.zeros((4, 8)),
        start_time=0., total_vehicle_count=3, decide=decide,
        session_protocol='event', interval_seconds=30.)
    assert len(callback_times) > 2
    assert rows[1]['remaining_distance'] == repeated_distance(30)
    assert rows[2]['junction_remaining_distance'] == repeated_distance(30)
    assert result.active_vehicle_seconds == 90.
    assert result.end_time == 30.


def test_static_route_cache_is_reused_across_windows_bounded_and_network_local():
    candidates = NextHopCandidates(network())
    candidates._event_route_cache_limit = 1
    candidates._event_route_cache[(99, 99, False)] = 0
    original = candidates.costs
    calls = []
    def counted(destination):
        calls.append(destination)
        return original(destination)
    candidates.costs = counted
    for start in (0., 30.):
        rows = [seed('a', current_edge=1, remaining_distance=100.,
                     committed_next_edge=None, pending_decision=True)]
        advance_imagined_window(rows, candidates=candidates,
            edge_speeds_mps=np.ones(4), frozen_flow=np.zeros((4, 8)), start_time=start,
            total_vehicle_count=1, session_protocol='event', interval_seconds=1.,
            decide=lambda states, flow, now: [NextHopDecision('a', 1, 2, now)])
        assert len(candidates._event_route_cache) == 1
        assert (99, 99, False) not in candidates._event_route_cache
    assert len(calls) == 1
    assert not NextHopCandidates(network())._event_route_cache
