"""Event imagination shares the real session's observation/action clock."""
from dataclasses import replace

import numpy as np
import pytest

from test_urban_dreamer_event_session import synthetic_network
from urban_dreamer.learning.imagination import advance_imagined_window
from urban_dreamer.traffic.environment import NextHopDecision
from urban_dreamer.traffic.topology import NextHopCandidates


def network(lengths=(10., 10., 10., 1000.), graph=None, junction_lengths=None):
    base = synthetic_network()
    graph = base.adjacency_list if graph is None else graph
    return replace(base, lengths_m=np.array(lengths), adjacency_list=graph,
        free_flow_travel_seconds=np.array(lengths) / 10.,
        connection_lengths_m=junction_lengths or {})


def seed(vehicle_id='v', **updates):
    row = dict(vehicle_id=vehicle_id, current_edge=0, destination_edge=3,
        departure_time=0., departed=True, arrived=False, remaining_distance=10.,
        committed_next_edge=1, pending_decision=False, in_junction=False,
        junction_remaining_distance=0., junction_speed=1.)
    row.update(updates)
    return row


def policy(rows, flow, now):
    return [NextHopDecision(row['vehicle_id'], row['current_edge'],
        3 if row['current_edge'] == 1 else 1, now) for row in rows]


def advance(rows, topology=None, start=0., seconds=30., **kwargs):
    topology = topology or network()
    return advance_imagined_window(rows, candidates=NextHopCandidates(topology),
        edge_speeds_mps=np.ones(topology.edge_count) * 10.,
        frozen_flow=np.zeros((topology.edge_count, 8)), start_time=start,
        total_vehicle_count=max(1, len(rows)), decide=policy,
        session_protocol='event', interval_seconds=seconds, **kwargs)


def test_singleton_route_is_automatic_and_only_observed_fork_decides():
    rows = [seed()]
    window = advance(rows, seconds=3.)
    assert [(item.current_edge, item.next_edge, item.decision_time) for item in window.decisions] == [(1, 3, 1.)]
    assert len(window.action_events) == 1
    event = window.action_events[0]
    assert (event.vehicle_id, event.event_index, event.destination_edge) == ('v', 0, 3)
    np.testing.assert_array_equal(window.action, [0., 0., 0., 1.])


def test_short_fork_crossed_within_observation_has_no_action():
    topology = network((1., 1., 1., 1000.))
    rows = [seed(remaining_distance=1.)]
    window = advance(rows, topology, seconds=1.)
    assert not window.decisions and not window.action_events
    assert rows[0]['current_edge'] == 3
    assert rows[0]['remaining_distance'] == pytest.approx(992.)


def test_observation_of_same_edge_after_unseen_loop_does_not_decide_again():
    topology = network((1., 10., 2., 1000.))
    # The committed detour returns to b inside this second, but neither b nor
    # c was a new observed edge. Its deterministic suffix then heads to d.
    rows = [seed(current_edge=1, remaining_distance=2., committed_next_edge=2)]
    window = advance(rows, topology, seconds=1.)
    assert not window.decisions
    assert rows[0]['current_edge'] == 1
    assert rows[0]['remaining_distance'] == pytest.approx(4.)


def test_return_to_last_external_edge_via_observed_internal_lane_decides():
    topology = network(junction_lengths={(2, 1): 20.})
    rows = [seed(current_edge=2, remaining_distance=0., committed_next_edge=1,
        in_junction=True, junction_remaining_distance=5., junction_speed=10.)]
    first = advance(rows, topology, seconds=1.)
    assert not first.decisions
    assert rows[0]['current_edge'] == 1 and rows[0]['pending_decision']
    second = advance(rows, topology, start=1., seconds=1.)
    assert second.decisions[0].decision_time == 1.


def test_same_vehicle_multiple_observed_events_keep_distinct_indices():
    rows = [seed(current_edge=1, remaining_distance=10., committed_next_edge=3, pending_decision=True)]
    calls = []
    def detour(rows, flow, now):
        calls.append(now)
        target = 2 if len(calls) == 1 else 3
        return [NextHopDecision(row['vehicle_id'], row['current_edge'], target, now) for row in rows]
    result = advance_imagined_window(rows, candidates=NextHopCandidates(network()),
        edge_speeds_mps=np.ones(4) * 10., frozen_flow=np.zeros((4, 8)),
        start_time=0., interval_seconds=4., total_vehicle_count=1, decide=detour,
        session_protocol='event', event_index_start=7)
    assert [event.decision_time for event in result.action_events] == [0., 2.]
    assert [event.event_index for event in result.action_events] == [7, 8]
    assert {event.vehicle_id for event in result.action_events} == {'v'}


def test_boundary_observation_defers_action_and_preserves_commitment():
    rows = [seed(remaining_distance=300.)]
    first = advance(rows)
    assert not first.decisions
    assert rows[0]['current_edge'] == 1 and rows[0]['pending_decision']
    assert rows[0]['committed_next_edge'] == 3
    second = advance(rows, start=30., event_index_start=4)
    assert second.action_events[0].decision_time == 30.
    assert second.action_events[0].event_index == 4


def test_long_junction_keeps_progress_and_commitment_across_windows():
    topology = network(junction_lengths={(0, 1): 400.})
    rows = [seed()]
    first = advance(rows, topology)
    assert not first.decisions and rows[0]['in_junction']
    assert rows[0]['committed_next_edge'] == 1
    assert rows[0]['junction_remaining_distance'] == pytest.approx(110.)
    second = advance(rows, topology, start=30.)
    assert second.decisions[0].decision_time == 41.


def test_departure_cost_uses_actual_time_and_arrival_is_observation_aligned():
    topology = network((1., 1., 1., 1.))
    rows = [seed(departure_time=.25, departed=False, remaining_distance=1., committed_next_edge=None)]
    result = advance(rows, topology, seconds=2., stop_when_empty=True)
    assert result.arrivals == {'v': 1.}
    assert result.active_vehicle_seconds == pytest.approx(.75)
    assert result.end_time == 1.
    assert rows[0]['observed_departure_time'] == 1.
    assert rows[0]['actual_departure_time'] == .25
    assert not result.decisions
    assert sum((part.end_time-part.start_time)*part.active_count for part in result.cost_segments) == .75


def test_arrival_in_middle_of_step_charges_until_observation():
    rows = [seed(current_edge=3, remaining_distance=2., committed_next_edge=None)]
    result = advance(rows, seconds=2., stop_when_empty=True)
    assert result.arrivals == {'v': 1.}
    assert result.active_vehicle_seconds == 1.


def test_departure_at_boundary_is_observed_in_prior_window_and_decides_next():
    rows = [seed(current_edge=1, departure_time=30., departed=False,
        committed_next_edge=None, remaining_distance=10.)]
    first = advance(rows)
    assert first.departure[1] == 1.
    assert first.active_vehicle_seconds == 0.
    assert not first.decisions and rows[0]['pending_decision']
    second = advance(rows, start=30.)
    assert second.decisions[0].decision_time == 30.


@pytest.mark.parametrize('updates', [dict(committed_next_edge=None),
    dict(in_junction=True, committed_next_edge=None, junction_remaining_distance=10.)])
def test_missing_nonpending_commitment_is_explicit_error(updates):
    with pytest.raises(ValueError, match='committed'):
        advance([seed(**updates)])


def test_event_protocol_requires_one_second_observations():
    with pytest.raises(ValueError, match='one-second'):
        advance([seed()], observation_step_seconds=.001)


def test_initial_first_hop_keeps_distinct_cost_rule():
    topology = network((10., 10., 10., 1000.), graph=((1, 2), (3,), (3,), ()),
        junction_lengths={(0, 1): 1000.})
    rows = [seed(departed=False, committed_next_edge=None)]
    # The initial route chooses 1 by downstream cost/tie, ignoring its long
    # first junction; its policy decision becomes pending at the first tick.
    result = advance(rows, topology, seconds=1.)
    assert not result.decisions
    assert rows[0]['committed_next_edge'] == 1
