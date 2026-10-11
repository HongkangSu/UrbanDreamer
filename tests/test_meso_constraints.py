from copy import deepcopy

import numpy as np
import pytest

from test_event_imagination_protocol import network, seed, policy
from urban_dreamer.learning.imagination import advance_imagined_window
from urban_dreamer.learning.meso_constraints import MesoReleaseConstraint, StaticSignalSchedule, UnsupportedConstraint
from urban_dreamer.traffic.topology import NextHopCandidates


class Signals:
    def __init__(self, release=100.):
        self.release = release

    def permits(self, edge, successor, timestamp):
        return (edge, successor) != (1, 3) or timestamp >= self.release


def advance(states, constraint=None, *, seconds=1., start=0., speed=10.):
    graph = network()
    return advance_imagined_window(states, candidates=NextHopCandidates(graph, traffic_model='meso'),
        edge_speeds_mps=np.full(4, speed), frozen_flow=np.zeros((4, 8)), start_time=start,
        interval_seconds=seconds, total_vehicle_count=len(states), decide=policy,
        session_protocol='event', traffic_model='meso', passage_constraint=constraint)


def queue():
    return [seed('z-front', current_edge=1, remaining_distance=0., committed_next_edge=3),
            seed('a-rear', current_edge=1, remaining_distance=2., committed_next_edge=2)]


def test_positive_speed_red_head_blocks_green_rear_without_vehicle_id_order():
    rows = queue()
    gate = MesoReleaseConstraint(Signals())
    result = advance(rows, gate, seconds=3.)
    assert [row['current_edge'] for row in rows] == [1, 1]
    assert [row['remaining_distance'] for row in rows] == [0., 0.]
    assert result.active_vehicle_seconds == 6.
    assert gate.blocked_signal and gate.blocked_fifo
    assert not result.departure.any()
    # Queue order survives equal positions created by our own zero-gap model.
    advance(rows, gate, seconds=1., start=3.)
    assert [row['current_edge'] for row in rows] == [1, 1]


def test_green_release_does_not_deadlock_after_red_queue():
    rows = queue()
    gate = MesoReleaseConstraint(Signals(release=2.))
    result = advance(rows, gate, seconds=4.)
    assert rows[0]['current_edge'] == 3
    assert rows[1]['current_edge'] != 1
    assert result.active_vehicle_seconds == 8.


def test_destination_arrival_cannot_bypass_blocked_head():
    rows = queue()
    rows[1].update(destination_edge=1, committed_next_edge=None)
    result = advance(rows, MesoReleaseConstraint(Signals()), seconds=2.)
    assert not rows[1]['arrived']
    assert result.arrivals == {} and result.active_vehicle_seconds == 4.


def test_all_green_nonqueue_matches_default_and_partial_window_cost():
    baseline = [seed('one', current_edge=1, remaining_distance=4., committed_next_edge=3)]
    constrained = deepcopy(baseline)
    before = advance(baseline, seconds=3.)
    after = advance(constrained, MesoReleaseConstraint(Signals(release=0.)), seconds=3.)
    assert constrained == baseline
    np.testing.assert_array_equal(before.action, after.action)
    assert before.cost_segments == after.cost_segments
    assert before.active_vehicle_seconds == after.active_vehicle_seconds == 3.
    assert before.action_events == after.action_events


def test_stopped_road_exposure_and_future_birth_rejection():
    rows = [seed(remaining_distance=10.)]
    result = advance(rows, MesoReleaseConstraint(Signals()), seconds=3., speed=0.)
    assert rows[0]['remaining_distance'] == 10.
    assert result.active_vehicle_seconds == 3.
    with pytest.raises(UnsupportedConstraint, match='future births'):
        advance([seed(departed=False)], MesoReleaseConstraint(Signals()))


def test_tied_seed_needs_authoritative_order():
    rows = queue()
    rows[1]['remaining_distance'] = 0.
    with pytest.raises(UnsupportedConstraint, match='authoritative FIFO'):
        advance(rows, MesoReleaseConstraint(Signals()))
    advance(rows, MesoReleaseConstraint(Signals(), fifo_order={1: ['z-front', 'a-rear']}))
    assert all(row['current_edge'] == 1 for row in rows)


def test_multiple_upstream_merge_is_not_silently_arbitrated():
    states = [seed('left', current_edge=0, remaining_distance=1.),
              seed('right', current_edge=2, remaining_distance=1., committed_next_edge=1)]
    with pytest.raises(UnsupportedConstraint, match='multi-upstream'):
        advance(states, MesoReleaseConstraint(Signals()))


def xml_schedule(tmp_path, *, offset='0', first='3', attrs='', state1='r', state2='G', links=None):
    path = tmp_path/'signals.xml'
    path.write_text(f'<net><tlLogic id="tls" type="static" offset="{offset}">'
        f'<phase duration="{first}" state="{state1}" {attrs}/><phase duration="2" state="{state2}"/>'
        '</tlLogic>' + (links or '<connection from="a" to="b" tl="tls" linkIndex="0"/>') + '</net>')
    return path


def test_static_signal_offset_and_half_open_tick_boundaries(tmp_path):
    path = xml_schedule(tmp_path, offset='1')
    signal = StaticSignalSchedule.from_network(path, ['a', 'b'], no_external_overrides=True)
    assert signal.permits(0, 1, 0.)  # (0 - 1) modulo 5 == 4: green
    assert not signal.permits(0, 1, 1.)
    assert not signal.permits(0, 1, 3.999)
    assert signal.permits(0, 1, 4.)
    assert signal.permits(1, 0, 2.)  # uncontrolled connection


@pytest.mark.parametrize('updates', [dict(first='3.5'), dict(offset='.5'), dict(attrs='next="1"'), dict(state2='g')])
def test_static_provider_rejects_unmodeled_phase_semantics(tmp_path, updates):
    path = xml_schedule(tmp_path, **updates)
    with pytest.raises(UnsupportedConstraint):
        StaticSignalSchedule.from_network(path, ['a', 'b'], no_external_overrides=True)


def test_external_programs_and_conflicting_lane_permissions_rejected(tmp_path):
    path = xml_schedule(tmp_path, state1='Gr', state2='rG', links=(
        '<connection from="a" to="b" tl="tls" linkIndex="0"/>'
        '<connection from="a" to="b" tl="tls" linkIndex="1"/>'))
    with pytest.raises(UnsupportedConstraint, match='overrides'):
        StaticSignalSchedule.from_network(path, ['a', 'b'])
    signal = StaticSignalSchedule.from_network(path, ['a', 'b'], no_external_overrides=True)
    with pytest.raises(UnsupportedConstraint, match='lane-dependent'):
        signal.permits(0, 1, 0.)


def test_empty_static_state_without_controlled_connections_is_rejected(tmp_path):
    path = tmp_path/'empty.xml'
    path.write_text('<net><tlLogic id="empty" type="static"><phase duration="1" state=""/></tlLogic></net>')
    with pytest.raises(UnsupportedConstraint, match='nonzero width'):
        StaticSignalSchedule.from_network(path, ['a', 'b'], no_external_overrides=True)


def test_actual_early_terminal_window_cost_excludes_unused_tail():
    graph = network((1., 1., 1., 1.))

    def run(constraint, remaining):
        return advance_imagined_window([seed(remaining_distance=remaining)],
            candidates=NextHopCandidates(graph, traffic_model='meso'),
            edge_speeds_mps=np.full(4, 10.), frozen_flow=np.zeros((4, 8)),
            start_time=0., interval_seconds=5., total_vehicle_count=1, decide=policy,
            session_protocol='event', traffic_model='meso', stop_when_empty=True,
            passage_constraint=constraint)

    original = run(None, 1.)
    green = run(MesoReleaseConstraint(Signals(release=0.)), 1.)
    assert original.arrivals == green.arrivals == {'v': 1.}
    assert original.end_time == green.end_time == 1.
    assert original.active_vehicle_seconds == green.active_vehicle_seconds == 1.
    assert original.cost_segments == green.cost_segments
    released = run(MesoReleaseConstraint(Signals(release=2.)), 0.)
    assert released.arrivals == {'v': 3.}
    assert released.end_time == released.active_vehicle_seconds == 3.
