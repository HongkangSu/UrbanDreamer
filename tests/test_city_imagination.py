"""Causal controls preserve the policy clock without claiming SUMO queue physics."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from test_event_imagination_protocol import network, policy, seed
from urban_dreamer.learning.city_imagination import (
    ChronologicalJunctionArbiter, advance_city_imagined_window,
)
from urban_dreamer.learning.imagination import advance_imagined_window
from urban_dreamer.traffic.movement_controls import CityControlTable, MovementControl, _Program, UnsupportedControl
from urban_dreamer.traffic.topology import NextHopCandidates


def table(topology, *, overrides=None, phases=None):
    overrides = overrides or {}
    movements = {}
    for source, targets in enumerate(topology.adjacency_list):
        for target in targets:
            movement = MovementControl(source, target, f'j{source}', 'priority', 0,
                None, None, 'M', frozenset(), frozenset(), 0, 0)
            movements[source, target] = replace(movement, **overrides.get((source, target), {}))
    return CityControlTable(movements, {} if phases is None else {'tls': _Program(0, tuple(phases))}, {})


def advance(rows, topology=None, *, arbiter=None, start=0., seconds=1., speeds=None,
            callback=policy, forbid_revisit=False, **kwargs):
    topology = topology or network()
    arbiter = arbiter or ChronologicalJunctionArbiter(table(topology), seed=17)
    result = advance_city_imagined_window(rows,
        candidates=NextHopCandidates(topology, traffic_model='meso', forbid_revisit=forbid_revisit),
        edge_speeds_mps=np.full(topology.edge_count, 10.) if speeds is None else speeds,
        frozen_flow=np.zeros((topology.edge_count, 8)), start_time=start,
        interval_seconds=seconds, total_vehicle_count=max(1, len(rows)),
        decide=callback, junction_arbiter=arbiter, **kwargs)
    return result, arbiter


def crossings(arbiter):
    return [event for event in arbiter.audit if event['event'] == 'crossing']


@pytest.mark.parametrize('reverse,rename', [(False, False), (True, False), (False, True), (True, True)])
def test_unequal_time_merge_is_chronological_not_id_or_input_order(reverse, rename):
    topology = network((10., 10., 100., 100.), graph=((2,), (2,), (3,), ()))
    late_id, early_id = ('a', 'z') if not rename else ('z', 'a')
    rows = [seed(late_id, remaining_distance=8., committed_next_edge=2),
            seed(early_id, current_edge=1, remaining_distance=2., committed_next_edge=2)]
    if reverse:
        rows.reverse()
    result, arbiter = advance(rows, topology, seconds=2.)
    events = crossings(arbiter)
    assert [(event['vehicle_id'], event['time']) for event in events] == [(early_id, .2), (late_id, 1.2)]
    assert result.active_vehicle_seconds == 4.
    assert [s['vehicle_id'] for s in arbiter.queues[2]] == [early_id, late_id]


def conflict_case(*, minor_state='g', major_distance=0., minor_distance=0.):
    topology = network((10., 10., 100., 100.), graph=((2,), (3,), (), ()))
    controls = table(topology, overrides={
        (0, 2): dict(junction_id='j', junction_type='traffic_light', request_index=0,
                     tls_id='tls', tls_index=0, yields_to=frozenset({1}), conflicts_with=frozenset({1})),
        (1, 3): dict(junction_id='j', junction_type='traffic_light', request_index=1,
                     tls_id='tls', tls_index=1, conflicts_with=frozenset({0}))},
        phases=[(100, minor_state + 'G')])
    rows = [seed('minor', destination_edge=2, remaining_distance=minor_distance, committed_next_edge=2),
            seed('major', current_edge=1, remaining_distance=major_distance, committed_next_edge=3)]
    return topology, controls, rows


@pytest.mark.parametrize('minor_state', ['g', 'y'])
def test_permissive_and_yellow_yield_to_conflicting_protected_movement(minor_state):
    topology, controls, rows = conflict_case(minor_state=minor_state)
    _, arbiter = advance(rows, topology, seconds=2., arbiter=ChronologicalJunctionArbiter(controls, seed=4))
    assert [(e['vehicle_id'], e['time']) for e in crossings(arbiter)] == [('major', 0.), ('minor', 1.)]


def test_approaching_major_head_is_considered_before_it_reaches_stop_line():
    topology, controls, rows = conflict_case(major_distance=5.)
    _, arbiter = advance(rows, topology, seconds=2., arbiter=ChronologicalJunctionArbiter(controls, seed=4))
    assert [(e['vehicle_id'], e['time']) for e in crossings(arbiter)] == [('major', .5), ('minor', 1.5)]


def test_nonconflicting_movements_cross_simultaneously():
    topology, controls, rows = conflict_case()
    controls = CityControlTable({pair: replace(m, conflicts_with=frozenset(), yields_to=frozenset())
                                for pair, m in controls.movements.items()}, controls.programs, {})
    _, arbiter = advance(rows, topology, arbiter=ChronologicalJunctionArbiter(controls, seed=4))
    assert sorted((e['vehicle_id'], e['time']) for e in crossings(arbiter)) == [('major', 0.), ('minor', 0.)]


def test_red_front_blocks_green_rear_and_fifo_persists_across_windows():
    topology = network((10., 100., 100., 100.), graph=((1, 2), (3,), (3,), ()))
    controls = table(topology, overrides={
        (0, 1): dict(tls_id='tls', tls_index=0), (0, 2): dict(tls_id='tls', tls_index=1)},
        phases=[(2, 'rG'), (100, 'GG')])
    rows = [seed('front', remaining_distance=0., committed_next_edge=1),
            seed('rear', remaining_distance=1., committed_next_edge=2)]
    arbiter = ChronologicalJunctionArbiter(controls, seed=4)
    first, _ = advance(rows, topology, arbiter=arbiter)
    assert not crossings(arbiter) and first.active_vehicle_seconds == 2.
    rows.reverse()
    second, _ = advance(rows, topology, arbiter=arbiter, start=1., seconds=2.)
    assert [(e['vehicle_id'], e['time']) for e in crossings(arbiter)] == [('front', 2.), ('rear', 3.)]
    assert second.active_vehicle_seconds == 4.


@pytest.mark.parametrize('start,duration', [(0., 1.), (0., 7.), (3., 4.)])
def test_deadline_tail_and_zero_speed_charge_active_cohort(start, duration):
    rows = [seed(remaining_distance=5.)]
    result, _ = advance(rows, start=start, seconds=duration, speeds=np.zeros(4), stop_when_empty=True)
    assert result.end_time == start + duration
    assert result.active_vehicle_seconds == duration
    assert not result.arrivals and rows[0]['remaining_distance'] == 5.


@pytest.mark.parametrize('lengths,seconds', [((10., 10., 10., 1000.), 5.), ((1., 1., 1., 1.), 3.)])
def test_all_green_noninteracting_path_matches_legacy_observations_and_cost(lengths, seconds):
    topology = network(lengths)
    original = [seed(remaining_distance=lengths[0])]
    city = deepcopy(original)
    expected = advance_imagined_window(original,
        candidates=NextHopCandidates(topology, traffic_model='meso'), edge_speeds_mps=np.full(4, 10.),
        frozen_flow=np.zeros((4, 8)), start_time=0., interval_seconds=seconds,
        total_vehicle_count=1, decide=policy, traffic_model='meso', session_protocol='event', stop_when_empty=True)
    actual, _ = advance(city, topology, seconds=seconds, stop_when_empty=True)
    assert actual.arrivals == expected.arrivals
    assert actual.decisions == expected.decisions
    assert actual.action_events == expected.action_events
    assert actual.active_vehicle_seconds == expected.active_vehicle_seconds
    assert actual.end_time == expected.end_time
    np.testing.assert_equal(actual.action, expected.action)
    assert city[0]['current_edge'] == original[0]['current_edge']
    assert city[0]['remaining_distance'] == pytest.approx(original[0]['remaining_distance'])


def test_arrival_observation_and_cost_wait_until_next_tick():
    rows = [seed(current_edge=3, remaining_distance=2., committed_next_edge=None)]
    result, arbiter = advance(rows, seconds=8., stop_when_empty=True)
    assert result.arrivals == {'v': 1.} and result.end_time == 1.
    assert result.active_vehicle_seconds == 1.
    assert [e['time'] for e in arbiter.audit if e['event'] == 'physical_arrival'] == [.2]


def test_endpoint_decision_remains_pending_until_next_window():
    rows = [seed(remaining_distance=10.)]
    first, arbiter = advance(rows)
    assert not first.decisions and rows[0]['pending_decision']
    second, _ = advance(rows, arbiter=arbiter, start=1., event_index_start=8)
    assert [(event.decision_time, event.event_index) for event in second.action_events] == [(1., 8)]


def test_rejects_future_births_before_any_physical_mutation():
    rows = [seed(), seed('future', departed=False, departure_time=500.)]
    before = deepcopy(rows)
    with pytest.raises(ValueError, match='no future births'):
        advance(rows)
    assert rows == before


def test_departure_schedule_is_not_consulted_for_active_cohort():
    first, second = [seed(departure_time=0.)], [seed(departure_time=1e20)]
    a, _ = advance(first, seconds=3.)
    b, _ = advance(second, seconds=3.)
    assert a.decisions == b.decisions and a.active_vehicle_seconds == b.active_vehicle_seconds
    assert first[0]['remaining_distance'] == second[0]['remaining_distance']


def test_unknown_and_unsupported_controls_fail_explicitly():
    topology = network()
    controls = table(topology, overrides={(0, 1): dict(support='unsupported', unsupported_reason='actuated TLS')})
    with pytest.raises(UnsupportedControl, match='actuated TLS'):
        advance([seed()], arbiter=ChronologicalJunctionArbiter(controls, seed=1))
    with pytest.raises(UnsupportedControl, match='unregistered movement'):
        advance([seed()], arbiter=ChronologicalJunctionArbiter(CityControlTable({}, {}, {}), seed=1))


def test_tied_positions_require_authority_or_explicit_seeded_uncertainty():
    rows = [seed('a'), seed('z')]
    with pytest.raises(ValueError, match='tied positions'):
        advance(rows)
    for state, order in zip(rows, (1, 0)):
        state['city_queue_order'] = order
    _, arbiter = advance(rows)
    assert crossings(arbiter)[0]['vehicle_id'] == 'z'
    assert arbiter.audit[0]['kind'] == 'authoritative'
    runs = []
    for reverse in (False, True):
        rows = [seed('a'), seed('z')]
        if reverse:
            rows.reverse()
        _, arbiter = advance(rows, arbiter=ChronologicalJunctionArbiter(table(network()), seed=24, uncertain_order=True))
        runs.append(crossings(arbiter))
        assert arbiter.audit[0]['kind'] == 'seeded_uncertain_order'
    assert runs[0] == runs[1]


def test_equal_time_merge_uses_logged_reproducible_fairness():
    topology = network((10., 10., 100., 100.), graph=((2,), (2,), (3,), ()))
    runs = []
    for reverse in (False, True):
        rows = [seed('a', remaining_distance=0., committed_next_edge=2),
                seed('b', current_edge=1, remaining_distance=0., committed_next_edge=2)]
        if reverse:
            rows.reverse()
        _, arbiter = advance(rows, topology, seconds=2.)
        runs.append(crossings(arbiter))
        assert any(e['event'] == 'equal_time_tie' for e in arbiter.audit)
    assert runs[0] == runs[1]
    assert [e['time'] for e in runs[0]] == [0., 1.]


def test_right_before_left_cycle_has_logged_break_and_eventual_progress():
    topology, controls, rows = conflict_case()
    controls = CityControlTable({pair: replace(m, junction_type='right_before_left',
        tls_id=None, tls_index=None, off_state='=', yields_to=m.conflicts_with)
        for pair, m in controls.movements.items()}, {}, {})
    _, arbiter = advance(rows, topology, seconds=2., arbiter=ChronologicalJunctionArbiter(controls, seed=2))
    assert len(crossings(arbiter)) == 2
    assert [e['time'] for e in crossings(arbiter)] == [0., 1.]
    assert len([e for e in arbiter.audit if e['event'] == 'priority_cycle_break']) == 1


def test_no_revisit_and_runtime_route_audit_records_hidden_entries(monkeypatch):
    import urban_dreamer.learning.imagination as runtime
    original, entered = runtime._record_physical_entry, []
    def record(state, edge, candidates):
        entered.append(edge)
        original(state, edge, candidates)
    monkeypatch.setattr(runtime, '_record_physical_entry', record)
    topology = network((1., 1., 1., 1.), graph=((1,), (2,), (3,), ()))
    rows = [seed(remaining_distance=1., visited_edges=(0,))]
    result, _ = advance(rows, topology, forbid_revisit=True)
    assert result.arrivals == {'v': 1.}
    assert rows[0]['visited_edges'] == (0, 1, 2, 3)
    assert entered == [1, 2, 3]


def test_illegal_policy_action_is_rejected_before_commitment():
    rows = [seed(current_edge=1, remaining_distance=10., committed_next_edge=3, pending_decision=True)]
    from urban_dreamer.traffic.environment import NextHopDecision
    def bad_policy(rows, flow, now):
        return [NextHopDecision('v', 1, 0, now)]
    with pytest.raises(ValueError, match='legal candidate'):
        advance(rows, callback=bad_policy)
    assert rows[0]['committed_next_edge'] == 3


def test_new_window_cannot_reset_reservations_or_change_cohort():
    rows = [seed()]
    _, arbiter = advance(rows)
    with pytest.raises(ValueError, match='same cohort and contiguous'):
        advance(rows, arbiter=arbiter, start=4.)


def test_priority_cycle_break_cannot_ignore_another_downstream_cycle():
    movements = {}
    dependencies = {0: {1, 2}, 1: {0}, 2: {3}, 3: {2}}
    for source in range(4):
        movements[source, 4] = MovementControl(source, 4, 'junction', 'right_before_left', source,
            None, None, '=', frozenset(dependencies[source]), frozenset(set(range(4)) - {source}), 0, 0)
    controls = CityControlTable(movements, {}, {})
    rows = [seed(chr(ord('a') + edge), current_edge=edge, destination_edge=4,
                 remaining_distance=0., committed_next_edge=4) for edge in range(4)]
    arbiter = ChronologicalJunctionArbiter(controls, seed=1)
    arbiter.prepare(rows, 0.)
    grants = arbiter.grants(0., np.ones(5))
    assert len(grants) == 1
    assert grants[0][0]['vehicle_id'] in {'c', 'd'}
    assert arbiter.audit[-1]['contenders'] == ['c', 'd']


def test_edge_renumbering_cannot_reverse_unequal_time_merge():
    for mapping in ({0: 0, 1: 1, 2: 2, 3: 3}, {0: 3, 1: 2, 2: 1, 3: 0}):
        graph, lengths = [None] * 4, [None] * 4
        for edge, targets in enumerate(((2,), (2,), (3,), ())):
            graph[mapping[edge]] = tuple(mapping[target] for target in targets)
            lengths[mapping[edge]] = (10., 10., 100., 100.)[edge]
        topology = network(tuple(lengths), graph=tuple(graph))
        rows = [seed('late', current_edge=mapping[0], destination_edge=mapping[3],
                     remaining_distance=8., committed_next_edge=mapping[2]),
                seed('early', current_edge=mapping[1], destination_edge=mapping[3],
                     remaining_distance=2., committed_next_edge=mapping[2])]
        _, arbiter = advance(rows, topology, seconds=2.)
        assert [(event['vehicle_id'], event['time']) for event in crossings(arbiter)] == [('early', .2), ('late', 1.2)]


def test_split_windows_keep_fractional_receiving_reservations_with_copied_states():
    topology = network((10., 10., 100., 100.), graph=((2,), (2,), (3,), ()))
    rows = [seed('late', remaining_distance=8., committed_next_edge=2),
            seed('early', current_edge=1, remaining_distance=2., committed_next_edge=2)]
    continuous = deepcopy(rows)
    complete, expected = advance(continuous, topology, seconds=2.)
    first, arbiter = advance(rows, topology)
    rows = deepcopy(rows)
    second, _ = advance(rows, topology, arbiter=arbiter, start=1.)
    assert crossings(arbiter) == crossings(expected)
    assert rows == continuous
    assert first.active_vehicle_seconds + second.active_vehicle_seconds == complete.active_vehicle_seconds
    with pytest.raises(ValueError, match='persistent arbiter and reservations'):
        advance(rows, topology, start=2.)


def test_new_tick_signal_release_cannot_change_the_prior_observation():
    topology = network()
    controls = table(topology, overrides={(0, 1): dict(tls_id='tls', tls_index=0)},
                     phases=[(1, 'r'), (10, 'G')])
    rows = [seed(remaining_distance=0.)]
    arbiter = ChronologicalJunctionArbiter(controls, seed=3)
    first, _ = advance(rows, topology, arbiter=arbiter)
    assert rows[0]['current_edge'] == 0 and not crossings(arbiter)
    assert not first.decisions
    second, _ = advance(rows, topology, arbiter=arbiter, start=1.)
    assert crossings(arbiter)[0]['time'] == 1.
    assert rows[0]['current_edge'] == 3
    assert not second.decisions


def test_equal_time_disjoint_priority_cycles_share_seeded_arbitration():
    dependencies = {0: {1}, 1: {0}, 2: {3}, 3: {2}}
    movements = {(source, 4): MovementControl(source, 4, 'junction', 'right_before_left', source,
        None, None, '=', frozenset(dependencies[source]), frozenset(set(range(4)) - {source}), 0, 0)
        for source in range(4)}
    controls = CityControlTable(movements, {}, {})
    winners = set()
    for random_seed in range(32):
        rows = [seed(chr(ord('a') + edge), current_edge=edge, destination_edge=4,
            remaining_distance=0., committed_next_edge=4) for edge in range(4)]
        arbiter = ChronologicalJunctionArbiter(controls, seed=random_seed)
        arbiter.prepare(rows, 0.)
        grants = arbiter.grants(0., np.ones(5))
        winners.add(grants[0][0]['vehicle_id'])
        assert arbiter.audit[-1]['contenders'] == ['a', 'b', 'c', 'd']
    assert winners == {'a', 'b', 'c', 'd'}
