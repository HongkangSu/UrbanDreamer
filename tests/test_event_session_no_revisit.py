"""Physical history includes roads hidden between fixed SUMO observations."""
from dataclasses import replace

import numpy as np
import pytest

from test_event_session_commitments import session_at
from test_urban_dreamer_event_session import toy_setup
from urban_dreamer.traffic.environment import EnvironmentContractError, NextHopDecision
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.topology import NextHopCandidates


def hard_session(road='d', route=(0, 1, 3), index=2):
    session = session_at(road, tuple('abcd'[i] for i in route), index)
    session.forbid_revisit = True
    session.candidates = NextHopCandidates(session.network, forbid_revisit=True)
    session._commanded_routes = {'v': route}
    session._route_cursors = {'v': 0}
    session._vehicles['v']['visited_edges'] = (0,)
    session.metrics.update(physical_edge_entries=1, physical_revisit_count=0)
    return session


def test_skipped_roads_recorded_once_and_snapshot_history_immutable():
    session = hard_session()
    old_snapshot = dict(session._vehicles['v'])
    session._observe(np.zeros(4))
    assert session._vehicles['v']['visited_edges'] == (0, 1, 3)
    assert old_snapshot['visited_edges'] == (0,)
    session._observe(np.zeros(4))
    assert session.metrics['physical_edge_entries'] == 3
    assert session.metrics['physical_revisit_count'] == 0


def test_internal_lane_consumes_incoming_skipped_road_not_successor():
    session = hard_session(':j', (0, 1, 3), 1)
    session._observe(np.zeros(4))
    row = session._vehicles['v']
    assert row['visited_edges'] == (0, 1)
    assert row['current_edge'] == 1 and row['committed_next_edge'] == 3
    assert row['in_junction']


def test_arrival_consumes_entire_unobserved_suffix():
    session = hard_session('a', (0, 1, 3), 0)
    session._connection.simulation.getArrivedIDList = lambda: ('v',)
    session._observe(np.zeros(4))
    row = session._vehicles['v']
    assert row['visited_edges'] == (0, 1, 3)
    assert row['arrived'] and row['current_edge'] == row['destination_edge'] == 3
    assert not session._active


@pytest.mark.parametrize('arrived', [False, True])
def test_hidden_repeat_raises_even_when_observed_edge_is_new(arrived):
    session = hard_session('d', (0, 1, 2, 1, 3), 4)
    session._connection.simulation.getArrivedIDList = lambda: ('v',) if arrived else ()
    with pytest.raises(EnvironmentContractError, match='revisited external edge 1'):
        session._observe(np.zeros(4))
    assert session.metrics['physical_revisit_count'] == 1


@pytest.mark.parametrize("retain_prefix", [False, True])
def test_setroute_resets_cursor_without_recounting_current_road(retain_prefix):
    session = hard_session('b', (0, 1, 3), 1)
    session._observe(np.zeros(4))
    row = session._vehicles['v']
    # Pruning the trapped detour makes this a singleton. Force the existing
    # callback surface to exercise cursor reset at a valid route replacement.
    row['pending_decision'] = True
    session._pending.add('v')
    assigned = []
    def set_route(_, route):
        assigned.append(route)
        session._connection.vehicle.getRoute = lambda _: ["a", *route] if retain_prefix else route
        session._connection.vehicle.getRouteIndex = lambda _: int(retain_prefix)
    session._connection.vehicle.setRoute = set_route
    session._apply_decisions(lambda rows, flow, now: [NextHopDecision('v', 1, 3, now)], np.zeros((4, 4)))
    assert assigned == [['b', 'd']]
    assert session._route_cursors['v'] == int(retain_prefix)
    session._observe(np.zeros(4))
    assert row['visited_edges'] == (0, 1)
    session._connection.simulation.getArrivedIDList = lambda: ('v',)
    session._observe(np.zeros(4))
    assert row['visited_edges'] == (0, 1, 3)


def test_departure_history_is_private_to_each_vehicle_and_session():
    session = hard_session('a', (0, 1, 3), 0)
    session._vehicles['other'] = dict(session._vehicles['v'], visited_edges=())
    session._record_route_progress('v', 1)
    assert session._vehicles['other']['visited_edges'] == ()
    assert hard_session()._vehicles['v']['visited_edges'] == (0,)


def test_real_libsumo_no_revisit_full_history_and_zero_population():
    pytest.importorskip('libsumo')
    config, prepared, topology = toy_setup()
    for population in (0, 8):
        small = replace(topology, vehicles=topology.vehicles[:population])
        with SumoEventSession(config, prepared, small, backend='libsumo', forbid_revisit=True) as session:
            initial = session.reset()
            assert all(row['visited_edges'] == () for row in initial['vehicles'])
            def policy(rows, flow, now):
                return [NextHopDecision(row['vehicle_id'], row['current_edge'],
                    session.candidates.legal_candidates(row['current_edge'], row['destination_edge'], row['visited_edges'])[-1], now)
                    for row in rows]
            for _ in range(160):
                if session.done:
                    break
                state = session.advance_window(policy).next_state
                for row in state['vehicles']:
                    history = row['visited_edges']
                    assert len(history) == len(set(history))
                    if row['departed']:
                        assert history[-1] == row['current_edge']
                    if row['arrived']:
                        assert history[-1] == row['destination_edge']
            assert session.done
            assert session.metrics['physical_revisit_count'] == 0
            assert len(session.travel_times) == population
