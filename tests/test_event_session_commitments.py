"""Full SUMO routes must remain usable as physical imagination seeds."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_urban_dreamer_event_session import greedy, synthetic_network, toy_setup
from urban_dreamer.learning.imagination import advance_imagined_window
from urban_dreamer.traffic.environment import EnvironmentContractError, NextHopDecision
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.topology import NextHopCandidates


def session_at(road, route, index, *, current=0, pending=False):
    network = replace(synthetic_network(),
        external_lane_lengths_m={f'{edge}_0': 25. for edge in 'abcd'},
        internal_lane_speeds_mps={':j_0': 7.},
        internal_lane_remaining_lengths_m={':j_0': 12.})
    session = object.__new__(SumoEventSession)
    session._time_ms, session.max_speed = 30000, 40.
    session._tc = SimpleNamespace(VAR_ROAD_ID=1, VAR_LANEPOSITION=2, VAR_LANE_ID=3)
    session._active, session._pending = {'v'}, {'v'} if pending else set()
    session.travel_times, session.metrics = {}, {'events': 0}
    session.network, session.candidates = network, NextHopCandidates(network)
    session._vehicles = {'v': dict(vehicle_id='v', current_edge=current, destination_edge=3,
        departed=True, arrived=False, actual_departure_time=0., departure_time=0.,
        committed_next_edge=None, pending_decision=pending, in_junction=False,
        remaining_distance=0., junction_remaining_distance=0.)}
    session._connection = SimpleNamespace(
        simulation=SimpleNamespace(getStartingTeleportIDList=lambda: (), getEndingTeleportIDList=lambda: (),
                                   getDepartedIDList=lambda: (), getArrivedIDList=lambda: ()),
        vehicle=SimpleNamespace(getAllSubscriptionResults=lambda: {'v': {1: road, 2: 2., 3: road + '_0'}},
                                getRoute=lambda _: route, getRouteIndex=lambda _: index))
    return session


def test_singleton_commitment_is_observed_without_action():
    session = session_at('c', ('a', 'b', 'c', 'b', 'd'), 2)
    session._observe(np.zeros(4))
    row = session._vehicles['v']
    assert row['current_edge'] == 2 and row['committed_next_edge'] == 1
    assert not row['pending_decision'] and not session._pending
    assert session.metrics['events'] == 0


def test_pending_fork_preserves_route_but_policy_can_replace_it():
    session = session_at('b', ('a', 'b', 'd'), 1)
    session._observe(np.zeros(4))
    row = session._vehicles['v']
    assert row['committed_next_edge'] == 3 and row['pending_decision']
    routes = []
    session._connection.vehicle.setRoute = lambda vid, route: routes.append(route)
    actions = session._apply_decisions(
        lambda rows, flow, now: [NextHopDecision('v', 1, 2, now)], np.zeros((4, 4)))
    assert row['committed_next_edge'] == 2 and not row['pending_decision']
    assert routes == [['b', 'c', 'b', 'd']]
    assert len(actions) == session.metrics['events'] == 1


def test_internal_lane_after_skipped_edge_recovers_actual_route_pair():
    session = session_at(':j', ('a', 'b', 'c', 'b', 'd'), 2, current=0, pending=True)
    session._observe(np.zeros(4))
    row = session._vehicles['v']
    assert row['current_edge'] == 2 and row['committed_next_edge'] == 1
    assert row['junction_connection'] == (2, 1)
    assert row['junction_remaining_distance'] == 10.
    assert not row['pending_decision'] and not session._pending
    assert session.metrics['events'] == 0


def test_destination_has_no_successor_or_decision():
    session = session_at('d', ('a', 'b', 'd'), 2)
    session._observe(np.zeros(4))
    row = session._vehicles['v']
    assert row['current_edge'] == 3 and row['committed_next_edge'] is None
    assert not row['pending_decision'] and not row['arrived']
    assert row['remaining_distance'] == 23.


@pytest.mark.parametrize('route,index,message', [
    (('a', 'b', 'd'), -1, 'invalid route index'),
    (('a', 'd'), 0, 'illegal successor'),
    (('a', 'b'), 0, 'does not end'),
    (('a', 'unknown', 'd'), 0, 'unknown external edge'),
])
def test_invalid_actual_routes_are_not_guessed(route, index, message):
    session = session_at('a', route, index)
    with pytest.raises(EnvironmentContractError, match=message):
        session._observe(np.zeros(4))


def test_real_libsumo_internal_route_indices_and_imagination_seeds():
    pytest.importorskip('libsumo')
    config, prepared, topology = toy_setup()
    topology = replace(topology, vehicles=topology.vehicles[:3])
    seen = set()
    with SumoEventSession(config, prepared, topology, backend='libsumo') as session:
        session.reset()
        for _ in range(240):
            session._apply_decisions(greedy(session), session._flow())
            session._connection.simulationStep()
            session._time_ms += 1000
            session._observe(np.zeros(session.network.edge_count))
            api = session._connection.vehicle
            for vid in session._active:
                row = session._vehicles[vid]
                route, index = api.getRoute(vid), api.getRouteIndex(vid)
                assert row['current_edge'] == session.network.edge_to_index[route[index]]
                if row['current_edge'] == row['destination_edge']:
                    assert row['committed_next_edge'] is None
                    seen.add('destination')
                else:
                    assert row['committed_next_edge'] == session.network.edge_to_index[route[index + 1]]
                    if row['in_junction']:
                        assert not row['pending_decision']
                        seen.add('internal')
                    elif row['pending_decision']:
                        seen.add('fork')
                    elif len(session.candidates.candidates(row['current_edge'], row['destination_edge'])) == 1:
                        seen.add('singleton')
            if session._time_ms % 30000 == 0:
                seeds = [dict(session._vehicles[vid]) for vid in session._active]
                if seeds:
                    result = advance_imagined_window(seeds, candidates=session.candidates,
                        edge_speeds_mps=session.network.speed_limits_mps,
                        frozen_flow=session._flow(), start_time=session._time_ms / 1000.,
                        total_vehicle_count=len(topology.vehicles), decide=greedy(session),
                        vehicle_max_speed_mps=session.max_speed)
                    assert result.end_time == session._time_ms / 1000. + 30.
            if session.done:
                break
    assert seen == {'internal', 'fork', 'singleton', 'destination'}
