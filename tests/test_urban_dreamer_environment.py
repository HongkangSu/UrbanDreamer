import pytest

from urban_dreamer.traffic.environment import EnvironmentContractError, NextHopDecision, safe_microstep_seconds


def test_plain_state_preserves_native_numpy_mapping_and_dataclass_values():
    from collections import UserDict
    from pathlib import Path
    import numpy as np
    from urban_dreamer.traffic.environment import to_plain_state
    value = UserDict({7: {'sequence':(np.int64(2), np.float32(.25), np.bool_(True), None),
                         'array':np.array([[1.,2.]],np.float32), 'path':Path('a/b'),
                         'decision':NextHopDecision('v',0,1,2.)}})
    expected = {'7':{'sequence':[2,.25,True,None], 'array':[[1.,2.]], 'path':'a/b',
                     'decision':dict(vehicle_id='v',current_edge=0,next_edge=1,decision_time=2.)}}
    plain = to_plain_state(value)
    assert plain == expected
    value[7]['array'][0,0]=99.
    value[7]['sequence']=(88,)
    assert plain == expected
    assert to_plain_state({np.int64(4):[np.float64(.5)]}) == {'4':[.5]}
    with pytest.raises(TypeError,match='cannot serialize'):
        to_plain_state(object())


def test_microstep_divides_a_second_and_observes_short_edges():
    assert safe_microstep_seconds([100., 80.], 40.) == 1.
    assert safe_microstep_seconds([.1, 80.], 40.) == .001
    value = safe_microstep_seconds([10.], 40.)
    assert 1 / value == int(1 / value)
    assert value <= 10 / 80
    with pytest.raises(EnvironmentContractError, match='millisecond'):
        safe_microstep_seconds([.01], 40.)


def test_next_hop_record_contains_only_executed_event():
    decision = NextHopDecision('v', 2, 4, 30.)
    assert set(vars(decision)) == {'vehicle_id', 'current_edge', 'next_edge', 'decision_time'}


def test_boundary_departure_arrival_and_pending_decision_are_not_shifted():
    from types import SimpleNamespace
    import time
    import numpy as np
    from urban_dreamer.traffic.environment import SumoSession
    from urban_dreamer.traffic.topology import TrafficTopology, EdgeNetwork
    graph = ((1,), ())
    network = EdgeNetwork(('a', 'b'), {'a': 0, 'b': 1}, np.ones((2, 4)), np.array([10., 10.]),
                          np.ones(2), np.array([1000., 1000.]), np.array([[0., 1.], [0., 0.]]),
                          graph, np.ones(2, dtype=bool), np.array([100., 100.]))
    config = SimpleNamespace(simulation=SimpleNamespace(vehicle_max_speed_mps=40.,
        episode_wall_timeout_seconds=120., incident_start_seconds=0., incident_end_seconds=0., incident_speed_mps=.1))
    session = SumoSession(config, None, TrafficTopology(network, ('a',), ('b',), ()))
    session._started_at = time.monotonic()
    session._tc = SimpleNamespace(VAR_ROAD_ID=1, VAR_LANEPOSITION=2, VAR_LANE_ID=3)
    def row(vehicle_id, current, departed):
        return dict(vehicle_id=vehicle_id, current_edge=current, destination_edge=1, departed=departed,
                    arrived=False, departure_time=0., actual_departure_time=0. if departed else None,
                    observed_departure_time=0. if departed else None, committed_next_edge=None,
                    pending_decision=False, in_junction=False, remaining_distance=1000.,
                    junction_remaining_distance=0.)
    session._vehicles = {'arriving': row('arriving', 1, True), 'departing': row('departing', 0, False)}
    session._active.add('arriving')
    class Simulation:
        def getDepartedIDList(self):
            return ['departing'] if session._time_ms == 30000 else []
        def getArrivedIDList(self):
            return ['arriving'] if session._time_ms == 30000 else []
        def getStartingTeleportIDList(self):
            return []
        def getEndingTeleportIDList(self):
            return []
    class Vehicle:
        def getDeparture(self, vehicle_id):
            return 29.
        def subscribe(self, *args):
            pass
        def getAllSubscriptionResults(self):
            return {key: {1: network.edge_ids[value['current_edge']], 2: 0., 3: network.edge_ids[value['current_edge']]+'_0'}
                    for key, value in session._vehicles.items() if value['departed'] and not value['arrived']}
        def setRoute(self, vehicle_id, route):
            assert vehicle_id == 'departing' and route == ['a', 'b']
    session._connection = SimpleNamespace(simulation=Simulation(), vehicle=Vehicle(), simulationStep=lambda: None)
    session._state = lambda: {'flow': np.zeros((2, 8)), 'vehicles': [dict(value) for value in session._vehicles.values()]}
    def callback(rows, flow, now):
        return [NextHopDecision(value['vehicle_id'], value['current_edge'], 1, now) for value in rows]
    first = session.advance_window(callback)
    assert first.decisions == ()
    assert first.departure_counts.tolist() == [1., 0.]
    assert first.active_vehicle_seconds == 31.
    assert session._vehicles['departing']['pending_decision']
    second = session.advance_window(callback)
    assert len(second.decisions) == 1 and second.decisions[0].decision_time == 30.
    assert second.departure_counts.sum() == 0
    assert second.active_vehicle_seconds == 30.
    third = session.advance_window(callback)
    assert third.decisions == ()
    assert session._vehicles['departing']['committed_next_edge'] == 1


@pytest.mark.parametrize('scenario', ['boundary_entry', 'missed_edge', 'premature_arrival', 'internal_suffix'])
def test_physical_observation_preserves_commitments_or_rejects_lost_events(scenario):
    from types import SimpleNamespace
    import numpy as np
    from urban_dreamer.traffic.environment import SumoSession
    session = object.__new__(SumoSession)
    session._time_ms = 30000
    session.max_speed = 40.
    session._tc = SimpleNamespace(VAR_ROAD_ID=1, VAR_LANEPOSITION=2, VAR_LANE_ID=3)
    session._active, session._pending = {'v'}, set()
    session.travel_times = {}
    row = dict(vehicle_id='v', current_edge=0, destination_edge=2, departed=True, arrived=False,
               actual_departure_time=0., committed_next_edge=1, pending_decision=False,
               in_junction=False, remaining_distance=0., junction_remaining_distance=0.)
    session._vehicles = {'v': row}
    session.network = SimpleNamespace(edge_to_index={'a': 0, 'b': 1, 'c': 2}, lengths_m=np.array([100., 100., 100.]),
        external_lane_lengths_m={'b_1': 25.}, internal_lane_speeds_mps={':j_1_0': 7.},
        internal_lane_remaining_lengths_m={':j_1_0': 12.})
    observed = {1: 'b', 2: 2., 3: 'b_1'}
    if scenario == 'missed_edge':
        observed[1] = 'c'
    elif scenario == 'internal_suffix':
        observed = {1: ':j_1', 2: 2., 3: ':j_1_0'}
    simulation = SimpleNamespace(getStartingTeleportIDList=lambda: (), getEndingTeleportIDList=lambda: (),
                                 getDepartedIDList=lambda: (), getArrivedIDList=lambda: ('v',) if scenario == 'premature_arrival' else ())
    session._connection = SimpleNamespace(simulation=simulation,
        vehicle=SimpleNamespace(getAllSubscriptionResults=lambda: {'v': observed}))
    departures = np.zeros(3)
    if scenario in {'missed_edge', 'premature_arrival'}:
        with pytest.raises(EnvironmentContractError):
            session._observe(departures)
    else:
        session._observe(departures)
        assert departures.sum() == 0
        if scenario == 'boundary_entry':
            assert row['remaining_distance'] == 23.
            assert row['pending_decision'] and row['committed_next_edge'] is None
            assert session._pending == {'v'}
        else:
            assert row['in_junction'] and not row['pending_decision']
            assert row['committed_next_edge'] == 1
            assert row['junction_remaining_distance'] == 10.
