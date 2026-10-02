from types import SimpleNamespace
import numpy as np
import pytest
from urban_dreamer.learning.imagination import advance_imagined_window


class Candidates:
    def __init__(self,lengths=(10.,10.,10.),junction=0.):
        self.network=SimpleNamespace(edge_count=3,lengths_m=np.array(lengths))
        self.junction_length=junction
    def candidates(self,current,destination):
        return (current+1,) if current < destination else ()
    def junction(self,current,next_edge):
        return self.junction_length, 1.


def vehicle(vehicle_id='v',departure=0.,**updates):
    return dict(dict(vehicle_id=vehicle_id,current_edge=0,destination_edge=2,departure_time=departure,departed=departure<=0,arrived=False,remaining_distance=10.,committed_next_edge=None,pending_decision=departure<=0,in_junction=False,junction_remaining_distance=0.,junction_speed=1.),**updates)


def decide(states, flow, time):
    return [SimpleNamespace(vehicle_id=s['vehicle_id'],current_edge=s['current_edge'],next_edge=s['current_edge']+1,decision_time=time) for s in states]


def run(states,candidates=None,start=0.,duration=30.,speeds=None,callback=decide):
    return advance_imagined_window(states,candidates=candidates or Candidates(),edge_speeds_mps=np.ones(3) if speeds is None else speeds,frozen_flow=np.zeros((3,8)),start_time=start,interval_seconds=duration,total_vehicle_count=len(states),decide=callback)


def test_multiple_hops_counted_and_terminal_edge_time_charged():
    states=[vehicle()]
    result=run(states)
    assert [x.decision_time for x in result.decisions] == [0.,10.]
    np.testing.assert_array_equal(result.action,[0,1,1])
    np.testing.assert_array_equal(result.departure,[0,0,0])
    assert result.arrivals == {'v':30.}
    assert result.active_vehicle_seconds == 30.


def test_end_boundary_physics_precedes_next_window_decisions_and_departures():
    states=[vehicle(remaining_distance=30.),vehicle('future',departure=30.)]
    first=run(states,candidates=Candidates((30.,10.,10.)))
    assert [x.vehicle_id for x in first.decisions] == ['v']
    assert states[0]['current_edge']==1 and states[0]['pending_decision']
    assert states[1]['departed'] and states[1]['pending_decision']
    np.testing.assert_array_equal(first.departure,[.5,0,0])
    second=run(states,candidates=Candidates((30.,10.,10.)),start=30.)
    assert [(x.vehicle_id,x.decision_time) for x in second.decisions[:2]] == [('future',30.),('v',30.)]
    np.testing.assert_array_equal(second.departure,[0,0,0])


def test_long_edge_keeps_committed_action_and_distance_with_new_speed():
    states=[vehicle(remaining_distance=100.)]
    first=run(states,candidates=Candidates((100.,10.,10.)))
    assert len(first.decisions)==1 and states[0]['remaining_distance']==70.
    second=run(states,candidates=Candidates((100.,10.,10.)),start=30.,speeds=np.array([2.,1.,1.]))
    assert len(second.decisions)==0 and states[0]['remaining_distance']==10.
    assert states[0]['committed_next_edge']==1 and not states[0]['arrived']


def test_junction_progress_survives_boundary_without_resampling():
    states=[vehicle(remaining_distance=25.)]
    first=run(states,candidates=Candidates(junction=10.))
    assert len(first.decisions)==1
    assert states[0]['in_junction'] and states[0]['junction_remaining_distance']==5.
    second=run(states,candidates=Candidates(junction=10.),start=30.)
    assert second.decisions[0].decision_time == 35.
    assert not second.arrivals
    assert states[0]['current_edge']==2 and states[0]['remaining_distance']==5.


def test_future_departures_keep_empty_windows_valid_and_stopped_traffic_stays_active():
    states=[vehicle(departure=60.)]
    first=run(states)
    assert first.active_vehicle_seconds==0 and not first.decisions
    second=run(states,start=30.)
    assert second.departure[0]==1 and not second.decisions
    third=run(states,start=60.,speeds=np.zeros(3))
    assert len(third.decisions)==1 and third.active_vehicle_seconds==30
    assert states[0]['remaining_distance']==10


def test_invalid_decision_raises_without_counting_action():
    def invalid(states,flow,time):
        return [SimpleNamespace(vehicle_id='v',current_edge=0,next_edge=2,decision_time=time)]
    with pytest.raises(ValueError,match='legal'):
        run([vehicle()],callback=invalid)


def test_simultaneous_boundary_arrival_departure_and_entry_are_not_lost():
    states = [vehicle('arrive',current_edge=2,remaining_distance=30.,departed=True,pending_decision=False),
              vehicle('enter',remaining_distance=30.), vehicle('depart',departure=30.)]
    first = run(states,candidates=Candidates((30.,10.,30.)))
    assert first.arrivals == {'arrive':30.}
    assert first.active_vehicle_seconds == 60.
    assert states[0]['arrived'] and not states[0]['pending_decision']
    assert states[1]['current_edge'] == 1 and states[1]['pending_decision']
    assert states[2]['departed'] and states[2]['pending_decision']
    np.testing.assert_allclose(first.departure,[1/3,0,0])
    assert [d.vehicle_id for d in first.decisions] == ['enter']
    second = run(states,candidates=Candidates((30.,10.,30.)),start=30.)
    assert [(d.vehicle_id,d.decision_time) for d in second.decisions[:2]] == [('depart',30.),('enter',30.)]
    assert second.arrivals == {}


def test_junction_exit_exactly_on_boundary_defers_decision():
    states = [vehicle(remaining_distance=20.)]
    first = run(states,candidates=Candidates(junction=10.))
    assert len(first.decisions) == 1
    assert not states[0]['in_junction'] and states[0]['current_edge'] == 1
    assert states[0]['remaining_distance'] == 10. and states[0]['pending_decision']
    second = run(states,candidates=Candidates(junction=10.),start=30.)
    assert len(second.decisions) == 1 and second.decisions[0].decision_time == 30.
    assert second.arrivals == {'v':60.}


def test_empty_initial_state_departure_is_conditioned_in_first_window():
    state = vehicle(departed=False,pending_decision=False,remaining_distance=100.)
    result = run([state],candidates=Candidates((100.,10.,10.)))
    np.testing.assert_array_equal(result.departure,[1.,0.,0.])
    assert result.decisions[0].decision_time == pytest.approx(.001)
    assert result.active_vehicle_seconds == pytest.approx(29.999)
    assert state['departure_time'] == 0.
    assert state['observed_departure_time'] == pytest.approx(.001)


def test_overdue_queued_departure_is_not_silently_inserted_into_seed_state():
    state = vehicle(departure=5.,departed=False,pending_decision=False,remaining_distance=100.)
    result = advance_imagined_window([state], candidates=Candidates((100.,10.,10.)),
        edge_speeds_mps=np.ones(3),frozen_flow=np.zeros((3,8)),start_time=30.,
        total_vehicle_count=1,decide=decide,observation_step_seconds=.1)
    np.testing.assert_array_equal(result.departure,[1.,0.,0.])
    assert result.decisions[0].decision_time == pytest.approx(30.1)
    assert result.active_vehicle_seconds == pytest.approx(29.9)
    assert state['departure_time'] == 5.
    assert state['observed_departure_time'] == pytest.approx(30.1)


def test_vehicle_speed_cap_applies_to_new_and_seeded_junctions():
    class FastJunction(Candidates):
        def junction(self,current,next_edge):
            return 100., 10.
    newly_entered = vehicle(remaining_distance=10.)
    seeded = vehicle('seeded',remaining_distance=0.,committed_next_edge=1,
        pending_decision=False,in_junction=True,junction_remaining_distance=100.,junction_speed=10.)
    result = advance_imagined_window([newly_entered,seeded],candidates=FastJunction(),
        edge_speeds_mps=np.ones(3),frozen_flow=np.zeros((3,8)),start_time=0.,
        total_vehicle_count=2,decide=decide,vehicle_max_speed_mps=2.)
    assert not result.arrivals
    assert newly_entered['junction_speed'] == seeded['junction_speed'] == 2.
    assert newly_entered['junction_remaining_distance'] == 60.
    assert seeded['junction_remaining_distance'] == 40.
    assert len(result.decisions) == 1


def test_physical_cost_segments_preserve_counts_and_exact_suffix_cost():
    states = [vehicle('first', current_edge=2, remaining_distance=5., pending_decision=False),
              vehicle('second', current_edge=2, remaining_distance=12., pending_decision=False)]
    result = run(states)
    assert [(s.start_time, s.end_time, s.active_count) for s in result.cost_segments] == [
        (0., 5., 2), (5., 12., 1), (12., 30., 0)]
    assert result.active_vehicle_seconds == 17.
    assert result.end_time == 30.
    assert sum((s.end_time-s.start_time)*s.active_count for s in result.cost_segments) == 17.


def test_stop_when_empty_reports_actual_terminal_time_without_padded_tail():
    result = advance_imagined_window([vehicle(current_edge=2, remaining_distance=7., pending_decision=False)],
        candidates=Candidates(), edge_speeds_mps=np.ones(3), frozen_flow=np.zeros((3,8)),
        start_time=0., total_vehicle_count=1, decide=decide, stop_when_empty=True)
    assert result.end_time == 7. and result.active_vehicle_seconds == 7.
    assert [(s.start_time, s.end_time, s.active_count) for s in result.cost_segments] == [(0., 7., 1)]
