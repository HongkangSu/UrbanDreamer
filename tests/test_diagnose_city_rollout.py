from copy import deepcopy

import numpy as np
import pytest

from test_event_imagination_protocol import network, seed
from tools.diagnose_city_rollout import FrozenRouteController, cohort_truth, run_free_rollout
from tools.diagnose_component_executor import RouteMismatch
from urban_dreamer.learning import imagination as physical_engine
from urban_dreamer.traffic.topology import NextHopCandidates


def test_truth_cost_uses_observed_arrival_and_requires_complete_endpoint():
    seeds = [seed('a'), seed('b'), seed('c')]
    endpoint = {s['vehicle_id']: deepcopy(s) for s in seeds}
    endpoint['a'].update(arrived=True, arrival_time=31.)
    endpoint['b'].update(arrived=True, arrival_time=60.)
    truth = cohort_truth(seeds, endpoint, start_time=30., horizon=30.)
    assert truth['active_seconds'] == 61.
    assert truth['arrivals'] == 2
    with pytest.raises(ValueError, match='endpoint'):
        cohort_truth(seeds, {'a': endpoint['a']}, start_time=30., horizon=120.)


def test_frozen_route_controller_rejects_unrecorded_entry_and_handles_repeated_edge():
    controller = FrozenRouteController({'v': (0, 1, 0, 2)})
    state = seed('v', current_edge=0)
    controller.entry(state, 1)
    state['current_edge'] = 1
    controller.entry(state, 0)
    state['current_edge'] = 0
    assert controller.decide([state], None, 40.)[0].next_edge == 2
    with pytest.raises(RouteMismatch):
        controller.entry(state, 3)


def test_free_rollout_keeps_cumulative_cost_and_completed_state_across_horizons():
    graph = network((1., 1., 1., 1.))
    states = [seed('v', current_edge=1, remaining_distance=.5, committed_next_edge=3)]
    before = deepcopy(states)
    result = run_free_rollout(states, routes={'v': (1, 3)}, implementation='legacy',
        candidates=NextHopCandidates(graph, traffic_model='meso'), controls=None,
        speeds=np.full(4, 10.), flow=np.zeros((4, 8)), start_time=0., population=1,
        horizons=(30, 60, 120), max_speed=40., seed=610004)
    assert [r['horizon'] for r in result] == [30, 60, 120]
    assert all(r['status'] == 'ok' for r in result)
    assert [r['active_seconds'] for r in result] == [1., 1., 1.]
    assert all(r['states'][0]['arrived'] for r in result)
    assert states == before


def test_frozen_zero_speed_is_not_refreshed_from_future_truth():
    graph = network()
    states = [seed('v', current_edge=1, remaining_distance=5., committed_next_edge=3)]
    result = run_free_rollout(states, routes={'v': (1, 3)}, implementation='legacy',
        candidates=NextHopCandidates(graph, traffic_model='meso'), controls=None,
        speeds=np.zeros(4), flow=np.zeros((4, 8)), start_time=0., population=1,
        horizons=(30, 60, 120), max_speed=40., seed=610004)
    assert [r['active_seconds'] for r in result] == [30., 60., 120.]
    assert all(r['states'][0]['remaining_distance'] == 5. for r in result)


def test_route_failure_retains_every_later_horizon_and_restores_entry_hook():
    graph = network()
    states = [seed('v', current_edge=1, remaining_distance=0., committed_next_edge=3)]
    original = physical_engine._record_physical_entry
    result = run_free_rollout(states, routes={'v': (1, 2, 3)}, implementation='legacy',
        candidates=NextHopCandidates(graph, traffic_model='meso'), controls=None,
        speeds=np.full(4, 10.), flow=np.zeros((4, 8)), start_time=0., population=1,
        horizons=(30, 60, 120), max_speed=40., seed=610004)
    assert len(result) == 3
    assert all(r['status'] == 'error' and r['error']['type'] == 'RouteMismatch' for r in result)
    assert all(r['error']['first_failed_horizon'] == 30 and r['active_seconds'] is None for r in result)
    assert physical_engine._record_physical_entry is original
