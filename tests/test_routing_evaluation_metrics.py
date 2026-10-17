from types import SimpleNamespace

import numpy as np
import pytest

from urban_dreamer.learning.routing_evaluation_metrics import DiversionCost, all_demand_censored_metrics
from urban_dreamer.traffic.topology import VehicleDemand


class Candidates:
    network = SimpleNamespace(free_flow_travel_seconds=np.array([0., 2., 4., 0.]))

    def candidates(self, source, destination):
        return (1, 2)

    def costs(self, destination):
        return np.array([0., 10. if destination == 3 else 20., 12., 0.])

    def junction(self, source, edge):
        return (6., 2.) if edge == 1 else (8., 2.)


def decision(edge):
    return SimpleNamespace(vehicle_id='v', current_edge=0, next_edge=edge)


def test_diversion_matches_historical_junction_edge_remaining_formula():
    cost = DiversionCost(Candidates())
    cost.add([decision(2), decision(1), decision(2)], {'v': 3})
    result = cost.metrics(2)
    # Chosen2 = 4+4+12=20; chosen1=3+2+10=15; two excesses of5.
    assert result['branch_freeflow_excess_seconds_total'] == 10.
    assert result['branch_freeflow_excess_seconds_per_started_vehicle'] == 5.
    assert result['diversion_decision_count'] == 3
    assert result['nonminimal_freeflow_choices'] == 2
    # The same branch can be optimal for another destination (cache key includes it).
    cost.add([decision(2)], {'v': 0})
    assert cost.metrics(2)['branch_freeflow_excess_seconds_total'] == 10.


def test_diversion_empty_and_invalid_choice():
    cost = DiversionCost(Candidates())
    assert cost.metrics(0)['branch_freeflow_excess_seconds_per_started_vehicle'] == 0.
    with pytest.raises(ValueError, match='legal candidate'):
        cost.add([decision(3)], {'v': 3})


def test_all_demand_includes_insertion_wait_and_unfinished_and_never_departed():
    demand = [VehicleDemand('a', 0, 1, 0.), VehicleDemand('b', 0, 1, 10.),
              VehicleDemand('c', 0, 1, 20.)]
    states = [dict(vehicle_id='a', departed=True, arrived=True, arrival_time=30., departure_time=5.),
              dict(vehicle_id='b', departed=True, arrived=False, departure_time=50.),
              dict(vehicle_id='c', departed=False, arrived=False, departure_time=20.)]
    result = all_demand_censored_metrics(demand, states, 100.)
    values = [30., 90., 80.]
    assert result['all_demand_censored_att_seconds'] == pytest.approx(np.mean(values))
    assert result['all_demand_censored_p90_seconds'] == pytest.approx(np.percentile(values, 90))
    assert result['all_demand_censored_p95_seconds'] == pytest.approx(np.percentile(values, 95))
    assert result['all_demand_censored_vehicle_seconds'] == 200.
    assert result['all_demand_vehicle_count'] == 3
    assert result['all_demand_unstarted_count'] == 1


def test_all_arrived_and_horizon_clipping_and_empty_demand():
    demand = [VehicleDemand('a', 0, 1, 10.)]
    state = [dict(vehicle_id='a', departed=True, arrived=True, arrival_time=40.)]
    assert all_demand_censored_metrics(demand, state, 100.)['all_demand_censored_att_seconds'] == 30.
    assert all_demand_censored_metrics(demand, state, 30.)['all_demand_censored_att_seconds'] == 20.
    assert all_demand_censored_metrics([], [], 100.)['all_demand_censored_att_seconds'] is None
