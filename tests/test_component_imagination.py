from copy import deepcopy

import numpy as np
import pytest
import torch

from test_urban_dreamer_event_session import synthetic_network
from tools.analyze_component_imagination import (
    DynamicShortestController, active_cohort, consistency_record, diagnose_rollout, external_counts,
)
from urban_dreamer.traffic.topology import NextHopCandidates


def vehicle(name='a', **kwargs):
    return dict(dict(vehicle_id=name, current_edge=3, destination_edge=3,
        departure_time=0., departed=True, arrived=False, remaining_distance=2.,
        committed_next_edge=None, pending_decision=False, in_junction=False,
        junction_remaining_distance=0., junction_speed=1.), **kwargs)


class ConstantModel:
    def __init__(self):
        self.events = []

    def encode(self, flow):
        return torch.nn.functional.pad(flow, (0, 56))

    def decode(self, latent):
        return latent[..., :4]

    def transition(self, z, action, *, action_events, **kwargs):
        self.events.append(action_events)
        return z


def test_external_counts_sum_duplicate_edges_and_exclude_internal_unborn_arrived():
    states = [vehicle('a'), vehicle('b'), vehicle('internal', in_junction=True),
              vehicle('future', departed=False), vehicle('done', arrived=True)]
    assert external_counts(states, 4).tolist() == [0, 0, 0, 2]
    cohort = active_cohort(states)
    assert {s['vehicle_id'] for s in cohort} == {'a', 'b', 'internal'}
    cohort[0]['current_edge'] = 0
    assert states[0]['current_edge'] == 3


def test_no_future_births_correct_cost_and_postterminal_label():
    states = [vehicle('a'), vehicle('b', remaining_distance=12.),
              vehicle('future', departed=False, departure_time=5.)]
    saved = deepcopy(states)
    flow = torch.zeros(4, 8)
    flow[:, 1] = 1.
    flow[3, 2] = 2.
    model = ConstantModel()
    records, traces = diagnose_rollout(model, flow, states, network=synthetic_network(),
        start_time=0., population=100, horizons=[1, 2, 4, 8, 16], steps=20,
        vehicle_max_speed_mps=10.)
    assert states == saved
    assert len(model.events) == 16 and len(records) == 6
    for record in records[1:]:
        assert record['initial_cohort'] == 2
        assert record['cumulative_arrivals'] == 2
        assert record['explicit_active'] == 0 and record['post_terminal_stress']
        assert record['cumulative_active_vehicle_seconds'] == pytest.approx(3.)
        assert record['mean_remaining_minutes'] == pytest.approx(.025)
        assert record['total_mass_absolute_error'] == 2.
    assert all('future' not in event.vehicle_ids for event in model.events)
    assert traces[-1]['external_counts'] == [0, 0, 0, 0]
    assert records[-1]['window_active_vehicle_seconds'] == 0.


def test_dynamic_controller_uses_legal_current_speed_costs():
    candidates = NextHopCandidates(synthetic_network())
    controller = DynamicShortestController(candidates, deadline=1800., initial_cohort=1,
                                         vehicle_max_speed_mps=10.)
    flow = np.zeros((4, 4))
    flow[:, 1] = 1.
    controller.prepare(flow)
    states = [vehicle(current_edge=1, destination_edge=3, pending_decision=True)]
    selected = controller.decide(states, flow, 5., states)
    assert selected[0].next_edge == 3 and selected[0].decision_time == 5.


def test_internal_junction_exclusion_and_clipping_are_reported_without_hiding_raw_values():
    decoded = np.zeros((4, 4))
    decoded[:, 1] = [-1., .01, 2., 1.]
    decoded[0, 2] = -.1
    states = [vehicle('junction', in_junction=True), vehicle('road')]
    record = consistency_record(decoded, states, capacities=np.ones(4), speed_limits=np.ones(4)*10,
        initial_size=2, cumulative_seconds=0., cumulative_arrivals=0, elapsed=0., horizon=0,
        timestamp=0., deadline=1800., traffic_model='micro', vehicle_max_speed_mps=10.)
    assert record['explicit_external'] == 1 and record['explicit_in_junction'] == 1
    assert record['negative_decoded_count_fraction'] == .25
    assert record['negative_decoded_speed_fraction'] == .25
    assert record['speed_fraction_clipped_fraction'] == .75
    assert record['total_mass_bias'] == pytest.approx(-1.1)


def test_empty_or_duplicate_cohort_is_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        active_cohort([vehicle(), vehicle()])
    with pytest.raises(ValueError, match='active seed'):
        diagnose_rollout(ConstantModel(), torch.zeros(4, 8), [vehicle(departed=False)],
            network=synthetic_network(), start_time=0., population=1, horizons=[1], steps=20)
