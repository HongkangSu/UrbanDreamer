from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from test_urban_dreamer_event_session import synthetic_network
from tools.analyze_component_imagination import (
    DynamicShortestController, active_cohort, consistency_record, diagnose_rollout, external_counts,
    resummarize, summarize_consistency, validate_condition_identity,
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


def summary_rows():
    return [dict(demand_group=f'g{i}', episode=f'archive{i}::episode=0', start=i,
        condition_seed=10+i, sample=0, horizon=h, timestamp=i*50+h*30,
        training_deadline_seconds=60., beyond_training_deadline=i*50+h*30 > 60,
        post_terminal_stress=False, total_mass_absolute_error=float(i+1))
        for h in (0, 1, 2) for i in (0, 1)]


def test_fixed_horizon_pair_population_does_not_depend_on_model_terminal_time():
    first, second = summary_rows(), summary_rows()
    for row in first:
        row['post_terminal_stress'] = row['horizon'] >= 1
    for row in second:
        row['post_terminal_stress'] = row['horizon'] >= 2
    a, b = summarize_consistency(first, seed=1), summarize_consistency(second, seed=1)
    assert a['complete_condition_sample_keys'] == b['complete_condition_sample_keys']
    for left, right in zip(a['fixed_horizon'], b['fixed_horizon']):
        assert left['matching_condition_sample_keys'] == right['matching_condition_sample_keys']
        assert left['matching_keys_sha256'] == right['matching_keys_sha256']
        assert left['states'] == right['states']
        assert left['metrics'] == right['metrics']
    assert a['terminal_diagnostics'] != b['terminal_diagnostics']
    with pytest.raises(ValueError, match='same complete'):
        summarize_consistency(first[:-1], seed=1)


def test_deadline_exceedance_is_separate_from_primary_fixed_horizon():
    result = summarize_consistency(summary_rows(), seed=1)
    later = [row for row in result['fixed_horizon'] if row['horizon'] == 2]
    assert {row['deadline_scope'] for row in later} == {'within_deadline', 'after_deadline'}
    assert all(row['states'] == 1 for row in later)
    primary = next(row for row in later if row['deadline_scope'] == 'within_deadline')
    stress = next(row for row in later if row['deadline_scope'] == 'after_deadline')
    assert primary['role'] == 'primary_paired_fixed_horizon'
    assert stress['role'] == 'after_deadline_stress'
    assert primary['metrics']['total_mass_absolute_error']['estimate'] == 1.
    assert stress['metrics']['total_mass_absolute_error']['estimate'] == 2.
    assert not set(primary['matching_condition_sample_keys']) & set(stress['matching_condition_sample_keys'])


def test_summary_revision_preserves_original_artifacts_and_records_hashes(tmp_path):
    source, revised = tmp_path/'original', tmp_path/'revision'
    source.mkdir()
    files = {'protocol.json': json.dumps({'campaign': {'selection_seed': 1}}),
        'states.jsonl': '\n'.join(json.dumps(row) for row in summary_rows()),
        'summary.json': '{"schema_version": 1}'}
    for name, contents in files.items():
        (source/name).write_text(contents)
    result = resummarize(source, revised)
    assert result['schema_version'] == 2
    for name, contents in files.items():
        assert (source/name).read_text() == contents
    revision = json.loads((revised/'revision.json').read_text())
    assert revision['kind'] == 'summary_only_revision_no_model_rerun'
    assert len(revision['code_sha256']) == len(revision['source_states_sha256']) == 64
    with pytest.raises(ValueError, match='new output'):
        resummarize(source, revised)


def test_external_conditions_validate_archive_episode_demand_and_start():
    entries = [dict(path='cache/a', episode=3, demand_group='group1', windows=10)]
    valid = dict(episode_index=0, start=9, source_path='cache/a',
                 episode='cache/a::episode=3', demand_group='group1')
    validate_condition_identity(valid, entries)
    for key, value in [('source_path', 'cache/b'), ('episode', 'cache/a::episode=4'),
                       ('demand_group', 'other'), ('start', 10), ('episode_index', 1)]:
        with pytest.raises(ValueError):
            validate_condition_identity(dict(valid, **{key: value}), entries)
