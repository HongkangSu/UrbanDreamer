import copy
import json

import pytest

from tools.summarize_component_predictions import (
    COMMON_KEYS, CAMPAIGN_KEYS, METRICS, validate_pairing, demand_bootstrap,
    paired_metric, create_report, read_evaluation,
    KNOWN_HORIZON_POLICIES, COMPLETE_HORIZON_POLICY,
)


def evaluation(variant='graph_fm'):
    protocol = {name: None for name in COMMON_KEYS}
    protocol.update(campaign={name:name for name in CAMPAIGN_KEYS},
                    model={'variant':variant}, role='development_not_final_test',samples=8,
                    steps=20,scales={'scales':[1.,2.],'congestion_edges':[.1,.2]},horizons=[1,4,8],
                    missing_horizons='omit endpoints beyond recorded trajectory, never extrapolate truth')
    conditions, states = [], {}
    for group in range(2):
        condition = dict(source_path=f'path{group}',episode=f'ep{group}',demand_group=f'g{group}',
                         start=1,condition_seed=group,requested_horizon=8,traffic_phase='peak')
        conditions.append(condition)
        forecasts = [('one_step_previous_state', 1)] + [('recorded_action_recursive_initial_cohort', h) for h in (1,4,8)]
        for forecast_protocol, horizon in forecasts:
            state_key = (f'path{group}',f'ep{group}',f'g{group}',1,forecast_protocol,horizon)
            states[state_key] = {}
            for predictor in ('predictor','persistence','endpoint_codec'):
                row = dict(condition,protocol=forecast_protocol,horizon=horizon,predictor=predictor,variant=variant,
                           roads=3,count_valid_roads_all=3,speed_valid_roads_all=3,
                           draws=8 if predictor=='predictor' else 1,
                           disagreement_independent_samples=horizon==1 and predictor=='predictor')
                row.update({name:float(group+1) for name in METRICS})
                states[state_key][predictor] = row
    return dict(directory='test',label=variant,protocol=protocol,conditions=conditions,states=states,
                variant=variant,training_seed=None,hashes={})


@pytest.mark.parametrize('field,value', [('steps',100),('samples',32),('target_speed','instantaneous'),
                                        ('scales',{'scales':[5.,6.]}),('solver','heun')])
def test_rejects_evaluation_protocol_mismatch(field,value):
    first, second = evaluation(), evaluation('regression')
    second['protocol'][field] = value
    with pytest.raises(ValueError,match='incompatible'):
        validate_pairing([first,second])


def test_rejects_changed_codec_manifest_or_config():
    for field in CAMPAIGN_KEYS:
        first, second = evaluation(), evaluation('regression')
        second['protocol']['campaign'][field] = 'different'
        with pytest.raises(ValueError,match='incompatible'):
            validate_pairing([first,second])


def test_rejects_condition_subsets_and_target_mask_changes():
    first, second = evaluation(), evaluation('regression')
    second['conditions'][0]['start'] += 1
    with pytest.raises(ValueError,match='conditions'):
        validate_pairing([first,second])
    second = evaluation('regression')
    next(iter(second['states'].values()))['predictor']['occupied_roads'] = 99
    with pytest.raises(ValueError,match='target mask'):
        validate_pairing([first,second])


def test_rejects_changed_deterministic_baseline():
    first, second = evaluation(), evaluation('regression')
    next(iter(second['states'].values()))['persistence']['energy_score'] += 1
    with pytest.raises(ValueError,match='baseline/codec'):
        validate_pairing([first,second])


def test_paired_bootstrap_conditions_on_checkpoint_and_keeps_clusters():
    rows = [dict(demand_group='g1',episode='e1',value=0.)]*100 + [dict(demand_group='g2',episode='e2',value=10.)]
    result = demand_bootstrap(rows,'value',repetitions=1000)
    assert result['estimate'] == 5.
    assert result['demand_groups'] == 2
    single = demand_bootstrap(rows[:100],'value')
    assert single['ci_low'] is None and single['ci_high'] is None
    left = {0:dict(demand_group='g1',episode='e1',value=1),1:dict(demand_group='g2',episode='e2',value=3)}
    right = {0:dict(demand_group='g1',episode='e1',value=2),1:dict(demand_group='g2',episode='e2',value=4)}
    delta = paired_metric(left,right,'value',repetitions=1000,seed=2)
    assert delta['estimate'] == delta['ci_low'] == delta['ci_high'] == -1


def test_empty_stress_separated_and_single_checkpoint_does_not_claim_comparison():
    data = evaluation()
    data['conditions'][0]['traffic_phase'] = 'empty_stress'
    for state in data['states'].values():
        for row in state.values():
            if row['demand_group'] == 'g0':
                row['traffic_phase']='empty_stress'
    report = create_report([data], repetitions=10)
    assert report['comparison_complete'] is False
    assert report['report_scope'] == 'single_checkpoint_diagnostic'
    assert {row['stratum'] for row in report['metrics']} == {'traffic_phases','empty_stress'}
    assert {row['reference'] for row in report['paired_differences']} == {'persistence'}


def test_pair_delta_uses_same_state_and_handles_missing_labels():
    first, second = evaluation(), evaluation('regression')
    for state in second['states'].values():
        state['predictor']['count_mae_all'] -= .5
    report = create_report([first,second], repetitions=10)
    rows=[r for r in report['paired_differences'] if r['reference']=='graph_fm' and r['metric']=='count_mae_all']
    assert len(rows)==4 and all(row['estimate']==-.5 for row in rows)
    left={0:dict(demand_group='g',episode='e',x=None)}
    right={0:dict(demand_group='g',episode='e',x=1)}
    with pytest.raises(ValueError,match='availability'):
        paired_metric(left,right,'x',repetitions=10,seed=0)


def test_reader_rejects_incomplete_or_final_evaluation(tmp_path):
    with pytest.raises(ValueError,match='incomplete'):
        read_evaluation(tmp_path)
    for name in ['protocol.json','conditions.json','states.jsonl','summary.json']:
        (tmp_path/name).write_text('{}')
    (tmp_path/'summary.json').write_text(json.dumps({'status':'complete'}))
    (tmp_path/'protocol.json').write_text(json.dumps({'role':'final_test'}))
    with pytest.raises(ValueError,match='development'):
        read_evaluation(tmp_path)


def test_known_horizon_policies_pair_only_after_record_completeness_proof():
    first, second = evaluation(), evaluation('regression')
    second['protocol']['missing_horizons'] = 'CLI rejects incomplete batches; direct evaluate_condition supports missing-label diagnostics'
    report = create_report([first, second], repetitions=10)
    assert report['compatibility']['missing_horizons'] == COMPLETE_HORIZON_POLICY
    proof = report['missing_horizon_equivalence']
    assert {row['original_policy'] for row in proof} == KNOWN_HORIZON_POLICIES
    assert all(row['predictor_states_checked'] == 8 and row['conditions_checked'] == 2 for row in proof)
    assert all(row['all_count_and_speed_roads_valid'] for row in proof)


@pytest.mark.parametrize('missing', ['h8', 'speed', 'both_partial_count_speed'])
def test_available_horizon_policy_cannot_hide_missing_forecasts_or_labels(missing):
    first, second = evaluation(), evaluation('regression')
    for data in (first, second):
        if missing == 'h8':
            # Identically incomplete result sets are also invalid; intersection is not proof.
            data['states'] = {k:v for k,v in data['states'].items() if k[-1] != 8}
        else:
            row = next(iter(data['states'].values()))['predictor']
            row['speed_valid_roads_all'] = 0 if missing == 'speed' else 2
            if missing == 'both_partial_count_speed':
                row['count_valid_roads_all'] = 2
    with pytest.raises(ValueError, match='complete|labels'):
        validate_pairing([first, second])


def test_unknown_horizon_policy_remains_rejected_with_full_records():
    first, second = evaluation(), evaluation('regression')
    second['protocol']['missing_horizons'] = 'new undocumented selection policy'
    with pytest.raises(ValueError, match='unknown missing_horizons'):
        validate_pairing([first, second])
