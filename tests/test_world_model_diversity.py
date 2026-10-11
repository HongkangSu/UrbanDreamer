import numpy as np
import pytest

from tools.analyze_world_model_diversity import empirical_crps, summarize_samples, select_conditions, select_stress_conditions, aggregate_rows


def test_crps_deterministic_equals_absolute_error():
    samples=np.array([[2.,5.]]*32)
    np.testing.assert_array_equal(empirical_crps(samples,np.array([1.,7.])),[1.,2.])


def test_crps_has_pairwise_spread_correction():
    samples=np.array([[0.],[2.]])
    assert empirical_crps(samples,np.array([1.]))[0]==.5


def test_crps_rejects_nonfinite_values():
    with pytest.raises(ValueError,match='finite'):
        empirical_crps(np.array([[float('nan')]]),np.array([1.]))


def test_ensemble_error_does_not_hide_single_draw_error_and_cohort_limit():
    decoded=np.zeros((2,2,4));decoded[:,:,1]=1.
    decoded[:,:,2]=[[0.,0.],[2.,0.]]
    target=np.zeros((2,8));target[0,2]=1.
    result=summarize_samples(np.zeros((2,2,64)),np.zeros((2,2,64)),decoded,target,
        np.ones(2),np.ones(2),np.ones(2),1)
    assert result['ensemble_count_mae_occupied']==0.
    assert result['single_sample_count_mae_occupied']==1.
    assert result['count_std_occupied']==1.
    assert result['interval90_coverage_occupied']==1.
    assert result['interval90_width_occupied']==pytest.approx(1.8)
    assert result['total_count_over_previous_cohort_fraction']==.5
    assert result['total_count_over_previous_cohort_plus_one_vehicle_fraction']==0.
    assert result['latent_std_contraction_ratio'] is None


def test_empty_target_has_no_fabricated_occupied_metrics():
    result=summarize_samples(np.zeros((32,2,64)),np.zeros((32,2,64)),np.zeros((32,2,4)),
        np.zeros((2,8)),None,np.ones(2),np.ones(2),0)
    assert result['occupied_roads']==0
    assert result['crps_count_occupied'] is None
    assert result['count_std_all']==0.


def test_drain_selection_requires_actual_decline_and_skips_censored_plateau():
    from types import SimpleNamespace
    entries=[dict(demand_group=str(index),path=str(index),method='toy',episode=index) for index in range(7)]
    flows=[]
    for index in range(7):
        flow=np.zeros((22,1,8))
        flow[:,0,2]=50. if index==0 else [5.]*10+[50.,30.,15.,10.,8.,6.,4.,3.,2.,1.,0.,0.]
        flows.append(dict(flow=flow,time=np.arange(22)*30.))
    source=SimpleNamespace(manifest=dict(entries=entries),_episode=lambda index:flows[index])
    rows=select_conditions(source,np.ones(1),180.)
    assert len(rows)==12
    assert len({row['demand_group'] for row in rows})==6
    assert all(row['episode']!=0 for row in rows)
    drains=[row for row in rows if row['phase']=='drain']
    assert len(drains)==4
    assert all(row['input_road_count']<=.6*row['peak_road_count'] for row in drains)
    assert all(row['decision_time']>300. for row in drains)


def test_multimethod_stress_keeps_late_congestion_label_and_distinct_groups():
    from types import SimpleNamespace
    methods=('online_ac','dreamer','mbpo','random')
    entries=[]
    for method in methods:
        for group in ('already_used',method):
            entries.append(dict(method=method,demand_group=group,path=group,episode=1))
    flow=np.zeros((15,1,8));flow[:,0,2]=[1.]*8+[50.]+[35.]*6
    source=SimpleNamespace(manifest=dict(entries=entries),
                           _episode=lambda index:dict(flow=flow,time=np.arange(15)*30.))
    selected=select_stress_conditions(source,np.ones(1),180.,{'already_used'})
    assert {row['method'] for row in selected}==set(methods)
    assert len({row['demand_group'] for row in selected})==4
    assert all(row['phase']=='congested_late' for row in selected)
    assert all(row['decision_time']==240. for row in selected)


def test_coverage_aggregate_weights_occupied_roads():
    rows=[]
    for occupied,coverage in ((1,1.),(3,0.)):
        rows.append(dict(sigmas=[dict(sigma=sigma,occupied_roads=occupied,
            interval90_coverage_occupied=coverage) for sigma in (0.,.05,.1,.2)]))
    result=aggregate_rows(rows)
    assert all(row['conditions']==2 for row in result)
    assert all(row['interval90_coverage_occupied']==.5 for row in result)
    assert all(row['occupied_road_weighted_coverage90']==.25 for row in result)
