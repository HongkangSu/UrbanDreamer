import json

import numpy as np
import pytest

from tools.component_prediction_metrics import (
    disagreement_analysis,
    empirical_crps,
    empirical_energy_score,
    evaluate_prediction_samples,
    grouped_bootstrap,
    summarize_state_records,
)


def test_deterministic_crps_equals_absolute_error():
    target = np.array([[1., 4.], [2., 3.]])
    sample = np.array([[2., 1.], [1., 8.]])
    np.testing.assert_array_equal(empirical_crps(np.repeat(sample[None], 4, axis=0), target),
                                  np.abs(sample-target))


def test_sorted_crps_matches_pairwise_definition_with_ties():
    samples = np.array([[0., 2.], [2., 2.], [2., 4.]])
    target = np.array([1., 3.])
    expected = (np.abs(samples-target).mean(0)
                - .5*np.abs(samples[:, None]-samples[None, :]).mean((0, 1)))
    np.testing.assert_allclose(empirical_crps(samples, target), expected)
    assert empirical_crps([[0.], [2.]], [1.])[0] == .5


def test_joint_energy_score_has_hand_calculated_pairwise_correction():
    assert empirical_energy_score([[0., 0.], [2., 0.]], [1., 0.]) == .5
    assert empirical_energy_score([[0., 0.]], [3., 4.]) == 5.


def test_physical_metrics_distinguish_mean_and_sample_error_and_count_cohort():
    target = np.array([[[1., 10.], [0., 20.]]])
    samples = np.array([[[[0., 8.], [0., 20.]]], [[[2., 12.], [0., 20.]]]])
    result = evaluate_prediction_samples(samples, target, scales=[1., 2.], cohort_count=1,
                                         latent_samples=samples,
                                         latent_disagreement=[.2],
                                         upper_bounds=[1., 15.])
    summary = result['summary']
    assert summary['count_mae_occupied'] == 0.
    assert summary['count_sample_mae_occupied'] == 1.
    assert summary['count_crps_occupied'] == .5
    assert summary['speed_crps_occupied'] == 1.
    assert summary['count_sample_variance_occupied'] == 1.
    assert summary['count_coverage90_occupied'] == 1.
    assert summary['count_width90_occupied'] == pytest.approx(1.8)
    assert summary['total_count_absolute_error'] == 0.
    assert summary['total_count_sample_absolute_error'] == 1.
    assert summary['total_count_over_cohort_fraction'] == .5
    assert summary['total_count_max_cohort_excess'] == 1.
    assert summary['speed_above_upper_fraction_all'] == .5
    assert summary['count_above_upper_fraction_all'] == .25
    assert summary['energy_score'] == pytest.approx(np.sqrt(2)/2)
    assert summary['latent_disagreement'] == .2
    # Sample variance over count/speed, occupied/empty road: (1+4+0+0)/4.
    assert summary['latent_sample_variance'] == 1.25
    json.dumps(result, allow_nan=False)


def test_nan_speed_is_not_zero_and_energy_uses_only_observed_coordinates():
    target = np.array([[[1., np.nan], [0., np.nan]]])
    samples = np.array([[[[2., 30.], [0., 20.]]]])
    result = evaluate_prediction_samples(samples, target, scales=[2., 10.])
    summary = result['summary']
    assert summary['speed_mae_all'] is None
    assert summary['speed_valid_roads_all'] == 0
    assert summary['count_mae_all'] == .5
    assert summary['energy_score'] == .5
    assert summary['energy_score_dimensions'] == 2
    assert summary['true_total_count'] == 1.
    json.dumps(result, allow_nan=False)


def test_missing_counts_do_not_fabricate_total_mass():
    target = np.array([[[np.nan, 10.], [1., 10.]]])
    result = evaluate_prediction_samples(np.zeros((2, 1, 2, 2)), target, scales=[1., 1.])
    assert result['summary']['true_total_count'] is None
    assert result['summary']['total_count_absolute_error'] is None


def test_empty_roads_have_no_occupied_metrics():
    result = evaluate_prediction_samples(np.zeros((1, 1, 2, 2)), np.zeros((1, 2, 2)), scales=[1., 1.])
    assert result['summary']['count_mae_occupied'] is None
    assert result['summary']['count_coverage90_occupied'] is None
    assert result['summary']['count_coverage90_all'] == 1.


def test_rmse_aggregation_is_root_mean_squared_error_not_mean_root():
    samples = np.array([[[[0., 0.]], [[4., 0.]]]])
    result = evaluate_prediction_samples(samples, np.zeros((2, 1, 2)), scales=[1., 1.],
        state_metadata=[{'episode': 0, 'demand_group': 'a'}, {'episode': 1, 'demand_group': 'b'}])
    assert result['summary']['count_rmse_all'] == pytest.approx(np.sqrt(8))
    assert summarize_state_records(result['states']) == result['summary']
    assert grouped_bootstrap(result['states'], 'count_rmse_all', n_bootstrap=10)['estimate'] == pytest.approx(np.sqrt(8))


def test_valid_masks_and_metadata_are_preserved():
    target = np.array([[[1., 10.], [0., 20.]]])
    mask = np.ones_like(target, dtype=bool)
    mask[0, 0, 1] = False
    result = evaluate_prediction_samples(np.zeros((2, 1, 2, 2)), target, scales=[1., 1.],
        valid_mask=mask, state_metadata=[{'variant': 'test', 'horizon': 4}])
    assert result['states'][0]['variant'] == 'test'
    assert result['summary']['speed_mae_all'] == 20.
    assert result['summary']['speed_mae_occupied'] is None


@pytest.mark.parametrize('kwargs,match', [
    ({'scales': [0., 1.]}, 'scales'),
    ({'occupied_mask': np.ones((1, 2))}, 'boolean'),
    ({'latent_disagreement': [-1.]}, 'latent_disagreement'),
    ({'state_metadata': [{'count_mae_all': 9}]}, 'collides'),
    ({'interval_levels': [.9, .9]}, 'unique'),
    ({'upper_bounds': [-1., 1.]}, 'lower bounds'),
])
def test_invalid_protocol_inputs_fail(kwargs, match):
    parameters = {'scales': [1., 1.], **kwargs}
    with pytest.raises(ValueError, match=match):
        evaluate_prediction_samples(np.zeros((2, 1, 2, 2)), np.zeros((1, 2, 2)), **parameters)


def test_nonfinite_predictions_are_failures_even_for_unavailable_labels():
    samples = np.zeros((2, 1, 2, 2))
    samples[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        evaluate_prediction_samples(samples, np.full((1, 2, 2), np.nan), scales=[1., 1.])


def test_bootstrap_retains_windows_and_weights_demands_then_episodes():
    rows = [dict(demand_group='a', episode=0, loss=0.) for _ in range(100)]
    rows += [dict(demand_group='a', episode=1, loss=4.),
             dict(demand_group='b', episode=0, loss=10.)]
    result = grouped_bootstrap(rows, 'loss', n_bootstrap=100, seed=1)
    assert result['estimate'] == 6.  # ((0+4)/2+10)/2, not 14/102.
    duplicate_windows = rows + [dict(row) for row in rows if row['demand_group'] == 'a']
    other = grouped_bootstrap(duplicate_windows, 'loss', n_bootstrap=100, seed=1)
    assert other['estimate'] == result['estimate']
    assert other['ci_low'] == result['ci_low']
    assert other['ci_high'] == result['ci_high']


def test_single_cluster_has_no_fabricated_confidence_interval():
    rows = [dict(demand_group='a', episode=i, loss=float(i)) for i in range(20)]
    result = grouped_bootstrap(rows, 'loss')
    assert result['estimate'] == 9.5
    assert result['ci_low'] is None
    assert result['ci_high'] is None
    with pytest.raises(ValueError, match='grouping keys'):
        grouped_bootstrap([{'loss': 1.}], 'loss')


def test_risk_analysis_respects_ties_and_congestion_strata():
    rows = [dict(demand_group=i//2, episode=i, latent_disagreement=float(i),
                 count_mae_all=float(i*2), congestion=float(i)) for i in range(12)]
    result = disagreement_analysis(rows, congestion_edges=[3., 7.], n_bootstrap=20)
    assert result['all']['spearman_descriptive_windows'] == pytest.approx(1.)
    assert result['all']['spearman_episode_means'] == pytest.approx(1.)
    assert [item['states'] for item in result['congestion_strata']] == [4, 4, 4]
    assert all(item['spearman_descriptive_windows'] == pytest.approx(1.)
               for item in result['congestion_strata'])
    for row in rows:
        row['latent_disagreement'] = 1.
    result = disagreement_analysis(rows, n_bootstrap=20)
    assert result['all']['spearman_descriptive_windows'] is None
    assert len(result['all']['deciles']) == 1
    json.dumps(result, allow_nan=False)


def test_risk_spearman_average_ranks_for_tied_values():
    rows = [dict(demand_group=i, episode=0, latent_disagreement=x,
                 count_mae_all=y) for i, (x, y) in enumerate([(1., 2.), (1., 1.), (3., 3.)])]
    result = disagreement_analysis(rows, n_bootstrap=10)
    assert result['all']['spearman_descriptive_windows'] == pytest.approx(np.sqrt(3)/2)
