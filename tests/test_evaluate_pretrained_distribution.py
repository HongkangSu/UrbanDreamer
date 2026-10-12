from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.evaluate_pretrained_distribution import (
    CHECKPOINT_SHA256, nested_mean_diagnostics, parser, run, score_bank,
    supplementary_scores, trajectory_bank,
)
from urban_dreamer.learning.action_events import pack_action_events


class NonlinearToy:
    def __init__(self):
        self.dynamics = SimpleNamespace(graph_prior=SimpleNamespace(rank=1))
        self.noises = []

    def transition(self, current, actions, *, noise, action_events, steps, solver):
        assert action_events.batch_size == current.shape[0]
        assert action_events.time_steps == 1
        assert steps == 100 and solver == 'euler'
        self.noises.append(noise.clone())
        return current + noise + .1

    def decode(self, z):
        decoded = z.new_zeros((*z.shape[:-1], 4))
        decoded[..., 2] = z[..., 0].square()
        decoded[..., 1] = z[..., 1] + 10
        return decoded


def bank_options():
    return dict(initial=torch.zeros(1, 2, 2), actions=torch.zeros(1, 2, 2),
        events=pack_action_events([[(), ()]]), capacities=torch.ones(2),
        limits=torch.ones(2), horizons=(1, 2), seed=19)


def test_draw_ordinal_noise_is_independent_of_chunk_and_global_rng():
    first = trajectory_bank(NonlinearToy(), chunk=4, **bank_options())
    torch.rand(2000)
    second = trajectory_bank(NonlinearToy(), chunk=3, **bank_options())
    assert first[2].shape == (32, 1, 2, 2)
    for horizon in (1, 2):
        np.testing.assert_array_equal(first[horizon], second[horizon])
    assert not np.array_equal(first[1][0], first[1][1])
    assert first[2][..., 1].var() > first[1][..., 1].var()


def test_zero_source_is_same_transition_with_exact_zero_coefficients():
    model = NonlinearToy()
    actual = trajectory_bank(model, samples=1, zero_source=True, **bank_options())
    assert all(torch.count_nonzero(noise) == 0 for noise in model.noises)
    np.testing.assert_allclose(actual[1][..., 0], .01)
    np.testing.assert_allclose(actual[2][..., 0], .04)
    np.testing.assert_allclose(actual[2][..., 1], 10.2)


def test_nonlinear_decoder_draws_are_averaged_after_decoding():
    model = NonlinearToy()
    bank = trajectory_bank(model, chunk=32, **bank_options())
    noise = model.noises[0][:, 0, 0].numpy()
    expected_mean = np.square(noise+.1).mean()
    wrong_latent_mean_decode = np.square((noise+.1).mean())
    np.testing.assert_allclose(bank[1][..., 0].mean(), expected_mean)
    assert abs(expected_mean-wrong_latent_mean_decode) > .1


def test_u_stat_scores_have_exact_two_draw_solution_and_k1_unavailable():
    samples = np.array([[[[0., 1.]]], [[[2., 1.]]]])
    target = np.array([[[1., 1.]]])
    result = supplementary_scores(samples, target, [1., 1.])
    assert result['count_crps_u_all'] == 0.
    assert result['energy_score_u'] == 0.
    assert supplementary_scores(samples[:1], target, [1., 1.])['energy_score_u'] is None
    deterministic = np.repeat(samples[:1], 8, axis=0)
    assert supplementary_scores(deterministic, target, [1., 1.])['count_crps_u_all'] == 1.
    assert supplementary_scores(deterministic, target, [1., 1.])['energy_score_u'] == 1.


def test_congestion_brier_thresholds_and_missing_speed_mask():
    target = np.array([[[5., 2.], [5., np.nan], [4.99, 2.]]])
    draws = np.array([[[[5., 2.], [10., 1.], [5., 2.]]],
                      [[[4., 1.], [10., 1.], [5., 2.]]]])
    result = supplementary_scores(draws, target, [1., 1.])
    assert result['congestion_valid_roads_all'] == 2
    assert result['congestion_brier_all'] == .625  # ((.5-1)^2 + (1-0)^2)/2
    assert result['congestion_prevalence_all'] == .5
    assert result['congestion_probability_all'] == .75


def test_k_prefixes_point_reference_and_no_single_draw_interval():
    bank = np.array([[[[float(i % 2)*2, 1.]]] for i in range(32)])
    target = np.array([[[1., 1.]]])
    rows = score_bank(bank, target, bank[:1], scales=[1., 1.], cohort_count=1,
        limits=np.array([10.]), metadata={'demand_group': 'g', 'episode': 'e', 'horizon': 1})
    assert len(rows) == 9
    first = next(row for row in rows if row['predictor'] == 'empirical_ensemble' and row['k'] == 1)
    assert first['count_mae_all'] == 1.
    assert first['count_coverage90_all'] is None and first['count_width90_all'] is None
    assert first['count_sample_variance_all'] is None
    full = next(row for row in rows if row['predictor'] == 'empirical_ensemble' and row['k'] == 32)
    point = next(row for row in rows if row['predictor'] == 'ensemble_mean_point' and row['k'] == 32)
    assert full['count_mae_all'] == point['count_mae_all'] == 0.
    assert full['count_sample_mae_all'] == 1.
    assert full['count_crps_all'] == .5
    assert point['count_crps_all'] == 0.
    assert point['count_coverage90_all'] is None
    assert full['total_count_over_cohort_fraction'] == .5
    assert full['count_above_upper_fraction_all'] == .5
    with pytest.raises(ValueError, match='32 trajectories'):
        score_bank(bank[:16], target, bank[:1], scales=[1., 1.], cohort_count=1,
                   limits=np.array([10.]), metadata={})


def test_nested_subsets_are_predeclared_permutations_and_full_mean_is_finite_reference():
    samples = np.arange(64, dtype=float).reshape(32, 1, 1, 2)
    rows, orders = nested_mean_diagnostics(samples, seed=7)
    repeated, same = nested_mean_diagnostics(samples, seed=7)
    assert rows == repeated
    np.testing.assert_array_equal(orders, same)
    assert all(sorted(order) == list(range(32)) for order in orders.tolist())
    assert all(row['mean_absolute_deviation_from_full_bank'] == 0 for row in rows if row['k'] == 32)
    assert all(row['disjoint_equal_k_mean_absolute_difference'] is None for row in rows if row['k'] == 32)
    assert any(row['disjoint_equal_k_mean_absolute_difference'] > 0 for row in rows if row['k'] == 16)


def test_reject_wrong_checkpoint_before_loading_or_using_gpu(tmp_path):
    source = tmp_path/'snapshot/checkpoints'
    source.mkdir(parents=True)
    (source/'latest.pt').write_bytes(b'screen model is not the pretrained checkpoint')
    args = parser().parse_args(['--experiment-dir', str(tmp_path), '--output-dir', str(tmp_path/'out')])
    with pytest.raises(ValueError, match='fully pretrained'):
        run(args)
    assert not args.output_dir.exists()
    assert CHECKPOINT_SHA256 == '2f63f0b2a5fe18f3a3bd9be27ffd1e63a2755464e7f2944e1227334f0ffe6af6'


def test_previous_artifacts_are_not_overwritten(tmp_path):
    args = parser().parse_args(['--experiment-dir', str(tmp_path), '--output-dir', str(tmp_path)])
    with pytest.raises(ValueError, match='preserve'):
        run(args)
