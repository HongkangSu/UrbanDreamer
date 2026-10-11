import json

import numpy as np
import pytest

from tools.counterfactual_statistics import cluster_paired_summary, paired_summary


def test_paired_summary_matches_sample_statistics_and_counts():
    values = [-1., 0., 2., 3.]
    actual = paired_summary(values)
    assert actual["n"] == 4
    assert actual["mean"] == 1.
    assert actual["sd"] == pytest.approx(np.sqrt(10 / 3))
    assert actual["se"] == pytest.approx(np.sqrt(10 / 3) / 2)
    assert actual["signal_to_noise"] == pytest.approx(1 / np.sqrt(10 / 3))
    assert actual["positive_count"] == 2
    assert actual["negative_count"] == 1
    assert actual["zero_count"] == 1


def test_bootstrap_is_reproducible_and_matches_fixed_seed_reference():
    values = np.array([.2, .5, -.3, .9, 1.1])
    actual = paired_summary(values)
    rng = np.random.default_rng(0)
    reference = values[rng.integers(0, 5, size=(20_000, 5))].mean(axis=1)
    low, high = np.quantile(reference, [.025, .975])
    assert actual == paired_summary(iter(values))
    assert actual["ci_low"] == pytest.approx(low)
    assert actual["ci_high"] == pytest.approx(high)
    assert actual["uncertainty_method"] == "paired_percentile_bootstrap"
    assert actual["bootstrap_draws"] == 20_000


@pytest.mark.parametrize("values", [[], [2.]])
def test_fewer_than_two_units_have_no_interval(values):
    actual = paired_summary(values)
    assert actual["ci_low"] is None
    assert actual["ci_high"] is None
    assert actual["sd"] is None
    assert actual["se"] is None
    assert actual["uncertainty_method"] == "insufficient_independent_units"
    assert actual["mean"] == (2. if values else None)
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize("value", [0., .02, -1.])
def test_zero_variance_is_explicit_and_serializes_without_infinity(value):
    actual = paired_summary([value] * 5)
    assert actual["variance_degenerate"]
    assert actual["ci_low"] == actual["ci_high"] == value
    assert actual["sd"] == actual["se"] == 0.
    assert actual["signal_to_noise"] is None
    assert "deterministic replays" in actual["uncertainty_note"]
    json.dumps(actual, allow_nan=False)


def test_cluster_means_receive_equal_weight_despite_different_repeat_counts():
    rows = [{"demand_seed": 1, "difference": 1.},
            {"demand_seed": 1, "difference": 3.},
            {"demand_seed": 2, "difference": 6.}]
    actual = cluster_paired_summary(rows)
    assert actual["n_pairs"] == 3
    assert actual["n_clusters"] == actual["n"] == 2
    assert actual["mean"] == 4.
    assert actual["sd"] == pytest.approx(np.sqrt(8))
    assert actual["se"] == 2.
    assert actual["ci_low"] == 2.
    assert actual["ci_high"] == 6.


def test_same_state_replays_do_not_inflate_independent_sample_size():
    rows = [{"demand_seed": 7, "difference": .02}] * 20
    actual = cluster_paired_summary(rows)
    assert actual["n_pairs"] == 20
    assert actual["n"] == actual["n_clusters"] == 1
    assert actual["ci_low"] is None
    assert actual["ci_high"] is None


def test_cluster_keys_and_confidence_can_be_customized():
    actual = cluster_paired_summary([{"seed": 2, "delta": -1.},
                                    {"seed": 3, "delta": 1.}],
                                   cluster_key="seed", value_key="delta", confidence=.8)
    assert actual["cluster_key"] == "seed"
    assert actual["value_key"] == "delta"
    assert actual["confidence"] == .8


@pytest.mark.parametrize("confidence", [0., 1., -.1, 1.1, np.nan, np.inf])
def test_invalid_confidence_fails(confidence):
    with pytest.raises(ValueError, match="confidence"):
        paired_summary([1., 2.], confidence=confidence)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-np.inf], [[1., 2.]]])
def test_nonfinite_or_nonscalar_inputs_fail(values):
    with pytest.raises(ValueError, match="finite scalar"):
        paired_summary(values)


def test_cluster_values_are_validated_before_aggregation():
    with pytest.raises(ValueError, match="finite scalar"):
        cluster_paired_summary([{"demand_seed": 1, "difference": np.inf}])
