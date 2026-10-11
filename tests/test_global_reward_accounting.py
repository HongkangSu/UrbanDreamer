import numpy as np
import pytest

from urban_dreamer.learning.global_reward import (
    RewardAccounting, causal_context, observed_mean_seconds, started_count_from_state,
)


def test_new_starts_receive_exact_normalization_correction():
    ledger = RewardAccounting()
    rewards = [ledger.update(60., 1), ledger.update(0., 2), ledger.update(60., 2)]
    assert rewards == pytest.approx([-1., .5, -.5])
    assert sum(rewards) == pytest.approx(-ledger.mean_seconds / 60.)


def test_return_invariant_to_accounting_partition_and_unfinished_trips():
    split = RewardAccounting()
    rewards = [split.update(30., 1), split.update(30., 2), split.update(60., 2)]
    whole = RewardAccounting()
    assert sum(rewards) == pytest.approx(whole.update(120., 2))
    assert whole.mean_seconds == 60.
    assert RewardAccounting().update(0., 0) == 0.


def test_cohort_excludes_never_started_and_starts_at_cutoff():
    state = dict(vehicles=[
        dict(departed=True, actual_departure_time=0.),
        dict(departed=True, actual_departure_time=30.),
        dict(departed=False, departure_time=0.),
    ])
    assert started_count_from_state(state, deadline=30.) == 1
    assert observed_mean_seconds(60., started_count_from_state(state, deadline=30.)) == 60.


def test_causal_context_carries_invertible_accounting():
    context = causal_context(30., 60., 1, 3, 120.)
    np.testing.assert_allclose(context[:2], [.5, 1/3])
    assert np.expm1(context[2]) * 60 == pytest.approx(120.)
    assert np.expm1(context[3]) == pytest.approx(3.)


@pytest.mark.parametrize('seconds,count', [(float('nan'), 1), (-1., 1), (1., -1), (1., True)])
def test_invalid_accounting_rejected(seconds, count):
    with pytest.raises(ValueError):
        observed_mean_seconds(seconds, count)
