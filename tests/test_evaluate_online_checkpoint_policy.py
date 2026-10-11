import pytest

from tools.evaluate_online_checkpoint import validate_baseline_policy
from urban_dreamer.learning.training import evaluation_protocol


def test_legacy_argmax_baseline_cannot_be_paired_with_explicit_sampled_evaluation():
    with pytest.raises(ValueError, match='Baseline evaluation policy differs'):
        validate_baseline_policy([{'metrics': {}}], evaluation_protocol(2.))
    validate_baseline_policy([{'metrics': {}}], evaluation_protocol(None))


def test_sampled_temperature_must_match_baseline_policy():
    rows = [{'metrics': {'evaluation_policy': 'zero_prior_sampled_temperature_2'}}]
    validate_baseline_policy(rows, evaluation_protocol(2.))
    with pytest.raises(ValueError, match='Baseline evaluation policy differs'):
        validate_baseline_policy(rows, evaluation_protocol(.2))
