import json

import numpy as np

from tools.plot_city_baseline_dataset import complete_evaluations, read_records, rolling


def test_log_reader_deduplicates_resume_and_ignores_partial_append(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    row = dict(phase='train', episode=1, policy='stochastic', seed=43,
               completion_rate=.2, choices={'large_dictionary': 1})
    path.write_text(json.dumps(row) + '\n' + json.dumps(dict(row, completion_rate=.7))
                    + '\n' + '{"phase":')
    rows = read_records(path)
    assert len(rows) == 1 and rows[0]['completion_rate'] == .7
    assert 'choices' not in rows[0]


def test_evaluation_requires_complete_seed_set_and_preserves_undefined_arrival_time():
    rows = [dict(phase='evaluation', episode=100, policy='deterministic', seed=seed,
                 completion_rate=completion, mean_travel_seconds=travel)
            for seed, completion, travel in [(10, 1., 100.), (11, 0., None), (12, .5, 200.)]]
    assert complete_evaluations(rows[:2], [10, 11, 12]) == []
    result = complete_evaluations(rows, [10, 11, 12])
    assert len(result) == 1
    assert result[0]['completion_rate'] == .5
    assert result[0]['mean_travel_seconds'] is None
    assert result[0]['mean_travel_seconds_std'] is None


def test_smoothing_uses_past_values_and_never_future_episodes():
    np.testing.assert_allclose(rolling([2., 4., 100., 6.], 2), [2., 3., 52., 53.])
    np.testing.assert_allclose(rolling([np.nan, 4., 6.], 2), [np.nan, 4., 5.], equal_nan=True)


def test_scheduled_and_bare_evaluations_do_not_mix():
    rows = [dict(phase='evaluation', episode=100, policy=policy, seed=seed,
                 completion_rate=value) for policy, value in [('deterministic', 1.),
                 ('deterministic_no_prior', .2)] for seed in (10, 11, 12)]
    scheduled = complete_evaluations(rows, [10, 11, 12])
    bare = complete_evaluations(rows, [10, 11, 12], policy='deterministic_no_prior')
    assert scheduled[0]['completion_rate'] == 1.
    np.testing.assert_allclose(bare[0]['completion_rate'], .2)
