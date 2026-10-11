import json

import numpy as np
import pytest

from tools import plot_city_baseline_dataset as plotting
from tools.plot_city_baseline_dataset import complete_evaluations, load_reference_results, read_records, rolling


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


def test_manifest_sampled_policy_and_historical_fallback_are_truthful():
    assert plotting.evaluation_settings({'evaluation_policy': 'sampled_temperature_0.2'}) == (
        'sampled_temperature_0.2', 'sampled, T=0.2')
    assert plotting.evaluation_settings({'evaluation_policy': 'sampled_temperature_2'}) == (
        'sampled_temperature_2', 'sampled, T=2')
    assert plotting.evaluation_settings({'evaluation_policy': 'sampled_temperature_2_no_prior',
                                         'policy_prior': {'enabled': True}}) == (
        'sampled_temperature_2_no_prior', 'sampled, T=2, prior=0')
    assert plotting.evaluation_settings({}) == ('deterministic', 'argmax')
    assert plotting.evaluation_settings({'policy_prior': {'enabled': True}}) == (
        'deterministic_no_prior', 'argmax, prior=0')


def test_mixed_action_policies_are_never_combined():
    rows = [dict(phase='evaluation', episode=100, policy=policy, seed=seed,
                 completion_rate=value) for policy, value in [('deterministic', 1.),
                 ('sampled_temperature_2', .2)] for seed in (10, 11, 12)]
    result = complete_evaluations(rows, [10, 11, 12], policy='sampled_temperature_2')
    assert result[0]['completion_rate'] == pytest.approx(.2)


def reference_payload():
    return dict(evaluation_seeds=[10, 11, 12], runs=[dict(
        city='city', method=method, training_episodes=episodes,
        evaluations=[dict(seed=seed, completion_rate=value, mean_exposure_seconds=120.,
                          mean_travel_seconds=60.) for seed, value in [(10, .5), (11, .75), (12, 1.)]])
        for method, episodes in [('shortest', 0), ('dta_so', 1)]])


def test_reference_results_average_same_demand_seeds(tmp_path):
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(reference_payload()))
    result = load_reference_results(path, ['city'], [10, 11, 12])
    assert result[('city', 'shortest')]['completion_rate'] == .75
    assert result[('city', 'dta_so')]['seed_count'] == 3


def test_reference_path_manifest_default_and_explicit_override(tmp_path, monkeypatch):
    root = tmp_path / 'dataset'
    monkeypatch.chdir(tmp_path)
    assert plotting.reference_results_path(root, {}) is None
    assert plotting.reference_results_path(root, {'reference_results': 'references/results.json'}) == root / 'references/results.json'
    absolute = tmp_path / 'shared.json'
    assert plotting.reference_results_path(root, {'reference_results': str(absolute)}) == absolute
    assert plotting.reference_results_path(root, {'reference_results': 'references/results.json'},
                                           plotting.Path('override.json')) == tmp_path / 'override.json'


@pytest.mark.parametrize('mutation', ['partial', 'mismatch', 'duplicate_seed', 'duplicate_run', 'missing_run', 'budget'])
def test_reference_results_reject_incompatible_comparisons(tmp_path, mutation):
    payload = reference_payload()
    if mutation == 'partial':
        payload['runs'][0]['evaluations'].pop()
    elif mutation == 'mismatch':
        payload['evaluation_seeds'] = [10, 11, 13]
    elif mutation == 'duplicate_seed':
        payload['runs'][0]['evaluations'].append(payload['runs'][0]['evaluations'][0])
    elif mutation == 'duplicate_run':
        payload['runs'].append(payload['runs'][0])
    elif mutation == 'missing_run':
        payload['runs'].pop()
    else:
        payload['runs'][1]['training_episodes'] = 2
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_reference_results(path, ['city'], [10, 11, 12])


def test_evaluation_reference_lines_and_legend_leave_training_unchanged(tmp_path, monkeypatch):
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(reference_payload()))
    references = load_reference_results(path, ['city'], [10, 11, 12])
    row = dict(episode=100, completion_rate=.8, completion_rate_std=.01,
               mean_exposure_seconds=130., mean_exposure_seconds_std=1.,
               mean_travel_seconds=70., mean_travel_seconds_std=2.)
    data = {('city', method): dict(train=[row], evaluation=[row]) for method in plotting.METHODS}
    figures = []
    monkeypatch.setattr(plotting, 'save_figure', lambda fig, *_: figures.append(fig))
    plotting.plot_grid(data, ['city'], tmp_path, 'now', evaluation=True, references=references)
    fig = figures[-1]
    labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert 'Random' not in labels
    assert set(labels) == {'Online AC', 'GNAC', 'Dreamer', 'MBPO', 'IPPO', 'MAPPO', 'IQL', 'MAMBA',
                           'Free-flow shortest path', 'DTA-SO (1 iteration)'}
    for ax in fig.axes:
        assert not any('Random' in text.get_text() for text in ax.texts)
        assert ax.get_xlim()[1] < 200
        assert len(ax.patches) == 2
    plotting.plot_grid(data, ['city'], tmp_path, 'now')
    assert 'Random' in [text.get_text() for text in figures[-1].legends[0].get_texts()]
    assert not figures[-1].axes[0].patches
    for fig in figures:
        plotting.plt.close(fig)
