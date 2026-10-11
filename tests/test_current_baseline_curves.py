import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('current_curves', Path(__file__).parents[1] / 'tools' / 'plot_current_three_city_baselines.py')
curves = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curves)


def row(episode=1, seed=10042, **kwargs):
    return dict(phase='evaluation', episode=episode, policy='deterministic', seed=seed,
                completion_rate=.5, **kwargs)


def test_canonical_reward_uses_started_not_total_or_ambiguous_mean():
    normalized = curves.normalize(row(exposure_seconds=1200, vehicle_count=20, never_departed=10,
                                      mean_exposure_seconds=60))
    assert normalized['att_seconds'] == 120
    assert normalized['canonical_reward_minutes'] == -2
    assert curves.canonical_att({'mean_exposure_seconds': 60, 'vehicle_count': 20})[0] is None
    assert curves.canonical_att({'exposure_seconds': 1200, 'vehicle_count': 20})[0] is None


def test_observed_and_started_accounting_precedence():
    assert curves.canonical_att({'observed_mean_seconds': 90, 'exposure_seconds': 1200,
                                 'started_vehicle_count': 10})[0] == 90
    assert curves.canonical_att({'observed_mean_seconds': float('nan'), 'cumulative_vehicle_seconds': 1800,
                                 'exposure_seconds': 1200, 'started_vehicle_count': 10})[0] == 180
    assert curves.canonical_att({'exposure_seconds': 0, 'started_vehicle_count': 0})[0] is None
    assert curves.canonical_att({'exposure_seconds': 600, 'vehicle_count': 5, 'never_departed': 6})[0] is None


def test_snapshot_last_duplicate_wins_and_partial_tail_ignored(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    first = row(observed_mean_seconds=60)
    replay = row(observed_mean_seconds=120)
    path.write_text(json.dumps(first) + '\n' + json.dumps(replay) + '\n' + '{"phase":')
    records, flags = curves.read_snapshot(path)
    assert len(records) == 1
    assert records[0]['canonical_reward_minutes'] == -2
    assert flags['duplicate_rows'] == 1
    assert flags['partial_lines_ignored'] == 1
    assert flags['complete_lines'] == 2


def test_three_unique_eval_seeds_and_no_optimistic_missing_metric_average():
    seeds = [10042, 10043, 10044]
    records = [curves.normalize(row(seed=seed, observed_mean_seconds=60 * index))
               for index, seed in enumerate(seeds, start=1)]
    records += [curves.normalize(row(episode=2, seed=seed, observed_mean_seconds=60)) for seed in seeds[:2]]
    records += [curves.normalize(row(episode=3, seed=seed, **({'observed_mean_seconds': 60} if seed != seeds[0] else {})))
                for seed in seeds]
    complete, incomplete = curves.complete_evaluations(records, seeds)
    assert [r['episode'] for r in complete] == [1, 3]
    assert incomplete == 1
    assert complete[0]['canonical_reward_minutes'] == -2
    assert complete[0]['canonical_reward_minutes_std'] == pytest.approx((2 / 3) ** .5)
    assert complete[1]['canonical_reward_minutes'] is None
    assert complete[1]['completion_rate'] == .5


def test_eval_repeated_seed_does_not_replace_missing_seed():
    seeds = [10042, 10043, 10044]
    records = [curves.normalize(row(seed=seed, observed_mean_seconds=60)) for seed in [10042, 10042, 10043]]
    assert curves.complete_evaluations(records, seeds) == ([], 1)


def test_smoothing_resets_at_objective_episode_and_unknown_gaps():
    rows = [dict(episode=episode, objective=objective, score=score) for episode, objective, score in
            [(1, 'legacy_unmarked', 10), (2, 'legacy_unmarked', 20),
             (3, 'observed_mean', 100), (5, 'observed_mean', 200),
             (6, 'observed_mean', None), (7, 'observed_mean', 400), (8, 'observed_mean', 600)]]
    assert curves.smooth_segments(rows, 'score', 50) == [10, 15, 100, 200, None, 400, 500]
    assert [len(chunk) for chunk in curves.segments(rows)] == [2, 1, 4]


def test_missing_file_is_an_explicit_empty_run(tmp_path):
    records, flags = curves.read_snapshot(tmp_path / 'absent.jsonl')
    assert records == []
    assert flags['exists'] is False
