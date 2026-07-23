from pathlib import Path

import pytest

from scripts.augment_three_city_eval_extended_metrics import (
    EpisodeData,
    Roles,
    _linear_percentile,
    _reference_stats,
    _tail_metrics,
    _training_cost,
)


def _episode(episode_id, seen, times, *, cav_ids=None, hdv_ids=None):
    return EpisodeData(
        episode_id=episode_id,
        path=Path(f"ep{episode_id}.csv"),
        seen_ids=frozenset(seen),
        travel_time_seconds=times,
        cav_ids=None if cav_ids is None else frozenset(cav_ids),
        hdv_ids=None if hdv_ids is None else frozenset(hdv_ids),
    )


def test_reference_stats_use_group_specific_completion_adjustment():
    roles = Roles(cav=frozenset({"c"}), hdv=frozenset({"h1", "h2"}))
    episodes = [
        _episode(1, roles.all, {"c": 60.0, "h1": 120.0, "h2": 180.0}),
        _episode(2, {"c", "h1"}, {"c": 60.0, "h1": 120.0}),
    ]

    means, audit = _reference_stats(episodes, roles)

    assert means["CAV"] == pytest.approx(60.0)
    assert means["HDV"] == pytest.approx((150.0 + 240.0) / 2.0)
    assert means["all"] == pytest.approx((120.0 + 135.0) / 2.0)
    assert audit["mean_completion_rate"]["HDV"] == pytest.approx(0.75)
    assert audit["all_reference_episodes_complete"] is False


def test_training_cost_does_not_treat_missing_travel_time_as_zero():
    roles = Roles(cav=frozenset({"c"}), hdv=frozenset({"h1", "h2"}))
    episodes = [
        _episode(1, roles.all, {"c": 120.0, "h1": 180.0}),
    ]
    reference = {"all": 100.0, "CAV": 60.0, "HDV": 120.0}

    result = _training_cost(episodes, roles, reference)

    assert result["c_seconds"]["CAV"] == pytest.approx(60.0)
    assert result["c_seconds"]["HDV"] == pytest.approx(240.0)
    assert result["c_seconds"]["all"] == pytest.approx(125.0)
    assert result["arrival_completion_rate"] == pytest.approx(2.0 / 3.0)


def test_training_cost_matches_reference_to_observed_episode_role_bundle():
    roles = Roles(cav=frozenset({"c1"}), hdv=frozenset({"c2", "h"}))
    reference_episodes = [
        _episode(1, roles.all, {"c1": 60.0, "c2": 120.0, "h": 180.0}),
    ]
    reference, _ = _reference_stats(reference_episodes, roles)
    training_episodes = [
        _episode(
            1,
            roles.all,
            {"c1": 120.0, "c2": 240.0, "h": 360.0},
            cav_ids={"c2"},
            hdv_ids={"c1", "h"},
        )
    ]

    result = _training_cost(
        training_episodes,
        roles,
        reference,
        reference_episodes=reference_episodes,
    )

    assert result["c_seconds"] == pytest.approx({"all": 120.0, "CAV": 120.0, "HDV": 120.0})
    assert result["reference_t_pre_minutes"]["CAV"] == pytest.approx(2.0)
    assert result["role_alignment"] == "per_episode_vType_bundle"
    assert result["role_bundles"] == 1


def test_linear_percentile_matches_standard_linear_interpolation():
    assert _linear_percentile([0.0, 10.0, 20.0, 30.0, 40.0], 95.0) == pytest.approx(38.0)


def test_tail_metrics_exclude_missing_times_and_mark_right_censoring():
    roles = Roles(cav=frozenset({"c"}), hdv=frozenset({"h1", "h2"}))
    episodes = [
        _episode(1, roles.all, {"c": 60.0, "h1": 120.0}),
        _episode(2, roles.all, {"c": 180.0, "h1": 240.0, "h2": 300.0}),
    ]

    result = _tail_metrics(episodes, roles)

    assert result["arrived_rows"] == 5
    assert result["expected_rows"] == 6
    assert result["p95_minutes"] == pytest.approx(4.8)
    assert result["tt_max_minutes"] == pytest.approx(5.0)
    assert result["scope"] == "arrived_only"
    assert result["right_censored"] is True
