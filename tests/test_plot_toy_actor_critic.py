import pytest

pytest.importorskip("matplotlib")

from tools.plot_toy_actor_critic import compare_runs, plot, report, summarize, write_comparison


def record(phase, policy="deterministic", episode=0, seed=1, **kwargs):
    return dict(phase=phase, policy=policy, episode=episode, seed=seed,
                completion_rate=1., mean_travel_seconds=10., reward_per_vehicle=-1.,
                actor_gradient_norm=1., critic_gradient_norm=1., **kwargs)


def test_partial_run_requires_complete_holdout_suite():
    protocol = {"args": {"episodes": 1, "eval_seeds": 2}}
    rows = [record("train", "stochastic", episode=1)]
    for policy in ("random", "shortest", "deterministic", "stochastic"):
        rows.extend(record("holdout_final", policy, episode=1, seed=seed) for seed in (1, 2))
    rows.append(record("holdout_best", episode=0))
    assert summarize(rows, protocol)["status"] == "incomplete"
    rows.append(record("holdout_best", episode=0, seed=2))
    summary = summarize(rows, protocol)
    assert summary["status"] == "completed"
    assert summary["train_first_20_travel"]["count"] == 1
    assert summary["gradients_finite"] is True


def test_plateau_excludes_partially_logged_evaluation():
    protocol = {"args": {"episodes": 30, "eval_seeds": 2}}
    rows = [record("evaluation", episode=10, seed=seed) for seed in (1, 2)]
    rows.append(record("evaluation", episode=20))
    summary = summarize(rows, protocol)
    assert summary["recent_evaluation_episodes"] == [10]
    assert summary["gradients_finite"] is None
    assert summary["status"] == "incomplete"


def protocol(global_weight=0):
    return {"args": {"episodes": 30, "eval_seeds": 2, "seed": 42},
            "vehicle_count": 320, "deadline_seconds": 3600,
            "reward": "-elapsed/30", "gamma": 1., "lambda_return": .95,
            "global_reward_weight": global_weight,
            "global_reward": "-active_seconds/(N*60)",
            "global_critic_observation": "flattened physical dynamic flow + progress/activefraction"}


def test_report_describes_actual_reward_mode():
    local, global_ = protocol(), protocol(1.)
    summary = summarize([], local)
    assert "no learned macro critic" in report(summary, local)
    text = report(summary, global_)
    assert "no learned macro critic" not in text
    assert "learned global critic" in text
    assert "-active_seconds/(N*60)" in text
    assert "Run incomplete" in text


def test_comparison_matches_evaluation_seeds_and_marks_partial(tmp_path):
    baseline = [record("evaluation", episode=10, seed=seed) for seed in (1, 2)]
    current = [record("evaluation", episode=10, seed=seed) for seed in (2, 3)]
    comparison = compare_runs(baseline, protocol(), current, protocol(1.))
    assert comparison["matched_record_count"] == 1
    assert comparison["baseline_groups"][0]["seeds"] == [2]
    assert comparison["current_groups"][0]["seeds"] == [2]
    assert comparison["same_training_seed"] is True
    assert comparison["status"] == "incomplete"
    write_comparison(baseline, protocol(), current, protocol(1.), tmp_path)
    assert (tmp_path / "comparison.png").stat().st_size > 0
    assert (tmp_path / "comparison.json").exists()
    assert "Partial snapshot" in (tmp_path / "comparison.md").read_text()


def test_global_only_missing_local_critic_is_expected(tmp_path):
    p = protocol(1.)
    p["local_reward_weight"] = 0.
    p["advantage_normalization"] = "scale_only_std_floor_1"
    row = record("train", "stochastic", episode=1, global_critic_gradient_norm=.2,
                 global_critic_loss=2., global_explained_variance=.1)
    del row["critic_gradient_norm"]
    summary = summarize([row], p)
    assert summary["gradients_finite"] is True
    assert "Global-only advantage" in report(summary, p)
    plot([row], summary, tmp_path)
    assert (tmp_path / "learning_curves.png").stat().st_size > 0
    comparison = compare_runs([], protocol(), [row], p)
    assert comparison["baseline_reward_mode"] == "Local only"
    assert comparison["current_reward_mode"] == "Global-only"
    write_comparison([], protocol(), [row], p, tmp_path)
    assert "versus Global-only" in (tmp_path / "comparison.md").read_text()
    assert "not a single-factor reward ablation" in (tmp_path / "comparison.md").read_text()


def test_uniform_entropy_accounts_for_toy_branch_mix():
    import math
    from tools.plot_toy_actor_critic import uniform_branch_entropy
    row=dict(vehicle_count=320,branch_event_count=640,completion_rate=1.)
    assert uniform_branch_entropy(row)==pytest.approx((math.log(3)+math.log(2))/2)
    assert uniform_branch_entropy(dict(row,completion_rate=.9)) is None
    assert uniform_branch_entropy({}) is None
