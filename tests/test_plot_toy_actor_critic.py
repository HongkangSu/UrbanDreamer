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


def global_protocol():
    p = protocol()
    del p["global_reward_weight"]
    p.update(reward_mode="global_only", training_policy="stochastic", evaluation_policy="argmax",
             advantage_normalization="scale_only_std_floor_1",
             reward="-active_seconds/(N*60) - 5000*unfinished",
             global_reward="-active_seconds/(N*60) - 5000*unfinished")
    return p


def test_fixed_global_protocol_labels_team_reward_and_global_gradients():
    p = global_protocol()
    row = record("train", "stochastic", episode=1, global_reward_sum=-5001.,
                 global_critic_gradient_norm=.2, global_critic_loss=2.,
                 global_explained_variance=.1)
    del row["critic_gradient_norm"]
    summary = summarize([row], p)
    assert summary["gradients_finite"] is True
    assert summary["reward_quantity"] == "global_reward_sum"
    assert summary["train_last_20_global"]["global_reward_sum"]["mean"] == -5001.
    assert "local_advantage_std" not in summary["train_last_20_global"]
    text = report(summary, p)
    assert "Global-only advantage" in text
    assert "Team reward" in text
    assert "Training policy: stochastic; evaluation policy: argmax" in text
    assert "local episodic reward" not in text
    assert "local weight" not in text
    del row["global_critic_gradient_norm"]
    assert summarize([row], p)["gradients_finite"] is False


def test_poisson_protocol_reports_expected_population_and_argmax_only_holdout():
    p = global_protocol()
    del p['vehicle_count']
    p['demand'] = dict(expected_vehicle_count=320., law='independent_od_poisson')
    p['evaluation_policies'] = ['deterministic']
    rows = [record('train', 'stochastic', episode=30, global_reward_sum=-3., global_critic_gradient_norm=.1)]
    rows.extend(record('holdout_final', episode=30, seed=seed, global_reward_sum=-2.) for seed in (1, 2))
    rows.extend(record('holdout_best', episode=0, seed=seed, global_reward_sum=-2.) for seed in (1, 2))
    summary = summarize(rows, p)
    assert summary['status'] == 'completed'
    text = report(summary, p)
    assert 'expected vehicles: 320 (Poisson)' in text


def test_individual_timeout_report_does_not_call_penalties_shared_rewards():
    p = global_protocol()
    p['reward_mode'] = 'global_with_individual_timeout'
    p['global_reward'] = '-active_seconds/(N*60)'
    row = record('train', 'stochastic', episode=1, global_reward_sum=-2.,
                 vehicle_timeout_penalty_sum=-5000., selection_score=-5002., global_critic_gradient_norm=.1)
    summary = summarize([row], p)
    assert summary['reward_quantity'] == 'global_reward_sum'
    assert summary['train_last_20_global']['vehicle_timeout_penalty_sum']['mean'] == -5000.
    text = report(summary, p)
    assert 'individual timeout' in text.lower()
    assert 'not a shared training reward' in text


def test_fixed_global_plot_uses_global_quantities(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    p = global_protocol()
    row = record("train", "stochastic", episode=1, global_reward_sum=-5001.,
                 global_critic_gradient_norm=.2, global_critic_loss=2.,
                 global_explained_variance=.1, branch_entropy=.4)
    del row["critic_gradient_norm"]
    figures = []
    close = plt.close
    monkeypatch.setattr(plt, "close", figures.append)
    plot([row], summarize([row], p), tmp_path)
    axes = {ax.get_title(): ax for ax in figures[-1].axes}
    assert list(axes["Team episode reward"].lines[0].get_ydata()) == [-5001.]
    assert list(axes["Global critic regression loss"].lines[0].get_ydata()) == [2.]
    assert list(axes["Global critic explained variance"].lines[0].get_ydata()) == [.1]
    assert list(axes["Global critic gradient norm"].lines[0].get_ydata()) == [.2]
    assert not any("Vehicle critic" in text.get_text() for ax in axes.values() for text in ax.texts)
    close(figures[-1])


def test_final_holdout_must_match_actual_last_training_episode():
    p = global_protocol()
    rows = [record("train", "stochastic", episode=episode, global_critic_gradient_norm=.2)
            for episode in (30, 31)]
    for policy in ("random", "shortest", "deterministic"):
        rows.extend(record("holdout_final", policy, episode=30, seed=seed) for seed in (1, 2))
    rows.extend(record("holdout_best", episode=0, seed=seed) for seed in (1, 2))
    assert summarize(rows, p)["status"] == "incomplete"
    for policy in ("random", "shortest", "deterministic"):
        rows.extend(record("holdout_final", policy, episode=31, seed=seed,
                           global_reward_sum=-3.) for seed in (1, 2))
    summary = summarize(rows, p)
    assert summary["status"] == "completed"
    assert summary["last_training_episode"] == 31
    final = [g for g in summary["groups"] if g["phase"] == "holdout_final" and g["episode"] == 31]
    assert final[0]["global_reward_sum"]["mean"] == -3.


def test_archived_resumed_run_ignores_stale_final_holdout():
    p = protocol(1.)
    p["local_reward_weight"] = 0.
    rows = [record("train", "stochastic", episode=31, global_critic_gradient_norm=.2)]
    for policy in ("random", "shortest", "deterministic", "stochastic"):
        rows.extend(record("holdout_final", policy, episode=30, seed=seed) for seed in (1, 2))
    rows.extend(record("holdout_best", episode=0, seed=seed) for seed in (1, 2))
    assert summarize(rows, p)["status"] == "incomplete"
    for policy in ("random", "shortest", "deterministic", "stochastic"):
        rows.extend(record("holdout_final", policy, episode=31, seed=seed) for seed in (1, 2))
    assert summarize(rows, p)["status"] == "completed"


def test_comparison_separates_new_team_reward_and_archived_local_reward(tmp_path):
    old_rows = [record("holdout_final", episode=30)]
    new_rows = [record("holdout_final", episode=30, global_reward_sum=-5001.)]
    write_comparison(old_rows, protocol(), new_rows, global_protocol(), tmp_path)
    text = (tmp_path / "comparison.md").read_text()
    assert "Logged reward mean" in text
    assert "| Local only | holdout_final | deterministic | 30 | [1] | 10.0000 ± 0.0000 | -1.0000" in text
    assert "| Global-only | holdout_final | deterministic | 30 | [1] | 10.0000 ± 0.0000 | -5001.0000" in text


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
