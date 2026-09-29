#!/usr/bin/env python3
"""Summarize a real-SUMO Actor–Critic diagnostic, including partial runs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"count": len(values), "mean": float(np.mean(values)),
            "std": float(np.std(values)), "min": min(values), "max": max(values)}


def grouped_metrics(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["phase"], record["policy"], record["episode"])].append(record)
    return [dict(phase=phase, policy=policy, episode=episode, seeds=[r["seed"] for r in rows],
                 **{metric: stats([r.get(metric) for r in rows]) for metric in
                    ("mean_travel_seconds", "reward_per_vehicle", "completion_rate", "branch_entropy")})
            for (phase, policy, episode), rows in sorted(groups.items())]


def uniform_branch_entropy(record):
    # In this acyclic Toy, each completed trip has one 3-way fork and at most
    # one 2-way fork. Account for changing branch composition before judging
    # whether a trained distribution is more confident than uniform sampling.
    vehicles = record.get("vehicle_count", 0)
    branches = record.get("branch_event_count", 0)
    if record.get("completion_rate") != 1. or not 0 < vehicles <= branches <= 2 * vehicles:
        return None
    return (vehicles * math.log(3.) + (branches - vehicles) * math.log(2.)) / branches


def summarize(records, protocol):
    train = sorted((r for r in records if r["phase"] == "train"), key=lambda r: r["episode"])
    groups = grouped_metrics([r for r in records if r["phase"] != "train"])
    evaluation = [g for g in groups if g["phase"] == "evaluation" and g["policy"] == "deterministic"]
    expected_seeds = protocol["args"]["eval_seeds"]
    complete_evaluation = [g for g in evaluation if len(set(g["seeds"])) == expected_seeds]
    last_three = complete_evaluation[-3:]
    intended = protocol["args"]["episodes"]
    last = train[-1]["episode"] if train else 0
    holdouts = [g for g in groups if g["phase"] == "holdout_final" and g["episode"] == intended]
    best_holdouts = [g for g in groups if g["phase"] == "holdout_best"]
    complete = (last >= intended and len(holdouts) == 4
                and all(len(set(g["seeds"])) == expected_seeds for g in holdouts)
                and any(len(set(g["seeds"])) == expected_seeds for g in best_holdouts))
    local_weight = protocol.get("local_reward_weight", protocol["args"].get("local_reward_weight", 1.))
    gradient_keys = ("actor_gradient_norm", "critic_gradient_norm") if local_weight else ("actor_gradient_norm",)
    gradients = [r.get(key) for r in train for key in gradient_keys]
    gradients.extend(r["global_critic_gradient_norm"] for r in train if "global_critic_gradient_norm" in r)
    return dict(status="completed" if complete else "incomplete", world_model=False,
                intended_training_episodes=intended, completed_training_episodes=len(train),
                last_training_episode=last, groups=groups,
                train_first_20_travel=stats([r.get("mean_travel_seconds") for r in train[:20]]),
                train_last_20_travel=stats([r.get("mean_travel_seconds") for r in train[-20:]]),
                train_last_20_global={key: stats([r.get(key) for r in train[-20:]]) for key in
                                      ("global_critic_loss", "global_explained_variance", "global_advantage_std",
                                       "local_advantage_std", "window_count", "global_reward_sum")},
                recent_evaluation_episodes=[g["episode"] for g in last_three],
                recent_evaluation_travel=stats([g["mean_travel_seconds"]["mean"] for g in last_three]),
                last_20_entropy_deficit=stats([
                    max(0., uniform_branch_entropy(r) - r["branch_entropy"])
                    for r in train[-20:] if uniform_branch_entropy(r) is not None]),
                gradients_finite=all(v is not None and math.isfinite(v) for v in gradients) if gradients else None,
                every_episode_fully_completed=all(r["completion_rate"] == 1 for r in records) if records else None,
                noncompletion_records=sum(r["completion_rate"] != 1 for r in records),
                logged_episodes=len(records), total_logged_wall_seconds=sum(r.get("wall_seconds", 0) for r in records))


def plot(records, summary, output):
    train = sorted((r for r in records if r["phase"] == "train"), key=lambda r: r["episode"])
    x = [r["episode"] for r in train]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    ax = axes[0, 0]
    travel = np.asarray([r.get("mean_travel_seconds") if r.get("mean_travel_seconds") is not None else np.nan for r in train])
    ax.plot(x, travel, alpha=.3, label="Stochastic training episodes")
    rolling = [float(np.mean(travel[max(0, i-9):i+1])) for i in range(len(travel))]
    ax.plot(x, rolling, label="Trailing mean (up to 10 episodes)")
    ax.set(title="Training travel time (arrived vehicles only)", ylabel="seconds")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    evaluations = [g for g in summary["groups"] if g["phase"] == "evaluation" and g["policy"] == "deterministic"]
    ex = [g["episode"] for g in evaluations]
    means = np.array([g["mean_travel_seconds"]["mean"] for g in evaluations], dtype=float)
    std = np.array([g["mean_travel_seconds"]["std"] for g in evaluations], dtype=float)
    ax.plot(ex, means, marker="o", label="Deterministic evaluation mean")
    ax.fill_between(ex, means-std, means+std, alpha=.2, label="±1 SD across evaluation seeds")
    labels = {"random": "Uniform random", "shortest": "Static shortest-cost", "deterministic": "Initial deterministic actor"}
    for group in summary["groups"]:
        if group["phase"] == "baseline" and group["mean_travel_seconds"]["mean"] is not None:
            ax.axhline(group["mean_travel_seconds"]["mean"], linestyle="--", label=labels[group["policy"]])
    ax.set(title="Evaluation travel time (arrived vehicles only)", ylabel="seconds")
    ax.legend(fontsize=8)
    for ax, key, title, unit in (
        (axes[1, 0], "reward_per_vehicle", "Local episode reward per vehicle (diagnostic)", "reward"),
        (axes[1, 1], "critic_loss", "Critic regression loss", "MSE"),
        (axes[2, 0], "explained_variance", "Critic explained variance", "fraction"),
        (axes[2, 1], "branch_entropy", "Branch entropy of underlying policy distribution", "nats"),
    ):
        if any(key in r for r in train):
            ax.plot(x, [r.get(key, np.nan) for r in train])
        elif key in ("critic_loss", "explained_variance"):
            ax.text(.5, .95, "Vehicle critic not trained / not logged", transform=ax.transAxes,
                    ha="center", va="top", fontsize=8)
        ax.set(title=title, ylabel=unit)
    ceiling = [uniform_branch_entropy(r) for r in train]
    if any(value is not None for value in ceiling):
        if axes[2, 1].lines:
            axes[2, 1].lines[0].set_label("Observed policy entropy")
        axes[2, 1].plot(x, ceiling, linestyle="--", label="Uniform upper bound for branch mix")
        axes[2, 1].legend(fontsize=8)
    if any("global_critic_loss" in r for r in train):
        if axes[1, 1].lines:
            axes[1, 1].lines[0].set_label("Vehicle critic")
        axes[1, 1].plot(x, [r.get("global_critic_loss", np.nan) for r in train], label="Global critic")
        axes[1, 1].legend(fontsize=8)
        if axes[2, 0].lines:
            axes[2, 0].lines[0].set_label("Vehicle critic")
        axes[2, 0].plot(x, [r.get("global_explained_variance", np.nan) for r in train], label="Global critic")
        axes[2, 0].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Training episode")
        ax.grid(alpha=.2)
    fig.suptitle(f"Toy real-SUMO Actor–Critic · no world model · {summary['status']}")
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)


def number(value):
    return "n/a" if value is None else f"{value:.4f}"


def reward_label(protocol):
    local = protocol.get("local_reward_weight", protocol["args"].get("local_reward_weight", 1.))
    global_ = protocol.get("global_reward_weight", protocol["args"].get("global_reward_weight", 0.))
    return "Global-only" if global_ and not local else "Local + global" if global_ else "Local only"


def report(summary, protocol):
    first, last = summary["train_first_20_travel"], summary["train_last_20_travel"]
    recent = summary["recent_evaluation_travel"]
    weight = protocol.get("global_reward_weight", protocol["args"].get("global_reward_weight", 0.))
    local_weight = protocol.get("local_reward_weight", protocol["args"].get("local_reward_weight", 1.))
    reward_mode = (f"{reward_label(protocol)} advantage (local weight {local_weight}, global weight {weight}); learned global critic. "
                   f"Global reward: `{protocol.get('global_reward', '-active_seconds/(N*60)')}`. "
                   f"Global critic observation: {protocol.get('global_critic_observation', 'not recorded')}."
                   if weight else "Local reward only; no learned macro critic.")
    lines = ["# Toy real-SUMO Actor–Critic diagnostic", "",
             f"Status: **{summary['status']}**. Logged {summary['completed_training_episodes']} training episodes; "
             f"requested {summary['intended_training_episodes']}. This report is a snapshot of the available log.", "",
             f"Training seed: {protocol['args']['seed']}; vehicles: {protocol['vehicle_count']}; "
             f"deadline: {protocol['deadline_seconds']} seconds. No world model or imagination. {reward_mode}", "",
             f"Reward: `{protocol['reward']}`; gamma={protocol['gamma']}, lambda={protocol['lambda_return']}.", "",
             f"Advantage normalization: `{protocol.get('advantage_normalization', 'center_and_scale')}`.", "",
             "## Baseline, evaluation, and holdout results", "",
             "Travel statistics below use arrived vehicles only. SD is population SD across SUMO evaluation seeds, "
             "not uncertainty across independent training runs. Holdout seeds differ from checkpoint-selection seeds.", "",
             "| Phase | Policy | Checkpoint episode | Runs | Travel seconds mean ± SD | Reward mean ± SD | Completion mean |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for g in summary["groups"]:
        t, r = g["mean_travel_seconds"], g["reward_per_vehicle"]
        lines.append(f"| {g['phase']} | {g['policy']} | {g['episode']} | {len(g['seeds'])} | "
                     f"{number(t['mean'])} ± {number(t['std'])} | {number(r['mean'])} ± {number(r['std'])} | "
                     f"{number(g['completion_rate']['mean'])} |")
    lines += ["", "## Learning diagnostics", "",
              f"- First {first['count']} training episodes: travel {number(first['mean'])} ± {number(first['std'])} seconds.",
              f"- Last {last['count']} training episodes: travel {number(last['mean'])} ± {number(last['std'])} seconds.",
              f"- Last up to three fully evaluated checkpoints: {summary['recent_evaluation_episodes']}; "
              f"mean travel range [{number(recent['min'])}, {number(recent['max'])}] seconds.",
              f"- All logged training gradient norms finite: {summary['gradients_finite']}.",
              f"- All vehicles completed in every logged episode: {summary['every_episode_fully_completed']}; "
              f"episodes with noncompletion: {summary['noncompletion_records']}.",
              f"- Sum of logged episode wall time: {summary['total_logged_wall_seconds']:.1f} seconds (excludes update overhead).", "",
              "![Learning curves](learning_curves.png)", "",
              "## Interpretation limits", "",
              "This experiment uses one training seed. Decreasing critic loss alone does not demonstrate policy convergence. "
              "A stable deterministic evaluation curve can reflect unchanged argmax choices while policy probabilities continue to change. "
              "Branch entropy measures the underlying policy distribution even during deterministic evaluation; it does not imply "
              "random actions were executed in that evaluation. The static shortest-cost baseline is not proven optimal under congestion. "
              "Checkpoint selection includes the untrained episode-0 actor, so the best checkpoint may be episode 0.", "",
              "For fewer than 40 training episodes, the first/last 20 groups may overlap; fewer than three completed evaluation "
              "checkpoints are insufficient to assess a three-checkpoint plateau. The report does not automatically declare convergence."]
    lines += ["", "Mean entropy deficit from the uniform branch-mix ceiling in the last 20 training episodes: "
              f"{summary['last_20_entropy_deficit']['mean']}. This is the average KL divergence from each legal "
              "uniform candidate distribution when the completed Toy branch-count contract holds. "
              "A near-zero gap indicates probabilities remain nearly uniform despite argmax route changes."]
    if summary["status"] != "completed":
        lines += ["", "**Run incomplete:** training or the full final/best holdout suite is missing from this snapshot. "
                  "Regenerate the report after the runner completes before drawing final conclusions."]
    if weight:
        lines += ["", "## Global reward diagnostics", "",
                  "Means below cover the last up to 20 logged training episodes. Reward per vehicle above remains "
                  "the local episodic reward as a comparable travel metric; it contributes to policy updates only when "
                  "local reward weight is nonzero. Global-only runs do not train a vehicle critic.", ""]
        lines.extend(f"- {key}: {number(value['mean'])} ({value['count']} episodes)."
                     for key, value in summary["train_last_20_global"].items())
    return "\n".join(lines) + "\n"


def load_run(output):
    protocol = json.loads((output / "protocol.json").read_text())
    raw = (output / "metrics.jsonl").read_text()
    records = []
    for i, line in enumerate(raw.splitlines()):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(raw.splitlines()) - 1 or raw.endswith("\n"):
                raise
    return protocol, records


def compare_runs(baseline_records, baseline_protocol, records, protocol):
    """Compare matched evaluation seeds, never silently pool different seed sets."""
    summaries = [summarize(rows, p) for rows, p in
                 ((baseline_records, baseline_protocol), (records, protocol))]
    by_key = []
    for rows in (baseline_records, records):
        by_key.append({(r["phase"], r["policy"], r["episode"], r["seed"]): r for r in rows
                       if r["phase"] in ("evaluation", "baseline", "holdout_final")})
    shared = sorted(by_key[0].keys() & by_key[1].keys())
    matched = [grouped_metrics([index[key] for key in shared]) for index in by_key]
    return {"status": "completed" if all(s["status"] == "completed" for s in summaries) else "incomplete",
            "baseline_reward_mode": reward_label(baseline_protocol), "current_reward_mode": reward_label(protocol),
            "training_seeds": [p["args"]["seed"] for p in (baseline_protocol, protocol)],
            "same_training_seed": baseline_protocol["args"]["seed"] == protocol["args"]["seed"],
            "advantage_normalization": [p.get("advantage_normalization", "center_and_scale")
                                         for p in (baseline_protocol, protocol)],
            "baseline_status": summaries[0]["status"], "current_status": summaries[1]["status"],
            "baseline_groups": matched[0], "current_groups": matched[1],
            "baseline_holdouts": [g for g in summaries[0]["groups"] if g["phase"].startswith("holdout")],
            "current_holdouts": [g for g in summaries[1]["groups"] if g["phase"].startswith("holdout")],
            "matched_record_count": len(shared)}


def write_comparison(baseline_records, baseline_protocol, records, protocol, output):
    comparison = compare_runs(baseline_records, baseline_protocol, records, protocol)
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for rows, groups, label in ((baseline_records, comparison["baseline_groups"], comparison["baseline_reward_mode"]),
                               (records, comparison["current_groups"], comparison["current_reward_mode"])):
        train = sorted((r for r in rows if r["phase"] == "train"), key=lambda r: r["episode"])
        values = [np.nan if r.get("mean_travel_seconds") is None else r["mean_travel_seconds"] for r in train]
        rolling = [float(np.mean(values[max(0, i-9):i+1])) for i in range(len(values))]
        axes[0].plot([r["episode"] for r in train], rolling, label=label)
        evaluation = [g for g in groups if g["phase"] == "evaluation" and g["policy"] == "deterministic"]
        axes[1].plot([g["episode"] for g in evaluation],
                     [g["mean_travel_seconds"]["mean"] for g in evaluation], marker="o", label=label)
    axes[0].set_title("Training travel: trailing mean of up to 10 episodes")
    axes[1].set_title("Deterministic evaluation: shared checkpoints and seeds")
    for ax in axes:
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Travel seconds (arrived vehicles only)")
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle(f"Toy real SUMO · no world model · comparison {comparison['status']}")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    lines = [f"# Toy {comparison['baseline_reward_mode']} versus {comparison['current_reward_mode']} reward", "",
             f"Status: **{comparison['status']}**. Baseline: {comparison['baseline_status']}; "
             f"current: {comparison['current_status']}. Training seeds: {comparison['training_seeds']}.", "",
             "Training curves use trailing means of up to 10 episodes. Evaluation curves match checkpoint episode "
             "and SUMO seed. Travel times include arrived vehicles only; always inspect completion alongside travel. "
             "This is one training seed per condition, not evidence of general convergence or a replicated causal estimate.", "",
             "![Reward comparison](comparison.png)", "", "## Holdout results", "",
             "Holdouts below are listed separately for each run. Best checkpoints may have different episode numbers. "
             "Compare final policies on the same holdout seeds; best checkpoint selection uses evaluation seeds.", "",
             "| Run | Phase | Policy | Episode | Seeds | Travel seconds mean ± SD | Local reward mean | Completion |",
             "|---|---|---|---:|---|---:|---:|---:|"]
    for label, key in ((comparison["baseline_reward_mode"], "baseline_holdouts"),
                       (comparison["current_reward_mode"], "current_holdouts")):
        for g in comparison[key]:
            travel = g["mean_travel_seconds"]
            lines.append(f"| {label} | {g['phase']} | {g['policy']} | {g['episode']} | {g['seeds']} | "
                         f"{number(travel['mean'])} ± {number(travel['std'])} | "
                         f"{number(g['reward_per_vehicle']['mean'])} | {number(g['completion_rate']['mean'])} |")
    if comparison["status"] != "completed":
        lines += ["", "**Partial snapshot:** final training and/or holdout results are missing. No final conclusion is warranted."]
    if comparison["advantage_normalization"][0] != comparison["advantage_normalization"][1]:
        lines += ["", "**Comparison limitation:** reward mode and advantage normalization both changed. "
                  f"Baseline uses `{comparison['advantage_normalization'][0]}`; current uses "
                  f"`{comparison['advantage_normalization'][1]}`. This is not a single-factor reward ablation. "
                  "The global-only update retains the value-critic baseline but avoids batch mean-centering, "
                  "which would erase an advantage shared by every sampled decision."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--compare-dir", type=Path, help="Existing local-only diagnostic directory")
    args = parser.parse_args(argv)
    output = args.input_dir.resolve()
    protocol, records = load_run(output)
    summary = summarize(records, protocol)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(report(summary, protocol))
    plot(records, summary, output)
    if args.compare_dir:
        baseline_protocol, baseline_records = load_run(args.compare_dir.resolve())
        write_comparison(baseline_records, baseline_protocol, records, protocol, output)
    print(json.dumps({"status": summary["status"], "training_episodes": summary["completed_training_episodes"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
