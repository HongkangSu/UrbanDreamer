#!/usr/bin/env python3
"""Dynamic-only SUMO evaluation for the toy online actor-critic policy."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "dreamer4", REPO_ROOT / "dreamer4" / "urban_routing"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline.run_toy_cross_bottleneck_online_ac import (  # noqa: E402
    TOY_NETWORK_NAME,
    ToyOnlineAcTrainer,
    build_config as build_train_config,
    validate_paths,
)
from scripts.eval_sumo_hdv_state import _load_hdv_learning_state  # noqa: E402


def _mean_complete(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    complete = df[df["completion_rate"] >= 0.999999].copy()
    if complete.empty:
        complete = df.copy()
    out: Dict[str, float] = {
        "episodes": float(len(df)),
        "complete_episodes": float(len(complete)),
    }
    for key in ("t_all", "t_CAV", "t_HDV", "reward_mean", "samples", "completion_rate"):
        if key in complete.columns:
            out[f"{key}_mean"] = float(complete[key].mean())
            out[f"{key}_std"] = float(complete[key].std(ddof=0))
    return out


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-dir", default=str(REPO_ROOT / "networks" / TOY_NETWORK_NAME))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / TOY_NETWORK_NAME / "online_ac_dynamic_eval"))
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument("--resume-start-episode", type=int, default=0)
    parser.add_argument("--post-mutation-state-in", default="")
    parser.add_argument("--hdv-state-in", required=True)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio-machines", type=float, default=0.4)
    parser.add_argument("--episode-steps", type=int, default=60)
    parser.add_argument("--sim-buffer-after-last-depart", type=int, default=3000)
    parser.add_argument("--max-steps-per-episode", type=int, default=0)
    parser.add_argument("--reward-mode", default="capped_step_cost", choices=["progress_fraction", "capped_step_cost"])
    parser.add_argument("--reward-temperature", type=float, default=60.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--replay-buffer-episodes", type=int, default=8)
    parser.add_argument("--replay-update-epochs", type=int, default=2)
    parser.add_argument("--replay-actor-max-age-episodes", type=int, default=8)
    parser.add_argument("--return-horizon-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--advantage-clip", type=float, default=2.5)
    parser.add_argument("--ppo-clip-ratio", type=float, default=0.075)
    parser.add_argument("--entropy-target-frac", type=float, default=0.65)
    parser.add_argument("--entropy-floor-coef", type=float, default=0.02)
    parser.add_argument("--hdv-gawron-mode", default="routerl")
    parser.add_argument("--hdv-gawron-alpha-zero", type=float, default=0.2)
    parser.add_argument("--hdv-gawron-beta", type=float, default=-1.5)
    parser.add_argument("--hdv-gawron-update-mode", default="expected")
    parser.add_argument("--incident-edge", default="")
    parser.add_argument("--incident-start", type=float, default=0.0)
    parser.add_argument("--incident-end", type=float, default=0.0)
    parser.add_argument("--incident-speed", type=float, default=0.0)
    parser.add_argument("--sumo-time-to-teleport", default="-1")
    parser.add_argument("--sumo-extra-args", default="")
    parser.add_argument(
        "--disable-human-learning-after-mutation",
        dest="disable_human_learning_after_mutation",
        action="store_true",
        help="Freeze HDV learning during dynamic evaluation.",
    )
    parser.add_argument(
        "--keep-human-learning-after-mutation",
        dest="disable_human_learning_after_mutation",
        action="store_false",
        help="Keep HDV learning enabled during dynamic evaluation.",
    )
    parser.set_defaults(
        settings="",
        av_execution_mode="dynamic",
        generate_network=False,
        n_agents=320,
        depart_span=420,
        train_episodes=1,
        checkpoint_episodes="",
        dry_run=False,
        disable_human_learning_after_mutation=False,
    )
    return parser.parse_args(argv)


def build_eval_config(args: argparse.Namespace):
    """Resolve the training-compatible config with deterministic evaluation actions."""
    cfg = build_train_config(args)
    cfg.disable_human_learning_after_mutation = bool(args.disable_human_learning_after_mutation)
    cfg.real_policy_route_selection = "argmax"
    return cfg


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    if _load_hdv_learning_state(Path(args.hdv_state_in)) is None:
        raise FileNotFoundError(f"hdv state could not be loaded: {args.hdv_state_in}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_eval_config(args)
    validate_paths(cfg)
    trainer = ToyOnlineAcTrainer(cfg)
    collector = trainer._build_collector()
    collector._av_execution_mode = "dynamic"
    rows: List[Dict[str, Any]] = []
    started = time.time()
    try:
        for idx in range(int(args.eval_episodes)):
            episode = int(args.resume_start_episode) + idx + 1
            _samples, row = collector.run_train_episode(episode, int(trainer.rollout_max_steps))
            row["eval_mode"] = "dynamic"
            rows.append(row)
            print(
                f"[toy-online-ac-dynamic-eval] ep={idx + 1:03d}/{int(args.eval_episodes):03d} "
                f"t_all={row.get('t_all', float('nan')):.3f} "
                f"t_CAV={row.get('t_CAV', float('nan')):.3f} "
                f"t_HDV={row.get('t_HDV', float('nan')):.3f} "
                f"completion={row.get('completion_rate', float('nan')):.6f}",
                flush=True,
            )
    finally:
        try:
            collector.finalize_protocol_xml_aliases()
        finally:
            close = getattr(getattr(collector, "connector", None), "close", None)
            if callable(close):
                close()

    pd.DataFrame(rows).to_csv(output_dir / "dynamic_episodes.csv", index=False)
    decision_df = pd.DataFrame(getattr(collector, "_dynamic_decision_rows", []))
    if not decision_df.empty:
        decision_df.to_csv(output_dir / "dynamic_decisions.csv", index=False)
    agent_df = pd.DataFrame(getattr(collector, "_dynamic_agent_decision_rows", []))
    if not agent_df.empty:
        agent_df.to_csv(output_dir / "dynamic_agent_decisions.csv", index=False)

    summary = {
        "checkpoint": str(args.resume_checkpoint),
        "post_mutation_state_in": str(args.post_mutation_state_in),
        "hdv_state_in": str(args.hdv_state_in),
        "eval_episodes": int(args.eval_episodes),
        "dynamic_routing": True,
        "hdv_updates_disabled": bool(args.disable_human_learning_after_mutation),
        "human_learning_after_mutation": not bool(args.disable_human_learning_after_mutation),
        "summary": _mean_complete(rows),
        "episodes_csv": str(output_dir / "dynamic_episodes.csv"),
        "decisions_csv": str(output_dir / "dynamic_decisions.csv"),
        "agent_decisions_csv": str(output_dir / "dynamic_agent_decisions.csv"),
        "elapsed_sec": float(time.time() - started),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
