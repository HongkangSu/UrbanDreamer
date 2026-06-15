#!/usr/bin/env python3
"""Run the Gretz online actor-critic no-CFM ablation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ablation.online_no_cfm_actor_critic import (
    DEFAULT_CHECKPOINT_EPISODES,
    GRETZ_TAG,
    OnlineNoCfmConfig,
    OnlineNoCfmTrainer,
    build_eval_command,
    build_gretz_ablation_plan,
    validate_required_paths,
)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gretz online actor-critic ablation without WM CFM transitions.",
    )
    parser.add_argument("--settings", default=None, help="Gretz .env settings file.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--resume-checkpoint", default="", help="Optional no-CFM actor checkpoint to continue from.")
    parser.add_argument(
        "--resume-start-episode",
        type=int,
        default=0,
        help="Episode number already completed in --resume-checkpoint; defaults to checkpoint metadata.",
    )
    parser.add_argument(
        "--checkpoint-episodes",
        default=",".join(str(v) for v in DEFAULT_CHECKPOINT_EPISODES),
        help="Comma-separated online training episodes to checkpoint and evaluate.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reward-mode",
        default="progress_fraction",
        choices=["progress_fraction", "capped_step_cost"],
        help="Online no-CFM vehicle reward. capped_step_cost uses -min(step_interval, remaining_route_tt).",
    )
    parser.add_argument(
        "--reward-temperature",
        type=float,
        default=60.0,
        help="Seconds scale for capped_step_cost; reward is -min(step_interval, remaining_route_tt) / temperature.",
    )
    parser.add_argument(
        "--replay-buffer-episodes",
        type=int,
        default=1,
        help="Number of recent real SUMO training episodes kept for actor-critic updates.",
    )
    parser.add_argument(
        "--replay-update-epochs",
        type=int,
        default=1,
        help="Gradient epochs per online update over the replay buffer.",
    )
    parser.add_argument(
        "--replay-actor-max-age-episodes",
        type=int,
        default=8,
        help="Maximum policy age, in episodes, for samples used in a replay update.",
    )
    parser.add_argument(
        "--return-horizon-steps",
        type=int,
        default=8,
        help="Future real vehicle decisions included in online returns; <=0 uses the full remaining vehicle trajectory.",
    )
    parser.add_argument("--eval-python", default=sys.executable)
    parser.add_argument(
        "--sumo-extra-args",
        default=None,
        help="Extra SUMO arguments used during online training and appended to eval_sumo.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan and exit.")
    parser.add_argument("--skip-eval", action="store_true", help="Train and save checkpoints, but do not run eval_sumo.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow dry-run output even when Gretz artifacts are missing.",
    )
    parser.add_argument(
        "eval_passthrough",
        nargs=argparse.REMAINDER,
        help="Extra arguments after -- are appended to eval_sumo.py.",
    )
    return parser.parse_args(argv)


def _parse_checkpoint_episodes(raw: str) -> tuple[int, ...]:
    out = []
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return tuple(sorted(set(out))) or DEFAULT_CHECKPOINT_EPISODES


def build_config(args: argparse.Namespace) -> OnlineNoCfmConfig:
    passthrough = list(args.eval_passthrough or [])
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    overrides: Dict[str, object] = {
        "train_episodes": int(args.train_episodes),
        "eval_episodes": int(args.eval_episodes),
        "checkpoint_episodes": _parse_checkpoint_episodes(args.checkpoint_episodes),
        "resume_checkpoint": str(args.resume_checkpoint or ""),
        "resume_start_episode": int(args.resume_start_episode),
        "device": str(args.device),
        "seed": int(args.seed),
        "reward_mode": str(args.reward_mode),
        "reward_temperature": float(args.reward_temperature),
        "replay_buffer_episodes": int(args.replay_buffer_episodes),
        "replay_update_epochs": int(args.replay_update_epochs),
        "replay_actor_max_age_episodes": int(args.replay_actor_max_age_episodes),
        "return_horizon_steps": int(args.return_horizon_steps),
        "eval_python": str(args.eval_python),
        "extra_eval_args": tuple(passthrough),
    }
    if args.sumo_extra_args is not None:
        overrides["sumo_extra_args"] = str(args.sumo_extra_args)
    if args.output_dir:
        overrides["output_dir"] = str(args.output_dir)
    if args.settings:
        return OnlineNoCfmConfig.from_settings(str(args.settings), **overrides)
    return OnlineNoCfmConfig(**overrides).resolve()


def run_evaluations(cfg: OnlineNoCfmConfig, train_summary: Dict[str, object]) -> Dict[str, object]:
    eval_root = Path(cfg.output_dir) / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, object] = {}
    checkpoint_paths = train_summary.get("checkpoint_paths", {})
    if not isinstance(checkpoint_paths, dict):
        checkpoint_paths = {}
    for episode in cfg.checkpoint_episodes:
        ckpt = checkpoint_paths.get(int(episode)) or checkpoint_paths.get(str(int(episode)))
        if not ckpt:
            ckpt = str(Path(cfg.output_dir) / "checkpoints" / f"checkpoint_ep{int(episode):05d}.pt")
        results_dir = eval_root / f"checkpoint_ep{int(episode):05d}"
        cmd = build_eval_command(cfg=cfg, checkpoint_path=ckpt, results_dir=results_dir)
        print("[online-no-cfm] eval command:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        summary_path = results_dir / "collector_eval_summary.json"
        outputs[str(int(episode))] = {
            "checkpoint": str(ckpt),
            "results_dir": str(results_dir),
            "summary_json": str(summary_path),
        }
    out_path = Path(cfg.output_dir) / "online_eval_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    return outputs


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    plan = build_gretz_ablation_plan(cfg)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        if not args.allow_missing:
            validate_required_paths(plan)
        return 0
    validate_required_paths(plan)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "online_no_cfm_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    with (output_dir / "online_no_cfm_plan.json").open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    trainer = OnlineNoCfmTrainer(cfg)
    train_summary = trainer.train()
    if not args.skip_eval:
        eval_summary = run_evaluations(cfg, train_summary)
        train_summary["eval"] = eval_summary
    with (output_dir / "online_no_cfm_final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
