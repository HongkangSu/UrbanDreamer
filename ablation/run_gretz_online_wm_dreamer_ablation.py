#!/usr/bin/env python3
"""Run the Gretz online world-model Dreamer-style actor-critic ablation."""

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

from ablation.online_wm_dreamer_actor_critic import (
    DEFAULT_ONLINE_WM_CHECKPOINT_EPISODES,
    OnlineWmDreamerConfig,
    OnlineWmDreamerTrainer,
    build_eval_command,
    build_gretz_online_wm_dreamer_plan,
    validate_required_paths,
)


def _parse_checkpoint_episodes(raw: str) -> tuple[int, ...]:
    out: List[int] = []
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return tuple(sorted(set(out))) or DEFAULT_ONLINE_WM_CHECKPOINT_EPISODES


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gretz online Dreamer-style actor-critic with online CFM world-model replay.",
    )
    parser.add_argument("--settings", default=None, help="Gretz .env settings file.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--network-name", default=None)
    parser.add_argument("--network-dir", default=None)
    parser.add_argument("--edge-registry", default=None)
    parser.add_argument("--route-artifacts-dir", default=None)
    parser.add_argument("--post-mutation-state-in", default=None)
    parser.add_argument("--hdv-state-in", default=None)
    parser.add_argument("--projector-ckpt", default=None)
    parser.add_argument("--dynamics-ckpt", default=None)
    parser.add_argument("--ratio-machines", type=float, default=None)
    parser.add_argument("--av-candidate-source", default=None)
    parser.add_argument("--route-generator", default=None)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--resume-checkpoint", default="", help="Optional online-WM checkpoint to continue from.")
    parser.add_argument("--resume-start-episode", type=int, default=0)
    parser.add_argument(
        "--checkpoint-episodes",
        default=",".join(str(v) for v in DEFAULT_ONLINE_WM_CHECKPOINT_EPISODES),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reward-mode",
        default="capped_step_cost",
        choices=["capped_step_cost"],
        help="Imagined vehicle reward; fixed to -min(step_interval, remaining_route_tt)/temperature.",
    )
    parser.add_argument("--reward-temperature", type=float, default=60.0)
    parser.add_argument("--wm-replay-buffer-episodes", type=int, default=8)
    parser.add_argument("--wm-update-epochs", type=int, default=1)
    parser.add_argument("--wm-update-every-episodes", type=int, default=1)
    parser.add_argument("--wm-replay-completion-threshold", type=float, default=1.0)
    parser.add_argument(
        "--allow-incomplete-wm-replay",
        action="store_true",
        help="Allow incomplete SUMO episodes into WM replay. Default is to require completion.",
    )
    parser.add_argument("--wm-batch-size", type=int, default=16)
    parser.add_argument("--wm-lr", type=float, default=1e-5)
    parser.add_argument("--wm-grad-clip", type=float, default=10.0)
    parser.add_argument("--wm-transition-decoded-supervision-weight", type=float, default=1.0)
    parser.add_argument("--wm-hdv-policy-weight", type=float, default=0.05)
    parser.add_argument("--wm-hdv-policy-target-mode", default="episode_mean")
    parser.add_argument("--wm-hdv-policy-action-mode", default="window_mean")
    parser.add_argument("--wm-transition-hdv-context-mode", default="episode_response")
    parser.add_argument("--wm-transition-hdv-context-weight", type=float, default=1.0)
    parser.add_argument("--wm-transition-hdv-context-scale", type=float, default=1.0)
    parser.add_argument("--wm-transition-hdv-action-mode", default="none")
    parser.add_argument("--wm-transition-hdv-action-weight", type=float, default=0.0)
    parser.add_argument("--wm-transition-hdv-action-scale", type=float, default=1.0)
    parser.add_argument("--imag-horizon", type=int, default=8)
    parser.add_argument("--imagined-updates-per-episode", type=int, default=1)
    parser.add_argument("--imagined-rollouts-per-update", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--sim-buffer-after-last-depart", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--eval-python", default=sys.executable)
    parser.add_argument(
        "--sumo-extra-args",
        default=None,
        help="Extra SUMO arguments used during online training and appended to eval_sumo.",
    )
    parser.add_argument(
        "--keep-human-learning-after-mutation",
        dest="disable_human_learning_after_mutation",
        action="store_false",
        help="Keep HDV/RouterL learning active after the post-mutation state during online training and eval.",
    )
    parser.add_argument(
        "--disable-human-learning-after-mutation",
        dest="disable_human_learning_after_mutation",
        action="store_true",
        help="Freeze HDV/RouterL learning after the post-mutation state during online training and eval.",
    )
    parser.set_defaults(disable_human_learning_after_mutation=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "eval_passthrough",
        nargs=argparse.REMAINDER,
        help="Extra arguments after -- are appended to eval_sumo.py.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> OnlineWmDreamerConfig:
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
        "wm_replay_buffer_episodes": int(args.wm_replay_buffer_episodes),
        "wm_update_epochs": int(args.wm_update_epochs),
        "wm_update_every_episodes": int(args.wm_update_every_episodes),
        "wm_replay_require_completion": not bool(args.allow_incomplete_wm_replay),
        "wm_replay_completion_threshold": float(args.wm_replay_completion_threshold),
        "wm_batch_size": int(args.wm_batch_size),
        "wm_lr": float(args.wm_lr),
        "wm_grad_clip": float(args.wm_grad_clip),
        "wm_transition_decoded_supervision_weight": float(args.wm_transition_decoded_supervision_weight),
        "wm_hdv_policy_weight": float(args.wm_hdv_policy_weight),
        "wm_hdv_policy_target_mode": str(args.wm_hdv_policy_target_mode),
        "wm_hdv_policy_action_mode": str(args.wm_hdv_policy_action_mode),
        "wm_transition_hdv_context_mode": str(args.wm_transition_hdv_context_mode),
        "wm_transition_hdv_context_weight": float(args.wm_transition_hdv_context_weight),
        "wm_transition_hdv_context_scale": float(args.wm_transition_hdv_context_scale),
        "wm_transition_hdv_action_mode": str(args.wm_transition_hdv_action_mode),
        "wm_transition_hdv_action_weight": float(args.wm_transition_hdv_action_weight),
        "wm_transition_hdv_action_scale": float(args.wm_transition_hdv_action_scale),
        "imag_horizon": int(args.imag_horizon),
        "imagined_updates_per_episode": int(args.imagined_updates_per_episode),
        "imagined_rollouts_per_update": int(args.imagined_rollouts_per_update),
        "disable_human_learning_after_mutation": bool(args.disable_human_learning_after_mutation),
        "eval_python": str(args.eval_python),
        "extra_eval_args": tuple(passthrough),
    }
    if args.sumo_extra_args is not None:
        overrides["sumo_extra_args"] = str(args.sumo_extra_args)
    if args.episode_steps is not None:
        overrides["episode_steps"] = int(args.episode_steps)
    if args.sim_buffer_after_last_depart is not None:
        overrides["sim_buffer_after_last_depart"] = int(args.sim_buffer_after_last_depart)
    if args.max_steps_per_episode is not None:
        overrides["max_steps_per_episode"] = int(args.max_steps_per_episode)
    if args.output_dir:
        overrides["output_dir"] = str(args.output_dir)
    if args.tag:
        overrides["tag"] = str(args.tag)
    for arg_name, field_name in (
        ("network_name", "network_name"),
        ("network_dir", "network_dir"),
        ("edge_registry", "edge_registry"),
        ("route_artifacts_dir", "route_artifacts_dir"),
        ("post_mutation_state_in", "post_mutation_state_in"),
        ("hdv_state_in", "hdv_state_in"),
        ("projector_ckpt", "projector_ckpt"),
        ("dynamics_ckpt", "dynamics_ckpt"),
        ("av_candidate_source", "av_candidate_source"),
        ("route_generator", "route_generator"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = str(value)
    if args.ratio_machines is not None:
        overrides["ratio_machines"] = float(args.ratio_machines)
    if args.settings:
        return OnlineWmDreamerConfig.from_settings(str(args.settings), **overrides)
    return OnlineWmDreamerConfig(**overrides).resolve()


def run_evaluations(cfg: OnlineWmDreamerConfig, train_summary: Dict[str, object]) -> Dict[str, object]:
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
        print("[online-wm-dreamer] eval command:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        outputs[str(int(episode))] = {
            "checkpoint": str(ckpt),
            "results_dir": str(results_dir),
            "summary_json": str(results_dir / "collector_eval_summary.json"),
        }
    out_path = Path(cfg.output_dir) / "online_wm_eval_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    return outputs


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    plan = build_gretz_online_wm_dreamer_plan(cfg)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        if not args.allow_missing:
            validate_required_paths(plan)
        return 0
    validate_required_paths(plan)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "online_wm_dreamer_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    with (output_dir / "online_wm_dreamer_plan.json").open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    trainer = OnlineWmDreamerTrainer(cfg)
    train_summary = trainer.train()
    if not args.skip_eval:
        train_summary["eval"] = run_evaluations(cfg, train_summary)
    with (output_dir / "online_wm_dreamer_final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
