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
        choices=["progress_fraction", "capped_step_cost"],
        help=(
            "Vehicle reward for real/imagined actor updates. progress_fraction uses route progress; "
            "capped_step_cost uses -min(step_interval, remaining_route_tt)/temperature."
        ),
    )
    parser.add_argument("--reward-temperature", type=float, default=60.0)
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount for real/imagined vehicle returns; omitted uses settings/default config.",
    )
    parser.add_argument("--wm-replay-buffer-episodes", type=int, default=8)
    parser.add_argument("--wm-update-epochs", type=int, default=1)
    parser.add_argument("--wm-update-every-episodes", type=int, default=1)
    parser.add_argument(
        "--wm-updates-per-collect",
        type=int,
        default=1,
        help="Number of replay-sampled WM optimizer updates after each real SUMO collection episode.",
    )
    parser.add_argument("--wm-replay-completion-threshold", type=float, default=1.0)
    parser.add_argument(
        "--allow-incomplete-wm-replay",
        action="store_true",
        help="Allow incomplete SUMO episodes into WM replay. Default is to require completion.",
    )
    parser.add_argument("--wm-batch-size", type=int, default=16)
    parser.add_argument(
        "--wm-gradient-checkpointing",
        action="store_true",
        help="Recompute online-WM transformer layers during backward to reduce GPU memory.",
    )
    parser.add_argument(
        "--wm-space-attention-batch-chunk-size",
        type=int,
        default=0,
        help=(
            "Maximum flattened batch/time items per online-WM spatial-attention call; "
            "0 disables chunking."
        ),
    )
    parser.add_argument(
        "--wm-mlp-token-chunk-size",
        type=int,
        default=0,
        help="Maximum flattened tokens per online-WM transformer MLP call.",
    )
    parser.add_argument(
        "--wm-time-attention-batch-chunk-size",
        type=int,
        default=0,
        help="Maximum sequences per online-WM temporal-attention call.",
    )
    parser.add_argument("--wm-lr", type=float, default=1e-5)
    parser.add_argument("--wm-grad-clip", type=float, default=10.0)
    parser.add_argument("--wm-transition-decoded-supervision-weight", type=float, default=1.0)
    parser.add_argument(
        "--wm-t-all-supervision-weight",
        type=float,
        default=0.2,
        help="Online supervision weight for the completed-vehicle episode t_all head.",
    )
    parser.add_argument("--wm-hdv-policy-weight", type=float, default=0.05)
    parser.add_argument(
        "--wm-hdv-target-source",
        default="episode_mean_current_occupancy",
        choices=["episode_mean_current_occupancy", "static_edge_demand"],
        help=(
            "Edge-level HDV target/context source for online WM replay. "
            "episode_mean_current_occupancy preserves the legacy current-occupancy mean; "
            "static_edge_demand projects each HDV episode route decision onto edge demand."
        ),
    )
    parser.add_argument("--wm-hdv-policy-target-mode", default="episode_mean")
    parser.add_argument("--wm-hdv-policy-action-mode", default="window_mean")
    parser.add_argument("--wm-transition-hdv-context-mode", default="episode_response")
    parser.add_argument("--wm-transition-hdv-context-weight", type=float, default=1.0)
    parser.add_argument("--wm-transition-hdv-context-scale", type=float, default=1.0)
    parser.add_argument("--wm-transition-hdv-action-mode", default="none")
    parser.add_argument("--wm-transition-hdv-action-weight", type=float, default=0.0)
    parser.add_argument("--wm-transition-hdv-action-scale", type=float, default=1.0)
    parser.add_argument(
        "--imagined-hdv-context-source",
        default="oracle_current_occupancy",
        choices=[
            "oracle_current_occupancy",
            "oracle_current",
            "oracle_target",
            "oracle_static",
            "causal_response_predictor",
            "none",
        ],
        help=(
            "HDV context used inside imagined actor rollouts. The causal_response_predictor "
            "mode uses a pre-episode prediction from the previous episode AV/HDV summary."
        ),
    )
    hdv_response_group = parser.add_mutually_exclusive_group()
    hdv_response_group.add_argument(
        "--enable-hdv-response-predictor",
        dest="hdv_response_predictor_enabled",
        action="store_true",
        default=False,
        help=(
            "Use a causal episode-level HDV response predictor. OD-route v3 "
            "checkpoints can be fine-tuned with the separate online-update flag."
        ),
    )
    hdv_response_group.add_argument(
        "--disable-hdv-response-predictor",
        dest="hdv_response_predictor_enabled",
        action="store_false",
        help="Disable the causal static-HDV-demand response predictor.",
    )
    parser.add_argument("--hdv-response-predictor-lr", type=float, default=2e-3)
    parser.add_argument("--hdv-response-predictor-weight-decay", type=float, default=1e-4)
    parser.add_argument("--hdv-response-predictor-hidden-dim", type=int, default=128)
    parser.add_argument("--hdv-response-predictor-depth", type=int, default=2)
    parser.add_argument("--hdv-response-predictor-update-epochs", type=int, default=64)
    parser.add_argument("--hdv-response-predictor-min-pairs", type=int, default=2)
    hdv_online_group = parser.add_mutually_exclusive_group()
    hdv_online_group.add_argument(
        "--enable-hdv-response-online-update",
        dest="hdv_response_predictor_online_update_enabled",
        action="store_true",
        default=False,
        help=(
            "Fine-tune a loaded OD-route v3 predictor after each target episode. "
            "The update affects only subsequent episodes."
        ),
    )
    hdv_online_group.add_argument(
        "--disable-hdv-response-online-update",
        dest="hdv_response_predictor_online_update_enabled",
        action="store_false",
    )
    parser.add_argument(
        "--hdv-response-predictor-online-lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--hdv-response-predictor-online-update-epochs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--hdv-response-predictor-online-buffer-episodes",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--hdv-response-predictor-ckpt",
        default="",
        help=(
            "Optional pretrained causal HDV response checkpoint. The online-WM "
            "mainline accepts validation-selected OD-route v3 checkpoints here."
        ),
    )
    parser.add_argument(
        "--hdv-response-predictor-progress-denominator",
        type=int,
        default=0,
        help="Episode-progress denominator for the response predictor. 0 uses train_episodes or checkpoint metadata.",
    )
    parser.add_argument(
        "--hdv-response-predictor-parameterization",
        choices=["residual_logit", "absolute_logits"],
        default="absolute_logits",
        help=(
            "Legacy edge predictor parameterization. The repaired HDV response "
            "path is evaluated separately in OD-route space."
        ),
    )
    parser.add_argument(
        "--hdv-response-predictor-residual-scale",
        type=float,
        default=1.0,
        help="Multiplier on the learned residual logits; offline validation may select 0 for persistence.",
    )
    parser.add_argument(
        "--hdv-response-predictor-fallback",
        default="prev_hdv",
        choices=["prev_hdv", "previous_hdv", "prev_static", "zero", "none", "oracle_target", "oracle_current"],
    )
    parser.add_argument(
        "--wm-cfm-init-mode",
        choices=["randn", "zero", "current"],
        default=None,
        help="CFM rollout source for the world model. current uses z_t as x0 and removes z_t as a condition token.",
    )
    parser.add_argument("--imag-horizon", type=int, default=8)
    parser.add_argument("--imagined-updates-per-episode", type=int, default=1)
    parser.add_argument(
        "--imagined-update-passes-per-collect",
        type=int,
        default=1,
        help="Number of imagined actor-critic update calls after each real SUMO collection episode.",
    )
    parser.add_argument("--imagined-rollouts-per-update", type=int, default=1)
    parser.add_argument(
        "--imagined-rollout-batch-size",
        type=int,
        default=1,
        help="Number of independent imagined rollouts advanced together at each causal horizon step.",
    )
    parser.add_argument("--min-replay-episodes-before-actor", type=int, default=1)
    parser.add_argument("--ac-lambda", type=float, default=0.95)
    parser.add_argument(
        "--global-advantage-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the state-only global-value lambda advantage in every vehicle actor sample. "
            "Default 0 keeps the term disabled until the t_all readout passes held-out calibration."
        ),
    )
    parser.add_argument(
        "--global-reward-scale",
        type=float,
        default=1.0,
        help="Scale applied to the objective-aligned t_all potential difference.",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Vehicle actor learning rate.")
    parser.add_argument("--critic-lr", type=float, default=3e-4, help="Vehicle critic learning rate.")
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--advantage-clip", type=float, default=2.5)
    parser.add_argument("--ppo-clip-ratio", type=float, default=0.075)
    parser.add_argument("--entropy-target-frac", type=float, default=0.65)
    parser.add_argument("--entropy-floor-coef", type=float, default=0.02)
    parser.add_argument("--replay-buffer-episodes", type=int, default=1)
    parser.add_argument("--replay-update-epochs", type=int, default=1)
    parser.add_argument("--replay-actor-max-age-episodes", type=int, default=8)
    parser.add_argument("--return-horizon-steps", type=int, default=0)
    parser.add_argument("--min-samples-per-update", type=int, default=1)
    parser.add_argument("--route-cost-logit-bias", type=float, default=0.0)
    parser.add_argument("--policy-sample-temperature", type=float, default=1.0)
    parser.add_argument(
        "--real-policy-route-selection",
        choices=["argmax", "sample"],
        default="sample",
        help=(
            "Action selection used by real SUMO collection. sample preserves policy exploration; "
            "the sampled action is reused for simulator execution, replay, and the WM one-hot condition."
        ),
    )
    parser.add_argument(
        "--eval-policy-route-selection",
        choices=["argmax", "sample"],
        default="sample",
        help=(
            "Route selection used for checkpoint evaluation. The mainline samples from the "
            "masked actor distribution; argmax is retained as an explicit ablation."
        ),
    )
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument(
        "--no-normalize-advantages",
        dest="ac_normalize_advantages",
        action="store_false",
        help="Disable imagined actor-critic advantage normalization.",
    )
    parser.set_defaults(ac_normalize_advantages=True)
    parser.add_argument("--wm-prediction-diagnostic-every-episodes", type=int, default=1)
    parser.add_argument("--wm-prediction-diagnostic-max-steps", type=int, default=16)
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument(
        "--enable-wm-hdv-pressure-trace",
        dest="wm_hdv_pressure_trace_enabled",
        action="store_true",
        default=True,
        help="Persist per-decision true HDV edge occupancy traces for HDV-pressure diagnostics.",
    )
    trace_group.add_argument(
        "--disable-wm-hdv-pressure-trace",
        dest="wm_hdv_pressure_trace_enabled",
        action="store_false",
        help="Do not persist per-decision HDV edge occupancy trace npz files.",
    )
    parser.add_argument("--vehicle-arrival-loss-coef", type=float, default=1.0)
    parser.add_argument("--vehicle-arrival-diag-horizon-seconds", type=float, default=14.0)
    arrival_head_group = parser.add_mutually_exclusive_group()
    arrival_head_group.add_argument(
        "--enable-vehicle-arrival-head",
        dest="vehicle_arrival_head_enabled",
        action="store_true",
        default=False,
        help="Enable the vehicle-level route-conditioned arrival auxiliary head.",
    )
    arrival_head_group.add_argument(
        "--disable-vehicle-arrival-head",
        dest="vehicle_arrival_head_enabled",
        action="store_false",
        help="Keep the vehicle-level route-conditioned arrival auxiliary head disabled.",
    )
    parser.add_argument(
        "--direct-real-a2c-update",
        action="store_true",
        help="Also run one real-SUMO A2C actor/critic update after each collection episode.",
    )
    parser.add_argument(
        "--freeze-policy-obs-world-model",
        action="store_true",
        help="Use a frozen checkpoint copy for real-SUMO actor observations while online WM updates train a separate model.",
    )
    parser.add_argument(
        "--policy-obs-mode",
        default="wm_latent",
        choices=["wm_latent", "toy_live"],
        help="Observation path for real-SUMO actor decisions; toy_live matches the toy no-WM AC collector.",
    )
    parser.add_argument(
        "--actor-update-schedule",
        default="fixed",
        choices=["fixed", "linear", "stair", "diagnostic"],
        help="Schedule for imagined actor-critic update passes per collected real episode.",
    )
    parser.add_argument("--actor-update-min-passes", type=int, default=0)
    parser.add_argument("--actor-update-max-passes", type=int, default=0)
    parser.add_argument("--actor-update-warmup-episodes", type=int, default=0)
    parser.add_argument("--actor-update-ramp-episodes", type=int, default=0)
    parser.add_argument("--actor-update-gate-enabled", action="store_true")
    parser.add_argument("--actor-update-gate-min-diag-samples", type=int, default=1)
    parser.add_argument("--actor-update-gate-min-top1-match", type=float, default=0.0)
    parser.add_argument("--actor-update-gate-max-route-mae", type=float, default=0.0)
    parser.add_argument("--actor-update-gate-max-current-edge-mae", type=float, default=0.0)
    parser.add_argument("--actor-update-gate-max-regret", type=float, default=0.0)
    parser.add_argument("--actor-update-gate-fail-passes", type=int, default=0)
    parser.add_argument("--actor-update-gate-missing-passes", type=int, default=0)
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
        "wm_updates_per_collect": int(args.wm_updates_per_collect),
        "wm_replay_require_completion": not bool(args.allow_incomplete_wm_replay),
        "wm_replay_completion_threshold": float(args.wm_replay_completion_threshold),
        "wm_batch_size": int(args.wm_batch_size),
        "wm_gradient_checkpointing": bool(args.wm_gradient_checkpointing),
        "wm_space_attention_batch_chunk_size": int(
            args.wm_space_attention_batch_chunk_size
        ),
        "wm_mlp_token_chunk_size": int(args.wm_mlp_token_chunk_size),
        "wm_time_attention_batch_chunk_size": int(
            args.wm_time_attention_batch_chunk_size
        ),
        "wm_lr": float(args.wm_lr),
        "wm_grad_clip": float(args.wm_grad_clip),
        "wm_transition_decoded_supervision_weight": float(args.wm_transition_decoded_supervision_weight),
        "wm_t_all_supervision_weight": float(args.wm_t_all_supervision_weight),
        "wm_hdv_policy_weight": float(args.wm_hdv_policy_weight),
        "wm_hdv_target_source": str(args.wm_hdv_target_source),
        "wm_hdv_policy_target_mode": str(args.wm_hdv_policy_target_mode),
        "wm_hdv_policy_action_mode": str(args.wm_hdv_policy_action_mode),
        "wm_transition_hdv_context_mode": str(args.wm_transition_hdv_context_mode),
        "wm_transition_hdv_context_weight": float(args.wm_transition_hdv_context_weight),
        "wm_transition_hdv_context_scale": float(args.wm_transition_hdv_context_scale),
        "wm_transition_hdv_action_mode": str(args.wm_transition_hdv_action_mode),
        "wm_transition_hdv_action_weight": float(args.wm_transition_hdv_action_weight),
        "wm_transition_hdv_action_scale": float(args.wm_transition_hdv_action_scale),
        "imagined_hdv_context_source": str(args.imagined_hdv_context_source),
        "hdv_response_predictor_enabled": bool(args.hdv_response_predictor_enabled),
        "hdv_response_predictor_lr": float(args.hdv_response_predictor_lr),
        "hdv_response_predictor_weight_decay": float(args.hdv_response_predictor_weight_decay),
        "hdv_response_predictor_hidden_dim": int(args.hdv_response_predictor_hidden_dim),
        "hdv_response_predictor_depth": int(args.hdv_response_predictor_depth),
        "hdv_response_predictor_update_epochs": int(args.hdv_response_predictor_update_epochs),
        "hdv_response_predictor_min_pairs": int(args.hdv_response_predictor_min_pairs),
        "hdv_response_predictor_online_update_enabled": bool(
            args.hdv_response_predictor_online_update_enabled
        ),
        "hdv_response_predictor_online_lr": float(
            args.hdv_response_predictor_online_lr
        ),
        "hdv_response_predictor_online_update_epochs": int(
            args.hdv_response_predictor_online_update_epochs
        ),
        "hdv_response_predictor_online_buffer_episodes": int(
            args.hdv_response_predictor_online_buffer_episodes
        ),
        "hdv_response_predictor_ckpt": str(args.hdv_response_predictor_ckpt),
        "hdv_response_predictor_progress_denominator": int(args.hdv_response_predictor_progress_denominator),
        "hdv_response_predictor_parameterization": str(
            args.hdv_response_predictor_parameterization
        ),
        "hdv_response_predictor_residual_scale": float(
            args.hdv_response_predictor_residual_scale
        ),
        "hdv_response_predictor_fallback": str(args.hdv_response_predictor_fallback),
        "imag_horizon": int(args.imag_horizon),
        "imagined_updates_per_episode": int(args.imagined_updates_per_episode),
        "imagined_update_passes_per_collect": int(args.imagined_update_passes_per_collect),
        "imagined_rollouts_per_update": int(args.imagined_rollouts_per_update),
        "imagined_rollout_batch_size": int(args.imagined_rollout_batch_size),
        "min_replay_episodes_before_actor": int(args.min_replay_episodes_before_actor),
        "ac_lambda": float(args.ac_lambda),
        "ac_global_advantage_weight": float(args.global_advantage_weight),
        "global_reward_scale": float(args.global_reward_scale),
        "lr": float(args.lr),
        "critic_lr": float(args.critic_lr),
        "value_coef": float(args.value_coef),
        "entropy_coef": float(args.entropy_coef),
        "ac_advantage_clip": float(args.advantage_clip),
        "ac_ppo_clip_ratio": float(args.ppo_clip_ratio),
        "ac_normalize_advantages": bool(args.ac_normalize_advantages),
        "ac_entropy_target_frac": float(args.entropy_target_frac),
        "ac_entropy_floor_coef": float(args.entropy_floor_coef),
        "replay_buffer_episodes": int(args.replay_buffer_episodes),
        "replay_update_epochs": int(args.replay_update_epochs),
        "replay_actor_max_age_episodes": int(args.replay_actor_max_age_episodes),
        "return_horizon_steps": int(args.return_horizon_steps),
        "min_samples_per_update": int(args.min_samples_per_update),
        "route_cost_logit_bias": float(args.route_cost_logit_bias),
        "av_policy_sample_temperature": float(args.policy_sample_temperature),
        "real_policy_route_selection": str(args.real_policy_route_selection),
        "eval_policy_route_selection": str(args.eval_policy_route_selection),
        "max_grad_norm": float(args.max_grad_norm),
        "wm_prediction_diagnostic_every_episodes": int(args.wm_prediction_diagnostic_every_episodes),
        "wm_prediction_diagnostic_max_steps": int(args.wm_prediction_diagnostic_max_steps),
        "wm_hdv_pressure_trace_enabled": bool(args.wm_hdv_pressure_trace_enabled),
        "vehicle_arrival_head_enabled": bool(args.vehicle_arrival_head_enabled),
        "vehicle_arrival_loss_coef": float(args.vehicle_arrival_loss_coef),
        "vehicle_arrival_diag_horizon_seconds": float(args.vehicle_arrival_diag_horizon_seconds),
        "direct_real_a2c_update": bool(args.direct_real_a2c_update),
        "freeze_policy_obs_world_model": bool(args.freeze_policy_obs_world_model),
        "policy_obs_mode": str(args.policy_obs_mode),
        "actor_update_schedule": str(args.actor_update_schedule),
        "actor_update_min_passes": int(args.actor_update_min_passes),
        "actor_update_max_passes": int(args.actor_update_max_passes),
        "actor_update_warmup_episodes": int(args.actor_update_warmup_episodes),
        "actor_update_ramp_episodes": int(args.actor_update_ramp_episodes),
        "actor_update_gate_enabled": bool(args.actor_update_gate_enabled),
        "actor_update_gate_min_diag_samples": int(args.actor_update_gate_min_diag_samples),
        "actor_update_gate_min_top1_match": float(args.actor_update_gate_min_top1_match),
        "actor_update_gate_max_route_mae": float(args.actor_update_gate_max_route_mae),
        "actor_update_gate_max_current_edge_mae": float(args.actor_update_gate_max_current_edge_mae),
        "actor_update_gate_max_regret": float(args.actor_update_gate_max_regret),
        "actor_update_gate_fail_passes": int(args.actor_update_gate_fail_passes),
        "actor_update_gate_missing_passes": int(args.actor_update_gate_missing_passes),
        "disable_human_learning_after_mutation": bool(args.disable_human_learning_after_mutation),
        "eval_python": str(args.eval_python),
        "extra_eval_args": tuple(passthrough),
    }
    if args.gamma is not None:
        overrides["gamma"] = float(args.gamma)
    if args.wm_cfm_init_mode is not None:
        overrides["wm_cfm_init_mode"] = str(args.wm_cfm_init_mode)
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
    final_hdv_state = Path(
        str(
            train_summary.get("final_hdv_state")
            or (Path(cfg.output_dir) / "hdv_learning_state_after_online_wm_train.json")
        )
    )
    if not final_hdv_state.exists():
        raise FileNotFoundError(f"Final online-training HDV state not found: {final_hdv_state}")
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
        hdv_state_index = cmd.index("--hdv_state_in") + 1
        cmd[hdv_state_index] = str(final_hdv_state)
        print("[online-wm-dreamer] eval command:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        outputs[str(int(episode))] = {
            "checkpoint": str(ckpt),
            "hdv_state_in": str(final_hdv_state),
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
