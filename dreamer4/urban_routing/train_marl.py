"""
train_marl.py — Dreamer CTDE vehicle-level training for Urban Routing.

Episode structure (100% offline, flow-only):
  1. Sample initial state from pre-collected flow dataset:
       z_0      = wm.encode_observation(flow_0)   (first flow snapshot of an episode)
       hdv_frac = uniform(K) or dataset-provided HDV fraction (if available)
  2. Build vehicle-level OD suffix candidates and graph-conditioned edge actions.
  3. Run Dreamer-style imagined rollouts with a vehicle route actor and centralized critic.
  4. Update actor/critic from imagined trajectories.

No SUMO, no traci, no connector needed at training time.
The world model IS the environment.

Data required (produced by data_collector.py):
  <data_dir>/flow_round1.pt            — episode, av_planned_edge_flow/av_edge_action, hdv_edge_action, reward
  <data_dir>/flows/<task>/shard_*.pt   — flow snapshots (shard_size, E, 8) float
"""

import os
import sys
import glob
import json
import logging
import argparse
import random
import csv
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from .urban_world_model import UrbanWorldModel
    from .rollout_buffer import RolloutBuffer
    from .dreamer_actor_critic import (
        CentralizedLatentValueCritic,
        CentralizedVehicleValueCritic,
        VehicleRouteActor,
        freeze_module,
        lambda_return,
    )
    from .dreamer_rollout import (
        aggregate_route_probs_to_action_edge,
        build_history_packed,
    )
    from . import dreamer_hdv_response as hdv_response
    from . import dreamer_policy_gradient as dreamer_pg
    from . import dreamer_reward_readouts as rollout_rewards
    from .dreamer_checkpoint import (
        _restore_dreamer_extra_state,
        _move_nested_tensors,
        capture_rng_state,
        count_fresh_pending_trajs,
        load_dreamer_checkpoint,
        maybe_resume_dreamer_checkpoint as _checkpoint_maybe_resume_dreamer_checkpoint,
        normalize_stats_history_defaults,
        restore_rng_state,
        save_dreamer_checkpoint,
    )
    from .dreamer_route_objectives import (
        _combined_grpo_policy_loss,
        _counterfactual_fleet_pressure_cost_matrix,
        _dreamer_action_route_probs,
        _expected_cost_policy_loss,
        _expected_route_cost_from_cost_matrix,
        _expected_route_cost_from_paths,
        _expected_route_regret_from_cost_matrix,
        _expected_route_regret_from_paths,
        _free_flow_edge_tt_tensor,
        _grpo_route_cost_policy_loss,
        _marginal_av_pressure_cost_matrix,
        _marginal_pressure_reference_probs,
        _masked_route_log_probs,
        _mean_path_invalid_rate,
        _planned_av_pressure_cost_matrix,
        _planned_av_pressure_edge_weights,
        _rollout_baseline_relative_advantages,
        _rollout_group_relative_advantages,
        _route_credit_base_cost_matrix,
        _route_cost_matrix_from_paths,
        _route_cost_policy_aux_loss,
        _route_prob_churn,
        _route_share_penalty_cost_matrix,
        _sample_masked_route_actions,
        _selected_route_cost_advantages,
        _sequential_marginal_av_pressure_cost_matrix,
        _sequential_planned_pressure_greedy_action_probs,
        _standardize_with_detached_stats,
        _valid_route_mask_from_paths,
        _vehicle_potential_progress_terms_from_paths,
    )
    from .experiment_settings import add_settings_arg, apply_stage_defaults, parse_settings_arg
    from .flow_projector import PathTopology
    from .departure_schedule import (
        build_departure_schedule, rebuild_active_set,
        compute_avg_edge_time, compute_avg_route_steps,
    )
    from .routerl_shared import (
        load_od_data,
        load_agents_dataframe,
        generate_dijkstra_routes,
        imagined_gawron_mode,
    )
    from .vehicle_imagination import (
        LazyVehiclePathCache,
        VehiclePathCache,
        _ensure_active_vehicle_current_edges,
        _od_suffix_path_batch,
        aggregate_vehicle_actions_to_edge_distribution,
        build_diverse_paths_current_to_dest,
        build_lightweight_vehicle_path_topology,
        build_lightweight_vehicle_route_registry,
        build_vehicle_fleet_pressure_obs_features,
        build_vehicle_obs_from_path_batch,
        build_vehicle_obs_vectorized,
        build_vehicle_route_cost_obs_features,
        advance_vehicle_along_path_time_budget,
        vehicle_route_feature_dim,
    )
except ImportError:
    from urban_world_model import UrbanWorldModel
    from rollout_buffer import RolloutBuffer
    from dreamer_actor_critic import (
        CentralizedLatentValueCritic,
        CentralizedVehicleValueCritic,
        VehicleRouteActor,
        freeze_module,
        lambda_return,
    )
    from dreamer_rollout import (
        aggregate_route_probs_to_action_edge,
        build_history_packed,
    )
    import dreamer_hdv_response as hdv_response
    import dreamer_policy_gradient as dreamer_pg
    import dreamer_reward_readouts as rollout_rewards
    from dreamer_checkpoint import (
        _restore_dreamer_extra_state,
        _move_nested_tensors,
        capture_rng_state,
        count_fresh_pending_trajs,
        load_dreamer_checkpoint,
        maybe_resume_dreamer_checkpoint as _checkpoint_maybe_resume_dreamer_checkpoint,
        normalize_stats_history_defaults,
        restore_rng_state,
        save_dreamer_checkpoint,
    )
    from dreamer_route_objectives import (
        _combined_grpo_policy_loss,
        _counterfactual_fleet_pressure_cost_matrix,
        _dreamer_action_route_probs,
        _expected_cost_policy_loss,
        _expected_route_cost_from_cost_matrix,
        _expected_route_cost_from_paths,
        _expected_route_regret_from_cost_matrix,
        _expected_route_regret_from_paths,
        _free_flow_edge_tt_tensor,
        _grpo_route_cost_policy_loss,
        _marginal_av_pressure_cost_matrix,
        _marginal_pressure_reference_probs,
        _masked_route_log_probs,
        _mean_path_invalid_rate,
        _planned_av_pressure_cost_matrix,
        _planned_av_pressure_edge_weights,
        _rollout_baseline_relative_advantages,
        _rollout_group_relative_advantages,
        _route_credit_base_cost_matrix,
        _route_cost_matrix_from_paths,
        _route_cost_policy_aux_loss,
        _route_prob_churn,
        _route_share_penalty_cost_matrix,
        _sample_masked_route_actions,
        _selected_route_cost_advantages,
        _sequential_marginal_av_pressure_cost_matrix,
        _sequential_planned_pressure_greedy_action_probs,
        _standardize_with_detached_stats,
        _valid_route_mask_from_paths,
        _vehicle_potential_progress_terms_from_paths,
    )
    from experiment_settings import add_settings_arg, apply_stage_defaults, parse_settings_arg
    from flow_projector import PathTopology
    from departure_schedule import (
        build_departure_schedule, rebuild_active_set,
        compute_avg_edge_time, compute_avg_route_steps,
    )
    from routerl_shared import (
        load_od_data,
        load_agents_dataframe,
        generate_dijkstra_routes,
        imagined_gawron_mode,
    )
    from vehicle_imagination import (
        LazyVehiclePathCache,
        VehiclePathCache,
        _ensure_active_vehicle_current_edges,
        _od_suffix_path_batch,
        aggregate_vehicle_actions_to_edge_distribution,
        build_diverse_paths_current_to_dest,
        build_lightweight_vehicle_path_topology,
        build_lightweight_vehicle_route_registry,
        build_vehicle_fleet_pressure_obs_features,
        build_vehicle_obs_from_path_batch,
        build_vehicle_obs_vectorized,
        build_vehicle_route_cost_obs_features,
        advance_vehicle_along_path_time_budget,
        vehicle_route_feature_dim,
    )

logger = logging.getLogger(__name__)

ROUTE_COST_RETURN_DISCOUNT = 1.0
LEGACY_ACTOR_ONLY_LOSS_WEIGHTS = (
    "dreamer_ac_route_advantage_weight",
    "dreamer_expected_cost_weight",
    "dreamer_grpo_route_cost_weight",
    "dreamer_grpo_rollout_weight",
    "dreamer_grpo_rollout_vehicle_cost_weight",
    "dreamer_route_cost_aux_weight",
)


class DreamerReturnNormalizer:
    """EMA return normalizer used to stabilize critic targets and advantages."""

    def __init__(
        self,
        *,
        decay: float = 0.99,
        eps: float = 1e-6,
        enabled: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        self.decay = float(decay)
        self.eps = float(eps)
        self.enabled = bool(enabled)
        target_device = device if device is not None else torch.device("cpu")
        self.mean = torch.zeros((), device=target_device, dtype=torch.float32)
        self.var = torch.ones((), device=target_device, dtype=torch.float32)
        self.count = torch.zeros((), device=target_device, dtype=torch.float32)

    def to(self, device: torch.device) -> "DreamerReturnNormalizer":
        self.mean = self.mean.to(device=device)
        self.var = self.var.to(device=device)
        self.count = self.count.to(device=device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": float(self.decay),
            "eps": float(self.eps),
            "enabled": bool(self.enabled),
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "count": self.count.detach().cpu(),
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        device = self.mean.device
        self.decay = float(state.get("decay", self.decay))
        self.eps = float(state.get("eps", self.eps))
        self.enabled = bool(state.get("enabled", self.enabled))
        self.mean = torch.as_tensor(state.get("mean", self.mean), device=device, dtype=torch.float32).reshape(())
        self.var = torch.as_tensor(state.get("var", self.var), device=device, dtype=torch.float32).reshape(()).clamp_min(self.eps)
        self.count = torch.as_tensor(state.get("count", self.count), device=device, dtype=torch.float32).reshape(())

    def update(self, values: torch.Tensor) -> None:
        if not self.enabled:
            return
        finite = values.detach().reshape(-1)
        finite = finite[torch.isfinite(finite)]
        if int(finite.numel()) <= 0:
            return
        batch_mean = finite.mean().to(device=self.mean.device, dtype=self.mean.dtype)
        batch_var = finite.var(unbiased=False).to(device=self.var.device, dtype=self.var.dtype).clamp_min(self.eps)
        if float(self.count.detach().cpu().item()) <= 0.0:
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var)
        else:
            decay = min(max(float(self.decay), 0.0), 0.999999)
            self.mean.mul_(decay).add_(batch_mean, alpha=1.0 - decay)
            self.var.mul_(decay).add_(batch_var, alpha=1.0 - decay)
        self.count.add_(float(int(finite.numel())))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return values
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.var.to(device=values.device, dtype=values.dtype).clamp_min(self.eps).sqrt()
        return (values - mean) / std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return values
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.var.to(device=values.device, dtype=values.dtype).clamp_min(self.eps).sqrt()
        return values * std + mean


def _polyak_update_module(target: nn.Module, source: nn.Module, tau: float) -> None:
    tau = min(max(float(tau), 0.0), 1.0)
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)
        for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
            if torch.is_floating_point(target_buffer):
                target_buffer.data.mul_(1.0 - tau).add_(source_buffer.data, alpha=tau)
            else:
                target_buffer.copy_(source_buffer)


def _scheduled_actor_lr(cfg: Any, base_lr: float, episode: int) -> float:
    start = int(getattr(cfg, "dreamer_actor_lr_decay_start_episode", -1))
    final_scale = max(float(getattr(cfg, "dreamer_actor_lr_final_scale", 1.0)), 0.0)
    if start < 0 or final_scale == 1.0:
        return float(base_lr)
    end = int(getattr(cfg, "dreamer_actor_lr_decay_end_episode", start))
    if int(episode) < start:
        scale = 1.0
    elif end <= start:
        scale = final_scale
    else:
        progress = min(max((float(episode) - float(start)) / float(end - start), 0.0), 1.0)
        scale = 1.0 + (final_scale - 1.0) * progress
    return max(float(base_lr) * float(scale), 0.0)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _restore_optimizer_lr(optimizer: torch.optim.Optimizer, lrs: List[float]) -> None:
    for group, lr in zip(optimizer.param_groups, lrs):
        group["lr"] = float(lr)


def _actor_update_gate_enabled(cfg: Any) -> bool:
    return (
        int(getattr(cfg, "dreamer_actor_update_gate_start_episode", -1)) >= 0
        or float(getattr(cfg, "dreamer_actor_update_gate_max_route_frac", 0.0)) > 0.0
        or float(getattr(cfg, "dreamer_actor_update_gate_route0_frac", 0.0)) > 0.0
        or float(getattr(cfg, "dreamer_actor_update_gate_top1_prob", 0.0)) > 0.0
    )


def _actor_update_gate_reasons(
    cfg: Any,
    *,
    episode: int,
    max_route_frac: float,
    route0_frac: float,
    top1_prob: float,
    sample_count: float,
    gate_latched: bool = False,
) -> List[str]:
    if not _actor_update_gate_enabled(cfg):
        return []
    start = int(getattr(cfg, "dreamer_actor_update_gate_start_episode", -1))
    if start >= 0 and int(episode) < start:
        return []
    if bool(gate_latched):
        return ["latched"]
    min_samples = max(float(getattr(cfg, "dreamer_actor_update_gate_min_samples", 1.0)), 0.0)
    if float(sample_count) < min_samples:
        return []
    reasons: List[str] = []
    max_route_threshold = float(getattr(cfg, "dreamer_actor_update_gate_max_route_frac", 0.0))
    if max_route_threshold > 0.0 and float(max_route_frac) >= max_route_threshold:
        reasons.append("policy_argmax_max_route_frac")
    route0_threshold = float(getattr(cfg, "dreamer_actor_update_gate_route0_frac", 0.0))
    if route0_threshold > 0.0 and float(route0_frac) >= route0_threshold:
        reasons.append("policy_argmax_route0_frac")
    top1_threshold = float(getattr(cfg, "dreamer_actor_update_gate_top1_prob", 0.0))
    if top1_threshold > 0.0 and float(top1_prob) >= top1_threshold:
        reasons.append("policy_top1_prob")
    return reasons


def _actor_trust_region_accepts(
    *,
    kl: float,
    argmax_change_frac: float,
    new_argmax_max_route_frac: float,
    old_argmax_max_route_frac: float,
    new_argmax_route0_frac: float = 0.0,
    old_argmax_route0_frac: float = 0.0,
    kl_limit: float = 0.0,
    argmax_change_limit: float = 0.0,
    max_route_frac_limit: float = 0.0,
    route0_frac_limit: float = 0.0,
    vehicle_argmax_change_max: float = 0.0,
    vehicle_group_count: float = 0.0,
    vehicle_argmax_change_limit: float = 0.0,
    eps: float = 1e-6,
) -> bool:
    kl_ok = (
        float(kl_limit) <= 0.0
        or (np.isfinite(float(kl)) and float(kl) <= float(kl_limit) + float(eps))
    )
    argmax_change_ok = (
        float(argmax_change_limit) <= 0.0
        or (
            np.isfinite(float(argmax_change_frac))
            and float(argmax_change_frac) <= float(argmax_change_limit) + float(eps)
        )
    )
    vehicle_argmax_change_ok = (
        float(vehicle_argmax_change_limit) <= 0.0
        or (
            np.isfinite(float(vehicle_argmax_change_max))
            and np.isfinite(float(vehicle_group_count))
            and float(vehicle_group_count) > 0.0
            and float(vehicle_argmax_change_max) <= float(vehicle_argmax_change_limit) + float(eps)
        )
    )
    if float(max_route_frac_limit) <= 0.0:
        max_route_ok = True
    elif not np.isfinite(float(new_argmax_max_route_frac)):
        max_route_ok = False
    elif float(new_argmax_max_route_frac) <= float(max_route_frac_limit) + float(eps):
        max_route_ok = True
    elif (
        np.isfinite(float(old_argmax_max_route_frac))
        and float(old_argmax_max_route_frac) >= float(max_route_frac_limit) - float(eps)
        and float(new_argmax_max_route_frac) <= float(old_argmax_max_route_frac) + float(eps)
    ):
        max_route_ok = True
    else:
        max_route_ok = False
    if float(route0_frac_limit) <= 0.0:
        route0_ok = True
    elif not np.isfinite(float(new_argmax_route0_frac)):
        route0_ok = False
    elif float(new_argmax_route0_frac) <= float(route0_frac_limit) + float(eps):
        route0_ok = True
    elif (
        np.isfinite(float(old_argmax_route0_frac))
        and float(old_argmax_route0_frac) >= float(route0_frac_limit) - float(eps)
        and float(new_argmax_route0_frac) <= float(old_argmax_route0_frac) + float(eps)
    ):
        route0_ok = True
    else:
        route0_ok = False
    return bool(kl_ok and argmax_change_ok and vehicle_argmax_change_ok and max_route_ok and route0_ok)


def _actor_anchor_enabled(cfg: Any) -> bool:
    return (
        float(getattr(cfg, "dreamer_actor_anchor_kl", 0.0)) > 0.0
        or float(getattr(cfg, "dreamer_actor_anchor_max_argmax_change_frac", 0.0)) > 0.0
        or float(getattr(cfg, "dreamer_actor_anchor_max_vehicle_argmax_change_frac", 0.0)) > 0.0
        or float(getattr(cfg, "dreamer_actor_anchor_max_route_frac", 0.0)) > 0.0
        or float(getattr(cfg, "dreamer_actor_anchor_max_route0_frac", 0.0)) > 0.0
    )


def _dreamer_global_return_discount(cfg: Any) -> float:
    """Use unit discount only when the scalar reward is route-duration based.

    Route cost and route-progress terms are measured in arrival-time/travel-time
    units, so their physical total is an undiscounted sum. Other global reward
    terms are allowed to use cfg.gamma because their semantics may not be total
    trip duration.
    """
    non_route_weights = (
        "dreamer_all_reward_weight",
        "dreamer_current_edge_reward_weight",
        "dreamer_cav_reward_weight",
        "dreamer_action_congestion_weight",
        "dreamer_route_balance_weight",
        "dreamer_route_max_share_weight",
        "dreamer_horizon_proxy_reward_weight",
        "dreamer_rollout_cost_reward_weight",
        "terminal_tall_reward_weight",
    )
    has_non_route_reward = any(
        abs(float(getattr(cfg, name, 0.0))) > 1e-12
        for name in non_route_weights
    )
    route_scope = str(getattr(cfg, "dreamer_route_cost_reward_scope", "both") or "both").strip().lower()
    route_cost_reaches_global = route_scope in {"both", "global", "global_critic", "step"}
    has_route_duration_reward = (
        (
            route_cost_reaches_global
            and abs(float(getattr(cfg, "dreamer_route_cost_reward_weight", 0.0))) > 1e-12
        )
        or abs(float(getattr(cfg, "dreamer_pred_t_cav_proxy_reward_weight", 0.0))) > 1e-12
        or abs(float(getattr(cfg, "dreamer_remaining_time_progress_weight", 0.0))) > 1e-12
    )
    if has_route_duration_reward and not has_non_route_reward:
        return ROUTE_COST_RETURN_DISCOUNT
    return float(getattr(cfg, "gamma", 0.99))


def _vehicle_critic_reward_enabled(cfg: Any) -> bool:
    route_scope = str(getattr(cfg, "dreamer_route_cost_reward_scope", "both") or "both").strip().lower()
    route_cost_reaches_vehicle = route_scope in {"both", "vehicle", "vehicle_critic", "per_vehicle"}
    return (
        (
            route_cost_reaches_vehicle
            and abs(float(getattr(cfg, "dreamer_route_cost_reward_weight", 0.0))) > 1e-12
        )
        or abs(float(getattr(cfg, "dreamer_vehicle_route_credit_reward_weight", 0.0))) > 1e-12
        or abs(float(getattr(cfg, "dreamer_vehicle_social_reward_weight", 0.0))) > 1e-12
    )


def _dreamer_vehicle_return_discount(cfg: Any) -> float:
    """Discount for the per-vehicle critic target.

    Route progress/cost reward keeps unit discount because it represents total
    trip-duration progress.  Pure social reward ablations may use a conventional
    discounted return.  Mixed route+social runs intentionally keep the unit
    discount so the route-duration term retains its physical meaning.
    """
    route_scope = str(getattr(cfg, "dreamer_route_cost_reward_scope", "both") or "both").strip().lower()
    route_cost_reaches_vehicle = route_scope in {"both", "vehicle", "vehicle_critic", "per_vehicle"}
    if (
        route_cost_reaches_vehicle
        and abs(float(getattr(cfg, "dreamer_route_cost_reward_weight", 0.0))) > 1e-12
    ) or _vehicle_route_credit_has_unit_discount(cfg):
        return ROUTE_COST_RETURN_DISCOUNT
    if abs(float(getattr(cfg, "dreamer_vehicle_route_credit_reward_weight", 0.0))) > 1e-12:
        return float(
            getattr(
                cfg,
                "dreamer_vehicle_route_credit_reward_discount",
                getattr(cfg, "gamma", 0.99),
            )
        )
    if abs(float(getattr(cfg, "dreamer_vehicle_social_reward_weight", 0.0))) > 1e-12:
        return float(getattr(cfg, "dreamer_vehicle_social_reward_discount", getattr(cfg, "gamma", 0.99)))
    return ROUTE_COST_RETURN_DISCOUNT


def _vehicle_route_credit_has_unit_discount(cfg: Any) -> bool:
    if abs(float(getattr(cfg, "dreamer_vehicle_route_credit_reward_weight", 0.0))) <= 1e-12:
        return False
    base_mode = str(
        getattr(
            cfg,
            "dreamer_vehicle_route_credit_base_cost_mode",
            getattr(cfg, "dreamer_ac_route_credit_base_cost_mode", "none"),
        )
        or "none"
    ).strip().lower()
    return base_mode in {
        "progress_fraction",
        "elapsed_progress",
        "trip_progress_fraction",
        "remaining_tt_inverse",
        "neg_inverse_remaining_tt",
        "step_progress",
    }


def _global_critic_reward_enabled(cfg: Any) -> bool:
    """Whether imagined step rewards contain a critic-visible global signal."""
    route_scope = str(getattr(cfg, "dreamer_route_cost_reward_scope", "both") or "both").strip().lower()
    route_cost_reaches_global = route_scope in {"both", "global", "global_critic", "step"}
    weighted_terms = (
        "dreamer_all_reward_weight",
        "dreamer_remaining_time_progress_weight",
        "dreamer_current_edge_reward_weight",
        "dreamer_cav_reward_weight",
        "dreamer_pred_t_cav_proxy_reward_weight",
        "dreamer_action_congestion_weight",
        "dreamer_route_balance_weight",
        "dreamer_route_max_share_weight",
        "dreamer_horizon_proxy_reward_weight",
        "dreamer_rollout_cost_reward_weight",
        "terminal_tall_reward_weight",
    )
    if route_cost_reaches_global and abs(float(getattr(cfg, "dreamer_route_cost_reward_weight", 0.0))) > 1e-12:
        return True
    return any(abs(float(getattr(cfg, name, 0.0))) > 1e-12 for name in weighted_terms)


def _enabled_legacy_actor_only_losses(cfg: Any) -> List[str]:
    return [
        name for name in LEGACY_ACTOR_ONLY_LOSS_WEIGHTS
        if abs(float(getattr(cfg, name, 0.0))) > 1e-12
    ]


def _validate_legacy_actor_only_losses(cfg: Any) -> None:
    enabled = _enabled_legacy_actor_only_losses(cfg)
    if not enabled:
        return
    if bool(getattr(cfg, "dreamer_allow_legacy_actor_only_losses", False)):
        logger.warning(
            "Deprecated actor-only Dreamer losses enabled for legacy reproduction: %s. "
            "Current training should route reward signals through critics instead.",
            ", ".join(enabled),
        )
        return
    raise ValueError(
        "Actor-only Dreamer losses are deprecated and disabled on the current training mainline: "
        + ", ".join(enabled)
        + ". Set these weights to 0 and route rewards through a critic, or pass "
        "--dreamer_allow_legacy_actor_only_losses only when reproducing legacy checkpoints."
    )


def _dreamer_critic_context_vector(
    *,
    step: int,
    episode_steps: int,
    n_active_av: int,
    n_active_hdv: int,
    n_total_av: int,
    n_total_hdv: int,
    valid_path_agents: int,
    valid_path_count: int,
    elapsed_time_np: Optional[np.ndarray],
    depart_np: Optional[np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    active_av_denom = float(max(int(n_total_av), 1))
    active_hdv_denom = float(max(int(n_total_hdv), 1))
    valid_agent_denom = float(max(int(n_active_av), 1))
    valid_route_denom = float(max(int(n_active_av), 1))
    elapsed_mean = 0.0
    elapsed_max = 0.0
    if elapsed_time_np is not None and np.asarray(elapsed_time_np).size > 0:
        elapsed = np.asarray(elapsed_time_np, dtype=np.float32)
        scale = max(float(episode_steps), 1.0)
        elapsed_mean = float(np.mean(elapsed) / scale)
        elapsed_max = float(np.max(elapsed) / scale)
    depart_sum = 0.0
    if depart_np is not None:
        depart_sum = float(np.asarray(depart_np, dtype=np.float32).sum() / active_av_denom)
    values = [
        float(step) / float(max(int(episode_steps), 1)),
        float(n_active_av) / active_av_denom,
        float(n_active_hdv) / active_hdv_denom,
        float(valid_path_agents) / valid_agent_denom,
        float(valid_path_count) / valid_route_denom,
        elapsed_mean,
        elapsed_max,
        depart_sum,
    ]
    return torch.tensor(values, device=device, dtype=dtype)


def _derive_agent_counts_from_csv(agents_csv: str, ratio_machines: float) -> Tuple[int, int]:
    """Return full-schedule AV/HDV counts from agents.csv and ratio_machines."""
    if not agents_csv or not os.path.exists(agents_csv):
        return 0, 0
    try:
        with open(agents_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            n_total = sum(1 for row in reader if row)
    except OSError:
        return 0, 0
    if n_total <= 0:
        return 0, 0
    n_av = int(n_total * float(ratio_machines))
    return n_av, max(0, n_total - n_av)


class ActionHorizonProxyReward(nn.Module):
    """Frozen long-horizon t_all proxy conditioned on decoded state and edge action."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        target_mean: float,
        target_std: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.register_buffer("target_mean", torch.tensor(float(target_mean), dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(float(target_std), dtype=torch.float32))
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: torch.device) -> "ActionHorizonProxyReward":
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = cls(
            input_dim=int(obj["input_dim"]),
            hidden_dim=int(obj.get("hidden_dim", 256)),
            target_mean=float(obj.get("target_mean", 0.0)),
            target_std=float(obj.get("target_std", 1.0)),
        ).to(device)
        model.load_state_dict(obj["model"], strict=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model

    @staticmethod
    def build_features(
        decoded_dyn: torch.Tensor,
        static_features: torch.Tensor,
        action_edge: torch.Tensor,
    ) -> torch.Tensor:
        dyn = decoded_dyn.float()
        static = static_features.to(device=dyn.device, dtype=dyn.dtype)
        action = action_edge.to(device=dyn.device, dtype=dyn.dtype).reshape(-1)
        active = (action.abs() > 1e-8).to(dtype=dyn.dtype)
        action_mass = action.sum().clamp_min(1e-8)
        action_dist = action / action_mass
        flow_stats = torch.cat(
            [
                dyn.mean(dim=0),
                dyn.std(dim=0),
                dyn.amax(dim=0),
                static.mean(dim=0),
                static.std(dim=0),
            ],
            dim=0,
        )
        action_stats = torch.stack(
            [
                action.sum(),
                active.mean(),
                action.max(),
                (action_dist * dyn[:, 2].clamp_min(0.0)).sum(),
                (action_dist * dyn[:, 1].clamp_min(0.0)).sum(),
            ]
        )
        return torch.cat([flow_stats, action_stats, action], dim=0)

    def predict_minutes(
        self,
        decoded_dyn: torch.Tensor,
        static_features: torch.Tensor,
        action_edge: torch.Tensor,
    ) -> torch.Tensor:
        feat = self.build_features(decoded_dyn, static_features, action_edge)
        if int(feat.numel()) != self.input_dim:
            raise ValueError(
                f"horizon proxy input_dim mismatch: checkpoint expects {self.input_dim}, "
                f"built {int(feat.numel())}"
            )
        pred_norm = self.net(feat.unsqueeze(0)).squeeze()
        target_std = self.target_std.to(device=pred_norm.device, dtype=pred_norm.dtype)
        target_mean = self.target_mean.to(device=pred_norm.device, dtype=pred_norm.dtype)
        return pred_norm * target_std + target_mean


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    # ── offline dataset ──────────────────────────────────────────────────────
    data_dir: str = "data/urban_routing_flow/round1"
    # name of the .pt file inside data_dir that holds episode/action metadata
    meta_file: str = "flow_round1.pt"
    # glob pattern for flow shards relative to data_dir
    flow_glob: str = "flows/*/shard_*.pt"

    # ── world model ──────────────────────────────────────────────────────────
    dynamics_ckpt: str = ""
    projector_ckpt: str = ""
    edge_registry: str = ""
    history_length: int = 8
    wm_schedule: str = "shortcut"
    wm_schedule_d: float = 0.25
    wm_cfm_init_mode: str = "zero"
    wm_policy_obs_source: str = "decoder_volume_tt"  # decoder_volume_tt | decoder_volume_tt_route_head | decoder_volume | decoder | obs_proj
    wm_edge_cost_source: str = "decoded_speed"  # bpr_volume | decoded_speed | route_head
    wm_candidate_edge_cost_source: str = "auto"  # auto | bpr_volume | decoded_speed | route_head
    wm_decoded_speed_min_ratio: float = 0.05
    wm_decoded_speed_max_ratio: float = 1.50
    wm_decoded_speed_excess_scale: float = 1.0

    # ── MARL / action ────────────────────────────────────────────────────────
    n_routes: int = 4           # K — must match data collection number_of_paths
    av_fraction: float = 0.3
    n_av: int = 0               # deprecated compatibility argument; ignored by training
    n_hdv: int = 0              # deprecated compatibility argument; ignored by training

    # ── episode / training ───────────────────────────────────────────────────
    episode_len: int = 8        # ≤ history_length，确保 z_0_real 始终在 context 内
    n_total_episodes: int = 10000

    # ── PPO ──────────────────────────────────────────────────────────────────
    training_family: str = "dreamer_ctde"
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.001      # Gaussian 策略熵无界，不能用 Categorical 时的 0.05
    value_coef: float = 0.25       # 降低避免 vf 振荡
    n_ppo_epochs: int = 4
    minibatch_size: int = 64
    n_episodes_per_update: int = 32    # 单智能体每步只有 8 条样本，需要 32 episode 才有 256 条

    # ── policy network ───────────────────────────────────────────────────────
    hidden_dim: int = 256
    n_hidden_layers: int = 2

    # ── vehicle-level network topology ───────────────────────────────────────
    agents_csv: str = ""            # path to agents.csv (vehicle counts)
    ratio_machines: float = 0.2     # AV fraction for vehicle split
    network_folder: str = ""        # network folder for vehicle-level routing
    network_name: str = ""          # e.g. "ingolstadt_custom"
    include_intention_obs: bool = True  # append previous/global edge-level routing context
    agent_level: str = "vehicle"        # only vehicle-level training is supported
    use_stackelberg: bool = False       # use Stackelberg equilibrium layer in WM
    stackelberg_n_iters: int = 10
    stackelberg_damping: float = 0.5
    vehicle_path_penalty_factor: float = 3.0
    vehicle_candidate_source: str = "current_to_dest"  # current_to_dest | od_suffix
    vehicle_path_cache_mode: str = "lazy"  # lazy | precompute
    vehicle_path_cache_max_entries: int = 250000
    vehicle_routes_csv: str = ""       # optional route artifacts CSV for od_suffix vehicle candidates
    vehicle_obs_include_global: bool = True
    vehicle_obs_route_stats: str = "mean"
    vehicle_obs_include_route_costs: bool = False
    vehicle_obs_include_fleet_pressure: bool = False
    vehicle_actor_architecture: str = "mlp"
    vehicle_actor_route_cost_logit_bias: float = 0.0
    vehicle_reward_scale: float = 1.0
    critic_target_variant: str = "baseline"  # baseline | multistep_imagined
    critic_target_horizon: int = 3

    # ── HDV Gawron dynamic response (BPR-based travel time estimation) ───────
    # When enabled, hdv_frac evolves within each imagined episode via Gawron,
    # letting HDVs respond to changes in joint traffic load caused by AV routing.
    hdv_gawron:  bool  = True   # enable HDV dynamic update
    gawron_lr:   float = 0.1    # Gawron learning rate (RouteRL default ≈ 0.1)
    gawron_mode: str = "replicator"  # replicator | routerl | routerl_blackbox
    gawron_alpha_zero: float = 0.2   # RouteRL-compatible alpha_zero
    gawron_beta: float = -1.5        # RouteRL-compatible beta
    gawron_update_mode: str = "expected"  # expected | sample
    hdv_update_momentum: float = 0.2  # inter-episode HDV update smoothing
    hdv_min_prob: float = 0.02        # floor to avoid route-collapse artifacts
    bpr_alpha:   float = 0.15   # BPR volume-delay function alpha
    bpr_beta:    float = 4.0    # BPR volume-delay function beta
    hdv_internalized: bool = False  # WM internalizes HDV response; no explicit Gawron in imagined rollout

    # ── Dreamer reward shaping ───────────────────────────────────────────────
    conservative_penalty_coef: float = 0.02  # penalize large AV/HDV intent divergence
    use_terminal_tall_reward: bool = False  # legacy terminal reward from WM t_all head
    terminal_tall_reward_weight: float = 0.0  # keep Dreamer reward as step-level seconds BPR by default

    # ── checkpoint ───────────────────────────────────────────────────────────
    checkpoint_dir: str = "logs/marl"
    checkpoint_every: int = 100
    resume_ckpt: str = ""
    resume_reset_actor_optimizer: bool = False

    # ── dynamic imagination (time-stepped, async departures) ──────────────────
    imagination_horizon: int = 15     # T — number of WM steps per imagined episode
    episode_steps: int = 60           # total steps in full episode (for random start)
    step_interval: float = 30.0       # Δt seconds per decision step
    dreamer_departure_cap_mode: str = "none"  # none | cfg; none uses full agents.csv schedule

    dreamer_batch_episodes: int = 2
    dreamer_actor_horizon: int = 8
    dreamer_value_coef: float = 1.0
    dreamer_entropy_coef: float = 1e-3
    dreamer_critic_variant: str = "v_only"  # v_only | action_conditioned
    dreamer_critic_aggregation: str = "mean"  # mean | global_attention
    dreamer_critic_attention_heads: int = 4
    dreamer_critic_context: bool = True
    dreamer_critic_context_dim: int = 8
    dreamer_slow_critic_tau: float = 0.01
    dreamer_return_normalizer: bool = True
    dreamer_return_normalizer_decay: float = 0.99
    dreamer_vehicle_critic: bool = True
    dreamer_vehicle_critic_lr: float = 0.0
    dreamer_vehicle_critic_context: bool = True
    dreamer_vehicle_critic_architecture: str = "self_attention"
    dreamer_vehicle_critic_attention_heads: int = 4
    dreamer_vehicle_critic_value_coef: float = 1.0
    dreamer_vehicle_critic_return_normalizer: bool = True
    dreamer_allow_legacy_actor_only_losses: bool = False
    dreamer_actor_lr: float = 0.0  # 0 keeps legacy cfg.lr
    dreamer_actor_lr_decay_start_episode: int = -1
    dreamer_actor_lr_decay_end_episode: int = -1
    dreamer_actor_lr_final_scale: float = 1.0
    dreamer_actor_update_gate_start_episode: int = -1
    dreamer_actor_update_gate_max_route_frac: float = 0.0
    dreamer_actor_update_gate_route0_frac: float = 0.0
    dreamer_actor_update_gate_top1_prob: float = 0.0
    dreamer_actor_update_gate_min_samples: float = 128.0
    dreamer_actor_update_gate_latch: bool = False
    dreamer_actor_trust_region_kl: float = 0.0
    dreamer_actor_trust_region_max_argmax_change_frac: float = 0.0
    dreamer_actor_trust_region_max_vehicle_argmax_change_frac: float = 0.0
    dreamer_actor_trust_region_max_route_frac: float = 0.0
    dreamer_actor_trust_region_max_route0_frac: float = 0.0
    dreamer_actor_trust_region_max_retries: int = 0
    dreamer_actor_trust_region_lr_shrink: float = 0.5
    dreamer_actor_trust_region_min_samples: int = 128
    dreamer_actor_trust_region_max_samples: int = 4096
    dreamer_actor_anchor_start_episode: int = -1
    dreamer_actor_anchor_update_interval: int = 0
    dreamer_actor_anchor_kl: float = 0.0
    dreamer_actor_anchor_max_argmax_change_frac: float = 0.0
    dreamer_actor_anchor_max_vehicle_argmax_change_frac: float = 0.0
    dreamer_actor_anchor_max_route_frac: float = 0.0
    dreamer_actor_anchor_max_route0_frac: float = 0.0
    dreamer_critic_lr: float = 0.0  # 0 keeps legacy cfg.lr
    dreamer_actor_objective: str = "discrete_actor_critic"  # direct_return | critic_q | direct_plus_q | discrete_actor_critic
    dreamer_actor_q_weight: float = 1.0
    dreamer_ac_advantage_clip: float = 5.0
    dreamer_ac_ppo_clip_ratio: float = 0.1
    dreamer_ac_normalize_combined_advantages: bool = True
    dreamer_ac_weight_by_active_agents: bool = True
    dreamer_ac_step_advantage_weight: float = 1.0
    dreamer_ac_route_advantage_weight: float = 0.0
    dreamer_ac_entropy_target: float = 0.0
    dreamer_ac_entropy_target_frac: float = 0.0
    dreamer_ac_entropy_floor_coef: float = 0.0
    dreamer_ac_entropy_cost_gap_max: float = -1.0
    dreamer_ac_entropy_cost_min_routes: int = 1
    dreamer_ac_rollout_action_mode: str = "sample"  # sample | argmax | scheduled_argmax
    dreamer_ac_rollout_argmax_warmup_episodes: int = 0
    dreamer_ac_rollout_argmax_ramp_episodes: int = 0
    dreamer_ac_rollout_argmax_final_mix: float = 1.0
    dreamer_ac_route_credit_mode: str = "route_cost"  # route_cost | planned_pressure | marginal_pressure | sequential_marginal | counterfactual_pressure
    dreamer_ac_route_credit_pressure_weight: float = 1.0
    dreamer_ac_route_credit_pressure_mode: str = "clipped_excess_tt_ratio"
    dreamer_ac_route_credit_pressure_reference: str = "soft"  # soft | argmax | sequential_argmax
    dreamer_ac_route_credit_normalize_by_path_length: bool = False
    dreamer_ac_route_credit_base_cost_mode: str = "progress_fraction"
    dreamer_ac_route_credit_remaining_tt_eps: float = 1.0
    dreamer_ac_route_credit_progress_cap: float = 1.0
    dreamer_ac_route_credit_include_base_cost: bool = True
    dreamer_ac_route_advantage_temperature: float = 60.0
    dreamer_ac_normalize_route_advantages: bool = True
    dreamer_action_mode: str = "expected"  # expected | straight_through_sample | straight_through_argmax
    dreamer_action_temperature: float = 1.0
    dreamer_all_reward_weight: float = 1.0
    dreamer_remaining_time_progress_weight: float = 0.0
    dreamer_current_edge_reward_weight: float = 0.0
    dreamer_cav_reward_weight: float = 0.0
    dreamer_route_cost_reward_weight: float = 0.0
    dreamer_route_cost_reward_scope: str = "both"
    dreamer_route_cost_reward_mode: str = "progress_fraction"
    dreamer_pred_t_cav_proxy_reward_weight: float = 0.0
    dreamer_route_cost_reward_normalize_regret: bool = False
    dreamer_route_cost_reward_temperature: float = 60.0
    dreamer_route_cost_reward_planned_av_pressure_weight: float = 0.0
    dreamer_route_cost_reward_planned_av_pressure_mode: str = "overlap"
    dreamer_vehicle_route_credit_reward_weight: float = 0.0
    dreamer_vehicle_route_credit_reward_discount: float = 0.99
    dreamer_vehicle_route_credit_reward_transform: str = "negative_cost"
    dreamer_vehicle_route_credit_reward_normalize: bool = False
    dreamer_vehicle_route_credit_reward_temperature: float = 1.0
    dreamer_vehicle_route_credit_mode: str = "counterfactual_pressure"
    dreamer_vehicle_route_credit_pressure_weight: float = 1.0
    dreamer_vehicle_route_credit_pressure_mode: str = "clipped_excess_tt_ratio"
    dreamer_vehicle_route_credit_pressure_reference: str = "soft"
    dreamer_vehicle_route_credit_pressure_scale: str = "mean"
    dreamer_vehicle_route_credit_normalize_by_path_length: bool = False
    dreamer_vehicle_route_credit_base_cost_mode: str = "none"
    dreamer_vehicle_route_credit_include_base_cost: bool = False
    dreamer_vehicle_route_credit_cost_gate_max_seconds: float = -1.0
    dreamer_vehicle_route_credit_cost_gate_max_ratio: float = 0.0
    dreamer_vehicle_route_credit_cost_gate_min_routes: int = 1
    dreamer_vehicle_social_reward_weight: float = 0.0
    dreamer_vehicle_social_reward_source: str = "excess_all_tt"
    dreamer_vehicle_social_reward_distribution: str = "per_vehicle"
    dreamer_vehicle_social_reward_discount: float = 0.99
    dreamer_grpo_route_cost_weight: float = 0.0
    dreamer_grpo_planned_av_pressure_weight: float = 0.0
    dreamer_grpo_planned_av_pressure_mode: str = "overlap"
    dreamer_grpo_marginal_av_pressure_weight: float = 0.0
    dreamer_grpo_marginal_av_pressure_mode: str = "excess_tt_ratio"
    dreamer_grpo_marginal_av_pressure_reference: str = "soft"  # soft | argmax | sequential_argmax
    dreamer_grpo_marginal_av_pressure_normalize_by_path_length: bool = False
    dreamer_grpo_advantage_temperature: float = 60.0
    dreamer_grpo_normalize_advantages: bool = True
    dreamer_grpo_route_cost_pairwise_weight: float = 0.0
    dreamer_grpo_route_cost_pairwise_margin: float = 0.0
    dreamer_grpo_route_share_penalty_weight: float = 0.0
    dreamer_grpo_route_share_penalty_cap: float = 1.0
    dreamer_grpo_combined_advantage_clip: float = 5.0
    dreamer_grpo_policy_clip_ratio: float = 0.2
    dreamer_grpo_policy_kl_coef: float = 0.0
    dreamer_grpo_rollout_weight: float = 0.0
    dreamer_grpo_rollout_normalize_advantages: bool = True
    dreamer_grpo_rollout_cost_source: str = "reward"  # reward | pressure | cav_proxy | pressure_cav
    dreamer_grpo_rollout_discount: float = 0.99
    dreamer_grpo_rollout_t_cav_weight: float = 0.0
    dreamer_grpo_rollout_group_size: int = 1
    dreamer_grpo_rollout_vehicle_cost_weight: float = 0.0
    dreamer_grpo_rollout_vehicle_planned_av_pressure_weight: float = 0.0
    dreamer_grpo_rollout_baseline_policy: str = "none"  # none | route_cost_greedy | route_cost_pressure_greedy
    dreamer_grpo_rollout_baseline_planned_av_pressure_weight: float = 0.0
    dreamer_rollout_cost_reward_weight: float = 0.0
    dreamer_rollout_cost_reward_source: str = "pressure_cav"  # pressure | cav_proxy | pressure_cav
    dreamer_rollout_cost_reward_t_cav_weight: float = 0.0
    dreamer_rollout_cost_reward_scale_seconds: float = 60.0
    dreamer_hdv_background_mode: str = "none"  # none | uniform | adapted_state | episode_av_pressure_response | episode_hdv_head_response
    dreamer_hdv_background_state: str = ""
    dreamer_hdv_background_weight: float = 1.0
    dreamer_hdv_background_combine: str = "additive"  # additive | population
    dreamer_hdv_response_pressure_weight: float = 60.0
    dreamer_hdv_response_temperature: float = 60.0
    dreamer_hdv_response_prior_mix: float = 0.0
    dreamer_hdv_response_head_fallback: str = "analytic"  # analytic | prior | error
    dreamer_expected_cost_weight: float = 0.0
    dreamer_expected_cost_temperature: float = 60.0
    dreamer_expected_cost_planned_av_pressure_weight: float = 0.0
    dreamer_expected_cost_planned_av_pressure_mode: str = "overlap"
    dreamer_expected_cost_marginal_av_pressure_weight: float = 0.0
    dreamer_expected_cost_marginal_av_pressure_mode: str = "excess_tt_ratio"
    dreamer_expected_cost_marginal_av_pressure_reference: str = "soft"  # soft | argmax | sequential_argmax
    dreamer_expected_cost_marginal_av_pressure_normalize_by_path_length: bool = False
    dreamer_expected_cost_detach_cost: bool = True
    dreamer_expected_cost_balance_best_classes: bool = False
    dreamer_expected_cost_pairwise: bool = False
    dreamer_expected_cost_pairwise_margin: float = 0.0
    dreamer_expected_cost_route0_margin_weight: float = 0.0
    dreamer_expected_cost_route0_margin: float = 0.0
    dreamer_expected_cost_route0_margin_min_gap: float = 0.0
    dreamer_expected_cost_nonzero_best_boost: float = 0.0
    dreamer_expected_cost_nonzero_best_min_gap: float = 0.0
    dreamer_expected_cost_route_share_penalty_weight: float = 0.0
    dreamer_expected_cost_route_share_penalty_cap: float = 1.0
    dreamer_action_congestion_weight: float = 0.0
    dreamer_route_balance_weight: float = 0.0
    dreamer_route_max_share_weight: float = 0.0
    dreamer_route_max_share_cap: float = 1.0
    dreamer_route_cost_aux_weight: float = 0.0
    dreamer_route_cost_aux_balance: bool = False
    dreamer_route_cost_aux_hard_target: bool = False
    dreamer_route_cost_aux_nonzero_margin: float = 0.0
    dreamer_horizon_proxy_ckpt: str = ""
    dreamer_horizon_proxy_reward_weight: float = 0.0
    dreamer_remaining_time_softmin_temperature: float = 30.0
    dreamer_route_cost_aux_temperature: float = 30.0
    dreamer_actor_min_valid_agents_ratio: float = 0.0
    dreamer_normalize_critic_targets: bool = False
    dreamer_normalize_actor_advantages: bool = False
    dreamer_disable_actor_updates: bool = False
    dreamer_preserve_partial_batch_on_exit: bool = False

    # ── misc ─────────────────────────────────────────────────────────────────
    device: str = "cuda"
    use_amp: bool = True
    seed: int = 42
    log_every: int = 10


DREAMER_UPDATE_SCALAR_KEYS = (
    "pg_loss", "value_loss", "entropy", "approx_kl",
    "route_cost_aux_loss", "route_cost_aux_acc", "route_cost_aux_margin",
    "expected_cost_loss", "expected_cost_raw", "expected_cost_best_prob",
    "expected_cost_best_route0_frac", "expected_cost_nonzero_best_frac", "expected_cost_margin",
    "grpo_route_cost_loss", "grpo_route_cost_raw", "grpo_route_cost_best_prob",
    "grpo_route_cost_best_route0_frac", "grpo_route_cost_nonzero_best_frac",
    "grpo_route_cost_margin", "grpo_route_cost_adv_mean", "grpo_route_cost_adv_std",
    "grpo_rollout_loss", "grpo_rollout_cost", "grpo_rollout_adv_mean",
    "grpo_rollout_adv_std", "grpo_rollout_logprob", "grpo_rollout_selected_nonzero_frac",
    "grpo_combined_loss", "grpo_combined_pg_loss", "grpo_combined_kl_loss",
    "grpo_combined_adv_mean", "grpo_combined_adv_std", "grpo_combined_clip_frac",
    "grpo_policy_kl",
    "grpo_rollout_vehicle_cost_loss", "grpo_rollout_vehicle_cost",
    "grpo_rollout_vehicle_cost_adv_mean", "grpo_rollout_vehicle_cost_adv_std",
    "dreamer_ac_loss", "dreamer_ac_adv_mean", "dreamer_ac_adv_std",
    "dreamer_ac_adv_min", "dreamer_ac_adv_max",
    "dreamer_ac_route_adv_mean", "dreamer_ac_route_adv_std",
    "dreamer_ac_combined_adv_mean", "dreamer_ac_combined_adv_std",
    "dreamer_ac_selected_nonzero_frac",
    "dreamer_ac_selected_route0_frac", "dreamer_ac_selected_route1_frac",
    "dreamer_ac_selected_route2_frac", "dreamer_ac_selected_route3_frac",
    "dreamer_ac_policy_argmax_nonzero_frac",
    "dreamer_ac_policy_argmax_route0_frac", "dreamer_ac_policy_argmax_route1_frac",
    "dreamer_ac_policy_argmax_route2_frac", "dreamer_ac_policy_argmax_route3_frac",
    "dreamer_ac_policy_argmax_max_route_frac", "dreamer_ac_policy_top1_prob_mean",
    "dreamer_actor_gate_skipped", "dreamer_actor_gate_max_route_frac",
    "dreamer_actor_gate_route0_frac", "dreamer_actor_gate_top1_prob",
    "dreamer_actor_gate_sample_count", "dreamer_actor_gate_latched",
    "dreamer_actor_trust_region_accepted", "dreamer_actor_trust_region_kl",
    "dreamer_actor_trust_region_retries", "dreamer_actor_trust_region_sample_count",
    "dreamer_actor_trust_region_argmax_change_frac",
    "dreamer_actor_trust_region_vehicle_argmax_change_max",
    "dreamer_actor_trust_region_vehicle_argmax_change_mean",
    "dreamer_actor_trust_region_vehicle_argmax_change_p95",
    "dreamer_actor_trust_region_vehicle_group_count",
    "dreamer_actor_trust_region_argmax_max_route_frac",
    "dreamer_actor_trust_region_argmax_route0_frac",
    "dreamer_actor_trust_region_top1_prob",
    "dreamer_actor_anchor_active", "dreamer_actor_anchor_accepted",
    "dreamer_actor_anchor_kl", "dreamer_actor_anchor_argmax_change_frac",
    "dreamer_actor_anchor_vehicle_argmax_change_max",
    "dreamer_actor_anchor_vehicle_argmax_change_mean",
    "dreamer_actor_anchor_vehicle_argmax_change_p95",
    "dreamer_actor_anchor_vehicle_group_count",
    "dreamer_actor_anchor_argmax_max_route_frac",
    "dreamer_actor_anchor_argmax_route0_frac", "dreamer_actor_anchor_top1_prob",
    "dreamer_actor_anchor_sample_count",
    "dreamer_ac_sample_count",
    "dreamer_ac_policy_kl", "dreamer_ac_clip_frac", "dreamer_ac_weight_mean",
    "dreamer_ac_uses_sample_adv", "dreamer_ac_entropy_loss",
    "dreamer_ac_entropy_floor_loss", "dreamer_ac_entropy_target",
    "dreamer_ac_entropy_valid_target_mean",
    "dreamer_ac_entropy_effective_route_count", "dreamer_ac_entropy_cost_mask_frac",
    "vehicle_value_loss", "vehicle_critic_target_mean", "vehicle_critic_target_std",
    "vehicle_critic_explained_variance", "vehicle_return_norm_mean", "vehicle_return_norm_std",
    "vehicle_reward_mean", "vehicle_reward_std",
    "vehicle_route_progress_reward_mean", "vehicle_route_progress_reward_std",
    "vehicle_route_credit_reward_mean", "vehicle_route_credit_reward_std",
    "vehicle_route_credit_gate_frac", "vehicle_route_credit_selected_gate_frac",
    "vehicle_social_reward_mean", "vehicle_social_reward_std",
    "vehicle_return_discount",
    "critic_target_mean", "critic_target_std", "critic_explained_variance",
    "critic_return_discount",
    "return_norm_mean", "return_norm_std",
)


def resolve_wm_candidate_edge_cost_source(cfg: TrainConfig) -> str:
    """Return the edge-cost source used only for imagined candidate path generation."""
    src = str(getattr(cfg, "wm_candidate_edge_cost_source", "auto") or "auto").strip().lower()
    if src == "auto":
        src = str(getattr(cfg, "wm_edge_cost_source", "decoded_speed") or "decoded_speed").strip().lower()
    if src not in {"bpr_volume", "decoded_speed", "route_head"}:
        raise ValueError(f"Unsupported wm_candidate_edge_cost_source={src!r}")
    return src


def uses_explicit_dynamic_candidate_edge_cost_source(cfg: TrainConfig) -> bool:
    """True when the user explicitly asks candidate generation to track WM state costs."""
    src = str(getattr(cfg, "wm_candidate_edge_cost_source", "auto") or "auto").strip().lower()
    return src != "auto"


# ══════════════════════════════════════════════════════════════════════════════
# Offline initial-state dataset
# ══════════════════════════════════════════════════════════════════════════════

class OfflineInitDataset:
    """
    Provides initial edge-flow states sampled from the pre-collected flow dataset.

    The active MARL path is vehicle-level / edge-level only. Route-level
    HDV fractions are not required at initialization time.
    """

    def __init__(
        self,
        data_dir: str,
        meta_file: str,
        flow_glob: str,
        K: int,
        hdv_internalized: bool = False,
    ):
        meta_path = os.path.join(data_dir, meta_file)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"OfflineInitDataset: meta_file not found: {meta_path}"
            )
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)

        self.episode_ids: torch.Tensor = meta["episode"]          # (N,)
        self._K = int(K)
        self._hdv_internalized = bool(hdv_internalized)
        self.hdv_action = None
        logger.info(
            "OfflineInitDataset: vehicle/edge mode initialization active — "
            "route-level hdv_action is ignored."
        )

        # Load flow shards
        flow_shard_paths = sorted(glob.glob(os.path.join(data_dir, flow_glob)))
        assert flow_shard_paths, (
            f"No flow shards found at {os.path.join(data_dir, flow_glob)}"
        )
        flows_list = []
        for p in flow_shard_paths:
            shard = torch.load(p, map_location="cpu", weights_only=False)
            f = shard["flows"] if isinstance(shard, dict) else shard
            flows_list.append(f.float())
        self.flow_frames: torch.Tensor = torch.cat(flows_list, dim=0)  # (N_total, E, 8)

        N_flows = self.flow_frames.shape[0]
        N_meta  = self.episode_ids.shape[0]
        N = min(N_flows, N_meta)
        if N_flows != N_meta:
            logger.warning(
                f"OfflineInitDataset: flow count ({N_flows}) != "
                f"meta count ({N_meta}); truncating to {N}"
            )
        self.flow_frames = self.flow_frames[:N]
        self.episode_ids = self.episode_ids[:N]
        # Find episode boundaries: indices where episode_id changes
        ep_ids = self.episode_ids.numpy()
        changes = np.where(np.diff(ep_ids) != 0)[0] + 1
        self.episode_starts = np.concatenate([[0], changes])  # index of first flow per episode
        self.episode_ends = np.concatenate([changes, [N]])     # one-past-last flow per episode
        self.n_episodes = len(self.episode_starts)

        logger.info(
            f"OfflineInitDataset: {N} flows, "
            f"{self.n_episodes} episodes, K={K}"
        )

    def sample(self, rng: np.random.Generator) -> torch.Tensor:
        """
        Sample a random episode's initial state.

        Returns:
            flow_0   : (E, 8)   float32
        """
        ep_idx  = int(rng.integers(0, self.n_episodes))
        start   = self.episode_starts[ep_idx]
        flow_0   = self.flow_frames[start]    # (E, 8)
        return flow_0

    def sample_at_step(self, rng: np.random.Generator, step: int) -> torch.Tensor:
        """
        Sample a random episode's frame aligned to the requested decision step.

        Dynamic vehicle imagination rebuilds active vehicles at a sampled
        simulation step ``s``.  The latent traffic state must come from the same
        offset; otherwise incident-window decisions can be paired with a
        free-flow episode-start state.
        """
        ep_idx = int(rng.integers(0, self.n_episodes))
        start = int(self.episode_starts[ep_idx])
        end = int(self.episode_ends[ep_idx])
        if end <= start:
            idx = start
        else:
            offset = int(np.clip(int(step), 0, max(0, end - start - 1)))
            idx = start + offset
        return self.flow_frames[idx]


def edge_tt_from_latent_state(
    wm: UrbanWorldModel,
    z: torch.Tensor,
    bpr_alpha: float,
    bpr_beta: float,
    edge_cost_source: str = "decoded_speed",
    decoded_speed_min_ratio: float = 0.05,
    decoded_speed_max_ratio: float = 1.50,
    decoded_speed_excess_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode latent z to edge dynamic features and derive edge travel times.

    By default this preserves the legacy decoded-volume -> BPR path.  The
    ``decoded_speed`` source is used for route-rank-speed checkpoints.  The
    ``route_head`` source uses a planner-cost adapter trained from dynamic
    state-aligned route-cost distillation without changing decoder speed/volume.

    Returns:
        edge_dyn: (E, n_dynamic)
        v_e: (E,) estimated edge vehicle volumes
        free_flow: (E,) seconds
        t_e: (E,) seconds
    """
    edge_dyn = wm.flow_projector.decode(z).detach().cpu().numpy().astype(np.float32)
    capacities = np.asarray(wm._registry.capacities, dtype=np.float32)
    v_e = np.maximum(edge_dyn[:, 2], 0.0) * np.maximum(capacities, 1.0)
    lengths = np.asarray(
        wm._registry.lengths if wm._registry.lengths is not None
        else np.ones((wm.n_edges,), dtype=np.float32) * 100.0,
        dtype=np.float32,
    )
    speed_limits = np.asarray(wm._registry.speed_limits, dtype=np.float32)
    free_flow = lengths / np.maximum(speed_limits, 0.1)

    source = str(edge_cost_source or "decoded_speed").strip().lower()
    if source in {"route_head", "route_cost_head", "planner_head"}:
        free_flow_t = torch.as_tensor(free_flow, device=z.device, dtype=z.dtype)
        if z.dim() == 3 and z.shape[0] == 1:
            z_head = z[0]
        else:
            z_head = z
        with torch.no_grad():
            t_e = wm.flow_projector.predict_edge_route_cost(z_head, free_flow_t).detach().cpu().numpy().astype(np.float32)
        return edge_dyn, v_e.astype(np.float32), free_flow.astype(np.float32), t_e.astype(np.float32)

    if source in {"decoded_speed", "decoder_speed", "speed"}:
        min_ratio = max(float(decoded_speed_min_ratio), 1e-4)
        max_ratio = max(float(decoded_speed_max_ratio), min_ratio)
        speed_ratio = np.nan_to_num(
            edge_dyn[:, 1],
            nan=1.0,
            posinf=max_ratio,
            neginf=min_ratio,
        ).astype(np.float32)
        speed_ratio = np.clip(speed_ratio, min_ratio, max_ratio)
        excess_scale = max(float(decoded_speed_excess_scale), 0.0)
        t_e = free_flow * (1.0 + excess_scale * ((1.0 / speed_ratio) - 1.0))
        return edge_dyn, v_e.astype(np.float32), free_flow.astype(np.float32), t_e.astype(np.float32)

    if source not in {"bpr_volume", "volume_bpr", "bpr"}:
        raise ValueError(
            f"Unsupported edge_cost_source={edge_cost_source!r}; expected 'bpr_volume', 'decoded_speed', or 'route_head'"
        )

    if hasattr(wm, "bpr") and wm.bpr is not None:
        t_e = wm.bpr.numpy_forward(v_e)
        free_flow = wm.bpr.free_flow_time.detach().cpu().numpy().astype(np.float32)
    else:
        t_e = free_flow * (
            1.0 + float(bpr_alpha) * (v_e / np.maximum(capacities, 1.0)) ** float(bpr_beta)
        )
    return edge_dyn, v_e.astype(np.float32), free_flow.astype(np.float32), t_e.astype(np.float32)


def _edge_tt_tensor_from_decoded(
    *,
    wm: UrbanWorldModel,
    decoded: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    edge_cost_source: str,
    decoded_speed_min_ratio: float,
    decoded_speed_max_ratio: float,
    decoded_speed_excess_scale: float = 1.0,
) -> torch.Tensor:
    source = str(edge_cost_source or "decoded_speed").strip().lower()
    if source in {"route_head", "route_cost_head", "planner_head"}:
        if z is None:
            raise ValueError("edge_cost_source='route_head' requires the latent state z.")
        lengths = torch.as_tensor(
            np.asarray(
                wm._registry.lengths if wm._registry.lengths is not None
                else np.ones((wm.n_edges,), dtype=np.float32) * 100.0,
                dtype=np.float32,
            ),
            device=z.device,
            dtype=z.dtype,
        )
        speed_limits = torch.as_tensor(
            np.asarray(wm._registry.speed_limits, dtype=np.float32),
            device=z.device,
            dtype=z.dtype,
        )
        free_flow = lengths / speed_limits.clamp_min(0.1)
        return wm.flow_projector.predict_edge_route_cost(z, free_flow)

    if source in {"decoded_speed", "decoder_speed", "speed"}:
        lengths = torch.as_tensor(
            np.asarray(
                wm._registry.lengths if wm._registry.lengths is not None
                else np.ones((wm.n_edges,), dtype=np.float32) * 100.0,
                dtype=np.float32,
            ),
            device=decoded.device,
            dtype=decoded.dtype,
        )
        speed_limits = torch.as_tensor(
            np.asarray(wm._registry.speed_limits, dtype=np.float32),
            device=decoded.device,
            dtype=decoded.dtype,
        )
        min_ratio = max(float(decoded_speed_min_ratio), 1e-4)
        max_ratio = max(float(decoded_speed_max_ratio), min_ratio)
        speed_ratio = torch.nan_to_num(
            decoded[:, 1],
            nan=1.0,
            posinf=max_ratio,
            neginf=min_ratio,
        ).clamp(min=min_ratio, max=max_ratio)
        free_flow = lengths / speed_limits.clamp_min(0.1)
        excess_scale = max(float(decoded_speed_excess_scale), 0.0)
        return free_flow * (1.0 + excess_scale * ((1.0 / speed_ratio) - 1.0))

    if source in {"bpr_volume", "volume_bpr", "bpr"}:
        capacities = wm.bpr.capacities.to(device=decoded.device, dtype=decoded.dtype)
        v_total = torch.relu(decoded[:, 2]) * capacities
        return wm.bpr(v_total)

    raise ValueError(
        f"Unsupported edge_cost_source={edge_cost_source!r}; expected 'bpr_volume', 'decoded_speed', or 'route_head'"
    )


def marginal_delay_reward_on_edge(
    edge_idx: int,
    v_e: np.ndarray,
    free_flow: np.ndarray,
    capacities: np.ndarray,
    bpr_alpha: float,
    bpr_beta: float,
    reward_scale: float,
) -> float:
    """
    Per-vehicle marginal delay contribution approximation on one traversed edge.
    """
    e = int(edge_idx)
    if e < 0 or e >= v_e.shape[0]:
        return 0.0
    v0 = max(float(v_e[e]), 0.0)
    c = max(float(capacities[e]), 1.0)
    ff = max(float(free_flow[e]), 1e-6)
    a = float(bpr_alpha)
    b = float(bpr_beta)
    v1 = max(v0 - 1.0, 0.0)
    delay0 = v0 * ff * a * ((v0 / c) ** b)
    delay1 = v1 * ff * a * ((v1 / c) ** b)
    marginal = max(delay0 - delay1, 0.0)
    return -float(reward_scale) * float(marginal)


def vehicle_mode_action_supported(
    *,
    n_edges: int,
    action_dim: int,
    raw_action_dim: int,
) -> bool:
    if int(action_dim) == int(n_edges):
        return True
    if int(raw_action_dim) == int(n_edges):
        return True
    return False


def vehicle_mode_obs_dim(
    *,
    n_routes: int,
    d_obs: int,
    include_global: bool,
    route_stats: str = "mean",
    include_route_costs: bool = False,
    include_fleet_pressure: bool = False,
) -> int:
    route_feat_dim = vehicle_route_feature_dim(int(d_obs), route_stats)
    route_cost_dim = 3 * int(n_routes) if bool(include_route_costs) else 0
    fleet_pressure_dim = 2 * int(n_routes) if bool(include_fleet_pressure) else 0
    return int(n_routes) * route_feat_dim + (route_feat_dim if include_global else 0) + fleet_pressure_dim + route_cost_dim + 3


def candidate_actor_start_steps(
    *,
    departure_schedule: Optional[Dict],
    episode_steps: int,
    horizon: int,
    avg_route_steps: int,
) -> List[int]:
    """Return start steps likely to contain at least one actor-controlled AV step."""
    if not departure_schedule:
        return []
    max_start = max(1, int(episode_steps) - int(horizon))
    candidates: List[int] = []
    for s in range(max_start):
        active_avs, _ = rebuild_active_set(departure_schedule, s, avg_route_steps)
        if active_avs:
            candidates.append(int(s))
            continue
        for step in range(int(s), min(int(s + horizon), int(episode_steps))):
            dep = departure_schedule.get(step)
            if dep is not None and int(dep.get("n_av", 0)) > 0:
                candidates.append(int(s))
                break
    return candidates



def _dreamer_step_reward(
    wm: UrbanWorldModel,
    z_next: torch.Tensor,
    reward_scale: float,
    z_current: Optional[torch.Tensor] = None,
    action_edge: Optional[torch.Tensor] = None,
    current_edge_indices: Optional[Union[torch.Tensor, np.ndarray, List[int]]] = None,
    route_probs: Optional[torch.Tensor] = None,
    route_balance_probs: Optional[torch.Tensor] = None,
    path_batch: Optional[List[List[List[int]]]] = None,
    horizon_proxy: Optional[ActionHorizonProxyReward] = None,
    *,
    all_reward_weight: float = 1.0,
    remaining_time_progress_weight: float = 0.0,
    current_edge_reward_weight: float = 0.0,
    cav_reward_weight: float = 0.0,
    route_cost_reward_weight: float = 0.0,
    pred_t_cav_proxy_reward_weight: float = 0.0,
    route_cost_reward_normalize_regret: bool = False,
    route_cost_reward_temperature: float = 60.0,
    route_cost_reward_planned_av_pressure_weight: float = 0.0,
    route_cost_reward_planned_av_pressure_mode: str = "overlap",
    action_congestion_weight: float = 0.0,
    route_balance_weight: float = 0.0,
    route_max_share_weight: float = 0.0,
    route_max_share_cap: float = 1.0,
    horizon_proxy_reward_weight: float = 0.0,
    remaining_time_softmin_temperature: float = 30.0,
    remaining_time_progress_step_seconds: float = 0.0,
    cav_route_steps: float = 1.0,
    invalid_route_cost_factor: float = 3.0,
    edge_cost_source: str = "decoded_speed",
    decoded_speed_min_ratio: float = 0.05,
    decoded_speed_max_ratio: float = 1.50,
    decoded_speed_excess_scale: float = 1.0,
) -> Tuple[torch.Tensor, float, float]:
    edge_dyn_next = wm.flow_projector.decode(z_next)
    capacities = wm.bpr.capacities.to(device=z_next.device, dtype=z_next.dtype)
    v_total = torch.relu(edge_dyn_next[:, 2]) * capacities
    edge_tt = _edge_tt_tensor_from_decoded(
        wm=wm,
        decoded=edge_dyn_next,
        z=z_next,
        edge_cost_source=edge_cost_source,
        decoded_speed_min_ratio=decoded_speed_min_ratio,
        decoded_speed_max_ratio=decoded_speed_max_ratio,
        decoded_speed_excess_scale=decoded_speed_excess_scale,
    )
    total_tt = torch.sum(v_total * edge_tt)
    n_veh = v_total.sum().clamp_min(1.0)
    all_tt = total_tt / n_veh

    remaining_progress = torch.zeros((), device=z_next.device, dtype=z_next.dtype)
    remaining_next = None
    if (
        z_current is not None
        and route_probs is not None
        and path_batch is not None
        and float(remaining_time_progress_weight) != 0.0
    ):
        edge_dyn_current = wm.flow_projector.decode(z_current)
        edge_tt_current = _edge_tt_tensor_from_decoded(
            wm=wm,
            decoded=edge_dyn_current,
            z=z_current,
            edge_cost_source=edge_cost_source,
            decoded_speed_min_ratio=decoded_speed_min_ratio,
            decoded_speed_max_ratio=decoded_speed_max_ratio,
            decoded_speed_excess_scale=decoded_speed_excess_scale,
        )
        fallback_current = (
            edge_tt_current.mean()
            * float(max(cav_route_steps, 1.0))
            * float(invalid_route_cost_factor)
        )
        fallback_next = (
            edge_tt.mean()
            * float(max(cav_route_steps, 1.0))
            * float(invalid_route_cost_factor)
        )
        remaining_current, remaining_next = _vehicle_potential_progress_terms_from_paths(
            route_probs=route_probs,
            path_batch=path_batch,
            edge_cost_current=edge_tt_current,
            edge_cost_next=edge_tt,
            n_routes=int(route_probs.shape[1]),
            invalid_route_cost_current=fallback_current,
            invalid_route_cost_next=fallback_next,
            temperature=float(remaining_time_softmin_temperature),
            step_seconds=float(remaining_time_progress_step_seconds),
        )
        if remaining_current is not None and remaining_next is not None:
            remaining_progress = remaining_current - remaining_next

    current_edge_tt = all_tt
    if current_edge_indices is not None and float(current_edge_reward_weight) != 0.0:
        current_idx = torch.as_tensor(
            current_edge_indices,
            device=z_next.device,
            dtype=torch.long,
        ).reshape(-1)
        valid = (current_idx >= 0) & (current_idx < int(edge_tt.shape[0]))
        if bool(valid.any().detach().cpu().item()):
            current_edge_tt = edge_tt[current_idx[valid]].mean()

    cav_tt = all_tt
    if action_edge is not None and float(cav_reward_weight) != 0.0:
        action = torch.relu(action_edge.to(device=z_next.device, dtype=z_next.dtype))
        action_sum = action.sum()
        if bool(torch.isfinite(action_sum).detach().cpu().item()) and float(action_sum.detach().cpu().item()) > 1e-8:
            action_dist = action / action_sum.clamp_min(1e-8)
            cav_tt = torch.sum(action_dist * edge_tt) * float(max(cav_route_steps, 1e-6))

    route_tt = None
    pred_t_cav_proxy_tt = None
    if (
        route_probs is not None
        and path_batch is not None
        and (
            float(route_cost_reward_weight) != 0.0
            or float(pred_t_cav_proxy_reward_weight) != 0.0
        )
    ):
        invalid_route_cost = (
            edge_tt.mean()
            * float(max(cav_route_steps, 1.0))
            * float(invalid_route_cost_factor)
        )
        pressure_weight = float(route_cost_reward_planned_av_pressure_weight)
        route_plan_abs_tt = None
        route_plan_reward_cost = None
        if pressure_weight != 0.0:
            route_costs, route_valid = _route_cost_matrix_from_paths(
                path_batch=path_batch,
                edge_cost=edge_tt,
                n_routes=int(route_probs.shape[1]),
                invalid_route_cost=invalid_route_cost,
            )
            if route_costs is not None and route_valid is not None:
                pressure_edge_weights = _planned_av_pressure_edge_weights(
                    wm=wm,
                    edge_tt=edge_tt,
                    v_total=v_total,
                    capacities=capacities,
                    mode=route_cost_reward_planned_av_pressure_mode,
                )
                pressure_costs = _planned_av_pressure_cost_matrix(
                    route_probs=route_probs,
                    path_batch=path_batch,
                    route_valid=route_valid,
                    n_edges=int(edge_tt.shape[0]),
                    n_routes=int(route_probs.shape[1]),
                    edge_weights=pressure_edge_weights,
                    detach_pressure=True,
                )
                if pressure_costs is not None:
                    route_costs = route_costs + pressure_weight * pressure_costs.to(
                        device=route_costs.device,
                        dtype=route_costs.dtype,
                    )
                route_plan_abs_tt = _expected_route_cost_from_cost_matrix(
                    route_probs=route_probs,
                    route_costs=route_costs,
                    route_valid=route_valid,
                )
                route_plan_reward_cost = (
                    _expected_route_regret_from_cost_matrix(
                        route_probs=route_probs,
                        route_costs=route_costs,
                        route_valid=route_valid,
                        temperature=float(route_cost_reward_temperature),
                    )
                    if bool(route_cost_reward_normalize_regret)
                    else route_plan_abs_tt
                )
        else:
            route_plan_abs_tt = _expected_route_cost_from_paths(
                route_probs=route_probs,
                path_batch=path_batch,
                edge_cost=edge_tt,
                n_routes=int(route_probs.shape[1]),
                invalid_route_cost=invalid_route_cost,
            )
            route_plan_reward_cost = route_plan_abs_tt
            if bool(route_cost_reward_normalize_regret):
                route_plan_reward_cost = _expected_route_regret_from_paths(
                    route_probs=route_probs,
                    path_batch=path_batch,
                    edge_cost=edge_tt,
                    n_routes=int(route_probs.shape[1]),
                    invalid_route_cost=invalid_route_cost,
                    temperature=float(route_cost_reward_temperature),
                )
        if route_plan_abs_tt is not None:
            cav_tt = route_plan_abs_tt
        if route_plan_reward_cost is not None:
            if float(route_cost_reward_weight) != 0.0:
                route_tt = route_plan_reward_cost
            if float(pred_t_cav_proxy_reward_weight) != 0.0:
                pred_t_cav_proxy_tt = route_plan_reward_cost

    route_balance = torch.zeros((), device=z_next.device, dtype=z_next.dtype)
    route_max_share = torch.zeros((), device=z_next.device, dtype=z_next.dtype)
    balance_source_probs = route_balance_probs if route_balance_probs is not None else route_probs
    if balance_source_probs is not None and (
        float(route_balance_weight) != 0.0 or float(route_max_share_weight) != 0.0
    ):
        probs = torch.relu(balance_source_probs.to(device=z_next.device, dtype=z_next.dtype))
        prob_sum = probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        probs = probs / prob_sum
        fleet_probs = probs.mean(dim=0)
        if float(route_balance_weight) != 0.0:
            uniform_floor = 1.0 / float(max(int(probs.shape[-1]), 1))
            route_balance = (fleet_probs.pow(2).sum() - uniform_floor).clamp_min(0.0)
        if float(route_max_share_weight) != 0.0:
            cap = max(0.0, min(float(route_max_share_cap), 1.0))
            route_max_share = (fleet_probs.max() - cap).clamp_min(0.0).pow(2)

    horizon_proxy_tt = torch.zeros((), device=z_next.device, dtype=z_next.dtype)
    if (
        horizon_proxy is not None
        and action_edge is not None
        and float(horizon_proxy_reward_weight) != 0.0
    ):
        proxy_z = z_current if z_current is not None else z_next
        proxy_dyn = wm.flow_projector.decode(proxy_z)
        static_np = np.asarray(wm._registry.static_features, dtype=np.float32)
        static_t = torch.from_numpy(static_np).to(device=z_next.device, dtype=z_next.dtype)
        horizon_proxy_tt = horizon_proxy.predict_minutes(
            decoded_dyn=proxy_dyn,
            static_features=static_t,
            action_edge=action_edge,
        ) * 60.0

    action_congestion = torch.zeros((), device=z_next.device, dtype=z_next.dtype)
    if action_edge is not None and float(action_congestion_weight) != 0.0:
        action = torch.relu(action_edge.to(device=z_next.device, dtype=z_next.dtype))
        action_sum = action.sum()
        if bool(torch.isfinite(action_sum).detach().cpu().item()) and float(action_sum.detach().cpu().item()) > 1e-8:
            action_dist = action / action_sum.clamp_min(1e-8)
            cap_scale = capacities / capacities.mean().clamp_min(1.0)
            volume_ratio = (v_total / capacities.clamp_min(1.0)).clamp_min(0.0)
            # Penalize planned AV load on edges that the WM already sees as busy,
            # with a small concentration term to discourage route collapse.  This
            # uses only imagined state/action/capacity, not route labels or SUMO
            # future outcomes.
            pressure = volume_ratio.pow(2) / cap_scale.clamp_min(1e-3)
            pressure_cost = torch.sum(action_dist * pressure)
            concentration_cost = torch.sum((action_dist / cap_scale.clamp_min(1e-3)) ** 2)
            action_congestion = (pressure_cost + 0.1 * concentration_cost) * float(max(cav_route_steps, 1.0))

    reward_cost = (
        float(all_reward_weight) * all_tt
        + float(current_edge_reward_weight) * current_edge_tt
        + float(cav_reward_weight) * cav_tt
        + float(route_cost_reward_weight) * (route_tt if route_tt is not None else cav_tt)
        + float(pred_t_cav_proxy_reward_weight) * (
            pred_t_cav_proxy_tt if pred_t_cav_proxy_tt is not None else cav_tt
        )
        + float(action_congestion_weight) * action_congestion
        + float(route_balance_weight) * route_balance
        + float(route_max_share_weight) * route_max_share
        + float(horizon_proxy_reward_weight) * horizon_proxy_tt
    )
    reward = (
        -reward_cost
        + float(remaining_time_progress_weight) * remaining_progress
    ) * float(reward_scale)
    t_all_step_min = float(all_tt.detach().cpu().item() / 60.0)
    if remaining_next is not None and float(remaining_time_progress_weight) != 0.0:
        t_cav_step_min = float(remaining_next.detach().cpu().item() / 60.0)
    elif float(current_edge_reward_weight) != 0.0:
        t_cav_step_min = float(current_edge_tt.detach().cpu().item() / 60.0)
    else:
        t_cav_step_min = float(cav_tt.detach().cpu().item() / 60.0)
    return reward, t_all_step_min, t_cav_step_min


def _selected_route_vehicle_rewards_from_cost_matrix(
    *,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
    selected_idx: torch.Tensor,
    reward_weight: float,
    reward_scale: float,
    reward_mode: str,
    step_interval: float,
    elapsed_time: Optional[torch.Tensor],
    progress_cap: float,
    remaining_tt_eps: float,
    normalize_regret: bool,
    temperature: float,
) -> torch.Tensor:
    """Per-vehicle counterpart of the route-cost Dreamer reward."""
    n = min(
        int(route_costs.shape[0]),
        int(route_valid.shape[0]),
        int(selected_idx.reshape(-1).shape[0]),
    )
    if n <= 0:
        return torch.zeros((0,), device=route_costs.device, dtype=route_costs.dtype)
    costs = route_costs[:n]
    valid = route_valid[:n].to(device=route_costs.device)
    selected = selected_idx.reshape(-1)[:n].to(device=route_costs.device, dtype=torch.long)
    selected = selected.clamp(0, max(int(costs.shape[1]) - 1, 0))
    rows = torch.arange(n, device=route_costs.device)
    selected_valid = valid[rows, selected]
    mode_key = str(reward_mode or "progress_fraction").strip().lower()
    if mode_key in {"progress_fraction", "step_progress", "trip_progress_fraction"}:
        progress_costs = _route_credit_base_cost_matrix(
            route_tt_costs=costs,
            route_valid=valid,
            mode="progress_fraction",
            step_interval=float(step_interval),
            elapsed_time=elapsed_time,
            remaining_tt_eps=float(remaining_tt_eps),
            progress_cap=float(progress_cap),
        )
        # _route_credit_base_cost_matrix returns negative progress as a cost,
        # so negating gives a critic-visible reward in [0, progress_cap].
        reward = -float(reward_weight) * progress_costs[rows, selected] * float(reward_scale)
    elif bool(normalize_regret):
        selected_cost = costs[rows, selected]
        large = torch.finfo(costs.dtype).max / 4.0
        min_cost = costs.masked_fill(~valid, large).min(dim=-1).values
        selected_cost = (selected_cost - min_cost) / float(max(temperature, 1e-6))
        reward = -float(reward_weight) * selected_cost * float(reward_scale)
    else:
        if mode_key not in {"", "route_tt", "remaining_tt", "full_path", "legacy"}:
            raise ValueError(
                f"Unsupported dreamer_route_cost_reward_mode={reward_mode!r}; "
                "expected progress_fraction or route_tt"
            )
        selected_cost = costs[rows, selected]
        reward = -float(reward_weight) * selected_cost * float(reward_scale)
    reward = torch.where(selected_valid, reward, torch.zeros_like(reward))
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def _selected_route_credit_vehicle_rewards_from_cost_matrix(
    *,
    route_credit_costs: torch.Tensor,
    route_valid: torch.Tensor,
    selected_idx: torch.Tensor,
    reward_weight: float,
    reward_scale: float,
    reward_transform: str = "negative_cost",
    normalize: bool = False,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Convert selected route-credit costs into critic-visible rewards."""
    n = min(
        int(route_credit_costs.shape[0]),
        int(route_valid.shape[0]),
        int(selected_idx.reshape(-1).shape[0]),
    )
    if n <= 0 or float(reward_weight) == 0.0:
        return torch.zeros((max(n, 0),), device=route_credit_costs.device, dtype=route_credit_costs.dtype)
    costs = route_credit_costs[:n]
    valid = route_valid[:n].to(device=route_credit_costs.device)
    selected = selected_idx.reshape(-1)[:n].to(device=route_credit_costs.device, dtype=torch.long)
    selected = selected.clamp(0, max(int(costs.shape[1]) - 1, 0))
    rows = torch.arange(n, device=route_credit_costs.device)
    selected_valid = valid[rows, selected]
    selected_cost = costs[rows, selected]
    transform_key = str(reward_transform or "negative_cost").strip().lower()
    if transform_key in {"", "negative_cost", "cost", "legacy", "absolute_cost"}:
        score = -selected_cost
    elif transform_key in {"relative_mean", "mean", "mean_baseline", "advantage"}:
        valid_f = valid.to(dtype=costs.dtype)
        valid_count = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        cost_mean = (costs.masked_fill(~valid, 0.0) * valid_f).sum(dim=-1, keepdim=True) / valid_count
        score = cost_mean.squeeze(-1) - selected_cost
        if bool(normalize):
            centered = (costs - cost_mean).masked_fill(~valid, 0.0)
            cost_std = torch.sqrt(
                (centered.square() * valid_f).sum(dim=-1) / valid_count.squeeze(-1)
            ).clamp_min(1e-6)
            score = score / cost_std
        else:
            score = score / float(max(temperature, 1e-6))
    elif transform_key in {"relative_min", "min_regret", "best_regret"}:
        large = torch.finfo(costs.dtype).max / 4.0
        min_cost = costs.masked_fill(~valid, large).min(dim=-1).values
        score = min_cost - selected_cost
        if bool(normalize):
            valid_f = valid.to(dtype=costs.dtype)
            valid_count = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
            cost_mean = (costs.masked_fill(~valid, 0.0) * valid_f).sum(dim=-1, keepdim=True) / valid_count
            centered = (costs - cost_mean).masked_fill(~valid, 0.0)
            cost_std = torch.sqrt(
                (centered.square() * valid_f).sum(dim=-1) / valid_count.squeeze(-1)
            ).clamp_min(1e-6)
            score = score / cost_std
        else:
            score = score / float(max(temperature, 1e-6))
    elif transform_key in {"zscore", "relative_zscore", "standardized_mean"}:
        valid_f = valid.to(dtype=costs.dtype)
        valid_count = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        cost_mean = (costs.masked_fill(~valid, 0.0) * valid_f).sum(dim=-1, keepdim=True) / valid_count
        centered = (costs - cost_mean).masked_fill(~valid, 0.0)
        cost_std = torch.sqrt(
            (centered.square() * valid_f).sum(dim=-1) / valid_count.squeeze(-1)
        ).clamp_min(1e-6)
        score = (cost_mean.squeeze(-1) - selected_cost) / cost_std
    else:
        raise ValueError(
            f"Unsupported route-credit reward transform {reward_transform!r}; "
            "expected negative_cost, relative_mean, relative_min, or zscore"
        )
    reward = float(reward_weight) * score * float(reward_scale)
    reward = torch.where(selected_valid, reward, torch.zeros_like(reward))
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def _route_credit_cost_gate_mask(
    *,
    route_tt_costs: torch.Tensor,
    route_valid: torch.Tensor,
    max_seconds: float,
    max_ratio: float,
    min_routes: int,
) -> torch.Tensor:
    """Mask candidates that are close enough to the local fastest route.

    The gate is intentionally based on local travel-time cost, not pressure
    cost.  It lets critic-visible externality rewards break near ties while
    preventing them from making clearly slower routes look attractive.
    """
    n = min(int(route_tt_costs.shape[0]), int(route_valid.shape[0]))
    if n <= 0:
        return torch.zeros(
            tuple(route_valid.shape),
            dtype=torch.bool,
            device=route_tt_costs.device,
        )
    costs = route_tt_costs[:n]
    valid = route_valid[:n].to(device=route_tt_costs.device, dtype=torch.bool)
    if float(max_seconds) < 0.0 and float(max_ratio) <= 0.0 and int(min_routes) <= 1:
        return valid

    large = torch.finfo(costs.dtype).max / 4.0
    masked_costs = costs.masked_fill(~valid, large)
    best_cost = masked_costs.min(dim=-1, keepdim=True).values
    finite_best = torch.isfinite(best_cost) & (best_cost < large / 2.0)
    threshold = torch.zeros_like(best_cost)
    if float(max_seconds) >= 0.0:
        threshold = torch.maximum(
            threshold,
            torch.full_like(threshold, float(max_seconds)),
        )
    if float(max_ratio) > 0.0:
        threshold = torch.maximum(
            threshold,
            best_cost.clamp_min(0.0) * float(max_ratio),
        )
    gated = valid & finite_best & ((costs - best_cost) <= threshold)

    keep_count = max(int(min_routes), 1)
    if keep_count > 1 and int(costs.shape[1]) > 1:
        valid_count = valid.sum(dim=-1)
        keep = torch.minimum(
            valid_count,
            torch.full_like(valid_count, keep_count),
        ).clamp_min(1)
        order = torch.argsort(masked_costs, dim=-1, stable=True)
        rank = torch.arange(int(costs.shape[1]), device=costs.device).unsqueeze(0).expand_as(order)
        top_sorted = rank < keep.unsqueeze(-1)
        top_keep = torch.zeros_like(valid)
        top_keep.scatter_(1, order, top_sorted)
        gated = gated | (top_keep & valid)

    no_gated = valid.any(dim=-1) & ~gated.any(dim=-1)
    if bool(no_gated.any().detach().cpu().item()):
        fallback_idx = masked_costs.argmin(dim=-1)
        gated[no_gated, fallback_idx[no_gated]] = True
    return gated


def _selected_route_credit_cost_gate_stats(
    *,
    route_tt_costs: torch.Tensor,
    route_valid: torch.Tensor,
    selected_idx: torch.Tensor,
    max_seconds: float,
    max_ratio: float,
    min_routes: int,
) -> Tuple[float, float]:
    if float(max_seconds) < 0.0 and float(max_ratio) <= 0.0 and int(min_routes) <= 1:
        return 1.0, 1.0
    n = min(
        int(route_tt_costs.shape[0]),
        int(route_valid.shape[0]),
        int(selected_idx.reshape(-1).shape[0]),
    )
    if n <= 0:
        return 0.0, 0.0
    gate = _route_credit_cost_gate_mask(
        route_tt_costs=route_tt_costs[:n],
        route_valid=route_valid[:n],
        max_seconds=float(max_seconds),
        max_ratio=float(max_ratio),
        min_routes=int(min_routes),
    )
    valid = route_valid[:n].to(device=gate.device, dtype=torch.bool)
    valid_count = valid.sum().clamp_min(1)
    gate_frac = float((gate & valid).to(dtype=torch.float32).sum().div(valid_count).detach().cpu().item())
    selected = selected_idx.reshape(-1)[:n].to(device=gate.device, dtype=torch.long)
    selected = selected.clamp(0, max(int(gate.shape[1]) - 1, 0))
    rows = torch.arange(n, device=gate.device)
    selected_valid = valid[rows, selected]
    selected_denom = selected_valid.to(dtype=torch.float32).sum().clamp_min(1.0)
    selected_frac = float(
        (gate[rows, selected] & selected_valid)
        .to(dtype=torch.float32)
        .sum()
        .div(selected_denom)
        .detach()
        .cpu()
        .item()
    )
    return gate_frac, selected_frac


def _apply_route_credit_cost_gate(
    *,
    route_credit_costs: torch.Tensor,
    route_tt_costs: torch.Tensor,
    route_valid: torch.Tensor,
    max_seconds: float,
    max_ratio: float,
    min_routes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if float(max_seconds) < 0.0 and float(max_ratio) <= 0.0 and int(min_routes) <= 1:
        return route_credit_costs, route_valid
    n = min(
        int(route_credit_costs.shape[0]),
        int(route_tt_costs.shape[0]),
        int(route_valid.shape[0]),
    )
    if n <= 0:
        return route_credit_costs, route_valid
    gated_valid = route_valid.clone()
    gate = _route_credit_cost_gate_mask(
        route_tt_costs=route_tt_costs[:n].to(
            device=route_credit_costs.device,
            dtype=route_credit_costs.dtype,
        ),
        route_valid=route_valid[:n].to(device=route_credit_costs.device),
        max_seconds=float(max_seconds),
        max_ratio=float(max_ratio),
        min_routes=int(min_routes),
    )
    gated_valid[:n] = route_valid[:n].to(device=gated_valid.device) & gate.to(device=gated_valid.device)
    return route_credit_costs, gated_valid


def _vehicle_social_reward_from_decoded(
    *,
    wm: UrbanWorldModel,
    decoded: torch.Tensor,
    edge_tt: torch.Tensor,
    selected_idx: torch.Tensor,
    valid_action_mask: torch.Tensor,
    reward_weight: float,
    reward_scale: float,
    source: str,
    distribution: str,
    n_active_av: int,
) -> torch.Tensor:
    """Broadcast critic-mediated global/social WM reward to vehicle samples."""
    n = min(
        int(selected_idx.reshape(-1).shape[0]),
        int(valid_action_mask.shape[0]),
    )
    if n <= 0 or float(reward_weight) == 0.0:
        return torch.zeros((max(n, 0),), device=edge_tt.device, dtype=edge_tt.dtype)
    valid_vehicle = valid_action_mask[:n].to(device=edge_tt.device).any(dim=-1)
    source_key = str(source or "excess_all_tt").strip().lower()
    capacities = wm.bpr.capacities.to(device=edge_tt.device, dtype=edge_tt.dtype)
    v_total = torch.relu(decoded[:, 2].to(device=edge_tt.device, dtype=edge_tt.dtype)) * capacities
    n_veh = v_total.sum().clamp_min(1.0)
    all_tt = torch.sum(v_total * edge_tt.to(device=edge_tt.device, dtype=edge_tt.dtype)) / n_veh
    if source_key in {"all_tt", "mean_tt", "global_tt", "system_tt"}:
        social_cost = all_tt
    elif source_key in {"excess_all_tt", "delay", "mean_delay", "excess_tt"}:
        free_flow = _free_flow_edge_tt_tensor(wm, device=edge_tt.device, dtype=edge_tt.dtype)
        social_cost = torch.sum(v_total * (edge_tt - free_flow).clamp_min(0.0)) / n_veh
    elif source_key in {"total_tt", "system_total_tt"}:
        social_cost = torch.sum(v_total * edge_tt.to(device=edge_tt.device, dtype=edge_tt.dtype))
    elif source_key in {"total_excess_tt", "system_delay", "total_delay"}:
        free_flow = _free_flow_edge_tt_tensor(wm, device=edge_tt.device, dtype=edge_tt.dtype)
        social_cost = torch.sum(v_total * (edge_tt - free_flow).clamp_min(0.0))
    else:
        raise ValueError(
            f"Unsupported dreamer_vehicle_social_reward_source={source!r}; "
            "expected all_tt, excess_all_tt, total_tt, or total_excess_tt"
        )
    distribution_key = str(distribution or "per_vehicle").strip().lower()
    if distribution_key in {"per_vehicle", "broadcast", "each"}:
        denom = 1.0
    elif distribution_key in {"active_av", "active_vehicle", "active_av_mean"}:
        denom = float(max(int(n_active_av), 1))
    elif distribution_key in {"valid_vehicle", "valid_av", "valid_agent"}:
        denom = float(max(int(valid_vehicle.detach().sum().cpu().item()), 1))
    else:
        raise ValueError(
            f"Unsupported dreamer_vehicle_social_reward_distribution={distribution!r}; "
            "expected per_vehicle, active_av, or valid_vehicle"
        )
    reward_scalar = -float(reward_weight) * social_cost * float(reward_scale) / denom
    reward = torch.full((n,), 0.0, device=edge_tt.device, dtype=edge_tt.dtype)
    reward = torch.where(valid_vehicle, reward_scalar.expand_as(reward), reward)
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def _planned_av_pressure_edge_weights_from_latent(
    *,
    wm: UrbanWorldModel,
    z: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
) -> Optional[torch.Tensor]:
    """Build edge weights for planned-pressure costs from the current WM state."""
    mode_key = str(mode or "overlap").strip().lower()
    if mode_key in {"", "overlap", "plain", "none", "sum_overlap", "mean_overlap"}:
        return None
    capacities_t = torch.as_tensor(
        np.asarray(wm._registry.capacities, dtype=np.float32),
        device=device,
        dtype=dtype,
    )
    edge_dyn_rc, v_e_rc, _, t_e_rc = edge_tt_from_latent_state(
        wm=wm,
        z=z.detach(),
        bpr_alpha=cfg.bpr_alpha,
        bpr_beta=cfg.bpr_beta,
        edge_cost_source=cfg.wm_edge_cost_source,
        decoded_speed_min_ratio=cfg.wm_decoded_speed_min_ratio,
        decoded_speed_max_ratio=cfg.wm_decoded_speed_max_ratio,
        decoded_speed_excess_scale=cfg.wm_decoded_speed_excess_scale,
    )
    del edge_dyn_rc
    return _planned_av_pressure_edge_weights(
        wm=wm,
        edge_tt=torch.from_numpy(np.asarray(t_e_rc, dtype=np.float32)).to(device=device, dtype=dtype),
        v_total=torch.from_numpy(np.asarray(v_e_rc, dtype=np.float32)).to(device=device, dtype=dtype),
        capacities=capacities_t,
        mode=mode_key,
    )


def _route_credit_cost_matrix(
    *,
    base_route_costs: torch.Tensor,
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    route_valid: torch.Tensor,
    wm: UrbanWorldModel,
    z: torch.Tensor,
    cfg: TrainConfig,
    n_edges: int,
    n_routes: int,
    device: torch.device,
    mode: str,
    pressure_weight: float,
    pressure_mode: str,
    pressure_reference: str,
    pressure_scale: str,
    normalize_by_path_length: bool,
    include_base_cost: bool = True,
    detach_pressure: bool = True,
) -> torch.Tensor:
    """Build the route-credit cost matrix used by per-vehicle policy gradients.

    ``route_cost`` preserves the legacy local candidate-route advantage.  The
    pressure modes turn the cost into a centralized externality proxy by
    scoring each candidate route against the current fleet's planned edge load.
    """
    mode_key = str(mode or "route_cost").strip().lower()
    if mode_key in {"", "route_cost", "base", "local", "legacy"}:
        return base_route_costs

    pressure_weight = float(pressure_weight)
    if pressure_weight == 0.0:
        return base_route_costs

    edge_weights = _planned_av_pressure_edge_weights_from_latent(
        wm=wm,
        z=z,
        cfg=cfg,
        device=device,
        dtype=base_route_costs.dtype,
        mode=pressure_mode,
    )
    if mode_key in {"planned_pressure", "planned_av_pressure"}:
        credit_costs = _planned_av_pressure_cost_matrix(
            route_probs=route_probs,
            path_batch=path_batch,
            route_valid=route_valid,
            n_edges=n_edges,
            n_routes=n_routes,
            edge_weights=edge_weights,
            normalize_by_path_length=bool(normalize_by_path_length),
            pressure_scale=str(pressure_scale),
            detach_pressure=bool(detach_pressure),
        )
    elif mode_key in {"marginal_pressure", "marginal_av_pressure", "externality"}:
        reference_key = str(pressure_reference or "soft").strip().lower()
        if reference_key in {"sequential_argmax", "sequential", "seq_argmax"}:
            credit_costs = _sequential_marginal_av_pressure_cost_matrix(
                route_probs=route_probs,
                path_batch=path_batch,
                route_valid=route_valid,
                n_edges=n_edges,
                n_routes=n_routes,
                edge_weights=edge_weights,
                normalize_by_path_length=bool(normalize_by_path_length),
            )
        else:
            credit_costs = _marginal_av_pressure_cost_matrix(
                route_probs=route_probs,
                path_batch=path_batch,
                route_valid=route_valid,
                n_edges=n_edges,
                n_routes=n_routes,
                edge_weights=edge_weights,
                pressure_route_probs=_marginal_pressure_reference_probs(
                    route_probs,
                    route_valid,
                    reference=reference_key,
                ),
                normalize_by_path_length=bool(normalize_by_path_length),
                detach_pressure=bool(detach_pressure),
            )
    elif mode_key in {"sequential_marginal", "sequential_marginal_pressure"}:
        credit_costs = _sequential_marginal_av_pressure_cost_matrix(
            route_probs=route_probs,
            path_batch=path_batch,
            route_valid=route_valid,
            n_edges=n_edges,
            n_routes=n_routes,
            edge_weights=edge_weights,
            normalize_by_path_length=bool(normalize_by_path_length),
        )
    elif mode_key in {
        "counterfactual_pressure",
        "counterfactual_fleet_pressure",
        "coma_pressure",
        "rollout_pressure",
    }:
        _, _, _, t_e_cf = edge_tt_from_latent_state(
            wm=wm,
            z=z,
            bpr_alpha=cfg.bpr_alpha,
            bpr_beta=cfg.bpr_beta,
            edge_cost_source=cfg.wm_edge_cost_source,
            decoded_speed_min_ratio=cfg.wm_decoded_speed_min_ratio,
            decoded_speed_max_ratio=cfg.wm_decoded_speed_max_ratio,
            decoded_speed_excess_scale=cfg.wm_decoded_speed_excess_scale,
        )
        edge_tt_cf = torch.from_numpy(np.asarray(t_e_cf, dtype=np.float32)).to(
            device=device,
            dtype=base_route_costs.dtype,
        )
        credit_costs = _counterfactual_fleet_pressure_cost_matrix(
            route_probs=route_probs,
            path_batch=path_batch,
            route_valid=route_valid,
            edge_tt=edge_tt_cf,
            n_edges=n_edges,
            n_routes=n_routes,
            edge_weights=edge_weights,
            normalize_by_path_length=bool(normalize_by_path_length),
            detach_pressure=bool(detach_pressure),
        )
    else:
        raise ValueError(
            "Unsupported route credit mode "
            f"{mode!r}; expected route_cost, planned_pressure, marginal_pressure, "
            "sequential_marginal, or counterfactual_pressure"
        )

    if credit_costs is None:
        return base_route_costs
    pressure_costs = pressure_weight * credit_costs.to(
        device=base_route_costs.device,
        dtype=base_route_costs.dtype,
    )
    if not bool(include_base_cost):
        return pressure_costs
    return base_route_costs + pressure_costs


def imagined_episode_dynamic_dreamer(
    *,
    wm: UrbanWorldModel,
    actor: VehicleRouteActor,
    dataset: OfflineInitDataset,
    route_registry: PathTopology,
    cfg: TrainConfig,
    rng: np.random.Generator,
    device: torch.device,
    departure_schedule: Optional[Dict] = None,
    avg_route_steps: int = 7,
    avg_edge_time: float = 8.0,
    path_cache: Optional[Union["VehiclePathCache", "LazyVehiclePathCache"]] = None,
    traj_id: Optional[int] = None,
    start_step_candidates: Optional[List[int]] = None,
    fixed_start_step: Optional[int] = None,
    rollout_policy_override: str = "actor",
    horizon_proxy: Optional[ActionHorizonProxyReward] = None,
    hdv_background_edge: Optional[torch.Tensor] = None,
    episode_index: Optional[int] = None,
) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
    T = min(int(cfg.imagination_horizon), int(cfg.dreamer_actor_horizon))
    episode_steps = int(cfg.episode_steps)
    K = int(cfg.n_routes)
    n_edges = int(wm.n_edges)
    step_interval = float(cfg.step_interval)

    if departure_schedule is None:
        departure_schedule = {}

    if fixed_start_step is not None:
        s = int(fixed_start_step)
    elif start_step_candidates:
        s = int(start_step_candidates[int(rng.integers(0, len(start_step_candidates)))])
    else:
        s = int(rng.integers(0, max(1, episode_steps - T)))
    flow_0 = dataset.sample_at_step(rng, s)
    z = wm.encode_observation(flow_0.numpy()).to(device=device, dtype=torch.float32)

    active_avs, active_hdvs = rebuild_active_set(departure_schedule, s, avg_route_steps)
    adj_list = PathTopology._build_adj_list(wm._registry.adjacency)
    capacities_np = np.asarray(wm._registry.capacities, dtype=np.float32)
    candidate_edge_cost_source = resolve_wm_candidate_edge_cost_source(cfg)

    use_od_suffix_candidates = (
        str(getattr(cfg, "vehicle_candidate_source", "current_to_dest")).strip().lower() == "od_suffix"
    )

    if path_cache is None and not use_od_suffix_candidates:
        _, _, _, t_e_ref = edge_tt_from_latent_state(
            wm=wm, z=z.detach(), bpr_alpha=cfg.bpr_alpha, bpr_beta=cfg.bpr_beta,
            edge_cost_source=candidate_edge_cost_source,
            decoded_speed_min_ratio=cfg.wm_decoded_speed_min_ratio,
            decoded_speed_max_ratio=cfg.wm_decoded_speed_max_ratio,
            decoded_speed_excess_scale=cfg.wm_decoded_speed_excess_scale,
        )
        dest_edges_unique = list({
            int(v.get("dest_edge", 0))
            for dep in departure_schedule.values()
            for v in dep.get("av_vehicles", [])
        })
        if dest_edges_unique:
            path_cache = VehiclePathCache(
                adj_list=adj_list,
                edge_tt_ref=t_e_ref,
                dest_edges_unique=dest_edges_unique,
                n_edges=n_edges,
                K=K,
                penalty_factor=cfg.vehicle_path_penalty_factor,
            )

    latent_history: List[torch.Tensor] = [z]
    z_states: List[torch.Tensor] = [z]
    rewards: List[torch.Tensor] = []
    base_rewards: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    action_edges: List[torch.Tensor] = []
    rollout_logprob_steps: List[torch.Tensor] = []
    rollout_selected_nonzero_fracs: List[float] = []
    rollout_cost_steps: List[torch.Tensor] = []
    grpo_sample_obs_steps: List[torch.Tensor] = []
    grpo_sample_selected_steps: List[torch.Tensor] = []
    grpo_sample_old_logprob_steps: List[torch.Tensor] = []
    grpo_sample_valid_steps: List[torch.Tensor] = []
    grpo_sample_route_adv_steps: List[torch.Tensor] = []
    rollout_vehicle_cost_losses: List[torch.Tensor] = []
    rollout_vehicle_cost_values: List[float] = []
    rollout_vehicle_cost_adv_means: List[float] = []
    rollout_vehicle_cost_adv_stds: List[float] = []
    dreamer_ac_obs_steps: List[torch.Tensor] = []
    dreamer_ac_selected_steps: List[torch.Tensor] = []
    dreamer_ac_valid_steps: List[torch.Tensor] = []
    dreamer_ac_step_index_steps: List[torch.Tensor] = []
    dreamer_ac_route_adv_steps: List[torch.Tensor] = []
    dreamer_ac_old_logprob_steps: List[torch.Tensor] = []
    dreamer_ac_sample_weight_steps: List[torch.Tensor] = []
    dreamer_ac_vehicle_id_steps: List[torch.Tensor] = []
    dreamer_ac_vehicle_reward_steps: List[torch.Tensor] = []
    dreamer_ac_vehicle_route_progress_reward_steps: List[torch.Tensor] = []
    dreamer_ac_vehicle_route_credit_reward_steps: List[torch.Tensor] = []
    dreamer_ac_vehicle_social_reward_steps: List[torch.Tensor] = []
    vehicle_route_credit_gate_fracs: List[float] = []
    vehicle_route_credit_selected_gate_fracs: List[float] = []
    dreamer_ac_selected_nonzero_fracs: List[float] = []
    critic_contexts: List[torch.Tensor] = []
    rollout_reward_stats_steps: List[Dict[str, float]] = []
    prev_route_probs: Optional[torch.Tensor] = None
    invalid_rates: List[float] = []
    route_churns: List[float] = []
    route_cost_aux_losses: List[torch.Tensor] = []
    route_cost_aux_accs: List[float] = []
    route_cost_aux_margins: List[float] = []
    expected_cost_losses: List[torch.Tensor] = []
    expected_cost_raw_values: List[float] = []
    expected_cost_best_probs: List[float] = []
    expected_cost_best_route0_fracs: List[float] = []
    expected_cost_nonzero_best_fracs: List[float] = []
    expected_cost_margins: List[float] = []
    grpo_route_cost_losses: List[torch.Tensor] = []
    grpo_route_cost_raw_values: List[float] = []
    grpo_route_cost_best_probs: List[float] = []
    grpo_route_cost_best_route0_fracs: List[float] = []
    grpo_route_cost_nonzero_best_fracs: List[float] = []
    grpo_route_cost_margins: List[float] = []
    grpo_route_cost_adv_means: List[float] = []
    grpo_route_cost_adv_stds: List[float] = []
    active_counts: List[int] = []
    max_active_av = 0
    valid_path_counts: List[int] = []
    valid_path_agent_counts: List[int] = []
    actor_step_flags: List[int] = []
    reward_grad_steps = 0
    entropy_grad_steps = 0
    sum_t_all = 0.0
    sum_t_av = 0.0
    route_probs_for_reward: Optional[torch.Tensor]
    path_batch_for_reward: Optional[List[List[List[int]]]]
    rollout_policy = str(rollout_policy_override or "actor").strip().lower()
    use_discrete_actor_critic = bool(dreamer_pg.uses_discrete_actor_critic(cfg) and rollout_policy == "actor")
    ac_rollout_action_mode = str(
        getattr(cfg, "dreamer_ac_rollout_action_mode", "sample") or "sample"
    ).strip().lower()
    if ac_rollout_action_mode in {"stochastic", "sampled", "sampling"}:
        ac_rollout_action_mode = "sample"
    elif ac_rollout_action_mode in {"deterministic", "greedy", "eval", "auto"}:
        ac_rollout_action_mode = "argmax"
    elif ac_rollout_action_mode in {"mixed", "scheduled", "scheduled_argmax", "annealed_argmax"}:
        ac_rollout_action_mode = "scheduled_argmax"
    if ac_rollout_action_mode not in {"sample", "argmax", "scheduled_argmax"}:
        raise ValueError(
            "Unsupported dreamer_ac_rollout_action_mode="
            f"{getattr(cfg, 'dreamer_ac_rollout_action_mode', None)!r}; "
            "expected sample, argmax, or scheduled_argmax"
        )
    ac_rollout_effective_mode = ac_rollout_action_mode
    ac_rollout_argmax_mix_prob = 0.0
    if ac_rollout_action_mode == "scheduled_argmax":
        ep_for_schedule = max(int(episode_index) if episode_index is not None else 0, 0)
        warmup = max(int(getattr(cfg, "dreamer_ac_rollout_argmax_warmup_episodes", 0)), 0)
        ramp = max(int(getattr(cfg, "dreamer_ac_rollout_argmax_ramp_episodes", 0)), 0)
        final_mix = min(max(float(getattr(cfg, "dreamer_ac_rollout_argmax_final_mix", 1.0)), 0.0), 1.0)
        if ep_for_schedule < warmup:
            ac_rollout_argmax_mix_prob = 0.0
        elif ramp <= 0:
            ac_rollout_argmax_mix_prob = final_mix
        else:
            progress = min(max(float(ep_for_schedule - warmup + 1) / float(ramp), 0.0), 1.0)
            ac_rollout_argmax_mix_prob = final_mix * progress
        ac_rollout_effective_mode = "mixed_argmax" if ac_rollout_argmax_mix_prob > 0.0 else "sample"
    elif ac_rollout_action_mode == "argmax":
        ac_rollout_argmax_mix_prob = 1.0
    ac_rollout_argmax_uses: List[float] = []
    route_adv_route_cost_matrix_needed = (
        use_discrete_actor_critic
        and float(getattr(cfg, "dreamer_ac_route_advantage_weight", 0.0)) != 0.0
    )
    route_reward_scope = str(getattr(cfg, "dreamer_route_cost_reward_scope", "both") or "both").strip().lower()
    route_cost_reaches_global = route_reward_scope in {"both", "global", "global_critic", "step"}
    route_cost_reaches_vehicle = route_reward_scope in {"both", "vehicle", "vehicle_critic", "per_vehicle"}
    vehicle_critic_reward_needed = (
        use_discrete_actor_critic
        and bool(getattr(cfg, "dreamer_vehicle_critic", True))
        and _vehicle_critic_reward_enabled(cfg)
    )
    direct_reward_requires_dynamics_grad = any(
        float(getattr(cfg, name, 0.0)) != 0.0
        for name in (
            "dreamer_all_reward_weight",
            "dreamer_remaining_time_progress_weight",
            "dreamer_current_edge_reward_weight",
            "dreamer_cav_reward_weight",
            "dreamer_pred_t_cav_proxy_reward_weight",
            "dreamer_action_congestion_weight",
            "dreamer_route_balance_weight",
            "dreamer_route_max_share_weight",
            "dreamer_horizon_proxy_reward_weight",
        )
    ) or (
        route_cost_reaches_global
        and float(getattr(cfg, "dreamer_route_cost_reward_weight", 0.0)) != 0.0
    ) or rollout_rewards.rollout_reward_requires_dynamics_grad(cfg)
    if use_discrete_actor_critic:
        direct_reward_requires_dynamics_grad = False

    n_total_av = max(sum(d.get("n_av", 0) for d in departure_schedule.values()), 1)
    n_total_hdv = max(sum(d.get("n_hdv", 0) for d in departure_schedule.values()), 1)

    for step in range(s, s + T):
        dep = departure_schedule.get(step, {
            "depart_edge": np.zeros(n_edges, dtype=np.float32),
            "av_vehicles": [],
            "hdv_vehicles": [],
            "n_av": 0,
            "n_hdv": 0,
        })
        for veh in dep.get("av_vehicles", []):
            av_new = {**veh, "depart_step": step}
            if route_registry is not None:
                od_idx_new = int(av_new.get("od_idx", -1))
                if 0 <= od_idx_new < int(route_registry.n_od):
                    od_routes_new = route_registry.route_edge_indices[od_idx_new]
                    if od_routes_new:
                        av_new["current_route"] = list(od_routes_new[0])
            active_avs.append(av_new)
        for veh in dep.get("hdv_vehicles", []):
            active_hdvs.append({**veh, "depart_step": step})

        max_life = int(avg_route_steps * 1.5)
        active_avs = [
            av for av in active_avs
            if (step - av["depart_step"]) < max_life and not bool(av.get("arrived", False))
        ]
        active_hdvs = [hdv for hdv in active_hdvs if (step - hdv["depart_step"]) < max_life]
        _ensure_active_vehicle_current_edges(
            active_avs=active_avs,
            current_step=step,
            avg_edge_time=avg_edge_time,
            step_interval=step_interval,
            route_registry=route_registry,
        )

        n_active_av = len(active_avs)
        n_active_hdv = len(active_hdvs)
        active_counts.append(n_active_av)
        max_active_av = max(max_active_av, n_active_av)
        depart_np = dep.get("depart_edge", np.zeros(n_edges, dtype=np.float32))
        selected_log_probs = None
        selected_route_idx_t_for_update = None
        selected_route_idx_for_env = None
        route_adv_for_ac = None
        vehicle_reward_for_ac = None
        vehicle_route_progress_reward_for_ac = None
        vehicle_route_credit_reward_for_ac = None
        vehicle_social_reward_for_ac = None
        valid_action_mask_for_update = None
        sampled_obs_t = None

        if n_active_av > 0:
            edge_feats_obs = wm.latent_edge_features(z).detach().cpu().numpy().astype(np.float32)
            _, _, _, t_e_curr = edge_tt_from_latent_state(
                wm=wm,
                z=z.detach(),
                bpr_alpha=cfg.bpr_alpha,
                bpr_beta=cfg.bpr_beta,
                edge_cost_source=candidate_edge_cost_source,
                decoded_speed_min_ratio=cfg.wm_decoded_speed_min_ratio,
                decoded_speed_max_ratio=cfg.wm_decoded_speed_max_ratio,
                decoded_speed_excess_scale=cfg.wm_decoded_speed_excess_scale,
            )

            current_edges_np = np.asarray(
                [int(av.get("current_edge", av["origin_edge"])) for av in active_avs],
                dtype=np.int64,
            )
            dest_edges_np = np.asarray([int(av["dest_edge"]) for av in active_avs], dtype=np.int64)
            elapsed_time_np = np.asarray(
                [
                    max(0.0, float(step - int(av.get("depart_step", step)))) * float(step_interval)
                    for av in active_avs
                ],
                dtype=np.float32,
            )
            done_np = np.zeros((n_active_av,), dtype=bool)
            if str(getattr(cfg, "vehicle_candidate_source", "current_to_dest")).strip().lower() == "od_suffix":
                path_batch = _od_suffix_path_batch(
                    route_registry=route_registry,
                    active_avs=active_avs,
                    current_edges=current_edges_np,
                    done=done_np,
                    K=K,
                )
            elif path_cache is not None:
                if isinstance(path_cache, LazyVehiclePathCache):
                    assert t_e_curr is not None
                    path_batch = path_cache.lookup_batch(current_edges_np, dest_edges_np, done_np, t_e_curr)
                else:
                    path_batch = path_cache.lookup_batch(current_edges_np, dest_edges_np, done_np)
            else:
                assert t_e_curr is not None
                path_batch = []
                for i in range(n_active_av):
                    path_batch.append(
                        build_diverse_paths_current_to_dest(
                            adj_list=adj_list,
                            edge_tt=t_e_curr,
                            src_idx=int(current_edges_np[i]),
                            dst_idx=int(dest_edges_np[i]),
                            K=K,
                            penalty_factor=cfg.vehicle_path_penalty_factor,
                        )
                    )
            need_route_cost_matrix = (
                float(getattr(cfg, "dreamer_route_cost_aux_weight", 0.0)) != 0.0
                or float(getattr(cfg, "dreamer_expected_cost_weight", 0.0)) != 0.0
                or float(getattr(cfg, "dreamer_grpo_route_cost_weight", 0.0)) != 0.0
                or route_adv_route_cost_matrix_needed
                or rollout_policy in {"route_cost_greedy", "route_cost_pressure_greedy"}
            )
            if (
                need_route_cost_matrix
                and (t_e_curr is None or candidate_edge_cost_source != str(cfg.wm_edge_cost_source).strip().lower())
            ):
                _, _, _, t_e_route_cost = edge_tt_from_latent_state(
                    wm=wm,
                    z=z.detach(),
                    bpr_alpha=cfg.bpr_alpha,
                    bpr_beta=cfg.bpr_beta,
                    edge_cost_source=cfg.wm_edge_cost_source,
                    decoded_speed_min_ratio=cfg.wm_decoded_speed_min_ratio,
                    decoded_speed_max_ratio=cfg.wm_decoded_speed_max_ratio,
                    decoded_speed_excess_scale=cfg.wm_decoded_speed_excess_scale,
                )
            else:
                t_e_route_cost = t_e_curr

            valid_path_count = 0
            valid_path_agents = 0
            for paths in path_batch:
                k_eff = min(len(paths), K)
                valid_here = sum(1 for k in range(k_eff) if len(paths[k]) > 0)
                valid_path_count += valid_here
                if valid_here > 0:
                    valid_path_agents += 1
            valid_path_counts.append(valid_path_count)
            valid_path_agent_counts.append(valid_path_agents)

            obs_np = build_vehicle_obs_vectorized(
                edge_feats_dyn=edge_feats_obs,
                path_batch=path_batch,
                include_global=bool(cfg.vehicle_obs_include_global),
                route_stats=str(cfg.vehicle_obs_route_stats),
                edge_cost=t_e_curr,
                include_route_costs=bool(cfg.vehicle_obs_include_route_costs),
                include_fleet_pressure=False,
            )
            first_depart = active_avs[0]["depart_step"] if active_avs else step
            context_features = np.array([
                step / max(episode_steps, 1),
                n_active_av / n_total_av,
                (step - first_depart) / max(avg_route_steps, 1),
            ], dtype=np.float32)
            ctx_tiled = np.tile(context_features, (n_active_av, 1))
            use_fleet_pressure_obs = bool(getattr(cfg, "vehicle_obs_include_fleet_pressure", False))
            if use_fleet_pressure_obs:
                fleet_counts_np = np.zeros((K,), dtype=np.float32)
                fleet_pressure_np = np.zeros((n_edges,), dtype=np.float32)
                logits_parts: List[torch.Tensor] = []
                route_prob_parts: List[torch.Tensor] = []
                action_prob_parts: List[torch.Tensor] = []
                sampled_idx_parts: List[torch.Tensor] = []
                sampled_logprob_parts: List[torch.Tensor] = []
                sampled_obs_parts: List[torch.Tensor] = []
                sampled_valid_parts: List[torch.Tensor] = []
                selected_indices_for_obs: List[int] = []
                for i_agent in range(n_active_av):
                    fleet_features_i = build_vehicle_fleet_pressure_obs_features(
                        path_batch=[path_batch[i_agent]],
                        n_edges=n_edges,
                        n_routes=K,
                        selected_route_counts=fleet_counts_np,
                        selected_edge_pressure=fleet_pressure_np,
                    )
                    obs_i_np = np.concatenate(
                        [obs_np[i_agent:i_agent + 1], fleet_features_i, ctx_tiled[i_agent:i_agent + 1]],
                        axis=1,
                    )
                    obs_i = torch.from_numpy(obs_i_np).to(device=device, dtype=torch.float32)
                    logits_i = actor.forward_logits(obs_i)
                    logits_parts.append(logits_i)
                    route_prob_i = torch.softmax(logits_i, dim=-1)
                    route_prob_parts.append(route_prob_i)
                    valid_i = _valid_route_mask_from_paths([path_batch[i_agent]], n_routes=K, device=device)
                    sampled_obs_parts.append(obs_i.squeeze(0))
                    sampled_valid_parts.append(valid_i.squeeze(0))
                    if rollout_policy in {"route_cost_greedy", "route_cost_pressure_greedy"}:
                        # Baseline policies are computed below in the shared branch.
                        action_i = route_prob_i
                        selected_i_t = torch.argmax(action_i.detach(), dim=-1)
                        logprob_i = torch.zeros((1,), device=device, dtype=logits_i.dtype)
                    elif (
                        (
                            float(getattr(cfg, "dreamer_grpo_route_cost_weight", 0.0)) != 0.0
                            or
                            float(getattr(cfg, "dreamer_grpo_rollout_weight", 0.0)) != 0.0
                            or float(getattr(cfg, "dreamer_grpo_rollout_vehicle_cost_weight", 0.0)) != 0.0
                            or use_discrete_actor_critic
                        )
                        and rollout_policy == "actor"
                    ):
                        action_i, selected_i_t, logprob_i, used_argmax_i = _sample_masked_route_actions(
                            logits_i,
                            valid_i,
                            temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                            mode=ac_rollout_effective_mode,
                            argmax_mix_prob=ac_rollout_argmax_mix_prob,
                        )
                        if bool(valid_i.any().detach().cpu().item()):
                            ac_rollout_argmax_uses.append(float(used_argmax_i.float().mean().detach().cpu().item()))
                    else:
                        action_i = _dreamer_action_route_probs(
                            logits_i,
                            mode=str(getattr(cfg, "dreamer_action_mode", "expected")),
                            temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                        )
                        selected_i_t = torch.argmax(action_i.detach(), dim=-1)
                        logprob_i = torch.zeros((1,), device=device, dtype=logits_i.dtype)
                    action_prob_parts.append(action_i)
                    sampled_idx_parts.append(selected_i_t)
                    sampled_logprob_parts.append(logprob_i)
                    selected_i = int(selected_i_t.detach().cpu().item())
                    selected_indices_for_obs.append(selected_i)
                    if 0 <= selected_i < K:
                        fleet_counts_np[selected_i] += 1.0
                        if i_agent < len(path_batch) and selected_i < len(path_batch[i_agent]):
                            for edge_idx in path_batch[i_agent][selected_i]:
                                e = int(edge_idx)
                                if 0 <= e < n_edges:
                                    fleet_pressure_np[e] += 1.0
                obs = None
                logits = torch.cat(logits_parts, dim=0) if logits_parts else torch.zeros((0, K), device=device)
                route_probs = torch.cat(route_prob_parts, dim=0) if route_prob_parts else torch.zeros((0, K), device=device)
                sequential_action_route_probs = torch.cat(action_prob_parts, dim=0) if action_prob_parts else torch.zeros((0, K), device=device)
                sequential_selected_route_idx_t = torch.cat(sampled_idx_parts, dim=0) if sampled_idx_parts else torch.zeros((0,), device=device, dtype=torch.long)
                sequential_selected_log_probs = torch.cat(sampled_logprob_parts, dim=0) if sampled_logprob_parts else torch.zeros((0,), device=device)
                sampled_obs_t = torch.stack(sampled_obs_parts, dim=0) if sampled_obs_parts else torch.zeros((0, obs_np.shape[1] + 0), device=device)
                sampled_valid_t = torch.stack(sampled_valid_parts, dim=0) if sampled_valid_parts else torch.zeros((0, K), device=device, dtype=torch.bool)
            else:
                obs_np = np.concatenate([obs_np, ctx_tiled], axis=1)
                obs = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)

                logits = actor.forward_logits(obs)
                route_probs = torch.softmax(logits, dim=-1)
                sequential_action_route_probs = None
                sequential_selected_route_idx_t = None
                sequential_selected_log_probs = None
                sampled_obs_t = obs.detach()
                sampled_valid_t = _valid_route_mask_from_paths(path_batch, n_routes=K, device=device)
            use_rollout_grpo = (
                (
                    float(getattr(cfg, "dreamer_grpo_rollout_weight", 0.0)) != 0.0
                    or float(getattr(cfg, "dreamer_grpo_rollout_vehicle_cost_weight", 0.0)) != 0.0
                )
                and rollout_policy == "actor"
            )
            use_grpo_policy_gradient = (
                (
                    float(getattr(cfg, "dreamer_grpo_route_cost_weight", 0.0)) != 0.0
                    or use_rollout_grpo
                    or use_discrete_actor_critic
                )
                and rollout_policy == "actor"
            )
            if sequential_action_route_probs is not None and rollout_policy == "actor":
                action_route_probs = sequential_action_route_probs
                if sequential_selected_route_idx_t is not None:
                    selected_route_idx_t_for_update = sequential_selected_route_idx_t
                    selected_route_idx_for_env = sequential_selected_route_idx_t.detach().cpu().numpy().astype(np.int64)
                if sequential_selected_log_probs is not None:
                    valid_action_mask = _valid_route_mask_from_paths(path_batch, n_routes=K, device=device)
                    valid_action_mask_for_update = valid_action_mask
                    valid_selected = valid_action_mask.any(dim=-1)
                    selected_log_probs = sequential_selected_log_probs
                    if use_rollout_grpo and bool(valid_selected.any().detach().cpu().item()):
                        rollout_logprob_steps.append(selected_log_probs[valid_selected].mean())
                        rollout_selected_nonzero_fracs.append(
                            float(
                                (sequential_selected_route_idx_t[valid_selected] != 0)
                                .float()
                                .mean()
                                .detach()
                                .cpu()
                                .item()
                            )
                        )
            elif rollout_policy in {"route_cost_greedy", "route_cost_pressure_greedy"}:
                if t_e_route_cost is None:
                    t_e_route_cost = t_e_curr
                edge_cost_t = torch.from_numpy(np.asarray(t_e_route_cost, dtype=np.float32)).to(
                    device=device,
                    dtype=logits.dtype,
                )
                route_costs_for_action, route_valid_for_action = _route_cost_matrix_from_paths(
                    path_batch=path_batch,
                    edge_cost=edge_cost_t,
                    n_routes=K,
                    invalid_route_cost=edge_cost_t.mean()
                    * float(max(avg_route_steps, 1))
                    * float(cfg.vehicle_path_penalty_factor),
                )
                if route_costs_for_action is None or route_valid_for_action is None:
                    action_route_probs = _dreamer_action_route_probs(
                        logits,
                        mode=str(getattr(cfg, "dreamer_action_mode", "expected")),
                        temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                    )
                elif rollout_policy == "route_cost_pressure_greedy":
                    action_route_probs = _sequential_planned_pressure_greedy_action_probs(
                        route_costs=route_costs_for_action,
                        route_valid=route_valid_for_action,
                        path_batch=path_batch,
                        n_edges=n_edges,
                        n_routes=K,
                        pressure_weight=float(
                            getattr(cfg, "dreamer_grpo_rollout_baseline_planned_av_pressure_weight", 0.0)
                        ),
                    )
                    selected_route_idx_for_env = torch.argmax(action_route_probs.detach(), dim=-1).cpu().numpy().astype(np.int64)
                else:
                    large_cost = torch.finfo(route_costs_for_action.dtype).max / 4.0
                    masked_costs = route_costs_for_action.masked_fill(~route_valid_for_action, large_cost)
                    selected_route_idx_t = torch.argmin(masked_costs, dim=-1)
                    selected_route_idx_t_for_update = selected_route_idx_t
                    selected_route_idx_for_env = selected_route_idx_t.detach().cpu().numpy().astype(np.int64)
                    action_route_probs = torch.zeros_like(logits)
                    action_route_probs.scatter_(1, selected_route_idx_t.unsqueeze(-1), 1.0)
            elif use_grpo_policy_gradient:
                valid_action_mask = _valid_route_mask_from_paths(path_batch, n_routes=K, device=device)
                valid_action_mask_for_update = valid_action_mask
                action_route_probs, selected_route_idx_t, selected_log_probs, used_argmax_t = _sample_masked_route_actions(
                    logits,
                    valid_action_mask,
                    temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                    mode=ac_rollout_effective_mode,
                    argmax_mix_prob=ac_rollout_argmax_mix_prob,
                )
                selected_route_idx_t_for_update = selected_route_idx_t
                selected_route_idx_for_env = selected_route_idx_t.detach().cpu().numpy().astype(np.int64)
                valid_selected = valid_action_mask.any(dim=-1)
                if bool(valid_selected.any().detach().cpu().item()):
                    ac_rollout_argmax_uses.append(
                        float(used_argmax_t[valid_selected].float().mean().detach().cpu().item())
                    )
                if use_rollout_grpo and bool(valid_selected.any().detach().cpu().item()):
                    rollout_logprob_steps.append(selected_log_probs[valid_selected].mean())
                    rollout_selected_nonzero_fracs.append(
                        float((selected_route_idx_t[valid_selected] != 0).float().mean().detach().cpu().item())
                    )
            elif sampled_obs_t is not None:
                action_route_probs = _dreamer_action_route_probs(
                    logits,
                    mode=str(getattr(cfg, "dreamer_action_mode", "expected")),
                    temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                )
                valid_action_mask = _valid_route_mask_from_paths(path_batch, n_routes=K, device=device)
                valid_action_mask_for_update = valid_action_mask
                selected_route_idx_t = torch.argmax(action_route_probs.detach(), dim=-1)
                selected_route_idx_t_for_update = selected_route_idx_t
                selected_log_probs = _masked_route_log_probs(
                    logits,
                    valid_action_mask,
                    temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                ).gather(1, selected_route_idx_t.unsqueeze(-1)).squeeze(-1)
                selected_log_probs = selected_log_probs.detach()
            else:
                action_route_probs = _dreamer_action_route_probs(
                    logits,
                    mode=str(getattr(cfg, "dreamer_action_mode", "expected")),
                    temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                )
            route_log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -(route_probs * route_log_probs).sum(dim=-1).mean()
            if need_route_cost_matrix and t_e_route_cost is not None:
                edge_cost_t = torch.from_numpy(np.asarray(t_e_route_cost, dtype=np.float32)).to(
                    device=device,
                    dtype=logits.dtype,
                )
                route_costs, route_valid = _route_cost_matrix_from_paths(
                    path_batch=path_batch,
                    edge_cost=edge_cost_t,
                    n_routes=K,
                    invalid_route_cost=edge_cost_t.mean()
                    * float(max(avg_route_steps, 1))
                    * float(cfg.vehicle_path_penalty_factor),
                )
                if route_costs is not None and route_valid is not None:
                    if float(getattr(cfg, "dreamer_route_cost_aux_weight", 0.0)) != 0.0:
                        aux_loss, aux_stats = _route_cost_policy_aux_loss(
                            logits=logits,
                            route_costs=route_costs,
                            route_valid=route_valid,
                            temperature=float(cfg.dreamer_route_cost_aux_temperature),
                            balance_best_classes=bool(cfg.dreamer_route_cost_aux_balance),
                            hard_target=bool(cfg.dreamer_route_cost_aux_hard_target),
                            nonzero_margin=float(cfg.dreamer_route_cost_aux_nonzero_margin),
                        )
                        if aux_loss is not None:
                            route_cost_aux_losses.append(aux_loss)
                            route_cost_aux_accs.append(float(aux_stats.get("route_cost_aux_acc", 0.0)))
                            route_cost_aux_margins.append(float(aux_stats.get("route_cost_aux_margin", 0.0)))
                    if float(getattr(cfg, "dreamer_expected_cost_weight", 0.0)) != 0.0:
                        expected_route_costs = route_costs
                        pressure_weight = float(getattr(cfg, "dreamer_expected_cost_planned_av_pressure_weight", 0.0))
                        if pressure_weight != 0.0:
                            pressure_mode = str(
                                getattr(cfg, "dreamer_expected_cost_planned_av_pressure_mode", "overlap")
                            ).strip().lower()
                            if pressure_mode not in {"", "overlap", "sum_overlap", "mean_overlap"}:
                                raise ValueError(
                                    "Unsupported dreamer_expected_cost_planned_av_pressure_mode="
                                    f"{pressure_mode!r}"
                                )
                            pressure_costs = _planned_av_pressure_cost_matrix(
                                route_probs=route_probs,
                                path_batch=path_batch,
                                route_valid=route_valid,
                                n_edges=n_edges,
                                n_routes=K,
                                normalize_by_path_length=(pressure_mode == "mean_overlap"),
                                detach_pressure=bool(getattr(cfg, "dreamer_expected_cost_detach_cost", True)),
                            )
                            if pressure_costs is not None:
                                expected_route_costs = route_costs + pressure_weight * pressure_costs.to(
                                    device=route_costs.device,
                                    dtype=route_costs.dtype,
                                )
                        marginal_pressure_weight = float(
                            getattr(cfg, "dreamer_expected_cost_marginal_av_pressure_weight", 0.0)
                        )
                        if marginal_pressure_weight != 0.0:
                            marginal_mode = str(
                                getattr(cfg, "dreamer_expected_cost_marginal_av_pressure_mode", "excess_tt_ratio")
                            )
                            marginal_reference = str(
                                getattr(
                                    cfg,
                                    "dreamer_expected_cost_marginal_av_pressure_reference",
                                    "soft",
                                )
                            ).strip().lower()
                            marginal_edge_weights = _planned_av_pressure_edge_weights_from_latent(
                                wm=wm,
                                z=z,
                                cfg=cfg,
                                device=device,
                                dtype=edge_cost_t.dtype,
                                mode=marginal_mode,
                            )
                            if marginal_reference in {"sequential_argmax", "sequential", "seq_argmax"}:
                                marginal_costs = _sequential_marginal_av_pressure_cost_matrix(
                                    route_probs=route_probs,
                                    path_batch=path_batch,
                                    route_valid=route_valid,
                                    n_edges=n_edges,
                                    n_routes=K,
                                    edge_weights=marginal_edge_weights,
                                    normalize_by_path_length=bool(
                                        getattr(
                                            cfg,
                                            "dreamer_expected_cost_marginal_av_pressure_normalize_by_path_length",
                                            False,
                                        )
                                    ),
                                )
                            else:
                                marginal_costs = _marginal_av_pressure_cost_matrix(
                                    route_probs=route_probs,
                                    path_batch=path_batch,
                                    route_valid=route_valid,
                                    n_edges=n_edges,
                                    n_routes=K,
                                    edge_weights=marginal_edge_weights,
                                    pressure_route_probs=_marginal_pressure_reference_probs(
                                        route_probs,
                                        route_valid,
                                        reference=marginal_reference,
                                    ),
                                    normalize_by_path_length=bool(
                                        getattr(
                                            cfg,
                                            "dreamer_expected_cost_marginal_av_pressure_normalize_by_path_length",
                                            False,
                                        )
                                    ),
                                    detach_pressure=bool(getattr(cfg, "dreamer_expected_cost_detach_cost", True)),
                                )
                            if marginal_costs is not None:
                                expected_route_costs = expected_route_costs + (
                                    marginal_pressure_weight
                                    * marginal_costs.to(
                                        device=expected_route_costs.device,
                                        dtype=expected_route_costs.dtype,
                                    )
                                )
                        share_penalty_weight = float(
                            getattr(cfg, "dreamer_expected_cost_route_share_penalty_weight", 0.0)
                        )
                        if share_penalty_weight != 0.0:
                            share_costs = _route_share_penalty_cost_matrix(
                                route_probs=route_probs,
                                route_valid=route_valid,
                                cap=float(getattr(cfg, "dreamer_expected_cost_route_share_penalty_cap", 1.0)),
                                detach_share=bool(getattr(cfg, "dreamer_expected_cost_detach_cost", True)),
                            )
                            if share_costs is not None:
                                expected_route_costs = expected_route_costs + (
                                    share_penalty_weight
                                    * share_costs.to(
                                        device=expected_route_costs.device,
                                        dtype=expected_route_costs.dtype,
                                    )
                                )
                        expected_loss, expected_stats = _expected_cost_policy_loss(
                            logits=logits,
                            route_costs=expected_route_costs,
                            route_valid=route_valid,
                            temperature=float(cfg.dreamer_expected_cost_temperature),
                            detach_cost=bool(getattr(cfg, "dreamer_expected_cost_detach_cost", True)),
                            balance_best_classes=bool(getattr(cfg, "dreamer_expected_cost_balance_best_classes", False)),
                            pairwise=bool(getattr(cfg, "dreamer_expected_cost_pairwise", False)),
                            pairwise_margin=float(getattr(cfg, "dreamer_expected_cost_pairwise_margin", 0.0)),
                            route0_margin_weight=float(
                                getattr(cfg, "dreamer_expected_cost_route0_margin_weight", 0.0)
                            ),
                            route0_margin=float(
                                getattr(cfg, "dreamer_expected_cost_route0_margin", 0.0)
                            ),
                            route0_margin_min_gap=float(
                                getattr(cfg, "dreamer_expected_cost_route0_margin_min_gap", 0.0)
                            ),
                            nonzero_best_boost=float(
                                getattr(cfg, "dreamer_expected_cost_nonzero_best_boost", 0.0)
                            ),
                            nonzero_best_min_gap=float(
                                getattr(cfg, "dreamer_expected_cost_nonzero_best_min_gap", 0.0)
                            ),
                        )
                        if expected_loss is not None:
                            expected_cost_losses.append(expected_loss)
                            expected_cost_raw_values.append(float(expected_stats.get("expected_cost_raw", 0.0)))
                            expected_cost_best_probs.append(float(expected_stats.get("expected_cost_best_prob", 0.0)))
                            expected_cost_best_route0_fracs.append(float(expected_stats.get("expected_cost_best_route0_frac", 0.0)))
                            expected_cost_nonzero_best_fracs.append(float(expected_stats.get("expected_cost_nonzero_best_frac", 0.0)))
                            expected_cost_margins.append(float(expected_stats.get("expected_cost_margin", 0.0)))
                    if float(getattr(cfg, "dreamer_grpo_route_cost_weight", 0.0)) != 0.0:
                        grpo_route_costs = route_costs
                        pressure_weight = float(getattr(cfg, "dreamer_grpo_planned_av_pressure_weight", 0.0))
                        if pressure_weight != 0.0:
                            pressure_mode = str(getattr(cfg, "dreamer_grpo_planned_av_pressure_mode", "overlap"))
                            pressure_edge_weights = _planned_av_pressure_edge_weights_from_latent(
                                wm=wm,
                                z=z,
                                cfg=cfg,
                                device=device,
                                dtype=edge_cost_t.dtype,
                                mode=pressure_mode,
                            )
                            pressure_costs = _planned_av_pressure_cost_matrix(
                                route_probs=route_probs,
                                path_batch=path_batch,
                                route_valid=route_valid,
                                n_edges=n_edges,
                                n_routes=K,
                                edge_weights=pressure_edge_weights,
                                detach_pressure=True,
                            )
                            if pressure_costs is not None:
                                grpo_route_costs = route_costs + pressure_weight * pressure_costs.to(
                                    device=route_costs.device,
                                    dtype=route_costs.dtype,
                                )
                        marginal_pressure_weight = float(
                            getattr(cfg, "dreamer_grpo_marginal_av_pressure_weight", 0.0)
                        )
                        if marginal_pressure_weight != 0.0:
                            marginal_mode = str(
                                getattr(cfg, "dreamer_grpo_marginal_av_pressure_mode", "excess_tt_ratio")
                            )
                            marginal_reference = str(
                                getattr(cfg, "dreamer_grpo_marginal_av_pressure_reference", "soft")
                            ).strip().lower()
                            marginal_edge_weights = _planned_av_pressure_edge_weights_from_latent(
                                wm=wm,
                                z=z,
                                cfg=cfg,
                                device=device,
                                dtype=edge_cost_t.dtype,
                                mode=marginal_mode,
                            )
                            if marginal_reference in {"sequential_argmax", "sequential", "seq_argmax"}:
                                marginal_costs = _sequential_marginal_av_pressure_cost_matrix(
                                    route_probs=route_probs,
                                    path_batch=path_batch,
                                    route_valid=route_valid,
                                    n_edges=n_edges,
                                    n_routes=K,
                                    edge_weights=marginal_edge_weights,
                                    normalize_by_path_length=bool(
                                        getattr(
                                            cfg,
                                            "dreamer_grpo_marginal_av_pressure_normalize_by_path_length",
                                            False,
                                        )
                                    ),
                                )
                            else:
                                marginal_costs = _marginal_av_pressure_cost_matrix(
                                    route_probs=route_probs,
                                    path_batch=path_batch,
                                    route_valid=route_valid,
                                    n_edges=n_edges,
                                    n_routes=K,
                                    edge_weights=marginal_edge_weights,
                                    pressure_route_probs=_marginal_pressure_reference_probs(
                                        route_probs,
                                        route_valid,
                                        reference=marginal_reference,
                                    ),
                                    normalize_by_path_length=bool(
                                        getattr(
                                            cfg,
                                            "dreamer_grpo_marginal_av_pressure_normalize_by_path_length",
                                            False,
                                        )
                                    ),
                                    detach_pressure=True,
                                )
                            if marginal_costs is not None:
                                grpo_route_costs = grpo_route_costs + (
                                    marginal_pressure_weight
                                    * marginal_costs.to(
                                        device=grpo_route_costs.device,
                                        dtype=grpo_route_costs.dtype,
                                    )
                                )
                        share_penalty_weight = float(
                            getattr(cfg, "dreamer_grpo_route_share_penalty_weight", 0.0)
                        )
                        if share_penalty_weight != 0.0:
                            share_costs = _route_share_penalty_cost_matrix(
                                route_probs=route_probs,
                                route_valid=route_valid,
                                cap=float(getattr(cfg, "dreamer_grpo_route_share_penalty_cap", 1.0)),
                                detach_share=True,
                            )
                            if share_costs is not None:
                                grpo_route_costs = grpo_route_costs + (
                                    share_penalty_weight
                                    * share_costs.to(device=grpo_route_costs.device, dtype=grpo_route_costs.dtype)
                                )
                        grpo_loss, grpo_stats = _grpo_route_cost_policy_loss(
                            logits=logits,
                            route_costs=grpo_route_costs,
                            route_valid=route_valid,
                            temperature=float(cfg.dreamer_grpo_advantage_temperature),
                            normalize_advantages=bool(getattr(cfg, "dreamer_grpo_normalize_advantages", True)),
                            detach_cost=True,
                            pairwise_weight=float(
                                getattr(cfg, "dreamer_grpo_route_cost_pairwise_weight", 0.0)
                            ),
                            pairwise_margin=float(
                                getattr(cfg, "dreamer_grpo_route_cost_pairwise_margin", 0.0)
                            ),
                        )
                        if grpo_loss is not None:
                            grpo_route_cost_losses.append(grpo_loss)
                            grpo_route_cost_raw_values.append(float(grpo_stats.get("grpo_route_cost_raw", 0.0)))
                            grpo_route_cost_best_probs.append(float(grpo_stats.get("grpo_route_cost_best_prob", 0.0)))
                            grpo_route_cost_best_route0_fracs.append(float(grpo_stats.get("grpo_route_cost_best_route0_frac", 0.0)))
                            grpo_route_cost_nonzero_best_fracs.append(float(grpo_stats.get("grpo_route_cost_nonzero_best_frac", 0.0)))
                            grpo_route_cost_margins.append(float(grpo_stats.get("grpo_route_cost_margin", 0.0)))
                            grpo_route_cost_adv_means.append(float(grpo_stats.get("grpo_route_cost_adv_mean", 0.0)))
                            grpo_route_cost_adv_stds.append(float(grpo_stats.get("grpo_route_cost_adv_std", 0.0)))
                        if (
                            selected_log_probs is not None
                            and selected_route_idx_t_for_update is not None
                            and sampled_obs_t is not None
                            and sampled_valid_t is not None
                        ):
                            n_sample = min(
                                int(sampled_obs_t.shape[0]),
                                int(sampled_valid_t.shape[0]),
                                int(selected_route_idx_t_for_update.reshape(-1).shape[0]),
                                int(selected_log_probs.reshape(-1).shape[0]),
                                int(grpo_route_costs.shape[0]),
                                int(route_valid.shape[0]),
                            )
                            if n_sample > 0:
                                route_adv_t = _selected_route_cost_advantages(
                                    route_costs=grpo_route_costs[:n_sample].detach(),
                                    route_valid=route_valid[:n_sample],
                                    selected_idx=selected_route_idx_t_for_update[:n_sample],
                                    temperature=float(cfg.dreamer_grpo_advantage_temperature),
                                    normalize_advantages=bool(
                                        getattr(cfg, "dreamer_grpo_normalize_advantages", True)
                                    ),
                                )
                                valid_selected_t = sampled_valid_t[:n_sample].any(dim=-1)
                                if bool(valid_selected_t.any().detach().cpu().item()):
                                    grpo_sample_obs_steps.append(sampled_obs_t[:n_sample].detach())
                                    grpo_sample_valid_steps.append(sampled_valid_t[:n_sample].detach())
                                    grpo_sample_selected_steps.append(
                                        selected_route_idx_t_for_update[:n_sample].detach()
                                    )
                                    grpo_sample_old_logprob_steps.append(
                                        selected_log_probs[:n_sample].detach()
                                    )
                                    grpo_sample_route_adv_steps.append(route_adv_t.detach())
                    if (
                        route_adv_route_cost_matrix_needed
                        and selected_route_idx_t_for_update is not None
                    ):
                        ac_base_route_costs = _route_credit_base_cost_matrix(
                            route_tt_costs=route_costs.detach(),
                            route_valid=route_valid,
                            mode=str(getattr(cfg, "dreamer_ac_route_credit_base_cost_mode", "progress_fraction")),
                            step_interval=float(getattr(cfg, "step_interval", 30.0)),
                            elapsed_time=torch.from_numpy(elapsed_time_np).to(
                                device=route_costs.device,
                                dtype=route_costs.dtype,
                            ),
                            remaining_tt_eps=float(
                                getattr(cfg, "dreamer_ac_route_credit_remaining_tt_eps", 1.0)
                            ),
                            progress_cap=float(
                                getattr(cfg, "dreamer_ac_route_credit_progress_cap", 1.0)
                            ),
                        )
                        ac_route_costs = _route_credit_cost_matrix(
                            base_route_costs=ac_base_route_costs.detach(),
                            route_probs=route_probs.detach(),
                            path_batch=path_batch,
                            route_valid=route_valid,
                            wm=wm,
                            z=z.detach(),
                            cfg=cfg,
                            n_edges=n_edges,
                            n_routes=K,
                            device=device,
                            mode=str(getattr(cfg, "dreamer_ac_route_credit_mode", "route_cost")),
                            pressure_weight=float(
                                getattr(cfg, "dreamer_ac_route_credit_pressure_weight", 1.0)
                            ),
                            pressure_mode=str(
                                getattr(cfg, "dreamer_ac_route_credit_pressure_mode", "clipped_excess_tt_ratio")
                            ),
                            pressure_reference=str(
                                getattr(cfg, "dreamer_ac_route_credit_pressure_reference", "soft")
                            ),
                            pressure_scale="mean",
                            normalize_by_path_length=bool(
                                getattr(cfg, "dreamer_ac_route_credit_normalize_by_path_length", False)
                            ),
                            include_base_cost=bool(
                                getattr(cfg, "dreamer_ac_route_credit_include_base_cost", True)
                            ),
                            detach_pressure=True,
                        )
                        route_adv_for_ac = _selected_route_cost_advantages(
                            route_costs=ac_route_costs.detach(),
                            route_valid=route_valid,
                            selected_idx=selected_route_idx_t_for_update,
                            temperature=float(getattr(cfg, "dreamer_ac_route_advantage_temperature", 60.0)),
                            normalize_advantages=bool(
                                getattr(cfg, "dreamer_ac_normalize_route_advantages", True)
                            ),
                        )
            entropies.append(entropy)
            if bool(entropy.requires_grad):
                entropy_grad_steps += 1
            invalid_rates.append(_mean_path_invalid_rate(path_batch, K))
            route_churns.append(_route_prob_churn(prev_route_probs, route_probs))
            prev_route_probs = route_probs.detach()
            action_edge = aggregate_route_probs_to_action_edge(
                action_route_probs,
                path_batch,
                n_edges=n_edges,
                n_routes=K,
                current_edges=torch.from_numpy(current_edges_np).to(device=device),
                edge_travel_times=torch.from_numpy(np.asarray(t_e_curr, dtype=np.float32)).to(device=device),
                horizon_seconds=float(cfg.step_interval),
            )
            with torch.no_grad():
                if selected_route_idx_for_env is None:
                    selected_route_idx = torch.argmax(action_route_probs.detach(), dim=-1).cpu().numpy().astype(np.int64)
                else:
                    selected_route_idx = selected_route_idx_for_env
            for i, av in enumerate(active_avs):
                if i >= len(path_batch):
                    continue
                paths = path_batch[i]
                if not paths:
                    continue
                k_sel = int(np.clip(selected_route_idx[i], 0, max(0, min(len(paths), K) - 1)))
                chosen_path = paths[k_sel]
                if not chosen_path:
                    continue
                av["last_route_idx"] = int(k_sel)
                av["current_route"] = list(chosen_path)
                advance_vehicle_along_path_time_budget(
                    av,
                    current_edge=int(current_edges_np[i]),
                    chosen_path=chosen_path,
                    edge_travel_times=np.asarray(t_e_curr, dtype=np.float32),
                    step_interval=float(cfg.step_interval),
                    n_edges=n_edges,
                )
            route_probs_for_reward = action_route_probs
            path_batch_for_reward = path_batch
            actor_step_flags.append(1 if valid_path_agents > 0 else 0)
        else:
            entropies.append(torch.zeros((), device=device, dtype=z.dtype))
            invalid_rates.append(0.0)
            route_churns.append(0.0)
            action_edge = torch.zeros((n_edges,), device=device, dtype=z.dtype)
            route_probs_for_reward = None
            path_batch_for_reward = None
            valid_path_counts.append(0)
            valid_path_agent_counts.append(0)
            actor_step_flags.append(0)

        context_np = np.array([
            step / max(episode_steps, 1),
            n_active_av / n_total_av,
            n_active_hdv / max(n_total_hdv, 1),
        ], dtype=np.float32)
        context_t = torch.from_numpy(context_np).to(device=device, dtype=z.dtype)
        critic_contexts.append(
            _dreamer_critic_context_vector(
                step=step,
                episode_steps=episode_steps,
                n_active_av=n_active_av,
                n_active_hdv=n_active_hdv,
                n_total_av=n_total_av,
                n_total_hdv=n_total_hdv,
                valid_path_agents=valid_path_agents if n_active_av > 0 else 0,
                valid_path_count=valid_path_count if n_active_av > 0 else 0,
                elapsed_time_np=elapsed_time_np if n_active_av > 0 else None,
                depart_np=depart_np,
                device=device,
                dtype=z.dtype,
            )
        )
        depart_t = torch.from_numpy(depart_np).to(device=device, dtype=z.dtype)
        past_packed = build_history_packed(latent_history, history_length=cfg.history_length)
        hdv_context_for_wm = None
        if bool(getattr(wm, "use_hdv_context", False)) and hdv_background_edge is not None:
            hdv_context_for_wm = hdv_background_edge.to(device=device, dtype=z.dtype).clamp_min(0.0)
            hdv_context_for_wm = hdv_context_for_wm / hdv_context_for_wm.sum().clamp_min(1e-8)
            hdv_context_for_wm = hdv_context_for_wm * float(getattr(cfg, "dreamer_hdv_background_weight", 1.0))
            action_edge_for_wm = action_edge
        else:
            action_edge_for_wm = hdv_response.combine_av_hdv_edge_background(
                action_edge,
                hdv_background_edge,
                n_active_av=n_active_av,
                n_total_hdv=n_total_hdv,
                mode=str(getattr(cfg, "dreamer_hdv_background_combine", "additive")),
                weight=float(getattr(cfg, "dreamer_hdv_background_weight", 1.0)),
            )
        if direct_reward_requires_dynamics_grad:
            z_next, _info = wm.predict_next_state_differentiable(
                z,
                action_edge_for_wm,
                past_packed=past_packed,
                depart=depart_t,
                context=context_t,
                hdv_context=hdv_context_for_wm,
            )
        else:
            # Score-function objectives below use detached rollout costs/log-probs.
            # Avoid retaining a full WM autograd graph when no direct reward term
            # differentiates through imagined dynamics.
            with torch.no_grad():
                z_next, _info = wm.predict_next_state_differentiable(
                    z,
                    action_edge_for_wm,
                    past_packed=past_packed,
                    depart=depart_t,
                    context=context_t,
                    hdv_context=hdv_context_for_wm,
                )
            z_next = z_next.detach()
        reward_t, t_all_step_min, t_cav_step_min = _dreamer_step_reward(
            wm=wm,
            z_next=z_next,
            reward_scale=float(cfg.vehicle_reward_scale),
            z_current=z,
            action_edge=action_edge,
            current_edge_indices=current_edges_np if n_active_av > 0 else None,
            route_probs=route_probs_for_reward,
            route_balance_probs=route_probs if n_active_av > 0 else None,
            path_batch=path_batch_for_reward,
            horizon_proxy=horizon_proxy,
            all_reward_weight=float(cfg.dreamer_all_reward_weight),
            remaining_time_progress_weight=float(cfg.dreamer_remaining_time_progress_weight),
            current_edge_reward_weight=float(cfg.dreamer_current_edge_reward_weight),
            cav_reward_weight=float(cfg.dreamer_cav_reward_weight),
            route_cost_reward_weight=(
                float(cfg.dreamer_route_cost_reward_weight)
                if route_cost_reaches_global
                else 0.0
            ),
            pred_t_cav_proxy_reward_weight=float(cfg.dreamer_pred_t_cav_proxy_reward_weight),
            route_cost_reward_normalize_regret=bool(cfg.dreamer_route_cost_reward_normalize_regret),
            route_cost_reward_temperature=float(cfg.dreamer_route_cost_reward_temperature),
            route_cost_reward_planned_av_pressure_weight=float(
                getattr(cfg, "dreamer_route_cost_reward_planned_av_pressure_weight", 0.0)
            ),
            route_cost_reward_planned_av_pressure_mode=str(
                getattr(cfg, "dreamer_route_cost_reward_planned_av_pressure_mode", "overlap")
            ),
            action_congestion_weight=float(cfg.dreamer_action_congestion_weight),
            route_balance_weight=float(cfg.dreamer_route_balance_weight),
            route_max_share_weight=float(cfg.dreamer_route_max_share_weight),
            route_max_share_cap=float(cfg.dreamer_route_max_share_cap),
            horizon_proxy_reward_weight=float(cfg.dreamer_horizon_proxy_reward_weight),
            remaining_time_softmin_temperature=float(cfg.dreamer_remaining_time_softmin_temperature),
            remaining_time_progress_step_seconds=float(cfg.step_interval),
            cav_route_steps=float(avg_route_steps),
            invalid_route_cost_factor=float(cfg.vehicle_path_penalty_factor),
            edge_cost_source=cfg.wm_edge_cost_source,
            decoded_speed_min_ratio=cfg.wm_decoded_speed_min_ratio,
            decoded_speed_max_ratio=cfg.wm_decoded_speed_max_ratio,
            decoded_speed_excess_scale=cfg.wm_decoded_speed_excess_scale,
        )
        if (
            use_discrete_actor_critic
            and selected_route_idx_t_for_update is not None
            and sampled_obs_t is not None
            and valid_action_mask_for_update is not None
        ):
            if vehicle_critic_reward_needed and path_batch_for_reward is not None:
                with torch.no_grad():
                    edge_dyn_next_for_vehicle_reward = wm.flow_projector.decode(z_next.detach())
                    edge_tt_next_for_vehicle_reward = _edge_tt_tensor_from_decoded(
                        wm=wm,
                        decoded=edge_dyn_next_for_vehicle_reward,
                        z=z_next.detach(),
                        edge_cost_source=cfg.wm_edge_cost_source,
                        decoded_speed_min_ratio=cfg.wm_decoded_speed_min_ratio,
                        decoded_speed_max_ratio=cfg.wm_decoded_speed_max_ratio,
                        decoded_speed_excess_scale=cfg.wm_decoded_speed_excess_scale,
                    )
                    invalid_vehicle_route_cost = (
                        edge_tt_next_for_vehicle_reward.mean()
                        * float(max(avg_route_steps, 1))
                        * float(cfg.vehicle_path_penalty_factor)
                    )
                    vehicle_reward_costs, vehicle_reward_valid = _route_cost_matrix_from_paths(
                        path_batch=path_batch_for_reward,
                        edge_cost=edge_tt_next_for_vehicle_reward.detach(),
                        n_routes=K,
                        invalid_route_cost=invalid_vehicle_route_cost,
                    )
                    if vehicle_reward_costs is not None and vehicle_reward_valid is not None:
                        vehicle_route_tt_costs = vehicle_reward_costs.detach()
                        vehicle_pressure_weight = float(
                            getattr(cfg, "dreamer_route_cost_reward_planned_av_pressure_weight", 0.0)
                        )
                        if vehicle_pressure_weight != 0.0:
                            pressure_edge_weights = _planned_av_pressure_edge_weights(
                                wm=wm,
                                edge_tt=edge_tt_next_for_vehicle_reward,
                                v_total=torch.relu(edge_dyn_next_for_vehicle_reward[:, 2])
                                * wm.bpr.capacities.to(
                                    device=z_next.device,
                                    dtype=z_next.dtype,
                                ),
                                capacities=wm.bpr.capacities.to(
                                    device=z_next.device,
                                    dtype=z_next.dtype,
                                ),
                                mode=str(
                                    getattr(
                                        cfg,
                                        "dreamer_route_cost_reward_planned_av_pressure_mode",
                                        "overlap",
                                    )
                                ),
                            )
                            pressure_costs = _planned_av_pressure_cost_matrix(
                                route_probs=action_route_probs.detach(),
                                path_batch=path_batch_for_reward,
                                route_valid=vehicle_reward_valid,
                                n_edges=n_edges,
                                n_routes=K,
                                edge_weights=pressure_edge_weights,
                                detach_pressure=True,
                            )
                            if pressure_costs is not None:
                                vehicle_reward_costs = vehicle_reward_costs + (
                                    vehicle_pressure_weight
                                    * pressure_costs.to(
                                        device=vehicle_reward_costs.device,
                                        dtype=vehicle_reward_costs.dtype,
                                    )
                                )
                        vehicle_reward_parts: List[torch.Tensor] = []
                        route_reward_weight = (
                            float(getattr(cfg, "dreamer_route_cost_reward_weight", 0.0))
                            if route_cost_reaches_vehicle
                            else 0.0
                        )
                        if route_reward_weight != 0.0:
                            vehicle_route_progress_reward_for_ac = (
                                _selected_route_vehicle_rewards_from_cost_matrix(
                                    route_costs=vehicle_reward_costs,
                                    route_valid=vehicle_reward_valid,
                                    selected_idx=selected_route_idx_t_for_update,
                                    reward_weight=route_reward_weight,
                                    reward_scale=float(getattr(cfg, "vehicle_reward_scale", 1.0)),
                                    reward_mode=str(getattr(cfg, "dreamer_route_cost_reward_mode", "progress_fraction")),
                                    step_interval=float(getattr(cfg, "step_interval", 30.0)),
                                    elapsed_time=torch.from_numpy(elapsed_time_np).to(
                                        device=vehicle_reward_costs.device,
                                        dtype=vehicle_reward_costs.dtype,
                                    ),
                                    progress_cap=float(getattr(cfg, "dreamer_ac_route_credit_progress_cap", 1.0)),
                                    remaining_tt_eps=float(getattr(cfg, "dreamer_ac_route_credit_remaining_tt_eps", 1.0)),
                                    normalize_regret=bool(getattr(cfg, "dreamer_route_cost_reward_normalize_regret", False)),
                                    temperature=float(getattr(cfg, "dreamer_route_cost_reward_temperature", 60.0)),
                                )
                            )
                            vehicle_reward_parts.append(vehicle_route_progress_reward_for_ac)
                        route_credit_reward_weight = float(
                            getattr(cfg, "dreamer_vehicle_route_credit_reward_weight", 0.0)
                        )
                        if route_credit_reward_weight != 0.0:
                            route_credit_base_mode = str(
                                getattr(
                                    cfg,
                                    "dreamer_vehicle_route_credit_base_cost_mode",
                                    getattr(cfg, "dreamer_ac_route_credit_base_cost_mode", "none"),
                                )
                            )
                            route_credit_base_costs = _route_credit_base_cost_matrix(
                                route_tt_costs=vehicle_route_tt_costs,
                                route_valid=vehicle_reward_valid,
                                mode=route_credit_base_mode,
                                step_interval=float(getattr(cfg, "step_interval", 30.0)),
                                elapsed_time=torch.from_numpy(elapsed_time_np).to(
                                    device=vehicle_reward_costs.device,
                                    dtype=vehicle_reward_costs.dtype,
                                ),
                                remaining_tt_eps=float(
                                    getattr(cfg, "dreamer_ac_route_credit_remaining_tt_eps", 1.0)
                                ),
                                progress_cap=float(
                                    getattr(cfg, "dreamer_ac_route_credit_progress_cap", 1.0)
                                ),
                            )
                            route_credit_costs = _route_credit_cost_matrix(
                                base_route_costs=route_credit_base_costs.detach(),
                                route_probs=action_route_probs.detach(),
                                path_batch=path_batch_for_reward,
                                route_valid=vehicle_reward_valid,
                                wm=wm,
                                z=z_next.detach(),
                                cfg=cfg,
                                n_edges=n_edges,
                                n_routes=K,
                                device=device,
                                mode=str(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_route_credit_mode",
                                        getattr(cfg, "dreamer_ac_route_credit_mode", "counterfactual_pressure"),
                                    )
                                ),
                                pressure_weight=float(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_route_credit_pressure_weight",
                                        getattr(cfg, "dreamer_ac_route_credit_pressure_weight", 1.0),
                                    )
                                ),
                                pressure_mode=str(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_route_credit_pressure_mode",
                                        getattr(
                                            cfg,
                                            "dreamer_ac_route_credit_pressure_mode",
                                            "clipped_excess_tt_ratio",
                                        ),
                                    )
                                ),
                                pressure_reference=str(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_route_credit_pressure_reference",
                                        getattr(cfg, "dreamer_ac_route_credit_pressure_reference", "soft"),
                                    )
                                ),
                                pressure_scale=str(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_route_credit_pressure_scale",
                                        "mean",
                                    )
                                ),
                                normalize_by_path_length=bool(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_route_credit_normalize_by_path_length",
                                        getattr(
                                            cfg,
                                            "dreamer_ac_route_credit_normalize_by_path_length",
                                            False,
                                        ),
                                    )
                                ),
                                include_base_cost=bool(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_route_credit_include_base_cost",
                                        False,
                                    )
                                ),
                                detach_pressure=True,
                            )
                            route_credit_costs, route_credit_valid = _apply_route_credit_cost_gate(
                                route_credit_costs=route_credit_costs.detach(),
                                route_tt_costs=vehicle_route_tt_costs.detach(),
                                route_valid=vehicle_reward_valid,
                                max_seconds=float(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_route_credit_cost_gate_max_seconds",
                                        -1.0,
                                    )
                                ),
                                max_ratio=float(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_route_credit_cost_gate_max_ratio",
                                        0.0,
                                    )
                                ),
                                min_routes=int(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_route_credit_cost_gate_min_routes",
                                        1,
                                    )
                                ),
                            )
                            gate_frac, selected_gate_frac = _selected_route_credit_cost_gate_stats(
                                route_tt_costs=vehicle_route_tt_costs.detach(),
                                route_valid=vehicle_reward_valid,
                                selected_idx=selected_route_idx_t_for_update,
                                max_seconds=float(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_route_credit_cost_gate_max_seconds",
                                        -1.0,
                                    )
                                ),
                                max_ratio=float(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_route_credit_cost_gate_max_ratio",
                                        0.0,
                                    )
                                ),
                                min_routes=int(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_route_credit_cost_gate_min_routes",
                                        1,
                                    )
                                ),
                            )
                            vehicle_route_credit_gate_fracs.append(gate_frac)
                            vehicle_route_credit_selected_gate_fracs.append(selected_gate_frac)
                            vehicle_route_credit_reward_for_ac = (
                                _selected_route_credit_vehicle_rewards_from_cost_matrix(
                                    route_credit_costs=route_credit_costs,
                                    route_valid=route_credit_valid,
                                    selected_idx=selected_route_idx_t_for_update,
                                    reward_weight=route_credit_reward_weight,
                                    reward_scale=float(getattr(cfg, "vehicle_reward_scale", 1.0)),
                                    reward_transform=str(
                                        getattr(
                                            cfg,
                                            "dreamer_vehicle_route_credit_reward_transform",
                                            "negative_cost",
                                        )
                                    ),
                                    normalize=bool(
                                        getattr(
                                            cfg,
                                            "dreamer_vehicle_route_credit_reward_normalize",
                                            False,
                                        )
                                    ),
                                    temperature=float(
                                        getattr(
                                            cfg,
                                            "dreamer_vehicle_route_credit_reward_temperature",
                                            1.0,
                                        )
                                    ),
                                )
                            )
                            vehicle_reward_parts.append(vehicle_route_credit_reward_for_ac)
                        social_reward_weight = float(
                            getattr(cfg, "dreamer_vehicle_social_reward_weight", 0.0)
                        )
                        if social_reward_weight != 0.0:
                            vehicle_social_reward_for_ac = _vehicle_social_reward_from_decoded(
                                wm=wm,
                                decoded=edge_dyn_next_for_vehicle_reward,
                                edge_tt=edge_tt_next_for_vehicle_reward,
                                selected_idx=selected_route_idx_t_for_update,
                                valid_action_mask=valid_action_mask_for_update,
                                reward_weight=social_reward_weight,
                                reward_scale=float(getattr(cfg, "vehicle_reward_scale", 1.0)),
                                source=str(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_social_reward_source",
                                        "excess_all_tt",
                                    )
                                ),
                                distribution=str(
                                    getattr(
                                        cfg,
                                        "dreamer_vehicle_social_reward_distribution",
                                        "per_vehicle",
                                    )
                                ),
                                n_active_av=n_active_av,
                            )
                            vehicle_reward_parts.append(vehicle_social_reward_for_ac)
                        if vehicle_reward_parts:
                            vehicle_reward_for_ac = torch.stack(
                                [
                                    part.to(
                                        device=vehicle_reward_costs.device,
                                        dtype=vehicle_reward_costs.dtype,
                                    )
                                    for part in vehicle_reward_parts
                                ],
                                dim=0,
                            ).sum(dim=0)
            ac_sample_weight = None
            if bool(getattr(cfg, "dreamer_ac_weight_by_active_agents", True)):
                n_weight = int(selected_route_idx_t_for_update.reshape(-1).shape[0])
                if n_weight > 0:
                    valid_agents_for_weight = max(float(valid_path_agents), 1.0)
                    ac_sample_weight = torch.full(
                        (n_weight,),
                        1.0 / valid_agents_for_weight,
                        device=device,
                        dtype=z.dtype,
                    )
            vehicle_id_t = torch.as_tensor(
                [int(av.get("id", -1)) for av in active_avs[:n_active_av]],
                device=device,
                dtype=torch.long,
            )
            dreamer_pg.append_discrete_actor_critic_samples(
                obs_steps=dreamer_ac_obs_steps,
                selected_steps=dreamer_ac_selected_steps,
                valid_steps=dreamer_ac_valid_steps,
                step_index_steps=dreamer_ac_step_index_steps,
                route_adv_steps=dreamer_ac_route_adv_steps,
                old_log_prob_steps=dreamer_ac_old_logprob_steps,
                sample_weight_steps=dreamer_ac_sample_weight_steps,
                vehicle_id_steps=dreamer_ac_vehicle_id_steps,
                vehicle_reward_steps=dreamer_ac_vehicle_reward_steps,
                vehicle_route_progress_reward_steps=dreamer_ac_vehicle_route_progress_reward_steps,
                vehicle_route_credit_reward_steps=dreamer_ac_vehicle_route_credit_reward_steps,
                vehicle_social_reward_steps=dreamer_ac_vehicle_social_reward_steps,
                selected_nonzero_fracs=dreamer_ac_selected_nonzero_fracs,
                obs=sampled_obs_t,
                selected_idx=selected_route_idx_t_for_update,
                route_valid=valid_action_mask_for_update,
                step_index=len(rewards),
                route_advantage=route_adv_for_ac,
                old_log_prob=selected_log_probs,
                sample_weight=ac_sample_weight,
                vehicle_id=vehicle_id_t,
                vehicle_reward=vehicle_reward_for_ac,
                vehicle_route_progress_reward=vehicle_route_progress_reward_for_ac,
                vehicle_route_credit_reward=vehicle_route_credit_reward_for_ac,
                vehicle_social_reward=vehicle_social_reward_for_ac,
            )
        reward_requires_grad = bool(reward_t.requires_grad)
        if reward_requires_grad:
            reward_grad_steps += 1
        base_reward_t = reward_t
        base_rewards.append(base_reward_t)
        rollout_reward_t, rollout_reward_stats = rollout_rewards.dreamer_rollout_cost_reward_from_config(
            wm=wm,
            z_next=z_next,
            action_edge=action_edge,
            route_probs=route_probs_for_reward,
            path_batch=path_batch_for_reward,
            n_routes=K,
            avg_route_steps=float(avg_route_steps),
            cfg=cfg,
        )
        reward_t = reward_t + rollout_reward_t
        rollout_reward_stats_steps.append(rollout_reward_stats)
        if bool(rollout_reward_t.requires_grad) and not reward_requires_grad:
            reward_grad_steps += 1
        rewards.append(reward_t)
        rollout_weight = float(getattr(cfg, "dreamer_grpo_rollout_weight", 0.0))
        rollout_vehicle_cost_weight = float(getattr(cfg, "dreamer_grpo_rollout_vehicle_cost_weight", 0.0))
        if rollout_weight != 0.0 or rollout_vehicle_cost_weight != 0.0:
            cost_source = str(getattr(cfg, "dreamer_grpo_rollout_cost_source", "reward")).strip().lower()
            with torch.no_grad():
                edge_dyn_next = wm.flow_projector.decode(z_next.detach())
                edge_tt_next = _edge_tt_tensor_from_decoded(
                    wm=wm,
                    decoded=edge_dyn_next,
                    z=z_next.detach(),
                    edge_cost_source=cfg.wm_edge_cost_source,
                    decoded_speed_min_ratio=cfg.wm_decoded_speed_min_ratio,
                    decoded_speed_max_ratio=cfg.wm_decoded_speed_max_ratio,
                    decoded_speed_excess_scale=cfg.wm_decoded_speed_excess_scale,
                )
                if rollout_weight != 0.0 and cost_source not in {"", "reward", "dreamer_reward"}:
                    pressure_cost = torch.zeros((), device=device, dtype=z.dtype)
                    if cost_source in {"pressure", "pressure_cav", "planned_pressure", "pred_t_all_pressure"}:
                        pressure = torch.relu(action_edge.detach().to(device=device, dtype=edge_tt_next.dtype))
                        pressure_sum = pressure.sum()
                        if bool(torch.isfinite(pressure_sum).detach().cpu().item()) and float(pressure_sum.detach().cpu().item()) > 1e-8:
                            pressure_dist = pressure / pressure_sum.clamp_min(1e-8)
                            pressure_cost = torch.sum(pressure_dist * edge_tt_next) * float(max(avg_route_steps, 1))
                    cav_proxy_cost = torch.zeros((), device=device, dtype=z.dtype)
                    if (
                        cost_source in {"cav_proxy", "pressure_cav", "pred_t_cav_proxy"}
                        and route_probs_for_reward is not None
                        and path_batch_for_reward is not None
                    ):
                        invalid_route_cost = (
                            edge_tt_next.mean()
                            * float(max(avg_route_steps, 1))
                            * float(cfg.vehicle_path_penalty_factor)
                        )
                        cav_val = _expected_route_cost_from_paths(
                            route_probs=route_probs_for_reward.detach(),
                            path_batch=path_batch_for_reward,
                            edge_cost=edge_tt_next,
                            n_routes=int(route_probs_for_reward.shape[1]),
                            invalid_route_cost=invalid_route_cost,
                        )
                        if cav_val is not None:
                            cav_proxy_cost = cav_val.to(device=device, dtype=z.dtype)
                    if cost_source in {"pressure", "planned_pressure", "pred_t_all_pressure"}:
                        rollout_cost_steps.append(pressure_cost.to(device=device, dtype=z.dtype))
                    elif cost_source in {"cav_proxy", "pred_t_cav_proxy"}:
                        rollout_cost_steps.append(cav_proxy_cost.to(device=device, dtype=z.dtype))
                    elif cost_source == "pressure_cav":
                        rollout_cost_steps.append(
                            pressure_cost.to(device=device, dtype=z.dtype)
                            + float(getattr(cfg, "dreamer_grpo_rollout_t_cav_weight", 0.0))
                            * cav_proxy_cost.to(device=device, dtype=z.dtype)
                        )
            if (
                rollout_vehicle_cost_weight != 0.0
                and selected_log_probs is not None
                and selected_route_idx_for_env is not None
                and path_batch_for_reward is not None
            ):
                invalid_route_cost = (
                    edge_tt_next.mean()
                    * float(max(avg_route_steps, 1))
                    * float(cfg.vehicle_path_penalty_factor)
                )
                vehicle_route_costs, vehicle_route_valid = _route_cost_matrix_from_paths(
                    path_batch=path_batch_for_reward,
                    edge_cost=edge_tt_next.detach(),
                    n_routes=K,
                    invalid_route_cost=invalid_route_cost,
                )
                if vehicle_route_costs is not None and vehicle_route_valid is not None:
                    vehicle_pressure_weight = float(
                        getattr(cfg, "dreamer_grpo_rollout_vehicle_planned_av_pressure_weight", 0.0)
                    )
                    if vehicle_pressure_weight != 0.0:
                        vehicle_pressure_costs = _planned_av_pressure_cost_matrix(
                            route_probs=action_route_probs.detach(),
                            path_batch=path_batch_for_reward,
                            route_valid=vehicle_route_valid,
                            n_edges=n_edges,
                            n_routes=K,
                            detach_pressure=True,
                        )
                        if vehicle_pressure_costs is not None:
                            vehicle_route_costs = vehicle_route_costs + (
                                vehicle_pressure_weight
                                * vehicle_pressure_costs.to(
                                    device=vehicle_route_costs.device,
                                    dtype=vehicle_route_costs.dtype,
                                )
                            )
                    n_vehicle = min(
                        int(vehicle_route_costs.shape[0]),
                        int(vehicle_route_valid.shape[0]),
                        int(selected_log_probs.shape[0]),
                        int(np.asarray(selected_route_idx_for_env).reshape(-1).shape[0]),
                    )
                    if n_vehicle > 0:
                        route_costs_v = vehicle_route_costs[:n_vehicle].to(device=device, dtype=z.dtype)
                        route_valid_v = vehicle_route_valid[:n_vehicle].to(device=device)
                        selected_idx_v = torch.as_tensor(
                            np.asarray(selected_route_idx_for_env).reshape(-1)[:n_vehicle],
                            device=device,
                            dtype=torch.long,
                        ).clamp(0, K - 1)
                        selected_logp_v = selected_log_probs[:n_vehicle]
                        rows = torch.arange(n_vehicle, device=device)
                        selected_valid = route_valid_v[rows, selected_idx_v]
                        selected_cost = route_costs_v[rows, selected_idx_v]
                        valid_f = route_valid_v.to(dtype=z.dtype)
                        valid_count = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
                        cost_sum = (route_costs_v.masked_fill(~route_valid_v, 0.0) * valid_f).sum(
                            dim=-1,
                            keepdim=True,
                        )
                        group_mean = cost_sum / valid_count
                        vehicle_adv = group_mean.squeeze(-1) - selected_cost
                        if bool(getattr(cfg, "dreamer_grpo_normalize_advantages", True)):
                            centered = (route_costs_v - group_mean).masked_fill(~route_valid_v, 0.0)
                            group_std = torch.sqrt(
                                (centered.square() * valid_f).sum(dim=-1) / valid_count.squeeze(-1)
                            ).clamp_min(1e-6)
                            vehicle_adv = vehicle_adv / group_std
                        else:
                            vehicle_adv = vehicle_adv / float(
                                max(float(getattr(cfg, "dreamer_grpo_advantage_temperature", 60.0)), 1e-6)
                            )
                        usable = (
                            selected_valid
                            & torch.isfinite(selected_cost)
                            & torch.isfinite(selected_logp_v)
                            & torch.isfinite(vehicle_adv)
                        )
                        if bool(usable.any().detach().cpu().item()):
                            vehicle_adv_used = vehicle_adv[usable].detach()
                            vehicle_logp_used = selected_logp_v[usable]
                            vehicle_loss = -(vehicle_adv_used * vehicle_logp_used).mean()
                            rollout_vehicle_cost_losses.append(vehicle_loss)
                            rollout_vehicle_cost_values.append(
                                float(selected_cost[usable].detach().mean().cpu().item())
                            )
                            rollout_vehicle_cost_adv_means.append(
                                float(vehicle_adv_used.detach().mean().cpu().item())
                            )
                            rollout_vehicle_cost_adv_stds.append(
                                float(vehicle_adv_used.detach().std(unbiased=False).cpu().item())
                                if int(vehicle_adv_used.numel()) > 1 else 0.0
                            )
        action_edges.append(action_edge)
        sum_t_all += t_all_step_min
        sum_t_av += t_cav_step_min
        z_states.append(z_next)
        latent_history.append(z_next)
        z = z_next

    if rewards:
        reward_tensor = torch.stack(rewards)
        entropy_tensor = torch.stack(entropies)
    else:
        reward_tensor = torch.zeros((0,), device=device)
        entropy_tensor = torch.zeros((0,), device=device)
    if rollout_logprob_steps:
        rollout_logprob_tensor = torch.stack(rollout_logprob_steps)
        rollout_logprob_mean = rollout_logprob_tensor.mean()
    else:
        rollout_logprob_tensor = torch.zeros((0,), device=device, dtype=z.dtype)
        rollout_logprob_mean = torch.zeros((), device=device, dtype=z.dtype)
    if rollout_cost_steps:
        cost_tensor = torch.stack(rollout_cost_steps).detach()
        discount = float(getattr(cfg, "dreamer_grpo_rollout_discount", 0.99))
        weights = torch.as_tensor(
            [discount ** i for i in range(int(cost_tensor.numel()))],
            device=cost_tensor.device,
            dtype=cost_tensor.dtype,
        )
        rollout_cost = (cost_tensor * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        rollout_cost = -reward_tensor.detach().mean() if reward_tensor.numel() > 0 else torch.zeros((), device=device, dtype=z.dtype)
    base_reward_mean = (
        float(torch.stack([r.detach() for r in base_rewards]).mean().cpu().item())
        if base_rewards else 0.0
    )
    rollout_reward_stats_mean = rollout_rewards.mean_rollout_reward_stats(rollout_reward_stats_steps)
    ac_samples = dreamer_pg.pack_discrete_actor_critic_samples(
        obs_steps=dreamer_ac_obs_steps,
        selected_steps=dreamer_ac_selected_steps,
        valid_steps=dreamer_ac_valid_steps,
        step_index_steps=dreamer_ac_step_index_steps,
        route_adv_steps=dreamer_ac_route_adv_steps,
        old_log_prob_steps=dreamer_ac_old_logprob_steps,
        sample_weight_steps=dreamer_ac_sample_weight_steps,
        vehicle_id_steps=dreamer_ac_vehicle_id_steps,
        vehicle_reward_steps=dreamer_ac_vehicle_reward_steps,
        vehicle_route_progress_reward_steps=dreamer_ac_vehicle_route_progress_reward_steps,
        vehicle_route_credit_reward_steps=dreamer_ac_vehicle_route_credit_reward_steps,
        vehicle_social_reward_steps=dreamer_ac_vehicle_social_reward_steps,
        device=device,
        dtype=z.dtype,
        obs_dim=int(getattr(actor, "obs_dim", 0)),
        n_routes=K,
    )
    ac_selected_route_fracs = [0.0, 0.0, 0.0, 0.0]
    ac_selected_for_diag = ac_samples.get("dreamer_ac_selected")
    if torch.is_tensor(ac_selected_for_diag) and ac_selected_for_diag.numel() > 0:
        selected_diag = ac_selected_for_diag.detach()
        for route_idx in range(min(K, len(ac_selected_route_fracs))):
            ac_selected_route_fracs[route_idx] = float(
                (selected_diag == route_idx).float().mean().cpu().item()
            )
    if grpo_sample_obs_steps:
        grpo_sample_obs_tensor = torch.cat(grpo_sample_obs_steps, dim=0)
        grpo_sample_selected_tensor = torch.cat(grpo_sample_selected_steps, dim=0)
        grpo_sample_old_logprob_tensor = torch.cat(grpo_sample_old_logprob_steps, dim=0)
        grpo_sample_valid_tensor = torch.cat(grpo_sample_valid_steps, dim=0)
        grpo_sample_route_adv_tensor = torch.cat(grpo_sample_route_adv_steps, dim=0)
    else:
        obs_dim = int(vehicle_mode_obs_dim(
            n_routes=K,
            d_obs=int(getattr(actor, "d_obs", 0)),
            include_global=bool(cfg.vehicle_obs_include_global),
            include_route_costs=bool(cfg.vehicle_obs_include_route_costs),
            include_fleet_pressure=bool(getattr(cfg, "vehicle_obs_include_fleet_pressure", False)),
        ))
        grpo_sample_obs_tensor = torch.zeros((0, obs_dim), device=device, dtype=z.dtype)
        grpo_sample_selected_tensor = torch.zeros((0,), device=device, dtype=torch.long)
        grpo_sample_old_logprob_tensor = torch.zeros((0,), device=device, dtype=z.dtype)
        grpo_sample_valid_tensor = torch.zeros((0, K), device=device, dtype=torch.bool)
        grpo_sample_route_adv_tensor = torch.zeros((0,), device=device, dtype=z.dtype)

    ep_metrics = {
        "t_av": sum_t_av / max(1, len(rewards)),
        "t_hdv": sum_t_all / max(1, len(rewards)),
        "t_all": sum_t_all / max(1, len(rewards)),
        "hdv_drift": 0.0,
        "hdv_head_entropy": 0.0,
        "avg_active_av": float(np.mean(active_counts)) if active_counts else 0.0,
        "max_active_av": float(max_active_av),
        "n_valid_paths": float(np.sum(valid_path_counts)) if valid_path_counts else 0.0,
        "n_valid_path_agents": float(np.sum(valid_path_agent_counts)) if valid_path_agent_counts else 0.0,
        "n_actor_steps": float(np.sum(actor_step_flags)) if actor_step_flags else 0.0,
        "reward_grad_steps": float(reward_grad_steps),
        "entropy_grad_steps": float(entropy_grad_steps),
        "invalid_route_rate": float(np.mean(invalid_rates)) if invalid_rates else 0.0,
        "route_churn": float(np.mean(route_churns)) if route_churns else 0.0,
        "route_cost_aux_loss": float(
            torch.stack([x.detach() for x in route_cost_aux_losses]).mean().cpu().item()
        ) if route_cost_aux_losses else 0.0,
        "route_cost_aux_acc": float(np.mean(route_cost_aux_accs)) if route_cost_aux_accs else 0.0,
        "route_cost_aux_margin": float(np.mean(route_cost_aux_margins)) if route_cost_aux_margins else 0.0,
        "expected_cost_loss": float(
            torch.stack([x.detach() for x in expected_cost_losses]).mean().cpu().item()
        ) if expected_cost_losses else 0.0,
        "expected_cost_raw": float(np.mean(expected_cost_raw_values)) if expected_cost_raw_values else 0.0,
        "expected_cost_best_prob": float(np.mean(expected_cost_best_probs)) if expected_cost_best_probs else 0.0,
        "expected_cost_best_route0_frac": float(np.mean(expected_cost_best_route0_fracs)) if expected_cost_best_route0_fracs else 0.0,
        "expected_cost_nonzero_best_frac": float(np.mean(expected_cost_nonzero_best_fracs)) if expected_cost_nonzero_best_fracs else 0.0,
        "expected_cost_margin": float(np.mean(expected_cost_margins)) if expected_cost_margins else 0.0,
        "grpo_route_cost_loss": float(
            torch.stack([x.detach() for x in grpo_route_cost_losses]).mean().cpu().item()
        ) if grpo_route_cost_losses else 0.0,
        "grpo_route_cost_raw": float(np.mean(grpo_route_cost_raw_values)) if grpo_route_cost_raw_values else 0.0,
        "grpo_route_cost_best_prob": float(np.mean(grpo_route_cost_best_probs)) if grpo_route_cost_best_probs else 0.0,
        "grpo_route_cost_best_route0_frac": float(np.mean(grpo_route_cost_best_route0_fracs)) if grpo_route_cost_best_route0_fracs else 0.0,
        "grpo_route_cost_nonzero_best_frac": float(np.mean(grpo_route_cost_nonzero_best_fracs)) if grpo_route_cost_nonzero_best_fracs else 0.0,
        "grpo_route_cost_margin": float(np.mean(grpo_route_cost_margins)) if grpo_route_cost_margins else 0.0,
        "grpo_route_cost_adv_mean": float(np.mean(grpo_route_cost_adv_means)) if grpo_route_cost_adv_means else 0.0,
        "grpo_route_cost_adv_std": float(np.mean(grpo_route_cost_adv_stds)) if grpo_route_cost_adv_stds else 0.0,
        "grpo_rollout_cost": float(rollout_cost.detach().cpu().item()),
        "grpo_rollout_logprob": float(rollout_logprob_mean.detach().cpu().item()) if rollout_logprob_steps else 0.0,
        "grpo_rollout_selected_nonzero_frac": float(np.mean(rollout_selected_nonzero_fracs)) if rollout_selected_nonzero_fracs else 0.0,
        "grpo_rollout_vehicle_cost_loss": float(
            torch.stack([x.detach() for x in rollout_vehicle_cost_losses]).mean().cpu().item()
        ) if rollout_vehicle_cost_losses else 0.0,
        "grpo_rollout_vehicle_cost": float(np.mean(rollout_vehicle_cost_values)) if rollout_vehicle_cost_values else 0.0,
        "grpo_rollout_vehicle_cost_adv_mean": float(np.mean(rollout_vehicle_cost_adv_means)) if rollout_vehicle_cost_adv_means else 0.0,
        "grpo_rollout_vehicle_cost_adv_std": float(np.mean(rollout_vehicle_cost_adv_stds)) if rollout_vehicle_cost_adv_stds else 0.0,
        "dreamer_base_reward": base_reward_mean,
        "dreamer_ac_rollout_argmax_mix_prob": float(ac_rollout_argmax_mix_prob),
        "dreamer_ac_rollout_argmax_frac": float(np.mean(ac_rollout_argmax_uses)) if ac_rollout_argmax_uses else 0.0,
        "dreamer_ac_sample_count": float(ac_samples["dreamer_ac_selected"].numel()),
        "dreamer_ac_selected_nonzero_frac": (
            float(np.mean(dreamer_ac_selected_nonzero_fracs))
            if dreamer_ac_selected_nonzero_fracs else 0.0
        ),
        "dreamer_ac_selected_route0_frac": ac_selected_route_fracs[0],
        "dreamer_ac_selected_route1_frac": ac_selected_route_fracs[1],
        "dreamer_ac_selected_route2_frac": ac_selected_route_fracs[2],
        "dreamer_ac_selected_route3_frac": ac_selected_route_fracs[3],
        "vehicle_reward_mean": (
            float(ac_samples["dreamer_ac_vehicle_reward"].detach().mean().cpu().item())
            if ac_samples["dreamer_ac_vehicle_reward"].numel() > 0 else 0.0
        ),
        "vehicle_reward_std": (
            float(ac_samples["dreamer_ac_vehicle_reward"].detach().std(unbiased=False).cpu().item())
            if ac_samples["dreamer_ac_vehicle_reward"].numel() > 1 else 0.0
        ),
        "vehicle_route_progress_reward_mean": (
            float(ac_samples["dreamer_ac_vehicle_route_progress_reward"].detach().mean().cpu().item())
            if ac_samples["dreamer_ac_vehicle_route_progress_reward"].numel() > 0 else 0.0
        ),
        "vehicle_route_progress_reward_std": (
            float(ac_samples["dreamer_ac_vehicle_route_progress_reward"].detach().std(unbiased=False).cpu().item())
            if ac_samples["dreamer_ac_vehicle_route_progress_reward"].numel() > 1 else 0.0
        ),
        "vehicle_route_credit_reward_mean": (
            float(ac_samples["dreamer_ac_vehicle_route_credit_reward"].detach().mean().cpu().item())
            if ac_samples["dreamer_ac_vehicle_route_credit_reward"].numel() > 0 else 0.0
        ),
        "vehicle_route_credit_reward_std": (
            float(ac_samples["dreamer_ac_vehicle_route_credit_reward"].detach().std(unbiased=False).cpu().item())
            if ac_samples["dreamer_ac_vehicle_route_credit_reward"].numel() > 1 else 0.0
        ),
        "vehicle_route_credit_gate_frac": (
            float(np.mean(vehicle_route_credit_gate_fracs))
            if vehicle_route_credit_gate_fracs else 0.0
        ),
        "vehicle_route_credit_selected_gate_frac": (
            float(np.mean(vehicle_route_credit_selected_gate_fracs))
            if vehicle_route_credit_selected_gate_fracs else 0.0
        ),
        "vehicle_social_reward_mean": (
            float(ac_samples["dreamer_ac_vehicle_social_reward"].detach().mean().cpu().item())
            if ac_samples["dreamer_ac_vehicle_social_reward"].numel() > 0 else 0.0
        ),
        "vehicle_social_reward_std": (
            float(ac_samples["dreamer_ac_vehicle_social_reward"].detach().std(unbiased=False).cpu().item())
            if ac_samples["dreamer_ac_vehicle_social_reward"].numel() > 1 else 0.0
        ),
        "hdv_background_sum": float(hdv_background_edge.detach().sum().cpu().item()) if hdv_background_edge is not None else 0.0,
        "hdv_background_weight": float(getattr(cfg, "dreamer_hdv_background_weight", 0.0)) if hdv_background_edge is not None else 0.0,
        "terminal_t_all_pred": wm.predict_t_all_from_latent(z.detach()),
        "terminal_bonus": 0.0,
        "start_step": s,
    }
    rollout_rewards.add_rollout_reward_metrics(ep_metrics, rollout_reward_stats_mean)
    trajectory = {
        "traj_id": int(traj_id) if traj_id is not None else int(s),
        "episode_index": int(episode_index) if episode_index is not None else -1,
        "z_states": z_states,
        "action_edges": action_edges,
        "critic_contexts": (
            torch.stack(critic_contexts)
            if critic_contexts
            else torch.zeros((0, 8), device=device, dtype=z.dtype)
        ),
        "rewards": reward_tensor,
        "entropies": entropy_tensor,
        "n_active_av_mean": ep_metrics["avg_active_av"],
        "n_active_av_max": ep_metrics["max_active_av"],
        "n_valid_paths": ep_metrics["n_valid_paths"],
        "n_valid_path_agents": ep_metrics["n_valid_path_agents"],
        "n_actor_steps": ep_metrics["n_actor_steps"],
        "reward_grad_steps": ep_metrics["reward_grad_steps"],
        "entropy_grad_steps": ep_metrics["entropy_grad_steps"],
        "invalid_route_rate": ep_metrics["invalid_route_rate"],
        "route_churn": ep_metrics["route_churn"],
        "route_cost_aux_losses": (
            torch.stack(route_cost_aux_losses)
            if route_cost_aux_losses
            else torch.zeros((0,), device=device, dtype=z.dtype)
        ),
        "route_cost_aux_acc": ep_metrics["route_cost_aux_acc"],
        "route_cost_aux_margin": ep_metrics["route_cost_aux_margin"],
        "expected_cost_losses": (
            torch.stack([x.detach() for x in expected_cost_losses])
            if expected_cost_losses
            else torch.zeros((0,), device=device, dtype=z.dtype)
        ),
        "expected_cost_loss": ep_metrics["expected_cost_loss"],
        "expected_cost_raw": ep_metrics["expected_cost_raw"],
        "expected_cost_best_prob": ep_metrics["expected_cost_best_prob"],
        "expected_cost_best_route0_frac": ep_metrics["expected_cost_best_route0_frac"],
        "expected_cost_nonzero_best_frac": ep_metrics["expected_cost_nonzero_best_frac"],
        "expected_cost_margin": ep_metrics["expected_cost_margin"],
        "grpo_route_cost_losses": (
            torch.stack(grpo_route_cost_losses)
            if grpo_route_cost_losses
            else torch.zeros((0,), device=device, dtype=z.dtype)
        ),
        "grpo_route_cost_loss": ep_metrics["grpo_route_cost_loss"],
        "grpo_route_cost_raw": ep_metrics["grpo_route_cost_raw"],
        "grpo_route_cost_best_prob": ep_metrics["grpo_route_cost_best_prob"],
        "grpo_route_cost_best_route0_frac": ep_metrics["grpo_route_cost_best_route0_frac"],
        "grpo_route_cost_nonzero_best_frac": ep_metrics["grpo_route_cost_nonzero_best_frac"],
        "grpo_route_cost_margin": ep_metrics["grpo_route_cost_margin"],
        "grpo_route_cost_adv_mean": ep_metrics["grpo_route_cost_adv_mean"],
        "grpo_route_cost_adv_std": ep_metrics["grpo_route_cost_adv_std"],
        "grpo_rollout_logprob_mean": rollout_logprob_mean,
        "grpo_rollout_logprob_steps": rollout_logprob_tensor,
        "grpo_rollout_cost": ep_metrics["grpo_rollout_cost"],
        "grpo_rollout_selected_nonzero_frac": ep_metrics["grpo_rollout_selected_nonzero_frac"],
        "grpo_sample_obs": grpo_sample_obs_tensor,
        "grpo_sample_selected": grpo_sample_selected_tensor,
        "grpo_sample_old_logprob": grpo_sample_old_logprob_tensor,
        "grpo_sample_valid": grpo_sample_valid_tensor,
        "grpo_sample_route_adv": grpo_sample_route_adv_tensor,
        "grpo_rollout_vehicle_cost_losses": (
            torch.stack(rollout_vehicle_cost_losses)
            if rollout_vehicle_cost_losses
            else torch.zeros((0,), device=device, dtype=z.dtype)
        ),
        "grpo_rollout_vehicle_cost_loss": ep_metrics["grpo_rollout_vehicle_cost_loss"],
        "grpo_rollout_vehicle_cost": ep_metrics["grpo_rollout_vehicle_cost"],
        "grpo_rollout_vehicle_cost_adv_mean": ep_metrics["grpo_rollout_vehicle_cost_adv_mean"],
        "grpo_rollout_vehicle_cost_adv_std": ep_metrics["grpo_rollout_vehicle_cost_adv_std"],
        "dreamer_base_reward": ep_metrics["dreamer_base_reward"],
        "dreamer_ac_rollout_argmax_mix_prob": ep_metrics["dreamer_ac_rollout_argmax_mix_prob"],
        "dreamer_ac_rollout_argmax_frac": ep_metrics["dreamer_ac_rollout_argmax_frac"],
        "dreamer_ac_sample_count": ep_metrics["dreamer_ac_sample_count"],
        "dreamer_ac_selected_nonzero_frac": ep_metrics["dreamer_ac_selected_nonzero_frac"],
        "dreamer_ac_selected_route0_frac": ep_metrics["dreamer_ac_selected_route0_frac"],
        "dreamer_ac_selected_route1_frac": ep_metrics["dreamer_ac_selected_route1_frac"],
        "dreamer_ac_selected_route2_frac": ep_metrics["dreamer_ac_selected_route2_frac"],
        "dreamer_ac_selected_route3_frac": ep_metrics["dreamer_ac_selected_route3_frac"],
        "vehicle_route_credit_gate_frac": ep_metrics["vehicle_route_credit_gate_frac"],
        "vehicle_route_credit_selected_gate_frac": ep_metrics["vehicle_route_credit_selected_gate_frac"],
        **ac_samples,
    }
    rollout_rewards.add_rollout_reward_trajectory_fields(trajectory, ep_metrics)
    ep_reward = float(reward_tensor.detach().mean().cpu().item()) if reward_tensor.numel() > 0 else 0.0
    return trajectory, ep_reward, ep_metrics


def dreamer_update(
    cfg: TrainConfig,
    actor: VehicleRouteActor,
    critic: CentralizedLatentValueCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    trajectories: List[Dict[str, Any]],
    update_id: Optional[int] = None,
    target_critic: Optional[CentralizedLatentValueCritic] = None,
    return_normalizer: Optional[DreamerReturnNormalizer] = None,
    vehicle_critic: Optional[CentralizedVehicleValueCritic] = None,
    vehicle_critic_optimizer: Optional[torch.optim.Optimizer] = None,
    target_vehicle_critic: Optional[CentralizedVehicleValueCritic] = None,
    vehicle_return_normalizer: Optional[DreamerReturnNormalizer] = None,
    actor_gate_state: Optional[Dict[str, Any]] = None,
    actor_anchor: Optional[VehicleRouteActor] = None,
) -> Dict[str, float]:
    actor_losses: List[torch.Tensor] = []
    critic_losses: List[torch.Tensor] = []
    vehicle_critic_losses: List[torch.Tensor] = []
    q_actor_items: List[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]
    ] = []
    entropies: List[torch.Tensor] = []
    invalid_rates: List[float] = []
    route_churns: List[float] = []
    route_cost_aux_loss_values: List[float] = []
    route_cost_aux_accs: List[float] = []
    route_cost_aux_margins: List[float] = []
    expected_cost_loss_values: List[float] = []
    expected_cost_raw_values: List[float] = []
    expected_cost_best_probs: List[float] = []
    expected_cost_best_route0_fracs: List[float] = []
    expected_cost_nonzero_best_fracs: List[float] = []
    expected_cost_margins: List[float] = []
    grpo_route_cost_loss_values: List[float] = []
    grpo_route_cost_raw_values: List[float] = []
    grpo_route_cost_best_probs: List[float] = []
    grpo_route_cost_best_route0_fracs: List[float] = []
    grpo_route_cost_nonzero_best_fracs: List[float] = []
    grpo_route_cost_margins: List[float] = []
    grpo_route_cost_adv_means: List[float] = []
    grpo_route_cost_adv_stds: List[float] = []
    grpo_policy_items: List[Dict[str, Any]] = []
    grpo_rollout_items: List[Tuple[float, int, int]] = []
    grpo_rollout_cost_values: List[float] = []
    grpo_rollout_baseline_cost_values: List[float] = []
    grpo_rollout_group_ids: List[int] = []
    grpo_rollout_logprob_values: List[float] = []
    grpo_rollout_selected_nonzero_fracs: List[float] = []
    grpo_rollout_adv_values: List[float] = []
    grpo_combined_loss_values: List[float] = []
    grpo_combined_pg_loss_values: List[float] = []
    grpo_combined_kl_loss_values: List[float] = []
    grpo_combined_adv_means: List[float] = []
    grpo_combined_adv_stds: List[float] = []
    grpo_combined_clip_fracs: List[float] = []
    grpo_policy_kl_values: List[float] = []
    grpo_rollout_vehicle_cost_loss_values: List[float] = []
    grpo_rollout_vehicle_cost_values: List[float] = []
    grpo_rollout_vehicle_cost_adv_means: List[float] = []
    grpo_rollout_vehicle_cost_adv_stds: List[float] = []
    dreamer_ac_loss_values: List[float] = []
    dreamer_ac_adv_means: List[float] = []
    dreamer_ac_adv_stds: List[float] = []
    dreamer_ac_adv_mins: List[float] = []
    dreamer_ac_adv_maxs: List[float] = []
    dreamer_ac_route_adv_means: List[float] = []
    dreamer_ac_route_adv_stds: List[float] = []
    dreamer_ac_combined_adv_means: List[float] = []
    dreamer_ac_combined_adv_stds: List[float] = []
    dreamer_ac_selected_nonzero_fracs: List[float] = []
    dreamer_ac_selected_route0_fracs: List[float] = []
    dreamer_ac_selected_route1_fracs: List[float] = []
    dreamer_ac_selected_route2_fracs: List[float] = []
    dreamer_ac_selected_route3_fracs: List[float] = []
    dreamer_ac_policy_argmax_nonzero_fracs: List[float] = []
    dreamer_ac_policy_argmax_route0_fracs: List[float] = []
    dreamer_ac_policy_argmax_route1_fracs: List[float] = []
    dreamer_ac_policy_argmax_route2_fracs: List[float] = []
    dreamer_ac_policy_argmax_route3_fracs: List[float] = []
    dreamer_ac_policy_argmax_max_route_fracs: List[float] = []
    dreamer_ac_policy_top1_prob_means: List[float] = []
    dreamer_ac_sample_counts: List[float] = []
    dreamer_ac_policy_kls: List[float] = []
    dreamer_ac_clip_fracs: List[float] = []
    dreamer_ac_weight_means: List[float] = []
    dreamer_ac_uses_sample_advs: List[float] = []
    dreamer_ac_entropy_losses: List[float] = []
    dreamer_ac_entropy_floor_losses: List[float] = []
    dreamer_ac_entropy_targets: List[float] = []
    dreamer_ac_entropy_valid_target_means: List[float] = []
    dreamer_ac_entropy_effective_route_counts: List[float] = []
    dreamer_ac_entropy_cost_mask_fracs: List[float] = []
    critic_target_means: List[float] = []
    critic_target_stds: List[float] = []
    critic_explained_variances: List[float] = []
    vehicle_critic_target_means: List[float] = []
    vehicle_critic_target_stds: List[float] = []
    vehicle_critic_explained_variances: List[float] = []
    vehicle_reward_means: List[float] = []
    vehicle_reward_stds: List[float] = []
    vehicle_route_progress_reward_means: List[float] = []
    vehicle_route_progress_reward_stds: List[float] = []
    vehicle_route_credit_reward_means: List[float] = []
    vehicle_route_credit_reward_stds: List[float] = []
    vehicle_route_credit_gate_fracs: List[float] = []
    vehicle_route_credit_selected_gate_fracs: List[float] = []
    vehicle_social_reward_means: List[float] = []
    vehicle_social_reward_stds: List[float] = []
    skipped_actor_trajs = 0
    skip_reason_counts: Dict[str, int] = {}
    skipped_traj_ids: List[int] = []
    used_traj_ids: List[int] = []
    consumed_traj_ids: List[int] = []
    consumed_restored_traj_count = 0
    actor_valid_agent_counts = [
        float(traj.get("n_valid_path_agents", 0.0))
        for traj in trajectories
        if not bool(traj.get("restored_from_checkpoint", False))
    ]
    max_valid_agent_count = max(actor_valid_agent_counts) if actor_valid_agent_counts else 0.0
    actor_objective = str(getattr(cfg, "dreamer_actor_objective", "direct_return")).strip().lower()
    if not dreamer_pg.supports_actor_objective(actor_objective):
        raise ValueError(f"Unsupported dreamer_actor_objective={actor_objective!r}")
    use_discrete_actor_critic = bool(dreamer_pg.uses_discrete_actor_critic(cfg))
    use_vehicle_critic = bool(
        use_discrete_actor_critic
        and bool(getattr(cfg, "dreamer_vehicle_critic", True))
        and vehicle_critic is not None
        and vehicle_critic_optimizer is not None
    )
    train_global_critic = _global_critic_reward_enabled(cfg)
    critic_return_discount = _dreamer_global_return_discount(cfg)
    vehicle_return_discount = _dreamer_vehicle_return_discount(cfg)

    for traj in trajectories:
        traj_id = int(traj.get("traj_id", -1))
        consumed_traj_ids.append(traj_id)
        if bool(traj.get("restored_from_checkpoint", False)):
            consumed_restored_traj_count += 1
            skipped_actor_trajs += 1
            reason = "restored_traj_no_grad_graph"
            skip_reason_counts[reason] = int(skip_reason_counts.get(reason, 0)) + 1
            skipped_traj_ids.append(traj_id)
            continue
        z_states = traj["z_states"]
        action_edges = traj.get("action_edges", [])
        rewards = traj["rewards"]
        if len(z_states) < 2 or rewards.numel() == 0:
            continue
        current_z = torch.stack([z.detach() for z in z_states[:-1]], dim=0)
        next_z = torch.stack([z.detach() for z in z_states[1:]], dim=0)
        critic_context_tensor = traj.get("critic_contexts")
        current_context = None
        next_context = None
        if (
            int(getattr(critic, "context_dim", 0)) > 0
            and torch.is_tensor(critic_context_tensor)
            and int(critic_context_tensor.shape[0]) >= int(rewards.shape[0])
        ):
            current_context = critic_context_tensor[: rewards.shape[0]].detach()
            if int(critic_context_tensor.shape[0]) > int(rewards.shape[0]):
                next_context = critic_context_tensor[1: rewards.shape[0] + 1].detach()
            else:
                next_context = torch.cat([current_context[1:], current_context[-1:]], dim=0)
        current_action = None
        current_action_for_actor = None
        next_action = None
        if int(getattr(critic, "action_dim", 0)) > 0:
            if len(action_edges) != rewards.shape[0]:
                raise ValueError(
                    f"action-conditioned critic expects one action_edge per reward step, got {len(action_edges)} vs {rewards.shape[0]}"
                )
            current_action = torch.stack([a.detach() for a in action_edges], dim=0)
            current_action_for_actor = torch.stack(action_edges, dim=0)
            bootstrap_actions = list(action_edges[1:]) + [action_edges[-1]]
            next_action = torch.stack([a.detach() for a in bootstrap_actions], dim=0)

        bootstrap_critic = target_critic if target_critic is not None else critic
        if train_global_critic:
            values = critic(current_z, action=current_action, context=current_context).squeeze(-1)
            with torch.no_grad():
                next_values = bootstrap_critic(next_z, action=next_action, context=next_context).squeeze(-1)
                critic_targets = lambda_return(
                    rewards.detach(),
                    next_values,
                    gamma=critic_return_discount,
                    lambda_=float(cfg.gae_lambda),
                )
            critic_target_means.append(float(critic_targets.detach().mean().cpu().item()))
            critic_target_stds.append(
                float(critic_targets.detach().std(unbiased=False).cpu().item())
                if int(critic_targets.numel()) > 1 else 0.0
            )
            target_var = torch.var(critic_targets.detach(), unbiased=False)
            if bool(torch.isfinite(target_var).detach().cpu().item()) and float(target_var.detach().cpu().item()) > 1e-8:
                residual_var = torch.var((critic_targets.detach() - values.detach()), unbiased=False)
                critic_explained_variances.append(
                    float((1.0 - residual_var / target_var.clamp_min(1e-8)).detach().cpu().item())
                )
            if return_normalizer is not None and bool(getattr(cfg, "dreamer_return_normalizer", True)):
                return_normalizer.update(critic_targets)
                critic_values_for_loss = return_normalizer.normalize(values)
                critic_targets_for_loss = return_normalizer.normalize(critic_targets.detach())
            elif bool(getattr(cfg, "dreamer_normalize_critic_targets", False)):
                critic_target_ref = critic_targets.detach()
                critic_target_mean = critic_target_ref.mean()
                critic_target_std = critic_target_ref.std(unbiased=False).clamp_min(1e-6)
                critic_values_for_loss = (values - critic_target_mean) / critic_target_std
                critic_targets_for_loss = (critic_targets - critic_target_mean) / critic_target_std
            else:
                critic_values_for_loss = values
                critic_targets_for_loss = critic_targets
            critic_loss = torch.mean((critic_values_for_loss - critic_targets_for_loss) ** 2) * float(cfg.dreamer_value_coef)
            critic_losses.append(critic_loss)

            with torch.no_grad():
                actor_bootstrap = bootstrap_critic(next_z, action=next_action, context=next_context).squeeze(-1)
            actor_returns = lambda_return(
                rewards,
                actor_bootstrap,
                gamma=critic_return_discount,
                lambda_=float(cfg.gae_lambda),
            )
            if return_normalizer is not None and bool(getattr(cfg, "dreamer_return_normalizer", True)):
                actor_advantages = return_normalizer.normalize(actor_returns) - return_normalizer.normalize(values.detach())
            else:
                actor_advantages = actor_returns - values.detach()
        else:
            actor_advantages = torch.zeros(
                (int(rewards.reshape(-1).shape[0]),),
                device=current_z.device,
                dtype=current_z.dtype,
            )
        if bool(getattr(cfg, "dreamer_normalize_actor_advantages", False)):
            actor_advantages = _standardize_with_detached_stats(actor_advantages, center=True)
        if use_vehicle_critic:
            vehicle_obs = traj.get("dreamer_ac_obs")
            vehicle_reward = traj.get("dreamer_ac_vehicle_reward")
            vehicle_ids = traj.get("dreamer_ac_vehicle_id")
            vehicle_step_index = traj.get("dreamer_ac_step_index")
            vehicle_sample_weight = traj.get("dreamer_ac_sample_weight")
            if (
                torch.is_tensor(vehicle_obs)
                and torch.is_tensor(vehicle_reward)
                and torch.is_tensor(vehicle_ids)
                and torch.is_tensor(vehicle_step_index)
                and vehicle_obs.numel() > 0
                and vehicle_reward.numel() > 0
                and vehicle_ids.numel() > 0
            ):
                n_vehicle_samples = min(
                    int(vehicle_obs.shape[0]),
                    int(vehicle_reward.reshape(-1).shape[0]),
                    int(vehicle_ids.reshape(-1).shape[0]),
                    int(vehicle_step_index.reshape(-1).shape[0]),
                )
                if n_vehicle_samples > 0:
                    vehicle_obs_n = vehicle_obs[:n_vehicle_samples]
                    vehicle_reward_n = vehicle_reward.reshape(-1)[:n_vehicle_samples].to(
                        device=vehicle_obs_n.device,
                        dtype=vehicle_obs_n.dtype,
                    )
                    vehicle_ids_n = vehicle_ids.reshape(-1)[:n_vehicle_samples].to(
                        device=vehicle_obs_n.device,
                        dtype=torch.long,
                    )
                    vehicle_step_n = vehicle_step_index.reshape(-1)[:n_vehicle_samples].to(
                        device=vehicle_obs_n.device,
                        dtype=torch.long,
                    )
                    finite_vehicle = (
                        (vehicle_ids_n >= 0)
                        & torch.isfinite(vehicle_reward_n)
                        & torch.isfinite(vehicle_obs_n).all(dim=-1)
                    )
                    if bool(finite_vehicle.any().detach().cpu().item()):
                        obs_used = vehicle_obs_n[finite_vehicle]
                        reward_used = vehicle_reward_n[finite_vehicle]
                        ids_used = vehicle_ids_n[finite_vehicle]
                        step_used = vehicle_step_n[finite_vehicle]
                        vehicle_context = None
                        if int(getattr(vehicle_critic, "context_dim", 0)) > 0:
                            if (
                                torch.is_tensor(critic_context_tensor)
                                and int(critic_context_tensor.shape[0]) > 0
                            ):
                                ctx_idx = step_used.clamp(0, int(critic_context_tensor.shape[0]) - 1)
                                vehicle_context = critic_context_tensor[ctx_idx].detach()
                        bootstrap_vehicle_critic = (
                            target_vehicle_critic if target_vehicle_critic is not None else vehicle_critic
                        )
                        vehicle_values = vehicle_critic(
                            obs_used,
                            context=vehicle_context,
                            group_index=step_used,
                        ).squeeze(-1)
                        with torch.no_grad():
                            bootstrap_vehicle_values = bootstrap_vehicle_critic(
                                obs_used,
                                context=vehicle_context,
                                group_index=step_used,
                            ).squeeze(-1)
                            vehicle_targets = torch.zeros_like(vehicle_values)
                            for veh_id in torch.unique(ids_used, sorted=True):
                                mask = ids_used == veh_id
                                idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
                                if int(idx.numel()) <= 0:
                                    continue
                                order = torch.argsort(step_used[idx], stable=True)
                                ordered_idx = idx[order]
                                rewards_i = reward_used.detach()[ordered_idx]
                                next_i = torch.zeros_like(rewards_i)
                                if int(ordered_idx.numel()) > 1:
                                    next_i[:-1] = bootstrap_vehicle_values.detach()[ordered_idx[1:]]
                                vehicle_targets[ordered_idx] = lambda_return(
                                    rewards_i,
                                    next_i,
                                    gamma=vehicle_return_discount,
                                    lambda_=float(cfg.gae_lambda),
                                )
                        vehicle_critic_target_means.append(float(vehicle_targets.detach().mean().cpu().item()))
                        vehicle_critic_target_stds.append(
                            float(vehicle_targets.detach().std(unbiased=False).cpu().item())
                            if int(vehicle_targets.numel()) > 1 else 0.0
                        )
                        vehicle_reward_means.append(float(reward_used.detach().mean().cpu().item()))
                        vehicle_reward_stds.append(
                            float(reward_used.detach().std(unbiased=False).cpu().item())
                            if int(reward_used.numel()) > 1 else 0.0
                        )
                        vehicle_route_progress_reward = traj.get("dreamer_ac_vehicle_route_progress_reward")
                        if torch.is_tensor(vehicle_route_progress_reward) and vehicle_route_progress_reward.numel() > 0:
                            progress_used = vehicle_route_progress_reward.reshape(-1)[:n_vehicle_samples].to(
                                device=vehicle_obs_n.device,
                                dtype=vehicle_obs_n.dtype,
                            )[finite_vehicle]
                            vehicle_route_progress_reward_means.append(
                                float(progress_used.detach().mean().cpu().item())
                            )
                            vehicle_route_progress_reward_stds.append(
                                float(progress_used.detach().std(unbiased=False).cpu().item())
                                if int(progress_used.numel()) > 1 else 0.0
                            )
                        vehicle_route_credit_reward = traj.get("dreamer_ac_vehicle_route_credit_reward")
                        if torch.is_tensor(vehicle_route_credit_reward) and vehicle_route_credit_reward.numel() > 0:
                            route_credit_used = vehicle_route_credit_reward.reshape(-1)[:n_vehicle_samples].to(
                                device=vehicle_obs_n.device,
                                dtype=vehicle_obs_n.dtype,
                            )[finite_vehicle]
                            vehicle_route_credit_reward_means.append(
                                float(route_credit_used.detach().mean().cpu().item())
                            )
                            vehicle_route_credit_reward_stds.append(
                                float(route_credit_used.detach().std(unbiased=False).cpu().item())
                                if int(route_credit_used.numel()) > 1 else 0.0
                            )
                        vehicle_social_reward = traj.get("dreamer_ac_vehicle_social_reward")
                        if torch.is_tensor(vehicle_social_reward) and vehicle_social_reward.numel() > 0:
                            social_used = vehicle_social_reward.reshape(-1)[:n_vehicle_samples].to(
                                device=vehicle_obs_n.device,
                                dtype=vehicle_obs_n.dtype,
                            )[finite_vehicle]
                            vehicle_social_reward_means.append(
                                float(social_used.detach().mean().cpu().item())
                            )
                            vehicle_social_reward_stds.append(
                                float(social_used.detach().std(unbiased=False).cpu().item())
                                if int(social_used.numel()) > 1 else 0.0
                            )
                        vehicle_target_var = torch.var(vehicle_targets.detach(), unbiased=False)
                        if (
                            bool(torch.isfinite(vehicle_target_var).detach().cpu().item())
                            and float(vehicle_target_var.detach().cpu().item()) > 1e-8
                        ):
                            vehicle_residual_var = torch.var(
                                vehicle_targets.detach() - vehicle_values.detach(),
                                unbiased=False,
                            )
                            vehicle_critic_explained_variances.append(
                                float(
                                    (
                                        1.0
                                        - vehicle_residual_var / vehicle_target_var.clamp_min(1e-8)
                                    )
                                    .detach()
                                    .cpu()
                                    .item()
                                )
                            )
                        if (
                            vehicle_return_normalizer is not None
                            and bool(getattr(cfg, "dreamer_vehicle_critic_return_normalizer", True))
                        ):
                            vehicle_return_normalizer.update(vehicle_targets)
                            vehicle_values_for_loss = vehicle_return_normalizer.normalize(vehicle_values)
                            vehicle_targets_for_loss = vehicle_return_normalizer.normalize(vehicle_targets.detach())
                        else:
                            vehicle_values_for_loss = vehicle_values
                            vehicle_targets_for_loss = vehicle_targets
                        vehicle_loss = torch.mean((vehicle_values_for_loss - vehicle_targets_for_loss) ** 2)
                        vehicle_loss = vehicle_loss * float(getattr(cfg, "dreamer_vehicle_critic_value_coef", 1.0))
                        vehicle_critic_losses.append(vehicle_loss)

                        sample_adv = torch.zeros(
                            (n_vehicle_samples,),
                            device=vehicle_obs_n.device,
                            dtype=vehicle_obs_n.dtype,
                        )
                        if (
                            vehicle_return_normalizer is not None
                            and bool(getattr(cfg, "dreamer_vehicle_critic_return_normalizer", True))
                        ):
                            adv_used = (
                                vehicle_return_normalizer.normalize(vehicle_targets)
                                - vehicle_return_normalizer.normalize(vehicle_values.detach())
                            )
                        else:
                            adv_used = vehicle_targets - vehicle_values.detach()
                        if bool(getattr(cfg, "dreamer_normalize_actor_advantages", False)):
                            adv_used = _standardize_with_detached_stats(adv_used, center=True)
                        sample_adv[finite_vehicle] = adv_used.detach().to(dtype=sample_adv.dtype)
                        traj["dreamer_ac_sample_adv"] = sample_adv
                    else:
                        traj["dreamer_ac_sample_adv"] = torch.zeros(
                            (n_vehicle_samples,),
                            device=vehicle_obs_n.device,
                            dtype=vehicle_obs_n.dtype,
                        )
        entropy = traj["entropies"].mean()
        route_cost_aux_loss = None
        route_cost_aux_tensor = traj.get("route_cost_aux_losses")
        if (
            float(getattr(cfg, "dreamer_route_cost_aux_weight", 0.0)) != 0.0
            and torch.is_tensor(route_cost_aux_tensor)
            and route_cost_aux_tensor.numel() > 0
        ):
            route_cost_aux_loss = route_cost_aux_tensor.mean()
            route_cost_aux_loss_values.append(float(route_cost_aux_loss.detach().cpu().item()))
            route_cost_aux_accs.append(float(traj.get("route_cost_aux_acc", 0.0)))
            route_cost_aux_margins.append(float(traj.get("route_cost_aux_margin", 0.0)))
        expected_cost_loss = None
        expected_cost_tensor = traj.get("expected_cost_losses")
        if (
            float(getattr(cfg, "dreamer_expected_cost_weight", 0.0)) != 0.0
            and torch.is_tensor(expected_cost_tensor)
            and expected_cost_tensor.numel() > 0
        ):
            expected_cost_loss = expected_cost_tensor.mean()
            expected_cost_loss_values.append(float(expected_cost_loss.detach().cpu().item()))
            expected_cost_raw_values.append(float(traj.get("expected_cost_raw", 0.0)))
            expected_cost_best_probs.append(float(traj.get("expected_cost_best_prob", 0.0)))
            expected_cost_best_route0_fracs.append(float(traj.get("expected_cost_best_route0_frac", 0.0)))
            expected_cost_nonzero_best_fracs.append(float(traj.get("expected_cost_nonzero_best_frac", 0.0)))
            expected_cost_margins.append(float(traj.get("expected_cost_margin", 0.0)))
        grpo_route_cost_loss_values.append(float(traj.get("grpo_route_cost_loss", 0.0)))
        grpo_route_cost_raw_values.append(float(traj.get("grpo_route_cost_raw", 0.0)))
        grpo_route_cost_best_probs.append(float(traj.get("grpo_route_cost_best_prob", 0.0)))
        grpo_route_cost_best_route0_fracs.append(float(traj.get("grpo_route_cost_best_route0_frac", 0.0)))
        grpo_route_cost_nonzero_best_fracs.append(float(traj.get("grpo_route_cost_nonzero_best_frac", 0.0)))
        grpo_route_cost_margins.append(float(traj.get("grpo_route_cost_margin", 0.0)))
        grpo_route_cost_adv_means.append(float(traj.get("grpo_route_cost_adv_mean", 0.0)))
        grpo_route_cost_adv_stds.append(float(traj.get("grpo_route_cost_adv_std", 0.0)))
        grpo_rollout_logprob = traj.get("grpo_rollout_logprob_mean")
        grpo_rollout_enabled = bool(
            float(getattr(cfg, "dreamer_grpo_rollout_weight", 0.0)) != 0.0
        )
        grpo_sample_obs = traj.get("grpo_sample_obs")
        grpo_sample_selected = traj.get("grpo_sample_selected")
        grpo_sample_old_logprob = traj.get("grpo_sample_old_logprob")
        grpo_sample_valid = traj.get("grpo_sample_valid")
        grpo_sample_route_adv = traj.get("grpo_sample_route_adv")
        grpo_policy_has_samples = bool(
            torch.is_tensor(grpo_sample_obs)
            and torch.is_tensor(grpo_sample_selected)
            and torch.is_tensor(grpo_sample_old_logprob)
            and torch.is_tensor(grpo_sample_valid)
            and torch.is_tensor(grpo_sample_route_adv)
            and grpo_sample_obs.numel() > 0
            and grpo_sample_selected.numel() > 0
        )
        grpo_cost = float(traj.get("grpo_rollout_cost", 0.0)) if grpo_rollout_enabled else 0.0
        grpo_rollout_vehicle_cost_loss = None
        grpo_rollout_vehicle_cost_tensor = traj.get("grpo_rollout_vehicle_cost_losses")
        if (
            float(getattr(cfg, "dreamer_grpo_rollout_vehicle_cost_weight", 0.0)) != 0.0
            and torch.is_tensor(grpo_rollout_vehicle_cost_tensor)
            and grpo_rollout_vehicle_cost_tensor.numel() > 0
        ):
            grpo_rollout_vehicle_cost_loss = grpo_rollout_vehicle_cost_tensor.mean()
            grpo_rollout_vehicle_cost_loss_values.append(
                float(grpo_rollout_vehicle_cost_loss.detach().cpu().item())
            )
            grpo_rollout_vehicle_cost_values.append(float(traj.get("grpo_rollout_vehicle_cost", 0.0)))
            grpo_rollout_vehicle_cost_adv_means.append(
                float(traj.get("grpo_rollout_vehicle_cost_adv_mean", 0.0))
            )
            grpo_rollout_vehicle_cost_adv_stds.append(
                float(traj.get("grpo_rollout_vehicle_cost_adv_std", 0.0))
            )
        discrete_ac_loss = None
        discrete_ac_has_samples = False
        if use_discrete_actor_critic:
            discrete_ac_loss, discrete_ac_stats = dreamer_pg.discrete_actor_critic_policy_loss(
                actor=actor,
                traj=traj,
                step_advantages=actor_advantages,
                entropy=entropy,
                entropy_coef=float(cfg.dreamer_entropy_coef),
                temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                advantage_clip=float(getattr(cfg, "dreamer_ac_advantage_clip", 5.0)),
                step_advantage_weight=float(getattr(cfg, "dreamer_ac_step_advantage_weight", 1.0)),
                route_advantage_weight=float(getattr(cfg, "dreamer_ac_route_advantage_weight", 0.0)),
                ppo_clip_ratio=float(getattr(cfg, "dreamer_ac_ppo_clip_ratio", 0.0)),
                normalize_combined_advantage=bool(
                    getattr(cfg, "dreamer_ac_normalize_combined_advantages", False)
                ),
                entropy_target=float(getattr(cfg, "dreamer_ac_entropy_target", 0.0)),
                entropy_target_frac=float(getattr(cfg, "dreamer_ac_entropy_target_frac", 0.0)),
                entropy_floor_coef=float(getattr(cfg, "dreamer_ac_entropy_floor_coef", 0.0)),
                entropy_cost_gap_max=float(getattr(cfg, "dreamer_ac_entropy_cost_gap_max", -1.0)),
                entropy_cost_min_routes=int(getattr(cfg, "dreamer_ac_entropy_cost_min_routes", 1)),
            )
            discrete_ac_has_samples = discrete_ac_loss is not None
            dreamer_ac_loss_values.append(float(discrete_ac_stats.get("dreamer_ac_loss", 0.0)))
            dreamer_ac_adv_means.append(float(discrete_ac_stats.get("dreamer_ac_adv_mean", 0.0)))
            dreamer_ac_adv_stds.append(float(discrete_ac_stats.get("dreamer_ac_adv_std", 0.0)))
            dreamer_ac_adv_mins.append(float(discrete_ac_stats.get("dreamer_ac_adv_min", 0.0)))
            dreamer_ac_adv_maxs.append(float(discrete_ac_stats.get("dreamer_ac_adv_max", 0.0)))
            dreamer_ac_route_adv_means.append(float(discrete_ac_stats.get("dreamer_ac_route_adv_mean", 0.0)))
            dreamer_ac_route_adv_stds.append(float(discrete_ac_stats.get("dreamer_ac_route_adv_std", 0.0)))
            dreamer_ac_combined_adv_means.append(float(discrete_ac_stats.get("dreamer_ac_combined_adv_mean", 0.0)))
            dreamer_ac_combined_adv_stds.append(float(discrete_ac_stats.get("dreamer_ac_combined_adv_std", 0.0)))
            dreamer_ac_selected_nonzero_fracs.append(
                float(discrete_ac_stats.get("dreamer_ac_selected_nonzero_frac", 0.0))
            )
            dreamer_ac_selected_route0_fracs.append(
                float(discrete_ac_stats.get("dreamer_ac_selected_route0_frac", 0.0))
            )
            dreamer_ac_selected_route1_fracs.append(
                float(discrete_ac_stats.get("dreamer_ac_selected_route1_frac", 0.0))
            )
            dreamer_ac_selected_route2_fracs.append(
                float(discrete_ac_stats.get("dreamer_ac_selected_route2_frac", 0.0))
            )
            dreamer_ac_selected_route3_fracs.append(
                float(discrete_ac_stats.get("dreamer_ac_selected_route3_frac", 0.0))
            )
            dreamer_ac_policy_argmax_nonzero_fracs.append(
                float(discrete_ac_stats.get("dreamer_ac_policy_argmax_nonzero_frac", 0.0))
            )
            dreamer_ac_policy_argmax_route0_fracs.append(
                float(discrete_ac_stats.get("dreamer_ac_policy_argmax_route0_frac", 0.0))
            )
            dreamer_ac_policy_argmax_route1_fracs.append(
                float(discrete_ac_stats.get("dreamer_ac_policy_argmax_route1_frac", 0.0))
            )
            dreamer_ac_policy_argmax_route2_fracs.append(
                float(discrete_ac_stats.get("dreamer_ac_policy_argmax_route2_frac", 0.0))
            )
            dreamer_ac_policy_argmax_route3_fracs.append(
                float(discrete_ac_stats.get("dreamer_ac_policy_argmax_route3_frac", 0.0))
            )
            dreamer_ac_policy_argmax_max_route_fracs.append(
                float(discrete_ac_stats.get("dreamer_ac_policy_argmax_max_route_frac", 0.0))
            )
            dreamer_ac_policy_top1_prob_means.append(
                float(discrete_ac_stats.get("dreamer_ac_policy_top1_prob_mean", 0.0))
            )
            dreamer_ac_sample_counts.append(float(discrete_ac_stats.get("dreamer_ac_sample_count", 0.0)))
            dreamer_ac_policy_kls.append(float(discrete_ac_stats.get("dreamer_ac_policy_kl", 0.0)))
            dreamer_ac_clip_fracs.append(float(discrete_ac_stats.get("dreamer_ac_clip_frac", 0.0)))
            dreamer_ac_weight_means.append(float(discrete_ac_stats.get("dreamer_ac_weight_mean", 0.0)))
            dreamer_ac_uses_sample_advs.append(float(discrete_ac_stats.get("dreamer_ac_uses_sample_adv", 0.0)))
            dreamer_ac_entropy_losses.append(float(discrete_ac_stats.get("dreamer_ac_entropy_loss", 0.0)))
            dreamer_ac_entropy_floor_losses.append(
                float(discrete_ac_stats.get("dreamer_ac_entropy_floor_loss", 0.0))
            )
            dreamer_ac_entropy_targets.append(float(discrete_ac_stats.get("dreamer_ac_entropy_target", 0.0)))
            dreamer_ac_entropy_valid_target_means.append(
                float(discrete_ac_stats.get("dreamer_ac_entropy_valid_target_mean", 0.0))
            )
            dreamer_ac_entropy_effective_route_counts.append(
                float(discrete_ac_stats.get("dreamer_ac_entropy_effective_route_count", 0.0))
            )
            dreamer_ac_entropy_cost_mask_fracs.append(
                float(discrete_ac_stats.get("dreamer_ac_entropy_cost_mask_frac", 0.0))
            )
        direct_actor_loss = -(actor_advantages.mean() + float(cfg.dreamer_entropy_coef) * entropy)
        actor_loss = direct_actor_loss
        if route_cost_aux_loss is not None:
            actor_loss = actor_loss + float(cfg.dreamer_route_cost_aux_weight) * route_cost_aux_loss
        if grpo_rollout_vehicle_cost_loss is not None:
            actor_loss = (
                actor_loss
                + float(cfg.dreamer_grpo_rollout_vehicle_cost_weight)
                * grpo_rollout_vehicle_cost_loss
            )
        low_valid_agent_coverage = (
            float(getattr(cfg, "dreamer_actor_min_valid_agents_ratio", 0.0)) > 0.0
            and max_valid_agent_count > 0.0
            and float(traj.get("n_valid_path_agents", 0.0))
            < float(cfg.dreamer_actor_min_valid_agents_ratio) * float(max_valid_agent_count)
        )
        actor_update_disabled = bool(getattr(cfg, "dreamer_disable_actor_updates", False))
        use_direct_actor = actor_objective in {"direct_return", "direct_plus_q"}
        use_q_actor = actor_objective in {"critic_q", "direct_plus_q"}
        if (
            use_q_actor
            and int(getattr(critic, "action_dim", 0)) <= 0
        ):
            raise ValueError("dreamer_actor_objective=critic_q/direct_plus_q requires --dreamer_critic_variant action_conditioned")
        q_actor_has_grad = bool(
            use_q_actor
            and current_action_for_actor is not None
            and current_action_for_actor.requires_grad
        )
        direct_actor_has_grad = bool(use_direct_actor and actor_loss.requires_grad)
        if (
            direct_actor_has_grad
            or q_actor_has_grad
            or grpo_policy_has_samples
            or discrete_ac_has_samples
        ) and not low_valid_agent_coverage and not actor_update_disabled:
            if use_direct_actor:
                actor_losses.append(actor_loss)
            if discrete_ac_loss is not None:
                actor_losses.append(discrete_ac_loss)
            if q_actor_has_grad:
                q_actor_items.append((
                    current_z,
                    current_action_for_actor,
                    current_context,
                    entropy,
                    route_cost_aux_loss,
                ))
            if grpo_policy_has_samples:
                grpo_group_id = int(traj.get("grpo_rollout_group_id", traj_id))
                grpo_policy_items.append({
                    "traj_id": traj_id,
                    "group_id": grpo_group_id,
                    "obs": grpo_sample_obs,
                    "selected": grpo_sample_selected,
                    "old_logprob": grpo_sample_old_logprob,
                    "valid": grpo_sample_valid,
                    "route_adv": grpo_sample_route_adv,
                    "rollout_logprob": grpo_rollout_logprob,
                })
                if grpo_rollout_enabled:
                    grpo_rollout_items.append((grpo_cost, grpo_group_id, len(grpo_policy_items) - 1))
                    grpo_rollout_cost_values.append(grpo_cost)
                    baseline_cost = traj.get("grpo_rollout_baseline_cost", float("nan"))
                    grpo_rollout_baseline_cost_values.append(float(baseline_cost))
                    grpo_rollout_group_ids.append(grpo_group_id)
                if torch.is_tensor(grpo_rollout_logprob):
                    grpo_rollout_logprob_values.append(float(grpo_rollout_logprob.detach().cpu().item()))
                grpo_rollout_selected_nonzero_fracs.append(float(traj.get("grpo_rollout_selected_nonzero_frac", 0.0)))
            entropies.append(entropy.detach())
            used_traj_ids.append(traj_id)
        else:
            skipped_actor_trajs += 1
            if actor_update_disabled:
                reason = "actor_update_disabled"
            elif low_valid_agent_coverage:
                reason = "low_valid_agent_coverage"
            elif float(traj.get("n_actor_steps", 0.0)) <= 0.0:
                reason = "no_actor_steps"
            elif float(traj.get("n_valid_path_agents", 0.0)) <= 0.0:
                reason = "no_valid_path_agents"
            elif bool(traj.get("restored_from_checkpoint", False)):
                reason = "restored_traj_no_grad_graph"
            elif (
                float(traj.get("reward_grad_steps", 0.0)) <= 0.0
                and float(traj.get("entropy_grad_steps", 0.0)) <= 0.0
                and route_cost_aux_loss is None
                and grpo_rollout_vehicle_cost_loss is None
                and not grpo_policy_has_samples
                and not discrete_ac_has_samples
            ):
                reason = "no_reward_entropy_or_aux_grad"
            elif float(traj.get("reward_grad_steps", 0.0)) <= 0.0:
                reason = "reward_path_no_grad"
            elif float(traj.get("entropy_grad_steps", 0.0)) <= 0.0:
                reason = "entropy_path_no_grad"
            else:
                reason = "unexpected_no_grad"
            skip_reason_counts[reason] = int(skip_reason_counts.get(reason, 0)) + 1
            skipped_traj_ids.append(traj_id)
        invalid_rates.append(float(traj.get("invalid_route_rate", 0.0)))
        route_churns.append(float(traj.get("route_churn", 0.0)))

    update_episode = max(
        [int(traj.get("episode_index", traj.get("traj_id", -1))) for traj in trajectories]
        or [-1]
    )
    consumed_fresh_traj_count = max(0, len(consumed_traj_ids) - int(consumed_restored_traj_count))
    skipped_restored_traj_count = int(skip_reason_counts.get("restored_traj_no_grad_graph", 0))
    skipped_fresh_traj_count = max(0, len(skipped_traj_ids) - skipped_restored_traj_count)

    if grpo_rollout_items:
        baseline_arr = np.asarray(grpo_rollout_baseline_cost_values, dtype=np.float32)
        if baseline_arr.size == len(grpo_rollout_cost_values) and np.isfinite(baseline_arr).all():
            advantages_np = _rollout_baseline_relative_advantages(
                grpo_rollout_cost_values,
                grpo_rollout_baseline_cost_values,
                grpo_rollout_group_ids,
                normalize=bool(getattr(cfg, "dreamer_grpo_rollout_normalize_advantages", True)),
            )
        else:
            advantages_np = _rollout_group_relative_advantages(
                grpo_rollout_cost_values,
                grpo_rollout_group_ids,
                normalize=bool(getattr(cfg, "dreamer_grpo_rollout_normalize_advantages", True)),
            )
        grpo_rollout_adv_values = [float(x) for x in advantages_np.cpu().numpy().tolist()]
    else:
        grpo_rollout_adv_values = []
    rollout_adv_by_policy_idx: Dict[int, float] = {}
    for (_, _, policy_idx), adv in zip(grpo_rollout_items, grpo_rollout_adv_values):
        rollout_adv_by_policy_idx[int(policy_idx)] = float(adv)

    if grpo_policy_items:
        combined_losses: List[torch.Tensor] = []
        for policy_idx, item in enumerate(grpo_policy_items):
            obs = item["obs"]
            selected = item["selected"]
            old_logprob = item["old_logprob"]
            valid = item["valid"]
            route_adv = item["route_adv"]
            if not (
                torch.is_tensor(obs)
                and torch.is_tensor(selected)
                and torch.is_tensor(old_logprob)
                and torch.is_tensor(valid)
                and torch.is_tensor(route_adv)
                and obs.numel() > 0
            ):
                continue
            logits_now = actor.forward_logits(obs)
            loss, stats = _combined_grpo_policy_loss(
                logits=logits_now,
                selected_idx=selected,
                old_log_probs=old_logprob,
                route_valid=valid,
                route_advantages=route_adv,
                rollout_advantage=rollout_adv_by_policy_idx.get(int(policy_idx), 0.0),
                route_weight=float(getattr(cfg, "dreamer_grpo_route_cost_weight", 0.0)),
                rollout_weight=float(getattr(cfg, "dreamer_grpo_rollout_weight", 0.0)),
                temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                policy_clip_ratio=float(getattr(cfg, "dreamer_grpo_policy_clip_ratio", 0.2)),
                advantage_clip=float(getattr(cfg, "dreamer_grpo_combined_advantage_clip", 5.0)),
                kl_coef=float(getattr(cfg, "dreamer_grpo_policy_kl_coef", 0.0)),
            )
            if loss is not None:
                combined_losses.append(loss)
                grpo_combined_loss_values.append(float(stats.get("grpo_combined_loss", 0.0)))
                grpo_combined_pg_loss_values.append(float(stats.get("grpo_combined_pg_loss", 0.0)))
                grpo_combined_kl_loss_values.append(float(stats.get("grpo_combined_kl_loss", 0.0)))
                grpo_combined_adv_means.append(float(stats.get("grpo_combined_adv_mean", 0.0)))
                grpo_combined_adv_stds.append(float(stats.get("grpo_combined_adv_std", 0.0)))
                grpo_combined_clip_fracs.append(float(stats.get("grpo_combined_clip_frac", 0.0)))
                grpo_policy_kl_values.append(float(stats.get("grpo_policy_kl", 0.0)))
        if combined_losses:
            actor_losses.append(torch.stack(combined_losses).mean())

    if not critic_losses and not vehicle_critic_losses:
        return {
            "pg_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "invalid_route_rate": 0.0,
            "route_churn": 0.0,
            "route_cost_aux_loss": 0.0,
            "route_cost_aux_acc": 0.0,
            "route_cost_aux_margin": 0.0,
            "expected_cost_loss": 0.0,
            "expected_cost_raw": 0.0,
            "expected_cost_best_prob": 0.0,
            "expected_cost_best_route0_frac": 0.0,
            "expected_cost_nonzero_best_frac": 0.0,
            "expected_cost_margin": 0.0,
            "grpo_route_cost_loss": 0.0,
            "grpo_route_cost_raw": 0.0,
            "grpo_route_cost_best_prob": 0.0,
            "grpo_route_cost_best_route0_frac": 0.0,
            "grpo_route_cost_nonzero_best_frac": 0.0,
            "grpo_route_cost_margin": 0.0,
            "grpo_route_cost_adv_mean": 0.0,
            "grpo_route_cost_adv_std": 0.0,
            "grpo_rollout_loss": 0.0,
            "grpo_combined_loss": 0.0,
            "grpo_combined_pg_loss": 0.0,
            "grpo_combined_kl_loss": 0.0,
            "grpo_combined_adv_mean": 0.0,
            "grpo_combined_adv_std": 0.0,
            "grpo_combined_clip_frac": 0.0,
            "grpo_policy_kl": 0.0,
            "grpo_rollout_cost": 0.0,
            "grpo_rollout_adv_mean": 0.0,
            "grpo_rollout_adv_std": 0.0,
            "grpo_rollout_logprob": 0.0,
            "grpo_rollout_selected_nonzero_frac": 0.0,
            "grpo_rollout_vehicle_cost_loss": 0.0,
            "grpo_rollout_vehicle_cost": 0.0,
            "grpo_rollout_vehicle_cost_adv_mean": 0.0,
            "grpo_rollout_vehicle_cost_adv_std": 0.0,
            "dreamer_ac_loss": 0.0,
            "dreamer_ac_adv_mean": 0.0,
            "dreamer_ac_adv_std": 0.0,
            "dreamer_ac_adv_min": 0.0,
            "dreamer_ac_adv_max": 0.0,
            "dreamer_ac_route_adv_mean": 0.0,
            "dreamer_ac_route_adv_std": 0.0,
            "dreamer_ac_combined_adv_mean": 0.0,
            "dreamer_ac_combined_adv_std": 0.0,
            "dreamer_ac_selected_nonzero_frac": 0.0,
            "dreamer_ac_selected_route0_frac": 0.0,
            "dreamer_ac_selected_route1_frac": 0.0,
            "dreamer_ac_selected_route2_frac": 0.0,
            "dreamer_ac_selected_route3_frac": 0.0,
            "dreamer_ac_policy_argmax_nonzero_frac": 0.0,
            "dreamer_ac_policy_argmax_route0_frac": 0.0,
            "dreamer_ac_policy_argmax_route1_frac": 0.0,
            "dreamer_ac_policy_argmax_route2_frac": 0.0,
            "dreamer_ac_policy_argmax_route3_frac": 0.0,
            "dreamer_ac_policy_argmax_max_route_frac": 0.0,
            "dreamer_ac_policy_top1_prob_mean": 0.0,
            "dreamer_actor_gate_skipped": 0.0,
            "dreamer_actor_gate_max_route_frac": 0.0,
            "dreamer_actor_gate_route0_frac": 0.0,
            "dreamer_actor_gate_top1_prob": 0.0,
            "dreamer_actor_gate_sample_count": 0.0,
            "dreamer_actor_gate_latched": 0.0,
            "dreamer_actor_trust_region_accepted": 0.0,
            "dreamer_actor_trust_region_kl": 0.0,
            "dreamer_actor_trust_region_retries": 0.0,
            "dreamer_actor_trust_region_sample_count": 0.0,
            "dreamer_actor_trust_region_argmax_change_frac": 0.0,
            "dreamer_actor_trust_region_vehicle_argmax_change_max": 0.0,
            "dreamer_actor_trust_region_vehicle_argmax_change_mean": 0.0,
            "dreamer_actor_trust_region_vehicle_argmax_change_p95": 0.0,
            "dreamer_actor_trust_region_vehicle_group_count": 0.0,
            "dreamer_actor_trust_region_argmax_max_route_frac": 0.0,
            "dreamer_actor_trust_region_argmax_route0_frac": 0.0,
            "dreamer_actor_trust_region_top1_prob": 0.0,
            "dreamer_actor_anchor_active": 0.0,
            "dreamer_actor_anchor_accepted": 0.0,
            "dreamer_actor_anchor_kl": 0.0,
            "dreamer_actor_anchor_argmax_change_frac": 0.0,
            "dreamer_actor_anchor_vehicle_argmax_change_max": 0.0,
            "dreamer_actor_anchor_vehicle_argmax_change_mean": 0.0,
            "dreamer_actor_anchor_vehicle_argmax_change_p95": 0.0,
            "dreamer_actor_anchor_vehicle_group_count": 0.0,
            "dreamer_actor_anchor_argmax_max_route_frac": 0.0,
            "dreamer_actor_anchor_argmax_route0_frac": 0.0,
            "dreamer_actor_anchor_top1_prob": 0.0,
            "dreamer_actor_anchor_sample_count": 0.0,
            "dreamer_ac_sample_count": 0.0,
            "dreamer_ac_policy_kl": 0.0,
            "dreamer_ac_clip_frac": 0.0,
            "dreamer_ac_weight_mean": 0.0,
            "dreamer_ac_uses_sample_adv": 0.0,
            "dreamer_ac_entropy_loss": 0.0,
            "dreamer_ac_entropy_floor_loss": 0.0,
            "dreamer_ac_entropy_target": 0.0,
            "dreamer_ac_entropy_valid_target_mean": 0.0,
            "dreamer_ac_entropy_effective_route_count": 0.0,
            "dreamer_ac_entropy_cost_mask_frac": 0.0,
            "vehicle_value_loss": 0.0,
            "vehicle_critic_target_mean": 0.0,
            "vehicle_critic_target_std": 0.0,
            "vehicle_critic_explained_variance": 0.0,
            "vehicle_return_norm_mean": 0.0,
            "vehicle_return_norm_std": 1.0,
            "vehicle_reward_mean": 0.0,
            "vehicle_reward_std": 0.0,
            "vehicle_route_progress_reward_mean": 0.0,
            "vehicle_route_progress_reward_std": 0.0,
            "vehicle_route_credit_reward_mean": 0.0,
            "vehicle_route_credit_reward_std": 0.0,
            "vehicle_social_reward_mean": 0.0,
            "vehicle_social_reward_std": 0.0,
            "vehicle_return_discount": float(vehicle_return_discount),
            "critic_target_mean": 0.0,
            "critic_target_std": 0.0,
            "critic_explained_variance": 0.0,
            "critic_return_discount": float(critic_return_discount),
            "return_norm_mean": 0.0,
            "return_norm_std": 1.0,
            "skipped_actor_trajs": float(skipped_actor_trajs),
            "skip_reason_counts": skip_reason_counts,
            "skipped_traj_ids": skipped_traj_ids,
            "used_traj_ids": used_traj_ids,
            "consumed_traj_ids": consumed_traj_ids,
            "consumed_fresh_traj_count": int(consumed_fresh_traj_count),
            "consumed_restored_traj_count": int(consumed_restored_traj_count),
            "used_fresh_traj_count": 0,
            "used_restored_traj_count": 0,
            "skipped_fresh_traj_count": int(skipped_fresh_traj_count),
            "skipped_restored_traj_count": int(skipped_restored_traj_count),
            "update_id": int(update_id) if update_id is not None else -1,
        }

    if critic_losses:
        critic_optimizer.zero_grad()
        critic_loss_total = torch.stack(critic_losses).mean()
        critic_loss_total.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optimizer.step()
        if target_critic is not None and float(getattr(cfg, "dreamer_slow_critic_tau", 0.0)) > 0.0:
            _polyak_update_module(target_critic, critic, tau=float(cfg.dreamer_slow_critic_tau))
    elif vehicle_critic_losses:
        critic_loss_total = torch.zeros(
            (),
            device=vehicle_critic_losses[0].device,
            dtype=vehicle_critic_losses[0].dtype,
        )
    else:
        critic_ref = next(critic.parameters())
        critic_loss_total = torch.zeros((), device=critic_ref.device, dtype=critic_ref.dtype)

    if vehicle_critic_losses and vehicle_critic is not None and vehicle_critic_optimizer is not None:
        vehicle_critic_optimizer.zero_grad()
        vehicle_critic_loss_total = torch.stack(vehicle_critic_losses).mean()
        vehicle_critic_loss_total.backward()
        nn.utils.clip_grad_norm_(vehicle_critic.parameters(), 1.0)
        vehicle_critic_optimizer.step()
        if target_vehicle_critic is not None and float(getattr(cfg, "dreamer_slow_critic_tau", 0.0)) > 0.0:
            _polyak_update_module(
                target_vehicle_critic,
                vehicle_critic,
                tau=float(cfg.dreamer_slow_critic_tau),
            )
    else:
        vehicle_critic_loss_total = torch.zeros((), device=critic_loss_total.device, dtype=critic_loss_total.dtype)

    if q_actor_items:
        old_requires_grad = [p.requires_grad for p in critic.parameters()]
        for p in critic.parameters():
            p.requires_grad_(False)
        try:
            for current_z, current_action_for_actor, current_context, entropy, route_cost_aux_loss in q_actor_items:
                q_values = critic(
                    current_z,
                    action=current_action_for_actor,
                    context=current_context,
                ).squeeze(-1)
                q_actor_loss = -float(getattr(cfg, "dreamer_actor_q_weight", 1.0)) * q_values.mean()
                q_actor_loss = q_actor_loss - float(cfg.dreamer_entropy_coef) * entropy
                if route_cost_aux_loss is not None:
                    q_actor_loss = q_actor_loss + float(cfg.dreamer_route_cost_aux_weight) * route_cost_aux_loss
                actor_losses.append(q_actor_loss)
        finally:
            for p, requires_grad in zip(critic.parameters(), old_requires_grad):
                p.requires_grad_(requires_grad)

    gate_max_route_frac = (
        float(np.mean(dreamer_ac_policy_argmax_max_route_fracs))
        if dreamer_ac_policy_argmax_max_route_fracs else 0.0
    )
    gate_route0_frac = (
        float(np.mean(dreamer_ac_policy_argmax_route0_fracs))
        if dreamer_ac_policy_argmax_route0_fracs else 0.0
    )
    gate_top1_prob = (
        float(np.mean(dreamer_ac_policy_top1_prob_means))
        if dreamer_ac_policy_top1_prob_means else 0.0
    )
    gate_sample_count = float(np.mean(dreamer_ac_sample_counts)) if dreamer_ac_sample_counts else 0.0
    actor_gate_reasons = _actor_update_gate_reasons(
        cfg,
        episode=int(update_episode),
        max_route_frac=gate_max_route_frac,
        route0_frac=gate_route0_frac,
        top1_prob=gate_top1_prob,
        sample_count=gate_sample_count,
        gate_latched=(
            bool(actor_gate_state.get("latched", False))
            if actor_gate_state is not None else False
        ),
    )
    actor_gate_skip = bool(actor_gate_reasons)
    if (
        actor_gate_skip
        and actor_gate_state is not None
        and bool(getattr(cfg, "dreamer_actor_update_gate_latch", False))
    ):
        actor_gate_state["latched"] = True
    if actor_gate_skip and actor_losses:
        skipped_by_gate = max(1, len(actor_losses))
        skipped_actor_trajs += skipped_by_gate
        reason = "actor_update_gate:" + ",".join(actor_gate_reasons)
        skip_reason_counts[reason] = int(skip_reason_counts.get(reason, 0)) + skipped_by_gate

    actor_trust_region_accepted = 0.0
    actor_trust_region_kl = 0.0
    actor_trust_region_retries = 0.0
    actor_trust_region_sample_count = 0.0
    actor_trust_region_argmax_change_frac = 0.0
    actor_trust_region_vehicle_argmax_change_max = 0.0
    actor_trust_region_vehicle_argmax_change_mean = 0.0
    actor_trust_region_vehicle_argmax_change_p95 = 0.0
    actor_trust_region_vehicle_group_count = 0.0
    actor_trust_region_argmax_max_route_frac = 0.0
    actor_trust_region_argmax_route0_frac = 0.0
    actor_trust_region_top1_prob = 0.0
    actor_anchor_active = 0.0
    actor_anchor_accepted = 0.0
    actor_anchor_kl = 0.0
    actor_anchor_argmax_change_frac = 0.0
    actor_anchor_vehicle_argmax_change_max = 0.0
    actor_anchor_vehicle_argmax_change_mean = 0.0
    actor_anchor_vehicle_argmax_change_p95 = 0.0
    actor_anchor_vehicle_group_count = 0.0
    actor_anchor_argmax_max_route_frac = 0.0
    actor_anchor_argmax_route0_frac = 0.0
    actor_anchor_top1_prob = 0.0
    actor_anchor_sample_count = 0.0

    if actor_losses and not actor_gate_skip:
        actor_loss_total = torch.stack(actor_losses).mean()
        trust_region_kl_limit = float(getattr(cfg, "dreamer_actor_trust_region_kl", 0.0))
        trust_region_snapshot: Dict[str, object] = {"sample_count": 0.0}
        trust_region_enabled = False
        trust_region_argmax_change_limit = float(
            getattr(cfg, "dreamer_actor_trust_region_max_argmax_change_frac", 0.0)
        )
        trust_region_vehicle_argmax_change_limit = float(
            getattr(cfg, "dreamer_actor_trust_region_max_vehicle_argmax_change_frac", 0.0)
        )
        trust_region_max_route_limit = float(
            getattr(cfg, "dreamer_actor_trust_region_max_route_frac", 0.0)
        )
        trust_region_route0_limit = float(
            getattr(cfg, "dreamer_actor_trust_region_max_route0_frac", 0.0)
        )
        if (
            (
                trust_region_kl_limit > 0.0
                or trust_region_argmax_change_limit > 0.0
                or trust_region_vehicle_argmax_change_limit > 0.0
                or trust_region_max_route_limit > 0.0
                or trust_region_route0_limit > 0.0
            )
            and use_discrete_actor_critic
        ):
            trust_region_snapshot = dreamer_pg.discrete_ac_policy_snapshot(
                actor,
                trajectories,
                temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                max_samples=int(getattr(cfg, "dreamer_actor_trust_region_max_samples", 4096)),
            )
            actor_trust_region_sample_count = float(trust_region_snapshot.get("sample_count", 0.0))
            trust_region_enabled = (
                actor_trust_region_sample_count
                >= float(getattr(cfg, "dreamer_actor_trust_region_min_samples", 128))
            )
        anchor_snapshot: Dict[str, object] = {"sample_count": 0.0}
        anchor_enabled = False
        anchor_start = int(getattr(cfg, "dreamer_actor_anchor_start_episode", -1))
        if (
            _actor_anchor_enabled(cfg)
            and actor_anchor is not None
            and use_discrete_actor_critic
            and int(update_episode) >= max(anchor_start, 0)
        ):
            actor_anchor.eval()
            anchor_snapshot = dreamer_pg.discrete_ac_policy_snapshot(
                actor_anchor,
                trajectories,
                temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                max_samples=int(getattr(cfg, "dreamer_actor_trust_region_max_samples", 4096)),
            )
            actor_anchor_sample_count = float(anchor_snapshot.get("sample_count", 0.0))
            anchor_enabled = (
                actor_anchor_sample_count
                >= float(getattr(cfg, "dreamer_actor_trust_region_min_samples", 128))
            )
            actor_anchor_active = 1.0 if anchor_enabled else 0.0

        if trust_region_enabled or anchor_enabled:
            actor_state_before = copy.deepcopy(actor.state_dict())
            actor_optimizer_state_before = copy.deepcopy(actor_optimizer.state_dict())
            optimizer_lrs_before = [float(group.get("lr", 0.0)) for group in actor_optimizer.param_groups]
            actor_optimizer.zero_grad()
            actor_loss_total.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            saved_actor_grads = [
                None if p.grad is None else p.grad.detach().clone()
                for p in actor.parameters()
            ]
            max_retries = max(int(getattr(cfg, "dreamer_actor_trust_region_max_retries", 0)), 0)
            lr_shrink = float(getattr(cfg, "dreamer_actor_trust_region_lr_shrink", 0.5))
            if not np.isfinite(lr_shrink) or lr_shrink <= 0.0 or lr_shrink >= 1.0:
                lr_shrink = 0.5
            accepted = False
            last_anchor_accepted = True
            for attempt in range(max_retries + 1):
                actor.load_state_dict(actor_state_before)
                actor_optimizer.load_state_dict(actor_optimizer_state_before)
                _restore_optimizer_lr(
                    actor_optimizer,
                    [lr * (lr_shrink ** int(attempt)) for lr in optimizer_lrs_before],
                )
                actor_optimizer.zero_grad(set_to_none=True)
                for p, grad in zip(actor.parameters(), saved_actor_grads):
                    if grad is not None:
                        p.grad = grad.to(device=p.device, dtype=p.dtype).clone()
                actor_optimizer.step()
                if trust_region_enabled:
                    shift_stats = dreamer_pg.discrete_ac_policy_shift_from_snapshot(
                        actor,
                        trust_region_snapshot,
                        temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                        group_min_samples=1,
                    )
                    actor_trust_region_kl = float(shift_stats.get("kl", 0.0))
                    actor_trust_region_argmax_change_frac = float(
                        shift_stats.get("argmax_change_frac", 0.0)
                    )
                    actor_trust_region_vehicle_argmax_change_max = float(
                        shift_stats.get("vehicle_argmax_change_max", 0.0)
                    )
                    actor_trust_region_vehicle_argmax_change_mean = float(
                        shift_stats.get("vehicle_argmax_change_mean", 0.0)
                    )
                    actor_trust_region_vehicle_argmax_change_p95 = float(
                        shift_stats.get("vehicle_argmax_change_p95", 0.0)
                    )
                    actor_trust_region_vehicle_group_count = float(
                        shift_stats.get("vehicle_group_count", 0.0)
                    )
                    actor_trust_region_argmax_max_route_frac = float(
                        shift_stats.get("argmax_max_route_frac", 0.0)
                    )
                    actor_trust_region_argmax_route0_frac = float(
                        shift_stats.get("argmax_route0_frac", 0.0)
                    )
                    actor_trust_region_top1_prob = float(shift_stats.get("top1_prob_mean", 0.0))
                    trust_accepted = _actor_trust_region_accepts(
                        kl=actor_trust_region_kl,
                        argmax_change_frac=actor_trust_region_argmax_change_frac,
                        vehicle_argmax_change_max=actor_trust_region_vehicle_argmax_change_max,
                        vehicle_group_count=actor_trust_region_vehicle_group_count,
                        new_argmax_max_route_frac=actor_trust_region_argmax_max_route_frac,
                        old_argmax_max_route_frac=float(
                            trust_region_snapshot.get("old_argmax_max_route_frac", 0.0)
                        ),
                        new_argmax_route0_frac=actor_trust_region_argmax_route0_frac,
                        old_argmax_route0_frac=float(
                            trust_region_snapshot.get("old_argmax_route0_frac", 0.0)
                        ),
                        kl_limit=trust_region_kl_limit,
                        argmax_change_limit=trust_region_argmax_change_limit,
                        vehicle_argmax_change_limit=trust_region_vehicle_argmax_change_limit,
                        max_route_frac_limit=trust_region_max_route_limit,
                        route0_frac_limit=trust_region_route0_limit,
                    )
                else:
                    trust_accepted = True
                actor_trust_region_retries = float(attempt)
                if anchor_enabled:
                    anchor_shift_stats = dreamer_pg.discrete_ac_policy_shift_from_snapshot(
                        actor,
                        anchor_snapshot,
                        temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                        group_min_samples=1,
                    )
                    actor_anchor_kl = float(anchor_shift_stats.get("kl", 0.0))
                    actor_anchor_argmax_change_frac = float(
                        anchor_shift_stats.get("argmax_change_frac", 0.0)
                    )
                    actor_anchor_vehicle_argmax_change_max = float(
                        anchor_shift_stats.get("vehicle_argmax_change_max", 0.0)
                    )
                    actor_anchor_vehicle_argmax_change_mean = float(
                        anchor_shift_stats.get("vehicle_argmax_change_mean", 0.0)
                    )
                    actor_anchor_vehicle_argmax_change_p95 = float(
                        anchor_shift_stats.get("vehicle_argmax_change_p95", 0.0)
                    )
                    actor_anchor_vehicle_group_count = float(
                        anchor_shift_stats.get("vehicle_group_count", 0.0)
                    )
                    actor_anchor_argmax_max_route_frac = float(
                        anchor_shift_stats.get("argmax_max_route_frac", 0.0)
                    )
                    actor_anchor_argmax_route0_frac = float(
                        anchor_shift_stats.get("argmax_route0_frac", 0.0)
                    )
                    actor_anchor_top1_prob = float(anchor_shift_stats.get("top1_prob_mean", 0.0))
                    anchor_accepted = _actor_trust_region_accepts(
                        kl=actor_anchor_kl,
                        argmax_change_frac=actor_anchor_argmax_change_frac,
                        vehicle_argmax_change_max=actor_anchor_vehicle_argmax_change_max,
                        vehicle_group_count=actor_anchor_vehicle_group_count,
                        new_argmax_max_route_frac=actor_anchor_argmax_max_route_frac,
                        old_argmax_max_route_frac=float(
                            anchor_snapshot.get("old_argmax_max_route_frac", 0.0)
                        ),
                        new_argmax_route0_frac=actor_anchor_argmax_route0_frac,
                        old_argmax_route0_frac=float(
                            anchor_snapshot.get("old_argmax_route0_frac", 0.0)
                        ),
                        kl_limit=float(getattr(cfg, "dreamer_actor_anchor_kl", 0.0)),
                        argmax_change_limit=float(
                            getattr(cfg, "dreamer_actor_anchor_max_argmax_change_frac", 0.0)
                        ),
                        vehicle_argmax_change_limit=float(
                            getattr(cfg, "dreamer_actor_anchor_max_vehicle_argmax_change_frac", 0.0)
                        ),
                        max_route_frac_limit=float(
                            getattr(cfg, "dreamer_actor_anchor_max_route_frac", 0.0)
                        ),
                        route0_frac_limit=float(
                            getattr(cfg, "dreamer_actor_anchor_max_route0_frac", 0.0)
                        ),
                    )
                    last_anchor_accepted = bool(anchor_accepted)
                    actor_anchor_accepted = 1.0 if anchor_accepted else 0.0
                else:
                    actor_anchor_accepted = 1.0
                    anchor_accepted = True
                    last_anchor_accepted = True
                if trust_accepted and anchor_accepted:
                    accepted = True
                    actor_anchor_accepted = 1.0 if anchor_enabled else actor_anchor_accepted
                    break
            if accepted:
                actor_trust_region_accepted = 1.0
                _restore_optimizer_lr(actor_optimizer, optimizer_lrs_before)
            else:
                actor.load_state_dict(actor_state_before)
                actor_optimizer.load_state_dict(actor_optimizer_state_before)
                _restore_optimizer_lr(actor_optimizer, optimizer_lrs_before)
                actor_loss_total = torch.zeros(
                    (),
                    device=critic_loss_total.device,
                    dtype=critic_loss_total.dtype,
                )
                skipped_actor_trajs += max(1, len(actor_losses))
                if anchor_enabled and not last_anchor_accepted:
                    reason = "actor_anchor_reject"
                else:
                    reason = "actor_trust_region_reject"
                skip_reason_counts[reason] = int(skip_reason_counts.get(reason, 0)) + max(1, len(actor_losses))
        else:
            actor_optimizer.zero_grad()
            actor_loss_total.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()
            actor_trust_region_accepted = 1.0
            actor_anchor_accepted = 1.0
    else:
        actor_loss_total = torch.zeros((), device=critic_loss_total.device, dtype=critic_loss_total.dtype)

    return {
        "pg_loss": float(actor_loss_total.detach().cpu().item()),
        "value_loss": float(critic_loss_total.detach().cpu().item()),
        "entropy": float(torch.stack(entropies).mean().cpu().item()) if entropies else 0.0,
        "approx_kl": float(np.mean(grpo_policy_kl_values)) if grpo_policy_kl_values else 0.0,
        "invalid_route_rate": float(np.mean(invalid_rates)) if invalid_rates else 0.0,
        "route_churn": float(np.mean(route_churns)) if route_churns else 0.0,
        "route_cost_aux_loss": float(np.mean(route_cost_aux_loss_values)) if route_cost_aux_loss_values else 0.0,
        "route_cost_aux_acc": float(np.mean(route_cost_aux_accs)) if route_cost_aux_accs else 0.0,
        "route_cost_aux_margin": float(np.mean(route_cost_aux_margins)) if route_cost_aux_margins else 0.0,
        "expected_cost_loss": float(np.mean(expected_cost_loss_values)) if expected_cost_loss_values else 0.0,
        "expected_cost_raw": float(np.mean(expected_cost_raw_values)) if expected_cost_raw_values else 0.0,
        "expected_cost_best_prob": float(np.mean(expected_cost_best_probs)) if expected_cost_best_probs else 0.0,
        "expected_cost_best_route0_frac": float(np.mean(expected_cost_best_route0_fracs)) if expected_cost_best_route0_fracs else 0.0,
        "expected_cost_nonzero_best_frac": float(np.mean(expected_cost_nonzero_best_fracs)) if expected_cost_nonzero_best_fracs else 0.0,
        "expected_cost_margin": float(np.mean(expected_cost_margins)) if expected_cost_margins else 0.0,
        "grpo_route_cost_loss": float(np.mean(grpo_route_cost_loss_values)) if grpo_route_cost_loss_values else 0.0,
        "grpo_route_cost_raw": float(np.mean(grpo_route_cost_raw_values)) if grpo_route_cost_raw_values else 0.0,
        "grpo_route_cost_best_prob": float(np.mean(grpo_route_cost_best_probs)) if grpo_route_cost_best_probs else 0.0,
        "grpo_route_cost_best_route0_frac": float(np.mean(grpo_route_cost_best_route0_fracs)) if grpo_route_cost_best_route0_fracs else 0.0,
        "grpo_route_cost_nonzero_best_frac": float(np.mean(grpo_route_cost_nonzero_best_fracs)) if grpo_route_cost_nonzero_best_fracs else 0.0,
        "grpo_route_cost_margin": float(np.mean(grpo_route_cost_margins)) if grpo_route_cost_margins else 0.0,
        "grpo_route_cost_adv_mean": float(np.mean(grpo_route_cost_adv_means)) if grpo_route_cost_adv_means else 0.0,
        "grpo_route_cost_adv_std": float(np.mean(grpo_route_cost_adv_stds)) if grpo_route_cost_adv_stds else 0.0,
        "grpo_rollout_loss": (
            float(np.mean([-adv * lp for adv, lp in zip(grpo_rollout_adv_values, grpo_rollout_logprob_values)]))
            if grpo_rollout_adv_values and grpo_rollout_logprob_values else 0.0
        ),
        "grpo_combined_loss": float(np.mean(grpo_combined_loss_values)) if grpo_combined_loss_values else 0.0,
        "grpo_combined_pg_loss": float(np.mean(grpo_combined_pg_loss_values)) if grpo_combined_pg_loss_values else 0.0,
        "grpo_combined_kl_loss": float(np.mean(grpo_combined_kl_loss_values)) if grpo_combined_kl_loss_values else 0.0,
        "grpo_combined_adv_mean": float(np.mean(grpo_combined_adv_means)) if grpo_combined_adv_means else 0.0,
        "grpo_combined_adv_std": float(np.mean(grpo_combined_adv_stds)) if grpo_combined_adv_stds else 0.0,
        "grpo_combined_clip_frac": float(np.mean(grpo_combined_clip_fracs)) if grpo_combined_clip_fracs else 0.0,
        "grpo_policy_kl": float(np.mean(grpo_policy_kl_values)) if grpo_policy_kl_values else 0.0,
        "grpo_rollout_cost": float(np.mean(grpo_rollout_cost_values)) if grpo_rollout_cost_values else 0.0,
        "grpo_rollout_adv_mean": float(np.mean(grpo_rollout_adv_values)) if grpo_rollout_adv_values else 0.0,
        "grpo_rollout_adv_std": float(np.std(grpo_rollout_adv_values)) if grpo_rollout_adv_values else 0.0,
        "grpo_rollout_logprob": float(np.mean(grpo_rollout_logprob_values)) if grpo_rollout_logprob_values else 0.0,
        "grpo_rollout_selected_nonzero_frac": float(np.mean(grpo_rollout_selected_nonzero_fracs)) if grpo_rollout_selected_nonzero_fracs else 0.0,
        "grpo_rollout_vehicle_cost_loss": float(np.mean(grpo_rollout_vehicle_cost_loss_values)) if grpo_rollout_vehicle_cost_loss_values else 0.0,
        "grpo_rollout_vehicle_cost": float(np.mean(grpo_rollout_vehicle_cost_values)) if grpo_rollout_vehicle_cost_values else 0.0,
        "grpo_rollout_vehicle_cost_adv_mean": float(np.mean(grpo_rollout_vehicle_cost_adv_means)) if grpo_rollout_vehicle_cost_adv_means else 0.0,
        "grpo_rollout_vehicle_cost_adv_std": float(np.mean(grpo_rollout_vehicle_cost_adv_stds)) if grpo_rollout_vehicle_cost_adv_stds else 0.0,
        "dreamer_ac_loss": float(np.mean(dreamer_ac_loss_values)) if dreamer_ac_loss_values else 0.0,
        "dreamer_ac_adv_mean": float(np.mean(dreamer_ac_adv_means)) if dreamer_ac_adv_means else 0.0,
        "dreamer_ac_adv_std": float(np.mean(dreamer_ac_adv_stds)) if dreamer_ac_adv_stds else 0.0,
        "dreamer_ac_adv_min": float(np.mean(dreamer_ac_adv_mins)) if dreamer_ac_adv_mins else 0.0,
        "dreamer_ac_adv_max": float(np.mean(dreamer_ac_adv_maxs)) if dreamer_ac_adv_maxs else 0.0,
        "dreamer_ac_route_adv_mean": (
            float(np.mean(dreamer_ac_route_adv_means)) if dreamer_ac_route_adv_means else 0.0
        ),
        "dreamer_ac_route_adv_std": (
            float(np.mean(dreamer_ac_route_adv_stds)) if dreamer_ac_route_adv_stds else 0.0
        ),
        "dreamer_ac_combined_adv_mean": (
            float(np.mean(dreamer_ac_combined_adv_means)) if dreamer_ac_combined_adv_means else 0.0
        ),
        "dreamer_ac_combined_adv_std": (
            float(np.mean(dreamer_ac_combined_adv_stds)) if dreamer_ac_combined_adv_stds else 0.0
        ),
        "dreamer_ac_selected_nonzero_frac": (
            float(np.mean(dreamer_ac_selected_nonzero_fracs))
            if dreamer_ac_selected_nonzero_fracs else 0.0
        ),
        "dreamer_ac_selected_route0_frac": (
            float(np.mean(dreamer_ac_selected_route0_fracs))
            if dreamer_ac_selected_route0_fracs else 0.0
        ),
        "dreamer_ac_selected_route1_frac": (
            float(np.mean(dreamer_ac_selected_route1_fracs))
            if dreamer_ac_selected_route1_fracs else 0.0
        ),
        "dreamer_ac_selected_route2_frac": (
            float(np.mean(dreamer_ac_selected_route2_fracs))
            if dreamer_ac_selected_route2_fracs else 0.0
        ),
        "dreamer_ac_selected_route3_frac": (
            float(np.mean(dreamer_ac_selected_route3_fracs))
            if dreamer_ac_selected_route3_fracs else 0.0
        ),
        "dreamer_ac_policy_argmax_nonzero_frac": (
            float(np.mean(dreamer_ac_policy_argmax_nonzero_fracs))
            if dreamer_ac_policy_argmax_nonzero_fracs else 0.0
        ),
        "dreamer_ac_policy_argmax_route0_frac": gate_route0_frac,
        "dreamer_ac_policy_argmax_route1_frac": (
            float(np.mean(dreamer_ac_policy_argmax_route1_fracs))
            if dreamer_ac_policy_argmax_route1_fracs else 0.0
        ),
        "dreamer_ac_policy_argmax_route2_frac": (
            float(np.mean(dreamer_ac_policy_argmax_route2_fracs))
            if dreamer_ac_policy_argmax_route2_fracs else 0.0
        ),
        "dreamer_ac_policy_argmax_route3_frac": (
            float(np.mean(dreamer_ac_policy_argmax_route3_fracs))
            if dreamer_ac_policy_argmax_route3_fracs else 0.0
        ),
        "dreamer_ac_policy_argmax_max_route_frac": gate_max_route_frac,
        "dreamer_ac_policy_top1_prob_mean": gate_top1_prob,
        "dreamer_actor_gate_skipped": 1.0 if actor_gate_skip else 0.0,
        "dreamer_actor_gate_max_route_frac": gate_max_route_frac,
        "dreamer_actor_gate_route0_frac": gate_route0_frac,
        "dreamer_actor_gate_top1_prob": gate_top1_prob,
        "dreamer_actor_gate_sample_count": gate_sample_count,
        "dreamer_actor_gate_latched": (
            1.0
            if actor_gate_state is not None and bool(actor_gate_state.get("latched", False))
            else 0.0
        ),
        "dreamer_actor_trust_region_accepted": actor_trust_region_accepted,
        "dreamer_actor_trust_region_kl": actor_trust_region_kl,
        "dreamer_actor_trust_region_retries": actor_trust_region_retries,
        "dreamer_actor_trust_region_sample_count": actor_trust_region_sample_count,
        "dreamer_actor_trust_region_argmax_change_frac": actor_trust_region_argmax_change_frac,
        "dreamer_actor_trust_region_vehicle_argmax_change_max": actor_trust_region_vehicle_argmax_change_max,
        "dreamer_actor_trust_region_vehicle_argmax_change_mean": actor_trust_region_vehicle_argmax_change_mean,
        "dreamer_actor_trust_region_vehicle_argmax_change_p95": actor_trust_region_vehicle_argmax_change_p95,
        "dreamer_actor_trust_region_vehicle_group_count": actor_trust_region_vehicle_group_count,
        "dreamer_actor_trust_region_argmax_max_route_frac": actor_trust_region_argmax_max_route_frac,
        "dreamer_actor_trust_region_argmax_route0_frac": actor_trust_region_argmax_route0_frac,
        "dreamer_actor_trust_region_top1_prob": actor_trust_region_top1_prob,
        "dreamer_actor_anchor_active": actor_anchor_active,
        "dreamer_actor_anchor_accepted": actor_anchor_accepted,
        "dreamer_actor_anchor_kl": actor_anchor_kl,
        "dreamer_actor_anchor_argmax_change_frac": actor_anchor_argmax_change_frac,
        "dreamer_actor_anchor_vehicle_argmax_change_max": actor_anchor_vehicle_argmax_change_max,
        "dreamer_actor_anchor_vehicle_argmax_change_mean": actor_anchor_vehicle_argmax_change_mean,
        "dreamer_actor_anchor_vehicle_argmax_change_p95": actor_anchor_vehicle_argmax_change_p95,
        "dreamer_actor_anchor_vehicle_group_count": actor_anchor_vehicle_group_count,
        "dreamer_actor_anchor_argmax_max_route_frac": actor_anchor_argmax_max_route_frac,
        "dreamer_actor_anchor_argmax_route0_frac": actor_anchor_argmax_route0_frac,
        "dreamer_actor_anchor_top1_prob": actor_anchor_top1_prob,
        "dreamer_actor_anchor_sample_count": actor_anchor_sample_count,
        "dreamer_ac_sample_count": float(np.mean(dreamer_ac_sample_counts)) if dreamer_ac_sample_counts else 0.0,
        "dreamer_ac_policy_kl": float(np.mean(dreamer_ac_policy_kls)) if dreamer_ac_policy_kls else 0.0,
        "dreamer_ac_clip_frac": float(np.mean(dreamer_ac_clip_fracs)) if dreamer_ac_clip_fracs else 0.0,
        "dreamer_ac_weight_mean": float(np.mean(dreamer_ac_weight_means)) if dreamer_ac_weight_means else 0.0,
        "dreamer_ac_uses_sample_adv": (
            float(np.mean(dreamer_ac_uses_sample_advs)) if dreamer_ac_uses_sample_advs else 0.0
        ),
        "dreamer_ac_entropy_loss": (
            float(np.mean(dreamer_ac_entropy_losses)) if dreamer_ac_entropy_losses else 0.0
        ),
        "dreamer_ac_entropy_floor_loss": (
            float(np.mean(dreamer_ac_entropy_floor_losses)) if dreamer_ac_entropy_floor_losses else 0.0
        ),
        "dreamer_ac_entropy_target": (
            float(np.mean(dreamer_ac_entropy_targets)) if dreamer_ac_entropy_targets else 0.0
        ),
        "dreamer_ac_entropy_valid_target_mean": (
            float(np.mean(dreamer_ac_entropy_valid_target_means))
            if dreamer_ac_entropy_valid_target_means else 0.0
        ),
        "dreamer_ac_entropy_effective_route_count": (
            float(np.mean(dreamer_ac_entropy_effective_route_counts))
            if dreamer_ac_entropy_effective_route_counts else 0.0
        ),
        "dreamer_ac_entropy_cost_mask_frac": (
            float(np.mean(dreamer_ac_entropy_cost_mask_fracs))
            if dreamer_ac_entropy_cost_mask_fracs else 0.0
        ),
        "vehicle_value_loss": float(vehicle_critic_loss_total.detach().cpu().item()),
        "vehicle_critic_target_mean": (
            float(np.mean(vehicle_critic_target_means)) if vehicle_critic_target_means else 0.0
        ),
        "vehicle_critic_target_std": (
            float(np.mean(vehicle_critic_target_stds)) if vehicle_critic_target_stds else 0.0
        ),
        "vehicle_critic_explained_variance": (
            float(np.mean(vehicle_critic_explained_variances))
            if vehicle_critic_explained_variances else 0.0
        ),
        "vehicle_return_norm_mean": (
            float(vehicle_return_normalizer.mean.detach().cpu().item())
            if vehicle_return_normalizer is not None else 0.0
        ),
        "vehicle_return_norm_std": (
            float(vehicle_return_normalizer.var.detach().clamp_min(1e-6).sqrt().cpu().item())
            if vehicle_return_normalizer is not None else 1.0
        ),
        "vehicle_reward_mean": float(np.mean(vehicle_reward_means)) if vehicle_reward_means else 0.0,
        "vehicle_reward_std": float(np.mean(vehicle_reward_stds)) if vehicle_reward_stds else 0.0,
        "vehicle_route_progress_reward_mean": (
            float(np.mean(vehicle_route_progress_reward_means))
            if vehicle_route_progress_reward_means else 0.0
        ),
        "vehicle_route_progress_reward_std": (
            float(np.mean(vehicle_route_progress_reward_stds))
            if vehicle_route_progress_reward_stds else 0.0
        ),
        "vehicle_route_credit_reward_mean": (
            float(np.mean(vehicle_route_credit_reward_means))
            if vehicle_route_credit_reward_means else 0.0
        ),
        "vehicle_route_credit_reward_std": (
            float(np.mean(vehicle_route_credit_reward_stds))
            if vehicle_route_credit_reward_stds else 0.0
        ),
        "vehicle_route_credit_gate_frac": (
            float(np.mean([
                float(traj.get("vehicle_route_credit_gate_frac", 0.0))
                for traj in trajectories
                if "vehicle_route_credit_gate_frac" in traj
            ]))
            if any("vehicle_route_credit_gate_frac" in traj for traj in trajectories) else 0.0
        ),
        "vehicle_route_credit_selected_gate_frac": (
            float(np.mean([
                float(traj.get("vehicle_route_credit_selected_gate_frac", 0.0))
                for traj in trajectories
                if "vehicle_route_credit_selected_gate_frac" in traj
            ]))
            if any("vehicle_route_credit_selected_gate_frac" in traj for traj in trajectories) else 0.0
        ),
        "vehicle_social_reward_mean": (
            float(np.mean(vehicle_social_reward_means)) if vehicle_social_reward_means else 0.0
        ),
        "vehicle_social_reward_std": (
            float(np.mean(vehicle_social_reward_stds)) if vehicle_social_reward_stds else 0.0
        ),
        "vehicle_return_discount": float(vehicle_return_discount),
        "critic_target_mean": float(np.mean(critic_target_means)) if critic_target_means else 0.0,
        "critic_target_std": float(np.mean(critic_target_stds)) if critic_target_stds else 0.0,
        "critic_explained_variance": (
            float(np.mean(critic_explained_variances)) if critic_explained_variances else 0.0
        ),
        "critic_return_discount": float(critic_return_discount),
        "return_norm_mean": (
            float(return_normalizer.mean.detach().cpu().item()) if return_normalizer is not None else 0.0
        ),
        "return_norm_std": (
            float(return_normalizer.var.detach().clamp_min(1e-6).sqrt().cpu().item())
            if return_normalizer is not None else 1.0
        ),
        "skipped_actor_trajs": float(skipped_actor_trajs),
        "skip_reason_counts": skip_reason_counts,
        "skipped_traj_ids": skipped_traj_ids,
        "used_traj_ids": used_traj_ids,
        "consumed_traj_ids": consumed_traj_ids,
        "consumed_fresh_traj_count": int(consumed_fresh_traj_count),
        "consumed_restored_traj_count": int(consumed_restored_traj_count),
        "used_fresh_traj_count": int(len(used_traj_ids)),
        "used_restored_traj_count": 0,
        "skipped_fresh_traj_count": int(skipped_fresh_traj_count),
        "skipped_restored_traj_count": int(skipped_restored_traj_count),
        "update_id": int(update_id) if update_id is not None else -1,
    }


def maybe_resume_dreamer_checkpoint(
    *,
    resume_ckpt,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    device,
    reset_actor_optimizer=False,
    target_critic=None,
    return_normalizer=None,
    vehicle_critic=None,
    vehicle_critic_optimizer=None,
    target_vehicle_critic=None,
    vehicle_return_normalizer=None,
):
    return _checkpoint_maybe_resume_dreamer_checkpoint(
        resume_ckpt=resume_ckpt,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        device=device,
        reset_actor_optimizer=reset_actor_optimizer,
        target_critic=target_critic,
        return_normalizer=return_normalizer,
        vehicle_critic=vehicle_critic,
        vehicle_critic_optimizer=vehicle_critic_optimizer,
        target_vehicle_critic=target_vehicle_critic,
        vehicle_return_normalizer=vehicle_return_normalizer,
        load_checkpoint_fn=load_dreamer_checkpoint,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg: TrainConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    _validate_legacy_actor_only_losses(cfg)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(
        "Runtime device: requested=%s resolved=%s cuda_available=%s visible_devices=%s current_cuda=%s",
        cfg.device,
        str(device),
        bool(torch.cuda.is_available()),
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        (int(torch.cuda.current_device()) if device.type == "cuda" else "cpu"),
    )
    rng    = np.random.default_rng(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ── offline dataset ────────────────────────────────────────────────────────
    logger.info("Loading offline dataset...")
    dataset = OfflineInitDataset(
        data_dir=cfg.data_dir,
        meta_file=cfg.meta_file,
        flow_glob=cfg.flow_glob,
        K=cfg.n_routes,
        hdv_internalized=cfg.hdv_internalized,
    )

    # ── world model (frozen) ──────────────────────────────────────────────────
    logger.info("Loading world model...")
    stackelberg_cfg = None
    if bool(cfg.use_stackelberg):
        stackelberg_av_count, stackelberg_hdv_count = _derive_agent_counts_from_csv(
            cfg.agents_csv or os.path.join(cfg.network_folder, "agents.csv"),
            float(cfg.ratio_machines),
        )
        stackelberg_cfg = {
            "enabled": True,
            "n_iters": int(cfg.stackelberg_n_iters),
            "damping": float(cfg.stackelberg_damping),
            "total_av_count": float(stackelberg_av_count),
            "total_hdv_count": float(stackelberg_hdv_count),
        }
    wm = UrbanWorldModel(
        edge_registry_path=cfg.edge_registry,
        flow_projector_ckpt=cfg.projector_ckpt,
        dynamics_ckpt=cfg.dynamics_ckpt,
        config={"schedule": cfg.wm_schedule, "schedule_d": cfg.wm_schedule_d},
        device=str(device),
        use_amp=cfg.use_amp,
        stackelberg_cfg=stackelberg_cfg,
        cfm_init_mode=cfg.wm_cfm_init_mode,
        policy_obs_source=cfg.wm_policy_obs_source,
    )
    logger.info(
        "World model capabilities: use_cfm=%s, hdv_policy_head=%s (mode=%s, dim=%d)",
        wm.use_cfm,
        "on" if wm.hdv_policy_dim > 0 else "off",
        getattr(wm, "hdv_policy_mode", "edge"),
        wm.hdv_policy_dim,
    )

    # ── Current training mainline: Dreamer CTDE vehicle routing ───────────────
    route_registry: Optional[PathTopology] = None
    if not (cfg.network_folder and cfg.network_name):
        raise ValueError("Vehicle-level Dreamer CTDE training requires --network_folder and --network_name.")

    import ast as _ast
    od_path = os.path.join(cfg.network_folder, f"od_{cfg.network_name}.txt")
    with open(od_path, "r") as f:
        od_data = _ast.literal_eval(f.read())
    agents_path = cfg.agents_csv or os.path.join(cfg.network_folder, "agents.csv")
    route_registry = build_lightweight_vehicle_path_topology(
        origins=list(od_data["origins"]),
        destinations=list(od_data["destinations"]),
        n_edges=int(wm.n_edges),
        K=int(cfg.n_routes),
        agents_csv=agents_path if os.path.exists(agents_path) else None,
        ratio_machines=float(cfg.ratio_machines),
        edge_to_idx=wm._registry._edge_to_idx,
        routes_csv=cfg.vehicle_routes_csv,
    )
    route_topology_av_count = int(route_registry.od_av_counts.sum())
    route_topology_hdv_count = int(route_registry.od_hdv_counts.sum())
    logger.info(
        "Built lightweight vehicle path topology: N_od=%d, K=%d, AVs=%d, HDVs=%d, route_edges=%d",
        route_registry.n_od,
        route_registry.K,
        route_topology_av_count,
        route_topology_hdv_count,
        int(np.count_nonzero(route_registry.route_masks)),
    )

    obs_dim = vehicle_mode_obs_dim(
        n_routes=cfg.n_routes,
        d_obs=wm.d_obs,
        include_global=bool(cfg.vehicle_obs_include_global),
        route_stats=str(cfg.vehicle_obs_route_stats),
        include_route_costs=bool(cfg.vehicle_obs_include_route_costs),
        include_fleet_pressure=bool(getattr(cfg, "vehicle_obs_include_fleet_pressure", False)),
    )
    logger.info(
        "Vehicle mode: obs_dim=%d, topology_av=%d, topology_hdv=%d, K=%d, global_obs=%s, route_stats=%s, route_cost_obs=%s, fleet_pressure_obs=%s",
        obs_dim,
        route_topology_av_count,
        route_topology_hdv_count,
        route_registry.K,
        bool(cfg.vehicle_obs_include_global),
        str(cfg.vehicle_obs_route_stats),
        bool(cfg.vehicle_obs_include_route_costs),
        bool(getattr(cfg, "vehicle_obs_include_fleet_pressure", False)),
    )
    wm_action_dim = int(getattr(wm, "action_dim", -1))
    raw_action_dim = int(getattr(wm, "raw_action_dim", -1) or -1)
    if not vehicle_mode_action_supported(
        n_edges=int(wm.n_edges),
        action_dim=wm_action_dim,
        raw_action_dim=raw_action_dim,
    ):
        raise RuntimeError(
            "agent_level=vehicle requires raw edge action support. "
            f"Expected raw_action_dim==n_edges ({wm.n_edges}), "
            f"got action_dim={wm_action_dim}, raw_action_dim={raw_action_dim}."
        )

    actor = VehicleRouteActor(
        obs_dim=obs_dim,
        n_routes=cfg.n_routes,
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        route_cost_logit_bias=(
            float(cfg.vehicle_actor_route_cost_logit_bias)
            if bool(cfg.vehicle_obs_include_route_costs)
            else 0.0
        ),
        architecture=str(getattr(cfg, "vehicle_actor_architecture", "mlp")),
        include_global=bool(getattr(cfg, "vehicle_obs_include_global", True)),
        include_route_costs=bool(getattr(cfg, "vehicle_obs_include_route_costs", False)),
        include_fleet_pressure=bool(getattr(cfg, "vehicle_obs_include_fleet_pressure", False)),
    ).to(device)
    critic = CentralizedLatentValueCritic(
        d_spatial=wm.d_spatial,
        action_dim=(int(wm.n_edges) if str(cfg.dreamer_critic_variant).lower() == "action_conditioned" else 0),
        context_dim=(
            int(getattr(cfg, "dreamer_critic_context_dim", 8))
            if bool(getattr(cfg, "dreamer_critic_context", True))
            else 0
        ),
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        aggregation=str(getattr(cfg, "dreamer_critic_aggregation", "mean")),
        n_attention_heads=int(getattr(cfg, "dreamer_critic_attention_heads", 4)),
    ).to(device)
    target_critic: Optional[CentralizedLatentValueCritic] = None
    if float(getattr(cfg, "dreamer_slow_critic_tau", 0.0)) > 0.0:
        target_critic = copy.deepcopy(critic).to(device)
        target_critic.eval()
        for param in target_critic.parameters():
            param.requires_grad_(False)
    vehicle_critic: Optional[CentralizedVehicleValueCritic] = None
    target_vehicle_critic: Optional[CentralizedVehicleValueCritic] = None
    if bool(getattr(cfg, "dreamer_vehicle_critic", True)):
        vehicle_critic = CentralizedVehicleValueCritic(
            obs_dim=obs_dim,
            context_dim=(
                int(getattr(cfg, "dreamer_critic_context_dim", 8))
                if bool(getattr(cfg, "dreamer_vehicle_critic_context", True))
                else 0
            ),
            hidden_dim=cfg.hidden_dim,
            n_hidden_layers=cfg.n_hidden_layers,
            architecture=str(getattr(cfg, "dreamer_vehicle_critic_architecture", "self_attention")),
            n_attention_heads=int(getattr(cfg, "dreamer_vehicle_critic_attention_heads", 4)),
        ).to(device)
        if float(getattr(cfg, "dreamer_slow_critic_tau", 0.0)) > 0.0:
            target_vehicle_critic = copy.deepcopy(vehicle_critic).to(device)
            target_vehicle_critic.eval()
            for param in target_vehicle_critic.parameters():
                param.requires_grad_(False)
    actor_anchor: Optional[VehicleRouteActor] = None
    if _actor_anchor_enabled(cfg):
        actor_anchor = copy.deepcopy(actor).to(device)
        actor_anchor.eval()
        for param in actor_anchor.parameters():
            param.requires_grad_(False)
    return_normalizer = DreamerReturnNormalizer(
        decay=float(getattr(cfg, "dreamer_return_normalizer_decay", 0.99)),
        enabled=bool(getattr(cfg, "dreamer_return_normalizer", True)),
        device=device,
    )
    vehicle_return_normalizer = DreamerReturnNormalizer(
        decay=float(getattr(cfg, "dreamer_return_normalizer_decay", 0.99)),
        enabled=bool(getattr(cfg, "dreamer_vehicle_critic_return_normalizer", True)),
        device=device,
    )
    freeze_module(wm)
    policy = actor
    actor_lr = float(cfg.dreamer_actor_lr) if float(getattr(cfg, "dreamer_actor_lr", 0.0)) > 0.0 else float(cfg.lr)
    critic_lr = float(cfg.dreamer_critic_lr) if float(getattr(cfg, "dreamer_critic_lr", 0.0)) > 0.0 else float(cfg.lr)
    actor_base_lr = float(actor_lr)
    vehicle_critic_lr = (
        float(cfg.dreamer_vehicle_critic_lr)
        if float(getattr(cfg, "dreamer_vehicle_critic_lr", 0.0)) > 0.0
        else critic_lr
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    vehicle_critic_optimizer = (
        torch.optim.Adam(vehicle_critic.parameters(), lr=vehicle_critic_lr)
        if vehicle_critic is not None
        else None
    )
    critic_return_discount = _dreamer_global_return_discount(cfg)
    logger.info(
        "Dreamer CTDE mainline active: actor_obs_dim=%d n_routes=%d d_spatial=%d batch_eps=%d actor_h=%d critic_variant=%s critic_aggregation=%s critic_context_dim=%d slow_critic_tau=%.4f return_norm=%s actor_lr=%.2e actor_lr_decay=%d->%d@%.3f critic_lr=%.2e vehicle_critic=%s vehicle_critic_context_dim=%d vehicle_critic_lr=%.2e critic_discount=%.4f vehicle_discount=%.4f",
        obs_dim,
        int(cfg.n_routes),
        int(wm.d_spatial),
        int(cfg.dreamer_batch_episodes),
        min(int(cfg.imagination_horizon), int(cfg.dreamer_actor_horizon)),
        str(cfg.dreamer_critic_variant),
        str(getattr(cfg, "dreamer_critic_aggregation", "mean")),
        int(getattr(critic, "context_dim", 0)),
        float(getattr(cfg, "dreamer_slow_critic_tau", 0.0)) if target_critic is not None else 0.0,
        str(bool(getattr(cfg, "dreamer_return_normalizer", True))),
        actor_lr,
        int(getattr(cfg, "dreamer_actor_lr_decay_start_episode", -1)),
        int(getattr(cfg, "dreamer_actor_lr_decay_end_episode", -1)),
        float(getattr(cfg, "dreamer_actor_lr_final_scale", 1.0)),
        critic_lr,
        str(vehicle_critic is not None),
        int(getattr(vehicle_critic, "context_dim", 0)) if vehicle_critic is not None else 0,
        vehicle_critic_lr,
        float(critic_return_discount),
        float(ROUTE_COST_RETURN_DISCOUNT),
    )

    horizon_proxy: Optional[ActionHorizonProxyReward] = None
    if str(getattr(cfg, "dreamer_horizon_proxy_ckpt", "") or "").strip():
        ckpt_path = str(cfg.dreamer_horizon_proxy_ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"dreamer_horizon_proxy_ckpt not found: {ckpt_path}")
        horizon_proxy = ActionHorizonProxyReward.from_checkpoint(ckpt_path, device)
        expected_dim = int(wm.n_edges) + 25
        if int(horizon_proxy.input_dim) != expected_dim:
            raise ValueError(
                f"horizon proxy was trained for input_dim={horizon_proxy.input_dim}, "
                f"but current network expects {expected_dim} (= n_edges + 25)."
            )
        logger.info(
            "Action horizon proxy loaded: ckpt=%s input_dim=%d reward_weight=%.4f",
            ckpt_path,
            int(horizon_proxy.input_dim),
            float(cfg.dreamer_horizon_proxy_reward_weight),
        )

    # ── optional resume ───────────────────────────────────────────────────────
    start_episode = 0
    stats_history: Dict[str, Any] = normalize_stats_history_defaults({})
    stats_history = normalize_stats_history_defaults(stats_history)

    target_episode = int(cfg.n_total_episodes)

    # ── training loop ─────────────────────────────────────────────────────────
    logger.info("Starting offline MARL (dreamer_ctde): ep %d -> %d", start_episode, target_episode)
    logger.info(
        "  Vehicle mode: topology_av=%d topology_hdv=%d K=%d H=%d episode_steps=%d path_penalty=%.2f reward_scale=%.3f",
        route_topology_av_count, route_topology_hdv_count, cfg.n_routes, int(cfg.imagination_horizon), int(cfg.episode_steps),
        float(cfg.vehicle_path_penalty_factor), float(cfg.vehicle_reward_scale),
    )

    rew_running_mean = 0.0
    rew_running_std  = 1.0
    rew_alpha = 0.05
    gawron_mode_imagined = imagined_gawron_mode(cfg.gawron_mode)
    use_sumo_synced_hdv_policy = False
    if cfg.gawron_mode == "routerl_blackbox" and not cfg.hdv_internalized:
        logger.info(
            "  gawron_mode=routerl_blackbox: imagined rollouts use RouteRL-compatible "
            "mean-field surrogate (mode=routerl); SUMO eval uses shared black-box humans."
        )
    if cfg.hdv_internalized:
        logger.info(
            "  HDV internalized mode enabled: imagined rollout conditions WM on AV-only actions."
        )
    if use_sumo_synced_hdv_policy:
        logger.info(
            "  HDV policy source for imagined rollout: SUMO-evaluated strategy "
            "(fixed within episode, synced after each eval)."
        )

    pending_bufs: list  = []
    last_update_stats: Dict[str, float] = {
        key: (1.0 if key == "return_norm_std" else 0.0)
        for key in DREAMER_UPDATE_SCALAR_KEYS
    }

    adaptive_penalty_coef = float(cfg.conservative_penalty_coef)
    # Pre-compute vehicle path cache for vehicle-level imagination
    _vehicle_path_cache: Optional[Union[VehiclePathCache, LazyVehiclePathCache]] = None
    _departure_schedule: Optional[Dict] = None
    _avg_route_steps: int = 7
    _avg_edge_time: float = 8.0
    _actor_start_step_candidates: Optional[List[int]] = None
    _hdv_background_edge: Optional[torch.Tensor] = None
    _hdv_background_route_prior: Optional[np.ndarray] = None
    _edge_tt_ref = np.ones((int(wm.n_edges),), dtype=np.float32)
    if route_registry is not None and wm._registry.adjacency is not None:
        import time as _time
        _pc_t0 = _time.time()
        _adj_list = PathTopology._build_adj_list(wm._registry.adjacency)
        # Use free-flow travel times as reference for path pre-computation
        _lengths = np.asarray(wm._registry.lengths, dtype=np.float64)
        _speeds = np.asarray(wm._registry.speed_limits, dtype=np.float64)
        _edge_tt_ref = np.where(_speeds > 0.0, _lengths / _speeds, 1e6)
        _dest_edges_unique = list(set(
            int(wm._registry._edge_to_idx.get(d, -1))
            for d in route_registry.destinations
        ))
        _dest_edges_unique = [d for d in _dest_edges_unique if d >= 0]
        _candidate_source = str(getattr(cfg, "vehicle_candidate_source", "current_to_dest")).strip().lower()
        if _candidate_source == "od_suffix":
            _vehicle_path_cache = None
            logger.info("Vehicle path cache disabled: using precomputed od_suffix candidate routes.")
        elif _candidate_source == "current_to_dest":
            _cache_mode = str(getattr(cfg, "vehicle_path_cache_mode", "lazy") or "lazy").strip().lower()
            if _cache_mode in {"lazy", "on_demand", "ondemand"}:
                _vehicle_path_cache = LazyVehiclePathCache(
                    adj_list=_adj_list,
                    n_edges=int(wm.n_edges),
                    K=int(cfg.n_routes),
                    penalty_factor=float(cfg.vehicle_path_penalty_factor),
                    max_entries=int(getattr(cfg, "vehicle_path_cache_max_entries", 250000)),
                )
                logger.info(
                    "Vehicle path cache enabled: current_edge->dest lazy candidates max_entries=%d",
                    int(getattr(cfg, "vehicle_path_cache_max_entries", 250000)),
                )
            elif _cache_mode in {"precompute", "eager", "full"}:
                _vehicle_path_cache = VehiclePathCache(
                    adj_list=_adj_list,
                    edge_tt_ref=_edge_tt_ref,
                    dest_edges_unique=_dest_edges_unique,
                    n_edges=int(wm.n_edges),
                    K=int(cfg.n_routes),
                    penalty_factor=float(cfg.vehicle_path_penalty_factor),
                )
                logger.info(
                    "Vehicle path cache enabled: current_edge->dest precomputed candidates destinations=%d entries=%d",
                    len(_dest_edges_unique),
                    len(getattr(_vehicle_path_cache, "_cache", {})),
                )
            else:
                raise ValueError(
                    "vehicle_path_cache_mode must be one of {'lazy', 'precompute'}, "
                    f"got {_cache_mode!r}"
                )
        else:
            raise ValueError(
                "vehicle_candidate_source must be one of {'current_to_dest', 'od_suffix'}, "
                f"got {_candidate_source!r}"
            )
        agents_path = cfg.agents_csv or os.path.join(cfg.network_folder, "agents.csv")
        if agents_path and os.path.exists(agents_path):
            _departure_cap_mode = str(getattr(cfg, "dreamer_departure_cap_mode", "cfg")).strip().lower()
            if _departure_cap_mode not in {"cfg", "none"}:
                raise ValueError(
                    "dreamer_departure_cap_mode must be one of {'cfg', 'none'}, "
                    f"got {_departure_cap_mode!r}"
                )
            if _departure_cap_mode == "cfg":
                logger.warning(
                    "dreamer_departure_cap_mode=cfg is deprecated and ignored; "
                    "MARL now derives all AV/HDV counts from the full departure schedule."
                )
            _max_av_count = None
            _max_hdv_count = None
            _departure_schedule = build_departure_schedule(
                agents_csv=agents_path,
                step_interval=float(cfg.step_interval),
                edge_to_idx=wm._registry._edge_to_idx,
                origins=list(route_registry.origins),
                destinations=list(route_registry.destinations),
                ratio_machines=float(cfg.ratio_machines),
                n_edges=int(wm.n_edges),
                max_av_count=_max_av_count,
                max_hdv_count=_max_hdv_count,
            )
            _sched_av_count = sum(int(d.get("n_av", 0)) for d in _departure_schedule.values())
            _sched_hdv_count = sum(int(d.get("n_hdv", 0)) for d in _departure_schedule.values())
            logger.info(
                "Vehicle departure schedule: cap_mode=%s max_av=%s max_hdv=%s scheduled_av=%d scheduled_hdv=%d",
                _departure_cap_mode,
                str(_max_av_count),
                str(_max_hdv_count),
                _sched_av_count,
                _sched_hdv_count,
            )
            _avg_edge_time = float(compute_avg_edge_time(wm._registry.lengths, wm._registry.speed_limits))
            route_lengths = [
                len(route)
                for od_routes in route_registry.route_edge_indices
                for route in od_routes
                if route
            ]
            if route_lengths:
                _avg_route_edges = float(np.mean(route_lengths))
                _avg_route_steps = int(
                    max(1, compute_avg_route_steps(_avg_route_edges, _avg_edge_time, float(cfg.step_interval)))
                )
            else:
                logger.info(
                    "Vehicle mode using lightweight path topology; route_masks are empty, "
                    "keeping fallback avg_route_steps=%d",
                    _avg_route_steps,
                )
            logger.info(
                "Vehicle dynamic imagination prepared: dep_steps=%d avg_edge_time=%.2fs avg_route_steps=%d",
                len(_departure_schedule), _avg_edge_time, _avg_route_steps,
            )
            _actor_start_step_candidates = candidate_actor_start_steps(
                departure_schedule=_departure_schedule,
                episode_steps=int(cfg.episode_steps),
                horizon=min(int(cfg.imagination_horizon), int(cfg.dreamer_actor_horizon)),
                avg_route_steps=_avg_route_steps,
            )
            if _actor_start_step_candidates:
                logger.info(
                    "Vehicle dynamic imagination start-step candidates prepared: %d/%d valid starts",
                    len(_actor_start_step_candidates),
                    max(1, int(cfg.episode_steps) - min(int(cfg.imagination_horizon), int(cfg.dreamer_actor_horizon))),
                )
            else:
                logger.warning(
                    "Vehicle dynamic imagination found no actor-qualified start steps; falling back to unconstrained random starts."
                )
            hdv_bg_mode = str(getattr(cfg, "dreamer_hdv_background_mode", "none") or "none").strip().lower()
            if hdv_bg_mode not in {"", "none", "off"}:
                if hdv_bg_mode in {"adapted_state", "episode_av_pressure_response", "episode_hdv_head_response"}:
                    bg_state_path = str(getattr(cfg, "dreamer_hdv_background_state", "") or "").strip()
                    hdv_route_fracs = hdv_response.load_hdv_route_fracs_from_state(
                        bg_state_path,
                        route_registry,
                        n_routes=int(cfg.n_routes),
                    )
                    if hdv_route_fracs is None:
                        logger.warning(
                            "HDV background mode adapted_state requested but no usable state was loaded; using uniform HDV routes."
                        )
                        hdv_route_fracs = np.ones(
                            (int(route_registry.n_od), int(cfg.n_routes)),
                            dtype=np.float32,
                        ) / float(max(int(cfg.n_routes), 1))
                elif hdv_bg_mode == "uniform":
                    hdv_route_fracs = np.ones(
                        (int(route_registry.n_od), int(cfg.n_routes)),
                        dtype=np.float32,
                    ) / float(max(int(cfg.n_routes), 1))
                else:
                    raise ValueError(
                        "Unsupported dreamer_hdv_background_mode="
                        f"{hdv_bg_mode!r}; expected none, uniform, adapted_state, "
                        "episode_av_pressure_response, or episode_hdv_head_response"
                    )
                _hdv_background_route_prior = hdv_route_fracs.astype(np.float32, copy=True)
                hdv_bg_np = hdv_response.build_hdv_background_edge_action(
                    route_registry,
                    hdv_route_fracs,
                    n_edges=int(wm.n_edges),
                    edge_travel_times=_edge_tt_ref,
                    horizon_seconds=float(cfg.step_interval),
                )
                _hdv_background_edge = torch.from_numpy(hdv_bg_np).to(device=device, dtype=torch.float32)
                logger.info(
                    "HDV episode background enabled: mode=%s combine=%s weight=%.3f sum=%.4f max=%.4f active_edges=%d",
                    hdv_bg_mode,
                    str(getattr(cfg, "dreamer_hdv_background_combine", "additive")),
                    float(getattr(cfg, "dreamer_hdv_background_weight", 1.0)),
                    float(hdv_bg_np.sum()),
                    float(hdv_bg_np.max(initial=0.0)),
                    int(np.count_nonzero(hdv_bg_np > 0.0)),
                )
                if hdv_bg_mode in {"episode_av_pressure_response", "episode_hdv_head_response"}:
                    logger.info(
                        "HDV episode response enabled: mode=%s response=f(episode_AV_pressure) pressure_weight=%.3f temperature=%.3f prior_mix=%.3f fallback=%s",
                        hdv_bg_mode,
                        float(getattr(cfg, "dreamer_hdv_response_pressure_weight", 0.0)),
                        float(getattr(cfg, "dreamer_hdv_response_temperature", 60.0)),
                        float(getattr(cfg, "dreamer_hdv_response_prior_mix", 0.0)),
                        str(getattr(cfg, "dreamer_hdv_response_head_fallback", "analytic")),
                    )
        else:
            logger.warning("vehicle_mode enabled but agents.csv not found; dynamic imagination will use empty departure schedule")

        if actor is None or critic is None or actor_optimizer is None or critic_optimizer is None:
            raise RuntimeError("dreamer_ctde expected actor/critic/optimizers to be initialized")
        if bool(getattr(cfg, "dreamer_vehicle_critic", True)) and (
            vehicle_critic is None or vehicle_critic_optimizer is None
        ):
            raise RuntimeError("dreamer_vehicle_critic enabled but vehicle critic/optimizer was not initialized")
        pending_trajs: List[Dict[str, Any]] = []
        update_id = 0
        last_update_stats: Dict[str, float] = {
            key: (1.0 if key == "return_norm_std" else 0.0)
            for key in DREAMER_UPDATE_SCALAR_KEYS
        }
        last_update_stats.update({
            "invalid_route_rate": 0.0,
            "route_churn": 0.0,
            "skipped_actor_trajs": 0.0,
        })
        actor_gate_state: Dict[str, Any] = {"latched": False}
        actor_anchor_initialized = bool(
            actor_anchor is not None
            and int(getattr(cfg, "dreamer_actor_anchor_start_episode", -1)) < 0
        )

        if cfg.resume_ckpt and os.path.exists(cfg.resume_ckpt):
            resume_state = maybe_resume_dreamer_checkpoint(
                resume_ckpt=cfg.resume_ckpt,
                actor=actor,
                critic=critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                device=device,
                reset_actor_optimizer=bool(getattr(cfg, "resume_reset_actor_optimizer", False)),
                target_critic=target_critic,
                return_normalizer=return_normalizer,
                vehicle_critic=vehicle_critic,
                vehicle_critic_optimizer=vehicle_critic_optimizer,
                target_vehicle_critic=target_vehicle_critic,
                vehicle_return_normalizer=vehicle_return_normalizer,
            )
            if resume_state is not None:
                start_episode = int(resume_state["start_episode"])
                stats_history = resume_state["stats_history"]
                pending_trajs = list(resume_state["pending_trajs"])
                update_id = int(resume_state["update_id"])
                if resume_state.get("restored_rng") is not None:
                    rng = resume_state["restored_rng"]
                extra_state = dict(resume_state.get("extra_state", {}) or {})
                actor_anchor_state = extra_state.get("actor_anchor")
                if actor_anchor is not None and actor_anchor_state is not None:
                    try:
                        actor_anchor.load_state_dict(actor_anchor_state)
                        actor_anchor.to(device)
                        actor_anchor.eval()
                        for param in actor_anchor.parameters():
                            param.requires_grad_(False)
                        logger.info("Restored actor anchor state from Dreamer checkpoint extra_state.")
                    except RuntimeError as exc:
                        logger.warning(
                            "Failed to restore actor anchor state; using current actor as anchor. Error: %s",
                            exc,
                        )
                        actor_anchor.load_state_dict(actor.state_dict())
                if actor_anchor is not None:
                    actor_anchor_initialized = bool(
                        extra_state.get("actor_anchor_initialized", actor_anchor_state is not None)
                    )
        if start_episode > 0 and int(cfg.n_total_episodes) <= start_episode:
            logger.warning(
                "n_total_episodes=%d <= resumed Dreamer episode=%d; interpret as additional episodes "
                "(target episode -> %d).",
                int(cfg.n_total_episodes),
                start_episode,
                start_episode + int(cfg.n_total_episodes),
            )
            target_episode = start_episode + int(cfg.n_total_episodes)

        def _dreamer_extra_state() -> Dict[str, Any]:
            extra_state: Dict[str, Any] = {}
            if bool(cfg.dreamer_preserve_partial_batch_on_exit):
                extra_state["pending_trajs"] = _move_nested_tensors(pending_trajs, cpu=True)
            if actor_anchor is not None:
                extra_state["actor_anchor"] = _move_nested_tensors(actor_anchor.state_dict(), cpu=True)
                extra_state["actor_anchor_initialized"] = bool(actor_anchor_initialized)
            extra_state["update_id"] = int(update_id)
            extra_state.update(capture_rng_state(rng=rng))
            return extra_state

        for episode in range(start_episode, target_episode):
            actor_lr_episode = _scheduled_actor_lr(cfg, actor_base_lr, episode)
            _set_optimizer_lr(actor_optimizer, actor_lr_episode)
            actor.train()
            if (
                actor_anchor is not None
                and not bool(actor_anchor_initialized)
                and int(episode) >= max(int(getattr(cfg, "dreamer_actor_anchor_start_episode", -1)), 0)
            ):
                actor_anchor.load_state_dict(actor.state_dict())
                actor_anchor.eval()
                for param in actor_anchor.parameters():
                    param.requires_grad_(False)
                actor_anchor_initialized = True
                logger.info("Initialized actor anchor at episode %d.", int(episode))
            critic.train()
            if target_critic is not None:
                target_critic.eval()
            _episode_wall_t0 = _time.perf_counter()
            _rollout_wall_sec = 0.0
            _hdv_response_wall_sec = 0.0
            _baseline_wall_sec = 0.0
            _update_wall_sec = 0.0

            rollout_group_size = 1
            if float(getattr(cfg, "dreamer_grpo_rollout_weight", 0.0)) != 0.0:
                rollout_group_size = max(1, int(getattr(cfg, "dreamer_grpo_rollout_group_size", 1)))
            fixed_start_step = None
            if rollout_group_size > 1:
                T_group = min(int(cfg.imagination_horizon), int(cfg.dreamer_actor_horizon))
                if _actor_start_step_candidates:
                    fixed_start_step = int(
                        _actor_start_step_candidates[int(rng.integers(0, len(_actor_start_step_candidates)))]
                    )
                else:
                    fixed_start_step = int(rng.integers(0, max(1, int(cfg.episode_steps) - T_group)))

            traj_group: List[Dict[str, Any]] = []
            ep_reward_group: List[float] = []
            ep_metric_group: List[Dict[str, float]] = []
            hdv_response_mode = str(
                getattr(cfg, "dreamer_hdv_background_mode", "none") or "none"
            ).strip().lower()

            def _compute_episode_hdv_response_background(traj_id: int) -> Tuple[torch.Tensor, Dict[str, float]]:
                with torch.no_grad():
                    probe_traj, _, _ = imagined_episode_dynamic_dreamer(
                        wm=wm,
                        actor=actor,
                        dataset=dataset,
                        route_registry=route_registry,
                        cfg=cfg,
                        rng=rng,
                        device=device,
                        departure_schedule=_departure_schedule,
                        avg_route_steps=_avg_route_steps,
                        avg_edge_time=_avg_edge_time,
                        path_cache=_vehicle_path_cache,
                        traj_id=traj_id,
                        start_step_candidates=_actor_start_step_candidates,
                        fixed_start_step=fixed_start_step,
                        horizon_proxy=horizon_proxy,
                        hdv_background_edge=_hdv_background_edge,
                        episode_index=episode,
                    )
                episode_av_pressure = hdv_response.episode_av_pressure_from_action_edges(
                    list(probe_traj.get("action_edges", [])),
                    n_edges=int(wm.n_edges),
                )
                hdv_response_bg_np: Optional[np.ndarray] = None
                if hdv_response_mode == "episode_hdv_head_response":
                    head_pred = wm.predict_episode_hdv_response(
                        episode_av_pressure,
                        z_context=probe_traj["z_states"][0] if probe_traj.get("z_states") else None,
                        normalize=True,
                    )
                    if head_pred is not None:
                        pred_np = np.clip(head_pred.detach().cpu().numpy().astype(np.float32), 0.0, None)
                        pred_sum = float(pred_np.sum())
                        if pred_sum > 1e-8:
                            pred_np = pred_np / pred_sum
                            scale = float(_hdv_background_edge.detach().sum().cpu().item()) if _hdv_background_edge is not None else float(max(_avg_route_steps, 1))
                            hdv_response_bg_np = (pred_np * scale).astype(np.float32)
                if hdv_response_bg_np is None:
                    fallback = str(getattr(cfg, "dreamer_hdv_response_head_fallback", "analytic")).strip().lower()
                    if hdv_response_mode == "episode_hdv_head_response" and fallback in {"", "error", "raise"}:
                        raise RuntimeError(
                            "episode_hdv_head_response requested, but the world model has no usable episode HDV head"
                        )
                    if hdv_response_mode == "episode_hdv_head_response" and fallback in {"prior", "adapted_state"} and _hdv_background_edge is not None:
                        hdv_response_bg_np = _hdv_background_edge.detach().cpu().numpy().astype(np.float32)
                    else:
                        hdv_response_fracs = hdv_response.build_hdv_route_fracs_from_episode_av_pressure(
                            route_registry,
                            episode_av_pressure,
                            base_edge_tt=_edge_tt_ref,
                            n_routes=int(cfg.n_routes),
                            pressure_weight=float(getattr(cfg, "dreamer_hdv_response_pressure_weight", 60.0)),
                            temperature=float(getattr(cfg, "dreamer_hdv_response_temperature", 60.0)),
                            prior_route_fracs=_hdv_background_route_prior,
                            prior_mix=float(getattr(cfg, "dreamer_hdv_response_prior_mix", 0.0)),
                        )
                        hdv_response_bg_np = hdv_response.build_hdv_background_edge_action(
                            route_registry,
                            hdv_response_fracs,
                            n_edges=int(wm.n_edges),
                            edge_travel_times=_edge_tt_ref,
                            horizon_seconds=float(cfg.step_interval),
                        )
                hdv_response_bg = torch.from_numpy(hdv_response_bg_np).to(
                    device=device,
                    dtype=torch.float32,
                )
                hdv_response_metrics = {
                    "hdv_response_av_pressure_sum": float(episode_av_pressure.sum()),
                    "hdv_response_background_sum": float(hdv_response_bg_np.sum()),
                    "hdv_response_background_active_edges": float(np.count_nonzero(hdv_response_bg_np > 0.0)),
                }
                return hdv_response_bg, hdv_response_metrics

            shared_hdv_response_bg: Optional[torch.Tensor] = None
            shared_hdv_response_metrics: Dict[str, float] = {}
            if rollout_group_size > 1 and hdv_response_mode in {"episode_av_pressure_response", "episode_hdv_head_response"} and route_registry is not None:
                shared_hdv_response_bg, shared_hdv_response_metrics = _compute_episode_hdv_response_background(
                    traj_id=-(episode * rollout_group_size + 1)
                )
            for rollout_group_idx in range(rollout_group_size):
                hdv_background_for_rollout = _hdv_background_edge
                hdv_response_metrics: Dict[str, float] = {}
                if hdv_response_mode in {"episode_av_pressure_response", "episode_hdv_head_response"} and route_registry is not None:
                    if shared_hdv_response_bg is not None:
                        hdv_background_for_rollout = shared_hdv_response_bg
                        hdv_response_metrics = dict(shared_hdv_response_metrics)
                    else:
                        _hdv_t0 = _time.perf_counter()
                        hdv_background_for_rollout, hdv_response_metrics = _compute_episode_hdv_response_background(
                            traj_id=-(episode * rollout_group_size + rollout_group_idx + 1)
                        )
                        _hdv_response_wall_sec += _time.perf_counter() - _hdv_t0
                _rollout_t0 = _time.perf_counter()
                traj, ep_reward_i, ep_metrics_i = imagined_episode_dynamic_dreamer(
                    wm=wm,
                    actor=actor,
                    dataset=dataset,
                    route_registry=route_registry,
                    cfg=cfg,
                    rng=rng,
                    device=device,
                    departure_schedule=_departure_schedule,
                    avg_route_steps=_avg_route_steps,
                    avg_edge_time=_avg_edge_time,
                    path_cache=_vehicle_path_cache,
                    traj_id=episode * rollout_group_size + rollout_group_idx,
                    start_step_candidates=_actor_start_step_candidates,
                    fixed_start_step=fixed_start_step,
                    horizon_proxy=horizon_proxy,
                    hdv_background_edge=hdv_background_for_rollout,
                    episode_index=episode,
                )
                _rollout_wall_sec += _time.perf_counter() - _rollout_t0
                if hdv_response_metrics:
                    ep_metrics_i.update(hdv_response_metrics)
                    traj.update(hdv_response_metrics)
                traj["grpo_rollout_group_id"] = int(episode)
                traj["grpo_rollout_group_index"] = int(rollout_group_idx)
                traj["grpo_rollout_group_size"] = int(rollout_group_size)
                traj_group.append(traj)
                ep_reward_group.append(float(ep_reward_i))
                ep_metric_group.append(ep_metrics_i)

                ep_r = traj["rewards"].detach()
                if ep_r.numel() > 0:
                    rew_running_mean = (1 - rew_alpha) * rew_running_mean + rew_alpha * ep_r.mean().item()
                    rew_running_std = (1 - rew_alpha) * rew_running_std + rew_alpha * ep_r.std().clamp_min(1e-8).item()

            baseline_policy = str(getattr(cfg, "dreamer_grpo_rollout_baseline_policy", "none")).strip().lower()
            if (
                float(getattr(cfg, "dreamer_grpo_rollout_weight", 0.0)) != 0.0
                and baseline_policy not in {"", "none", "off"}
                and fixed_start_step is not None
            ):
                _baseline_t0 = _time.perf_counter()
                with torch.no_grad():
                    baseline_traj, _, baseline_metrics = imagined_episode_dynamic_dreamer(
                        wm=wm,
                        actor=actor,
                        dataset=dataset,
                        route_registry=route_registry,
                        cfg=cfg,
                        rng=rng,
                        device=device,
                        departure_schedule=_departure_schedule,
                        avg_route_steps=_avg_route_steps,
                        avg_edge_time=_avg_edge_time,
                        path_cache=_vehicle_path_cache,
                        traj_id=-(episode + 1),
                        start_step_candidates=_actor_start_step_candidates,
                        fixed_start_step=fixed_start_step,
                        rollout_policy_override=baseline_policy,
                        horizon_proxy=horizon_proxy,
                        hdv_background_edge=_hdv_background_edge,
                        episode_index=episode,
                    )
                baseline_cost = float(baseline_traj.get("grpo_rollout_cost", baseline_metrics.get("grpo_rollout_cost", 0.0)))
                for traj_i in traj_group:
                    traj_i["grpo_rollout_baseline_policy"] = baseline_policy
                    traj_i["grpo_rollout_baseline_cost"] = baseline_cost
                _baseline_wall_sec += _time.perf_counter() - _baseline_t0

            traj = traj_group[0]
            ep_reward = float(np.mean(ep_reward_group)) if ep_reward_group else 0.0
            ep_metrics = dict(ep_metric_group[0]) if ep_metric_group else {}
            for metric_key in set().union(*(m.keys() for m in ep_metric_group)):
                vals = []
                for metrics_i in ep_metric_group:
                    value_i = metrics_i.get(metric_key)
                    if isinstance(value_i, (int, float, np.integer, np.floating)) and np.isfinite(float(value_i)):
                        vals.append(float(value_i))
                if vals:
                    ep_metrics[metric_key] = float(np.mean(vals))

            pending_trajs.extend(traj_group)
            stats_history["ep_reward"].append(ep_reward)
            stats_history["t_av"].append(ep_metrics["t_av"])
            stats_history["t_hdv"].append(ep_metrics["t_hdv"])
            stats_history["t_all"].append(ep_metrics["t_all"])
            stats_history["hdv_drift"].append(ep_metrics["hdv_drift"])
            stats_history["hdv_head_entropy"].append(ep_metrics.get("hdv_head_entropy", 0.0))
            stats_history["terminal_t_all_pred"].append(ep_metrics.get("terminal_t_all_pred", float("nan")))
            stats_history["terminal_bonus"].append(ep_metrics.get("terminal_bonus", 0.0))
            stats_history["dreamer_base_reward"].append(ep_metrics.get("dreamer_base_reward", 0.0))
            stats_history["adaptive_penalty"].append(adaptive_penalty_coef)
            stats_history["invalid_route_rate"].append(ep_metrics.get("invalid_route_rate", 0.0))
            stats_history["route_churn"].append(ep_metrics.get("route_churn", 0.0))
            stats_history["route_cost_aux_loss"].append(ep_metrics.get("route_cost_aux_loss", 0.0))
            stats_history["route_cost_aux_acc"].append(ep_metrics.get("route_cost_aux_acc", 0.0))
            stats_history["route_cost_aux_margin"].append(ep_metrics.get("route_cost_aux_margin", 0.0))
            stats_history["expected_cost_loss"].append(ep_metrics.get("expected_cost_loss", 0.0))
            stats_history["expected_cost_raw"].append(ep_metrics.get("expected_cost_raw", 0.0))
            stats_history["expected_cost_best_prob"].append(ep_metrics.get("expected_cost_best_prob", 0.0))
            stats_history["expected_cost_best_route0_frac"].append(ep_metrics.get("expected_cost_best_route0_frac", 0.0))
            stats_history["expected_cost_nonzero_best_frac"].append(ep_metrics.get("expected_cost_nonzero_best_frac", 0.0))
            stats_history["expected_cost_margin"].append(ep_metrics.get("expected_cost_margin", 0.0))
            stats_history["avg_active_av"].append(ep_metrics.get("avg_active_av", 0.0))
            stats_history["max_active_av"].append(ep_metrics.get("max_active_av", 0.0))
            stats_history["n_valid_paths"].append(ep_metrics.get("n_valid_paths", 0.0))
            stats_history["n_valid_path_agents"].append(ep_metrics.get("n_valid_path_agents", 0.0))
            stats_history["n_actor_steps"].append(ep_metrics.get("n_actor_steps", 0.0))
            stats_history["reward_grad_steps"].append(ep_metrics.get("reward_grad_steps", 0.0))
            stats_history["entropy_grad_steps"].append(ep_metrics.get("entropy_grad_steps", 0.0))
            stats_history["dreamer_ac_rollout_argmax_frac"].append(
                ep_metrics.get("dreamer_ac_rollout_argmax_frac", 0.0)
            )
            stats_history["dreamer_ac_rollout_argmax_mix_prob"].append(
                ep_metrics.get("dreamer_ac_rollout_argmax_mix_prob", 0.0)
            )
            stats_history["hdv_response_av_pressure_sum"].append(ep_metrics.get("hdv_response_av_pressure_sum", 0.0))
            stats_history["hdv_response_background_sum"].append(ep_metrics.get("hdv_response_background_sum", 0.0))
            stats_history["hdv_response_background_active_edges"].append(ep_metrics.get("hdv_response_background_active_edges", 0.0))
            stats_history["vehicle_route_credit_gate_frac"].append(
                ep_metrics.get("vehicle_route_credit_gate_frac", 0.0)
            )
            stats_history["vehicle_route_credit_selected_gate_frac"].append(
                ep_metrics.get("vehicle_route_credit_selected_gate_frac", 0.0)
            )
            rollout_rewards.append_rollout_reward_stats(stats_history, ep_metrics)
            stats_history["traj_diagnostics"].append({
                "traj_id": int(traj.get("traj_id", episode)),
                "episode": int(episode),
                "start_step": int(ep_metrics.get("start_step", -1)),
                "avg_active_av": float(ep_metrics.get("avg_active_av", 0.0)),
                "max_active_av": float(ep_metrics.get("max_active_av", 0.0)),
                "n_valid_paths": int(ep_metrics.get("n_valid_paths", 0.0)),
                "n_valid_path_agents": int(ep_metrics.get("n_valid_path_agents", 0.0)),
                "n_actor_steps": int(ep_metrics.get("n_actor_steps", 0.0)),
                "reward_grad_steps": int(ep_metrics.get("reward_grad_steps", 0.0)),
                "entropy_grad_steps": int(ep_metrics.get("entropy_grad_steps", 0.0)),
                "route_cost_aux_loss": float(ep_metrics.get("route_cost_aux_loss", 0.0)),
                "route_cost_aux_acc": float(ep_metrics.get("route_cost_aux_acc", 0.0)),
                "route_cost_aux_margin": float(ep_metrics.get("route_cost_aux_margin", 0.0)),
                "expected_cost_loss": float(ep_metrics.get("expected_cost_loss", 0.0)),
                "expected_cost_raw": float(ep_metrics.get("expected_cost_raw", 0.0)),
                "expected_cost_best_prob": float(ep_metrics.get("expected_cost_best_prob", 0.0)),
                "expected_cost_best_route0_frac": float(ep_metrics.get("expected_cost_best_route0_frac", 0.0)),
                "expected_cost_nonzero_best_frac": float(ep_metrics.get("expected_cost_nonzero_best_frac", 0.0)),
                "expected_cost_margin": float(ep_metrics.get("expected_cost_margin", 0.0)),
                "grpo_route_cost_loss": float(ep_metrics.get("grpo_route_cost_loss", 0.0)),
                "grpo_route_cost_raw": float(ep_metrics.get("grpo_route_cost_raw", 0.0)),
                "grpo_route_cost_best_prob": float(ep_metrics.get("grpo_route_cost_best_prob", 0.0)),
                "grpo_route_cost_best_route0_frac": float(ep_metrics.get("grpo_route_cost_best_route0_frac", 0.0)),
                "grpo_route_cost_nonzero_best_frac": float(ep_metrics.get("grpo_route_cost_nonzero_best_frac", 0.0)),
                "grpo_route_cost_margin": float(ep_metrics.get("grpo_route_cost_margin", 0.0)),
                "grpo_route_cost_adv_mean": float(ep_metrics.get("grpo_route_cost_adv_mean", 0.0)),
                "grpo_route_cost_adv_std": float(ep_metrics.get("grpo_route_cost_adv_std", 0.0)),
                "grpo_rollout_cost": float(ep_metrics.get("grpo_rollout_cost", 0.0)),
                "grpo_rollout_logprob": float(ep_metrics.get("grpo_rollout_logprob", 0.0)),
                "grpo_rollout_selected_nonzero_frac": float(ep_metrics.get("grpo_rollout_selected_nonzero_frac", 0.0)),
                "dreamer_base_reward": float(ep_metrics.get("dreamer_base_reward", 0.0)),
                "dreamer_ac_rollout_argmax_frac": float(
                    ep_metrics.get("dreamer_ac_rollout_argmax_frac", 0.0)
                ),
                "dreamer_ac_rollout_argmax_mix_prob": float(
                    ep_metrics.get("dreamer_ac_rollout_argmax_mix_prob", 0.0)
                ),
                "dreamer_ac_sample_count": float(ep_metrics.get("dreamer_ac_sample_count", 0.0)),
                "dreamer_ac_selected_nonzero_frac": float(ep_metrics.get("dreamer_ac_selected_nonzero_frac", 0.0)),
                "dreamer_ac_selected_route0_frac": float(ep_metrics.get("dreamer_ac_selected_route0_frac", 0.0)),
                "dreamer_ac_selected_route1_frac": float(ep_metrics.get("dreamer_ac_selected_route1_frac", 0.0)),
                "dreamer_ac_selected_route2_frac": float(ep_metrics.get("dreamer_ac_selected_route2_frac", 0.0)),
                "dreamer_ac_selected_route3_frac": float(ep_metrics.get("dreamer_ac_selected_route3_frac", 0.0)),
                "vehicle_reward_mean": float(ep_metrics.get("vehicle_reward_mean", 0.0)),
                "vehicle_reward_std": float(ep_metrics.get("vehicle_reward_std", 0.0)),
                "vehicle_route_progress_reward_mean": float(
                    ep_metrics.get("vehicle_route_progress_reward_mean", 0.0)
                ),
                "vehicle_route_progress_reward_std": float(
                    ep_metrics.get("vehicle_route_progress_reward_std", 0.0)
                ),
                "vehicle_route_credit_reward_mean": float(
                    ep_metrics.get("vehicle_route_credit_reward_mean", 0.0)
                ),
                "vehicle_route_credit_reward_std": float(
                    ep_metrics.get("vehicle_route_credit_reward_std", 0.0)
                ),
                "vehicle_route_credit_gate_frac": float(
                    ep_metrics.get("vehicle_route_credit_gate_frac", 0.0)
                ),
                "vehicle_route_credit_selected_gate_frac": float(
                    ep_metrics.get("vehicle_route_credit_selected_gate_frac", 0.0)
                ),
                "vehicle_social_reward_mean": float(ep_metrics.get("vehicle_social_reward_mean", 0.0)),
                "vehicle_social_reward_std": float(ep_metrics.get("vehicle_social_reward_std", 0.0)),
                "dreamer_ac_entropy_loss": float(ep_metrics.get("dreamer_ac_entropy_loss", 0.0)),
                "dreamer_ac_entropy_floor_loss": float(ep_metrics.get("dreamer_ac_entropy_floor_loss", 0.0)),
                "dreamer_ac_entropy_target": float(ep_metrics.get("dreamer_ac_entropy_target", 0.0)),
                "dreamer_ac_entropy_valid_target_mean": float(
                    ep_metrics.get("dreamer_ac_entropy_valid_target_mean", 0.0)
                ),
                "dreamer_ac_route_adv_mean": float(ep_metrics.get("dreamer_ac_route_adv_mean", 0.0)),
                "dreamer_ac_route_adv_std": float(ep_metrics.get("dreamer_ac_route_adv_std", 0.0)),
                "dreamer_ac_combined_adv_mean": float(ep_metrics.get("dreamer_ac_combined_adv_mean", 0.0)),
                "dreamer_ac_combined_adv_std": float(ep_metrics.get("dreamer_ac_combined_adv_std", 0.0)),
                **rollout_rewards.rollout_reward_diagnostics(ep_metrics),
                "hdv_response_av_pressure_sum": float(ep_metrics.get("hdv_response_av_pressure_sum", 0.0)),
                "hdv_response_background_sum": float(ep_metrics.get("hdv_response_background_sum", 0.0)),
                "hdv_response_background_active_edges": float(ep_metrics.get("hdv_response_background_active_edges", 0.0)),
            })

            fresh_pending_traj_count = count_fresh_pending_trajs(pending_trajs)
            if fresh_pending_traj_count >= max(1, int(cfg.dreamer_batch_episodes)):
                _update_t0 = _time.perf_counter()
                last_update_stats = dreamer_update(
                    cfg,
                    actor,
                    critic,
                    actor_optimizer,
                    critic_optimizer,
                    pending_trajs,
                    update_id=update_id,
                    target_critic=target_critic,
                    return_normalizer=return_normalizer,
                    vehicle_critic=vehicle_critic,
                    vehicle_critic_optimizer=vehicle_critic_optimizer,
                    target_vehicle_critic=target_vehicle_critic,
                    vehicle_return_normalizer=vehicle_return_normalizer,
                    actor_gate_state=actor_gate_state,
                    actor_anchor=actor_anchor,
                )
                _update_wall_sec += _time.perf_counter() - _update_t0
                anchor_interval = int(getattr(cfg, "dreamer_actor_anchor_update_interval", 0))
                if (
                    actor_anchor is not None
                    and anchor_interval > 0
                    and int(last_update_stats.get("dreamer_actor_anchor_accepted", 0.0)) > 0
                    and (update_id + 1) % anchor_interval == 0
                ):
                    actor_anchor.load_state_dict(actor.state_dict())
                    actor_anchor.eval()
                    actor_anchor_initialized = True
                pending_trajs.clear()
                update_id += 1
                for k in DREAMER_UPDATE_SCALAR_KEYS:
                    stats_history[k].append(last_update_stats.get(k, 0.0))
                stats_history["skipped_actor_trajs"].append(last_update_stats.get("skipped_actor_trajs", 0.0))
                stats_history["update_diagnostics"].append({
                    "update_id": int(last_update_stats.get("update_id", update_id - 1)),
                    "consumed_traj_ids": list(last_update_stats.get("consumed_traj_ids", [])),
                    "used_traj_ids": list(last_update_stats.get("used_traj_ids", [])),
                    "skipped_traj_ids": list(last_update_stats.get("skipped_traj_ids", [])),
                    "skip_reason_counts": dict(last_update_stats.get("skip_reason_counts", {})),
                    "actor_gate_skipped": bool(last_update_stats.get("dreamer_actor_gate_skipped", 0.0)),
                    "actor_gate_max_route_frac": float(last_update_stats.get("dreamer_actor_gate_max_route_frac", 0.0)),
                    "actor_gate_route0_frac": float(last_update_stats.get("dreamer_actor_gate_route0_frac", 0.0)),
                    "actor_gate_top1_prob": float(last_update_stats.get("dreamer_actor_gate_top1_prob", 0.0)),
                    "actor_gate_sample_count": float(last_update_stats.get("dreamer_actor_gate_sample_count", 0.0)),
                    "actor_gate_latched": bool(last_update_stats.get("dreamer_actor_gate_latched", 0.0)),
                    "actor_trust_region_accepted": bool(last_update_stats.get("dreamer_actor_trust_region_accepted", 0.0)),
                    "actor_trust_region_kl": float(last_update_stats.get("dreamer_actor_trust_region_kl", 0.0)),
                    "actor_trust_region_retries": float(last_update_stats.get("dreamer_actor_trust_region_retries", 0.0)),
                    "actor_trust_region_argmax_route0_frac": float(
                        last_update_stats.get("dreamer_actor_trust_region_argmax_route0_frac", 0.0)
                    ),
                    "actor_trust_region_vehicle_argmax_change_max": float(
                        last_update_stats.get("dreamer_actor_trust_region_vehicle_argmax_change_max", 0.0)
                    ),
                    "actor_anchor_active": bool(last_update_stats.get("dreamer_actor_anchor_active", 0.0)),
                    "actor_anchor_accepted": bool(last_update_stats.get("dreamer_actor_anchor_accepted", 0.0)),
                    "actor_anchor_kl": float(last_update_stats.get("dreamer_actor_anchor_kl", 0.0)),
                    "actor_anchor_argmax_change_frac": float(
                        last_update_stats.get("dreamer_actor_anchor_argmax_change_frac", 0.0)
                    ),
                    "actor_anchor_vehicle_argmax_change_max": float(
                        last_update_stats.get("dreamer_actor_anchor_vehicle_argmax_change_max", 0.0)
                    ),
                    "actor_anchor_argmax_route0_frac": float(
                        last_update_stats.get("dreamer_actor_anchor_argmax_route0_frac", 0.0)
                    ),
                    "skipped_actor_trajs": float(last_update_stats.get("skipped_actor_trajs", 0.0)),
                    "consumed_fresh_traj_count": int(last_update_stats.get("consumed_fresh_traj_count", 0)),
                    "consumed_restored_traj_count": int(last_update_stats.get("consumed_restored_traj_count", 0)),
                    "used_fresh_traj_count": int(last_update_stats.get("used_fresh_traj_count", 0)),
                    "used_restored_traj_count": int(last_update_stats.get("used_restored_traj_count", 0)),
                    "skipped_fresh_traj_count": int(last_update_stats.get("skipped_fresh_traj_count", 0)),
                    "skipped_restored_traj_count": int(last_update_stats.get("skipped_restored_traj_count", 0)),
                })

            if episode % cfg.log_every == 0:
                recent_reward = np.mean(stats_history["ep_reward"][-cfg.log_every:])
                recent_t_av = np.mean(stats_history["t_av"][-cfg.log_every:])
                recent_t_hdv = np.mean(stats_history["t_hdv"][-cfg.log_every:])
                recent_t_all = np.mean(stats_history["t_all"][-cfg.log_every:])
                recent_invalid = np.mean(stats_history["invalid_route_rate"][-cfg.log_every:])
                recent_churn = np.mean(stats_history["route_churn"][-cfg.log_every:])
                recent_route_cost_aux = np.mean(stats_history["route_cost_aux_loss"][-cfg.log_every:]) if stats_history["route_cost_aux_loss"] else 0.0
                recent_route_cost_aux_acc = np.mean(stats_history["route_cost_aux_acc"][-cfg.log_every:]) if stats_history["route_cost_aux_acc"] else 0.0
                recent_expected_cost = np.mean(stats_history["expected_cost_loss"][-cfg.log_every:]) if stats_history["expected_cost_loss"] else 0.0
                recent_expected_best_prob = np.mean(stats_history["expected_cost_best_prob"][-cfg.log_every:]) if stats_history["expected_cost_best_prob"] else 0.0
                recent_expected_route0 = np.mean(stats_history["expected_cost_best_route0_frac"][-cfg.log_every:]) if stats_history["expected_cost_best_route0_frac"] else 0.0
                recent_expected_nonzero = np.mean(stats_history["expected_cost_nonzero_best_frac"][-cfg.log_every:]) if stats_history["expected_cost_nonzero_best_frac"] else 0.0
                recent_expected_margin = np.mean(stats_history["expected_cost_margin"][-cfg.log_every:]) if stats_history["expected_cost_margin"] else 0.0
                recent_grpo = np.mean(stats_history["grpo_route_cost_loss"][-cfg.log_every:]) if stats_history["grpo_route_cost_loss"] else 0.0
                recent_grpo_best_prob = np.mean(stats_history["grpo_route_cost_best_prob"][-cfg.log_every:]) if stats_history["grpo_route_cost_best_prob"] else 0.0
                recent_grpo_route0 = np.mean(stats_history["grpo_route_cost_best_route0_frac"][-cfg.log_every:]) if stats_history["grpo_route_cost_best_route0_frac"] else 0.0
                recent_grpo_nonzero = np.mean(stats_history["grpo_route_cost_nonzero_best_frac"][-cfg.log_every:]) if stats_history["grpo_route_cost_nonzero_best_frac"] else 0.0
                recent_grpo_margin = np.mean(stats_history["grpo_route_cost_margin"][-cfg.log_every:]) if stats_history["grpo_route_cost_margin"] else 0.0
                recent_rollout_grpo = np.mean(stats_history["grpo_rollout_loss"][-cfg.log_every:]) if stats_history["grpo_rollout_loss"] else 0.0
                recent_combined_grpo = np.mean(stats_history["grpo_combined_loss"][-cfg.log_every:]) if stats_history["grpo_combined_loss"] else 0.0
                recent_combined_clip = np.mean(stats_history["grpo_combined_clip_frac"][-cfg.log_every:]) if stats_history["grpo_combined_clip_frac"] else 0.0
                recent_policy_kl = np.mean(stats_history["grpo_policy_kl"][-cfg.log_every:]) if stats_history["grpo_policy_kl"] else 0.0
                recent_rollout_cost = np.mean(stats_history["grpo_rollout_cost"][-cfg.log_every:]) if stats_history["grpo_rollout_cost"] else 0.0
                recent_rollout_adv_std = np.mean(stats_history["grpo_rollout_adv_std"][-cfg.log_every:]) if stats_history["grpo_rollout_adv_std"] else 0.0
                recent_rollout_nonzero = np.mean(stats_history["grpo_rollout_selected_nonzero_frac"][-cfg.log_every:]) if stats_history["grpo_rollout_selected_nonzero_frac"] else 0.0
                recent_rollout_vehicle = np.mean(stats_history["grpo_rollout_vehicle_cost_loss"][-cfg.log_every:]) if stats_history["grpo_rollout_vehicle_cost_loss"] else 0.0
                recent_rollout_vehicle_adv_std = np.mean(stats_history["grpo_rollout_vehicle_cost_adv_std"][-cfg.log_every:]) if stats_history["grpo_rollout_vehicle_cost_adv_std"] else 0.0
                recent_rollout_reward = rollout_rewards.recent_rollout_reward_stats(stats_history, cfg.log_every)
                recent_base_reward = np.mean(stats_history["dreamer_base_reward"][-cfg.log_every:]) if stats_history["dreamer_base_reward"] else 0.0
                recent_ac_adv_mean = np.mean(stats_history["dreamer_ac_adv_mean"][-cfg.log_every:]) if stats_history["dreamer_ac_adv_mean"] else 0.0
                recent_ac_adv_std = np.mean(stats_history["dreamer_ac_adv_std"][-cfg.log_every:]) if stats_history["dreamer_ac_adv_std"] else 0.0
                recent_ac_route_adv_mean = np.mean(stats_history["dreamer_ac_route_adv_mean"][-cfg.log_every:]) if stats_history["dreamer_ac_route_adv_mean"] else 0.0
                recent_ac_route_adv_std = np.mean(stats_history["dreamer_ac_route_adv_std"][-cfg.log_every:]) if stats_history["dreamer_ac_route_adv_std"] else 0.0
                recent_ac_combined_adv_mean = np.mean(stats_history["dreamer_ac_combined_adv_mean"][-cfg.log_every:]) if stats_history["dreamer_ac_combined_adv_mean"] else 0.0
                recent_ac_combined_adv_std = np.mean(stats_history["dreamer_ac_combined_adv_std"][-cfg.log_every:]) if stats_history["dreamer_ac_combined_adv_std"] else 0.0
                recent_ac_argmax_mix = np.mean(stats_history["dreamer_ac_rollout_argmax_mix_prob"][-cfg.log_every:]) if stats_history["dreamer_ac_rollout_argmax_mix_prob"] else 0.0
                recent_ac_argmax_frac = np.mean(stats_history["dreamer_ac_rollout_argmax_frac"][-cfg.log_every:]) if stats_history["dreamer_ac_rollout_argmax_frac"] else 0.0
                recent_ac_nonzero = np.mean(stats_history["dreamer_ac_selected_nonzero_frac"][-cfg.log_every:]) if stats_history["dreamer_ac_selected_nonzero_frac"] else 0.0
                recent_ac_route0 = np.mean(stats_history["dreamer_ac_selected_route0_frac"][-cfg.log_every:]) if stats_history["dreamer_ac_selected_route0_frac"] else 0.0
                recent_ac_route1 = np.mean(stats_history["dreamer_ac_selected_route1_frac"][-cfg.log_every:]) if stats_history["dreamer_ac_selected_route1_frac"] else 0.0
                recent_ac_route2 = np.mean(stats_history["dreamer_ac_selected_route2_frac"][-cfg.log_every:]) if stats_history["dreamer_ac_selected_route2_frac"] else 0.0
                recent_ac_route3 = np.mean(stats_history["dreamer_ac_selected_route3_frac"][-cfg.log_every:]) if stats_history["dreamer_ac_selected_route3_frac"] else 0.0
                recent_ac_policy_argmax_nonzero = np.mean(stats_history["dreamer_ac_policy_argmax_nonzero_frac"][-cfg.log_every:]) if stats_history["dreamer_ac_policy_argmax_nonzero_frac"] else 0.0
                recent_ac_policy_argmax_route0 = np.mean(stats_history["dreamer_ac_policy_argmax_route0_frac"][-cfg.log_every:]) if stats_history["dreamer_ac_policy_argmax_route0_frac"] else 0.0
                recent_ac_policy_argmax_max = np.mean(stats_history["dreamer_ac_policy_argmax_max_route_frac"][-cfg.log_every:]) if stats_history["dreamer_ac_policy_argmax_max_route_frac"] else 0.0
                recent_ac_policy_top1 = np.mean(stats_history["dreamer_ac_policy_top1_prob_mean"][-cfg.log_every:]) if stats_history["dreamer_ac_policy_top1_prob_mean"] else 0.0
                recent_actor_gate = np.mean(stats_history["dreamer_actor_gate_skipped"][-cfg.log_every:]) if stats_history["dreamer_actor_gate_skipped"] else 0.0
                recent_actor_gate_latched = np.mean(stats_history["dreamer_actor_gate_latched"][-cfg.log_every:]) if stats_history["dreamer_actor_gate_latched"] else 0.0
                recent_actor_tr_accept = np.mean(stats_history["dreamer_actor_trust_region_accepted"][-cfg.log_every:]) if stats_history["dreamer_actor_trust_region_accepted"] else 0.0
                recent_actor_tr_kl = np.mean(stats_history["dreamer_actor_trust_region_kl"][-cfg.log_every:]) if stats_history["dreamer_actor_trust_region_kl"] else 0.0
                recent_actor_tr_retries = np.mean(stats_history["dreamer_actor_trust_region_retries"][-cfg.log_every:]) if stats_history["dreamer_actor_trust_region_retries"] else 0.0
                recent_actor_tr_argmax_change = np.mean(stats_history["dreamer_actor_trust_region_argmax_change_frac"][-cfg.log_every:]) if stats_history["dreamer_actor_trust_region_argmax_change_frac"] else 0.0
                recent_actor_tr_vehicle_change = np.mean(stats_history["dreamer_actor_trust_region_vehicle_argmax_change_max"][-cfg.log_every:]) if stats_history["dreamer_actor_trust_region_vehicle_argmax_change_max"] else 0.0
                recent_actor_tr_route0 = np.mean(stats_history["dreamer_actor_trust_region_argmax_route0_frac"][-cfg.log_every:]) if stats_history["dreamer_actor_trust_region_argmax_route0_frac"] else 0.0
                recent_actor_anchor_active = np.mean(stats_history["dreamer_actor_anchor_active"][-cfg.log_every:]) if stats_history["dreamer_actor_anchor_active"] else 0.0
                recent_actor_anchor_accept = np.mean(stats_history["dreamer_actor_anchor_accepted"][-cfg.log_every:]) if stats_history["dreamer_actor_anchor_accepted"] else 0.0
                recent_actor_anchor_kl = np.mean(stats_history["dreamer_actor_anchor_kl"][-cfg.log_every:]) if stats_history["dreamer_actor_anchor_kl"] else 0.0
                recent_actor_anchor_change = np.mean(stats_history["dreamer_actor_anchor_argmax_change_frac"][-cfg.log_every:]) if stats_history["dreamer_actor_anchor_argmax_change_frac"] else 0.0
                recent_actor_anchor_vehicle_change = np.mean(stats_history["dreamer_actor_anchor_vehicle_argmax_change_max"][-cfg.log_every:]) if stats_history["dreamer_actor_anchor_vehicle_argmax_change_max"] else 0.0
                recent_actor_anchor_route0 = np.mean(stats_history["dreamer_actor_anchor_argmax_route0_frac"][-cfg.log_every:]) if stats_history["dreamer_actor_anchor_argmax_route0_frac"] else 0.0
                recent_ac_samples = np.mean(stats_history["dreamer_ac_sample_count"][-cfg.log_every:]) if stats_history["dreamer_ac_sample_count"] else 0.0
                recent_ac_kl = np.mean(stats_history["dreamer_ac_policy_kl"][-cfg.log_every:]) if stats_history["dreamer_ac_policy_kl"] else 0.0
                recent_ac_clip = np.mean(stats_history["dreamer_ac_clip_frac"][-cfg.log_every:]) if stats_history["dreamer_ac_clip_frac"] else 0.0
                recent_ac_weight = np.mean(stats_history["dreamer_ac_weight_mean"][-cfg.log_every:]) if stats_history["dreamer_ac_weight_mean"] else 0.0
                recent_ac_sample_adv = np.mean(stats_history["dreamer_ac_uses_sample_adv"][-cfg.log_every:]) if stats_history["dreamer_ac_uses_sample_adv"] else 0.0
                recent_ac_entropy_floor = np.mean(stats_history["dreamer_ac_entropy_floor_loss"][-cfg.log_every:]) if stats_history["dreamer_ac_entropy_floor_loss"] else 0.0
                recent_ac_entropy_target = np.mean(stats_history["dreamer_ac_entropy_target"][-cfg.log_every:]) if stats_history["dreamer_ac_entropy_target"] else 0.0
                recent_ac_entropy_routes = np.mean(stats_history["dreamer_ac_entropy_effective_route_count"][-cfg.log_every:]) if stats_history["dreamer_ac_entropy_effective_route_count"] else 0.0
                recent_ac_entropy_mask = np.mean(stats_history["dreamer_ac_entropy_cost_mask_frac"][-cfg.log_every:]) if stats_history["dreamer_ac_entropy_cost_mask_frac"] else 0.0
                recent_vehicle_vf = np.mean(stats_history["vehicle_value_loss"][-cfg.log_every:]) if stats_history["vehicle_value_loss"] else 0.0
                recent_vehicle_target_std = np.mean(stats_history["vehicle_critic_target_std"][-cfg.log_every:]) if stats_history["vehicle_critic_target_std"] else 0.0
                recent_vehicle_reward_mean = np.mean(stats_history["vehicle_reward_mean"][-cfg.log_every:]) if stats_history["vehicle_reward_mean"] else 0.0
                recent_vehicle_reward_std = np.mean(stats_history["vehicle_reward_std"][-cfg.log_every:]) if stats_history["vehicle_reward_std"] else 0.0
                recent_vehicle_route_progress_mean = np.mean(stats_history["vehicle_route_progress_reward_mean"][-cfg.log_every:]) if stats_history["vehicle_route_progress_reward_mean"] else 0.0
                recent_vehicle_route_progress_std = np.mean(stats_history["vehicle_route_progress_reward_std"][-cfg.log_every:]) if stats_history["vehicle_route_progress_reward_std"] else 0.0
                recent_vehicle_route_credit_mean = np.mean(stats_history["vehicle_route_credit_reward_mean"][-cfg.log_every:]) if stats_history["vehicle_route_credit_reward_mean"] else 0.0
                recent_vehicle_route_credit_std = np.mean(stats_history["vehicle_route_credit_reward_std"][-cfg.log_every:]) if stats_history["vehicle_route_credit_reward_std"] else 0.0
                recent_vehicle_route_credit_gate = np.mean(stats_history["vehicle_route_credit_gate_frac"][-cfg.log_every:]) if stats_history["vehicle_route_credit_gate_frac"] else 0.0
                recent_vehicle_route_credit_selected_gate = np.mean(stats_history["vehicle_route_credit_selected_gate_frac"][-cfg.log_every:]) if stats_history["vehicle_route_credit_selected_gate_frac"] else 0.0
                recent_vehicle_social_mean = np.mean(stats_history["vehicle_social_reward_mean"][-cfg.log_every:]) if stats_history["vehicle_social_reward_mean"] else 0.0
                recent_vehicle_social_std = np.mean(stats_history["vehicle_social_reward_std"][-cfg.log_every:]) if stats_history["vehicle_social_reward_std"] else 0.0
                recent_vehicle_ret_std = np.mean(stats_history["vehicle_return_norm_std"][-cfg.log_every:]) if stats_history["vehicle_return_norm_std"] else 1.0
                recent_v_ev = np.mean(stats_history["critic_explained_variance"][-cfg.log_every:]) if stats_history["critic_explained_variance"] else 0.0
                recent_ret_std = np.mean(stats_history["return_norm_std"][-cfg.log_every:]) if stats_history["return_norm_std"] else 1.0
                recent_hdv_response_bg = np.mean(stats_history["hdv_response_background_sum"][-cfg.log_every:]) if stats_history["hdv_response_background_sum"] else 0.0
                recent_active_av = np.mean(stats_history["avg_active_av"][-cfg.log_every:])
                recent_actor_steps = np.mean(stats_history["n_actor_steps"][-cfg.log_every:]) if stats_history["n_actor_steps"] else 0.0
                recent_valid_agents = np.mean(stats_history["n_valid_path_agents"][-cfg.log_every:]) if stats_history["n_valid_path_agents"] else 0.0
                recent_skipped = (
                    np.mean(stats_history["skipped_actor_trajs"][-cfg.log_every:])
                    if stats_history["skipped_actor_trajs"] else 0.0
                )
                last_skip_reasons = (
                    stats_history["update_diagnostics"][-1].get("skip_reason_counts", {})
                    if stats_history["update_diagnostics"] else {}
                )
                if stats_history["terminal_t_all_pred"]:
                    tail_vals = np.asarray(stats_history["terminal_t_all_pred"][-cfg.log_every:], dtype=np.float32)
                    finite_tail = tail_vals[np.isfinite(tail_vals)]
                    recent_tall_pred = float(finite_tail.mean()) if finite_tail.size > 0 else float("nan")
                else:
                    recent_tall_pred = float("nan")
                episode_wall_sec = _time.perf_counter() - _episode_wall_t0
                print(
                    f"ep {episode:05d} | "
                    f"reward={ep_reward:8.3f} (avg{cfg.log_every}={recent_reward:.3f}) | "
                    f"actor={last_update_stats['pg_loss']:.4f} | "
                    f"vf={last_update_stats['value_loss']:.4f} | "
                    f"ent={last_update_stats['entropy']:.4f} | "
                    f"t_av={recent_t_av:.4f} t_hdv={recent_t_hdv:.4f} t_all={recent_t_all:.4f} | "
                    f"active_av={recent_active_av:.2f} valid_agents={recent_valid_agents:.2f} actor_steps={recent_actor_steps:.2f} skipped_actor={recent_skipped:.2f} | "
                    f"invalid={recent_invalid:.4f} churn={recent_churn:.4f} | "
                    f"route_aux={recent_route_cost_aux:.4f}/{recent_route_cost_aux_acc:.3f} | "
                    f"expected_cost={recent_expected_cost:.4f}/{recent_expected_best_prob:.3f} "
                    f"best_r0/non0/margin={recent_expected_route0:.3f}/{recent_expected_nonzero:.3f}/{recent_expected_margin:.2f} | "
                    f"grpo={recent_grpo:.4f}/{recent_grpo_best_prob:.3f} "
                    f"best_r0/non0/margin={recent_grpo_route0:.3f}/{recent_grpo_nonzero:.3f}/{recent_grpo_margin:.2f} | "
                    f"combined_grpo={recent_combined_grpo:.4f} clip={recent_combined_clip:.3f} kl={recent_policy_kl:.4f} | "
                    f"rollout_grpo={recent_rollout_grpo:.4f} cost={recent_rollout_cost:.4f} "
                    f"advstd={recent_rollout_adv_std:.3f} sampled_non0={recent_rollout_nonzero:.3f} "
                    f"veh={recent_rollout_vehicle:.4f}/{recent_rollout_vehicle_adv_std:.3f} | "
                    f"base_reward={recent_base_reward:.4f} "
                    f"{rollout_rewards.format_recent_rollout_reward(recent_rollout_reward)} | "
                    f"dreamer_ac_adv={recent_ac_adv_mean:.4f}/{recent_ac_adv_std:.4f} "
                    f"route_adv={recent_ac_route_adv_mean:.4f}/{recent_ac_route_adv_std:.4f} "
                    f"combined_adv={recent_ac_combined_adv_mean:.4f}/{recent_ac_combined_adv_std:.4f} "
                    f"ac_argmax={recent_ac_argmax_frac:.3f}/{recent_ac_argmax_mix:.3f} "
                    f"ac_non0={recent_ac_nonzero:.3f} "
                    f"ac_r={recent_ac_route0:.2f}/{recent_ac_route1:.2f}/{recent_ac_route2:.2f}/{recent_ac_route3:.2f} "
                    f"ac_policy_argmax={recent_ac_policy_argmax_max:.2f}/r0={recent_ac_policy_argmax_route0:.2f}/non0={recent_ac_policy_argmax_nonzero:.2f} "
                    f"top1={recent_ac_policy_top1:.2f} gate={recent_actor_gate:.1f}/{recent_actor_gate_latched:.1f} "
                    f"trust={recent_actor_tr_accept:.1f}/kl={recent_actor_tr_kl:.4f}/retry={recent_actor_tr_retries:.1f}/chg={recent_actor_tr_argmax_change:.3f}/vchg={recent_actor_tr_vehicle_change:.3f}/r0={recent_actor_tr_route0:.2f} "
                    f"anchor={recent_actor_anchor_active:.1f}/{recent_actor_anchor_accept:.1f}/kl={recent_actor_anchor_kl:.4f}/chg={recent_actor_anchor_change:.3f}/vchg={recent_actor_anchor_vehicle_change:.3f}/r0={recent_actor_anchor_route0:.2f} "
                    f"ac_n={recent_ac_samples:.1f} "
                    f"ac_kl/clip/w/sample_adv={recent_ac_kl:.4f}/{recent_ac_clip:.3f}/{recent_ac_weight:.3f}/{recent_ac_sample_adv:.1f} "
                    f"ent_floor={recent_ac_entropy_floor:.4f}/{recent_ac_entropy_target:.3f}/k={recent_ac_entropy_routes:.2f}/mask={recent_ac_entropy_mask:.2f} "
                    f"v_ev={recent_v_ev:.3f} ret_std={recent_ret_std:.3f} | "
                    f"veh_vf={recent_vehicle_vf:.4f} veh_rew={recent_vehicle_reward_mean:.2f}/{recent_vehicle_reward_std:.2f} "
                    f"veh_route_progress={recent_vehicle_route_progress_mean:.2f}/{recent_vehicle_route_progress_std:.2f} "
                    f"veh_route_credit={recent_vehicle_route_credit_mean:.2f}/{recent_vehicle_route_credit_std:.2f} "
                    f"credit_gate={recent_vehicle_route_credit_gate:.2f}/{recent_vehicle_route_credit_selected_gate:.2f} "
                    f"veh_social={recent_vehicle_social_mean:.2f}/{recent_vehicle_social_std:.2f} "
                    f"veh_tgt_std={recent_vehicle_target_std:.2f} veh_ret_std={recent_vehicle_ret_std:.2f} | "
                    f"hdv_resp_bg={recent_hdv_response_bg:.4f} | "
                    f"timing(ep={episode_wall_sec:.2f}s rollout={_rollout_wall_sec:.2f}s hdv={_hdv_response_wall_sec:.2f}s baseline={_baseline_wall_sec:.2f}s update={_update_wall_sec:.2f}s group={rollout_group_size}) | "
                    f"skip_reasons={last_skip_reasons} | "
                    f"tall_pred={recent_tall_pred:.4f} | "
                    f"actor_lr={actor_optimizer.param_groups[0]['lr']:.2e} critic_lr={critic_optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

            if cfg.checkpoint_every > 0 and (episode + 1) % cfg.checkpoint_every == 0:
                extra_state = _dreamer_extra_state()
                path = save_dreamer_checkpoint(
                    cfg,
                    actor,
                    critic,
                    actor_optimizer,
                    critic_optimizer,
                    episode + 1,
                    stats_history,
                    extra_state=extra_state,
                    target_critic=target_critic,
                    return_normalizer=return_normalizer,
                    vehicle_critic=vehicle_critic,
                    vehicle_critic_optimizer=vehicle_critic_optimizer,
                    target_vehicle_critic=target_vehicle_critic,
                    vehicle_return_normalizer=vehicle_return_normalizer,
                )
                logger.info("  Dreamer checkpoint saved: %s", path)

        if pending_trajs and not bool(cfg.dreamer_preserve_partial_batch_on_exit):
            last_update_stats = dreamer_update(
                cfg,
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                pending_trajs,
                update_id=update_id,
                target_critic=target_critic,
                return_normalizer=return_normalizer,
                vehicle_critic=vehicle_critic,
                vehicle_critic_optimizer=vehicle_critic_optimizer,
                target_vehicle_critic=target_vehicle_critic,
                vehicle_return_normalizer=vehicle_return_normalizer,
                actor_gate_state=actor_gate_state,
                actor_anchor=actor_anchor,
            )
            anchor_interval = int(getattr(cfg, "dreamer_actor_anchor_update_interval", 0))
            if (
                actor_anchor is not None
                and anchor_interval > 0
                and int(last_update_stats.get("dreamer_actor_anchor_accepted", 0.0)) > 0
                and (update_id + 1) % anchor_interval == 0
            ):
                actor_anchor.load_state_dict(actor.state_dict())
                actor_anchor.eval()
                actor_anchor_initialized = True
            for k in DREAMER_UPDATE_SCALAR_KEYS:
                stats_history[k].append(last_update_stats.get(k, 0.0))
            stats_history["skipped_actor_trajs"].append(last_update_stats.get("skipped_actor_trajs", 0.0))
            stats_history["update_diagnostics"].append({
                "update_id": int(last_update_stats.get("update_id", update_id)),
                "consumed_traj_ids": list(last_update_stats.get("consumed_traj_ids", [])),
                "used_traj_ids": list(last_update_stats.get("used_traj_ids", [])),
                "skipped_traj_ids": list(last_update_stats.get("skipped_traj_ids", [])),
                "skip_reason_counts": dict(last_update_stats.get("skip_reason_counts", {})),
                "actor_gate_skipped": bool(last_update_stats.get("dreamer_actor_gate_skipped", 0.0)),
                "actor_gate_latched": bool(last_update_stats.get("dreamer_actor_gate_latched", 0.0)),
                "actor_trust_region_accepted": bool(last_update_stats.get("dreamer_actor_trust_region_accepted", 0.0)),
                "actor_trust_region_kl": float(last_update_stats.get("dreamer_actor_trust_region_kl", 0.0)),
                "actor_trust_region_retries": float(last_update_stats.get("dreamer_actor_trust_region_retries", 0.0)),
                "actor_trust_region_argmax_route0_frac": float(
                    last_update_stats.get("dreamer_actor_trust_region_argmax_route0_frac", 0.0)
                ),
                "actor_trust_region_vehicle_argmax_change_max": float(
                    last_update_stats.get("dreamer_actor_trust_region_vehicle_argmax_change_max", 0.0)
                ),
                "actor_anchor_active": bool(last_update_stats.get("dreamer_actor_anchor_active", 0.0)),
                "actor_anchor_accepted": bool(last_update_stats.get("dreamer_actor_anchor_accepted", 0.0)),
                "actor_anchor_kl": float(last_update_stats.get("dreamer_actor_anchor_kl", 0.0)),
                "actor_anchor_argmax_change_frac": float(
                    last_update_stats.get("dreamer_actor_anchor_argmax_change_frac", 0.0)
                ),
                "actor_anchor_vehicle_argmax_change_max": float(
                    last_update_stats.get("dreamer_actor_anchor_vehicle_argmax_change_max", 0.0)
                ),
                "actor_anchor_argmax_route0_frac": float(
                    last_update_stats.get("dreamer_actor_anchor_argmax_route0_frac", 0.0)
                ),
                "skipped_actor_trajs": float(last_update_stats.get("skipped_actor_trajs", 0.0)),
                "consumed_fresh_traj_count": int(last_update_stats.get("consumed_fresh_traj_count", 0)),
                "consumed_restored_traj_count": int(last_update_stats.get("consumed_restored_traj_count", 0)),
                "used_fresh_traj_count": int(last_update_stats.get("used_fresh_traj_count", 0)),
                "used_restored_traj_count": int(last_update_stats.get("used_restored_traj_count", 0)),
                "skipped_fresh_traj_count": int(last_update_stats.get("skipped_fresh_traj_count", 0)),
                "skipped_restored_traj_count": int(last_update_stats.get("skipped_restored_traj_count", 0)),
            })

        final_extra_state = _dreamer_extra_state()

        save_dreamer_checkpoint(
            cfg,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            target_episode,
            stats_history,
            extra_state=final_extra_state,
            target_critic=target_critic,
            return_normalizer=return_normalizer,
            vehicle_critic=vehicle_critic,
            vehicle_critic_optimizer=vehicle_critic_optimizer,
            target_vehicle_critic=target_vehicle_critic,
            vehicle_return_normalizer=vehicle_return_normalizer,
        )
        logger.info("Dreamer-style training complete.")
        return


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fully offline IPPO for urban routing using Dreamer4 world model"
    )
    add_settings_arg(p)
    p.add_argument("--data_dir",          type=str,   default="data/urban_routing_flow/round1")
    p.add_argument("--meta_file",         type=str,   default="flow_round1.pt")
    p.add_argument("--dynamics_ckpt",     type=str,   required=False, default="")
    p.add_argument("--projector_ckpt",    type=str, default="",
                   help="FlowProjector checkpoint path (required)")
    p.add_argument("--edge_registry",     type=str, default="",
                   help="EdgeRegistry JSON path (required)")
    p.add_argument("--flow_glob",         type=str, default="flows/*/shard_*.pt",
                   help="Glob pattern for flow shards under data_dir")
    p.add_argument("--history_length",    type=int,   default=8)
    p.add_argument("--wm_schedule",       type=str,   default="shortcut")
    p.add_argument("--wm_schedule_d",     type=float, default=0.25)
    p.add_argument("--wm_cfm_init_mode",  choices=["randn", "zero"], default="zero",
                   help="CFM rollout source for MARL imagination.")
    p.add_argument("--n_routes",          type=int,   default=4)
    p.add_argument("--n_av",              type=int,   default=0,
                   help="Deprecated compatibility cap used only when --dreamer_departure_cap_mode=cfg.")
    p.add_argument("--n_hdv",             type=int,   default=0,
                   help="Deprecated compatibility cap used only when --dreamer_departure_cap_mode=cfg.")
    p.add_argument("--episode_len",       type=int,   default=50)
    p.add_argument("--n_total_episodes",  type=int,   default=2000)
    p.add_argument("--training_family",   type=str,   default="dreamer_ctde",
                   choices=["dreamer_ctde"],
                   help="Downstream training family. UrbanDreamer keeps the Dreamer CTDE mainline only.")
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--gae_lambda",        type=float, default=0.95)
    p.add_argument("--clip_eps",          type=float, default=0.2)
    p.add_argument("--entropy_coef",      type=float, default=0.05)
    p.add_argument("--value_coef",           type=float, default=0.25)
    p.add_argument("--n_ppo_epochs",         type=int,   default=4)
    p.add_argument("--minibatch_size",       type=int,   default=64)
    p.add_argument("--n_episodes_per_update",type=int,   default=32)
    p.add_argument("--hidden_dim",        type=int,   default=256)
    p.add_argument("--n_hidden_layers",   type=int,   default=2)
    p.add_argument("--critic_target_variant", type=str, default="baseline",
                   choices=["baseline", "multistep_imagined"],
                   help="Critic-target variant for downstream comparison. 'multistep_imagined' changes value targets only.")
    p.add_argument("--critic_target_horizon", type=int, default=3,
                   help="Short imagined horizon used when critic_target_variant=multistep_imagined.")
    p.add_argument("--use_stackelberg", action="store_true",
                   help="Enable edge-only Stackelberg response inside the world model.")
    p.add_argument("--stackelberg_n_iters", type=int, default=10,
                   help="Fixed-point iterations for edge-level Stackelberg response.")
    p.add_argument("--stackelberg_damping", type=float, default=0.5,
                   help="Damping factor for edge-level Stackelberg fixed-point updates.")
    # ── HDV Gawron dynamic response ───────────────────────────────────────────
    p.add_argument("--no_hdv_gawron",  action="store_true",
                   help="Disable BPR-Gawron HDV dynamic update within imagined episodes")
    p.add_argument("--gawron_lr",      type=float, default=0.1,
                   help="Gawron learning rate for HDV route update")
    p.add_argument("--gawron_mode",    type=str, default="replicator",
                   choices=["replicator", "routerl", "routerl_blackbox"],
                   help="HDV backend: replicator/routerl for imagined updates; "
                        "routerl_blackbox uses shared RouteRL human black-box in SUMO eval")
    p.add_argument("--gawron_alpha_zero", type=float, default=0.2,
                   help="RouteRL-compatible alpha_zero (used when gawron_mode=routerl)")
    p.add_argument("--gawron_beta",    type=float, default=-1.5,
                   help="RouteRL-compatible beta (used when gawron_mode=routerl)")
    p.add_argument("--gawron_update_mode", type=str, default="expected",
                   choices=["expected", "sample"],
                   help="RouteRL-compatible update style: expected (fast) or sample (stochastic)")
    p.add_argument("--hdv_update_momentum", type=float, default=0.2,
                   help="Inter-episode HDV update smoothing (0=no update, 1=full update)")
    p.add_argument("--hdv_min_prob",   type=float, default=0.02,
                   help="Minimum probability floor per HDV route to avoid collapse")
    p.add_argument("--bpr_alpha",      type=float, default=0.15,
                   help="BPR volume-delay function alpha")
    p.add_argument("--bpr_beta",       type=float, default=4.0,
                   help="BPR volume-delay function beta")
    p.add_argument("--conservative_penalty_coef", type=float, default=0.02,
                   help="Penalty on AV-HDV intent divergence to reduce model exploitation")
    p.add_argument("--terminal_tall_reward_weight", type=float, default=0.0,
                   help="Legacy terminal reward weight for -t_all_pred. Default 0 keeps Dreamer reward as seconds-level BPR.")
    p.add_argument("--no_terminal_tall_reward", action="store_true",
                   help="Disable scheme-B terminal reward from WM t_all predictor.")
    p.add_argument("--wm_policy_obs_source", type=str, default="decoder_volume_tt",
                   choices=["decoder_volume_tt", "decoder_volume_tt_route_head", "decoder_volume", "decoder", "obs_proj"],
                   help="Source for MARL edge observations: compact decoded routing features, full decoder dynamics, or legacy obs_proj.")
    p.add_argument("--wm_edge_cost_source", type=str, default="decoded_speed",
                   choices=["bpr_volume", "decoded_speed", "route_head"],
                   help="Source for WM-derived edge travel times used by imagined progress rewards and route-cost auxiliary ranking.")
    p.add_argument("--wm_candidate_edge_cost_source", type=str, default="auto",
                   choices=["auto", "bpr_volume", "decoded_speed", "route_head"],
                   help="WM edge cost source used only for imagined candidate path generation. auto preserves the default cached candidate behavior.")
    p.add_argument("--wm_decoded_speed_min_ratio", type=float, default=0.05,
                   help="Minimum decoded speed/speed_limit ratio when --wm_edge_cost_source=decoded_speed.")
    p.add_argument("--wm_decoded_speed_max_ratio", type=float, default=1.50,
                   help="Maximum decoded speed/speed_limit ratio when --wm_edge_cost_source=decoded_speed.")
    p.add_argument("--wm_decoded_speed_excess_scale", type=float, default=1.0,
                   help="Blend factor for decoded-speed excess delay: 1.0 gives free_flow/speed_ratio; lower values damp global low-speed bias.")
    p.add_argument("--hdv_internalized", action="store_true",
                   help="Use AV-only actions in imagined rollout; HDV response is internalized in world model")
    p.add_argument("--dreamer_batch_episodes", type=int, default=2,
                   help="Number of imagined episodes to accumulate before a Dreamer actor/critic update.")
    p.add_argument("--dreamer_actor_horizon", type=int, default=8,
                   help="Truncated differentiable imagination horizon for Dreamer actor updates.")
    p.add_argument("--dreamer_departure_cap_mode", type=str, default="none",
                   choices=["cfg", "none"],
                   help=(
                       "Deprecated compatibility switch. Training now always uses the full "
                       "agents.csv schedule and active vehicle counts."
                   ))
    p.add_argument("--dreamer_value_coef", type=float, default=1.0,
                   help="Scale factor for Dreamer critic regression loss.")
    p.add_argument("--dreamer_entropy_coef", type=float, default=1e-3,
                   help="Entropy bonus coefficient for the Dreamer actor objective.")
    p.add_argument("--dreamer_critic_variant", type=str, default="v_only",
                   choices=["v_only", "action_conditioned"],
                   help="Dreamer critic variant. 'action_conditioned' concatenates action_edge to the pooled latent critic input.")
    p.add_argument("--dreamer_critic_aggregation", type=str, default="mean",
                   choices=["mean", "global_attention", "attention", "transformer"],
                   help=(
                       "Global latent aggregation used by the centralized critic. "
                       "global_attention uses a learned query over edge tokens instead "
                       "of mean pooling, closer to a GAWM-style centralized world-state critic."
                   ))
    p.add_argument("--dreamer_critic_attention_heads", type=int, default=4,
                   help="Attention heads for --dreamer_critic_aggregation=global_attention.")
    p.add_argument("--dreamer_critic_context", action=argparse.BooleanOptionalAction, default=True,
                   help="Condition the centralized critic on rollout context such as time, active vehicles, valid routes, and elapsed trip state.")
    p.add_argument("--dreamer_critic_context_dim", type=int, default=8,
                   help="Number of critic context features to consume. The current rollout builder emits 8 features.")
    p.add_argument("--dreamer_slow_critic_tau", type=float, default=0.01,
                   help="Polyak update rate for the target critic used for lambda-return bootstrapping. <=0 disables the target critic.")
    p.add_argument("--dreamer_return_normalizer", action=argparse.BooleanOptionalAction, default=True,
                   help="Use an EMA normalizer for Dreamer critic targets and actor advantages.")
    p.add_argument("--dreamer_return_normalizer_decay", type=float, default=0.99,
                   help="EMA decay for --dreamer_return_normalizer.")
    p.add_argument("--dreamer_vehicle_critic", action=argparse.BooleanOptionalAction, default=True,
                   help="Train a per-vehicle critic for discrete actor-critic sample advantages.")
    p.add_argument("--dreamer_vehicle_critic_lr", type=float, default=0.0,
                   help="Per-vehicle critic learning rate override. <=0 uses the global critic LR.")
    p.add_argument("--dreamer_vehicle_critic_context", action=argparse.BooleanOptionalAction, default=True,
                   help="Condition the per-vehicle critic on the same rollout context as the global critic.")
    p.add_argument("--dreamer_vehicle_critic_architecture", type=str, default="self_attention",
                   choices=["mlp", "self_attention", "attention", "transformer"],
                   help="Per-vehicle critic architecture. self_attention lets active vehicles in the same step attend to each other before value prediction.")
    p.add_argument("--dreamer_vehicle_critic_attention_heads", type=int, default=4,
                   help="Attention heads for the self-attention per-vehicle critic.")
    p.add_argument("--dreamer_vehicle_critic_value_coef", type=float, default=1.0,
                   help="Scale factor for the per-vehicle critic regression loss.")
    p.add_argument("--dreamer_vehicle_critic_return_normalizer",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Use a separate EMA normalizer for per-vehicle critic targets and advantages.")
    p.add_argument("--dreamer_allow_legacy_actor_only_losses", action="store_true",
                   help=(
                       "Allow deprecated actor-only losses for reproducing legacy checkpoints. "
                       "Current training should keep this off and route reward signals through critics."
                   ))
    p.add_argument("--dreamer_actor_lr", type=float, default=0.0,
                   help="Actor learning rate override for Dreamer CTDE. <=0 uses --lr.")
    p.add_argument("--dreamer_actor_lr_decay_start_episode", type=int, default=-1,
                   help="Episode index where actor LR starts linearly decaying. <0 disables decay.")
    p.add_argument("--dreamer_actor_lr_decay_end_episode", type=int, default=-1,
                   help="Episode index where actor LR reaches --dreamer_actor_lr_final_scale * base LR.")
    p.add_argument("--dreamer_actor_lr_final_scale", type=float, default=1.0,
                   help="Final actor LR scale after decay. 0 freezes actor updates while critics can continue.")
    p.add_argument("--dreamer_actor_update_gate_start_episode", type=int, default=-1,
                   help=(
                       "Episode from which route-sharpness actor update gating can skip actor "
                       "optimizer steps while critics continue. <0 disables the start gate."
                   ))
    p.add_argument("--dreamer_actor_update_gate_max_route_frac", type=float, default=0.0,
                   help=(
                       "Skip actor optimizer steps when the current actor's deterministic argmax "
                       "route distribution has any route at or above this fraction. <=0 disables."
                   ))
    p.add_argument("--dreamer_actor_update_gate_route0_frac", type=float, default=0.0,
                   help=(
                       "Skip actor optimizer steps when the current actor's deterministic argmax "
                       "route0 fraction is at or above this value. <=0 disables."
                   ))
    p.add_argument("--dreamer_actor_update_gate_top1_prob", type=float, default=0.0,
                   help=(
                       "Skip actor optimizer steps when mean top-1 policy probability on AC "
                       "samples is at or above this value. <=0 disables."
                   ))
    p.add_argument("--dreamer_actor_update_gate_min_samples", type=float, default=128.0,
                   help="Minimum discrete AC samples required before route-sharpness actor gating can fire.")
    p.add_argument("--dreamer_actor_update_gate_latch",
                   action=argparse.BooleanOptionalAction, default=False,
                   help=(
                       "Once a route-sharpness gate fires, keep skipping later actor optimizer "
                       "steps while critics continue. This prevents small later batches from "
                       "unlocking actor over-sharpening."
                   ))
    p.add_argument("--dreamer_actor_trust_region_kl", type=float, default=0.0,
                   help=(
                       "If >0, measure full categorical KL(old actor || post-step actor) on "
                       "discrete AC samples after each actor step. Reject or retry steps whose "
                       "KL exceeds this trust-region limit. This constrains actor update size "
                       "without adding any direct reward path."
                   ))
    p.add_argument("--dreamer_actor_trust_region_max_argmax_change_frac", type=float, default=0.0,
                   help=(
                       "If >0, reject/retry actor steps whose deterministic route argmax changes "
                       "for more than this weighted fraction of AC snapshot samples."
                   ))
    p.add_argument("--dreamer_actor_trust_region_max_vehicle_argmax_change_frac", type=float, default=0.0,
                   help=(
                       "If >0, reject/retry actor steps when any vehicle group changes its "
                       "deterministic route argmax on more than this fraction of sampled AC "
                       "states. This constrains fine-grained policy drift without adding a "
                       "reward or actor-only loss."
                   ))
    p.add_argument("--dreamer_actor_trust_region_max_route_frac", type=float, default=0.0,
                   help=(
                       "If >0, reject/retry actor steps whose post-step deterministic argmax "
                       "distribution places any route at or above this fraction."
                   ))
    p.add_argument("--dreamer_actor_trust_region_max_route0_frac", type=float, default=0.0,
                   help=(
                       "If >0, reject/retry actor steps whose post-step deterministic argmax "
                       "distribution places route0 at or above this fraction. This constrains "
                       "route0 collapse without adding any actor-only reward term."
                   ))
    p.add_argument("--dreamer_actor_trust_region_max_retries", type=int, default=0,
                   help="Number of smaller-LR retries after a trust-region KL violation.")
    p.add_argument("--dreamer_actor_trust_region_lr_shrink", type=float, default=0.5,
                   help="LR multiplier applied for each actor trust-region retry.")
    p.add_argument("--dreamer_actor_trust_region_min_samples", type=int, default=128,
                   help="Minimum AC snapshot samples required before actor trust-region checks run.")
    p.add_argument("--dreamer_actor_trust_region_max_samples", type=int, default=4096,
                   help="Maximum deterministic AC samples used for actor trust-region KL checks.")
    p.add_argument("--dreamer_actor_anchor_start_episode", type=int, default=-1,
                   help=(
                       "Episode at which to start constraining cumulative actor drift against a "
                       "frozen reference actor. <0 disables the anchor start gate unless another "
                       "anchor limit is enabled with start treated as episode 0."
                   ))
    p.add_argument("--dreamer_actor_anchor_update_interval", type=int, default=0,
                   help=(
                       "If >0, refresh the frozen actor anchor every N accepted actor updates. "
                       "0 keeps the initial/resumed anchor fixed."
                   ))
    p.add_argument("--dreamer_actor_anchor_kl", type=float, default=0.0,
                   help=(
                       "If >0, reject actor steps whose cumulative full categorical KL from the "
                       "frozen actor anchor exceeds this value on discrete AC samples. This is a "
                       "trust-region constraint, not a reward or actor-only route-cost loss."
                   ))
    p.add_argument("--dreamer_actor_anchor_max_argmax_change_frac", type=float, default=0.0,
                   help="If >0, reject actor steps whose argmax differs from the actor anchor on more than this weighted sample fraction.")
    p.add_argument("--dreamer_actor_anchor_max_vehicle_argmax_change_frac", type=float, default=0.0,
                   help=(
                       "If >0, reject actor steps when any vehicle group differs from the "
                       "frozen anchor argmax on more than this fraction of sampled AC states. "
                       "Use this to preserve a SUMO-validated checkpoint while critics keep "
                       "training."
                   ))
    p.add_argument("--dreamer_actor_anchor_max_route_frac", type=float, default=0.0,
                   help=(
                       "If >0, reject actor steps whose post-step deterministic argmax "
                       "distribution places any route above this fraction relative to the anchor samples."
                   ))
    p.add_argument("--dreamer_actor_anchor_max_route0_frac", type=float, default=0.0,
                   help=(
                       "If >0, reject actor steps whose post-step deterministic argmax "
                       "distribution places route0 above this fraction relative to the anchor samples."
                   ))
    p.add_argument("--dreamer_critic_lr", type=float, default=0.0,
                   help="Critic learning rate override for Dreamer CTDE. <=0 uses --lr.")
    p.add_argument("--dreamer_actor_objective", type=str, default="discrete_actor_critic",
                   choices=["direct_return", "critic_q", "direct_plus_q", "discrete_actor_critic", "discrete_ac", "score_function", "reinforce"],
                   help="Dreamer actor objective. direct_return differentiates rewards; discrete_actor_critic uses score-function gradients on sampled route actions.")
    p.add_argument("--dreamer_actor_q_weight", type=float, default=1.0,
                   help="Weight for the action-conditioned critic-Q actor objective when enabled.")
    p.add_argument("--dreamer_ac_advantage_clip", type=float, default=5.0,
                   help="Absolute clip for discrete_actor_critic lambda-return advantages. <=0 disables clipping.")
    p.add_argument("--dreamer_ac_ppo_clip_ratio", type=float, default=0.1,
                   help="PPO-style selected-action probability ratio clip for discrete_actor_critic. <=0 disables clipping.")
    p.add_argument("--dreamer_ac_normalize_combined_advantages",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Normalize combined step and route advantages before discrete_actor_critic clipping.")
    p.add_argument("--dreamer_ac_weight_by_active_agents",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Weight replicated per-vehicle actor samples by inverse active valid-agent count per imagined step.")
    p.add_argument("--dreamer_ac_step_advantage_weight", type=float, default=1.0,
                   help="Weight for step-level Dreamer lambda-return advantage in discrete_actor_critic.")
    p.add_argument("--dreamer_ac_route_advantage_weight", type=float, default=0.0,
                   help=(
                       "Deprecated actor-only selected-route cost advantage. Keep 0.0; "
                       "requires --dreamer_allow_legacy_actor_only_losses for legacy reproduction."
                   ))
    p.add_argument("--dreamer_ac_entropy_target", type=float, default=0.0,
                   help=(
                       "Absolute target entropy for discrete_actor_critic floor regularization. "
                       "<=0 disables the absolute target unless --dreamer_ac_entropy_target_frac is set."
                   ))
    p.add_argument("--dreamer_ac_entropy_target_frac", type=float, default=0.0,
                   help=(
                       "Target entropy as a per-sample fraction of log(valid route count) for "
                       "discrete_actor_critic. Useful for preventing premature categorical "
                       "policy collapse in long training."
                   ))
    p.add_argument("--dreamer_ac_entropy_floor_coef", type=float, default=0.0,
                   help=(
                       "Quadratic penalty coefficient applied only when selected-sample entropy "
                       "falls below the configured entropy target."
                   ))
    p.add_argument("--dreamer_ac_entropy_cost_gap_max", type=float, default=-1.0,
                   help=(
                       "If >=0 and route-cost observations are enabled, compute the entropy-floor "
                       "route count only over candidates whose relative current-cost gap is at most "
                       "this value. This keeps exploration on near-best candidates without adding "
                       "a route-cost actor loss."
                   ))
    p.add_argument("--dreamer_ac_entropy_cost_min_routes", type=int, default=1,
                   help=(
                       "Minimum number of valid candidates kept in the cost-aware entropy-floor "
                       "target when --dreamer_ac_entropy_cost_gap_max is enabled."
                   ))
    p.add_argument("--dreamer_ac_rollout_action_mode", type=str, default="sample",
                   choices=[
                       "sample", "argmax", "scheduled_argmax",
                       "stochastic", "sampled", "sampling",
                       "deterministic", "greedy", "eval", "auto",
                       "mixed", "scheduled", "annealed_argmax",
                   ],
                   help=(
                       "Route action selection used by discrete actor-critic imagined rollouts. "
                       "sample preserves historical stochastic rollout exploration; argmax matches "
                       "deterministic SUMO eval/deployment while still training through the selected "
                       "action log-probability; scheduled_argmax gradually mixes argmax rows after "
                       "a configurable warmup/ramp."
                   ))
    p.add_argument("--dreamer_ac_rollout_argmax_warmup_episodes", type=int, default=0,
                   help="Number of episodes to keep scheduled_argmax rollout fully stochastic.")
    p.add_argument("--dreamer_ac_rollout_argmax_ramp_episodes", type=int, default=0,
                   help="Episodes over which scheduled_argmax linearly ramps toward final argmax mix.")
    p.add_argument("--dreamer_ac_rollout_argmax_final_mix", type=float, default=1.0,
                   help="Final per-vehicle argmax probability for scheduled_argmax rollouts.")
    p.add_argument("--dreamer_ac_route_credit_mode", type=str, default="route_cost",
                   choices=[
                       "route_cost", "planned_pressure", "marginal_pressure",
                       "sequential_marginal", "counterfactual_pressure",
                       "counterfactual_fleet_pressure", "coma_pressure", "rollout_pressure",
                   ],
                   help=(
                       "Cost matrix used for discrete_actor_critic per-vehicle route credit. "
                       "route_cost is the legacy local candidate-path cost; pressure modes use "
                       "centralized planned-fleet externality costs for GAWM-style credit assignment. "
                       "counterfactual_pressure scores same-state per-route fleet pressure cost "
                       "under COMA-style replacement of one vehicle's route."
                   ))
    p.add_argument("--dreamer_ac_route_credit_pressure_weight", type=float, default=1.0,
                   help="Scale for pressure/externality costs when --dreamer_ac_route_credit_mode is not route_cost.")
    p.add_argument("--dreamer_ac_route_credit_pressure_mode", type=str, default="clipped_excess_tt_ratio",
                   choices=[
                       "overlap", "tt_ratio", "excess_tt_ratio", "clipped_excess_tt_ratio",
                       "threshold_excess_tt_ratio", "volume_ratio", "excess_volume_ratio",
                   ],
                   help="Edge weighting mode for discrete_actor_critic planned/marginal pressure route credit.")
    p.add_argument("--dreamer_ac_route_credit_pressure_reference", type=str, default="soft",
                   choices=["soft", "argmax", "sequential_argmax"],
                   help=(
                       "Fleet baseline used by discrete_actor_critic marginal pressure credit. "
                       "soft uses current route probabilities; argmax fixes other vehicles to their "
                       "current deterministic route; sequential_argmax matches sequential fleet pressure."
                   ))
    p.add_argument("--dreamer_ac_route_credit_normalize_by_path_length",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Normalize discrete_actor_critic pressure route credit by candidate path edge count.")
    p.add_argument("--dreamer_ac_route_credit_base_cost_mode", type=str, default="progress_fraction",
                   choices=[
                       "progress_fraction", "remaining_tt_inverse", "neg_inverse_remaining_tt",
                       "step_progress", "full_path", "remaining_tt", "route_tt", "legacy",
                       "none", "zero", "disabled",
                   ],
                   help=(
                       "Base cost used by AC route credit before pressure/externality terms. "
                       "progress_fraction uses -step_interval / (elapsed_time + remaining_route_tt) "
                       "so one decision contributes trip-progress credit without over-rewarding "
                       "short late-trip suffixes."
                   ))
    p.add_argument("--dreamer_ac_route_credit_remaining_tt_eps", type=float, default=1.0,
                   help="Minimum elapsed-plus-remaining route TT in seconds for progress-fraction AC route credit.")
    p.add_argument("--dreamer_ac_route_credit_progress_cap", type=float, default=1.0,
                   help="Upper clip for step_interval / (elapsed_time + remaining_route_tt) in AC progress-fraction route credit; <=0 disables the cap.")
    p.add_argument("--dreamer_ac_route_credit_include_base_cost",
                   action=argparse.BooleanOptionalAction, default=True,
                   help=(
                       "Include base route travel-time cost in pressure-mode discrete_actor_critic "
                       "route credit. Disable to ablate pure planned/marginal pressure credit."
                   ))
    p.add_argument("--dreamer_ac_route_advantage_temperature", type=float, default=60.0,
                   help="Seconds scale for discrete_actor_critic route advantages when normalization is disabled.")
    p.add_argument("--dreamer_ac_normalize_route_advantages", action=argparse.BooleanOptionalAction, default=True,
                   help="Normalize discrete_actor_critic per-vehicle route advantages by valid candidate cost std.")
    p.add_argument("--dreamer_action_mode", type=str, default="expected",
                   choices=["expected", "straight_through_sample", "straight_through_argmax", "st_sample", "st_argmax"],
                   help="How imagined rollouts convert actor logits to edge actions. expected is the legacy soft mixture; straight-through modes use discrete forward route choices with softmax gradients.")
    p.add_argument("--dreamer_action_temperature", type=float, default=1.0,
                   help="Softmax temperature for the imagined action distribution used by --dreamer_action_mode.")
    p.add_argument("--dreamer_all_reward_weight", type=float, default=1.0,
                   help="Weight for all-vehicle BPR travel-time cost in differentiable Dreamer reward.")
    p.add_argument("--dreamer_remaining_time_progress_weight", type=float, default=0.0,
                   help="Weight for softmin remaining-time progress reward R(s_k)-R(s_{k+1}) from current-edge candidate path costs.")
    p.add_argument("--dreamer_remaining_time_softmin_temperature", type=float, default=30.0,
                   help="Temperature in seconds for remaining-time softmin over candidate route costs; lower values approximate hard shortest path.")
    p.add_argument("--dreamer_current_edge_reward_weight", type=float, default=0.0,
                   help="Weight for active AV current-edge BPR travel-time cost in differentiable Dreamer reward.")
    p.add_argument("--dreamer_cav_reward_weight", type=float, default=0.0,
                   help="Weight for CAV/action-edge route travel-time cost in differentiable Dreamer reward.")
    p.add_argument("--dreamer_route_cost_reward_weight", type=float, default=0.0,
                   help="Weight for non-leaky candidate-path route cost computed from WM edge travel times.")
    p.add_argument("--dreamer_route_cost_reward_scope", type=str, default="both",
                   choices=["both", "global", "global_critic", "step", "vehicle", "vehicle_critic", "per_vehicle"],
                   help=(
                       "Where --dreamer_route_cost_reward_weight is applied. Use vehicle_critic "
                       "for the current per-vehicle actor-critic mainline so route progress "
                       "trains the vehicle critic without creating a noisy global step-value target."
                   ))
    p.add_argument("--dreamer_route_cost_reward_mode", type=str, default="progress_fraction",
                   choices=[
                       "progress_fraction", "step_progress", "trip_progress_fraction",
                       "route_tt", "remaining_tt", "full_path", "legacy",
                   ],
                   help=(
                       "Per-vehicle critic route reward readout. progress_fraction rewards "
                       "min(step_interval, remaining_route_tt)/(elapsed+remaining_route_tt), "
                       "so final steps that arrive early cannot exceed one step of progress; "
                       "route_tt/legacy keeps the old negative remaining-route-seconds reward."
                   ))
    p.add_argument("--dreamer_pred_t_cav_proxy_reward_weight", type=float, default=0.0,
                   help=(
                       "Weight for the pred_t_cav_proxy-style CAV route-plan cost in Dreamer reward. "
                       "This is the actor expected selected-route cost under WM edge travel times, "
                       "matching the reranker proxy semantics without CE/distillation or route-slot reordering."
                   ))
    p.add_argument("--dreamer_route_cost_reward_normalize_regret", action="store_true",
                   help=(
                       "Use per-agent normalized route regret (C-min(C))/temperature for route-cost "
                       "Dreamer reward terms instead of raw seconds."
                   ))
    p.add_argument("--dreamer_route_cost_reward_temperature", type=float, default=60.0,
                   help="Seconds temperature used by --dreamer_route_cost_reward_normalize_regret.")
    p.add_argument("--dreamer_route_cost_reward_planned_av_pressure_weight", type=float, default=0.0,
                   help=(
                       "Seconds penalty for same-tick planned AV edge-pressure overlap added to "
                       "route-cost Dreamer reward terms. This enters reward/advantage, not the "
                       "actor-only expected-cost loss."
                   ))
    p.add_argument("--dreamer_route_cost_reward_planned_av_pressure_mode", type=str, default="overlap",
                   choices=[
                       "overlap", "tt_ratio", "excess_tt_ratio", "clipped_excess_tt_ratio",
                       "threshold_excess_tt_ratio", "volume_ratio", "excess_volume_ratio",
                   ],
                   help=(
                       "How to weight planned AV pressure in route-cost reward. overlap is pure "
                       "planned path overlap; excess_tt_ratio weights pressure by WM decoded "
                       "travel-time excess over free-flow."
                   ))
    p.add_argument("--dreamer_vehicle_route_credit_reward_weight", type=float, default=0.0,
                   help=(
                       "Critic-mediated selected-route externality reward weight. This converts "
                       "the selected route-credit cost matrix into dreamer_ac_vehicle_reward, so "
                       "the actor sees it only through the per-vehicle critic advantage."
                   ))
    p.add_argument("--dreamer_vehicle_route_credit_reward_discount", type=float, default=0.99,
                   help=(
                       "Per-vehicle critic discount for pure non-progress route-credit ablations. "
                       "If a progress-fraction route credit base is enabled, discount remains 1.0."
                   ))
    p.add_argument("--dreamer_vehicle_route_credit_reward_transform", type=str, default="negative_cost",
                   choices=[
                       "negative_cost", "cost", "legacy", "absolute_cost",
                       "relative_mean", "mean", "mean_baseline", "advantage",
                       "relative_min", "min_regret", "best_regret",
                       "zscore", "relative_zscore", "standardized_mean",
                   ],
                   help=(
                       "How critic-routed selected route-credit costs become per-vehicle rewards. "
                       "negative_cost keeps the old absolute-cost target; relative_mean/zscore "
                       "use same-vehicle candidate baselines so the vehicle critic sees "
                       "action-discriminative credit without any actor-only loss."
                   ))
    p.add_argument("--dreamer_vehicle_route_credit_reward_normalize",
                   action=argparse.BooleanOptionalAction, default=False,
                   help=(
                       "Normalize relative critic-routed route-credit rewards by per-vehicle "
                       "candidate cost std before they enter the vehicle critic target."
                   ))
    p.add_argument("--dreamer_vehicle_route_credit_reward_temperature", type=float, default=1.0,
                   help="Scale divisor for non-normalized relative route-credit rewards.")
    p.add_argument("--dreamer_vehicle_route_credit_mode", type=str, default="counterfactual_pressure",
                   choices=[
                       "route_cost", "planned_pressure", "marginal_pressure",
                       "sequential_marginal", "counterfactual_pressure",
                       "counterfactual_fleet_pressure", "coma_pressure", "rollout_pressure",
                   ],
                   help=(
                       "Route-credit cost mode used for critic-routed selected-route vehicle reward. "
                       "Pressure modes are action-conditioned per vehicle; counterfactual_pressure "
                       "scores each candidate as a same-state fleet externality."
                   ))
    p.add_argument("--dreamer_vehicle_route_credit_pressure_weight", type=float, default=1.0,
                   help="Scale for pressure/externality costs in critic-routed route-credit reward.")
    p.add_argument("--dreamer_vehicle_route_credit_pressure_mode", type=str, default="clipped_excess_tt_ratio",
                   choices=[
                       "overlap", "tt_ratio", "excess_tt_ratio", "clipped_excess_tt_ratio",
                       "threshold_excess_tt_ratio", "volume_ratio", "excess_volume_ratio",
                   ],
                   help="Edge weighting mode for critic-routed route-credit pressure/externality costs.")
    p.add_argument("--dreamer_vehicle_route_credit_pressure_reference", type=str, default="soft",
                   choices=["soft", "argmax", "sequential_argmax"],
                   help="Fleet baseline for critic-routed marginal route-credit modes.")
    p.add_argument("--dreamer_vehicle_route_credit_pressure_scale", type=str, default="mean",
                   choices=["mean", "avg", "average", "per_vehicle", "sum", "total", "system", "system_total", "fleet_total"],
                   help=(
                       "Scale for critic-routed counterfactual pressure costs. mean preserves the "
                       "old per-vehicle-average pressure; system_total removes the active-agent "
                       "normalization so each vehicle critic sees system-level externality scale."
                   ))
    p.add_argument("--dreamer_vehicle_route_credit_normalize_by_path_length",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Normalize critic-routed pressure route credit by candidate path edge count.")
    p.add_argument("--dreamer_vehicle_route_credit_base_cost_mode", type=str, default="none",
                   choices=[
                       "progress_fraction", "remaining_tt_inverse", "neg_inverse_remaining_tt",
                       "step_progress", "full_path", "remaining_tt", "route_tt", "legacy",
                       "none", "zero", "disabled",
                   ],
                   help=(
                       "Base route cost mixed into critic-routed route credit before pressure "
                       "terms. Default none keeps existing route progress reward separate."
                   ))
    p.add_argument("--dreamer_vehicle_route_credit_include_base_cost",
                   action=argparse.BooleanOptionalAction, default=False,
                   help=(
                       "Include the base route cost in critic-routed pressure route credit. "
                       "Default false makes this a pure action-conditioned social externality."
                   ))
    p.add_argument("--dreamer_vehicle_route_credit_cost_gate_max_seconds", type=float, default=-1.0,
                   help=(
                       "Enable a local route-cost gate for critic-routed route-credit reward. "
                       "When >=0, externality credit is only assigned to candidates within this "
                       "many seconds of the fastest valid candidate, so pressure can break near "
                       "ties but cannot reward clearly slower routes."
                   ))
    p.add_argument("--dreamer_vehicle_route_credit_cost_gate_max_ratio", type=float, default=0.0,
                   help=(
                       "Optional relative route-cost gate for critic-routed route-credit reward. "
                       "A value of 0.10 keeps candidates within 10%% of the fastest local route. "
                       "The effective gate uses the looser of seconds and ratio thresholds."
                   ))
    p.add_argument("--dreamer_vehicle_route_credit_cost_gate_min_routes", type=int, default=1,
                   help=(
                       "Minimum number of valid candidates kept by the route-credit cost gate. "
                       "Use 1 to keep only near-tie routes; larger values preserve more "
                       "externality exploration while still ordering by local route cost."
                   ))
    p.add_argument("--dreamer_vehicle_social_reward_weight", type=float, default=0.0,
                   help=(
                       "Critic-mediated per-vehicle social reward weight. This broadcasts a WM "
                       "global congestion/travel-time readout into dreamer_ac_vehicle_reward, so "
                       "it affects the actor only through the per-vehicle critic advantage."
                   ))
    p.add_argument("--dreamer_vehicle_social_reward_source", type=str, default="excess_all_tt",
                   choices=[
                       "all_tt", "mean_tt", "global_tt", "system_tt",
                       "excess_all_tt", "delay", "mean_delay", "excess_tt",
                       "total_tt", "system_total_tt",
                       "total_excess_tt", "system_delay", "total_delay",
                   ],
                   help=(
                       "WM decoded social-cost readout for --dreamer_vehicle_social_reward_weight. "
                       "excess_all_tt is mean vehicle delay over free-flow; all_tt is mean travel "
                       "time; total_* variants use system totals and should normally be paired "
                       "with active_av/valid_vehicle distribution."
                   ))
    p.add_argument("--dreamer_vehicle_social_reward_distribution", type=str, default="per_vehicle",
                   choices=["per_vehicle", "broadcast", "each", "active_av", "active_vehicle", "active_av_mean", "valid_vehicle", "valid_av", "valid_agent"],
                   help=(
                       "How the scalar social reward is assigned to active vehicle samples. "
                       "per_vehicle broadcasts the same team reward; active_av/valid_vehicle "
                       "divide by a vehicle count to keep the sum near the scalar global reward."
                   ))
    p.add_argument("--dreamer_vehicle_social_reward_discount", type=float, default=0.99,
                   help=(
                       "Per-vehicle critic discount used only for pure social-reward ablations. "
                       "If route-cost/progress reward is enabled, vehicle return discount remains 1.0."
                   ))
    p.add_argument("--dreamer_expected_cost_weight", type=float, default=0.0,
                   help=(
                       "Deprecated actor-only expected-cost loss weight. Keep 0.0; "
                       "requires --dreamer_allow_legacy_actor_only_losses for legacy reproduction."
                   ))
    p.add_argument("--dreamer_expected_cost_temperature", type=float, default=60.0,
                   help="Seconds scale for the actor expected-cost loss; costs are normalized as regret / temperature.")
    p.add_argument("--dreamer_expected_cost_planned_av_pressure_weight", type=float, default=0.0,
                   help="Seconds penalty for same-tick planned AV edge-pressure overlap added to expected route costs.")
    p.add_argument("--dreamer_expected_cost_planned_av_pressure_mode", type=str, default="overlap",
                   choices=["overlap", "sum_overlap", "mean_overlap"],
                   help=(
                       "How to aggregate planned AV pressure for expected-cost training. "
                       "overlap/sum_overlap sums edge pressure along the route; mean_overlap "
                       "divides by route edge count to avoid length-driven pressure blow-up."
                   ))
    p.add_argument("--dreamer_expected_cost_marginal_av_pressure_weight", type=float, default=0.0,
                   help=(
                       "Seconds penalty for COMA-style edge externality in expected-cost actor loss. "
                       "Each candidate route is scored against other vehicles' current planned edge pressure."
                   ))
    p.add_argument("--dreamer_expected_cost_marginal_av_pressure_mode", type=str, default="excess_tt_ratio",
                   choices=[
                       "overlap", "tt_ratio", "excess_tt_ratio", "clipped_excess_tt_ratio",
                       "threshold_excess_tt_ratio", "volume_ratio", "excess_volume_ratio",
                   ],
                   help=(
                       "Edge weights for expected-cost marginal AV pressure. excess_tt_ratio focuses "
                       "credit on currently delayed WM-decoded edges."
                   ))
    p.add_argument("--dreamer_expected_cost_marginal_av_pressure_reference", type=str, default="soft",
                   choices=["soft", "argmax", "sequential_argmax"],
                   help=(
                       "Fleet baseline used by expected-cost marginal AV pressure. soft uses the "
                       "current policy distribution; argmax fixes other vehicles to their current "
                       "deterministic route plan; sequential_argmax matches the actor's sequential "
                       "fleet-pressure observation."
                   ))
    p.add_argument("--dreamer_expected_cost_marginal_av_pressure_normalize_by_path_length",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Normalize expected-cost marginal AV pressure by candidate path edge count.")
    p.add_argument("--dreamer_expected_cost_detach_cost", action=argparse.BooleanOptionalAction, default=True,
                   help="Detach expected route costs so gradients flow through actor probabilities, not through the cost readout.")
    p.add_argument("--dreamer_expected_cost_balance_best_classes", action="store_true",
                   help="Inverse-frequency weight actor expected-cost samples by current best route slot. This remains an expected-cost objective, not CE/distillation.")
    p.add_argument("--dreamer_expected_cost_pairwise", action="store_true",
                   help=(
                       "Use pairwise cost-ranking loss under the expected-cost objective: "
                       "lower-cost candidate routes are pushed to have higher logits than "
                       "higher-cost routes, without CE labels or route-slot distillation."
                   ))
    p.add_argument("--dreamer_expected_cost_pairwise_margin", type=float, default=0.0,
                   help="Optional logit margin for --dreamer_expected_cost_pairwise.")
    p.add_argument("--dreamer_expected_cost_route0_margin_weight", type=float, default=0.0,
                   help=(
                       "Cost-derived boundary-crossing loss weight. When a nonzero route is "
                       "cheaper than route0 by --dreamer_expected_cost_route0_margin_min_gap, "
                       "push that route's logit above route0 by --dreamer_expected_cost_route0_margin. "
                       "This is based on the current cost matrix, not CE/distillation labels."
                   ))
    p.add_argument("--dreamer_expected_cost_route0_margin", type=float, default=0.0,
                   help="Logit margin used by --dreamer_expected_cost_route0_margin_weight.")
    p.add_argument("--dreamer_expected_cost_route0_margin_min_gap", type=float, default=0.0,
                   help=(
                       "Minimum seconds gap C(route0)-C(best_nonzero) required before "
                       "--dreamer_expected_cost_route0_margin_weight applies."
                   ))
    p.add_argument("--dreamer_expected_cost_nonzero_best_boost", type=float, default=0.0,
                   help=(
                       "Finite sample-weight boost for expected-cost samples whose best route "
                       "slot is nonzero. This is cost-gap gated and avoids inverse-frequency "
                       "overcorrection from full class balancing."
                   ))
    p.add_argument("--dreamer_expected_cost_nonzero_best_min_gap", type=float, default=0.0,
                   help=(
                       "Minimum best-vs-second-best cost gap in seconds required before "
                       "--dreamer_expected_cost_nonzero_best_boost is applied."
                   ))
    p.add_argument("--dreamer_expected_cost_route_share_penalty_weight", type=float, default=0.0,
                   help=(
                       "Seconds penalty added to expected-cost candidate costs when the current "
                       "fleet probability share of that route slot exceeds "
                       "--dreamer_expected_cost_route_share_penalty_cap."
                   ))
    p.add_argument("--dreamer_expected_cost_route_share_penalty_cap", type=float, default=1.0,
                   help="Fleet route-slot share cap used by --dreamer_expected_cost_route_share_penalty_weight.")
    p.add_argument("--dreamer_grpo_route_cost_weight", type=float, default=0.0,
                   help=(
                       "Deprecated actor-only per-vehicle GRPO route-cost loss weight. Keep 0.0; "
                       "requires --dreamer_allow_legacy_actor_only_losses for legacy reproduction."
                   ))
    p.add_argument("--dreamer_grpo_planned_av_pressure_weight", type=float, default=0.0,
                   help="Seconds penalty for same-tick planned AV edge-pressure overlap added to GRPO route costs.")
    p.add_argument("--dreamer_grpo_planned_av_pressure_mode", type=str, default="overlap",
                   choices=[
                       "overlap", "tt_ratio", "excess_tt_ratio", "clipped_excess_tt_ratio",
                       "threshold_excess_tt_ratio", "volume_ratio", "excess_volume_ratio",
                   ],
                   help=(
                       "How to weight planned AV pressure in GRPO costs. overlap is pure planned "
                       "path overlap; excess_tt_ratio weights pressure by WM decoded travel-time excess."
                   ))
    p.add_argument("--dreamer_grpo_marginal_av_pressure_weight", type=float, default=0.0,
                   help=(
                       "Seconds penalty for COMA-style edge externality in GRPO route costs. "
                       "Each vehicle candidate is scored against other vehicles' expected planned edge pressure."
                   ))
    p.add_argument("--dreamer_grpo_marginal_av_pressure_mode", type=str, default="excess_tt_ratio",
                   choices=[
                       "overlap", "tt_ratio", "excess_tt_ratio", "clipped_excess_tt_ratio",
                       "threshold_excess_tt_ratio", "volume_ratio", "excess_volume_ratio",
                   ],
                   help=(
                       "Edge weights for GRPO marginal AV pressure. excess_tt_ratio focuses "
                       "credit on currently delayed WM-decoded edges."
                   ))
    p.add_argument("--dreamer_grpo_marginal_av_pressure_reference", type=str, default="soft",
                   choices=["soft", "argmax", "sequential_argmax"],
                   help=(
                       "Fleet baseline used by GRPO marginal AV pressure. soft uses the current "
                       "policy distribution; argmax fixes other vehicles to their current "
                       "deterministic route plan; sequential_argmax matches the actor's sequential "
                       "fleet-pressure observation."
                   ))
    p.add_argument("--dreamer_grpo_marginal_av_pressure_normalize_by_path_length",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Normalize GRPO marginal AV pressure by candidate path edge count.")
    p.add_argument("--dreamer_grpo_advantage_temperature", type=float, default=60.0,
                   help="Seconds scale for GRPO advantages when advantage normalization is disabled.")
    p.add_argument("--dreamer_grpo_normalize_advantages", action=argparse.BooleanOptionalAction, default=True,
                   help="Normalize GRPO advantages by each vehicle group's valid route-cost standard deviation.")
    p.add_argument("--dreamer_grpo_route_cost_pairwise_weight", type=float, default=0.0,
                   help=(
                       "Extra non-distillation pairwise ranking term inside GRPO route-cost loss. "
                       "For valid route pairs, lower-cost routes are encouraged to have higher logits."
                   ))
    p.add_argument("--dreamer_grpo_route_cost_pairwise_margin", type=float, default=0.0,
                   help="Optional logit margin for --dreamer_grpo_route_cost_pairwise_weight.")
    p.add_argument("--dreamer_grpo_route_share_penalty_weight", type=float, default=0.0,
                   help=(
                       "Seconds penalty added to GRPO candidate costs when the current fleet probability "
                       "share of that original route slot exceeds --dreamer_grpo_route_share_penalty_cap."
                   ))
    p.add_argument("--dreamer_grpo_route_share_penalty_cap", type=float, default=1.0,
                   help="Fleet route-slot share cap used by --dreamer_grpo_route_share_penalty_weight.")
    p.add_argument("--dreamer_grpo_combined_advantage_clip", type=float, default=5.0,
                   help=(
                       "Clip absolute value of the combined route-cost and rollout advantage before "
                       "policy-gradient updates. Set <=0 to disable."
                   ))
    p.add_argument("--dreamer_grpo_policy_clip_ratio", type=float, default=0.2,
                   help=(
                       "PPO-style probability-ratio clip for the combined GRPO policy-gradient update. "
                       "Set <=0 to disable clipping."
                   ))
    p.add_argument("--dreamer_grpo_policy_kl_coef", type=float, default=0.0,
                   help=(
                       "Optional KL penalty coefficient for the combined GRPO policy gradient. "
                       "The KL is measured against the trajectory's stored selected-action logprob."
                   ))
    p.add_argument("--dreamer_grpo_rollout_weight", type=float, default=0.0,
                   help=(
                       "Deprecated critic-free stochastic rollout GRPO weight. Keep 0.0; "
                       "requires --dreamer_allow_legacy_actor_only_losses for legacy reproduction."
                   ))
    p.add_argument("--dreamer_grpo_rollout_normalize_advantages", action=argparse.BooleanOptionalAction, default=True,
                   help="Normalize rollout GRPO advantages across the imagined rollout group/batch.")
    p.add_argument("--dreamer_grpo_rollout_cost_source", type=str, default="reward",
                   choices=["reward", "pressure", "cav_proxy", "pressure_cav"],
                   help=(
                       "Rollout-level GRPO cost readout. reward uses -Dreamer reward; pressure uses "
                       "planned edge pressure dotted with rollout edge travel times; cav_proxy uses "
                       "sampled-route cost; pressure_cav adds pressure + t_cav_weight*cav_proxy."
                   ))
    p.add_argument("--dreamer_grpo_rollout_discount", type=float, default=0.99,
                   help="Discount for averaging per-step rollout GRPO costs.")
    p.add_argument("--dreamer_grpo_rollout_t_cav_weight", type=float, default=0.0,
                   help="CAV proxy weight used when --dreamer_grpo_rollout_cost_source=pressure_cav.")
    p.add_argument("--dreamer_grpo_rollout_group_size", type=int, default=1,
                   help=(
                       "Number of stochastic imagined rollouts to sample from the same start state "
                       "for rollout GRPO. Values >1 use same-state group-relative advantages."
                   ))
    p.add_argument("--dreamer_grpo_rollout_vehicle_cost_weight", type=float, default=0.0,
                   help=(
                       "Deprecated actor-only per-vehicle score-function loss inside imagined rollouts. "
                       "Keep 0.0; requires --dreamer_allow_legacy_actor_only_losses for legacy reproduction."
                   ))
    p.add_argument("--dreamer_grpo_rollout_vehicle_planned_av_pressure_weight", type=float, default=0.0,
                   help=(
                       "Optional sampled-fleet planned-pressure add-on for "
                       "--dreamer_grpo_rollout_vehicle_cost_weight."
                   ))
    p.add_argument("--dreamer_grpo_rollout_baseline_policy", type=str, default="none",
                   choices=["none", "route_cost_greedy", "route_cost_pressure_greedy"],
                   help=(
                       "Optional same-start counterfactual rollout baseline for rollout GRPO. "
                       "The baseline cost is used only for advantage = baseline_cost - sampled_cost; "
                       "its route choices are not used as supervised labels."
                   ))
    p.add_argument("--dreamer_grpo_rollout_baseline_planned_av_pressure_weight", type=float, default=0.0,
                   help=(
                       "Sequential planned-AV pressure penalty used only when "
                       "--dreamer_grpo_rollout_baseline_policy=route_cost_pressure_greedy."
                   ))
    rollout_rewards.add_rollout_reward_args(p)
    p.add_argument("--dreamer_hdv_background_mode", type=str, default="none",
                   choices=["none", "uniform", "adapted_state", "episode_av_pressure_response", "episode_hdv_head_response"],
                   help=(
                       "Episode-level fixed HDV background injected into vehicle-mode world-model dynamics. "
                       "uniform uses per-OD uniform static route fractions; adapted_state loads a RouteRL/collector "
                       "HDV learning-state JSON and projects its static route distribution to one edge-aligned field; "
                       "episode_av_pressure_response first probes the actor's full-episode AV pressure and builds "
                       "a Wardrop-style HDV response background from that pressure; episode_hdv_head_response "
                       "uses the world-model HDV response head for that same episode-level mapping."
                   ))
    p.add_argument("--dreamer_hdv_background_state", type=str, default="",
                   help="Optional HDV learning-state JSON for --dreamer_hdv_background_mode=adapted_state.")
    p.add_argument("--dreamer_hdv_background_weight", type=float, default=1.0,
                   help="Multiplier for the fixed HDV background before it is injected into WM dynamics.")
    p.add_argument("--dreamer_hdv_background_combine", type=str, default="additive",
                   choices=["additive", "population"],
                   help=(
                       "How to combine AV planned edge action with fixed HDV background for WM dynamics. "
                       "additive keeps AV action scale and adds weight*HDV; population forms a population-weighted blend."
                   ))
    p.add_argument("--dreamer_hdv_response_pressure_weight", type=float, default=60.0,
                   help="Seconds penalty per edge of episode-level AV planned pressure when building HDV response routes.")
    p.add_argument("--dreamer_hdv_response_temperature", type=float, default=60.0,
                   help="Soft Wardrop/Gawron temperature in seconds for the episode-level HDV response.")
    p.add_argument("--dreamer_hdv_response_prior_mix", type=float, default=0.0,
                   help="Blend factor toward adapted_state/uniform HDV prior when using episode_av_pressure_response.")
    p.add_argument("--dreamer_hdv_response_head_fallback", type=str, default="analytic",
                   choices=["analytic", "prior", "error"],
                   help=(
                       "Fallback when --dreamer_hdv_background_mode=episode_hdv_head_response has no usable head. "
                       "analytic uses the Wardrop-style response; prior uses the loaded adapted/uniform background; "
                       "error raises immediately."
                   ))
    p.add_argument("--dreamer_action_congestion_weight", type=float, default=0.0,
                   help="Weight for non-leaky action-edge concentration penalty computed from actor route load and edge capacities.")
    p.add_argument("--dreamer_route_balance_weight", type=float, default=0.0,
                   help="Weight for non-leaky fleet-level route concentration penalty sum(mean_route_prob^2)-1/K.")
    p.add_argument("--dreamer_route_max_share_weight", type=float, default=0.0,
                   help="Weight for non-leaky max-route-share cap penalty relu(max(mean_route_prob)-cap)^2.")
    p.add_argument("--dreamer_route_max_share_cap", type=float, default=1.0,
                   help="Allowed fleet-level max route share before --dreamer_route_max_share_weight applies.")
    p.add_argument("--dreamer_route_cost_aux_weight", type=float, default=0.0,
                   help=(
                       "Deprecated actor-only route-cost auxiliary supervision weight. Keep 0.0; "
                       "requires --dreamer_allow_legacy_actor_only_losses for legacy reproduction."
                   ))
    p.add_argument("--dreamer_route_cost_aux_balance", action="store_true",
                   help="Balance route-cost auxiliary samples by the current best-route class to avoid route0-majority collapse.")
    p.add_argument("--dreamer_route_cost_aux_hard_target", action="store_true",
                   help="Use hard current-best route slot labels for route-cost auxiliary supervision instead of soft cost targets.")
    p.add_argument("--dreamer_route_cost_aux_nonzero_margin", type=float, default=0.0,
                   help="Extra logit margin penalty applied when the current best valid route slot is not route0.")
    p.add_argument("--dreamer_route_cost_aux_temperature", type=float, default=30.0,
                   help="Temperature in seconds for soft route-cost auxiliary targets; lower values make shortest-route targets sharper.")
    p.add_argument("--dreamer_horizon_proxy_ckpt", type=str, default="",
                   help="Optional frozen action-conditioned horizon proxy checkpoint for long-horizon t_all cost.")
    p.add_argument("--dreamer_horizon_proxy_reward_weight", type=float, default=0.0,
                   help="Weight on the horizon proxy predicted t_all cost (seconds) inside Dreamer reward.")
    p.add_argument("--dreamer_actor_min_valid_agents_ratio", type=float, default=0.0,
                   help="Optional actor-only trajectory quality gate: skip actor loss for fresh trajectories whose n_valid_path_agents is below this fraction of the batch max. 0 disables the gate.")
    p.add_argument("--dreamer_normalize_critic_targets", action="store_true",
                   help=(
                       "Normalize Dreamer lambda-return critic targets with detached per-trajectory "
                       "mean/std before value loss to reduce return-scale variance."
                   ))
    p.add_argument("--dreamer_normalize_actor_advantages", action="store_true",
                   help=(
                       "Normalize Dreamer actor advantages with detached per-trajectory mean/std. "
                       "Detached stats preserve gradients through differentiable rewards while stabilizing scale."
                   ))
    p.add_argument("--dreamer_disable_actor_updates", action="store_true",
                   help="Disable Dreamer actor optimizer updates while keeping critic training and rollout generation active.")
    p.add_argument("--dreamer_preserve_partial_batch_on_exit", action="store_true",
                   help="Preserve leftover Dreamer pending trajectories in checkpoint extra_state instead of force-flushing them at chunk end.")
    p.add_argument("--checkpoint_dir",    type=str,   default="logs/marl")
    p.add_argument("--checkpoint_every",  type=int,   default=100)
    p.add_argument("--resume_ckpt",       type=str,   default="")
    p.add_argument("--resume_reset_actor_optimizer", action="store_true",
                   help="Load Dreamer actor/critic weights from resume_ckpt but reinitialize the actor optimizer state.")
    # ── dynamic imagination (vehicle mode) ──────────────────────────────────
    p.add_argument("--imagination_horizon", type=int, default=15,
                   help="Vehicle-mode imagined rollout horizon.")
    p.add_argument("--episode_steps", type=int, default=60,
                   help="Vehicle-mode full episode length used for random-start replay.")
    p.add_argument("--step_interval", type=float, default=30.0,
                   help="Vehicle-mode decision interval in seconds.")
    # ── mean field multi-agent ────────────────────────────────────────────────
    p.add_argument("--agents_csv",         type=str,   default="",
                   help="agents.csv for vehicle counts")
    p.add_argument("--ratio_machines",     type=float, default=0.2,
                   help="AV fraction for vehicle split")
    p.add_argument("--network_folder",     type=str,   default="",
                   help="Network folder for vehicle-level routing")
    p.add_argument("--network_name",       type=str,   default="",
                   help="Network name (e.g. ingolstadt_custom)")
    p.add_argument("--agent_level", type=str, default="vehicle", choices=["vehicle"],
                   help="Agent granularity for training. Only vehicle-level mode is supported.")
    p.add_argument("--vehicle_path_penalty_factor", type=float, default=3.0,
                   help="Penalty multiplier for diverse K-path generation in vehicle mode.")
    p.add_argument("--vehicle_candidate_source", type=str, default="current_to_dest",
                   choices=["current_to_dest", "od_suffix"],
                   help="Vehicle imagined candidate action space. Use od_suffix to match urb_fair od_suffix evaluation.")
    p.add_argument("--vehicle_path_cache_mode", type=str, default="lazy",
                   choices=["lazy", "precompute"],
                   help=(
                       "Path cache strategy for current_to_dest vehicle candidates. "
                       "lazy computes only queried (current_edge, dest_edge) pairs; "
                       "precompute eagerly builds all edge-destination pairs and is only "
                       "reasonable for small networks."
                   ))
    p.add_argument("--vehicle_path_cache_max_entries", type=int, default=250000,
                   help="Maximum cached current_to_dest path entries when --vehicle_path_cache_mode=lazy.")
    p.add_argument("--vehicle_routes_csv", type=str, default="",
                   help="routes.csv artifact used when --vehicle_candidate_source=od_suffix.")
    p.add_argument("--vehicle_reward_scale", type=float, default=1.0,
                   help="Scale factor for per-vehicle marginal-delay reward in vehicle mode.")
    p.add_argument("--no_vehicle_obs_global", action="store_true",
                   help="In vehicle mode, do not append global edge summary to per-vehicle observation.")
    p.add_argument("--vehicle_obs_route_stats", type=str, default="mean",
                   choices=["mean", "bottleneck"],
                   help=(
                       "Route feature aggregation for vehicle actor observations. "
                       "'mean' preserves old checkpoint shape; 'bottleneck' adds route max/q90/sum/length features."
                   ))
    p.add_argument("--vehicle_obs_include_route_costs", action="store_true",
                   help=(
                       "Append current-state candidate route-cost features to vehicle actor observations. "
                       "Uses WM-derived edge costs only; no future labels or route-cost auxiliary teacher."
                   ))
    p.add_argument("--vehicle_obs_include_fleet_pressure", action="store_true",
                   help=(
                       "Append same-decision sequential fleet context to each vehicle observation: "
                       "already-selected route shares and candidate overlap with already-planned AV edge pressure."
                   ))
    p.add_argument("--vehicle_actor_architecture", type=str, default="mlp",
                   choices=["mlp", "route_shared"],
                   help=(
                       "Vehicle actor architecture. mlp preserves the old concatenated-observation actor; "
                       "route_shared scores each candidate slot with shared parameters over its own path/cost/fleet features."
                   ))
    p.add_argument("--vehicle_actor_route_cost_logit_bias", type=float, default=0.0,
                   help=(
                       "When route-cost obs is enabled, subtract this multiple of the "
                       "per-route relative cost gap from actor logits."
                   ))
    p.add_argument("--no_intention_obs",   action="store_true",
                   help="Disable intent-enriched observation (prev AV + current HDV fracs)")
    p.add_argument("--device",            type=str,   default="cuda")
    p.add_argument("--no_amp",            action="store_true")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--log_every",         type=int,   default=10)
    return p


def main() -> None:
    p = build_arg_parser()
    settings_path = parse_settings_arg()
    apply_stage_defaults(p, settings_path, stage="marl")
    args = p.parse_args()
    d = vars(args)
    d.pop("settings", None)
    d["use_amp"]        = not d.pop("no_amp")
    d["hdv_gawron"]     = not d.pop("no_hdv_gawron")
    d["include_intention_obs"] = not d.pop("no_intention_obs")
    d["vehicle_obs_include_global"] = not d.pop("no_vehicle_obs_global")
    d["use_terminal_tall_reward"] = (
        not d.pop("no_terminal_tall_reward")
        and abs(float(d.get("terminal_tall_reward_weight", 0.0))) > 0.0
    )
    # Validate: projector_ckpt and edge_registry are required
    if not d.get("projector_ckpt") or not d.get("edge_registry"):
        p.error("--projector_ckpt and --edge_registry are required")
    cfg = TrainConfig(**d)
    try:
        _validate_legacy_actor_only_losses(cfg)
    except ValueError as exc:
        p.error(str(exc))
    train(cfg)


if __name__ == "__main__":
    main()
