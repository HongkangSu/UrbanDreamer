"""
train_marl.py — Fully Offline IPPO for Urban Routing (no SUMO during training)

Episode structure (100% offline, flow-only):
  1. Sample initial state from pre-collected flow dataset:
       z_0      = wm.encode_observation(flow_0)   (first flow snapshot of an episode)
       hdv_frac = uniform(K) or dataset-provided HDV fraction (if available)
  2. Imagined rollout for episode_len steps:
       policy(z_t.flatten()) -> av_frac ∈ Δ^{K-1}  (Gaussian-in-logit-space → softmax)
       compute_joint_frac(av_frac, hdv_frac)  -> (K,) float
       wm.predict_next_state(buffer, joint_frac) -> z_{t+1}
       reward from BPR travel-time model or LatentRewardPredictor
  3. PPO update after each episode

No SUMO, no traci, no connector needed at training time.
The world model IS the environment.

Data required (produced by data_collector.py):
  <data_dir>/flow_round1.pt            — episode, av_planned_edge_flow/av_edge_action, hdv_edge_action, reward
  <data_dir>/flows/<task>/shard_*.pt   — flow snapshots (shard_size, E, 8) float
"""

import os
import sys
import glob
import copy
import math
import json
import logging
import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from .urban_world_model import UrbanWorldModel
    from .traffic_reward_model import TrafficMetricsRewardModel
    from .rollout_buffer import RolloutBuffer
    from .dreamer_actor_critic import (
        CentralizedLatentValueCritic,
        VehicleRouteActor,
        freeze_module,
        lambda_return,
    )
    from .dreamer_rollout import (
        aggregate_route_probs_to_action_edge,
        build_history_packed,
    )
    from .critics import (
        LatentGlobalCritic,
        append_global_summary_to_obs_np,
        latent_global_summary,
        latent_global_summary_dim,
    )
    from .experiment_settings import add_settings_arg, apply_stage_defaults, parse_settings_arg
    from .flow_projector import PathTopology, split_od_agent_counts
    from .gawron import gawron_step_from_route_tt, route_tt_from_joint_frac
    from .departure_schedule import (
        build_departure_schedule, rebuild_active_set,
        estimate_vehicle_position, compute_avg_edge_time, compute_avg_route_steps,
    )
    from .routerl_shared import (
        load_od_data,
        load_agents_dataframe,
        compute_agent_counts,
        ensure_agents_csv_in_records,
        generate_dijkstra_routes,
        create_routerl_environment,
    )
    from .routerl_hdv_blackbox import (
        build_human_agent_lookup,
        choose_routerl_hdv_action,
        apply_routerl_hdv_learning as apply_routerl_hdv_learning_shared,
    )
except ImportError:
    from urban_world_model import UrbanWorldModel
    from traffic_reward_model import TrafficMetricsRewardModel
    from rollout_buffer import RolloutBuffer
    from dreamer_actor_critic import (
        CentralizedLatentValueCritic,
        VehicleRouteActor,
        freeze_module,
        lambda_return,
    )
    from dreamer_rollout import (
        aggregate_route_probs_to_action_edge,
        build_history_packed,
    )
    from critics import (
        LatentGlobalCritic,
        append_global_summary_to_obs_np,
        latent_global_summary,
        latent_global_summary_dim,
    )
    from experiment_settings import add_settings_arg, apply_stage_defaults, parse_settings_arg
    from flow_projector import PathTopology, split_od_agent_counts
    from gawron import gawron_step_from_route_tt, route_tt_from_joint_frac
    from departure_schedule import (
        build_departure_schedule, rebuild_active_set,
        estimate_vehicle_position, compute_avg_edge_time, compute_avg_route_steps,
    )
    from routerl_shared import (
        load_od_data,
        load_agents_dataframe,
        compute_agent_counts,
        ensure_agents_csv_in_records,
        generate_dijkstra_routes,
        create_routerl_environment,
    )
    from routerl_hdv_blackbox import (
        build_human_agent_lookup,
        choose_routerl_hdv_action,
        apply_routerl_hdv_learning as apply_routerl_hdv_learning_shared,
    )

try:
    from routerl import TrafficEnvironment as _TrafficEnvironment
    _ROUTERL_AVAILABLE = True
except ImportError:
    _ROUTERL_AVAILABLE = False

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Latent-space reward predictor (trained by scripts/train_latent_reward.py)
# ══════════════════════════════════════════════════════════════════════════════

class LatentRewardPredictor(nn.Module):
    """
    小型 MLP，将 z.flatten() 映射到标准化 reward，再还原到原始尺度。
    由 train_latent_reward.py 离线训练，避免 decoder 平滑效应。
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
        return self.net(z_flat).squeeze(-1)  # (B,) or scalar

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device) -> "LatentRewardPredictor":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ckpt["latent_dim"], ckpt["hidden_dim"])
        model.load_state_dict(ckpt["model"])
        model.r_mean = float(ckpt["r_mean"])
        model.r_std  = float(ckpt["r_std"])
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        logger.info(
            f"LatentRewardPredictor loaded: latent_dim={ckpt['latent_dim']}, "
            f"val_loss={ckpt.get('val_loss', float('nan')):.4f}, "
            f"r_mean={ckpt['r_mean']:.2f}, r_std={ckpt['r_std']:.2f}"
        )
        return model

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> float:
        """z: (n_spatial, d_spatial) → scalar reward（原始尺度）"""
        z_flat = z.flatten().unsqueeze(0)  # (1, D)
        r_norm = self.forward(z_flat).item()
        return r_norm * self.r_std + self.r_mean


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
    av_fraction: float = 0.3    # notional AV fraction used to compute n_av / n_hdv
    n_av: int = 98              # int(491 * 0.2) — matches agents.csv + ratio_machines=0.2
    n_hdv: int = 393            # 491 - 98

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
    vehicle_routes_csv: str = ""       # optional route artifacts CSV for od_suffix vehicle candidates
    vehicle_obs_include_global: bool = True
    vehicle_obs_route_stats: str = "mean"
    vehicle_obs_include_route_costs: bool = False
    vehicle_obs_include_fleet_pressure: bool = False
    vehicle_actor_architecture: str = "mlp"
    vehicle_actor_route_cost_logit_bias: float = 0.0
    vehicle_reward_scale: float = 1.0
    critic_variant: str = "baseline"  # baseline | latent_global
    critic_global_summary_variant: str = "mean"  # mean | mean_std
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

    # ── BPR reward (recommended) ─────────────────────────────────────────────
    # When True, reward = -BPR_travel_time(av_frac, joint_frac) computed
    # directly from the action at each step. This is guaranteed to be
    # action-dependent and optimises AV travel time explicitly.
    # When False, falls back to LatentRewardPredictor or image-based model.
    use_bpr_reward: bool = True
    reward_mode: str = "difference"  # global | local | difference
    conservative_penalty_coef: float = 0.02  # penalize large AV/HDV intent divergence
    dense_bpr_reward_weight: float = 0.1  # scheme-B shaping weight for step-level BPR reward
    use_terminal_tall_reward: bool = False  # legacy terminal reward from WM t_all head
    terminal_tall_reward_weight: float = 0.0  # keep Dreamer reward as step-level seconds BPR by default

    # ── latent reward predictor（可选，替换基于图像的 reward）────────────────
    # 由 scripts/train_latent_reward.py 生成；若为空则退回到图像颜色分析
    latent_reward_ckpt: str = ""

    # ── checkpoint ───────────────────────────────────────────────────────────
    checkpoint_dir: str = "logs/marl"
    checkpoint_every: int = 100
    resume_ckpt: str = ""
    resume_reset_actor_optimizer: bool = False

    # ── periodic SUMO evaluation (requires routerl + SUMO) ───────────────────
    # 0 = disabled.  Set to e.g. 1000 to evaluate every 1000 offline episodes.
    eval_every: int = 0
    eval_eps: int = 10               # SUMO test episodes per eval call
    eval_human_eps: int = 300        # human stabilisation episodes
    eval_ratio_machines: float = 0.4 # AV fraction for repo-local ingolstadt_custom
    eval_network: str = "ingolstadt_custom"
    eval_network_folder: str = "networks/ingolstadt_custom"
    eval_results_dir: str = "results/marl_eval"
    eval_seed: int = 42
    eval_disable_human_learning_after_mutation: bool = True
    eval_use_fixed_hdv_fracs: bool = True
    eval_stochastic_policy: bool = False
    eval_hdv_state_in: str = ""
    eval_hdv_state_out: str = ""
    discrepancy_threshold: float = 0.25
    lr_decay_on_discrepancy: float = 0.7
    max_lr_decays: int = 4

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
    dreamer_actor_objective: str = "direct_return"  # direct_return | critic_q | direct_plus_q
    dreamer_actor_q_weight: float = 1.0
    dreamer_action_mode: str = "expected"  # expected | straight_through_sample | straight_through_argmax
    dreamer_action_temperature: float = 1.0
    dreamer_all_reward_weight: float = 1.0
    dreamer_remaining_time_progress_weight: float = 0.0
    dreamer_current_edge_reward_weight: float = 0.0
    dreamer_cav_reward_weight: float = 0.0
    dreamer_route_cost_reward_weight: float = 0.0
    dreamer_pred_t_cav_proxy_reward_weight: float = 0.0
    dreamer_route_cost_reward_normalize_regret: bool = False
    dreamer_route_cost_reward_temperature: float = 60.0
    dreamer_route_cost_reward_planned_av_pressure_weight: float = 0.0
    dreamer_route_cost_reward_planned_av_pressure_mode: str = "overlap"
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
    dreamer_grpo_policy_objective: str = "combined"  # combined | legacy_separate
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


# ══════════════════════════════════════════════════════════════════════════════
# Periodic SUMO evaluation harness
# ══════════════════════════════════════════════════════════════════════════════

class SUMOEvalHarness:
    """
    Manages a long-lived RouteRL TrafficEnvironment for periodic SUMO evaluation.

    Lifecycle:
      __init__  — reads OD/agents, starts SUMO, runs eval_human_eps human-
                  stabilisation episodes, calls mutation() once.  Done once.
      evaluate() — runs eval_eps test episodes with the supplied policy, reads
                   RouteRL episode CSVs, returns real travel times.
      close()   — stops SUMO.

    HDV learning is frozen after mutation by default. In `routerl_blackbox` mode,
    RouteRL human action selection uses the same shared black-box functions as
    data collection.

    In mean field mode (world_model + route_registry provided), encodes actual
    SUMO state via traci and computes per-OD av_fracs.
    In single-agent mode, uses deterministic mean action at z = 0.
    """

    def __init__(
        self,
        n_routes:            int,
        obs_dim:             int,
        device:              torch.device,
        network:             str,
        network_folder:      str,
        eval_results_dir:    str,
        eval_human_eps:      int,
        eval_eps:            int,
        ratio_machines:      float = 0.2,
        seed:                int   = 42,
        gawron_mode:         str   = "replicator",
        world_model:         Optional["UrbanWorldModel"] = None,
        route_registry:      Optional["PathTopology"] = None,
        fixed_hdv_fracs:     Optional[np.ndarray] = None,
        disable_human_learning_after_mutation: bool = True,
        stochastic_policy: bool = False,
        hdv_state_in_path: str = "",
        hdv_state_out_path: str = "",
    ):
        if not _ROUTERL_AVAILABLE:
            raise RuntimeError("routerl is not installed — cannot use SUMOEvalHarness")

        self.n_routes  = n_routes
        self.obs_dim   = obs_dim
        self.device    = device
        self.eval_eps  = eval_eps
        self._gawron_mode = str(gawron_mode).lower()
        self._use_routerl_hdv_blackbox = (self._gawron_mode == "routerl_blackbox")
        self._disable_human_learning_after_mutation = bool(disable_human_learning_after_mutation)
        self._eval_stochastic_policy = bool(stochastic_policy)
        self._hdv_state_in_path = self._resolve_optional_path(hdv_state_in_path)
        self._hdv_state_out_path = self._resolve_optional_path(hdv_state_out_path)
        self._loaded_hdv_state = self._load_hdv_learning_state(self._hdv_state_in_path)
        self._pre_mutation_hdv_agents: Dict[str, Dict[str, Any]] = {}
        self._wm       = world_model
        self._rr       = route_registry
        self._mean_field = (world_model is not None and route_registry is not None)
        self._rng_np = np.random.default_rng(seed + 2026)
        self._rng_py = random.Random(seed + 2026)
        self._use_intention_obs = False
        self._eval_hdv_fracs: Optional[np.ndarray] = None
        self._eval_prev_av_fracs: Optional[np.ndarray] = None
        self._fixed_hdv_fracs: Optional[np.ndarray] = None
        self._human_agents_by_id: Dict[str, Any] = {}
        self._human_episode_actions: Dict[str, int] = {}
        if self._mean_field:
            base_obs_dim = self.n_routes * self._wm.d_obs + self._wm.d_obs
            self._use_intention_obs = (self.obs_dim == base_obs_dim + 2 * self.n_routes)
            self._eval_hdv_fracs = np.ones((self._rr.n_od, self.n_routes), dtype=np.float32) / self.n_routes
            self._eval_prev_av_fracs = self._eval_hdv_fracs.copy()

        self._episodes_folder = os.path.join(eval_results_dir, "episodes")
        plots_folder          = os.path.join(eval_results_dir, "plots")
        os.makedirs(self._episodes_folder, exist_ok=True)
        os.makedirs(plots_folder,          exist_ok=True)

        od_data = load_od_data(network_folder, network)
        agents_df = load_agents_dataframe(network_folder)
        num_agents, num_machines, max_start_time = compute_agent_counts(agents_df, ratio_machines)
        ensure_agents_csv_in_records(network_folder, eval_results_dir)

        # ── build OD→agent map for mean field mode ──────────────────────────
        self._od_agent_map = {}  # agent_id (int) → od_idx
        if self._mean_field:
            n_dest = len(od_data["destinations"])
            for _, row in agents_df.iterrows():
                oi = int(row["origin"])
                di = int(row["destination"])
                self._od_agent_map[int(row["id"])] = oi * n_dest + di

        if self._use_routerl_hdv_blackbox and fixed_hdv_fracs is not None:
            logger.info(
                "[SUMOEval] gawron_mode=routerl_blackbox -> ignore fixed_hdv_fracs "
                "(human routes come from RouteRL black-box model)."
            )
        elif self._mean_field and fixed_hdv_fracs is not None:
            fixed_arr = np.asarray(fixed_hdv_fracs, dtype=np.float32)
            if fixed_arr.shape == (self._rr.n_od, self.n_routes):
                fixed_arr = np.clip(fixed_arr, 1e-6, None)
                fixed_arr = fixed_arr / fixed_arr.sum(axis=1, keepdims=True)
                self._fixed_hdv_fracs = fixed_arr
                self._eval_hdv_fracs = fixed_arr.copy()
                self._eval_prev_av_fracs = fixed_arr.copy()
            else:
                logger.warning(
                    "[SUMOEval] Ignore fixed_hdv_fracs due to shape mismatch: got %s, expected (%d, %d)",
                    tuple(fixed_arr.shape), self._rr.n_od, self.n_routes,
                )

        effective_human_eps = int(eval_human_eps)
        if self._fixed_hdv_fracs is not None:
            effective_human_eps = 0
            logger.info("[SUMOEval] Fixed HDV strategy provided -> skip human re-learning phase.")

        total_eps   = effective_human_eps + eval_eps
        phases      = [1, effective_human_eps, effective_human_eps, total_eps]
        phase_names = ["Human stab.", "Mutation", "AV eval", "End"]

        logger.info(
            f"[SUMOEval] Initialising TrafficEnvironment "
            f"({num_agents} agents, {num_machines} AVs = {ratio_machines*100:.0f}%) …"
        )

        # ── generate Dijkstra routes ────────────────────────────────────────
        routes_df = generate_dijkstra_routes(
            network_folder=network_folder,
            network_name=network,
            origins=od_data["origins"],
            destinations=od_data["destinations"],
            n_routes=n_routes,
            records_folder=eval_results_dir,
        )
        n_distinct = (
            len(set(frozenset(p.split()) for p in routes_df["path"].values))
            if len(routes_df) > 0 else 0
        )
        logger.info(
            "[SUMOEval] Dijkstra routes: %d total, %d distinct -> %s",
            len(routes_df), n_distinct, eval_results_dir,
        )

        self._env = create_routerl_environment(
            _TrafficEnvironment,
            seed=seed,
            create_agents=False,
            create_paths=False,
            network_name=network,
            network_folder=network_folder,
            simulation_timesteps=max_start_time,
            records_folder=eval_results_dir,
            plots_folder=plots_folder,
            origins=od_data["origins"],
            destinations=od_data["destinations"],
            number_of_paths=n_routes,
            new_machines_after_mutation=num_machines,
            machine_behavior="selfish",
            machine_observation_type="previous_agents_plus_start_time",
            save_every=1,
            phases=phases,
            phase_names=phase_names,
            smooth_by=5,
            plot_choices="none",
            sumo_type="sumo",
            save_detectors_info=False,
        )

        self._env.start()
        self._env.reset()
        if self._loaded_hdv_state is not None:
            self._restore_hdv_learning_state(self._loaded_hdv_state, label="pre-phase1")
        self._pre_mutation_hdv_agents = self._snapshot_hdv_learning_state()

        # ── human stabilisation (runs once) ──────────────────────────────────
        if effective_human_eps > 0:
            logger.info(f"[SUMOEval] Human stabilisation: {effective_human_eps} episodes …")
            human_t_all_series: List[float] = []
            missing_t_all_count = 0
            log_every = 10 if effective_human_eps >= 10 else 1
            for i in range(effective_human_eps):
                ep_snapshot = self._snapshot_episodes()
                self._env.step()
                ep_csvs = self._new_csv_paths_since(ep_snapshot, fallback_latest=1)
                ep_metrics = self._metrics_from_csvs(ep_csvs) if ep_csvs else None
                if ep_metrics is not None and np.isfinite(ep_metrics["t_all"]):
                    human_t_all_series.append(float(ep_metrics["t_all"]))
                else:
                    missing_t_all_count += 1

                should_log = ((i + 1) % log_every == 0) or ((i + 1) == effective_human_eps)
                if should_log:
                    if human_t_all_series:
                        window = min(50, len(human_t_all_series))
                        logger.info(
                            "[SUMOEval]   … %d/%d episodes done | t_all(last)=%.2f min t_all(avg%d)=%.2f min",
                            i + 1,
                            effective_human_eps,
                            human_t_all_series[-1],
                            window,
                            float(np.mean(human_t_all_series[-window:])),
                        )
                    else:
                        logger.info(
                            "[SUMOEval]   … %d/%d episodes done | t_all=N/A (no episode CSV metric yet)",
                            i + 1,
                            effective_human_eps,
                        )
            if human_t_all_series:
                logger.info(
                    "[SUMOEval] Human stabilisation summary | t_all(mean)=%.2f min t_all(last)=%.2f min (n=%d, missing=%d)",
                    float(np.mean(human_t_all_series)),
                    human_t_all_series[-1],
                    len(human_t_all_series),
                    missing_t_all_count,
                )
            else:
                logger.info(
                    "[SUMOEval] Human stabilisation summary | t_all unavailable (n=0, missing=%d)",
                    missing_t_all_count,
                )
        else:
            logger.info("[SUMOEval] Human stabilisation skipped (fixed HDV strategy).")

        # ── mutation (runs once) ──────────────────────────────────────────────
        logger.info("[SUMOEval] Mutation → introducing AVs …")
        self._env.mutation(
            disable_human_learning=self._disable_human_learning_after_mutation,
            mutation_start_percentile=-1,
        )
        if self._loaded_hdv_state is not None:
            self._restore_hdv_learning_state(self._loaded_hdv_state, label="post-mutation")
        if self._use_routerl_hdv_blackbox:
            logger.info("[SUMOEval] HDV routing backend: RouteRL black-box human model (shared)")
        else:
            logger.info(
                "[SUMOEval] HDV routing backend: RouteRL native (%s after mutation)",
                "freeze" if self._disable_human_learning_after_mutation else "keep learning",
            )
        self._apply_fixed_hdv_defaults()
        self._agent_lookup = {str(a.id): a for a in self._env.machine_agents}
        logger.info(
            f"[SUMOEval] Ready: {len(self._env.machine_agents)} AVs, "
            f"{len(self._env.human_agents)} HDVs"
        )

    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(self, policy: "SharedPolicy") -> Optional[Dict[str, float]]:
        """
        Run eval_eps SUMO test episodes with the current policy.

        In mean field mode: encodes actual SUMO state via traci, computes
        per-OD observations, and maps each AV to its OD pair.
        In single-agent mode: uses deterministic mean action at z = 0.

        Returns dict {t_CAV, t_HDV, t_all} in minutes, or None if no CSVs found.
        """
        existing = self._snapshot_episodes()
        rng_np   = np.random.default_rng()

        policy.eval()
        with torch.no_grad():
            if self._mean_field:
                route_masks_t = self._rr.get_route_masks_tensor(str(self.device))
                for _ in range(self.eval_eps):
                    ep_snapshot = self._snapshot_episodes() if self._use_routerl_hdv_blackbox else {}
                    self._apply_fixed_hdv_defaults()
                    self._env.reset()
                    if self._eval_hdv_fracs is not None:
                        if self._fixed_hdv_fracs is not None:
                            self._eval_hdv_fracs = self._fixed_hdv_fracs.copy()
                        self._eval_prev_av_fracs = self._eval_hdv_fracs.copy()
                    # Encode SUMO state and compute per-OD av_fracs
                    av_fracs_np = self._encode_and_get_av_fracs(policy, route_masks_t)
                    for agent_id in self._env.agent_iter():
                        _, _, term, trunc, _ = self._env.last()
                        if term or trunc:
                            action = None
                        else:
                            # Map agent to its OD pair
                            try:
                                aid_int = int(str(agent_id).split("_")[-1])
                            except (ValueError, IndexError):
                                aid_int = -1
                            od_idx = self._od_agent_map.get(aid_int, 0)
                            action = int(rng_np.choice(self.n_routes, p=av_fracs_np[od_idx]))
                        self._env.step(action)
                    if self._use_routerl_hdv_blackbox:
                        self._apply_routerl_hdv_blackbox_learning(ep_snapshot)
            else:
                # Single-agent mode: one global av_frac
                av_frac_np = self._policy_mean_frac(policy)
                for _ in range(self.eval_eps):
                    ep_snapshot = self._snapshot_episodes() if self._use_routerl_hdv_blackbox else {}
                    self._apply_fixed_hdv_defaults()
                    self._env.reset()
                    for agent_id in self._env.agent_iter():
                        _, _, term, trunc, _ = self._env.last()
                        if term or trunc:
                            action = None
                        else:
                            action = int(rng_np.choice(self.n_routes, p=av_frac_np))
                        self._env.step(action)
                    if self._use_routerl_hdv_blackbox:
                        self._apply_routerl_hdv_blackbox_learning(ep_snapshot)

        changed_paths = self._new_csv_paths_since(existing, fallback_latest=self.eval_eps)
        return self._metrics_from_csvs(changed_paths)

    def set_fixed_hdv_fracs(self, hdv_fracs: Optional[np.ndarray]) -> None:
        """Set per-OD HDV route fractions for SUMO eval and disable HDV re-learning."""
        if self._use_routerl_hdv_blackbox:
            if hdv_fracs is not None:
                logger.debug("[SUMOEval] Ignore fixed_hdv_fracs in routerl_blackbox mode.")
            self._fixed_hdv_fracs = None
            return
        if not self._mean_field:
            return
        if hdv_fracs is None:
            self._fixed_hdv_fracs = None
            base = np.ones((self._rr.n_od, self.n_routes), dtype=np.float32) / self.n_routes
            self._eval_hdv_fracs = base
            self._eval_prev_av_fracs = base.copy()
            self._apply_fixed_hdv_defaults()
            return

        arr = np.asarray(hdv_fracs, dtype=np.float32)
        if arr.shape != (self._rr.n_od, self.n_routes):
            raise ValueError(
                f"fixed_hdv_fracs shape mismatch: got {arr.shape}, "
                f"expected {(self._rr.n_od, self.n_routes)}"
            )
        arr = np.clip(arr, 1e-6, None)
        arr = arr / arr.sum(axis=1, keepdims=True)
        self._fixed_hdv_fracs = arr
        self._eval_hdv_fracs = arr.copy()
        self._eval_prev_av_fracs = arr.copy()
        self._apply_fixed_hdv_defaults()

    def _apply_fixed_hdv_defaults(self) -> None:
        """Apply fixed HDV actions for the next episode (one sample per human agent)."""
        if not hasattr(self, "_env"):
            return
        humans = getattr(self._env, "human_agents", [])
        if not humans:
            return

        if self._use_routerl_hdv_blackbox:
            self._human_agents_by_id = build_human_agent_lookup(humans)
            self._human_episode_actions = {}
            sampled_od_actions: list[tuple[int, int]] = []
            for human in humans:
                hid = str(getattr(human, "id", ""))
                action = choose_routerl_hdv_action(
                    human_lookup=self._human_agents_by_id,
                    agent_id=hid,
                    n_routes=self.n_routes,
                    rng=self._rng_py,
                    episode_actions=self._human_episode_actions,
                )
                human.default_action = action
                if self._mean_field and self._eval_hdv_fracs is not None:
                    try:
                        aid_int = int(float(hid))
                    except Exception:
                        aid_int = -1
                    od_idx = self._od_agent_map.get(aid_int)
                    if od_idx is not None and 0 <= od_idx < self._eval_hdv_fracs.shape[0]:
                        sampled_od_actions.append((od_idx, int(action)))
            if self._mean_field and self._eval_hdv_fracs is not None:
                counts = np.full_like(self._eval_hdv_fracs, 1e-6, dtype=np.float32)
                for od_idx, action in sampled_od_actions:
                    counts[od_idx, action] += 1.0
                counts = counts / counts.sum(axis=1, keepdims=True)
                self._eval_hdv_fracs = counts.astype(np.float32)
            return

        if self._fixed_hdv_fracs is None:
            for human in humans:
                human.default_action = None
            return

        uniform = np.full(self.n_routes, 1.0 / self.n_routes, dtype=np.float32)
        for human in humans:
            try:
                aid_int = int(human.id)
            except Exception:
                aid_int = -1
            od_idx = self._od_agent_map.get(aid_int, None)
            if od_idx is None or od_idx < 0 or od_idx >= self._fixed_hdv_fracs.shape[0]:
                probs = uniform
            else:
                probs = self._fixed_hdv_fracs[od_idx]
            human.default_action = int(self._rng_np.choice(self.n_routes, p=probs))

    def _collect_human_travel_times_from_csvs(self, csv_paths: set[str]) -> Dict[str, float]:
        """Collect per-human travel time in minutes from RouteRL episode CSV files."""
        import pandas as pd

        out: Dict[str, float] = {}
        for path in csv_paths:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty or "travel_time" not in df.columns or "id" not in df.columns:
                continue
            work = df.copy()
            work["travel_time"] = pd.to_numeric(work["travel_time"], errors="coerce")
            work = work.dropna(subset=["travel_time"])
            if work.empty:
                continue
            kind_col = next((c for c in ("vType", "kind", "type") if c in work.columns), None)
            if kind_col:
                kinds = work[kind_col].astype(str).str.lower()
                work = work[kinds.isin({"human", "hdv"})]
            if work.empty:
                continue
            for _, row in work.iterrows():
                out[str(row["id"])] = float(row["travel_time"])
        return out

    def _apply_routerl_hdv_blackbox_learning(self, snapshot_before_ep: Dict[str, int]) -> int:
        """Apply shared RouteRL black-box human learning for one finished eval episode."""
        if not self._use_routerl_hdv_blackbox or not self._human_episode_actions:
            return 0
        self._human_agents_by_id = build_human_agent_lookup(
            getattr(self._env, "human_agents", [])
        )
        changed_paths = self._new_csv_paths_since(snapshot_before_ep, fallback_latest=1)
        if not changed_paths:
            return 0

        human_tts_min = self._collect_human_travel_times_from_csvs(changed_paths)
        if not human_tts_min:
            return 0
        return apply_routerl_hdv_learning_shared(
            human_lookup=self._human_agents_by_id,
            episode_actions=self._human_episode_actions,
            travel_times=human_tts_min,
            travel_time_unit="minutes",
        )

    # ──────────────────────────────────────────────────────────────────────────

    def _encode_and_get_av_fracs(
        self, policy: "MeanFieldRoutePolicy", route_masks_t: torch.Tensor
    ) -> np.ndarray:
        """Encode current SUMO state via traci → per-OD av_fracs (N_od, K)."""
        try:
            import traci
            edges_state = {}
            for edge_id in traci.edge.getIDList():
                if edge_id.startswith(":"):
                    continue
                try:
                    edges_state[edge_id] = {
                        "occupancy":      traci.edge.getLastStepOccupancy(edge_id),
                        "mean_speed":     traci.edge.getLastStepMeanSpeed(edge_id),
                        "vehicle_number": traci.edge.getLastStepVehicleNumber(edge_id),
                        "halting_number": traci.edge.getLastStepHaltingNumber(edge_id),
                    }
                except Exception:
                    pass
            if edges_state:
                z = self._wm.encode_observation({"edges": edges_state})
                obs_mf = self._wm.local_policy_obs(
                    z,
                    route_masks_t,
                    prev_av_fracs=self._eval_prev_av_fracs if self._use_intention_obs else None,
                    hdv_fracs=self._eval_hdv_fracs if self._use_intention_obs else None,
                )
                av_np = policy.get_av_frac(
                    obs_mf,
                    deterministic=not self._eval_stochastic_policy,
                ).cpu().numpy()
                if self._eval_prev_av_fracs is not None:
                    self._eval_prev_av_fracs = av_np.copy()
                return av_np
        except Exception:
            pass
        # Fallback: uniform distribution for all OD pairs
        n_od = route_masks_t.shape[0]
        fallback = np.full((n_od, self.n_routes), 1.0 / self.n_routes, dtype=np.float32)
        if self._eval_prev_av_fracs is not None:
            self._eval_prev_av_fracs = fallback.copy()
        return fallback

    # ──────────────────────────────────────────────────────────────────────────

    def _policy_mean_frac(self, policy: "SharedPolicy") -> np.ndarray:
        """Return softmax(μ_logits) at z = 0 — the policy's "default" distribution."""
        with torch.no_grad():
            obs = torch.zeros(1, self.obs_dim, device=self.device)
            av_frac = policy.get_av_frac(
                obs,
                deterministic=not self._eval_stochastic_policy,
            )   # (1, K)
        return av_frac[0].cpu().numpy()                  # (K,)

    def _resolve_optional_path(self, path_str: str) -> Optional[Path]:
        if not path_str:
            return None
        p = Path(path_str)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p.resolve()

    def _load_hdv_learning_state(self, state_path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if state_path is None:
            return None
        if not state_path.exists():
            logger.info("[SUMOEval] HDV state input not found, skip restore: %s", state_path)
            return None
        try:
            with state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            if not isinstance(state, dict):
                raise ValueError("top-level JSON is not an object")
            agents = state.get("agents", {})
            if not isinstance(agents, dict):
                raise ValueError("`agents` is not an object")
            logger.info("[SUMOEval] HDV state loaded: %d agents <- %s", len(agents), state_path)
            return state
        except Exception as exc:
            logger.warning("[SUMOEval] Failed to load HDV state %s: %s", state_path, exc)
            return None

    @staticmethod
    def _numeric_to_list(value: Any) -> Optional[list]:
        try:
            arr = np.asarray(value, dtype=np.float64)
            return arr.tolist()
        except Exception:
            return None

    def _snapshot_hdv_learning_state(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for human in getattr(self._env, "human_agents", []):
            model = getattr(human, "model", None)
            if model is None:
                continue
            aid = str(getattr(human, "id", ""))
            if not aid:
                continue
            entry: Dict[str, Any] = {"model_class": model.__class__.__name__}
            if hasattr(model, "cost"):
                cost_list = self._numeric_to_list(getattr(model, "cost"))
                if cost_list is not None:
                    entry["cost"] = cost_list
            if hasattr(model, "memory"):
                mem_list = self._numeric_to_list(getattr(model, "memory"))
                if mem_list is not None:
                    entry["memory"] = mem_list
            if hasattr(model, "beta"):
                try:
                    entry["beta"] = float(getattr(model, "beta"))
                except Exception:
                    pass
            out[aid] = entry
        return out

    @staticmethod
    def _merge_hdv_learning_state(
        base_state: Optional[Dict[str, Any]],
        update_agents: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged_agents: Dict[str, Dict[str, Any]] = {}
        if isinstance(base_state, dict):
            base_agents = base_state.get("agents", {})
            if isinstance(base_agents, dict):
                merged_agents.update(base_agents)
        merged_agents.update(update_agents)
        return {"version": 1, "agents": merged_agents}

    def _restore_hdv_learning_state(self, state: Optional[Dict[str, Any]], *, label: str) -> None:
        if not state:
            return
        agents_state = state.get("agents", {})
        if not isinstance(agents_state, dict):
            return

        humans = list(getattr(self._env, "human_agents", []))
        restored = 0
        missing = 0
        mismatched = 0
        no_model = 0

        for human in humans:
            aid = str(getattr(human, "id", ""))
            entry = agents_state.get(aid)
            if entry is None:
                missing += 1
                continue
            model = getattr(human, "model", None)
            if model is None:
                no_model += 1
                continue

            applied = False
            if "cost" in entry and hasattr(model, "cost"):
                try:
                    cur = np.asarray(getattr(model, "cost"))
                    new = np.asarray(entry["cost"], dtype=cur.dtype)
                    if cur.shape == new.shape:
                        setattr(model, "cost", new.copy())
                        applied = True
                    else:
                        mismatched += 1
                except Exception:
                    mismatched += 1
            if "memory" in entry and hasattr(model, "memory"):
                try:
                    cur_mem = np.asarray(getattr(model, "memory"))
                    new_mem = np.asarray(entry["memory"], dtype=cur_mem.dtype)
                    if cur_mem.shape == new_mem.shape:
                        setattr(model, "memory", new_mem.copy())
                        applied = True
                    else:
                        mismatched += 1
                except Exception:
                    mismatched += 1
            if "beta" in entry and hasattr(model, "beta"):
                try:
                    setattr(model, "beta", float(entry["beta"]))
                    applied = True
                except Exception:
                    pass

            if applied:
                restored += 1

        logger.info(
            "[SUMOEval] HDV state/%s restored=%d/%d missing=%d mismatched=%d no_model=%d",
            label,
            restored,
            len(humans),
            missing,
            mismatched,
            no_model,
        )

    def _save_hdv_learning_state(self, state: Optional[Dict[str, Any]]) -> None:
        if self._hdv_state_out_path is None or not state:
            return
        agents = state.get("agents", {})
        if not isinstance(agents, dict):
            return
        try:
            self._hdv_state_out_path.parent.mkdir(parents=True, exist_ok=True)
            with self._hdv_state_out_path.open("w", encoding="utf-8") as f:
                json.dump(state, f)
            logger.info(
                "[SUMOEval] HDV state saved: %d agents -> %s",
                len(agents),
                self._hdv_state_out_path,
            )
        except Exception as exc:
            logger.warning("[SUMOEval] Failed to save HDV state %s: %s", self._hdv_state_out_path, exc)

    def _snapshot_episodes(self) -> Dict[str, int]:
        """Return ep*.csv absolute path -> mtime_ns map."""
        if not os.path.isdir(self._episodes_folder):
            return {}
        out: Dict[str, int] = {}
        for f in os.listdir(self._episodes_folder):
            if not (f.startswith("ep") and f.endswith(".csv")):
                continue
            p = os.path.join(self._episodes_folder, f)
            try:
                out[p] = os.stat(p).st_mtime_ns
            except OSError:
                continue
        return out

    def _new_csv_paths_since(self, snapshot_before: Dict[str, int], fallback_latest: int = 0) -> set[str]:
        """
        Return csv paths changed since `snapshot_before`.

        If RouteRL overwrites files in-place and no mtime changed path is found,
        optionally fall back to the latest `fallback_latest` files by mtime.
        """
        latest = self._snapshot_episodes()
        changed_paths = {
            p for p, m in latest.items()
            if (p not in snapshot_before) or (m > snapshot_before[p])
        }
        if (not changed_paths) and int(fallback_latest) > 0:
            ordered = sorted(latest.items(), key=lambda kv: kv[1], reverse=True)
            changed_paths = {p for p, _ in ordered[: int(fallback_latest)]}
        return changed_paths

    def _metrics_from_csvs(self, csv_paths: set) -> Optional[Dict[str, float]]:
        """Read travel times from a set of RouteRL episode CSVs."""
        import pandas as pd

        tt_av, tt_hdv, tt_all = [], [], []
        for path in csv_paths:
            try:
                df = pd.read_csv(path)
                if df.empty or "travel_time" not in df.columns:
                    continue
                df["travel_time"] = pd.to_numeric(df["travel_time"], errors="coerce")
                df = df.dropna(subset=["travel_time"])

                kind_col = next(
                    (c for c in ("vType", "kind", "type") if c in df.columns), None
                )
                if kind_col:
                    kinds    = df[kind_col].astype(str).str.lower()
                    av_mask  = kinds.isin({"av", "machine", "cav"})
                    hdv_mask = kinds.isin({"human", "hdv"})
                    tt_av.extend(df.loc[av_mask,  "travel_time"].tolist())
                    tt_hdv.extend(df.loc[hdv_mask, "travel_time"].tolist())
                tt_all.extend(df["travel_time"].tolist())
            except Exception:
                continue

        if not tt_all:
            return None

        def _min(lst):
            # RouteRL already stores travel_time in minutes — no conversion needed
            return float(np.mean(lst)) if lst else float("nan")

        return {"t_CAV": _min(tt_av), "t_HDV": _min(tt_hdv), "t_all": _min(tt_all)}

    def close(self):
        post_rollout_hdv_agents = self._snapshot_hdv_learning_state()
        merged_hdv_state = self._merge_hdv_learning_state(
            self._loaded_hdv_state,
            self._pre_mutation_hdv_agents,
        )
        merged_hdv_state = self._merge_hdv_learning_state(
            merged_hdv_state,
            post_rollout_hdv_agents,
        )
        self._save_hdv_learning_state(merged_hdv_state)
        try:
            self._env.stop_simulation()
        except Exception:
            pass

    def get_current_hdv_fracs(self) -> Optional[np.ndarray]:
        """
        Return current HDV route fractions used by SUMO eval harness.

        In black-box mode, this tracks the latest sampled/observed OD-level
        HDV strategy. In fixed-fraction mode, this returns the fixed fractions.
        """
        if not self._mean_field:
            return None
        if self._eval_hdv_fracs is None:
            return None
        arr = np.asarray(self._eval_hdv_fracs, dtype=np.float32).copy()
        arr = np.clip(arr, 1e-6, None)
        arr = arr / arr.sum(axis=1, keepdims=True)
        return arr

    @property
    def uses_routerl_hdv_blackbox(self) -> bool:
        return bool(self._use_routerl_hdv_blackbox)


# ══════════════════════════════════════════════════════════════════════════════
# Policy network
# ══════════════════════════════════════════════════════════════════════════════

class SharedPolicy(nn.Module):
    """
    单智能体连续动作 actor-critic，用于车队路线分配。

    obs    = z_t.flatten()          (obs_dim,)
    action = av_frac ∈ Δ^{K-1}     通过 logit 空间 Gaussian → softmax 产生

    网络输出 μ_logits ∈ R^K（logit 均值）；
    log_std 为可学习参数；
    采样时在 logit 空间加 Gaussian 噪声后 softmax，保证输出为合法概率分布。
    log_prob 在 logit 空间计算（Gaussian），PPO ratio 仍然正确（Jacobian 在 ratio 中抵消）。
    """

    def __init__(
        self,
        obs_dim: int,
        n_routes: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
        critic_obs_dim: Optional[int] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_routes = n_routes
        self.critic_obs_dim = int(critic_obs_dim or obs_dim)

        layers: List[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)

        self.policy_head = nn.Linear(hidden_dim, n_routes)   # 输出 μ_logits
        self.log_std     = nn.Parameter(torch.zeros(n_routes))  # 可学习 log σ
        self.critic = None
        if self.critic_obs_dim != int(obs_dim):
            self.critic = LatentGlobalCritic(
                input_dim=self.critic_obs_dim,
                hidden_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
            )
            self.value_head = None
        else:
            self.value_head  = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        if self.critic is None:
            nn.init.orthogonal_(self.value_head.weight,  gain=1.0)

    def _value_forward(
        self,
        obs: torch.Tensor,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.critic is None:
            h = self.trunk(obs)
            return self.value_head(h)
        critic_in = critic_obs if critic_obs is not None else obs
        return self.critic(critic_in)

    def forward(
        self,
        obs: torch.Tensor,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.policy_head(h), self._value_forward(obs, critic_obs=critic_obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        raw_action: Optional[torch.Tensor] = None,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            av_frac   : (B, K)  softmax 后的路线分布（传给 world model）
            raw       : (B, K)  logit 空间的采样值（存入 buffer 供 PPO recompute）
            log_prob  : (B,)    Gaussian log_prob，在 PPO ratio 中使用
            entropy   : scalar  策略熵（标量）
            value     : (B, 1)  状态价值
        """
        mu_logits, value = self.forward(obs, critic_obs=critic_obs)
        std  = self.log_std.clamp(-3.0, 0.5).exp()  # std ∈ [0.05, 1.65]，防止熵无限增大
        dist = Normal(mu_logits, std)

        if raw_action is None:
            raw = dist.rsample()          # (B, K)，可微采样
        else:
            raw = raw_action

        log_prob = dist.log_prob(raw).sum(-1)        # (B,)，对 K 维求和
        entropy  = dist.entropy().sum(-1).mean()     # scalar
        av_frac  = torch.softmax(raw, dim=-1)        # (B, K)

        return av_frac, raw, log_prob, entropy, value


# ══════════════════════════════════════════════════════════════════════════════
# Mean-field multi-agent policy
# ══════════════════════════════════════════════════════════════════════════════

class MeanFieldRoutePolicy(nn.Module):
    """
    Shared-weight actor-critic for mean field multi-agent routing.
    Input per agent: (K*4 + 4) — K route features + global summary.
    Output: μ_logits (K,), value (1,). Same Gaussian→softmax as SharedPolicy.
    """

    def __init__(
        self,
        obs_dim: int,
        n_routes: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        critic_obs_dim: Optional[int] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_routes = n_routes
        self.critic_obs_dim = int(critic_obs_dim or obs_dim)

        layers: List[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)

        self.policy_head = nn.Linear(hidden_dim, n_routes)
        self.log_std     = nn.Parameter(torch.zeros(n_routes))
        self.critic = None
        if self.critic_obs_dim != int(obs_dim):
            self.critic = LatentGlobalCritic(
                input_dim=self.critic_obs_dim,
                hidden_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
            )
            self.value_head = None
        else:
            self.value_head  = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        if self.critic is None:
            nn.init.orthogonal_(self.value_head.weight,  gain=1.0)

    def _value_forward(
        self,
        obs: torch.Tensor,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.critic is None:
            h = self.trunk(obs)
            return self.value_head(h)
        critic_in = critic_obs if critic_obs is not None else obs
        return self.critic(critic_in)

    def forward(
        self,
        obs: torch.Tensor,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.policy_head(h), self._value_forward(obs, critic_obs=critic_obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        raw_action: Optional[torch.Tensor] = None,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, obs_dim) where B can be N_od or minibatch
            raw_action: optional (B, K) for recomputing log_prob in PPO

        Returns:
            av_fracs  : (B, K) softmax route distributions
            raw       : (B, K) logit-space samples
            log_prob  : (B,) Gaussian log_prob
            entropy   : scalar
            values    : (B, 1)
        """
        mu_logits, value = self.forward(obs, critic_obs=critic_obs)
        std  = self.log_std.clamp(-3.0, 0.5).exp()
        dist = Normal(mu_logits, std)

        if raw_action is None:
            raw = dist.rsample()
        else:
            raw = raw_action

        log_prob = dist.log_prob(raw).sum(-1)
        entropy  = dist.entropy().sum(-1).mean()
        av_fracs = torch.softmax(raw, dim=-1)

        return av_fracs, raw, log_prob, entropy, value


# ══════════════════════════════════════════════════════════════════════════════
# PPO rollout buffer
# ══════════════════════════════════════════════════════════════════════════════

class PPOBuffer:
    """单智能体 PPO buffer（每步一个集体决策）。"""

    def __init__(self, episode_len: int, obs_dim: int, n_routes: int, critic_obs_dim: Optional[int] = None):
        self.T, self.obs_dim, self.K = episode_len, obs_dim, n_routes
        self.critic_obs_dim = int(critic_obs_dim or obs_dim)
        self.ptr = 0

        self.obs        = torch.zeros(episode_len, obs_dim)
        self.critic_obs = torch.zeros(episode_len, self.critic_obs_dim)
        self.actions    = torch.zeros(episode_len, n_routes)   # raw logit 样本（float）
        self.rewards    = torch.zeros(episode_len)
        self.values     = torch.zeros(episode_len)
        self.logprobs   = torch.zeros(episode_len)
        self.advantages = torch.zeros(episode_len)
        self.returns    = torch.zeros(episode_len)
        self.value_targets = torch.zeros(episode_len)

    def store(
        self,
        obs: torch.Tensor,       # (obs_dim,)
        raw_action: torch.Tensor, # (K,)  logit 空间样本
        reward: float,
        value: torch.Tensor,     # scalar 或 (1,)
        logprob: torch.Tensor,   # scalar 或 ()
        critic_obs: Optional[torch.Tensor] = None,
    ):
        t = self.ptr
        self.obs[t]      = obs.cpu()
        self.critic_obs[t] = (critic_obs if critic_obs is not None else obs).cpu()
        self.actions[t]  = raw_action.cpu()
        self.rewards[t]  = float(reward)
        self.values[t]   = value.squeeze().cpu()
        self.logprobs[t] = logprob.squeeze().cpu()
        self.ptr += 1

    def compute_gae(
        self,
        last_value: torch.Tensor,   # scalar 或 (1,)
        gamma: float,
        gae_lambda: float,
        reward_mean: float = 0.0,
        reward_std:  float = 1.0,
    ):
        rewards_norm = (self.rewards - reward_mean) / max(reward_std, 1e-8)

        last_val = last_value.squeeze().cpu().item()
        gae = 0.0
        for t in reversed(range(self.T)):
            next_val = last_val if t == self.T - 1 else self.values[t + 1].item()
            delta = rewards_norm[t].item() + gamma * next_val - self.values[t].item()
            gae = delta + gamma * gae_lambda * gae
            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t]
        self.value_targets.copy_(self.returns)

    def get_flat_batches(self) -> Dict[str, torch.Tensor]:
        return {
            "obs":        self.obs,        # (T, obs_dim)
            "critic_obs": self.critic_obs, # (T, critic_obs_dim)
            "actions":    self.actions,    # (T, K)  raw logit 样本
            "logprobs":   self.logprobs,   # (T,)
            "values":     self.values,     # (T,)
            "advantages": self.advantages, # (T,)
            "returns":    self.returns,    # (T,)
            "value_targets": self.value_targets,  # (T,)
        }

    @staticmethod
    def merge(bufs: list) -> Dict[str, torch.Tensor]:
        keys = ["obs", "critic_obs", "actions", "logprobs", "values", "advantages", "returns", "value_targets"]
        batches = [b.get_flat_batches() for b in bufs]
        return {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}

    def compute_value_targets_multistep(
        self,
        last_value: torch.Tensor,
        gamma: float,
        horizon: int,
        reward_mean: float = 0.0,
        reward_std: float = 1.0,
    ) -> None:
        rewards_norm = (self.rewards - reward_mean) / max(reward_std, 1e-8)
        last_val = last_value.squeeze().cpu().item()
        H = max(1, int(horizon))
        for t in range(self.T):
            target = 0.0
            discount = 1.0
            for h in range(H):
                ti = t + h
                if ti >= self.T:
                    break
                target += discount * rewards_norm[ti].item()
                discount *= gamma
            boot_idx = t + H
            boot_val = last_val if boot_idx >= self.T else self.values[boot_idx].item()
            target += discount * boot_val
            self.value_targets[t] = target


class MultiAgentPPOBuffer:
    """Multi-agent PPO buffer: per-agent obs/actions/values/rewards."""

    def __init__(self, episode_len: int, n_od: int, obs_dim: int, n_routes: int, critic_obs_dim: Optional[int] = None):
        self.T = episode_len
        self.N_od = n_od
        self.obs_dim = obs_dim
        self.critic_obs_dim = int(critic_obs_dim or obs_dim)
        self.K = n_routes
        self.ptr = 0

        self.obs      = torch.zeros(episode_len, n_od, obs_dim)   # per-agent obs
        self.critic_obs = torch.zeros(episode_len, n_od, self.critic_obs_dim)
        self.actions  = torch.zeros(episode_len, n_od, n_routes)   # raw logit samples
        self.rewards  = torch.zeros(episode_len, n_od)             # per-agent reward
        self.values   = torch.zeros(episode_len, n_od)             # per-agent values
        self.logprobs = torch.zeros(episode_len, n_od)             # per-agent log probs
        self.advantages = torch.zeros(episode_len, n_od)
        self.returns    = torch.zeros(episode_len, n_od)
        self.value_targets = torch.zeros(episode_len, n_od)
        self.masks      = torch.zeros(episode_len, n_od, dtype=torch.bool)

    def store(
        self,
        obs: torch.Tensor,         # (N_od, obs_dim)
        raw_actions: torch.Tensor,  # (N_od, K)
        reward: Union[float, np.ndarray, torch.Tensor],  # scalar or (N_od,)
        values: torch.Tensor,       # (N_od, 1) or (N_od,)
        logprobs: torch.Tensor,     # (N_od,)
        active_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        critic_obs: Optional[torch.Tensor] = None,
    ):
        t = self.ptr
        self.obs[t]      = obs.cpu()
        self.critic_obs[t] = (critic_obs if critic_obs is not None else obs).cpu()
        self.actions[t]  = raw_actions.cpu()
        if torch.is_tensor(reward):
            rew_t = reward.detach().float().cpu().reshape(-1)
            if rew_t.numel() == 1:
                self.rewards[t] = rew_t.item()
            else:
                if rew_t.numel() != self.N_od:
                    raise ValueError(
                        f"reward shape mismatch: expected {self.N_od}, got {rew_t.numel()}"
                    )
                self.rewards[t] = rew_t
        elif isinstance(reward, np.ndarray):
            rew_np = reward.astype(np.float32).reshape(-1)
            if rew_np.size == 1:
                self.rewards[t] = float(rew_np.item())
            else:
                if rew_np.size != self.N_od:
                    raise ValueError(
                        f"reward shape mismatch: expected {self.N_od}, got {rew_np.size}"
                    )
                self.rewards[t] = torch.from_numpy(rew_np)
        else:
            self.rewards[t] = float(reward)
        self.values[t]   = values.squeeze(-1).cpu()
        self.logprobs[t] = logprobs.cpu()
        if active_mask is None:
            self.masks[t] = True
        elif torch.is_tensor(active_mask):
            self.masks[t] = active_mask.detach().cpu().reshape(-1).bool()
        else:
            self.masks[t] = torch.from_numpy(np.asarray(active_mask).reshape(-1).astype(np.bool_))
        self.ptr += 1

    def compute_gae(
        self,
        last_values: torch.Tensor,  # (N_od,) or (N_od, 1)
        gamma: float,
        gae_lambda: float,
        reward_mean: float = 0.0,
        reward_std:  float = 1.0,
    ):
        """GAE per agent: δ_i[t] = r_i[t] + γ*V_i[t+1] - V_i[t]."""
        rewards_norm = (self.rewards - reward_mean) / max(reward_std, 1e-8)
        last_vals = last_values.squeeze(-1).cpu().reshape(-1)
        if last_vals.numel() < self.N_od:
            padded_last = torch.zeros(self.N_od, dtype=last_vals.dtype)
            padded_last[: last_vals.numel()] = last_vals
            last_vals = padded_last
        elif last_vals.numel() > self.N_od:
            last_vals = last_vals[: self.N_od]

        gae = torch.zeros(self.N_od)
        for t in reversed(range(self.T)):
            cur_mask = self.masks[t].float()
            next_vals = last_vals if t == self.T - 1 else self.values[t + 1]
            next_mask = cur_mask if t == self.T - 1 else self.masks[t + 1].float()
            delta = rewards_norm[t] + gamma * next_vals * next_mask - self.values[t]
            gae = (delta + gamma * gae_lambda * gae * next_mask) * cur_mask
            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t] * cur_mask
        self.value_targets.copy_(self.returns)

    def get_flat_batches(self) -> Dict[str, torch.Tensor]:
        """Flatten (T, N_od) → (T*N_od,) for PPO minibatching."""
        T, N = self.T, self.N_od
        flat_mask = self.masks.reshape(T * N)
        return {
            "obs":        self.obs.reshape(T * N, self.obs_dim)[flat_mask],
            "critic_obs": self.critic_obs.reshape(T * N, self.critic_obs_dim)[flat_mask],
            "actions":    self.actions.reshape(T * N, self.K)[flat_mask],
            "logprobs":   self.logprobs.reshape(T * N)[flat_mask],
            "values":     self.values.reshape(T * N)[flat_mask],
            "advantages": self.advantages.reshape(T * N)[flat_mask],
            "returns":    self.returns.reshape(T * N)[flat_mask],
            "value_targets": self.value_targets.reshape(T * N)[flat_mask],
        }

    @staticmethod
    def merge(bufs: list) -> Dict[str, torch.Tensor]:
        keys = ["obs", "critic_obs", "actions", "logprobs", "values", "advantages", "returns", "value_targets"]
        batches = [b.get_flat_batches() for b in bufs]
        return {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}

    def compute_value_targets_multistep(
        self,
        last_values: torch.Tensor,
        gamma: float,
        horizon: int,
        reward_mean: float = 0.0,
        reward_std: float = 1.0,
    ) -> None:
        rewards_norm = (self.rewards - reward_mean) / max(reward_std, 1e-8)
        last_vals = last_values.squeeze(-1).cpu().reshape(-1)
        if last_vals.numel() < self.N_od:
            padded_last = torch.zeros(self.N_od, dtype=last_vals.dtype)
            padded_last[: last_vals.numel()] = last_vals
            last_vals = padded_last
        elif last_vals.numel() > self.N_od:
            last_vals = last_vals[: self.N_od]

        H = max(1, int(horizon))
        for t in range(self.T):
            cur_mask = self.masks[t].float()
            alive = cur_mask.clone()
            target = torch.zeros(self.N_od)
            discount = 1.0
            for h in range(H):
                ti = t + h
                if ti >= self.T:
                    break
                step_mask = alive * self.masks[ti].float()
                target += discount * rewards_norm[ti] * step_mask
                alive = step_mask
                discount *= gamma
                if float(alive.sum().item()) <= 0.0:
                    break
            boot_idx = t + H
            if boot_idx >= self.T:
                target += discount * last_vals * alive
            else:
                boot_mask = alive * self.masks[boot_idx].float()
                target += discount * self.values[boot_idx] * boot_mask
            self.value_targets[t] = target * cur_mask


# ══════════════════════════════════════════════════════════════════════════════
# Helper: joint route fraction
# ══════════════════════════════════════════════════════════════════════════════

def compute_joint_frac(
    av_frac: np.ndarray,   # (K,) float，AV 路线分布（softmax 输出）
    hdv_frac: np.ndarray,  # (K,) float
    n_av: int,
    n_hdv: int,
) -> np.ndarray:
    """Population-weighted joint route fraction passed to the world model."""
    total = n_av + n_hdv
    if total == 0:
        return np.zeros_like(av_frac)
    return (n_av * av_frac + n_hdv * hdv_frac) / total


# ══════════════════════════════════════════════════════════════════════════════
# Helper: Gawron HDV update via BPR travel-time estimate
# ══════════════════════════════════════════════════════════════════════════════

def gawron_update(
    hdv_frac:  np.ndarray,  # (K,) current HDV route distribution
    joint_frac: np.ndarray, # (K,) joint distribution (AV+HDV), for BPR load
    K:         int,         # number of routes
    lr:        float = 0.1, # Gawron learning rate
    bpr_alpha: float = 0.15,# BPR volume-delay alpha (standard value)
    bpr_beta:  float = 4.0, # BPR volume-delay beta  (standard value)
) -> np.ndarray:
    """
    One Gawron step: update hdv_frac based on BPR-estimated per-route travel times.

    Travel time model (BPR, t_free = 1 normalised):
        t_r = 1 + alpha * (K * joint_frac[r]) ^ beta

    K * joint_frac[r] is the volume-to-capacity ratio when each route has
    equal capacity = 1/K of total traffic.

    Gawron update (logit-space):
        logit_r += lr * (avg_tt - t_r)
        hdv_frac = softmax(logit)

    Routes with below-average travel time gain probability mass.
    """
    t_per_route = route_tt_from_joint_frac(joint_frac, K, bpr_alpha=bpr_alpha, bpr_beta=bpr_beta)
    new_frac, _ = gawron_step_from_route_tt(
        probs=hdv_frac,
        route_tt=t_per_route,
        mode="replicator",
        lr=lr,
    )
    return new_frac


# ══════════════════════════════════════════════════════════════════════════════
# Edge-level BPR reward (mean field)
# ══════════════════════════════════════════════════════════════════════════════

def edge_bpr_reward(
    av_fracs: np.ndarray,        # (N_od, K) AV route fractions per OD
    hdv_fracs: np.ndarray,       # (N_od, K) HDV route fractions per OD
    od_av_counts: np.ndarray,    # (N_od,) AV count per OD
    od_hdv_counts: np.ndarray,   # (N_od,) HDV count per OD
    route_masks: np.ndarray,     # (N_od, K, E) route edge masks
    lengths: np.ndarray,         # (E,) edge lengths in meters
    speed_limits: np.ndarray,    # (E,) speed limits in m/s
    capacities: np.ndarray,      # (E,) edge vehicle capacities
    bpr_alpha: float = 0.15,
    bpr_beta: float = 4.0,
) -> Tuple[float, np.ndarray]:
    """
    Edge-level BPR travel time reward.

    Returns:
        reward: -total_travel_time / N_total (negative, to minimize)
        t_e: (E,) per-edge travel times in seconds
    """
    # Vehicles per route: (N_od, K)
    veh = av_fracs * od_av_counts[:, None] + hdv_fracs * od_hdv_counts[:, None]
    # Edge volumes: sum over all OD pairs and routes
    v_e = np.einsum('ok,oke->e', veh, route_masks)                   # (E,)
    # Free-flow travel time per edge
    free_flow = lengths / np.maximum(speed_limits, 0.1)               # (E,) seconds
    # BPR volume-delay
    t_e = free_flow * (1.0 + bpr_alpha * (v_e / np.maximum(capacities, 1.0)) ** bpr_beta)
    # Total travel time = sum of edge_volume * edge_travel_time
    total_tt = np.dot(v_e, t_e)
    n_total = max(float(veh.sum()), 1.0)
    return -total_tt / n_total, t_e


def gawron_update_edge_bpr(
    hdv_frac: np.ndarray,   # (K,) current HDV route fraction for one OD pair
    route_tt: np.ndarray,    # (K,) pre-computed route travel times
    lr: float = 0.1,
) -> np.ndarray:
    """Gawron update using pre-computed route travel times from edge BPR."""
    new_frac, _ = gawron_step_from_route_tt(
        probs=hdv_frac,
        route_tt=route_tt,
        mode="replicator",
        lr=lr,
    )
    return new_frac


def gawron_step_compatible(
    hdv_frac: np.ndarray,
    route_tt: np.ndarray,
    mode: str,
    lr: float,
    alpha_zero: float,
    beta: float,
    update_mode: str,
    min_prob: float,
    cost_state: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Unified wrapper: project replicator mode + RouteRL-compatible mode."""
    return gawron_step_from_route_tt(
        probs=hdv_frac,
        route_tt=route_tt,
        mode=mode,
        lr=lr,
        cost=cost_state,
        alpha_zero=alpha_zero,
        beta=beta,
        update_mode=update_mode,
        min_prob=min_prob,
    )


def imagined_gawron_mode(mode: str) -> str:
    """
    World-model imagined rollout cannot execute RouteRL per-agent black-box.
    Use the RouteRL-compatible mean-field surrogate when black-box mode is requested.
    """
    mode_l = str(mode).lower()
    if mode_l == "routerl_blackbox":
        return "routerl"
    return mode_l


def aggregate_route_fracs(
    av_fracs: np.ndarray,        # (N_od, K)
    hdv_fracs: np.ndarray,       # (N_od, K)
    od_av_counts: np.ndarray,    # (N_od,)
    od_hdv_counts: np.ndarray,   # (N_od,)
) -> np.ndarray:
    """Population-weighted average route fractions for dynamics action. Returns (K,)."""
    veh = av_fracs * od_av_counts[:, None] + hdv_fracs * od_hdv_counts[:, None]  # (N_od, K)
    total = veh.sum()
    if total < 1e-8:
        return np.ones(av_fracs.shape[1]) / av_fracs.shape[1]
    return veh.sum(axis=0) / total  # (K,)


def route_tt_from_edge_times(
    route_registry: PathTopology,
    edge_tt: np.ndarray,  # (E,) edge travel time (seconds)
) -> np.ndarray:
    """
    Build per-route travel time from route edge indices.

    Note: route_masks are normalized (sum=1 per route) for feature aggregation.
    For travel-time metrics and Gawron route costs we need physical route TT,
    i.e. sum of edge travel times along each route.
    """
    t_e = np.asarray(edge_tt, dtype=np.float32).reshape(-1)
    n_od = int(route_registry.n_od)
    k = int(route_registry.K)
    fallback = float(np.mean(t_e)) if t_e.size > 0 else 0.0
    route_tt = np.full((n_od, k), fallback, dtype=np.float32)
    for i in range(n_od):
        od_routes = route_registry.route_edge_indices[i]
        for j in range(min(k, len(od_routes))):
            idxs = od_routes[j]
            if idxs:
                route_tt[i, j] = float(np.sum(t_e[np.asarray(idxs, dtype=np.int64)]))
    return route_tt


def load_hdv_route_fracs_from_state(
    state_path: str,
    route_registry: PathTopology,
    *,
    n_routes: int,
) -> Optional[np.ndarray]:
    """Load a static episode-level HDV route distribution from RouteRL state JSON.

    The serialized ``route_policy`` is a dense OD table, not a time-varying set of
    policies.  Only ODs with nonzero HDV counts affect the fixed edge background
    that is later injected into the world-model dynamics.
    """
    path = str(state_path or "").strip()
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load HDV background state %s: %s", path, exc)
        return None

    n_od = int(route_registry.n_od)
    k_eff = int(n_routes)
    out = np.ones((n_od, k_eff), dtype=np.float32) / float(max(k_eff, 1))
    route_policy = state.get("route_policy") if isinstance(state, dict) else None
    if not isinstance(route_policy, list):
        logger.warning("HDV background state has no route_policy list: %s", path)
        return out

    n_dest = max(1, len(route_registry.destinations))
    loaded = 0
    loaded_active = 0
    active_od_mask = np.asarray(route_registry.od_hdv_counts, dtype=np.float32).reshape(-1) > 0.0
    active_od_total = int(active_od_mask[:n_od].sum())
    for row in route_policy:
        if not isinstance(row, dict):
            continue
        try:
            oi = int(row.get("origin", -1))
            di = int(row.get("destination", -1))
        except (TypeError, ValueError):
            continue
        od_idx = oi * n_dest + di
        if not (0 <= od_idx < n_od):
            continue
        probs = np.asarray(row.get("probs", []), dtype=np.float32).reshape(-1)
        if probs.size <= 0:
            continue
        row_probs = np.zeros((k_eff,), dtype=np.float32)
        take = min(k_eff, int(probs.size))
        row_probs[:take] = np.clip(probs[:take], 0.0, None)
        total = float(row_probs.sum())
        if total <= 1e-8:
            row_probs[:] = 1.0 / float(max(k_eff, 1))
        else:
            row_probs /= total
        out[od_idx] = row_probs
        loaded += 1
        if bool(active_od_mask[od_idx]):
            loaded_active += 1

    logger.info(
        "HDV static route distribution loaded: dense_od_rows=%d/%d active_hdv_od_rows=%d/%d <- %s",
        loaded,
        n_od,
        loaded_active,
        active_od_total,
        path,
    )
    return out


def build_hdv_background_edge_action(
    route_registry: PathTopology,
    hdv_route_fracs: np.ndarray,
    *,
    n_edges: int,
    edge_travel_times: Optional[np.ndarray] = None,
    horizon_seconds: Optional[float] = None,
) -> np.ndarray:
    """Project static episode-level HDV route distribution to one fixed edge background."""
    n_edges = int(n_edges)
    out = np.zeros((n_edges,), dtype=np.float32)
    if n_edges <= 0:
        return out
    hdv_counts = np.asarray(route_registry.od_hdv_counts, dtype=np.float32).reshape(-1)
    probs_all = np.asarray(hdv_route_fracs, dtype=np.float32)
    if probs_all.ndim != 2:
        return out

    use_horizon = edge_travel_times is not None and horizon_seconds is not None
    edge_tt = None
    if use_horizon:
        edge_tt = np.asarray(edge_travel_times, dtype=np.float32).reshape(-1)
        use_horizon = int(edge_tt.size) >= n_edges
    horizon = max(float(horizon_seconds or 0.0), 1.0)

    total_hdv = 0.0
    for od_idx, count in enumerate(hdv_counts[: int(route_registry.n_od)]):
        c = float(max(count, 0.0))
        if c <= 0.0:
            continue
        routes = route_registry.route_edge_indices[od_idx]
        if not routes:
            continue
        probs = probs_all[od_idx, : min(int(route_registry.K), probs_all.shape[1])].astype(np.float32)
        probs = np.clip(probs, 0.0, None)
        ps = float(probs.sum())
        if ps <= 1e-8:
            probs = np.ones((min(int(route_registry.K), len(routes)),), dtype=np.float32)
            probs /= float(max(int(probs.size), 1))
        else:
            probs /= ps
        total_hdv += c
        for k, prob in enumerate(probs[: len(routes)]):
            p = float(prob)
            if p <= 0.0:
                continue
            raw_idxs = [int(e) for e in routes[k] if 0 <= int(e) < n_edges]
            if not raw_idxs:
                continue
            if use_horizon:
                eta = 0.0
                keep: List[int] = []
                for edge_idx in raw_idxs:
                    if eta >= horizon:
                        break
                    keep.append(edge_idx)
                    eta += float(edge_tt[edge_idx])
                raw_idxs = keep
            if raw_idxs:
                np.add.at(out, np.asarray(raw_idxs, dtype=np.int64), c * p)

    if total_hdv > 1e-8:
        out /= float(total_hdv)
    return out.astype(np.float32)


def episode_av_pressure_from_action_edges(
    action_edges: List[torch.Tensor],
    *,
    n_edges: int,
) -> np.ndarray:
    """Aggregate a rollout's AV edge actions into one episode-level pressure field."""
    if not action_edges:
        return np.zeros((int(n_edges),), dtype=np.float32)
    arrs: List[np.ndarray] = []
    for action in action_edges:
        if action is None:
            continue
        a = action.detach().float().cpu().numpy().reshape(-1)
        if a.size < int(n_edges):
            padded = np.zeros((int(n_edges),), dtype=np.float32)
            padded[: a.size] = a.astype(np.float32)
            a = padded
        arrs.append(np.clip(a[: int(n_edges)], 0.0, None).astype(np.float32))
    if not arrs:
        return np.zeros((int(n_edges),), dtype=np.float32)
    return np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float32)


def build_hdv_route_fracs_from_episode_av_pressure(
    route_registry: PathTopology,
    episode_av_pressure: np.ndarray,
    *,
    base_edge_tt: np.ndarray,
    n_routes: int,
    pressure_weight: float,
    temperature: float,
    prior_route_fracs: Optional[np.ndarray] = None,
    prior_mix: float = 0.0,
) -> np.ndarray:
    """Approximate the converged HDV route response to an episode-level AV pressure field.

    This is an explicit Wardrop/Gawron-style response model, not a per-step HDV
    action.  Each HDV OD receives route costs from current WM edge travel times
    plus a penalty for overlap with the full-episode AV planned pressure, then
    converts those route costs into a static route distribution.
    """
    n_od = int(route_registry.n_od)
    k_eff = int(n_routes)
    edge_tt = np.asarray(base_edge_tt, dtype=np.float32).reshape(-1)
    pressure = np.asarray(episode_av_pressure, dtype=np.float32).reshape(-1)
    n_edges = int(route_registry.n_edges)
    if edge_tt.size < n_edges:
        edge_tt = np.pad(edge_tt, (0, n_edges - edge_tt.size), constant_values=float(edge_tt.mean() if edge_tt.size else 1.0))
    if pressure.size < n_edges:
        pressure = np.pad(pressure, (0, n_edges - pressure.size), constant_values=0.0)
    edge_cost = edge_tt[:n_edges] + float(pressure_weight) * np.clip(pressure[:n_edges], 0.0, None)
    temp = max(float(temperature), 1e-6)
    out = np.ones((n_od, k_eff), dtype=np.float32) / float(max(k_eff, 1))
    prior = None
    if prior_route_fracs is not None:
        prior = np.asarray(prior_route_fracs, dtype=np.float32)
        if prior.shape[0] != n_od:
            prior = None
    mix = float(np.clip(float(prior_mix), 0.0, 1.0))

    for od_idx in range(n_od):
        routes = route_registry.route_edge_indices[od_idx]
        costs = np.full((k_eff,), np.inf, dtype=np.float32)
        for k in range(min(k_eff, len(routes))):
            idxs = [int(e) for e in routes[k] if 0 <= int(e) < n_edges]
            if idxs:
                costs[k] = float(np.sum(edge_cost[np.asarray(idxs, dtype=np.int64)]))
        valid = np.isfinite(costs)
        if not bool(valid.any()):
            if prior is not None and prior.shape[1] >= k_eff:
                row = np.clip(prior[od_idx, :k_eff], 0.0, None)
                total = float(row.sum())
                out[od_idx] = row / total if total > 1e-8 else out[od_idx]
            continue
        min_cost = float(np.min(costs[valid]))
        logits = np.zeros((k_eff,), dtype=np.float32)
        logits[valid] = -(costs[valid] - min_cost) / temp
        logits[~valid] = -80.0
        logits = logits - float(np.max(logits))
        probs = np.exp(logits).astype(np.float32)
        probs[~valid] = 0.0
        total = float(probs.sum())
        probs = probs / total if total > 1e-8 else out[od_idx]
        if prior is not None and prior.shape[1] >= k_eff and mix > 0.0:
            prior_row = np.clip(prior[od_idx, :k_eff], 0.0, None)
            prior_total = float(prior_row.sum())
            if prior_total > 1e-8:
                prior_row = prior_row / prior_total
                probs = (1.0 - mix) * probs + mix * prior_row
                probs = probs / float(max(float(probs.sum()), 1e-8))
        out[od_idx] = probs.astype(np.float32)
    return out


def combine_av_hdv_edge_background(
    av_action_edge: torch.Tensor,
    hdv_background_edge: Optional[torch.Tensor],
    *,
    n_active_av: int,
    n_total_hdv: int,
    mode: str,
    weight: float,
) -> torch.Tensor:
    """Combine controllable AV edge action with fixed episode-level HDV background."""
    if hdv_background_edge is None or float(weight) == 0.0:
        return av_action_edge
    hdv = hdv_background_edge.to(device=av_action_edge.device, dtype=av_action_edge.dtype)
    mode_norm = str(mode or "additive").strip().lower()
    w = float(weight)
    if mode_norm == "population":
        av_count = float(max(int(n_active_av), 0))
        hdv_count = float(max(int(n_total_hdv), 0)) * w
        denom = max(av_count + hdv_count, 1e-8)
        return (av_action_edge * av_count + hdv * hdv_count) / denom
    if mode_norm != "additive":
        raise ValueError(f"Unsupported dreamer_hdv_background_combine={mode!r}")
    return av_action_edge + w * hdv


def build_vehicle_od_assignments(
    route_registry: PathTopology,
    n_av: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build per-vehicle OD assignment indices for AV agents.

    Uses route_registry.od_av_counts as prior and adjusts to exact n_av.
    Returns:
        od_idx_per_vehicle: (n_av,) int64
    """
    n_av = int(max(1, n_av))
    n_od = int(route_registry.n_od)
    counts = np.asarray(route_registry.od_av_counts, dtype=np.float64).reshape(-1)
    if counts.shape[0] != n_od:
        counts = np.ones((n_od,), dtype=np.float64)
    counts = np.clip(counts, 0.0, None)
    if float(counts.sum()) <= 1e-8:
        probs = np.ones((n_od,), dtype=np.float64) / float(n_od)
    else:
        probs = counts / float(counts.sum())

    expected = probs * float(n_av)
    base = np.floor(expected).astype(np.int64)
    frac = expected - base.astype(np.float64)
    remain = int(n_av - int(base.sum()))
    if remain > 0:
        order = np.argsort(-frac)
        for idx in order[:remain]:
            base[int(idx)] += 1
    elif remain < 0:
        order = np.argsort(-base)
        need = -remain
        for idx in order:
            if need <= 0:
                break
            take = min(int(base[int(idx)]), need)
            base[int(idx)] -= take
            need -= take

    od_idx = np.repeat(np.arange(n_od, dtype=np.int64), base)
    if od_idx.shape[0] < n_av:
        extra = rng.choice(np.arange(n_od, dtype=np.int64), size=(n_av - od_idx.shape[0],), p=probs)
        od_idx = np.concatenate([od_idx, extra.astype(np.int64)], axis=0)
    elif od_idx.shape[0] > n_av:
        rng.shuffle(od_idx)
        od_idx = od_idx[:n_av]
    rng.shuffle(od_idx)
    return od_idx.astype(np.int64)


def build_vehicle_edge_states(
    route_registry: PathTopology,
    edge_to_idx: Dict[str, int],
    n_av: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-vehicle (OD assignment, current edge, destination edge).
    """
    od_idx = build_vehicle_od_assignments(route_registry, n_av, rng)
    n_dest = max(1, len(route_registry.destinations))
    cur_edge = np.zeros((od_idx.shape[0],), dtype=np.int64)
    dst_edge = np.zeros((od_idx.shape[0],), dtype=np.int64)
    for i, od in enumerate(od_idx.tolist()):
        oi = int(od // n_dest)
        di = int(od % n_dest)
        orig_id = route_registry.origins[oi] if oi < len(route_registry.origins) else route_registry.origins[0]
        dest_id = route_registry.destinations[di] if di < len(route_registry.destinations) else route_registry.destinations[0]
        c = int(edge_to_idx.get(orig_id, 0))
        d = int(edge_to_idx.get(dest_id, c))
        cur_edge[i] = c
        dst_edge[i] = d
    return od_idx, cur_edge, dst_edge


def build_lightweight_vehicle_path_topology(
    *,
    origins: List[str],
    destinations: List[str],
    n_edges: int,
    K: int,
    agents_csv: Optional[str],
    ratio_machines: float,
    edge_to_idx: Optional[Dict[str, int]] = None,
    routes_csv: Optional[str] = None,
) -> PathTopology:
    """
    Build a lightweight path topology for vehicle mode without precomputing K OD routes.

    Vehicle-level imagination constructs candidate paths online from the current edge to
    the destination edge, so it only needs:
      - origin/destination edge vocabularies
      - per-OD AV/HDV counts for sampling vehicle assignments
      - a lightweight topology container to keep the rest of the pipeline unchanged
    """
    n_orig = len(origins)
    n_dest = len(destinations)
    n_od = n_orig * n_dest
    route_edge_indices = [[[] for _ in range(int(K))] for _ in range(n_od)]
    route_masks = np.zeros((n_od, int(K), int(n_edges)), dtype=np.float32)
    od_av_counts = np.zeros((n_od,), dtype=np.float32)
    od_hdv_counts = np.zeros((n_od,), dtype=np.float32)

    if routes_csv and edge_to_idx and os.path.exists(routes_csv):
        import csv

        route_offsets: Dict[int, int] = {}
        with open(routes_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    oi_raw = int(row.get("origins", row.get("origin", -1)))
                    di_raw = int(row.get("destinations", row.get("destination", -1)))
                except (TypeError, ValueError):
                    continue
                if not (0 <= oi_raw < n_orig and 0 <= di_raw < n_dest):
                    continue
                od_idx = oi_raw * n_dest + di_raw
                k = int(route_offsets.get(od_idx, 0))
                if k >= int(K):
                    continue
                path_names = str(row.get("path", "")).split()
                idxs = [
                    int(edge_to_idx[e])
                    for e in path_names
                    if e in edge_to_idx
                ]
                if not idxs:
                    continue
                route_edge_indices[od_idx][k] = idxs
                w = 1.0 / float(len(idxs))
                for idx in idxs:
                    if 0 <= int(idx) < int(n_edges):
                        route_masks[od_idx, k, int(idx)] = w
                route_offsets[od_idx] = k + 1

    if agents_csv and os.path.exists(agents_csv):
        import csv

        with open(agents_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                oi_raw = int(row["origin"])
                di_raw = int(row["destination"])
                if oi_raw < n_orig and di_raw < n_dest:
                    od_idx = oi_raw * n_dest + di_raw
                    od_hdv_counts[od_idx] += 1.0

        total_agents = float(od_hdv_counts.sum())
        if total_agents > 0.0:
            od_av_counts, od_hdv_counts = split_od_agent_counts(
                od_hdv_counts,
                float(ratio_machines),
            )

    return PathTopology(
        origins=origins,
        destinations=destinations,
        K=int(K),
        route_edge_indices=route_edge_indices,
        route_masks=route_masks,
        od_av_counts=od_av_counts,
        od_hdv_counts=od_hdv_counts,
        n_edges=int(n_edges),
    )


# Backward-compatible alias during internal rename.
build_lightweight_vehicle_route_registry = build_lightweight_vehicle_path_topology


def build_diverse_paths_current_to_dest(
    adj_list: List[List[int]],
    edge_tt: np.ndarray,
    src_idx: int,
    dst_idx: int,
    K: int,
    penalty_factor: float,
) -> List[List[int]]:
    """
    Build K diverse shortest paths on edge graph from current edge to destination edge.
    """
    src = int(src_idx)
    dst = int(dst_idx)
    K = int(max(1, K))
    if src == dst:
        return [[src] for _ in range(K)]

    weights = np.asarray(edge_tt, dtype=np.float64).copy()
    penalty = float(max(1.0, penalty_factor))
    paths: List[List[int]] = []
    seen: set = set()
    for _ in range(K):
        path = PathTopology._dijkstra(adj_list, weights, src, dst)
        if path is None or len(path) == 0:
            break
        key = tuple(int(x) for x in path)
        tries = 0
        while key in seen and tries < 3:
            for ei in path:
                weights[int(ei)] *= penalty
            path = PathTopology._dijkstra(adj_list, weights, src, dst)
            if path is None or len(path) == 0:
                break
            key = tuple(int(x) for x in path)
            tries += 1
        if path is None or len(path) == 0 or key in seen:
            break
        p_int = [int(x) for x in path]
        paths.append(p_int)
        seen.add(key)
        for ei in p_int:
            weights[int(ei)] *= penalty

    if not paths:
        paths = [[src]]
    while len(paths) < K:
        paths.append(list(paths[0]))
    return paths[:K]


# ---------------------------------------------------------------------------
# Pre-computed path cache for vehicle-level imagination
# ---------------------------------------------------------------------------

class VehiclePathCache:
    """
    Pre-computes K diverse paths from every reachable edge to each unique
    destination, so that per-step imagination only needs a dict lookup
    instead of running Dijkstra per vehicle.
    """

    def __init__(
        self,
        adj_list: List[List[int]],
        edge_tt_ref: np.ndarray,
        dest_edges_unique: List[int],
        n_edges: int,
        K: int,
        penalty_factor: float = 3.0,
    ):
        self.K = int(K)
        self.n_edges = int(n_edges)
        self.dest_edges = list(dest_edges_unique)
        # cache: (src, dst) -> List[List[int]]  (K paths)
        self._cache: Dict[Tuple[int, int], List[List[int]]] = {}
        self._build(adj_list, edge_tt_ref, penalty_factor)

    def _build(
        self,
        adj_list: List[List[int]],
        edge_tt_ref: np.ndarray,
        penalty_factor: float,
    ) -> None:
        import heapq

        n = self.n_edges
        penalty = float(max(1.0, penalty_factor))

        # Build reverse adjacency list: rev_adj[v] = list of u where u→v
        rev_adj: List[List[int]] = [[] for _ in range(n)]
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                rev_adj[v].append(u)

        def _reverse_dijkstra(
            weights: np.ndarray, dst: int,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Run Dijkstra on reversed graph from dst.
            Returns (dist, prev) arrays for ALL sources."""
            dist = np.full(n, np.inf)
            prev = np.full(n, -1, dtype=np.int64)
            dist[dst] = weights[dst]
            heap: List[tuple] = [(float(dist[dst]), dst)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue
                for v in rev_adj[u]:
                    nd = d + weights[v]
                    if nd < dist[v]:
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(heap, (nd, v))
            return dist, prev

        def _extract_path(prev: np.ndarray, src: int, dst: int) -> Optional[List[int]]:
            """Extract path from src to dst using prev from reverse Dijkstra."""
            if prev[src] == -1 and src != dst:
                return None
            path = [src]
            u = src
            while u != dst:
                u = int(prev[u])
                if u == -1:
                    return None
                path.append(u)
            return path

        for dst in self.dest_edges:
            # Round 0: compute shortest path tree (unpenalized)
            base_weights = np.asarray(edge_tt_ref, dtype=np.float64).copy()
            dist0, prev0 = _reverse_dijkstra(base_weights, dst)

            # For each reachable source, collect K diverse paths via per-src penalty
            for src in range(n):
                if src == dst:
                    self._cache[(src, dst)] = [[src] for _ in range(self.K)]
                    continue
                if np.isinf(dist0[src]):
                    continue

                # First path from the shared shortest path tree
                path0 = _extract_path(prev0, src, dst)
                if path0 is None:
                    continue

                paths: List[List[int]] = [path0]
                seen: set = {tuple(path0)}

                # Subsequent paths: per-src penalty on individual Dijkstra
                if self.K > 1:
                    w = base_weights.copy()
                    for ei in path0:
                        w[ei] *= penalty
                    for _ in range(self.K - 1):
                        p = PathTopology._dijkstra(adj_list, w, src, dst)
                        if p is None:
                            break
                        key = tuple(p)
                        if key in seen:
                            break
                        paths.append([int(x) for x in p])
                        seen.add(key)
                        for ei in p:
                            w[int(ei)] *= penalty

                while len(paths) < self.K:
                    paths.append(list(paths[0]))
                self._cache[(src, dst)] = paths[:self.K]

    def lookup(self, src: int, dst: int) -> List[List[int]]:
        cached = self._cache.get((int(src), int(dst)))
        if cached is not None:
            return cached
        # Unreachable pair — return trivial
        return [[int(src)] for _ in range(self.K)]

    def lookup_batch(
        self,
        current_edges: np.ndarray,
        dest_edges: np.ndarray,
        done: np.ndarray,
    ) -> List[List[List[int]]]:
        """Vectorized lookup for all vehicles at once."""
        n = len(current_edges)
        batch: List[List[List[int]]] = []
        for i in range(n):
            if bool(done[i]):
                c = int(current_edges[i])
                batch.append([[c] for _ in range(self.K)])
            else:
                batch.append(self.lookup(int(current_edges[i]), int(dest_edges[i])))
        return batch


class LazyVehiclePathCache:
    """
    On-demand path cache for large networks where precomputing all (src, dst)
    pairs is too expensive. Paths are generated once per queried pair and then
    reused across later rollout steps and episodes.
    """

    def __init__(
        self,
        adj_list: List[List[int]],
        n_edges: int,
        K: int,
        penalty_factor: float = 3.0,
        max_entries: int = 250000,
    ):
        self.adj_list = adj_list
        self.n_edges = int(n_edges)
        self.K = int(K)
        self.penalty_factor = float(penalty_factor)
        self.max_entries = int(max_entries)
        self._cache: Dict[Tuple[int, int], List[List[int]]] = {}

    def _compute(self, src: int, dst: int, edge_tt: np.ndarray) -> List[List[int]]:
        return build_diverse_paths_current_to_dest(
            adj_list=self.adj_list,
            edge_tt=edge_tt,
            src_idx=int(src),
            dst_idx=int(dst),
            K=self.K,
            penalty_factor=self.penalty_factor,
        )

    def lookup(self, src: int, dst: int, edge_tt: np.ndarray) -> List[List[int]]:
        key = (int(src), int(dst))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        paths = self._compute(int(src), int(dst), edge_tt)
        if len(self._cache) < self.max_entries:
            self._cache[key] = paths
        return paths

    def lookup_batch(
        self,
        current_edges: np.ndarray,
        dest_edges: np.ndarray,
        done: np.ndarray,
        edge_tt: np.ndarray,
    ) -> List[List[List[int]]]:
        batch: List[List[List[int]]] = []
        n = len(current_edges)
        for i in range(n):
            cur = int(current_edges[i])
            if bool(done[i]):
                batch.append([[cur] for _ in range(self.K)])
            else:
                batch.append(self.lookup(cur, int(dest_edges[i]), edge_tt))
        return batch


def _od_suffix_path_batch(
    *,
    route_registry: PathTopology,
    active_avs: List[Dict[str, Any]],
    current_edges: np.ndarray,
    done: np.ndarray,
    K: int,
) -> List[List[List[int]]]:
    """Return OD-route suffix candidates matching the SUMO od_suffix action space."""
    batch: List[List[List[int]]] = []
    n_od = int(route_registry.n_od)
    k_eff = int(K)
    for i, av in enumerate(active_avs):
        cur = int(current_edges[i])
        if bool(done[i]):
            batch.append([[cur] for _ in range(k_eff)])
            continue
        od_idx = int(av.get("od_idx", -1))
        if od_idx < 0 or od_idx >= n_od:
            batch.append([[cur] for _ in range(k_eff)])
            continue
        suffixes: List[List[int]] = []
        for route in route_registry.route_edge_indices[od_idx][:k_eff]:
            matches = [j for j, edge_idx in enumerate(route) if int(edge_idx) == cur]
            if not matches:
                suffixes.append([])
                continue
            suffix = list(route[int(matches[-1]):])
            if suffix and int(suffix[0]) == cur:
                suffixes.append(suffix)
            else:
                suffixes.append([])
        while len(suffixes) < k_eff:
            suffixes.append([])
        batch.append(suffixes[:k_eff])
    return batch


def _ensure_active_vehicle_current_edges(
    *,
    active_avs: List[Dict[str, Any]],
    current_step: int,
    avg_edge_time: float,
    step_interval: float,
    route_registry: Optional[PathTopology],
) -> None:
    """Fill current_edge for vehicles reconstructed mid-episode.

    ``rebuild_active_set`` can recreate vehicles that departed before the
    sampled imagination start step.  Those vehicles must not be treated as if
    they were still at the OD origin; otherwise current-edge route suffixes are
    built from the wrong action space.  Use route slot 0 only as a position
    prior for reconstructed vehicles.  Candidate order remains unchanged.
    """
    if not active_avs:
        return
    for av in active_avs:
        if "current_edge" in av:
            av["current_edge"] = int(av.get("current_edge", av.get("origin_edge", 0)))
            continue
        ref_route: List[int] = []
        if route_registry is not None:
            od_idx = int(av.get("od_idx", -1))
            if 0 <= od_idx < int(route_registry.n_od):
                od_routes = route_registry.route_edge_indices[od_idx]
                if od_routes:
                    ref_route = list(od_routes[0])
        av["current_edge"] = int(
            estimate_vehicle_position(
                av,
                int(current_step),
                float(avg_edge_time),
                float(step_interval),
                ref_route,
            )
        )


def _vehicle_obs_route_stats_mode(route_stats: str) -> str:
    mode = str(route_stats or "mean").strip().lower()
    aliases = {
        "": "mean",
        "avg": "mean",
        "average": "mean",
        "bottleneck_cost": "bottleneck",
        "route_cost": "bottleneck",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"mean", "bottleneck"}:
        raise ValueError(f"Unsupported vehicle_obs_route_stats={route_stats!r}; expected 'mean' or 'bottleneck'")
    return mode


def vehicle_route_feature_dim(d_obs: int, route_stats: str = "mean") -> int:
    """Feature width for one candidate route before global/context features."""
    d = int(d_obs)
    mode = _vehicle_obs_route_stats_mode(route_stats)
    if mode == "mean":
        return d
    # mean, max, q90, sum per dynamic feature + one normalized route-length feature.
    return 4 * d + 1


def _path_feature(
    edge_feats_dyn: np.ndarray,
    idxs: List[int],
    *,
    route_stats: str,
) -> np.ndarray:
    mode = _vehicle_obs_route_stats_mode(route_stats)
    d_dyn = int(edge_feats_dyn.shape[1])
    feat_dim = vehicle_route_feature_dim(d_dyn, mode)
    if not idxs:
        return np.zeros((feat_dim,), dtype=np.float32)

    idx_arr = np.asarray(idxs, dtype=np.int64)
    idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < int(edge_feats_dyn.shape[0]))]
    if idx_arr.size == 0:
        return np.zeros((feat_dim,), dtype=np.float32)

    vals = edge_feats_dyn[idx_arr].astype(np.float32, copy=False)
    if mode == "mean":
        return vals.mean(axis=0).astype(np.float32)

    length_norm = np.array(
        [min(float(np.log1p(idx_arr.size) / np.log1p(128.0)), 2.0)],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            vals.mean(axis=0),
            vals.max(axis=0),
            np.quantile(vals, 0.90, axis=0).astype(np.float32),
            vals.sum(axis=0),
            length_norm,
        ],
        axis=0,
    ).astype(np.float32)


def _global_route_feature(edge_feats_dyn: np.ndarray, *, route_stats: str) -> np.ndarray:
    mode = _vehicle_obs_route_stats_mode(route_stats)
    d_dyn = int(edge_feats_dyn.shape[1])
    feat_dim = vehicle_route_feature_dim(d_dyn, mode)
    if edge_feats_dyn.size == 0:
        return np.zeros((feat_dim,), dtype=np.float32)
    vals = edge_feats_dyn.astype(np.float32, copy=False)
    if mode == "mean":
        return vals.mean(axis=0).astype(np.float32)
    active = (np.abs(vals).sum(axis=1) > 1e-6).astype(np.float32)
    active_frac = np.array([float(active.mean())], dtype=np.float32)
    return np.concatenate(
        [
            vals.mean(axis=0),
            vals.max(axis=0),
            np.quantile(vals, 0.90, axis=0).astype(np.float32),
            vals.std(axis=0),
            active_frac,
        ],
        axis=0,
    ).astype(np.float32)


def build_vehicle_route_cost_obs_features(
    path_batch: List[List[List[int]]],
    edge_cost: Optional[np.ndarray],
    n_routes: int,
) -> np.ndarray:
    """Build non-leaky candidate route-cost features from current edge costs.

    Features are per candidate route: relative cost gap, log absolute cost, and
    valid-route mask.  Costs come from the current WM/SUMO state only; no
    future rollout labels or OD-specific best-route table are used.
    """
    n_agents = len(path_batch)
    k_eff = int(n_routes)
    out = np.zeros((n_agents, 3 * k_eff), dtype=np.float32)
    if n_agents <= 0 or k_eff <= 0 or edge_cost is None:
        return out

    edge_cost_arr = np.asarray(edge_cost, dtype=np.float32).reshape(-1)
    n_edges = int(edge_cost_arr.shape[0])
    costs = np.full((n_agents, k_eff), np.nan, dtype=np.float32)
    valid = np.zeros((n_agents, k_eff), dtype=bool)

    for i, paths in enumerate(path_batch):
        for k in range(min(k_eff, len(paths))):
            idxs = [int(e) for e in paths[k] if 0 <= int(e) < n_edges]
            if not idxs:
                continue
            vals = edge_cost_arr[np.asarray(idxs, dtype=np.int64)]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            costs[i, k] = float(np.maximum(vals, 0.0).sum())
            valid[i, k] = True

    for i in range(n_agents):
        finite = np.isfinite(costs[i]) & valid[i]
        if not finite.any():
            continue
        row = costs[i].copy()
        finite_vals = row[finite]
        min_cost = float(finite_vals.min())
        scale = max(float(np.median(finite_vals)), 60.0)
        fallback = float(finite_vals.max() + scale)
        row[~finite] = fallback

        rel_gap = np.clip((row - min_cost) / scale, 0.0, 20.0)
        log_cost = np.log1p(np.maximum(row, 0.0)) / np.log1p(3600.0)
        valid_f = valid[i].astype(np.float32)
        out[i, 0:k_eff] = rel_gap.astype(np.float32)
        out[i, k_eff:2 * k_eff] = log_cost.astype(np.float32)
        out[i, 2 * k_eff:3 * k_eff] = valid_f

    return np.nan_to_num(out, nan=0.0, posinf=20.0, neginf=0.0).astype(np.float32)


def build_vehicle_fleet_pressure_obs_features(
    path_batch: List[List[List[int]]],
    *,
    n_edges: int,
    n_routes: int,
    selected_route_counts: Optional[np.ndarray] = None,
    selected_edge_pressure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build same-tick fleet context features for marginal route decisions.

    Features are [route_share_0..K-1, planned_pressure_overlap_0..K-1].
    They summarize only routes selected earlier in the current decision tick,
    so they are non-leaky under a sequential actor evaluation order.
    """
    n_agents = len(path_batch)
    k_eff = int(n_routes)
    out = np.zeros((n_agents, 2 * k_eff), dtype=np.float32)
    if n_agents <= 0 or k_eff <= 0:
        return out

    counts = (
        np.asarray(selected_route_counts, dtype=np.float32).reshape(-1)
        if selected_route_counts is not None
        else np.zeros((k_eff,), dtype=np.float32)
    )
    if counts.shape[0] < k_eff:
        padded = np.zeros((k_eff,), dtype=np.float32)
        padded[: counts.shape[0]] = counts
        counts = padded
    else:
        counts = counts[:k_eff]
    denom = max(float(np.maximum(counts.sum(), 0.0)), 1.0)
    route_share = np.clip(counts / denom, 0.0, 1.0).astype(np.float32)

    pressure = (
        np.asarray(selected_edge_pressure, dtype=np.float32).reshape(-1)
        if selected_edge_pressure is not None
        else np.zeros((max(int(n_edges), 0),), dtype=np.float32)
    )
    if pressure.shape[0] < int(n_edges):
        padded = np.zeros((max(int(n_edges), 0),), dtype=np.float32)
        padded[: pressure.shape[0]] = pressure
        pressure = padded
    else:
        pressure = pressure[: max(int(n_edges), 0)]
    pressure_norm = max(denom, 1.0)

    for i, paths in enumerate(path_batch):
        out[i, :k_eff] = route_share
        for k in range(min(k_eff, len(paths))):
            idxs = sorted({int(e) for e in paths[k] if 0 <= int(e) < int(pressure.shape[0])})
            if idxs:
                out[i, k_eff + k] = float(pressure[np.asarray(idxs, dtype=np.int64)].sum()) / pressure_norm

    return np.nan_to_num(out, nan=0.0, posinf=20.0, neginf=0.0).astype(np.float32)


def build_vehicle_obs_vectorized(
    edge_feats_dyn: np.ndarray,    # (E, n_dynamic)
    path_batch: List[List[List[int]]],  # [N_av][K][edge_indices]
    include_global: bool,
    route_stats: str = "mean",
    edge_cost: Optional[np.ndarray] = None,
    include_route_costs: bool = False,
    include_fleet_pressure: bool = False,
    fleet_route_counts: Optional[np.ndarray] = None,
    fleet_edge_pressure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorized version of build_vehicle_obs_from_path_batch."""
    n_agents = len(path_batch)
    if n_agents == 0:
        return np.zeros((0, 0), dtype=np.float32)
    K = len(path_batch[0])
    d_dyn = int(edge_feats_dyn.shape[1])
    route_feat_dim = vehicle_route_feature_dim(d_dyn, route_stats)
    route_cost_dim = 3 * K if bool(include_route_costs) else 0
    fleet_pressure_dim = 2 * K if bool(include_fleet_pressure) else 0
    obs_dim = K * route_feat_dim + (route_feat_dim if include_global else 0) + fleet_pressure_dim + route_cost_dim
    out = np.zeros((n_agents, obs_dim), dtype=np.float32)
    global_dyn = _global_route_feature(edge_feats_dyn, route_stats=route_stats)
    base_dim = K * route_feat_dim + (route_feat_dim if include_global else 0)

    for i in range(n_agents):
        off = 0
        for k in range(K):
            out[i, off: off + route_feat_dim] = _path_feature(
                edge_feats_dyn,
                path_batch[i][k],
                route_stats=route_stats,
            )
            off += route_feat_dim
        if include_global:
            out[i, off: off + route_feat_dim] = global_dyn
    if include_fleet_pressure:
        fleet_features = build_vehicle_fleet_pressure_obs_features(
            path_batch=path_batch,
            n_edges=int(edge_feats_dyn.shape[0]),
            n_routes=K,
            selected_route_counts=fleet_route_counts,
            selected_edge_pressure=fleet_edge_pressure,
        )
    if include_route_costs:
        cost_features = build_vehicle_route_cost_obs_features(
            path_batch=path_batch,
            edge_cost=edge_cost,
            n_routes=K,
        )
        out[:, base_dim:base_dim + route_cost_dim] = cost_features
    if include_fleet_pressure:
        fleet_start = base_dim + route_cost_dim
        out[:, fleet_start:fleet_start + fleet_pressure_dim] = fleet_features
    return out


def build_vehicle_obs_from_path_batch(
    edge_feats_dyn: np.ndarray,  # (E, n_dynamic)
    path_batch: List[List[List[int]]],  # [N_av][K][edge_idx]
    include_global: bool,
    route_stats: str = "mean",
    edge_cost: Optional[np.ndarray] = None,
    include_route_costs: bool = False,
    include_fleet_pressure: bool = False,
    fleet_route_counts: Optional[np.ndarray] = None,
    fleet_edge_pressure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build per-vehicle observations from candidate paths.
    """
    return build_vehicle_obs_vectorized(
        edge_feats_dyn=edge_feats_dyn,
        path_batch=path_batch,
        include_global=include_global,
        route_stats=route_stats,
        edge_cost=edge_cost,
        include_route_costs=include_route_costs,
        include_fleet_pressure=include_fleet_pressure,
        fleet_route_counts=fleet_route_counts,
        fleet_edge_pressure=fleet_edge_pressure,
    )


def aggregate_vehicle_actions_to_edge_distribution(
    av_vehicle_fracs: np.ndarray,   # (N_av, K)
    path_batch: List[List[List[int]]],
    n_edges: int,
    n_routes: int,
    *,
    current_edges: Optional[np.ndarray] = None,
    edge_travel_times: Optional[np.ndarray] = None,
    horizon_seconds: Optional[float] = None,
) -> np.ndarray:
    """
    Aggregate per-vehicle path probabilities into planned edge-flow action.

    This mirrors collection-time ``av_planned_edge_flow``: count edges expected
    to be entered within the next decision interval, then divide by active AVs.
    """
    if av_vehicle_fracs.size == 0 or not path_batch or int(n_edges) <= 0:
        return np.zeros((int(n_edges),), dtype=np.float32)

    edge_chunks: List[np.ndarray] = []
    weight_chunks: List[np.ndarray] = []
    n_agents = min(int(av_vehicle_fracs.shape[0]), len(path_batch))
    current_arr = None if current_edges is None else np.asarray(current_edges, dtype=np.int64).reshape(-1)
    edge_tt = None
    use_horizon = edge_travel_times is not None and horizon_seconds is not None
    if use_horizon:
        edge_tt = np.asarray(edge_travel_times, dtype=np.float32).reshape(-1)
        if edge_tt.size < int(n_edges):
            use_horizon = False
    horizon = max(float(horizon_seconds or 0.0), 1.0)
    n_controlled = 0
    for i in range(n_agents):
        paths = path_batch[i]
        if not paths:
            continue
        k_eff = int(min(len(paths), int(n_routes), int(av_vehicle_fracs.shape[1])))
        if k_eff <= 0:
            continue
        n_controlled += 1
        probs = np.asarray(av_vehicle_fracs[i][:k_eff], dtype=np.float32)
        probs = np.clip(probs, 0.0, None)
        prob_sum = float(probs.sum())
        if prob_sum <= 1e-8:
            probs = np.ones((k_eff,), dtype=np.float32) / float(k_eff)
        else:
            probs = probs / prob_sum
        for k in range(k_eff):
            path = paths[k]
            if not path:
                continue
            idxs = np.asarray(path, dtype=np.int64)
            idxs = idxs[(idxs >= 0) & (idxs < int(n_edges))]
            if idxs.size == 0:
                continue
            if use_horizon:
                start_pos = 0
                current_edge = None
                if current_arr is not None and i < current_arr.shape[0]:
                    current_edge = int(current_arr[i])
                    matches = np.flatnonzero(idxs == current_edge)
                    if matches.size > 0:
                        start_pos = int(matches[0])
                eta = 0.0
                keep: List[int] = []
                for pos in range(start_pos, int(idxs.size)):
                    edge_idx = int(idxs[pos])
                    if current_edge is not None and pos == start_pos and edge_idx == current_edge:
                        eta += float(edge_tt[edge_idx])
                        continue
                    if eta >= horizon:
                        break
                    keep.append(edge_idx)
                    eta += float(edge_tt[edge_idx])
                if not keep:
                    continue
                idxs = np.asarray(keep, dtype=np.int64)
            edge_chunks.append(idxs)
            weight_chunks.append(np.full((idxs.size,), float(probs[k]), dtype=np.float32))

    action_edge = np.zeros((int(n_edges),), dtype=np.float32)
    if edge_chunks:
        all_edges = np.concatenate(edge_chunks, axis=0)
        all_weights = np.concatenate(weight_chunks, axis=0)
        np.add.at(action_edge, all_edges, all_weights)

    if n_controlled > 0:
        action_edge /= float(n_controlled)
    return action_edge


def choose_vehicle_next_edge(current_edge: int, chosen_path: List[int]) -> int:
    """
    Convert chosen path into one-step next-hop edge index.
    """
    if not chosen_path:
        return int(current_edge)
    if int(chosen_path[0]) == int(current_edge):
        if len(chosen_path) >= 2:
            return int(chosen_path[1])
        return int(chosen_path[0])
    return int(chosen_path[0])


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


def compute_mean_field_agent_rewards(
    mode: str,
    global_reward: float,
    av_fracs: np.ndarray,      # (N_od, K)
    hdv_fracs: np.ndarray,     # (N_od, K)
    route_tt: np.ndarray,      # (N_od, K)
    od_av_counts: np.ndarray,  # (N_od,)
    od_hdv_counts: np.ndarray, # (N_od,)
    route_masks: np.ndarray,   # (N_od, K, E)
    lengths: np.ndarray,       # (E,)
    speed_limits: np.ndarray,  # (E,)
    capacities: np.ndarray,    # (E,)
    bpr_alpha: float,
    bpr_beta: float,
) -> np.ndarray:
    """
    Compute per-OD rewards for credit assignment.

    Modes:
      - global: each OD agent gets the same global reward.
      - local: per-OD AV expected route travel time.
      - difference: counterfactual difference reward
          D_i = R_global - R_cf(i uses uniform AV distribution).
    """
    n_od, k = av_fracs.shape
    mode = str(mode).lower()

    if mode == "global":
        return np.full(n_od, global_reward, dtype=np.float32)

    if mode == "local":
        local = -(av_fracs * route_tt).sum(axis=1)
        return local.astype(np.float32)

    if mode == "difference":
        uniform = np.ones(k, dtype=np.float32) / float(k)
        rewards = np.zeros(n_od, dtype=np.float32)
        for i in range(n_od):
            av_cf = av_fracs.copy()
            av_cf[i] = uniform
            cf_reward, _ = edge_bpr_reward(
                av_cf,
                hdv_fracs,
                od_av_counts,
                od_hdv_counts,
                route_masks,
                lengths,
                speed_limits,
                capacities,
                bpr_alpha,
                bpr_beta,
            )
            rewards[i] = float(global_reward - cf_reward)
        return rewards

    raise ValueError(f"Unknown reward_mode='{mode}'. Expected one of: global, local, difference")


def smooth_distribution(
    previous: np.ndarray,
    updated: np.ndarray,
    momentum: float,
    min_prob: float,
) -> np.ndarray:
    """Blend and clamp a probability vector to avoid route-collapse artifacts."""
    m = float(np.clip(momentum, 0.0, 1.0))
    out = (1.0 - m) * previous + m * updated
    out = np.maximum(out, float(min_prob))
    s = out.sum()
    if s <= 0:
        return np.ones_like(out) / out.size
    return out / s


# ══════════════════════════════════════════════════════════════════════════════
# Core: imagined episode (fully offline, no SUMO)
# ══════════════════════════════════════════════════════════════════════════════

def imagined_episode(
    cfg: TrainConfig,
    wm,
    policy: SharedPolicy,
    reward_model,           # TrafficMetricsRewardModel 或 LatentRewardPredictor
    dataset: OfflineInitDataset,
    rng: np.random.Generator,
    device: torch.device,
    reward_mean: float = 0.0,
    reward_std:  float = 1.0,
) -> Tuple[PPOBuffer, float, Dict[str, float]]:
    """
    One fully imagined episode:
      1. Sample (flow_0, hdv_frac) from offline dataset — no SUMO call
      2. Encode flow_0 -> z_0 via world model
      3. Roll out episode_len steps entirely in latent space
      4. Return filled PPOBuffer, total reward, and travel-time proxy metrics

    Travel-time proxy metrics (load-index, dimensionless):
      t_av  = mean_t( av_frac · joint_frac )   — AV mean route load
      t_hdv = mean_t( hdv_frac · joint_frac )  — HDV mean route load
      t_all = mean_t( joint_frac · joint_frac ) — all-agent mean route load
    For K=4 uniform routing the baseline is 0.25; lower is better.
    """
    K     = cfg.n_routes
    n_av  = cfg.n_av
    n_hdv = cfg.n_hdv

    # ── 1. Sample initial state from offline dataset ──────────────────────────
    flow_0 = dataset.sample(rng)               # (E, 8) float tensor
    flow_np  = flow_0.numpy()                  # (E, 8) float32
    hdv_frac = np.ones((K,), dtype=np.float32) / float(K)
    hdv_frac_init = hdv_frac.copy()            # keep for drift tracking
    hdv_cost_state: Optional[np.ndarray] = None

    # ── 2. Encode initial flow -> latent ──────────────────────────────────
    z = wm.encode_observation(flow_np)         # (E, 64) float32 numpy

    wm_buffer = RolloutBuffer(
        max_length=cfg.history_length,
        n_spatial=wm.n_edges,
        d_spatial=wm.d_spatial,
        device=cfg.device,
    )
    wm_buffer.add(z)

    obs_dim = wm.n_edges * wm.d_obs
    ppo_buf = PPOBuffer(cfg.episode_len, obs_dim, K)

    total_reward = 0.0

    # accumulators for travel-time proxy
    sum_t_av  = 0.0
    sum_t_hdv = 0.0
    sum_t_all = 0.0

    # ── 3. Imagined rollout ───────────────────────────────────────────────────
    for _ in range(cfg.episode_len):
        obs_flat = wm.policy_obs(z).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, E*n_dynamic)

        with torch.no_grad():
            av_frac, raw, log_prob, _, value = policy.get_action_and_value(obs_flat)

        # av_frac: (1, K) → (K,) numpy；传入 world model
        av_frac_np = av_frac[0].cpu().numpy()
        joint_frac = compute_joint_frac(av_frac_np, hdv_frac, n_av, n_hdv)
        joint_frac_t = torch.from_numpy(joint_frac)   # (K,)
        bpr_route_tt = route_tt_from_joint_frac(
            joint_frac, K, bpr_alpha=cfg.bpr_alpha, bpr_beta=cfg.bpr_beta
        )

        # ── travel-time proxy: route load index per group ────────────────────
        # load_g = sum_r( group_frac[r] * joint_frac[r] )
        # = weighted avg route congestion experienced by group g
        sum_t_av  += float(np.dot(av_frac_np, joint_frac))
        sum_t_hdv += float(np.dot(hdv_frac,   joint_frac))
        sum_t_all += float(np.dot(joint_frac,  joint_frac))

        z_next = wm.predict_next_state(wm_buffer, action=joint_frac_t)
        wm_buffer.add(z_next)

        # ── HDV Gawron update: let HDVs respond to current traffic load ──────
        if cfg.hdv_gawron:
            hdv_frac, hdv_cost_state = gawron_step_compatible(
                hdv_frac=hdv_frac,
                route_tt=bpr_route_tt,
                mode=imagined_gawron_mode(cfg.gawron_mode),
                lr=cfg.gawron_lr,
                alpha_zero=cfg.gawron_alpha_zero,
                beta=cfg.gawron_beta,
                update_mode=cfg.gawron_update_mode,
                min_prob=cfg.hdv_min_prob,
                cost_state=hdv_cost_state,
            )

        if cfg.use_bpr_reward:
            # BPR travel time — directly action-dependent, no decoder needed
            # t_r = 1 + alpha * (K * joint_frac[r])^beta  (BPR volume-delay)
            # reward = -AV_weighted_travel_time  (minimise AV congestion)
            reward = -float(np.dot(joint_frac, bpr_route_tt))
            reward *= float(cfg.dense_bpr_reward_weight)
        elif isinstance(reward_model, LatentRewardPredictor):
            reward = reward_model.predict(z_next)
        else:
            reward = 0.0  # fallback — BPR or LatentRewardPredictor recommended

        ppo_buf.store(obs_flat[0].cpu(), raw[0].cpu(), reward, value[0].cpu(), log_prob[0].cpu())

        total_reward += reward
        z             = z_next

    # ── 4. Bootstrap value at final state ────────────────────────────────────
    obs_last = wm.policy_obs(z).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        _, last_value = policy.forward(obs_last)   # (1, 1)

    terminal_tall_pred = float("nan")
    terminal_bonus = 0.0
    if bool(cfg.use_terminal_tall_reward):
        t_pred = wm.predict_t_all_from_latent(z)
        if t_pred is not None and np.isfinite(t_pred):
            terminal_tall_pred = float(t_pred)
            terminal_bonus = -float(cfg.terminal_tall_reward_weight) * terminal_tall_pred
            ppo_buf.rewards[cfg.episode_len - 1] += float(terminal_bonus)
            total_reward += float(terminal_bonus)

    ppo_buf.compute_gae(last_value, cfg.gamma, cfg.gae_lambda,
                        reward_mean=reward_mean, reward_std=reward_std)
    if str(cfg.critic_target_variant).lower() == "multistep_imagined":
        ppo_buf.compute_value_targets_multistep(
            last_value,
            gamma=float(cfg.gamma),
            horizon=int(cfg.critic_target_horizon),
            reward_mean=reward_mean,
            reward_std=reward_std,
        )

    hdv_drift = float(np.abs(hdv_frac - hdv_frac_init).sum())  # L1 distance from initial HDV frac

    ep_metrics = {
        "t_av":      sum_t_av  / cfg.episode_len,
        "t_hdv":     sum_t_hdv / cfg.episode_len,
        "t_all":     sum_t_all / cfg.episode_len,
        "hdv_drift": hdv_drift,
        "terminal_t_all_pred": terminal_tall_pred,
        "terminal_bonus": terminal_bonus,
    }
    return ppo_buf, total_reward, ep_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Core: imagined episode — mean field multi-agent
# ══════════════════════════════════════════════════════════════════════════════

def imagined_episode_mf(
    cfg: TrainConfig,
    wm,
    policy: MeanFieldRoutePolicy,
    route_registry: PathTopology,
    dataset: OfflineInitDataset,
    rng: np.random.Generator,
    device: torch.device,
    reward_mean: float = 0.0,
    reward_std:  float = 1.0,
    hdv_fracs_override: Optional[np.ndarray] = None,  # (N_od, K) persistent HDV state
    conservative_penalty_coef: float = 0.0,
) -> Tuple[MultiAgentPPOBuffer, float, Dict[str, float]]:
    """
    One imagined episode with mean field multi-agent routing.

    Each OD pair is an independent agent with shared policy weights.
    HDV fractions are constant within the episode (no intra-episode Gawron).
    Inter-episode HDV updates are handled by the caller.
    Reward can be global/local/difference according to cfg.reward_mode.
    """
    K     = cfg.n_routes
    N_od  = route_registry.n_od

    route_masks_np   = route_registry.route_masks                         # (N_od, K, E)
    route_masks_t    = route_registry.get_route_masks_tensor(str(device))  # (N_od, K, E)
    od_av_counts     = route_registry.od_av_counts                        # (N_od,)
    od_hdv_counts    = route_registry.od_hdv_counts                       # (N_od,)
    lengths          = wm._registry.lengths                               # (E,)
    speed_limits     = wm._registry.speed_limits                          # (E,)
    capacities       = wm._registry.capacities                            # (E,)

    # Fallback if lengths not stored in edge registry
    if lengths is None:
        lengths = np.ones(wm.n_edges, dtype=np.float32) * 100.0

    # ── 1. Sample initial state ──────────────────────────────────────────────
    flow_0 = dataset.sample(rng)  # flow_0: (E, 8)
    flow_np  = flow_0.numpy()

    # Use persistent HDV state if provided, otherwise tile from dataset
    if hdv_fracs_override is not None:
        hdv_fracs = hdv_fracs_override.copy()  # (N_od, K)
    else:
        hdv_fracs = np.ones((N_od, K), dtype=np.float32) / float(K)
    hdv_fracs_init = hdv_fracs.copy()  # for drift tracking

    # ── 2. Encode initial flow ───────────────────────────────────────────────
    z = wm.encode_observation(flow_np)  # (E, 64)

    wm_buffer = RolloutBuffer(
        max_length=cfg.history_length,
        n_spatial=wm.n_edges,
        d_spatial=wm.d_spatial,
        device=cfg.device,
    )
    wm_buffer.add(z)

    obs_dim = K * wm.d_obs + wm.d_obs
    if cfg.include_intention_obs:
        obs_dim += 2 * K
    ppo_buf = MultiAgentPPOBuffer(cfg.episode_len, N_od, obs_dim, K)

    total_reward = 0.0
    sum_t_av  = 0.0
    sum_t_hdv = 0.0
    sum_t_all = 0.0
    sum_t_e = np.zeros(wm.n_edges, dtype=np.float32)
    sum_hdv_head_entropy = 0.0
    n_hdv_head = 0
    prev_av_fracs = np.ones((N_od, K), dtype=np.float32) / float(K)

    # ── 3. Imagined rollout ──────────────────────────────────────────────────
    for _ in range(cfg.episode_len):
        # Per-OD observations
        obs = wm.local_policy_obs(
            z,
            route_masks_t,
            prev_av_fracs=prev_av_fracs if cfg.include_intention_obs else None,
            hdv_fracs=hdv_fracs if cfg.include_intention_obs else None,
        )
        obs = obs.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            av_fracs_t, raws, log_probs, _, values = policy.get_action_and_value(obs)

        av_fracs_np = av_fracs_t.cpu().numpy()  # (N_od, K)

        # Aggregate action for dynamics.
        # Internalized mode: AV-only action conditions WM; HDV response is learned inside WM.
        # Legacy mode: joint AV+HDV route fractions.
        if cfg.hdv_internalized:
            av_veh = av_fracs_np * od_av_counts[:, None]  # (N_od, K)
            total_av = float(av_veh.sum())
            if total_av < 1e-8:
                action_for_wm = np.ones(K, dtype=np.float32) / float(K)
            else:
                action_for_wm = av_veh.sum(axis=0) / total_av
        else:
            action_for_wm = aggregate_route_fracs(
                av_fracs_np, hdv_fracs, od_av_counts, od_hdv_counts,
            ).astype(np.float32)
        action_for_wm_t = torch.from_numpy(action_for_wm)

        # Dynamics: predict next state
        z_next = wm.predict_next_state(wm_buffer, action=action_for_wm_t)

        # Optional: update HDV intent from world-model hdv_policy_head.
        # Only consume it when the head is route-level (dim == K).
        if cfg.hdv_internalized:
            try:
                hdv_pred = wm.predict_hdv_policy(wm_buffer, action_for_wm_t)
            except Exception:
                hdv_pred = None
            if hdv_pred is not None and int(hdv_pred.numel()) == int(K):
                hdv_global = hdv_pred.numpy().astype(np.float32)  # (K,)
                hdv_global = np.clip(hdv_global, 1e-6, None)
                hdv_global = hdv_global / hdv_global.sum()
                hdv_fracs = np.tile(hdv_global, (N_od, 1))
                sum_hdv_head_entropy += float(-(hdv_global * np.log(hdv_global + 1e-10)).sum())
                n_hdv_head += 1
        wm_buffer.add(z_next)

        # Reward + metrics
        if cfg.hdv_internalized:
            edge_feats = wm.flow_projector.decode(z_next).detach().cpu().numpy()  # (E, n_dynamic)
            if edge_feats.shape[1] < 3:
                raise RuntimeError(
                    f"flow_projector.decode(z) expected >=3 features, got {edge_feats.shape[1]}"
                )
            v_e = edge_feats[:, 2] * np.maximum(capacities, 1.0)
            free_flow = lengths / np.maximum(speed_limits, 0.1)
            t_e = free_flow * (1.0 + cfg.bpr_alpha * (v_e / np.maximum(capacities, 1.0)) ** cfg.bpr_beta)
            total_tt = float(np.dot(np.maximum(v_e, 0.0), t_e))
            n_total = max(float(od_av_counts.sum() + od_hdv_counts.sum()), 1.0)
            reward_global = -total_tt / n_total
            reward_agents = np.full(N_od, reward_global, dtype=np.float32)
        else:
            reward_global, t_e = edge_bpr_reward(
                av_fracs_np, hdv_fracs, od_av_counts, od_hdv_counts,
                route_masks_np, lengths, speed_limits, capacities,
                cfg.bpr_alpha, cfg.bpr_beta,
            )
            route_tt = route_tt_from_edge_times(route_registry, t_e)       # (N_od, K), seconds
            reward_agents = compute_mean_field_agent_rewards(
                mode=cfg.reward_mode,
                global_reward=reward_global,
                av_fracs=av_fracs_np,
                hdv_fracs=hdv_fracs,
                route_tt=route_tt,
                od_av_counts=od_av_counts,
                od_hdv_counts=od_hdv_counts,
                route_masks=route_masks_np,
                lengths=lengths,
                speed_limits=speed_limits,
                capacities=capacities,
                bpr_alpha=cfg.bpr_alpha,
                bpr_beta=cfg.bpr_beta,
            )

        # Travel-time proxy metrics from edge-level route times
        av_veh  = av_fracs_np * od_av_counts[:, None]                   # (N_od, K)
        hdv_veh = hdv_fracs   * od_hdv_counts[:, None]                  # (N_od, K)
        all_veh = av_veh + hdv_veh
        route_tt = route_tt_from_edge_times(route_registry, t_e)        # (N_od, K), seconds
        av_tt_sec  = float((av_veh  * route_tt).sum()) / max(float(av_veh.sum()), 1.0)
        hdv_tt_sec = float((hdv_veh * route_tt).sum()) / max(float(hdv_veh.sum()), 1.0)
        all_tt_sec = float((all_veh * route_tt).sum()) / max(float(all_veh.sum()), 1.0)
        sec_to_min = 1.0 / 60.0
        sum_t_av  += av_tt_sec * sec_to_min
        sum_t_hdv += hdv_tt_sec * sec_to_min
        sum_t_all += all_tt_sec * sec_to_min
        sum_t_e += t_e.astype(np.float32)
        if conservative_penalty_coef > 0.0:
            intent_gap = np.mean(np.abs(av_fracs_np - hdv_fracs), axis=1).astype(np.float32)
            reward_agents = reward_agents - float(conservative_penalty_coef) * intent_gap
            reward_global -= float(conservative_penalty_coef) * float(intent_gap.mean())
        reward_agents = reward_agents * float(cfg.dense_bpr_reward_weight)
        reward_global = reward_global * float(cfg.dense_bpr_reward_weight)

        # HDV fractions are constant within this episode (no intra-episode Gawron).
        # Inter-episode updates are handled by the caller via avg_t_e.

        ppo_buf.store(obs, raws, reward_agents, values, log_probs)

        total_reward += reward_global
        prev_av_fracs = av_fracs_np
        z = z_next

    # ── 4. Bootstrap value at final state ────────────────────────────────────
    obs_last = wm.local_policy_obs(
        z,
        route_masks_t,
        prev_av_fracs=prev_av_fracs if cfg.include_intention_obs else None,
        hdv_fracs=hdv_fracs if cfg.include_intention_obs else None,
    ).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        _, last_values = policy.forward(obs_last)  # (N_od, 1)

    terminal_tall_pred = float("nan")
    terminal_bonus = 0.0
    if bool(cfg.use_terminal_tall_reward):
        t_pred = wm.predict_t_all_from_latent(z)
        if t_pred is not None and np.isfinite(t_pred):
            terminal_tall_pred = float(t_pred)
            terminal_bonus = -float(cfg.terminal_tall_reward_weight) * terminal_tall_pred
            ppo_buf.rewards[cfg.episode_len - 1] += float(terminal_bonus)
            total_reward += float(terminal_bonus)

    ppo_buf.compute_gae(last_values, cfg.gamma, cfg.gae_lambda,
                        reward_mean=reward_mean, reward_std=reward_std)
    if str(cfg.critic_target_variant).lower() == "multistep_imagined":
        ppo_buf.compute_value_targets_multistep(
            last_values,
            gamma=float(cfg.gamma),
            horizon=int(cfg.critic_target_horizon),
            reward_mean=reward_mean,
            reward_std=reward_std,
        )

    # Average edge travel times across the episode (for inter-episode HDV update)
    avg_t_e = sum_t_e / max(cfg.episode_len, 1)

    hdv_drift = float(np.abs(hdv_fracs - hdv_fracs_init).sum())

    ep_metrics = {
        "t_av":      sum_t_av  / cfg.episode_len,
        "t_hdv":     sum_t_hdv / cfg.episode_len,
        "t_all":     sum_t_all / cfg.episode_len,
        "hdv_drift": hdv_drift,
        "avg_t_e":   avg_t_e,   # (E,) for inter-episode HDV update
        "hdv_head_entropy": (sum_hdv_head_entropy / max(1, n_hdv_head)),
        "terminal_t_all_pred": terminal_tall_pred,
        "terminal_bonus": terminal_bonus,
    }
    return ppo_buf, total_reward, ep_metrics


def imagined_episode_vehicle(
    cfg: TrainConfig,
    wm,
    policy: MeanFieldRoutePolicy,
    route_registry: PathTopology,
    dataset: OfflineInitDataset,
    rng: np.random.Generator,
    device: torch.device,
    reward_mean: float = 0.0,
    reward_std: float = 1.0,
    path_cache: Optional[Union[VehiclePathCache, LazyVehiclePathCache]] = None,
) -> Tuple[MultiAgentPPOBuffer, float, Dict[str, float]]:
    """
    Vehicle-level imagined rollout.

    - Agent: each AV vehicle.
    - Action: choose one of K candidate paths from current edge to destination, then execute one next-hop edge.
    - World-model action: aggregated AV edge-visit vector (E,) for current step.
    - Reward: per-vehicle marginal delay contribution on traversed edge.
    """
    K = int(cfg.n_routes)
    n_av = int(max(1, cfg.n_av))
    if wm._registry.adjacency is None:
        raise RuntimeError("Vehicle mode requires edge adjacency in edge_registry.")
    wm_action_dim = int(getattr(wm, "action_dim", -1))
    comp_in = int(getattr(wm, "action_compressor_input_dim", -1) or -1)
    comp_out = int(getattr(wm, "action_compressor_output_dim", -1) or -1)
    action_ok = (wm_action_dim == int(wm.n_edges)) or (
        comp_in == int(wm.n_edges) and comp_out == wm_action_dim
    )
    if not action_ok:
        raise RuntimeError(
            "Vehicle mode requires raw edge action support. "
            f"Expected action_dim==n_edges ({wm.n_edges}) or compressor {wm.n_edges}->action_dim, "
            f"got action_dim={wm_action_dim}, compressor_in={comp_in}, compressor_out={comp_out}."
        )

    # sample initial latent state
    flow_0 = dataset.sample_at_step(rng, s)
    z = wm.encode_observation(flow_0.numpy())  # (E, 64)

    wm_buffer = RolloutBuffer(
        max_length=cfg.history_length,
        n_spatial=wm.n_edges,
        d_spatial=wm.d_spatial,
        device=cfg.device,
    )
    wm_buffer.add(z)

    # build per-vehicle route intent state
    edge_to_idx = wm._registry._edge_to_idx
    _, current_edges, dest_edges = build_vehicle_edge_states(
        route_registry=route_registry,
        edge_to_idx=edge_to_idx,
        n_av=n_av,
        rng=rng,
    )
    done = (current_edges == dest_edges)
    capacities = np.asarray(wm._registry.capacities, dtype=np.float32)
    adj_list = PathTopology._build_adj_list(wm._registry.adjacency)

    obs_dim = vehicle_mode_obs_dim(
        n_routes=K,
        d_obs=wm.d_obs,
        include_global=bool(cfg.vehicle_obs_include_global),
        route_stats=str(cfg.vehicle_obs_route_stats),
        include_route_costs=bool(cfg.vehicle_obs_include_route_costs),
    ) - 3
    ppo_buf = MultiAgentPPOBuffer(cfg.episode_len, n_av, obs_dim, K)

    # HDV equilibrium: compute once per episode (HDVs choose routes at start).
    # Condition on the episode's initial AV edge occupancy instead of a uniform prior.
    active_mask = ~done
    if np.any(active_mask):
        av_edge_init = np.bincount(current_edges[active_mask], minlength=wm.n_edges).astype(np.float32)
    else:
        av_edge_init = np.bincount(current_edges, minlength=wm.n_edges).astype(np.float32)
    av_edge_init_sum = float(av_edge_init.sum())
    if av_edge_init_sum > 0.0:
        av_edge_init /= av_edge_init_sum
    else:
        av_edge_init.fill(1.0 / float(wm.n_edges))
    hdv_edge_fixed = wm.compute_hdv_equilibrium(
        av_action=av_edge_init,
        z_current=z,
    )  # (E,) normalized or None

    total_reward = 0.0
    sum_t_av = 0.0
    sum_t_hdv = 0.0
    sum_t_all = 0.0

    candidate_edge_cost_source = resolve_wm_candidate_edge_cost_source(cfg)

    for _ in range(cfg.episode_len):
        edge_dyn, _, free_flow, t_e = edge_tt_from_latent_state(
            wm=wm,
            z=z,
            bpr_alpha=cfg.bpr_alpha,
            bpr_beta=cfg.bpr_beta,
            edge_cost_source=candidate_edge_cost_source,
            decoded_speed_min_ratio=cfg.wm_decoded_speed_min_ratio,
            decoded_speed_max_ratio=cfg.wm_decoded_speed_max_ratio,
            decoded_speed_excess_scale=cfg.wm_decoded_speed_excess_scale,
        )

        # Per-vehicle candidate path lookup using either a dense precomputed cache or
        # the lazy on-demand cache used on larger networks.
        if path_cache is not None:
            if isinstance(path_cache, LazyVehiclePathCache):
                path_batch = path_cache.lookup_batch(current_edges, dest_edges, done, t_e)
            else:
                path_batch = path_cache.lookup_batch(current_edges, dest_edges, done)
        else:
            path_batch: List[List[List[int]]] = []
            for i in range(n_av):
                cur = int(current_edges[i])
                dst = int(dest_edges[i])
                if bool(done[i]):
                    path_batch.append([[cur] for _k in range(K)])
                else:
                    pths = build_diverse_paths_current_to_dest(
                        adj_list=adj_list,
                        edge_tt=t_e,
                        src_idx=cur,
                        dst_idx=dst,
                        K=K,
                        penalty_factor=cfg.vehicle_path_penalty_factor,
                    )
                    path_batch.append(pths)

        edge_feats_obs = wm.latent_edge_features(z).detach().cpu().numpy().astype(np.float32)
        obs_np = build_vehicle_obs_vectorized(
            edge_feats_dyn=edge_feats_obs,
            path_batch=path_batch,
            include_global=bool(cfg.vehicle_obs_include_global),
            route_stats=str(cfg.vehicle_obs_route_stats),
            edge_cost=t_e,
            include_route_costs=bool(cfg.vehicle_obs_include_route_costs),
        )
        obs = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            av_fracs_t, raws, log_probs, _, values = policy.get_action_and_value(obs)
        av_fracs_np = av_fracs_t.detach().cpu().numpy().astype(np.float32)

        # Vectorized route sampling + next-edge selection
        active_mask = ~done  # (n_av,)
        probs_all = np.clip(av_fracs_np, 1e-8, None)
        probs_all = probs_all / probs_all.sum(axis=1, keepdims=True)
        # Sample route index for each vehicle in one shot
        uniform = rng.random((n_av,))
        cum_probs = np.cumsum(probs_all, axis=1)
        act_indices = (uniform[:, None] >= cum_probs).sum(axis=1).astype(np.int64)
        act_indices = np.clip(act_indices, 0, K - 1)

        chosen_edges = np.full((n_av,), -1, dtype=np.int64)
        for i in range(n_av):
            if not active_mask[i]:
                continue
            chosen_path = path_batch[i][int(act_indices[i])]
            nxt = choose_vehicle_next_edge(int(current_edges[i]), chosen_path)
            chosen_edges[i] = int(nxt)

        # Aggregate the chosen route plans, not only the immediate next-hop
        # edges, so imagined actions keep the same route-intent semantics as
        # the world-model planner/evaluator.
        sampled_route_probs = np.zeros_like(probs_all, dtype=np.float32)
        for i in range(n_av):
            if active_mask[i]:
                sampled_route_probs[i, int(act_indices[i])] = 1.0
        action_edge = aggregate_vehicle_actions_to_edge_distribution(
            av_vehicle_fracs=sampled_route_probs,
            path_batch=path_batch,
            n_edges=wm.n_edges,
            n_routes=K,
            current_edges=current_edges,
            edge_travel_times=t_e,
            horizon_seconds=float(cfg.step_interval),
        )

        # Combine AV action with pre-computed HDV equilibrium using real counts.
        if hdv_edge_fixed is not None:
            hdv_np = hdv_edge_fixed.cpu().numpy().astype(np.float32)
            if wm.stackelberg is None:
                raise RuntimeError("hdv_edge_fixed is present but wm.stackelberg is unavailable.")
            av_total = float(wm.stackelberg.total_av_count.detach().cpu().item())
            hdv_total = float(wm.stackelberg.total_hdv_count.detach().cpu().item())
            joint_action = action_edge * av_total + hdv_np * hdv_total
            _s = float(joint_action.sum())
            if _s > 1e-8:
                joint_action /= _s
            z_next = wm.predict_next_state(
                wm_buffer, action=torch.from_numpy(joint_action),
                skip_stackelberg=True,
            )
        else:
            z_next = wm.predict_next_state(
                wm_buffer, action=torch.from_numpy(action_edge),
            )
        wm_buffer.add(z_next)

        # ── BPR reward from WM-predicted next state ─────────────────────
        # Decode z_next to get v_e directly from world model prediction.
        edge_dyn_next = wm.flow_projector.decode(z_next).detach().cpu().numpy().astype(np.float32)
        v_total = np.maximum(edge_dyn_next[:, 2], 0.0) * capacities    # (E,) vehicle counts

        # Use shared BPRLayer for travel time computation
        t_e_a = wm.bpr.numpy_forward(v_total)
        free_flow_a = wm.bpr.free_flow_time.detach().cpu().numpy().astype(np.float32)
        _alpha = float(wm.bpr.alpha.detach().cpu())
        _beta = float(wm.bpr.beta.detach().cpu())

        total_tt = float(np.dot(v_total, t_e_a))
        n_total_veh = max(float(v_total.sum()), 1.0)
        reward_global = -total_tt / n_total_veh

        # Per-vehicle reward: marginal contribution on chosen edge
        rewards = np.zeros((n_av,), dtype=np.float32)
        av_t_edges: List[float] = []
        for i in range(n_av):
            e = int(chosen_edges[i])
            if e < 0:
                continue
            # Counterfactual: travel time if this vehicle had NOT chosen edge e
            v_cf = max(v_total[e] - 1.0, 0.0)
            t_cf = float(free_flow_a[e]) * (
                1.0 + _alpha * (v_cf / max(float(capacities[e]), 1.0)) ** _beta
            )
            marginal_delay = float(t_e_a[e]) - t_cf
            rewards[i] = -float(cfg.vehicle_reward_scale) * marginal_delay
            current_edges[i] = e
            if int(current_edges[i]) == int(dest_edges[i]):
                done[i] = True
            av_t_edges.append(float(t_e_a[e]))

        rewards *= float(cfg.dense_bpr_reward_weight)
        reward_global *= float(cfg.dense_bpr_reward_weight)

        t_all_step_min = (total_tt / n_total_veh) / 60.0
        t_av_step_min = (float(np.mean(av_t_edges)) / 60.0) if av_t_edges else t_all_step_min
        sum_t_av += t_av_step_min
        sum_t_hdv += t_all_step_min
        sum_t_all += t_all_step_min

        ppo_buf.store(obs, raws, rewards, values, log_probs)
        total_reward += reward_global
        z = z_next

    # bootstrap
    edge_dyn_last, _, _, t_e_last = edge_tt_from_latent_state(
        wm=wm,
        z=z,
        bpr_alpha=cfg.bpr_alpha,
        bpr_beta=cfg.bpr_beta,
        edge_cost_source=candidate_edge_cost_source,
        decoded_speed_min_ratio=cfg.wm_decoded_speed_min_ratio,
        decoded_speed_max_ratio=cfg.wm_decoded_speed_max_ratio,
        decoded_speed_excess_scale=cfg.wm_decoded_speed_excess_scale,
    )
    if path_cache is not None:
        if isinstance(path_cache, LazyVehiclePathCache):
            path_batch_last = path_cache.lookup_batch(current_edges, dest_edges, done, t_e_last)
        else:
            path_batch_last = path_cache.lookup_batch(current_edges, dest_edges, done)
    else:
        path_batch_last = []
        for i in range(n_av):
            cur = int(current_edges[i])
            dst = int(dest_edges[i])
            if bool(done[i]):
                path_batch_last.append([[cur] for _k in range(K)])
            else:
                path_batch_last.append(
                    build_diverse_paths_current_to_dest(
                        adj_list=adj_list,
                        edge_tt=t_e_last,
                        src_idx=cur,
                        dst_idx=dst,
                        K=K,
                        penalty_factor=cfg.vehicle_path_penalty_factor,
                    )
                )
    edge_feats_obs_last = wm.latent_edge_features(z).detach().cpu().numpy().astype(np.float32)
    obs_last_np = build_vehicle_obs_vectorized(
        edge_feats_dyn=edge_feats_obs_last,
        path_batch=path_batch_last,
        include_global=bool(cfg.vehicle_obs_include_global),
        route_stats=str(cfg.vehicle_obs_route_stats),
        edge_cost=t_e_last,
        include_route_costs=bool(cfg.vehicle_obs_include_route_costs),
    )
    obs_last = torch.from_numpy(obs_last_np).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        _, last_values = policy.forward(obs_last)

    terminal_tall_pred = float("nan")
    terminal_bonus = 0.0
    if bool(cfg.use_terminal_tall_reward):
        t_pred = wm.predict_t_all_from_latent(z)
        if t_pred is not None and np.isfinite(t_pred):
            terminal_tall_pred = float(t_pred)
            terminal_bonus = -float(cfg.terminal_tall_reward_weight) * terminal_tall_pred
            ppo_buf.rewards[cfg.episode_len - 1] += float(terminal_bonus)
            total_reward += float(terminal_bonus)

    ppo_buf.compute_gae(
        last_values,
        cfg.gamma,
        cfg.gae_lambda,
        reward_mean=reward_mean,
        reward_std=reward_std,
    )
    if str(cfg.critic_target_variant).lower() == "multistep_imagined":
        ppo_buf.compute_value_targets_multistep(
            last_values,
            gamma=float(cfg.gamma),
            horizon=int(cfg.critic_target_horizon),
            reward_mean=reward_mean,
            reward_std=reward_std,
        )

    ep_metrics = {
        "t_av": sum_t_av / max(1, cfg.episode_len),
        "t_hdv": sum_t_hdv / max(1, cfg.episode_len),
        "t_all": sum_t_all / max(1, cfg.episode_len),
        "hdv_drift": 0.0,
        "terminal_t_all_pred": terminal_tall_pred,
        "terminal_bonus": terminal_bonus,
        "hdv_head_entropy": 0.0,
    }
    return ppo_buf, total_reward, ep_metrics
def select_imagination_mode(mean_field: bool, agent_level: str) -> str:
    level = str(agent_level).lower()
    if level not in {"vehicle"}:
        raise ValueError(f"Unknown agent_level={agent_level!r}; expected 'vehicle'.")
    if mean_field and level == "vehicle":
        return "vehicle_dynamic"
    return "single_agent"


def vehicle_mode_action_supported(
    *,
    n_edges: int,
    action_dim: int,
    raw_action_dim: int,
    action_compressor_input_dim: int,
    action_compressor_output_dim: int,
) -> bool:
    if int(action_dim) == int(n_edges):
        return True
    if int(raw_action_dim) == int(n_edges):
        return True
    if int(action_compressor_input_dim) == int(n_edges) and int(action_compressor_output_dim) > 0:
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



def imagined_episode_dynamic(
    wm: UrbanWorldModel,
    policy: nn.Module,
    dataset: OfflineInitDataset,
    route_registry: PathTopology,
    cfg: TrainConfig,
    rng: np.random.Generator,
    device: str = "cuda",
    departure_schedule: Optional[Dict] = None,
    avg_route_steps: int = 7,
    avg_edge_time: float = 8.0,
    path_cache: Optional[Union[VehiclePathCache, LazyVehiclePathCache]] = None,
    reward_mean: float = 0.0,
    reward_std: float = 1.0,
    start_step_candidates: Optional[List[int]] = None,
) -> Tuple[MultiAgentPPOBuffer, float, Dict[str, float]]:
    """
    Dynamic imagined rollout with async vehicle departures.

    Key differences from imagined_episode_vehicle:
      - Fixed time steps (step_interval seconds per step, episode_steps total)
      - Vehicles depart at different times based on departure_schedule
      - Each active AV re-routes every step
      - Random start for imagination (start from random step s, imagine T steps)
      - Vehicle positions tracked analytically (not via WM)

    Args:
        wm: Urban world model.
        policy: MeanFieldRoutePolicy (vehicle-level).
        dataset: Offline flow dataset for initial state sampling.
        route_registry: Pre-built route registry with OD/route info.
        cfg: Training config.
        rng: Random number generator.
        device: Torch device string.
        departure_schedule: Pre-built schedule from build_departure_schedule().
        avg_route_steps: Estimated number of steps for average route completion.
        avg_edge_time: Average edge traversal time in seconds.
        path_cache: Pre-built VehiclePathCache for fast route lookup.
        reward_mean: Running mean for reward normalization.
        reward_std: Running std for reward normalization.

    Returns:
        (ppo_buf, total_reward, ep_metrics)
    """
    T = int(cfg.imagination_horizon)
    episode_steps = int(cfg.episode_steps)
    K = int(cfg.n_routes)
    n_edges = int(wm.n_edges)
    step_interval = float(cfg.step_interval)

    if departure_schedule is None:
        departure_schedule = {}

    # ── Sample random starting step ──────────────────────────────────────
    if start_step_candidates:
        s = int(start_step_candidates[int(rng.integers(0, len(start_step_candidates)))])
    else:
        s = int(rng.integers(0, max(1, episode_steps - T)))

    # ── Get initial state from training data ─────────────────────────────
    flow_0 = dataset.sample(rng)
    z = wm.encode_observation(flow_0.numpy())  # (E, 64)

    wm_buffer = RolloutBuffer(
        max_length=cfg.history_length,
        n_spatial=n_edges,
        d_spatial=wm.d_spatial,
        device=device,
    )
    wm_buffer.add(z)

    # ── Rebuild active vehicle set at step s ──────────────────────────────
    active_avs, active_hdvs = rebuild_active_set(departure_schedule, s, avg_route_steps)

    # ── Pre-build helpers ─────────────────────────────────────────────────
    adj_list = PathTopology._build_adj_list(wm._registry.adjacency)
    edge_to_idx = wm._registry._edge_to_idx
    capacities = np.asarray(wm._registry.capacities, dtype=np.float32)
    candidate_edge_cost_source = resolve_wm_candidate_edge_cost_source(cfg)

    use_od_suffix_candidates = (
        str(getattr(cfg, "vehicle_candidate_source", "current_to_dest")).strip().lower() == "od_suffix"
    )

    # Build path cache on the fly if not provided.  The od_suffix mode uses the
    # precomputed OD candidate routes directly, so building a current-to-dest
    # Dijkstra cache here is both unused and very expensive on large networks.
    if path_cache is None and not use_od_suffix_candidates:
        _, _, _, t_e_ref = edge_tt_from_latent_state(
            wm=wm, z=z, bpr_alpha=cfg.bpr_alpha, bpr_beta=cfg.bpr_beta,
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

    # ── PPO storage: variable n_agents per step ──────────────────────────
    all_obs: List[torch.Tensor] = []
    all_critic_obs: List[torch.Tensor] = []
    all_raws: List[torch.Tensor] = []
    all_log_probs: List[torch.Tensor] = []
    all_values: List[torch.Tensor] = []
    all_rewards: List[torch.Tensor] = []

    total_reward = 0.0
    sum_t_all = 0.0
    sum_t_av = 0.0

    # Precompute total vehicle counts for normalization
    n_total_av = max(sum(d.get("n_av", 0) for d in departure_schedule.values()), 1)
    n_total_hdv = max(sum(d.get("n_hdv", 0) for d in departure_schedule.values()), 1)

    for step in range(s, s + T):
        # ── Departures ────────────────────────────────────────────────
        dep = departure_schedule.get(step, {
            "depart_edge": np.zeros(n_edges, dtype=np.float32),
            "av_vehicles": [],
            "hdv_vehicles": [],
            "n_av": 0,
            "n_hdv": 0,
        })
        for veh in dep.get("av_vehicles", []):
            active_avs.append({**veh, "depart_step": step, "current_edge": int(veh.get("origin_edge", 0))})
        for veh in dep.get("hdv_vehicles", []):
            active_hdvs.append({**veh, "depart_step": step})

        # ── Arrival detection (remove vehicles past expected duration) ─
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

        depart_np = dep.get("depart_edge", np.zeros(n_edges, dtype=np.float32))

        if n_active_av == 0:
            # No active AVs — still advance WM with zero AV action
            context_np = np.array([
                step / max(episode_steps, 1),
                0.0,
                n_active_hdv / max(n_total_hdv, 1),
            ], dtype=np.float32)
            action_edge = np.zeros(n_edges, dtype=np.float32)
            z = wm.predict_next_state(
                wm_buffer,
                action=torch.from_numpy(action_edge),
                depart=torch.from_numpy(depart_np),
                context=torch.from_numpy(context_np),
                skip_stackelberg=True,
            )
            wm_buffer.add(z)
            continue

        # ── Build per-vehicle observations ────────────────────────────
        edge_feats_obs = wm.latent_edge_features(z).detach().cpu().numpy().astype(np.float32)
        t_e_curr: Optional[np.ndarray] = None
        _, _, _, t_e_curr = edge_tt_from_latent_state(
            wm=wm,
            z=z,
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
        done_np = np.zeros((n_active_av,), dtype=bool)
        if path_cache is not None:
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

        obs_np = build_vehicle_obs_vectorized(
            edge_feats_dyn=edge_feats_obs,
            path_batch=path_batch,
            include_global=bool(cfg.vehicle_obs_include_global),
            route_stats=str(cfg.vehicle_obs_route_stats),
            edge_cost=t_e_curr,
            include_route_costs=bool(cfg.vehicle_obs_include_route_costs),
        )

        # Append context features: time_frac, active_av_frac, progress_frac
        first_depart = active_avs[0]["depart_step"] if active_avs else step
        context_features = np.array([
            step / max(episode_steps, 1),
            n_active_av / n_total_av,
            (step - first_depart) / max(avg_route_steps, 1),
        ], dtype=np.float32)
        ctx_tiled = np.tile(context_features, (n_active_av, 1))
        obs_np = np.concatenate([obs_np, ctx_tiled], axis=1)

        obs = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
        critic_obs = obs
        if str(cfg.critic_variant).lower() == "latent_global":
            latent_summary_np = (
                latent_global_summary(
                    z.detach().to(device=device, dtype=torch.float32),
                    variant=str(cfg.critic_global_summary_variant),
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            critic_obs_np = append_global_summary_to_obs_np(obs_np, latent_summary_np)
            critic_obs = torch.from_numpy(critic_obs_np).to(device=device, dtype=torch.float32)

        # ── Policy forward ────────────────────────────────────────────
        with torch.no_grad():
            av_fracs_t, raws, log_probs, _, values = policy.get_action_and_value(
                obs,
                critic_obs=critic_obs,
            )
        av_fracs_np = av_fracs_t.detach().cpu().numpy().astype(np.float32)

        # ── Aggregate to edge action ──────────────────────────────────
        action_edge = aggregate_vehicle_actions_to_edge_distribution(
            av_vehicle_fracs=av_fracs_np,
            path_batch=path_batch,
            n_edges=n_edges,
            n_routes=K,
            current_edges=current_edges_np,
            edge_travel_times=t_e_curr,
            horizon_seconds=float(cfg.step_interval),
        )

        # ── WM predict ────────────────────────────────────────────────
        context_np = np.array([
            step / max(episode_steps, 1),
            n_active_av / n_total_av,
            n_active_hdv / max(n_total_hdv, 1),
        ], dtype=np.float32)

        z_next = wm.predict_next_state(
            wm_buffer,
            action=torch.from_numpy(action_edge),
            depart=torch.from_numpy(depart_np),
            context=torch.from_numpy(context_np),
            skip_stackelberg=True,
        )
        wm_buffer.add(z_next)

        # ── Reward from WM-predicted state ────────────────────────────
        edge_dyn_next = wm.flow_projector.decode(z_next).detach().cpu().numpy().astype(np.float32)
        v_total = np.maximum(edge_dyn_next[:, 2], 0.0) * capacities
        t_e = wm.bpr.numpy_forward(v_total)
        total_tt = float(np.dot(v_total, t_e))
        n_veh = max(float(v_total.sum()), 1.0)
        step_reward = -total_tt / n_veh

        # Travel time metrics (minutes)
        t_all_step_min = (total_tt / n_veh) / 60.0
        sum_t_all += t_all_step_min
        valid_current = (current_edges_np >= 0) & (current_edges_np < int(t_e.shape[0]))
        current_edge_tt = t_e[current_edges_np[valid_current]] if np.any(valid_current) else np.asarray([], dtype=np.float32)
        if current_edge_tt.size > 0:
            t_av_step_min = float(np.mean(current_edge_tt) / 60.0)
        else:
            t_av_step_min = t_all_step_min
        sum_t_av += t_av_step_min

        # ── Store PPO data ────────────────────────────────────────────
        if float(getattr(cfg, "dreamer_current_edge_reward_weight", 0.0)) != 0.0 and current_edge_tt.size > 0:
            reward_per_agent = np.full(n_active_av, 0.0, dtype=np.float32)
            reward_per_agent[valid_current] = (
                -float(cfg.vehicle_reward_scale)
                * float(cfg.dreamer_current_edge_reward_weight)
                * t_e[current_edges_np[valid_current]].astype(np.float32)
            )
            if float(getattr(cfg, "dreamer_all_reward_weight", 0.0)) != 0.0:
                reward_per_agent += float(cfg.dreamer_all_reward_weight) * float(step_reward)
        else:
            reward_per_agent = np.full(n_active_av, step_reward, dtype=np.float32)
        total_reward += float(np.mean(reward_per_agent)) if reward_per_agent.size > 0 else float(step_reward)
        all_obs.append(obs.detach().cpu())
        all_critic_obs.append(critic_obs.detach().cpu())
        all_raws.append(raws.detach().cpu())
        all_log_probs.append(log_probs.detach().cpu())
        all_values.append(values.detach().cpu())
        all_rewards.append(torch.from_numpy(reward_per_agent))

        z = z_next

    # ── Build PPO buffer ──────────────────────────────────────────────────
    n_stored_steps = len(all_obs)
    if n_stored_steps == 0:
        # Edge case: no active AVs in any step
        obs_dim = vehicle_mode_obs_dim(
            n_routes=K,
            d_obs=wm.d_obs,
            include_global=bool(cfg.vehicle_obs_include_global),
            route_stats=str(cfg.vehicle_obs_route_stats),
            include_route_costs=bool(cfg.vehicle_obs_include_route_costs),
        )
        critic_obs_dim = obs_dim
        if str(cfg.critic_variant).lower() == "latent_global":
            critic_obs_dim = obs_dim + latent_global_summary_dim(
                wm.d_spatial,
                variant=str(cfg.critic_global_summary_variant),
            )
        buf = MultiAgentPPOBuffer(1, 1, obs_dim, K, critic_obs_dim=critic_obs_dim)
        ep_metrics = {
            "t_av": 0.0, "t_hdv": 0.0, "t_all": 0.0,
            "hdv_drift": 0.0, "n_steps": 0, "avg_active_av": 0.0,
            "start_step": s,
        }
        return buf, 0.0, ep_metrics

    # Use max agents across steps as N_od (buffer pads smaller steps)
    max_agents = max(o.shape[0] for o in all_obs)
    obs_dim = all_obs[0].shape[1]
    critic_obs_dim = all_critic_obs[0].shape[1]
    buf = MultiAgentPPOBuffer(
        episode_len=n_stored_steps,
        n_od=max_agents,
        obs_dim=obs_dim,
        critic_obs_dim=critic_obs_dim,
        n_routes=K,
    )

    for t_idx in range(n_stored_steps):
        n_a = all_obs[t_idx].shape[0]
        # Pad to max_agents if needed
        if n_a < max_agents:
            pad_obs = torch.zeros(max_agents, obs_dim)
            pad_obs[:n_a] = all_obs[t_idx]
            pad_critic_obs = torch.zeros(max_agents, critic_obs_dim)
            pad_critic_obs[:n_a] = all_critic_obs[t_idx]
            pad_raws = torch.zeros(max_agents, K)
            pad_raws[:n_a] = all_raws[t_idx]
            pad_lp = torch.zeros(max_agents)
            pad_lp[:n_a] = all_log_probs[t_idx]
            pad_vals = torch.zeros(max_agents)
            pad_vals[:n_a] = all_values[t_idx].squeeze(-1)
            pad_rew = np.zeros(max_agents, dtype=np.float32)
            pad_rew[:n_a] = all_rewards[t_idx].numpy()
        else:
            pad_obs = all_obs[t_idx]
            pad_critic_obs = all_critic_obs[t_idx]
            pad_raws = all_raws[t_idx]
            pad_lp = all_log_probs[t_idx]
            pad_vals = all_values[t_idx]
            pad_rew = all_rewards[t_idx].numpy()

        buf.store(
            obs=pad_obs,
            critic_obs=pad_critic_obs,
            raw_actions=pad_raws,
            reward=pad_rew,
            values=pad_vals,
            logprobs=pad_lp,
            active_mask=np.arange(max_agents) < n_a,
        )

    # Bootstrap value from last state
    last_vals = all_values[-1]
    last_vals_flat = last_vals.squeeze(-1).reshape(-1)
    if last_vals_flat.shape[0] < max_agents:
        padded_last = torch.zeros(max_agents)
        padded_last[: last_vals_flat.shape[0]] = last_vals_flat
        last_vals = padded_last
    else:
        last_vals = last_vals_flat
    buf.compute_gae(
        last_values=last_vals,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        reward_mean=reward_mean,
        reward_std=reward_std,
    )
    if str(cfg.critic_target_variant).lower() == "multistep_imagined":
        buf.compute_value_targets_multistep(
            last_vals,
            gamma=float(cfg.gamma),
            horizon=int(cfg.critic_target_horizon),
            reward_mean=reward_mean,
            reward_std=reward_std,
        )

    avg_active = float(np.mean([o.shape[0] for o in all_obs]))
    ep_metrics = {
        "t_av": sum_t_av / max(1, n_stored_steps),
        "t_hdv": sum_t_all / max(1, n_stored_steps),
        "t_all": sum_t_all / max(1, n_stored_steps),
        "hdv_drift": 0.0,
        "n_steps": n_stored_steps,
        "avg_active_av": avg_active,
        "start_step": s,
    }
    return buf, total_reward / max(1, n_stored_steps), ep_metrics


def ppo_update(
    cfg: TrainConfig,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],   # 合并后的 flat batch（支持多 episode）
    device: torch.device,
) -> Dict[str, float]:
    N = batch["obs"].shape[0]

    obs        = batch["obs"].to(device)
    critic_obs = batch.get("critic_obs", batch["obs"]).to(device)
    actions    = batch["actions"].to(device)
    old_lp     = batch["logprobs"].to(device)
    advantages = batch["advantages"].to(device)
    returns    = batch["returns"].to(device)
    value_targets = batch.get("value_targets", returns).to(device)

    adv_std = advantages.std(unbiased=False).clamp_min(1e-8)
    advantages = (advantages - advantages.mean()) / adv_std

    # 归一化 returns → value_loss 量级 O(1)，避免被 vf 梯度淹没 pg 梯度
    ret_mean    = value_targets.mean()
    ret_std     = value_targets.std(unbiased=False).clamp_min(1e-8)
    returns_norm = (value_targets - ret_mean) / ret_std

    stats = {"pg_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    n_updates = 0

    for _ in range(cfg.n_ppo_epochs):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, cfg.minibatch_size):
            idx = perm[start : start + cfg.minibatch_size]

            # actions[idx]: (mb, K) raw logit 样本；传入 raw_action 供 log_prob 重算
            _, _, new_lp, ent, new_val = policy.get_action_and_value(
                obs[idx],
                raw_action=actions[idx],
                critic_obs=critic_obs[idx],
            )

            ratio   = (new_lp - old_lp[idx]).exp()
            adv     = advantages[idx]
            pg_loss = -torch.min(
                ratio * adv,
                ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv,
            ).mean()

            # Returns 归一化后再计算 vf_loss，消除绝对尺度导致的发散
            v = new_val.squeeze(-1)
            value_loss = (v - returns_norm[idx]).pow(2).mean()
            loss       = pg_loss + cfg.value_coef * value_loss - cfg.entropy_coef * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_lp[idx] - new_lp).mean().item()

            stats["pg_loss"]    += pg_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"]    += ent.item()
            stats["approx_kl"]  += approx_kl
            n_updates += 1

    for k in stats:
        stats[k] /= max(1, n_updates)
    return stats


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


def _vehicle_potential_progress_terms_from_paths(
    *,
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    edge_cost_current: torch.Tensor,
    edge_cost_next: torch.Tensor,
    n_routes: int,
    invalid_route_cost_current: torch.Tensor,
    invalid_route_cost_next: torch.Tensor,
    temperature: float,
    step_seconds: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Vehicle-level potential terms for progress reward.

    Current potential is a soft shortest-route cost from the vehicle's current
    edge to its destination.  Next potential advances each candidate route by
    one decision-step time budget, then takes the actor-distribution-weighted
    remaining suffix cost.  This makes route choices affect the progress signal
    through the vehicle's predicted position, not only through congestion-field
    changes in the world model.
    """
    if route_probs.dim() != 2:
        raise ValueError(f"route_probs must be 2D, got {tuple(route_probs.shape)}")
    if edge_cost_current.dim() != 1 or edge_cost_next.dim() != 1:
        raise ValueError("edge costs must be 1D tensors")

    current_costs, route_valid = _route_cost_matrix_from_paths(
        path_batch=path_batch,
        edge_cost=edge_cost_current,
        n_routes=n_routes,
        invalid_route_cost=invalid_route_cost_current,
    )
    if current_costs is None or route_valid is None:
        return None, None

    device = route_probs.device
    dtype = route_probs.dtype
    current_costs = current_costs.to(device=device, dtype=dtype)
    route_valid = route_valid.to(device=device)
    fallback_cost = invalid_route_cost_current.to(device=device, dtype=dtype)
    valid_any = route_valid.any(dim=1)
    if not bool(valid_any.any().detach().cpu().item()):
        return None, None

    large_cost = fallback_cost + current_costs.detach().abs().max().clamp_min(1.0) * 1000.0
    masked_current = torch.where(route_valid, current_costs, large_cost)
    temp = float(temperature)
    if temp <= 1e-8:
        current_potential = masked_current.min(dim=1).values
    else:
        tau = torch.as_tensor(temp, device=device, dtype=dtype)
        current_potential = -tau * torch.logsumexp(-masked_current / tau, dim=1)

    suffix_costs = _advanced_suffix_cost_matrix_from_paths(
        path_batch=path_batch,
        edge_cost_current=edge_cost_current,
        edge_cost_next=edge_cost_next,
        n_routes=n_routes,
        invalid_route_cost=invalid_route_cost_next,
        step_seconds=step_seconds,
    )
    if suffix_costs is None:
        return None, None
    suffix_costs = suffix_costs.to(device=device, dtype=dtype)

    k_eff = min(int(route_probs.shape[1]), int(suffix_costs.shape[1]))
    probs = route_probs[: int(suffix_costs.shape[0]), :k_eff].to(device=device, dtype=dtype)
    valid = route_valid[: int(suffix_costs.shape[0]), :k_eff]
    probs = torch.where(valid, torch.relu(probs), torch.zeros_like(probs))
    prob_sum = probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    next_potential = torch.sum((probs / prob_sum) * suffix_costs[:, :k_eff], dim=1)

    mask = valid_any[: int(next_potential.shape[0])]
    return current_potential[: int(next_potential.shape[0])][mask].mean(), next_potential[mask].mean()


def _advanced_suffix_cost_matrix_from_paths(
    *,
    path_batch: List[List[List[int]]],
    edge_cost_current: torch.Tensor,
    edge_cost_next: torch.Tensor,
    n_routes: int,
    invalid_route_cost: torch.Tensor,
    step_seconds: float,
) -> Optional[torch.Tensor]:
    """Return route suffix costs after moving along each candidate path.

    The suffix start is chosen with detached current edge travel times because
    it is a discrete vehicle-position update.  The returned costs still depend
    on ``edge_cost_next``, so gradients flow through the next-state travel-time
    prediction.
    """
    if edge_cost_next.dim() != 1:
        raise ValueError(f"edge_cost_next must be 1D, got {tuple(edge_cost_next.shape)}")

    n_agents = len(path_batch)
    k_eff = int(n_routes)
    if n_agents <= 0 or k_eff <= 0:
        return None

    device = edge_cost_next.device
    dtype = edge_cost_next.dtype
    n_edges = int(edge_cost_next.shape[0])
    fallback_cost = invalid_route_cost.to(device=device, dtype=dtype)
    current_np = edge_cost_current.detach().cpu().numpy()
    costs = torch.empty((n_agents, k_eff), device=device, dtype=dtype)

    for i in range(n_agents):
        paths = path_batch[i] if i < len(path_batch) else []
        for k in range(k_eff):
            path = paths[k] if k < len(paths) else []
            idxs = [int(e) for e in path if 0 <= int(e) < n_edges]
            if not idxs:
                costs[i, k] = fallback_cost
                continue
            suffix = _advance_path_by_seconds(idxs, current_np, float(step_seconds), n_edges)
            if not suffix:
                costs[i, k] = torch.zeros((), device=device, dtype=dtype)
                continue
            edge_idx = torch.tensor(suffix, device=device, dtype=torch.long)
            costs[i, k] = edge_cost_next.index_select(0, edge_idx).sum().to(dtype=dtype)

    return costs


def _advance_path_by_seconds(
    path: List[int],
    edge_cost_seconds: np.ndarray,
    step_seconds: float,
    n_edges: int,
) -> List[int]:
    remaining = max(float(step_seconds), 0.0)
    valid_path = [int(e) for e in path if 0 <= int(e) < int(n_edges)]
    if not valid_path or remaining <= 1e-8:
        return valid_path

    for pos, edge_idx in enumerate(valid_path):
        edge_seconds = float(edge_cost_seconds[int(edge_idx)])
        if not np.isfinite(edge_seconds) or edge_seconds <= 1e-8:
            edge_seconds = 1e-8
        if remaining < edge_seconds:
            return valid_path[pos:]
        remaining -= edge_seconds
    return []


def _softmin_route_cost_from_paths(
    *,
    path_batch: List[List[List[int]]],
    edge_cost: torch.Tensor,
    n_routes: int,
    invalid_route_cost: torch.Tensor,
    temperature: float,
) -> Optional[torch.Tensor]:
    """Policy-independent soft shortest-path cost over current candidates.

    This defines the per-agent potential R_i(s_k): each active vehicle only
    looks at candidate paths from its current edge to its destination, then
    takes a soft minimum over candidate route costs.  The actor probabilities
    are intentionally not used here, so progress reward measures state
    improvement instead of rewarding the current policy's own route mixture.
    """
    route_costs, route_valid = _route_cost_matrix_from_paths(
        path_batch=path_batch,
        edge_cost=edge_cost,
        n_routes=n_routes,
        invalid_route_cost=invalid_route_cost,
    )
    if route_costs is None or route_valid is None:
        return None

    dtype = route_costs.dtype
    device = route_costs.device
    fallback_cost = invalid_route_cost.to(device=device, dtype=dtype)
    valid_any = route_valid.any(dim=1)
    if not bool(valid_any.any().detach().cpu().item()):
        return None

    large_cost = fallback_cost + route_costs.detach().abs().max().clamp_min(1.0) * 1000.0
    masked_costs = torch.where(route_valid, route_costs, large_cost)

    temp = float(temperature)
    if temp <= 1e-8:
        per_agent = masked_costs.min(dim=1).values
    else:
        tau = torch.as_tensor(temp, device=device, dtype=dtype)
        per_agent = -tau * torch.logsumexp(-masked_costs / tau, dim=1)
    per_agent = torch.where(valid_any, per_agent, fallback_cost)
    return per_agent[valid_any].mean()


def _expected_route_cost_from_paths(
    *,
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    edge_cost: torch.Tensor,
    n_routes: int,
    invalid_route_cost: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Expected candidate-path cost under the actor distribution.

    This uses only current candidate route topology plus WM-predicted edge costs.
    It deliberately does not encode any externally observed best route label.
    """
    if route_probs.dim() != 2:
        raise ValueError(f"route_probs must be 2D, got {tuple(route_probs.shape)}")
    if edge_cost.dim() != 1:
        raise ValueError(f"edge_cost must be 1D, got {tuple(edge_cost.shape)}")

    n_agents = min(int(route_probs.shape[0]), len(path_batch))
    k_eff = min(int(route_probs.shape[1]), int(n_routes))
    if n_agents <= 0 or k_eff <= 0:
        return None

    device = route_probs.device
    dtype = route_probs.dtype
    n_edges = int(edge_cost.shape[0])
    fallback_cost = invalid_route_cost.to(device=device, dtype=dtype)
    agent_costs: List[torch.Tensor] = []

    for i in range(n_agents):
        paths = path_batch[i] if i < len(path_batch) else []
        costs: List[torch.Tensor] = []
        for k in range(k_eff):
            path = paths[k] if k < len(paths) else []
            idxs = [int(e) for e in path if 0 <= int(e) < n_edges]
            if idxs:
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                costs.append(edge_cost.index_select(0, edge_idx).sum().to(dtype=dtype))
            else:
                costs.append(fallback_cost)
        route_costs = torch.stack(costs)
        probs = route_probs[i, :k_eff]
        prob_sum = probs.sum().clamp_min(1e-8)
        agent_costs.append(torch.sum((probs / prob_sum) * route_costs))

    return torch.stack(agent_costs).mean()


def _expected_route_cost_from_cost_matrix(
    *,
    route_probs: torch.Tensor,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Expected candidate-route cost from a precomputed cost matrix."""
    n = min(int(route_probs.shape[0]), int(route_costs.shape[0]), int(route_valid.shape[0]))
    k = min(int(route_probs.shape[1]), int(route_costs.shape[1]), int(route_valid.shape[1]))
    if n <= 0 or k <= 0:
        return None
    probs = route_probs[:n, :k].to(device=route_costs.device, dtype=route_costs.dtype)
    costs = route_costs[:n, :k]
    valid = route_valid[:n, :k].to(device=route_costs.device)
    agent_mask = valid.any(dim=-1)
    if not bool(agent_mask.any().detach().cpu().item()):
        return None
    probs = torch.where(valid, torch.relu(probs), torch.zeros_like(probs))
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    per_agent = (probs * costs.masked_fill(~valid, 0.0)).sum(dim=-1)
    return per_agent[agent_mask].mean()


def _expected_route_regret_from_cost_matrix(
    *,
    route_probs: torch.Tensor,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
    temperature: float,
) -> Optional[torch.Tensor]:
    """Expected per-agent route regret from a precomputed cost matrix."""
    n = min(int(route_probs.shape[0]), int(route_costs.shape[0]), int(route_valid.shape[0]))
    k = min(int(route_probs.shape[1]), int(route_costs.shape[1]), int(route_valid.shape[1]))
    if n <= 0 or k <= 0:
        return None
    probs = route_probs[:n, :k].to(device=route_costs.device, dtype=route_costs.dtype)
    valid = route_valid[:n, :k].to(device=route_costs.device)
    costs = route_costs[:n, :k].to(device=route_costs.device, dtype=route_costs.dtype)
    agent_mask = valid.any(dim=-1)
    if not bool(agent_mask.any().detach().cpu().item()):
        return None

    probs = torch.where(valid, torch.relu(probs), torch.zeros_like(probs))
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    large = torch.finfo(costs.dtype).max / 4.0
    masked_costs = costs.masked_fill(~valid, large)
    min_cost = masked_costs.min(dim=-1, keepdim=True).values
    regret = (masked_costs - min_cost) / float(max(temperature, 1e-6))
    regret = regret.masked_fill(~valid, 0.0)
    per_agent = (probs * regret).sum(dim=-1)
    return per_agent[agent_mask].mean()


def _expected_route_regret_from_paths(
    *,
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    edge_cost: torch.Tensor,
    n_routes: int,
    invalid_route_cost: torch.Tensor,
    temperature: float,
) -> Optional[torch.Tensor]:
    """Expected per-agent route regret normalized by a seconds temperature."""
    route_costs, route_valid = _route_cost_matrix_from_paths(
        path_batch=path_batch,
        edge_cost=edge_cost,
        n_routes=n_routes,
        invalid_route_cost=invalid_route_cost,
    )
    if route_costs is None or route_valid is None:
        return None
    n = min(int(route_probs.shape[0]), int(route_costs.shape[0]))
    k = min(int(route_probs.shape[1]), int(route_costs.shape[1]))
    if n <= 0 or k <= 0:
        return None

    return _expected_route_regret_from_cost_matrix(
        route_probs=route_probs[:n, :k],
        route_costs=route_costs[:n, :k],
        route_valid=route_valid[:n, :k],
        temperature=float(temperature),
    )


def _route_cost_matrix_from_paths(
    *,
    path_batch: List[List[List[int]]],
    edge_cost: torch.Tensor,
    n_routes: int,
    invalid_route_cost: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return per-agent candidate route costs and a valid-path mask.

    Costs are computed from the current imagined state and candidate path
    topology only.  No observed best-route labels or future SUMO outcomes are
    used.
    """
    if edge_cost.dim() != 1:
        raise ValueError(f"edge_cost must be 1D, got {tuple(edge_cost.shape)}")

    n_agents = len(path_batch)
    k_eff = int(n_routes)
    if n_agents <= 0 or k_eff <= 0:
        return None, None

    device = edge_cost.device
    dtype = edge_cost.dtype
    n_edges = int(edge_cost.shape[0])
    fallback_cost = invalid_route_cost.to(device=device, dtype=dtype)
    costs = torch.empty((n_agents, k_eff), device=device, dtype=dtype)
    valid = torch.zeros((n_agents, k_eff), device=device, dtype=torch.bool)

    for i in range(n_agents):
        paths = path_batch[i] if i < len(path_batch) else []
        for k in range(k_eff):
            path = paths[k] if k < len(paths) else []
            idxs = [int(e) for e in path if 0 <= int(e) < n_edges]
            if idxs:
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                costs[i, k] = edge_cost.index_select(0, edge_idx).sum().to(dtype=dtype)
                valid[i, k] = True
            else:
                costs[i, k] = fallback_cost

    if not bool(valid.any().detach().cpu().item()):
        return None, None
    return costs, valid


def _route_cost_policy_aux_loss(
    *,
    logits: torch.Tensor,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
    temperature: float,
    balance_best_classes: bool = False,
    hard_target: bool = False,
    nonzero_margin: float = 0.0,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Cross-entropy from actor logits to current best route slots.

    Candidate slot order is preserved by callers.  The target route is the
    argmin within those original slots after masking invalid candidates; this
    function never reorders or renumbers routes.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if route_costs.dim() != 2:
        raise ValueError(f"route_costs must be 2D, got {tuple(route_costs.shape)}")
    if route_valid.shape != route_costs.shape:
        raise ValueError(
            f"route_valid shape {tuple(route_valid.shape)} must match route_costs {tuple(route_costs.shape)}"
        )

    n = min(int(logits.shape[0]), int(route_costs.shape[0]))
    k = min(int(logits.shape[1]), int(route_costs.shape[1]))
    if n <= 0 or k <= 0:
        return None, {"route_cost_aux_acc": 0.0, "route_cost_aux_margin": 0.0}

    logits = logits[:n, :k]
    route_costs = route_costs[:n, :k].to(device=logits.device, dtype=logits.dtype)
    route_valid = route_valid[:n, :k].to(device=logits.device)
    agent_mask = route_valid.any(dim=-1)
    if not bool(agent_mask.any().detach().cpu().item()):
        return None, {"route_cost_aux_acc": 0.0, "route_cost_aux_margin": 0.0}

    logits = logits[agent_mask]
    route_costs = route_costs[agent_mask]
    route_valid = route_valid[agent_mask]
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0).clamp(min=-50.0, max=50.0)
    large = torch.finfo(route_costs.dtype).max / 4.0
    masked_costs = route_costs.masked_fill(~route_valid, large)
    best = torch.argmin(masked_costs, dim=-1)

    masked_logits = logits.masked_fill(~route_valid, -large)
    if bool(hard_target):
        per_agent_loss = F.cross_entropy(masked_logits, best, reduction="none")
    else:
        min_cost = masked_costs.gather(1, best.unsqueeze(-1))
        normalized_costs = masked_costs - min_cost
        temp = float(max(temperature, 1e-6))
        target_logits = -(normalized_costs / temp).masked_fill(~route_valid, large)
        target = torch.softmax(target_logits, dim=-1).detach()
        log_probs = F.log_softmax(masked_logits, dim=-1)
        per_agent_loss = -(target * log_probs).sum(dim=-1)
    if bool(balance_best_classes):
        counts = torch.bincount(best, minlength=k).to(device=logits.device, dtype=logits.dtype)
        present = counts > 0
        weights = torch.zeros_like(counts)
        weights[present] = counts[present].sum() / (present.to(logits.dtype).sum().clamp_min(1.0) * counts[present])
        sample_weights = weights.index_select(0, best).detach()
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)
        loss = (per_agent_loss * sample_weights).mean()
    else:
        loss = per_agent_loss.mean()

    margin_value = float(nonzero_margin)
    nonzero_best_mask = best != 0
    if margin_value > 0.0 and k > 1 and bool(nonzero_best_mask.any().detach().cpu().item()):
        best_logits = masked_logits.gather(1, best.unsqueeze(-1)).squeeze(-1)
        route0_logits = masked_logits[:, 0]
        margin_terms = F.relu(route0_logits - best_logits + margin_value)[nonzero_best_mask]
        if margin_terms.numel() > 0:
            loss = loss + margin_terms.mean()

    pred = torch.argmax(masked_logits.detach(), dim=-1)
    acc = float((pred == best).float().mean().detach().cpu().item())
    nonzero_best_frac = float((best != 0).float().mean().detach().cpu().item())
    valid_counts = route_valid.sum(dim=-1)
    if bool((valid_counts > 1).any().detach().cpu().item()):
        top2 = torch.topk(masked_costs, k=2, largest=False, dim=-1).values
        margin_t = (top2[:, 1] - top2[:, 0])[valid_counts > 1]
        margin = float(margin_t.mean().detach().cpu().item()) if margin_t.numel() > 0 else 0.0
    else:
        margin = 0.0
    return loss, {
        "route_cost_aux_acc": acc,
        "route_cost_aux_margin": margin,
        "route_cost_aux_nonzero_best_frac": nonzero_best_frac,
    }


def _planned_av_pressure_cost_matrix(
    *,
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    route_valid: torch.Tensor,
    n_edges: int,
    n_routes: int,
    edge_weights: Optional[torch.Tensor] = None,
    normalize_by_path_length: bool = False,
    detach_pressure: bool = True,
) -> Optional[torch.Tensor]:
    """Return per-route overlap with the current expected planned AV edge load.

    Optional edge weights turn pure path-overlap pressure into a WM-weighted
    externality proxy, e.g. planned load on edges with high decoded delay.
    """
    if route_probs.dim() != 2:
        raise ValueError(f"route_probs must be 2D, got {tuple(route_probs.shape)}")
    if route_valid.dim() != 2:
        raise ValueError(f"route_valid must be 2D, got {tuple(route_valid.shape)}")

    n_agents = min(int(route_probs.shape[0]), len(path_batch), int(route_valid.shape[0]))
    k_eff = min(int(route_probs.shape[1]), int(route_valid.shape[1]), int(n_routes))
    if n_agents <= 0 or k_eff <= 0 or int(n_edges) <= 0:
        return None

    device = route_probs.device
    dtype = route_probs.dtype
    if edge_weights is not None:
        edge_weights = torch.nan_to_num(
            edge_weights.to(device=device, dtype=dtype).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        if int(edge_weights.numel()) < int(n_edges):
            padded = torch.zeros((int(n_edges),), device=device, dtype=dtype)
            padded[: int(edge_weights.numel())] = edge_weights
            edge_weights = padded
        else:
            edge_weights = edge_weights[: int(n_edges)]
    probs = torch.relu(route_probs[:n_agents, :k_eff])
    valid = route_valid[:n_agents, :k_eff].to(device=device)
    probs = probs * valid.to(dtype=dtype)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    edge_pressure = torch.zeros((int(n_edges),), device=device, dtype=dtype)
    route_edge_indices: List[List[List[int]]] = []
    for i in range(n_agents):
        paths = path_batch[i] if i < len(path_batch) else []
        per_agent: List[List[int]] = []
        for k in range(k_eff):
            path = paths[k] if k < len(paths) else []
            idxs = sorted({int(e) for e in path if 0 <= int(e) < int(n_edges)})
            per_agent.append(idxs)
            if idxs and bool(valid[i, k].detach().cpu().item()):
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                edge_pressure.index_add_(0, edge_idx, probs[i, k].expand(len(idxs)))
        route_edge_indices.append(per_agent)
    edge_pressure = edge_pressure / float(max(n_agents, 1))
    if bool(detach_pressure):
        edge_pressure = edge_pressure.detach()

    pressure_costs = torch.zeros((n_agents, k_eff), device=device, dtype=dtype)
    for i in range(n_agents):
        for k in range(k_eff):
            idxs = route_edge_indices[i][k]
            if idxs and bool(valid[i, k].detach().cpu().item()):
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                pressure_values = edge_pressure.index_select(0, edge_idx)
                if edge_weights is not None:
                    pressure_values = pressure_values * edge_weights.index_select(0, edge_idx)
                cost_value = pressure_values.sum()
                if bool(normalize_by_path_length):
                    cost_value = cost_value / float(max(len(idxs), 1))
                pressure_costs[i, k] = cost_value
    return pressure_costs


def _marginal_av_pressure_cost_matrix(
    *,
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    route_valid: torch.Tensor,
    n_edges: int,
    n_routes: int,
    edge_weights: Optional[torch.Tensor] = None,
    pressure_route_probs: Optional[torch.Tensor] = None,
    normalize_by_path_length: bool = False,
    detach_pressure: bool = True,
) -> Optional[torch.Tensor]:
    """Return per-route overlap with other vehicles' planned AV edge load.

    This is a COMA/difference-reward style externality proxy.  Unlike
    ``_planned_av_pressure_cost_matrix``, it removes vehicle i's expected edge
    contribution before scoring vehicle i's candidates, so the cost is
    state/edge-specific rather than a global route-slot share rule.
    """
    if route_probs.dim() != 2:
        raise ValueError(f"route_probs must be 2D, got {tuple(route_probs.shape)}")
    if route_valid.dim() != 2:
        raise ValueError(f"route_valid must be 2D, got {tuple(route_valid.shape)}")
    if pressure_route_probs is not None and pressure_route_probs.dim() != 2:
        raise ValueError(
            f"pressure_route_probs must be 2D, got {tuple(pressure_route_probs.shape)}"
        )

    n_agents = min(int(route_probs.shape[0]), len(path_batch), int(route_valid.shape[0]))
    if pressure_route_probs is not None:
        n_agents = min(n_agents, int(pressure_route_probs.shape[0]))
    k_eff = min(int(route_probs.shape[1]), int(route_valid.shape[1]), int(n_routes))
    if pressure_route_probs is not None:
        k_eff = min(k_eff, int(pressure_route_probs.shape[1]))
    if n_agents <= 0 or k_eff <= 0 or int(n_edges) <= 0:
        return None

    device = route_probs.device
    dtype = route_probs.dtype
    if edge_weights is not None:
        edge_weights = torch.nan_to_num(
            edge_weights.to(device=device, dtype=dtype).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        if int(edge_weights.numel()) < int(n_edges):
            padded = torch.zeros((int(n_edges),), device=device, dtype=dtype)
            padded[: int(edge_weights.numel())] = edge_weights
            edge_weights = padded
        else:
            edge_weights = edge_weights[: int(n_edges)]

    pressure_source = route_probs if pressure_route_probs is None else pressure_route_probs
    probs = torch.relu(pressure_source[:n_agents, :k_eff].to(device=device, dtype=dtype))
    valid = route_valid[:n_agents, :k_eff].to(device=device)
    probs = probs * valid.to(dtype=dtype)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    edge_pressure = torch.zeros((int(n_edges),), device=device, dtype=dtype)
    agent_edge_pressure = torch.zeros((n_agents, int(n_edges)), device=device, dtype=dtype)
    route_edge_indices: List[List[List[int]]] = []
    for i in range(n_agents):
        paths = path_batch[i] if i < len(path_batch) else []
        per_agent: List[List[int]] = []
        for k in range(k_eff):
            path = paths[k] if k < len(paths) else []
            idxs = sorted({int(e) for e in path if 0 <= int(e) < int(n_edges)})
            per_agent.append(idxs)
            if idxs and bool(valid[i, k].detach().cpu().item()):
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                contribution = probs[i, k].expand(len(idxs))
                edge_pressure.index_add_(0, edge_idx, contribution)
                agent_edge_pressure[i].index_add_(0, edge_idx, contribution)
        route_edge_indices.append(per_agent)

    denom = float(max(n_agents, 1))
    edge_pressure = edge_pressure / denom
    agent_edge_pressure = agent_edge_pressure / denom
    if bool(detach_pressure):
        edge_pressure = edge_pressure.detach()
        agent_edge_pressure = agent_edge_pressure.detach()

    pressure_costs = torch.zeros((n_agents, k_eff), device=device, dtype=dtype)
    for i in range(n_agents):
        other_pressure = (edge_pressure - agent_edge_pressure[i]).clamp_min(0.0)
        for k in range(k_eff):
            idxs = route_edge_indices[i][k]
            if idxs and bool(valid[i, k].detach().cpu().item()):
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                pressure_values = other_pressure.index_select(0, edge_idx)
                if edge_weights is not None:
                    pressure_values = pressure_values * edge_weights.index_select(0, edge_idx)
                cost_value = pressure_values.sum()
                if bool(normalize_by_path_length):
                    cost_value = cost_value / float(max(len(idxs), 1))
                pressure_costs[i, k] = cost_value
    return pressure_costs


def _sequential_marginal_av_pressure_cost_matrix(
    *,
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    route_valid: torch.Tensor,
    n_edges: int,
    n_routes: int,
    edge_weights: Optional[torch.Tensor] = None,
    normalize_by_path_length: bool = False,
) -> Optional[torch.Tensor]:
    """Return candidate overlap with previously fixed fleet route pressure.

    This matches the vehicle actor's sequential fleet-pressure observation:
    vehicle i is scored against the routes selected by vehicles < i, then its
    own argmax route is added to the pressure field for later vehicles.
    """
    if route_probs.dim() != 2 or route_valid.dim() != 2:
        return None
    n_agents = min(int(route_probs.shape[0]), len(path_batch), int(route_valid.shape[0]))
    k_eff = min(int(route_probs.shape[1]), int(route_valid.shape[1]), int(n_routes))
    if n_agents <= 0 or k_eff <= 0 or int(n_edges) <= 0:
        return None

    device = route_probs.device
    dtype = route_probs.dtype
    valid = route_valid[:n_agents, :k_eff].to(device=device)
    if edge_weights is not None:
        edge_weights = torch.nan_to_num(
            edge_weights.to(device=device, dtype=dtype).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        if int(edge_weights.numel()) < int(n_edges):
            padded = torch.zeros((int(n_edges),), device=device, dtype=dtype)
            padded[: int(edge_weights.numel())] = edge_weights
            edge_weights = padded
        else:
            edge_weights = edge_weights[: int(n_edges)]

    pressure = torch.zeros((int(n_edges),), device=device, dtype=dtype)
    costs = torch.zeros((n_agents, k_eff), device=device, dtype=dtype)
    for i in range(n_agents):
        paths = path_batch[i] if i < len(path_batch) else []
        for k in range(k_eff):
            path = paths[k] if k < len(paths) else []
            idxs = sorted({int(e) for e in path if 0 <= int(e) < int(n_edges)})
            if idxs and bool(valid[i, k].detach().cpu().item()):
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                pressure_values = pressure.index_select(0, edge_idx)
                if edge_weights is not None:
                    pressure_values = pressure_values * edge_weights.index_select(0, edge_idx)
                cost_value = pressure_values.sum()
                if bool(normalize_by_path_length):
                    cost_value = cost_value / float(max(len(idxs), 1))
                costs[i, k] = cost_value

        valid_i = valid[i]
        if bool(valid_i.any().detach().cpu().item()):
            large_neg = torch.finfo(dtype).min / 4.0
            logits_i = route_probs[i, :k_eff].masked_fill(~valid_i, large_neg)
            selected = int(torch.argmax(logits_i.detach()).cpu().item())
            if selected < len(paths):
                idxs = sorted({int(e) for e in paths[selected] if 0 <= int(e) < int(n_edges)})
                if idxs:
                    edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                    pressure.index_add_(0, edge_idx, torch.ones((len(idxs),), device=device, dtype=dtype))
    denom = torch.arange(n_agents, device=device, dtype=dtype).clamp_min(1.0).unsqueeze(-1)
    return costs / denom


def _route_share_penalty_cost_matrix(
    *,
    route_probs: torch.Tensor,
    route_valid: torch.Tensor,
    cap: float,
    detach_share: bool = True,
) -> Optional[torch.Tensor]:
    """Return per-candidate marginal cost from fleet-level route-slot share.

    This is a lightweight externality proxy: if the actor currently places too
    much fleet probability on candidate slot k, all candidates in that slot get
    a positive marginal cost. It keeps the original slot order intact and does
    not construct labels.
    """
    if route_probs.dim() != 2 or route_valid.dim() != 2:
        return None
    n = min(int(route_probs.shape[0]), int(route_valid.shape[0]))
    k = min(int(route_probs.shape[1]), int(route_valid.shape[1]))
    if n <= 0 or k <= 0:
        return None

    probs = torch.relu(route_probs[:n, :k])
    valid = route_valid[:n, :k].to(device=route_probs.device)
    probs = probs * valid.to(dtype=probs.dtype)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    fleet_share = probs.mean(dim=0)
    excess = (fleet_share - float(max(0.0, min(float(cap), 1.0)))).clamp_min(0.0)
    if bool(detach_share):
        excess = excess.detach()
    return excess.unsqueeze(0).expand(n, k).masked_fill(~valid, 0.0)


def _planned_av_pressure_edge_weights(
    *,
    wm: UrbanWorldModel,
    edge_tt: torch.Tensor,
    v_total: torch.Tensor,
    capacities: torch.Tensor,
    mode: str,
) -> Optional[torch.Tensor]:
    """Edge weights for turning planned route overlap into WM congestion pressure."""
    mode_key = str(mode or "overlap").strip().lower()
    if mode_key in {"", "overlap", "plain", "none"}:
        return None
    if mode_key in {"volume_ratio", "v_over_c"}:
        return (v_total / capacities.clamp_min(1.0)).clamp_min(0.0)
    if mode_key in {"excess_volume_ratio", "volume_excess"}:
        return (v_total / capacities.clamp_min(1.0) - 1.0).clamp_min(0.0)
    if mode_key in {"tt_ratio", "travel_time_ratio"}:
        free_flow = _free_flow_edge_tt_tensor(wm, device=edge_tt.device, dtype=edge_tt.dtype)
        return (edge_tt / free_flow.clamp_min(1e-6)).clamp_min(0.0)
    if mode_key in {"excess_tt_ratio", "tt_excess", "wm_excess_tt"}:
        free_flow = _free_flow_edge_tt_tensor(wm, device=edge_tt.device, dtype=edge_tt.dtype)
        return (edge_tt / free_flow.clamp_min(1e-6) - 1.0).clamp_min(0.0)
    if mode_key in {"clipped_excess_tt_ratio", "clipped_tt_excess", "wm_clipped_excess_tt"}:
        free_flow = _free_flow_edge_tt_tensor(wm, device=edge_tt.device, dtype=edge_tt.dtype)
        return (edge_tt / free_flow.clamp_min(1e-6) - 1.0).clamp(0.0, 3.0)
    if mode_key in {"threshold_excess_tt_ratio", "thresholded_excess_tt_ratio", "wm_threshold_excess_tt"}:
        free_flow = _free_flow_edge_tt_tensor(wm, device=edge_tt.device, dtype=edge_tt.dtype)
        return (edge_tt / free_flow.clamp_min(1e-6) - 1.25).clamp(0.0, 3.0)
    raise ValueError(f"Unsupported planned AV pressure mode: {mode!r}")


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


def _marginal_pressure_reference_probs(
    route_probs: torch.Tensor,
    route_valid: torch.Tensor,
    *,
    reference: str,
) -> torch.Tensor:
    """Return detached route probabilities used as the fixed fleet baseline."""
    ref = str(reference or "soft").strip().lower()
    valid = route_valid[: route_probs.shape[0], : route_probs.shape[1]].to(device=route_probs.device)
    if ref in {"", "soft", "policy", "expected"}:
        probs = torch.relu(route_probs) * valid.to(dtype=route_probs.dtype)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs.detach()
    if ref in {"argmax", "deterministic", "hard"}:
        large_neg = torch.finfo(route_probs.dtype).min / 4.0
        masked_logits = route_probs.masked_fill(~valid, large_neg)
        best = torch.argmax(masked_logits, dim=-1)
        one_hot = F.one_hot(best, num_classes=int(route_probs.shape[1])).to(
            device=route_probs.device,
            dtype=route_probs.dtype,
        )
        one_hot = one_hot * valid.to(dtype=route_probs.dtype)
        return one_hot.detach()
    raise ValueError(f"Unsupported marginal AV pressure reference: {reference!r}")


def _free_flow_edge_tt_tensor(
    wm: UrbanWorldModel,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    lengths = torch.as_tensor(
        np.asarray(
            wm._registry.lengths if wm._registry.lengths is not None
            else np.ones((wm.n_edges,), dtype=np.float32) * 100.0,
            dtype=np.float32,
        ),
        device=device,
        dtype=dtype,
    )
    speed_limits = torch.as_tensor(
        np.asarray(wm._registry.speed_limits, dtype=np.float32),
        device=device,
        dtype=dtype,
    )
    return lengths / speed_limits.clamp_min(0.1)


def _expected_cost_policy_loss(
    *,
    logits: torch.Tensor,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
    temperature: float,
    detach_cost: bool = True,
    balance_best_classes: bool = False,
    pairwise: bool = False,
    pairwise_margin: float = 0.0,
    route0_margin_weight: float = 0.0,
    route0_margin: float = 0.0,
    route0_margin_min_gap: float = 0.0,
    nonzero_best_boost: float = 0.0,
    nonzero_best_min_gap: float = 0.0,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Direct actor objective for minimizing E_pi[C_i(a)].

    This is not a distillation/CE target: no best-route label is constructed.
    The loss is the actor's expected normalized route cost under its own
    probabilities while preserving the original candidate slot order.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if route_costs.dim() != 2:
        raise ValueError(f"route_costs must be 2D, got {tuple(route_costs.shape)}")
    if route_valid.shape != route_costs.shape:
        raise ValueError(
            f"route_valid shape {tuple(route_valid.shape)} must match route_costs {tuple(route_costs.shape)}"
        )

    n = min(int(logits.shape[0]), int(route_costs.shape[0]))
    k = min(int(logits.shape[1]), int(route_costs.shape[1]))
    empty_stats = {
        "expected_cost_loss": 0.0,
        "expected_cost_raw": 0.0,
        "expected_cost_best_prob": 0.0,
        "expected_cost_best_route0_frac": 0.0,
        "expected_cost_nonzero_best_frac": 0.0,
        "expected_cost_margin": 0.0,
        "expected_cost_route0_margin_loss": 0.0,
        "expected_cost_route0_margin_frac": 0.0,
    }
    if n <= 0 or k <= 0:
        return None, empty_stats

    logits = logits[:n, :k]
    route_costs = route_costs[:n, :k].to(device=logits.device, dtype=logits.dtype)
    route_valid = route_valid[:n, :k].to(device=logits.device)
    agent_mask = route_valid.any(dim=-1)
    if not bool(agent_mask.any().detach().cpu().item()):
        return None, empty_stats

    logits = logits[agent_mask]
    route_costs = route_costs[agent_mask]
    route_valid = route_valid[agent_mask]
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0).clamp(min=-50.0, max=50.0)
    large = torch.finfo(route_costs.dtype).max / 4.0
    masked_costs = route_costs.masked_fill(~route_valid, large)
    masked_logits = logits.masked_fill(~route_valid, -large)
    probs = torch.softmax(masked_logits, dim=-1)
    best = torch.argmin(masked_costs, dim=-1)
    min_cost = masked_costs.gather(1, best.unsqueeze(-1))
    valid_counts = route_valid.sum(dim=-1)
    if k > 1:
        top2 = torch.topk(masked_costs, k=2, largest=False, dim=-1).values
        best_gap = top2[:, 1] - top2[:, 0]
    else:
        best_gap = torch.zeros_like(min_cost.squeeze(-1))
    scaled_costs = (masked_costs - min_cost) / float(max(temperature, 1e-6))
    scaled_costs = scaled_costs.masked_fill(~route_valid, 0.0)
    raw_expected = (probs * masked_costs.masked_fill(~route_valid, 0.0)).sum(dim=-1)
    if bool(detach_cost):
        scaled_costs = scaled_costs.detach()
    if bool(pairwise):
        cost_i = masked_costs.unsqueeze(2)
        cost_j = masked_costs.unsqueeze(1)
        valid_pairs = route_valid.unsqueeze(2) & route_valid.unsqueeze(1) & (cost_i < cost_j)
        cost_gap = ((cost_j - cost_i) / float(max(temperature, 1e-6))).clamp_min(0.0)
        cost_gap = cost_gap.masked_fill(~valid_pairs, 0.0)
        if bool(detach_cost):
            cost_gap = cost_gap.detach()
        logit_diff = masked_logits.unsqueeze(2) - masked_logits.unsqueeze(1)
        margin = float(max(pairwise_margin, 0.0))
        pair_losses = F.softplus(-(logit_diff - margin)) * cost_gap
        gap_sum = cost_gap.sum(dim=(1, 2))
        pair_sum = pair_losses.sum(dim=(1, 2))
        expected_fallback = (probs * scaled_costs).sum(dim=-1)
        per_agent_loss = torch.where(gap_sum > 1e-8, pair_sum / gap_sum.clamp_min(1e-8), expected_fallback)
    else:
        per_agent_loss = (probs * scaled_costs).sum(dim=-1)
    route0_margin_losses = torch.zeros_like(per_agent_loss)
    r0_margin_weight = float(max(route0_margin_weight, 0.0))
    if r0_margin_weight > 0.0 and k > 1:
        route0_valid = route_valid[:, 0]
        best_valid = route_valid.gather(1, best.unsqueeze(-1)).squeeze(-1)
        route0_cost = masked_costs[:, 0]
        best_cost = masked_costs.gather(1, best.unsqueeze(-1)).squeeze(-1)
        route0_gap = route0_cost - best_cost
        margin_mask = (
            (best != 0)
            & route0_valid
            & best_valid
            & (route0_gap >= float(max(route0_margin_min_gap, 0.0)))
        )
        if bool(margin_mask.any().detach().cpu().item()):
            best_logits = masked_logits.gather(1, best.unsqueeze(-1)).squeeze(-1)
            route0_logits = masked_logits[:, 0]
            route0_margin_losses = F.relu(
                route0_logits - best_logits + float(max(route0_margin, 0.0))
            )
            route0_margin_losses = route0_margin_losses * margin_mask.to(dtype=route0_margin_losses.dtype)
            route0_margin_losses = route0_margin_losses / margin_mask.to(dtype=route0_margin_losses.dtype).mean().clamp_min(1e-8)
            per_agent_loss = per_agent_loss + r0_margin_weight * route0_margin_losses
    boost = float(max(nonzero_best_boost, 0.0))
    min_gap = float(max(nonzero_best_min_gap, 0.0))
    if boost > 0.0:
        nonzero_high_conf = (best != 0) & (best_gap >= min_gap) & (valid_counts > 1)
        if bool(nonzero_high_conf.any().detach().cpu().item()):
            sample_weights = torch.ones_like(per_agent_loss)
            sample_weights = sample_weights + boost * nonzero_high_conf.to(dtype=sample_weights.dtype)
            sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)
            per_agent_loss = per_agent_loss * sample_weights
    if bool(balance_best_classes):
        counts = torch.bincount(best, minlength=k).to(device=logits.device, dtype=logits.dtype)
        present = counts > 0
        weights = torch.zeros_like(counts)
        weights[present] = counts[present].sum() / (present.to(logits.dtype).sum().clamp_min(1.0) * counts[present])
        sample_weights = weights.index_select(0, best).detach()
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)
        per_agent_loss = per_agent_loss * sample_weights
    stats = {
        "expected_cost_loss": float(per_agent_loss.detach().mean().cpu().item()),
        "expected_cost_raw": float(raw_expected.detach().mean().cpu().item()),
        "expected_cost_best_prob": float(
            probs.gather(1, best.unsqueeze(-1)).detach().mean().cpu().item()
        ),
        "expected_cost_best_route0_frac": float((best == 0).float().mean().detach().cpu().item()),
        "expected_cost_nonzero_best_frac": float((best != 0).float().mean().detach().cpu().item()),
        "expected_cost_route0_margin_loss": float(route0_margin_losses.detach().mean().cpu().item()),
        "expected_cost_route0_margin_frac": float(
            ((best != 0) & route_valid[:, 0]).float().mean().detach().cpu().item()
        ) if k > 1 else 0.0,
    }
    if bool((valid_counts > 1).any().detach().cpu().item()):
        margin_t = (top2[:, 1] - top2[:, 0])[valid_counts > 1]
        stats["expected_cost_margin"] = float(margin_t.mean().detach().cpu().item()) if margin_t.numel() > 0 else 0.0
    else:
        stats["expected_cost_margin"] = 0.0
    return per_agent_loss.mean(), stats


def _grpo_route_cost_policy_loss(
    *,
    logits: torch.Tensor,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
    temperature: float,
    normalize_advantages: bool = True,
    detach_cost: bool = True,
    pairwise_weight: float = 0.0,
    pairwise_margin: float = 0.0,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Per-vehicle route-group policy loss from relative route costs.

    Each active vehicle's valid candidate routes form one GRPO-style group.
    The advantage is the within-vehicle mean route cost minus the candidate
    cost, so lower-cost routes get positive advantage and higher-cost routes
    get negative advantage.  This keeps original route slots intact and does
    not build a hard best-route supervision label.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if route_costs.dim() != 2:
        raise ValueError(f"route_costs must be 2D, got {tuple(route_costs.shape)}")
    if route_valid.shape != route_costs.shape:
        raise ValueError(
            f"route_valid shape {tuple(route_valid.shape)} must match route_costs {tuple(route_costs.shape)}"
        )

    n = min(int(logits.shape[0]), int(route_costs.shape[0]))
    k = min(int(logits.shape[1]), int(route_costs.shape[1]))
    empty_stats = {
        "grpo_route_cost_loss": 0.0,
        "grpo_route_cost_raw": 0.0,
        "grpo_route_cost_best_prob": 0.0,
        "grpo_route_cost_best_route0_frac": 0.0,
        "grpo_route_cost_nonzero_best_frac": 0.0,
        "grpo_route_cost_margin": 0.0,
        "grpo_route_cost_adv_mean": 0.0,
        "grpo_route_cost_adv_std": 0.0,
        "grpo_route_cost_pairwise_loss": 0.0,
    }
    if n <= 0 or k <= 0:
        return None, empty_stats

    logits = logits[:n, :k]
    route_costs = route_costs[:n, :k].to(device=logits.device, dtype=logits.dtype)
    route_valid = route_valid[:n, :k].to(device=logits.device)
    agent_mask = route_valid.any(dim=-1)
    if not bool(agent_mask.any().detach().cpu().item()):
        return None, empty_stats

    logits = logits[agent_mask]
    route_costs = route_costs[agent_mask]
    route_valid = route_valid[agent_mask]
    large = torch.finfo(route_costs.dtype).max / 4.0
    masked_costs = route_costs.masked_fill(~route_valid, large)
    masked_logits = logits.masked_fill(~route_valid, -large)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    probs = torch.softmax(masked_logits, dim=-1)

    valid_f = route_valid.to(dtype=route_costs.dtype)
    valid_count = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    cost_sum = (route_costs.masked_fill(~route_valid, 0.0) * valid_f).sum(dim=-1, keepdim=True)
    group_mean = cost_sum / valid_count
    advantages = (group_mean - route_costs) / float(max(temperature, 1e-6))
    advantages = advantages.masked_fill(~route_valid, 0.0)
    if bool(normalize_advantages):
        centered_cost = (route_costs - group_mean).masked_fill(~route_valid, 0.0)
        group_std = torch.sqrt((centered_cost.square() * valid_f).sum(dim=-1, keepdim=True) / valid_count).clamp_min(1e-6)
        advantages = ((group_mean - route_costs) / group_std).masked_fill(~route_valid, 0.0)
    if bool(detach_cost):
        advantages = advantages.detach()

    group_weights = valid_f / valid_count
    per_agent_loss = -(group_weights * advantages * log_probs).sum(dim=-1)
    pairwise_losses = torch.zeros_like(per_agent_loss)
    pw_weight = float(max(pairwise_weight, 0.0))
    if pw_weight > 0.0 and k > 1:
        cost_low = masked_costs.unsqueeze(2)
        cost_high = masked_costs.unsqueeze(1)
        valid_pairs = (
            route_valid.unsqueeze(2)
            & route_valid.unsqueeze(1)
            & (cost_low < cost_high)
        )
        if bool(valid_pairs.any().detach().cpu().item()):
            cost_gap = ((cost_high - cost_low) / float(max(temperature, 1e-6))).clamp_min(0.0)
            if bool(normalize_advantages):
                centered_cost = (route_costs - group_mean).masked_fill(~route_valid, 0.0)
                group_std = torch.sqrt(
                    (centered_cost.square() * valid_f).sum(dim=-1, keepdim=True)
                    / valid_count
                ).clamp_min(1e-6)
                cost_gap = ((cost_high - cost_low) / group_std.unsqueeze(2)).clamp_min(0.0)
            cost_gap = cost_gap.masked_fill(~valid_pairs, 0.0)
            if bool(detach_cost):
                cost_gap = cost_gap.detach()
            logit_low = masked_logits.unsqueeze(2)
            logit_high = masked_logits.unsqueeze(1)
            margin = float(max(pairwise_margin, 0.0))
            pair_loss = F.softplus(-(logit_low - logit_high - margin)) * cost_gap
            pair_sum = pair_loss.sum(dim=(1, 2))
            pair_weight_sum = cost_gap.sum(dim=(1, 2)).clamp_min(1e-8)
            pairwise_losses = pair_sum / pair_weight_sum
            per_agent_loss = per_agent_loss + pw_weight * pairwise_losses
    best = torch.argmin(masked_costs, dim=-1)
    raw_expected = (probs * route_costs.masked_fill(~route_valid, 0.0)).sum(dim=-1)
    adv_valid = advantages.detach()[route_valid]
    stats = {
        "grpo_route_cost_loss": float(per_agent_loss.detach().mean().cpu().item()),
        "grpo_route_cost_raw": float(raw_expected.detach().mean().cpu().item()),
        "grpo_route_cost_best_prob": float(
            probs.gather(1, best.unsqueeze(-1)).detach().mean().cpu().item()
        ),
        "grpo_route_cost_best_route0_frac": float((best == 0).float().mean().detach().cpu().item()),
        "grpo_route_cost_nonzero_best_frac": float((best != 0).float().mean().detach().cpu().item()),
        "grpo_route_cost_adv_mean": float(adv_valid.mean().detach().cpu().item()) if adv_valid.numel() > 0 else 0.0,
        "grpo_route_cost_adv_std": float(adv_valid.std(unbiased=False).detach().cpu().item()) if adv_valid.numel() > 0 else 0.0,
        "grpo_route_cost_pairwise_loss": float(pairwise_losses.detach().mean().cpu().item()),
    }
    valid_counts = route_valid.sum(dim=-1)
    if bool((valid_counts > 1).any().detach().cpu().item()):
        top2 = torch.topk(masked_costs, k=2, largest=False, dim=-1).values
        margin_t = (top2[:, 1] - top2[:, 0])[valid_counts > 1]
        stats["grpo_route_cost_margin"] = float(margin_t.mean().detach().cpu().item()) if margin_t.numel() > 0 else 0.0
    else:
        stats["grpo_route_cost_margin"] = 0.0
    return per_agent_loss.mean(), stats


def _dreamer_action_route_probs(
    logits: torch.Tensor,
    *,
    mode: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return route probabilities used for the imagined environment action.

    ``expected`` preserves the old fully-soft action. Straight-through modes
    use a discrete forward route choice to match SUMO execution while keeping
    the softmax gradient for actor learning.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"st_sample", "straight_through_stochastic"}:
        mode_norm = "straight_through_sample"
    elif mode_norm in {"st_argmax", "straight_through_deterministic"}:
        mode_norm = "straight_through_argmax"
    if mode_norm not in {"expected", "straight_through_sample", "straight_through_argmax"}:
        raise ValueError(f"unsupported dreamer_action_mode={mode!r}")

    temp = max(float(temperature), 1e-6)
    soft_probs = torch.softmax(logits / temp, dim=-1)
    if mode_norm == "expected":
        return soft_probs

    if mode_norm == "straight_through_sample":
        idx = torch.multinomial(soft_probs.detach().clamp_min(0.0), num_samples=1).squeeze(-1)
    else:
        idx = torch.argmax(soft_probs.detach(), dim=-1)
    hard_probs = torch.zeros_like(soft_probs)
    hard_probs.scatter_(1, idx.unsqueeze(-1), 1.0)
    return hard_probs + soft_probs - soft_probs.detach()


def _valid_route_mask_from_paths(
    path_batch: List[List[List[int]]],
    *,
    n_routes: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((len(path_batch), int(n_routes)), device=device, dtype=torch.bool)
    for i, paths in enumerate(path_batch):
        k_eff = min(len(paths), int(n_routes))
        for k in range(k_eff):
            if paths[k]:
                mask[i, k] = True
    return mask


def _sample_masked_route_actions(
    logits: torch.Tensor,
    route_valid: torch.Tensor,
    *,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample route actions from valid candidates and return hard probs/log-probs.

    The hard one-hot route probabilities drive the imagined environment.  The
    selected log-probabilities are used by rollout-level GRPO, keeping the
    training signal stochastic instead of differentiating through argmax.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if route_valid.shape != logits.shape:
        raise ValueError(f"route_valid shape {tuple(route_valid.shape)} must match logits {tuple(logits.shape)}")
    large = torch.finfo(logits.dtype).max / 4.0
    valid = route_valid.to(device=logits.device)
    agent_has_valid = valid.any(dim=-1)
    safe_valid = valid.clone()
    if not bool(agent_has_valid.all().detach().cpu().item()):
        safe_valid[~agent_has_valid, 0] = True
    masked_logits = (logits / float(max(temperature, 1e-6))).masked_fill(~safe_valid, -large)
    dist = torch.distributions.Categorical(logits=masked_logits)
    selected = dist.sample()
    log_probs = dist.log_prob(selected)
    hard_probs = torch.zeros_like(logits)
    hard_probs.scatter_(1, selected.unsqueeze(-1), 1.0)
    log_probs = log_probs.masked_fill(~agent_has_valid, 0.0)
    return hard_probs, selected, log_probs


def _masked_route_log_probs(
    logits: torch.Tensor,
    route_valid: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Log probabilities over valid route candidates."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if route_valid.shape != logits.shape:
        raise ValueError(f"route_valid shape {tuple(route_valid.shape)} must match logits {tuple(logits.shape)}")
    large = torch.finfo(logits.dtype).max / 4.0
    valid = route_valid.to(device=logits.device)
    safe_valid = valid.clone()
    agent_has_valid = safe_valid.any(dim=-1)
    if not bool(agent_has_valid.all().detach().cpu().item()):
        safe_valid[~agent_has_valid, 0] = True
    masked_logits = (logits / float(max(temperature, 1e-6))).masked_fill(~safe_valid, -large)
    return torch.log_softmax(masked_logits, dim=-1)


def _selected_route_cost_advantages(
    *,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
    selected_idx: torch.Tensor,
    temperature: float,
    normalize_advantages: bool,
) -> torch.Tensor:
    """Return per-sampled-route relative route-cost advantages."""
    if route_costs.dim() != 2:
        raise ValueError(f"route_costs must be 2D, got {tuple(route_costs.shape)}")
    if route_valid.shape != route_costs.shape:
        raise ValueError(f"route_valid shape {tuple(route_valid.shape)} must match route_costs {tuple(route_costs.shape)}")
    n = min(int(route_costs.shape[0]), int(route_valid.shape[0]), int(selected_idx.reshape(-1).shape[0]))
    if n <= 0:
        return torch.zeros((0,), device=route_costs.device, dtype=route_costs.dtype)
    k = int(route_costs.shape[1])
    costs = route_costs[:n]
    valid = route_valid[:n].to(device=costs.device)
    selected = selected_idx.reshape(-1)[:n].to(device=costs.device, dtype=torch.long).clamp(0, max(k - 1, 0))
    rows = torch.arange(n, device=costs.device)
    valid_f = valid.to(dtype=costs.dtype)
    valid_count = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    cost_sum = (costs.masked_fill(~valid, 0.0) * valid_f).sum(dim=-1, keepdim=True)
    group_mean = cost_sum / valid_count
    selected_cost = costs[rows, selected]
    adv = group_mean.squeeze(-1) - selected_cost
    if bool(normalize_advantages):
        centered = (costs - group_mean).masked_fill(~valid, 0.0)
        group_std = torch.sqrt((centered.square() * valid_f).sum(dim=-1) / valid_count.squeeze(-1)).clamp_min(1e-6)
        adv = adv / group_std
    else:
        adv = adv / float(max(temperature, 1e-6))
    selected_valid = valid[rows, selected]
    return adv.masked_fill(~selected_valid, 0.0)


def _combined_grpo_policy_loss(
    *,
    logits: torch.Tensor,
    selected_idx: torch.Tensor,
    old_log_probs: torch.Tensor,
    route_valid: torch.Tensor,
    route_advantages: Optional[torch.Tensor],
    rollout_advantage: Union[float, torch.Tensor] = 0.0,
    route_weight: float = 1.0,
    rollout_weight: float = 0.0,
    temperature: float = 1.0,
    policy_clip_ratio: float = 0.2,
    advantage_clip: float = 5.0,
    kl_coef: float = 0.0,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """PPO-style policy-gradient loss on route-cost plus rollout advantages.

    The selected action and old log-probability come from the imagined rollout.
    The current log-probability is recomputed from the actor during update, so
    route-cost and rollout signals share one on-policy score-function update
    instead of being applied as separate auxiliary losses.
    """
    empty_stats = {
        "grpo_combined_loss": 0.0,
        "grpo_combined_pg_loss": 0.0,
        "grpo_combined_kl_loss": 0.0,
        "grpo_combined_adv_mean": 0.0,
        "grpo_combined_adv_std": 0.0,
        "grpo_combined_clip_frac": 0.0,
        "grpo_policy_kl": 0.0,
    }
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if route_valid.shape != logits.shape:
        raise ValueError(f"route_valid shape {tuple(route_valid.shape)} must match logits {tuple(logits.shape)}")

    n = min(
        int(logits.shape[0]),
        int(route_valid.shape[0]),
        int(selected_idx.reshape(-1).shape[0]),
        int(old_log_probs.reshape(-1).shape[0]),
    )
    if route_advantages is not None:
        n = min(n, int(route_advantages.reshape(-1).shape[0]))
    if n <= 0:
        return None, empty_stats

    logits = logits[:n]
    valid = route_valid[:n].to(device=logits.device)
    selected = selected_idx.reshape(-1)[:n].to(device=logits.device, dtype=torch.long)
    selected = selected.clamp(0, max(int(logits.shape[1]) - 1, 0))
    old_logp = old_log_probs.reshape(-1)[:n].to(device=logits.device, dtype=logits.dtype).detach()
    if route_advantages is None:
        route_adv = torch.zeros((n,), device=logits.device, dtype=logits.dtype)
    else:
        route_adv = route_advantages.reshape(-1)[:n].to(device=logits.device, dtype=logits.dtype).detach()

    rows = torch.arange(n, device=logits.device)
    selected_valid = valid[rows, selected]
    current_logp_all = _masked_route_log_probs(
        logits,
        valid,
        temperature=float(max(temperature, 1e-6)),
    )
    current_logp = current_logp_all[rows, selected]
    rollout_adv = torch.as_tensor(
        rollout_advantage,
        device=logits.device,
        dtype=logits.dtype,
    ).reshape(-1)
    rollout_scalar = rollout_adv[0] if int(rollout_adv.numel()) > 0 else torch.zeros((), device=logits.device, dtype=logits.dtype)
    combined_adv = (
        float(route_weight) * route_adv
        + float(rollout_weight) * rollout_scalar.detach()
    )
    finite = (
        selected_valid
        & torch.isfinite(current_logp)
        & torch.isfinite(old_logp)
        & torch.isfinite(combined_adv)
    )
    if not bool(finite.any().detach().cpu().item()):
        return None, empty_stats

    current_logp = current_logp[finite]
    old_logp = old_logp[finite]
    combined_adv = combined_adv[finite].detach()
    adv_clip = float(advantage_clip)
    if adv_clip > 0.0:
        combined_adv = combined_adv.clamp(min=-adv_clip, max=adv_clip)

    log_ratio = current_logp - old_logp
    ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
    clip_eps = float(max(policy_clip_ratio, 0.0))
    if clip_eps > 0.0:
        clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
        pg_terms = torch.minimum(ratio * combined_adv, clipped_ratio * combined_adv)
        clip_frac = (torch.abs(ratio - 1.0) > clip_eps).to(dtype=logits.dtype).mean()
    else:
        pg_terms = ratio * combined_adv
        clip_frac = torch.zeros((), device=logits.device, dtype=logits.dtype)
    pg_loss = -pg_terms.mean()
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    kl_loss = float(max(kl_coef, 0.0)) * approx_kl
    loss = pg_loss + kl_loss

    stats = {
        "grpo_combined_loss": float(loss.detach().cpu().item()),
        "grpo_combined_pg_loss": float(pg_loss.detach().cpu().item()),
        "grpo_combined_kl_loss": float(kl_loss.detach().cpu().item()),
        "grpo_combined_adv_mean": float(combined_adv.detach().mean().cpu().item()),
        "grpo_combined_adv_std": float(
            combined_adv.detach().std(unbiased=False).cpu().item()
        ) if int(combined_adv.numel()) > 1 else 0.0,
        "grpo_combined_clip_frac": float(clip_frac.detach().cpu().item()),
        "grpo_policy_kl": float(approx_kl.detach().cpu().item()),
    }
    return loss, stats


def _rollout_group_relative_advantages(
    costs: Union[torch.Tensor, List[float]],
    group_ids: Optional[List[int]],
    *,
    normalize: bool,
) -> torch.Tensor:
    """Return cost-minimizing advantages, preferably within same-start groups."""
    costs_t = torch.as_tensor(costs, dtype=torch.float32)
    if costs_t.numel() == 0:
        return costs_t
    if group_ids is None or len(group_ids) != int(costs_t.numel()):
        group_ids = [0 for _ in range(int(costs_t.numel()))]

    advantages = torch.zeros_like(costs_t)
    used_group_baseline = False
    for gid in sorted(set(int(g) for g in group_ids)):
        idx = [i for i, g in enumerate(group_ids) if int(g) == gid]
        if len(idx) < 2:
            continue
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=costs_t.device)
        group_costs = costs_t.index_select(0, idx_t)
        group_adv = group_costs.mean() - group_costs
        if normalize:
            group_adv = group_adv / group_costs.std(unbiased=False).clamp_min(1e-6)
        advantages.index_copy_(0, idx_t, group_adv)
        used_group_baseline = True

    if used_group_baseline:
        return advantages

    advantages = costs_t.mean() - costs_t
    if normalize and int(costs_t.numel()) > 1:
        advantages = advantages / costs_t.std(unbiased=False).clamp_min(1e-6)
    return advantages


def _rollout_baseline_relative_advantages(
    costs: Union[torch.Tensor, List[float]],
    baseline_costs: Union[torch.Tensor, List[float]],
    group_ids: Optional[List[int]],
    *,
    normalize: bool,
) -> torch.Tensor:
    """Return advantages against same-state counterfactual rollout costs."""
    costs_t = torch.as_tensor(costs, dtype=torch.float32)
    baseline_t = torch.as_tensor(baseline_costs, dtype=torch.float32)
    if costs_t.shape != baseline_t.shape:
        raise ValueError(f"costs shape {tuple(costs_t.shape)} must match baseline {tuple(baseline_t.shape)}")
    if costs_t.numel() == 0:
        return costs_t

    advantages = baseline_t - costs_t
    if not normalize:
        return advantages
    if group_ids is None or len(group_ids) != int(costs_t.numel()):
        scale = costs_t.std(unbiased=False).clamp_min(1e-6) if int(costs_t.numel()) > 1 else torch.ones_like(costs_t)
        return advantages / scale

    scaled = torch.zeros_like(advantages)
    for gid in sorted(set(int(g) for g in group_ids)):
        idx = [i for i, g in enumerate(group_ids) if int(g) == gid]
        if not idx:
            continue
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=costs_t.device)
        group_costs = costs_t.index_select(0, idx_t)
        if len(idx) > 1:
            scale = group_costs.std(unbiased=False).clamp_min(1e-6)
        else:
            scale = torch.ones((), dtype=costs_t.dtype, device=costs_t.device)
        scaled.index_copy_(0, idx_t, advantages.index_select(0, idx_t) / scale)
    return scaled


def _sequential_planned_pressure_greedy_action_probs(
    *,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
    path_batch: List[List[List[int]]],
    n_edges: int,
    n_routes: int,
    pressure_weight: float,
) -> torch.Tensor:
    """Greedily select route-cost routes with sequential planned-AV pressure."""
    if route_costs.dim() != 2:
        raise ValueError(f"route_costs must be 2D, got {tuple(route_costs.shape)}")
    if route_valid.shape != route_costs.shape:
        raise ValueError(f"route_valid shape {tuple(route_valid.shape)} must match route_costs {tuple(route_costs.shape)}")
    n_agents = min(int(route_costs.shape[0]), len(path_batch))
    k_eff = min(int(route_costs.shape[1]), int(n_routes))
    action_probs = torch.zeros_like(route_costs)
    if n_agents <= 0 or k_eff <= 0:
        return action_probs

    device = route_costs.device
    dtype = route_costs.dtype
    edge_pressure = torch.zeros((max(int(n_edges), 0),), device=device, dtype=dtype)
    norm = float(max(n_agents, 1))
    large = torch.finfo(dtype).max / 4.0
    penalty = max(float(pressure_weight), 0.0)

    for i in range(n_agents):
        valid_i = route_valid[i, :k_eff].to(device=device)
        if not bool(valid_i.any().detach().cpu().item()):
            continue
        row = route_costs[i, :k_eff].to(device=device, dtype=dtype).clone()
        if penalty > 0.0 and int(edge_pressure.numel()) > 0:
            routes = path_batch[i] if i < len(path_batch) else []
            for k in range(min(k_eff, len(routes))):
                if not bool(valid_i[k].detach().cpu().item()):
                    continue
                idxs = sorted({int(e) for e in routes[k] if 0 <= int(e) < int(edge_pressure.numel())})
                if idxs:
                    edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                    row[k] = row[k] + penalty * edge_pressure.index_select(0, edge_idx).sum() / norm
        masked = row.masked_fill(~valid_i, large)
        choice = int(torch.argmin(masked).detach().cpu().item())
        action_probs[i, choice] = 1.0
        routes = path_batch[i] if i < len(path_batch) else []
        if choice < len(routes) and int(edge_pressure.numel()) > 0:
            idxs = sorted({int(e) for e in routes[choice] if 0 <= int(e) < int(edge_pressure.numel())})
            if idxs:
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                edge_pressure.index_add_(0, edge_idx, torch.ones((len(idxs),), device=device, dtype=dtype))
    return action_probs


def _standardize_with_detached_stats(
    values: torch.Tensor,
    *,
    center: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize values while keeping gradients through the original tensor."""
    if values.numel() <= 1:
        return values
    ref = values.detach()
    std = ref.std(unbiased=False).clamp_min(float(eps))
    if bool(center):
        return (values - ref.mean()) / std
    return values / std


def _mean_path_invalid_rate(
    path_batch: List[List[List[int]]],
    n_routes: int,
) -> float:
    total = 0
    invalid = 0
    for agent_paths in path_batch:
        k_eff = min(len(agent_paths), int(n_routes))
        for k in range(k_eff):
            total += 1
            if not agent_paths[k]:
                invalid += 1
    if total <= 0:
        return 0.0
    return float(invalid) / float(total)


def _route_prob_churn(
    prev_probs: Optional[torch.Tensor],
    cur_probs: torch.Tensor,
) -> float:
    if prev_probs is None:
        return 0.0
    n = min(int(prev_probs.shape[0]), int(cur_probs.shape[0]))
    if n <= 0:
        return 0.0
    churn = (cur_probs[:n].detach() - prev_probs[:n]).abs().mean().item()
    return float(churn)


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
    direct_reward_requires_dynamics_grad = any(
        float(getattr(cfg, name, 0.0)) != 0.0
        for name in (
            "dreamer_all_reward_weight",
            "dreamer_remaining_time_progress_weight",
            "dreamer_current_edge_reward_weight",
            "dreamer_cav_reward_weight",
            "dreamer_route_cost_reward_weight",
            "dreamer_pred_t_cav_proxy_reward_weight",
            "dreamer_action_congestion_weight",
            "dreamer_route_balance_weight",
            "dreamer_route_max_share_weight",
            "dreamer_horizon_proxy_reward_weight",
        )
    )

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
            active_avs.append({**veh, "depart_step": step})
        for veh in dep.get("hdv_vehicles", []):
            active_hdvs.append({**veh, "depart_step": step})

        max_life = int(avg_route_steps * 1.5)
        active_avs = [av for av in active_avs if (step - av["depart_step"]) < max_life]
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
                        )
                        and rollout_policy == "actor"
                    ):
                        action_i, selected_i_t, logprob_i = _sample_masked_route_actions(
                            logits_i,
                            valid_i,
                            temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                        )
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
                )
                and rollout_policy == "actor"
            )
            selected_log_probs = None
            selected_route_idx_t_for_update = None
            selected_route_idx_for_env = None
            if sequential_action_route_probs is not None and rollout_policy == "actor":
                action_route_probs = sequential_action_route_probs
                if sequential_selected_route_idx_t is not None:
                    selected_route_idx_t_for_update = sequential_selected_route_idx_t
                    selected_route_idx_for_env = sequential_selected_route_idx_t.detach().cpu().numpy().astype(np.int64)
                if sequential_selected_log_probs is not None:
                    valid_action_mask = _valid_route_mask_from_paths(path_batch, n_routes=K, device=device)
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
                action_route_probs, selected_route_idx_t, selected_log_probs = _sample_masked_route_actions(
                    logits,
                    valid_action_mask,
                    temperature=float(getattr(cfg, "dreamer_action_temperature", 1.0)),
                )
                selected_route_idx_t_for_update = selected_route_idx_t
                selected_route_idx_for_env = selected_route_idx_t.detach().cpu().numpy().astype(np.int64)
                valid_selected = valid_action_mask.any(dim=-1)
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
                next_edge = choose_vehicle_next_edge(int(current_edges_np[i]), chosen_path)
                av["current_edge"] = int(next_edge)
                av["last_route_idx"] = int(k_sel)
                av["arrived"] = int(next_edge) == int(av.get("dest_edge", next_edge))
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
        depart_t = torch.from_numpy(depart_np).to(device=device, dtype=z.dtype)
        past_packed = build_history_packed(latent_history, history_length=cfg.history_length)
        hdv_context_for_wm = None
        if bool(getattr(wm, "use_hdv_context", False)) and hdv_background_edge is not None:
            hdv_context_for_wm = hdv_background_edge.to(device=device, dtype=z.dtype).clamp_min(0.0)
            hdv_context_for_wm = hdv_context_for_wm / hdv_context_for_wm.sum().clamp_min(1e-8)
            hdv_context_for_wm = hdv_context_for_wm * float(getattr(cfg, "dreamer_hdv_background_weight", 1.0))
            action_edge_for_wm = action_edge
        else:
            action_edge_for_wm = combine_av_hdv_edge_background(
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
            route_cost_reward_weight=float(cfg.dreamer_route_cost_reward_weight),
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
        if bool(reward_t.requires_grad):
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
        "hdv_background_sum": float(hdv_background_edge.detach().sum().cpu().item()) if hdv_background_edge is not None else 0.0,
        "hdv_background_weight": float(getattr(cfg, "dreamer_hdv_background_weight", 0.0)) if hdv_background_edge is not None else 0.0,
        "terminal_t_all_pred": wm.predict_t_all_from_latent(z.detach()),
        "terminal_bonus": 0.0,
        "start_step": s,
    }
    trajectory = {
        "traj_id": int(traj_id) if traj_id is not None else int(s),
        "z_states": z_states,
        "action_edges": action_edges,
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
    }
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
) -> Dict[str, float]:
    actor_losses: List[torch.Tensor] = []
    critic_losses: List[torch.Tensor] = []
    q_actor_items: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = []
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
    if actor_objective not in {"direct_return", "critic_q", "direct_plus_q"}:
        raise ValueError(f"Unsupported dreamer_actor_objective={actor_objective!r}")

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

        values = critic(current_z, action=current_action).squeeze(-1)
        with torch.no_grad():
            next_values = critic(next_z, action=next_action).squeeze(-1)
            critic_targets = lambda_return(
                rewards.detach(),
                next_values,
                gamma=float(cfg.gamma),
                lambda_=float(cfg.gae_lambda),
            )
        if bool(getattr(cfg, "dreamer_normalize_critic_targets", False)):
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

        actor_bootstrap = critic(next_z, action=next_action).squeeze(-1).detach()
        actor_returns = lambda_return(
            rewards,
            actor_bootstrap,
            gamma=float(cfg.gamma),
            lambda_=float(cfg.gae_lambda),
        )
        actor_advantages = actor_returns - values.detach()
        if bool(getattr(cfg, "dreamer_normalize_actor_advantages", False)):
            actor_advantages = _standardize_with_detached_stats(actor_advantages, center=True)
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
        grpo_policy_objective = str(
            getattr(cfg, "dreamer_grpo_policy_objective", "combined")
        ).strip().lower()
        if grpo_policy_objective not in {"combined", "legacy_separate"}:
            raise ValueError(
                "dreamer_grpo_policy_objective must be one of {'combined', 'legacy_separate'}, "
                f"got {grpo_policy_objective!r}"
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
        direct_actor_loss = -(actor_advantages.mean() + float(cfg.dreamer_entropy_coef) * entropy)
        actor_loss = direct_actor_loss
        if route_cost_aux_loss is not None:
            actor_loss = actor_loss + float(cfg.dreamer_route_cost_aux_weight) * route_cost_aux_loss
        if grpo_policy_objective == "legacy_separate":
            grpo_route_cost_tensor = traj.get("grpo_route_cost_losses")
            if (
                float(getattr(cfg, "dreamer_grpo_route_cost_weight", 0.0)) != 0.0
                and torch.is_tensor(grpo_route_cost_tensor)
                and grpo_route_cost_tensor.numel() > 0
            ):
                actor_loss = (
                    actor_loss
                    + float(getattr(cfg, "dreamer_grpo_route_cost_weight", 0.0))
                    * grpo_route_cost_tensor.mean()
                )
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
        if (direct_actor_has_grad or q_actor_has_grad or grpo_policy_has_samples) and not low_valid_agent_coverage and not actor_update_disabled:
            if use_direct_actor:
                actor_losses.append(actor_loss)
            if q_actor_has_grad:
                q_actor_items.append((
                    current_z,
                    current_action_for_actor,
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

    grpo_policy_objective = str(
        getattr(cfg, "dreamer_grpo_policy_objective", "combined")
    ).strip().lower()
    if grpo_policy_objective not in {"combined", "legacy_separate"}:
        raise ValueError(
            "dreamer_grpo_policy_objective must be one of {'combined', 'legacy_separate'}, "
            f"got {grpo_policy_objective!r}"
        )

    if grpo_policy_items and grpo_policy_objective == "combined":
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
    elif grpo_policy_items and grpo_policy_objective == "legacy_separate":
        rollout_weight = float(getattr(cfg, "dreamer_grpo_rollout_weight", 0.0))
        if rollout_weight != 0.0:
            legacy_rollout_losses: List[torch.Tensor] = []
            for policy_idx, item in enumerate(grpo_policy_items):
                rollout_logprob = item.get("rollout_logprob")
                if not torch.is_tensor(rollout_logprob) or rollout_logprob.numel() <= 0:
                    continue
                adv = float(rollout_adv_by_policy_idx.get(int(policy_idx), 0.0))
                legacy_rollout_losses.append(
                    -rollout_weight
                    * torch.as_tensor(adv, device=rollout_logprob.device, dtype=rollout_logprob.dtype)
                    * rollout_logprob.reshape(-1).mean()
                )
            if legacy_rollout_losses:
                actor_losses.append(torch.stack(legacy_rollout_losses).mean())

    if not critic_losses:
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

    critic_optimizer.zero_grad()
    critic_loss_total = torch.stack(critic_losses).mean()
    critic_loss_total.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_optimizer.step()

    if q_actor_items:
        old_requires_grad = [p.requires_grad for p in critic.parameters()]
        for p in critic.parameters():
            p.requires_grad_(False)
        try:
            for current_z, current_action_for_actor, entropy, route_cost_aux_loss in q_actor_items:
                q_values = critic(current_z, action=current_action_for_actor).squeeze(-1)
                q_actor_loss = -float(getattr(cfg, "dreamer_actor_q_weight", 1.0)) * q_values.mean()
                q_actor_loss = q_actor_loss - float(cfg.dreamer_entropy_coef) * entropy
                if route_cost_aux_loss is not None:
                    q_actor_loss = q_actor_loss + float(cfg.dreamer_route_cost_aux_weight) * route_cost_aux_loss
                actor_losses.append(q_actor_loss)
        finally:
            for p, requires_grad in zip(critic.parameters(), old_requires_grad):
                p.requires_grad_(requires_grad)

    if actor_losses:
        actor_optimizer.zero_grad()
        actor_loss_total = torch.stack(actor_losses).mean()
        actor_loss_total.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optimizer.step()
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


def save_dreamer_checkpoint(
    cfg,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    episode,
    stats,
    extra_state=None,
) -> str:
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    obj = {
        "episode": episode,
        "model_family": "dreamer_ctde",
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "policy": actor.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "stats": stats,
        "cfg": cfg.__dict__,
        "extra_state": extra_state or {},
    }
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_ep{episode:05d}.pt")
    torch.save(obj, path)
    torch.save(obj, os.path.join(cfg.checkpoint_dir, "latest.pt"))
    return path


def load_dreamer_checkpoint(
    path,
    actor,
    critic,
    actor_optimizer=None,
    critic_optimizer=None,
    *,
    reset_actor_optimizer: bool = False,
):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    actor_state = ckpt.get("actor", ckpt.get("policy"))
    if actor_state is None:
        raise RuntimeError(f"checkpoint has no actor/policy state: {path}")
    actor.load_state_dict(actor_state)
    critic_state = ckpt.get("critic")
    if critic_state is not None:
        critic.load_state_dict(critic_state)
    if actor_optimizer is not None and "actor_optimizer" in ckpt and not reset_actor_optimizer:
        actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
    if critic_optimizer is not None and "critic_optimizer" in ckpt:
        critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
    return ckpt.get("episode", 0), ckpt.get("stats", {}), dict(ckpt.get("extra_state", {}) or {})


def _move_nested_tensors(value, *, device=None, cpu: bool = False):
    if torch.is_tensor(value):
        target = "cpu" if cpu else device
        return value.detach().to(target) if target is not None else value.detach()
    if isinstance(value, list):
        return [_move_nested_tensors(item, device=device, cpu=cpu) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested_tensors(item, device=device, cpu=cpu) for item in value)
    if isinstance(value, dict):
        return {
            key: _move_nested_tensors(item, device=device, cpu=cpu)
            for key, item in value.items()
        }
    return value


def _restore_dreamer_extra_state(extra_state, *, device):
    extra_state = dict(extra_state or {})
    pending_trajs = [
        _move_nested_tensors(traj, device=device, cpu=False)
        for traj in list(extra_state.get("pending_trajs", []))
    ]
    for traj in pending_trajs:
        traj["restored_from_checkpoint"] = True
    update_id = int(extra_state.get("update_id", 0))
    return pending_trajs, update_id


def count_fresh_pending_trajs(pending_trajs):
    return int(sum(0 if bool(traj.get("restored_from_checkpoint", False)) else 1 for traj in pending_trajs))


def capture_rng_state(*, rng):
    state = {
        "numpy_bit_generator_state": copy.deepcopy(rng.bit_generator.state),
        "python_random_state": copy.deepcopy(random.getstate()),
        "torch_cpu_rng_state": torch.random.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(extra_state):
    extra_state = dict(extra_state or {})
    np_state = extra_state.get("numpy_bit_generator_state")
    py_state = extra_state.get("python_random_state")
    torch_cpu_state = extra_state.get("torch_cpu_rng_state")
    torch_cuda_states = extra_state.get("torch_cuda_rng_state_all")
    rng = None
    if np_state is not None:
        rng = np.random.default_rng()
        rng.bit_generator.state = np_state
    if py_state is not None:
        random.setstate(py_state)
    if torch_cpu_state is not None:
        torch.random.set_rng_state(torch_cpu_state)
    if torch_cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_states)
    return rng


def normalize_stats_history_defaults(stats_history):
    stats_history = dict(stats_history or {})
    scalar_series = (
        "ep_reward", "pg_loss", "value_loss", "entropy", "approx_kl",
        "t_av", "t_hdv", "t_all", "hdv_drift", "hdv_head_entropy",
        "imagined_eval_t_all", "discrepancy_t_all", "adaptive_penalty",
        "terminal_t_all_pred", "terminal_bonus", "invalid_route_rate", "route_churn",
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
        "hdv_response_av_pressure_sum", "hdv_response_background_sum", "hdv_response_background_active_edges",
        "avg_active_av", "skipped_actor_trajs", "n_valid_paths", "n_valid_path_agents",
        "n_actor_steps", "reward_grad_steps", "entropy_grad_steps", "max_active_av",
    )
    for key in scalar_series:
        stats_history.setdefault(key, [])
    stats_history.setdefault("traj_diagnostics", [])
    stats_history.setdefault("update_diagnostics", [])
    return stats_history


def maybe_resume_dreamer_checkpoint(
    *,
    resume_ckpt,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    device,
    reset_actor_optimizer=False,
):
    if not resume_ckpt or not os.path.exists(resume_ckpt):
        return None
    ckpt_state = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
    if ckpt_state.get("model_family") != "dreamer_ctde" and "actor" not in ckpt_state:
        return None
    actor_state_before = copy.deepcopy(actor.state_dict())
    critic_state_before = copy.deepcopy(critic.state_dict())
    actor_opt_state_before = copy.deepcopy(actor_optimizer.state_dict()) if actor_optimizer is not None else None
    critic_opt_state_before = copy.deepcopy(critic_optimizer.state_dict()) if critic_optimizer is not None else None
    try:
        start_episode, stats_history, extra_state = load_dreamer_checkpoint(
            resume_ckpt,
            actor,
            critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            reset_actor_optimizer=reset_actor_optimizer,
        )
        logger.info("Resumed Dreamer checkpoint from ep %d", start_episode)
        if reset_actor_optimizer:
            logger.info("Actor optimizer state reset on resume; actor weights were still loaded.")
        stats_history = normalize_stats_history_defaults(stats_history)
        pending_trajs, update_id = _restore_dreamer_extra_state(extra_state, device=device)
        return {
            "start_episode": int(start_episode),
            "stats_history": stats_history,
            "pending_trajs": pending_trajs,
            "update_id": int(update_id),
            "restored_rng": restore_rng_state(extra_state),
        }
    except (RuntimeError, ValueError) as exc:
        actor.load_state_dict(actor_state_before)
        critic.load_state_dict(critic_state_before)
        if actor_optimizer is not None and actor_opt_state_before is not None:
            actor_optimizer.load_state_dict(actor_opt_state_before)
        if critic_optimizer is not None and critic_opt_state_before is not None:
            critic_optimizer.load_state_dict(critic_opt_state_before)
        logger.warning(
            "Failed to resume Dreamer checkpoint %s; starting fresh. Error: %s",
            resume_ckpt,
            exc,
        )
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    cfg,
    policy,
    optimizer,
    episode,
    stats,
    hdv_fracs_state: Optional[np.ndarray] = None,
    hdv_costs_state: Optional[np.ndarray] = None,
) -> str:
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    obj = {
        "episode":   episode,
        "policy":    policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stats":     stats,
        "cfg":       cfg.__dict__,
        "hdv_fracs_state": None if hdv_fracs_state is None else np.asarray(hdv_fracs_state, dtype=np.float32),
        "hdv_costs_state": None if hdv_costs_state is None else np.asarray(hdv_costs_state, dtype=np.float32),
    }
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_ep{episode:05d}.pt")
    torch.save(obj, path)
    torch.save(obj, os.path.join(cfg.checkpoint_dir, "latest.pt"))
    return path


def load_checkpoint(path, policy, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    hdv_state = ckpt.get("hdv_fracs_state")
    if hdv_state is not None:
        hdv_state = np.asarray(hdv_state, dtype=np.float32)
    hdv_cost_state = ckpt.get("hdv_costs_state")
    if hdv_cost_state is not None:
        hdv_cost_state = np.asarray(hdv_cost_state, dtype=np.float32)
    return ckpt.get("episode", 0), ckpt.get("stats", {}), hdv_state, hdv_cost_state


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg: TrainConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

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
        stackelberg_cfg = {
            "enabled": True,
            "n_iters": int(cfg.stackelberg_n_iters),
            "damping": float(cfg.stackelberg_damping),
            "total_av_count": float(cfg.n_av),
            "total_hdv_count": float(cfg.n_hdv),
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

    # ── Detect vehicle-level network mode ────────────────────────────────────
    mean_field = bool(cfg.network_folder)
    agent_level = str(getattr(cfg, "agent_level", "vehicle")).lower()
    imagination_mode = select_imagination_mode(mean_field, agent_level)
    vehicle_mode = imagination_mode == "vehicle_dynamic"
    training_family = str(getattr(cfg, "training_family", "dreamer_ctde")).strip().lower()
    route_registry: Optional[PathTopology] = None

    if mean_field:
        if cfg.network_folder and cfg.network_name:
            import ast as _ast
            od_path = os.path.join(cfg.network_folder, f"od_{cfg.network_name}.txt")
            with open(od_path, "r") as f:
                od_data = _ast.literal_eval(f.read())
            agents_path = cfg.agents_csv or os.path.join(cfg.network_folder, "agents.csv")
            if vehicle_mode:
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
                logger.info(
                    "Built lightweight vehicle path topology: N_od=%d, K=%d, AVs=%d, HDVs=%d, route_edges=%d",
                    route_registry.n_od,
                    route_registry.K,
                    int(route_registry.od_av_counts.sum()),
                    int(route_registry.od_hdv_counts.sum()),
                    int(np.count_nonzero(route_registry.route_masks)),
                )
            else:
                raise RuntimeError("Route/OD-level imagination has been removed; use vehicle-level training.")
        else:
            raise ValueError("Vehicle-level training requires --network_folder and --network_name.")

        if vehicle_mode:
            obs_dim = vehicle_mode_obs_dim(
                n_routes=cfg.n_routes,
                d_obs=wm.d_obs,
                include_global=bool(cfg.vehicle_obs_include_global),
                route_stats=str(cfg.vehicle_obs_route_stats),
                include_route_costs=bool(cfg.vehicle_obs_include_route_costs),
                include_fleet_pressure=bool(getattr(cfg, "vehicle_obs_include_fleet_pressure", False)),
            )
            logger.info(
                "Vehicle mode: obs_dim=%d, N_av=%d, K=%d, global_obs=%s, route_stats=%s, route_cost_obs=%s, fleet_pressure_obs=%s",
                obs_dim,
                int(cfg.n_av),
                route_registry.K,
                bool(cfg.vehicle_obs_include_global),
                str(cfg.vehicle_obs_route_stats),
                bool(cfg.vehicle_obs_include_route_costs),
                bool(getattr(cfg, "vehicle_obs_include_fleet_pressure", False)),
            )
            wm_action_dim = int(getattr(wm, "action_dim", -1))
            raw_action_dim = int(getattr(wm, "raw_action_dim", -1) or -1)
            comp_in = int(getattr(wm, "action_compressor_input_dim", -1) or -1)
            comp_out = int(getattr(wm, "action_compressor_output_dim", -1) or -1)
            action_ok = vehicle_mode_action_supported(
                n_edges=int(wm.n_edges),
                action_dim=wm_action_dim,
                raw_action_dim=raw_action_dim,
                action_compressor_input_dim=comp_in,
                action_compressor_output_dim=comp_out,
            )
            if not action_ok:
                raise RuntimeError(
                    "agent_level=vehicle requires raw edge action support. "
                    f"Expected raw_action_dim==n_edges ({wm.n_edges}) or compatible compressor input, "
                    f"got action_dim={wm_action_dim}, raw_action_dim={raw_action_dim}, "
                    f"compressor_in={comp_in}, compressor_out={comp_out}."
                )
        else:
            raise RuntimeError("Route/OD-level imagination has been removed; use agent_level=vehicle.")

        if training_family == "dreamer_ctde":
            if not vehicle_mode:
                raise RuntimeError("dreamer_ctde currently supports vehicle-level training only.")
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
                hidden_dim=cfg.hidden_dim,
                n_hidden_layers=cfg.n_hidden_layers,
            ).to(device)
            freeze_module(wm)
            policy = actor
            optimizer = None
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
            logger.info(
                "Dreamer CTDE mainline active: actor_obs_dim=%d n_routes=%d d_spatial=%d batch_eps=%d actor_h=%d critic_variant=%s",
                obs_dim,
                int(cfg.n_routes),
                int(wm.d_spatial),
                int(cfg.dreamer_batch_episodes),
                min(int(cfg.imagination_horizon), int(cfg.dreamer_actor_horizon)),
                str(cfg.dreamer_critic_variant),
            )
        else:
            # Use MeanFieldRoutePolicy
            critic_obs_dim = obs_dim
            if vehicle_mode and str(cfg.critic_variant).lower() == "latent_global":
                critic_obs_dim = obs_dim + latent_global_summary_dim(
                    wm.d_spatial,
                    variant=str(cfg.critic_global_summary_variant),
                )
            policy = MeanFieldRoutePolicy(
                obs_dim=obs_dim,
                n_routes=cfg.n_routes,
                hidden_dim=cfg.hidden_dim,
                n_hidden_layers=cfg.n_hidden_layers,
                critic_obs_dim=critic_obs_dim,
            ).to(device)
            optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
            actor = None
            critic = None
            actor_optimizer = None
            critic_optimizer = None
            logger.info(
                "Policy critic_variant=%s critic_summary=%s critic_obs_dim=%d actor_obs_dim=%d critic_target_variant=%s target_h=%d training_family=%s",
                cfg.critic_variant,
                cfg.critic_global_summary_variant,
                critic_obs_dim,
                obs_dim,
                cfg.critic_target_variant,
                int(cfg.critic_target_horizon),
                training_family,
            )
    else:
        obs_dim = wm.n_edges * wm.d_obs
        logger.info(f"Single-agent mode: obs_dim={obs_dim}")
        if training_family == "dreamer_ctde":
            raise RuntimeError("dreamer_ctde currently supports vehicle-level training only.")
        policy = SharedPolicy(
            obs_dim=obs_dim,
            n_routes=cfg.n_routes,
            hidden_dim=cfg.hidden_dim,
            n_hidden_layers=cfg.n_hidden_layers,
            critic_obs_dim=obs_dim,
        ).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        actor = None
        critic = None
        actor_optimizer = None
        critic_optimizer = None

    # ── reward model (only used in single-agent mode) ─────────────────────────
    reward_model = None
    if not mean_field:
        if cfg.latent_reward_ckpt and os.path.exists(cfg.latent_reward_ckpt):
            reward_model = LatentRewardPredictor.from_checkpoint(cfg.latent_reward_ckpt, device)
        else:
            reward_model = TrafficMetricsRewardModel(use_latent=False)

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
    if training_family != "dreamer_ctde" and cfg.resume_ckpt and os.path.exists(cfg.resume_ckpt):
        try:
            start_episode, stats_history, resume_hdv_state, resume_hdv_cost_state = load_checkpoint(
                cfg.resume_ckpt, policy, optimizer
            )
            logger.info(f"Resumed from ep {start_episode}")
        except RuntimeError as exc:
            logger.warning(
                "Failed to resume checkpoint %s due to incompatible policy shape; "
                "starting fresh. Error: %s",
                cfg.resume_ckpt,
                exc,
            )
            resume_hdv_state = None
            resume_hdv_cost_state = None
    else:
        resume_hdv_state = None
        resume_hdv_cost_state = None
    stats_history = normalize_stats_history_defaults(stats_history)

    target_episode = int(cfg.n_total_episodes)
    if start_episode > 0 and target_episode <= start_episode:
        # Backward-compatible guard: many scripts pass n_total_episodes as a per-round
        # budget even when resuming from a larger absolute episode index.
        logger.warning(
            "n_total_episodes=%d <= resumed episode=%d; interpret as additional episodes "
            "(target episode -> %d).",
            target_episode,
            start_episode,
            start_episode + target_episode,
        )
        target_episode = start_episode + target_episode

    # ── training loop ─────────────────────────────────────────────────────────
    logger.info("Starting offline MARL (%s): ep %d -> %d", training_family, start_episode, target_episode)
    if vehicle_mode:
        logger.info(
            "  Vehicle mode: N_av=%d, K=%d, H=%d, episode_steps=%d, path_penalty=%.2f, reward_scale=%.3f",
            int(cfg.n_av), cfg.n_routes, int(cfg.imagination_horizon), int(cfg.episode_steps),
            float(cfg.vehicle_path_penalty_factor), float(cfg.vehicle_reward_scale),
        )
    elif mean_field:
        logger.info(
            f"  Mean field: N_od={route_registry.n_od}, K={cfg.n_routes}, T={cfg.episode_len}"
        )
        logger.info(
            "  reward_mode=%s, base_penalty=%.4f, discrepancy_threshold=%.3f",
            cfg.reward_mode, cfg.conservative_penalty_coef, cfg.discrepancy_threshold,
        )
    else:
        logger.info(f"  n_av={cfg.n_av}, n_hdv={cfg.n_hdv}, K={cfg.n_routes}, T={cfg.episode_len}")

    rew_running_mean = 0.0
    rew_running_std  = 1.0
    if not mean_field and isinstance(reward_model, LatentRewardPredictor):
        rew_running_mean = reward_model.r_mean
        rew_running_std  = reward_model.r_std
    rew_alpha = 0.05
    gawron_mode_imagined = imagined_gawron_mode(cfg.gawron_mode)
    use_sumo_synced_hdv_policy = bool(
        mean_field and (not cfg.hdv_internalized) and cfg.gawron_mode == "routerl_blackbox"
    )
    if vehicle_mode:
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
        "pg_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0
    }

    # Persistent HDV state for inter-episode Gawron update (mean field only)
    hdv_fracs_state: Optional[np.ndarray] = None
    hdv_costs_state: Optional[np.ndarray] = None
    if mean_field and not vehicle_mode:
        expected_shape = (route_registry.n_od, cfg.n_routes)
        if resume_hdv_state is not None and tuple(resume_hdv_state.shape) == expected_shape:
            hdv_fracs_state = np.asarray(resume_hdv_state, dtype=np.float32)
            hdv_fracs_state = np.clip(hdv_fracs_state, 1e-6, None)
            hdv_fracs_state = hdv_fracs_state / hdv_fracs_state.sum(axis=1, keepdims=True)
            logger.info(f"  HDV state restored from checkpoint: shape={hdv_fracs_state.shape}")
        else:
            hdv_fracs_state = np.ones(expected_shape, dtype=np.float32) / cfg.n_routes
        if gawron_mode_imagined == "routerl":
            if resume_hdv_cost_state is not None and tuple(resume_hdv_cost_state.shape) == expected_shape:
                hdv_costs_state = np.asarray(resume_hdv_cost_state, dtype=np.float32)
                logger.info(f"  HDV cost state restored from checkpoint: shape={hdv_costs_state.shape}")
            else:
                hdv_costs_state = np.zeros(expected_shape, dtype=np.float32)
        logger.info(
            "  HDV inter-episode Gawron: requested_mode=%s imagined_mode=%s lr=%.4f shape=%s",
            cfg.gawron_mode, gawron_mode_imagined, cfg.gawron_lr, hdv_fracs_state.shape,
        )

    # ── optional SUMO eval harness ────────────────────────────────────────────
    sumo_eval: Optional[SUMOEvalHarness] = None
    if cfg.eval_every > 0:
        if vehicle_mode:
            logger.warning(
                "eval_every > 0 but agent_level=vehicle: disable built-in SUMO eval "
                "(current harness is OD-observation based)."
            )
        elif not _ROUTERL_AVAILABLE:
            logger.warning("eval_every > 0 but routerl is not installed — SUMO eval disabled")
        else:
            sumo_eval = SUMOEvalHarness(
                n_routes          = cfg.n_routes,
                obs_dim           = obs_dim,
                device            = device,
                network           = cfg.eval_network,
                network_folder    = cfg.eval_network_folder,
                eval_results_dir  = cfg.eval_results_dir,
                eval_human_eps    = cfg.eval_human_eps,
                eval_eps          = cfg.eval_eps,
                ratio_machines    = cfg.eval_ratio_machines,
                seed              = cfg.eval_seed,
                gawron_mode       = cfg.gawron_mode,
                world_model       = wm if mean_field else None,
                route_registry    = route_registry if mean_field else None,
                fixed_hdv_fracs   = (
                    hdv_fracs_state
                    if (mean_field and cfg.eval_use_fixed_hdv_fracs and cfg.gawron_mode != "routerl_blackbox")
                    else None
                ),
                disable_human_learning_after_mutation=cfg.eval_disable_human_learning_after_mutation,
                stochastic_policy=cfg.eval_stochastic_policy,
                hdv_state_in_path=cfg.eval_hdv_state_in,
                hdv_state_out_path=cfg.eval_hdv_state_out,
            )
    if use_sumo_synced_hdv_policy and sumo_eval is None:
        logger.warning(
            "  routerl_blackbox requested but eval_every=0 or SUMO unavailable; "
            "imagined HDV strategy will stay fixed at its initial state."
        )
    if use_sumo_synced_hdv_policy and sumo_eval is not None and hdv_fracs_state is not None:
        synced = sumo_eval.get_current_hdv_fracs()
        if synced is not None and synced.shape == hdv_fracs_state.shape:
            hdv_fracs_state = synced.astype(np.float32)
            hdv_ent = -np.sum(hdv_fracs_state * np.log(hdv_fracs_state + 1e-10), axis=1)
            logger.info(
                "  Initial HDV strategy synced from SUMO harness: mean_entropy=%.4f",
                float(hdv_ent.mean()),
            )

    adaptive_penalty_coef = float(cfg.conservative_penalty_coef)
    n_lr_decays = 0

    # Pre-compute vehicle path cache for vehicle-level imagination
    _vehicle_path_cache: Optional[Union[VehiclePathCache, LazyVehiclePathCache]] = None
    _departure_schedule: Optional[Dict] = None
    _avg_route_steps: int = 7
    _avg_edge_time: float = 8.0
    _actor_start_step_candidates: Optional[List[int]] = None
    _hdv_background_edge: Optional[torch.Tensor] = None
    _hdv_background_route_prior: Optional[np.ndarray] = None
    _edge_tt_ref = np.ones((int(wm.n_edges),), dtype=np.float32)
    if vehicle_mode and route_registry is not None and wm._registry.adjacency is not None:
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
        _cache_pairs = len(_dest_edges_unique) * int(wm.n_edges)
        _max_cache_pairs = 50000
        if str(getattr(cfg, "vehicle_candidate_source", "current_to_dest")).strip().lower() == "od_suffix":
            _vehicle_path_cache = None
            logger.info("Vehicle path cache disabled: using precomputed od_suffix candidate routes.")
        elif uses_explicit_dynamic_candidate_edge_cost_source(cfg):
            _vehicle_path_cache = LazyVehiclePathCache(
                adj_list=_adj_list,
                n_edges=wm.n_edges,
                K=cfg.n_routes,
                penalty_factor=cfg.vehicle_path_penalty_factor,
            )
            logger.info(
                "Vehicle path cache set to dynamic lazy mode: candidate_edge_cost_source=%s",
                resolve_wm_candidate_edge_cost_source(cfg),
            )
        elif _cache_pairs <= _max_cache_pairs:
            _vehicle_path_cache = VehiclePathCache(
                adj_list=_adj_list,
                edge_tt_ref=_edge_tt_ref,
                dest_edges_unique=_dest_edges_unique,
                n_edges=wm.n_edges,
                K=cfg.n_routes,
                penalty_factor=cfg.vehicle_path_penalty_factor,
            )
            logger.info(
                "Vehicle path cache built: %d destinations, %d cached pairs in %.1fs",
                len(_dest_edges_unique), len(_vehicle_path_cache._cache),
                _time.time() - _pc_t0,
            )
        else:
            _vehicle_path_cache = LazyVehiclePathCache(
                adj_list=_adj_list,
                n_edges=wm.n_edges,
                K=cfg.n_routes,
                penalty_factor=cfg.vehicle_path_penalty_factor,
            )
            logger.info(
                "Vehicle path cache skipped: %d destinations x %d edges = %d pairs exceeds limit %d; "
                "using lazy on-demand cache.",
                len(_dest_edges_unique),
                int(wm.n_edges),
                _cache_pairs,
                _max_cache_pairs,
            )
        agents_path = cfg.agents_csv or os.path.join(cfg.network_folder, "agents.csv")
        if agents_path and os.path.exists(agents_path):
            _departure_cap_mode = str(getattr(cfg, "dreamer_departure_cap_mode", "cfg")).strip().lower()
            if _departure_cap_mode not in {"cfg", "none"}:
                raise ValueError(
                    "dreamer_departure_cap_mode must be one of {'cfg', 'none'}, "
                    f"got {_departure_cap_mode!r}"
                )
            _max_av_count = int(cfg.n_av) if (_departure_cap_mode == "cfg" and int(cfg.n_av) > 0) else None
            _max_hdv_count = int(cfg.n_hdv) if (_departure_cap_mode == "cfg" and int(cfg.n_hdv) > 0) else None
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
                horizon=min(int(cfg.imagination_horizon), int(cfg.dreamer_actor_horizon))
                if training_family == "dreamer_ctde" else int(cfg.imagination_horizon),
                avg_route_steps=_avg_route_steps,
            )
            if _actor_start_step_candidates:
                logger.info(
                    "Vehicle dynamic imagination start-step candidates prepared: %d/%d valid starts",
                    len(_actor_start_step_candidates),
                    max(1, int(cfg.episode_steps) - (min(int(cfg.imagination_horizon), int(cfg.dreamer_actor_horizon)) if training_family == 'dreamer_ctde' else int(cfg.imagination_horizon))),
                )
            else:
                logger.warning(
                    "Vehicle dynamic imagination found no actor-qualified start steps; falling back to unconstrained random starts."
                )
            hdv_bg_mode = str(getattr(cfg, "dreamer_hdv_background_mode", "none") or "none").strip().lower()
            if hdv_bg_mode not in {"", "none", "off"}:
                if hdv_bg_mode in {"adapted_state", "episode_av_pressure_response", "episode_hdv_head_response"}:
                    bg_state_path = str(getattr(cfg, "dreamer_hdv_background_state", "") or "").strip()
                    if not bg_state_path:
                        bg_state_path = str(getattr(cfg, "eval_hdv_state_in", "") or "").strip()
                    hdv_route_fracs = load_hdv_route_fracs_from_state(
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
                hdv_bg_np = build_hdv_background_edge_action(
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

        if training_family == "dreamer_ctde":
            if actor is None or critic is None or actor_optimizer is None or critic_optimizer is None:
                raise RuntimeError("dreamer_ctde expected actor/critic/optimizers to be initialized")
            if not vehicle_mode:
                raise RuntimeError("dreamer_ctde currently requires vehicle_mode")
            pending_trajs: List[Dict[str, Any]] = []
            update_id = 0
            last_update_stats: Dict[str, float] = {
                "pg_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
                "invalid_route_rate": 0.0, "route_churn": 0.0, "skipped_actor_trajs": 0.0,
                "route_cost_aux_loss": 0.0, "route_cost_aux_acc": 0.0, "route_cost_aux_margin": 0.0,
                "expected_cost_loss": 0.0, "expected_cost_raw": 0.0, "expected_cost_best_prob": 0.0,
                "expected_cost_best_route0_frac": 0.0,
                "expected_cost_nonzero_best_frac": 0.0,
                "expected_cost_margin": 0.0,
                "grpo_combined_loss": 0.0, "grpo_combined_pg_loss": 0.0, "grpo_combined_kl_loss": 0.0,
                "grpo_combined_adv_mean": 0.0, "grpo_combined_adv_std": 0.0, "grpo_combined_clip_frac": 0.0,
                "grpo_policy_kl": 0.0,
            }

        if cfg.resume_ckpt and os.path.exists(cfg.resume_ckpt):
            resume_state = maybe_resume_dreamer_checkpoint(
                resume_ckpt=cfg.resume_ckpt,
                actor=actor,
                critic=critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                device=device,
                reset_actor_optimizer=bool(getattr(cfg, "resume_reset_actor_optimizer", False)),
            )
            if resume_state is not None:
                start_episode = int(resume_state["start_episode"])
                stats_history = resume_state["stats_history"]
                pending_trajs = list(resume_state["pending_trajs"])
                update_id = int(resume_state["update_id"])
                if resume_state.get("restored_rng") is not None:
                    rng = resume_state["restored_rng"]
        if start_episode > 0 and int(cfg.n_total_episodes) <= start_episode:
            logger.warning(
                "n_total_episodes=%d <= resumed Dreamer episode=%d; interpret as additional episodes "
                "(target episode -> %d).",
                int(cfg.n_total_episodes),
                start_episode,
                start_episode + int(cfg.n_total_episodes),
            )
            target_episode = start_episode + int(cfg.n_total_episodes)

        for episode in range(start_episode, target_episode):
            actor.train()
            critic.train()
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
                    )
                episode_av_pressure = episode_av_pressure_from_action_edges(
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
                        hdv_response_fracs = build_hdv_route_fracs_from_episode_av_pressure(
                            route_registry,
                            episode_av_pressure,
                            base_edge_tt=_edge_tt_ref,
                            n_routes=int(cfg.n_routes),
                            pressure_weight=float(getattr(cfg, "dreamer_hdv_response_pressure_weight", 60.0)),
                            temperature=float(getattr(cfg, "dreamer_hdv_response_temperature", 60.0)),
                            prior_route_fracs=_hdv_background_route_prior,
                            prior_mix=float(getattr(cfg, "dreamer_hdv_response_prior_mix", 0.0)),
                        )
                        hdv_response_bg_np = build_hdv_background_edge_action(
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
            stats_history["hdv_response_av_pressure_sum"].append(ep_metrics.get("hdv_response_av_pressure_sum", 0.0))
            stats_history["hdv_response_background_sum"].append(ep_metrics.get("hdv_response_background_sum", 0.0))
            stats_history["hdv_response_background_active_edges"].append(ep_metrics.get("hdv_response_background_active_edges", 0.0))
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
                )
                _update_wall_sec += _time.perf_counter() - _update_t0
                pending_trajs.clear()
                update_id += 1
                for k in (
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
                ):
                    stats_history[k].append(last_update_stats[k])
                stats_history["skipped_actor_trajs"].append(last_update_stats.get("skipped_actor_trajs", 0.0))
                stats_history["update_diagnostics"].append({
                    "update_id": int(last_update_stats.get("update_id", update_id - 1)),
                    "consumed_traj_ids": list(last_update_stats.get("consumed_traj_ids", [])),
                    "used_traj_ids": list(last_update_stats.get("used_traj_ids", [])),
                    "skipped_traj_ids": list(last_update_stats.get("skipped_traj_ids", [])),
                    "skip_reason_counts": dict(last_update_stats.get("skip_reason_counts", {})),
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
                    f"hdv_resp_bg={recent_hdv_response_bg:.4f} | "
                    f"timing(ep={episode_wall_sec:.2f}s rollout={_rollout_wall_sec:.2f}s hdv={_hdv_response_wall_sec:.2f}s baseline={_baseline_wall_sec:.2f}s update={_update_wall_sec:.2f}s group={rollout_group_size}) | "
                    f"skip_reasons={last_skip_reasons} | "
                    f"tall_pred={recent_tall_pred:.4f} | "
                    f"actor_lr={actor_optimizer.param_groups[0]['lr']:.2e} critic_lr={critic_optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

            if cfg.checkpoint_every > 0 and (episode + 1) % cfg.checkpoint_every == 0:
                extra_state = {}
                if bool(cfg.dreamer_preserve_partial_batch_on_exit):
                    extra_state = {
                        "pending_trajs": _move_nested_tensors(pending_trajs, cpu=True),
                        "update_id": int(update_id),
                        **capture_rng_state(rng=rng),
                    }
                path = save_dreamer_checkpoint(
                    cfg,
                    actor,
                    critic,
                    actor_optimizer,
                    critic_optimizer,
                    episode + 1,
                    stats_history,
                    extra_state=extra_state,
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
            )
            for k in (
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
            ):
                stats_history[k].append(last_update_stats[k])
            stats_history["skipped_actor_trajs"].append(last_update_stats.get("skipped_actor_trajs", 0.0))
            stats_history["update_diagnostics"].append({
                "update_id": int(last_update_stats.get("update_id", update_id)),
                "consumed_traj_ids": list(last_update_stats.get("consumed_traj_ids", [])),
                "used_traj_ids": list(last_update_stats.get("used_traj_ids", [])),
                "skipped_traj_ids": list(last_update_stats.get("skipped_traj_ids", [])),
                "skip_reason_counts": dict(last_update_stats.get("skip_reason_counts", {})),
                "skipped_actor_trajs": float(last_update_stats.get("skipped_actor_trajs", 0.0)),
                "consumed_fresh_traj_count": int(last_update_stats.get("consumed_fresh_traj_count", 0)),
                "consumed_restored_traj_count": int(last_update_stats.get("consumed_restored_traj_count", 0)),
                "used_fresh_traj_count": int(last_update_stats.get("used_fresh_traj_count", 0)),
                "used_restored_traj_count": int(last_update_stats.get("used_restored_traj_count", 0)),
                "skipped_fresh_traj_count": int(last_update_stats.get("skipped_fresh_traj_count", 0)),
                "skipped_restored_traj_count": int(last_update_stats.get("skipped_restored_traj_count", 0)),
            })

        final_extra_state = {}
        if bool(cfg.dreamer_preserve_partial_batch_on_exit):
            final_extra_state = {
                "pending_trajs": _move_nested_tensors(pending_trajs, cpu=True),
                "update_id": int(update_id),
                **capture_rng_state(rng=rng),
            }

        save_dreamer_checkpoint(
            cfg,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            target_episode,
            stats_history,
            extra_state=final_extra_state,
        )
        logger.info("Dreamer-style training complete.")
        return

    for episode in range(start_episode, target_episode):
        policy.train()

        if vehicle_mode:
            ppo_buf, ep_reward, ep_metrics = imagined_episode_dynamic(
                wm=wm,
                policy=policy,
                dataset=dataset,
                route_registry=route_registry,
                cfg=cfg,
                rng=rng,
                device=device,
                departure_schedule=_departure_schedule,
                avg_route_steps=_avg_route_steps,
                avg_edge_time=_avg_edge_time,
                path_cache=_vehicle_path_cache,
                reward_mean=rew_running_mean,
                reward_std=rew_running_std,
                start_step_candidates=_actor_start_step_candidates,
            )
        elif mean_field:
            ppo_buf, ep_reward, ep_metrics = imagined_episode_mf(
                cfg, wm, policy, route_registry, dataset, rng, device,
                reward_mean=rew_running_mean, reward_std=rew_running_std,
                hdv_fracs_override=hdv_fracs_state,
                conservative_penalty_coef=adaptive_penalty_coef,
            )
        else:
            ppo_buf, ep_reward, ep_metrics = imagined_episode(
                cfg, wm, policy, reward_model, dataset, rng, device,
                reward_mean=rew_running_mean, reward_std=rew_running_std,
            )

        ep_r = ppo_buf.rewards
        rew_running_mean = (1 - rew_alpha) * rew_running_mean + rew_alpha * ep_r.mean().item()
        rew_running_std  = (1 - rew_alpha) * rew_running_std  + rew_alpha * ep_r.std().clamp_min(1e-8).item()

        pending_bufs.append(ppo_buf)
        stats_history["ep_reward"].append(ep_reward)
        stats_history["t_av"].append(ep_metrics["t_av"])
        stats_history["t_hdv"].append(ep_metrics["t_hdv"])
        stats_history["t_all"].append(ep_metrics["t_all"])
        stats_history["hdv_drift"].append(ep_metrics["hdv_drift"])
        stats_history["hdv_head_entropy"].append(ep_metrics.get("hdv_head_entropy", 0.0))
        stats_history["terminal_t_all_pred"].append(ep_metrics.get("terminal_t_all_pred", float("nan")))
        stats_history["terminal_bonus"].append(ep_metrics.get("terminal_bonus", 0.0))
        stats_history["adaptive_penalty"].append(adaptive_penalty_coef)

        # ── Inter-episode HDV Gawron update ──────────────────────────────────
        if (
            mean_field
            and (not vehicle_mode)
            and hdv_fracs_state is not None
            and cfg.hdv_gawron
            and not cfg.hdv_internalized
            and not use_sumo_synced_hdv_policy
        ):
            avg_t_e = ep_metrics["avg_t_e"]                            # (E,)
            route_tt_all = route_tt_from_edge_times(route_registry, avg_t_e)  # (N_od, K), seconds
            for i in range(route_registry.n_od):
                route_tt_i = route_tt_all[i]                           # (K,)
                updated, updated_cost = gawron_step_compatible(
                    hdv_frac=hdv_fracs_state[i],
                    route_tt=route_tt_i,
                    mode=gawron_mode_imagined,
                    lr=cfg.gawron_lr,
                    alpha_zero=cfg.gawron_alpha_zero,
                    beta=cfg.gawron_beta,
                    update_mode=cfg.gawron_update_mode,
                    min_prob=cfg.hdv_min_prob,
                    cost_state=hdv_costs_state[i] if hdv_costs_state is not None else None,
                )
                hdv_fracs_state[i] = smooth_distribution(
                    previous=hdv_fracs_state[i],
                    updated=updated,
                    momentum=cfg.hdv_update_momentum,
                    min_prob=cfg.hdv_min_prob,
                )
                if hdv_costs_state is not None and updated_cost is not None:
                    hdv_costs_state[i] = updated_cost.astype(np.float32)
            # Track HDV concentration (mean entropy across OD pairs)
            hdv_ent = -np.sum(hdv_fracs_state * np.log(hdv_fracs_state + 1e-10), axis=1)
            ep_metrics["hdv_entropy"] = float(hdv_ent.mean())
            # Overwrite hdv_drift with inter-episode drift from uniform
            uniform = 1.0 / cfg.n_routes
            stats_history["hdv_drift"][-1] = float(np.abs(hdv_fracs_state - uniform).sum())

        if len(pending_bufs) >= cfg.n_episodes_per_update:
            if mean_field:
                merged_batch = MultiAgentPPOBuffer.merge(pending_bufs)
            else:
                merged_batch = PPOBuffer.merge(pending_bufs)
            last_update_stats = ppo_update(cfg, policy, optimizer, merged_batch, device)
            pending_bufs.clear()

            for k in ("pg_loss", "value_loss", "entropy", "approx_kl"):
                stats_history[k].append(last_update_stats[k])

        if episode % cfg.log_every == 0:
            recent_reward   = np.mean(stats_history["ep_reward"][-cfg.log_every:])
            recent_t_av     = np.mean(stats_history["t_av"][-cfg.log_every:])
            recent_t_hdv    = np.mean(stats_history["t_hdv"][-cfg.log_every:])
            recent_t_all    = np.mean(stats_history["t_all"][-cfg.log_every:])
            recent_hdv_drift = np.mean(stats_history["hdv_drift"][-cfg.log_every:])
            recent_hdv_head_ent = np.mean(stats_history["hdv_head_entropy"][-cfg.log_every:])
            if stats_history["terminal_t_all_pred"]:
                tail_vals = np.asarray(stats_history["terminal_t_all_pred"][-cfg.log_every:], dtype=np.float32)
                finite_tail = tail_vals[np.isfinite(tail_vals)]
                recent_tall_pred = float(finite_tail.mean()) if finite_tail.size > 0 else float("nan")
            else:
                recent_tall_pred = float("nan")
            recent_terminal_bonus = (
                float(np.mean(stats_history["terminal_bonus"][-cfg.log_every:]))
                if stats_history["terminal_bonus"] else 0.0
            )
            print(
                f"ep {episode:05d} | "
                f"reward={ep_reward:8.3f} (avg{cfg.log_every}={recent_reward:.3f}) | "
                f"pg={last_update_stats['pg_loss']:.4f} | "
                f"vf={last_update_stats['value_loss']:.4f} | "
                f"ent={last_update_stats['entropy']:.4f} | "
                f"kl={last_update_stats['approx_kl']:.4f} | "
                f"t_av={recent_t_av:.4f} t_hdv={recent_t_hdv:.4f} t_all={recent_t_all:.4f} | "
                f"hdv_drift={recent_hdv_drift:.4f} | "
                f"hdv_head_ent={recent_hdv_head_ent:.4f} | "
                f"tall_pred={recent_tall_pred:.4f} tall_bonus={recent_terminal_bonus:.4f} | "
                f"penalty={adaptive_penalty_coef:.4f} lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

        if cfg.checkpoint_every > 0 and (episode + 1) % cfg.checkpoint_every == 0:
            path = save_checkpoint(
                cfg, policy, optimizer, episode + 1, stats_history,
                hdv_fracs_state=hdv_fracs_state if (mean_field and (not vehicle_mode)) else None,
                hdv_costs_state=hdv_costs_state if (mean_field and (not vehicle_mode)) else None,
            )
            logger.info(f"  Checkpoint saved: {path}")

        # ── periodic SUMO evaluation ──────────────────────────────────────────
        if sumo_eval is not None and (episode + 1) % cfg.eval_every == 0:
            logger.info(f"[SUMOEval] Running {cfg.eval_eps} SUMO episodes at ep {episode+1} …")
            if (
                mean_field
                and hdv_fracs_state is not None
                and not sumo_eval.uses_routerl_hdv_blackbox
            ):
                sumo_eval.set_fixed_hdv_fracs(hdv_fracs_state)
            real = sumo_eval.evaluate(policy)
            policy.train()
            if use_sumo_synced_hdv_policy and mean_field and hdv_fracs_state is not None:
                synced = sumo_eval.get_current_hdv_fracs()
                if synced is not None and synced.shape == hdv_fracs_state.shape:
                    hdv_fracs_state = synced.astype(np.float32)
                    hdv_ent = -np.sum(hdv_fracs_state * np.log(hdv_fracs_state + 1e-10), axis=1)
                    logger.info(
                        "[SUMOEval] Synced HDV strategy -> imagined rollout (mean_entropy=%.4f)",
                        float(hdv_ent.mean()),
                    )
                else:
                    logger.warning(
                        "[SUMOEval] Failed to sync HDV strategy from SUMO eval; keep previous imagined HDV state."
                    )
            if real is not None:
                if stats_history["t_all"]:
                    recent_proxy = np.asarray(stats_history["t_all"][-cfg.eval_every:], dtype=np.float32)
                    imagined_t_all_proxy = float(np.mean(recent_proxy))
                else:
                    imagined_t_all_proxy = float("nan")

                imagined_t_all = float("nan")
                imagined_source = "proxy"
                if stats_history.get("terminal_t_all_pred"):
                    recent_pred = np.asarray(
                        stats_history["terminal_t_all_pred"][-cfg.eval_every:],
                        dtype=np.float32,
                    )
                    recent_pred = recent_pred[np.isfinite(recent_pred)]
                    recent_pred = recent_pred[recent_pred >= 0.0]
                    if recent_pred.size > 0:
                        imagined_t_all = float(recent_pred.mean())
                        imagined_source = "t_all_pred"

                if not np.isfinite(imagined_t_all):
                    imagined_t_all = imagined_t_all_proxy

                if np.isfinite(imagined_t_all):
                    discrepancy = abs(imagined_t_all - real["t_all"]) / max(abs(real["t_all"]), 1e-6)
                else:
                    discrepancy = float("nan")

                stats_history["imagined_eval_t_all"].append(imagined_t_all)
                stats_history["discrepancy_t_all"].append(discrepancy)

                if "sumo_eval_log" not in stats_history:
                    stats_history["sumo_eval_log"] = []
                stats_history["sumo_eval_log"].append(
                    (episode + 1, real["t_CAV"], real["t_HDV"], real["t_all"], imagined_t_all, discrepancy)
                )
                print(
                    f"  *** SUMO eval ep {episode+1:05d} | "
                    f"t_CAV={real['t_CAV']:.2f} min  "
                    f"t_HDV={real['t_HDV']:.2f} min  "
                    f"t_all={real['t_all']:.2f} min | "
                    f"imagined_t_all={imagined_t_all:.2f} ({imagined_source}, proxy={imagined_t_all_proxy:.2f}) | "
                    f"discrepancy={discrepancy:.3f} ***",
                    flush=True,
                )

                if np.isfinite(discrepancy) and discrepancy > cfg.discrepancy_threshold:
                    adaptive_penalty_coef = min(1.0, adaptive_penalty_coef * 1.25 + 1e-4)
                    if n_lr_decays < cfg.max_lr_decays:
                        for group in optimizer.param_groups:
                            group["lr"] *= cfg.lr_decay_on_discrepancy
                        n_lr_decays += 1
                        logger.warning(
                            "[SUMOEval] discrepancy %.3f > %.3f -> lr decay applied (%d/%d), new_lr=%.2e, penalty=%.4f",
                            discrepancy, cfg.discrepancy_threshold,
                            n_lr_decays, cfg.max_lr_decays,
                            optimizer.param_groups[0]["lr"], adaptive_penalty_coef,
                        )
                    else:
                        logger.warning(
                            "[SUMOEval] discrepancy %.3f > %.3f -> penalty increased to %.4f (lr decay budget exhausted)",
                            discrepancy, cfg.discrepancy_threshold, adaptive_penalty_coef,
                        )
                elif np.isfinite(discrepancy):
                    adaptive_penalty_coef = max(
                        float(cfg.conservative_penalty_coef),
                        adaptive_penalty_coef * 0.98,
                    )
            else:
                logger.warning("[SUMOEval] No episode CSVs found — check save_every setting")

    save_checkpoint(
        cfg, policy, optimizer, target_episode, stats_history,
        hdv_fracs_state=hdv_fracs_state if (mean_field and (not vehicle_mode)) else None,
        hdv_costs_state=hdv_costs_state if (mean_field and (not vehicle_mode)) else None,
    )
    logger.info("Training complete.")

    if sumo_eval is not None:
        sumo_eval.close()


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
    p.add_argument("--n_av",              type=int,   default=30)
    p.add_argument("--n_hdv",             type=int,   default=70)
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
    p.add_argument("--critic_variant",    type=str,   default="baseline",
                   choices=["baseline", "latent_global"],
                   help="Critic variant for downstream comparison. 'latent_global' keeps actor unchanged and augments critic input with a pooled latent global summary.")
    p.add_argument("--critic_global_summary_variant", type=str, default="mean",
                   choices=["mean", "mean_std"],
                   help="Global latent summary ablation for A1 latent_global critic.")
    p.add_argument("--critic_target_variant", type=str, default="baseline",
                   choices=["baseline", "multistep_imagined"],
                   help="Critic-target variant for downstream comparison. 'multistep_imagined' changes value targets only.")
    p.add_argument("--critic_target_horizon", type=int, default=3,
                   help="Short imagined horizon used when critic_target_variant=multistep_imagined.")
    p.add_argument("--latent_reward_ckpt",type=str,   default="",
                   help="latent reward 预测器路径（由 train_latent_reward.py 生成）")
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
    p.add_argument("--no_bpr_reward",  action="store_true",
                   help="Disable BPR reward; fall back to LatentRewardPredictor / image model")
    p.add_argument("--reward_mode",    type=str, default="difference",
                   choices=["global", "local", "difference"],
                   help="Mean-field per-agent reward mode")
    p.add_argument("--conservative_penalty_coef", type=float, default=0.02,
                   help="Penalty on AV-HDV intent divergence to reduce model exploitation")
    p.add_argument("--dense_bpr_reward_weight", type=float, default=0.1,
                   help="Scheme-B: step-level dense BPR shaping weight.")
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
                       "Vehicle departure schedule cap for Dreamer imagination. none uses the "
                       "full agents.csv schedule and is the mainline behavior; cfg caps scheduled "
                       "vehicles to --n_av/--n_hdv for small-scale ablations."
                   ))
    p.add_argument("--dreamer_value_coef", type=float, default=1.0,
                   help="Scale factor for Dreamer critic regression loss.")
    p.add_argument("--dreamer_entropy_coef", type=float, default=1e-3,
                   help="Entropy bonus coefficient for the Dreamer actor objective.")
    p.add_argument("--dreamer_critic_variant", type=str, default="v_only",
                   choices=["v_only", "action_conditioned"],
                   help="Dreamer critic variant. 'action_conditioned' concatenates action_edge to the pooled latent critic input.")
    p.add_argument("--dreamer_actor_objective", type=str, default="direct_return",
                   choices=["direct_return", "critic_q", "direct_plus_q"],
                   help="Dreamer actor objective. direct_return is the existing differentiable reward objective; critic_q maximizes an action-conditioned Q critic.")
    p.add_argument("--dreamer_actor_q_weight", type=float, default=1.0,
                   help="Weight for the action-conditioned critic-Q actor objective when enabled.")
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
    p.add_argument("--dreamer_expected_cost_weight", type=float, default=0.0,
                   help="Actor-only expected-cost loss weight for minimizing sum_k pi_k C_k without CE/distillation targets.")
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
                       "Actor-only per-vehicle GRPO route-cost loss weight. Each vehicle's candidate "
                       "routes form a group, with advantage = group mean cost - candidate cost."
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
    p.add_argument("--dreamer_grpo_policy_objective", type=str, default="combined",
                   choices=["combined", "legacy_separate"],
                   help=(
                       "GRPO actor objective. combined uses one PPO-style clipped objective over "
                       "route-cost and rollout advantages. legacy_separate reproduces the older "
                       "behavior: add differentiable route-cost GRPO and rollout advantage log-prob "
                       "terms separately, without PPO ratio/clip/KL."
                   ))
    p.add_argument("--dreamer_grpo_rollout_weight", type=float, default=0.0,
                   help=(
                       "Critic-free stochastic rollout GRPO weight. Sample joint vehicle route actions, "
                       "score whole imagined rollouts with Dreamer reward, and update by batch-relative advantage."
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
                       "Optional per-vehicle score-function loss inside imagined rollouts. Uses WM-decoded "
                       "next-step route costs to give each sampled vehicle route its own relative advantage."
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
                   help=(
                       "Optional HDV learning-state JSON for --dreamer_hdv_background_mode=adapted_state. "
                       "Falls back to --eval_hdv_state_in when omitted."
                   ))
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
                   help="Auxiliary actor-logit supervision weight toward soft shortest candidate routes from current WM/BPR route costs.")
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
    # ── periodic SUMO evaluation ──────────────────────────────────────────────
    p.add_argument("--eval_every",         type=int,   default=0,
                   help="Run SUMO eval every N offline episodes (0 = disabled)")
    p.add_argument("--eval_eps",           type=int,   default=10,
                   help="SUMO test episodes per eval call")
    p.add_argument("--eval_human_eps",     type=int,   default=300,
                   help="Human stabilisation episodes before mutation (done once)")
    p.add_argument("--eval_ratio_machines",type=float, default=0.4,
                   help="AV fraction for SUMO eval.")
    p.add_argument("--eval_network",       type=str,   default="ingolstadt_custom")
    p.add_argument("--eval_network_folder",type=str,   default="networks/ingolstadt_custom")
    p.add_argument("--eval_results_dir",   type=str,   default="results/marl_eval")
    p.add_argument("--eval_seed",          type=int,   default=42)
    p.add_argument("--eval_keep_human_learning_after_mutation",
                   dest="eval_disable_human_learning_after_mutation",
                   action="store_false",
                   help="In SUMO eval, keep RouteRL native human learning active after mutation.")
    p.add_argument("--eval_disable_human_learning_after_mutation",
                   dest="eval_disable_human_learning_after_mutation",
                   action="store_true",
                   help="In SUMO eval, freeze human learning after mutation (default).")
    p.set_defaults(eval_disable_human_learning_after_mutation=True)
    p.add_argument("--eval_no_fixed_hdv_fracs", action="store_true",
                   help="In SUMO eval, do not overwrite HDV defaults from checkpoint fractions; use RouteRL native human model.")
    p.add_argument("--eval_stochastic_policy", action="store_true",
                   help="In SUMO eval, sample AV fractions via Gaussian logits instead of deterministic softmax(mean logits).")
    p.add_argument("--eval_hdv_state_in", type=str, default="",
                   help="Optional JSON path to restore RouteRL human learning state for periodic SUMO eval.")
    p.add_argument("--eval_hdv_state_out", type=str, default="",
                   help="Optional JSON path to save RouteRL human learning state after periodic SUMO eval.")
    p.add_argument("--discrepancy_threshold", type=float, default=0.25,
                   help="Relative |imagined-real|/|real| threshold for adaptive controls")
    p.add_argument("--lr_decay_on_discrepancy", type=float, default=0.7,
                   help="LR decay multiplier when discrepancy is above threshold")
    p.add_argument("--max_lr_decays", type=int, default=4,
                   help="Maximum number of discrepancy-triggered LR decays")
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
    d["use_bpr_reward"] = not d.pop("no_bpr_reward")
    d["include_intention_obs"] = not d.pop("no_intention_obs")
    d["vehicle_obs_include_global"] = not d.pop("no_vehicle_obs_global")
    d["use_terminal_tall_reward"] = (
        not d.pop("no_terminal_tall_reward")
        and abs(float(d.get("terminal_tall_reward_weight", 0.0))) > 0.0
    )
    d["eval_use_fixed_hdv_fracs"] = not d.pop("eval_no_fixed_hdv_fracs")
    # Validate: projector_ckpt and edge_registry are required
    if not d.get("projector_ckpt") or not d.get("edge_registry"):
        p.error("--projector_ckpt and --edge_registry are required")
    cfg = TrainConfig(**d)
    train(cfg)


if __name__ == "__main__":
    main()
