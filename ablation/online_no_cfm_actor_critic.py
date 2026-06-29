"""Online Gretz actor-critic ablation without world-model CFM transitions.

The ablation keeps the world-model encoder and policy-observation decoder, but
the actor-critic update consumes only real SUMO decisions and live route-cost
rewards.  No world-model transition, rollout, or reranking is used here.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DREAMER4_DIR = _REPO_ROOT / "dreamer4"
_URBAN_DIR = _DREAMER4_DIR / "urban_routing"
for _path in (str(_DREAMER4_DIR), str(_URBAN_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from dreamer4.urban_routing.dreamer_actor_critic import (
        CentralizedVehicleValueCritic,
        VehicleRouteActor,
    )
    from dreamer4.urban_routing.experiment_settings import load_settings_file
    from dreamer4.urban_routing.routerl_shared import (
        compute_agent_counts,
        load_od_data,
    )
    from dreamer4.urban_routing.train_marl import (
        build_diverse_paths_current_to_dest,
        build_vehicle_fleet_pressure_obs_features,
        build_vehicle_obs_vectorized,
        edge_tt_from_latent_state,
        vehicle_mode_obs_dim,
    )
except Exception:
    from dreamer_actor_critic import CentralizedVehicleValueCritic, VehicleRouteActor
    from experiment_settings import load_settings_file
    from routerl_shared import compute_agent_counts, load_od_data
    from train_marl import (
        build_diverse_paths_current_to_dest,
        build_vehicle_fleet_pressure_obs_features,
        build_vehicle_obs_vectorized,
        edge_tt_from_latent_state,
        vehicle_mode_obs_dim,
    )

from scripts.eval_sumo_hdv_state import (
    _load_hdv_learning_state,
    _merge_hdv_learning_state,
    _save_hdv_learning_state,
    _snapshot_hdv_learning_state,
)
from scripts.eval_sumo_io import (
    _episode_complete_for_dataset,
    _new_sumo_xmls_since,
    _reuse_route_artifacts,
    _reuse_route_artifacts_from_refs,
    _snapshot_sumo_xmls,
    _wait_for_expected_sumo_xml,
)
from scripts.eval_sumo_routing import (
    _build_edge_adj_list,
    _edge_route_cost_matrix,
    _route_batch_uniqueness_summary,
    _temperature_adjust_probs,
)
from ablation.traci_port_fallback import patch_traci_start_port_fallback
from dreamer4.urban_routing.data_collector import DataCollector, serialize_hdv_route_state


GRETZ_TAG = (
    "gretz_armainvilliers_current_to_dest_incident_completion100_"
    "stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1"
)
GRETZ_POST_MUTATION_TAG = "gretz_armainvilliers_current_to_dest_incident_20260530_odcompact"
DEFAULT_CHECKPOINT_EPISODES = (25, 50, 100)


def _append_vehicle_context_features(
    obs_vehicle_np: np.ndarray,
    *,
    step_idx: int,
    episode_steps: int,
    n_active_av: int,
    n_total_av: int,
) -> np.ndarray:
    """Append the three context features used by vehicle-mode train_marl."""
    if obs_vehicle_np.ndim != 2:
        return obs_vehicle_np
    context = np.array(
        [
            float(step_idx) / max(int(episode_steps), 1),
            float(n_active_av) / max(int(n_total_av), 1),
            0.0,
        ],
        dtype=np.float32,
    )
    if int(obs_vehicle_np.shape[0]) == 0:
        return np.zeros((0, int(obs_vehicle_np.shape[1]) + 3), dtype=np.float32)
    return np.concatenate([obs_vehicle_np, np.tile(context, (obs_vehicle_np.shape[0], 1))], axis=1)


@dataclass
class OnlineNoCfmConfig:
    """Configuration for the Gretz online no-CFM actor-critic ablation."""

    tag: str = GRETZ_TAG
    settings: str = str(
        _REPO_ROOT
        / ".urbandreamer"
        / "generated"
        / GRETZ_TAG
        / "urb_fair_eval.env"
    )
    output_dir: str = str(_REPO_ROOT / "results" / GRETZ_TAG / "ablation_online_no_cfm_ac")
    network_name: str = "gretz_armainvilliers"
    network_dir: str = str(_REPO_ROOT / "networks" / "gretz_armainvilliers")
    edge_registry: str = str(_REPO_ROOT / "networks" / "gretz_armainvilliers" / "edge_registry.json")
    projector_ckpt: str = str(
        _REPO_ROOT
        / "logs"
        / GRETZ_TAG
        / "dynamics_urb_fair_mainline_8k"
        / "flow_projector_latest.pt"
    )
    dynamics_ckpt: str = str(
        _REPO_ROOT
        / "logs"
        / GRETZ_TAG
        / "dynamics_urb_fair_mainline_8k"
        / "latest.pt"
    )
    route_artifacts_dir: str = str(
        _REPO_ROOT / "results" / GRETZ_TAG / "collect_urb_fair_eval_postmutation_records"
    )
    post_mutation_state_in: str = str(
        _REPO_ROOT
        / "results"
        / GRETZ_POST_MUTATION_TAG
        / "collect_urb_fair_eval_postmutation_records"
        / "post_mutation_state.json"
    )
    hdv_state_in: str = str(
        _REPO_ROOT
        / "results"
        / GRETZ_POST_MUTATION_TAG
        / "collect_urb_fair_eval_postmutation_records"
        / "post_mutation_hdv_learning_state.json"
    )
    ratio_machines: float = 0.4
    n_routes: int = 4
    train_episodes: int = 100
    eval_episodes: int = 3
    checkpoint_episodes: Tuple[int, ...] = DEFAULT_CHECKPOINT_EPISODES
    resume_checkpoint: str = ""
    resume_start_episode: int = 0
    episode_steps: int = 60
    sim_buffer_after_last_depart: int = 1000
    max_steps_per_episode: int = 0
    seed: int = 42
    device: str = "cuda"
    deterministic_eval: bool = True
    hidden_dim: int = 256
    n_hidden_layers: int = 2
    actor_architecture: str = "route_shared"
    critic_architecture: str = "self_attention"
    critic_attention_heads: int = 4
    include_global: bool = True
    include_route_costs: bool = True
    include_fleet_pressure: bool = True
    route_stats: str = "mean"
    route_cost_logit_bias: float = 0.0
    wm_policy_obs_source: str = "decoder_volume_tt"
    wm_edge_cost_source: str = "decoded_speed"
    wm_candidate_edge_cost_source: str = "freeflow"
    wm_decoded_speed_min_ratio: float = 0.05
    wm_decoded_speed_max_ratio: float = 1.50
    wm_decoded_speed_excess_scale: float = 1.0
    wm_cfm_init_mode: str = "zero"
    av_candidate_source: str = "current_shortest_freeflow"
    av_candidate_path_penalty_factor: float = 3.0
    av_policy_sample_temperature: float = 1.0
    route_cost_length_norm: str = "sum"
    reward_mode: str = "progress_fraction"
    reward_weight: float = 1.0
    reward_scale: float = 1.0
    step_interval: float = 30.0
    progress_cap: float = 1.0
    remaining_tt_eps: float = 1.0
    normalize_regret: bool = False
    reward_temperature: float = 60.0
    gamma: float = 1.0
    lr: float = 2e-4
    critic_lr: float = 3e-4
    value_coef: float = 1.0
    entropy_coef: float = 0.002
    ac_advantage_clip: float = 2.5
    ac_ppo_clip_ratio: float = 0.075
    ac_normalize_advantages: bool = True
    ac_entropy_target_frac: float = 0.65
    ac_entropy_floor_coef: float = 0.02
    replay_buffer_episodes: int = 1
    replay_update_epochs: int = 1
    replay_actor_max_age_episodes: int = 8
    return_horizon_steps: int = 8
    max_grad_norm: float = 10.0
    min_samples_per_update: int = 1
    hdv_gawron_mode: str = "routerl"
    hdv_gawron_alpha_zero: float = 0.2
    hdv_gawron_beta: float = -1.5
    hdv_gawron_update_mode: str = "expected"
    disable_human_learning_after_mutation: bool = True
    incident_edge: str = "-986733117#1"
    incident_start: float = 500.0
    incident_end: float = 2500.0
    incident_speed: float = 0.1
    route_generator: str = "dijkstra"
    path_gen_beta: float = -5.0
    path_num_samples: int = 10
    sumo_depart_lane: str = ""
    sumo_depart_pos: str = ""
    sumo_depart_speed: str = ""
    sumo_time_to_teleport: str = "-1"
    sumo_eager_insert: bool = False
    sumo_sloppy_insert: bool = False
    sumo_emergency_insert: bool = False
    sumo_extra_args: str = ""
    eval_python: str = sys.executable
    extra_eval_args: Tuple[str, ...] = field(default_factory=tuple)

    def resolve(self) -> "OnlineNoCfmConfig":
        values = asdict(self)
        for key in (
            "settings",
            "output_dir",
            "network_dir",
            "edge_registry",
            "projector_ckpt",
            "dynamics_ckpt",
            "route_artifacts_dir",
            "post_mutation_state_in",
            "hdv_state_in",
            "resume_checkpoint",
        ):
            raw = values.get(key, "")
            if raw:
                values[key] = str(_resolve_path(raw))
        return OnlineNoCfmConfig(**values)

    @classmethod
    def from_settings(cls, settings_path: str, **overrides: Any) -> "OnlineNoCfmConfig":
        cfg = cls(settings=str(settings_path))
        settings = load_settings_file(settings_path)
        updates: Dict[str, Any] = {
            "network_name": settings.get("NETWORK_NAME", cfg.network_name),
            "network_dir": settings.get("NETWORK_DIR", cfg.network_dir),
            "edge_registry": settings.get("EDGE_REGISTRY", cfg.edge_registry),
            "projector_ckpt": settings.get("PROJECTOR_CKPT", cfg.projector_ckpt),
            "dynamics_ckpt": settings.get("DYNAMICS_CKPT", cfg.dynamics_ckpt),
            "route_artifacts_dir": settings.get("ROUTE_ARTIFACTS_DIR", cfg.route_artifacts_dir),
            "ratio_machines": _as_float(settings.get("RATIO_MACHINES"), cfg.ratio_machines),
            "n_routes": _as_int(settings.get("N_ROUTES"), cfg.n_routes),
            "episode_steps": _as_int(settings.get("EPISODE_STEPS"), cfg.episode_steps),
            "sim_buffer_after_last_depart": _as_int(
                settings.get("SIM_BUFFER_AFTER_LAST_DEPART"),
                cfg.sim_buffer_after_last_depart,
            ),
            "incident_edge": settings.get("INCIDENT_EDGE", cfg.incident_edge),
            "incident_start": _as_float(settings.get("INCIDENT_START"), cfg.incident_start),
            "incident_end": _as_float(settings.get("INCIDENT_END"), cfg.incident_end),
            "incident_speed": _as_float(settings.get("INCIDENT_SPEED"), cfg.incident_speed),
            "route_generator": settings.get("ROUTE_GENERATOR", cfg.route_generator),
            "path_gen_beta": _as_float(settings.get("PATH_GEN_BETA"), cfg.path_gen_beta),
            "path_num_samples": _as_int(settings.get("PATH_NUM_SAMPLES"), cfg.path_num_samples),
            "hdv_gawron_mode": settings.get("HDV_GAWRON_MODE", cfg.hdv_gawron_mode),
            "hdv_gawron_alpha_zero": _as_float(
                settings.get("HDV_GAWRON_ALPHA_ZERO"),
                cfg.hdv_gawron_alpha_zero,
            ),
            "hdv_gawron_beta": _as_float(settings.get("HDV_GAWRON_BETA"), cfg.hdv_gawron_beta),
            "hdv_gawron_update_mode": settings.get(
                "HDV_GAWRON_UPDATE_MODE",
                cfg.hdv_gawron_update_mode,
            ),
            "wm_policy_obs_source": settings.get("WM_POLICY_OBS_SOURCE", cfg.wm_policy_obs_source),
            "wm_edge_cost_source": settings.get("WM_EDGE_COST_SOURCE", cfg.wm_edge_cost_source),
            "wm_candidate_edge_cost_source": settings.get(
                "WM_CANDIDATE_EDGE_COST_SOURCE",
                cfg.wm_candidate_edge_cost_source,
            ),
            "reward_mode": settings.get(
                "VEHICLE_REWARD_MODE",
                settings.get("DREAMER_ROUTE_COST_REWARD_MODE", cfg.reward_mode),
            ),
            "reward_scale": _as_float(settings.get("VEHICLE_REWARD_SCALE"), cfg.reward_scale),
            "reward_temperature": _as_float(
                settings.get(
                    "VEHICLE_REWARD_TEMPERATURE",
                    settings.get("DREAMER_ROUTE_COST_REWARD_TEMPERATURE"),
                ),
                cfg.reward_temperature,
            ),
        }
        updates.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**{**asdict(cfg), **updates}).resolve()


@dataclass
class OnlineDecisionSample:
    episode: int
    decision_idx: int
    sim_step: int
    veh_id: str
    obs: np.ndarray
    action: int
    old_log_prob: float
    reward: float
    valid_mask: np.ndarray
    route_costs: np.ndarray
    elapsed_time: float
    path_batch: Optional[List[List[int]]] = None
    current_edge_idx: int = -1
    dest_edge_idx: int = -1


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path


def _as_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return int(default)
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return float(default)
    return float(value)


def _jsonable_config(cfg: OnlineNoCfmConfig) -> Dict[str, Any]:
    values = asdict(cfg)
    values["checkpoint_episodes"] = list(cfg.checkpoint_episodes)
    values["extra_eval_args"] = list(cfg.extra_eval_args)
    return values


def build_gretz_ablation_plan(cfg: Optional[OnlineNoCfmConfig] = None) -> Dict[str, Any]:
    resolved = (cfg or OnlineNoCfmConfig()).resolve()
    output_dir = Path(resolved.output_dir)
    checkpoints = {
        int(ep): str(output_dir / "checkpoints" / f"checkpoint_ep{int(ep):05d}.pt")
        for ep in resolved.checkpoint_episodes
    }
    eval_dirs = {
        int(ep): str(output_dir / "eval" / f"checkpoint_ep{int(ep):05d}")
        for ep in resolved.checkpoint_episodes
    }
    required_paths = {
        "settings": resolved.settings,
        "network_dir": resolved.network_dir,
        "edge_registry": resolved.edge_registry,
        "projector_ckpt": resolved.projector_ckpt,
        "dynamics_ckpt": resolved.dynamics_ckpt,
        "route_artifacts_dir": resolved.route_artifacts_dir,
        "post_mutation_state_in": resolved.post_mutation_state_in,
        "hdv_state_in": resolved.hdv_state_in,
    }
    missing = [
        name
        for name, value in required_paths.items()
        if value and not Path(str(value)).exists()
    ]
    return {
        "ablation": "gretz_online_no_cfm_actor_critic",
        "network": resolved.network_name,
        "tag": resolved.tag,
        "reward_semantics": {
            "scope": "vehicle_critic",
            "state": "current",
            "mode": resolved.reward_mode,
            "weight": resolved.reward_weight,
            "scale": resolved.reward_scale,
            "temperature": float(resolved.reward_temperature),
            "source": "live_sumo_current_edge_travel_time",
        },
        "scenario": {
            "source_note": "matches experiment_result/gretz_stronglivebad_completion100_full_cycle_20260602.md",
            "incident_edge": resolved.incident_edge,
            "incident_start": float(resolved.incident_start),
            "incident_end": float(resolved.incident_end),
            "incident_speed": float(resolved.incident_speed),
            "disable_human_learning_after_mutation": bool(
                resolved.disable_human_learning_after_mutation
            ),
            "post_mutation_state_in": resolved.post_mutation_state_in,
            "hdv_state_in": resolved.hdv_state_in,
        },
        "world_model_usage": {
            "encoder_for_actor_obs": True,
            "policy_obs_source": resolved.wm_policy_obs_source,
            "cfm_transition_in_training": False,
            "wm_rollout_or_rerank_in_training": False,
        },
        "training": {
            "algorithm": "online_actor_critic",
            "from_scratch": True,
            "episodes": resolved.train_episodes,
            "resume_checkpoint": resolved.resume_checkpoint,
            "resume_start_episode": int(resolved.resume_start_episode),
            "checkpoint_episodes": list(resolved.checkpoint_episodes),
            "checkpoint_paths": checkpoints,
            "replay_buffer_episodes": int(resolved.replay_buffer_episodes),
            "replay_update_epochs": int(resolved.replay_update_epochs),
            "replay_actor_max_age_episodes": int(resolved.replay_actor_max_age_episodes),
            "return_horizon_steps": int(resolved.return_horizon_steps),
        },
        "evaluation": {
            "episodes_per_checkpoint": resolved.eval_episodes,
            "eval_dirs": eval_dirs,
        },
        "paths": required_paths,
        "output_dir": str(output_dir),
        "missing": missing,
    }


def route_valid_mask_from_path_batch(
    path_batch: Sequence[Sequence[Sequence[int]]],
    *,
    n_routes: int,
) -> np.ndarray:
    mask = np.zeros((len(path_batch), int(n_routes)), dtype=bool)
    for i, paths in enumerate(path_batch):
        for k, path in enumerate(list(paths)[: int(n_routes)]):
            mask[i, k] = len(path) > 0
    return mask


def selected_vehicle_route_rewards(
    *,
    route_costs: np.ndarray | torch.Tensor,
    route_valid: np.ndarray | torch.Tensor,
    selected_idx: np.ndarray | torch.Tensor,
    reward_mode: str = "progress_fraction",
    reward_weight: float = 1.0,
    reward_scale: float = 1.0,
    step_interval: float = 30.0,
    elapsed_time: Optional[np.ndarray | torch.Tensor] = None,
    progress_cap: float = 1.0,
    remaining_tt_eps: float = 1.0,
    reward_temperature: float = 60.0,
) -> torch.Tensor:
    """Vehicle reward from live route costs for the no-CFM online ablation.

    ``progress_fraction`` matches offline current-state route progress:
    ``min(step_interval, remaining_route_tt) / (elapsed_time + remaining_route_tt)``.
    ``capped_step_cost`` is a cost-minimizing ablation:
    ``-min(step_interval, remaining_route_tt) / reward_temperature``.
    Invalid selected routes receive zero reward.
    """
    costs = torch.as_tensor(route_costs, dtype=torch.float32)
    valid = torch.as_tensor(route_valid, dtype=torch.bool, device=costs.device)
    selected = torch.as_tensor(selected_idx, dtype=torch.long, device=costs.device).reshape(-1)
    n = min(int(costs.shape[0]), int(valid.shape[0]), int(selected.shape[0]))
    if n <= 0:
        return torch.zeros((0,), dtype=costs.dtype, device=costs.device)
    costs = costs[:n]
    valid = valid[:n]
    selected = selected[:n].clamp(0, max(int(costs.shape[1]) - 1, 0))
    rows = torch.arange(n, device=costs.device)
    if elapsed_time is None:
        elapsed = torch.zeros((n, 1), dtype=costs.dtype, device=costs.device)
    else:
        elapsed = torch.as_tensor(elapsed_time, dtype=costs.dtype, device=costs.device)
        if elapsed.dim() == 0:
            elapsed = elapsed.reshape(1, 1).expand(n, 1)
        elif elapsed.dim() == 1:
            elapsed = elapsed.reshape(-1, 1)
        if int(elapsed.shape[0]) == 1 and n != 1:
            elapsed = elapsed.expand(n, int(elapsed.shape[1]))
        elapsed = elapsed[:n]
    remaining = costs.clamp_min(0.0)
    effective = torch.minimum(
        torch.full_like(remaining, float(max(step_interval, 1e-6))),
        remaining,
    )
    mode_key = str(reward_mode or "progress_fraction").strip().lower()
    if mode_key in {"progress_fraction", "step_progress", "trip_progress_fraction"}:
        denom = (remaining + elapsed.clamp_min(0.0)).clamp_min(float(max(remaining_tt_eps, 1e-6)))
        reward_matrix = effective / denom
        if float(progress_cap) > 0.0:
            reward_matrix = reward_matrix.clamp(max=float(progress_cap))
    elif mode_key in {"capped_step_cost", "capped_cost", "step_cost", "min_step_remaining"}:
        reward_matrix = -effective / float(max(reward_temperature, 1e-6))
    else:
        raise ValueError(
            f"Unsupported online no-CFM reward_mode={reward_mode!r}; "
            "expected progress_fraction or capped_step_cost"
        )
    selected_valid = valid[rows, selected]
    rewards = float(reward_weight) * float(reward_scale) * reward_matrix[rows, selected]
    rewards = torch.where(selected_valid, rewards, torch.zeros_like(rewards))
    return torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)


def selected_progress_fraction_rewards(
    *,
    route_costs: np.ndarray | torch.Tensor,
    route_valid: np.ndarray | torch.Tensor,
    selected_idx: np.ndarray | torch.Tensor,
    reward_weight: float = 1.0,
    reward_scale: float = 1.0,
    step_interval: float = 30.0,
    elapsed_time: Optional[np.ndarray | torch.Tensor] = None,
    progress_cap: float = 1.0,
    remaining_tt_eps: float = 1.0,
) -> torch.Tensor:
    return selected_vehicle_route_rewards(
        route_costs=route_costs,
        route_valid=route_valid,
        selected_idx=selected_idx,
        reward_mode="progress_fraction",
        reward_weight=reward_weight,
        reward_scale=reward_scale,
        step_interval=step_interval,
        elapsed_time=elapsed_time,
        progress_cap=progress_cap,
        remaining_tt_eps=remaining_tt_eps,
        reward_temperature=60.0,
    )


def compute_a2c_loss(
    *,
    actor: VehicleRouteActor,
    critic: CentralizedVehicleValueCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    valid_mask: torch.Tensor,
    old_log_probs: Optional[torch.Tensor] = None,
    group_index: Optional[torch.Tensor] = None,
    entropy_coef: float = 0.01,
    value_coef: float = 1.0,
    temperature: float = 1.0,
    advantage_clip: float = 0.0,
    ppo_clip_ratio: float = 0.0,
    normalize_advantages: bool = False,
    entropy_target_frac: float = 0.0,
    entropy_floor_coef: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compute a single online A2C loss from recorded real SUMO samples."""
    if obs.numel() == 0 or int(obs.shape[0]) == 0:
        zero = torch.zeros((), dtype=torch.float32, requires_grad=True)
        return {
            "loss": zero,
            "policy_loss": zero,
            "value_loss": zero,
            "entropy": zero.detach(),
            "n_samples": torch.tensor(0),
        }
    logits = actor.forward_logits(obs)
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    if valid.shape != logits.shape:
        raise ValueError(f"valid_mask shape {tuple(valid.shape)} must match logits {tuple(logits.shape)}")
    has_valid = valid.any(dim=-1)
    if not bool(has_valid.any().detach().cpu().item()):
        zero = logits.sum() * 0.0
        return {
            "loss": zero,
            "policy_loss": zero,
            "value_loss": zero,
            "entropy": zero.detach(),
            "n_samples": torch.tensor(0),
        }
    safe_valid = valid.clone()
    safe_valid[~has_valid, 0] = True
    temp = float(max(temperature, 1e-6))
    masked_logits = (logits / temp).masked_fill(~safe_valid, -1.0e9)
    dist = Categorical(logits=masked_logits)
    action_t = actions.to(device=logits.device, dtype=torch.long).reshape(-1)
    action_t = action_t.clamp(0, int(logits.shape[1]) - 1)
    returns_t = returns.to(device=logits.device, dtype=obs.dtype).reshape(-1)
    values = critic(obs, group_index=group_index).reshape(-1)
    rows = torch.arange(int(logits.shape[0]), device=logits.device)
    selected_valid = safe_valid[rows, action_t[: int(logits.shape[0])]]
    active = has_valid & selected_valid & torch.isfinite(returns_t)
    if not bool(active.any().detach().cpu().item()):
        zero = logits.sum() * 0.0 + values.sum() * 0.0
        return {
            "loss": zero,
            "policy_loss": zero,
            "value_loss": zero,
            "entropy": zero.detach(),
            "n_samples": torch.tensor(0),
            "return_mean": zero.detach(),
            "return_std": zero.detach(),
            "adv_mean": zero.detach(),
            "adv_std": zero.detach(),
            "combined_adv_mean": zero.detach(),
            "combined_adv_std": zero.detach(),
            "clip_frac": zero.detach(),
            "selected_valid_frac": zero.detach(),
            "entropy_floor": zero.detach(),
        }
    log_prob = dist.log_prob(action_t)
    entropy = dist.entropy()
    advantage = returns_t - values.detach()
    combined_adv = advantage.detach()
    if bool(normalize_advantages) and int(combined_adv[active].numel()) > 1:
        ref = combined_adv[active].detach()
        combined_adv = (combined_adv - ref.mean()) / ref.std(unbiased=False).clamp_min(1e-6)
    if float(advantage_clip) > 0.0:
        combined_adv = combined_adv.clamp(min=-float(advantage_clip), max=float(advantage_clip))
    old_logp_used = None
    if old_log_probs is not None and torch.is_tensor(old_log_probs) and old_log_probs.numel() > 0:
        old_candidate = old_log_probs.to(device=logits.device, dtype=obs.dtype).reshape(-1)
        if int(old_candidate.numel()) >= int(log_prob.numel()):
            old_logp_used = old_candidate[: int(log_prob.numel())].detach()
            active = active & torch.isfinite(old_logp_used)
    if old_logp_used is not None and float(ppo_clip_ratio) > 0.0:
        ratio = torch.exp((log_prob - old_logp_used).clamp(min=-20.0, max=20.0))
        clipped_ratio = ratio.clamp(1.0 - float(ppo_clip_ratio), 1.0 + float(ppo_clip_ratio))
        surrogate = torch.minimum(ratio * combined_adv, clipped_ratio * combined_adv)
        policy_loss = -surrogate[active].mean()
        clip_frac = ((ratio[active] - 1.0).abs() > float(ppo_clip_ratio)).to(dtype=obs.dtype).mean()
    else:
        policy_loss = -(log_prob[active] * combined_adv[active]).mean()
        clip_frac = torch.zeros((), device=logits.device, dtype=obs.dtype)
    value_loss = 0.5 * torch.square(values[active] - returns_t[active]).mean()
    entropy_mean = entropy[active].mean()
    entropy_floor = torch.zeros((), device=logits.device, dtype=obs.dtype)
    if float(entropy_floor_coef) > 0.0 and float(entropy_target_frac) > 0.0:
        valid_count = safe_valid[active].sum(dim=-1).to(dtype=obs.dtype).clamp_min(1.0)
        target_entropy = float(entropy_target_frac) * torch.log(valid_count)
        entropy_gap = torch.relu(target_entropy - entropy[active])
        entropy_floor = entropy_gap.square().mean()
    loss = (
        policy_loss
        + float(value_coef) * value_loss
        - float(entropy_coef) * entropy_mean
        + float(entropy_floor_coef) * entropy_floor
    )
    active_returns = returns_t[active].detach()
    active_adv = advantage[active].detach()
    active_combined = combined_adv[active].detach()
    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy_mean.detach(),
        "n_samples": active.sum().detach().cpu(),
        "return_mean": active_returns.mean(),
        "return_std": active_returns.std(unbiased=False) if int(active_returns.numel()) > 1 else torch.zeros_like(active_returns.mean()),
        "adv_mean": active_adv.mean(),
        "adv_std": active_adv.std(unbiased=False) if int(active_adv.numel()) > 1 else torch.zeros_like(active_adv.mean()),
        "combined_adv_mean": active_combined.mean(),
        "combined_adv_std": active_combined.std(unbiased=False) if int(active_combined.numel()) > 1 else torch.zeros_like(active_combined.mean()),
        "clip_frac": clip_frac.detach(),
        "selected_valid_frac": selected_valid.to(dtype=obs.dtype).mean().detach(),
        "entropy_floor": entropy_floor.detach(),
    }


def discounted_vehicle_returns(
    samples: Sequence[OnlineDecisionSample],
    gamma: float,
    *,
    horizon_steps: int = 0,
) -> np.ndarray:
    returns = np.zeros((len(samples),), dtype=np.float32)
    by_vehicle: Dict[Tuple[int, str], List[int]] = {}
    for idx, sample in enumerate(samples):
        by_vehicle.setdefault((int(sample.episode), str(sample.veh_id)), []).append(idx)
    for indices in by_vehicle.values():
        ordered = sorted(indices, key=lambda i: (int(samples[i].decision_idx), int(samples[i].sim_step)))
        if int(horizon_steps) <= 0:
            acc = 0.0
            for idx in reversed(ordered):
                acc = float(samples[idx].reward) + float(gamma) * acc
                returns[idx] = float(acc)
            continue
        for pos, idx in enumerate(ordered):
            acc = 0.0
            discount = 1.0
            for future_idx in ordered[pos : pos + int(horizon_steps)]:
                acc += discount * float(samples[future_idx].reward)
                discount *= float(gamma)
            returns[idx] = float(acc)
    return returns


def save_policy_checkpoint(
    *,
    path: str | Path,
    actor: VehicleRouteActor,
    critic: Optional[CentralizedVehicleValueCritic],
    cfg: OnlineNoCfmConfig,
    episode: int,
    obs_dim: int,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a checkpoint compatible with ``scripts.eval_sumo_policy.load_policy``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg_dict = {
        "training_family": "dreamer_ctde",
        "agent_level": "vehicle",
        "n_routes": int(cfg.n_routes),
        "hidden_dim": int(cfg.hidden_dim),
        "n_hidden_layers": int(cfg.n_hidden_layers),
        "vehicle_actor_architecture": str(cfg.actor_architecture),
        "vehicle_actor_route_cost_logit_bias": float(cfg.route_cost_logit_bias),
        "vehicle_obs_include_global": bool(cfg.include_global),
        "vehicle_obs_include_route_costs": bool(cfg.include_route_costs),
        "vehicle_obs_include_fleet_pressure": bool(cfg.include_fleet_pressure),
        "vehicle_obs_route_stats": str(cfg.route_stats),
        "vehicle_path_penalty_factor": float(cfg.av_candidate_path_penalty_factor),
        "vehicle_reward_scale": float(cfg.reward_scale),
        "gamma": float(cfg.gamma),
        "dreamer_ac_advantage_clip": float(cfg.ac_advantage_clip),
        "dreamer_ac_ppo_clip_ratio": float(cfg.ac_ppo_clip_ratio),
        "dreamer_ac_normalize_combined_advantages": bool(cfg.ac_normalize_advantages),
        "dreamer_ac_entropy_target_frac": float(cfg.ac_entropy_target_frac),
        "dreamer_ac_entropy_floor_coef": float(cfg.ac_entropy_floor_coef),
        "online_replay_buffer_episodes": int(cfg.replay_buffer_episodes),
        "online_replay_update_epochs": int(cfg.replay_update_epochs),
        "online_replay_actor_max_age_episodes": int(cfg.replay_actor_max_age_episodes),
        "online_return_horizon_steps": int(cfg.return_horizon_steps),
        "dreamer_route_cost_reward_scope": "vehicle_critic",
        "dreamer_route_cost_reward_state": "current",
        "dreamer_route_cost_reward_mode": str(cfg.reward_mode),
        "dreamer_route_cost_reward_weight": float(cfg.reward_weight),
        "dreamer_route_cost_reward_normalize_regret": bool(cfg.normalize_regret),
        "dreamer_route_cost_reward_temperature": float(cfg.reward_temperature),
        "dreamer_ac_route_credit_progress_cap": float(cfg.progress_cap),
        "dreamer_ac_route_credit_remaining_tt_eps": float(cfg.remaining_tt_eps),
        "wm_policy_obs_source": str(cfg.wm_policy_obs_source),
        "wm_edge_cost_source": str(cfg.wm_edge_cost_source),
        "wm_candidate_edge_cost_source": str(cfg.wm_candidate_edge_cost_source),
        "step_interval": float(cfg.step_interval),
        "ablation": "online_no_cfm_actor_critic",
        "obs_dim": int(obs_dim),
    }
    state = {
        "model_family": "dreamer_ctde",
        "episode": int(episode),
        "actor": actor.state_dict(),
        "policy": actor.state_dict(),
        "critic": critic.state_dict() if critic is not None else None,
        "vehicle_critic": critic.state_dict() if critic is not None else None,
        "cfg": cfg_dict,
        "metrics": dict(metrics or {}),
    }
    torch.save(state, out)
    return out


class OnlineNoCfmCollector(DataCollector):
    """Real-SUMO online sampler for the no-CFM actor-critic ablation."""

    def __init__(
        self,
        *,
        wm,
        actor: VehicleRouteActor,
        cfg: OnlineNoCfmConfig,
        loaded_hdv_state: Optional[Dict[str, Any]] = None,
        **collector_kwargs: Any,
    ):
        collector_kwargs.setdefault("loaded_hdv_state", loaded_hdv_state)
        candidate_source = str(cfg.av_candidate_source or "current_shortest_freeflow").strip().lower()
        base_candidate_source = "od_suffix" if candidate_source == "od_suffix" else "current_to_dest"
        collector_kwargs.setdefault("av_candidate_source", base_candidate_source)
        super().__init__(**collector_kwargs)
        self._wm_eval = wm
        self._policy_eval = actor
        self._policy_eval_device = next(actor.parameters()).device
        self._cfg = cfg
        self._loaded_hdv_state_eval = loaded_hdv_state
        self._av_candidate_source = candidate_source
        if self._av_candidate_source not in {
            "od_suffix",
            "current_shortest_live",
            "current_shortest_wm",
            "current_shortest_freeflow",
        }:
            raise ValueError(f"Unsupported av_candidate_source={cfg.av_candidate_source!r}")
        self._av_candidate_path_penalty_factor = max(float(cfg.av_candidate_path_penalty_factor), 1.0)
        self._av_candidate_adj_list = _build_edge_adj_list(
            getattr(self._edge_registry, "adjacency", None),
            getattr(self._edge_registry, "passenger_allowed", None),
        )
        self._wm_edge_cost_source = str(cfg.wm_edge_cost_source or "decoded_speed").strip().lower()
        self._wm_candidate_edge_cost_source = str(cfg.wm_candidate_edge_cost_source or "freeflow").strip().lower()
        if self._wm_candidate_edge_cost_source in {"auto", "free_flow", "free-flow", "ff"}:
            self._wm_candidate_edge_cost_source = "freeflow"
        self._wm_decoded_speed_min_ratio = float(cfg.wm_decoded_speed_min_ratio)
        self._wm_decoded_speed_max_ratio = float(cfg.wm_decoded_speed_max_ratio)
        self._wm_decoded_speed_excess_scale = float(cfg.wm_decoded_speed_excess_scale)
        self._vehicle_obs_route_stats = str(cfg.route_stats or "mean").strip().lower()
        self._vehicle_obs_include_route_costs = bool(cfg.include_route_costs)
        self._vehicle_obs_include_fleet_pressure = bool(cfg.include_fleet_pressure)
        self._episode_samples: List[OnlineDecisionSample] = []
        self._all_training_samples: List[OnlineDecisionSample] = []
        self._dynamic_decision_rows: List[Dict[str, Any]] = []
        self._dynamic_agent_decision_rows: List[Dict[str, Any]] = []
        self._current_episode_id_eval = -1
        self._current_decision_idx_eval = 0
        self._av_decision_skip_stats: Dict[str, int] = {}
        self._last_ep_av_travel_times: List[float] = []
        self._last_ep_hdv_travel_times: List[float] = []
        self._last_ep_vehicle_tt: Dict[str, float] = {}
        self._protocol_xml_alias_map: Dict[int, int] = {}

    def _should_update_hdv_after_episode(self, desc: str) -> bool:
        if self._hdv_route_probs is None or self._route_registry is None:
            return False
        if (
            bool(self._cfg.disable_human_learning_after_mutation)
            and str(desc).strip().lower() in {"train", "test"}
        ):
            return False
        return True

    def _should_apply_routerl_hdv_learning(
        self,
        desc: str,
        veh_travel_times: Dict[str, float],
    ) -> bool:
        if not self._should_update_hdv_after_episode(desc):
            return False
        total_agents = int(getattr(self, "_num_agents", 0) or 0)
        if total_agents <= 0:
            return bool(veh_travel_times)
        return len(veh_travel_times) >= total_agents

    def _wm_current_edge_tt(self, z_obs: torch.Tensor, edge_cost_source: Optional[str] = None) -> Optional[np.ndarray]:
        try:
            _edge_dyn, _v_e, _free_flow, edge_tt = edge_tt_from_latent_state(
                wm=self._wm_eval,
                z=z_obs,
                bpr_alpha=0.15,
                bpr_beta=4.0,
                edge_cost_source=str(edge_cost_source or self._wm_edge_cost_source),
                decoded_speed_min_ratio=float(self._wm_decoded_speed_min_ratio),
                decoded_speed_max_ratio=float(self._wm_decoded_speed_max_ratio),
                decoded_speed_excess_scale=float(self._wm_decoded_speed_excess_scale),
            )
        except Exception:
            return None
        out = np.asarray(edge_tt, dtype=np.float32).reshape(-1)
        return out if out.size > 0 else None

    def _live_current_edge_tt(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        edge_ids = list(getattr(self._edge_registry, "edge_ids", []))
        if not edge_ids:
            return None
        lengths = getattr(self._edge_registry, "lengths", None)
        speed_limits = getattr(self._edge_registry, "speed_limits", None)
        lengths_arr = (
            np.asarray(lengths, dtype=np.float32).reshape(-1)
            if lengths is not None
            else np.ones((len(edge_ids),), dtype=np.float32) * 100.0
        )
        speed_limits_arr = (
            np.asarray(speed_limits, dtype=np.float32).reshape(-1)
            if speed_limits is not None
            else np.ones((len(edge_ids),), dtype=np.float32) * 13.89
        )
        n = min(len(edge_ids), int(lengths_arr.shape[0]), int(speed_limits_arr.shape[0]))
        if n <= 0:
            return None
        edge_state = state.get("edges", {}) if isinstance(state, dict) else {}
        out = np.zeros((n,), dtype=np.float32)
        for i, edge_id in enumerate(edge_ids[:n]):
            data = edge_state.get(edge_id, {}) if isinstance(edge_state, dict) else {}
            speed = float(data.get("mean_speed", 0.0) or 0.0) if isinstance(data, dict) else 0.0
            if not np.isfinite(speed) or speed <= 0.1:
                speed = float(speed_limits_arr[i])
            out[i] = float(lengths_arr[i]) / max(float(speed), 0.1)
        return out

    def _freeflow_edge_tt(self) -> Optional[np.ndarray]:
        lengths = getattr(self._edge_registry, "lengths", None)
        speed_limits = getattr(self._edge_registry, "speed_limits", None)
        if lengths is None or speed_limits is None:
            return None
        lengths_arr = np.asarray(lengths, dtype=np.float32).reshape(-1)
        speed_limits_arr = np.asarray(speed_limits, dtype=np.float32).reshape(-1)
        n = min(int(lengths_arr.shape[0]), int(speed_limits_arr.shape[0]))
        if n <= 0:
            return None
        return lengths_arr[:n] / np.maximum(speed_limits_arr[:n], 0.1)

    def _current_to_dest_route_candidates(
        self,
        veh_id: str,
        current_edge: str,
        edge_cost: Optional[np.ndarray],
    ) -> List[List[str]]:
        if build_diverse_paths_current_to_dest is None:
            return []
        if not self._av_candidate_adj_list or edge_cost is None:
            return []
        if (not current_edge) or current_edge.startswith(":"):
            return []
        dest_edge = str(self._veh_dest.get(veh_id, ""))
        if (not dest_edge) or dest_edge.startswith(":"):
            return []
        edge_to_idx = getattr(self._edge_registry, "_edge_to_idx", {})
        edge_ids = list(getattr(self._edge_registry, "edge_ids", []))
        if current_edge not in edge_to_idx or dest_edge not in edge_to_idx:
            return []
        costs = np.asarray(edge_cost, dtype=np.float32).reshape(-1)
        n_edges = len(edge_ids)
        if n_edges <= 0 or costs.shape[0] < n_edges:
            return []
        paths = build_diverse_paths_current_to_dest(
            adj_list=self._av_candidate_adj_list,
            edge_tt=costs[:n_edges],
            src_idx=int(edge_to_idx[current_edge]),
            dst_idx=int(edge_to_idx[dest_edge]),
            K=int(self.n_routes),
            penalty_factor=float(self._av_candidate_path_penalty_factor),
        )
        named: List[List[str]] = []
        for path in paths[: int(self.n_routes)]:
            route = [
                edge_ids[int(edge_idx)]
                for edge_idx in path
                if 0 <= int(edge_idx) < n_edges
            ]
            if route and route[0] == current_edge:
                named.append(route)
        if not named:
            return []
        while len(named) < int(self.n_routes):
            named.append(list(named[0]))
        return named[: int(self.n_routes)]

    def _vehicle_elapsed_seconds(self, veh_ids: Sequence[str], sim_step: int) -> np.ndarray:
        agents_df = getattr(self.connector, "_agents_df", None)
        start_by_id: Dict[str, float] = {}
        if agents_df is not None and {"id", "start_time"}.issubset(set(agents_df.columns)):
            try:
                ids = agents_df["id"].astype(str).tolist()
                starts = agents_df["start_time"].astype(float).tolist()
                start_by_id = {str(veh_id): float(start) for veh_id, start in zip(ids, starts)}
            except Exception:
                start_by_id = {}
        elapsed: List[float] = []
        for veh_id in veh_ids:
            depart = start_by_id.get(str(veh_id), float(sim_step))
            elapsed.append(max(0.0, float(sim_step) - float(depart)))
        return np.asarray(elapsed, dtype=np.float32)

    def _select_av_actions(self, state: Dict[str, Any], rng: np.random.RandomState) -> Dict[str, List[str]]:
        stats = getattr(self, "_av_decision_skip_stats", None)
        if stats is None:
            stats = {}
            self._av_decision_skip_stats = stats

        def _inc(name: str, amount: int = 1) -> None:
            stats[name] = int(stats.get(name, 0)) + int(amount)

        _inc("calls")
        in_network = state.get("vehicles", {})
        active_avs = sorted(self._av_vehicles & set(in_network.keys()))
        _inc("active_av_seen", len(active_avs))
        if not active_avs:
            _inc("no_active_av_calls")
            return {}

        try:
            with torch.no_grad():
                z_obs = self._wm_eval.encode_observation(state)
                edge_feats_dyn = (
                    self._wm_eval.latent_edge_features(z_obs)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
        except Exception:
            _inc("wm_obs_failed_calls")
            return {}

        edge_to_idx = getattr(self._edge_registry, "_edge_to_idx", {})
        wm_edge_cost = self._wm_current_edge_tt(z_obs)
        live_edge_cost = self._live_current_edge_tt(state)
        freeflow_edge_cost = self._freeflow_edge_tt()
        obs_edge_cost = wm_edge_cost if wm_edge_cost is not None else freeflow_edge_cost
        if self._wm_candidate_edge_cost_source == "freeflow":
            candidate_edge_cost = freeflow_edge_cost
        elif self._wm_candidate_edge_cost_source == self._wm_edge_cost_source:
            candidate_edge_cost = wm_edge_cost
        else:
            candidate_edge_cost = self._wm_current_edge_tt(z_obs, self._wm_candidate_edge_cost_source)
        if self._av_candidate_source == "current_shortest_live":
            candidate_edge_cost = live_edge_cost
        elif self._av_candidate_source == "current_shortest_freeflow":
            candidate_edge_cost = freeflow_edge_cost
        if candidate_edge_cost is None:
            _inc("candidate_edge_cost_missing_calls")

        path_batch: List[List[List[int]]] = []
        route_batch: List[List[List[str]]] = []
        active_ids: List[str] = []
        active_current_edges: List[str] = []
        candidate_dynamic_count = 0
        candidate_fallback_count = 0
        for veh_id in active_avs:
            veh_info = in_network.get(veh_id)
            if not isinstance(veh_info, dict):
                _inc("skip_bad_vehicle_info")
                continue
            current_edge = str(veh_info.get("edge", ""))
            if (not current_edge) or current_edge.startswith(":"):
                _inc("skip_bad_current_edge")
                continue
            candidates: List[List[str]] = []
            if self._av_candidate_source != "od_suffix":
                candidates = self._current_to_dest_route_candidates(
                    veh_id,
                    current_edge,
                    candidate_edge_cost,
                )
                if candidates:
                    candidate_dynamic_count += 1
                else:
                    candidate_fallback_count += 1
            if not candidates:
                candidates = self._candidate_route_suffixes(veh_id, current_edge)
            if candidates:
                candidates = self._filter_safe_reroute_candidates(veh_id, current_edge, candidates)
            if not candidates:
                _inc("skip_no_candidates")
                continue
            idx_paths: List[List[int]] = []
            named_paths: List[List[str]] = []
            for route in candidates[: int(self.n_routes)]:
                idxs = [int(edge_to_idx[e]) for e in route if e in edge_to_idx]
                if idxs:
                    idx_paths.append(idxs)
                    named_paths.append(list(route))
            if not idx_paths:
                _inc("skip_no_idx_paths")
                continue
            while len(idx_paths) < int(self.n_routes):
                idx_paths.append(list(idx_paths[0]))
                named_paths.append(list(named_paths[0]))
            path_batch.append(idx_paths[: int(self.n_routes)])
            route_batch.append(named_paths[: int(self.n_routes)])
            active_ids.append(veh_id)
            active_current_edges.append(current_edge)

        if not path_batch:
            _inc("no_path_batch_calls")
            return {}
        _inc("selected_vehicle_count", len(active_ids))

        obs_base = build_vehicle_obs_vectorized(
            edge_feats_dyn=edge_feats_dyn,
            path_batch=path_batch,
            include_global=bool(self._cfg.include_global),
            route_stats=self._vehicle_obs_route_stats,
            edge_cost=obs_edge_cost,
            include_route_costs=bool(self._vehicle_obs_include_route_costs),
            include_fleet_pressure=False,
        )

        selected_indices = np.zeros((len(active_ids),), dtype=np.int64)
        policy_probs: List[np.ndarray] = []
        obs_rows: List[np.ndarray] = []
        if self._vehicle_obs_include_fleet_pressure:
            fleet_counts = np.zeros((int(self.n_routes),), dtype=np.float32)
            fleet_pressure = np.zeros((len(getattr(self._edge_registry, "edge_ids", [])),), dtype=np.float32)
            for i in range(len(active_ids)):
                fleet_i = build_vehicle_fleet_pressure_obs_features(
                    path_batch=[path_batch[i]],
                    n_edges=len(getattr(self._edge_registry, "edge_ids", [])),
                    n_routes=int(self.n_routes),
                    selected_route_counts=fleet_counts,
                    selected_edge_pressure=fleet_pressure,
                )
                obs_i_np = _append_vehicle_context_features(
                    np.concatenate([obs_base[i : i + 1], fleet_i], axis=1),
                    step_idx=int(getattr(self.connector, "current_step", 0)),
                    episode_steps=int(self._cfg.episode_steps),
                    n_active_av=len(active_ids),
                    n_total_av=max(len(self._av_vehicles), 1),
                )
                obs_i = torch.from_numpy(obs_i_np).to(device=self._policy_eval_device, dtype=torch.float32)
                with torch.no_grad():
                    logits_i = self._policy_eval.forward_logits(obs_i)
                    probs_i = torch.softmax(logits_i, dim=-1).detach().cpu().numpy().astype(np.float32)[0]
                probs_i = _temperature_adjust_probs(
                    probs_i,
                    float(self._cfg.av_policy_sample_temperature),
                    int(self.n_routes),
                )
                selected_i = int(rng.choice(int(self.n_routes), p=probs_i))
                selected_i = int(np.clip(selected_i, 0, len(route_batch[i]) - 1))
                selected_indices[i] = selected_i
                policy_probs.append(probs_i.astype(np.float32))
                obs_rows.append(obs_i_np[0].astype(np.float32))
                fleet_counts[selected_i] += 1.0
                for edge_idx in path_batch[i][selected_i]:
                    edge_i = int(edge_idx)
                    if 0 <= edge_i < int(fleet_pressure.shape[0]):
                        fleet_pressure[edge_i] += 1.0
            obs_vehicle_np = np.stack(obs_rows, axis=0).astype(np.float32)
            probs_np = np.stack(policy_probs, axis=0).astype(np.float32)
        else:
            obs_vehicle_np = _append_vehicle_context_features(
                obs_base,
                step_idx=int(getattr(self.connector, "current_step", 0)),
                episode_steps=int(self._cfg.episode_steps),
                n_active_av=len(active_ids),
                n_total_av=max(len(self._av_vehicles), 1),
            )
            obs_t = torch.from_numpy(obs_vehicle_np).to(device=self._policy_eval_device, dtype=torch.float32)
            with torch.no_grad():
                logits = self._policy_eval.forward_logits(obs_t)
                probs_np = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
            for i in range(len(active_ids)):
                probs = _temperature_adjust_probs(
                    probs_np[i],
                    float(self._cfg.av_policy_sample_temperature),
                    int(self.n_routes),
                )
                selected_indices[i] = int(rng.choice(int(self.n_routes), p=probs))
                selected_indices[i] = int(np.clip(selected_indices[i], 0, len(route_batch[i]) - 1))
                probs_np[i] = probs.astype(np.float32)

        live_cost_matrix = (
            _edge_route_cost_matrix(
                path_batch,
                live_edge_cost,
                n_routes=int(self.n_routes),
                length_norm=str(self._cfg.route_cost_length_norm),
            )
            if live_edge_cost is not None
            else np.full((len(active_ids), int(self.n_routes)), np.inf, dtype=np.float32)
        )
        valid_mask = route_valid_mask_from_path_batch(path_batch, n_routes=int(self.n_routes))
        elapsed = self._vehicle_elapsed_seconds(
            active_ids,
            int(getattr(self.connector, "current_step", 0)),
        )
        rewards = selected_vehicle_route_rewards(
            route_costs=live_cost_matrix,
            route_valid=valid_mask,
            selected_idx=selected_indices,
            reward_mode=str(self._cfg.reward_mode),
            reward_weight=float(self._cfg.reward_weight),
            reward_scale=float(self._cfg.reward_scale),
            step_interval=float(self._cfg.step_interval),
            elapsed_time=elapsed,
            progress_cap=float(self._cfg.progress_cap),
            remaining_tt_eps=float(self._cfg.remaining_tt_eps),
            reward_temperature=float(self._cfg.reward_temperature),
        ).detach().cpu().numpy().astype(np.float32)

        actions: Dict[str, List[str]] = {}
        for i, veh_id in enumerate(active_ids):
            ridx = int(np.clip(int(selected_indices[i]), 0, len(route_batch[i]) - 1))
            self._av_route_idx[veh_id] = ridx
            actions[veh_id] = route_batch[i][ridx]
            sample = OnlineDecisionSample(
                episode=int(self._current_episode_id_eval),
                decision_idx=int(self._current_decision_idx_eval),
                sim_step=int(getattr(self.connector, "current_step", -1)),
                veh_id=str(veh_id),
                obs=obs_vehicle_np[i].astype(np.float32),
                action=int(ridx),
                old_log_prob=float(np.log(max(float(probs_np[i, ridx]), 1e-8))),
                reward=float(rewards[i]),
                valid_mask=valid_mask[i].astype(bool),
                route_costs=live_cost_matrix[i].astype(np.float32),
                elapsed_time=float(elapsed[i]),
                path_batch=[list(path) for path in path_batch[i]],
                current_edge_idx=int(edge_to_idx.get(active_current_edges[i], -1))
                if i < len(active_current_edges)
                else -1,
                dest_edge_idx=int(edge_to_idx.get(str(self._veh_dest.get(veh_id, "")), -1)),
            )
            self._episode_samples.append(sample)
            self._all_training_samples.append(sample)

        unique_mean, unique_min, unique_max = _route_batch_uniqueness_summary(route_batch)
        counts = np.bincount(selected_indices.clip(0, int(self.n_routes) - 1), minlength=int(self.n_routes))
        mean_probs = np.mean(probs_np[:, : int(self.n_routes)], axis=0)
        decision_row: Dict[str, Any] = {
            "episode": int(self._current_episode_id_eval),
            "decision_idx": int(self._current_decision_idx_eval),
            "sim_step": int(getattr(self.connector, "current_step", -1)),
            "active_av": int(len(active_ids)),
            "candidate_dynamic_count": int(candidate_dynamic_count),
            "candidate_fallback_count": int(candidate_fallback_count),
            "candidate_unique_mean": float(unique_mean),
            "candidate_unique_min": int(unique_min),
            "candidate_unique_max": int(unique_max),
            "reward_mean": float(np.mean(rewards)) if len(rewards) else float("nan"),
            "reward_sum": float(np.sum(rewards)) if len(rewards) else 0.0,
            "cfm_transition_used": 0,
            "wm_rollout_rerank_used": 0,
        }
        for k in range(int(self.n_routes)):
            decision_row[f"mean_prob_route{k}"] = float(mean_probs[k])
            decision_row[f"selected_count_route{k}"] = int(counts[k])
            vals = live_cost_matrix[:, k]
            vals = vals[np.isfinite(vals)]
            decision_row[f"live_mean_route_cost_route{k}"] = float(vals.mean()) if vals.size else float("nan")
        self._dynamic_decision_rows.append(decision_row)

        for i, veh_id in enumerate(active_ids):
            agent_row: Dict[str, Any] = {
                "episode": int(self._current_episode_id_eval),
                "decision_idx": int(self._current_decision_idx_eval),
                "sim_step": int(getattr(self.connector, "current_step", -1)),
                "veh_id": str(veh_id),
                "current_edge": active_current_edges[i] if i < len(active_current_edges) else "",
                "dest_edge": str(self._veh_dest.get(veh_id, "")),
                "selected_route": int(selected_indices[i]),
                "reward": float(rewards[i]),
                "elapsed_time": float(elapsed[i]),
            }
            for k in range(int(self.n_routes)):
                agent_row[f"prob_route{k}"] = float(probs_np[i, k])
                agent_row[f"valid_route{k}"] = int(bool(valid_mask[i, k]))
                agent_row[f"live_cost_route{k}"] = (
                    float(live_cost_matrix[i, k])
                    if np.isfinite(live_cost_matrix[i, k])
                    else float("nan")
                )
            self._dynamic_agent_decision_rows.append(agent_row)
        return actions

    def _write_urb_episode_csv(self, episode_id: int, episode_reward: float) -> Optional[Path]:
        agents_df = getattr(self.connector, "_agents_df", None)
        if agents_df is None or len(agents_df) == 0:
            return None
        records_folder = Path(getattr(self.connector, "records_folder", ""))
        episodes_dir = records_folder / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        fixed_machine_ids = set(getattr(self, "_fixed_machine_ids", set()) or set())
        fixed_human_ids = set(getattr(self, "_fixed_human_ids", set()) or set())
        rows: List[Dict[str, Any]] = []
        initial_route_choices = getattr(self.connector, "_episode_route_choices", {}) or {}
        for _, row in agents_df.iterrows():
            veh_id = str(row.get("id"))
            if veh_id in fixed_machine_ids or veh_id in self._av_vehicles:
                vtype = "AV"
            elif veh_id in fixed_human_ids or veh_id in self._hdv_vehicles:
                vtype = "Human"
            else:
                vtype = str(row.get("kind", "Human"))
            route_idx = self._av_route_idx.get(veh_id, self._hdv_route_idx.get(veh_id, -1))
            rows.append(
                {
                    "id": int(float(row.get("id", -1))),
                    "travel_time": float(self._last_ep_vehicle_tt.get(veh_id, np.nan)),
                    "action": int(route_idx),
                    "initial_route_action": int(initial_route_choices.get(veh_id, -1)),
                    "policy_action": int(self._av_route_idx.get(veh_id, -1)),
                    "hdv_action": int(self._hdv_route_idx.get(veh_id, -1)),
                    "seen_in_episode": int(veh_id in self._av_vehicles or veh_id in self._hdv_vehicles),
                    "origin": int(float(row.get("origin", -1))),
                    "destination": int(float(row.get("destination", -1))),
                    "start_time": int(float(row.get("start_time", 0))),
                    "reward": float(episode_reward),
                    "vType": vtype,
                }
            )
        out_path = episodes_dir / f"ep{int(episode_id)}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        return out_path

    def _register_protocol_xml_aliases(self, xml_paths: set[str], episode_id: int) -> None:
        for src_str in sorted(xml_paths):
            stem = Path(src_str).stem
            _prefix, sep, suffix = stem.rpartition("_")
            if sep and suffix.isdigit():
                self._protocol_xml_alias_map[int(suffix)] = int(episode_id)

    def finalize_protocol_xml_aliases(self) -> None:
        records_folder = str(getattr(self.connector, "records_folder", ""))
        if not records_folder:
            return
        sumo_output_folder = Path(records_folder) / "SUMO_output"
        for src in sorted(sumo_output_folder.glob("*.xml")):
            stem = src.stem
            prefix, sep, suffix = stem.rpartition("_")
            if (not sep) or (not suffix.isdigit()):
                continue
            protocol_episode_id = self._protocol_xml_alias_map.get(int(suffix))
            if protocol_episode_id is None:
                continue
            _wait_for_expected_sumo_xml(src)
            dst = src.with_name(f"{prefix}_{protocol_episode_id}{src.suffix}")
            if src.resolve() != dst.resolve():
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    import shutil

                    shutil.copy2(src, dst)
                except Exception:
                    pass

    def _collect_single_episode(self, episode_id: int, max_steps: int) -> Tuple[List[OnlineDecisionSample], Dict[str, Any]]:
        rng = np.random.RandomState((self.seed or 0) + int(episode_id))
        self._episode_samples = []
        self._av_decision_skip_stats = {}
        self._av_vehicles.clear()
        self._hdv_vehicles.clear()
        self._random_avs.clear()
        self._greedy_avs.clear()
        self._av_route_idx.clear()
        self._hdv_route_idx.clear()
        self._veh_dest.clear()
        self._route_cache.clear()
        self._veh_candidates.clear()
        self._veh_od.clear()
        self._last_ep_travel_times.clear()
        self._last_ep_av_travel_times.clear()
        self._last_ep_hdv_travel_times.clear()
        self._last_ep_vehicle_tt.clear()
        self._last_ep_route_tts.clear()

        veh_depart: Dict[str, float] = {}
        veh_travel_times: Dict[str, float] = {}
        self._seed_depart_times_from_agents(veh_depart)
        state = self.connector.reset(
            route_probs=self._hdv_route_probs,
            human_learning_state=self._loaded_hdv_state_eval,
        )
        if self._fixed_machine_ids or self._fixed_human_ids:
            self._assign_roles(list(self._fixed_machine_ids | self._fixed_human_ids), rng)
        self._register_vehicles(list(state.get("vehicles", {}).keys()), rng)
        self._record_depart_times(state, veh_depart)
        reset_arrivals = getattr(self.connector, "_last_reset_arrival_events", None)
        if reset_arrivals:
            self._record_arrivals(
                veh_depart,
                veh_travel_times,
                track_role_breakdown=True,
                arrival_events=reset_arrivals,
            )

        decision_ticks = sorted(self._decision_tick_map(max_steps).keys())
        if not decision_ticks:
            decision_ticks = list(range(max_steps))
        current_tick = 0
        ep_rewards: List[float] = []
        ep_decisions = 0
        for decision_idx, tick in enumerate(decision_ticks):
            steps_to_boundary = max(int(tick) - int(current_tick), 0)
            if steps_to_boundary > 0:
                arrivals = self.connector.advance_steps(steps_to_boundary)
                current_tick += steps_to_boundary
                self._record_arrivals(
                    veh_depart,
                    veh_travel_times,
                    track_role_breakdown=True,
                    arrival_events=arrivals,
                )
                state = self.connector.get_traffic_state()
            self._register_vehicles(list(state.get("vehicles", {}).keys()), rng)
            self._record_depart_times(state, veh_depart)
            self._current_episode_id_eval = int(episode_id)
            self._current_decision_idx_eval = int(decision_idx)
            before = len(self._episode_samples)
            av_actions = self._select_av_actions(state, rng)
            if len(self._episode_samples) > before:
                ep_rewards.extend([s.reward for s in self._episode_samples[before:]])
            next_tick = (
                int(decision_ticks[decision_idx + 1])
                if (decision_idx + 1) < len(decision_ticks)
                else int(max_steps)
            )
            advance_steps = max(next_tick - int(tick), 1)
            if av_actions:
                self._apply_reroutings(av_actions)
            arrivals = self.connector.advance_steps(advance_steps)
            current_tick = int(tick) + int(advance_steps)
            self._record_arrivals(
                veh_depart,
                veh_travel_times,
                track_role_breakdown=True,
                arrival_events=arrivals,
            )
            state = self.connector.get_traffic_state()
            ep_decisions += 1
            if self.connector.is_done() or (len(state.get("vehicles", {})) == 0 and current_tick > 10):
                break

        if (
            self._use_routerl_hdv_blackbox
            and hasattr(self.connector, "apply_routerl_hdv_learning")
            and self._should_apply_routerl_hdv_learning("train", veh_travel_times)
        ):
            self.connector.apply_routerl_hdv_learning(veh_travel_times)

        episode_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        self._write_urb_episode_csv(int(episode_id), episode_reward)
        total_agents = int(getattr(self, "_num_agents", 0) or 0)
        row = {
            "episode": int(episode_id),
            "phase": "train",
            "steps": int(ep_decisions),
            "samples": int(len(self._episode_samples)),
            "reward_mean": float(np.mean(ep_rewards)) if ep_rewards else float("nan"),
            "reward_sum": float(np.sum(ep_rewards)) if ep_rewards else 0.0,
            "arrived_all": len(self._last_ep_travel_times),
            "arrived_av": len(self._last_ep_av_travel_times),
            "arrived_hdv": len(self._last_ep_hdv_travel_times),
            "total_agents": total_agents,
            "unfinished_all": max(total_agents - len(self._last_ep_travel_times), 0),
            "completion_rate": float(len(self._last_ep_travel_times)) / float(max(total_agents, 1)),
            "t_all": float(np.mean(self._last_ep_travel_times) / 60.0) if self._last_ep_travel_times else float("nan"),
            "t_CAV": float(np.mean(self._last_ep_av_travel_times) / 60.0) if self._last_ep_av_travel_times else float("nan"),
            "t_HDV": float(np.mean(self._last_ep_hdv_travel_times) / 60.0) if self._last_ep_hdv_travel_times else float("nan"),
            "episode_complete": int(
                _episode_complete_for_dataset(
                    n_arrived=len(self._last_ep_travel_times),
                    total_agents=total_agents,
                    n_steps=int(ep_decisions),
                )
            ),
        }
        for key, value in sorted(getattr(self, "_av_decision_skip_stats", {}).items()):
            row[f"av_diag_{key}"] = int(value)
        return list(self._episode_samples), row

    def run_train_episode(self, episode_id: int, max_steps_per_episode: int) -> Tuple[List[OnlineDecisionSample], Dict[str, Any]]:
        records_folder = str(getattr(self.connector, "records_folder", ""))
        sumo_output_folder = os.path.join(records_folder, "SUMO_output")
        xml_snapshot_before = _snapshot_sumo_xmls(sumo_output_folder)
        samples, row = self._collect_single_episode(
            episode_id=int(episode_id),
            max_steps=int(max_steps_per_episode),
        )
        if self._should_update_hdv_after_episode("train"):
            if self._last_ep_route_tts:
                self._update_hdv_gawron_actual()
            elif samples:
                state = self.connector.get_traffic_state()
                self._update_hdv_gawron(self._collect_flow(state))
        new_xmls = _new_sumo_xmls_since(xml_snapshot_before, sumo_output_folder, fallback_latest=2)
        self._register_protocol_xml_aliases(new_xmls, int(episode_id))
        return samples, row


def samples_to_tensors(
    samples: Sequence[OnlineDecisionSample],
    *,
    device: torch.device,
    gamma: float,
    return_horizon_steps: int = 0,
) -> Dict[str, torch.Tensor]:
    if not samples:
        return {
            "obs": torch.zeros((0, 0), device=device),
            "actions": torch.zeros((0,), dtype=torch.long, device=device),
            "returns": torch.zeros((0,), device=device),
            "old_log_probs": torch.zeros((0,), device=device),
            "valid_mask": torch.zeros((0, 0), dtype=torch.bool, device=device),
            "group_index": torch.zeros((0,), dtype=torch.long, device=device),
        }
    returns = discounted_vehicle_returns(
        samples,
        gamma=float(gamma),
        horizon_steps=int(return_horizon_steps),
    )
    obs_np = np.stack([s.obs for s in samples], axis=0).astype(np.float32)
    valid_np = np.stack([s.valid_mask for s in samples], axis=0).astype(bool)
    group_ids: Dict[Tuple[int, int], int] = {}
    groups: List[int] = []
    for sample in samples:
        key = (int(sample.episode), int(sample.decision_idx))
        if key not in group_ids:
            group_ids[key] = len(group_ids)
        groups.append(group_ids[key])
    group_np = np.asarray(groups, dtype=np.int64)
    return {
        "obs": torch.from_numpy(obs_np).to(device=device, dtype=torch.float32),
        "actions": torch.tensor([int(s.action) for s in samples], device=device, dtype=torch.long),
        "returns": torch.from_numpy(returns).to(device=device, dtype=torch.float32),
        "old_log_probs": torch.tensor([float(s.old_log_prob) for s in samples], device=device, dtype=torch.float32),
        "valid_mask": torch.from_numpy(valid_np).to(device=device, dtype=torch.bool),
        "group_index": torch.from_numpy(group_np).to(device=device, dtype=torch.long),
    }


class OnlineNoCfmTrainer:
    """Owns actor-critic construction, SUMO collection, and A2C updates."""

    def __init__(self, cfg: OnlineNoCfmConfig):
        self.cfg = cfg.resolve()
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(int(self.cfg.seed))
        np.random.seed(int(self.cfg.seed))
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self.wm = self._load_world_model()
        self.obs_dim = vehicle_mode_obs_dim(
            n_routes=int(self.cfg.n_routes),
            d_obs=int(self.wm.d_obs),
            include_global=bool(self.cfg.include_global),
            route_stats=str(self.cfg.route_stats),
            include_route_costs=bool(self.cfg.include_route_costs),
            include_fleet_pressure=bool(self.cfg.include_fleet_pressure),
        )
        self.actor = VehicleRouteActor(
            self.obs_dim,
            int(self.cfg.n_routes),
            int(self.cfg.hidden_dim),
            int(self.cfg.n_hidden_layers),
            route_cost_logit_bias=float(self.cfg.route_cost_logit_bias),
            architecture=str(self.cfg.actor_architecture),
            include_global=bool(self.cfg.include_global),
            include_route_costs=bool(self.cfg.include_route_costs),
            include_fleet_pressure=bool(self.cfg.include_fleet_pressure),
        ).to(self.device)
        self.critic = CentralizedVehicleValueCritic(
            self.obs_dim,
            context_dim=0,
            hidden_dim=int(self.cfg.hidden_dim),
            n_hidden_layers=int(self.cfg.n_hidden_layers),
            architecture=str(self.cfg.critic_architecture),
            n_attention_heads=int(self.cfg.critic_attention_heads),
            output_dim=1,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(self.cfg.lr))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(self.cfg.critic_lr))
        self.collector: Optional[OnlineNoCfmCollector] = None
        self.loaded_resume_episode = self._load_resume_checkpoint()

    def _load_resume_checkpoint(self) -> int:
        ckpt_path = str(getattr(self.cfg, "resume_checkpoint", "") or "").strip()
        if not ckpt_path:
            return 0
        path = _resolve_path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"resume_checkpoint does not exist: {path}")
        state = torch.load(path, map_location=self.device)
        actor_state = state.get("actor") or state.get("policy")
        if actor_state is None:
            raise KeyError(f"resume checkpoint has no actor/policy state: {path}")
        self.actor.load_state_dict(actor_state)
        critic_state = state.get("critic") or state.get("vehicle_critic")
        if critic_state is not None:
            self.critic.load_state_dict(critic_state)
        episode = int(state.get("episode", 0) or 0)
        print(
            f"[online-no-cfm] resumed actor/critic from {path} at episode {episode}",
            flush=True,
        )
        return episode

    def _load_world_model(self):
        from dreamer4.urban_routing.urban_world_model import UrbanWorldModel

        wm = UrbanWorldModel(
            flow_projector_ckpt=str(self.cfg.projector_ckpt),
            dynamics_ckpt=str(self.cfg.dynamics_ckpt),
            edge_registry_path=str(self.cfg.edge_registry),
            device=str(self.device),
            use_amp=True,
            cfm_init_mode=str(self.cfg.wm_cfm_init_mode),
            policy_obs_source=str(self.cfg.wm_policy_obs_source),
        )
        wm.eval()
        return wm

    def _build_collector(self) -> OnlineNoCfmCollector:
        patch_traci_start_port_fallback()
        output_dir = self.output_dir / "collector_runtime"
        output_dir.mkdir(parents=True, exist_ok=True)
        post_state = _load_json_object(Path(self.cfg.post_mutation_state_in))
        if post_state is not None:
            reused = _reuse_route_artifacts_from_refs(post_state.get("artifact_refs", {}), output_dir)
        else:
            reused = _reuse_route_artifacts(Path(self.cfg.route_artifacts_dir).resolve(), output_dir)
        loaded_hdv_state = _load_hdv_learning_state(Path(self.cfg.hdv_state_in))
        network_folder = str(Path(self.cfg.network_dir).resolve())
        od_data = load_od_data(network_folder, self.cfg.network_name)
        agents_df = pd.read_csv(reused["agents_csv"])
        num_agents, _num_machines, max_start_time = compute_agent_counts(
            agents_df,
            float(self.cfg.ratio_machines),
        )
        sim_length = int(
            max(
                float(max_start_time),
                float(agents_df["start_time"].max()) if "start_time" in agents_df.columns else float(max_start_time),
            )
            + float(self.cfg.sim_buffer_after_last_depart)
        )
        collector = OnlineNoCfmCollector(
            wm=self.wm,
            actor=self.actor,
            cfg=self.cfg,
            loaded_hdv_state=loaded_hdv_state,
            output_dir=str(output_dir),
            shard_size=2048,
            seed=int(self.cfg.seed),
            backend="routerl",
            network_name=str(self.cfg.network_name),
            network_folder=network_folder,
            origins=od_data["origins"],
            destinations=od_data["destinations"],
            number_of_paths=int(self.cfg.n_routes),
            num_agents=int(num_agents),
            simulation_timesteps=int(sim_length),
            decision_steps=int(self.cfg.episode_steps),
            sim_buffer_after_last_depart=int(self.cfg.sim_buffer_after_last_depart),
            sumo_depart_lane=str(self.cfg.sumo_depart_lane),
            sumo_depart_pos=str(self.cfg.sumo_depart_pos),
            sumo_depart_speed=str(self.cfg.sumo_depart_speed),
            sumo_time_to_teleport=str(self.cfg.sumo_time_to_teleport),
            sumo_eager_insert=bool(self.cfg.sumo_eager_insert),
            sumo_sloppy_insert=bool(self.cfg.sumo_sloppy_insert),
            sumo_emergency_insert=bool(self.cfg.sumo_emergency_insert),
            sumo_extra_args=str(self.cfg.sumo_extra_args),
            incident_edge=str(self.cfg.incident_edge),
            incident_start=float(self.cfg.incident_start),
            incident_end=float(self.cfg.incident_end),
            incident_speed=float(self.cfg.incident_speed),
            records_folder=str(output_dir),
            av_fraction=float(self.cfg.ratio_machines),
            random_av_fraction=0.0,
            epsilon=0.0,
            agents_csv=str(reused["agents_csv"]),
            edge_registry_path=str(self.cfg.edge_registry),
            hdv_gawron_mode=str(self.cfg.hdv_gawron_mode),
            hdv_gawron_alpha_zero=float(self.cfg.hdv_gawron_alpha_zero),
            hdv_gawron_beta=float(self.cfg.hdv_gawron_beta),
            hdv_gawron_update_mode=str(self.cfg.hdv_gawron_update_mode),
            od_file=os.path.join(network_folder, f"od_{self.cfg.network_name}.txt"),
            route_registry_path=os.path.join(network_folder, "route_registry.json"),
            skip_hdv_route_registry=False,
            use_light_hdv_route_registry=True,
        )
        if post_state is not None:
            collector.set_fixed_roles(
                machine_ids=[str(x) for x in post_state.get("machine_agent_ids", [])],
                human_ids=[str(x) for x in post_state.get("human_agent_ids", [])],
            )
            if loaded_hdv_state:
                collector._loaded_hdv_state_eval = loaded_hdv_state
            if hasattr(collector.connector, "_episode_count"):
                collector.connector._episode_count = int(post_state.get("mutation_episode_index", 0))
        self.collector = collector
        self.rollout_max_steps = (
            int(self.cfg.max_steps_per_episode)
            if int(self.cfg.max_steps_per_episode) > 0
            else int(sim_length)
        )
        return collector

    def update_from_episode(self, samples: Sequence[OnlineDecisionSample]) -> Dict[str, float]:
        return self.update_from_replay([samples])

    def update_from_replay(self, episode_batches: Sequence[Sequence[OnlineDecisionSample]]) -> Dict[str, float]:
        samples: List[OnlineDecisionSample] = [
            sample
            for batch in episode_batches
            for sample in batch
        ]
        if len(samples) < int(self.cfg.min_samples_per_update):
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "n_samples": float(len(samples)),
                "replay_episodes": float(len(episode_batches)),
                "update_epochs": 0.0,
                "skipped": 1.0,
            }
        tensors = samples_to_tensors(
            samples,
            device=self.device,
            gamma=float(self.cfg.gamma),
            return_horizon_steps=int(self.cfg.return_horizon_steps),
        )
        self.actor.train()
        self.critic.train()
        results: List[Dict[str, torch.Tensor]] = []
        update_epochs = max(int(self.cfg.replay_update_epochs), 1)
        for _epoch in range(update_epochs):
            result = compute_a2c_loss(
                actor=self.actor,
                critic=self.critic,
                obs=tensors["obs"],
                actions=tensors["actions"],
                returns=tensors["returns"],
                valid_mask=tensors["valid_mask"],
                old_log_probs=tensors["old_log_probs"],
                group_index=tensors["group_index"],
                entropy_coef=float(self.cfg.entropy_coef),
                value_coef=float(self.cfg.value_coef),
                temperature=float(self.cfg.av_policy_sample_temperature),
                advantage_clip=float(self.cfg.ac_advantage_clip),
                ppo_clip_ratio=float(self.cfg.ac_ppo_clip_ratio),
                normalize_advantages=bool(self.cfg.ac_normalize_advantages),
                entropy_target_frac=float(self.cfg.ac_entropy_target_frac),
                entropy_floor_coef=float(self.cfg.ac_entropy_floor_coef),
            )
            loss = result["loss"]
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.cfg.max_grad_norm))
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.cfg.max_grad_norm))
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            results.append(result)

        result = results[-1]

        def _mean_result(name: str) -> float:
            values = [
                float(item[name].detach().cpu().item())
                for item in results
                if name in item and torch.is_tensor(item[name])
            ]
            return float(np.mean(values)) if values else 0.0

        return {
            "loss": _mean_result("loss"),
            "policy_loss": _mean_result("policy_loss"),
            "value_loss": _mean_result("value_loss"),
            "entropy": _mean_result("entropy"),
            "n_samples": float(result["n_samples"].detach().cpu().item()),
            "replay_episodes": float(len(episode_batches)),
            "update_epochs": float(update_epochs),
            "return_mean": _mean_result("return_mean"),
            "return_std": _mean_result("return_std"),
            "adv_mean": _mean_result("adv_mean"),
            "adv_std": _mean_result("adv_std"),
            "combined_adv_mean": _mean_result("combined_adv_mean"),
            "combined_adv_std": _mean_result("combined_adv_std"),
            "clip_frac": _mean_result("clip_frac"),
            "selected_valid_frac": _mean_result("selected_valid_frac"),
            "entropy_floor": _mean_result("entropy_floor"),
            "skipped": 0.0,
        }

    def save_checkpoint(self, episode: int, metrics: Optional[Dict[str, Any]] = None) -> Path:
        path = self.output_dir / "checkpoints" / f"checkpoint_ep{int(episode):05d}.pt"
        return save_policy_checkpoint(
            path=path,
            actor=self.actor,
            critic=self.critic,
            cfg=self.cfg,
            episode=int(episode),
            obs_dim=int(self.obs_dim),
            metrics=metrics,
        )

    def train(self) -> Dict[str, Any]:
        collector = self._build_collector()
        ckpt_eps = {int(v) for v in self.cfg.checkpoint_episodes}
        train_rows: List[Dict[str, Any]] = []
        update_rows: List[Dict[str, Any]] = []
        checkpoint_paths: Dict[int, str] = {}
        start_episode = int(getattr(self.cfg, "resume_start_episode", 0) or 0)
        if start_episode <= 0:
            start_episode = int(self.loaded_resume_episode)
        start_episode = max(0, int(start_episode))
        start = time.time()
        try:
            replay_buffer: List[List[OnlineDecisionSample]] = []
            replay_capacity = max(int(self.cfg.replay_buffer_episodes), 1)
            replay_max_age = max(int(self.cfg.replay_actor_max_age_episodes), 1)
            for ep in range(start_episode + 1, int(self.cfg.train_episodes) + 1):
                samples, row = collector.run_train_episode(ep, int(self.rollout_max_steps))
                replay_buffer.append(list(samples))
                if len(replay_buffer) > replay_capacity:
                    replay_buffer = replay_buffer[-replay_capacity:]
                min_episode = int(ep) - int(replay_max_age) + 1
                replay_batches = [
                    batch
                    for batch in replay_buffer
                    if batch and int(batch[0].episode) >= min_episode
                ]
                if not replay_batches:
                    replay_batches = [list(samples)]
                update = self.update_from_replay(replay_batches)
                row.update({f"update_{k}": v for k, v in update.items()})
                train_rows.append(row)
                update_rows.append({"episode": ep, **update})
                print(
                    f"[online-no-cfm] ep={ep:04d}/{int(self.cfg.train_episodes):04d} "
                    f"samples={len(samples)} reward_mean={row.get('reward_mean', float('nan')):.4f} "
                    f"loss={update['loss']:.4f} t_all_arrived={row.get('t_all', float('nan')):.3f} "
                    f"completion={row.get('completion_rate', float('nan')):.6f} "
                    f"unfinished={int(row.get('unfinished_all', 0))} "
                    f"replay={update.get('replay_episodes', 0.0):.0f}x{update.get('update_epochs', 0.0):.0f} "
                    f"entropy={update.get('entropy', 0.0):.4f} clip={update.get('clip_frac', 0.0):.3f} "
                    f"adv_std={update.get('adv_std', 0.0):.4f}",
                    flush=True,
                )
                if ep in ckpt_eps:
                    ckpt = self.save_checkpoint(ep, metrics=row)
                    checkpoint_paths[int(ep)] = str(ckpt)
                    print(f"[online-no-cfm] saved checkpoint: {ckpt}", flush=True)
        finally:
            try:
                collector.finalize_protocol_xml_aliases()
            finally:
                close = getattr(getattr(collector, "connector", None), "close", None)
                if callable(close):
                    close()
        train_df = pd.DataFrame(train_rows)
        train_df.to_csv(self.output_dir / "online_train_episodes.csv", index=False)
        pd.DataFrame(update_rows).to_csv(self.output_dir / "online_train_updates.csv", index=False)
        decision_df = pd.DataFrame(getattr(collector, "_dynamic_decision_rows", []))
        if len(decision_df) > 0:
            decision_df.to_csv(self.output_dir / "online_train_decisions.csv", index=False)
        agent_df = pd.DataFrame(getattr(collector, "_dynamic_agent_decision_rows", []))
        if len(agent_df) > 0:
            agent_df.to_csv(self.output_dir / "online_train_agent_decisions.csv", index=False)
        latest_hdv_state = _merge_hdv_learning_state(
            getattr(collector, "_loaded_hdv_state_eval", None),
            _snapshot_hdv_learning_state(collector.connector.env) if hasattr(collector.connector, "env") else {},
            route_state=serialize_hdv_route_state(
                getattr(collector, "_hdv_route_probs", None),
                getattr(collector, "_hdv_route_costs", None),
            ),
        )
        _save_hdv_learning_state(self.output_dir / "hdv_learning_state_after_online_train.json", latest_hdv_state)
        summary = {
            "ablation": "gretz_online_no_cfm_actor_critic",
            "config": _jsonable_config(self.cfg),
            "obs_dim": int(self.obs_dim),
            "checkpoint_paths": checkpoint_paths,
            "train_episode_csv": str(self.output_dir / "online_train_episodes.csv"),
            "train_update_csv": str(self.output_dir / "online_train_updates.csv"),
            "elapsed_seconds": float(time.time() - start),
        }
        with (self.output_dir / "online_train_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


def _load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def build_eval_command(
    *,
    cfg: OnlineNoCfmConfig,
    checkpoint_path: str | Path,
    results_dir: str | Path,
) -> List[str]:
    cmd = [
        str(cfg.eval_python),
        "-u",
        str(_REPO_ROOT / "scripts" / "eval_sumo.py"),
        "--settings",
        str(cfg.settings),
        "--policy_ckpt",
        str(checkpoint_path),
        "--projector_ckpt",
        str(cfg.projector_ckpt),
        "--dynamics_ckpt",
        str(cfg.dynamics_ckpt),
        "--edge_registry",
        str(cfg.edge_registry),
        "--network",
        str(cfg.network_name),
        "--network_folder",
        str(cfg.network_dir),
        "--n_routes",
        str(int(cfg.n_routes)),
        "--ratio_machines",
        str(float(cfg.ratio_machines)),
        "--human_learning_episodes",
        "0",
        "--test_eps",
        str(int(cfg.eval_episodes)),
        "--results_dir",
        str(results_dir),
        "--device",
        str(cfg.device),
        "--seed",
        str(int(cfg.seed)),
        "--episode_steps",
        str(int(cfg.episode_steps)),
        "--sim_buffer_after_last_depart",
        str(int(cfg.sim_buffer_after_last_depart)),
        "--reuse_artifacts_dir",
        str(cfg.route_artifacts_dir),
        "--post_mutation_state_in",
        str(cfg.post_mutation_state_in),
        "--hdv_state_in",
        str(cfg.hdv_state_in),
        "--av_candidate_source",
        str(cfg.av_candidate_source),
        "--av_policy_route_selection",
        "argmax" if bool(cfg.deterministic_eval) else "sample",
        "--wm_policy_obs_source",
        str(cfg.wm_policy_obs_source),
        "--wm_edge_cost_source",
        str(cfg.wm_edge_cost_source),
        "--wm_candidate_edge_cost_source",
        str(cfg.wm_candidate_edge_cost_source),
        "--wm_cfm_init_mode",
        str(cfg.wm_cfm_init_mode),
        "--mpc_horizon",
        "0",
        "--av_route_cost_policy",
        "none",
        "--sumo_time_to_teleport",
        str(cfg.sumo_time_to_teleport),
        f"--incident_edge={cfg.incident_edge}",
        "--incident_start",
        str(float(cfg.incident_start)),
        "--incident_end",
        str(float(cfg.incident_end)),
        "--incident_speed",
        str(float(cfg.incident_speed)),
    ]
    if bool(cfg.disable_human_learning_after_mutation):
        cmd.append("--disable_human_learning_after_mutation")
    else:
        cmd.append("--keep_human_learning_after_mutation")
    if bool(cfg.deterministic_eval):
        cmd.append("--deterministic")
    else:
        cmd.append("--stochastic")
    if str(cfg.sumo_extra_args).strip():
        cmd.extend(["--sumo_extra_args", str(cfg.sumo_extra_args)])
    cmd.extend(list(cfg.extra_eval_args))
    return cmd


def validate_required_paths(plan: Dict[str, Any]) -> None:
    missing = list(plan.get("missing", []))
    if missing:
        details = {name: plan["paths"].get(name) for name in missing}
        raise FileNotFoundError("Missing required Gretz ablation artifacts: " + json.dumps(details, indent=2))
