"""Online Gretz Dreamer-style world-model actor-critic ablation.

This ablation differs from ``baseline.online_actor_critic`` in two ways:

* Real SUMO episodes train an online replay-updated world model
  (FlowProjector + CFM Dynamics).
* The actor/critic update is computed from CFM imagined rollouts, not from
  direct real-SUMO A2C samples.
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

from baseline.online_actor_critic import (  # noqa: E402
    DEFAULT_CHECKPOINT_EPISODES,
    GRETZ_TAG,
    OnlineDecisionSample,
    OnlineNoCfmCollector,
    OnlineNoCfmConfig,
    _append_vehicle_context_features,
    _as_float,
    _as_int,
    _jsonable_config,
    _load_json_object,
    _resolve_path,
    _select_real_policy_route_index,
    build_diverse_paths_current_to_dest,
    build_eval_command as build_no_cfm_eval_command,
    compute_a2c_loss,
    route_valid_mask_from_path_batch,
    save_policy_checkpoint,
    samples_to_tensors,
    selected_vehicle_route_rewards,
    normalize_real_policy_route_selection,
)

try:
    from dreamer4.urban_routing.dreamer_actor_critic import (
        CentralizedLatentValueCritic,
        CentralizedVehicleValueCritic,
        VehicleRouteActor,
        lambda_return,
    )
    from dreamer4.urban_routing.dreamer_rollout import aggregate_route_probs_to_action_edge
    from dreamer4.urban_routing.experiment_settings import load_settings_file
    from dreamer4.urban_routing.routerl_shared import compute_agent_counts, load_od_data
    from dreamer4.urban_routing.train_marl import (
        build_vehicle_fleet_pressure_obs_features,
        build_vehicle_obs_vectorized,
        edge_tt_from_latent_state,
        vehicle_mode_obs_dim,
    )
    from dreamer4.dynamics_losses import transition_decoded_supervision_loss
    from dreamer4.train_dynamics import cfm_loss, cfm_predict_one_timestep
except Exception:
    from dreamer_actor_critic import (
        CentralizedLatentValueCritic,
        CentralizedVehicleValueCritic,
        VehicleRouteActor,
        lambda_return,
    )
    from dreamer_rollout import aggregate_route_probs_to_action_edge
    from experiment_settings import load_settings_file
    from routerl_shared import compute_agent_counts, load_od_data
    from train_marl import (
        build_vehicle_fleet_pressure_obs_features,
        build_vehicle_obs_vectorized,
        edge_tt_from_latent_state,
        vehicle_mode_obs_dim,
    )
    from dynamics_losses import transition_decoded_supervision_loss
    from train_dynamics import cfm_loss, cfm_predict_one_timestep

from scripts.eval_sumo_hdv_state import (  # noqa: E402
    _load_hdv_learning_state,
    _merge_hdv_learning_state,
    _save_hdv_learning_state,
    _snapshot_hdv_learning_state,
)
from scripts.eval_sumo_io import (  # noqa: E402
    _new_sumo_xmls_since,
    _reuse_route_artifacts,
    _reuse_route_artifacts_from_refs,
    _snapshot_sumo_xmls,
)
from scripts.eval_sumo_routing import (  # noqa: E402
    _build_edge_adj_list,
    _edge_route_cost_matrix,
    _route_batch_uniqueness_summary,
    _temperature_adjust_probs,
)
from baseline.traci_port_fallback import patch_traci_start_port_fallback  # noqa: E402
from dreamer4.urban_routing.data_collector import serialize_hdv_route_state  # noqa: E402


DEFAULT_ONLINE_WM_CHECKPOINT_EPISODES = (25, 50, 100)


@dataclass
class OnlineWmDreamerConfig(OnlineNoCfmConfig):
    """Configuration for online world-model Dreamer-style Gretz ablation."""

    output_dir: str = str(_REPO_ROOT / "results" / GRETZ_TAG / "ablation_online_wm_dreamer_ac")
    checkpoint_episodes: Tuple[int, ...] = DEFAULT_ONLINE_WM_CHECKPOINT_EPISODES
    eval_policy_route_selection: str = "sample"
    reward_mode: str = "capped_step_cost"
    return_horizon_steps: int = 0
    wm_replay_buffer_episodes: int = 8
    wm_update_epochs: int = 1
    wm_update_every_episodes: int = 1
    wm_updates_per_collect: int = 1
    wm_replay_require_completion: bool = True
    wm_replay_completion_threshold: float = 1.0
    wm_batch_size: int = 16
    wm_lr: float = 1e-5
    wm_grad_clip: float = 10.0
    wm_transition_decoded_supervision_weight: float = 1.0
    wm_hdv_policy_weight: float = 0.05
    wm_hdv_target_source: str = "episode_mean_current_occupancy"
    wm_hdv_policy_target_mode: str = "episode_mean"
    wm_hdv_policy_action_mode: str = "window_mean"
    wm_transition_hdv_context_mode: str = "episode_response"
    wm_transition_hdv_context_weight: float = 1.0
    wm_transition_hdv_context_scale: float = 1.0
    wm_transition_hdv_action_mode: str = "none"
    wm_transition_hdv_action_weight: float = 0.0
    wm_transition_hdv_action_scale: float = 1.0
    imagined_hdv_context_source: str = "oracle_current_occupancy"
    hdv_response_predictor_enabled: bool = False
    hdv_response_predictor_lr: float = 2e-3
    hdv_response_predictor_weight_decay: float = 1e-4
    hdv_response_predictor_hidden_dim: int = 128
    hdv_response_predictor_depth: int = 2
    hdv_response_predictor_update_epochs: int = 64
    hdv_response_predictor_min_pairs: int = 2
    hdv_response_predictor_fallback: str = "prev_hdv"
    hdv_response_predictor_ckpt: str = ""
    hdv_response_predictor_progress_denominator: int = 0
    wm_cfm_active_weight: float = 0.0
    wm_cfm_busy_weight: float = 0.0
    wm_cfm_busy_threshold: float = 0.05
    wm_cfm_active_threshold: float = 0.0
    wm_cfm_max_weight: float = 50.0
    imag_horizon: int = 8
    imagined_updates_per_episode: int = 1
    imagined_update_passes_per_collect: int = 1
    imagined_rollouts_per_update: int = 1
    imagined_max_start_samples: int = 1
    min_replay_episodes_before_actor: int = 1
    ac_lambda: float = 0.95
    ac_global_advantage_weight: float = 1.0
    wm_prediction_diagnostic_every_episodes: int = 1
    wm_prediction_diagnostic_max_steps: int = 16
    wm_hdv_pressure_trace_enabled: bool = True
    direct_real_a2c_update: bool = False
    freeze_policy_obs_world_model: bool = False
    policy_obs_mode: str = "wm_latent"
    actor_update_schedule: str = "fixed"
    actor_update_min_passes: int = 0
    actor_update_max_passes: int = 0
    actor_update_warmup_episodes: int = 0
    actor_update_ramp_episodes: int = 0
    actor_update_gate_enabled: bool = False
    actor_update_gate_min_diag_samples: int = 1
    actor_update_gate_min_top1_match: float = 0.0
    actor_update_gate_max_route_mae: float = 0.0
    actor_update_gate_max_current_edge_mae: float = 0.0
    actor_update_gate_max_regret: float = 0.0
    actor_update_gate_fail_passes: int = 0
    actor_update_gate_missing_passes: int = 0
    vehicle_arrival_head_enabled: bool = False
    vehicle_arrival_loss_coef: float = 1.0
    vehicle_arrival_diag_horizon_seconds: float = 14.0

    def resolve(self) -> "OnlineWmDreamerConfig":
        values = asdict(self)
        values["real_policy_route_selection"] = normalize_real_policy_route_selection(
            values.get("real_policy_route_selection")
        )
        values["eval_policy_route_selection"] = normalize_real_policy_route_selection(
            values.get("eval_policy_route_selection")
        )
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
        return OnlineWmDreamerConfig(**values)

    @classmethod
    def from_settings(cls, settings_path: str, **overrides: Any) -> "OnlineWmDreamerConfig":
        no_cfm_fields = set(asdict(OnlineNoCfmConfig()).keys())
        base_overrides = {
            key: value
            for key, value in overrides.items()
            if key in no_cfm_fields and value is not None
        }
        base = OnlineNoCfmConfig.from_settings(settings_path, **base_overrides)
        values = asdict(base)
        own_defaults = asdict(cls())
        for key in (
            "output_dir",
            "checkpoint_episodes",
            "eval_policy_route_selection",
            "reward_mode",
            "return_horizon_steps",
            "wm_replay_buffer_episodes",
            "wm_update_epochs",
            "wm_update_every_episodes",
            "wm_updates_per_collect",
            "wm_replay_require_completion",
            "wm_replay_completion_threshold",
            "wm_batch_size",
            "wm_lr",
            "wm_grad_clip",
            "wm_transition_decoded_supervision_weight",
            "wm_hdv_policy_weight",
            "wm_hdv_policy_target_mode",
            "wm_hdv_policy_action_mode",
            "wm_transition_hdv_context_mode",
            "wm_transition_hdv_context_weight",
            "wm_transition_hdv_context_scale",
            "wm_transition_hdv_action_mode",
            "wm_transition_hdv_action_weight",
            "wm_transition_hdv_action_scale",
            "imagined_hdv_context_source",
            "hdv_response_predictor_enabled",
            "hdv_response_predictor_lr",
            "hdv_response_predictor_weight_decay",
            "hdv_response_predictor_hidden_dim",
            "hdv_response_predictor_depth",
            "hdv_response_predictor_update_epochs",
            "hdv_response_predictor_min_pairs",
            "hdv_response_predictor_fallback",
            "hdv_response_predictor_ckpt",
            "hdv_response_predictor_progress_denominator",
            "wm_cfm_active_weight",
            "wm_cfm_busy_weight",
            "wm_cfm_busy_threshold",
            "wm_cfm_active_threshold",
            "wm_cfm_max_weight",
            "imag_horizon",
            "imagined_updates_per_episode",
            "imagined_update_passes_per_collect",
            "imagined_rollouts_per_update",
            "imagined_max_start_samples",
            "min_replay_episodes_before_actor",
            "ac_lambda",
            "ac_global_advantage_weight",
            "wm_prediction_diagnostic_every_episodes",
            "wm_prediction_diagnostic_max_steps",
            "wm_hdv_pressure_trace_enabled",
            "direct_real_a2c_update",
            "freeze_policy_obs_world_model",
            "policy_obs_mode",
            "actor_update_schedule",
            "actor_update_min_passes",
            "actor_update_max_passes",
            "actor_update_warmup_episodes",
            "actor_update_ramp_episodes",
            "actor_update_gate_enabled",
            "actor_update_gate_min_diag_samples",
            "actor_update_gate_min_top1_match",
            "actor_update_gate_max_route_mae",
            "actor_update_gate_max_current_edge_mae",
            "actor_update_gate_max_regret",
            "actor_update_gate_fail_passes",
            "actor_update_gate_missing_passes",
        ):
            if key not in overrides:
                values[key] = own_defaults[key]
        values.update({k: v for k, v in overrides.items() if v is not None})
        settings = load_settings_file(settings_path)
        if "wm_cfm_init_mode" not in overrides:
            values["wm_cfm_init_mode"] = settings.get("WM_CFM_INIT_MODE", values.get("wm_cfm_init_mode", "zero"))
        values["wm_policy_obs_source"] = settings.get(
            "WM_POLICY_OBS_SOURCE",
            values.get("wm_policy_obs_source", "decoder_volume_tt"),
        )
        values["wm_edge_cost_source"] = settings.get(
            "WM_EDGE_COST_SOURCE",
            values.get("wm_edge_cost_source", "decoded_speed"),
        )
        values["wm_candidate_edge_cost_source"] = settings.get(
            "WM_CANDIDATE_EDGE_COST_SOURCE",
            values.get("wm_candidate_edge_cost_source", "freeflow"),
        )
        values["wm_hdv_policy_target_mode"] = settings.get(
            "HDV_POLICY_TARGET_MODE",
            values.get("wm_hdv_policy_target_mode", "episode_mean"),
        )
        values["wm_hdv_target_source"] = settings.get(
            "HDV_TARGET_SOURCE",
            values.get("wm_hdv_target_source", "episode_mean_current_occupancy"),
        )
        values["wm_hdv_policy_action_mode"] = settings.get(
            "HDV_POLICY_ACTION_MODE",
            values.get("wm_hdv_policy_action_mode", "window_mean"),
        )
        values["wm_transition_hdv_context_mode"] = settings.get(
            "TRANSITION_HDV_CONTEXT_MODE",
            values.get("wm_transition_hdv_context_mode", "episode_response"),
        )
        values["wm_transition_hdv_context_weight"] = _as_float(
            settings.get("TRANSITION_HDV_CONTEXT_WEIGHT"),
            float(values.get("wm_transition_hdv_context_weight", 1.0)),
        )
        values["wm_transition_hdv_context_scale"] = _as_float(
            settings.get("TRANSITION_HDV_CONTEXT_SCALE"),
            float(values.get("wm_transition_hdv_context_scale", 1.0)),
        )
        values["wm_transition_hdv_action_mode"] = settings.get(
            "TRANSITION_HDV_ACTION_MODE",
            values.get("wm_transition_hdv_action_mode", "none"),
        )
        values["wm_transition_hdv_action_weight"] = _as_float(
            settings.get("TRANSITION_HDV_ACTION_WEIGHT"),
            float(values.get("wm_transition_hdv_action_weight", 0.0)),
        )
        values["wm_transition_hdv_action_scale"] = _as_float(
            settings.get("TRANSITION_HDV_ACTION_SCALE"),
            float(values.get("wm_transition_hdv_action_scale", 1.0)),
        )
        values["imagined_hdv_context_source"] = settings.get(
            "IMAGINED_HDV_CONTEXT_SOURCE",
            values.get("imagined_hdv_context_source", "oracle_current_occupancy"),
        )
        values["hdv_response_predictor_enabled"] = bool(
            _as_int(
                settings.get("HDV_RESPONSE_PREDICTOR_ENABLED"),
                int(bool(values.get("hdv_response_predictor_enabled", False))),
            )
        )
        values["hdv_response_predictor_lr"] = _as_float(
            settings.get("HDV_RESPONSE_PREDICTOR_LR"),
            float(values.get("hdv_response_predictor_lr", 2e-3)),
        )
        values["hdv_response_predictor_weight_decay"] = _as_float(
            settings.get("HDV_RESPONSE_PREDICTOR_WEIGHT_DECAY"),
            float(values.get("hdv_response_predictor_weight_decay", 1e-4)),
        )
        values["hdv_response_predictor_hidden_dim"] = _as_int(
            settings.get("HDV_RESPONSE_PREDICTOR_HIDDEN_DIM"),
            int(values.get("hdv_response_predictor_hidden_dim", 128)),
        )
        values["hdv_response_predictor_depth"] = _as_int(
            settings.get("HDV_RESPONSE_PREDICTOR_DEPTH"),
            int(values.get("hdv_response_predictor_depth", 2)),
        )
        values["hdv_response_predictor_update_epochs"] = _as_int(
            settings.get("HDV_RESPONSE_PREDICTOR_UPDATE_EPOCHS"),
            int(values.get("hdv_response_predictor_update_epochs", 64)),
        )
        values["hdv_response_predictor_min_pairs"] = _as_int(
            settings.get("HDV_RESPONSE_PREDICTOR_MIN_PAIRS"),
            int(values.get("hdv_response_predictor_min_pairs", 2)),
        )
        values["hdv_response_predictor_fallback"] = settings.get(
            "HDV_RESPONSE_PREDICTOR_FALLBACK",
            values.get("hdv_response_predictor_fallback", "prev_hdv"),
        )
        values["hdv_response_predictor_ckpt"] = settings.get(
            "HDV_RESPONSE_PREDICTOR_CKPT",
            values.get("hdv_response_predictor_ckpt", ""),
        )
        values["hdv_response_predictor_progress_denominator"] = _as_int(
            settings.get("HDV_RESPONSE_PREDICTOR_PROGRESS_DENOMINATOR"),
            int(values.get("hdv_response_predictor_progress_denominator", 0)),
        )
        values["reward_mode"] = settings.get(
            "VEHICLE_REWARD_MODE",
            settings.get("DREAMER_ROUTE_COST_REWARD_MODE", values.get("reward_mode", "capped_step_cost")),
        )
        values["reward_temperature"] = _as_float(
            settings.get(
                "VEHICLE_REWARD_TEMPERATURE",
                settings.get("DREAMER_ROUTE_COST_REWARD_TEMPERATURE"),
            ),
            float(values.get("reward_temperature", 60.0)),
        )
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values).resolve()


def _annotate_samples_with_arrival_remaining(
    samples: Sequence[OnlineDecisionSample],
    vehicle_travel_times: Dict[str, float],
) -> None:
    for sample in samples:
        veh_id = str(getattr(sample, "veh_id", ""))
        travel_time = float(vehicle_travel_times.get(veh_id, float("nan")))
        elapsed = float(getattr(sample, "elapsed_time", float("nan")))
        if np.isfinite(travel_time) and np.isfinite(elapsed):
            # TraCI arrival events are observed on the next tick relative to
            # tripinfo's arrival timestamp; align the target to arrival-sim_step.
            sample.arrival_remaining_time = float(max(travel_time - elapsed - 1.0, 0.0))
        else:
            sample.arrival_remaining_time = float("nan")


def _annotate_agent_decision_rows_with_arrival(
    rows: List[Dict[str, Any]],
    samples: Sequence[OnlineDecisionSample],
) -> None:
    by_key = {
        (int(s.episode), int(s.decision_idx), str(s.veh_id)): float(
            getattr(s, "arrival_remaining_time", float("nan"))
        )
        for s in samples
    }
    for row in rows:
        key = (
            int(row.get("episode", -1)),
            int(row.get("decision_idx", -1)),
            str(row.get("veh_id", "")),
        )
        value = float(by_key.get(key, float("nan")))
        if np.isfinite(value):
            row["arrival_remaining_time"] = value


@dataclass
class OnlineWorldModelStep:
    episode: int
    decision_idx: int
    sim_step: int
    flow: np.ndarray
    next_flow: np.ndarray
    av_action: np.ndarray
    hdv_action: np.ndarray
    depart: np.ndarray
    context: np.ndarray
    hdv_target: Optional[np.ndarray] = None
    hdv_static_edge_demand: Optional[np.ndarray] = None
    hdv_predicted_edge_demand: Optional[np.ndarray] = None
    hdv_prediction_source: str = ""
    samples: List[OnlineDecisionSample] = field(default_factory=list)


@dataclass
class OnlineHdvEpisodeSummary:
    episode: int
    av_edge_pressure: np.ndarray
    hdv_static_edge_demand: np.ndarray
    hdv_target: np.ndarray
    predicted_hdv_edge_demand: Optional[np.ndarray] = None
    prediction_source: str = ""


class OnlineWorldModelReplay:
    def __init__(self, capacity_episodes: int, *, hdv_target_source: str = "episode_mean_current_occupancy"):
        self.capacity_episodes = max(int(capacity_episodes), 1)
        self.hdv_target_source = str(hdv_target_source or "episode_mean_current_occupancy").strip().lower()
        self._episodes: List[List[OnlineWorldModelStep]] = []
        self._summaries: List[OnlineHdvEpisodeSummary] = []

    @staticmethod
    def _normalised_edge_vector(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).reshape(-1).clip(min=0.0)
        denom = float(arr.sum())
        if denom > 1e-8:
            return (arr / denom).astype(np.float32)
        return arr.astype(np.float32)

    def _episode_current_occupancy_mean(self, steps: Sequence[OnlineWorldModelStep]) -> Optional[np.ndarray]:
        hdv_stack = _stack_np([s.hdv_action for s in steps])
        if hdv_stack.size <= 0:
            return None
        return self._normalised_edge_vector(hdv_stack.clip(min=0.0).mean(axis=0))

    def _episode_static_edge_demand(self, steps: Sequence[OnlineWorldModelStep]) -> Optional[np.ndarray]:
        static_rows = [
            np.asarray(s.hdv_static_edge_demand, dtype=np.float32)
            for s in steps
            if s.hdv_static_edge_demand is not None
        ]
        if not static_rows:
            return None
        static_stack = _stack_np(static_rows)
        if static_stack.size <= 0:
            return None
        return self._normalised_edge_vector(static_stack.clip(min=0.0).mean(axis=0))

    def _episode_av_pressure_mean(self, steps: Sequence[OnlineWorldModelStep]) -> Optional[np.ndarray]:
        av_stack = _stack_np([s.av_action for s in steps])
        if av_stack.size <= 0:
            return None
        return self._normalised_edge_vector(av_stack.clip(min=0.0).mean(axis=0))

    def append_episode(
        self,
        steps: Sequence[OnlineWorldModelStep],
        *,
        predicted_hdv_edge_demand: Optional[np.ndarray] = None,
        prediction_source: str = "",
    ) -> None:
        if not steps:
            return
        episode_steps = list(steps)
        source = str(self.hdv_target_source or "episode_mean_current_occupancy").strip().lower()
        if source in {"static_edge_demand", "static_episode_edge_demand", "episode_static_edge_demand"}:
            hdv_mean = self._episode_static_edge_demand(episode_steps)
            if hdv_mean is None:
                hdv_mean = self._episode_current_occupancy_mean(episode_steps)
        elif source in {"episode_mean_current_occupancy", "current_occupancy_mean", "episode_mean", "legacy"}:
            hdv_mean = self._episode_current_occupancy_mean(episode_steps)
        else:
            raise ValueError(
                f"Unsupported wm_hdv_target_source={self.hdv_target_source!r}; "
                "expected episode_mean_current_occupancy or static_edge_demand"
            )
        if hdv_mean is not None:
            for step in episode_steps:
                step.hdv_target = hdv_mean.copy()
        pred = None
        if predicted_hdv_edge_demand is not None:
            pred = self._normalised_edge_vector(np.asarray(predicted_hdv_edge_demand, dtype=np.float32))
            for step in episode_steps:
                step.hdv_predicted_edge_demand = pred.copy()
                step.hdv_prediction_source = str(prediction_source or "causal_response_predictor")
        av_mean = self._episode_av_pressure_mean(episode_steps)
        static_mean = self._episode_static_edge_demand(episode_steps)
        if static_mean is None:
            static_mean = self._episode_current_occupancy_mean(episode_steps)
        if av_mean is not None and static_mean is not None and hdv_mean is not None:
            self._summaries.append(
                OnlineHdvEpisodeSummary(
                    episode=int(episode_steps[0].episode),
                    av_edge_pressure=av_mean.astype(np.float32),
                    hdv_static_edge_demand=static_mean.astype(np.float32),
                    hdv_target=hdv_mean.astype(np.float32),
                    predicted_hdv_edge_demand=None if pred is None else pred.astype(np.float32),
                    prediction_source=str(prediction_source or ""),
                )
            )
        self._episodes.append(episode_steps)
        if len(self._episodes) > self.capacity_episodes:
            self._episodes = self._episodes[-self.capacity_episodes :]

    @property
    def episodes(self) -> List[List[OnlineWorldModelStep]]:
        return list(self._episodes)

    @property
    def summaries(self) -> List[OnlineHdvEpisodeSummary]:
        return list(self._summaries)

    def episode_count(self) -> int:
        return int(len(self._episodes))

    def all_steps(self) -> List[OnlineWorldModelStep]:
        return [step for ep in self._episodes for step in ep]

    def sample_steps(self, rng: np.random.Generator, batch_size: int) -> List[OnlineWorldModelStep]:
        steps = self.all_steps()
        if not steps:
            return []
        n = min(max(int(batch_size), 1), len(steps))
        idx = rng.choice(len(steps), size=n, replace=len(steps) < n)
        return [steps[int(i)] for i in np.asarray(idx).reshape(-1)]


class CausalHdvResponsePredictor(torch.nn.Module):
    def __init__(self, n_edges: int, hidden_dim: int, depth: int):
        super().__init__()
        self.n_edges = int(n_edges)
        input_dim = 2 * self.n_edges + 1
        layers: List[torch.nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(int(depth), 1)):
            layers.append(torch.nn.Linear(dim, int(hidden_dim)))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.LayerNorm(int(hidden_dim)))
            dim = int(hidden_dim)
        layers.append(torch.nn.Linear(dim, self.n_edges))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalHdvResponseModel:
    def __init__(self, *, n_edges: int, cfg: OnlineWmDreamerConfig, device: torch.device):
        self.n_edges = int(n_edges)
        self.cfg = cfg
        self.device = device
        self.hidden_dim = int(cfg.hdv_response_predictor_hidden_dim)
        self.depth = int(cfg.hdv_response_predictor_depth)
        self.progress_denominator = int(
            getattr(cfg, "hdv_response_predictor_progress_denominator", 0)
            or getattr(cfg, "train_episodes", 1)
            or 1
        )
        self.model = CausalHdvResponsePredictor(
            n_edges=int(n_edges),
            hidden_dim=int(self.hidden_dim),
            depth=int(self.depth),
        ).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.hdv_response_predictor_lr),
            weight_decay=float(cfg.hdv_response_predictor_weight_decay),
        )
        self.trained_pairs = 0
        self.pretrained_pairs = 0

    @staticmethod
    def _normalise_np(values: np.ndarray, n_edges: int) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if int(arr.size) != int(n_edges):
            out = np.zeros((int(n_edges),), dtype=np.float32)
            take = min(int(arr.size), int(n_edges))
            if take > 0:
                out[:take] = arr[:take]
            arr = out
        arr = arr.clip(min=0.0)
        denom = float(arr.sum())
        if denom > 1e-8:
            arr = arr / denom
        elif int(n_edges) > 0:
            arr = np.full((int(n_edges),), 1.0 / float(int(n_edges)), dtype=np.float32)
        return arr.astype(np.float32)

    def _feature_np(
        self,
        *,
        prev_av_edge_pressure: np.ndarray,
        prev_hdv_static_edge_demand: np.ndarray,
        episode: int,
    ) -> np.ndarray:
        prev_av = self._normalise_np(prev_av_edge_pressure, self.n_edges)
        prev_hdv = self._normalise_np(prev_hdv_static_edge_demand, self.n_edges)
        progress = float(max(int(episode), 1)) / float(max(int(self.progress_denominator), 1))
        progress = float(np.clip(progress, 0.0, 1.0))
        return np.concatenate([prev_av, prev_hdv, np.asarray([progress], dtype=np.float32)], axis=0).astype(np.float32)

    def _pairs_from_summaries(
        self,
        summaries: Sequence[OnlineHdvEpisodeSummary],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        ordered = sorted(list(summaries), key=lambda s: int(s.episode))
        if len(ordered) < 2:
            return None, None
        by_ep = {int(s.episode): s for s in ordered}
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for current in ordered:
            prev = by_ep.get(int(current.episode) - 1)
            if prev is None:
                continue
            xs.append(
                self._feature_np(
                    prev_av_edge_pressure=prev.av_edge_pressure,
                    prev_hdv_static_edge_demand=prev.hdv_static_edge_demand,
                    episode=int(current.episode),
                )
            )
            ys.append(self._normalise_np(current.hdv_static_edge_demand, self.n_edges))
        if not xs:
            return None, None
        return (
            torch.from_numpy(np.stack(xs, axis=0)).to(device=self.device, dtype=torch.float32),
            torch.from_numpy(np.stack(ys, axis=0)).to(device=self.device, dtype=torch.float32),
        )

    def predict_for_next_episode(
        self,
        summaries: Sequence[OnlineHdvEpisodeSummary],
        *,
        episode: int,
    ) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        if not bool(getattr(self.cfg, "hdv_response_predictor_enabled", False)):
            return None, {"hdv_response_pred_enabled": 0.0, "hdv_response_pred_source_code": 0.0}
        ordered = sorted(list(summaries), key=lambda s: int(s.episode))
        prev = ordered[-1] if ordered else None
        if prev is None:
            return None, {"hdv_response_pred_enabled": 1.0, "hdv_response_pred_source_code": 1.0}
        fallback = str(getattr(self.cfg, "hdv_response_predictor_fallback", "prev_hdv") or "prev_hdv").strip().lower()
        can_predict = self.trained_pairs >= max(int(getattr(self.cfg, "hdv_response_predictor_min_pairs", 2)), 1)
        if can_predict:
            self.model.eval()
            x = torch.from_numpy(
                self._feature_np(
                    prev_av_edge_pressure=prev.av_edge_pressure,
                    prev_hdv_static_edge_demand=prev.hdv_static_edge_demand,
                    episode=int(episode),
                )
            ).to(device=self.device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = torch.softmax(self.model(x), dim=-1)[0].detach().cpu().numpy().astype(np.float32)
            return pred, {
                "hdv_response_pred_enabled": 1.0,
                "hdv_response_pred_source_code": 2.0,
                "hdv_response_pred_pairs": float(self.trained_pairs),
            }
        if fallback in {"prev_hdv", "previous_hdv", "prev_static"}:
            return self._normalise_np(prev.hdv_static_edge_demand, self.n_edges), {
                "hdv_response_pred_enabled": 1.0,
                "hdv_response_pred_source_code": 1.5,
                "hdv_response_pred_pairs": float(self.trained_pairs),
            }
        if fallback in {"zero", "none"}:
            return np.zeros((self.n_edges,), dtype=np.float32), {
                "hdv_response_pred_enabled": 1.0,
                "hdv_response_pred_source_code": 1.25,
                "hdv_response_pred_pairs": float(self.trained_pairs),
            }
        return None, {
            "hdv_response_pred_enabled": 1.0,
            "hdv_response_pred_source_code": 1.0,
            "hdv_response_pred_pairs": float(self.trained_pairs),
        }

    def update(self, summaries: Sequence[OnlineHdvEpisodeSummary]) -> Dict[str, float]:
        if not bool(getattr(self.cfg, "hdv_response_predictor_enabled", False)):
            return {
                "hdv_response_loss": 0.0,
                "hdv_response_updates": 0.0,
                "hdv_response_pairs": float(self.trained_pairs),
            }
        x, target = self._pairs_from_summaries(summaries)
        if x is None or target is None:
            return {
                "hdv_response_loss": 0.0,
                "hdv_response_updates": 0.0,
                "hdv_response_pairs": float(self.trained_pairs),
                "hdv_response_online_pairs": 0.0,
            }
        self.model.train()
        losses: List[float] = []
        for _ in range(max(int(getattr(self.cfg, "hdv_response_predictor_update_epochs", 64)), 1)):
            logits = self.model(x)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            loss = -(target * log_probs).sum(dim=-1).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        self.model.eval()
        online_pairs = int(x.shape[0])
        self.trained_pairs = int(self.pretrained_pairs) + int(online_pairs)
        with torch.no_grad():
            pred = torch.softmax(self.model(x), dim=-1)
            tv = 0.5 * torch.abs(pred - target).sum(dim=-1).mean()
        return {
            "hdv_response_loss": float(np.mean(losses)) if losses else 0.0,
            "hdv_response_updates": float(len(losses)),
            "hdv_response_pairs": float(self.trained_pairs),
            "hdv_response_online_pairs": float(online_pairs),
            "hdv_response_train_tv": float(tv.detach().cpu().item()),
        }

    def save_checkpoint(self, path: str | Path, *, metrics: Optional[Dict[str, Any]] = None) -> Path:
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "causal_hdv_response_predictor_v1",
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "trained_pairs": int(self.trained_pairs),
                "pretrained_pairs": int(self.pretrained_pairs),
                "n_edges": int(self.n_edges),
                "hidden_dim": int(self.hidden_dim),
                "depth": int(self.depth),
                "progress_denominator": int(self.progress_denominator),
                "metrics": dict(metrics or {}),
            },
            ckpt_path,
        )
        return ckpt_path

    def load_checkpoint(self, path: str | Path, *, load_optimizer: bool = False) -> Dict[str, Any]:
        ckpt_path = Path(path)
        state = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        if not isinstance(state, dict):
            raise RuntimeError(f"HDV response predictor checkpoint is not a dict: {ckpt_path}")
        expected_edges = int(state.get("n_edges", self.n_edges))
        if expected_edges != int(self.n_edges):
            raise RuntimeError(
                f"HDV response predictor n_edges mismatch: checkpoint={expected_edges}, model={self.n_edges}"
            )
        model_state = state.get("model_state_dict") or state.get("model")
        if model_state is None:
            raise RuntimeError(f"HDV response predictor checkpoint has no model_state_dict: {ckpt_path}")
        self.model.load_state_dict(model_state, strict=True)
        if bool(load_optimizer) and state.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        saved_trained_pairs = int(state.get("trained_pairs", self.trained_pairs) or 0)
        saved_pretrained_pairs = state.get("pretrained_pairs")
        if saved_pretrained_pairs is None or int(saved_pretrained_pairs or 0) <= 0:
            self.pretrained_pairs = int(saved_trained_pairs)
        else:
            self.pretrained_pairs = int(saved_pretrained_pairs)
        self.trained_pairs = int(self.pretrained_pairs)
        self.progress_denominator = int(state.get("progress_denominator", self.progress_denominator) or 1)
        return state


@dataclass
class ImaginedVehicleRouteState:
    candidate_paths: List[List[int]]
    current_edge: int
    dest_edge: int
    current_edge_remaining: float
    elapsed_time: float
    arrived: bool = False


@dataclass
class ImaginedRouteAdvance:
    elapsed_cost: float
    current_edge: int
    current_edge_remaining: float
    arrived: bool
    remaining_path: List[int]


def _imagined_route_remaining_time(
    *,
    path: Sequence[int],
    edge_tt: np.ndarray,
    current_edge: int,
    current_edge_remaining: float,
) -> float:
    n_edges = int(np.asarray(edge_tt).reshape(-1).shape[0])
    route = _sanitise_path(path, n_edges=n_edges)
    if not route:
        return 0.0
    suffix = _suffix_path_from_edge(route, int(current_edge))
    if suffix:
        route = suffix
    if not route:
        return 0.0
    total = (
        float(current_edge_remaining)
        if int(route[0]) == int(current_edge) and float(current_edge_remaining) > 0.0
        else _edge_tt_seconds(edge_tt, int(route[0]))
    )
    for edge in route[1:]:
        total += _edge_tt_seconds(edge_tt, int(edge))
    return float(max(total, 0.0))


def _stack_np(items: Sequence[np.ndarray], *, dtype=np.float32) -> np.ndarray:
    if not items:
        return np.zeros((0,), dtype=dtype)
    return np.stack([np.asarray(x, dtype=dtype) for x in items], axis=0).astype(dtype)


def _paths_from_samples(samples: Sequence[OnlineDecisionSample]) -> List[List[List[int]]]:
    out: List[List[List[int]]] = []
    for sample in samples:
        paths = sample.path_batch or []
        out.append([[int(e) for e in path] for path in paths])
    return out


def _valid_mask_from_samples(samples: Sequence[OnlineDecisionSample], n_routes: int) -> np.ndarray:
    if not samples:
        return np.zeros((0, int(n_routes)), dtype=bool)
    paths = _paths_from_samples(samples)
    return route_valid_mask_from_path_batch(paths, n_routes=int(n_routes))


def _enabled_mode(value: str) -> bool:
    return str(value or "none").strip().lower() not in {"", "none", "off", "disabled"}


def _normalise_edge_distribution(values: torch.Tensor) -> torch.Tensor:
    clipped = values.to(dtype=torch.float32).clamp_min(0.0)
    denom = clipped.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return clipped / denom


def _causal_pair(current: torch.Tensor) -> torch.Tensor:
    """Align a one-step action/context with CFM targets [S_t, S_{t+1}]."""
    return torch.stack([torch.zeros_like(current), current], dim=1)


def _edge_tt_seconds(edge_tt: np.ndarray, edge_idx: int, *, default: float = 1.0) -> float:
    edge_arr = np.asarray(edge_tt, dtype=np.float32).reshape(-1)
    idx = int(edge_idx)
    if 0 <= idx < int(edge_arr.shape[0]):
        value = float(edge_arr[idx])
        if np.isfinite(value) and value > 0.0:
            return value
    return float(default)


def _sanitise_path(path: Sequence[int], *, n_edges: int) -> List[int]:
    return [int(edge) for edge in path if 0 <= int(edge) < int(n_edges)]


def _suffix_path_from_edge(path: Sequence[int], current_edge: int) -> List[int]:
    current = int(current_edge)
    if current < 0:
        return []
    clean = [int(edge) for edge in path]
    for pos, edge in enumerate(clean):
        if int(edge) == current:
            return clean[pos:]
    return []


def _candidate_suffixes_from_current(
    candidate_paths: Sequence[Sequence[int]],
    *,
    current_edge: int,
    n_routes: int,
    n_edges: int,
    fallback_path: Optional[Sequence[int]] = None,
) -> List[List[int]]:
    suffixes: List[List[int]] = []
    for path in list(candidate_paths)[: int(n_routes)]:
        suffixes.append(
            _sanitise_path(
                _suffix_path_from_edge(path, int(current_edge)),
                n_edges=int(n_edges),
            )
        )
    while len(suffixes) < int(n_routes):
        suffixes.append([])
    if any(len(path) > 0 for path in suffixes):
        return suffixes[: int(n_routes)]
    if fallback_path is not None:
        fallback = _sanitise_path(
            _suffix_path_from_edge(fallback_path, int(current_edge)),
            n_edges=int(n_edges),
        )
        if fallback:
            suffixes[0] = fallback
    return suffixes[: int(n_routes)]


def _current_to_dest_candidate_paths(
    *,
    adj_list: Sequence[Sequence[int]],
    edge_tt: np.ndarray,
    current_edge: int,
    dest_edge: int,
    n_routes: int,
    n_edges: int,
    penalty_factor: float,
    fallback_paths: Optional[Sequence[Sequence[int]]] = None,
) -> List[List[int]]:
    current = int(current_edge)
    dest = int(dest_edge)
    if current < 0 or dest < 0 or current >= int(n_edges) or dest >= int(n_edges):
        return _candidate_suffixes_from_current(
            fallback_paths or [],
            current_edge=current,
            n_routes=int(n_routes),
            n_edges=int(n_edges),
            fallback_path=(fallback_paths or [[]])[0] if fallback_paths else None,
        )
    if current == dest:
        return [[current] for _ in range(int(n_routes))]
    costs = np.asarray(edge_tt, dtype=np.float32).reshape(-1)
    if costs.shape[0] < int(n_edges) or build_diverse_paths_current_to_dest is None or not adj_list:
        return _candidate_suffixes_from_current(
            fallback_paths or [],
            current_edge=current,
            n_routes=int(n_routes),
            n_edges=int(n_edges),
            fallback_path=(fallback_paths or [[]])[0] if fallback_paths else None,
        )
    try:
        generated = build_diverse_paths_current_to_dest(
            adj_list=adj_list,
            edge_tt=costs[: int(n_edges)],
            src_idx=current,
            dst_idx=dest,
            K=int(n_routes),
            penalty_factor=float(max(penalty_factor, 1.0)),
        )
    except Exception:
        generated = []
    paths: List[List[int]] = []
    for path in list(generated)[: int(n_routes)]:
        clean = _sanitise_path(path, n_edges=int(n_edges))
        if clean and int(clean[0]) == current:
            paths.append(clean)
    if not paths:
        return _candidate_suffixes_from_current(
            fallback_paths or [],
            current_edge=current,
            n_routes=int(n_routes),
            n_edges=int(n_edges),
            fallback_path=(fallback_paths or [[]])[0] if fallback_paths else None,
        )
    while len(paths) < int(n_routes):
        paths.append(list(paths[0]))
    return paths[: int(n_routes)]


def _advance_imagined_route_progress(
    *,
    path: Sequence[int],
    edge_tt: np.ndarray,
    current_edge: int,
    current_edge_remaining: float,
    step_interval: float,
) -> ImaginedRouteAdvance:
    """Advance one imagined vehicle along one selected route.

    This is intentionally route-progress based: a vehicle pays only for the
    simulated seconds until arrival, then stops contributing reward on later
    imagined steps.  That matches the real online SUMO reward semantics where
    shorter trips accumulate fewer negative step costs.
    """
    n_edges = int(np.asarray(edge_tt).reshape(-1).shape[0])
    route = _sanitise_path(path, n_edges=n_edges)
    if not route:
        return ImaginedRouteAdvance(0.0, -1, 0.0, True, [])
    suffix = _suffix_path_from_edge(route, int(current_edge))
    if suffix:
        route = suffix
    budget = float(max(step_interval, 0.0))
    elapsed = 0.0
    pos = 0
    remaining_on_edge = (
        float(current_edge_remaining)
        if route and int(route[0]) == int(current_edge) and float(current_edge_remaining) > 0.0
        else _edge_tt_seconds(edge_tt, int(route[0]))
    )
    while pos < len(route):
        edge = int(route[pos])
        spend = min(float(budget), float(max(remaining_on_edge, 0.0)))
        elapsed += spend
        budget -= spend
        remaining_on_edge -= spend
        if remaining_on_edge > 1e-6:
            return ImaginedRouteAdvance(
                elapsed_cost=float(elapsed),
                current_edge=edge,
                current_edge_remaining=float(remaining_on_edge),
                arrived=False,
                remaining_path=list(route[pos:]),
            )
        pos += 1
        if pos >= len(route):
            return ImaginedRouteAdvance(float(elapsed), -1, 0.0, True, [])
        if budget <= 1e-6:
            next_edge = int(route[pos])
            return ImaginedRouteAdvance(
                elapsed_cost=float(elapsed),
                current_edge=next_edge,
                current_edge_remaining=float(_edge_tt_seconds(edge_tt, next_edge)),
                arrived=False,
                remaining_path=list(route[pos:]),
            )
        remaining_on_edge = _edge_tt_seconds(edge_tt, int(route[pos]))
    return ImaginedRouteAdvance(float(elapsed), -1, 0.0, True, [])


def _discounted_returns_by_vehicle_ids(
    rewards: torch.Tensor,
    vehicle_ids: torch.Tensor,
    *,
    gamma: float,
    horizon_steps: int = 0,
) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    if int(rewards.numel()) == 0:
        return returns
    ids = vehicle_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    by_vehicle: Dict[int, List[int]] = {}
    for idx, vehicle_id in enumerate(ids[: int(rewards.numel())]):
        by_vehicle.setdefault(int(vehicle_id), []).append(int(idx))
    if int(horizon_steps) <= 0:
        for indices in by_vehicle.values():
            acc = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
            for idx in reversed(indices):
                acc = rewards[idx] + float(gamma) * acc
                returns[idx] = acc
        return returns
    for indices in by_vehicle.values():
        for pos, idx in enumerate(indices):
            acc = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
            discount = 1.0
            for future_idx in indices[pos : pos + int(horizon_steps)]:
                acc = acc + float(discount) * rewards[future_idx]
                discount *= float(gamma)
            returns[idx] = acc
    return returns


def _lambda_returns_by_vehicle_ids(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    next_valid: torch.Tensor,
    vehicle_ids: torch.Tensor,
    *,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    if int(rewards.numel()) == 0:
        return returns
    ids = vehicle_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    by_vehicle: Dict[int, List[int]] = {}
    for idx, vehicle_id in enumerate(ids[: int(rewards.numel())]):
        by_vehicle.setdefault(int(vehicle_id), []).append(int(idx))
    masked_next = torch.where(next_valid.to(device=next_values.device), next_values, torch.zeros_like(next_values))
    for indices in by_vehicle.values():
        if not indices:
            continue
        idx_t = torch.tensor(indices, device=rewards.device, dtype=torch.long)
        returns.index_copy_(
            0,
            idx_t,
            lambda_return(
                rewards.index_select(0, idx_t),
                masked_next.index_select(0, idx_t),
                gamma=float(gamma),
                lambda_=float(lambda_),
            ),
        )
    return returns


def _mean_update_stats(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    out: Dict[str, float] = {}
    for key in keys:
        if key in {"update_phase", "update_pass"}:
            continue
        vals: List[float] = []
        for row in rows:
            try:
                value = float(row.get(key, float("nan")))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                vals.append(value)
        if vals:
            out[key] = float(np.mean(vals))
    return out


def _finite_metric(row: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not row:
        return None
    try:
        value = float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _pearson_np(x: Sequence[float], y: Sequence[float]) -> float:
    xs = np.asarray([float(v) for v in x], dtype=np.float64)
    ys = np.asarray([float(v) for v in y], dtype=np.float64)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if int(finite.sum()) < 2:
        return float("nan")
    xs = xs[finite]
    ys = ys[finite]
    x_std = float(xs.std())
    y_std = float(ys.std())
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


class OnlineWmDreamerCollector(OnlineNoCfmCollector):
    """Online collector that also records replay transitions for WM training."""

    def _collect_single_episode(  # type: ignore[override]
        self,
        episode_id: int,
        max_steps: int,
    ) -> Tuple[List[OnlineDecisionSample], Dict[str, Any]]:
        self._wm_episode_steps: List[OnlineWorldModelStep] = []
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

            frame_t = self._collect_flow(state)
            before = len(self._episode_samples)
            av_actions = self._select_av_actions(state, rng)
            new_samples = list(self._episode_samples[before:])
            if new_samples:
                ep_rewards.extend([s.reward for s in new_samples])
            next_tick = (
                int(decision_ticks[decision_idx + 1])
                if (decision_idx + 1) < len(decision_ticks)
                else int(max_steps)
            )
            advance_steps = max(next_tick - int(tick), 1)
            av_edge_frac = self._compute_planned_av_edge_flow(
                state,
                av_actions,
                horizon_seconds=float(advance_steps),
            )
            hdv_edge_frac = self._compute_hdv_edge_distribution(state)
            decision_depart = self._decision_depart_edge(int(decision_idx))
            decision_context = self._decision_context(state, int(tick))
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
            frame_next = self._collect_flow(state)
            self._wm_episode_steps.append(
                OnlineWorldModelStep(
                    episode=int(episode_id),
                    decision_idx=int(decision_idx),
                    sim_step=int(tick),
                    flow=np.asarray(frame_t, dtype=np.float32),
                    next_flow=np.asarray(frame_next, dtype=np.float32),
                    av_action=np.asarray(av_edge_frac, dtype=np.float32),
                    hdv_action=np.asarray(hdv_edge_frac, dtype=np.float32),
                    depart=np.asarray(decision_depart, dtype=np.float32),
                    context=np.asarray(decision_context, dtype=np.float32),
                    samples=new_samples,
                )
            )
            ep_decisions += 1
            if self.connector.is_done() or (len(state.get("vehicles", {})) == 0 and current_tick > 10):
                break

        if (
            self._use_routerl_hdv_blackbox
            and hasattr(self.connector, "apply_routerl_hdv_learning")
            and self._should_apply_routerl_hdv_learning("train", veh_travel_times)
        ):
            self.connector.apply_routerl_hdv_learning(veh_travel_times)
        _annotate_samples_with_arrival_remaining(self._episode_samples, self._last_ep_vehicle_tt)
        hdv_static_edge_demand = self._compute_hdv_static_edge_demand()
        if hdv_static_edge_demand is not None:
            for step in self._wm_episode_steps:
                step.hdv_static_edge_demand = hdv_static_edge_demand.copy()

        episode_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        self._write_urb_episode_csv(int(episode_id), episode_reward)
        total_agents = int(getattr(self, "_num_agents", 0) or 0)
        row = {
            "episode": int(episode_id),
            "phase": "train",
            "steps": int(ep_decisions),
            "samples": int(len(self._episode_samples)),
            "wm_transitions": int(len(self._wm_episode_steps)),
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
            "episode_complete": int(total_agents > 0 and len(self._last_ep_travel_times) >= total_agents),
            "hdv_static_edge_demand_sum": float(np.sum(hdv_static_edge_demand))
            if hdv_static_edge_demand is not None
            else 0.0,
            "hdv_static_edge_demand_edges": int(np.count_nonzero(hdv_static_edge_demand > 0.0))
            if hdv_static_edge_demand is not None
            else 0,
        }
        for key, value in sorted(getattr(self, "_av_decision_skip_stats", {}).items()):
            row[f"av_diag_{key}"] = int(value)
        return list(self._episode_samples), row

    def _compute_hdv_static_edge_demand(self) -> Optional[np.ndarray]:
        edge_to_idx = getattr(self._edge_registry, "_edge_to_idx", {})
        n_edges = int(getattr(self._edge_registry, "n_edges", 0) or 0)
        if n_edges <= 0 or not edge_to_idx:
            return None
        out = np.zeros((n_edges,), dtype=np.float32)
        used = 0
        for veh_id in sorted(getattr(self, "_hdv_vehicles", set())):
            ridx = getattr(self, "_hdv_route_idx", {}).get(veh_id)
            candidates = getattr(self, "_veh_candidates", {}).get(veh_id)
            if ridx is None or not candidates:
                continue
            ridx_i = int(ridx)
            if ridx_i < 0 or ridx_i >= len(candidates):
                continue
            route = candidates[ridx_i]
            added = False
            for edge_id in route:
                ei = edge_to_idx.get(str(edge_id))
                if ei is None:
                    continue
                out[int(ei)] += 1.0
                added = True
            if added:
                used += 1
        if used <= 0:
            return None
        denom = float(out.sum())
        if denom > 1e-8:
            out /= denom
        return out.astype(np.float32)

    def run_train_episode(
        self,
        episode_id: int,
        max_steps_per_episode: int,
    ) -> Tuple[List[OnlineDecisionSample], Dict[str, Any], List[OnlineWorldModelStep]]:
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
        _annotate_agent_decision_rows_with_arrival(
            getattr(self, "_dynamic_agent_decision_rows", []),
            samples,
        )
        return samples, row, list(getattr(self, "_wm_episode_steps", []))


class OnlineWmToyLiveCollector(OnlineWmDreamerCollector):
    """Online-WM collector whose real-SUMO policy observations match toy no-WM AC."""

    def _policy_logits_for_active_ids(
        self,
        obs: torch.Tensor,
        active_ids: Sequence[str],
    ) -> torch.Tensor:
        active_logits = getattr(self._policy_eval, "forward_logits_for_active_ids", None)
        if callable(active_logits):
            return active_logits(obs, active_ids)
        return self._policy_eval.forward_logits(obs)

    def _live_edge_features(self, state: Dict[str, Any]) -> np.ndarray:
        edge_state = state.get("edges", {}) if isinstance(state, dict) else {}
        feats = self._edge_registry.extract_flow(edge_state if isinstance(edge_state, dict) else {})
        return np.asarray(feats, dtype=np.float32)

    def _collect_single_episode(  # type: ignore[override]
        self,
        episode_id: int,
        max_steps: int,
    ) -> Tuple[List[OnlineDecisionSample], Dict[str, Any]]:
        self._first_decision_routes: Dict[str, List[str]] = {}
        return super()._collect_single_episode(episode_id, max_steps)

    def _select_av_actions(self, state: Dict[str, Any], rng: np.random.RandomState) -> Dict[str, List[str]]:
        stats = getattr(self, "_av_decision_skip_stats", None)
        if stats is None:
            stats = {}
            self._av_decision_skip_stats = stats

        def inc(name: str, amount: int = 1) -> None:
            stats[name] = int(stats.get(name, 0)) + int(amount)

        inc("calls")
        in_network = state.get("vehicles", {})
        active_avs = sorted(self._av_vehicles & set(in_network.keys()))
        inc("active_av_seen", len(active_avs))
        if not active_avs:
            inc("no_active_av_calls")
            return {}

        edge_feats_dyn = self._live_edge_features(state)
        edge_to_idx = getattr(self._edge_registry, "_edge_to_idx", {})
        live_edge_cost = self._live_current_edge_tt(state)
        freeflow_edge_cost = self._freeflow_edge_tt()
        obs_edge_cost = live_edge_cost if live_edge_cost is not None else freeflow_edge_cost
        candidate_edge_cost = freeflow_edge_cost
        if self._av_candidate_source == "current_shortest_live":
            candidate_edge_cost = live_edge_cost
        elif self._av_candidate_source == "current_shortest_freeflow":
            candidate_edge_cost = freeflow_edge_cost
        elif self._av_candidate_source == "current_shortest_wm":
            candidate_edge_cost = freeflow_edge_cost
            inc("candidate_wm_source_fallback_freeflow")
        if self._av_candidate_source != "od_suffix" and candidate_edge_cost is None:
            inc("candidate_edge_cost_missing_calls")

        path_batch: List[List[List[int]]] = []
        route_batch: List[List[List[str]]] = []
        active_ids: List[str] = []
        active_current_edges: List[str] = []
        candidate_dynamic_count = 0
        candidate_fallback_count = 0
        for veh_id in active_avs:
            veh_info = in_network.get(veh_id)
            if not isinstance(veh_info, dict):
                inc("skip_bad_vehicle_info")
                continue
            current_edge = str(veh_info.get("edge", ""))
            if (not current_edge) or current_edge.startswith(":"):
                inc("skip_bad_current_edge")
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
                inc("skip_no_candidates")
                continue
            idx_paths: List[List[int]] = []
            named_paths: List[List[str]] = []
            for route in candidates[: int(self.n_routes)]:
                idxs = [int(edge_to_idx[e]) for e in route if e in edge_to_idx]
                if idxs:
                    idx_paths.append(idxs)
                    named_paths.append(list(route))
            if not idx_paths:
                inc("skip_no_idx_paths")
                continue
            while len(idx_paths) < int(self.n_routes):
                idx_paths.append(list(idx_paths[0]))
                named_paths.append(list(named_paths[0]))
            path_batch.append(idx_paths[: int(self.n_routes)])
            route_batch.append(named_paths[: int(self.n_routes)])
            active_ids.append(veh_id)
            active_current_edges.append(current_edge)

        if not path_batch:
            inc("no_path_batch_calls")
            return {}
        inc("selected_vehicle_count", len(active_ids))

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
                    logits_i = self._policy_logits_for_active_ids(obs_i, [active_ids[i]])
                    probs_i = torch.softmax(logits_i, dim=-1).detach().cpu().numpy().astype(np.float32)[0]
                probs_i = _temperature_adjust_probs(
                    probs_i,
                    float(self._cfg.av_policy_sample_temperature),
                    int(self.n_routes),
                )
                selected_i = _select_real_policy_route_index(
                    probs_i,
                    mode=self._cfg.real_policy_route_selection,
                    rng=rng,
                )
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
                logits = self._policy_logits_for_active_ids(obs_t, active_ids)
                probs_np = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
            for i in range(len(active_ids)):
                probs = _temperature_adjust_probs(
                    probs_np[i],
                    float(self._cfg.av_policy_sample_temperature),
                    int(self.n_routes),
                )
                selected_indices[i] = _select_real_policy_route_index(
                    probs,
                    mode=self._cfg.real_policy_route_selection,
                    rng=rng,
                )
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
        argmax_indices = np.argmax(probs_np[:, : int(self.n_routes)], axis=1).astype(np.int64)
        argmax_matches = selected_indices == argmax_indices
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
            "obs_source": "live_edge_registry_extract_flow",
            "real_policy_route_selection": str(self._cfg.real_policy_route_selection),
            "selected_argmax_match_rate": float(np.mean(argmax_matches)),
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
                "argmax_route": int(argmax_indices[i]),
                "selected_matches_argmax": int(bool(argmax_matches[i])),
                "real_policy_route_selection": str(self._cfg.real_policy_route_selection),
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

        if str(getattr(self, "_av_execution_mode", "dynamic")).lower() in {
            "first_only",
            "first_decision_only",
            "static_first",
        }:
            locked = getattr(self, "_first_decision_routes", None)
            if locked is None:
                locked = {}
                self._first_decision_routes = locked
            filtered_actions: Dict[str, List[str]] = {}
            suppressed = 0
            for veh_id, route in actions.items():
                if veh_id not in locked:
                    locked[veh_id] = list(route)
                    filtered_actions[veh_id] = route
                else:
                    suppressed += 1
            if suppressed:
                inc("first_decision_only_suppressed", suppressed)
            return filtered_actions
        return actions


class OnlineWorldModelUpdater:
    def __init__(self, wm, cfg: OnlineWmDreamerConfig, device: torch.device):
        self.wm = wm
        self.cfg = cfg
        self.device = device
        for module in (wm.flow_projector, wm.dynamics):
            for param in module.parameters():
                param.requires_grad_(True)
        params = list(wm.flow_projector.parameters()) + list(wm.dynamics.parameters())
        self.optimizer = torch.optim.Adam(params, lr=float(cfg.wm_lr))

    def _batch_tensors(self, steps: Sequence[OnlineWorldModelStep]) -> Dict[str, torch.Tensor]:
        flows = _stack_np([s.flow for s in steps])
        next_flows = _stack_np([s.next_flow for s in steps])
        actions = _stack_np([s.av_action for s in steps])
        hdv = _stack_np([s.hdv_action for s in steps])
        hdv_targets = _stack_np([s.hdv_target if s.hdv_target is not None else s.hdv_action for s in steps])
        depart = _stack_np([s.depart for s in steps])
        context = _stack_np([s.context for s in steps])
        flow_pair = np.stack([flows, next_flows], axis=1).astype(np.float32)
        return {
            "flows": torch.from_numpy(flow_pair).to(device=self.device, dtype=torch.float32),
            "actions": torch.from_numpy(actions).to(device=self.device, dtype=torch.float32),
            "hdv": torch.from_numpy(hdv).to(device=self.device, dtype=torch.float32),
            "hdv_target": torch.from_numpy(hdv_targets).to(device=self.device, dtype=torch.float32),
            "depart": torch.from_numpy(depart).to(device=self.device, dtype=torch.float32),
            "context": torch.from_numpy(context).to(device=self.device, dtype=torch.float32),
        }

    def _transition_hdv_context(self, action: torch.Tensor, hdv_target: torch.Tensor) -> Optional[torch.Tensor]:
        if not _enabled_mode(str(self.cfg.wm_transition_hdv_context_mode)):
            return None
        if float(self.cfg.wm_transition_hdv_context_weight) == 0.0:
            return None
        hdv = hdv_target.to(device=action.device, dtype=action.dtype).clamp_min(0.0)
        mode = str(self.cfg.wm_transition_hdv_context_mode).strip().lower()
        if mode in {"step", "per_step", "episode_mean", "episode_response", "response"}:
            hdv = _normalise_edge_distribution(hdv).to(device=action.device, dtype=action.dtype)
        elif mode in {"raw", "raw_step"}:
            pass
        else:
            raise ValueError(f"Unsupported wm_transition_hdv_context_mode={self.cfg.wm_transition_hdv_context_mode!r}")
        return float(self.cfg.wm_transition_hdv_context_weight) * float(self.cfg.wm_transition_hdv_context_scale) * hdv

    def _transition_action(self, action: torch.Tensor, hdv_target: torch.Tensor) -> torch.Tensor:
        if not _enabled_mode(str(self.cfg.wm_transition_hdv_action_mode)):
            return action
        if float(self.cfg.wm_transition_hdv_action_weight) == 0.0:
            return action
        hdv = hdv_target.to(device=action.device, dtype=action.dtype).clamp_min(0.0)
        mode = str(self.cfg.wm_transition_hdv_action_mode).strip().lower()
        if mode in {"step", "per_step", "episode_mean", "episode_response", "response"}:
            hdv = _normalise_edge_distribution(hdv).to(device=action.device, dtype=action.dtype)
        elif mode in {"raw", "raw_step"}:
            pass
        else:
            raise ValueError(f"Unsupported wm_transition_hdv_action_mode={self.cfg.wm_transition_hdv_action_mode!r}")
        return action + float(self.cfg.wm_transition_hdv_action_weight) * float(self.cfg.wm_transition_hdv_action_scale) * hdv

    def _hdv_policy_actions(
        self,
        action: torch.Tensor,
        depart: torch.Tensor,
        context: torch.Tensor,
        z_dtype: torch.dtype,
    ) -> Optional[torch.Tensor | Tuple[torch.Tensor, ...]]:
        mode = str(self.cfg.wm_hdv_policy_action_mode or "step").strip().lower()
        if mode in {"", "step", "per_step", "legacy"}:
            return None
        if mode not in {"window_mean", "episode_mean", "episode_response", "response"}:
            raise ValueError(f"Unsupported wm_hdv_policy_action_mode={self.cfg.wm_hdv_policy_action_mode!r}")
        act_cond = action
        if mode in {"episode_mean", "episode_response", "response"}:
            dep_cond = None
            ctx_cond = None
        else:
            dep_cond = depart
            ctx_cond = context
        action_pair = torch.stack([act_cond, act_cond], dim=1)
        depart_pair = None if dep_cond is None else torch.stack([dep_cond, dep_cond], dim=1)
        context_pair = None if ctx_cond is None else torch.stack([ctx_cond, ctx_cond], dim=1)
        return self.wm._prepare_model_actions(
            batch_size=int(action.shape[0]),
            seq_len=2,
            action=action_pair,
            depart=depart_pair,
            context=context_pair,
            hdv_context=None,
            device=self.device,
            dtype=z_dtype,
        )

    def update(self, replay: OnlineWorldModelReplay, rng: np.random.Generator) -> Dict[str, float]:
        if not replay.all_steps():
            return {"wm_loss": 0.0, "wm_updates": 0.0, "wm_replay_transitions": 0.0, "wm_skipped": 1.0}
        self.wm.flow_projector.train()
        self.wm.dynamics.train()
        stats: List[Dict[str, float]] = []
        for _ in range(max(int(self.cfg.wm_update_epochs), 1)):
            steps = replay.sample_steps(rng, int(self.cfg.wm_batch_size))
            if not steps:
                continue
            batch = self._batch_tensors(steps)
            flows = batch["flows"]
            actions_raw = batch["actions"]
            hdv = batch["hdv"]
            hdv_target_current = batch["hdv_target"]
            depart = batch["depart"]
            context = batch["context"]
            z_pair = self.wm.flow_projector(flows)
            transition_action = self._transition_action(actions_raw, hdv_target_current)
            transition_hdv_context = self._transition_hdv_context(transition_action, hdv_target_current)
            model_actions = self.wm._prepare_model_actions(
                batch_size=int(flows.shape[0]),
                seq_len=2,
                action=_causal_pair(transition_action),
                depart=_causal_pair(depart),
                context=_causal_pair(context),
                hdv_context=_causal_pair(transition_hdv_context)
                if transition_hdv_context is not None and bool(getattr(self.wm, "use_hdv_context", False))
                else None,
                device=self.device,
                dtype=z_pair.dtype,
            )
            hdv_target_pair = torch.stack([hdv, hdv_target_current], dim=1)
            hdv_policy_mode = str(self.cfg.wm_hdv_policy_target_mode or "episode_mean").strip().lower()
            if hdv_policy_mode == "window_mean":
                hdv_target_pair = hdv_target_pair.mean(dim=1, keepdim=True).expand_as(hdv_target_pair)
            elif hdv_policy_mode in {"episode_mean", "episode_response", "response"}:
                hdv_target_pair = torch.stack([hdv_target_current, hdv_target_current], dim=1)
            elif hdv_policy_mode in {"", "step", "per_step", "legacy"}:
                hdv_target_pair = torch.stack([hdv, hdv], dim=1)
            else:
                raise ValueError(f"Unsupported wm_hdv_policy_target_mode={self.cfg.wm_hdv_policy_target_mode!r}")
            hdv_actions = self._hdv_policy_actions(actions_raw, depart, context, z_pair.dtype)
            base_loss, aux = cfm_loss(
                self.wm.dynamics,
                z1=z_pair,
                actions=model_actions,
                act_mask=None,
                hdv_actions=hdv_actions,
                agent_tokens=None,
                hdv_target=hdv_target_pair,
                hdv_policy_weight=float(self.cfg.wm_hdv_policy_weight)
                if int(getattr(self.wm, "hdv_policy_dim", 0)) > 0
                else 0.0,
                flow_targets=flows,
                cfm_active_weight=float(self.cfg.wm_cfm_active_weight),
                cfm_busy_weight=float(self.cfg.wm_cfm_busy_weight),
                cfm_busy_threshold=float(self.cfg.wm_cfm_busy_threshold),
                cfm_active_threshold=float(self.cfg.wm_cfm_active_threshold),
                cfm_max_weight=float(self.cfg.wm_cfm_max_weight),
                volume_index=2,
                init_mode=str(self.cfg.wm_cfm_init_mode),
            )
            initial_state = None
            if str(self.cfg.wm_cfm_init_mode).strip().lower() == "zero":
                initial_state = torch.zeros(
                    (int(z_pair.shape[0]), 1, int(z_pair.shape[2]), int(z_pair.shape[3])),
                    device=self.device,
                    dtype=z_pair.dtype,
                )
            z_pred_next = cfm_predict_one_timestep(
                self.wm.dynamics,
                past_packed=z_pair[:, :-1],
                actions=model_actions,
                act_mask=None,
                n_steps=int(getattr(self.wm, "n_ode_steps", 1)),
                initial_state=initial_state,
                init_mode=str(self.cfg.wm_cfm_init_mode),
            )
            transition_loss, transition_aux = transition_decoded_supervision_loss(
                z_pred_next=z_pred_next,
                flow_projector=self.wm.flow_projector,
                flow_target_next=flows[:, -1],
                supervision_weight=float(self.cfg.wm_transition_decoded_supervision_weight),
                speed_index=1,
                volume_index=2,
                speed_weight=4.0,
                volume_weight=2.0,
                busy_threshold=float(self.cfg.wm_cfm_busy_threshold),
            )
            loss = base_loss + transition_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.wm.flow_projector.parameters()) + list(self.wm.dynamics.parameters()),
                float(self.cfg.wm_grad_clip),
            )
            self.optimizer.step()
            row = {
                "wm_loss": float(loss.detach().cpu().item()),
                "wm_cfm_loss": float(base_loss.detach().cpu().item()),
                "wm_transition_loss": float(transition_loss.detach().cpu().item()),
                "wm_hdv_ce": float(aux.get("hdv_policy_ce", torch.zeros(())).detach().cpu().item())
                if isinstance(aux, dict)
                else 0.0,
                "wm_weighted_flow_mse": float(aux.get("weighted_flow_mse", torch.zeros(())).detach().cpu().item())
                if isinstance(aux, dict)
                else 0.0,
                "wm_transition_mse": float(
                    transition_aux.get("pred_next_decoded_total_mse", torch.zeros(())).detach().cpu().item()
                )
                if isinstance(transition_aux, dict)
                else 0.0,
            }
            stats.append(row)
        self.wm.flow_projector.eval()
        self.wm.dynamics.eval()
        if not stats:
            return {"wm_loss": 0.0, "wm_updates": 0.0, "wm_replay_transitions": float(len(replay.all_steps())), "wm_skipped": 1.0}
        out = {k: float(np.mean([row.get(k, 0.0) for row in stats])) for k in stats[-1].keys()}
        out["wm_updates"] = float(len(stats))
        out["wm_replay_transitions"] = float(len(replay.all_steps()))
        out["wm_skipped"] = 0.0
        return out


def capped_step_cost_reward_matrix(
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
    *,
    step_interval: float,
    reward_temperature: float,
) -> torch.Tensor:
    effective = torch.minimum(
        route_costs.clamp_min(0.0),
        torch.full_like(route_costs, float(max(step_interval, 1e-6))),
    )
    rewards = -effective / float(max(reward_temperature, 1e-6))
    return torch.where(route_valid, rewards, torch.zeros_like(rewards))


def _sample_masked_actions(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one masked route and expose the same one-hot WM action."""
    safe_valid = valid_mask.to(device=logits.device, dtype=torch.bool).clone()
    has_valid = safe_valid.any(dim=-1)
    safe_valid[~has_valid, 0] = True
    masked_logits = (logits / float(max(temperature, 1e-6))).masked_fill(~safe_valid, -1e9)
    dist = Categorical(logits=masked_logits)
    selected = dist.sample()
    one_hot = torch.nn.functional.one_hot(selected, num_classes=int(logits.shape[-1])).to(dtype=logits.dtype)
    return selected, one_hot, dist.log_prob(selected), dist.entropy()


def _combine_vehicle_global_advantages(
    vehicle_advantage: torch.Tensor,
    global_advantage: torch.Tensor,
    group_index: torch.Tensor,
    *,
    global_weight: float,
) -> torch.Tensor:
    """Add each imagined step's global advantage to its active CAV samples."""
    vehicle = vehicle_advantage.reshape(-1)
    groups = group_index.to(device=vehicle.device, dtype=torch.long).reshape(-1)
    global_step = global_advantage.to(device=vehicle.device, dtype=vehicle.dtype).reshape(-1)
    if int(groups.numel()) != int(vehicle.numel()):
        raise ValueError("group_index must align with vehicle advantages")
    if int(groups.numel()) == 0:
        return vehicle
    if int(global_step.numel()) <= int(groups.max().detach().cpu().item()):
        raise ValueError("global advantages do not cover all vehicle groups")
    return vehicle + float(global_weight) * global_step.index_select(0, groups)


def _global_network_reward_from_latent(
    wm,
    z_next: torch.Tensor,
    edge_tt: np.ndarray,
    *,
    reward_temperature: float,
) -> torch.Tensor:
    """Dense all-vehicle reward used by the existing global latent critic."""
    with torch.no_grad():
        decoded = wm.flow_projector.decode(z_next.detach())
        capacities = wm.bpr.capacities.to(device=z_next.device, dtype=z_next.dtype).reshape(-1)
        edge_time = torch.as_tensor(edge_tt, device=z_next.device, dtype=z_next.dtype).reshape(-1)
        n_edges = min(int(decoded.shape[0]), int(capacities.numel()), int(edge_time.numel()))
        if n_edges <= 0:
            return torch.zeros((), device=z_next.device, dtype=z_next.dtype)
        vehicle_count = torch.relu(decoded[:n_edges, 2]) * capacities[:n_edges]
        count_sum = vehicle_count.sum()
        if float(count_sum.detach().cpu().item()) > 1e-8:
            network_tt = torch.sum(vehicle_count * edge_time[:n_edges].clamp_min(0.0)) / count_sum
        else:
            network_tt = edge_time[:n_edges].clamp_min(0.0).mean()
        return -network_tt / float(max(reward_temperature, 1e-6))


def _safe_corr_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if int(a.numel()) <= 1 or int(b.numel()) <= 1:
        return torch.zeros((), device=a.device, dtype=a.dtype)
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = torch.sqrt(torch.square(a0).mean().clamp_min(1e-12)) * torch.sqrt(
        torch.square(b0).mean().clamp_min(1e-12)
    )
    if float(denom.detach().cpu().item()) <= 1e-12:
        return torch.zeros((), device=a.device, dtype=a.dtype)
    return (a0 * b0).mean() / denom


def _arrival_head_loss_and_diag(
    *,
    critic: CentralizedVehicleValueCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    valid_mask: torch.Tensor,
    group_index: Optional[torch.Tensor],
    arrival_remaining: torch.Tensor,
    route_costs: torch.Tensor,
    horizon_seconds: float,
) -> Dict[str, torch.Tensor]:
    zero = obs.sum() * 0.0
    predict = getattr(critic, "predict_arrival_log_time", None)
    if not callable(predict):
        return {"loss": zero, "n": zero.detach(), "skipped": torch.ones((), device=obs.device, dtype=obs.dtype)}
    pred_log_all = predict(obs, group_index=group_index)
    if pred_log_all.numel() == 0:
        return {"loss": zero, "n": zero.detach(), "skipped": torch.ones((), device=obs.device, dtype=obs.dtype)}
    n = min(
        int(obs.shape[0]),
        int(actions.reshape(-1).numel()),
        int(arrival_remaining.reshape(-1).numel()),
        int(valid_mask.shape[0]),
        int(pred_log_all.shape[0]),
    )
    if n <= 0:
        return {"loss": zero, "n": zero.detach(), "skipped": torch.ones((), device=obs.device, dtype=obs.dtype)}
    pred_log_all = pred_log_all[:n]
    action_t = actions.to(device=obs.device, dtype=torch.long).reshape(-1)[:n]
    action_t = action_t.clamp(0, int(pred_log_all.shape[1]) - 1)
    target = arrival_remaining.to(device=obs.device, dtype=obs.dtype).reshape(-1)[:n]
    valid = valid_mask.to(device=obs.device, dtype=torch.bool)[:n]
    rows = torch.arange(n, device=obs.device)
    selected_valid = valid[rows, action_t]
    active = selected_valid & torch.isfinite(target) & (target >= 0.0)
    if not bool(active.any().detach().cpu().item()):
        return {"loss": zero, "n": zero.detach(), "skipped": torch.ones((), device=obs.device, dtype=obs.dtype)}

    pred_selected_log = pred_log_all[rows, action_t]
    pred_selected = torch.expm1(pred_selected_log.clamp(min=0.0, max=12.0)).clamp_min(0.0)
    target_log = torch.log1p(target.clamp_min(0.0))
    loss = torch.nn.functional.smooth_l1_loss(pred_selected_log[active], target_log[active])

    pred_err = pred_selected[active].detach() - target[active].detach()
    pred_abs = pred_err.abs()
    route_cost_t = route_costs.to(device=obs.device, dtype=obs.dtype)
    if route_cost_t.dim() == 2 and int(route_cost_t.shape[0]) >= n and int(route_cost_t.shape[1]) >= int(pred_log_all.shape[1]):
        route_selected = route_cost_t[:n, : int(pred_log_all.shape[1])][rows, action_t]
    else:
        route_selected = torch.full((n,), float("nan"), device=obs.device, dtype=obs.dtype)
    progress_active = active & torch.isfinite(route_selected) & (route_selected >= 0.0)
    if bool(progress_active.any().detach().cpu().item()):
        progress_err = route_selected[progress_active].detach() - target[progress_active].detach()
        progress_abs = progress_err.abs()
        progress_mae = progress_abs.mean()
        progress_bias = progress_err.mean()
        progress_corr = _safe_corr_torch(route_selected[progress_active].detach(), target[progress_active].detach())
    else:
        progress_mae = zero.detach()
        progress_bias = zero.detach()
        progress_corr = zero.detach()

    horizon = float(max(horizon_seconds, 0.0))
    true_arrive = target[active] <= horizon
    pred_arrive = pred_selected[active] <= horizon
    tp = (true_arrive & pred_arrive).to(dtype=obs.dtype).sum()
    fp = ((~true_arrive) & pred_arrive).to(dtype=obs.dtype).sum()
    fn = (true_arrive & (~pred_arrive)).to(dtype=obs.dtype).sum()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    if bool(progress_active.any().detach().cpu().item()):
        true_arrive_p = target[progress_active] <= horizon
        progress_arrive = route_selected[progress_active] <= horizon
        tp_p = (true_arrive_p & progress_arrive).to(dtype=obs.dtype).sum()
        fp_p = ((~true_arrive_p) & progress_arrive).to(dtype=obs.dtype).sum()
        fn_p = (true_arrive_p & (~progress_arrive)).to(dtype=obs.dtype).sum()
        progress_precision = tp_p / (tp_p + fp_p).clamp_min(1.0)
        progress_recall = tp_p / (tp_p + fn_p).clamp_min(1.0)
    else:
        progress_precision = zero.detach()
        progress_recall = zero.detach()
    return {
        "loss": loss,
        "n": active.to(dtype=obs.dtype).sum().detach(),
        "skipped": torch.zeros((), device=obs.device, dtype=obs.dtype),
        "mae": pred_abs.mean().detach(),
        "bias": pred_err.mean().detach(),
        "corr": _safe_corr_torch(pred_selected[active].detach(), target[active].detach()).detach(),
        "target_mean": target[active].detach().mean(),
        "pred_mean": pred_selected[active].detach().mean(),
        "precision": precision.detach(),
        "recall": recall.detach(),
        "progress_n": progress_active.to(dtype=obs.dtype).sum().detach(),
        "progress_mae": progress_mae.detach(),
        "progress_bias": progress_bias.detach(),
        "progress_corr": progress_corr.detach(),
        "progress_precision": progress_precision.detach(),
        "progress_recall": progress_recall.detach(),
    }


def _edge_tt_numpy_from_z(wm, z: torch.Tensor, cfg: OnlineWmDreamerConfig) -> np.ndarray:
    try:
        _edge_dyn, _v_e, _free_flow, edge_tt = edge_tt_from_latent_state(
            wm=wm,
            z=z.detach(),
            bpr_alpha=0.15,
            bpr_beta=4.0,
            edge_cost_source=str(cfg.wm_edge_cost_source),
            decoded_speed_min_ratio=float(cfg.wm_decoded_speed_min_ratio),
            decoded_speed_max_ratio=float(cfg.wm_decoded_speed_max_ratio),
            decoded_speed_excess_scale=float(cfg.wm_decoded_speed_excess_scale),
        )
        return np.asarray(edge_tt, dtype=np.float32).reshape(-1)
    except Exception:
        lengths = getattr(wm._registry, "lengths", None)
        speeds = getattr(wm._registry, "speed_limits", None)
        if lengths is None or speeds is None:
            return np.ones((int(wm.n_edges),), dtype=np.float32)
        return np.asarray(lengths, dtype=np.float32) / np.maximum(np.asarray(speeds, dtype=np.float32), 0.1)


def _decoded_edge_features_numpy_from_z(
    wm,
    z: torch.Tensor,
    cfg: OnlineWmDreamerConfig,
    *,
    expected_dim: int,
) -> np.ndarray:
    n_edges = int(getattr(wm, "n_edges", 0) or 0)
    try:
        edge_dyn, _v_e, _free_flow, _edge_tt = edge_tt_from_latent_state(
            wm=wm,
            z=z.detach(),
            bpr_alpha=0.15,
            bpr_beta=4.0,
            edge_cost_source=str(cfg.wm_edge_cost_source),
            decoded_speed_min_ratio=float(cfg.wm_decoded_speed_min_ratio),
            decoded_speed_max_ratio=float(cfg.wm_decoded_speed_max_ratio),
            decoded_speed_excess_scale=float(cfg.wm_decoded_speed_excess_scale),
        )
        edge_feats = np.asarray(edge_dyn, dtype=np.float32)
    except Exception:
        edge_feats = np.zeros((n_edges, max(int(expected_dim), 1)), dtype=np.float32)
    if edge_feats.ndim != 2:
        edge_feats = np.asarray(edge_feats, dtype=np.float32).reshape(n_edges, -1)
    if n_edges > 0 and int(edge_feats.shape[0]) != n_edges:
        fixed = np.zeros((n_edges, int(edge_feats.shape[1])), dtype=np.float32)
        rows = min(n_edges, int(edge_feats.shape[0]))
        if rows > 0:
            fixed[:rows] = edge_feats[:rows]
        edge_feats = fixed
    expected = max(int(expected_dim), 1)
    if int(edge_feats.shape[1]) < expected:
        padded = np.zeros((int(edge_feats.shape[0]), expected), dtype=np.float32)
        padded[:, : int(edge_feats.shape[1])] = edge_feats
        edge_feats = padded
    elif int(edge_feats.shape[1]) > expected:
        edge_feats = edge_feats[:, :expected]
    return np.nan_to_num(edge_feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _route_cost_matrix_torch(
    path_batch: List[List[List[int]]],
    edge_cost: np.ndarray,
    *,
    n_routes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(path_batch)
    costs = torch.zeros((n, int(n_routes)), device=device, dtype=torch.float32)
    valid = torch.zeros((n, int(n_routes)), device=device, dtype=torch.bool)
    edge = torch.as_tensor(edge_cost, device=device, dtype=torch.float32).reshape(-1)
    for i, paths in enumerate(path_batch):
        for k, path in enumerate(paths[: int(n_routes)]):
            idxs = [int(e) for e in path if 0 <= int(e) < int(edge.numel())]
            if not idxs:
                continue
            idx_t = torch.tensor(idxs, device=device, dtype=torch.long)
            costs[i, k] = edge.index_select(0, idx_t).clamp_min(0.0).sum()
            valid[i, k] = True
    return costs, valid


def _best_index(values: Sequence[float]) -> Optional[int]:
    finite = [(idx, float(v)) for idx, v in enumerate(values) if np.isfinite(float(v))]
    if not finite:
        return None
    return min(finite, key=lambda item: (item[1], item[0]))[0]


class OnlineImaginedActorCritic:
    def __init__(
        self,
        *,
        wm,
        actor: VehicleRouteActor,
        critic: CentralizedVehicleValueCritic,
        global_critic: CentralizedLatentValueCritic,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        global_critic_optimizer: torch.optim.Optimizer,
        cfg: OnlineWmDreamerConfig,
        device: torch.device,
        obs_dim: int,
    ):
        self.wm = wm
        self.actor = actor
        self.critic = critic
        self.global_critic = global_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.global_critic_optimizer = global_critic_optimizer
        self.cfg = cfg
        self.device = device
        self.obs_dim = int(obs_dim)
        registry = getattr(wm, "_registry", None)
        self.adj_list = _build_edge_adj_list(
            getattr(registry, "adjacency", None),
            getattr(registry, "passenger_allowed", None),
        )

    def _hdv_context_for_step(self, step: OnlineWorldModelStep, dtype: torch.dtype) -> Tuple[Optional[torch.Tensor], str]:
        if not bool(getattr(self.wm, "use_hdv_context", False)):
            return None, "disabled"
        source = str(getattr(self.cfg, "imagined_hdv_context_source", "oracle_current_occupancy") or "oracle_current_occupancy")
        mode = source.strip().lower()
        arr: Optional[np.ndarray]
        label = mode
        if mode in {"", "none", "off", "disabled"}:
            return None, "none"
        if mode in {"causal_response_predictor", "response_predictor", "predicted", "causal"}:
            arr = step.hdv_predicted_edge_demand
            label = str(step.hdv_prediction_source or "causal_response_predictor")
            if arr is None:
                fallback = str(getattr(self.cfg, "hdv_response_predictor_fallback", "prev_hdv") or "prev_hdv").strip().lower()
                if fallback in {"oracle_target", "target"} and step.hdv_target is not None:
                    arr = step.hdv_target
                    label = "fallback_oracle_target"
                elif fallback in {"oracle_current", "current", "current_occupancy"}:
                    arr = step.hdv_action
                    label = "fallback_oracle_current"
                else:
                    return None, "missing_prediction"
        elif mode in {"oracle_target", "target", "static_target"}:
            arr = step.hdv_target if step.hdv_target is not None else step.hdv_action
        elif mode in {"oracle_static", "static_edge_demand"}:
            arr = step.hdv_static_edge_demand if step.hdv_static_edge_demand is not None else step.hdv_target
            if arr is None:
                arr = step.hdv_action
        elif mode in {"oracle_current", "oracle_current_occupancy", "current_occupancy", "legacy"}:
            arr = step.hdv_action
        else:
            raise ValueError(f"Unsupported imagined_hdv_context_source={source!r}")
        if arr is None:
            return None, "missing"
        hdv = torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device=self.device, dtype=dtype)
        if hdv.dim() != 1:
            hdv = hdv.reshape(-1)
        hdv = _normalise_edge_distribution(hdv.unsqueeze(0))[0].to(device=self.device, dtype=dtype)
        return hdv, label

    def diagnose_prediction(self, replay: OnlineWorldModelReplay, rng: np.random.Generator) -> Dict[str, float]:
        candidate_steps = [s for s in replay.all_steps() if s.samples]
        if not candidate_steps:
            return {"wm_pred_diag_samples": 0.0, "wm_pred_diag_skipped": 1.0}
        max_steps = max(int(self.cfg.wm_prediction_diagnostic_max_steps), 1)
        n_steps = min(max_steps, len(candidate_steps))
        idx = rng.choice(len(candidate_steps), size=n_steps, replace=len(candidate_steps) < n_steps)
        route_pred_values: List[float] = []
        route_true_values: List[float] = []
        route_abs_errors: List[float] = []
        current_edge_pred: List[float] = []
        current_edge_true: List[float] = []
        current_edge_abs_errors: List[float] = []
        top1_matches: List[float] = []
        wm_best_live_regrets: List[float] = []
        active_decisions = 0
        n_edges = int(self.wm.n_edges)
        n_routes = int(self.cfg.n_routes)
        self.wm.flow_projector.eval()
        self.wm.dynamics.eval()
        with torch.no_grad():
            for raw_i in np.asarray(idx).reshape(-1):
                step = candidate_steps[int(raw_i)]
                samples = [s for s in step.samples if s.path_batch]
                if not samples:
                    continue
                path_batch = _paths_from_samples(samples)
                z = self.wm.encode_observation(step.flow).to(device=self.device, dtype=torch.float32).clone()
                action_edge = torch.from_numpy(np.asarray(step.av_action, dtype=np.float32)).to(
                    device=self.device,
                    dtype=z.dtype,
                )
                depart_t = torch.from_numpy(np.asarray(step.depart, dtype=np.float32)).to(device=self.device, dtype=z.dtype)
                context_t = torch.from_numpy(np.asarray(step.context, dtype=np.float32)).to(device=self.device, dtype=z.dtype)
                hdv_t, _hdv_source = self._hdv_context_for_step(step, z.dtype)
                try:
                    z_pred, _info = self.wm.predict_next_state_differentiable(
                        z.detach(),
                        action_edge.detach(),
                        past_packed=z.detach().unsqueeze(0).unsqueeze(0),
                        depart=depart_t,
                        context=context_t,
                        hdv_context=hdv_t,
                    )
                    z_true = self.wm.encode_observation(step.next_flow).to(device=self.device, dtype=torch.float32).clone()
                except Exception:
                    continue
                edge_tt_pred = _edge_tt_numpy_from_z(self.wm, z_pred, self.cfg)
                edge_tt_true = _edge_tt_numpy_from_z(self.wm, z_true, self.cfg)
                for sample, paths in zip(samples, path_batch):
                    active_decisions += 1
                    current = int(sample.current_edge_idx)
                    if 0 <= current < n_edges:
                        pred_cur = float(edge_tt_pred[current])
                        true_cur = float(edge_tt_true[current])
                        if np.isfinite(pred_cur) and np.isfinite(true_cur):
                            current_edge_pred.append(pred_cur)
                            current_edge_true.append(true_cur)
                            current_edge_abs_errors.append(abs(pred_cur - true_cur))
                    costs_pred: List[float] = []
                    costs_true: List[float] = []
                    for path in list(paths)[:n_routes]:
                        clean = _sanitise_path(path, n_edges=n_edges)
                        if not clean:
                            costs_pred.append(float("nan"))
                            costs_true.append(float("nan"))
                            continue
                        p = float(np.asarray(edge_tt_pred, dtype=np.float32)[clean].clip(min=0.0).sum())
                        t = float(np.asarray(edge_tt_true, dtype=np.float32)[clean].clip(min=0.0).sum())
                        costs_pred.append(p)
                        costs_true.append(t)
                        if np.isfinite(p) and np.isfinite(t):
                            route_pred_values.append(p)
                            route_true_values.append(t)
                            route_abs_errors.append(abs(p - t))
                    pred_best = _best_index(costs_pred)
                    true_best = _best_index(costs_true)
                    true_finite = [float(v) for v in costs_true if np.isfinite(float(v))]
                    if pred_best is not None and true_best is not None:
                        top1_matches.append(float(int(pred_best == true_best)))
                    if pred_best is not None and true_finite and np.isfinite(float(costs_true[pred_best])):
                        wm_best_live_regrets.append(float(costs_true[pred_best]) - float(min(true_finite)))
        if not route_abs_errors and not current_edge_abs_errors:
            return {"wm_pred_diag_samples": 0.0, "wm_pred_diag_skipped": 1.0}
        return {
            "wm_pred_diag_samples": float(active_decisions),
            "wm_pred_diag_active_decisions": float(active_decisions),
            "wm_pred_diag_route_cost_samples": float(len(route_abs_errors)),
            "wm_pred_diag_steps": float(n_steps),
            "wm_pred_diag_route_mae": float(np.mean(route_abs_errors)) if route_abs_errors else float("nan"),
            "wm_pred_diag_route_corr": _pearson_np(route_pred_values, route_true_values),
            "wm_pred_diag_current_edge_mae": (
                float(np.mean(current_edge_abs_errors)) if current_edge_abs_errors else float("nan")
            ),
            "wm_pred_diag_current_edge_corr": _pearson_np(current_edge_pred, current_edge_true),
            "wm_pred_diag_top1_match": float(np.mean(top1_matches)) if top1_matches else float("nan"),
            "wm_pred_diag_wm_best_live_regret": (
                float(np.mean(wm_best_live_regrets)) if wm_best_live_regrets else float("nan")
            ),
            "wm_pred_diag_skipped": 0.0,
        }

    def _obs_from_paths(
        self,
        *,
        z: torch.Tensor,
        path_batch: List[List[List[int]]],
        step_idx: int,
        n_total_av: int,
        edge_tt_override: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        with torch.no_grad():
            z_obs = z.detach().clone()
            if str(getattr(self.cfg, "policy_obs_mode", "wm_latent") or "wm_latent").strip().lower() in {
                "toy_live",
                "live_toy",
                "live_edge_registry",
            }:
                edge_feats = _decoded_edge_features_numpy_from_z(
                    self.wm,
                    z_obs,
                    self.cfg,
                    expected_dim=8,
                )
            else:
                edge_feats = self.wm.latent_edge_features(z_obs).detach().cpu().numpy().astype(np.float32)
            edge_tt = (
                np.asarray(edge_tt_override, dtype=np.float32).reshape(-1)
                if edge_tt_override is not None
                else _edge_tt_numpy_from_z(self.wm, z_obs, self.cfg)
            )
        obs_base = build_vehicle_obs_vectorized(
            edge_feats_dyn=edge_feats,
            path_batch=path_batch,
            include_global=bool(self.cfg.include_global),
            route_stats=str(self.cfg.route_stats),
            edge_cost=edge_tt,
            include_route_costs=bool(self.cfg.include_route_costs),
            include_fleet_pressure=False,
        )
        if bool(self.cfg.include_fleet_pressure):
            fleet_counts = np.zeros((int(self.cfg.n_routes),), dtype=np.float32)
            fleet_pressure = np.zeros((int(self.wm.n_edges),), dtype=np.float32)
            obs_rows: List[np.ndarray] = []
            for i in range(len(path_batch)):
                fleet_i = build_vehicle_fleet_pressure_obs_features(
                    path_batch=[path_batch[i]],
                    n_edges=int(self.wm.n_edges),
                    n_routes=int(self.cfg.n_routes),
                    selected_route_counts=fleet_counts,
                    selected_edge_pressure=fleet_pressure,
                )
                row = _append_vehicle_context_features(
                    np.concatenate([obs_base[i : i + 1], fleet_i], axis=1),
                    step_idx=int(step_idx),
                    episode_steps=int(self.cfg.episode_steps),
                    n_active_av=len(path_batch),
                    n_total_av=max(int(n_total_av), 1),
                )
                obs_rows.append(row[0].astype(np.float32))
            obs_np = np.stack(obs_rows, axis=0).astype(np.float32) if obs_rows else np.zeros((0, self.obs_dim), dtype=np.float32)
        else:
            obs_np = _append_vehicle_context_features(
                obs_base,
                step_idx=int(step_idx),
                episode_steps=int(self.cfg.episode_steps),
                n_active_av=len(path_batch),
                n_total_av=max(int(n_total_av), 1),
            )
        valid_np = route_valid_mask_from_path_batch(path_batch, n_routes=int(self.cfg.n_routes))
        return (
            torch.from_numpy(obs_np).to(device=self.device, dtype=torch.float32),
            torch.from_numpy(valid_np).to(device=self.device, dtype=torch.bool),
            edge_tt,
        )

    def imagine_from_step(self, start: OnlineWorldModelStep) -> Dict[str, Any]:
        samples = [s for s in start.samples if s.path_batch]
        if not samples:
            return {"obs": torch.zeros((0, self.obs_dim), device=self.device), "reward_mean": 0.0, "n_samples": 0}
        path_batch = _paths_from_samples(samples)
        n_edges = int(self.wm.n_edges)
        n_routes = int(self.cfg.n_routes)
        with torch.no_grad():
            z = self.wm.encode_observation(start.flow).to(device=self.device, dtype=torch.float32).clone()
            initial_edge_tt = _edge_tt_numpy_from_z(self.wm, z, self.cfg)
        vehicle_states: List[ImaginedVehicleRouteState] = []
        for sample, paths in zip(samples, path_batch):
            current_edge = int(sample.current_edge_idx)
            if current_edge < 0 and paths and paths[0]:
                current_edge = int(paths[0][0])
            dest_edge = int(getattr(sample, "dest_edge_idx", -1))
            if dest_edge < 0:
                for path in paths:
                    if path:
                        dest_edge = int(path[-1])
                        break
            current_remaining = _edge_tt_seconds(initial_edge_tt, current_edge) if current_edge >= 0 else 0.0
            vehicle_states.append(
                ImaginedVehicleRouteState(
                    candidate_paths=[
                        _sanitise_path(path, n_edges=n_edges)
                        for path in list(paths)[:n_routes]
                    ],
                    current_edge=int(current_edge),
                    dest_edge=int(dest_edge),
                    current_edge_remaining=float(current_remaining),
                    elapsed_time=float(max(float(getattr(sample, "elapsed_time", 0.0)), 0.0)),
                    arrived=bool(current_edge < 0),
                )
            )
        latent_history: List[torch.Tensor] = [z]
        obs_steps: List[torch.Tensor] = []
        action_steps: List[torch.Tensor] = []
        old_logp_steps: List[torch.Tensor] = []
        valid_steps: List[torch.Tensor] = []
        reward_steps: List[torch.Tensor] = []
        group_steps: List[torch.Tensor] = []
        next_group_steps: List[torch.Tensor] = []
        vehicle_id_steps: List[torch.Tensor] = []
        next_obs_steps: List[torch.Tensor] = []
        next_valid_steps: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        action_edges: List[torch.Tensor] = []
        global_latent_steps: List[torch.Tensor] = []
        global_next_latent_steps: List[torch.Tensor] = []
        global_context_steps: List[torch.Tensor] = []
        global_reward_steps: List[torch.Tensor] = []
        arrived_events = 0
        horizon = max(int(self.cfg.imag_horizon), 1)
        n_agents = len(samples)
        for h in range(horizon):
            active_indices = [
                i
                for i, vehicle in enumerate(vehicle_states)
                if (not bool(vehicle.arrived)) and int(vehicle.current_edge) >= 0
            ]
            if not active_indices:
                break
            step_edge_tt = _edge_tt_numpy_from_z(self.wm, z.detach(), self.cfg)
            active_path_batch: List[List[List[int]]] = []
            active_current_edges: List[int] = []
            step_active_indices: List[int] = []
            for idx in active_indices:
                vehicle = vehicle_states[int(idx)]
                candidates = _current_to_dest_candidate_paths(
                    adj_list=self.adj_list,
                    edge_tt=step_edge_tt,
                    current_edge=int(vehicle.current_edge),
                    dest_edge=int(vehicle.dest_edge),
                    n_routes=n_routes,
                    n_edges=n_edges,
                    penalty_factor=float(self.cfg.av_candidate_path_penalty_factor),
                    fallback_paths=vehicle.candidate_paths,
                )
                if not any(len(path) > 0 for path in candidates):
                    continue
                active_path_batch.append(candidates)
                active_current_edges.append(int(vehicle.current_edge))
                step_active_indices.append(int(idx))
            active_indices = step_active_indices
            if not active_path_batch:
                break
            obs, valid, edge_tt = self._obs_from_paths(
                z=z.detach(),
                path_batch=active_path_batch,
                step_idx=int(start.decision_idx) + h,
                n_total_av=max(n_agents, 1),
                edge_tt_override=step_edge_tt,
            )
            if int(obs.shape[0]) <= 0:
                break
            logits = self.actor.forward_logits(obs)
            selected, executed_one_hot, old_logp, entropy_per_vehicle = _sample_masked_actions(
                logits,
                valid,
                temperature=float(self.cfg.av_policy_sample_temperature),
            )
            old_logp = old_logp.detach()
            entropy = entropy_per_vehicle.mean()
            reward_values: List[float] = []
            for local_i, state_i in enumerate(active_indices):
                vehicle = vehicle_states[int(state_i)]
                action_idx = int(selected[local_i].detach().cpu().item())
                action_idx = int(np.clip(action_idx, 0, n_routes - 1))
                chosen_paths = active_path_batch[local_i]
                chosen = chosen_paths[action_idx] if action_idx < len(chosen_paths) else []
                remaining_before = _imagined_route_remaining_time(
                    path=chosen,
                    edge_tt=edge_tt,
                    current_edge=int(vehicle.current_edge),
                    current_edge_remaining=float(vehicle.current_edge_remaining),
                )
                elapsed_before = float(max(vehicle.elapsed_time, 0.0))
                advance = _advance_imagined_route_progress(
                    path=chosen,
                    edge_tt=edge_tt,
                    current_edge=int(vehicle.current_edge),
                    current_edge_remaining=float(vehicle.current_edge_remaining),
                    step_interval=float(self.cfg.step_interval),
                )
                vehicle.current_edge = int(advance.current_edge)
                vehicle.current_edge_remaining = float(advance.current_edge_remaining)
                vehicle.elapsed_time = float(elapsed_before + max(float(advance.elapsed_cost), 0.0))
                vehicle.arrived = bool(advance.arrived)
                mode_key = str(self.cfg.reward_mode or "capped_step_cost").strip().lower()
                if mode_key in {"progress_fraction", "step_progress", "trip_progress_fraction"}:
                    denom = max(
                        float(remaining_before) + float(elapsed_before),
                        float(max(self.cfg.remaining_tt_eps, 1e-6)),
                    )
                    reward = float(max(float(advance.elapsed_cost), 0.0)) / denom
                    if float(self.cfg.progress_cap) > 0.0:
                        reward = min(reward, float(self.cfg.progress_cap))
                    reward_values.append(float(self.cfg.reward_weight) * float(self.cfg.reward_scale) * reward)
                else:
                    reward_values.append(
                        -float(self.cfg.reward_weight)
                        * float(self.cfg.reward_scale)
                        * float(advance.elapsed_cost)
                        / float(max(self.cfg.reward_temperature, 1e-6))
                    )
                if bool(advance.arrived):
                    arrived_events += 1
            rewards = torch.tensor(reward_values, device=self.device, dtype=obs.dtype)
            action_edge = aggregate_route_probs_to_action_edge(
                executed_one_hot,
                active_path_batch,
                n_edges=n_edges,
                n_routes=n_routes,
                current_edges=torch.tensor(active_current_edges, device=self.device, dtype=torch.long),
                edge_travel_times=torch.from_numpy(edge_tt).to(device=self.device, dtype=torch.float32),
                horizon_seconds=float(self.cfg.step_interval),
            )
            depart_t = torch.from_numpy(np.asarray(start.depart, dtype=np.float32)).to(device=self.device, dtype=z.dtype)
            context_t = torch.from_numpy(np.asarray(start.context, dtype=np.float32)).to(device=self.device, dtype=z.dtype)
            hdv_t, hdv_context_source = self._hdv_context_for_step(start, z.dtype)
            past_packed = torch.stack(latent_history[-1:], dim=0).unsqueeze(0)
            with torch.no_grad():
                z_next, _info = self.wm.predict_next_state_differentiable(
                    z.detach(),
                    action_edge.detach(),
                    past_packed=past_packed.detach(),
                    depart=depart_t,
                    context=context_t,
                    hdv_context=hdv_t,
                )
            next_obs = torch.zeros_like(obs)
            next_valid = torch.zeros((int(obs.shape[0]),), device=self.device, dtype=torch.bool)
            next_edge_tt = _edge_tt_numpy_from_z(self.wm, z_next.detach(), self.cfg)
            global_reward = _global_network_reward_from_latent(
                self.wm,
                z_next,
                next_edge_tt,
                reward_temperature=float(self.cfg.reward_temperature),
            )
            next_path_batch: List[List[List[int]]] = []
            next_positions: List[int] = []
            for local_i, state_i in enumerate(active_indices):
                vehicle = vehicle_states[int(state_i)]
                if bool(vehicle.arrived) or int(vehicle.current_edge) < 0:
                    continue
                candidates = _current_to_dest_candidate_paths(
                    adj_list=self.adj_list,
                    edge_tt=next_edge_tt,
                    current_edge=int(vehicle.current_edge),
                    dest_edge=int(vehicle.dest_edge),
                    n_routes=n_routes,
                    n_edges=n_edges,
                    penalty_factor=float(self.cfg.av_candidate_path_penalty_factor),
                    fallback_paths=vehicle.candidate_paths,
                )
                if not any(len(path) > 0 for path in candidates):
                    continue
                next_path_batch.append(candidates)
                next_positions.append(int(local_i))
            if next_path_batch:
                next_obs_used, next_route_valid, _ = self._obs_from_paths(
                    z=z_next.detach(),
                    path_batch=next_path_batch,
                    step_idx=int(start.decision_idx) + h + 1,
                    n_total_av=max(n_agents, 1),
                    edge_tt_override=next_edge_tt,
                )
                next_has_valid = next_route_valid.any(dim=-1)
                for src_i, dst_i in enumerate(next_positions):
                    if bool(next_has_valid[src_i].detach().cpu().item()):
                        next_obs[int(dst_i)] = next_obs_used[src_i]
                        next_valid[int(dst_i)] = True
            obs_steps.append(obs.detach())
            action_steps.append(selected.detach())
            old_logp_steps.append(old_logp.detach())
            valid_steps.append(valid.detach())
            reward_steps.append(rewards.detach())
            group_steps.append(torch.full((int(obs.shape[0]),), h, device=self.device, dtype=torch.long))
            next_group_steps.append(torch.full((int(obs.shape[0]),), h + 1, device=self.device, dtype=torch.long))
            vehicle_id_steps.append(torch.tensor(active_indices, device=self.device, dtype=torch.long))
            next_obs_steps.append(next_obs.detach())
            next_valid_steps.append(next_valid.detach())
            entropies.append(entropy.detach())
            action_edges.append(action_edge.detach())
            global_latent_steps.append(z.detach())
            global_next_latent_steps.append(z_next.detach())
            global_context_steps.append(context_t.detach())
            global_reward_steps.append(global_reward.detach())
            z = z_next.detach()
            latent_history.append(z)
        if not obs_steps:
            return {"obs": torch.zeros((0, self.obs_dim), device=self.device), "reward_mean": 0.0, "n_samples": 0}
        rewards_flat = torch.cat(reward_steps, dim=0)
        vehicle_ids_flat = torch.cat(vehicle_id_steps, dim=0)
        returns_flat = _discounted_returns_by_vehicle_ids(
            rewards_flat,
            vehicle_ids_flat,
            gamma=float(self.cfg.gamma),
            horizon_steps=int(self.cfg.return_horizon_steps),
        )
        arrived_frac = float(arrived_events) / float(max(n_agents, 1))
        return {
            "obs": torch.cat(obs_steps, dim=0),
            "actions": torch.cat(action_steps, dim=0),
            "old_log_probs": torch.cat(old_logp_steps, dim=0),
            "valid_mask": torch.cat(valid_steps, dim=0),
            "rewards": rewards_flat,
            "returns": returns_flat.detach(),
            "group_index": torch.cat(group_steps, dim=0),
            "next_group_index": torch.cat(next_group_steps, dim=0),
            "vehicle_index": vehicle_ids_flat,
            "next_obs": torch.cat(next_obs_steps, dim=0),
            "next_valid": torch.cat(next_valid_steps, dim=0),
            "entropy_rollout": torch.stack(entropies).mean() if entropies else torch.zeros((), device=self.device),
            "action_edges": action_edges,
            "global_latent_states": torch.stack(global_latent_steps, dim=0),
            "global_next_latent_states": torch.stack(global_next_latent_steps, dim=0),
            "global_contexts": torch.stack(global_context_steps, dim=0),
            "global_rewards": torch.stack(global_reward_steps, dim=0),
            "reward_mean": float(rewards_flat.mean().detach().cpu().item()),
            "return_mean": float(returns_flat.mean().detach().cpu().item()),
            "arrived_frac": float(min(max(arrived_frac, 0.0), 1.0)),
            "n_samples": int(torch.cat(action_steps).numel()),
        }

    def _global_critic_terms(
        self,
        rollout_parts: Sequence[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        losses: List[torch.Tensor] = []
        advantages: List[torch.Tensor] = []
        values_all: List[torch.Tensor] = []
        targets_all: List[torch.Tensor] = []
        rewards_all: List[torch.Tensor] = []
        for part in rollout_parts:
            z = part.get("global_latent_states")
            z_next = part.get("global_next_latent_states")
            contexts = part.get("global_contexts")
            global_rewards = part.get("global_rewards")
            action_list = part.get("action_edges")
            if not (
                torch.is_tensor(z)
                and torch.is_tensor(z_next)
                and torch.is_tensor(contexts)
                and torch.is_tensor(global_rewards)
                and isinstance(action_list, list)
                and action_list
            ):
                continue
            actions = torch.stack(action_list, dim=0).to(device=self.device, dtype=z.dtype)
            n = min(
                int(z.shape[0]),
                int(z_next.shape[0]),
                int(contexts.shape[0]),
                int(global_rewards.reshape(-1).shape[0]),
                int(actions.shape[0]),
            )
            if n <= 0:
                continue
            z = z[:n]
            z_next = z_next[:n]
            contexts = contexts[:n]
            actions = actions[:n]
            global_rewards = global_rewards.reshape(-1)[:n].to(device=self.device, dtype=z.dtype)
            values = self.global_critic(z, action=actions, context=contexts).reshape(-1)
            with torch.no_grad():
                next_actions = torch.cat([actions[1:], actions[-1:]], dim=0)
                next_contexts = torch.cat([contexts[1:], contexts[-1:]], dim=0)
                next_values = self.global_critic(
                    z_next,
                    action=next_actions,
                    context=next_contexts,
                ).reshape(-1)
                targets = lambda_return(
                    global_rewards,
                    next_values,
                    gamma=float(self.cfg.gamma),
                    lambda_=float(self.cfg.ac_lambda),
                ).detach()
            losses.append(0.5 * torch.square(values - targets).mean())
            advantages.append((targets - values.detach()).detach())
            values_all.append(values.detach())
            targets_all.append(targets)
            rewards_all.append(global_rewards.detach())

        if not losses:
            zero = next(self.global_critic.parameters()).sum() * 0.0
            return zero, torch.zeros((0,), device=self.device), {
                "imag_global_reward_mean": 0.0,
                "imag_global_adv_mean": 0.0,
                "imag_global_adv_std": 0.0,
                "imag_global_value_target_ev": 0.0,
            }
        values_cat = torch.cat(values_all, dim=0)
        targets_cat = torch.cat(targets_all, dim=0)
        advantages_cat = torch.cat(advantages, dim=0)
        target_var = torch.var(targets_cat, unbiased=False)
        value_ev = torch.zeros((), device=self.device, dtype=target_var.dtype)
        if float(target_var.detach().cpu().item()) > 1e-12:
            residual_var = torch.var(targets_cat - values_cat, unbiased=False)
            value_ev = 1.0 - residual_var / target_var.clamp_min(1e-12)
        return torch.stack(losses).mean(), advantages_cat, {
            "imag_global_reward_mean": float(torch.cat(rewards_all).mean().cpu().item()),
            "imag_global_adv_mean": float(advantages_cat.mean().cpu().item()),
            "imag_global_adv_std": float(advantages_cat.std(unbiased=False).cpu().item())
            if int(advantages_cat.numel()) > 1
            else 0.0,
            "imag_global_value_target_ev": float(value_ev.cpu().item()),
        }

    def update(self, replay: OnlineWorldModelReplay, rng: np.random.Generator) -> Dict[str, float]:
        steps = replay.all_steps()
        candidate_steps = [s for s in steps if s.samples]
        if not candidate_steps:
            return {"imag_loss": 0.0, "imag_n_samples": 0.0, "imag_skipped": 1.0}
        self.actor.train()
        self.critic.train()
        self.global_critic.train()
        rows: List[Dict[str, float]] = []
        for _ in range(max(int(self.cfg.imagined_updates_per_episode), 1)):
            rollout_parts: List[Dict[str, Any]] = []
            n_starts = max(int(self.cfg.imagined_rollouts_per_update), 1)
            start_idx = rng.choice(len(candidate_steps), size=n_starts, replace=len(candidate_steps) < n_starts)
            for idx in np.asarray(start_idx).reshape(-1):
                rollout_parts.append(self.imagine_from_step(candidate_steps[int(idx)]))
            obs_parts = [p["obs"] for p in rollout_parts if torch.is_tensor(p.get("obs")) and int(p["obs"].shape[0]) > 0]
            if not obs_parts:
                continue
            obs = torch.cat(obs_parts, dim=0)
            actions = torch.cat([p["actions"] for p in rollout_parts if "actions" in p], dim=0)
            rewards = torch.cat([p["rewards"] for p in rollout_parts if "rewards" in p], dim=0)
            valid = torch.cat([p["valid_mask"] for p in rollout_parts if "valid_mask" in p], dim=0)
            old_logp = torch.cat([p["old_log_probs"] for p in rollout_parts if "old_log_probs" in p], dim=0)
            group_parts: List[torch.Tensor] = []
            next_group_parts: List[torch.Tensor] = []
            vehicle_parts: List[torch.Tensor] = []
            part_group_bases: List[int] = []
            group_offset = 0
            vehicle_offset = 0
            for p in rollout_parts:
                group_i = p.get("group_index")
                next_group_i = p.get("next_group_index")
                vehicle_i = p.get("vehicle_index")
                if torch.is_tensor(group_i) and int(group_i.numel()) > 0:
                    base = int(group_offset)
                    part_group_bases.append(base)
                    group_parts.append(group_i.to(device=self.device, dtype=torch.long) + base)
                    max_group = int(group_i.max().detach().cpu().item())
                    if torch.is_tensor(next_group_i) and int(next_group_i.numel()) == int(group_i.numel()):
                        next_group_parts.append(next_group_i.to(device=self.device, dtype=torch.long) + base)
                        max_group = max(max_group, int(next_group_i.max().detach().cpu().item()))
                    group_offset += max_group + 1
                else:
                    part_group_bases.append(-1)
                if torch.is_tensor(vehicle_i) and int(vehicle_i.numel()) > 0:
                    vehicle_parts.append(vehicle_i.to(device=self.device, dtype=torch.long) + int(vehicle_offset))
                    vehicle_offset += int(vehicle_i.max().detach().cpu().item()) + 1
            group = (
                torch.cat(group_parts, dim=0)
                if group_parts and int(torch.cat(group_parts, dim=0).numel()) == int(obs.shape[0])
                else torch.arange(int(obs.shape[0]), device=self.device, dtype=torch.long)
            )
            next_group = (
                torch.cat(next_group_parts, dim=0)
                if next_group_parts and int(torch.cat(next_group_parts, dim=0).numel()) == int(obs.shape[0])
                else group
            )
            vehicle_ids = (
                torch.cat(vehicle_parts, dim=0)
                if vehicle_parts and int(torch.cat(vehicle_parts, dim=0).numel()) == int(obs.shape[0])
                else torch.arange(int(obs.shape[0]), device=self.device, dtype=torch.long)
            )
            next_obs = torch.cat(
                [
                    p["next_obs"] if "next_obs" in p else torch.zeros_like(p["obs"])
                    for p in rollout_parts
                    if "obs" in p and torch.is_tensor(p.get("obs")) and int(p["obs"].shape[0]) > 0
                ],
                dim=0,
            )
            next_valid = torch.cat(
                [
                    p["next_valid"]
                    if "next_valid" in p
                    else torch.zeros((int(p["obs"].shape[0]),), device=self.device, dtype=torch.bool)
                    for p in rollout_parts
                    if "obs" in p and torch.is_tensor(p.get("obs")) and int(p["obs"].shape[0]) > 0
                ],
                dim=0,
            ).to(device=self.device, dtype=torch.bool)
            with torch.no_grad():
                next_values = self.critic(next_obs, group_index=next_group).reshape(-1)
                returns = _lambda_returns_by_vehicle_ids(
                    rewards,
                    next_values,
                    next_valid,
                    vehicle_ids,
                    gamma=float(self.cfg.gamma),
                    lambda_=float(self.cfg.ac_lambda),
                ).detach()
            global_value_loss, compact_global_advantage, global_stats = self._global_critic_terms(rollout_parts)
            global_advantage = torch.zeros((max(int(group_offset), 1),), device=self.device, dtype=returns.dtype)
            global_cursor = 0
            for part, base in zip(rollout_parts, part_group_bases):
                rewards_i = part.get("global_rewards")
                n_global = int(rewards_i.reshape(-1).numel()) if torch.is_tensor(rewards_i) else 0
                if base < 0 or n_global <= 0:
                    continue
                global_advantage[base : base + n_global] = compact_global_advantage[
                    global_cursor : global_cursor + n_global
                ]
                global_cursor += n_global
            logits = self.actor.forward_logits(obs)
            safe_valid = valid.clone()
            has_valid = safe_valid.any(dim=-1)
            safe_valid[~has_valid, 0] = True
            masked_logits = (logits / float(max(self.cfg.av_policy_sample_temperature, 1e-6))).masked_fill(~safe_valid, -1e9)
            dist = Categorical(logits=masked_logits)
            logp = dist.log_prob(actions)
            entropy = dist.entropy()
            values = self.critic(obs, group_index=group).reshape(-1)
            vehicle_adv = returns - values.detach()
            adv = _combine_vehicle_global_advantages(
                vehicle_adv,
                global_advantage,
                group,
                global_weight=float(self.cfg.ac_global_advantage_weight),
            )
            if bool(self.cfg.ac_normalize_advantages) and int(adv.numel()) > 1:
                adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
            if float(self.cfg.ac_advantage_clip) > 0:
                adv = adv.clamp(-float(self.cfg.ac_advantage_clip), float(self.cfg.ac_advantage_clip))
            active = has_valid & torch.isfinite(returns) & torch.isfinite(old_logp)
            active = active & torch.isfinite(values) & torch.isfinite(logp)
            if not bool(active.any().detach().cpu().item()):
                continue
            ratio = torch.exp((logp - old_logp).clamp(-20.0, 20.0))
            if float(self.cfg.ac_ppo_clip_ratio) > 0:
                clipped = ratio.clamp(1.0 - float(self.cfg.ac_ppo_clip_ratio), 1.0 + float(self.cfg.ac_ppo_clip_ratio))
                policy_loss = -torch.minimum(ratio * adv, clipped * adv)[active].mean()
                clip_frac = ((ratio[active] - 1.0).abs() > float(self.cfg.ac_ppo_clip_ratio)).float().mean()
            else:
                policy_loss = -(logp[active] * adv[active]).mean()
                clip_frac = torch.zeros((), device=self.device)
            value_loss = 0.5 * torch.square(values[active] - returns[active]).mean()
            entropy_mean = entropy[active].mean()
            loss = (
                policy_loss
                + float(self.cfg.value_coef) * (value_loss + global_value_loss)
                - float(self.cfg.entropy_coef) * entropy_mean
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.global_critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.cfg.max_grad_norm))
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.cfg.max_grad_norm))
            global_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.global_critic.parameters(),
                float(self.cfg.max_grad_norm),
            )
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            self.global_critic_optimizer.step()
            target_var = torch.var(returns[active].detach(), unbiased=False)
            residual_var = torch.var((returns[active] - values[active]).detach(), unbiased=False)
            value_ev = torch.zeros((), device=self.device, dtype=values.dtype)
            if float(target_var.detach().cpu().item()) > 1e-12:
                value_ev = 1.0 - residual_var / target_var.clamp_min(1e-12)
            rows.append(
                {
                    "imag_loss": float(loss.detach().cpu().item()),
                    "imag_policy_loss": float(policy_loss.detach().cpu().item()),
                    "imag_value_loss": float(value_loss.detach().cpu().item()),
                    "imag_global_value_loss": float(global_value_loss.detach().cpu().item()),
                    "imag_entropy": float(entropy_mean.detach().cpu().item()),
                    "imag_clip_frac": float(clip_frac.detach().cpu().item()),
                    "imag_actor_grad_norm": float(actor_grad_norm.detach().cpu().item()),
                    "imag_critic_grad_norm": float(critic_grad_norm.detach().cpu().item()),
                    "imag_global_critic_grad_norm": float(global_critic_grad_norm.detach().cpu().item()),
                    "imag_adv_std": float(adv.std(unbiased=False).detach().cpu().item())
                    if int(adv.numel()) > 1
                    else 0.0,
                    "imag_adv_mean": float(adv[active].mean().detach().cpu().item()),
                    "imag_vehicle_adv_mean": float(vehicle_adv[active].mean().detach().cpu().item()),
                    "imag_value_mean": float(values[active].detach().mean().cpu().item()),
                    "imag_value_target_ev": float(value_ev.detach().cpu().item()),
                    "imag_next_valid_frac": float(next_valid.to(dtype=torch.float32).mean().detach().cpu().item()),
                    "imag_reward_mean": float(rewards.detach().mean().cpu().item()),
                    "imag_return_mean": float(returns.detach().mean().cpu().item()),
                    "imag_arrived_frac": float(
                        np.mean([float(p.get("arrived_frac", 0.0)) for p in rollout_parts])
                    ),
                    "imag_n_samples": float(int(obs.shape[0])),
                    "imag_skipped": 0.0,
                    **global_stats,
                }
            )
        if not rows:
            return {"imag_loss": 0.0, "imag_n_samples": 0.0, "imag_skipped": 1.0}
        return {k: float(np.mean([row.get(k, 0.0) for row in rows])) for k in rows[-1].keys()}


def build_gretz_online_wm_dreamer_plan(cfg: Optional[OnlineWmDreamerConfig] = None) -> Dict[str, Any]:
    resolved = (cfg or OnlineWmDreamerConfig()).resolve()
    output_dir = Path(resolved.output_dir)
    final_hdv_state = output_dir / "hdv_learning_state_after_online_wm_train.json"
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
    missing = [name for name, value in required_paths.items() if value and not Path(str(value)).exists()]
    reward_mode = str(resolved.reward_mode)
    reward_formula = (
        "min(step_interval, current_route_remaining_tt) / (current_route_remaining_tt + elapsed_time)"
        if reward_mode == "progress_fraction"
        else "-min(step_interval, current_imagined_route_remaining_tt) / reward_temperature"
    )
    return {
        "ablation": "gretz_online_wm_dreamer_actor_critic",
        "network": resolved.network_name,
        "tag": resolved.tag,
        "reward_semantics": {
            "scope": "imagined_vehicle_and_global_critics",
            "mode": resolved.reward_mode,
            "temperature": float(resolved.reward_temperature),
            "gamma": float(resolved.gamma),
            "formula": reward_formula,
            "source": "world_model_decoded_edge_travel_time",
            "progress_state": "per_vehicle_route_suffix_until_arrival",
            "global_formula": "-decoded_all_vehicle_count_weighted_edge_tt / reward_temperature",
        },
        "world_model_usage": {
            "encoder_for_actor_obs": True,
            "policy_obs_mode": str(resolved.policy_obs_mode),
            "cfm_transition_in_training": True,
            "imagined_action_mode": "stochastic_masked_sample_one_hot",
            "real_collection_action_mode": (
                "deterministic_argmax_one_hot"
                if str(resolved.real_policy_route_selection) == "argmax"
                else "stochastic_policy_sample_one_hot"
            ),
            "real_collection_wm_action_alignment": "same_selected_one_hot",
            "world_model_online_update": True,
            "updated_modules": ["flow_projector", "cfm_dynamics"],
            "direct_real_a2c_update": bool(resolved.direct_real_a2c_update),
            "freeze_policy_obs_world_model": bool(resolved.freeze_policy_obs_world_model),
            "wm_grad_enabled": True,
        },
        "training": {
            "algorithm": "online_dreamer_world_model_actor_critic",
            "actor_from_scratch": not bool(resolved.resume_checkpoint),
            "real_policy_route_selection": str(resolved.real_policy_route_selection),
            "episodes": int(resolved.train_episodes),
            "wm_replay_buffer_episodes": int(resolved.wm_replay_buffer_episodes),
            "wm_update_epochs": int(resolved.wm_update_epochs),
            "wm_update_every_episodes": int(resolved.wm_update_every_episodes),
            "wm_updates_per_collect": int(resolved.wm_updates_per_collect),
            "wm_replay_require_completion": bool(resolved.wm_replay_require_completion),
            "wm_replay_completion_threshold": float(resolved.wm_replay_completion_threshold),
            "wm_batch_size": int(resolved.wm_batch_size),
            "wm_hdv_target_source": str(resolved.wm_hdv_target_source),
            "wm_hdv_policy_target_mode": str(resolved.wm_hdv_policy_target_mode),
            "wm_hdv_policy_action_mode": str(resolved.wm_hdv_policy_action_mode),
            "wm_transition_hdv_context_mode": str(resolved.wm_transition_hdv_context_mode),
            "wm_transition_hdv_context_weight": float(resolved.wm_transition_hdv_context_weight),
            "wm_transition_hdv_context_scale": float(resolved.wm_transition_hdv_context_scale),
            "wm_transition_hdv_action_mode": str(resolved.wm_transition_hdv_action_mode),
            "wm_transition_hdv_action_weight": float(resolved.wm_transition_hdv_action_weight),
            "wm_transition_hdv_action_scale": float(resolved.wm_transition_hdv_action_scale),
            "imagined_hdv_context_source": str(resolved.imagined_hdv_context_source),
            "hdv_response_predictor_enabled": bool(resolved.hdv_response_predictor_enabled),
            "hdv_response_predictor_lr": float(resolved.hdv_response_predictor_lr),
            "hdv_response_predictor_weight_decay": float(resolved.hdv_response_predictor_weight_decay),
            "hdv_response_predictor_hidden_dim": int(resolved.hdv_response_predictor_hidden_dim),
            "hdv_response_predictor_depth": int(resolved.hdv_response_predictor_depth),
            "hdv_response_predictor_update_epochs": int(resolved.hdv_response_predictor_update_epochs),
            "hdv_response_predictor_min_pairs": int(resolved.hdv_response_predictor_min_pairs),
            "hdv_response_predictor_fallback": str(resolved.hdv_response_predictor_fallback),
            "hdv_response_predictor_ckpt": str(resolved.hdv_response_predictor_ckpt),
            "hdv_response_predictor_progress_denominator": int(resolved.hdv_response_predictor_progress_denominator),
            "imag_horizon": int(resolved.imag_horizon),
            "imagined_updates_per_episode": int(resolved.imagined_updates_per_episode),
            "imagined_update_passes_per_collect": int(resolved.imagined_update_passes_per_collect),
            "imagined_rollouts_per_update": int(resolved.imagined_rollouts_per_update),
            "min_replay_episodes_before_actor": int(resolved.min_replay_episodes_before_actor),
            "ac_lambda": float(resolved.ac_lambda),
            "global_critic_enabled": True,
            "ac_global_advantage_weight": float(resolved.ac_global_advantage_weight),
            "actor_advantage": "vehicle_plus_global_lambda_advantage",
            "actor_lr": float(resolved.lr),
            "critic_lr": float(resolved.critic_lr),
            "value_coef": float(resolved.value_coef),
            "entropy_coef": float(resolved.entropy_coef),
            "ac_advantage_clip": float(resolved.ac_advantage_clip),
            "ac_ppo_clip_ratio": float(resolved.ac_ppo_clip_ratio),
            "ac_normalize_advantages": bool(resolved.ac_normalize_advantages),
            "route_cost_logit_bias": float(resolved.route_cost_logit_bias),
            "av_policy_sample_temperature": float(resolved.av_policy_sample_temperature),
            "max_grad_norm": float(resolved.max_grad_norm),
            "wm_prediction_diagnostic_every_episodes": int(resolved.wm_prediction_diagnostic_every_episodes),
            "wm_prediction_diagnostic_max_steps": int(resolved.wm_prediction_diagnostic_max_steps),
            "wm_hdv_pressure_trace_enabled": bool(resolved.wm_hdv_pressure_trace_enabled),
            "actor_update_schedule": str(resolved.actor_update_schedule),
            "actor_update_min_passes": int(resolved.actor_update_min_passes),
            "actor_update_max_passes": int(resolved.actor_update_max_passes),
            "actor_update_warmup_episodes": int(resolved.actor_update_warmup_episodes),
            "actor_update_ramp_episodes": int(resolved.actor_update_ramp_episodes),
            "actor_update_gate_enabled": bool(resolved.actor_update_gate_enabled),
            "actor_update_gate_min_diag_samples": int(resolved.actor_update_gate_min_diag_samples),
            "actor_update_gate_min_top1_match": float(resolved.actor_update_gate_min_top1_match),
            "actor_update_gate_max_route_mae": float(resolved.actor_update_gate_max_route_mae),
            "actor_update_gate_max_current_edge_mae": float(resolved.actor_update_gate_max_current_edge_mae),
            "actor_update_gate_max_regret": float(resolved.actor_update_gate_max_regret),
            "actor_update_gate_fail_passes": int(resolved.actor_update_gate_fail_passes),
            "actor_update_gate_missing_passes": int(resolved.actor_update_gate_missing_passes),
            "checkpoint_paths": checkpoints,
        },
        "evaluation": {
            "episodes_per_checkpoint": int(resolved.eval_episodes),
            "action_mode": (
                "deterministic_masked_argmax"
                if str(resolved.eval_policy_route_selection) == "argmax"
                else "stochastic_masked_sample"
            ),
            "policy_route_selection": str(resolved.eval_policy_route_selection),
            "hdv_state_mode": "final_online_training_state",
            "hdv_state_path": str(final_hdv_state),
            "eval_dirs": eval_dirs,
            "uses_updated_world_model_checkpoints": True,
        },
        "paths": required_paths,
        "output_dir": str(output_dir),
        "missing": missing,
    }


class OnlineWmDreamerTrainer:
    def __init__(self, cfg: OnlineWmDreamerConfig):
        self.cfg = cfg.resolve()
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(int(self.cfg.seed))
        np.random.seed(int(self.cfg.seed))
        self.rng = np.random.default_rng(int(self.cfg.seed))
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self.wm = self._load_world_model()
        self.policy_obs_mode = str(getattr(self.cfg, "policy_obs_mode", "wm_latent") or "wm_latent").strip().lower()
        if self.policy_obs_mode in {"", "wm", "wm_latent", "latent"}:
            self.policy_obs_mode = "wm_latent"
        elif self.policy_obs_mode in {"toy_live", "live_toy", "live_edge_registry"}:
            self.policy_obs_mode = "toy_live"
        else:
            raise ValueError(f"Unsupported policy_obs_mode={self.cfg.policy_obs_mode!r}")
        # Loading the world model consumes torch RNG; reset before actor/critic init so
        # real-A2C-only can be compared against the no-WM online AC baseline.
        torch.manual_seed(int(self.cfg.seed))
        policy_d_obs = 8 if self.policy_obs_mode == "toy_live" else int(self.wm.d_obs)
        self.obs_dim = vehicle_mode_obs_dim(
            n_routes=int(self.cfg.n_routes),
            d_obs=int(policy_d_obs),
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
            arrival_output_dim=int(self.cfg.n_routes)
            if bool(getattr(self.cfg, "vehicle_arrival_head_enabled", False))
            else 0,
        ).to(self.device)
        self.global_critic = CentralizedLatentValueCritic(
            d_spatial=int(self.wm.d_spatial),
            action_dim=int(self.wm.n_edges),
            context_dim=3,
            hidden_dim=int(self.cfg.hidden_dim),
            n_hidden_layers=int(self.cfg.n_hidden_layers),
            aggregation="mean",
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(self.cfg.lr))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(self.cfg.critic_lr))
        self.global_critic_optimizer = torch.optim.Adam(
            self.global_critic.parameters(),
            lr=float(self.cfg.critic_lr),
        )
        self.wm_updater = OnlineWorldModelUpdater(self.wm, self.cfg, self.device)
        self.policy_obs_wm = self._load_world_model() if bool(self.cfg.freeze_policy_obs_world_model) else self.wm
        self.policy_obs_wm.eval()
        if bool(self.cfg.freeze_policy_obs_world_model):
            for param in self.policy_obs_wm.parameters():
                param.requires_grad_(False)
        self.imagined_ac = OnlineImaginedActorCritic(
            wm=self.wm,
            actor=self.actor,
            critic=self.critic,
            global_critic=self.global_critic,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            global_critic_optimizer=self.global_critic_optimizer,
            cfg=self.cfg,
            device=self.device,
            obs_dim=int(self.obs_dim),
        )
        hdv_response_ckpt = self._resolve_hdv_response_predictor_ckpt()
        self.hdv_response_predictor_ckpt_loaded = ""
        self.hdv_response_model: Optional[CausalHdvResponseModel] = None
        if bool(self.cfg.hdv_response_predictor_enabled) or hdv_response_ckpt is not None:
            self.hdv_response_model = CausalHdvResponseModel(
                n_edges=int(self.wm.n_edges),
                cfg=self.cfg,
                device=self.device,
            )
            if hdv_response_ckpt is not None:
                loaded_state = self.hdv_response_model.load_checkpoint(hdv_response_ckpt)
                self.hdv_response_predictor_ckpt_loaded = str(hdv_response_ckpt)
                print(
                    "[online-wm-dreamer] loaded HDV response predictor: "
                    f"{hdv_response_ckpt} pairs={int(loaded_state.get('trained_pairs', 0) or 0)} "
                    f"progress_den={int(loaded_state.get('progress_denominator', 0) or 0)}",
                    flush=True,
                )
        self.replay = OnlineWorldModelReplay(
            int(self.cfg.wm_replay_buffer_episodes),
            hdv_target_source=str(self.cfg.wm_hdv_target_source),
        )
        self.collector: Optional[OnlineWmDreamerCollector] = None
        self._dynamics_checkpoint_state = torch.load(str(self.cfg.dynamics_ckpt), map_location="cpu", weights_only=False)
        self.loaded_resume_episode = self._load_resume_checkpoint()

    def _resolve_hdv_response_predictor_ckpt(self) -> Optional[Path]:
        raw = str(getattr(self.cfg, "hdv_response_predictor_ckpt", "") or "").strip()
        explicit = bool(raw)
        if not raw:
            post_state_path = str(getattr(self.cfg, "post_mutation_state_in", "") or "").strip()
            if post_state_path:
                state_path = _resolve_path(post_state_path)
                state = _load_json_object(state_path)
                if isinstance(state, dict):
                    raw = str(state.get("hdv_response_predictor_ref", "") or "").strip()
                    if not raw:
                        refs = state.get("artifact_refs", {})
                        if isinstance(refs, dict):
                            raw = str(refs.get("hdv_response_predictor_ckpt", "") or "").strip()
        if not raw:
            return None
        path = _resolve_path(raw)
        if not path.exists():
            if explicit:
                raise FileNotFoundError(f"HDV response predictor checkpoint not found: {path}")
            return None
        return path

    def _save_hdv_pressure_trace(self, episode: int, steps: Sequence[OnlineWorldModelStep]) -> Optional[Path]:
        if not bool(getattr(self.cfg, "wm_hdv_pressure_trace_enabled", True)) or not steps:
            return None
        trace_dir = self.output_dir / "hdv_edge_pressure_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        hdv_stack = _stack_np([s.hdv_action for s in steps]).clip(min=0.0).astype(np.float32)
        hdv_mean = hdv_stack.mean(axis=0).astype(np.float32) if hdv_stack.size else np.zeros((0,), dtype=np.float32)
        denom = float(hdv_mean.sum())
        if denom > 1e-8:
            hdv_mean = hdv_mean / denom
        target_replay = OnlineWorldModelReplay(
            1,
            hdv_target_source=str(getattr(self.cfg, "wm_hdv_target_source", "episode_mean_current_occupancy")),
        )
        if str(target_replay.hdv_target_source) in {
            "static_edge_demand",
            "static_episode_edge_demand",
            "episode_static_edge_demand",
        }:
            hdv_target = target_replay._episode_static_edge_demand(steps)
            if hdv_target is None:
                hdv_target = target_replay._episode_current_occupancy_mean(steps)
        else:
            hdv_target = target_replay._episode_current_occupancy_mean(steps)
        if hdv_target is None:
            hdv_target = np.zeros_like(hdv_mean, dtype=np.float32)
        hdv_static_stack = _stack_np(
            [
                s.hdv_static_edge_demand
                if s.hdv_static_edge_demand is not None
                else np.zeros_like(s.hdv_action, dtype=np.float32)
                for s in steps
            ]
        ).astype(np.float32)
        hdv_static_mean = (
            hdv_static_stack.clip(min=0.0).mean(axis=0).astype(np.float32)
            if hdv_static_stack.size
            else np.zeros_like(hdv_target, dtype=np.float32)
        )
        static_denom = float(hdv_static_mean.sum())
        if static_denom > 1e-8:
            hdv_static_mean = hdv_static_mean / static_denom
        edge_registry = getattr(getattr(self, "collector", None), "_edge_registry", None)
        edge_ids = np.asarray(list(getattr(edge_registry, "edge_ids", [])), dtype=str)
        path = trace_dir / f"episode_{int(episode):05d}.npz"
        np.savez_compressed(
            path,
            episode=np.asarray([int(episode)], dtype=np.int64),
            decision_idx=np.asarray([int(s.decision_idx) for s in steps], dtype=np.int64),
            sim_step=np.asarray([int(s.sim_step) for s in steps], dtype=np.int64),
            hdv_action=hdv_stack,
            hdv_episode_mean=hdv_mean,
            hdv_target=np.asarray(hdv_target, dtype=np.float32),
            hdv_static_edge_demand=hdv_static_stack,
            hdv_static_edge_demand_mean=hdv_static_mean,
            hdv_target_source=np.asarray(
                [str(getattr(self.cfg, "wm_hdv_target_source", "episode_mean_current_occupancy"))],
                dtype=str,
            ),
            av_action=_stack_np([s.av_action for s in steps]).astype(np.float32),
            flow=_stack_np([s.flow for s in steps]).astype(np.float32),
            next_flow=_stack_np([s.next_flow for s in steps]).astype(np.float32),
            depart=_stack_np([s.depart for s in steps]).astype(np.float32),
            context=_stack_np([s.context for s in steps]).astype(np.float32),
            edge_ids=edge_ids,
        )
        return path

    def _episode_allowed_for_wm_replay(self, row: Dict[str, Any]) -> bool:
        if not bool(self.cfg.wm_replay_require_completion):
            return True
        threshold = float(self.cfg.wm_replay_completion_threshold)
        completion = float(row.get("completion_rate", 0.0) or 0.0)
        return completion >= threshold - 1e-12

    def _wm_update_due(self, episode: int) -> bool:
        every = max(int(self.cfg.wm_update_every_episodes), 1)
        return int(episode) % every == 0

    @staticmethod
    def _edge_distribution_tv(pred: Optional[np.ndarray], target: Optional[np.ndarray]) -> float:
        if pred is None or target is None:
            return 0.0
        p = np.asarray(pred, dtype=np.float32).reshape(-1).clip(min=0.0)
        q = np.asarray(target, dtype=np.float32).reshape(-1).clip(min=0.0)
        if int(p.size) != int(q.size) or int(p.size) <= 0:
            return 0.0
        p_sum = float(p.sum())
        q_sum = float(q.sum())
        if p_sum > 1e-8:
            p = p / p_sum
        if q_sum > 1e-8:
            q = q / q_sum
        return float(0.5 * np.abs(p - q).sum())

    def _episode_static_hdv_target(self, steps: Sequence[OnlineWorldModelStep]) -> Optional[np.ndarray]:
        static_rows = [
            np.asarray(step.hdv_static_edge_demand, dtype=np.float32)
            for step in steps
            if step.hdv_static_edge_demand is not None
        ]
        if static_rows:
            arr = np.mean(np.stack(static_rows, axis=0), axis=0).astype(np.float32)
        else:
            hdv_rows = [np.asarray(step.hdv_action, dtype=np.float32) for step in steps]
            if not hdv_rows:
                return None
            arr = np.mean(np.stack(hdv_rows, axis=0), axis=0).astype(np.float32)
        arr = arr.clip(min=0.0)
        denom = float(arr.sum())
        if denom > 1e-8:
            arr = arr / denom
        return arr.astype(np.float32)

    def _predict_hdv_response_for_episode(
        self,
        episode: int,
        steps: Sequence[OnlineWorldModelStep],
    ) -> Tuple[Optional[np.ndarray], str, Dict[str, float]]:
        metrics: Dict[str, float] = {
            "hdv_response_pred_enabled": float(bool(self.cfg.hdv_response_predictor_enabled)),
            "hdv_response_pred_source_code": 0.0,
            "hdv_response_pred_tv": 0.0,
        }
        if self.hdv_response_model is None:
            return None, "", metrics
        pred, pred_metrics = self.hdv_response_model.predict_for_next_episode(
            self.replay.summaries,
            episode=int(episode),
        )
        metrics.update(pred_metrics)
        source_code = float(metrics.get("hdv_response_pred_source_code", 0.0))
        if source_code == 2.0:
            source = "causal_response_predictor"
        elif source_code == 1.5:
            source = "fallback_prev_hdv"
        elif source_code == 1.25:
            source = "fallback_zero"
        else:
            source = ""
        target = self._episode_static_hdv_target(steps)
        metrics["hdv_response_pred_tv"] = self._edge_distribution_tv(pred, target)
        return pred, source, metrics

    def _update_hdv_response_predictor(self) -> Dict[str, float]:
        if self.hdv_response_model is None:
            return {
                "hdv_response_loss": 0.0,
                "hdv_response_updates": 0.0,
                "hdv_response_pairs": 0.0,
            }
        return self.hdv_response_model.update(self.replay.summaries)

    def _skipped_wm_update(self, *, reason: str) -> Dict[str, float]:
        return {
            "wm_loss": 0.0,
            "wm_cfm_loss": 0.0,
            "wm_transition_loss": 0.0,
            "wm_hdv_ce": 0.0,
            "wm_weighted_flow_mse": 0.0,
            "wm_transition_mse": 0.0,
            "wm_updates": 0.0,
            "wm_replay_transitions": float(len(self.replay.all_steps())),
            "wm_skipped": 1.0,
            "wm_skip_reason": reason,
        }

    def _direct_real_a2c_update(
        self,
        samples: Sequence[OnlineDecisionSample],
        *,
        replay_episodes: int = 1,
    ) -> Dict[str, float]:
        if not bool(getattr(self.cfg, "direct_real_a2c_update", False)):
            return {"real_a2c_enabled": 0.0, "real_a2c_skipped": 1.0, "real_a2c_skip_reason": "disabled"}
        if len(samples) < int(self.cfg.min_samples_per_update):
            return {
                "real_a2c_enabled": 1.0,
                "real_a2c_skipped": 1.0,
                "real_a2c_skip_reason": "min_samples",
                "real_a2c_n_samples": float(len(samples)),
                "real_a2c_replay_episodes": float(max(int(replay_episodes), 0)),
            }
        tensors = samples_to_tensors(
            samples,
            device=self.device,
            gamma=float(self.cfg.gamma),
            return_horizon_steps=int(self.cfg.return_horizon_steps),
        )
        self.actor.train()
        self.critic.train()
        rows: List[Dict[str, float]] = []
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
            arrival_diag = _arrival_head_loss_and_diag(
                critic=self.critic,
                obs=tensors["obs"],
                actions=tensors["actions"],
                valid_mask=tensors["valid_mask"],
                group_index=tensors["group_index"],
                arrival_remaining=tensors.get("arrival_remaining", torch.zeros((0,), device=self.device)),
                route_costs=tensors.get("route_costs", torch.zeros((0, 0), device=self.device)),
                horizon_seconds=float(getattr(self.cfg, "vehicle_arrival_diag_horizon_seconds", 14.0)),
            )
            arrival_loss = arrival_diag.get("loss", result["loss"] * 0.0)
            loss = result["loss"] + float(getattr(self.cfg, "vehicle_arrival_loss_coef", 1.0)) * arrival_loss
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.cfg.max_grad_norm))
            critic_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.cfg.max_grad_norm))
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            def _value(name: str) -> float:
                value = result.get(name)
                if torch.is_tensor(value):
                    return float(value.detach().cpu().item())
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0

            rows.append(
                {
                    "real_a2c_loss": _value("loss"),
                    "real_a2c_policy_loss": _value("policy_loss"),
                    "real_a2c_value_loss": _value("value_loss"),
                    "real_a2c_entropy": _value("entropy"),
                    "real_a2c_n_samples": _value("n_samples"),
                    "real_a2c_return_mean": _value("return_mean"),
                    "real_a2c_return_std": _value("return_std"),
                    "real_a2c_adv_mean": _value("adv_mean"),
                    "real_a2c_adv_std": _value("adv_std"),
                    "real_a2c_clip_frac": _value("clip_frac"),
                    "real_a2c_actor_grad_norm": float(actor_grad.detach().cpu().item()),
                    "real_a2c_critic_grad_norm": float(critic_grad.detach().cpu().item()),
                    "real_a2c_arrival_loss": float(arrival_loss.detach().cpu().item())
                    if torch.is_tensor(arrival_loss)
                    else 0.0,
                    "real_a2c_arrival_n": float(arrival_diag.get("n", torch.zeros(())).detach().cpu().item())
                    if torch.is_tensor(arrival_diag.get("n"))
                    else 0.0,
                    "real_a2c_arrival_mae": float(arrival_diag.get("mae", torch.zeros(())).detach().cpu().item())
                    if torch.is_tensor(arrival_diag.get("mae"))
                    else float("nan"),
                    "real_a2c_arrival_bias": float(arrival_diag.get("bias", torch.zeros(())).detach().cpu().item())
                    if torch.is_tensor(arrival_diag.get("bias"))
                    else float("nan"),
                    "real_a2c_arrival_corr": float(arrival_diag.get("corr", torch.zeros(())).detach().cpu().item())
                    if torch.is_tensor(arrival_diag.get("corr"))
                    else float("nan"),
                    "real_a2c_arrival_pred_mean": float(arrival_diag.get("pred_mean", torch.zeros(())).detach().cpu().item())
                    if torch.is_tensor(arrival_diag.get("pred_mean"))
                    else float("nan"),
                    "real_a2c_arrival_target_mean": float(arrival_diag.get("target_mean", torch.zeros(())).detach().cpu().item())
                    if torch.is_tensor(arrival_diag.get("target_mean"))
                    else float("nan"),
                    "real_a2c_arrival_precision": float(arrival_diag.get("precision", torch.zeros(())).detach().cpu().item())
                    if torch.is_tensor(arrival_diag.get("precision"))
                    else float("nan"),
                    "real_a2c_arrival_recall": float(arrival_diag.get("recall", torch.zeros(())).detach().cpu().item())
                    if torch.is_tensor(arrival_diag.get("recall"))
                    else float("nan"),
                    "real_a2c_progress_arrival_n": float(arrival_diag.get("progress_n", torch.zeros(())).detach().cpu().item())
                    if torch.is_tensor(arrival_diag.get("progress_n"))
                    else 0.0,
                    "real_a2c_progress_arrival_mae": float(arrival_diag.get("progress_mae", torch.zeros(())).detach().cpu().item())
                    if torch.is_tensor(arrival_diag.get("progress_mae"))
                    else float("nan"),
                    "real_a2c_progress_arrival_bias": float(arrival_diag.get("progress_bias", torch.zeros(())).detach().cpu().item())
                    if torch.is_tensor(arrival_diag.get("progress_bias"))
                    else float("nan"),
                    "real_a2c_progress_arrival_corr": float(arrival_diag.get("progress_corr", torch.zeros(())).detach().cpu().item())
                    if torch.is_tensor(arrival_diag.get("progress_corr"))
                    else float("nan"),
                    "real_a2c_progress_arrival_precision": float(arrival_diag.get("progress_precision", torch.zeros(())).detach().cpu().item())
                    if torch.is_tensor(arrival_diag.get("progress_precision"))
                    else float("nan"),
                    "real_a2c_progress_arrival_recall": float(arrival_diag.get("progress_recall", torch.zeros(())).detach().cpu().item())
                    if torch.is_tensor(arrival_diag.get("progress_recall"))
                    else float("nan"),
                }
            )

        def _mean(name: str) -> float:
            vals = [float(row[name]) for row in rows if name in row and np.isfinite(float(row[name]))]
            return float(np.mean(vals)) if vals else 0.0

        return {
            "real_a2c_enabled": 1.0,
            "real_a2c_skipped": 0.0,
            "real_a2c_replay_episodes": float(max(int(replay_episodes), 0)),
            "real_a2c_update_epochs": float(update_epochs),
            "real_a2c_loss": _mean("real_a2c_loss"),
            "real_a2c_policy_loss": _mean("real_a2c_policy_loss"),
            "real_a2c_value_loss": _mean("real_a2c_value_loss"),
            "real_a2c_entropy": _mean("real_a2c_entropy"),
            "real_a2c_n_samples": _mean("real_a2c_n_samples"),
            "real_a2c_return_mean": _mean("real_a2c_return_mean"),
            "real_a2c_return_std": _mean("real_a2c_return_std"),
            "real_a2c_adv_mean": _mean("real_a2c_adv_mean"),
            "real_a2c_adv_std": _mean("real_a2c_adv_std"),
            "real_a2c_clip_frac": _mean("real_a2c_clip_frac"),
            "real_a2c_actor_grad_norm": _mean("real_a2c_actor_grad_norm"),
            "real_a2c_critic_grad_norm": _mean("real_a2c_critic_grad_norm"),
            "real_a2c_arrival_loss": _mean("real_a2c_arrival_loss"),
            "real_a2c_arrival_n": _mean("real_a2c_arrival_n"),
            "real_a2c_arrival_mae": _mean("real_a2c_arrival_mae"),
            "real_a2c_arrival_bias": _mean("real_a2c_arrival_bias"),
            "real_a2c_arrival_corr": _mean("real_a2c_arrival_corr"),
            "real_a2c_arrival_pred_mean": _mean("real_a2c_arrival_pred_mean"),
            "real_a2c_arrival_target_mean": _mean("real_a2c_arrival_target_mean"),
            "real_a2c_arrival_precision": _mean("real_a2c_arrival_precision"),
            "real_a2c_arrival_recall": _mean("real_a2c_arrival_recall"),
            "real_a2c_progress_arrival_n": _mean("real_a2c_progress_arrival_n"),
            "real_a2c_progress_arrival_mae": _mean("real_a2c_progress_arrival_mae"),
            "real_a2c_progress_arrival_bias": _mean("real_a2c_progress_arrival_bias"),
            "real_a2c_progress_arrival_corr": _mean("real_a2c_progress_arrival_corr"),
            "real_a2c_progress_arrival_precision": _mean("real_a2c_progress_arrival_precision"),
            "real_a2c_progress_arrival_recall": _mean("real_a2c_progress_arrival_recall"),
        }

    def _actor_update_passes_for_episode(self, episode: int) -> Tuple[int, str]:
        base = max(int(self.cfg.imagined_update_passes_per_collect), 0)
        schedule = str(getattr(self.cfg, "actor_update_schedule", "fixed") or "fixed").strip().lower()
        if schedule in {"", "fixed", "constant"}:
            return base, "fixed"
        min_passes = max(int(getattr(self.cfg, "actor_update_min_passes", 0)), 0)
        max_passes_cfg = int(getattr(self.cfg, "actor_update_max_passes", 0))
        max_passes = max(base, max_passes_cfg, min_passes)
        warmup = max(int(getattr(self.cfg, "actor_update_warmup_episodes", 0)), 0)
        if int(episode) <= warmup:
            return min_passes, "warmup"
        if schedule == "linear":
            ramp = max(int(getattr(self.cfg, "actor_update_ramp_episodes", 0)), 1)
            progress = min(max((int(episode) - warmup) / float(ramp), 0.0), 1.0)
            passes = int(round(float(min_passes) + progress * float(max_passes - min_passes)))
            return max(min(passes, max_passes), 0), "linear"
        if schedule == "diagnostic":
            return max_passes, "diagnostic"
        if schedule == "stair":
            ramp = max(int(getattr(self.cfg, "actor_update_ramp_episodes", 0)), 1)
            if int(episode) <= warmup + ramp:
                return max(min(base, max_passes), min_passes), "stair_mid"
            return max_passes, "stair_high"
        return base, f"unknown_schedule:{schedule}"

    def _actor_gate_passes(
        self,
        *,
        target_passes: int,
        diag: Optional[Dict[str, Any]],
    ) -> Tuple[int, str]:
        target = max(int(target_passes), 0)
        if target <= 0:
            return 0, "target_zero"
        if not bool(getattr(self.cfg, "actor_update_gate_enabled", False)):
            return target, "gate_disabled"
        missing_passes = max(int(getattr(self.cfg, "actor_update_gate_missing_passes", 0)), 0)
        fail_passes = max(int(getattr(self.cfg, "actor_update_gate_fail_passes", 0)), 0)
        samples = _finite_metric(diag, "wm_pred_diag_samples")
        min_samples = max(int(getattr(self.cfg, "actor_update_gate_min_diag_samples", 1)), 0)
        if samples is None or samples < float(min_samples):
            return min(target, missing_passes), "missing_diag"

        failures: List[str] = []
        min_top1 = float(getattr(self.cfg, "actor_update_gate_min_top1_match", 0.0) or 0.0)
        top1 = _finite_metric(diag, "wm_pred_diag_top1_match")
        if min_top1 > 0.0 and (top1 is None or top1 < min_top1):
            failures.append("top1")

        max_route_mae = float(getattr(self.cfg, "actor_update_gate_max_route_mae", 0.0) or 0.0)
        route_mae = _finite_metric(diag, "wm_pred_diag_route_mae")
        if max_route_mae > 0.0 and (route_mae is None or route_mae > max_route_mae):
            failures.append("route_mae")

        max_current_mae = float(getattr(self.cfg, "actor_update_gate_max_current_edge_mae", 0.0) or 0.0)
        current_mae = _finite_metric(diag, "wm_pred_diag_current_edge_mae")
        if max_current_mae > 0.0 and (current_mae is None or current_mae > max_current_mae):
            failures.append("current_edge_mae")

        max_regret = float(getattr(self.cfg, "actor_update_gate_max_regret", 0.0) or 0.0)
        regret = _finite_metric(diag, "wm_pred_diag_wm_best_live_regret")
        if max_regret > 0.0 and (regret is None or regret > max_regret):
            failures.append("regret")

        if failures:
            return min(target, fail_passes), "gate_fail:" + ",".join(failures)
        return target, "gate_pass"

    def _load_world_model(self):
        from dreamer4.urban_routing.urban_world_model import UrbanWorldModel

        wm = UrbanWorldModel(
            flow_projector_ckpt=str(self.cfg.projector_ckpt),
            dynamics_ckpt=str(self.cfg.dynamics_ckpt),
            edge_registry_path=str(self.cfg.edge_registry),
            device=str(self.device),
            use_amp=False,
            cfm_init_mode=str(self.cfg.wm_cfm_init_mode),
            policy_obs_source=str(self.cfg.wm_policy_obs_source),
        )
        wm.eval()
        return wm

    def _load_resume_checkpoint(self) -> int:
        ckpt_path = str(getattr(self.cfg, "resume_checkpoint", "") or "").strip()
        if not ckpt_path:
            return 0
        path = _resolve_path(ckpt_path)
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(state.get("actor") or state.get("policy"))
        critic_state = state.get("critic") or state.get("vehicle_critic")
        if critic_state is not None:
            self.critic.load_state_dict(critic_state, strict=False)
        global_critic_state = state.get("global_critic")
        if global_critic_state is not None:
            self.global_critic.load_state_dict(global_critic_state, strict=True)
        if "flow_projector" in state:
            self.wm.flow_projector.load_state_dict(state["flow_projector"], strict=False)
        if "dynamics" in state:
            self.wm.dynamics.load_state_dict(state["dynamics"], strict=True)
        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "global_critic_optimizer" in state:
            self.global_critic_optimizer.load_state_dict(state["global_critic_optimizer"])
        if "wm_optimizer" in state:
            self.wm_updater.optimizer.load_state_dict(state["wm_optimizer"])
        if self.hdv_response_model is not None:
            hdv_response_state = state.get("hdv_response_predictor")
            if hdv_response_state is not None:
                self.hdv_response_model.model.load_state_dict(hdv_response_state, strict=True)
            hdv_response_optimizer_state = state.get("hdv_response_optimizer")
            if hdv_response_optimizer_state is not None:
                self.hdv_response_model.optimizer.load_state_dict(hdv_response_optimizer_state)
            self.hdv_response_model.trained_pairs = int(state.get("hdv_response_trained_pairs", 0) or 0)
        episode = int(state.get("episode", 0) or 0)
        print(f"[online-wm-dreamer] resumed from {path} at episode {episode}", flush=True)
        return episode

    def _build_collector(self) -> OnlineWmDreamerCollector:
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
        num_agents, _num_machines, max_start_time = compute_agent_counts(agents_df, float(self.cfg.ratio_machines))
        sim_length = int(
            max(
                float(max_start_time),
                float(agents_df["start_time"].max()) if "start_time" in agents_df.columns else float(max_start_time),
            )
            + float(self.cfg.sim_buffer_after_last_depart)
        )
        collector_cls = OnlineWmToyLiveCollector if self.policy_obs_mode == "toy_live" else OnlineWmDreamerCollector
        collector = collector_cls(
            wm=self.policy_obs_wm,
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
        if self.policy_obs_mode == "toy_live":
            collector._av_execution_mode = str(getattr(self.cfg, "av_execution_mode", "dynamic"))
        self.collector = collector
        self.rollout_max_steps = (
            int(self.cfg.max_steps_per_episode)
            if int(self.cfg.max_steps_per_episode) > 0
            else int(sim_length)
        )
        return collector

    def _save_world_model_files(self, episode: int) -> Tuple[Path, Path]:
        wm_dir = self.output_dir / "checkpoints" / f"wm_ep{int(episode):05d}"
        wm_dir.mkdir(parents=True, exist_ok=True)
        projector_path = wm_dir / f"wm_projector_ep{int(episode):05d}.pt"
        dynamics_path = wm_dir / f"wm_dynamics_ep{int(episode):05d}.pt"
        self.wm.flow_projector.save(str(projector_path))
        dyn_state = dict(self._dynamics_checkpoint_state)
        dyn_state["dynamics"] = self.wm.dynamics.state_dict()
        dyn_state["flow_projector"] = self.wm.flow_projector.state_dict()
        dyn_state["online_ablation"] = "gretz_online_wm_dreamer_actor_critic"
        torch.save(dyn_state, dynamics_path)
        return projector_path, dynamics_path

    def save_checkpoint(self, episode: int, metrics: Optional[Dict[str, Any]] = None) -> Path:
        projector_path, dynamics_path = self._save_world_model_files(int(episode))
        path = self.output_dir / "checkpoints" / f"checkpoint_ep{int(episode):05d}.pt"
        save_policy_checkpoint(
            path=path,
            actor=self.actor,
            critic=self.critic,
            cfg=self.cfg,
            episode=int(episode),
            obs_dim=int(self.obs_dim),
            metrics=metrics,
        )
        state = torch.load(path, map_location="cpu", weights_only=False)
        state.update(
            {
                "ablation": "gretz_online_wm_dreamer_actor_critic",
                "flow_projector": self.wm.flow_projector.state_dict(),
                "dynamics": self.wm.dynamics.state_dict(),
                "wm_projector_ckpt": str(projector_path),
                "wm_dynamics_ckpt": str(dynamics_path),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "global_critic": self.global_critic.state_dict(),
                "global_critic_optimizer": self.global_critic_optimizer.state_dict(),
                "wm_optimizer": self.wm_updater.optimizer.state_dict(),
                "hdv_response_predictor": None
                if self.hdv_response_model is None
                else self.hdv_response_model.model.state_dict(),
                "hdv_response_optimizer": None
                if self.hdv_response_model is None
                else self.hdv_response_model.optimizer.state_dict(),
                "hdv_response_trained_pairs": 0
                if self.hdv_response_model is None
                else int(self.hdv_response_model.trained_pairs),
            }
        )
        cfg_dict = dict(state.get("cfg", {}))
        cfg_dict.update(
            {
                "ablation": "gretz_online_wm_dreamer_actor_critic",
                "wm_projector_ckpt": str(projector_path),
                "wm_dynamics_ckpt": str(dynamics_path),
                "world_model_online_update": True,
                "direct_real_a2c_update": bool(self.cfg.direct_real_a2c_update),
                "policy_obs_mode": str(self.policy_obs_mode),
                "dreamer_route_cost_reward_mode": str(self.cfg.reward_mode),
                "dreamer_route_cost_reward_temperature": float(self.cfg.reward_temperature),
                "dreamer_route_cost_reward_progress_state": "per_vehicle_route_suffix_until_arrival",
                "dreamer_vehicle_return_target": "lambda_return_with_next_state_bootstrap",
                "dreamer_vehicle_return_lambda": float(self.cfg.ac_lambda),
                "dreamer_global_critic_enabled": True,
                "dreamer_global_advantage_weight": float(self.cfg.ac_global_advantage_weight),
                "dreamer_actor_advantage": "vehicle_plus_global_lambda_advantage",
                "imagined_hdv_context_source": str(self.cfg.imagined_hdv_context_source),
                "hdv_response_predictor_enabled": bool(self.cfg.hdv_response_predictor_enabled),
                "hdv_response_predictor_fallback": str(self.cfg.hdv_response_predictor_fallback),
                "hdv_response_predictor_ckpt": str(self.cfg.hdv_response_predictor_ckpt),
                "hdv_response_predictor_loaded_ckpt": str(self.hdv_response_predictor_ckpt_loaded),
            }
        )
        state["cfg"] = cfg_dict
        torch.save(state, path)
        return path

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
            real_a2c_replay: List[List[OnlineDecisionSample]] = []
            real_a2c_replay_capacity = max(int(self.cfg.replay_buffer_episodes), 1)
            real_a2c_replay_max_age = max(int(self.cfg.replay_actor_max_age_episodes), 1)
            for ep in range(start_episode + 1, int(self.cfg.train_episodes) + 1):
                samples, row, wm_steps = collector.run_train_episode(ep, int(self.rollout_max_steps))
                trace_path = self._save_hdv_pressure_trace(ep, wm_steps)
                if trace_path is not None:
                    row["hdv_pressure_trace"] = str(trace_path)
                real_a2c_replay.append(list(samples))
                if len(real_a2c_replay) > real_a2c_replay_capacity:
                    real_a2c_replay = real_a2c_replay[-real_a2c_replay_capacity:]
                real_a2c_min_episode = int(ep) - int(real_a2c_replay_max_age) + 1
                real_a2c_batches = [
                    batch
                    for batch in real_a2c_replay
                    if batch and int(batch[0].episode) >= real_a2c_min_episode
                ]
                if not real_a2c_batches:
                    real_a2c_batches = [list(samples)]
                real_a2c_samples = [
                    sample
                    for batch in real_a2c_batches
                    for sample in batch
                ]
                row["wm_replay_accepted"] = int(self._episode_allowed_for_wm_replay(row))
                if int(row["wm_replay_accepted"]):
                    hdv_pred, hdv_pred_source, hdv_pred_metrics = self._predict_hdv_response_for_episode(ep, wm_steps)
                    row.update(hdv_pred_metrics)
                    self.replay.append_episode(
                        wm_steps,
                        predicted_hdv_edge_demand=hdv_pred,
                        prediction_source=hdv_pred_source,
                    )
                    row["wm_replay_skip_reason"] = ""
                else:
                    row.update(
                        {
                            "hdv_response_pred_enabled": float(bool(self.cfg.hdv_response_predictor_enabled)),
                            "hdv_response_pred_source_code": 0.0,
                            "hdv_response_pred_tv": 0.0,
                        }
                    )
                    row["wm_replay_skip_reason"] = "incomplete_episode"
                per_pass_updates: List[Dict[str, float]] = []
                if int(row["wm_replay_accepted"]):
                    hdv_response_update = self._update_hdv_response_predictor()
                    hdv_response_update["update_phase"] = 0.25
                    hdv_response_update["update_pass"] = 0.0
                    per_pass_updates.append(hdv_response_update)
                real_a2c_update = self._direct_real_a2c_update(
                    real_a2c_samples,
                    replay_episodes=len(real_a2c_batches),
                )
                real_a2c_update["update_phase"] = 0.5
                real_a2c_update["update_pass"] = 0.0
                per_pass_updates.append(real_a2c_update)
                n_wm_updates = max(int(self.cfg.wm_updates_per_collect), 0)
                if self._wm_update_due(ep) and n_wm_updates > 0:
                    for pass_idx in range(n_wm_updates):
                        wm_update = self.wm_updater.update(self.replay, self.rng)
                        wm_update["update_phase"] = 0.0
                        wm_update["update_pass"] = float(pass_idx + 1)
                        per_pass_updates.append(wm_update)
                else:
                    wm_update = self._skipped_wm_update(
                        reason="disabled" if n_wm_updates <= 0 else "interval"
                    )
                    wm_update["update_phase"] = 0.0
                    wm_update["update_pass"] = 0.0
                    per_pass_updates.append(wm_update)
                pred_diag: Optional[Dict[str, float]] = None
                diag_every = max(int(self.cfg.wm_prediction_diagnostic_every_episodes), 0)
                if diag_every > 0 and int(ep) % diag_every == 0:
                    pred_diag = self.imagined_ac.diagnose_prediction(self.replay, self.rng)
                    pred_diag["update_phase"] = 2.0
                    pred_diag["update_pass"] = 0.0
                    per_pass_updates.append(pred_diag)

                target_actor_passes, actor_schedule_reason = self._actor_update_passes_for_episode(ep)
                actor_passes, actor_gate_reason = self._actor_gate_passes(
                    target_passes=int(target_actor_passes),
                    diag=pred_diag,
                )
                actor_control_row: Dict[str, float] = {
                    "actor_update_target_passes": float(target_actor_passes),
                    "actor_update_effective_passes": float(actor_passes),
                    "actor_update_schedule_reason": actor_schedule_reason,
                    "actor_update_gate_reason": actor_gate_reason,
                    "update_phase": 1.5,
                    "update_pass": 0.0,
                }
                per_pass_updates.append(actor_control_row)

                if self.replay.episode_count() >= max(int(self.cfg.min_replay_episodes_before_actor), 1) and actor_passes > 0:
                    for pass_idx in range(int(actor_passes)):
                        imag_update = self.imagined_ac.update(self.replay, self.rng)
                        imag_update["update_phase"] = 1.0
                        imag_update["update_pass"] = float(pass_idx + 1)
                        per_pass_updates.append(imag_update)
                else:
                    skip_reason = (
                        "min_replay_episodes"
                        if self.replay.episode_count() < max(int(self.cfg.min_replay_episodes_before_actor), 1)
                        else "actor_gate"
                    )
                    per_pass_updates.append(
                        {
                            "imag_loss": 0.0,
                            "imag_n_samples": 0.0,
                            "imag_skipped": 1.0,
                            "imag_skip_reason": skip_reason,
                            "update_phase": 1.0,
                            "update_pass": 0.0,
                        }
                    )
                update = _mean_update_stats(per_pass_updates)
                row.update({f"update_{k}": v for k, v in update.items()})
                train_rows.append(row)
                for update_idx, update_row in enumerate(per_pass_updates, 1):
                    update_rows.append({"episode": ep, "update_idx": int(update_idx), **update_row})
                print(
                    f"[online-wm-dreamer] ep={ep:04d}/{int(self.cfg.train_episodes):04d} "
                    f"samples={row.get('samples', 0)} wm_trans={row.get('wm_transitions', 0)} "
                    f"reward_mean={row.get('reward_mean', float('nan')):.4f} "
                    f"t_all_arrived={row.get('t_all', float('nan')):.3f} "
                    f"t_CAV={row.get('t_CAV', float('nan')):.3f} "
                    f"t_HDV={row.get('t_HDV', float('nan')):.3f} "
                    f"completion={row.get('completion_rate', float('nan')):.6f} "
                    f"unfinished={int(row.get('unfinished_all', 0))} "
                    f"wm_replay={int(row.get('wm_replay_accepted', 0))} "
                    f"replay_eps={self.replay.episode_count()} "
                    f"wm_skip={str(update.get('wm_skip_reason', ''))} "
                    f"wm_loss={update.get('wm_loss', 0.0):.4f} "
                    f"hdv_pred_tv={row.get('hdv_response_pred_tv', 0.0):.4f} "
                    f"hdv_pairs={int(update.get('hdv_response_pairs', 0.0))} "
                    f"real_ac={int(update.get('real_a2c_enabled', 0.0))}:{int(update.get('real_a2c_skipped', 1.0))} "
                    f"real_g={update.get('real_a2c_actor_grad_norm', 0.0):.4f} "
                    f"real_adv={update.get('real_a2c_adv_std', 0.0):.4f} "
                    f"real_ent={update.get('real_a2c_entropy', 0.0):.4f} "
                    f"actor_passes={int(actor_passes)}/{int(target_actor_passes)} "
                    f"gate={actor_gate_reason} "
                    f"imag_loss={update.get('imag_loss', 0.0):.4f} "
                    f"imag_reward={update.get('imag_reward_mean', 0.0):.4f} "
                    f"imag_return={update.get('imag_return_mean', 0.0):.4f} "
                    f"global_reward={update.get('imag_global_reward_mean', 0.0):.4f} "
                    f"global_adv={update.get('imag_global_adv_mean', 0.0):.4f} "
                    f"imag_arrived={update.get('imag_arrived_frac', 0.0):.3f} "
                    f"entropy={update.get('imag_entropy', 0.0):.4f} "
                    f"clip={update.get('imag_clip_frac', 0.0):.3f} "
                    f"adv_std={update.get('imag_adv_std', 0.0):.4f} "
                    f"actor_g={update.get('imag_actor_grad_norm', 0.0):.4f} "
                    f"value_ev={update.get('imag_value_target_ev', 0.0):.4f} "
                    f"global_ev={update.get('imag_global_value_target_ev', 0.0):.4f}",
                    flush=True,
                )
                if ep in ckpt_eps:
                    ckpt = self.save_checkpoint(ep, metrics=row)
                    checkpoint_paths[int(ep)] = str(ckpt)
                    print(f"[online-wm-dreamer] saved checkpoint: {ckpt}", flush=True)
        finally:
            try:
                collector.finalize_protocol_xml_aliases()
            finally:
                close = getattr(getattr(collector, "connector", None), "close", None)
                if callable(close):
                    close()
        pd.DataFrame(train_rows).to_csv(self.output_dir / "online_wm_train_episodes.csv", index=False)
        pd.DataFrame(update_rows).to_csv(self.output_dir / "online_wm_train_updates.csv", index=False)
        decision_df = pd.DataFrame(getattr(collector, "_dynamic_decision_rows", []))
        if len(decision_df) > 0:
            decision_df.to_csv(self.output_dir / "online_wm_train_decisions.csv", index=False)
        agent_df = pd.DataFrame(getattr(collector, "_dynamic_agent_decision_rows", []))
        if len(agent_df) > 0:
            agent_df.to_csv(self.output_dir / "online_wm_train_agent_decisions.csv", index=False)
        latest_hdv_state = _merge_hdv_learning_state(
            getattr(collector, "_loaded_hdv_state_eval", None),
            _snapshot_hdv_learning_state(collector.connector.env) if hasattr(collector.connector, "env") else {},
            route_state=serialize_hdv_route_state(
                getattr(collector, "_hdv_route_probs", None),
                getattr(collector, "_hdv_route_costs", None),
            ),
        )
        final_hdv_state = self.output_dir / "hdv_learning_state_after_online_wm_train.json"
        _save_hdv_learning_state(final_hdv_state, latest_hdv_state)
        summary = {
            "ablation": "gretz_online_wm_dreamer_actor_critic",
            "config": _jsonable_config(self.cfg),
            "obs_dim": int(self.obs_dim),
            "checkpoint_paths": checkpoint_paths,
            "train_episode_csv": str(self.output_dir / "online_wm_train_episodes.csv"),
            "train_update_csv": str(self.output_dir / "online_wm_train_updates.csv"),
            "final_hdv_state": str(final_hdv_state),
            "hdv_response_predictor_loaded_ckpt": str(self.hdv_response_predictor_ckpt_loaded),
            "elapsed_seconds": float(time.time() - start),
        }
        with (self.output_dir / "online_wm_train_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


def build_eval_command(
    *,
    cfg: OnlineWmDreamerConfig,
    checkpoint_path: str | Path,
    results_dir: str | Path,
) -> List[str]:
    ckpt_path = Path(checkpoint_path)
    projector_path = cfg.projector_ckpt
    dynamics_path = cfg.dynamics_ckpt
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        projector_path = str(state.get("wm_projector_ckpt", projector_path))
        dynamics_path = str(state.get("wm_dynamics_ckpt", dynamics_path))
    cmd = build_no_cfm_eval_command(cfg=cfg, checkpoint_path=checkpoint_path, results_dir=results_dir)
    route_selection_flag = "--av_policy_route_selection"
    if route_selection_flag in cmd:
        route_selection_idx = cmd.index(route_selection_flag)
        if route_selection_idx + 1 < len(cmd):
            cmd[route_selection_idx + 1] = str(cfg.eval_policy_route_selection)
    for flag, value in (("--projector_ckpt", projector_path), ("--dynamics_ckpt", dynamics_path)):
        if flag in cmd:
            idx = cmd.index(flag)
            if idx + 1 < len(cmd):
                cmd[idx + 1] = str(value)
    return cmd


def validate_required_paths(plan: Dict[str, Any]) -> None:
    missing = list(plan.get("missing", []))
    if missing:
        details = {name: plan["paths"].get(name) for name in missing}
        raise FileNotFoundError("Missing required Gretz online-WM ablation artifacts: " + json.dumps(details, indent=2))
