"""Online Gretz Dreamer-style world-model actor-critic ablation.

This ablation differs from ``online_no_cfm_actor_critic`` in two ways:

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

from ablation.online_no_cfm_actor_critic import (  # noqa: E402
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
    build_diverse_paths_current_to_dest,
    build_eval_command as build_no_cfm_eval_command,
    route_valid_mask_from_path_batch,
    save_policy_checkpoint,
)

try:
    from dreamer4.urban_routing.dreamer_actor_critic import (
        CentralizedVehicleValueCritic,
        VehicleRouteActor,
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
    from dreamer_actor_critic import CentralizedVehicleValueCritic, VehicleRouteActor
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
from scripts.eval_sumo_routing import _build_edge_adj_list  # noqa: E402
from ablation.traci_port_fallback import patch_traci_start_port_fallback  # noqa: E402
from dreamer4.urban_routing.data_collector import serialize_hdv_route_state  # noqa: E402


DEFAULT_ONLINE_WM_CHECKPOINT_EPISODES = (25, 50, 100)


@dataclass
class OnlineWmDreamerConfig(OnlineNoCfmConfig):
    """Configuration for online world-model Dreamer-style Gretz ablation."""

    output_dir: str = str(_REPO_ROOT / "results" / GRETZ_TAG / "ablation_online_wm_dreamer_ac")
    checkpoint_episodes: Tuple[int, ...] = DEFAULT_ONLINE_WM_CHECKPOINT_EPISODES
    reward_mode: str = "capped_step_cost"
    return_horizon_steps: int = 0
    wm_replay_buffer_episodes: int = 8
    wm_update_epochs: int = 1
    wm_update_every_episodes: int = 1
    wm_replay_require_completion: bool = True
    wm_replay_completion_threshold: float = 1.0
    wm_batch_size: int = 16
    wm_lr: float = 1e-5
    wm_grad_clip: float = 10.0
    wm_transition_decoded_supervision_weight: float = 1.0
    wm_hdv_policy_weight: float = 0.05
    wm_hdv_policy_target_mode: str = "episode_mean"
    wm_hdv_policy_action_mode: str = "window_mean"
    wm_transition_hdv_context_mode: str = "episode_response"
    wm_transition_hdv_context_weight: float = 1.0
    wm_transition_hdv_context_scale: float = 1.0
    wm_transition_hdv_action_mode: str = "none"
    wm_transition_hdv_action_weight: float = 0.0
    wm_transition_hdv_action_scale: float = 1.0
    wm_cfm_active_weight: float = 0.0
    wm_cfm_busy_weight: float = 0.0
    wm_cfm_busy_threshold: float = 0.05
    wm_cfm_active_threshold: float = 0.0
    wm_cfm_max_weight: float = 50.0
    imag_horizon: int = 8
    imagined_updates_per_episode: int = 1
    imagined_rollouts_per_update: int = 1
    imagined_max_start_samples: int = 1
    direct_real_a2c_update: bool = False

    def resolve(self) -> "OnlineWmDreamerConfig":
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
        return OnlineWmDreamerConfig(**values)

    @classmethod
    def from_settings(cls, settings_path: str, **overrides: Any) -> "OnlineWmDreamerConfig":
        base = OnlineNoCfmConfig.from_settings(settings_path, **overrides)
        values = asdict(base)
        own_defaults = asdict(cls())
        for key in (
            "output_dir",
            "checkpoint_episodes",
            "reward_mode",
            "return_horizon_steps",
            "wm_replay_buffer_episodes",
            "wm_update_epochs",
            "wm_update_every_episodes",
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
            "wm_cfm_active_weight",
            "wm_cfm_busy_weight",
            "wm_cfm_busy_threshold",
            "wm_cfm_active_threshold",
            "wm_cfm_max_weight",
            "imag_horizon",
            "imagined_updates_per_episode",
            "imagined_rollouts_per_update",
            "imagined_max_start_samples",
            "direct_real_a2c_update",
        ):
            if key not in overrides:
                values[key] = own_defaults[key]
        values.update({k: v for k, v in overrides.items() if v is not None})
        settings = load_settings_file(settings_path)
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
    samples: List[OnlineDecisionSample] = field(default_factory=list)


class OnlineWorldModelReplay:
    def __init__(self, capacity_episodes: int):
        self.capacity_episodes = max(int(capacity_episodes), 1)
        self._episodes: List[List[OnlineWorldModelStep]] = []

    def append_episode(self, steps: Sequence[OnlineWorldModelStep]) -> None:
        if not steps:
            return
        episode_steps = list(steps)
        hdv_stack = _stack_np([s.hdv_action for s in episode_steps])
        if hdv_stack.size > 0:
            hdv_mean = hdv_stack.clip(min=0.0).mean(axis=0).astype(np.float32)
            denom = float(hdv_mean.sum())
            if denom > 1e-8:
                hdv_mean = hdv_mean / denom
            for step in episode_steps:
                step.hdv_target = hdv_mean.copy()
        self._episodes.append(episode_steps)
        if len(self._episodes) > self.capacity_episodes:
            self._episodes = self._episodes[-self.capacity_episodes :]

    @property
    def episodes(self) -> List[List[OnlineWorldModelStep]]:
        return list(self._episodes)

    def all_steps(self) -> List[OnlineWorldModelStep]:
        return [step for ep in self._episodes for step in ep]

    def sample_steps(self, rng: np.random.Generator, batch_size: int) -> List[OnlineWorldModelStep]:
        steps = self.all_steps()
        if not steps:
            return []
        n = min(max(int(batch_size), 1), len(steps))
        idx = rng.choice(len(steps), size=n, replace=len(steps) < n)
        return [steps[int(i)] for i in np.asarray(idx).reshape(-1)]


@dataclass
class ImaginedVehicleRouteState:
    candidate_paths: List[List[int]]
    current_edge: int
    dest_edge: int
    current_edge_remaining: float
    arrived: bool = False


@dataclass
class ImaginedRouteAdvance:
    elapsed_cost: float
    current_edge: int
    current_edge_remaining: float
    arrived: bool
    remaining_path: List[int]


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

        if self._use_routerl_hdv_blackbox and hasattr(self.connector, "apply_routerl_hdv_learning"):
            self.connector.apply_routerl_hdv_learning(veh_travel_times)

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
        }
        for key, value in sorted(getattr(self, "_av_decision_skip_stats", {}).items()):
            row[f"av_diag_{key}"] = int(value)
        return list(self._episode_samples), row

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
        return samples, row, list(getattr(self, "_wm_episode_steps", []))


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


class OnlineImaginedActorCritic:
    def __init__(
        self,
        *,
        wm,
        actor: VehicleRouteActor,
        critic: CentralizedVehicleValueCritic,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        cfg: OnlineWmDreamerConfig,
        device: torch.device,
        obs_dim: int,
    ):
        self.wm = wm
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.cfg = cfg
        self.device = device
        self.obs_dim = int(obs_dim)
        registry = getattr(wm, "_registry", None)
        self.adj_list = _build_edge_adj_list(
            getattr(registry, "adjacency", None),
            getattr(registry, "passenger_allowed", None),
        )

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
        vehicle_id_steps: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        action_edges: List[torch.Tensor] = []
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
            safe_valid = valid.clone()
            has_valid = safe_valid.any(dim=-1)
            safe_valid[~has_valid, 0] = True
            masked_logits = (logits / float(max(self.cfg.av_policy_sample_temperature, 1e-6))).masked_fill(~safe_valid, -1e9)
            dist = Categorical(logits=masked_logits)
            selected = dist.sample()
            old_logp = dist.log_prob(selected).detach()
            entropy = dist.entropy().mean()
            reward_values: List[float] = []
            for local_i, state_i in enumerate(active_indices):
                vehicle = vehicle_states[int(state_i)]
                action_idx = int(selected[local_i].detach().cpu().item())
                action_idx = int(np.clip(action_idx, 0, n_routes - 1))
                chosen_paths = active_path_batch[local_i]
                chosen = chosen_paths[action_idx] if action_idx < len(chosen_paths) else []
                advance = _advance_imagined_route_progress(
                    path=chosen,
                    edge_tt=edge_tt,
                    current_edge=int(vehicle.current_edge),
                    current_edge_remaining=float(vehicle.current_edge_remaining),
                    step_interval=float(self.cfg.step_interval),
                )
                vehicle.current_edge = int(advance.current_edge)
                vehicle.current_edge_remaining = float(advance.current_edge_remaining)
                vehicle.arrived = bool(advance.arrived)
                reward_values.append(
                    -float(self.cfg.reward_weight)
                    * float(self.cfg.reward_scale)
                    * float(advance.elapsed_cost)
                    / float(max(self.cfg.reward_temperature, 1e-6))
                )
                if bool(advance.arrived):
                    arrived_events += 1
            rewards = torch.tensor(reward_values, device=self.device, dtype=obs.dtype)
            probs = torch.softmax(masked_logits, dim=-1)
            action_edge = aggregate_route_probs_to_action_edge(
                probs,
                active_path_batch,
                n_edges=n_edges,
                n_routes=n_routes,
                current_edges=torch.tensor(active_current_edges, device=self.device, dtype=torch.long),
                edge_travel_times=torch.from_numpy(edge_tt).to(device=self.device, dtype=torch.float32),
                horizon_seconds=float(self.cfg.step_interval),
            )
            depart_t = torch.from_numpy(np.asarray(start.depart, dtype=np.float32)).to(device=self.device, dtype=z.dtype)
            context_t = torch.from_numpy(np.asarray(start.context, dtype=np.float32)).to(device=self.device, dtype=z.dtype)
            hdv_t = torch.from_numpy(np.asarray(start.hdv_action, dtype=np.float32)).to(device=self.device, dtype=z.dtype)
            past_packed = torch.stack(latent_history[-1:], dim=0).unsqueeze(0)
            with torch.no_grad():
                z_next, _info = self.wm.predict_next_state_differentiable(
                    z.detach(),
                    action_edge.detach(),
                    past_packed=past_packed.detach(),
                    depart=depart_t,
                    context=context_t,
                    hdv_context=hdv_t if bool(getattr(self.wm, "use_hdv_context", False)) else None,
                )
            obs_steps.append(obs.detach())
            action_steps.append(selected.detach())
            old_logp_steps.append(old_logp.detach())
            valid_steps.append(valid.detach())
            reward_steps.append(rewards.detach())
            group_steps.append(torch.full((int(obs.shape[0]),), h, device=self.device, dtype=torch.long))
            vehicle_id_steps.append(torch.tensor(active_indices, device=self.device, dtype=torch.long))
            entropies.append(entropy.detach())
            action_edges.append(action_edge.detach())
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
            "vehicle_index": vehicle_ids_flat,
            "entropy_rollout": torch.stack(entropies).mean() if entropies else torch.zeros((), device=self.device),
            "action_edges": action_edges,
            "reward_mean": float(rewards_flat.mean().detach().cpu().item()),
            "return_mean": float(returns_flat.mean().detach().cpu().item()),
            "arrived_frac": float(min(max(arrived_frac, 0.0), 1.0)),
            "n_samples": int(torch.cat(action_steps).numel()),
        }

    def update(self, replay: OnlineWorldModelReplay, rng: np.random.Generator) -> Dict[str, float]:
        steps = replay.all_steps()
        candidate_steps = [s for s in steps if s.samples]
        if not candidate_steps:
            return {"imag_loss": 0.0, "imag_n_samples": 0.0, "imag_skipped": 1.0}
        self.actor.train()
        self.critic.train()
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
            returns = torch.cat(
                [
                    p["returns"] if "returns" in p else p["rewards"]
                    for p in rollout_parts
                    if "rewards" in p
                ],
                dim=0,
            )
            valid = torch.cat([p["valid_mask"] for p in rollout_parts if "valid_mask" in p], dim=0)
            old_logp = torch.cat([p["old_log_probs"] for p in rollout_parts if "old_log_probs" in p], dim=0)
            group_parts: List[torch.Tensor] = []
            group_offset = 0
            for p in rollout_parts:
                group_i = p.get("group_index")
                if torch.is_tensor(group_i) and int(group_i.numel()) > 0:
                    group_parts.append(group_i.to(device=self.device, dtype=torch.long) + int(group_offset))
                    group_offset += int(group_i.max().detach().cpu().item()) + 1
            group = (
                torch.cat(group_parts, dim=0)
                if group_parts and int(torch.cat(group_parts, dim=0).numel()) == int(obs.shape[0])
                else torch.arange(int(obs.shape[0]), device=self.device, dtype=torch.long)
            )
            logits = self.actor.forward_logits(obs)
            safe_valid = valid.clone()
            has_valid = safe_valid.any(dim=-1)
            safe_valid[~has_valid, 0] = True
            masked_logits = (logits / float(max(self.cfg.av_policy_sample_temperature, 1e-6))).masked_fill(~safe_valid, -1e9)
            dist = Categorical(logits=masked_logits)
            logp = dist.log_prob(actions)
            entropy = dist.entropy()
            values = self.critic(obs, group_index=group).reshape(-1)
            adv = returns - values.detach()
            if bool(self.cfg.ac_normalize_advantages) and int(adv.numel()) > 1:
                adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
            if float(self.cfg.ac_advantage_clip) > 0:
                adv = adv.clamp(-float(self.cfg.ac_advantage_clip), float(self.cfg.ac_advantage_clip))
            active = has_valid & torch.isfinite(returns) & torch.isfinite(old_logp)
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
            loss = policy_loss + float(self.cfg.value_coef) * value_loss - float(self.cfg.entropy_coef) * entropy_mean
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.cfg.max_grad_norm))
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.cfg.max_grad_norm))
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            rows.append(
                {
                    "imag_loss": float(loss.detach().cpu().item()),
                    "imag_policy_loss": float(policy_loss.detach().cpu().item()),
                    "imag_value_loss": float(value_loss.detach().cpu().item()),
                    "imag_entropy": float(entropy_mean.detach().cpu().item()),
                    "imag_clip_frac": float(clip_frac.detach().cpu().item()),
                    "imag_adv_std": float((returns - values.detach()).std(unbiased=False).detach().cpu().item())
                    if int(returns.numel()) > 1
                    else 0.0,
                    "imag_reward_mean": float(rewards.detach().mean().cpu().item()),
                    "imag_return_mean": float(returns.detach().mean().cpu().item()),
                    "imag_arrived_frac": float(
                        np.mean([float(p.get("arrived_frac", 0.0)) for p in rollout_parts])
                    ),
                    "imag_n_samples": float(int(obs.shape[0])),
                    "imag_skipped": 0.0,
                }
            )
        if not rows:
            return {"imag_loss": 0.0, "imag_n_samples": 0.0, "imag_skipped": 1.0}
        return {k: float(np.mean([row.get(k, 0.0) for row in rows])) for k in rows[-1].keys()}


def build_gretz_online_wm_dreamer_plan(cfg: Optional[OnlineWmDreamerConfig] = None) -> Dict[str, Any]:
    resolved = (cfg or OnlineWmDreamerConfig()).resolve()
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
    missing = [name for name, value in required_paths.items() if value and not Path(str(value)).exists()]
    return {
        "ablation": "gretz_online_wm_dreamer_actor_critic",
        "network": resolved.network_name,
        "tag": resolved.tag,
        "reward_semantics": {
            "scope": "imagined_vehicle_critic",
            "mode": resolved.reward_mode,
            "temperature": float(resolved.reward_temperature),
            "formula": "-min(step_interval, current_imagined_route_remaining_tt) / reward_temperature",
            "source": "world_model_decoded_edge_travel_time",
            "progress_state": "per_vehicle_route_suffix_until_arrival",
        },
        "world_model_usage": {
            "encoder_for_actor_obs": True,
            "cfm_transition_in_training": True,
            "world_model_online_update": True,
            "updated_modules": ["flow_projector", "cfm_dynamics"],
            "direct_real_a2c_update": bool(resolved.direct_real_a2c_update),
            "wm_grad_enabled": True,
        },
        "training": {
            "algorithm": "online_dreamer_world_model_actor_critic",
            "actor_from_scratch": not bool(resolved.resume_checkpoint),
            "episodes": int(resolved.train_episodes),
            "wm_replay_buffer_episodes": int(resolved.wm_replay_buffer_episodes),
            "wm_update_epochs": int(resolved.wm_update_epochs),
            "wm_update_every_episodes": int(resolved.wm_update_every_episodes),
            "wm_replay_require_completion": bool(resolved.wm_replay_require_completion),
            "wm_replay_completion_threshold": float(resolved.wm_replay_completion_threshold),
            "wm_batch_size": int(resolved.wm_batch_size),
            "wm_hdv_policy_target_mode": str(resolved.wm_hdv_policy_target_mode),
            "wm_hdv_policy_action_mode": str(resolved.wm_hdv_policy_action_mode),
            "wm_transition_hdv_context_mode": str(resolved.wm_transition_hdv_context_mode),
            "wm_transition_hdv_context_weight": float(resolved.wm_transition_hdv_context_weight),
            "wm_transition_hdv_context_scale": float(resolved.wm_transition_hdv_context_scale),
            "wm_transition_hdv_action_mode": str(resolved.wm_transition_hdv_action_mode),
            "wm_transition_hdv_action_weight": float(resolved.wm_transition_hdv_action_weight),
            "wm_transition_hdv_action_scale": float(resolved.wm_transition_hdv_action_scale),
            "imag_horizon": int(resolved.imag_horizon),
            "imagined_updates_per_episode": int(resolved.imagined_updates_per_episode),
            "imagined_rollouts_per_update": int(resolved.imagined_rollouts_per_update),
            "checkpoint_paths": checkpoints,
        },
        "evaluation": {
            "episodes_per_checkpoint": int(resolved.eval_episodes),
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
        self.wm_updater = OnlineWorldModelUpdater(self.wm, self.cfg, self.device)
        self.imagined_ac = OnlineImaginedActorCritic(
            wm=self.wm,
            actor=self.actor,
            critic=self.critic,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            cfg=self.cfg,
            device=self.device,
            obs_dim=int(self.obs_dim),
        )
        self.replay = OnlineWorldModelReplay(int(self.cfg.wm_replay_buffer_episodes))
        self.collector: Optional[OnlineWmDreamerCollector] = None
        self._dynamics_checkpoint_state = torch.load(str(self.cfg.dynamics_ckpt), map_location="cpu", weights_only=False)
        self.loaded_resume_episode = self._load_resume_checkpoint()

    def _episode_allowed_for_wm_replay(self, row: Dict[str, Any]) -> bool:
        if not bool(self.cfg.wm_replay_require_completion):
            return True
        threshold = float(self.cfg.wm_replay_completion_threshold)
        completion = float(row.get("completion_rate", 0.0) or 0.0)
        return completion >= threshold - 1e-12

    def _wm_update_due(self, episode: int) -> bool:
        every = max(int(self.cfg.wm_update_every_episodes), 1)
        return int(episode) % every == 0

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
            self.critic.load_state_dict(critic_state)
        if "flow_projector" in state:
            self.wm.flow_projector.load_state_dict(state["flow_projector"], strict=False)
        if "dynamics" in state:
            self.wm.dynamics.load_state_dict(state["dynamics"], strict=True)
        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "wm_optimizer" in state:
            self.wm_updater.optimizer.load_state_dict(state["wm_optimizer"])
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
        collector = OnlineWmDreamerCollector(
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
                "wm_optimizer": self.wm_updater.optimizer.state_dict(),
            }
        )
        cfg_dict = dict(state.get("cfg", {}))
        cfg_dict.update(
            {
                "ablation": "gretz_online_wm_dreamer_actor_critic",
                "wm_projector_ckpt": str(projector_path),
                "wm_dynamics_ckpt": str(dynamics_path),
                "world_model_online_update": True,
                "direct_real_a2c_update": False,
                "dreamer_route_cost_reward_mode": str(self.cfg.reward_mode),
                "dreamer_route_cost_reward_temperature": float(self.cfg.reward_temperature),
                "dreamer_route_cost_reward_progress_state": "per_vehicle_route_suffix_until_arrival",
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
            for ep in range(start_episode + 1, int(self.cfg.train_episodes) + 1):
                _samples, row, wm_steps = collector.run_train_episode(ep, int(self.rollout_max_steps))
                row["wm_replay_accepted"] = int(self._episode_allowed_for_wm_replay(row))
                if int(row["wm_replay_accepted"]):
                    self.replay.append_episode(wm_steps)
                    row["wm_replay_skip_reason"] = ""
                else:
                    row["wm_replay_skip_reason"] = "incomplete_episode"
                if self._wm_update_due(ep):
                    wm_update = self.wm_updater.update(self.replay, self.rng)
                else:
                    wm_update = self._skipped_wm_update(reason="interval")
                imag_update = self.imagined_ac.update(self.replay, self.rng)
                update = {**wm_update, **imag_update}
                row.update({f"update_{k}": v for k, v in update.items()})
                train_rows.append(row)
                update_rows.append({"episode": ep, **update})
                print(
                    f"[online-wm-dreamer] ep={ep:04d}/{int(self.cfg.train_episodes):04d} "
                    f"samples={row.get('samples', 0)} wm_trans={row.get('wm_transitions', 0)} "
                    f"reward_mean={row.get('reward_mean', float('nan')):.4f} "
                    f"t_all_arrived={row.get('t_all', float('nan')):.3f} "
                    f"completion={row.get('completion_rate', float('nan')):.6f} "
                    f"unfinished={int(row.get('unfinished_all', 0))} "
                    f"wm_replay={int(row.get('wm_replay_accepted', 0))} "
                    f"wm_skip={str(update.get('wm_skip_reason', ''))} "
                    f"wm_loss={update.get('wm_loss', 0.0):.4f} "
                    f"imag_loss={update.get('imag_loss', 0.0):.4f} "
                    f"imag_reward={update.get('imag_reward_mean', 0.0):.4f} "
                    f"imag_return={update.get('imag_return_mean', 0.0):.4f} "
                    f"imag_arrived={update.get('imag_arrived_frac', 0.0):.3f} "
                    f"entropy={update.get('imag_entropy', 0.0):.4f} "
                    f"clip={update.get('imag_clip_frac', 0.0):.3f} "
                    f"adv_std={update.get('imag_adv_std', 0.0):.4f}",
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
        _save_hdv_learning_state(self.output_dir / "hdv_learning_state_after_online_wm_train.json", latest_hdv_state)
        summary = {
            "ablation": "gretz_online_wm_dreamer_actor_critic",
            "config": _jsonable_config(self.cfg),
            "obs_dim": int(self.obs_dim),
            "checkpoint_paths": checkpoint_paths,
            "train_episode_csv": str(self.output_dir / "online_wm_train_episodes.csv"),
            "train_update_csv": str(self.output_dir / "online_wm_train_updates.csv"),
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
