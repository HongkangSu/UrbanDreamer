#!/usr/bin/env python3
"""Train the MAMBA model-based MARL baseline on fixed-role SUMO traffic.

The SUMO protocol intentionally mirrors the other baselines in this directory.
Only the policy/learner is different: every routing boundary advances one joint
RSSM prior, then corrects and acts for the vehicles that have a valid routing
observation.  Replay is stored as sparse fixed-identity frames by
``baseline.mamba``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import sysconfig
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MAMBA_CHECKPOINT_SCHEMA_VERSION = 3
COLLECTOR_PROTOCOL_STATE_VERSION = 1
for path in (REPO_ROOT, REPO_ROOT / "dreamer4", REPO_ROOT / "dreamer4" / "urban_routing"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _ensure_sumo_runtime_env() -> None:
    """Expose a packaged or system SUMO installation before collector imports."""
    prefix = Path(sys.executable).resolve().parent.parent
    purelib = Path(sysconfig.get_paths().get("purelib", ""))
    homes = [
        Path(os.environ["SUMO_HOME"]).expanduser() if os.environ.get("SUMO_HOME") else None,
        purelib / "sumo" if purelib else None,
        prefix / "share" / "sumo",
        Path("/usr/share/sumo"),
    ]
    sumo_home = next((item.resolve() for item in homes if item and item.is_dir()), None)
    if sumo_home is not None:
        os.environ.setdefault("SUMO_HOME", str(sumo_home))
        tools_dir = sumo_home / "tools"
        if tools_dir.is_dir() and str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
    path_parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    bins = [prefix / "bin", sumo_home / "bin" if sumo_home is not None else None]
    prepend = [
        str(item.resolve())
        for item in bins
        if item is not None and item.is_dir() and str(item.resolve()) not in path_parts
    ]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + path_parts)


_ensure_sumo_runtime_env()

from baseline.create_toy_cross_bottleneck_network import NETWORK_NAME as TOY_NETWORK_NAME
from baseline.create_toy_cross_bottleneck_network import generate as generate_toy_network
from baseline.online_actor_critic import (
    OnlineDecisionSample,
    OnlineNoCfmConfig,
    _append_vehicle_context_features,
    _resolve_path,
    selected_vehicle_route_rewards,
)
from baseline.run_toy_cross_bottleneck_online_ac import (
    ToyOnlineAcCollector,
    _apply_route_valid_mask_to_probs,
    validate_paths,
)
from baseline.traci_port_fallback import patch_traci_start_port_fallback
from dreamer4.urban_routing.data_collector import serialize_hdv_route_state
from dreamer4.urban_routing.flow_projector import EdgeRegistry
from dreamer4.urban_routing.routerl_shared import compute_agent_counts, load_od_data
from dreamer4.urban_routing.train_marl import (
    build_vehicle_fleet_pressure_obs_features,
    build_vehicle_obs_vectorized,
    vehicle_mode_obs_dim,
)
from dreamer4.urban_routing.urb_fair_protocol import derive_fixed_roles, load_post_mutation_state
from scripts.eval_sumo_hdv_state import (
    _load_hdv_learning_state,
    _merge_hdv_learning_state,
    _save_hdv_learning_state,
    _snapshot_hdv_learning_state,
)
from scripts.eval_sumo_io import _reuse_route_artifacts, _reuse_route_artifacts_from_refs
from scripts.eval_sumo_routing import _edge_route_cost_matrix, _route_batch_uniqueness_summary


@dataclass
class ToyMambaConfig(OnlineNoCfmConfig):
    """SUMO protocol plus upstream-compatible MAMBA hyperparameters."""

    train_episodes: int = 500
    checkpoint_episodes: Tuple[int, ...] = (100, 200, 300, 400, 500)
    gamma: float = 0.99
    disable_human_learning_after_mutation: bool = False
    checkpoint_interval: int = 100
    update_every_episodes: int = 1
    replay_episodes: int = 20
    sequence_length: int = 20
    burn_in: int = 5
    sequence_minibatch_size: int = 4
    communication_chunk_size: int = 32
    min_model_replay_frames: int = 100
    min_agent_replay_frames: int = 2500
    max_imagination_starts: int = 16
    imagination_alive_threshold: float = 0.5
    action_information_coef: float = 1.0
    availability_coef: float = 1.0
    availability_threshold: float = 0.5
    model_epochs: int = 4
    agent_epochs: int = 1
    critic_epochs: int = 5
    ppo_epochs: int = 5
    critic_warmup_updates: int = 20
    imagination_reward_scope: str = "individual"
    hidden_dim: int = 256
    model_hidden_dim: int = 256
    embed_dim: int = 256
    deter_dim: int = 256
    stoch_classes: int = 32
    stoch_categories: int = 32
    transformer_layers: int = 3
    attention_heads: int = 8
    imagination_horizon: int = 15
    model_lr: float = 2.0e-4
    actor_lr: float = 2.0e-4
    mamba_critic_lr: float = 3.0e-4
    gae_lambda: float = 0.95
    ppo_clip: float = 0.075
    target_policy_kl: float = 0.01
    critic_huber_delta: float = 1.0
    target_tau: float = 0.05
    entropy_coef: float = 5.0e-3
    amp_dtype: str = "bf16"
    av_execution_mode: str = "dynamic"

    def resolve(self) -> "ToyMambaConfig":
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
        return ToyMambaConfig(**values)

    @classmethod
    def from_settings(cls, settings_path: str, **overrides: Any) -> "ToyMambaConfig":
        base_names = set(OnlineNoCfmConfig.__dataclass_fields__)
        base_overrides = {key: value for key, value in overrides.items() if key in base_names}
        own_overrides = {key: value for key, value in overrides.items() if key not in base_names}
        base = OnlineNoCfmConfig.from_settings(settings_path, **base_overrides).resolve()
        return cls(**{**asdict(cls()), **asdict(base), **own_overrides}).resolve()


def _jsonable_config(cfg: ToyMambaConfig) -> Dict[str, Any]:
    values = asdict(cfg)
    values["checkpoint_episodes"] = list(cfg.checkpoint_episodes)
    values["extra_eval_args"] = list(cfg.extra_eval_args)
    return values


def _checkpoint_episodes(
    *, raw: str, train_episodes: int, checkpoint_interval: int
) -> Tuple[int, ...]:
    text = str(raw or "").strip()
    if text:
        episodes = {
            int(part.strip())
            for part in text.replace(";", ",").split(",")
            if part.strip()
        }
        return tuple(sorted(ep for ep in episodes if ep > 0))
    interval = int(checkpoint_interval)
    if interval <= 0:
        return tuple()
    return tuple(range(interval, int(train_episodes) + 1, interval))


def _resolve_fixed_roles(cfg: ToyMambaConfig) -> Tuple[List[str], List[str], str]:
    raw = str(cfg.post_mutation_state_in or "").strip()
    if raw:
        path = Path(raw).resolve()
        if path.exists():
            state = load_post_mutation_state(path)
            return (
                [str(item) for item in state.get("machine_agent_ids", [])],
                [str(item) for item in state.get("human_agent_ids", [])],
                str(path),
            )
    agents_csv = Path(cfg.route_artifacts_dir).resolve() / "agents.csv"
    machines, humans = derive_fixed_roles(
        agents_csv=agents_csv,
        ratio_machines=float(cfg.ratio_machines),
        seed=int(cfg.seed),
    )
    return [str(item) for item in machines], [str(item) for item in humans], "derived_from_agents_csv"


def fixed_role_hash(machine_ids: Sequence[str]) -> str:
    payload = json.dumps([str(item) for item in machine_ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _empty_sparse_rows(obs_dim: int, n_routes: int) -> Dict[str, np.ndarray]:
    return {
        "obs": np.zeros((0, int(obs_dim)), dtype=np.float32),
        "actions": np.zeros((0,), dtype=np.int64),
        "rewards": np.zeros((0,), dtype=np.float32),
        "old_log_probs": np.zeros((0,), dtype=np.float32),
        "action_valid": np.zeros((0, int(n_routes)), dtype=bool),
    }


def _pad_route_candidates(
    index_paths: Sequence[Sequence[int]],
    named_paths: Sequence[Sequence[str]],
    *,
    n_routes: int,
) -> Tuple[List[List[int]], List[List[str]], np.ndarray]:
    """Pad observation tensors without advertising duplicate aliases as actions."""
    if len(index_paths) != len(named_paths):
        raise ValueError("index_paths and named_paths must have the same length")
    limit = int(n_routes)
    pairs: List[Tuple[List[int], List[str]]] = []
    seen_paths: set[Tuple[int, ...]] = set()
    for indexed_path, named_path in zip(index_paths, named_paths, strict=True):
        indexed_item = list(indexed_path)
        identity = tuple(int(edge) for edge in indexed_item)
        if not indexed_item or identity in seen_paths:
            continue
        seen_paths.add(identity)
        pairs.append((indexed_item, list(named_path)))
        if len(pairs) >= limit:
            break
    indexed = [item[0] for item in pairs]
    named = [item[1] for item in pairs]
    if not indexed or len(named) != len(indexed):
        raise ValueError("at least one aligned non-empty route candidate is required")
    valid = np.zeros((limit,), dtype=bool)
    valid[:len(indexed)] = True
    while len(indexed) < limit:
        # Observation builders infer their valid-route feature from whether a
        # path is empty.  Keep padded tensor slots empty so that feature agrees
        # with the execution/replay mask; the named fallback is never selected
        # because the corresponding action remains invalid.
        indexed.append([])
        named.append(list(named[0]))
    return indexed, named, valid


class SparseDecisionBoundary:
    """Translate live vehicle identities into sparse fixed-slot MAMBA frames."""

    def __init__(
        self,
        *,
        learner: Any,
        machine_ids: Sequence[str],
        obs_dim: int,
        n_routes: int,
    ) -> None:
        self.learner = learner
        self.machine_ids = [str(item) for item in machine_ids]
        self.slot_by_vehicle = {veh_id: slot for slot, veh_id in enumerate(self.machine_ids)}
        if len(self.slot_by_vehicle) != len(self.machine_ids):
            raise ValueError("machine_agent_ids contains duplicates")
        self.obs_dim = int(obs_dim)
        self.n_routes = int(n_routes)
        self.reset_tracking()

    def reset_tracking(self) -> None:
        self._presence: set[int] = set()
        self._seen: set[int] = set()
        self._terminated: set[int] = set()
        self._context: Optional[Dict[str, Any]] = None

    def reset_episode(self) -> None:
        self.reset_tracking()
        self.learner.reset_episode()

    def _slots(self, vehicle_ids: Sequence[str]) -> List[int]:
        return sorted(
            self.slot_by_vehicle[str(veh_id)]
            for veh_id in vehicle_ids
            if str(veh_id) in self.slot_by_vehicle
        )

    def begin_decision(
        self,
        *,
        episode: int,
        decision_idx: int,
        sim_step: int,
        presence_vehicle_ids: Sequence[str],
    ) -> Dict[str, Any]:
        if self._context is not None:
            raise RuntimeError("previous MAMBA decision was not ended")
        presence = set(self._slots(presence_vehicle_ids))
        terminated = (self._presence - presence) - self._terminated
        self._terminated.update(terminated)
        self._seen.update(presence)
        self._presence = presence
        self.learner.begin_decision(sorted(presence))
        self._context = {
            "episode": int(episode),
            "decision_idx": int(decision_idx),
            "sim_step": int(sim_step),
            "presence_slots": sorted(presence),
            "terminated_slots": sorted(terminated),
            "truncated": False,
        }
        return dict(self._context)

    def end_decision(self, samples: Sequence[OnlineDecisionSample]) -> Dict[str, Any]:
        if self._context is None:
            raise RuntimeError("MAMBA decision ended without begin_decision")
        rows = list(samples)
        decision_slots = [self.slot_by_vehicle[str(sample.veh_id)] for sample in rows]
        if len(set(decision_slots)) != len(decision_slots):
            raise ValueError("a vehicle may contribute at most one sample per decision frame")
        if rows:
            obs = np.stack([np.asarray(sample.obs, dtype=np.float32) for sample in rows])
            if obs.shape != (len(rows), self.obs_dim):
                raise ValueError(
                    f"sparse obs has shape {obs.shape}; expected {(len(rows), self.obs_dim)}"
                )
            sparse = {
                "obs": obs,
                "actions": np.asarray([sample.action for sample in rows], dtype=np.int64),
                "rewards": np.asarray([sample.reward for sample in rows], dtype=np.float32),
                "old_log_probs": np.asarray(
                    [sample.old_log_prob for sample in rows], dtype=np.float32
                ),
                "action_valid": np.stack(
                    [np.asarray(sample.valid_mask, dtype=bool) for sample in rows]
                ),
            }
            if sparse["action_valid"].shape != (len(rows), self.n_routes):
                raise ValueError("sparse action_valid shape does not match action space")
        else:
            sparse = _empty_sparse_rows(self.obs_dim, self.n_routes)
        frame = {
            **self._context,
            "obs_slots": list(decision_slots),
            "decision_slots": list(decision_slots),
            **sparse,
        }
        self.learner.end_decision(frame)
        self._context = None
        return frame

    def end_episode(
        self,
        *,
        episode: int,
        decision_idx: int,
        sim_step: int,
        final_presence_vehicle_ids: Sequence[str],
        truncated: bool,
    ) -> Dict[str, Any]:
        if self._context is not None:
            raise RuntimeError("cannot end an episode during an open MAMBA decision")
        final_presence = set(self._slots(final_presence_vehicle_ids))
        terminated = (self._seen - final_presence) - self._terminated
        self._terminated.update(terminated)
        frame: Dict[str, Any] = {
            "episode": int(episode),
            "decision_idx": int(decision_idx),
            "sim_step": int(sim_step),
            "presence_slots": sorted(final_presence),
            "obs_slots": [],
            "decision_slots": [],
            "terminated_slots": sorted(terminated),
            "truncated": bool(truncated),
            **_empty_sparse_rows(self.obs_dim, self.n_routes),
        }
        self.learner.end_episode(frame)
        self._presence = final_presence
        return frame


def _normalized_action_result(
    result: Any,
    *,
    valid_mask: np.ndarray,
    n_routes: int,
) -> Tuple[int, float, np.ndarray]:
    """Normalize the stable learner act contract and validate masked actions."""
    if isinstance(result, Mapping):
        action = int(result["action"])
        log_prob = float(result["log_prob"])
        logits = result.get("logits")
        raw_probs = result.get("probs")
    elif isinstance(result, (tuple, list)) and len(result) >= 2:
        action, log_prob = int(result[0]), float(result[1])
        logits = result[2] if len(result) >= 3 else None
        raw_probs = None
    else:
        raise TypeError("MambaLearner.act must return a mapping or (action, log_prob[, logits])")
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid.shape != (int(n_routes),) or not bool(valid.any()):
        raise ValueError("MAMBA action mask must contain at least one valid route")
    if action < 0 or action >= int(n_routes) or not bool(valid[action]):
        raise ValueError(f"MAMBA selected invalid route action {action}")
    if not np.isfinite(log_prob):
        raise ValueError("MAMBA action log probability must be finite")
    if raw_probs is not None:
        probs = np.asarray(torch.as_tensor(raw_probs).detach().cpu(), dtype=np.float32).reshape(-1)
    elif logits is not None:
        logits_t = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
        if logits_t.numel() != int(n_routes):
            raise ValueError("MAMBA action logits have the wrong size")
        logits_t = logits_t.masked_fill(~torch.as_tensor(valid), -1.0e9)
        probs = torch.softmax(logits_t, dim=-1).cpu().numpy().astype(np.float32)
    else:
        probs = np.zeros((int(n_routes),), dtype=np.float32)
        probs[action] = 1.0
    probs = _apply_route_valid_mask_to_probs(probs, valid, int(n_routes))
    return action, log_prob, probs


def mamba_transition_diagnostics() -> Dict[str, int]:
    """Keep UrbanDreamer CFM and MAMBA RSSM accounting unambiguous."""
    return {
        "cfm_transition_used": 0,
        "mamba_transition_used": 1,
        "wm_rollout_rerank_used": 0,
    }


class _DeviceAnchor(torch.nn.Module):
    """Satisfy the legacy collector's actor-device plumbing without acting."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), device=device), requires_grad=False)


class ToyMambaCollector(ToyOnlineAcCollector):
    """MAMBA-specific collector with one joint prior per SUMO decision."""

    def __init__(
        self,
        *,
        learner: Any,
        machine_ids: Sequence[str],
        obs_dim: int,
        cfg: ToyMambaConfig,
        **collector_kwargs: Any,
    ) -> None:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        super().__init__(actor=_DeviceAnchor(device), cfg=cfg, **collector_kwargs)
        self._mamba = learner
        self._mamba_boundary = SparseDecisionBoundary(
            learner=learner,
            machine_ids=machine_ids,
            obs_dim=int(obs_dim),
            n_routes=int(cfg.n_routes),
        )
        self._mamba_machine_ids = {str(item) for item in machine_ids}

    def _collect_single_episode(
        self, episode_id: int, max_steps: int
    ) -> Tuple[List[OnlineDecisionSample], Dict[str, Any]]:
        self._mamba_boundary.reset_episode()
        samples, row = super()._collect_single_episode(episode_id, max_steps)
        state = self.connector.get_traffic_state()
        vehicles = state.get("vehicles", {}) if isinstance(state, dict) else {}
        final_ids = sorted(self._mamba_machine_ids & set(vehicles))
        truncated = bool(int(row.get("unfinished_all", 0)) > 0)
        self._mamba_boundary.end_episode(
            episode=int(episode_id),
            decision_idx=int(row.get("steps", 0)),
            sim_step=int(getattr(self.connector, "current_step", -1)),
            final_presence_vehicle_ids=final_ids,
            truncated=truncated,
        )
        row["mamba_sparse_frames"] = int(row.get("steps", 0)) + 1
        row["mamba_truncated"] = int(truncated)
        return samples, row

    def _select_av_actions(
        self, state: Dict[str, Any], rng: np.random.RandomState
    ) -> Dict[str, List[str]]:
        del rng  # Sampling is performed by MambaLearner from checkpointed torch RNG state.
        stats = getattr(self, "_av_decision_skip_stats", None)
        if stats is None:
            stats = {}
            self._av_decision_skip_stats = stats

        def inc(name: str, amount: int = 1) -> None:
            stats[name] = int(stats.get(name, 0)) + int(amount)

        inc("calls")
        in_network = state.get("vehicles", {}) if isinstance(state, dict) else {}
        active_avs = sorted(self._av_vehicles & set(in_network))
        self._mamba_boundary.begin_decision(
            episode=int(self._current_episode_id_eval),
            decision_idx=int(self._current_decision_idx_eval),
            sim_step=int(getattr(self.connector, "current_step", -1)),
            presence_vehicle_ids=active_avs,
        )
        inc("active_av_seen", len(active_avs))
        if not active_avs:
            inc("no_active_av_calls")
            self._mamba_boundary.end_decision([])
            return {}

        edge_feats_dyn = self._live_edge_features(state)
        edge_to_idx = getattr(self._edge_registry, "_edge_to_idx", {})
        live_edge_cost = self._live_current_edge_tt(state)
        freeflow_edge_cost = self._freeflow_edge_tt()
        obs_edge_cost = live_edge_cost if live_edge_cost is not None else freeflow_edge_cost
        candidate_edge_cost = freeflow_edge_cost
        if self._av_candidate_source == "current_shortest_live":
            candidate_edge_cost = live_edge_cost
        elif self._av_candidate_source == "current_shortest_wm":
            candidate_edge_cost = freeflow_edge_cost
            inc("candidate_wm_source_fallback_freeflow")
        if self._av_candidate_source != "od_suffix" and candidate_edge_cost is None:
            inc("candidate_edge_cost_missing_calls")

        path_batch: List[List[List[int]]] = []
        route_batch: List[List[List[str]]] = []
        active_ids: List[str] = []
        active_current_edges: List[str] = []
        valid_rows: List[np.ndarray] = []
        candidate_dynamic_count = 0
        candidate_fallback_count = 0
        for veh_id in active_avs:
            veh_info = in_network.get(veh_id)
            if not isinstance(veh_info, dict):
                inc("skip_bad_vehicle_info")
                continue
            current_edge = str(veh_info.get("edge", ""))
            if not current_edge or current_edge.startswith(":"):
                inc("skip_bad_current_edge")
                continue
            candidates: List[List[str]] = []
            if self._av_candidate_source != "od_suffix":
                candidates = self._current_to_dest_route_candidates(
                    veh_id, current_edge, candidate_edge_cost
                )
                if candidates:
                    candidate_dynamic_count += 1
                else:
                    candidate_fallback_count += 1
            if not candidates:
                candidates = self._candidate_route_suffixes(veh_id, current_edge)
            if candidates:
                candidates = self._filter_safe_reroute_candidates(
                    veh_id, current_edge, candidates
                )
            if not candidates:
                inc("skip_no_candidates")
                continue
            idx_paths: List[List[int]] = []
            named_paths: List[List[str]] = []
            for route in candidates[: int(self.n_routes)]:
                idxs = [int(edge_to_idx[edge]) for edge in route if edge in edge_to_idx]
                if idxs:
                    idx_paths.append(idxs)
                    named_paths.append(list(route))
            if not idx_paths:
                inc("skip_no_idx_paths")
                continue
            idx_paths, named_paths, valid_row = _pad_route_candidates(
                idx_paths,
                named_paths,
                n_routes=int(self.n_routes),
            )
            path_batch.append(idx_paths)
            route_batch.append(named_paths)
            valid_rows.append(valid_row)
            active_ids.append(str(veh_id))
            active_current_edges.append(current_edge)

        if not path_batch:
            inc("no_path_batch_calls")
            self._mamba_boundary.end_decision([])
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
        valid_mask = np.stack(valid_rows)
        selected_indices = np.zeros((len(active_ids),), dtype=np.int64)
        old_log_probs = np.zeros((len(active_ids),), dtype=np.float32)
        policy_probs: List[np.ndarray] = []
        obs_rows: List[np.ndarray] = []
        deterministic = str(self._cfg.real_policy_route_selection).lower() == "argmax"

        def act_one(index: int, obs_row: np.ndarray) -> Tuple[int, float, np.ndarray]:
            slot = self._mamba_boundary.slot_by_vehicle[active_ids[index]]
            result = self._mamba.act(
                slot=int(slot),
                obs=np.asarray(obs_row, dtype=np.float32),
                action_mask=np.asarray(valid_mask[index], dtype=bool),
                deterministic=bool(deterministic),
            )
            return _normalized_action_result(
                result,
                valid_mask=valid_mask[index],
                n_routes=int(self.n_routes),
            )

        if self._vehicle_obs_include_fleet_pressure:
            fleet_counts = np.zeros((int(self.n_routes),), dtype=np.float32)
            fleet_pressure = np.zeros(
                (len(getattr(self._edge_registry, "edge_ids", [])),), dtype=np.float32
            )
            for i in range(len(active_ids)):
                fleet_i = build_vehicle_fleet_pressure_obs_features(
                    path_batch=[path_batch[i]],
                    n_edges=len(getattr(self._edge_registry, "edge_ids", [])),
                    n_routes=int(self.n_routes),
                    selected_route_counts=fleet_counts,
                    selected_edge_pressure=fleet_pressure,
                )
                obs_i = _append_vehicle_context_features(
                    np.concatenate([obs_base[i : i + 1], fleet_i], axis=1),
                    step_idx=int(getattr(self.connector, "current_step", 0)),
                    episode_steps=int(self._cfg.episode_steps),
                    n_active_av=len(active_ids),
                    n_total_av=max(len(self._av_vehicles), 1),
                )[0].astype(np.float32)
                action, log_prob, probs = act_one(i, obs_i)
                selected_indices[i] = action
                old_log_probs[i] = log_prob
                policy_probs.append(probs)
                obs_rows.append(obs_i)
                fleet_counts[action] += 1.0
                for edge_idx in path_batch[i][action]:
                    if 0 <= int(edge_idx) < fleet_pressure.shape[0]:
                        fleet_pressure[int(edge_idx)] += 1.0
            obs_vehicle_np = np.stack(obs_rows).astype(np.float32)
            probs_np = np.stack(policy_probs).astype(np.float32)
        else:
            obs_vehicle_np = _append_vehicle_context_features(
                obs_base,
                step_idx=int(getattr(self.connector, "current_step", 0)),
                episode_steps=int(self._cfg.episode_steps),
                n_active_av=len(active_ids),
                n_total_av=max(len(self._av_vehicles), 1),
            ).astype(np.float32)
            for i in range(len(active_ids)):
                action, log_prob, probs = act_one(i, obs_vehicle_np[i])
                selected_indices[i] = action
                old_log_probs[i] = log_prob
                policy_probs.append(probs)
            probs_np = np.stack(policy_probs).astype(np.float32)

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
        elapsed = self._vehicle_elapsed_seconds(
            active_ids, int(getattr(self.connector, "current_step", 0))
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
        frame_samples: List[OnlineDecisionSample] = []
        for i, veh_id in enumerate(active_ids):
            route_idx = int(selected_indices[i])
            self._av_route_idx[veh_id] = route_idx
            actions[veh_id] = route_batch[i][route_idx]
            sample = OnlineDecisionSample(
                episode=int(self._current_episode_id_eval),
                decision_idx=int(self._current_decision_idx_eval),
                sim_step=int(getattr(self.connector, "current_step", -1)),
                veh_id=veh_id,
                obs=obs_vehicle_np[i],
                action=route_idx,
                old_log_prob=float(old_log_probs[i]),
                reward=float(rewards[i]),
                valid_mask=valid_mask[i].astype(bool),
                route_costs=live_cost_matrix[i],
                elapsed_time=float(elapsed[i]),
                path_batch=[list(path) for path in path_batch[i]],
                current_edge_idx=int(edge_to_idx.get(active_current_edges[i], -1)),
                dest_edge_idx=int(edge_to_idx.get(str(self._veh_dest.get(veh_id, "")), -1)),
            )
            frame_samples.append(sample)
            self._episode_samples.append(sample)
        self._mamba_boundary.end_decision(frame_samples)

        unique_mean, unique_min, unique_max = _route_batch_uniqueness_summary(route_batch)
        counts = np.bincount(selected_indices, minlength=int(self.n_routes))
        mean_probs = probs_np.mean(axis=0)
        decision_row: Dict[str, Any] = {
            "episode": int(self._current_episode_id_eval),
            "decision_idx": int(self._current_decision_idx_eval),
            "sim_step": int(getattr(self.connector, "current_step", -1)),
            "active_av": len(active_ids),
            "candidate_dynamic_count": candidate_dynamic_count,
            "candidate_fallback_count": candidate_fallback_count,
            "candidate_unique_mean": float(unique_mean),
            "candidate_unique_min": int(unique_min),
            "candidate_unique_max": int(unique_max),
            "reward_mean": float(rewards.mean()) if rewards.size else float("nan"),
            "reward_sum": float(rewards.sum()),
            **mamba_transition_diagnostics(),
            "obs_source": "live_edge_registry_extract_flow",
            "policy_route_selection": str(self._cfg.real_policy_route_selection),
        }
        for route in range(int(self.n_routes)):
            decision_row[f"mean_prob_route{route}"] = float(mean_probs[route])
            decision_row[f"selected_count_route{route}"] = int(counts[route])
            costs = live_cost_matrix[:, route]
            costs = costs[np.isfinite(costs)]
            decision_row[f"live_mean_route_cost_route{route}"] = (
                float(costs.mean()) if costs.size else float("nan")
            )
        self._dynamic_decision_rows.append(decision_row)

        for i, veh_id in enumerate(active_ids):
            row: Dict[str, Any] = {
                "episode": int(self._current_episode_id_eval),
                "decision_idx": int(self._current_decision_idx_eval),
                "sim_step": int(getattr(self.connector, "current_step", -1)),
                "veh_id": veh_id,
                "global_slot": self._mamba_boundary.slot_by_vehicle[veh_id],
                "current_edge": active_current_edges[i],
                "dest_edge": str(self._veh_dest.get(veh_id, "")),
                "selected_route": int(selected_indices[i]),
                "policy_route_selection": str(self._cfg.real_policy_route_selection),
                "reward": float(rewards[i]),
                "elapsed_time": float(elapsed[i]),
            }
            for route in range(int(self.n_routes)):
                row[f"prob_route{route}"] = float(probs_np[i, route])
                row[f"valid_route{route}"] = int(bool(valid_mask[i, route]))
                value = live_cost_matrix[i, route]
                row[f"live_cost_route{route}"] = (
                    float(value) if np.isfinite(value) else float("nan")
                )
            self._dynamic_agent_decision_rows.append(row)
        return actions


class ToyMambaTrainer:
    """Own the MAMBA learner while reusing the fixed-role SUMO protocol."""

    def __init__(self, cfg: ToyMambaConfig) -> None:
        self.cfg = cfg.resolve()
        if str(self.cfg.av_execution_mode).lower() != "dynamic":
            raise ValueError("MAMBA currently requires --av-execution-mode dynamic")
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(int(self.cfg.seed))
        np.random.seed(int(self.cfg.seed))
        torch.manual_seed(int(self.cfg.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.cfg.seed))
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self.edge_registry = EdgeRegistry.load(str(self.cfg.edge_registry))
        self.obs_dim = vehicle_mode_obs_dim(
            n_routes=int(self.cfg.n_routes),
            d_obs=8,
            include_global=bool(self.cfg.include_global),
            route_stats=str(self.cfg.route_stats),
            include_route_costs=bool(self.cfg.include_route_costs),
            include_fleet_pressure=bool(self.cfg.include_fleet_pressure),
        )
        self.machine_ids, self.human_ids, self.role_source = _resolve_fixed_roles(self.cfg)
        if not self.machine_ids:
            raise ValueError("MAMBA fixed-role baseline requires at least one machine agent")
        self.role_hash = fixed_role_hash(self.machine_ids)

        # Keep the core import lazy so --dry-run and runner-only tests do not
        # initialize CUDA or require the learner implementation to be imported.
        from baseline.mamba import MambaConfig, MambaLearner

        self.core_config = MambaConfig(
            obs_dim=int(self.obs_dim),
            action_dim=int(self.cfg.n_routes),
            n_slots=len(self.machine_ids),
            device=str(self.device),
            seed=int(self.cfg.seed),
            hidden_size=int(self.cfg.hidden_dim),
            model_hidden_size=int(self.cfg.model_hidden_dim),
            embedding_size=int(self.cfg.embed_dim),
            deterministic_size=int(self.cfg.deter_dim),
            n_categoricals=int(self.cfg.stoch_categories),
            n_classes=int(self.cfg.stoch_classes),
            transformer_layers=int(self.cfg.transformer_layers),
            transformer_heads=int(self.cfg.attention_heads),
            replay_capacity_episodes=int(self.cfg.replay_episodes),
            sequence_length=int(self.cfg.sequence_length),
            burn_in=int(self.cfg.burn_in),
            max_agents_per_chunk=int(self.cfg.communication_chunk_size),
            sequence_minibatch_size=int(self.cfg.sequence_minibatch_size),
            min_model_replay_frames=int(self.cfg.min_model_replay_frames),
            min_agent_replay_frames=int(self.cfg.min_agent_replay_frames),
            max_imagination_starts=int(self.cfg.max_imagination_starts),
            imagination_alive_threshold=float(self.cfg.imagination_alive_threshold),
            action_information_coef=float(self.cfg.action_information_coef),
            availability_coef=float(self.cfg.availability_coef),
            availability_threshold=float(self.cfg.availability_threshold),
            model_epochs=int(self.cfg.model_epochs),
            agent_epochs=int(self.cfg.agent_epochs),
            critic_epochs=int(self.cfg.critic_epochs),
            ppo_epochs=int(self.cfg.ppo_epochs),
            critic_warmup_updates=int(self.cfg.critic_warmup_updates),
            imagination_horizon=int(self.cfg.imagination_horizon),
            imagination_reward_scope=str(self.cfg.imagination_reward_scope),
            update_interval_episodes=int(self.cfg.update_every_episodes),
            model_lr=float(self.cfg.model_lr),
            actor_lr=float(self.cfg.actor_lr),
            critic_lr=float(self.cfg.mamba_critic_lr),
            gamma=float(self.cfg.gamma),
            gae_lambda=float(self.cfg.gae_lambda),
            ppo_clip=float(self.cfg.ppo_clip),
            target_policy_kl=float(self.cfg.target_policy_kl),
            entropy_coef=float(self.cfg.entropy_coef),
            value_coef=float(self.cfg.value_coef),
            max_grad_norm=float(self.cfg.max_grad_norm),
            policy_max_grad_norm=float(self.cfg.max_grad_norm),
            critic_max_grad_norm=float(self.cfg.max_grad_norm),
            critic_huber_delta=float(self.cfg.critic_huber_delta),
            target_tau=float(self.cfg.target_tau),
            normalize_advantages=bool(self.cfg.ac_normalize_advantages),
            advantage_clip=float(self.cfg.ac_advantage_clip),
            use_bfloat16=str(self.cfg.amp_dtype).lower() == "bf16",
        )
        self.learner = MambaLearner(self.core_config, self.machine_ids)
        self.collector: Optional[ToyMambaCollector] = None
        self._resume_collector_protocol_state: Optional[Dict[str, Any]] = None
        self.loaded_resume_episode = self._load_resume_checkpoint()

    @staticmethod
    def _validate_collector_protocol_state(
        raw_state: Optional[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if raw_state is None:
            return None
        if not isinstance(raw_state, Mapping):
            raise ValueError("checkpoint collector_protocol_state is not an object")
        try:
            version = int(raw_state.get("version", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint collector protocol version is invalid") from exc
        if version != COLLECTOR_PROTOCOL_STATE_VERSION:
            raise ValueError(f"unsupported collector protocol state version: {version}")

        hdv_state = raw_state.get("hdv_learning_state")
        if not isinstance(hdv_state, Mapping):
            raise ValueError("checkpoint collector protocol HDV state is not an object")
        if not isinstance(hdv_state.get("agents"), Mapping):
            raise ValueError("checkpoint collector protocol HDV agents are not an object")
        if "route_policy" in hdv_state and not isinstance(hdv_state["route_policy"], list):
            raise ValueError("checkpoint collector protocol route_policy is not a list")

        episode_count = raw_state.get("connector_episode_count")
        if isinstance(episode_count, bool) or not isinstance(episode_count, (int, np.integer)):
            raise ValueError("checkpoint connector_episode_count is not an integer")
        if int(episode_count) < 0:
            raise ValueError("checkpoint connector_episode_count must be non-negative")
        return {
            "version": version,
            "hdv_learning_state": deepcopy(dict(hdv_state)),
            "connector_episode_count": int(episode_count),
        }

    def _snapshot_collector_protocol_state(self) -> Dict[str, Any]:
        collector = self.collector
        if collector is None:
            raise RuntimeError("cannot checkpoint collector protocol state before collector setup")
        connector = getattr(collector, "connector", None)
        if connector is None or not hasattr(connector, "_episode_count"):
            raise RuntimeError("collector connector does not expose its episode counter")
        env = getattr(connector, "env", None)
        if env is None:
            raise RuntimeError("collector connector does not expose its HDV environment")
        hdv_state = _merge_hdv_learning_state(
            getattr(collector, "_loaded_hdv_state_eval", None),
            _snapshot_hdv_learning_state(env),
            route_state=serialize_hdv_route_state(
                getattr(collector, "_hdv_route_probs", None),
                getattr(collector, "_hdv_route_costs", None),
            ),
        )
        return {
            "version": COLLECTOR_PROTOCOL_STATE_VERSION,
            "hdv_learning_state": hdv_state,
            "connector_episode_count": int(connector._episode_count),
        }

    def _resume_hdv_state(
        self,
        fallback_state: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        protocol_state = self._resume_collector_protocol_state
        if protocol_state is None:
            return fallback_state
        return deepcopy(protocol_state["hdv_learning_state"])

    def _initialize_connector_episode_count(
        self,
        collector: ToyMambaCollector,
        post_state: Optional[Mapping[str, Any]],
    ) -> None:
        protocol_state = self._resume_collector_protocol_state
        if protocol_state is not None:
            episode_count = int(protocol_state["connector_episode_count"])
            exact_resume = True
        elif post_state is not None:
            episode_count = int(post_state.get("mutation_episode_index", 0) or 0)
            exact_resume = False
        else:
            return
        connector = getattr(collector, "connector", None)
        if connector is None or not hasattr(connector, "_episode_count"):
            if exact_resume:
                raise RuntimeError("collector connector cannot restore its checkpoint episode counter")
            return
        connector._episode_count = episode_count

    def _load_resume_checkpoint(self) -> int:
        raw = str(self.cfg.resume_checkpoint or "").strip()
        if not raw:
            return 0
        path = _resolve_path(raw)
        if not path.exists():
            raise FileNotFoundError(f"resume_checkpoint does not exist: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("model_family") != "mamba":
            raise ValueError(f"resume checkpoint is not a MAMBA checkpoint: {path}")
        if int(state.get("schema_version", -1)) != MAMBA_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported MAMBA checkpoint schema: {state.get('schema_version')}")
        checkpoint_ids = [str(item) for item in state.get("machine_agent_ids", [])]
        if checkpoint_ids != self.machine_ids:
            raise ValueError("resume checkpoint machine_agent_ids do not match fixed roles")
        if int(state.get("obs_dim", self.obs_dim)) != int(self.obs_dim):
            raise ValueError("resume checkpoint observation dimension does not match this run")
        if int(state.get("action_dim", self.cfg.n_routes)) != int(self.cfg.n_routes):
            raise ValueError("resume checkpoint action dimension does not match this run")
        protocol_state = self._validate_collector_protocol_state(
            state.get("collector_protocol_state")
        )
        loaded = self.learner.load_checkpoint_state(state)
        self._resume_collector_protocol_state = protocol_state
        episode = int(state.get("episode", 0) if loaded is None else loaded)
        print(f"[toy-mamba] resumed full learner state from {path} at episode {episode}", flush=True)
        return episode

    def _build_collector(self) -> ToyMambaCollector:
        patch_traci_start_port_fallback()
        runtime_dir = self.output_dir / "collector_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        post_state: Optional[Dict[str, Any]] = None
        post_path: Optional[Path] = None
        raw_post = str(self.cfg.post_mutation_state_in or "").strip()
        if raw_post:
            post_path = Path(raw_post).resolve()
        if post_path is not None and post_path.exists():
            post_state = load_post_mutation_state(post_path)
            reused = _reuse_route_artifacts_from_refs(post_state.get("artifact_refs", {}), runtime_dir)
        else:
            reused = _reuse_route_artifacts(Path(self.cfg.route_artifacts_dir).resolve(), runtime_dir)

        hdv_state_in = str(self.cfg.hdv_state_in or "").strip()
        if not hdv_state_in and post_state is not None:
            hdv_state_in = str(post_state.get("human_learning_state_ref", "") or "").strip()
        loaded_hdv_state = _load_hdv_learning_state(Path(hdv_state_in)) if hdv_state_in else None
        loaded_hdv_state = self._resume_hdv_state(loaded_hdv_state)
        network_folder = str(Path(self.cfg.network_dir).resolve())
        od_data = load_od_data(network_folder, self.cfg.network_name)
        agents_df = pd.read_csv(reused["agents_csv"])
        num_agents, _num_machines, max_start_time = compute_agent_counts(
            agents_df, float(self.cfg.ratio_machines)
        )
        last_start = (
            float(agents_df["start_time"].max())
            if "start_time" in agents_df.columns
            else float(max_start_time)
        )
        sim_length = int(
            max(float(max_start_time), last_start) + float(self.cfg.sim_buffer_after_last_depart)
        )
        collector = ToyMambaCollector(
            learner=self.learner,
            machine_ids=self.machine_ids,
            obs_dim=int(self.obs_dim),
            cfg=self.cfg,
            loaded_hdv_state=loaded_hdv_state,
            output_dir=str(runtime_dir),
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
            records_folder=str(runtime_dir),
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
        self._initialize_connector_episode_count(collector, post_state)
        collector.set_fixed_roles(machine_ids=self.machine_ids, human_ids=self.human_ids)
        manifest = {
            "source": self.role_source,
            "post_mutation_state_in": str(post_path) if post_path is not None else "",
            "agents_csv": str(reused["agents_csv"]),
            "ratio_machines": float(self.cfg.ratio_machines),
            "seed": int(self.cfg.seed),
            "role_hash": self.role_hash,
            "human_learning_enabled": not bool(self.cfg.disable_human_learning_after_mutation),
            "machine_count": len(self.machine_ids),
            "human_count": len(self.human_ids),
            "machine_agent_ids": self.machine_ids,
            "human_agent_ids": self.human_ids,
        }
        with (self.output_dir / "toy_mamba_fixed_roles.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        self.collector = collector
        self.rollout_max_steps = (
            int(self.cfg.max_steps_per_episode)
            if int(self.cfg.max_steps_per_episode) > 0
            else int(sim_length)
        )
        print(
            f"[toy-mamba] fixed roles machines={len(self.machine_ids)} "
            f"humans={len(self.human_ids)} source={self.role_source}",
            flush=True,
        )
        return collector

    @staticmethod
    def _pending_update_metrics(replay_episodes: int) -> Dict[str, float]:
        return {
            "loss": 0.0,
            "model_loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "replay_episodes": float(replay_episodes),
            "skipped": 1.0,
            "pending_update": 1.0,
        }

    @staticmethod
    def _float_metrics(values: Mapping[str, Any]) -> Dict[str, float]:
        output: Dict[str, float] = {}
        for key, value in values.items():
            if torch.is_tensor(value):
                value = value.detach().float().mean().cpu().item()
            if isinstance(value, (bool, int, float, np.number)):
                output[str(key)] = float(value)
        output.setdefault("loss", output.get("model_loss", 0.0))
        output.setdefault("skipped", 0.0)
        output["pending_update"] = 0.0
        return output

    @staticmethod
    def _write_rows_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        """Persist completed episode/update rows without waiting for run exit."""
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        try:
            pd.DataFrame(list(rows)).to_csv(temporary, index=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _append_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        pd.DataFrame(list(rows)).to_csv(
            path,
            mode="a",
            header=not path.exists(),
            index=False,
        )

    @staticmethod
    def _load_resume_rows(path: Path, start_episode: int) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        frame = pd.read_csv(path)
        rows = frame.to_dict(orient="records")
        if rows:
            last_episode = int(rows[-1].get("episode", -1))
            if last_episode != int(start_episode):
                raise ValueError(
                    f"resume CSV {path} ends at episode {last_episode}, "
                    f"but checkpoint/start episode is {start_episode}; use a new output "
                    "directory or a matching checkpoint"
                )
        return rows

    def save_checkpoint(self, episode: int, metrics: Optional[Mapping[str, Any]] = None) -> Path:
        path = self.output_dir / "checkpoints" / f"checkpoint_ep{int(episode):05d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        collector_protocol_state = self._snapshot_collector_protocol_state()
        state = self.learner.checkpoint_state(
            episode=int(episode),
            metrics=dict(metrics or {}),
            role_source=str(self.role_source),
            role_hash=str(self.role_hash),
            config_dict=_jsonable_config(self.cfg),
        )
        state.update(
            {
                "schema_version": MAMBA_CHECKPOINT_SCHEMA_VERSION,
                "model_family": "mamba",
                "episode": int(episode),
                "obs_dim": int(self.obs_dim),
                "action_dim": int(self.cfg.n_routes),
                "machine_agent_ids": list(self.machine_ids),
                "human_agent_ids": list(self.human_ids),
                "role_source": str(self.role_source),
                "role_hash": str(self.role_hash),
                "runner_config": _jsonable_config(self.cfg),
                "metrics": dict(metrics or {}),
                "collector_protocol_state": collector_protocol_state,
            }
        )
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def train(self) -> Dict[str, Any]:
        collector = self._build_collector()
        checkpoints = {int(item) for item in self.cfg.checkpoint_episodes}
        start_episode = int(self.cfg.resume_start_episode or self.loaded_resume_episode or 0)
        checkpoint_paths: Dict[int, str] = {}
        episode_csv = self.output_dir / "online_train_episodes.csv"
        update_csv = self.output_dir / "online_train_updates.csv"
        decision_csv = self.output_dir / "online_train_decisions.csv"
        agent_decision_csv = self.output_dir / "online_train_agent_decisions.csv"
        if start_episode > 0:
            train_rows = self._load_resume_rows(episode_csv, start_episode)
            update_rows = self._load_resume_rows(update_csv, start_episode)
        else:
            existing = [
                path
                for path in (episode_csv, update_csv, decision_csv, agent_decision_csv)
                if path.exists()
            ]
            if existing:
                raise FileExistsError(
                    "fresh MAMBA training refuses to mix with existing metric files: "
                    + ", ".join(str(path) for path in existing)
                )
            train_rows: List[Dict[str, Any]] = []
            update_rows: List[Dict[str, Any]] = []
        started = time.time()
        try:
            for episode in range(start_episode + 1, int(self.cfg.train_episodes) + 1):
                samples, row = collector.run_train_episode(episode, int(self.rollout_max_steps))
                should_update = (
                    episode % max(int(self.cfg.update_every_episodes), 1) == 0
                    or episode in checkpoints
                    or episode == int(self.cfg.train_episodes)
                )
                if should_update:
                    update = self._float_metrics(self.learner.update())
                else:
                    replay = getattr(self.learner, "replay", None)
                    update = self._pending_update_metrics(len(replay) if replay is not None else 0)
                row.update({f"update_{key}": value for key, value in update.items()})
                train_rows.append(row)
                update_rows.append({"episode": int(episode), **update})
                # These two compact tables are atomically rewritten after every
                # completed episode so SIGTERM/hard-gate runs retain all metrics
                # even when later update columns first appear after warm-up.
                self._write_rows_atomic(episode_csv, train_rows)
                self._write_rows_atomic(update_csv, update_rows)
                # Decision tables have stable schemas and are much larger; append
                # only this episode and release their in-memory rows immediately.
                decision_rows = list(getattr(collector, "_dynamic_decision_rows", []))
                agent_decision_rows = list(
                    getattr(collector, "_dynamic_agent_decision_rows", [])
                )
                self._append_rows(decision_csv, decision_rows)
                self._append_rows(agent_decision_csv, agent_decision_rows)
                getattr(collector, "_dynamic_decision_rows", []).clear()
                getattr(collector, "_dynamic_agent_decision_rows", []).clear()
                print(
                    f"[toy-mamba] ep={episode:04d}/{int(self.cfg.train_episodes):04d} "
                    f"samples={len(samples)} reward_mean={row.get('reward_mean', float('nan')):.4f} "
                    f"loss={update.get('loss', 0.0):.4f} "
                    f"model={update.get('model_loss', 0.0):.4f} "
                    f"actor={update.get('actor_loss', 0.0):.4f} "
                    f"critic={update.get('critic_loss', 0.0):.4f} "
                    f"ev={update.get('critic_explained_variance_post', 0.0):.3f} "
                    f"adv_std={update.get('advantage_std', 0.0):.3f} "
                    f"kl={update.get('policy_kl', 0.0):.4f} "
                    f"clip={update.get('clip_fraction', 0.0):.4f} "
                    f"ppo_n={update.get('ppo_epochs_completed', 0.0):.0f} "
                    f"ppo_reject={update.get('ppo_epochs_rejected', 0.0):.0f} "
                    f"critic_n={update.get('critic_update_count', 0.0):.0f} "
                    f"survival={update.get('survival_mass', 0.0):.1f} "
                    f"t_all={row.get('t_all', float('nan')):.3f} "
                    f"completion={row.get('completion_rate', float('nan')):.6f} "
                    f"pending={int(update.get('pending_update', 0.0))}",
                    flush=True,
                )
                if episode in checkpoints:
                    checkpoint = self.save_checkpoint(episode, row)
                    checkpoint_paths[int(episode)] = str(checkpoint)
                    print(f"[toy-mamba] saved checkpoint: {checkpoint}", flush=True)
        finally:
            try:
                collector.finalize_protocol_xml_aliases()
            finally:
                close = getattr(getattr(collector, "connector", None), "close", None)
                if callable(close):
                    close()

        latest_hdv_state = _merge_hdv_learning_state(
            getattr(collector, "_loaded_hdv_state_eval", None),
            _snapshot_hdv_learning_state(collector.connector.env)
            if hasattr(collector.connector, "env")
            else {},
            route_state=serialize_hdv_route_state(
                getattr(collector, "_hdv_route_probs", None),
                getattr(collector, "_hdv_route_costs", None),
            ),
        )
        hdv_out = self.output_dir / "hdv_learning_state_after_online_train.json"
        _save_hdv_learning_state(hdv_out, latest_hdv_state)
        summary = {
            "baseline": "toy_cross_bottleneck_mamba",
            "algorithm": "mamba",
            "config": _jsonable_config(self.cfg),
            "mamba_config": self.core_config.to_dict(),
            "fixed_roles": {
                "source": self.role_source,
                "role_hash": self.role_hash,
                "machine_count": len(self.machine_ids),
                "human_count": len(self.human_ids),
                "machine_agent_ids": self.machine_ids,
                "human_agent_ids": self.human_ids,
            },
            "obs_dim": int(self.obs_dim),
            "obs_source": "live_edge_registry_extract_flow",
            "world_model_usage": {
                "family": "mamba_rssm",
                "joint_prior_per_decision": True,
                "posterior_per_observed_vehicle": True,
                "urban_dreamer_cfm_transition": False,
                "world_model_route_reranking": False,
            },
            "checkpoint_paths": checkpoint_paths,
            "train_episode_csv": str(self.output_dir / "online_train_episodes.csv"),
            "train_update_csv": str(self.output_dir / "online_train_updates.csv"),
            "hdv_state_out": str(hdv_out),
            "elapsed_sec": float(time.time() - started),
        }
        with (self.output_dir / "online_train_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default="")
    parser.add_argument("--network-dir", default=str(REPO_ROOT / "networks" / TOY_NETWORK_NAME))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--generate-network", action="store_true")
    parser.add_argument("--n-agents", type=int, default=320)
    parser.add_argument("--depart-span", type=int, default=420)
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--checkpoint-episodes", default="")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-start-episode", type=int, default=0)
    parser.add_argument("--post-mutation-state-in", default="")
    parser.add_argument("--hdv-state-in", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio-machines", type=float, default=0.4)
    parser.add_argument("--episode-steps", type=int, default=60)
    parser.add_argument("--sim-buffer-after-last-depart", type=int, default=3000)
    parser.add_argument("--max-steps-per-episode", type=int, default=0)
    parser.add_argument(
        "--reward-mode",
        default="capped_step_cost",
        choices=["progress_fraction", "capped_step_cost"],
    )
    parser.add_argument("--reward-temperature", type=float, default=60.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--update-every-episodes", type=int, default=1)
    parser.add_argument("--replay-episodes", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--burn-in", type=int, default=5)
    parser.add_argument("--sequence-minibatch-size", type=int, default=4)
    parser.add_argument("--communication-chunk-size", type=int, default=32)
    parser.add_argument("--min-model-replay-frames", type=int, default=100)
    parser.add_argument("--min-agent-replay-frames", type=int, default=2500)
    parser.add_argument("--max-imagination-starts", type=int, default=16)
    parser.add_argument("--imagination-alive-threshold", type=float, default=0.5)
    parser.add_argument("--action-information-coef", type=float, default=1.0)
    parser.add_argument("--availability-coef", type=float, default=1.0)
    parser.add_argument("--availability-threshold", type=float, default=0.5)
    parser.add_argument("--model-epochs", type=int, default=4)
    parser.add_argument("--agent-epochs", type=int, default=1)
    parser.add_argument("--critic-epochs", type=int, default=5)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--critic-warmup-updates", type=int, default=20)
    parser.add_argument(
        "--imagination-reward-scope",
        choices=["individual", "team"],
        default="individual",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--model-hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--deter-dim", type=int, default=256)
    parser.add_argument("--stoch-classes", type=int, default=32)
    parser.add_argument("--stoch-categories", type=int, default=32)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--imag-horizon", type=int, default=15)
    parser.add_argument("--model-lr", type=float, default=2.0e-4)
    parser.add_argument("--actor-lr", type=float, default=2.0e-4)
    parser.add_argument("--critic-lr", type=float, default=3.0e-4)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-clip", type=float, default=0.075)
    parser.add_argument("--target-policy-kl", type=float, default=0.01)
    parser.add_argument("--critic-huber-delta", type=float, default=1.0)
    parser.add_argument("--target-tau", type=float, default=0.05)
    parser.add_argument("--advantage-clip", type=float, default=2.5)
    parser.add_argument(
        "--normalize-advantages",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--entropy-coef", type=float, default=5.0e-3)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--av-execution-mode", choices=["dynamic"], default="dynamic")
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ToyMambaConfig:
    has_settings = bool(str(args.settings or "").strip())
    network_dir = Path(args.network_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT
        / "results"
        / ("from_settings" if has_settings else TOY_NETWORK_NAME)
        / f"mamba_fixed_role_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    network_overrides: Dict[str, Any] = {}
    if not has_settings:
        network_overrides = {
            "tag": TOY_NETWORK_NAME,
            "network_name": TOY_NETWORK_NAME,
            "network_dir": str(network_dir),
            "edge_registry": str(network_dir / "edge_registry.json"),
            "projector_ckpt": "",
            "dynamics_ckpt": "",
            "route_artifacts_dir": str(network_dir),
            "ratio_machines": float(args.ratio_machines),
            "n_routes": 4,
            "episode_steps": int(args.episode_steps),
            "sim_buffer_after_last_depart": int(args.sim_buffer_after_last_depart),
            "incident_edge": str(args.incident_edge or ""),
            "incident_start": float(args.incident_start),
            "incident_end": float(args.incident_end),
            "incident_speed": float(args.incident_speed),
        }
    episodes = int(args.train_episodes)
    overrides = {
        **network_overrides,
        "output_dir": str(output_dir),
        "train_episodes": episodes,
        "checkpoint_interval": int(args.checkpoint_interval),
        "checkpoint_episodes": _checkpoint_episodes(
            raw=args.checkpoint_episodes,
            train_episodes=episodes,
            checkpoint_interval=int(args.checkpoint_interval),
        ),
        "resume_checkpoint": str(args.resume_checkpoint or ""),
        "resume_start_episode": int(args.resume_start_episode),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "seed": int(args.seed),
        "device": str(args.device),
        "reward_mode": str(args.reward_mode),
        "reward_temperature": float(args.reward_temperature),
        "gamma": float(args.gamma),
        "update_every_episodes": int(args.update_every_episodes),
        "replay_episodes": int(args.replay_episodes),
        "sequence_length": int(args.sequence_length),
        "burn_in": int(args.burn_in),
        "sequence_minibatch_size": int(args.sequence_minibatch_size),
        "communication_chunk_size": int(args.communication_chunk_size),
        "min_model_replay_frames": int(args.min_model_replay_frames),
        "min_agent_replay_frames": int(args.min_agent_replay_frames),
        "max_imagination_starts": int(args.max_imagination_starts),
        "imagination_alive_threshold": float(args.imagination_alive_threshold),
        "action_information_coef": float(args.action_information_coef),
        "availability_coef": float(args.availability_coef),
        "availability_threshold": float(args.availability_threshold),
        "model_epochs": int(args.model_epochs),
        "agent_epochs": int(args.agent_epochs),
        "critic_epochs": int(args.critic_epochs),
        "ppo_epochs": int(args.ppo_epochs),
        "critic_warmup_updates": int(args.critic_warmup_updates),
        "imagination_reward_scope": str(args.imagination_reward_scope),
        "hidden_dim": int(args.hidden_dim),
        "model_hidden_dim": int(args.model_hidden_dim),
        "embed_dim": int(args.embed_dim),
        "deter_dim": int(args.deter_dim),
        "stoch_classes": int(args.stoch_classes),
        "stoch_categories": int(args.stoch_categories),
        "transformer_layers": int(args.transformer_layers),
        "attention_heads": int(args.attention_heads),
        "imagination_horizon": int(args.imag_horizon),
        "model_lr": float(args.model_lr),
        "actor_lr": float(args.actor_lr),
        "mamba_critic_lr": float(args.critic_lr),
        "gae_lambda": float(args.gae_lambda),
        "ppo_clip": float(args.ppo_clip),
        "target_policy_kl": float(args.target_policy_kl),
        "critic_huber_delta": float(args.critic_huber_delta),
        "target_tau": float(args.target_tau),
        "ac_advantage_clip": float(args.advantage_clip),
        "ac_normalize_advantages": bool(args.normalize_advantages),
        "entropy_coef": float(args.entropy_coef),
        "value_coef": float(args.value_coef),
        "max_grad_norm": float(args.max_grad_norm),
        "amp_dtype": str(args.amp_dtype),
        "av_execution_mode": str(args.av_execution_mode),
        "hdv_gawron_mode": str(args.hdv_gawron_mode),
        "hdv_gawron_alpha_zero": float(args.hdv_gawron_alpha_zero),
        "hdv_gawron_beta": float(args.hdv_gawron_beta),
        "hdv_gawron_update_mode": str(args.hdv_gawron_update_mode),
        "disable_human_learning_after_mutation": False,
        "real_policy_route_selection": "sample",
        "sumo_time_to_teleport": str(args.sumo_time_to_teleport),
        "sumo_extra_args": str(args.sumo_extra_args or ""),
    }
    # With --settings, omitted CLI paths mean "use the settings value", not
    # "erase it".  Explicit launcher paths still take precedence.
    if str(args.post_mutation_state_in or "").strip() or not has_settings:
        overrides["post_mutation_state_in"] = str(args.post_mutation_state_in or "")
    if str(args.hdv_state_in or "").strip() or not has_settings:
        overrides["hdv_state_in"] = str(args.hdv_state_in or "")
    if has_settings:
        return ToyMambaConfig.from_settings(str(args.settings), **overrides)
    return ToyMambaConfig(settings="", **overrides).resolve()


def build_plan(cfg: ToyMambaConfig) -> Dict[str, Any]:
    return {
        "baseline": "toy_cross_bottleneck_mamba",
        "algorithm": "mamba",
        "network": cfg.network_name,
        "setting_match": {
            "reward_mode": cfg.reward_mode,
            "reward_temperature": cfg.reward_temperature,
            "ratio_machines": cfg.ratio_machines,
            "episode_steps": cfg.episode_steps,
            "av_execution_mode": cfg.av_execution_mode,
            "av_candidate_source": cfg.av_candidate_source,
            "route_generator": cfg.route_generator,
            "post_mutation_state_in": cfg.post_mutation_state_in,
            "hdv_state_in": cfg.hdv_state_in,
            "human_learning_after_mutation": not cfg.disable_human_learning_after_mutation,
            "incident_edge": cfg.incident_edge,
            "incident_start": cfg.incident_start,
            "incident_end": cfg.incident_end,
            "incident_speed": cfg.incident_speed,
        },
        "mamba": {
            "fixed_identity_slots": True,
            "sparse_replay": True,
            "train_episodes": cfg.train_episodes,
            "checkpoint_interval": cfg.checkpoint_interval,
            "checkpoint_episodes": list(cfg.checkpoint_episodes),
            "update_every_episodes": cfg.update_every_episodes,
            "replay_episodes": cfg.replay_episodes,
            "sequence_length": cfg.sequence_length,
            "burn_in": cfg.burn_in,
            "sequence_minibatch_size": cfg.sequence_minibatch_size,
            "communication_chunk_size": cfg.communication_chunk_size,
            "min_model_replay_frames": cfg.min_model_replay_frames,
            "min_agent_replay_frames": cfg.min_agent_replay_frames,
            "max_imagination_starts": cfg.max_imagination_starts,
            "imagination_alive_threshold": cfg.imagination_alive_threshold,
            "action_information_coef": cfg.action_information_coef,
            "availability_coef": cfg.availability_coef,
            "availability_threshold": cfg.availability_threshold,
            "model_epochs": cfg.model_epochs,
            "agent_epochs": cfg.agent_epochs,
            "critic_epochs": cfg.critic_epochs,
            "ppo_epochs": cfg.ppo_epochs,
            "critic_warmup_updates": cfg.critic_warmup_updates,
            "imagination_reward_scope": cfg.imagination_reward_scope,
            "hidden_dim": cfg.hidden_dim,
            "model_hidden_dim": cfg.model_hidden_dim,
            "embed_dim": cfg.embed_dim,
            "deter_dim": cfg.deter_dim,
            "stoch_classes": cfg.stoch_classes,
            "stoch_categories": cfg.stoch_categories,
            "transformer_layers": cfg.transformer_layers,
            "attention_heads": cfg.attention_heads,
            "imagination_horizon": cfg.imagination_horizon,
            "normalize_advantages": cfg.ac_normalize_advantages,
            "advantage_clip": cfg.ac_advantage_clip,
            "ppo_clip": cfg.ppo_clip,
            "target_policy_kl": cfg.target_policy_kl,
            "critic_huber_delta": cfg.critic_huber_delta,
            "target_tau": cfg.target_tau,
            "amp_dtype": cfg.amp_dtype,
        },
        "paths": {
            "network_dir": cfg.network_dir,
            "edge_registry": cfg.edge_registry,
            "route_artifacts_dir": cfg.route_artifacts_dir,
            "post_mutation_state_in": cfg.post_mutation_state_in,
            "output_dir": cfg.output_dir,
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.generate_network:
        generate_toy_network(
            output_dir=Path(args.network_dir).resolve(),
            n_agents=int(args.n_agents),
            depart_span=int(args.depart_span),
            netconvert="",
        )
    cfg = build_config(args)
    plan = build_plan(cfg)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    validate_paths(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "toy_mamba_config.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable_config(cfg), handle, indent=2)
    with (output_dir / "toy_mamba_plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False)
    trainer = ToyMambaTrainer(cfg)
    summary = trainer.train()
    with (output_dir / "toy_mamba_final_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
