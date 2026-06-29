#!/usr/bin/env python3
"""Run online actor-critic on the toy cross-bottleneck SUMO network."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "dreamer4", REPO_ROOT / "dreamer4" / "urban_routing"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ablation.create_toy_cross_bottleneck_network import NETWORK_NAME as TOY_NETWORK_NAME
from ablation.create_toy_cross_bottleneck_network import generate as generate_toy_network
from ablation.online_no_cfm_actor_critic import (
    OnlineDecisionSample,
    OnlineNoCfmCollector,
    OnlineNoCfmConfig,
    _append_vehicle_context_features,
    _as_float,
    _as_int,
    _jsonable_config,
    _resolve_path,
    compute_a2c_loss,
    route_valid_mask_from_path_batch,
    samples_to_tensors,
    save_policy_checkpoint,
    selected_vehicle_route_rewards,
)
from ablation.traci_port_fallback import patch_traci_start_port_fallback
from dreamer4.urban_routing.dreamer_actor_critic import (
    CentralizedVehicleValueCritic,
    VehicleRouteActor,
)
from dreamer4.urban_routing.data_collector import serialize_hdv_route_state
from dreamer4.urban_routing.flow_projector import EdgeRegistry
from dreamer4.urban_routing.routerl_shared import compute_agent_counts, load_od_data
from dreamer4.urban_routing.train_marl import (
    build_vehicle_fleet_pressure_obs_features,
    build_vehicle_obs_vectorized,
    vehicle_mode_obs_dim,
)
from scripts.eval_sumo_hdv_state import (
    _load_hdv_learning_state,
    _merge_hdv_learning_state,
    _save_hdv_learning_state,
    _snapshot_hdv_learning_state,
)
from scripts.eval_sumo_io import _reuse_route_artifacts, _reuse_route_artifacts_from_refs
from scripts.eval_sumo_routing import (
    _edge_route_cost_matrix,
    _route_batch_uniqueness_summary,
    _temperature_adjust_probs,
)
from dreamer4.urban_routing.urb_fair_protocol import derive_fixed_roles, load_post_mutation_state


class ToyOnlineAcCollector(OnlineNoCfmCollector):
    """Online no-CFM collector whose actor observes live toy edge features."""

    def __init__(self, *, actor: VehicleRouteActor, cfg: OnlineNoCfmConfig, **collector_kwargs: Any):
        super().__init__(wm=None, actor=actor, cfg=cfg, **collector_kwargs)

    def _live_edge_features(self, state: Dict[str, Any]) -> np.ndarray:
        edge_state = state.get("edges", {}) if isinstance(state, dict) else {}
        feats = self._edge_registry.extract_flow(edge_state if isinstance(edge_state, dict) else {})
        return np.asarray(feats, dtype=np.float32)

    def _collect_single_episode(
        self,
        episode_id: int,
        max_steps: int,
    ) -> tuple[List[OnlineDecisionSample], Dict[str, Any]]:
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

        path_batch: List[List[List[int]]] = []
        route_batch: List[List[List[str]]] = []
        active_ids: List[str] = []
        active_current_edges: List[str] = []
        for veh_id in active_avs:
            veh_info = in_network.get(veh_id)
            if not isinstance(veh_info, dict):
                inc("skip_bad_vehicle_info")
                continue
            current_edge = str(veh_info.get("edge", ""))
            if (not current_edge) or current_edge.startswith(":"):
                inc("skip_bad_current_edge")
                continue
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
            "candidate_dynamic_count": 0,
            "candidate_fallback_count": int(len(active_ids)),
            "candidate_unique_mean": float(unique_mean),
            "candidate_unique_min": int(unique_min),
            "candidate_unique_max": int(unique_max),
            "reward_mean": float(np.mean(rewards)) if len(rewards) else float("nan"),
            "reward_sum": float(np.sum(rewards)) if len(rewards) else 0.0,
            "cfm_transition_used": 0,
            "wm_rollout_rerank_used": 0,
            "obs_source": "live_edge_registry_extract_flow",
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


class ToyOnlineAcTrainer:
    """Actor-critic trainer for the toy cross-bottleneck network."""

    def __init__(self, cfg: OnlineNoCfmConfig):
        self.cfg = cfg.resolve()
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(int(self.cfg.seed))
        np.random.seed(int(self.cfg.seed))
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
        self.collector: Optional[ToyOnlineAcCollector] = None
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
        print(f"[toy-online-ac] resumed actor/critic from {path} at episode {episode}", flush=True)
        return episode

    def _build_collector(self) -> ToyOnlineAcCollector:
        patch_traci_start_port_fallback()
        output_dir = self.output_dir / "collector_runtime"
        output_dir.mkdir(parents=True, exist_ok=True)
        post_state = None
        post_state_in = str(getattr(self.cfg, "post_mutation_state_in", "") or "").strip()
        post_state_path = Path(post_state_in).resolve() if post_state_in else None
        if post_state_path is not None and post_state_path.exists():
            post_state = load_post_mutation_state(post_state_path)
            reused = _reuse_route_artifacts_from_refs(post_state.get("artifact_refs", {}), output_dir)
        else:
            reused = _reuse_route_artifacts(Path(self.cfg.route_artifacts_dir).resolve(), output_dir)
        hdv_state_in = str(getattr(self.cfg, "hdv_state_in", "") or "").strip()
        if not hdv_state_in and post_state is not None:
            state_ref = str(post_state.get("human_learning_state_ref", "") or "").strip()
            if state_ref:
                hdv_state_in = state_ref
        loaded_hdv_state = _load_hdv_learning_state(Path(hdv_state_in)) if hdv_state_in else None
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
        collector = ToyOnlineAcCollector(
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
            machine_ids = [str(x) for x in post_state.get("machine_agent_ids", [])]
            human_ids = [str(x) for x in post_state.get("human_agent_ids", [])]
            role_source = str(post_state_path)
            if hasattr(collector.connector, "_episode_count"):
                collector.connector._episode_count = int(post_state.get("mutation_episode_index", 0) or 0)
        else:
            machine_ids, human_ids = derive_fixed_roles(
                agents_csv=Path(str(reused["agents_csv"])),
                ratio_machines=float(self.cfg.ratio_machines),
                seed=int(self.cfg.seed),
            )
            role_source = "derived_from_agents_csv"
        collector.set_fixed_roles(machine_ids=machine_ids, human_ids=human_ids)
        fixed_roles_manifest = {
            "source": role_source,
            "post_mutation_state_in": str(post_state_path) if post_state_path is not None else "",
            "agents_csv": str(reused["agents_csv"]),
            "ratio_machines": float(self.cfg.ratio_machines),
            "seed": int(self.cfg.seed),
            "human_learning_enabled": not bool(getattr(self.cfg, "disable_human_learning_after_mutation", False)),
            "machine_count": len(machine_ids),
            "human_count": len(human_ids),
            "total_count": len(machine_ids) + len(human_ids),
            "machine_agent_ids": list(machine_ids),
            "human_agent_ids": list(human_ids),
        }
        with (self.output_dir / "toy_online_ac_fixed_roles.json").open("w", encoding="utf-8") as f:
            json.dump(fixed_roles_manifest, f, indent=2)
        print(
            "[toy-online-ac] fixed roles: "
            f"machines={len(machine_ids)} humans={len(human_ids)} source={role_source}",
            flush=True,
        )
        collector._av_execution_mode = str(getattr(self.cfg, "av_execution_mode", "dynamic"))
        self.collector = collector
        self.rollout_max_steps = (
            int(self.cfg.max_steps_per_episode)
            if int(self.cfg.max_steps_per_episode) > 0
            else int(sim_length)
        )
        return collector

    def update_from_replay(self, episode_batches: Sequence[Sequence[OnlineDecisionSample]]) -> Dict[str, float]:
        samples: List[OnlineDecisionSample] = [sample for batch in episode_batches for sample in batch]
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

        def mean_result(name: str) -> float:
            values = [
                float(item[name].detach().cpu().item())
                for item in results
                if name in item and torch.is_tensor(item[name])
            ]
            return float(np.mean(values)) if values else 0.0

        return {
            "loss": mean_result("loss"),
            "policy_loss": mean_result("policy_loss"),
            "value_loss": mean_result("value_loss"),
            "entropy": mean_result("entropy"),
            "n_samples": float(result["n_samples"].detach().cpu().item()),
            "replay_episodes": float(len(episode_batches)),
            "update_epochs": float(update_epochs),
            "return_mean": mean_result("return_mean"),
            "return_std": mean_result("return_std"),
            "adv_mean": mean_result("adv_mean"),
            "adv_std": mean_result("adv_std"),
            "combined_adv_mean": mean_result("combined_adv_mean"),
            "combined_adv_std": mean_result("combined_adv_std"),
            "clip_frac": mean_result("clip_frac"),
            "selected_valid_frac": mean_result("selected_valid_frac"),
            "entropy_floor": mean_result("entropy_floor"),
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
                    f"[toy-online-ac] ep={ep:04d}/{int(self.cfg.train_episodes):04d} "
                    f"samples={len(samples)} reward_mean={row.get('reward_mean', float('nan')):.4f} "
                    f"loss={update['loss']:.4f} t_all={row.get('t_all', float('nan')):.3f} "
                    f"t_CAV={row.get('t_CAV', float('nan')):.3f} t_HDV={row.get('t_HDV', float('nan')):.3f} "
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
                    print(f"[toy-online-ac] saved checkpoint: {ckpt}", flush=True)
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
            "ablation": "toy_cross_bottleneck_online_actor_critic",
            "config": {
                **_jsonable_config(self.cfg),
                "av_execution_mode": str(getattr(self.cfg, "av_execution_mode", "dynamic")),
            },
            "obs_dim": int(self.obs_dim),
            "obs_source": "live_edge_registry_extract_flow",
            "checkpoint_paths": checkpoint_paths,
            "train_episode_csv": str(self.output_dir / "online_train_episodes.csv"),
            "train_update_csv": str(self.output_dir / "online_train_updates.csv"),
            "elapsed_sec": float(time.time() - start),
        }
        with (self.output_dir / "online_train_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


def parse_checkpoint_episodes(raw: str) -> tuple[int, ...]:
    out = []
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return tuple(sorted(set(out))) or (25, 50, 100)


def build_config(args: argparse.Namespace) -> OnlineNoCfmConfig:
    network_dir = Path(args.network_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        REPO_ROOT
        / "results"
        / TOY_NETWORK_NAME
        / f"online_ac_hdvlearn_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    cfg = OnlineNoCfmConfig(
        tag=TOY_NETWORK_NAME,
        settings="",
        output_dir=str(output_dir),
        network_name=TOY_NETWORK_NAME,
        network_dir=str(network_dir),
        edge_registry=str(network_dir / "edge_registry.json"),
        projector_ckpt="",
        dynamics_ckpt="",
        route_artifacts_dir=str(network_dir),
        post_mutation_state_in=str(getattr(args, "post_mutation_state_in", "") or ""),
        hdv_state_in=str(args.hdv_state_in or ""),
        ratio_machines=float(args.ratio_machines),
        n_routes=4,
        train_episodes=int(args.train_episodes),
        eval_episodes=0,
        checkpoint_episodes=parse_checkpoint_episodes(args.checkpoint_episodes),
        resume_checkpoint=str(args.resume_checkpoint or ""),
        resume_start_episode=int(args.resume_start_episode),
        episode_steps=int(args.episode_steps),
        sim_buffer_after_last_depart=int(args.sim_buffer_after_last_depart),
        max_steps_per_episode=int(args.max_steps_per_episode),
        seed=int(args.seed),
        device=str(args.device),
        reward_mode=str(args.reward_mode),
        reward_temperature=float(args.reward_temperature),
        gamma=1.0,
        lr=float(args.lr),
        critic_lr=float(args.critic_lr),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        ac_advantage_clip=float(args.advantage_clip),
        ac_ppo_clip_ratio=float(args.ppo_clip_ratio),
        ac_normalize_advantages=True,
        ac_entropy_target_frac=float(args.entropy_target_frac),
        ac_entropy_floor_coef=float(args.entropy_floor_coef),
        replay_buffer_episodes=int(args.replay_buffer_episodes),
        replay_update_epochs=int(args.replay_update_epochs),
        replay_actor_max_age_episodes=int(args.replay_actor_max_age_episodes),
        return_horizon_steps=int(args.return_horizon_steps),
        min_samples_per_update=1,
        hdv_gawron_mode=str(args.hdv_gawron_mode),
        hdv_gawron_alpha_zero=float(args.hdv_gawron_alpha_zero),
        hdv_gawron_beta=float(args.hdv_gawron_beta),
        hdv_gawron_update_mode=str(args.hdv_gawron_update_mode),
        disable_human_learning_after_mutation=False,
        incident_edge=str(args.incident_edge or ""),
        incident_start=float(args.incident_start),
        incident_end=float(args.incident_end),
        incident_speed=float(args.incident_speed),
        sumo_time_to_teleport=str(args.sumo_time_to_teleport),
        sumo_extra_args=str(args.sumo_extra_args or ""),
    )
    resolved = cfg.resolve()
    setattr(resolved, "av_execution_mode", str(args.av_execution_mode))
    return resolved


def validate_paths(cfg: OnlineNoCfmConfig) -> None:
    required = [
        Path(cfg.network_dir) / f"{cfg.network_name}.net.xml",
        Path(cfg.network_dir) / f"od_{cfg.network_name}.txt",
        Path(cfg.edge_registry),
        Path(cfg.route_artifacts_dir) / "agents.csv",
        Path(cfg.route_artifacts_dir) / "routes.csv",
        Path(cfg.route_artifacts_dir) / "route.rou.xml",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing toy network artifacts: " + ", ".join(missing))


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-dir", default=str(REPO_ROOT / "networks" / TOY_NETWORK_NAME))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--generate-network", action="store_true", help="Generate toy network artifacts before training.")
    parser.add_argument("--n-agents", type=int, default=320)
    parser.add_argument("--depart-span", type=int, default=420)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--checkpoint-episodes", default="25,50,100")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-start-episode", type=int, default=0)
    parser.add_argument(
        "--post-mutation-state-in",
        default="",
        help=(
            "Optional URB-fair post_mutation_state.json. When omitted, toy online AC "
            "still derives deterministic fixed AV/HDV roles from agents.csv."
        ),
    )
    parser.add_argument("--hdv-state-in", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio-machines", type=float, default=0.4)
    parser.add_argument("--episode-steps", type=int, default=60)
    parser.add_argument("--sim-buffer-after-last-depart", type=int, default=3000)
    parser.add_argument("--max-steps-per-episode", type=int, default=0)
    parser.add_argument("--reward-mode", default="capped_step_cost", choices=["progress_fraction", "capped_step_cost"])
    parser.add_argument("--reward-temperature", type=float, default=60.0)
    parser.add_argument("--replay-buffer-episodes", type=int, default=8)
    parser.add_argument("--replay-update-epochs", type=int, default=2)
    parser.add_argument("--replay-actor-max-age-episodes", type=int, default=8)
    parser.add_argument("--return-horizon-steps", type=int, default=0)
    parser.add_argument("--av-execution-mode", default="dynamic", choices=["dynamic", "first_decision_only"])
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    network_dir = Path(args.network_dir).resolve()
    if args.generate_network:
        generate_toy_network(
            output_dir=network_dir,
            n_agents=int(args.n_agents),
            depart_span=int(args.depart_span),
            netconvert="",
        )
    cfg = build_config(args)
    plan = {
        "ablation": "toy_cross_bottleneck_online_actor_critic",
        "network": cfg.network_name,
        "setting_match": {
            "reward_mode": cfg.reward_mode,
            "reward_temperature": cfg.reward_temperature,
            "ratio_machines": cfg.ratio_machines,
            "episode_steps": cfg.episode_steps,
            "av_execution_mode": str(getattr(cfg, "av_execution_mode", "dynamic")),
            "post_mutation_state_in": cfg.post_mutation_state_in,
            "replay_buffer_episodes": cfg.replay_buffer_episodes,
            "replay_update_epochs": cfg.replay_update_epochs,
            "return_horizon_steps": cfg.return_horizon_steps,
            "hdv_learning": cfg.hdv_gawron_mode,
            "incident_edge": cfg.incident_edge,
        },
        "paths": {
            "network_dir": cfg.network_dir,
            "edge_registry": cfg.edge_registry,
            "route_artifacts_dir": cfg.route_artifacts_dir,
            "post_mutation_state_in": cfg.post_mutation_state_in,
            "output_dir": cfg.output_dir,
        },
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    validate_paths(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "toy_online_ac_config.json").open("w", encoding="utf-8") as f:
        cfg_json = _jsonable_config(cfg)
        cfg_json["av_execution_mode"] = str(getattr(cfg, "av_execution_mode", "dynamic"))
        json.dump(cfg_json, f, indent=2)
    with (output_dir / "toy_online_ac_plan.json").open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    trainer = ToyOnlineAcTrainer(cfg)
    summary = trainer.train()
    with (output_dir / "toy_online_ac_final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
