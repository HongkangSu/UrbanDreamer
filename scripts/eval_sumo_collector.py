"""Policy-guided RouteRL collector used by SUMO evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # noqa: E731

try:
    from data_collector import DataCollector
except ImportError:
    from dreamer4.urban_routing.data_collector import DataCollector

try:
    from train_marl import (
        build_diverse_paths_current_to_dest,
        build_vehicle_fleet_pressure_obs_features,
        build_vehicle_obs_vectorized,
        edge_tt_from_latent_state,
    )
except Exception:
    try:
        from dreamer4.urban_routing.train_marl import (
            build_diverse_paths_current_to_dest,
            build_vehicle_fleet_pressure_obs_features,
            build_vehicle_obs_vectorized,
            edge_tt_from_latent_state,
        )
    except Exception:
        build_diverse_paths_current_to_dest = None
        build_vehicle_fleet_pressure_obs_features = None
        build_vehicle_obs_vectorized = None
        edge_tt_from_latent_state = None

try:
    from wm_reranker import rerank_topk_candidates, rollout_candidate_metrics
except Exception:
    from dreamer4.urban_routing.wm_reranker import rerank_topk_candidates, rollout_candidate_metrics

try:
    from scripts.eval_sumo_io import (
        _copy_artifact_if_needed,
        _episode_complete_for_dataset,
        _new_sumo_xmls_since,
        _snapshot_sumo_xmls,
        _wait_for_expected_sumo_xml,
    )
    from scripts.eval_sumo_hdv_state import _restore_hdv_learning_state
    from scripts.eval_sumo_routing import (
        _build_edge_adj_list,
        _edge_route_cost_matrix,
        _route_batch_uniqueness_summary,
        _route_cost_matrix_summary,
        _select_min_cost_route_indices_with_planned_pressure,
        _select_quota_route_indices,
        _temperature_adjust_probs,
    )
except ImportError:
    from eval_sumo_io import (
        _copy_artifact_if_needed,
        _episode_complete_for_dataset,
        _new_sumo_xmls_since,
        _snapshot_sumo_xmls,
        _wait_for_expected_sumo_xml,
    )
    from eval_sumo_hdv_state import _restore_hdv_learning_state
    from eval_sumo_routing import (
        _build_edge_adj_list,
        _edge_route_cost_matrix,
        _route_batch_uniqueness_summary,
        _route_cost_matrix_summary,
        _select_min_cost_route_indices_with_planned_pressure,
        _select_quota_route_indices,
        _temperature_adjust_probs,
    )


def _append_vehicle_context_features(
    obs_vehicle_np: np.ndarray,
    *,
    step_idx: int,
    episode_steps: int,
    n_active_av: int,
    n_total_av: int,
) -> np.ndarray:
    """Match train_marl vehicle-mode obs shape by appending 3 context features."""
    if obs_vehicle_np.ndim != 2:
        return obs_vehicle_np
    context_features = np.array([
        float(step_idx) / max(int(episode_steps), 1),
        float(n_active_av) / max(int(n_total_av), 1),
        0.0,
    ], dtype=np.float32)
    if obs_vehicle_np.shape[0] == 0:
        return np.zeros((0, obs_vehicle_np.shape[1] + 3), dtype=np.float32)
    ctx_tiled = np.tile(context_features, (obs_vehicle_np.shape[0], 1))
    return np.concatenate([obs_vehicle_np, ctx_tiled], axis=1)


class PolicyGuidedCollectorEval(DataCollector):
    """SUMO evaluator that reuses the DataCollector rollout loop."""

    def __init__(
        self,
        *,
        wm,
        policy,
        deterministic: bool,
        episode_steps: int,
        disable_human_learning_after_mutation: bool = True,
        mpc_horizon: int = 0,
        mpc_candidates: int = 5,
        mpc_discount: float = 0.99,
        rerank_t_all_weight: float = 1.0,
        rerank_t_cav_weight: float = 2.0,
        rerank_cav_tolerance: float = 0.0,
        wm_volume_floor: float = 0.10,
        loaded_hdv_state: Optional[Dict[str, Any]] = None,
        av_od_route_overrides: Optional[Dict[Tuple[int, int], int]] = None,
        av_policy_route_selection: str = "auto",
        av_policy_sample_temperature: float = 1.0,
        av_policy_confidence_gate_source: str = "none",
        av_policy_confidence_gate_margin: float = 0.0,
        av_policy_confidence_gate_trigger_share: float = 0.0,
        av_policy_nonzero_route_share_cap: float = 0.0,
        av_policy_share_cap_fallback_source: str = "wm_current_bpr",
        av_candidate_source: str = "current_shortest_wm",
        av_candidate_path_penalty_factor: float = 3.0,
        av_route_cost_policy: str = "none",
        av_route_cost_length_norm: str = "sum",
        av_route_cost_temperature: float = 0.0,
        av_route_hdv_pressure_penalty: float = 0.0,
        av_route_wm_live_blend_weight: float = 0.5,
        av_route_planned_av_pressure_penalty: float = 0.0,
        wm_mcc_diagnostic_penalties: Optional[Sequence[float]] = None,
        wm_mcc_diagnostic_horizon: int = 1,
        bpr_alpha: float = 0.15,
        bpr_beta: float = 4.0,
        wm_edge_cost_source: str = "decoded_speed",
        wm_candidate_edge_cost_source: str = "auto",
        wm_decoded_speed_min_ratio: float = 0.05,
        wm_decoded_speed_max_ratio: float = 1.50,
        wm_decoded_speed_excess_scale: float = 1.0,
        vehicle_obs_route_stats: str = "mean",
        vehicle_obs_include_route_costs: bool = False,
        vehicle_obs_include_fleet_pressure: bool = False,
        **collector_kwargs,
    ):
        collector_kwargs.setdefault("loaded_hdv_state", loaded_hdv_state)
        eval_candidate_source = str(av_candidate_source or "current_shortest_wm").strip().lower()
        base_candidate_source = "od_suffix" if eval_candidate_source == "od_suffix" else "current_to_dest"
        collector_kwargs.setdefault("av_candidate_source", base_candidate_source)
        super().__init__(**collector_kwargs)
        self._wm_eval = wm
        self._policy_eval = policy
        self._policy_eval_device = next(policy.parameters()).device
        self._deterministic_eval = bool(deterministic)
        self._episode_steps_eval = max(int(episode_steps), 1)
        requested_h = max(int(mpc_horizon), 0)
        self._mpc_horizon = requested_h
        self._mpc_horizon_requested = requested_h
        self._mpc_candidates = max(int(mpc_candidates), 1)
        self._mpc_discount = float(mpc_discount)
        self._rerank_t_all_weight = float(rerank_t_all_weight)
        self._rerank_t_cav_weight = float(rerank_t_cav_weight)
        self._rerank_cav_tolerance = float(rerank_cav_tolerance)
        self._wm_volume_floor = max(float(wm_volume_floor), 0.0)
        self._rerank_enabled = self._mpc_horizon > 0 and self._mpc_candidates > 1
        self._av_od_route_overrides = {
            (int(k[0]), int(k[1])): int(v)
            for k, v in dict(av_od_route_overrides or {}).items()
        }
        self._av_policy_route_selection = str(av_policy_route_selection or "auto").strip().lower()
        if self._av_policy_route_selection not in {"auto", "argmax", "sample", "quota"}:
            raise ValueError(f"Unsupported av_policy_route_selection={av_policy_route_selection!r}")
        self._av_policy_sample_temperature = max(float(av_policy_sample_temperature), 1e-6)
        self._av_policy_confidence_gate_source = str(
            av_policy_confidence_gate_source or "none"
        ).strip().lower()
        if self._av_policy_confidence_gate_source not in {"none", "wm_current_bpr", "live_current_tt", "freeflow_greedy"}:
            raise ValueError(
                f"Unsupported av_policy_confidence_gate_source={av_policy_confidence_gate_source!r}"
            )
        self._av_policy_confidence_gate_margin = max(float(av_policy_confidence_gate_margin), 0.0)
        self._av_policy_confidence_gate_trigger_share = float(
            np.clip(float(av_policy_confidence_gate_trigger_share), 0.0, 1.0)
        )
        self._av_policy_nonzero_route_share_cap = float(
            np.clip(float(av_policy_nonzero_route_share_cap), 0.0, 1.0)
        )
        self._av_policy_share_cap_fallback_source = str(
            av_policy_share_cap_fallback_source or "wm_current_bpr"
        ).strip().lower()
        if self._av_policy_share_cap_fallback_source not in {"wm_current_bpr", "live_current_tt", "freeflow_greedy"}:
            raise ValueError(
                f"Unsupported av_policy_share_cap_fallback_source={av_policy_share_cap_fallback_source!r}"
            )
        self._av_candidate_source = eval_candidate_source
        if self._av_candidate_source not in {"od_suffix", "current_shortest_live", "current_shortest_wm", "current_shortest_freeflow"}:
            raise ValueError(f"Unsupported av_candidate_source={av_candidate_source!r}")
        self._av_candidate_path_penalty_factor = max(float(av_candidate_path_penalty_factor), 1.0)
        edge_registry = getattr(self, "_edge_registry", None)
        self._av_candidate_adj_list = _build_edge_adj_list(
            getattr(edge_registry, "adjacency", None),
            getattr(edge_registry, "passenger_allowed", None),
        )
        self._av_route_cost_policy = str(av_route_cost_policy or "none").strip().lower()
        if self._av_route_cost_policy not in {"none", "freeflow_greedy", "wm_current_bpr", "live_current_tt", "wm_live_blend"}:
            raise ValueError(f"Unsupported av_route_cost_policy={av_route_cost_policy!r}")
        self._av_route_cost_length_norm = str(av_route_cost_length_norm or "sum").strip().lower()
        if self._av_route_cost_length_norm not in {"sum", "mean"}:
            raise ValueError(f"Unsupported av_route_cost_length_norm={av_route_cost_length_norm!r}")
        self._av_route_cost_temperature = max(float(av_route_cost_temperature), 0.0)
        self._av_route_hdv_pressure_penalty = max(float(av_route_hdv_pressure_penalty), 0.0)
        self._av_route_wm_live_blend_weight = float(
            np.clip(float(av_route_wm_live_blend_weight), 0.0, 1.0)
        )
        self._av_route_planned_av_pressure_penalty = max(
            float(av_route_planned_av_pressure_penalty),
            0.0,
        )
        self._wm_mcc_diagnostic_penalties = [
            max(float(v), 0.0)
            for v in list(wm_mcc_diagnostic_penalties or [])
        ]
        self._wm_mcc_diagnostic_horizon = max(int(wm_mcc_diagnostic_horizon), 1)
        self._bpr_alpha_eval = float(bpr_alpha)
        self._bpr_beta_eval = float(bpr_beta)
        self._wm_edge_cost_source = str(wm_edge_cost_source or "decoded_speed").strip().lower()
        if self._wm_edge_cost_source not in {"bpr_volume", "decoded_speed", "route_head"}:
            raise ValueError(f"Unsupported wm_edge_cost_source={wm_edge_cost_source!r}")
        self._wm_candidate_edge_cost_source = str(wm_candidate_edge_cost_source or "auto").strip().lower()
        if self._wm_candidate_edge_cost_source == "auto":
            self._wm_candidate_edge_cost_source = "freeflow"
        if self._wm_candidate_edge_cost_source in {"free_flow", "free-flow", "ff"}:
            self._wm_candidate_edge_cost_source = "freeflow"
        if self._wm_candidate_edge_cost_source not in {"freeflow", "bpr_volume", "decoded_speed", "route_head"}:
            raise ValueError(f"Unsupported wm_candidate_edge_cost_source={wm_candidate_edge_cost_source!r}")
        self._wm_decoded_speed_min_ratio = max(float(wm_decoded_speed_min_ratio), 1e-4)
        self._wm_decoded_speed_max_ratio = max(
            float(wm_decoded_speed_max_ratio),
            self._wm_decoded_speed_min_ratio,
        )
        self._wm_decoded_speed_excess_scale = max(float(wm_decoded_speed_excess_scale), 0.0)
        self._vehicle_obs_route_stats = str(vehicle_obs_route_stats or "mean").strip().lower()
        self._vehicle_obs_include_route_costs = bool(vehicle_obs_include_route_costs)
        self._vehicle_obs_include_fleet_pressure = bool(vehicle_obs_include_fleet_pressure)
        self._loaded_hdv_state_eval = loaded_hdv_state
        self._use_policy_actions = True
        self._disable_human_learning_after_mutation_eval = bool(
            disable_human_learning_after_mutation
        )
        self._current_decision_idx_eval = 0
        self._current_episode_id_eval = -1
        self._last_ep_av_travel_times: List[float] = []
        self._last_ep_hdv_travel_times: List[float] = []
        self._last_ep_vehicle_tt: Dict[str, float] = {}
        self._protocol_xml_alias_map: Dict[int, int] = {}
        self._dynamic_decision_rows: List[Dict[str, Any]] = []
        self._dynamic_agent_decision_rows: List[Dict[str, Any]] = []
        self._wm_mcc_diagnostic_rows: List[Dict[str, Any]] = []

    def _should_update_hdv_after_episode(self, desc: str) -> bool:
        if self._hdv_route_probs is None or self._route_registry is None:
            return False
        if (
            bool(self._disable_human_learning_after_mutation_eval)
            and str(desc).strip().lower() == "test"
        ):
            return False
        return True

    def _write_urb_episode_csv(self, episode_id: int, episode_reward: float) -> Optional[Path]:
        agents_df = getattr(self.connector, "_agents_df", None)
        if agents_df is None or len(agents_df) == 0:
            return None

        records_folder = Path(getattr(self.connector, "records_folder", ""))
        if not records_folder:
            return None
        episodes_dir = records_folder / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        initial_route_choices = getattr(self.connector, "_episode_route_choices", {}) or {}
        fixed_machine_ids = set(getattr(self, "_fixed_machine_ids", set()) or set())
        fixed_human_ids = set(getattr(self, "_fixed_human_ids", set()) or set())
        for _, row in agents_df.iterrows():
            veh_id = str(row.get("id"))
            policy_route_idx = self._av_route_idx.get(veh_id, -1)
            hdv_route_idx = self._hdv_route_idx.get(veh_id, -1)
            route_idx = self._av_route_idx.get(veh_id, self._hdv_route_idx.get(veh_id, -1))
            initial_route_idx = int(initial_route_choices.get(veh_id, -1))
            if veh_id in fixed_machine_ids or veh_id in self._av_vehicles:
                vtype = "AV"
            elif veh_id in fixed_human_ids or veh_id in self._hdv_vehicles:
                vtype = "Human"
            else:
                vtype = str(row.get("kind", "Human"))
            rows.append(
                {
                    "id": int(float(row.get("id", -1))),
                    "travel_time": float(self._last_ep_vehicle_tt.get(veh_id, np.nan)),
                    "action": int(route_idx),
                    "initial_route_action": int(initial_route_idx),
                    "policy_action": int(policy_route_idx),
                    "hdv_action": int(hdv_route_idx),
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
            runtime_episode_id = int(suffix)
            protocol_episode_id = self._protocol_xml_alias_map.get(runtime_episode_id)
            if protocol_episode_id is None:
                continue
            _wait_for_expected_sumo_xml(src)
            dst = src.with_name(f"{prefix}_{protocol_episode_id}{src.suffix}")
            _copy_artifact_if_needed(src, dst)

    def _wm_current_edge_tt(self, z_obs: torch.Tensor, edge_cost_source: Optional[str] = None) -> Optional[np.ndarray]:
        if edge_tt_from_latent_state is None:
            return None
        try:
            _edge_dyn, _v_e, _free_flow, edge_tt = edge_tt_from_latent_state(
                wm=self._wm_eval,
                z=z_obs,
                bpr_alpha=float(self._bpr_alpha_eval),
                bpr_beta=float(self._bpr_beta_eval),
                edge_cost_source=str(edge_cost_source or self._wm_edge_cost_source),
                decoded_speed_min_ratio=self._wm_decoded_speed_min_ratio,
                decoded_speed_max_ratio=self._wm_decoded_speed_max_ratio,
                decoded_speed_excess_scale=self._wm_decoded_speed_excess_scale,
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
        if lengths is None:
            lengths_arr = np.ones((len(edge_ids),), dtype=np.float32) * 100.0
        else:
            lengths_arr = np.asarray(lengths, dtype=np.float32).reshape(-1)
        speed_limits = np.asarray(
            getattr(self._edge_registry, "speed_limits", np.ones((len(edge_ids),), dtype=np.float32) * 13.89),
            dtype=np.float32,
        ).reshape(-1)
        n = min(len(edge_ids), int(lengths_arr.shape[0]), int(speed_limits.shape[0]))
        if n <= 0:
            return None
        edge_state = state.get("edges", {}) if isinstance(state, dict) else {}
        out = np.zeros((n,), dtype=np.float32)
        for i, eid in enumerate(edge_ids[:n]):
            data = edge_state.get(eid, {}) if isinstance(edge_state, dict) else {}
            speed = float(data.get("mean_speed", 0.0) or 0.0) if isinstance(data, dict) else 0.0
            if not np.isfinite(speed) or speed <= 0.1:
                speed = float(speed_limits[i])
            speed = max(speed, 0.1)
            out[i] = float(lengths_arr[i]) / speed
        return out

    def _freeflow_edge_tt(self) -> Optional[np.ndarray]:
        edge_ids = list(getattr(self._edge_registry, "edge_ids", []))
        if not edge_ids:
            return None
        lengths = getattr(self._edge_registry, "lengths", None)
        speed_limits = getattr(self._edge_registry, "speed_limits", None)
        if lengths is None or speed_limits is None:
            return None
        lengths_arr = np.asarray(lengths, dtype=np.float32).reshape(-1)
        speed_limits_arr = np.asarray(speed_limits, dtype=np.float32).reshape(-1)
        n = min(len(edge_ids), int(lengths_arr.shape[0]), int(speed_limits_arr.shape[0]))
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
        if not self._av_candidate_adj_list:
            return []
        if edge_cost is None:
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
        n_edges = len(edge_ids)
        costs = np.asarray(edge_cost, dtype=np.float32).reshape(-1)
        if n_edges <= 0 or costs.shape[0] < n_edges:
            return []

        src_idx = int(edge_to_idx[current_edge])
        dst_idx = int(edge_to_idx[dest_edge])
        paths = build_diverse_paths_current_to_dest(
            adj_list=self._av_candidate_adj_list,
            edge_tt=costs[:n_edges],
            src_idx=src_idx,
            dst_idx=dst_idx,
            K=int(self.n_routes),
            penalty_factor=float(self._av_candidate_path_penalty_factor),
        )
        named_paths: List[List[str]] = []
        for path in paths[: int(self.n_routes)]:
            route = [
                edge_ids[int(edge_idx)]
                for edge_idx in path
                if 0 <= int(edge_idx) < n_edges
            ]
            if route and route[0] == current_edge:
                named_paths.append(route)
        if not named_paths:
            return []
        while len(named_paths) < int(self.n_routes):
            named_paths.append(list(named_paths[0]))
        return named_paths[: int(self.n_routes)]

    def _record_wm_mcc_diagnostics(
        self,
        *,
        z_obs: torch.Tensor,
        path_batch: List[List[List[int]]],
        baseline_route_indices: np.ndarray,
        live_route_cost_matrix: Optional[np.ndarray],
        depart_np: np.ndarray,
        context_np: np.ndarray,
        hdv_edge_np: np.ndarray,
        current_edges_np: np.ndarray,
        action_horizon_seconds: float,
        rng: np.random.RandomState,
    ) -> None:
        penalties = list(getattr(self, "_wm_mcc_diagnostic_penalties", []) or [])
        if not penalties or not path_batch:
            return

        def _append_metrics_row(
            *,
            candidate_kind: str,
            route_indices: np.ndarray,
            applied: int,
            penalty: float = float("nan"),
        ) -> None:
            metrics = rollout_candidate_metrics(
                wm=self._wm_eval,
                z_current=z_obs,
                path_batch=path_batch,
                route_indices=route_indices,
                horizon=int(self._wm_mcc_diagnostic_horizon),
                discount=float(self._mpc_discount),
                depart_edge_np=depart_np,
                context_np=context_np,
                hdv_edge_np=hdv_edge_np,
                total_av_count=max(float(len(self._av_vehicles)), 1.0),
                total_hdv_count=max(float(len(self._hdv_vehicles)), 1.0),
                volume_floor=float(self._wm_volume_floor),
                edge_cost_source=str(self._wm_edge_cost_source),
                current_edges=current_edges_np,
                action_horizon_seconds=float(action_horizon_seconds),
                decoded_speed_min_ratio=float(self._wm_decoded_speed_min_ratio),
                decoded_speed_max_ratio=float(self._wm_decoded_speed_max_ratio),
                decoded_speed_excess_scale=float(self._wm_decoded_speed_excess_scale),
            )
            counts = np.bincount(
                np.asarray(route_indices, dtype=np.int64).clip(0, int(self.n_routes) - 1),
                minlength=int(self.n_routes),
            )
            row: Dict[str, Any] = {
                "episode": int(self._current_episode_id_eval),
                "decision_idx": int(self._current_decision_idx_eval),
                "sim_step": int(getattr(self.connector, "current_step", -1)),
                "active_av": int(len(route_indices)),
                "candidate_kind": str(candidate_kind),
                "penalty": float(penalty),
                "applied": int(applied),
                "pred_t_all": float(metrics.get("pred_t_all", float("nan"))),
                "pred_t_all_pressure": float(metrics.get("pred_t_all_pressure", float("nan"))),
                "pred_t_all_head": float(metrics.get("pred_t_all_head", float("nan"))),
                "pred_t_all_decoded": float(metrics.get("pred_t_all_decoded", float("nan"))),
                "pred_t_cav_proxy": float(metrics.get("pred_t_cav_proxy", float("nan"))),
                "horizon": int(self._wm_mcc_diagnostic_horizon),
            }
            for k in range(int(self.n_routes)):
                row[f"selected_count_route{k}"] = int(counts[k])
            self._wm_mcc_diagnostic_rows.append(row)

        for penalty in penalties:
            try:
                if live_route_cost_matrix is None:
                    continue
                route_indices, applied = _select_min_cost_route_indices_with_planned_pressure(
                    live_route_cost_matrix,
                    baseline_route_indices,
                    path_batch,
                    n_edges=len(getattr(self._edge_registry, "edge_ids", [])),
                    deterministic=True,
                    rng=rng,
                    temperature=0.0,
                    pressure_penalty=float(penalty),
                )
                _append_metrics_row(
                    candidate_kind="penalty",
                    route_indices=route_indices,
                    applied=int(applied),
                    penalty=float(penalty),
                )
            except Exception as exc:
                self._wm_mcc_diagnostic_rows.append(
                    {
                        "episode": int(self._current_episode_id_eval),
                        "decision_idx": int(self._current_decision_idx_eval),
                        "sim_step": int(getattr(self.connector, "current_step", -1)),
                        "active_av": int(len(baseline_route_indices)),
                        "candidate_kind": "penalty",
                        "penalty": float(penalty),
                        "applied": 0,
                        "pred_t_all": float("nan"),
                        "pred_t_all_pressure": float("nan"),
                        "pred_t_all_head": float("nan"),
                        "pred_t_all_decoded": float("nan"),
                        "pred_t_cav_proxy": float("nan"),
                        "horizon": int(self._wm_mcc_diagnostic_horizon),
                        "error": str(exc),
                    }
                )
    def _select_av_actions(
        self, state: Dict[str, Any], rng: np.random.RandomState
    ) -> Dict[str, List[str]]:
        if not self._use_policy_actions:
            return {}

        in_network = state.get("vehicles", {})
        active_avs = sorted(self._av_vehicles & set(in_network.keys()))
        if not active_avs:
            return {}

        try:
            z_obs = self._wm_eval.encode_observation(state)
            edge_feats_dyn = (
                self._wm_eval.latent_edge_features(z_obs)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        except Exception:
            return {}

        edge_to_idx = getattr(self._edge_registry, "_edge_to_idx", {})
        wm_edge_cost = self._wm_current_edge_tt(z_obs)
        live_edge_cost = self._live_current_edge_tt(state)
        freeflow_edge_cost = self._freeflow_edge_tt()
        wm_candidate_edge_cost = None
        if self._wm_candidate_edge_cost_source == "freeflow":
            wm_candidate_edge_cost = freeflow_edge_cost
        elif self._wm_candidate_edge_cost_source == self._wm_edge_cost_source:
            wm_candidate_edge_cost = wm_edge_cost
        else:
            wm_candidate_edge_cost = self._wm_current_edge_tt(z_obs, edge_cost_source=self._wm_candidate_edge_cost_source)
        candidate_edge_cost: Optional[np.ndarray] = None
        if self._av_candidate_source == "current_shortest_wm":
            candidate_edge_cost = wm_candidate_edge_cost
        elif self._av_candidate_source == "current_shortest_live":
            candidate_edge_cost = live_edge_cost
        elif self._av_candidate_source == "current_shortest_freeflow":
            candidate_edge_cost = freeflow_edge_cost
        path_batch: List[List[List[int]]] = []
        route_batch: List[List[List[str]]] = []
        active_ids: List[str] = []
        active_current_edges: List[str] = []
        candidate_dynamic_count = 0
        candidate_fallback_count = 0

        for veh_id in active_avs:
            veh_info = in_network.get(veh_id)
            if not isinstance(veh_info, dict):
                continue
            current_edge = str(veh_info.get("edge", ""))
            if (not current_edge) or current_edge.startswith(":"):
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
                continue

            idx_paths: List[List[int]] = []
            named_paths: List[List[str]] = []
            for route in candidates[: self.n_routes]:
                idxs = [int(edge_to_idx[e]) for e in route if e in edge_to_idx]
                if idxs:
                    idx_paths.append(idxs)
                    named_paths.append(list(route))
            if not idx_paths:
                continue
            while len(idx_paths) < self.n_routes:
                idx_paths.append(list(idx_paths[0]))
                named_paths.append(list(named_paths[0]))
            path_batch.append(idx_paths[: self.n_routes])
            route_batch.append(named_paths[: self.n_routes])
            active_ids.append(veh_id)
            active_current_edges.append(current_edge)

        if not path_batch:
            return {}

        candidate_unique_mean, candidate_unique_min, candidate_unique_max = _route_batch_uniqueness_summary(
            route_batch
        )

        obs_vehicle_base_np = build_vehicle_obs_vectorized(
            edge_feats_dyn=edge_feats_dyn,
            path_batch=path_batch,
            include_global=True,
            route_stats=self._vehicle_obs_route_stats,
            edge_cost=wm_edge_cost,
            include_route_costs=self._vehicle_obs_include_route_costs,
            include_fleet_pressure=False,
        )
        if self._vehicle_obs_include_fleet_pressure:
            if build_vehicle_fleet_pressure_obs_features is None:
                return {}
            eval_selection_mode = str(self._av_policy_route_selection)
            if eval_selection_mode == "auto":
                eval_selection_mode = "argmax" if self._deterministic_eval else "sample"
            fleet_counts_np = np.zeros((int(self.n_routes),), dtype=np.float32)
            fleet_pressure_np = np.zeros((len(getattr(self._edge_registry, "edge_ids", [])),), dtype=np.float32)
            av_frac_rows: List[np.ndarray] = []
            for i in range(len(active_ids)):
                fleet_i = build_vehicle_fleet_pressure_obs_features(
                    path_batch=[path_batch[i]],
                    n_edges=len(getattr(self._edge_registry, "edge_ids", [])),
                    n_routes=int(self.n_routes),
                    selected_route_counts=fleet_counts_np,
                    selected_edge_pressure=fleet_pressure_np,
                )
                obs_i_np = _append_vehicle_context_features(
                    np.concatenate([obs_vehicle_base_np[i:i + 1], fleet_i], axis=1),
                    step_idx=int(getattr(self.connector, "current_step", 0)),
                    episode_steps=self._episode_steps_eval,
                    n_active_av=len(active_ids),
                    n_total_av=max(len(self._av_vehicles), 1),
                )
                obs_i = torch.from_numpy(obs_i_np).to(
                    device=self._policy_eval_device,
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    frac_i = self._policy_eval.get_av_frac(
                        obs_i,
                        deterministic=self._deterministic_eval,
                    ).detach().cpu().numpy().astype(np.float32)[0]
                probs_i = _temperature_adjust_probs(
                    frac_i,
                    self._av_policy_sample_temperature,
                    self.n_routes,
                )
                if eval_selection_mode == "sample":
                    selected_i = int(rng.choice(self.n_routes, p=probs_i))
                else:
                    selected_i = int(np.argmax(probs_i))
                selected_i = int(np.clip(selected_i, 0, len(route_batch[i]) - 1))
                av_frac_rows.append(frac_i)
                fleet_counts_np[selected_i] += 1.0
                if selected_i < len(path_batch[i]):
                    for edge_idx in path_batch[i][selected_i]:
                        e = int(edge_idx)
                        if 0 <= e < int(fleet_pressure_np.shape[0]):
                            fleet_pressure_np[e] += 1.0
            av_fracs_np = np.stack(av_frac_rows, axis=0).astype(np.float32)
        else:
            obs_vehicle_np = _append_vehicle_context_features(
                obs_vehicle_base_np,
                step_idx=int(getattr(self.connector, "current_step", 0)),
                episode_steps=self._episode_steps_eval,
                n_active_av=len(active_ids),
                n_total_av=max(len(self._av_vehicles), 1),
            )
            obs_online = torch.from_numpy(obs_vehicle_np).to(
                device=self._policy_eval_device,
                dtype=torch.float32,
            )
            with torch.no_grad():
                av_fracs = self._policy_eval.get_av_frac(
                    obs_online,
                    deterministic=self._deterministic_eval,
                )
            av_fracs_np = av_fracs.detach().cpu().numpy().astype(np.float32)

        baseline_route_indices = np.zeros((len(active_ids),), dtype=np.int64)
        selection_mode = str(self._av_policy_route_selection)
        if selection_mode == "auto":
            selection_mode = "argmax" if self._deterministic_eval else "sample"
        if selection_mode == "quota":
            av_fracs_for_selection = np.stack([
                _temperature_adjust_probs(row, self._av_policy_sample_temperature, self.n_routes)
                for row in av_fracs_np
            ], axis=0).astype(np.float32)
            argmax_baseline = np.zeros((len(active_ids),), dtype=np.int64)
            for i in range(len(active_ids)):
                probs = av_fracs_for_selection[i]
                argmax_baseline[i] = int(np.clip(int(np.argmax(probs)), 0, len(route_batch[i]) - 1))
            baseline_route_indices = _select_quota_route_indices(
                av_fracs_for_selection,
                argmax_baseline,
                n_routes=int(self.n_routes),
            )
            for i in range(len(active_ids)):
                baseline_route_indices[i] = int(np.clip(baseline_route_indices[i], 0, len(route_batch[i]) - 1))
        else:
            for i in range(len(active_ids)):
                probs = _temperature_adjust_probs(
                    av_fracs_np[i],
                    self._av_policy_sample_temperature,
                    self.n_routes,
                )
                if selection_mode == "sample":
                    ridx = int(rng.choice(self.n_routes, p=probs))
                else:
                    ridx = int(np.argmax(probs))
                baseline_route_indices[i] = int(np.clip(ridx, 0, len(route_batch[i]) - 1))

        selected_route_indices = baseline_route_indices.copy()
        counterfactual_mode = "none"
        counterfactual_applied = 0
        confidence_gate_applied = 0
        confidence_gate_triggered = False
        confidence_gate_max_nonzero_share = 0.0
        share_cap_applied = 0
        route_cost_matrix: Optional[np.ndarray] = None
        wm_route_cost_matrix: Optional[np.ndarray] = None
        live_route_cost_matrix: Optional[np.ndarray] = None
        freeflow_route_cost_matrix: Optional[np.ndarray] = None
        hdv_pressure_route_matrix: Optional[np.ndarray] = None
        route_cost_applied = 0
        rerank_trace: Optional[Dict[str, Any]] = None
        rerank_error = ""
        if wm_edge_cost is not None:
            wm_route_cost_matrix = _edge_route_cost_matrix(
                path_batch,
                wm_edge_cost,
                n_routes=int(self.n_routes),
                length_norm=self._av_route_cost_length_norm,
            )
        if live_edge_cost is not None:
            live_route_cost_matrix = _edge_route_cost_matrix(
                path_batch,
                live_edge_cost,
                n_routes=int(self.n_routes),
                length_norm=self._av_route_cost_length_norm,
            )
        if freeflow_edge_cost is not None:
            freeflow_route_cost_matrix = _edge_route_cost_matrix(
                path_batch,
                freeflow_edge_cost,
                n_routes=int(self.n_routes),
                length_norm=self._av_route_cost_length_norm,
            )
        if self._av_route_hdv_pressure_penalty > 0.0:
            hdv_edge_pressure = self._compute_hdv_edge_distribution(state)
            hdv_pressure_route_matrix = _edge_route_cost_matrix(
                path_batch,
                hdv_edge_pressure,
                n_routes=int(self.n_routes),
                length_norm="sum",
            )
        diagnostic_depart_np = self._decision_depart_edge(int(self._current_decision_idx_eval))
        diagnostic_context_np = np.array([
            float(self._current_decision_idx_eval) / max(int(self._episode_steps_eval), 1),
            float(len(active_ids)) / max(int(self._expected_n_av), 1),
            0.0,
        ], dtype=np.float32)
        diagnostic_hdv_edge_np = self._compute_hdv_edge_distribution(state)
        diagnostic_current_edge_idx = np.asarray(
            [int(edge_to_idx[e]) for e in active_current_edges if e in edge_to_idx],
            dtype=np.int64,
        )
        diagnostic_action_horizon_seconds = (
            float(getattr(self, "_decision_horizon_seconds", self._episode_steps_eval))
            / max(float(getattr(self, "_decision_steps", 1) or 1), 1.0)
        )
        self._record_wm_mcc_diagnostics(
            z_obs=z_obs,
            path_batch=path_batch,
            baseline_route_indices=baseline_route_indices,
            live_route_cost_matrix=live_route_cost_matrix,
            depart_np=diagnostic_depart_np,
            context_np=diagnostic_context_np,
            hdv_edge_np=diagnostic_hdv_edge_np,
            current_edges_np=diagnostic_current_edge_idx,
            action_horizon_seconds=float(diagnostic_action_horizon_seconds),
            rng=rng,
        )
        if len(active_ids) > 0:
            route_slots = np.clip(
                np.asarray(baseline_route_indices, dtype=np.int64),
                0,
                int(self.n_routes) - 1,
            )
            route_counts_for_gate = np.bincount(route_slots, minlength=int(self.n_routes))
            if int(self.n_routes) > 1:
                confidence_gate_max_nonzero_share = (
                    float(np.max(route_counts_for_gate[1:])) / float(len(active_ids))
                )
            trigger_share = float(self._av_policy_confidence_gate_trigger_share)
            confidence_gate_triggered = (
                trigger_share <= 0.0
                or confidence_gate_max_nonzero_share >= trigger_share
            )
        if self._av_od_route_overrides:
            counterfactual_mode = "od_override"
            selected_route_indices = baseline_route_indices.copy()
            for i, veh_id in enumerate(active_ids):
                od_key = self._veh_od.get(veh_id)
                if od_key is None:
                    continue
                override = self._av_od_route_overrides.get((int(od_key[0]), int(od_key[1])))
                if override is None:
                    continue
                selected_route_indices[i] = int(override)
                counterfactual_applied += 1
        elif (
            self._av_policy_confidence_gate_source != "none"
            and self._av_policy_confidence_gate_margin > 0.0
            and confidence_gate_triggered
            and selection_mode == "argmax"
        ):
            if self._av_policy_confidence_gate_source == "wm_current_bpr":
                gate_cost_matrix = wm_route_cost_matrix
            elif self._av_policy_confidence_gate_source == "live_current_tt":
                gate_cost_matrix = live_route_cost_matrix
            elif self._av_policy_confidence_gate_source == "freeflow_greedy":
                gate_cost_matrix = freeflow_route_cost_matrix
            else:
                gate_cost_matrix = None
            if gate_cost_matrix is not None:
                gated = selected_route_indices.copy()
                margin_threshold = float(self._av_policy_confidence_gate_margin)
                for i in range(min(len(active_ids), int(gate_cost_matrix.shape[0]))):
                    cost_row = np.asarray(gate_cost_matrix[i, : int(self.n_routes)], dtype=np.float32)
                    finite = np.isfinite(cost_row)
                    if not bool(finite.any()):
                        continue
                    masked_cost = np.where(finite, cost_row, np.inf)
                    fallback = int(np.argmin(masked_cost))
                    top = int(np.clip(int(gated[i]), 0, int(self.n_routes) - 1))
                    if fallback == top:
                        continue
                    probs = _temperature_adjust_probs(
                        av_fracs_np[i],
                        self._av_policy_sample_temperature,
                        self.n_routes,
                    )
                    if fallback >= int(probs.shape[0]) or top >= int(probs.shape[0]):
                        continue
                    policy_margin = float(probs[top] - probs[fallback])
                    if policy_margin < margin_threshold:
                        gated[i] = int(np.clip(fallback, 0, len(route_batch[i]) - 1))
                        confidence_gate_applied += 1
                selected_route_indices = gated
                if confidence_gate_applied > 0:
                    counterfactual_mode = f"policy_confidence_gate_{self._av_policy_confidence_gate_source}"
                    counterfactual_applied = int(confidence_gate_applied)
        elif self._av_route_cost_policy != "none":
            if self._av_route_cost_policy == "freeflow_greedy":
                route_cost_matrix = freeflow_route_cost_matrix
            elif self._av_route_cost_policy == "wm_current_bpr":
                route_cost_matrix = wm_route_cost_matrix
            elif self._av_route_cost_policy == "live_current_tt":
                route_cost_matrix = live_route_cost_matrix
            elif self._av_route_cost_policy == "wm_live_blend":
                if wm_route_cost_matrix is not None and live_route_cost_matrix is not None:
                    w = float(self._av_route_wm_live_blend_weight)
                    route_cost_matrix = (
                        (1.0 - w) * np.asarray(live_route_cost_matrix, dtype=np.float32)
                        + w * np.asarray(wm_route_cost_matrix, dtype=np.float32)
                    )
                else:
                    route_cost_matrix = None
            else:
                route_cost_matrix = None
            if (
                route_cost_matrix is not None
                and hdv_pressure_route_matrix is not None
                and self._av_route_hdv_pressure_penalty > 0.0
            ):
                route_cost_matrix = (
                    np.asarray(route_cost_matrix, dtype=np.float32)
                    + float(self._av_route_hdv_pressure_penalty)
                    * np.asarray(hdv_pressure_route_matrix, dtype=np.float32)
                )

            if route_cost_matrix is not None:
                selected_route_indices, route_cost_applied = _select_min_cost_route_indices_with_planned_pressure(
                    route_cost_matrix,
                    baseline_route_indices,
                    path_batch,
                    n_edges=len(getattr(self._edge_registry, "edge_ids", [])),
                    deterministic=bool(self._deterministic_eval),
                    rng=rng,
                    temperature=float(self._av_route_cost_temperature),
                    pressure_penalty=float(self._av_route_planned_av_pressure_penalty),
                )
                counterfactual_mode = f"route_cost_{self._av_route_cost_policy}"
                counterfactual_applied = int(route_cost_applied)

        if (
            self._av_policy_nonzero_route_share_cap > 0.0
            and counterfactual_mode in {"none", f"policy_confidence_gate_{self._av_policy_confidence_gate_source}"}
            and selection_mode == "argmax"
            and len(active_ids) > 0
        ):
            if self._av_policy_share_cap_fallback_source == "wm_current_bpr":
                cap_cost_matrix = wm_route_cost_matrix
            elif self._av_policy_share_cap_fallback_source == "live_current_tt":
                cap_cost_matrix = live_route_cost_matrix
            elif self._av_policy_share_cap_fallback_source == "freeflow_greedy":
                cap_cost_matrix = freeflow_route_cost_matrix
            else:
                cap_cost_matrix = None
            if cap_cost_matrix is not None:
                capped = selected_route_indices.copy()
                max_per_nonzero_route = int(
                    np.ceil(float(self._av_policy_nonzero_route_share_cap) * float(len(active_ids)))
                )
                max_per_nonzero_route = max(max_per_nonzero_route, 0)
                route_probs_for_cap = np.stack([
                    _temperature_adjust_probs(row, self._av_policy_sample_temperature, self.n_routes)
                    for row in av_fracs_np
                ], axis=0).astype(np.float32)
                for route_idx in range(1, int(self.n_routes)):
                    selected_for_route = np.where(capped == int(route_idx))[0]
                    if selected_for_route.size <= max_per_nonzero_route:
                        continue
                    candidates_to_rank: List[Tuple[float, int, int]] = []
                    mandatory_keep = 0
                    for i in selected_for_route.tolist():
                        if i >= int(cap_cost_matrix.shape[0]):
                            mandatory_keep += 1
                            continue
                        cost_row = np.asarray(cap_cost_matrix[i, : int(self.n_routes)], dtype=np.float32)
                        finite = np.isfinite(cost_row)
                        if not bool(finite.any()):
                            mandatory_keep += 1
                            continue
                        fallback = int(np.argmin(np.where(finite, cost_row, np.inf)))
                        if fallback == int(route_idx):
                            mandatory_keep += 1
                            continue
                        probs = route_probs_for_cap[i]
                        confidence = float(probs[int(route_idx)] - probs[int(fallback)])
                        candidates_to_rank.append((confidence, int(i), int(fallback)))
                    keep_budget = max(int(max_per_nonzero_route) - int(mandatory_keep), 0)
                    candidates_to_rank.sort(key=lambda x: x[0], reverse=True)
                    keep_ids = {int(i) for _conf, i, _fallback in candidates_to_rank[:keep_budget]}
                    for _conf, i, fallback in candidates_to_rank[keep_budget:]:
                        capped[int(i)] = int(np.clip(fallback, 0, len(route_batch[int(i)]) - 1))
                        share_cap_applied += 1
                if share_cap_applied > 0:
                    selected_route_indices = capped
                    if counterfactual_mode == "none":
                        counterfactual_mode = "policy_nonzero_share_cap"
                    else:
                        counterfactual_mode = f"{counterfactual_mode}+share_cap"
                    counterfactual_applied += int(share_cap_applied)

        if counterfactual_mode == "none" and self._rerank_enabled and len(active_ids) > 0:
            depart_np = self._decision_depart_edge(int(self._current_decision_idx_eval))
            context_np = np.array([
                float(self._current_decision_idx_eval) / max(int(self._episode_steps_eval), 1),
                float(len(active_ids)) / max(int(self._expected_n_av), 1),
                0.0,
            ], dtype=np.float32)
            hdv_edge_np = self._compute_hdv_edge_distribution(state)
            active_current_edge_idx = np.asarray(
                [int(edge_to_idx[e]) for e in active_current_edges if e in edge_to_idx],
                dtype=np.int64,
            )
            action_horizon_seconds = (
                float(getattr(self, "_decision_horizon_seconds", self._episode_steps_eval))
                / max(float(getattr(self, "_decision_steps", 1) or 1), 1.0)
            )
            try:
                rerank = rerank_topk_candidates(
                    wm=self._wm_eval,
                    z_current=z_obs,
                    route_probs=av_fracs_np,
                    baseline_route_indices=baseline_route_indices,
                    path_batch=path_batch,
                    horizon=self._mpc_horizon,
                    top_k=min(self._mpc_candidates, self.n_routes),
                    discount=self._mpc_discount,
                    depart_edge_np=depart_np,
                    context_np=context_np,
                    hdv_edge_np=hdv_edge_np,
                    total_av_count=float(max(int(self._expected_n_av), 1)),
                    total_hdv_count=float(max(int(self._expected_n_hdv), 1)),
                    t_all_weight=self._rerank_t_all_weight,
                    t_cav_weight=self._rerank_t_cav_weight,
                    cav_tolerance=self._rerank_cav_tolerance,
                    volume_floor=self._wm_volume_floor,
                    edge_cost_source=self._wm_edge_cost_source,
                    current_edges=active_current_edge_idx,
                    action_horizon_seconds=action_horizon_seconds,
                    decoded_speed_min_ratio=self._wm_decoded_speed_min_ratio,
                    decoded_speed_max_ratio=self._wm_decoded_speed_max_ratio,
                    decoded_speed_excess_scale=self._wm_decoded_speed_excess_scale,
                )
                rerank_trace = rerank
                selected_route_indices = np.asarray(rerank["selected"].route_indices, dtype=np.int64)
                cand_parts = []
                for d in rerank["decisions"]:
                    cand_parts.append(
                        f"c{d.candidate_id}:pressure={d.pred_t_all_pressure:.3f},"
                        f"head={d.pred_t_all_head:.3f},decoded={d.pred_t_all_decoded:.3f},"
                        f"cav_proxy={d.pred_t_cav_proxy:.3f},score={d.weighted_score:.3f},"
                        f"gate={int(d.gate_pass)}"
                    )
                print(
                    f"  [wm_rerank_local] step={int(self._current_decision_idx_eval):02d} "
                    f"selected={int(rerank['selected_idx'])} baseline=0 | " + " ; ".join(cand_parts)
                )
            except Exception as exc:
                rerank_error = str(exc)
                print(f"  [wm_rerank_local] step={int(self._current_decision_idx_eval):02d} fallback_to_baseline due to error: {exc}")
                selected_route_indices = baseline_route_indices

        actions: Dict[str, List[str]] = {}
        selected_counts = np.bincount(
            np.asarray(selected_route_indices, dtype=np.int64).clip(0, self.n_routes - 1),
            minlength=self.n_routes,
        )
        baseline_counts = np.bincount(
            np.asarray(baseline_route_indices, dtype=np.int64).clip(0, self.n_routes - 1),
            minlength=self.n_routes,
        )
        mean_probs = np.mean(av_fracs_np[:, : self.n_routes], axis=0) if av_fracs_np.size else np.zeros((self.n_routes,), dtype=np.float32)
        if route_cost_matrix is not None and route_cost_matrix.size:
            route_cost_mean = np.full((int(self.n_routes),), np.nan, dtype=np.float32)
            for k in range(int(self.n_routes)):
                vals = route_cost_matrix[:, k]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    route_cost_mean[k] = float(vals.mean())
        else:
            route_cost_mean = np.full((int(self.n_routes),), np.nan, dtype=np.float32)
        wm_route_cost_mean, wm_route_best_counts, wm_route_best_total = _route_cost_matrix_summary(
            wm_route_cost_matrix,
            int(self.n_routes),
        )
        live_route_cost_mean, live_route_best_counts, live_route_best_total = _route_cost_matrix_summary(
            live_route_cost_matrix,
            int(self.n_routes),
        )
        row: Dict[str, Any] = {
            "episode": int(self._current_episode_id_eval),
            "decision_idx": int(self._current_decision_idx_eval),
            "sim_step": int(getattr(self.connector, "current_step", -1)),
            "active_av": int(len(active_ids)),
            "deterministic": bool(self._deterministic_eval),
            "policy_route_selection": self._av_policy_route_selection,
            "rerank_enabled": bool(self._rerank_enabled),
            "counterfactual_mode": counterfactual_mode,
            "counterfactual_applied": int(counterfactual_applied),
            "av_candidate_source": self._av_candidate_source,
            "candidate_dynamic_count": int(candidate_dynamic_count),
            "candidate_fallback_count": int(candidate_fallback_count),
            "candidate_unique_mean": float(candidate_unique_mean),
            "candidate_unique_min": int(candidate_unique_min),
            "candidate_unique_max": int(candidate_unique_max),
            "route_cost_policy": self._av_route_cost_policy,
            "wm_edge_cost_source": self._wm_edge_cost_source,
            "wm_candidate_edge_cost_source": self._wm_candidate_edge_cost_source,
            "wm_decoded_speed_excess_scale": float(self._wm_decoded_speed_excess_scale),
            "route_cost_length_norm": self._av_route_cost_length_norm,
            "route_cost_temperature": float(self._av_route_cost_temperature),
            "route_hdv_pressure_penalty": float(self._av_route_hdv_pressure_penalty),
            "route_wm_live_blend_weight": float(self._av_route_wm_live_blend_weight),
            "route_planned_av_pressure_penalty": float(self._av_route_planned_av_pressure_penalty),
            "route_cost_applied": int(route_cost_applied),
            "policy_confidence_gate_source": self._av_policy_confidence_gate_source,
            "policy_confidence_gate_margin": float(self._av_policy_confidence_gate_margin),
            "policy_confidence_gate_trigger_share": float(self._av_policy_confidence_gate_trigger_share),
            "policy_confidence_gate_triggered": bool(confidence_gate_triggered),
            "policy_confidence_gate_max_nonzero_share": float(confidence_gate_max_nonzero_share),
            "policy_confidence_gate_applied": int(confidence_gate_applied),
            "policy_nonzero_route_share_cap": float(self._av_policy_nonzero_route_share_cap),
            "policy_share_cap_fallback_source": self._av_policy_share_cap_fallback_source,
            "policy_share_cap_applied": int(share_cap_applied),
            "wm_route_best_total": int(wm_route_best_total),
            "live_route_best_total": int(live_route_best_total),
            "rerank_selected_candidate": int(rerank_trace["selected_idx"]) if rerank_trace is not None else -1,
            "rerank_candidate_count": int(len(rerank_trace.get("decisions", []))) if rerank_trace is not None else 0,
            "rerank_baseline_t_cav_proxy": float(rerank_trace.get("baseline_t_cav_proxy", float("nan"))) if rerank_trace is not None else float("nan"),
            "rerank_error": rerank_error,
        }
        if rerank_trace is not None:
            for decision in rerank_trace.get("decisions", []):
                cid = int(decision.candidate_id)
                row[f"rerank_c{cid}_t_all"] = float(decision.pred_t_all)
                row[f"rerank_c{cid}_t_all_pressure"] = float(decision.pred_t_all_pressure)
                row[f"rerank_c{cid}_t_all_head"] = float(decision.pred_t_all_head)
                row[f"rerank_c{cid}_t_all_decoded"] = float(decision.pred_t_all_decoded)
                row[f"rerank_c{cid}_t_cav_proxy"] = float(decision.pred_t_cav_proxy)
                row[f"rerank_c{cid}_weighted_score"] = float(decision.weighted_score)
                row[f"rerank_c{cid}_gate_pass"] = int(bool(decision.gate_pass))
                row[f"rerank_c{cid}_gate_reason"] = str(decision.gate_reason)
        for k in range(self.n_routes):
            row[f"mean_prob_route{k}"] = float(mean_probs[k]) if k < len(mean_probs) else 0.0
            row[f"baseline_count_route{k}"] = int(baseline_counts[k])
            row[f"selected_count_route{k}"] = int(selected_counts[k])
            row[f"mean_route_cost_route{k}"] = float(route_cost_mean[k]) if k < len(route_cost_mean) and np.isfinite(route_cost_mean[k]) else float("nan")
            row[f"wm_mean_route_cost_route{k}"] = float(wm_route_cost_mean[k]) if k < len(wm_route_cost_mean) and np.isfinite(wm_route_cost_mean[k]) else float("nan")
            row[f"wm_best_count_route{k}"] = int(wm_route_best_counts[k]) if k < len(wm_route_best_counts) else 0
            row[f"live_mean_route_cost_route{k}"] = float(live_route_cost_mean[k]) if k < len(live_route_cost_mean) and np.isfinite(live_route_cost_mean[k]) else float("nan")
            row[f"live_best_count_route{k}"] = int(live_route_best_counts[k]) if k < len(live_route_best_counts) else 0
        self._dynamic_decision_rows.append(row)

        for i, veh_id in enumerate(active_ids):
            agent_row: Dict[str, Any] = {
                "episode": int(self._current_episode_id_eval),
                "decision_idx": int(self._current_decision_idx_eval),
                "sim_step": int(getattr(self.connector, "current_step", -1)),
                "veh_id": str(veh_id),
                "current_edge": active_current_edges[i] if i < len(active_current_edges) else "",
                "dest_edge": str(self._veh_dest.get(veh_id, "")),
                "deterministic": bool(self._deterministic_eval),
                "policy_route_selection": self._av_policy_route_selection,
                "av_candidate_source": self._av_candidate_source,
                "wm_candidate_edge_cost_source": self._wm_candidate_edge_cost_source,
                "route_cost_length_norm": self._av_route_cost_length_norm,
                "policy_confidence_gate_source": self._av_policy_confidence_gate_source,
                "policy_confidence_gate_margin": float(self._av_policy_confidence_gate_margin),
                "policy_confidence_gate_trigger_share": float(self._av_policy_confidence_gate_trigger_share),
                "policy_confidence_gate_triggered": bool(confidence_gate_triggered),
                "policy_confidence_gate_max_nonzero_share": float(confidence_gate_max_nonzero_share),
                "policy_nonzero_route_share_cap": float(self._av_policy_nonzero_route_share_cap),
                "policy_share_cap_fallback_source": self._av_policy_share_cap_fallback_source,
                "baseline_route": int(baseline_route_indices[i]) if i < len(baseline_route_indices) else -1,
                "selected_route": int(selected_route_indices[i]) if i < len(selected_route_indices) else -1,
                "wm_best_route": -1,
                "live_best_route": -1,
                "wm_live_best_match": 0,
            }
            wm_best_idx = -1
            live_best_idx = -1
            if wm_route_cost_matrix is not None and i < int(wm_route_cost_matrix.shape[0]):
                wm_row = np.asarray(wm_route_cost_matrix[i, : int(self.n_routes)], dtype=np.float32)
                finite = np.isfinite(wm_row)
                if bool(finite.any()):
                    finite_idx = np.flatnonzero(finite)
                    wm_best_idx = int(finite_idx[int(np.argmin(wm_row[finite]))])
            if live_route_cost_matrix is not None and i < int(live_route_cost_matrix.shape[0]):
                live_row = np.asarray(live_route_cost_matrix[i, : int(self.n_routes)], dtype=np.float32)
                finite = np.isfinite(live_row)
                if bool(finite.any()):
                    finite_idx = np.flatnonzero(finite)
                    live_best_idx = int(finite_idx[int(np.argmin(live_row[finite]))])
            agent_row["wm_best_route"] = int(wm_best_idx)
            agent_row["live_best_route"] = int(live_best_idx)
            agent_row["wm_live_best_match"] = int(wm_best_idx >= 0 and wm_best_idx == live_best_idx)
            for k in range(self.n_routes):
                agent_row[f"prob_route{k}"] = float(av_fracs_np[i, k]) if i < av_fracs_np.shape[0] and k < av_fracs_np.shape[1] else float("nan")
                agent_row[f"path_edges_route{k}"] = (
                    json.dumps(route_batch[i][k], separators=(",", ":"))
                    if i < len(route_batch) and k < len(route_batch[i])
                    else "[]"
                )
                agent_row[f"path_edge_indices_route{k}"] = (
                    json.dumps(path_batch[i][k], separators=(",", ":"))
                    if i < len(path_batch) and k < len(path_batch[i])
                    else "[]"
                )
                agent_row[f"wm_cost_route{k}"] = (
                    float(wm_route_cost_matrix[i, k])
                    if wm_route_cost_matrix is not None
                    and i < int(wm_route_cost_matrix.shape[0])
                    and k < int(wm_route_cost_matrix.shape[1])
                    and np.isfinite(wm_route_cost_matrix[i, k])
                    else float("nan")
                )
                agent_row[f"live_cost_route{k}"] = (
                    float(live_route_cost_matrix[i, k])
                    if live_route_cost_matrix is not None
                    and i < int(live_route_cost_matrix.shape[0])
                    and k < int(live_route_cost_matrix.shape[1])
                    and np.isfinite(live_route_cost_matrix[i, k])
                    else float("nan")
                )
            self._dynamic_agent_decision_rows.append(agent_row)

        for i, veh_id in enumerate(active_ids):
            ridx = int(np.clip(int(selected_route_indices[i]), 0, len(route_batch[i]) - 1))
            self._av_route_idx[veh_id] = ridx
            actions[veh_id] = route_batch[i][ridx]
        return actions

    def _collect_single_episode(
        self,
        episode_id: int,
        max_steps: int,
    ) -> Tuple[
        List[int],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[float],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[int],
        List[int],
    ]:
        rng = np.random.RandomState((self.seed or 0) + episode_id)

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

        ep_ids: List[int] = []
        ep_av_edge_acts: List[np.ndarray] = []
        ep_hdv_edge_acts: List[np.ndarray] = []
        ep_joint_edge_acts: List[np.ndarray] = []
        ep_rews: List[float] = []
        ep_frames: List[np.ndarray] = []
        ep_departs: List[np.ndarray] = []
        ep_contexts: List[np.ndarray] = []
        ep_decision_indices: List[int] = []
        ep_sim_steps: List[int] = []

        try:
            state = self.connector.reset(
                route_probs=self._hdv_route_probs,
                human_learning_state=self._loaded_hdv_state_eval,
            )
        except TypeError:
            state = self.connector.reset(route_probs=self._hdv_route_probs)
            if self._loaded_hdv_state_eval is not None and hasattr(self.connector, "env"):
                _restore_hdv_learning_state(
                    self.connector.env,
                    self._loaded_hdv_state_eval,
                    label=f"collector-ep{episode_id}",
                )
        except Exception as exc:
            print(f"  [episode {episode_id}] reset failed: {exc}")
            return (
                ep_ids,
                ep_av_edge_acts,
                ep_hdv_edge_acts,
                ep_joint_edge_acts,
                ep_rews,
                ep_frames,
                ep_departs,
                ep_contexts,
                ep_decision_indices,
                ep_sim_steps,
            )

        self._register_vehicles(list(state.get("vehicles", {}).keys()), rng)
        self._record_depart_times(state, veh_depart)

        decision_ticks = sorted(self._decision_tick_map(max_steps).keys())
        if not decision_ticks:
            decision_ticks = list(range(max_steps))

        current_tick = 0
        for decision_idx, t in enumerate(decision_ticks):
            steps_to_boundary = max(int(t) - int(current_tick), 0)
            if steps_to_boundary > 0:
                arrival_events = self.connector.advance_steps(steps_to_boundary)
                current_tick += steps_to_boundary
                self._record_arrivals(
                    veh_depart,
                    veh_travel_times,
                    track_role_breakdown=True,
                    arrival_events=arrival_events,
                )
                state = self.connector.get_traffic_state()

            frame_t = self._collect_flow(state)

            self._register_vehicles(list(state.get("vehicles", {}).keys()), rng)
            self._record_depart_times(state, veh_depart)

            self._current_episode_id_eval = int(episode_id)
            self._current_decision_idx_eval = int(decision_idx)
            av_actions = self._select_av_actions(state, rng)
            next_tick = int(decision_ticks[decision_idx + 1]) if (decision_idx + 1) < len(decision_ticks) else int(max_steps)
            advance_steps = max(next_tick - int(t), 1)
            av_edge_frac = self._compute_planned_av_edge_flow(
                state,
                av_actions,
                horizon_seconds=float(advance_steps),
            )
            hdv_edge_frac = self._compute_hdv_edge_distribution(state)
            joint_edge_frac = self._compute_joint_edge_distribution(state)
            decision_depart = self._decision_depart_edge(int(decision_idx))
            decision_context = self._decision_context(state, int(t))
            if av_actions:
                self._apply_reroutings(av_actions)

            try:
                arrival_events = self.connector.advance_steps(advance_steps)
            except Exception as exc:
                print(f"  [episode {episode_id}, decision {decision_idx}] advance failed: {exc}")
                break
            current_tick = int(t) + int(advance_steps)
            self._record_arrivals(
                veh_depart,
                veh_travel_times,
                track_role_breakdown=True,
                arrival_events=arrival_events,
            )
            state = self.connector.get_traffic_state()

            reward = self.connector._calculate_reward(state)

            ep_ids.append(episode_id)
            ep_frames.append(frame_t)
            ep_av_edge_acts.append(av_edge_frac)
            ep_hdv_edge_acts.append(hdv_edge_frac)
            ep_joint_edge_acts.append(joint_edge_frac)
            ep_rews.append(float(reward))
            ep_departs.append(decision_depart)
            ep_contexts.append(decision_context)
            ep_decision_indices.append(int(decision_idx))
            ep_sim_steps.append(int(t))

            if self.connector.is_done() or (len(state.get("vehicles", {})) == 0 and current_tick > 10):
                break

        if self._use_routerl_hdv_blackbox and hasattr(self.connector, "apply_routerl_hdv_learning"):
            try:
                self.connector.apply_routerl_hdv_learning(veh_travel_times)
            except Exception as exc:
                print(f"  [episode {episode_id}] RouteRL black-box HDV learning failed: {exc}")

        if self._decision_steps <= 0:
            ep_frames.append(self._collect_flow(state))

        if self._decision_steps > 0 and ep_rews:
            ep_reward = self._completion_aware_episode_reward(
                current_tick=int(current_tick),
                max_steps=int(max_steps),
                fallback_frame=ep_frames[-1] if ep_frames else None,
            )
            ep_rews = [ep_reward] * len(ep_rews)

        return (
            ep_ids,
            ep_av_edge_acts,
            ep_hdv_edge_acts,
            ep_joint_edge_acts,
            ep_rews,
            ep_frames,
            ep_departs,
            ep_contexts,
            ep_decision_indices,
            ep_sim_steps,
        )

    def run_eval_episodes(
        self,
        *,
        start_episode_id: int,
        num_episodes: int,
        max_steps_per_episode: int,
        use_policy_actions: bool,
        desc: str,
        log_every: int,
        collect_dataset: bool,
    ) -> Dict[str, Any]:
        self._use_policy_actions = bool(use_policy_actions)
        episode_rows: List[Dict[str, Any]] = []

        all_episodes: List[int] = []
        all_av_edge_actions: List[np.ndarray] = []
        all_hdv_edge_actions: List[np.ndarray] = []
        all_joint_edge_actions: List[np.ndarray] = []
        all_rewards: List[float] = []
        all_flows: List[np.ndarray] = []
        all_departs: List[np.ndarray] = []
        all_contexts: List[np.ndarray] = []
        all_decision_indices: List[int] = []
        all_sim_steps: List[int] = []
        dataset_saved_episodes = 0
        dataset_skipped_incomplete_episodes = 0

        iterator = tqdm(range(num_episodes), desc=desc)
        for offset in iterator:
            episode_id = int(start_episode_id + offset)
            records_folder = str(getattr(self.connector, "records_folder", ""))
            sumo_output_folder = os.path.join(records_folder, "SUMO_output")
            xml_snapshot_before = _snapshot_sumo_xmls(sumo_output_folder)
            (
                ep_ids,
                ep_av_edge_acts,
                ep_hdv_edge_acts,
                ep_joint_edge_acts,
                ep_rews,
                ep_frames,
                ep_departs,
                ep_contexts,
                ep_decision_indices,
                ep_sim_steps,
            ) = self._collect_single_episode(episode_id=episode_id, max_steps=max_steps_per_episode)

            if self._should_update_hdv_after_episode(desc):
                if self._last_ep_route_tts:
                    self._update_hdv_gawron_actual()
                elif len(ep_frames) > 0:
                    self._update_hdv_gawron(ep_frames[-1])

            t_all = float(np.mean(self._last_ep_travel_times) / 60.0) if self._last_ep_travel_times else float("nan")
            t_cav = float(np.mean(self._last_ep_av_travel_times) / 60.0) if self._last_ep_av_travel_times else float("nan")
            t_hdv = float(np.mean(self._last_ep_hdv_travel_times) / 60.0) if self._last_ep_hdv_travel_times else float("nan")
            ep_reward = float(ep_rews[-1]) if ep_rews else 0.0
            self._write_urb_episode_csv(episode_id, ep_reward)
            new_xmls = _new_sumo_xmls_since(xml_snapshot_before, sumo_output_folder, fallback_latest=2)
            self._register_protocol_xml_aliases(new_xmls, episode_id)
            row = {
                "episode": episode_id,
                "phase": desc,
                "steps": len(ep_ids),
                "arrived_all": len(self._last_ep_travel_times),
                "arrived_av": len(self._last_ep_av_travel_times),
                "arrived_hdv": len(self._last_ep_hdv_travel_times),
                "total_agents": int(getattr(self, "_num_agents", 0) or 0),
                "unfinished_all": max(int(getattr(self, "_num_agents", 0) or 0) - len(self._last_ep_travel_times), 0),
                "completion_rate": (
                    float(len(self._last_ep_travel_times)) / float(max(int(getattr(self, "_num_agents", 0) or 0), 1))
                ),
                "t_all": t_all,
                "t_CAV": t_cav,
                "t_HDV": t_hdv,
            }
            total_agents = int(getattr(self, "_num_agents", 0) or 0)
            episode_complete = _episode_complete_for_dataset(
                n_arrived=len(self._last_ep_travel_times),
                total_agents=total_agents,
                n_steps=len(ep_ids),
            )
            row["dataset_collected"] = int(bool(collect_dataset and episode_complete))
            episode_rows.append(row)

            if collect_dataset and episode_complete and ep_ids:
                all_episodes.extend(ep_ids)
                all_av_edge_actions.extend(ep_av_edge_acts)
                all_hdv_edge_actions.extend(ep_hdv_edge_acts)
                all_joint_edge_actions.extend(ep_joint_edge_acts)
                all_rewards.extend(ep_rews)
                all_flows.extend(ep_frames)
                all_departs.extend(ep_departs)
                all_contexts.extend(ep_contexts)
                all_decision_indices.extend(ep_decision_indices)
                all_sim_steps.extend(ep_sim_steps)
                dataset_saved_episodes += 1
            elif collect_dataset:
                dataset_skipped_incomplete_episodes += 1

            should_log = ((offset + 1) % max(int(log_every), 1) == 0) or ((offset + 1) == num_episodes)
            if should_log:
                if np.isfinite(t_all):
                    print(
                        f"  [{desc}] ep={offset+1:04d}/{num_episodes:04d} "
                        f"t_CAV={t_cav:.2f} min t_HDV={t_hdv:.2f} min t_all={t_all:.2f} min",
                        flush=True,
                    )
                else:
                    print(
                        f"  [{desc}] ep={offset+1:04d}/{num_episodes:04d} t_all=N/A",
                        flush=True,
                    )

        dataset = None
        if collect_dataset and all_episodes:
            dataset = {
                "episodes": all_episodes,
                "av_edge_actions": all_av_edge_actions,
                "hdv_edge_actions": all_hdv_edge_actions,
                "joint_edge_actions": all_joint_edge_actions,
                "rewards": all_rewards,
                "flows": all_flows,
                "departs": all_departs,
                "contexts": all_contexts,
                "decision_indices": all_decision_indices,
                "sim_steps": all_sim_steps,
            }
        return {
            "rows": episode_rows,
            "dataset": dataset,
            "dataset_saved_episodes": dataset_saved_episodes,
            "dataset_skipped_incomplete_episodes": dataset_skipped_incomplete_episodes,
        }
