#!/usr/bin/env python3
"""Policy-free SUMO baseline for current-to-destination live greedy routing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for rel in ("dreamer4", "dreamer4/urban_routing"):
    path = str(REPO_ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

from dreamer4.urban_routing.data_collector import DataCollector  # noqa: E402
from dreamer4.urban_routing.experiment_settings import load_settings_file  # noqa: E402
from dreamer4.urban_routing.urb_fair_protocol import load_post_mutation_state  # noqa: E402
from dreamer4.urban_routing.vehicle_imagination import build_diverse_paths_current_to_dest  # noqa: E402
from scripts.eval_sumo_hdv_state import _load_hdv_learning_state  # noqa: E402
from scripts.eval_sumo_io import _reuse_route_artifacts_from_refs  # noqa: E402
from scripts.eval_sumo_routing import (  # noqa: E402
    _build_edge_adj_list,
    _route_batch_uniqueness_summary,
)


class PolicyFreeLiveGreedyCollector(DataCollector):
    """Use free-flow current-to-dest candidates and live current TT costs."""

    def __init__(self, *args: Any, av_candidate_path_penalty_factor: float = 3.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._av_candidate_path_penalty_factor = max(float(av_candidate_path_penalty_factor), 1.0)
        self._av_candidate_adj_list = _build_edge_adj_list(
            getattr(self._edge_registry, "adjacency", None),
            getattr(self._edge_registry, "passenger_allowed", None),
        )
        self._dynamic_decision_rows: List[Dict[str, Any]] = []
        self._dynamic_agent_decision_rows: List[Dict[str, Any]] = []
        self._last_ep_av_travel_times: List[float] = []
        self._last_ep_hdv_travel_times: List[float] = []
        self._last_ep_vehicle_tt: Dict[str, float] = {}

    def _freeflow_edge_tt(self) -> Optional[np.ndarray]:
        lengths = getattr(self._edge_registry, "lengths", None)
        speed_limits = getattr(self._edge_registry, "speed_limits", None)
        edge_ids = list(getattr(self._edge_registry, "edge_ids", []))
        if lengths is None or speed_limits is None or not edge_ids:
            return None
        lengths_arr = np.asarray(lengths, dtype=np.float32).reshape(-1)
        speed_arr = np.asarray(speed_limits, dtype=np.float32).reshape(-1)
        n = min(len(edge_ids), int(lengths_arr.shape[0]), int(speed_arr.shape[0]))
        if n <= 0:
            return None
        return lengths_arr[:n] / np.maximum(speed_arr[:n], 0.1)

    def _live_current_edge_tt(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        edge_ids = list(getattr(self._edge_registry, "edge_ids", []))
        if not edge_ids:
            return None
        lengths = getattr(self._edge_registry, "lengths", None)
        speed_limits = getattr(self._edge_registry, "speed_limits", None)
        if lengths is None or speed_limits is None:
            return None
        lengths_arr = np.asarray(lengths, dtype=np.float32).reshape(-1)
        speed_arr = np.asarray(speed_limits, dtype=np.float32).reshape(-1)
        n = min(len(edge_ids), int(lengths_arr.shape[0]), int(speed_arr.shape[0]))
        if n <= 0:
            return None
        edge_state = state.get("edges", {}) if isinstance(state, dict) else {}
        out = np.zeros((n,), dtype=np.float32)
        for i, edge_id in enumerate(edge_ids[:n]):
            data = edge_state.get(edge_id, {}) if isinstance(edge_state, dict) else {}
            speed = float(data.get("mean_speed", 0.0) or 0.0) if isinstance(data, dict) else 0.0
            if not np.isfinite(speed) or speed <= 0.1:
                speed = float(speed_arr[i])
            out[i] = float(lengths_arr[i]) / max(float(speed), 0.1)
        return out

    def _current_to_dest_route_candidates(
        self,
        veh_id: str,
        current_edge: str,
        edge_cost: Optional[np.ndarray],
    ) -> List[List[str]]:
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

        src_idx = int(edge_to_idx[current_edge])
        dst_idx = int(edge_to_idx[dest_edge])
        paths = build_diverse_paths_current_to_dest(
            adj_list=self._av_candidate_adj_list,
            edge_tt=np.asarray(edge_cost, dtype=np.float32),
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
                if 0 <= int(edge_idx) < len(edge_ids)
            ]
            if route and route[0] == current_edge:
                named_paths.append(route)
        if not named_paths:
            return []
        while len(named_paths) < int(self.n_routes):
            named_paths.append(list(named_paths[0]))
        return named_paths[: int(self.n_routes)]

    def _route_costs(self, routes: List[List[str]], edge_cost: np.ndarray) -> List[float]:
        edge_to_idx = getattr(self._edge_registry, "_edge_to_idx", {})
        costs: List[float] = []
        for route in routes[: int(self.n_routes)]:
            idxs = [int(edge_to_idx[e]) for e in route if e in edge_to_idx]
            if not idxs:
                costs.append(float("inf"))
                continue
            vals = np.asarray(edge_cost[idxs], dtype=np.float32)
            vals = vals[np.isfinite(vals)]
            costs.append(float(vals.sum()) if vals.size else float("inf"))
        while len(costs) < int(self.n_routes):
            costs.append(float("inf"))
        return costs[: int(self.n_routes)]

    def _select_av_actions(self, state: Dict[str, Any], rng: np.random.RandomState) -> Dict[str, List[str]]:
        in_network = state.get("vehicles", {})
        active_avs = sorted(self._av_vehicles & set(in_network.keys()))
        actions: Dict[str, List[str]] = {}
        freeflow_edge_cost = self._freeflow_edge_tt()
        live_edge_cost = self._live_current_edge_tt(state)
        if freeflow_edge_cost is None or live_edge_cost is None:
            return actions

        route_batch: List[List[List[str]]] = []
        active_ids: List[str] = []
        selected_indices: List[int] = []
        live_cost_rows: List[List[float]] = []
        candidate_dynamic_count = 0
        candidate_fallback_count = 0

        for veh_id in active_avs:
            veh_info = in_network.get(veh_id, {})
            if not isinstance(veh_info, dict):
                continue
            current_edge = str(veh_info.get("edge", ""))
            if (not current_edge) or current_edge.startswith(":"):
                continue

            candidates = self._current_to_dest_route_candidates(veh_id, current_edge, freeflow_edge_cost)
            if candidates:
                candidate_dynamic_count += 1
            else:
                candidate_fallback_count += 1
                candidates = self._candidate_route_suffixes(veh_id, current_edge)
            if candidates:
                candidates = self._filter_safe_reroute_candidates(veh_id, current_edge, candidates)
            if not candidates:
                continue

            costs = self._route_costs(candidates, live_edge_cost)
            finite = np.asarray(np.isfinite(costs), dtype=bool)
            if bool(finite.any()):
                selected = int(np.flatnonzero(finite)[int(np.argmin(np.asarray(costs)[finite]))])
            else:
                selected = 0
            selected = int(np.clip(selected, 0, len(candidates) - 1))
            self._av_route_idx[veh_id] = selected
            actions[veh_id] = list(candidates[selected])
            route_batch.append([list(route) for route in candidates[: int(self.n_routes)]])
            active_ids.append(str(veh_id))
            selected_indices.append(selected)
            live_cost_rows.append(costs)

        if route_batch:
            uniq_mean, uniq_min, uniq_max = _route_batch_uniqueness_summary(route_batch)
            counts = np.bincount(
                np.asarray(selected_indices, dtype=np.int64).clip(0, int(self.n_routes) - 1),
                minlength=int(self.n_routes),
            )
            decision_row: Dict[str, Any] = {
                "episode": int(getattr(self, "_current_episode_id_eval", -1)),
                "decision_idx": int(getattr(self, "_current_decision_idx_eval", -1)),
                "sim_step": int(getattr(self.connector, "current_step", -1)),
                "active_av": int(len(active_ids)),
                "deterministic": True,
                "policy_route_selection": "route_cost",
                "av_candidate_source": "current_shortest_freeflow",
                "candidate_dynamic_count": int(candidate_dynamic_count),
                "candidate_fallback_count": int(candidate_fallback_count),
                "candidate_unique_mean": float(uniq_mean),
                "candidate_unique_min": int(uniq_min),
                "candidate_unique_max": int(uniq_max),
                "route_cost_policy": "live_current_tt",
            }
            live_cost_arr = np.asarray(live_cost_rows, dtype=np.float32)
            for k in range(int(self.n_routes)):
                decision_row[f"selected_count_route{k}"] = int(counts[k])
                vals = live_cost_arr[:, k] if live_cost_arr.ndim == 2 and k < live_cost_arr.shape[1] else []
                vals = np.asarray(vals, dtype=np.float32)
                vals = vals[np.isfinite(vals)]
                decision_row[f"live_mean_route_cost_route{k}"] = float(vals.mean()) if vals.size else float("nan")
                decision_row[f"live_best_count_route{k}"] = int(counts[k])
            self._dynamic_decision_rows.append(decision_row)

            edge_to_idx = getattr(self._edge_registry, "_edge_to_idx", {})
            for veh_id, routes, selected, costs in zip(active_ids, route_batch, selected_indices, live_cost_rows):
                row: Dict[str, Any] = {
                    "episode": decision_row["episode"],
                    "decision_idx": decision_row["decision_idx"],
                    "sim_step": decision_row["sim_step"],
                    "veh_id": str(veh_id),
                    "selected_route": int(selected),
                    "live_best_route": int(selected),
                    "av_candidate_source": "current_shortest_freeflow",
                    "route_cost_policy": "live_current_tt",
                }
                for k in range(int(self.n_routes)):
                    route = routes[k] if k < len(routes) else []
                    row[f"path_edges_route{k}"] = json.dumps(route)
                    row[f"path_edge_indices_route{k}"] = json.dumps([int(edge_to_idx[e]) for e in route if e in edge_to_idx])
                    row[f"live_cost_route{k}"] = float(costs[k]) if k < len(costs) else float("nan")
                self._dynamic_agent_decision_rows.append(row)

        return actions

    def _collect_single_episode(self, episode_id: int, max_steps: int):  # type: ignore[override]
        rng = np.random.RandomState((self.seed or 0) + int(episode_id))

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

        try:
            state = self.connector.reset(
                route_probs=self._hdv_route_probs,
                human_learning_state=self._loaded_hdv_state,
            )
        except TypeError:
            state = self.connector.reset(route_probs=self._hdv_route_probs)
        self._register_vehicles(list(state.get("vehicles", {}).keys()), rng)
        self._record_depart_times(state, veh_depart)

        decision_ticks = sorted(self._decision_tick_map(max_steps).keys())
        if not decision_ticks:
            decision_ticks = list(range(max_steps))

        current_tick = 0
        ep_rews: List[float] = []
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
            av_actions = self._select_av_actions(state, rng)
            if av_actions:
                self._apply_reroutings(av_actions)

            next_tick = int(decision_ticks[decision_idx + 1]) if decision_idx + 1 < len(decision_ticks) else int(max_steps)
            advance_steps = max(next_tick - int(tick), 1)
            try:
                arrivals = self.connector.advance_steps(advance_steps)
            except Exception as exc:
                print(f"  [episode {episode_id}, decision {decision_idx}] advance failed: {exc}")
                break
            current_tick = int(tick) + int(advance_steps)
            self._record_arrivals(
                veh_depart,
                veh_travel_times,
                track_role_breakdown=True,
                arrival_events=arrivals,
            )
            state = self.connector.get_traffic_state()
            ep_rews.append(float(self.connector._calculate_reward(state)))
            if self.connector.is_done() or (len(state.get("vehicles", {})) == 0 and current_tick > 10):
                break

        state, current_tick = self._drain_episode_to_completion(
            state=state,
            current_tick=int(current_tick),
            veh_depart=veh_depart,
            veh_travel_times=veh_travel_times,
        )
        self._record_arrivals(
            veh_depart,
            veh_travel_times,
            track_role_breakdown=True,
        )
        return current_tick, ep_rews

    def write_episode_csv(self, episode_id: int, episode_reward: float) -> Optional[Path]:
        agents_df = getattr(self.connector, "_agents_df", None)
        if agents_df is None or len(agents_df) == 0:
            return None
        episodes_dir = self.records_folder / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        fixed_machine_ids = set(getattr(self, "_fixed_machine_ids", set()) or set())
        fixed_human_ids = set(getattr(self, "_fixed_human_ids", set()) or set())
        for _, agent_row in agents_df.iterrows():
            veh_id = str(agent_row.get("id"))
            if veh_id in fixed_machine_ids or veh_id in self._av_vehicles:
                vtype = "AV"
            elif veh_id in fixed_human_ids or veh_id in self._hdv_vehicles:
                vtype = "Human"
            else:
                vtype = str(agent_row.get("kind", "Human"))
            rows.append(
                {
                    "id": int(float(agent_row.get("id", -1))),
                    "travel_time": float(self._last_ep_vehicle_tt.get(veh_id, np.nan)),
                    "action": int(self._av_route_idx.get(veh_id, self._hdv_route_idx.get(veh_id, -1))),
                    "policy_action": int(self._av_route_idx.get(veh_id, -1)),
                    "hdv_action": int(self._hdv_route_idx.get(veh_id, -1)),
                    "seen_in_episode": int(veh_id in self._av_vehicles or veh_id in self._hdv_vehicles),
                    "origin": int(float(agent_row.get("origin", -1))),
                    "destination": int(float(agent_row.get("destination", -1))),
                    "start_time": int(float(agent_row.get("start_time", 0))),
                    "reward": float(episode_reward),
                    "vType": vtype,
                }
            )
        out_path = episodes_dir / f"ep{int(episode_id)}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        return out_path


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _load_hdv_state_for_reuse(reuse_dir: Path, post_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = [
        reuse_dir / "hdv_learning_state_final.json",
        reuse_dir / "post_mutation_hdv_learning_state.json",
    ]
    state_ref = post_state.get("human_learning_state_ref")
    if state_ref:
        candidates.insert(0, _resolve(str(state_ref)))
    for path in candidates:
        state = _load_hdv_learning_state(path)
        if state is not None:
            return state
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--network_folder", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--reuse_artifacts_dir", required=True)
    parser.add_argument("--post_mutation_state_in", required=True)
    parser.add_argument("--test_eps", type=int, default=1)
    parser.add_argument("--ratio_machines", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps_per_episode", type=int, default=0)
    parser.add_argument("--sim_buffer_after_last_depart", type=int, default=1000)
    parser.add_argument("--episode_steps", type=int, default=60)
    parser.add_argument("--n_routes", type=int, default=4)
    parser.add_argument("--sumo_time_to_teleport", type=str, default="600")
    parser.add_argument("--safe_reroute_min_remaining_m", type=float, default=35.0)
    parser.add_argument("--candidate_path_penalty_factor", type=float, default=3.0)
    parser.add_argument("--completion_drain_steps", type=int, default=1200)
    parser.add_argument("--completion_drain_patience_steps", type=int, default=120)
    parser.add_argument("--min_arrival_fraction", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings_file(str(_resolve(args.settings)))
    network_folder = _resolve(args.network_folder)
    results_dir = _resolve(args.results_dir)
    reuse_dir = _resolve(args.reuse_artifacts_dir)
    post_state_path = _resolve(args.post_mutation_state_in)
    results_dir.mkdir(parents=True, exist_ok=True)

    post_state = load_post_mutation_state(post_state_path)
    reused = _reuse_route_artifacts_from_refs(post_state.get("artifact_refs", {}), results_dir)
    loaded_hdv_state = _load_hdv_state_for_reuse(reuse_dir, post_state)

    agents_df = pd.read_csv(reused["agents_csv"])
    max_start_time = int(float(agents_df["start_time"].max())) if "start_time" in agents_df.columns else 3600
    sim_length = int(max_start_time + int(args.sim_buffer_after_last_depart))
    rollout_max_steps = int(args.max_steps_per_episode) if int(args.max_steps_per_episode) > 0 else sim_length

    edge_registry = network_folder / "edge_registry.json"
    collector = PolicyFreeLiveGreedyCollector(
        output_dir=str(results_dir / "collector_runtime"),
        shard_size=2048,
        seed=int(args.seed),
        backend="routerl",
        network_name=str(args.network),
        network_folder=str(network_folder),
        origins=None,
        destinations=None,
        number_of_paths=int(args.n_routes),
        simulation_timesteps=sim_length,
        records_folder=str(results_dir),
        av_fraction=float(args.ratio_machines),
        random_av_fraction=0.0,
        epsilon=0.0,
        decision_steps=int(args.episode_steps),
        av_candidate_source="current_to_dest",
        safe_reroute_min_remaining_m=float(args.safe_reroute_min_remaining_m),
        completion_drain_steps=int(args.completion_drain_steps),
        completion_drain_patience_steps=int(args.completion_drain_patience_steps),
        min_arrival_fraction_for_incomplete=float(args.min_arrival_fraction),
        sumo_time_to_teleport=str(args.sumo_time_to_teleport or ""),
        sim_buffer_after_last_depart=int(args.sim_buffer_after_last_depart),
        agents_csv=str(reused["agents_csv"]),
        edge_registry_path=str(edge_registry),
        hdv_gawron_mode=str(settings.get("HDV_GAWRON_MODE", "routerl") or "routerl"),
        hdv_gawron_alpha_zero=float(settings.get("HDV_GAWRON_ALPHA_ZERO", "0.2") or 0.2),
        hdv_gawron_beta=float(settings.get("HDV_GAWRON_BETA", "-1.5") or -1.5),
        hdv_gawron_update_mode=str(settings.get("HDV_GAWRON_UPDATE_MODE", "expected") or "expected"),
        od_file=str(network_folder / f"od_{args.network}.txt"),
        route_registry_path=str(network_folder / "route_registry.json"),
        skip_hdv_route_registry=False,
        use_light_hdv_route_registry=True,
        loaded_hdv_state=loaded_hdv_state,
        av_candidate_path_penalty_factor=float(args.candidate_path_penalty_factor),
    )
    collector.set_fixed_roles(
        machine_ids=[str(x) for x in post_state.get("machine_agent_ids", [])],
        human_ids=[str(x) for x in post_state.get("human_agent_ids", [])],
    )
    if hasattr(collector.connector, "_episode_count"):
        collector.connector._episode_count = int(post_state.get("mutation_episode_index", 0))

    episode_rows: List[Dict[str, Any]] = []
    start_episode_id = int(post_state.get("mutation_episode_index", 0)) + 1
    for offset in range(int(args.test_eps)):
        episode_id = int(start_episode_id + offset)
        steps, rewards = collector._collect_single_episode(
            episode_id=episode_id,
            max_steps=int(rollout_max_steps),
        )
        reward = float(rewards[-1]) if rewards else 0.0
        collector.write_episode_csv(episode_id, reward)
        row = {
            "episode": int(episode_id),
            "phase": "test",
            "steps": int(steps),
            "arrived_all": int(len(collector._last_ep_travel_times)),
            "arrived_av": int(len(collector._last_ep_av_travel_times)),
            "arrived_hdv": int(len(collector._last_ep_hdv_travel_times)),
            "total_agents": int(getattr(collector, "_num_agents", 0) or 0),
            "completion_rate": float(len(collector._last_ep_travel_times))
            / max(float(getattr(collector, "_num_agents", 0) or 0), 1.0),
            "t_all": float(np.mean(collector._last_ep_travel_times) / 60.0)
            if collector._last_ep_travel_times
            else float("nan"),
            "t_CAV": float(np.mean(collector._last_ep_av_travel_times) / 60.0)
            if collector._last_ep_av_travel_times
            else float("nan"),
            "t_HDV": float(np.mean(collector._last_ep_hdv_travel_times) / 60.0)
            if collector._last_ep_hdv_travel_times
            else float("nan"),
        }
        episode_rows.append(row)
        print(
            "episode=%d t_all=%.4f t_CAV=%.4f t_HDV=%.4f completion=%.3f"
            % (row["episode"], row["t_all"], row["t_CAV"], row["t_HDV"], row["completion_rate"]),
            flush=True,
        )

    decision_df = pd.DataFrame(collector._dynamic_decision_rows)
    agent_df = pd.DataFrame(collector._dynamic_agent_decision_rows)
    episodes_df = pd.DataFrame(episode_rows)
    decision_path = results_dir / "dynamic_av_decisions.csv"
    agent_path = results_dir / "dynamic_av_agent_decisions.csv"
    episodes_path = results_dir / "collector_eval_episodes.csv"
    decision_df.to_csv(decision_path, index=False)
    agent_df.to_csv(agent_path, index=False)
    episodes_df.to_csv(episodes_path, index=False)

    route_counts: Dict[str, int] = {}
    live_best_counts: Dict[str, int] = {}
    candidate_dynamic_total = 0
    candidate_fallback_total = 0
    candidate_unique_mean = float("nan")
    candidate_unique_min = 0
    if len(decision_df) > 0:
        candidate_dynamic_total = int(decision_df["candidate_dynamic_count"].sum())
        candidate_fallback_total = int(decision_df["candidate_fallback_count"].sum())
        candidate_unique_mean = float(pd.to_numeric(decision_df["candidate_unique_mean"], errors="coerce").mean())
        candidate_unique_min = int(pd.to_numeric(decision_df["candidate_unique_min"], errors="coerce").min())
        for k in range(int(args.n_routes)):
            route_counts[f"route{k}"] = int(decision_df.get(f"selected_count_route{k}", pd.Series(dtype=float)).sum())
            live_best_counts[f"route{k}"] = int(decision_df.get(f"live_best_count_route{k}", pd.Series(dtype=float)).sum())
    total_selected = max(int(sum(route_counts.values())), 1)
    summary = {
        "network": str(args.network),
        "policy_ckpt": "",
        "test_eps": int(args.test_eps),
        "ratio_machines": float(args.ratio_machines),
        "dynamic_routing": True,
        "dynamic_decision_trace": str(decision_path),
        "dynamic_agent_decision_trace": str(agent_path),
        "dynamic_selected_route_counts": route_counts,
        "route0_share": float(route_counts.get("route0", 0)) / float(total_selected),
        "av_candidate_source": "current_shortest_freeflow",
        "candidate_dynamic_total": int(candidate_dynamic_total),
        "candidate_fallback_total": int(candidate_fallback_total),
        "candidate_unique_mean": float(candidate_unique_mean),
        "candidate_unique_min": int(candidate_unique_min),
        "live_route_best_counts": live_best_counts,
        "av_route_cost_policy": "live_current_tt",
        "wm_candidate_edge_cost_source": "freeflow",
        "t_all": float(episodes_df["t_all"].mean()) if len(episodes_df) else float("nan"),
        "t_CAV": float(episodes_df["t_CAV"].mean()) if len(episodes_df) else float("nan"),
        "t_HDV": float(episodes_df["t_HDV"].mean()) if len(episodes_df) else float("nan"),
        "completion_rate": float(episodes_df["completion_rate"].mean()) if len(episodes_df) else float("nan"),
    }
    summary_path = results_dir / "collector_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
