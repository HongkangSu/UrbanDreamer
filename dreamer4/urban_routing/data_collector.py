# data_collector.py
"""
Data Collection Pipeline for Urban Routing (World Model Pretraining)

Action/context format saved to disk:
  av_edge_action / av_planned_edge_flow
    : (N, E) float32 — pre-advance planned AV route-flow field for each decision
  hdv_edge_action
    : (N, E) float32 — HDV edge occupancy/context distribution at each decision
  joint_edge_action
    : (N, E) float32 — joint AV+HDV edge occupancy/context distribution

AV re-routing strategy (configurable per-episode mix):
  - random_av_fraction  of AVs: pick a uniformly random route from [0, K) every step
  - (1 - random_av_fraction) of AVs: epsilon-greedy — best route w.p. (1-eps), random otherwise
  HDVs: commit to departure-time route, never re-routed during the episode.

The world model consumes edge-level actions directly.
"""

import os
import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from tqdm import tqdm

try:
    from .experiment_settings import add_settings_arg, apply_stage_defaults, parse_settings_arg
except ImportError:
    from experiment_settings import add_settings_arg, apply_stage_defaults, parse_settings_arg

try:
    from .routerl_shared import load_od_data, ensure_agents_csv_in_records
except ImportError:
    from routerl_shared import load_od_data, ensure_agents_csv_in_records

try:
    from .routerl_connector import ROUTERL_AVAILABLE, RouteRLConnector
except ImportError:
    try:
        from routerl_connector import ROUTERL_AVAILABLE, RouteRLConnector
    except ImportError:
        ROUTERL_AVAILABLE = False

try:
    from .sumo_connector import SUMOConnector
except ImportError:
    from sumo_connector import SUMOConnector

try:
    from .gawron import gawron_step_from_route_tt
except ImportError:
    from gawron import gawron_step_from_route_tt

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False


class _LightRouteRegistry:
    """Minimal route-registry surface for HDV Gawron updates during eval/collection."""

    def __init__(
        self,
        *,
        origins: List[str],
        destinations: List[str],
        route_edge_indices: List[List[List[int]]],
        od_av_counts: np.ndarray,
        od_hdv_counts: np.ndarray,
        od_pairs: Optional[List[Tuple[int, int]]] = None,
    ):
        self.origins = list(origins)
        self.destinations = list(destinations)
        self.route_edge_indices = route_edge_indices
        self.od_av_counts = np.asarray(od_av_counts, dtype=np.float32)
        self.od_hdv_counts = np.asarray(od_hdv_counts, dtype=np.float32)
        self.n_od = int(len(route_edge_indices))
        self.od_pairs = (
            [(int(o), int(d)) for o, d in od_pairs]
            if od_pairs is not None
            else None
        )
        self.od_index_map = (
            {(int(o), int(d)): int(i) for i, (o, d) in enumerate(self.od_pairs)}
            if self.od_pairs is not None
            else None
        )
        self.route_masks = None

    def od_index(self, origin_idx: int, dest_idx: int) -> int:
        if self.od_index_map is not None:
            return int(self.od_index_map.get((int(origin_idx), int(dest_idx)), -1))
        n_dests = max(1, len(self.destinations))
        od_idx = int(origin_idx) * n_dests + int(dest_idx)
        return int(od_idx) if 0 <= od_idx < self.n_od else -1

    def iter_od_rows(self):
        if self.od_pairs is not None:
            for od_idx, od_key in enumerate(self.od_pairs):
                yield int(od_idx), (int(od_key[0]), int(od_key[1]))
            return
        n_origins = len(self.origins)
        n_dests = len(self.destinations)
        for oi in range(n_origins):
            for di in range(n_dests):
                yield oi * n_dests + di, (oi, di)


def _route_registry_od_index(rr: Any, oi: int, di: int) -> int:
    if hasattr(rr, "od_index"):
        try:
            return int(rr.od_index(int(oi), int(di)))
        except Exception:
            return -1
    n_dests = max(1, len(getattr(rr, "destinations", []) or []))
    od_idx = int(oi) * n_dests + int(di)
    n_od = int(getattr(rr, "n_od", len(getattr(rr, "route_edge_indices", []) or [])))
    return int(od_idx) if 0 <= od_idx < n_od else -1


def _iter_route_registry_od_rows(rr: Any):
    if hasattr(rr, "iter_od_rows"):
        yield from rr.iter_od_rows()
        return
    od_pairs = getattr(rr, "od_pairs", None)
    if od_pairs is not None:
        for od_idx, od_key in enumerate(od_pairs):
            yield int(od_idx), (int(od_key[0]), int(od_key[1]))
        return
    n_origins = len(getattr(rr, "origins", []) or [])
    n_dests = len(getattr(rr, "destinations", []) or [])
    for oi in range(n_origins):
        for di in range(n_dests):
            yield oi * n_dests + di, (oi, di)


def _copy_agents_csv_to_records(agents_csv: str, records_folder: str) -> str:
    """Make an explicit demand file visible to RouteRL without using network defaults."""
    dst = Path(records_folder) / "agents.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = Path(agents_csv)
    try:
        if src.resolve() == dst.resolve():
            return str(dst)
    except FileNotFoundError:
        pass
    shutil.copy2(src, dst)
    return str(dst)


def serialize_hdv_route_state(
    route_probs: Optional[Dict[Tuple[int, int], np.ndarray]],
    route_costs: Optional[Dict[Tuple[int, int], np.ndarray]],
) -> Optional[Dict[str, Any]]:
    if not route_probs:
        return None
    entries: List[Dict[str, Any]] = []
    for (oi, di), probs in sorted(route_probs.items(), key=lambda kv: (int(kv[0][0]), int(kv[0][1]))):
        entry: Dict[str, Any] = {
            "origin": int(oi),
            "destination": int(di),
            "probs": np.asarray(probs, dtype=np.float64).tolist(),
        }
        if route_costs is not None and (oi, di) in route_costs:
            entry["costs"] = np.asarray(route_costs[(oi, di)], dtype=np.float64).tolist()
        entries.append(entry)
    return {"route_policy": entries}


def restore_hdv_route_state(
    snapshot: Optional[Dict[str, Any]],
    route_probs: Optional[Dict[Tuple[int, int], np.ndarray]],
    route_costs: Optional[Dict[Tuple[int, int], np.ndarray]],
) -> Dict[str, int]:
    stats = {"restored_probs": 0, "restored_costs": 0, "skipped": 0}
    if not snapshot or route_probs is None:
        return stats
    entries = snapshot.get("route_policy", [])
    if not isinstance(entries, list):
        return stats
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            od_key = (int(entry["origin"]), int(entry["destination"]))
        except Exception:
            stats["skipped"] += 1
            continue
        probs = route_probs.get(od_key)
        if probs is None:
            stats["skipped"] += 1
            continue
        try:
            new_probs = np.asarray(entry.get("probs", []), dtype=probs.dtype)
        except Exception:
            stats["skipped"] += 1
            continue
        if new_probs.shape != probs.shape:
            stats["skipped"] += 1
            continue
        route_probs[od_key] = new_probs.copy()
        stats["restored_probs"] += 1
        if route_costs is not None and od_key in route_costs and "costs" in entry:
            try:
                new_costs = np.asarray(entry.get("costs", []), dtype=route_costs[od_key].dtype)
            except Exception:
                continue
            if new_costs.shape == route_costs[od_key].shape:
                route_costs[od_key] = new_costs.copy()
                stats["restored_costs"] += 1
    return stats


def _build_light_route_registry_from_routes_csv(
    *,
    routes_csv_path: str,
    agents_csv_path: str,
    edge_to_idx: Dict[str, int],
    origins: List[str],
    destinations: List[str],
    ratio_machines: float,
) -> _LightRouteRegistry:
    import pandas as pd
    try:
        from .flow_projector import split_od_agent_counts
    except ImportError:
        from flow_projector import split_od_agent_counts

    routes_df = pd.read_csv(routes_csv_path)
    agents_df = pd.read_csv(agents_csv_path)

    observed_pairs: set[Tuple[int, int]] = set()
    if {"origin", "destination"}.issubset(agents_df.columns):
        for oi, di in agents_df[["origin", "destination"]].itertuples(index=False, name=None):
            try:
                oi_i = int(oi)
                di_i = int(di)
            except Exception:
                continue
            if 0 <= oi_i < len(origins) and 0 <= di_i < len(destinations):
                observed_pairs.add((oi_i, di_i))

    route_rows: Dict[Tuple[int, int], List[List[int]]] = {}

    for (oi, di), group in routes_df.groupby(["origins", "destinations"], sort=True):
        oi_i = int(oi)
        di_i = int(di)
        if oi_i < 0 or oi_i >= len(origins) or di_i < 0 or di_i >= len(destinations):
            continue
        if observed_pairs and (oi_i, di_i) not in observed_pairs:
            continue
        paths: List[List[int]] = []
        for path_str in group["path"].tolist():
            idxs = [int(edge_to_idx[e]) for e in str(path_str).split() if e in edge_to_idx]
            if idxs:
                paths.append(idxs)
        if paths:
            route_rows[(oi_i, di_i)] = paths

    od_pairs = sorted(route_rows.keys())
    od_index_map = {od_key: i for i, od_key in enumerate(od_pairs)}
    route_edge_indices: List[List[List[int]]] = [route_rows[od_key] for od_key in od_pairs]

    total_counts = np.zeros((len(od_pairs),), dtype=np.float32)
    if {"origin", "destination"}.issubset(agents_df.columns):
        for (oi, di), group in agents_df.groupby(["origin", "destination"], sort=False):
            oi_i = int(oi)
            di_i = int(di)
            od_idx = od_index_map.get((oi_i, di_i))
            if od_idx is not None:
                total_counts[int(od_idx)] = float(len(group))

    od_av_counts, od_hdv_counts = split_od_agent_counts(total_counts, ratio_machines)
    return _LightRouteRegistry(
        origins=origins,
        destinations=destinations,
        route_edge_indices=route_edge_indices,
        od_av_counts=od_av_counts,
        od_hdv_counts=od_hdv_counts,
        od_pairs=od_pairs,
    )


class DataCollector:
    """
    Collects world-model pretraining data from SUMO simulations.

    Vehicles are randomly split into HDV and AV roles each episode:
      HDV: Gawron model selects route at departure; never re-routed.
      AV:  Re-routed every step using a mix of random and epsilon-greedy policies.

    Saved demo tensors per task:
      episode           : (N,)    int64
      av_edge_action    : (N, E)  float32  — pre-advance planned AV edge-flow action per step
      av_planned_edge_flow
                        : (N, E)  float32  — explicit alias for av_edge_action
      hdv_edge_action   : (N, E)  float32  — HDV edge occupancy distribution per step
      joint_edge_action : (N, E)  float32  — joint AV+HDV edge occupancy distribution
      reward            : (N,)    float32
    Flow shards saved separately under flows/<task_name>/shard_XXXX.pt.
    """

    def __init__(
        self,
        output_dir: str,
        img_size: int = 128,
        shard_size: int = 2048,
        seed: Optional[int] = None,
        # --- routerl backend parameters ---
        backend: str = "routerl",
        network_name: str = "ing_small",
        network_folder: str = "",
        origins: Optional[List[str]] = None,
        destinations: Optional[List[str]] = None,
        number_of_paths: int = 4,
        path_gen_beta: float = -5.0,
        num_samples: int = 10,
        num_agents: int = 100,
        simulation_timesteps: int = 1800,
        records_folder: str = "routerl_records",
        use_gui: bool = False,
        # --- legacy backend parameters ---
        config_file: str = "",
        # --- AV re-routing parameters ---
        av_fraction: float = 0.3,
        random_av_fraction: float = 0.5,
        epsilon: float = 0.3,
        decision_steps: int = 60,
        av_candidate_source: str = "current_to_dest",
        safe_reroute_min_remaining_m: float = 35.0,
        completion_drain_steps: int = 0,
        completion_drain_patience_steps: int = 0,
        min_arrival_fraction_for_incomplete: float = 1.0,
        fixed_av_route_index: int = -1,
        fixed_av_route_probs: Optional[np.ndarray] = None,
        sim_buffer_after_last_depart: int = 1000,
        sumo_depart_lane: str = "",
        sumo_depart_pos: str = "",
        sumo_depart_speed: str = "",
        sumo_time_to_teleport: str = "",
        sumo_eager_insert: bool = False,
        sumo_sloppy_insert: bool = False,
        sumo_emergency_insert: bool = False,
        sumo_extra_args: str = "",
        incident_edge: str = "",
        incident_start: float = 0.0,
        incident_end: float = 0.0,
        incident_speed: float = 0.0,
        # --- agents_csv alignment ---
        agents_csv: str = "",
        # --- DAgger policy guidance ---
        policy_ckpt: str = "",
        dagger_beta: float = 0.5,
        dagger_noise: float = 0.3,
        edge_registry_path: str = "",
        # --- HDV inter-episode Gawron update ---
        hdv_gawron_lr: float = 0.1,
        hdv_gawron_mode: str = "replicator",
        hdv_gawron_alpha_zero: float = 0.2,
        hdv_gawron_beta: float = -1.5,
        hdv_gawron_update_mode: str = "expected",
        hdv_min_prob: float = 0.0,
        od_file: str = "",
        route_registry_path: str = "",
        skip_hdv_route_registry: bool = False,
        use_light_hdv_route_registry: bool = False,
        fixed_machine_ids: Optional[List[str]] = None,
        fixed_human_ids: Optional[List[str]] = None,
        loaded_hdv_state: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            output_dir:          Directory to save collected data.
            img_size:            Rendered image size (square).
            shard_size:          Number of frames per shard file.
            seed:                Global random seed.
            backend:             "routerl" (default) or "legacy".
            network_name:        SUMO network name.
            network_folder:      Absolute path to network files (routerl backend).
            origins:             Origin edge IDs (routerl backend).
            destinations:        Destination edge IDs (routerl backend).
            number_of_paths:     K — number of candidate routes per OD pair.
            path_gen_beta:       Logit sensitivity for janux path generation.
            num_samples:         Candidate routes sampled before selecting K.
            num_agents:          Number of demand agents.
            simulation_timesteps: SUMO simulation length in seconds.
            records_folder:      Directory for routerl episode records.
            use_gui:             Launch SUMO-GUI.
            config_file:         .sumocfg path (legacy backend only).
            av_fraction:         Fraction of vehicles designated as AVs each episode.
            random_av_fraction:  Fraction of AVs using purely random re-routing.
            epsilon:             Exploration rate for epsilon-greedy AVs.
            av_candidate_source: "current_to_dest" re-searches routes from the
                                 current edge to the destination at each
                                 decision; "od_suffix" trims pre-generated OD
                                 routes and is deprecated legacy behavior.
            safe_reroute_min_remaining_m: Minimum distance remaining on the
                                 current lane before allowing a reroute whose
                                 next edge is reachable only from another lane
                                 on the current edge. Set <0 to disable this
                                 safety gate.
            completion_drain_steps: Extra SUMO ticks to advance after the
                                 recorded control horizon solely to observe
                                 late arrivals. No additional frames/actions
                                 are saved during the drain.
            completion_drain_patience_steps: Stop the completion drain early
                                 after this many consecutive ticks without any
                                 new arrivals. Set <=0 to disable patience.
            min_arrival_fraction_for_incomplete: If <1.0, stop completion drain
                                 once this arrived/total fraction is observed.
                                 This is intended for explicit incomplete-episode
                                 collection on heavily congested networks.
            fixed_av_route_index: If >= 0, override AV routing to always choose
                                  this candidate route index. Used only for
                                  SUMO counterfactual screening, not learning.
            fixed_av_route_probs: Optional route distribution for AV rerouting.
                                  Ignored when fixed_av_route_index >= 0.
            agents_csv:          Path to agents.csv — overrides num_agents and
                                 simulation_timesteps for the routerl backend.
            policy_ckpt:         MARL checkpoint for DAgger policy guidance.
            dagger_beta:         Fraction of steps where policy guides AV routing.
            dagger_noise:        Dirichlet noise weight mixed into policy av_frac.
            edge_registry_path:  Path to edge_registry.json — enables flow mode (saves flows/ instead of frames/).
        """
        self.output_dir = Path(output_dir)
        self.records_folder = Path(records_folder)
        self.img_size = img_size
        self.shard_size = shard_size
        self.seed = seed
        self.backend = backend
        self.n_routes = number_of_paths
        self.av_fraction = av_fraction
        self.random_av_fraction = random_av_fraction
        self.epsilon = epsilon
        self._decision_steps = max(int(decision_steps), 0)
        self._av_candidate_source = str(av_candidate_source or "current_to_dest").strip().lower()
        if self._av_candidate_source not in {"current_to_dest", "od_suffix"}:
            raise ValueError(f"Unsupported av_candidate_source={av_candidate_source!r}")
        self._safe_reroute_min_remaining_m = float(safe_reroute_min_remaining_m)
        self._completion_drain_steps = max(int(completion_drain_steps), 0)
        self._completion_drain_patience_steps = max(int(completion_drain_patience_steps), 0)
        self._min_arrival_fraction_for_incomplete = float(
            np.clip(float(min_arrival_fraction_for_incomplete), 0.0, 1.0)
        )
        self._fixed_av_route_index = int(fixed_av_route_index)
        self._fixed_av_route_probs: Optional[np.ndarray] = None
        if fixed_av_route_probs is not None:
            probs = np.asarray(fixed_av_route_probs, dtype=np.float64).reshape(-1)
            if probs.size != int(self.n_routes):
                raise ValueError(
                    f"fixed_av_route_probs expected {int(self.n_routes)} values, got {probs.size}"
                )
            probs = np.clip(probs, 0.0, None)
            total = float(probs.sum())
            if total <= 0.0:
                raise ValueError("fixed_av_route_probs must have positive sum")
            self._fixed_av_route_probs = (probs / total).astype(np.float64)
        self._sim_buffer_after_last_depart = max(int(sim_buffer_after_last_depart), 0)
        self._sumo_depart_lane = str(sumo_depart_lane or "").strip()
        self._sumo_depart_pos = str(sumo_depart_pos or "").strip()
        self._sumo_depart_speed = str(sumo_depart_speed or "").strip()
        self._sumo_time_to_teleport = str(sumo_time_to_teleport or "").strip()
        self._sumo_eager_insert = bool(sumo_eager_insert)
        self._sumo_sloppy_insert = bool(sumo_sloppy_insert)
        self._sumo_emergency_insert = bool(sumo_emergency_insert)
        self._sumo_extra_args = str(sumo_extra_args or "")
        self._incident_edge = str(incident_edge or "").strip()
        self._incident_start = float(incident_start or 0.0)
        self._incident_end = float(incident_end or 0.0)
        self._incident_speed = float(incident_speed or 0.0)
        self._decision_horizon_seconds = int(simulation_timesteps)
        self._num_agents = int(num_agents)
        self._expected_n_av = max(0, int(round(float(num_agents) * float(self.av_fraction))))
        self._expected_n_hdv = max(0, int(num_agents) - self._expected_n_av)
        self._departure_schedule = None
        self._fixed_machine_ids = set(str(x) for x in (fixed_machine_ids or []))
        self._fixed_human_ids = set(str(x) for x in (fixed_human_ids or []))
        self._loaded_hdv_state = loaded_hdv_state

        if (origins is None or destinations is None) and od_file:
            try:
                od_data = load_od_data(
                    network_folder,
                    network_name,
                ) if (network_folder and network_name and not os.path.samefile(od_file, os.path.join(network_folder, f"od_{network_name}.txt"))) else None
            except Exception:
                od_data = None
            if od_data is None:
                try:
                    with open(od_file, "r", encoding="utf-8") as f:
                        import ast as _ast
                        od_data = _ast.literal_eval(f.read())
                except Exception:
                    od_data = None
            if isinstance(od_data, dict):
                if origins is None:
                    origins = list(od_data.get("origins", []))
                if destinations is None:
                    destinations = list(od_data.get("destinations", []))

        # -- flow collection --
        try:
            from .flow_projector import EdgeRegistry, build_edge_registry
        except ImportError:
            from flow_projector import EdgeRegistry, build_edge_registry
        import os as _os
        if edge_registry_path and _os.path.exists(edge_registry_path):
            self._edge_registry = EdgeRegistry.load(edge_registry_path)
        elif edge_registry_path:
            # Auto-build from net.xml if registry JSON doesn't exist yet
            print(f"  edge_registry.json not found — building from net.xml ...")
            reg_dir = str(Path(edge_registry_path).parent)
            self._edge_registry = build_edge_registry(reg_dir, network_name)
            self._edge_registry.save(edge_registry_path)
        else:
            # Default: build from network_folder
            self._edge_registry = build_edge_registry(network_folder, network_name)
        self._flows_dir = self.output_dir / "flows"
        self._flows_dir.mkdir(parents=True, exist_ok=True)
        print(f"  EdgeRegistry loaded: {self._edge_registry.n_edges} edges")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # -- agents_csv alignment: override num_agents + simulation_timesteps
        if agents_csv:
            import pandas as pd
            df = pd.read_csv(agents_csv)
            num_agents = len(df)
            # Prefer "start_time" column; fall back to "max_start_time" if absent
            if "start_time" in df.columns:
                max_depart = int(df["start_time"].max())
            elif "max_start_time" in df.columns:
                max_depart = int(df["max_start_time"].max())
            else:
                max_depart = simulation_timesteps
            self._decision_horizon_seconds = int(max_depart)
            simulation_timesteps = max_depart + self._sim_buffer_after_last_depart
            self._num_agents = int(num_agents)
            self._expected_n_av = max(0, int(round(float(num_agents) * float(self.av_fraction))))
            self._expected_n_hdv = max(0, int(num_agents) - self._expected_n_av)
            print(f"  [agents_csv] num_agents={num_agents}, last_depart={max_depart}s, sim_ts={simulation_timesteps}s")
        else:
            self._decision_horizon_seconds = int(simulation_timesteps)

        if backend == "routerl":
            if not ROUTERL_AVAILABLE:
                raise RuntimeError(
                    "routerl is not installed. Run: python -m pip install "
                    "routerlurb==1.0.0  "
                    "or use --backend legacy"
                )
            if not network_folder:
                raise ValueError("network_folder must be provided for the routerl backend")
            if origins is None or destinations is None:
                raise ValueError("origins and destinations must be provided for the routerl backend")
            if agents_csv:
                _copy_agents_csv_to_records(agents_csv, records_folder)

            self.connector = RouteRLConnector(
                network_name=network_name,
                network_folder=network_folder,
                origins=origins,
                destinations=destinations,
                number_of_paths=number_of_paths,
                path_gen_beta=path_gen_beta,
                num_samples=num_samples,
                num_agents=num_agents,
                simulation_timesteps=simulation_timesteps,
                records_folder=records_folder,
                seed=seed or 42,
                use_gui=use_gui,
                use_routerl_hdv_blackbox=(str(hdv_gawron_mode).lower() == "routerl_blackbox"),
                sumo_depart_lane=self._sumo_depart_lane,
                sumo_depart_pos=self._sumo_depart_pos,
                sumo_depart_speed=self._sumo_depart_speed,
                sumo_time_to_teleport=self._sumo_time_to_teleport,
                sumo_eager_insert=self._sumo_eager_insert,
                sumo_sloppy_insert=self._sumo_sloppy_insert,
                sumo_emergency_insert=self._sumo_emergency_insert,
                sumo_extra_args=self._sumo_extra_args,
                incident_edge=self._incident_edge,
                incident_start=self._incident_start,
                incident_end=self._incident_end,
                incident_speed=self._incident_speed,
            )
        else:
            if not config_file:
                raise ValueError("config_file must be provided for the legacy backend")
            self.connector = SUMOConnector(
                config_file=config_file,
                use_gui=use_gui,
                seed=seed,
                enable_rerouting=False,
            )

        if (
            self._decision_steps > 0
            and agents_csv
            and origins is not None
            and destinations is not None
        ):
            try:
                from .departure_schedule import build_departure_schedule
            except ImportError:
                from departure_schedule import build_departure_schedule
            step_interval = float(self._decision_horizon_seconds) / max(self._decision_steps, 1)
            if step_interval > 0.0:
                self._departure_schedule = build_departure_schedule(
                    agents_csv=agents_csv,
                    step_interval=step_interval,
                    origins=origins,
                    destinations=destinations,
                    edge_to_idx=self._edge_registry._edge_to_idx,
                    n_edges=self._edge_registry.n_edges,
                    ratio_machines=float(self.av_fraction),
                )

        # Per-episode vehicle role assignments (cleared on each reset)
        self._av_vehicles: set = set()
        self._hdv_vehicles: set = set()
        self._random_avs: set = set()   # AV sub-group: random re-routing
        self._greedy_avs: set = set()   # AV sub-group: epsilon-greedy
        self._av_route_idx: Dict[str, int] = {}   # current route index for each AV
        self._hdv_route_idx: Dict[str, int] = {}  # departure route index for each HDV (fixed)
        self._veh_dest: Dict[str, str] = {}        # destination edge per vehicle (set at registration)
        # Route cache keyed by (from_edge_id, dest_edge_id) — shared across vehicles,
        # cleared each episode. Routes always start from from_edge_id, so they are
        # valid for any vehicle currently on that edge.
        self._route_cache: Dict[Tuple[str, str], List[List[str]]] = {}
        # Per-vehicle cached route candidates: computed once at registration time
        # from the vehicle's origin edge. This avoids per-step dijkstra calls.
        self._veh_candidates: Dict[str, List[List[str]]] = {}

        # Per-episode actual vehicle travel times (populated by _collect_single_episode)
        self._last_ep_travel_times: List[float] = []  # per-arrived-vehicle tt (seconds)
        self._last_av_decision_trace: List[Dict[str, Any]] = []
        self._collection_decision_trace_rows: List[Dict[str, Any]] = []
        self._last_ep_safe_reroutes = 0
        self._last_ep_unsafe_reroute_keeps = 0
        self._last_ep_completion_drain_steps = 0
        self._last_ep_completion_drain_stale_steps = 0
        self._last_ep_completion_drain_stale_steps = 0
        # Per-OD-route actual travel times: {(oi, di, k): [tt1, tt2, ...]}
        self._last_ep_route_tts: Dict[Tuple[int, int, int], List[float]] = {}
        # Vehicle OD mapping: {veh_id: (origin_idx, dest_idx)}
        self._veh_od: Dict[str, Tuple[int, int]] = {}

        # -- DAgger policy loading (CPU inference, lightweight)
        self._dagger_active = False
        self._dagger_encoder = None
        self._dagger_policy  = None
        self._dagger_beta    = dagger_beta
        self._dagger_noise   = dagger_noise
        self._step_av_policy_frac: Optional[np.ndarray] = None

        if policy_ckpt:
            # DAgger in flow mode not yet implemented
            pass

        # -- HDV inter-episode Gawron update --
        self._hdv_gawron_lr = hdv_gawron_lr
        self._hdv_gawron_mode = str(hdv_gawron_mode).lower()
        self._hdv_gawron_alpha_zero = float(hdv_gawron_alpha_zero)
        self._hdv_gawron_beta = float(hdv_gawron_beta)
        self._hdv_gawron_update_mode = str(hdv_gawron_update_mode).lower()
        self._hdv_min_prob = float(hdv_min_prob)
        self._use_routerl_hdv_blackbox = (self._hdv_gawron_mode == "routerl_blackbox")
        self._skip_hdv_route_registry = bool(skip_hdv_route_registry)
        self._use_light_hdv_route_registry = bool(use_light_hdv_route_registry)
        self._hdv_route_probs: Optional[Dict[Tuple[int, int], np.ndarray]] = None
        self._hdv_route_costs: Optional[Dict[Tuple[int, int], np.ndarray]] = None
        self._route_registry = None

        if (
            (not self._use_routerl_hdv_blackbox)
            and (self._use_light_hdv_route_registry)
            and agents_csv
            and origins is not None
            and destinations is not None
        ):
            routes_csv_path = os.path.join(records_folder, "routes.csv")
            if os.path.exists(routes_csv_path) and os.path.exists(agents_csv):
                self._route_registry = _build_light_route_registry_from_routes_csv(
                    routes_csv_path=routes_csv_path,
                    agents_csv_path=agents_csv,
                    edge_to_idx=self._edge_registry._edge_to_idx,
                    origins=list(origins),
                    destinations=list(destinations),
                    ratio_machines=float(self.av_fraction),
                )
                self._hdv_route_probs = {}
                self._hdv_route_costs = {} if self._hdv_gawron_mode == "routerl" else None
                for od_idx, od_key in _iter_route_registry_od_rows(self._route_registry):
                    if od_idx < 0 or od_idx >= len(self._route_registry.route_edge_indices):
                        continue
                    K = len(self._route_registry.route_edge_indices[od_idx])
                    if K <= 0:
                        continue
                    self._hdv_route_probs[od_key] = np.ones(K, dtype=np.float32) / K
                    if self._hdv_route_costs is not None:
                        self._hdv_route_costs[od_key] = np.zeros(K, dtype=np.float32)
                print(
                    "  HDV Gawron update (light): mode=%s lr=%.3f alpha_zero=%.3f beta=%.3f update=%s, %d OD pairs"
                    % (
                        self._hdv_gawron_mode,
                        self._hdv_gawron_lr,
                        self._hdv_gawron_alpha_zero,
                        self._hdv_gawron_beta,
                        self._hdv_gawron_update_mode,
                        len(self._hdv_route_probs),
                    )
                )
        elif (
            (not self._use_routerl_hdv_blackbox)
            and (not self._skip_hdv_route_registry)
            and od_file
            and network_folder
            and self._edge_registry is not None
        ):
            try:
                from .flow_projector import RouteRegistry, build_route_registry
            except ImportError:
                from flow_projector import RouteRegistry, build_route_registry
            rr_path = route_registry_path or os.path.join(network_folder, "route_registry.json")
            if os.path.exists(rr_path):
                self._route_registry = RouteRegistry.load(rr_path, self._edge_registry)
            else:
                self._route_registry = build_route_registry(
                    network_folder, network_name, self._edge_registry,
                    od_file, number_of_paths, agents_csv, av_fraction,
                )
            # Initialize uniform route probs keyed by (origin_idx, dest_idx) integers
            # matching agents.csv format used by routerl_connector
            self._hdv_route_probs = {}
            self._hdv_route_costs = {} if self._hdv_gawron_mode == "routerl" else None
            for od_idx, od_key in _iter_route_registry_od_rows(self._route_registry):
                if od_idx < 0 or od_idx >= len(self._route_registry.route_edge_indices):
                    continue
                K = len(self._route_registry.route_edge_indices[od_idx])
                if K <= 0:
                    continue
                self._hdv_route_probs[od_key] = np.ones(K, dtype=np.float32) / K
                if self._hdv_route_costs is not None:
                    self._hdv_route_costs[od_key] = np.zeros(K, dtype=np.float32)
            print(
                "  HDV Gawron update: mode=%s lr=%.3f alpha_zero=%.3f beta=%.3f update=%s, %d OD pairs"
                % (
                    self._hdv_gawron_mode,
                    self._hdv_gawron_lr,
                    self._hdv_gawron_alpha_zero,
                    self._hdv_gawron_beta,
                    self._hdv_gawron_update_mode,
                    len(self._hdv_route_probs),
                )
            )
        elif self._use_routerl_hdv_blackbox:
            print("  HDV update: RouteRL black-box human learning (per-agent internal state)")
        elif self._skip_hdv_route_registry:
            print("  HDV route-registry initialization skipped for this collector path.")

        if self._loaded_hdv_state is not None and self._hdv_route_probs is not None:
            restored = restore_hdv_route_state(
                self._loaded_hdv_state,
                self._hdv_route_probs,
                self._hdv_route_costs,
            )
            if restored["restored_probs"] > 0:
                print(
                    "  HDV route-policy restored: probs=%d costs=%d skipped=%d"
                    % (
                        restored["restored_probs"],
                        restored["restored_costs"],
                        restored["skipped"],
                    )
                )

        print(f"DataCollector initialized (backend={backend})")
        print(f"  Output: {output_dir}  |  K={number_of_paths}  |  img_size={img_size}")
        print(f"  av_fraction={av_fraction}  random_av_fraction={random_av_fraction}  epsilon={epsilon}")
        print(f"  av_candidate_source={self._av_candidate_source}")
        if self._safe_reroute_min_remaining_m >= 0.0:
            print(f"  safe_reroute_min_remaining_m={self._safe_reroute_min_remaining_m:.1f}")
        if self._completion_drain_steps > 0:
            print(f"  completion_drain_steps={self._completion_drain_steps}")
            if self._completion_drain_patience_steps > 0:
                print(f"  completion_drain_patience_steps={self._completion_drain_patience_steps}")
            if self._min_arrival_fraction_for_incomplete < 1.0:
                print(
                    "  min_arrival_fraction_for_incomplete="
                    f"{self._min_arrival_fraction_for_incomplete:.3f}"
                )
        if self._sumo_time_to_teleport:
            print(f"  sumo_time_to_teleport={self._sumo_time_to_teleport}")
        if self._dagger_active:
            print(f"  DAgger: beta={dagger_beta}  noise={dagger_noise}")

    # ------------------------------------------------------------------
    # DAgger policy loading
    # ------------------------------------------------------------------

    def _load_dagger_policy(self, policy_ckpt: str, n_routes: int) -> None:
        """Load world model encoder and MARL policy for DAgger guidance (CPU)."""
        try:
            from .urban_world_model import UrbanWorldModel
            from .train_marl import GaussianRoutePolicy
        except ImportError:
            from urban_world_model import UrbanWorldModel
            from train_marl import GaussianRoutePolicy

        wm = UrbanWorldModel(
            flow_projector_ckpt="",  # TODO: pass projector ckpt for DAgger
            dynamics_ckpt=None,
            edge_registry_path="",
            device="cpu",
            use_amp=False,
        )
        obs_dim = wm.n_spatial * wm.d_spatial
        self._dagger_encoder = wm

        ckpt = torch.load(policy_ckpt, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("cfg", {})
        hidden_dim      = cfg_dict.get("hidden_dim",      256)
        n_hidden_layers = cfg_dict.get("n_hidden_layers", 2)

        policy = GaussianRoutePolicy(
            obs_dim=obs_dim,
            n_routes=n_routes,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )
        policy.load_state_dict(ckpt["policy"])
        policy.eval()
        self._dagger_policy = policy
        self._dagger_active = True
        print(f"  [DAgger] policy loaded from {policy_ckpt}")

    # ------------------------------------------------------------------
    # Route candidate generation and matching
    # ------------------------------------------------------------------

    def _route_vclass(self) -> str:
        return str(getattr(self, "route_vclass", "passenger") or "passenger")

    def _route_is_valid_for_sumo(self, route: List[str]) -> bool:
        """Return True only if SUMO can accept this edge sequence via setRoute."""
        if not route:
            return False
        net = getattr(self.connector, "network", None)
        if net is None:
            return True
        vclass = self._route_vclass()
        try:
            edge_objs = [net.getEdge(str(edge_id)) for edge_id in route if str(edge_id)]
        except Exception:
            return False
        if len(edge_objs) != len(route):
            return False
        try:
            if any(not edge.allows(vclass) for edge in edge_objs):
                return False
        except Exception:
            pass
        for edge, next_edge in zip(edge_objs, edge_objs[1:]):
            try:
                allowed = edge.getAllowedOutgoing(vclass)
            except Exception:
                try:
                    allowed = edge.getOutgoing().keys()
                except Exception:
                    return False
            if next_edge not in allowed:
                return False
        return True

    @staticmethod
    def _edge_id_from_lane_id(lane_id: str) -> str:
        lane = str(lane_id or "")
        if not lane:
            return ""
        if lane.startswith(":"):
            return lane
        if "_" not in lane:
            return lane
        return lane.rsplit("_", 1)[0]

    def _lane_links_to_edge(self, lane_id: str, next_edge: str) -> Optional[bool]:
        """Return whether a SUMO lane has an outgoing link to ``next_edge``.

        ``None`` means the lane link data could not be inspected.
        """
        if not TRACI_AVAILABLE:
            return None
        try:
            links = traci.lane.getLinks(str(lane_id))
        except Exception:
            return None
        found_links = False
        for link in links or []:
            try:
                target_lane = str(link[0])
            except Exception:
                continue
            if not target_lane:
                continue
            found_links = True
            if self._edge_id_from_lane_id(target_lane) == str(next_edge):
                return True
        if not found_links:
            return None
        return False

    def _edge_has_lane_link_to_edge(self, edge_id: str, next_edge: str) -> Optional[bool]:
        """Return whether any lane on ``edge_id`` can continue to ``next_edge``."""
        if not TRACI_AVAILABLE:
            return None
        try:
            n_lanes = int(traci.edge.getLaneNumber(str(edge_id)))
        except Exception:
            return None
        inspected = False
        for lane_idx in range(max(n_lanes, 0)):
            lane_id = f"{edge_id}_{lane_idx}"
            lane_ok = self._lane_links_to_edge(lane_id, next_edge)
            if lane_ok is True:
                return True
            if lane_ok is not None:
                inspected = True
        return False if inspected else None

    def _lane_remaining_distance(self, veh_id: str) -> Optional[float]:
        if not TRACI_AVAILABLE:
            return None
        try:
            lane_id = traci.vehicle.getLaneID(str(veh_id))
            lane_len = float(traci.lane.getLength(lane_id))
            lane_pos = float(traci.vehicle.getLanePosition(str(veh_id)))
            remaining = lane_len - lane_pos
        except Exception:
            return None
        if not np.isfinite(remaining):
            return None
        return max(float(remaining), 0.0)

    def _current_route_suffix(self, veh_id: str, current_edge: str) -> List[str]:
        if not TRACI_AVAILABLE:
            return []
        try:
            full_route = [str(edge_id) for edge_id in traci.vehicle.getRoute(str(veh_id))]
        except Exception:
            return []
        if not full_route:
            return []
        try:
            route_idx = int(traci.vehicle.getRouteIndex(str(veh_id)))
        except Exception:
            route_idx = -1
        suffix: List[str] = []
        if 0 <= route_idx < len(full_route):
            suffix = list(full_route[route_idx:])
        if not suffix or suffix[0] != str(current_edge):
            matches = [idx for idx, edge_id in enumerate(full_route) if edge_id == str(current_edge)]
            if matches:
                suffix = list(full_route[int(matches[-1]):])
        if suffix and suffix[0] == str(current_edge):
            return suffix
        return []

    def _route_is_safe_for_current_lane(self, veh_id: str, route: List[str]) -> bool:
        """Conservative runtime guard before treating a candidate as controllable.

        The edge sequence may be globally valid while the vehicle's current lane
        can no longer reach the candidate's next edge.  In that case frequent
        random rerouting can leave SUMO vehicles stopped at a junction with
        "no connection to the next edge".  We only allow the reroute if the
        current lane directly reaches the next edge, or if there is enough
        remaining distance for a lane change and another lane on the same edge
        reaches it.
        """
        if not route or not self._route_is_valid_for_sumo(route):
            return False
        if self._safe_reroute_min_remaining_m < 0.0 or len(route) < 2:
            return True
        if not TRACI_AVAILABLE:
            return True
        try:
            current_edge = str(traci.vehicle.getRoadID(str(veh_id)))
            lane_id = str(traci.vehicle.getLaneID(str(veh_id)))
        except Exception:
            return False
        if (not current_edge) or current_edge.startswith(":") or route[0] != current_edge:
            return False
        if (not lane_id) or lane_id.startswith(":"):
            return False

        next_edge = str(route[1])
        lane_ok = self._lane_links_to_edge(lane_id, next_edge)
        if lane_ok is True:
            return True

        remaining = self._lane_remaining_distance(veh_id)
        if remaining is None or remaining < float(self._safe_reroute_min_remaining_m):
            return False

        edge_ok = self._edge_has_lane_link_to_edge(current_edge, next_edge)
        if edge_ok is not None:
            return bool(edge_ok)

        return lane_ok is None

    def _filter_safe_reroute_candidates(
        self,
        veh_id: str,
        current_edge: str,
        candidates: List[List[str]],
    ) -> List[List[str]]:
        fallback = self._current_route_suffix(veh_id, current_edge)
        out: List[List[str]] = []
        for route in candidates[: self.n_routes]:
            if route and self._route_is_safe_for_current_lane(veh_id, route):
                out.append(list(route))
            elif fallback:
                out.append(list(fallback))
            else:
                out.append([])
        while len(out) < self.n_routes:
            out.append(list(fallback) if fallback else [])
        first_valid = next((list(route) for route in out if route), [])
        if not first_valid:
            return []
        return [list(route) if route else list(first_valid) for route in out[: self.n_routes]]

    def _normalize_valid_candidate_bank(self, candidates: List[List[str]]) -> List[List[str]]:
        valid: List[List[str]] = []
        seen = set()
        for route in candidates:
            route_list = [str(edge_id) for edge_id in route if str(edge_id)]
            key = tuple(route_list)
            if not route_list or key in seen:
                continue
            if not self._route_is_valid_for_sumo(route_list):
                continue
            valid.append(route_list)
            seen.add(key)
            if len(valid) >= self.n_routes:
                break
        if not valid:
            return []
        while len(valid) < self.n_routes:
            valid.append(list(valid[0]))
        return valid[: self.n_routes]

    def _get_candidates(self, from_edge_id: str, dest_edge_id: str) -> List[List[str]]:
        """
        Return K candidate routes from from_edge_id to dest_edge_id.

        Cached by (from_edge_id, dest_edge_id) so that multiple vehicles on the
        same edge share the same candidates. Cache is cleared each episode.

        IMPORTANT: routes always start with from_edge_id, so they are valid for
        setRoute() only when the vehicle's current edge == from_edge_id.
        """
        key = (from_edge_id, dest_edge_id)
        if key in self._route_cache:
            return self._route_cache[key]

        net = self.connector.network
        if net is None:
            return []

        try:
            dest_edge = net.getEdge(dest_edge_id)
            src_edge  = net.getEdge(from_edge_id)
            vclass = self._route_vclass()
            try:
                outgoing = list(src_edge.getAllowedOutgoing(vclass))
            except Exception:
                outgoing = list(src_edge.getOutgoing().keys())

            routes: List[List[str]] = []
            for out_edge in outgoing:
                try:
                    result = net.getFastestPath(out_edge, dest_edge, vClass=vclass)
                except TypeError:
                    try:
                        result = net.getShortestPath(out_edge, dest_edge)
                    except Exception:
                        result = None
                except Exception:
                    result = None
                try:
                    if result and len(result[0]) > 0:
                        # result[0] starts with out_edge, prepend from_edge_id so
                        # the route begins at the vehicle's current position.
                        routes.append([from_edge_id] + [e.getID() for e in result[0]])
                except Exception:
                    continue
                if len(routes) >= self.n_routes:
                    break

            routes = self._normalize_valid_candidate_bank(routes)
            self._route_cache[key] = routes
            return routes
        except Exception:
            return []

    def _precomputed_od_candidates(
        self,
        od_key: Optional[Tuple[int, int]],
    ) -> List[List[str]]:
        """Return pre-generated OD routes from routes.csv when available."""
        if od_key is None:
            return []
        od_route_paths = getattr(self.connector, "_od_route_paths", None)
        if not isinstance(od_route_paths, dict):
            return []
        paths = od_route_paths.get(od_key, [])
        if not paths:
            return []
        return [list(path) for path in paths[: self.n_routes]]

    def _match_route_idx(self, candidates: List[List[str]], sumo_route: List[str]) -> int:
        """
        Match a SUMO route (edge list) to one of the K candidate routes.
        Uses maximum edge overlap. Returns 0 if candidates is empty.
        """
        if not candidates:
            return 0
        sumo_set = set(sumo_route)
        best_idx, best_overlap = 0, -1
        for i, cand in enumerate(candidates):
            overlap = len(set(cand) & sumo_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        return best_idx

    def _candidate_route_suffixes(
        self,
        veh_id: str,
        current_edge: str,
    ) -> List[List[str]]:
        """
        Build reroute-able suffixes from the vehicle's pre-computed full routes.

        The full candidate set is generated once when the vehicle first appears.
        During the episode we do not re-run shortest path search; we only select
        a route index and trim that route so it starts at the vehicle's current
        edge before calling ``setRoute``.
        """
        if (not current_edge) or current_edge.startswith(":"):
            return []

        full_candidates = self._veh_candidates.get(veh_id, [])
        suffixes: List[List[str]] = []
        fallback = self._current_route_suffix(veh_id, current_edge)
        for route in full_candidates[: self.n_routes]:
            matches = [idx for idx, edge_id in enumerate(route) if edge_id == current_edge]
            if not matches:
                suffixes.append(list(fallback) if fallback else [])
                continue
            start_idx = int(matches[-1])
            suffix = list(route[start_idx:])
            if suffix and suffix[0] == current_edge:
                suffixes.append(suffix)
            else:
                suffixes.append(list(fallback) if fallback else [])

        while len(suffixes) < self.n_routes:
            suffixes.append(list(fallback) if fallback else [])
        first_valid = next((list(route) for route in suffixes if route), [])
        if not first_valid:
            return []
        return self._normalize_valid_candidate_bank(
            [list(route) if route else list(first_valid) for route in suffixes[: self.n_routes]]
        )

    # ------------------------------------------------------------------
    # Per-episode vehicle role assignment
    # ------------------------------------------------------------------

    def _assign_roles(self, vehicle_ids: List[str], rng: np.random.RandomState) -> None:
        """Randomly assign vehicles as AV or HDV and split AVs into sub-groups."""
        if self._fixed_machine_ids or self._fixed_human_ids:
            av_ids = {vid for vid in vehicle_ids if vid in self._fixed_machine_ids}
            hdv_ids = {vid for vid in vehicle_ids if vid in self._fixed_human_ids}
            unresolved = set(vehicle_ids) - av_ids - hdv_ids
            hdv_ids |= unresolved
            self._av_vehicles |= av_ids
            self._hdv_vehicles |= hdv_ids
            # Legacy warmup semantics: fixed CAVs all use epsilon-greedy routing.
            self._greedy_avs |= av_ids
            return

        n_av = max(0, int(round(len(vehicle_ids) * self.av_fraction)))
        av_ids = set(rng.choice(vehicle_ids, size=n_av, replace=False).tolist() if n_av > 0 else [])

        n_random = max(0, int(round(n_av * self.random_av_fraction)))
        av_list = list(av_ids)
        rng.shuffle(av_list)

        self._av_vehicles |= av_ids
        self._hdv_vehicles |= set(vehicle_ids) - av_ids
        self._random_avs |= set(av_list[:n_random])
        self._greedy_avs |= set(av_list[n_random:])

    def set_fixed_roles(
        self,
        machine_ids: List[str],
        human_ids: Optional[List[str]] = None,
    ) -> None:
        self._fixed_machine_ids = set(str(x) for x in machine_ids)
        self._fixed_human_ids = set(str(x) for x in (human_ids or []))

    def _register_vehicles(self, vehicle_ids: List[str], rng: np.random.RandomState) -> None:
        """
        Assign roles and record departure info for newly seen vehicles.
        Pre-computes route candidates from origin edge (one-time dijkstra cost).
        """
        new_ids = [v for v in vehicle_ids if v not in self._av_vehicles and v not in self._hdv_vehicles]
        if new_ids:
            self._assign_roles(new_ids, rng)

        if not TRACI_AVAILABLE:
            return

        for veh_id in vehicle_ids:
            try:
                traci.vehicle.setType(
                    veh_id,
                    "AV" if veh_id in self._av_vehicles else "Human",
                )
            except Exception:
                pass
            if veh_id in self._veh_dest:
                continue  # already registered
            try:
                current_edge = traci.vehicle.getRoadID(veh_id)
                if not current_edge or current_edge.startswith(":"):
                    continue
                full_route = list(traci.vehicle.getRoute(veh_id))
                if len(full_route) < 2:
                    continue
                dest_edge_id = full_route[-1]
                self._veh_dest[veh_id] = dest_edge_id

                od_key: Optional[Tuple[int, int]] = None
                if hasattr(self.connector, '_agents_df') and veh_id not in self._veh_od:
                    try:
                        row = self.connector._agents_df[
                            self.connector._agents_df['id'].astype(str) == veh_id
                        ]
                        if len(row) > 0:
                            od_key = (
                                int(row.iloc[0]['origin']),
                                int(row.iloc[0]['destination']),
                            )
                            self._veh_od[veh_id] = od_key
                    except Exception:
                        od_key = None
                if od_key is None:
                    od_key = self._veh_od.get(veh_id)

                # Prefer the pre-generated OD route table from routes.csv so the
                # policy's route index always refers to the same candidate set.
                candidates = self._precomputed_od_candidates(od_key)
                if not candidates:
                    candidates = self._get_candidates(current_edge, dest_edge_id)
                self._veh_candidates[veh_id] = candidates

                # Record departure route index for HDVs (fixed for the episode)
                if veh_id in self._hdv_vehicles:
                    self._hdv_route_idx[veh_id] = self._match_route_idx(candidates, full_route)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # AV re-routing policy
    # ------------------------------------------------------------------

    def _select_av_actions(
        self, state: Dict[str, Any], rng: np.random.RandomState
    ) -> Dict[str, List[str]]:
        """
        Choose a route for every AV currently in the network.

        Route indices are selected fresh each step.  By default candidates are
        searched from the vehicle's current edge to its destination; od_suffix
        preserves deprecated legacy pre-generated OD-route suffix behavior.

        Returns {veh_id: route_edge_list}.
        """
        in_network = set(state["vehicles"].keys())
        actions: Dict[str, List[str]] = {}
        self._last_av_decision_trace = []

        for veh_id in self._av_vehicles & in_network:
            veh_info = state["vehicles"][veh_id]
            current_edge = veh_info.get("edge", "")
            if not current_edge or current_edge.startswith(":"):
                continue  # on junction-internal edge, cannot reroute

            candidates: List[List[str]] = []
            if self._av_candidate_source == "current_to_dest":
                dest_edge = self._veh_dest.get(veh_id, "")
                if dest_edge:
                    candidates = self._get_candidates(current_edge, dest_edge)
            if not candidates:
                candidates = self._candidate_route_suffixes(veh_id, current_edge)
            if not candidates:
                current_suffix = self._current_route_suffix(veh_id, current_edge)
                if current_suffix:
                    actions[veh_id] = current_suffix
                continue

            raw_candidates = list(candidates)
            raw_candidate_count = int(len(raw_candidates))
            safe_candidates = self._filter_safe_reroute_candidates(veh_id, current_edge, candidates)
            if not safe_candidates:
                current_suffix = self._current_route_suffix(veh_id, current_edge)
                if current_suffix:
                    actions[veh_id] = current_suffix
                    self._last_ep_unsafe_reroute_keeps += 1
                    incident_edge = str(getattr(self, "_incident_edge", "") or "")
                    self._last_av_decision_trace.append({
                        "veh_id": str(veh_id),
                        "current_edge": str(current_edge),
                        "dest_edge": str(self._veh_dest.get(veh_id, "")),
                        "selected_route": -1,
                        "candidate_count": raw_candidate_count,
                        "selected_contains_incident": int(bool(incident_edge and incident_edge in current_suffix)),
                        "candidate_contains_incident": int(
                            sum(1 for route in raw_candidates[: self.n_routes] if incident_edge and incident_edge in route)
                        ),
                        "reroute_safe": 0,
                        "selected_path": " ".join(map(str, current_suffix)),
                    })
                continue
            candidates = safe_candidates

            if self._fixed_av_route_index >= 0:
                idx = min(max(int(self._fixed_av_route_index), 0), len(candidates) - 1)
            elif self._fixed_av_route_probs is not None:
                K_eff = min(len(candidates), self.n_routes)
                probs = np.asarray(self._fixed_av_route_probs[:K_eff], dtype=np.float64)
                probs = np.clip(probs, 0.0, None)
                total = float(probs.sum())
                if total > 0.0:
                    probs = probs / total
                else:
                    probs = np.ones((K_eff,), dtype=np.float64) / float(K_eff)
                idx = int(rng.choice(K_eff, p=probs))
            elif self._step_av_policy_frac is not None:
                # DAgger: sample route from policy distribution (already noise-mixed)
                av_frac = self._step_av_policy_frac
                K_eff = min(len(candidates), self.n_routes)
                probs = av_frac[:K_eff].copy()
                probs = np.clip(probs, 0, None)
                s = probs.sum()
                if s > 0:
                    probs /= s
                else:
                    probs = np.ones(K_eff, dtype=np.float32) / K_eff
                idx = int(rng.choice(K_eff, p=probs))
            elif veh_id in self._random_avs:
                idx = int(rng.randint(0, len(candidates)))
            else:
                if rng.rand() < self.epsilon:
                    idx = int(rng.randint(0, len(candidates)))
                else:
                    idx = self._greedy_route_idx(candidates, state)

            self._av_route_idx[veh_id] = idx
            selected_route = candidates[idx]
            actions[veh_id] = selected_route
            self._last_ep_safe_reroutes += 1
            incident_edge = str(getattr(self, "_incident_edge", "") or "")
            self._last_av_decision_trace.append({
                "veh_id": str(veh_id),
                "current_edge": str(current_edge),
                "dest_edge": str(self._veh_dest.get(veh_id, "")),
                "selected_route": int(idx),
                "candidate_count": raw_candidate_count,
                "selected_contains_incident": int(bool(incident_edge and incident_edge in selected_route)),
                "candidate_contains_incident": int(
                    sum(1 for route in raw_candidates[: self.n_routes] if incident_edge and incident_edge in route)
                ),
                "reroute_safe": 1,
                "selected_path": " ".join(map(str, selected_route)),
            })

        return actions

    def _record_av_decision_trace(self, episode_id: int, decision_idx: int, sim_step: int) -> None:
        for row in getattr(self, "_last_av_decision_trace", []):
            out = {
                "episode": int(episode_id),
                "decision_idx": int(decision_idx),
                "sim_step": int(sim_step),
            }
            out.update(row)
            self._collection_decision_trace_rows.append(out)

    def _collect_flow(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract (E, 8) flow matrix from SUMO step state."""
        return self._edge_registry.extract_flow(state.get("edges", {}))

    def _compute_edge_distribution(
        self,
        state: Dict[str, Any],
        active_vehicle_ids: set[str],
    ) -> np.ndarray:
        """Compute normalized per-edge occupancy distribution for a vehicle subset."""
        out = np.zeros((self._edge_registry.n_edges,), dtype=np.float32)
        vehicles = state.get("vehicles", {})
        if not vehicles:
            return out
        edge_to_idx = getattr(self._edge_registry, "_edge_to_idx", {})
        in_network = set(vehicles.keys())
        active = active_vehicle_ids & in_network
        for veh_id in active:
            veh = vehicles.get(veh_id)
            if not isinstance(veh, dict):
                continue
            edge_id = str(veh.get("edge", ""))
            if (not edge_id) or edge_id.startswith(":"):
                continue
            ei = edge_to_idx.get(edge_id)
            if ei is None:
                continue
            out[int(ei)] += 1.0
        s = float(out.sum())
        if s > 0.0:
            out /= s
        return out

    def _compute_av_edge_distribution(self, state: Dict[str, Any]) -> np.ndarray:
        """Compute per-edge AV occupancy distribution p(edge) from current vehicles."""
        return self._compute_edge_distribution(state, self._av_vehicles)

    def _compute_hdv_edge_distribution(self, state: Dict[str, Any]) -> np.ndarray:
        """Compute per-edge HDV occupancy distribution p(edge) from current vehicles."""
        return self._compute_edge_distribution(state, self._hdv_vehicles)

    def _compute_joint_edge_distribution(self, state: Dict[str, Any]) -> np.ndarray:
        """Compute per-edge joint AV+HDV occupancy distribution p(edge)."""
        return self._compute_edge_distribution(state, self._av_vehicles | self._hdv_vehicles)

    def _edge_travel_time_estimate(self, edge_id: str, state: Dict[str, Any]) -> float:
        """Estimate edge traversal time from current state speed, with free-flow fallback."""
        edge_to_idx = getattr(self._edge_registry, "_edge_to_idx", {})
        ei = edge_to_idx.get(str(edge_id))
        if ei is None:
            return 1.0
        lengths = getattr(self._edge_registry, "lengths", None)
        speed_limits = getattr(self._edge_registry, "speed_limits", None)
        length = float(lengths[int(ei)]) if lengths is not None else 100.0
        speed_limit = float(speed_limits[int(ei)]) if speed_limits is not None else 13.89
        edge_state = state.get("edges", {}).get(str(edge_id), {}) if isinstance(state, dict) else {}
        speed = 0.0
        if isinstance(edge_state, dict):
            speed = float(edge_state.get("mean_speed", 0.0) or 0.0)
        if not np.isfinite(speed) or speed <= 0.1:
            speed = speed_limit
        return float(length / max(speed, 0.1))

    def _compute_planned_av_edge_flow(
        self,
        state: Dict[str, Any],
        av_actions: Dict[str, List[str]],
        *,
        horizon_seconds: float,
    ) -> np.ndarray:
        """
        Encode selected AV routes as a fixed edge-aligned controllable action.

        The saved action is computed before SUMO advances.  For every active AV
        with a selected route, edges expected to be entered within the next
        decision interval receive one count, then counts are normalized by the
        number of active controlled AVs.  This represents intended future route
        pressure, not post-action observed occupancy.
        """
        out = np.zeros((self._edge_registry.n_edges,), dtype=np.float32)
        if not av_actions:
            return out

        vehicles = state.get("vehicles", {}) if isinstance(state, dict) else {}
        edge_to_idx = getattr(self._edge_registry, "_edge_to_idx", {})
        horizon = max(float(horizon_seconds), 1.0)
        n_controlled = 0

        for veh_id, route in av_actions.items():
            veh = vehicles.get(veh_id)
            if not isinstance(veh, dict) or not route:
                continue
            current_edge = str(veh.get("edge", ""))
            if not current_edge or current_edge.startswith(":"):
                continue

            route_edges = [str(eid) for eid in route if str(eid)]
            if not route_edges:
                continue
            n_controlled += 1

            try:
                start_idx = route_edges.index(current_edge)
            except ValueError:
                start_idx = 0

            eta = 0.0
            for pos in range(start_idx, len(route_edges)):
                edge_id = route_edges[pos]

                # The vehicle already occupies current_edge at decision time;
                # count future planned entries, not current occupancy.
                if pos == start_idx and edge_id == current_edge:
                    eta += self._edge_travel_time_estimate(edge_id, state)
                    continue

                if eta >= horizon:
                    break
                ei = edge_to_idx.get(edge_id)
                if ei is not None:
                    out[int(ei)] += 1.0
                eta += self._edge_travel_time_estimate(edge_id, state)

        if n_controlled > 0:
            out /= float(n_controlled)
        return out

    def _greedy_route_idx(self, candidates: List[List[str]], state: Dict[str, Any]) -> int:
        """Score each candidate route by mean speed minus weighted occupancy."""
        best_idx, best_score = 0, -float("inf")
        edges_state = state["edges"]
        for i, route in enumerate(candidates):
            score = sum(
                edges_state.get(eid, {}).get("mean_speed", 0.0)
                - edges_state.get(eid, {}).get("occupancy", 0.0) * 10.0
                for eid in route
            )
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _apply_reroutings(self, actions: Dict[str, List[str]]) -> None:
        """Apply chosen routes to AV vehicles via traci."""
        if not TRACI_AVAILABLE:
            return
        for veh_id, route in actions.items():
            try:
                # Candidate routes are generated from a specific source edge.
                # Apply only when the vehicle is still on that source edge.
                current_edge = traci.vehicle.getRoadID(veh_id)
                if not route or not current_edge or current_edge.startswith(":"):
                    continue
                if route[0] != current_edge:
                    continue
                if not self._route_is_safe_for_current_lane(veh_id, route):
                    continue
                current_suffix = self._current_route_suffix(veh_id, current_edge)
                if current_suffix and tuple(current_suffix) == tuple(route):
                    continue
                traci.vehicle.setRoute(veh_id, route)
            except Exception:
                pass

    def _update_hdv_gawron(self, final_flow: np.ndarray) -> None:
        """Update HDV route probabilities using Gawron based on edge travel times.

        Args:
            final_flow: (E, 8) flow matrix from the last step of the episode.
        """
        rr = self._route_registry
        er = self._edge_registry

        t_e = self._edge_travel_times(final_flow)

        for od_idx, od_key in _iter_route_registry_od_rows(rr):
            if od_key not in self._hdv_route_probs:
                continue
            if od_idx < 0 or od_idx >= len(rr.route_edge_indices):
                continue
            route_masks_od = getattr(rr, "route_masks", None)
            if route_masks_od is not None:
                route_tt = route_masks_od[od_idx] @ t_e  # (K,)
            else:
                route_tt = np.asarray(
                    [sum(float(t_e[e]) for e in edges) for edges in rr.route_edge_indices[od_idx]],
                    dtype=np.float32,
                )
            hdv_frac = self._hdv_route_probs[od_key]
            hdv_cost = (
                self._hdv_route_costs.get(od_key)
                if self._hdv_route_costs is not None else None
            )
            new_probs, new_cost = gawron_step_from_route_tt(
                probs=hdv_frac,
                route_tt=route_tt,
                mode=self._hdv_gawron_mode,
                lr=self._hdv_gawron_lr,
                cost=hdv_cost,
                alpha_zero=self._hdv_gawron_alpha_zero,
                beta=self._hdv_gawron_beta,
                update_mode=self._hdv_gawron_update_mode,
                min_prob=self._hdv_min_prob,
            )
            self._hdv_route_probs[od_key] = new_probs.astype(np.float32)
            if self._hdv_route_costs is not None and new_cost is not None:
                self._hdv_route_costs[od_key] = new_cost.astype(np.float32)

    def _update_hdv_gawron_actual(self) -> None:
        """Update HDV route probabilities using actual per-route travel times.

        Uses self._last_ep_route_tts: {(oi, di, k): [tt1, tt2, ...]}
        collected from traci arrival data during the episode.
        """
        rr = self._route_registry
        # Build per-OD route travel times: {(oi, di): (K,) array}
        od_route_tt: Dict[Tuple[int, int], np.ndarray] = {}
        for (oi, di, k), tts in self._last_ep_route_tts.items():
            od_key = (oi, di)
            if od_key not in od_route_tt:
                od_idx = _route_registry_od_index(rr, oi, di)
                if od_idx < 0 or od_idx >= len(rr.route_edge_indices):
                    continue
                K = len(rr.route_edge_indices[od_idx])
                od_route_tt[od_key] = np.full(K, np.nan, dtype=np.float32)
            if int(k) < 0 or int(k) >= od_route_tt[od_key].shape[0]:
                continue
            od_route_tt[od_key][k] = float(np.mean(tts))

        # Update each OD pair that has observed travel times
        for od_key, route_tt in od_route_tt.items():
            if od_key not in self._hdv_route_probs:
                continue
            # Fill NaN routes (unobserved) with the max observed time
            observed = route_tt[~np.isnan(route_tt)]
            if len(observed) == 0:
                continue
            fill_val = float(observed.max()) * 1.5  # penalise unobserved routes
            route_tt = np.where(np.isnan(route_tt), fill_val, route_tt)

            hdv_frac = self._hdv_route_probs[od_key]
            hdv_cost = (
                self._hdv_route_costs.get(od_key)
                if self._hdv_route_costs is not None else None
            )
            new_probs, new_cost = gawron_step_from_route_tt(
                probs=hdv_frac,
                route_tt=route_tt,
                mode=self._hdv_gawron_mode,
                lr=self._hdv_gawron_lr,
                cost=hdv_cost,
                alpha_zero=self._hdv_gawron_alpha_zero,
                beta=self._hdv_gawron_beta,
                update_mode=self._hdv_gawron_update_mode,
                min_prob=self._hdv_min_prob,
            )
            self._hdv_route_probs[od_key] = new_probs.astype(np.float32)
            if self._hdv_route_costs is not None and new_cost is not None:
                self._hdv_route_costs[od_key] = new_cost.astype(np.float32)

    def _edge_travel_times(self, flow: np.ndarray) -> np.ndarray:
        """Compute per-edge travel times (seconds) from (E, 8) flow matrix."""
        er = self._edge_registry
        speeds = flow[:, 1] * er.speed_limits  # norm_speed * speed_limit
        speeds = np.maximum(speeds, 0.1)
        lengths = er.lengths if er.lengths is not None else np.ones(er.n_edges) * 100.0
        return lengths / speeds  # (E,)

    def _global_avg_travel_time(self, flow: np.ndarray) -> float:
        """Compute per-vehicle average full OD route travel time (seconds).

        For each OD pair, sums edge travel times along each route, weights
        by vehicle counts (HDV fracs from Gawron + AV assumed uniform),
        and averages across all vehicles.
        Requires RouteRegistry to be loaded.
        """
        rr = self._route_registry
        if rr is None:
            return 0.0

        t_e = self._edge_travel_times(flow)

        total_tt = 0.0
        total_veh = 0.0
        for od_idx, od_key in _iter_route_registry_od_rows(rr):
            if od_idx < 0 or od_idx >= len(rr.route_edge_indices):
                continue
            n_av = float(rr.od_av_counts[od_idx])
            n_hdv = float(rr.od_hdv_counts[od_idx])
            if n_av + n_hdv == 0:
                continue

            # HDV route fracs from Gawron state; AV assumed uniform
            K_od = len(rr.route_edge_indices[od_idx])
            if K_od <= 0:
                continue
            hdv_frac = self._hdv_route_probs.get(od_key) if self._hdv_route_probs else None
            if hdv_frac is None:
                hdv_frac = np.ones(K_od, dtype=np.float32) / K_od
            av_frac = np.ones(K_od, dtype=np.float32) / K_od

            for k in range(K_od):
                edges = rr.route_edge_indices[od_idx][k]
                route_tt = sum(t_e[e] for e in edges)  # full route travel time
                n_veh_k = n_hdv * hdv_frac[k] + n_av * av_frac[k]
                total_tt += n_veh_k * route_tt
                total_veh += n_veh_k

        return total_tt / max(total_veh, 1.0)

    def _decision_tick_map(self, max_steps: int) -> Dict[int, int]:
        if self._decision_steps <= 0 or max_steps <= 0:
            return {t: t for t in range(max_steps)}
        horizon = max(1, min(int(max_steps), int(self._decision_horizon_seconds)))
        ticks = np.linspace(
            0.0,
            float(max(horizon - 1, 0)),
            num=max(int(self._decision_steps), 1),
            endpoint=True,
        )
        tick_ids = np.clip(np.rint(ticks).astype(np.int64), 0, max_steps - 1)
        tick_ids = np.unique(tick_ids)
        return {int(tick): idx for idx, tick in enumerate(tick_ids.tolist())}

    def _decision_depart_edge(self, decision_idx: int) -> np.ndarray:
        if self._departure_schedule is None:
            return np.zeros((self._edge_registry.n_edges,), dtype=np.float32)
        dep = self._departure_schedule.get(int(decision_idx), {})
        depart_edge = dep.get("depart_edge")
        if depart_edge is None:
            return np.zeros((self._edge_registry.n_edges,), dtype=np.float32)
        return np.asarray(depart_edge, dtype=np.float32).copy()

    def _decision_context(self, state: Dict[str, Any], sim_step: int) -> np.ndarray:
        active_ids = set(state.get("vehicles", {}).keys())
        n_active_av = len(self._av_vehicles & active_ids)
        n_active_hdv = len(self._hdv_vehicles & active_ids)
        return np.array([
            float(sim_step) / max(float(self._decision_horizon_seconds), 1.0),
            float(n_active_av) / max(int(self._expected_n_av), 1),
            float(n_active_hdv) / max(int(self._expected_n_hdv), 1),
        ], dtype=np.float32)

    def _seed_depart_times_from_agents(self, veh_depart: Dict[str, float]) -> None:
        """Preload scheduled departures so between-decision arrivals are counted."""
        agents_df = getattr(self.connector, "_agents_df", None)
        if agents_df is None:
            return
        columns = set(getattr(agents_df, "columns", []))
        if "id" not in columns or "start_time" not in columns:
            return
        try:
            ids = agents_df["id"].astype(str).tolist()
            starts = agents_df["start_time"].astype(float).tolist()
        except Exception:
            return
        for veh_id, depart in zip(ids, starts):
            try:
                depart_f = float(depart)
            except Exception:
                continue
            if veh_id not in veh_depart and np.isfinite(depart_f):
                veh_depart[str(veh_id)] = depart_f

    def _record_depart_times(self, state: Dict[str, Any], veh_depart: Dict[str, float]) -> None:
        if not TRACI_AVAILABLE:
            return
        for veh_id in state.get("vehicles", {}):
            if veh_id in veh_depart:
                continue
            try:
                veh_depart[veh_id] = traci.vehicle.getDeparture(veh_id)
            except Exception:
                veh_depart[veh_id] = float(self.connector.current_step)

    def _record_arrivals(
        self,
        veh_depart: Dict[str, float],
        veh_travel_times: Dict[str, float],
        *,
        track_role_breakdown: bool = False,
        arrival_events: Optional[List[Tuple[str, float]]] = None,
    ) -> None:
        if not TRACI_AVAILABLE:
            return
        if arrival_events is None:
            try:
                sim_time = float(traci.simulation.getTime())
                arrived = traci.simulation.getArrivedIDList()
            except Exception:
                return
            events = [(str(veh_id), sim_time) for veh_id in arrived]
        else:
            events = [(str(veh_id), float(sim_time)) for veh_id, sim_time in arrival_events]
        for veh_id, sim_time in events:
            if veh_id in veh_travel_times:
                continue
            dep = veh_depart.get(veh_id)
            if dep is None:
                continue
            tt = float(sim_time - dep)
            self._last_ep_travel_times.append(tt)
            veh_travel_times[veh_id] = tt
            if track_role_breakdown:
                if hasattr(self, "_last_ep_vehicle_tt"):
                    self._last_ep_vehicle_tt[veh_id] = tt
                if hasattr(self, "_last_ep_av_travel_times") and veh_id in self._av_vehicles:
                    self._last_ep_av_travel_times.append(tt)
                elif hasattr(self, "_last_ep_hdv_travel_times") and veh_id in self._hdv_vehicles:
                    self._last_ep_hdv_travel_times.append(tt)
            od = self._veh_od.get(veh_id)
            ridx = self._hdv_route_idx.get(veh_id, self._av_route_idx.get(veh_id))
            if od is not None and ridx is not None:
                key = (od[0], od[1], ridx)
                self._last_ep_route_tts.setdefault(key, []).append(tt)

    def _sumo_min_expected(self) -> int:
        if not TRACI_AVAILABLE:
            return 0
        try:
            return int(traci.simulation.getMinExpectedNumber())
        except Exception:
            return 0

    def _drain_episode_to_completion(
        self,
        *,
        state: Dict[str, Any],
        current_tick: int,
        veh_depart: Dict[str, float],
        veh_travel_times: Dict[str, float],
    ) -> Tuple[Dict[str, Any], int]:
        """Advance SUMO after the recorded horizon until late arrivals clear."""
        total_agents = int(getattr(self, "_num_agents", 0) or 0)
        if self._completion_drain_steps <= 0 or total_agents <= 0:
            return state, int(current_tick)
        if len(self._last_ep_travel_times) >= total_agents:
            return state, int(current_tick)
        if (
            self._min_arrival_fraction_for_incomplete < 1.0
            and float(len(self._last_ep_travel_times)) / max(float(total_agents), 1.0)
            >= self._min_arrival_fraction_for_incomplete
        ):
            return state, int(current_tick)

        deadline = int(current_tick) + int(self._completion_drain_steps)
        tick = int(current_tick)
        stale_steps = 0
        while tick < deadline and len(self._last_ep_travel_times) < total_agents:
            pending = self._sumo_min_expected()
            active = len(state.get("vehicles", {})) if isinstance(state, dict) else 0
            if pending <= 0 and active <= 0:
                break
            step = min(60, max(deadline - tick, 0))
            if step <= 0:
                break
            try:
                arrival_events = self.connector.advance_steps(step)
            except Exception as exc:
                print(f"  [drain] advance failed: {exc}")
                break
            arrivals_before = len(self._last_ep_travel_times)
            tick += int(step)
            self._last_ep_completion_drain_steps += int(step)
            self._record_arrivals(
                veh_depart,
                veh_travel_times,
                arrival_events=arrival_events,
            )
            arrivals_after = len(self._last_ep_travel_times)
            if arrivals_after > arrivals_before:
                stale_steps = 0
            else:
                stale_steps += int(step)
                self._last_ep_completion_drain_stale_steps = int(stale_steps)
            if (
                self._min_arrival_fraction_for_incomplete < 1.0
                and float(arrivals_after) / max(float(total_agents), 1.0)
                >= self._min_arrival_fraction_for_incomplete
            ):
                break
            if (
                self._completion_drain_patience_steps > 0
                and stale_steps >= self._completion_drain_patience_steps
            ):
                break
            try:
                state = self.connector.get_traffic_state()
                self._record_depart_times(state, veh_depart)
            except Exception:
                state = {"vehicles": {}}
        return state, tick

    def _completion_aware_episode_reward(
        self,
        *,
        current_tick: int,
        max_steps: int,
        fallback_frame: Optional[np.ndarray],
    ) -> float:
        """
        Return -t_all minutes without dropping unfinished vehicles.

        Complete episodes use the actual arrived-vehicle mean.  In incomplete
        episodes, unfinished vehicles receive a conservative horizon travel-time
        penalty instead of disappearing from the label.
        """
        arrived = list(getattr(self, "_last_ep_travel_times", []))
        total_agents = int(getattr(self, "_num_agents", 0) or 0)
        if arrived and total_agents > 0:
            n_arrived = min(len(arrived), total_agents)
            unfinished = max(total_agents - n_arrived, 0)
            penalty_tt = float(max(int(current_tick), int(max_steps), 1))
            total_tt = float(np.sum(arrived[:n_arrived])) + float(unfinished) * penalty_tt
            return -float((total_tt / float(total_agents)) / 60.0)
        if arrived:
            return -float(np.mean(arrived) / 60.0)
        if fallback_frame is not None:
            return -float(self._global_avg_travel_time(fallback_frame) / 60.0)
        return 0.0

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    def collect_episodes(
        self,
        num_episodes: int,
        max_steps_per_episode: int = 1000,
        task_name: str = "urban_routing",
        warmup_episodes: int = 0,
    ) -> Dict[str, torch.Tensor]:
        all_episodes: List[int] = []
        all_av_edge_actions: List[np.ndarray] = []      # (E,) per step
        all_hdv_edge_actions: List[np.ndarray] = []     # (E,) per step
        all_joint_edge_actions: List[np.ndarray] = []   # (E,) per step
        all_rewards: List[float] = []
        all_flows: List[np.ndarray] = []
        all_departs: List[np.ndarray] = []
        all_contexts: List[np.ndarray] = []

        total_episodes = int(warmup_episodes) + int(num_episodes)
        print(
            f"Collecting {num_episodes} episodes (task={task_name})"
            + (f" after {warmup_episodes} warmup episodes" if warmup_episodes > 0 else "")
            + "..."
        )

        saved_episode_count = 0
        for ep in tqdm(range(total_episodes), desc="Episodes"):
            try:
                (
                    ep_ids,
                    ep_av_edge_acts,
                    ep_hdv_edge_acts,
                    ep_joint_edge_acts,
                    ep_rews,
                    ep_frames,
                    ep_departs,
                    ep_contexts,
                ) = self._collect_single_episode(episode_id=ep, max_steps=max_steps_per_episode)

                n_arrived = len(self._last_ep_travel_times)
                if n_arrived > 0:
                    avg_tt = np.mean(self._last_ep_travel_times)
                    tt_msg = f"arrived={n_arrived} | avg_tt={avg_tt:.1f}s ({avg_tt/60:.2f}min)"
                else:
                    tt_msg = "arrived=0 (no vehicles completed their route)"

                # Inter-episode HDV Gawron update using actual route travel times
                if self._hdv_route_probs is not None and self._route_registry is not None:
                    if self._last_ep_route_tts:
                        self._update_hdv_gawron_actual()
                    elif len(ep_frames) > 0:
                        # Fallback: use instantaneous edge speeds if no arrivals
                        self._update_hdv_gawron(ep_frames[-1])

                if ep < int(warmup_episodes):
                    tqdm.write(
                        f"  warmup {ep+1:3d}/{int(warmup_episodes):3d} | steps={len(ep_ids):3d} | {tt_msg}"
                    )
                    continue

                total_agents = int(getattr(self, "_num_agents", 0) or 0)
                episode_complete = total_agents > 0 and n_arrived >= total_agents
                if not episode_complete:
                    tqdm.write(
                        f"  skip incomplete attempt={ep-int(warmup_episodes):3d} | "
                        f"steps={len(ep_ids):3d} | arrived={n_arrived}/{total_agents}"
                    )
                    continue

                saved_ep_id = saved_episode_count
                all_episodes += [saved_ep_id] * len(ep_ids)
                all_av_edge_actions += ep_av_edge_acts
                all_hdv_edge_actions += ep_hdv_edge_acts
                all_joint_edge_actions += ep_joint_edge_acts
                all_rewards += ep_rews
                all_flows += ep_frames
                all_departs += ep_departs
                all_contexts += ep_contexts
                saved_episode_count += 1

                tqdm.write(
                    f"  collect {saved_ep_id:3d}/{int(num_episodes)-1:3d} | steps={len(ep_ids):3d} | {tt_msg}"
                )

                if saved_episode_count % 10 == 0:
                    self._save_intermediate(
                        task_name,
                        all_episodes,
                        all_av_edge_actions,
                        all_hdv_edge_actions,
                        all_joint_edge_actions,
                        all_rewards,
                        all_flows,
                        all_departs,
                        all_contexts,
                    )
            except Exception as e:
                print(f"  [episode {ep}] error: {e}")
                continue

        if not all_episodes:
            raise RuntimeError(
                "No complete episodes were collected; refusing to save an empty dataset."
            )
        print(f"Collection done — {len(all_flows)} flows across {len(set(all_episodes))} saved episodes")
        return self._save_final_data(
            task_name,
            all_episodes,
            all_av_edge_actions,
            all_hdv_edge_actions,
            all_joint_edge_actions,
            all_rewards,
            all_flows,
            all_departs,
            all_contexts,
        )

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
    ]:
        """
        Run a single episode and return collected data.

        Returns
            (
                ep_ids,
                ep_av_edge_acts,
                ep_hdv_edge_acts,
                ep_joint_edge_acts,
                ep_rewards,
                ep_frames,
            ).

        Frame index i is the observation at decision time i.  The matching
        av_edge_action is computed from the route decisions made from that
        same observation, before SUMO advances to the next decision boundary.
        """
        rng = np.random.RandomState((self.seed or 0) + episode_id)

        # Clear per-episode state
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
        self._last_ep_route_tts.clear()
        self._last_ep_safe_reroutes = 0
        self._last_ep_unsafe_reroute_keeps = 0
        self._last_ep_completion_drain_steps = 0

        # Vehicle departure times (SUMO simulation seconds)
        veh_depart: Dict[str, float] = {}
        veh_travel_times: Dict[str, float] = {}
        self._seed_depart_times_from_agents(veh_depart)

        ep_ids: List[int] = []
        ep_av_edge_acts: List[np.ndarray] = []
        ep_hdv_edge_acts: List[np.ndarray] = []
        ep_joint_edge_acts: List[np.ndarray] = []
        ep_rews: List[float] = []
        ep_frames: List[np.ndarray] = []
        ep_departs: List[np.ndarray] = []
        ep_contexts: List[np.ndarray] = []

        try:
            state = self.connector.reset(
                route_probs=self._hdv_route_probs,
                human_learning_state=self._loaded_hdv_state,
            )
        except TypeError:
            state = self.connector.reset(route_probs=self._hdv_route_probs)
        except Exception as e:
            print(f"  [episode {episode_id}] reset failed: {e}")
            return (
                ep_ids,
                ep_av_edge_acts,
                ep_hdv_edge_acts,
                ep_joint_edge_acts,
                ep_rews,
                ep_frames,
                ep_departs,
                ep_contexts,
            )

        # Register vehicles present after warm-up and assign roles
        self._register_vehicles(list(state.get("vehicles", {}).keys()), rng)
        self._record_depart_times(state, veh_depart)
        reset_arrivals = getattr(self.connector, "_last_reset_arrival_events", None)
        if reset_arrivals:
            self._record_arrivals(
                veh_depart,
                veh_travel_times,
                arrival_events=reset_arrivals,
            )

        decision_ticks = sorted(self._decision_tick_map(max_steps).keys())
        if not decision_ticks:
            decision_ticks = list(range(max_steps))

        current_tick = 0
        total_agents = int(getattr(self, "_num_agents", 0) or 0)
        for decision_idx, t in enumerate(decision_ticks):
            episode_complete = False
            steps_to_boundary = max(int(t) - int(current_tick), 0)
            if steps_to_boundary > 0:
                arrival_events = self.connector.advance_steps(steps_to_boundary)
                current_tick += steps_to_boundary
                self._record_arrivals(
                    veh_depart,
                    veh_travel_times,
                    arrival_events=arrival_events,
                )
                state = self.connector.get_traffic_state()
                if total_agents > 0 and len(self._last_ep_travel_times) >= total_agents:
                    episode_complete = True
                    break

            frame_t = self._collect_flow(state)

            # DAgger: encode frame with policy and compute guided av_frac
            self._step_av_policy_frac = None
            if self._dagger_active and rng.rand() < self._dagger_beta:
                frame_np = frame_t
                try:
                    with torch.no_grad():
                        z = self._dagger_encoder.encode_observation(frame_np)
                        z_flat = z.flatten().float().unsqueeze(0)   # (1, obs_dim)
                        av_frac, _, _, _, _ = self._dagger_policy.get_action_and_value(z_flat)
                        av_frac_np = av_frac.squeeze(0).cpu().numpy()   # (K,)
                    # Mix with Dirichlet noise for exploration
                    noise = rng.dirichlet([0.5] * self.n_routes)
                    av_frac_np = (1.0 - self._dagger_noise) * av_frac_np + self._dagger_noise * noise
                    av_frac_np = np.clip(av_frac_np, 0, None)
                    s = av_frac_np.sum()
                    if s > 0:
                        av_frac_np /= s
                    self._step_av_policy_frac = av_frac_np
                except Exception:
                    self._step_av_policy_frac = None

            self._register_vehicles(list(state["vehicles"].keys()), rng)
            self._record_depart_times(state, veh_depart)

            # AV re-routing: select route indices and apply best-effort rerouting.
            av_actions = self._select_av_actions(state, rng)
            self._record_av_decision_trace(episode_id, int(decision_idx), int(t))
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
            except Exception as e:
                print(f"  [episode {episode_id}, decision {decision_idx}] advance failed: {e}")
                break
            current_tick = int(t) + int(advance_steps)
            self._record_arrivals(
                veh_depart,
                veh_travel_times,
                arrival_events=arrival_events,
            )
            state = self.connector.get_traffic_state()
            if total_agents > 0 and len(self._last_ep_travel_times) >= total_agents:
                episode_complete = True

            reward = self.connector._calculate_reward(state)

            ep_ids.append(episode_id)
            ep_frames.append(frame_t)
            ep_av_edge_acts.append(av_edge_frac)
            ep_hdv_edge_acts.append(hdv_edge_frac)
            ep_joint_edge_acts.append(joint_edge_frac)
            ep_rews.append(float(reward))
            ep_departs.append(decision_depart)
            ep_contexts.append(decision_context)

            if episode_complete:
                break
            if self.connector.is_done() or (len(state.get("vehicles", {})) == 0 and current_tick > 10):
                break

        if total_agents <= 0 or len(self._last_ep_travel_times) < total_agents:
            state, current_tick = self._drain_episode_to_completion(
                state=state,
                current_tick=int(current_tick),
                veh_depart=veh_depart,
                veh_travel_times=veh_travel_times,
            )

        if self._use_routerl_hdv_blackbox and hasattr(self.connector, "apply_routerl_hdv_learning"):
            try:
                self.connector.apply_routerl_hdv_learning(veh_travel_times)
            except Exception as e:
                print(f"  [episode {episode_id}] RouteRL black-box HDV learning failed: {e}")

        if self._decision_steps <= 0:
            ep_frames.append(self._collect_flow(state))  # terminal frame

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
        )

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _build_demo_dict(
        self,
        episodes: List[int],
        av_edge_actions: List[np.ndarray],
        hdv_edge_actions: List[np.ndarray],
        joint_edge_actions: List[np.ndarray],
        rewards: List[float],
        departs: Optional[List[np.ndarray]] = None,
        contexts: Optional[List[np.ndarray]] = None,
        pair_ids: Optional[List[int]] = None,
        variant_ids: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        av_edge_action_tensor = torch.from_numpy(np.stack(av_edge_actions, axis=0)).float()
        hdv_edge_action_tensor = torch.from_numpy(np.stack(hdv_edge_actions, axis=0)).float()
        joint_edge_action_tensor = torch.from_numpy(np.stack(joint_edge_actions, axis=0)).float()
        demo = {
            "episode": torch.tensor(episodes, dtype=torch.int64),
            "av_edge_action": av_edge_action_tensor,
            "av_planned_edge_flow": av_edge_action_tensor,
            "hdv_edge_action": hdv_edge_action_tensor,
            "joint_edge_action": joint_edge_action_tensor,
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "action_schema_version": torch.tensor(2, dtype=torch.int64),
        }
        if departs:
            demo["depart"] = torch.from_numpy(np.stack(departs, axis=0)).float()
        if contexts:
            demo["context"] = torch.from_numpy(np.stack(contexts, axis=0)).float()
        if pair_ids is not None:
            demo["pair_id"] = torch.tensor(pair_ids, dtype=torch.int64)
        if variant_ids is not None:
            demo["variant_id"] = torch.tensor(variant_ids, dtype=torch.int64)
        return demo

    def _save_intermediate(
        self,
        task_name: str,
        episodes: List[int],
        av_edge_actions: List[np.ndarray],
        hdv_edge_actions: List[np.ndarray],
        joint_edge_actions: List[np.ndarray],
        rewards: List[float],
        obs_list: List[np.ndarray],
        departs: List[np.ndarray],
        contexts: List[np.ndarray],
        pair_ids: Optional[List[int]] = None,
        variant_ids: Optional[List[int]] = None,
    ) -> None:
        intermediate_task = f"{task_name}_intermediate"
        path = self.output_dir / f"{intermediate_task}.pt"
        torch.save(
            self._build_demo_dict(
                episodes,
                av_edge_actions,
                hdv_edge_actions,
                joint_edge_actions,
                rewards,
                departs,
                contexts,
                pair_ids,
                variant_ids,
            ),
            path,
        )
        task_flows_dir = self._flows_dir / intermediate_task
        task_flows_dir.mkdir(exist_ok=True)
        for old in task_flows_dir.glob("shard_*.pt"):
            old.unlink()
        num_shards = (len(obs_list) + self.shard_size - 1) // self.shard_size
        for si in range(num_shards):
            s, e = si * self.shard_size, min((si + 1) * self.shard_size, len(obs_list))
            arr = np.stack(obs_list[s:e], axis=0)
            torch.save(
                {"flows": torch.from_numpy(arr).float()},
                task_flows_dir / f"shard_{si:04d}.pt",
            )
        print(f"  [checkpoint] {len(episodes)} steps saved → {path}")
        print(f"  [checkpoint] {num_shards} flow shards saved → {task_flows_dir}")

    def _write_collection_decision_trace(self, task_name: str) -> None:
        rows = getattr(self, "_collection_decision_trace_rows", [])
        if not rows:
            return
        trace_path = self.output_dir / f"{task_name}_av_decision_trace.csv"
        fieldnames = [
            "episode",
            "decision_idx",
            "sim_step",
            "veh_id",
            "current_edge",
            "dest_edge",
            "selected_route",
            "candidate_count",
            "selected_contains_incident",
            "candidate_contains_incident",
            "reroute_safe",
            "selected_path",
        ]
        with trace_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        print(f"  AV decision trace: {trace_path} ({len(rows)} rows)")

    def _save_final_data(
        self,
        task_name: str,
        episodes: List[int],
        av_edge_actions: List[np.ndarray],
        hdv_edge_actions: List[np.ndarray],
        joint_edge_actions: List[np.ndarray],
        rewards: List[float],
        obs_list: List[np.ndarray],
        departs: List[np.ndarray],
        contexts: List[np.ndarray],
        pair_ids: Optional[List[int]] = None,
        variant_ids: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        print("Saving final dataset...")

        demo = self._build_demo_dict(
            episodes,
            av_edge_actions,
            hdv_edge_actions,
            joint_edge_actions,
            rewards,
            departs,
            contexts,
            pair_ids,
            variant_ids,
        )
        demo_path = self.output_dir / f"{task_name}.pt"
        torch.save(demo, demo_path)
        print(f"  Demo: {demo_path}")
        print(f"    av_edge_action    {tuple(demo['av_edge_action'].shape)}  (planned AV edge flow)")
        print(f"    av_planned_edge_flow {tuple(demo['av_planned_edge_flow'].shape)}  (planned AV edge flow)")
        print(f"    hdv_edge_action   {tuple(demo['hdv_edge_action'].shape)}  (HDV edge distribution)")
        print(f"    joint_edge_action {tuple(demo['joint_edge_action'].shape)}  (joint edge distribution)")

        # Flow shards
        task_flows_dir = self._flows_dir / task_name
        task_flows_dir.mkdir(exist_ok=True)
        num_shards = (len(obs_list) + self.shard_size - 1) // self.shard_size
        for si in tqdm(range(num_shards), desc="Flow shards"):
            s, e = si * self.shard_size, min((si + 1) * self.shard_size, len(obs_list))
            arr = np.stack(obs_list[s:e], axis=0)   # (N, E, d_feat)
            torch.save(
                {"flows": torch.from_numpy(arr).float()},
                task_flows_dir / f"shard_{si:04d}.pt",
            )
        print(f"  Flow shards: {num_shards} shards → {task_flows_dir}")
        self._write_collection_decision_trace(task_name)
        return demo


def main():
    parser = argparse.ArgumentParser(description="Collect urban routing data from SUMO")
    add_settings_arg(parser)

    # Common
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--warmup_episodes", type=int, default=0,
                        help="Run initial episodes for HDV adaptation only; do not save them to the dataset.")
    parser.add_argument("--max_steps",    type=int, default=1000)
    parser.add_argument("--task_name",    type=str, default="urban_routing")
    parser.add_argument("--img_size",     type=int, default=128)
    parser.add_argument("--shard_size",   type=int, default=2048)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--backend",      type=str, default="routerl",
                        choices=["routerl", "legacy"])

    # routerl backend
    parser.add_argument("--network_name",         type=str, default="ing_small")
    parser.add_argument("--network_folder",        type=str, default="")
    parser.add_argument("--origins",              type=str, default=None,
                        help="Comma-separated list of origin edge IDs (avoids argparse issue with IDs starting with '-')")
    parser.add_argument("--destinations",         type=str, default=None,
                        help="Comma-separated list of destination edge IDs")
    parser.add_argument("--number_of_paths",      type=int, default=4)
    parser.add_argument("--path_gen_beta",        type=float, default=-5.0)
    parser.add_argument("--num_samples",          type=int, default=10)
    parser.add_argument("--num_agents",           type=int, default=100)
    parser.add_argument("--simulation_timesteps", type=int, default=1800)
    parser.add_argument("--records_folder",       type=str, default="routerl_records")

    # Legacy backend
    parser.add_argument("--config", type=str, default="",
                        help="SUMO .sumocfg file — legacy backend only")

    # AV re-routing
    parser.add_argument("--av_fraction",        type=float, default=0.3,
                        help="Fraction of vehicles designated as AVs each episode")
    parser.add_argument("--random_av_fraction", type=float, default=0.5,
                        help="Fraction of AVs using purely random re-routing")
    parser.add_argument("--epsilon",            type=float, default=0.3,
                        help="Exploration rate for epsilon-greedy AVs")
    parser.add_argument("--decision_steps",     type=int, default=60,
                        help="Number of decision/recording steps per episode (SUMO still runs at native tick resolution).")
    parser.add_argument("--av_candidate_source", type=str, default="current_to_dest",
                        choices=["current_to_dest", "od_suffix"],
                        help="AV dynamic candidate source: current edge to destination search, or deprecated legacy OD route suffixes.")
    parser.add_argument("--safe_reroute_min_remaining_m", type=float, default=35.0,
                        help="Minimum lane distance remaining for reroutes that require a lane change. Set <0 to disable.")
    parser.add_argument("--completion_drain_steps", type=int, default=0,
                        help="Extra SUMO ticks after the recorded horizon used only to observe late arrivals.")
    parser.add_argument("--completion_drain_patience_steps", type=int, default=0,
                        help="Stop completion drain after this many consecutive ticks without new arrivals. <=0 disables.")
    parser.add_argument("--fixed_av_route_index", type=int, default=-1,
                        help="If >=0, force every AV decision to this route candidate index for counterfactual screening.")
    parser.add_argument("--fixed_av_route_probs", type=str, default="",
                        help="Comma-separated AV route probabilities for stochastic fixed-policy collection.")
    parser.add_argument("--sim_buffer_after_last_depart", type=int, default=1000,
                        help="Extra SUMO ticks to run after the last departure time.")
    parser.add_argument("--sumo_depart_lane", type=str, default="",
                        help="Optional traci.vehicle.add departLane value, e.g. best/free.")
    parser.add_argument("--sumo_depart_pos", type=str, default="",
                        help="Optional traci.vehicle.add departPos value, e.g. free/random_free.")
    parser.add_argument("--sumo_depart_speed", type=str, default="",
                        help="Optional traci.vehicle.add departSpeed value, e.g. max/desired.")
    parser.add_argument("--sumo_time_to_teleport", type=str, default="",
                        help="Override RouteRL's SUMO --time-to-teleport value. Empty preserves RouteRL default -1.")
    parser.add_argument("--sumo_eager_insert", action="store_true",
                        help="Pass SUMO --eager-insert for insertion diagnostics.")
    parser.add_argument("--sumo_sloppy_insert", action="store_true",
                        help="Pass SUMO --sloppy-insert for insertion diagnostics.")
    parser.add_argument("--sumo_emergency_insert", action="store_true",
                        help="Pass SUMO --emergency-insert for insertion diagnostics.")
    parser.add_argument("--sumo_extra_args", type=str, default="",
                        help="Extra SUMO CLI args parsed with shlex, e.g. '--ignore-route-errors'.")
    parser.add_argument("--incident_edge", type=str, default="",
                        help="Optional edge id whose lanes are slowed during [incident_start, incident_end).")
    parser.add_argument("--incident_start", type=float, default=0.0,
                        help="SUMO time in seconds when the optional incident starts.")
    parser.add_argument("--incident_end", type=float, default=0.0,
                        help="SUMO time in seconds when the optional incident ends.")
    parser.add_argument("--incident_speed", type=float, default=0.0,
                        help="Temporary incident lane max speed in m/s. <=0 disables the incident.")

    # agents_csv alignment
    parser.add_argument("--agents_csv",         type=str, default="",
                        help="Path to agents.csv — overrides num_agents + sim_timesteps")

    # DAgger policy guidance
    parser.add_argument("--policy_ckpt",        type=str, default="",
                        help="MARL policy checkpoint for DAgger guidance")
    parser.add_argument("--dagger_beta",        type=float, default=0.5,
                        help="Fraction of steps using policy guidance")
    parser.add_argument("--dagger_noise",       type=float, default=0.3,
                        help="Dirichlet noise weight on policy av_frac")
    parser.add_argument("--edge_registry", type=str, default="",
                        help="Path to edge_registry.json — enables flow collection mode")
    parser.add_argument("--hdv_gawron_lr", type=float, default=0.1,
                        help="Gawron learning rate for HDV inter-episode route update")
    parser.add_argument("--hdv_gawron_mode", type=str, default="replicator",
                        choices=["replicator", "routerl", "routerl_blackbox"],
                        help="HDV update mode: replicator/routerl (shared implementation) or routerl_blackbox (RouteRL internal per-agent learning)")
    parser.add_argument("--hdv_gawron_alpha_zero", type=float, default=0.2,
                        help="RouteRL-compatible alpha_zero for HDV Gawron (used when mode=routerl)")
    parser.add_argument("--hdv_gawron_beta", type=float, default=-1.5,
                        help="RouteRL-compatible beta for HDV route choice (used when mode=routerl)")
    parser.add_argument("--hdv_gawron_update_mode", type=str, default="expected",
                        choices=["expected", "sample"],
                        help="RouteRL-compatible update style: expected (fast) or sample (stochastic)")
    parser.add_argument("--hdv_min_prob", type=float, default=0.0,
                        help="Minimum probability floor per HDV route after Gawron update")
    parser.add_argument("--od_file", type=str, default="",
                        help="Path to OD file (JSON with origins/destinations) — enables HDV Gawron update")
    parser.add_argument("--route_registry", type=str, default="",
                        help="Path to route_registry.json (auto-built if absent)")

    settings_path = parse_settings_arg()
    apply_stage_defaults(parser, settings_path, stage="collect")
    args = parser.parse_args()
    fixed_av_route_probs = None
    if str(args.fixed_av_route_probs or "").strip():
        vals = np.asarray(
            [float(x.strip()) for x in str(args.fixed_av_route_probs).replace(";", ",").split(",") if x.strip()],
            dtype=np.float64,
        )
        if vals.size != int(args.number_of_paths):
            raise ValueError(
                f"--fixed_av_route_probs expected {int(args.number_of_paths)} values, got {vals.size}"
            )
        vals = np.clip(vals, 0.0, None)
        total = float(vals.sum())
        if total <= 0.0:
            raise ValueError("--fixed_av_route_probs must have positive sum")
        fixed_av_route_probs = vals / total

    collector = DataCollector(
        output_dir=args.output_dir,
        img_size=args.img_size,
        shard_size=args.shard_size,
        seed=args.seed,
        backend=args.backend,
        network_name=args.network_name,
        network_folder=args.network_folder,
        origins=[o.strip() for o in args.origins.split(",")] if args.origins else None,
        destinations=[d.strip() for d in args.destinations.split(",")] if args.destinations else None,
        number_of_paths=args.number_of_paths,
        path_gen_beta=args.path_gen_beta,
        num_samples=args.num_samples,
        num_agents=args.num_agents,
        simulation_timesteps=args.simulation_timesteps,
        records_folder=args.records_folder,
        config_file=args.config,
        av_fraction=args.av_fraction,
        random_av_fraction=args.random_av_fraction,
        epsilon=args.epsilon,
        decision_steps=args.decision_steps,
        av_candidate_source=args.av_candidate_source,
        safe_reroute_min_remaining_m=args.safe_reroute_min_remaining_m,
        completion_drain_steps=args.completion_drain_steps,
        completion_drain_patience_steps=args.completion_drain_patience_steps,
        fixed_av_route_index=args.fixed_av_route_index,
        fixed_av_route_probs=fixed_av_route_probs,
        sim_buffer_after_last_depart=args.sim_buffer_after_last_depart,
        sumo_depart_lane=args.sumo_depart_lane,
        sumo_depart_pos=args.sumo_depart_pos,
        sumo_depart_speed=args.sumo_depart_speed,
        sumo_time_to_teleport=args.sumo_time_to_teleport,
        sumo_eager_insert=args.sumo_eager_insert,
        sumo_sloppy_insert=args.sumo_sloppy_insert,
        sumo_emergency_insert=args.sumo_emergency_insert,
        sumo_extra_args=args.sumo_extra_args,
        incident_edge=args.incident_edge,
        incident_start=args.incident_start,
        incident_end=args.incident_end,
        incident_speed=args.incident_speed,
        agents_csv=args.agents_csv,
        policy_ckpt=args.policy_ckpt,
        dagger_beta=args.dagger_beta,
        dagger_noise=args.dagger_noise,
        edge_registry_path=args.edge_registry,
        hdv_gawron_lr=args.hdv_gawron_lr,
        hdv_gawron_mode=args.hdv_gawron_mode,
        hdv_gawron_alpha_zero=args.hdv_gawron_alpha_zero,
        hdv_gawron_beta=args.hdv_gawron_beta,
        hdv_gawron_update_mode=args.hdv_gawron_update_mode,
        hdv_min_prob=args.hdv_min_prob,
        od_file=args.od_file,
        route_registry_path=args.route_registry,
    )

    collector.collect_episodes(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        task_name=args.task_name,
        warmup_episodes=args.warmup_episodes,
    )
    print("Data collection complete.")


if __name__ == "__main__":
    main()
