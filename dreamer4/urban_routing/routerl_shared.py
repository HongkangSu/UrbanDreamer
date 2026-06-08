"""
Shared RouteRL/SUMO helpers used by data collection, training-time eval and scripts.

This module intentionally centralises:
1) Route generation (Dijkstra / RouteRL) for routes.csv + route.rou.xml
2) RouteRL TrafficEnvironment argument construction
3) OD / agents loading helpers
"""

import ast
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .flow_projector import RouteRegistry, build_edge_registry
except ImportError:
    from flow_projector import RouteRegistry, build_edge_registry


def load_od_data(network_folder: str, network_name: str) -> Dict[str, Any]:
    """Load `od_<network_name>.txt` and return its parsed dict."""
    od_file = os.path.join(network_folder, f"od_{network_name}.txt")
    with open(od_file, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


def load_agents_dataframe(network_folder: str, agents_csv: Optional[str] = None) -> pd.DataFrame:
    """Load an agents CSV, defaulting to `<network_folder>/agents.csv`."""
    path = agents_csv or os.path.join(network_folder, "agents.csv")
    return pd.read_csv(path)


def compute_agent_counts(agents_df: pd.DataFrame, ratio_machines: float) -> Tuple[int, int, int]:
    """
    Return (num_agents, num_machines, max_start_time).
    """
    num_agents = int(len(agents_df))
    num_machines = int(num_agents * float(ratio_machines))
    max_start_time = int(agents_df["start_time"].max()) if num_agents > 0 else 0
    return num_agents, num_machines, max_start_time


def imagined_gawron_mode(mode: str) -> str:
    """Return the differentiable imagination surrogate for a RouteRL mode."""
    mode_key = str(mode or "replicator").strip().lower()
    if mode_key == "routerl_blackbox":
        return "routerl"
    if mode_key in {"replicator", "routerl"}:
        return mode_key
    raise ValueError(
        f"Unsupported gawron_mode={mode!r}; expected replicator, routerl, or routerl_blackbox"
    )


def ensure_agents_csv_in_records(
    network_folder: str,
    records_folder: str,
    overwrite: bool = False,
    agents_csv: Optional[str] = None,
) -> str:
    """
    Ensure RouteRL-visible `agents.csv` exists under `records_folder`.
    Returns destination path.
    """
    os.makedirs(records_folder, exist_ok=True)
    src = agents_csv or os.path.join(network_folder, "agents.csv")
    dst = os.path.join(records_folder, "agents.csv")
    if overwrite or (not os.path.exists(dst)):
        shutil.copy2(src, dst)
    return dst


def _load_demand_od_pairs(agents_csv: Optional[str]) -> list[tuple[int, int]]:
    if not agents_csv or not os.path.exists(agents_csv):
        return []
    try:
        agents_df = pd.read_csv(agents_csv)
        if {"origin", "destination"}.issubset(agents_df.columns):
            return [
                (int(o), int(d))
                for o, d in agents_df[["origin", "destination"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            ]
    except Exception:
        return []
    return []


def _routes_meta_path(records_folder: str) -> str:
    return os.path.join(records_folder, "routes_meta.json")


def _cached_routes_match(
    records_folder: str,
    *,
    generator: str,
    expected_rows: int,
    expected_od_pairs: Optional[Sequence[tuple[int, int]]] = None,
) -> Optional[pd.DataFrame]:
    routes_csv = os.path.join(records_folder, "routes.csv")
    routes_xml = os.path.join(records_folder, "route.rou.xml")
    meta_path = _routes_meta_path(records_folder)
    if not (os.path.exists(routes_csv) and os.path.exists(routes_xml) and os.path.exists(meta_path)):
        return None
    try:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        if str(meta.get("generator", "")).strip().lower() != str(generator).strip().lower():
            return None
        if expected_od_pairs is not None:
            meta_od_pairs = meta.get("od_pairs")
            expected = [[int(oi), int(di)] for oi, di in expected_od_pairs]
            if meta_od_pairs != expected:
                return None
        cached_df = pd.read_csv(routes_csv)
        if len(cached_df) != int(expected_rows):
            return None
        return cached_df
    except Exception:
        return None


def _write_routes_meta(
    records_folder: str,
    *,
    generator: str,
    network_name: str,
    n_routes: int,
    n_origins: int,
    n_destinations: int,
    od_pairs: Optional[Sequence[tuple[int, int]]] = None,
) -> None:
    meta = {
        "generator": str(generator),
        "network_name": str(network_name),
        "n_routes": int(n_routes),
        "n_origins": int(n_origins),
        "n_destinations": int(n_destinations),
    }
    if od_pairs is not None:
        meta["od_pairs"] = [[int(oi), int(di)] for oi, di in od_pairs]
    Path(_routes_meta_path(records_folder)).write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def generate_dijkstra_routes(
    network_folder: str,
    network_name: str,
    origins: Sequence[str],
    destinations: Sequence[str],
    n_routes: int,
    records_folder: str,
    penalty_factor: float = 3.0,
    agents_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate penalty-based Dijkstra routes and save:
      - routes.csv
      - route.rou.xml
    Returns the generated routes dataframe.
    """
    os.makedirs(records_folder, exist_ok=True)
    routes_csv = os.path.join(records_folder, "routes.csv")
    routes_xml = os.path.join(records_folder, "route.rou.xml")
    demand_agents_csv = agents_csv or os.path.join(network_folder, "agents.csv")
    od_pairs = _load_demand_od_pairs(demand_agents_csv)
    if not od_pairs:
        od_pairs = [(oi, di) for oi in range(len(origins)) for di in range(len(destinations))]
    expected_rows = int(len(od_pairs)) * int(n_routes)

    er_path = os.path.join(network_folder, "edge_registry.json")
    edge_registry = build_edge_registry(network_folder, network_name)
    cached_df = _cached_routes_match(
        records_folder,
        generator="dijkstra_passenger",
        expected_rows=expected_rows,
        expected_od_pairs=od_pairs,
    )
    if cached_df is not None:
        return cached_df

    adj_list = RouteRegistry._build_vehicle_adj_list(
        edge_registry.adjacency,
        getattr(edge_registry, "passenger_allowed", None),
    )
    lengths = edge_registry.lengths
    speed_limits = edge_registry.speed_limits
    base_weights = lengths / np.maximum(speed_limits, 0.1)
    edge_to_idx = edge_registry._edge_to_idx
    edge_ids = edge_registry.edge_ids

    rows = []
    for oi, di in od_pairs:
        if oi >= len(origins) or di >= len(destinations):
            continue
        origin = origins[oi]
        dest = destinations[di]
        if origin not in edge_to_idx or dest not in edge_to_idx:
            continue
        src_idx, dst_idx = edge_to_idx[origin], edge_to_idx[dest]
        weights = base_weights.copy()
        routes_for_od = []
        seen_sets = []
        for _ in range(int(n_routes)):
            path = RouteRegistry._dijkstra(adj_list, weights, src_idx, dst_idx)
            if path is None:
                break
            path_set = frozenset(path)
            if path_set in seen_sets:
                for idx in path:
                    weights[idx] *= penalty_factor
                path = RouteRegistry._dijkstra(adj_list, weights, src_idx, dst_idx)
                if path is None:
                    break
                path_set = frozenset(path)
            routes_for_od.append(path)
            seen_sets.append(path_set)
            for idx in path:
                weights[idx] *= penalty_factor

        while len(routes_for_od) < int(n_routes) and routes_for_od:
            routes_for_od.append(routes_for_od[0])

        for route_indices in routes_for_od:
            route_edge_ids = [edge_ids[i] for i in route_indices]
            # RouteRL human reward uses minutes, so keep free-flow priors
            # on the same scale to avoid a large unit mismatch.
            fft = float(sum(base_weights[i] for i in route_indices) / 60.0)
            rows.append(
                {
                    "origins": oi,
                    "destinations": di,
                    "path": " ".join(route_edge_ids),
                    "free_flow_time": fft,
                }
            )

    df = pd.DataFrame(rows, columns=["origins", "destinations", "path", "free_flow_time"])
    if not df.empty:
        df.sort_values(by=["origins", "destinations"], inplace=True)

    df.to_csv(routes_csv, index=False)

    with open(routes_xml, "w", encoding="utf-8") as f:
        f.write("<routes>\n")
        f.write('<vType id="Human" color="red" guiShape="passenger/sedan"/>\n')
        f.write('<vType id="AV" color="yellow"/>\n')
        for oi, di in od_pairs:
            paths = df[(df["origins"] == oi) & (df["destinations"] == di)]["path"].values
            for idx, path in enumerate(paths):
                f.write(f'<route id="{oi}_{di}_{idx}" edges="\n')
                f.write(path + "\n")
                f.write('" />\n')
        f.write("</routes>\n")
    _write_routes_meta(
        records_folder,
        generator="dijkstra_passenger",
        network_name=network_name,
        n_routes=n_routes,
        n_origins=len(origins),
        n_destinations=len(destinations),
        od_pairs=od_pairs,
    )
    return df


def generate_routerl_routes(
    network_folder: str,
    network_name: str,
    origins: Sequence[str],
    destinations: Sequence[str],
    n_routes: int,
    records_folder: str,
    simulation_timesteps: int,
    num_agents: Optional[int] = None,
    beta: float = -5.0,
    num_samples: int = 10,
    agents_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate routes.csv + route.rou.xml via RouteRL/janux path generation.

    If a valid cached routes.csv/route.rou.xml pair already exists under
    ``records_folder``, reuse it.
    """
    os.makedirs(records_folder, exist_ok=True)
    routes_csv = os.path.join(records_folder, "routes.csv")
    routes_xml = os.path.join(records_folder, "route.rou.xml")
    records_agents_csv = os.path.join(records_folder, "agents.csv")
    expected_rows = int(len(origins)) * int(len(destinations)) * int(n_routes)
    demand_od_pairs = _load_demand_od_pairs(agents_csv or records_agents_csv)
    if demand_od_pairs:
        expected_rows = int(len(demand_od_pairs)) * int(n_routes)
    cached_df = _cached_routes_match(
        records_folder,
        generator="routerl",
        expected_rows=expected_rows,
        expected_od_pairs=demand_od_pairs or None,
    )
    if cached_df is not None:
        return cached_df

    try:
        from routerl import TrafficEnvironment
    except ImportError as exc:
        raise RuntimeError("routerl is not installed; cannot generate RouteRL routes") from exc

    create_agents = not os.path.exists(records_agents_csv)
    env = create_routerl_environment(
        TrafficEnvironment,
        seed=42,
        create_agents=create_agents,
        create_paths=True,
        network_name=network_name,
        network_folder=network_folder,
        simulation_timesteps=int(simulation_timesteps),
        records_folder=records_folder,
        plots_folder=os.path.join(records_folder, "plots"),
        origins=list(origins),
        destinations=list(destinations),
        number_of_paths=int(n_routes),
        num_agents=int(num_agents) if (create_agents and num_agents is not None) else None,
        new_machines_after_mutation=0,
        machine_behavior="selfish",
        machine_observation_type=None,
        save_every=1,
        phases=[0, 1],
        phase_names=["Route generation"],
        smooth_by=5,
        plot_choices="none",
        sumo_type="sumo",
        path_beta=float(beta),
        path_num_samples=int(num_samples),
        visualize_paths=False,
        save_detectors_info=False,
    )
    try:
        df = pd.read_csv(routes_csv)
    finally:
        try:
            env.close()
        except Exception:
            try:
                env.simulator.stop()
            except Exception:
                pass
    if not os.path.exists(routes_xml):
        raise FileNotFoundError(f"RouteRL did not produce route.rou.xml at {routes_xml}")
    _write_routes_meta(
        records_folder,
        generator="routerl",
        network_name=network_name,
        n_routes=n_routes,
        n_origins=len(origins),
        n_destinations=len(destinations),
        od_pairs=demand_od_pairs or None,
    )
    return df


def create_routerl_environment(
    traffic_environment_cls: Any,
    *,
    seed: int,
    create_agents: bool,
    create_paths: bool,
    network_name: str,
    network_folder: str,
    simulation_timesteps: int,
    records_folder: str,
    plots_folder: str,
    origins: Any,
    destinations: Any,
    number_of_paths: int,
    num_agents: Optional[int] = None,
    new_machines_after_mutation: int = 0,
    machine_behavior: str = "selfish",
    machine_observation_type: Optional[str] = "previous_agents_plus_start_time",
    human_model: str = "gawron",
    save_every: int = 1,
    phases: Optional[Sequence[int]] = None,
    phase_names: Optional[Sequence[str]] = None,
    smooth_by: int = 5,
    plot_choices: str = "none",
    sumo_type: str = "sumo",
    path_beta: Optional[float] = None,
    path_num_samples: Optional[int] = None,
    visualize_paths: bool = False,
    save_detectors_info: bool = False,
) -> Any:
    """Create a RouteRL TrafficEnvironment with unified parameter schema."""
    if create_agents and num_agents is None:
        raise ValueError("num_agents must be provided when create_agents=True")

    machine_parameters = {"behavior": machine_behavior}
    if machine_observation_type:
        machine_parameters["observation_type"] = machine_observation_type

    agent_parameters: Dict[str, Any] = {
        "new_machines_after_mutation": int(new_machines_after_mutation),
        "human_parameters": {"model": human_model},
        "machine_parameters": machine_parameters,
    }
    if create_agents:
        agent_parameters["num_agents"] = int(num_agents)

    path_generation_parameters: Dict[str, Any] = {
        "origins": origins,
        "destinations": destinations,
        "number_of_paths": int(number_of_paths),
    }
    if path_beta is not None:
        path_generation_parameters["beta"] = float(path_beta)
    if path_num_samples is not None:
        path_generation_parameters["num_samples"] = int(path_num_samples)
    if path_beta is not None or path_num_samples is not None:
        path_generation_parameters["visualize_paths"] = bool(visualize_paths)

    if phases is None:
        phases = [0, 1]
    if phase_names is None:
        phase_names = ["Run"]

    return traffic_environment_cls(
        seed=seed,
        create_agents=create_agents,
        create_paths=create_paths,
        save_detectors_info=save_detectors_info,
        agent_parameters=agent_parameters,
        environment_parameters={"save_every": int(save_every)},
        simulator_parameters={
            "network_name": network_name,
            "custom_network_folder": network_folder,
            "sumo_type": sumo_type,
            "simulation_timesteps": int(simulation_timesteps),
        },
        plotter_parameters={
            "records_folder": records_folder,
            "plots_folder": plots_folder,
            "plot_choices": plot_choices,
            "phases": list(phases),
            "phase_names": list(phase_names),
            "smooth_by": int(smooth_by),
        },
        path_generation_parameters=path_generation_parameters,
    )
