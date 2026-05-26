"""
Departure schedule utilities for time-stepped WM architecture.

Provides:
  build_departure_schedule  — pre-compute per-step departure info from agents.csv
  rebuild_active_set        — reconstruct active vehicle set at arbitrary step
  estimate_vehicle_position — analytical position estimate for a vehicle
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def build_departure_schedule(
    agents_csv: str,
    step_interval: float,
    origins: List[str],
    destinations: List[str],
    edge_to_idx: Dict[str, int],
    n_edges: int,
    ratio_machines: float = 0.6,
    max_av_count: Optional[int] = None,
    max_hdv_count: Optional[int] = None,
) -> Dict[int, Dict]:
    """Build per-step departure schedule from agents.csv.

    Args:
        agents_csv: path to agents.csv (columns: id, start_time, origin, destination, kind)
        step_interval: Δt in seconds per decision step
        origins: list of origin edge names (from route_registry)
        destinations: list of destination edge names
        edge_to_idx: edge name → index mapping
        n_edges: total number of edges
        ratio_machines: fraction of agents that are AV (Machine)
        max_av_count: optional cap on AV departures included in the schedule
        max_hdv_count: optional cap on HDV departures included in the schedule

    Returns:
        {step_idx: {
            "depart_edge": (E,) float32  — new departures per edge this step
            "av_vehicles": [{"id", "origin_idx", "dest_idx", "od_idx", "origin_edge", "dest_edge"}, ...]
            "hdv_vehicles": [{"id", "origin_idx", "dest_idx", "od_idx", "origin_edge", "dest_edge"}, ...]
            "n_av": int
            "n_hdv": int
        }}
    """
    df = pd.read_csv(agents_csv)
    n_dest = len(destinations)

    # Assign AV/HDV based on ratio_machines
    # RouteRL uses the first ratio_machines fraction as machines
    n_total = len(df)
    n_machines = int(n_total * ratio_machines)
    av_cap = None if max_av_count is None else max(0, int(max_av_count))
    hdv_cap = None if max_hdv_count is None else max(0, int(max_hdv_count))
    av_seen = 0
    hdv_seen = 0

    schedule: Dict[int, Dict] = {}

    for _, row in df.iterrows():
        agent_id = int(row["id"])
        start_time = float(row["start_time"])
        oi = int(row["origin"])  # index into origins list
        di = int(row["destination"])  # index into destinations list
        step = int(start_time / step_interval)

        # Determine AV or HDV
        is_av = agent_id < n_machines
        if is_av:
            if av_cap is not None and av_seen >= av_cap:
                continue
            av_seen += 1
        else:
            if hdv_cap is not None and hdv_seen >= hdv_cap:
                continue
            hdv_seen += 1

        # Map to edge indices
        origin_name = origins[oi] if oi < len(origins) else origins[0]
        dest_name = destinations[di] if di < len(destinations) else destinations[0]
        origin_edge = edge_to_idx.get(origin_name, 0)
        dest_edge = edge_to_idx.get(dest_name, 0)
        od_idx = oi * n_dest + di

        if step not in schedule:
            schedule[step] = {
                "depart_edge": np.zeros(n_edges, dtype=np.float32),
                "av_vehicles": [],
                "hdv_vehicles": [],
                "n_av": 0,
                "n_hdv": 0,
            }

        veh_info = {
            "id": agent_id,
            "origin_idx": oi,
            "dest_idx": di,
            "od_idx": od_idx,
            "origin_edge": origin_edge,
            "dest_edge": dest_edge,
        }

        schedule[step]["depart_edge"][origin_edge] += 1.0
        if is_av:
            schedule[step]["av_vehicles"].append(veh_info)
            schedule[step]["n_av"] += 1
        else:
            schedule[step]["hdv_vehicles"].append(veh_info)
            schedule[step]["n_hdv"] += 1

    # Normalize depart_edge per step
    for step_data in schedule.values():
        s = step_data["depart_edge"].sum()
        if s > 0:
            step_data["depart_edge"] = step_data["depart_edge"] / s

    # Fill missing steps with empty entries
    max_step = max(schedule.keys()) if schedule else 0
    empty = {
        "depart_edge": np.zeros(n_edges, dtype=np.float32),
        "av_vehicles": [],
        "hdv_vehicles": [],
        "n_av": 0,
        "n_hdv": 0,
    }
    for step in range(max_step + 1):
        if step not in schedule:
            schedule[step] = {k: (v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v)
                              for k, v in empty.items()}

    return schedule


def rebuild_active_set(
    departure_schedule: Dict[int, Dict],
    current_step: int,
    avg_route_steps: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Reconstruct the active vehicle sets at a given step.

    Replays the departure schedule from step 0 to current_step-1,
    removing vehicles that have likely arrived (departed > avg_route_steps ago).

    Returns:
        (active_avs, active_hdvs) — lists of vehicle info dicts with added "depart_step" key
    """
    active_avs: List[Dict] = []
    active_hdvs: List[Dict] = []

    for step in range(current_step):
        dep = departure_schedule.get(step)
        if dep is None:
            continue

        for veh in dep["av_vehicles"]:
            active_avs.append({**veh, "depart_step": step})
        for veh in dep["hdv_vehicles"]:
            active_hdvs.append({**veh, "depart_step": step})

    # Remove vehicles that have likely arrived
    cutoff = max(0, current_step - int(avg_route_steps * 1.5))
    active_avs = [av for av in active_avs if av["depart_step"] >= cutoff]
    active_hdvs = [hdv for hdv in active_hdvs if hdv["depart_step"] >= cutoff]

    return active_avs, active_hdvs


def estimate_vehicle_position(
    vehicle: Dict,
    current_step: int,
    avg_edge_time: float,
    step_interval: float,
    route: List[int],
) -> int:
    """Estimate which edge a vehicle is currently on.

    Uses departure time + average edge traversal time to estimate progress along route.

    Args:
        vehicle: dict with "depart_step" key
        current_step: current simulation step
        avg_edge_time: average time to traverse one edge (seconds)
        step_interval: Δt seconds per step
        route: list of edge indices from origin to destination

    Returns:
        Estimated current edge index
    """
    if not route:
        return vehicle.get("origin_edge", 0)

    time_elapsed = (current_step - vehicle["depart_step"]) * step_interval
    edges_traversed = int(time_elapsed / max(avg_edge_time, 0.1))
    idx = min(edges_traversed, len(route) - 1)
    return route[idx]


def compute_avg_edge_time(lengths: np.ndarray, speed_limits: np.ndarray) -> float:
    """Compute average free-flow edge traversal time in seconds."""
    valid = speed_limits > 0.1
    if not valid.any():
        return 10.0
    return float(np.mean(lengths[valid] / speed_limits[valid]))


def compute_avg_route_steps(
    avg_route_edges: float,
    avg_edge_time: float,
    step_interval: float,
) -> int:
    """Estimate how many decision steps it takes to complete an average route."""
    if isinstance(avg_route_edges, np.ndarray):
        route_masks = np.asarray(avg_route_edges, dtype=np.float32)
        if route_masks.ndim >= 2:
            avg_route_edges = float((route_masks > 0).sum(axis=-1).mean())
        else:
            avg_route_edges = float((route_masks > 0).sum())
    avg_route_time = float(avg_route_edges) * avg_edge_time
    return max(1, int(avg_route_time / step_interval))
