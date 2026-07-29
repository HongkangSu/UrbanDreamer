from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..traffic.environment import ActionDecision


ROUTE_COUNT = 4
EDGE_POLICY_FEATURES = 2
ROUTE_COST_FEATURES = 3
FLEET_PRESSURE_FEATURES = 2
ACTOR_OBSERVATION_DIM = 33
_COST_LOG_NORMALIZER = float(np.log1p(3600.0))


def build_actor_observations(
    *,
    policy_edge_features: np.ndarray,
    candidate_paths: Sequence[Sequence[Sequence[int]]],
    edge_travel_seconds: np.ndarray,
    assigned_route_counts: np.ndarray,
    assigned_edge_pressure: np.ndarray,
    episode_progress: float,
    active_av_fraction: float,
) -> np.ndarray:
    """Build the sole K=4 route-shared actor observation layout."""
    edge_features = np.asarray(policy_edge_features, dtype=np.float32)
    if edge_features.ndim != 2 or edge_features.shape[1] != EDGE_POLICY_FEATURES:
        raise ValueError(
            "policy_edge_features must have shape (E, 2), "
            f"got {edge_features.shape}"
        )
    vehicle_count = len(candidate_paths)
    if vehicle_count == 0:
        return np.zeros((0, ACTOR_OBSERVATION_DIM), dtype=np.float32)
    if any(len(paths) != ROUTE_COUNT for paths in candidate_paths):
        raise ValueError("every vehicle must have exactly four candidate paths")

    route_features = np.zeros(
        (vehicle_count, ROUTE_COUNT * EDGE_POLICY_FEATURES),
        dtype=np.float32,
    )
    for vehicle_index, paths in enumerate(candidate_paths):
        for route_index, path in enumerate(paths):
            edge_indices = _valid_edge_indices(path, edge_features.shape[0])
            if edge_indices.size:
                start = route_index * EDGE_POLICY_FEATURES
                route_features[
                    vehicle_index,
                    start : start + EDGE_POLICY_FEATURES,
                ] = edge_features[edge_indices].mean(axis=0)

    global_features = np.repeat(
        edge_features.mean(axis=0, keepdims=True),
        vehicle_count,
        axis=0,
    )
    route_cost_features = _route_cost_features(
        candidate_paths=candidate_paths,
        edge_travel_seconds=edge_travel_seconds,
        edge_count=edge_features.shape[0],
    )
    fleet_pressure_features = _fleet_pressure_features(
        candidate_paths=candidate_paths,
        edge_count=edge_features.shape[0],
        assigned_route_counts=assigned_route_counts,
        assigned_edge_pressure=assigned_edge_pressure,
    )
    context = np.repeat(
        np.asarray(
            [[float(episode_progress), float(active_av_fraction), 0.0]],
            dtype=np.float32,
        ),
        vehicle_count,
        axis=0,
    )
    observations = np.concatenate(
        (
            route_features,
            global_features,
            route_cost_features,
            fleet_pressure_features,
            context,
        ),
        axis=1,
    )
    if observations.shape != (vehicle_count, ACTOR_OBSERVATION_DIM):
        raise AssertionError(
            "actor observation layout drifted: "
            f"expected {(vehicle_count, ACTOR_OBSERVATION_DIM)}, "
            f"got {observations.shape}"
        )
    return observations.astype(np.float32, copy=False)


def aggregate_planned_edge_flow(
    decisions: Sequence[ActionDecision],
    *,
    edge_count: int,
) -> np.ndarray:
    """Mean selected future edge occupancy over controlled vehicles."""
    if edge_count <= 0:
        raise ValueError("edge_count must be positive")
    if not decisions:
        return np.zeros((edge_count,), dtype=np.float32)
    planned_flows = np.stack(
        [
            np.asarray(decision.planned_edge_flow, dtype=np.float32)
            for decision in decisions
        ],
        axis=0,
    )
    if planned_flows.shape[1:] != (edge_count,):
        raise ValueError(
            "planned edge flow width does not match the topology edge count"
        )
    return planned_flows.mean(axis=0, dtype=np.float32)


def advance_vehicle_on_path(
    vehicle_state: dict[str, Any],
    *,
    selected_path: Sequence[int],
    edge_travel_seconds: np.ndarray,
    interval_seconds: float,
    edge_count: int,
) -> None:
    """Advance one imagined vehicle through its selected path."""
    travel_seconds = np.asarray(edge_travel_seconds, dtype=np.float32).reshape(-1)
    if travel_seconds.shape != (edge_count,):
        raise ValueError("edge_travel_seconds must have one value per edge")
    path = _valid_edge_indices(selected_path, edge_count).tolist()
    if not path:
        raise ValueError("selected_path must contain a topology edge")

    current_edge = int(vehicle_state["current_edge"])
    try:
        path_position = path.index(current_edge)
    except ValueError:
        path_position = 0
    remaining_on_edge = float(
        vehicle_state.get(
            "remaining_edge_seconds",
            travel_seconds[path[path_position]],
        )
    )
    remaining_interval = max(float(interval_seconds), 0.0)
    while (
        remaining_interval >= remaining_on_edge
        and path_position < len(path) - 1
    ):
        remaining_interval -= remaining_on_edge
        path_position += 1
        remaining_on_edge = float(travel_seconds[path[path_position]])

    arrived = (
        path_position == len(path) - 1
        and remaining_interval >= remaining_on_edge
    )
    vehicle_state["current_edge"] = int(path[path_position])
    vehicle_state["remaining_edge_seconds"] = (
        0.0
        if arrived
        else max(remaining_on_edge - remaining_interval, 0.0)
    )
    vehicle_state["arrived"] = bool(arrived)


def _route_cost_features(
    *,
    candidate_paths: Sequence[Sequence[Sequence[int]]],
    edge_travel_seconds: np.ndarray,
    edge_count: int,
) -> np.ndarray:
    travel_seconds = np.asarray(edge_travel_seconds, dtype=np.float32).reshape(-1)
    if travel_seconds.shape != (edge_count,):
        raise ValueError("edge_travel_seconds must have one value per edge")
    vehicle_count = len(candidate_paths)
    route_costs = np.full(
        (vehicle_count, ROUTE_COUNT),
        np.nan,
        dtype=np.float32,
    )
    valid_routes = np.zeros((vehicle_count, ROUTE_COUNT), dtype=bool)
    for vehicle_index, paths in enumerate(candidate_paths):
        for route_index, path in enumerate(paths):
            edge_indices = _valid_edge_indices(path, edge_count)
            if not edge_indices.size:
                continue
            values = travel_seconds[edge_indices]
            if np.isfinite(values).all():
                route_costs[vehicle_index, route_index] = float(
                    np.maximum(values, 0.0).sum()
                )
                valid_routes[vehicle_index, route_index] = True

    output = np.zeros(
        (vehicle_count, ROUTE_COUNT * ROUTE_COST_FEATURES),
        dtype=np.float32,
    )
    for vehicle_index in range(vehicle_count):
        valid = valid_routes[vehicle_index]
        if not valid.any():
            continue
        costs = route_costs[vehicle_index].copy()
        finite_costs = costs[valid]
        minimum_cost = float(finite_costs.min())
        scale = max(float(np.median(finite_costs)), 60.0)
        costs[~valid] = float(finite_costs.max() + scale)
        relative_gap = np.clip(
            (costs - minimum_cost) / scale,
            0.0,
            20.0,
        )
        log_cost = np.log1p(np.maximum(costs, 0.0)) / _COST_LOG_NORMALIZER
        output[vehicle_index] = np.concatenate(
            (
                relative_gap,
                log_cost,
                valid.astype(np.float32),
            )
        )
    return output


def _fleet_pressure_features(
    *,
    candidate_paths: Sequence[Sequence[Sequence[int]]],
    edge_count: int,
    assigned_route_counts: np.ndarray,
    assigned_edge_pressure: np.ndarray,
) -> np.ndarray:
    route_counts = np.asarray(assigned_route_counts, dtype=np.float32).reshape(-1)
    edge_pressure = np.asarray(assigned_edge_pressure, dtype=np.float32).reshape(-1)
    if route_counts.shape != (ROUTE_COUNT,):
        raise ValueError("assigned_route_counts must have shape (4,)")
    if edge_pressure.shape != (edge_count,):
        raise ValueError(
            "assigned_edge_pressure must have one value per topology edge"
        )
    denominator = max(float(np.maximum(route_counts, 0.0).sum()), 1.0)
    route_share = np.clip(route_counts / denominator, 0.0, 1.0)
    output = np.zeros(
        (len(candidate_paths), ROUTE_COUNT * FLEET_PRESSURE_FEATURES),
        dtype=np.float32,
    )
    output[:, :ROUTE_COUNT] = route_share
    for vehicle_index, paths in enumerate(candidate_paths):
        for route_index, path in enumerate(paths):
            edge_indices = np.unique(_valid_edge_indices(path, edge_count))
            if edge_indices.size:
                output[vehicle_index, ROUTE_COUNT + route_index] = float(
                    edge_pressure[edge_indices].sum() / denominator
                )
    return np.nan_to_num(
        output,
        nan=0.0,
        posinf=20.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)


def _valid_edge_indices(path: Sequence[int], edge_count: int) -> np.ndarray:
    indices = np.asarray(tuple(path), dtype=np.int64).reshape(-1)
    return indices[(indices >= 0) & (indices < edge_count)]


__all__ = [
    "ACTOR_OBSERVATION_DIM",
    "ROUTE_COUNT",
    "advance_vehicle_on_path",
    "aggregate_planned_edge_flow",
    "build_actor_observations",
]
