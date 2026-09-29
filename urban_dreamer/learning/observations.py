from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

CANDIDATE_FEATURE_DIM = 10
VEHICLE_OBSERVATION_DIM = 14


@dataclass(frozen=True)
class CandidateObservations:
    features: np.ndarray
    mask: np.ndarray
    edge_indices: np.ndarray


def _log_distance(value: float) -> float:
    return float(np.log1p(max(float(value), 0.0)) / np.log1p(10000.0))


def _edge_features(value: np.ndarray, edge_count: int) -> np.ndarray:
    features = np.asarray(value, dtype=np.float32)
    if features.shape != (edge_count, 2) or not np.isfinite(features).all():
        raise ValueError("policy_edge_features must be finite with shape (E, 2)")
    return features


def build_actor_observations(*, policy_edge_features: np.ndarray,
                             vehicle_states: Sequence[dict[str, Any]], candidates: Any,
                             episode_progress: float, active_vehicle_fraction: float) -> CandidateObservations:
    """Score candidate traffic, static cost and destination distance without path materialization."""
    network = candidates.network
    edges = _edge_features(policy_edge_features, network.edge_count)
    count, width = len(vehicle_states), candidates.max_candidates
    features = np.zeros((count, width, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((count, width), dtype=bool)
    indices = np.full((count, width), -1, dtype=np.int64)
    global_features = edges.mean(axis=0)
    for row, state in enumerate(vehicle_states):
        current, destination = int(state['current_edge']), int(state['destination_edge'])
        choices = candidates.candidates(current, destination)
        if not choices:
            raise ValueError(f"vehicle {state.get('vehicle_id', row)!r} has no legal next-hop candidate")
        distances, costs = candidates.distances(destination), candidates.costs(destination)
        for column, edge in enumerate(choices):
            junction_length, junction_speed = candidates.junction(current, edge)
            indices[row, column], mask[row, column] = edge, True
            features[row, column] = (*edges[edge], *global_features,
                np.log1p(junction_length / junction_speed + network.free_flow_travel_seconds[edge] + costs[edge]) / np.log1p(3600.),
                _log_distance(junction_length + network.lengths_m[edge] + distances[edge]),
                _log_distance(state.get('remaining_distance', network.lengths_m[current])),
                float(edge == destination), episode_progress, active_vehicle_fraction)
    return CandidateObservations(features, mask, indices)


def build_vehicle_observations(*, policy_edge_features: np.ndarray,
                               vehicle_states: Sequence[dict[str, Any]], candidates: Any,
                               episode_progress: float, active_vehicle_fraction: float,
                               current_time: float) -> np.ndarray:
    """Vehicle value features remain defined between policy events and at horizon cuts."""
    network = candidates.network
    edges = _edge_features(policy_edge_features, network.edge_count)
    result = np.zeros((len(vehicle_states), VEHICLE_OBSERVATION_DIM), dtype=np.float32)
    global_features = edges.mean(axis=0)
    for row, state in enumerate(vehicle_states):
        current, destination = int(state['current_edge']), int(state['destination_edge'])
        committed = state.get('committed_next_edge')
        distances = candidates.distances(destination)
        remaining = float(state.get('remaining_distance', network.lengths_m[current]))
        junction = float(state.get('junction_remaining_distance', 0.0))
        junction_seconds = junction / max(float(state.get('junction_speed', 1.0)), 1e-6)
        committed_distance = 0.0 if committed is None else network.lengths_m[int(committed)] + distances[int(committed)]
        result[row] = (*edges[current], *global_features, _log_distance(remaining),
                       _log_distance(distances[current]), _log_distance(committed_distance),
                       junction_seconds / 30.0, float(state.get('in_junction', False)),
                       float(state.get('departed', False)), float(state.get('arrived', False)),
                       max(float(state.get('departure_time', current_time)) - current_time, 0.) / 30.,
                       episode_progress, active_vehicle_fraction)
    if not np.isfinite(result).all():
        raise ValueError("vehicle critic state must have a reachable destination and finite progress")
    return result


def aggregate_next_hop_actions(decisions: Sequence[Any], *, edge_count: int, total_vehicle_count: int) -> np.ndarray:
    if edge_count <= 0 or total_vehicle_count <= 0:
        raise ValueError("edge_count and total_vehicle_count must be positive")
    counts = np.zeros(edge_count, dtype=np.float32)
    for decision in decisions:
        edge = int(decision.next_edge)
        if not 0 <= edge < edge_count:
            raise ValueError("next edge is outside the network")
        counts[edge] += 1.
    return counts / float(total_vehicle_count)
