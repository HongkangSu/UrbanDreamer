from __future__ import annotations

import itertools
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..traffic.topology import TopologyError

CANDIDATE_FEATURE_DIM = 10

_LOG_DISTANCE_SCALE = float(np.log1p(10000.0))
_LOG_HOUR_SCALE = float(np.log1p(3600.0))
_NAN = float("nan")


@dataclass(frozen=True)
class CandidateObservations:
    features: np.ndarray
    mask: np.ndarray
    edge_indices: np.ndarray


def _log_distance(values: Any) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(values, dtype=np.float64), 0.0)) / _LOG_DISTANCE_SCALE


def _edge_features(value: np.ndarray, edge_count: int) -> np.ndarray:
    features = np.asarray(value, dtype=np.float32)
    if features.shape != (edge_count, 2) or not np.isfinite(features).all():
        raise ValueError("policy_edge_features must be finite with shape (E, 2)")
    return features


def _int_column(states: Sequence[dict[str, Any]], key: str, count: int) -> np.ndarray:
    return np.fromiter(map(operator.itemgetter(key), states), dtype=np.int64, count=count)


def _float_column(states: Sequence[dict[str, Any]], key: str, default: float, count: int) -> np.ndarray:
    return np.fromiter(map(dict.get, states, itertools.repeat(key), itertools.repeat(default)),
                       dtype=np.float64, count=count)


def build_actor_observations(*, policy_edge_features: np.ndarray,
                             vehicle_states: Sequence[dict[str, Any]], candidates: Any,
                             episode_progress: float, active_vehicle_fraction: float) -> CandidateObservations:
    """Score candidate traffic, static cost and destination distance without path materialization."""
    network = candidates.network
    edges = _edge_features(policy_edge_features, network.edge_count)
    count, width = len(vehicle_states), candidates.max_candidates
    if count == 0:
        return CandidateObservations(np.zeros((0, width, CANDIDATE_FEATURE_DIM), dtype=np.float32),
                                     np.zeros((0, width), dtype=bool),
                                     np.full((0, width), -1, dtype=np.int64))
    global_features = edges.mean(axis=0)
    edge_count = network.edge_count
    currents = _int_column(vehicle_states, 'current_edge', count)
    destinations = _int_column(vehicle_states, 'destination_edge', count)
    remaining = _float_column(vehicle_states, 'remaining_distance', _NAN, count)
    if ((currents < 0) | (currents >= edge_count)).any() or ((destinations < 0) | (destinations >= edge_count)).any():
        for row in range(count):
            current, destination = int(currents[row]), int(destinations[row])
            if not 0 <= current < edge_count:
                raise TopologyError(f"unknown current edge {current}")
            if not 0 <= destination < edge_count:
                raise TopologyError(f"unknown destination edge {destination}")
    packed = currents * edge_count + destinations
    unique_pairs, pair_inverse = np.unique(packed, return_inverse=True)
    pair_destinations = unique_pairs % edge_count
    pair_choices = list(map(candidates.candidates, unique_pairs // edge_count, pair_destinations))
    pair_counts = np.fromiter(map(len, pair_choices), dtype=np.int64, count=len(pair_choices))
    if (pair_counts == 0).any():
        row = int(np.flatnonzero(np.isin(pair_inverse, np.flatnonzero(pair_counts == 0)))[0])
        raise ValueError(f"vehicle {vehicle_states[row].get('vehicle_id', row)!r} has no legal next-hop candidate")
    pair_mask = np.arange(width) < pair_counts[:, None]
    pair_indices = np.full((len(pair_choices), width), -1, dtype=np.int64)
    pair_indices[pair_mask] = np.fromiter(itertools.chain.from_iterable(pair_choices),
                                          dtype=np.int64, count=int(pair_counts.sum()))
    mask = pair_mask[pair_inverse]
    indices = pair_indices[pair_inverse]
    pair_safe = np.where(pair_mask, pair_indices, 0)
    stacked = getattr(candidates, 'stacked_distance_tables', None)
    if callable(stacked):
        unique_destinations, destination_inverse, distance_tables = stacked(destinations, seconds=False)
        cost_tables = stacked(destinations, seconds=True)[2]
    else:
        unique_destinations, destination_inverse = np.unique(destinations, return_inverse=True)
        distance_tables = np.stack([candidates.distances(int(edge)) for edge in unique_destinations])
        cost_tables = np.stack([candidates.costs(int(edge)) for edge in unique_destinations])
    pair_dest_rows = np.searchsorted(unique_destinations, pair_destinations)
    pair_distances = distance_tables[pair_dest_rows[:, None], pair_safe]
    pair_costs = cost_tables[pair_dest_rows[:, None], pair_safe]
    pair_currents = unique_pairs // edge_count
    junction_matrices = getattr(candidates, 'junction_matrices', None)
    if callable(junction_matrices):
        dense_lengths, dense_speeds = junction_matrices()
        junction_lengths = dense_lengths[pair_currents[:, None], pair_safe].astype(np.float64)
        junction_speeds = dense_speeds[pair_currents[:, None], pair_safe].astype(np.float64)
    else:
        junction_lengths = np.zeros((len(pair_choices), width), dtype=np.float64)
        junction_speeds = np.ones((len(pair_choices), width), dtype=np.float64)
        for pair, choices in enumerate(pair_choices):
            for column, edge in enumerate(choices):
                length, speed = candidates.junction(pair_currents[pair], edge)
                junction_lengths[pair, column], junction_speeds[pair, column] = length, speed
    travel_seconds = (junction_lengths / junction_speeds
                      + network.free_flow_travel_seconds[pair_safe] + pair_costs)
    total_distance = junction_lengths + network.lengths_m[pair_safe] + pair_distances
    pair_features = np.zeros((len(pair_choices), width, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    np.copyto(pair_features[:, :, 0:2], edges[pair_safe], where=pair_mask[..., None])
    np.copyto(pair_features[:, :, 2:4], global_features, where=pair_mask[..., None])
    np.copyto(pair_features[:, :, 4], np.log1p(travel_seconds) / _LOG_HOUR_SCALE, where=pair_mask)
    np.copyto(pair_features[:, :, 5], _log_distance(total_distance), where=pair_mask)
    np.copyto(pair_features[:, :, 7], pair_indices == pair_destinations[:, None], where=pair_mask)
    np.copyto(pair_features[:, :, 8], episode_progress, where=pair_mask)
    np.copyto(pair_features[:, :, 9], active_vehicle_fraction, where=pair_mask)
    features = pair_features[pair_inverse]
    remaining = np.where(np.isnan(remaining), network.lengths_m[currents], remaining)
    np.copyto(features[:, :, 6], _log_distance(remaining)[:, None], where=mask)
    return CandidateObservations(features, mask, indices)


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
