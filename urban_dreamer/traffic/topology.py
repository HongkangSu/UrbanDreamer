from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .routing import PreparedInputs


class TopologyError(ValueError):
    """Raised when a validated artifact cannot form the runtime topology."""


@dataclass(frozen=True)
class VehicleDemand:
    vehicle_id: str
    origin_index: int
    destination_index: int
    departure_second: int
    is_automated: bool


@dataclass(frozen=True)
class EdgeNetwork:
    edge_ids: tuple[str, ...]
    edge_to_index: dict[str, int]
    static_features: np.ndarray
    speed_limits_mps: np.ndarray
    capacities: np.ndarray
    lengths_m: np.ndarray
    adjacency: np.ndarray
    adjacency_list: tuple[tuple[int, ...], ...]
    passenger_allowed: np.ndarray
    free_flow_travel_seconds: np.ndarray

    @property
    def edge_count(self) -> int:
        return len(self.edge_ids)

    def extract_flow(
        self,
        edge_state: Mapping[str, Mapping[str, Any]],
    ) -> np.ndarray:
        """Return the sole model input layout: four dynamic + four static fields."""
        flow = np.empty((self.edge_count, 8), dtype=np.float32)
        for edge_index, edge_id in enumerate(self.edge_ids):
            state = edge_state.get(edge_id)
            speed_limit = max(float(self.speed_limits_mps[edge_index]), 0.1)
            capacity = max(float(self.capacities[edge_index]), 1.0)
            if state is None:
                dynamic = (0.0, 1.0, 0.0, 0.0)
            else:
                raw_occupancy = float(state.get("occupancy", 0.0))
                occupancy = (
                    raw_occupancy / 100.0
                    if raw_occupancy > 1.0
                    else raw_occupancy
                )
                dynamic = (
                    float(np.clip(occupancy, 0.0, 1.0)),
                    float(state.get("mean_speed", speed_limit)) / speed_limit,
                    float(state.get("vehicle_number", 0.0)) / capacity,
                    float(state.get("halting_number", 0.0)) / capacity,
                )
            flow[edge_index, :4] = dynamic
            flow[edge_index, 4:] = self.static_features[edge_index]
        return flow


@dataclass(frozen=True)
class TrafficTopology:
    network: EdgeNetwork
    origins: tuple[str, ...]
    destinations: tuple[str, ...]
    vehicles: tuple[VehicleDemand, ...]
    od_pairs: tuple[tuple[int, int], ...]
    route_paths: tuple[tuple[tuple[int, ...], ...], ...]
    od_automated_counts: np.ndarray
    od_human_counts: np.ndarray

    @property
    def route_count(self) -> int:
        return len(self.route_paths[0])

    @property
    def od_to_index(self) -> dict[tuple[int, int], int]:
        return {pair: index for index, pair in enumerate(self.od_pairs)}


def load_traffic_topology(prepared: PreparedInputs) -> TrafficTopology:
    network = _load_edge_network(prepared.edge_registry)
    origins, destinations = _load_od_edges(prepared.od_file)
    unknown_od_edges = [
        edge_id
        for edge_id in (*origins, *destinations)
        if edge_id not in network.edge_to_index
    ]
    if unknown_od_edges:
        raise TopologyError(
            "OD file references edges absent from edge_registry.json: "
            + ", ".join(unknown_od_edges[:10])
        )
    machine_ids, human_ids = _load_fixed_roles(prepared.post_mutation_state)
    vehicles = _load_vehicle_demand(
        prepared.route_agents_csv,
        machine_ids=machine_ids,
        human_ids=human_ids,
        origin_count=len(origins),
        destination_count=len(destinations),
    )
    od_pairs, route_paths = _load_route_paths(
        prepared.routes_csv,
        edge_to_index=network.edge_to_index,
        expected_route_count=prepared.route_count,
    )
    pair_to_index = {pair: index for index, pair in enumerate(od_pairs)}
    automated_counts = np.zeros((len(od_pairs),), dtype=np.float32)
    human_counts = np.zeros((len(od_pairs),), dtype=np.float32)
    for vehicle in vehicles:
        pair = (vehicle.origin_index, vehicle.destination_index)
        if pair not in pair_to_index:
            raise TopologyError(
                f"vehicle {vehicle.vehicle_id!r} uses OD pair {pair} absent from routes.csv"
            )
        target = automated_counts if vehicle.is_automated else human_counts
        target[pair_to_index[pair]] += 1.0

    return TrafficTopology(
        network=network,
        origins=origins,
        destinations=destinations,
        vehicles=vehicles,
        od_pairs=od_pairs,
        route_paths=route_paths,
        od_automated_counts=automated_counts,
        od_human_counts=human_counts,
    )


def _load_edge_network(path: Path) -> EdgeNetwork:
    raw = _read_json_object(path)
    edge_ids = tuple(str(value) for value in _required_list(raw, "edge_ids"))
    edge_count = len(edge_ids)
    if edge_count == 0 or len(set(edge_ids)) != edge_count:
        raise TopologyError("edge registry must contain unique edge IDs")

    static_features = _float_array(
        raw, "static_features", expected_shape=(edge_count, 4)
    )
    speed_limits = _float_array(
        raw, "speed_limits", expected_shape=(edge_count,)
    )
    capacities = _float_array(raw, "capacities", expected_shape=(edge_count,))
    lengths = _float_array(raw, "lengths", expected_shape=(edge_count,))
    adjacency = _float_array(
        raw, "adjacency", expected_shape=(edge_count, edge_count)
    )
    passenger_allowed = np.asarray(
        _required_list(raw, "passenger_allowed"),
        dtype=np.bool_,
    )
    if passenger_allowed.shape != (edge_count,):
        raise TopologyError(
            "edge registry passenger_allowed must have one value per edge"
        )
    if (
        not np.all(np.isfinite(static_features))
        or not np.all(np.isfinite(speed_limits))
        or not np.all(np.isfinite(capacities))
        or not np.all(np.isfinite(lengths))
        or not np.all(np.isfinite(adjacency))
    ):
        raise TopologyError("edge registry numeric arrays must be finite")
    if np.any(speed_limits <= 0.0) or np.any(capacities <= 0.0):
        raise TopologyError("edge speed limits and capacities must be positive")
    if np.any(lengths < 0.0) or np.any(adjacency < 0.0):
        raise TopologyError("edge lengths and adjacency must be non-negative")

    adjacency_list: list[tuple[int, ...]] = []
    for source in range(edge_count):
        if not bool(passenger_allowed[source]):
            adjacency_list.append(())
            continue
        neighbors = np.flatnonzero(adjacency[source] > 0.0)
        adjacency_list.append(
            tuple(
                int(target)
                for target in neighbors
                if bool(passenger_allowed[int(target)])
            )
        )
    free_flow = lengths / np.maximum(speed_limits, 0.1)
    return EdgeNetwork(
        edge_ids=edge_ids,
        edge_to_index={edge_id: index for index, edge_id in enumerate(edge_ids)},
        static_features=static_features,
        speed_limits_mps=speed_limits,
        capacities=capacities,
        lengths_m=lengths,
        adjacency=adjacency,
        adjacency_list=tuple(adjacency_list),
        passenger_allowed=passenger_allowed,
        free_flow_travel_seconds=free_flow.astype(np.float32),
    )


def _load_od_edges(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        raw = ast.literal_eval(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError) as exc:
        raise TopologyError(f"cannot parse OD file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TopologyError("OD file must contain a mapping")
    origins = tuple(str(value) for value in _required_list(raw, "origins"))
    destinations = tuple(str(value) for value in _required_list(raw, "destinations"))
    if not origins or not destinations:
        raise TopologyError("OD file origins and destinations must be non-empty")
    return origins, destinations


def _load_fixed_roles(path: Path) -> tuple[set[str], set[str]]:
    state = _read_json_object(path)
    machine_ids = {str(value) for value in _required_list(state, "machine_agent_ids")}
    human_ids = {str(value) for value in _required_list(state, "human_agent_ids")}
    return machine_ids, human_ids


def _load_vehicle_demand(
    path: Path,
    *,
    machine_ids: set[str],
    human_ids: set[str],
    origin_count: int,
    destination_count: int,
) -> tuple[VehicleDemand, ...]:
    vehicles: list[VehicleDemand] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "origin", "destination", "start_time"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise TopologyError(
                "agents.csv must contain id,origin,destination,start_time"
            )
        for row in reader:
            vehicle_id = str(row["id"])
            try:
                origin = int(row["origin"])
                destination = int(row["destination"])
                departure = int(float(row["start_time"]))
            except (TypeError, ValueError) as exc:
                raise TopologyError(
                    f"invalid demand row for vehicle {vehicle_id!r}"
                ) from exc
            if not 0 <= origin < origin_count or not 0 <= destination < destination_count:
                raise TopologyError(
                    f"vehicle {vehicle_id!r} has out-of-range OD indices"
                )
            if departure < 0:
                raise TopologyError(
                    f"vehicle {vehicle_id!r} has a negative departure time"
                )
            if vehicle_id not in machine_ids and vehicle_id not in human_ids:
                raise TopologyError(
                    f"vehicle {vehicle_id!r} has no fixed post-mutation role"
                )
            vehicles.append(
                VehicleDemand(
                    vehicle_id=vehicle_id,
                    origin_index=origin,
                    destination_index=destination,
                    departure_second=departure,
                    is_automated=vehicle_id in machine_ids,
                )
            )
    if not vehicles:
        raise TopologyError("agents.csv contains no vehicles")
    return tuple(vehicles)


def _load_route_paths(
    path: Path,
    *,
    edge_to_index: Mapping[str, int],
    expected_route_count: int,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[tuple[int, ...], ...], ...],
]:
    pair_order: list[tuple[int, int]] = []
    grouped: dict[tuple[int, int], list[tuple[int, ...]]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"origins", "destinations", "path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise TopologyError(
                "routes.csv must contain origins,destinations,path"
            )
        for row in reader:
            try:
                pair = (int(row["origins"]), int(row["destinations"]))
            except (TypeError, ValueError) as exc:
                raise TopologyError("routes.csv contains invalid OD indices") from exc
            if pair not in grouped:
                pair_order.append(pair)
                grouped[pair] = []
            edge_names = str(row["path"]).split()
            try:
                edge_path = tuple(edge_to_index[name] for name in edge_names)
            except KeyError as exc:
                raise TopologyError(
                    f"routes.csv references unknown edge {exc.args[0]!r}"
                ) from exc
            if not edge_path:
                raise TopologyError("routes.csv contains an empty route")
            grouped[pair].append(edge_path)

    for pair, routes in grouped.items():
        if len(routes) != expected_route_count:
            raise TopologyError(
                f"OD pair {pair} has {len(routes)} routes; "
                f"expected {expected_route_count}"
            )
    if not pair_order:
        raise TopologyError("routes.csv contains no routes")
    return (
        tuple(pair_order),
        tuple(tuple(grouped[pair]) for pair in pair_order),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopologyError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TopologyError(f"{path} must contain a JSON object")
    return raw


def _required_list(raw: Mapping[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise TopologyError(f"missing or invalid list field {key!r}")
    return value


def _float_array(
    raw: Mapping[str, Any],
    key: str,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    try:
        value = np.asarray(raw[key], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise TopologyError(f"missing or invalid numeric field {key!r}") from exc
    if value.shape != expected_shape:
        raise TopologyError(
            f"edge registry {key} has shape {value.shape}; expected {expected_shape}"
        )
    return value
