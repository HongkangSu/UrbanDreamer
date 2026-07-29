from __future__ import annotations

import csv
import heapq
import json
import math
import os
import random
import shutil
import types
from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

_ROUTE_COUNT = 4
_DECISION_INTERVAL_SECONDS = 30
_PATH_PENALTY_FACTOR = 3.0
_PATH_CACHE_CAPACITY = 250_000
_HDV_GAWRON_ALPHA = 0.2
_HDV_GAWRON_BETA = -1.5


class EnvironmentContractError(ValueError):
    """Raised when environment-layer data becomes internally inconsistent."""


class SessionClosedError(RuntimeError):
    """Raised when a closed traffic session is used again."""


def _plain_key(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def to_plain_state(value: Any) -> Any:
    """Convert nested dataclasses/tuples into checkpoint-safe plain Python data."""
    if is_dataclass(value):
        return {
            item.name: to_plain_state(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {_plain_key(key): to_plain_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_plain_state(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize {type(value)!r} into plain checkpoint state")


def _softmin_route_probabilities(costs: Sequence[float]) -> list[float]:
    if len(costs) != _ROUTE_COUNT:
        raise EnvironmentContractError(
            f"mainline HDV costs must contain {_ROUTE_COUNT} routes"
        )
    scores = [_HDV_GAWRON_BETA * float(cost) for cost in costs]
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise EnvironmentContractError("cannot normalize HDV route probabilities")
    return [weight / total for weight in weights]


class CandidatePathGenerator:
    """Penalty-Dijkstra current-edge to destination candidate routes with bounded LRU cache."""

    def __init__(
        self,
        *,
        adjacency: Sequence[Sequence[int]],
        free_flow_costs: Sequence[float],
    ) -> None:
        costs = tuple(float(cost) for cost in free_flow_costs)
        if not costs or len(adjacency) != len(costs):
            raise EnvironmentContractError(
                "adjacency and free_flow_costs must have the same non-zero edge count"
            )
        if any(not math.isfinite(cost) or cost <= 0.0 for cost in costs):
            raise EnvironmentContractError(
                "free-flow costs must be finite and positive"
            )
        edge_count = len(costs)
        graph = {
            edge_index: tuple(int(neighbor) for neighbor in neighbors)
            for edge_index, neighbors in enumerate(adjacency)
        }
        if any(
            neighbor < 0 or neighbor >= edge_count
            for neighbors in graph.values()
            for neighbor in neighbors
        ):
            raise EnvironmentContractError(
                "adjacency references an edge outside free_flow_costs"
            )
        self._graph = graph
        self._costs = dict(enumerate(costs))
        self._cache: OrderedDict[
            tuple[int, int],
            tuple[tuple[int, ...], ...],
        ] = OrderedDict()

    def generate_paths(
        self,
        current_edge: int,
        destination_edge: int,
    ) -> tuple[tuple[int, ...], ...]:
        key = (current_edge, destination_edge)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        paths = self._compute_paths(current_edge, destination_edge)
        self._cache[key] = paths
        self._cache.move_to_end(key)
        if len(self._cache) > _PATH_CACHE_CAPACITY:
            self._cache.popitem(last=False)
        return paths

    def state_dict(self) -> dict[str, Any]:
        return {
            "cache": [
                {
                    "current_edge": current_edge,
                    "destination_edge": destination_edge,
                    "paths": [list(path) for path in paths],
                }
                for (current_edge, destination_edge), paths in self._cache.items()
            ]
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {"cache"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise EnvironmentContractError(
                "candidate-path cache state has an invalid schema"
            )
        cache_rows = state["cache"]
        if not isinstance(cache_rows, list):
            raise EnvironmentContractError(
                "candidate-path cache entries must be a list"
            )
        if len(cache_rows) > _PATH_CACHE_CAPACITY:
            raise EnvironmentContractError(
                "candidate-path cache exceeds configured capacity"
            )
        restored: OrderedDict[
            tuple[int, int],
            tuple[tuple[int, ...], ...],
        ] = OrderedDict()
        for row in cache_rows:
            if not isinstance(row, Mapping) or set(row) != {
                "current_edge",
                "destination_edge",
                "paths",
            }:
                raise EnvironmentContractError(
                    "candidate-path cache entry has an invalid schema"
                )
            current_edge = int(row["current_edge"])
            destination_edge = int(row["destination_edge"])
            if current_edge not in self._graph or destination_edge not in self._graph:
                raise EnvironmentContractError(
                    "candidate-path cache references an unknown edge"
                )
            paths = _normalize_path_collection(row["paths"])
            paths = tuple(
                tuple(int(edge) for edge in path)
                for path in paths
            )
            if len(paths) != _ROUTE_COUNT:
                raise EnvironmentContractError(
                    f"candidate-path cache must contain {_ROUTE_COUNT} routes"
                )
            for path in paths:
                if any(edge not in self._graph for edge in path):
                    raise EnvironmentContractError(
                        "candidate-path cache route references an unknown edge"
                    )
            restored[(current_edge, destination_edge)] = paths
        self._cache = restored

    def _compute_paths(
        self,
        current_edge: int,
        destination_edge: int,
    ) -> tuple[tuple[int, ...], ...]:
        if current_edge not in self._graph:
            raise EnvironmentContractError(f"unknown current edge {current_edge!r}")
        if destination_edge not in self._graph:
            raise EnvironmentContractError(f"unknown destination edge {destination_edge!r}")
        if current_edge == destination_edge:
            path = (current_edge,)
            return tuple(path for _ in range(_ROUTE_COUNT))

        costs = dict(self._costs)
        paths: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for _ in range(_ROUTE_COUNT):
            path = self._shortest_path(current_edge, destination_edge, costs)
            if path is None:
                break
            if path in seen:
                for edge in path:
                    costs[edge] *= _PATH_PENALTY_FACTOR
                path = self._shortest_path(current_edge, destination_edge, costs)
                if path is None or path in seen:
                    break
            paths.append(path)
            seen.add(path)
            for edge in path:
                costs[edge] *= _PATH_PENALTY_FACTOR

        if not paths:
            paths = [(current_edge,)]
        while len(paths) < _ROUTE_COUNT:
            paths.append(paths[0])
        return tuple(paths[:_ROUTE_COUNT])

    def _shortest_path(
        self,
        current_edge: int,
        destination_edge: int,
        costs: Mapping[int, float],
    ) -> tuple[int, ...] | None:
        best_cost: dict[int, float] = {current_edge: float(costs[current_edge])}
        previous: dict[int, int] = {}
        frontier: list[tuple[float, int, int]] = [
            (float(costs[current_edge]), 0, current_edge)
        ]
        sequence = 1
        while frontier:
            cumulative_cost, _, edge = heapq.heappop(frontier)
            if cumulative_cost > best_cost.get(edge, float("inf")):
                continue
            if edge == destination_edge:
                return self._reconstruct_path(previous, current_edge, destination_edge)
            for neighbor in self._graph.get(edge, ()):
                candidate_cost = cumulative_cost + float(costs[neighbor])
                if candidate_cost < best_cost.get(neighbor, float("inf")):
                    best_cost[neighbor] = candidate_cost
                    previous[neighbor] = edge
                    heapq.heappush(frontier, (candidate_cost, sequence, neighbor))
                    sequence += 1
        return None

    @staticmethod
    def _reconstruct_path(
        previous: Mapping[int, int],
        current_edge: int,
        destination_edge: int,
    ) -> tuple[int, ...]:
        path: list[int] = [destination_edge]
        cursor = destination_edge
        while cursor != current_edge:
            cursor = previous[cursor]
            path.append(cursor)
        path.reverse()
        return tuple(path)


def _normalize_path_collection(
    candidate_paths: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if not candidate_paths:
        raise EnvironmentContractError("candidate_paths must be non-empty")
    normalized = tuple(tuple(path) for path in candidate_paths)
    if any(not path for path in normalized):
        raise EnvironmentContractError("candidate_paths cannot contain an empty path")
    return normalized


def _normalize_edge_sequence(values: Sequence[int]) -> tuple[int, ...]:
    if not values:
        raise EnvironmentContractError("edge sequence must be non-empty")
    return tuple(values)


def _normalize_edge_travel_times(
    values: Sequence[float],
    *,
    expected_length: int | None = None,
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise EnvironmentContractError("edge travel times must be non-empty")
    if expected_length is not None and len(normalized) != expected_length:
        raise EnvironmentContractError(
            "edge travel-time count "
            f"{len(normalized)} does not match expected {expected_length}"
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in normalized):
        raise EnvironmentContractError(
            "edge travel times must be finite and positive"
        )
    return normalized


def _derive_planned_edge_flow(
    *,
    selected_path: Sequence[int],
    edge_order: Sequence[int],
    edge_travel_times: Sequence[float],
    decision_horizon_seconds: float,
) -> tuple[float, ...]:
    if float(decision_horizon_seconds) < 0.0:
        raise EnvironmentContractError("decision_horizon_seconds must be non-negative")
    edge_order_tuple = _normalize_edge_sequence(edge_order)
    edge_travel_times_tuple = _normalize_edge_travel_times(
        edge_travel_times,
        expected_length=len(edge_order_tuple),
    )
    selected_path_tuple = _normalize_edge_sequence(selected_path)
    edge_to_index = {edge: index for index, edge in enumerate(edge_order_tuple)}
    missing = [edge for edge in selected_path_tuple if edge not in edge_to_index]
    if missing:
        raise EnvironmentContractError(
            "selected_path contains edges outside edge_order: "
            + ", ".join(str(edge) for edge in missing)
        )

    planned = [0.0 for _ in edge_order_tuple]
    elapsed = 0.0
    for position, edge in enumerate(selected_path_tuple):
        travel_time = edge_travel_times_tuple[edge_to_index[edge]]
        if position == 0:
            elapsed += travel_time
            continue
        if elapsed >= float(decision_horizon_seconds):
            break
        planned[edge_to_index[edge]] += 1.0
        elapsed += travel_time
    return tuple(planned)


@dataclass(frozen=True)
class ActionDecision:
    route_index: int
    route_one_hot: tuple[int, ...]
    candidate_paths: tuple[tuple[int, ...], ...]
    selected_path: tuple[int, ...]
    edge_order: tuple[int, ...]
    edge_travel_times: tuple[float, ...]
    decision_horizon_seconds: float
    planned_edge_flow: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.candidate_paths) != _ROUTE_COUNT:
            raise EnvironmentContractError(
                f"mainline actions require exactly {_ROUTE_COUNT} candidate paths"
            )
        if not 0 <= int(self.route_index) < len(self.candidate_paths):
            raise EnvironmentContractError("route_index is outside candidate_paths")
        if len(self.route_one_hot) != len(self.candidate_paths):
            raise EnvironmentContractError(
                "route_one_hot length must match candidate_paths length"
            )
        if tuple(self.route_one_hot) != tuple(
            1 if index == int(self.route_index) else 0
            for index in range(len(self.candidate_paths))
        ):
            raise EnvironmentContractError(
                "route_one_hot must exactly encode route_index"
            )
        if self.selected_path != self.candidate_paths[int(self.route_index)]:
            raise EnvironmentContractError(
                "selected_path must exactly match candidate_paths[route_index]"
            )
        if len(self.edge_order) != len(self.edge_travel_times):
            raise EnvironmentContractError(
                "edge_order and edge_travel_times must have the same length"
            )
        expected_flow = _derive_planned_edge_flow(
            selected_path=self.selected_path,
            edge_order=self.edge_order,
            edge_travel_times=self.edge_travel_times,
            decision_horizon_seconds=self.decision_horizon_seconds,
        )
        if tuple(self.planned_edge_flow) != expected_flow:
            raise EnvironmentContractError(
                "planned_edge_flow must be derived from selected_path and decision horizon"
            )

    @classmethod
    def build(
        cls,
        *,
        route_index: int,
        candidate_paths: Sequence[Sequence[int]],
        edge_order: Sequence[int],
        edge_travel_times: Sequence[float],
        decision_horizon_seconds: float,
    ) -> ActionDecision:
        normalized_paths = _normalize_path_collection(candidate_paths)
        normalized_edge_order = _normalize_edge_sequence(edge_order)
        normalized_travel_times = _normalize_edge_travel_times(
            edge_travel_times,
            expected_length=len(normalized_edge_order),
        )
        selected = normalized_paths[int(route_index)]
        planned_edge_flow = _derive_planned_edge_flow(
            selected_path=selected,
            edge_order=normalized_edge_order,
            edge_travel_times=normalized_travel_times,
            decision_horizon_seconds=float(decision_horizon_seconds),
        )
        return cls(
            route_index=int(route_index),
            route_one_hot=tuple(
                1 if index == int(route_index) else 0
                for index in range(len(normalized_paths))
            ),
            candidate_paths=normalized_paths,
            selected_path=selected,
            edge_order=normalized_edge_order,
            edge_travel_times=normalized_travel_times,
            decision_horizon_seconds=float(decision_horizon_seconds),
            planned_edge_flow=planned_edge_flow,
        )

@dataclass(frozen=True)
class RouteRLSessionSpec:
    network_name: str
    network_directory: Path
    records_directory: Path
    agents_path: Path
    routes_path: Path
    route_xml_path: Path
    fixed_machine_ids: tuple[str, ...]
    fixed_human_ids: tuple[str, ...]
    simulation_timesteps: int
    seed: int
    origins: tuple[str, ...]
    destinations: tuple[str, ...]
    initial_hdv_learning_state: Mapping[str, Any]
    sumo_extra_args: tuple[str, ...] = ()
    incident_edge: str = ""
    incident_start_seconds: float = 0.0
    incident_end_seconds: float = 0.0
    incident_speed_mps: float = 0.0
    episode_counter: int = 0

    def __post_init__(self) -> None:
        if not self.network_name:
            raise EnvironmentContractError("network_name must be non-empty")
        if not self.agents_path:
            raise EnvironmentContractError("agents_path must be provided")
        if not self.routes_path:
            raise EnvironmentContractError("routes_path must be provided")
        if not self.route_xml_path:
            raise EnvironmentContractError("route_xml_path must be provided")
        if int(self.simulation_timesteps) <= 0:
            raise EnvironmentContractError("simulation_timesteps must be positive")
        if not self.origins or not self.destinations:
            raise EnvironmentContractError(
                "origins and destinations must be non-empty"
            )
        if set(self.fixed_machine_ids).intersection(self.fixed_human_ids):
            raise EnvironmentContractError(
                "fixed_machine_ids and fixed_human_ids must not overlap"
            )
        if float(self.incident_start_seconds) < 0.0:
            raise EnvironmentContractError("incident_start_seconds must be non-negative")
        if float(self.incident_end_seconds) < float(self.incident_start_seconds):
            raise EnvironmentContractError(
                "incident_end_seconds must be >= incident_start_seconds"
            )
        if float(self.incident_speed_mps) < 0.0:
            raise EnvironmentContractError("incident_speed_mps must be non-negative")


@runtime_checkable
class RouteRLBackend(Protocol):
    def reset(self) -> Mapping[str, Any]: ...

    def get_traffic_state(self) -> Mapping[str, Any]: ...

    def set_vehicle_route(
        self,
        vehicle_id: str,
        route_index: int,
        route_edges: Sequence[str],
    ) -> None: ...

    def advance(self, steps: int) -> None: ...

    def get_active_vehicle_exposure(self) -> float: ...

    def is_done(self) -> bool: ...

    def travel_times_seconds(self) -> Mapping[str, float]: ...

    def hdv_route_choices(self) -> Mapping[str, int]: ...

    def travel_time_route_choices(self) -> Mapping[str, int]: ...

    def hdv_learning_state(self) -> Mapping[str, Any]: ...

    def restore_hdv_learning_state(self, state: Mapping[str, Any]) -> None: ...

    def update_hdv_learning(
        self, travel_times_seconds: Mapping[str, float] | None = None
    ) -> int: ...

    def close(self) -> None: ...


class RouteRLSumoSession:
    """Minimal SUMO/RouteRL session wrapper with lazy dependency imports."""

    def __init__(
        self,
        *,
        edge_ids: Sequence[str],
        spec: RouteRLSessionSpec | None = None,
        backend_factory: Callable[[], RouteRLBackend] | None = None,
    ) -> None:
        if spec is None and backend_factory is None:
            raise EnvironmentContractError(
                "provide either a RouteRLSessionSpec or an explicit backend_factory"
            )
        normalized_edge_ids = tuple(str(edge_id) for edge_id in edge_ids)
        if not normalized_edge_ids or len(set(normalized_edge_ids)) != len(
            normalized_edge_ids
        ):
            raise EnvironmentContractError(
                "edge_ids must contain unique SUMO edge IDs"
            )
        self._spec = spec
        self._backend_factory = backend_factory
        self._edge_ids = normalized_edge_ids
        self._backend: RouteRLBackend | None = None
        self._closed = False
        self._last_active_vehicle_exposure = 0.0

    @property
    def last_active_vehicle_exposure(self) -> float:
        return self._last_active_vehicle_exposure

    def reset(self) -> dict[str, Any]:
        backend = self._ensure_backend()
        self._last_active_vehicle_exposure = 0.0
        return dict(backend.reset())

    def get_traffic_state(self) -> dict[str, Any]:
        return dict(self._ensure_backend().get_traffic_state())

    def is_done(self) -> bool:
        return bool(self._ensure_backend().is_done())

    def travel_times_seconds(self) -> dict[str, float]:
        return {
            str(vehicle_id): float(seconds)
            for vehicle_id, seconds in self._ensure_backend()
            .travel_times_seconds()
            .items()
        }

    def hdv_route_choices(self) -> dict[str, int]:
        return {
            str(vehicle_id): int(route_index)
            for vehicle_id, route_index in self._ensure_backend()
            .hdv_route_choices()
            .items()
        }

    def travel_time_route_choices(self) -> dict[str, int]:
        return {
            str(vehicle_id): int(route_index)
            for vehicle_id, route_index in self._ensure_backend()
            .travel_time_route_choices()
            .items()
        }

    def hdv_learning_state(self) -> dict[str, Any]:
        return dict(to_plain_state(self._ensure_backend().hdv_learning_state()))

    def update_hdv_learning(
        self, travel_times_seconds: Mapping[str, float] | None = None
    ) -> int:
        return int(
            self._ensure_backend().update_hdv_learning(travel_times_seconds)
        )

    def set_routes_safely(
        self,
        assignments: Mapping[str, ActionDecision],
    ) -> None:
        backend = self._ensure_backend()
        for vehicle_id, assignment in assignments.items():
            route_edges: list[str] = []
            for raw_edge_index in assignment.selected_path:
                edge_index = int(raw_edge_index)
                if edge_index < 0 or edge_index >= len(self._edge_ids):
                    raise EnvironmentContractError(
                        f"route references unknown edge index {edge_index}"
                    )
                route_edges.append(self._edge_ids[edge_index])
            try:
                backend.set_vehicle_route(
                    str(vehicle_id),
                    int(assignment.route_index),
                    route_edges,
                )
            except Exception:
                continue

    def advance(self, steps: int) -> None:
        backend = self._ensure_backend()
        backend.advance(int(steps))
        self._last_active_vehicle_exposure = float(
            backend.get_active_vehicle_exposure()
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._backend is not None:
            self._backend.close()
        self._backend = None
        self._closed = True

    def _ensure_backend(self) -> RouteRLBackend:
        if self._closed:
            raise SessionClosedError("RouteRLSumoSession is already closed")
        if self._backend is None:
            factory = self._backend_factory
            if factory is None:
                factory = self._build_default_backend
            self._backend = factory()
        return self._backend

    def _build_default_backend(self) -> RouteRLBackend:
        if self._spec is None:
            raise EnvironmentContractError("RouteRLSessionSpec is required for default backend creation")
        try:
            from routerl import TrafficEnvironment  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "routerl is required to build the default RouteRL backend"
            ) from exc
        try:
            import traci  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "traci is required to build the default RouteRL backend"
            ) from exc
        return _LiveRouteRLBackend(
            spec=self._spec,
            traffic_environment_cls=TrafficEnvironment,
            traci_module=traci,
        )


@dataclass(frozen=True)
class _DemandVehicle:
    vehicle_id: str
    origin: int
    destination: int
    depart_seconds: float
    kind: str


@dataclass(frozen=True)
class _StagedRouteRLArtifacts:
    agents_csv: Path
    routes_csv: Path
    route_xml: Path


class _LiveRouteRLBackend(RouteRLBackend):
    """Thin adapter over RouteRL TrafficEnvironment and a live traci module."""

    def __init__(self, *, spec: RouteRLSessionSpec, traffic_environment_cls: Any, traci_module: Any) -> None:
        self._spec = spec
        self._traci = traci_module
        self._machine_ids = {str(vehicle_id) for vehicle_id in spec.fixed_machine_ids}
        self._human_ids = {str(vehicle_id) for vehicle_id in spec.fixed_human_ids}
        self._artifacts = self._stage_routerl_artifacts()
        self._routes_by_od = self._load_routes_by_od(self._artifacts.routes_csv)
        self._demand = self._load_demand(self._artifacts.agents_csv)
        self._episode_count = int(spec.episode_counter)
        self._human_episode_actions: dict[str, int] = {}
        self._machine_episode_actions: dict[str, int] = {}
        self._depart_times_seconds: dict[str, float] = {}
        self._arrival_times_seconds: dict[str, float] = {}
        self._travel_times_seconds: dict[str, float] = {}
        self._active_vehicle_exposure = 0.0
        self._incident_lane_speeds: dict[str, float] = {}
        self._incident_active = False
        self._hdv_learning_state = self._normalized_hdv_learning_state(
            spec.initial_hdv_learning_state
        )
        self._route_policy_lookup = self._route_policy_by_od()
        self._episode_rng = random.Random(int(spec.seed) + self._episode_count)
        self._environment = traffic_environment_cls(
            seed=spec.seed,
            create_agents=False,
            create_paths=False,
            save_detectors_info=False,
            agent_parameters={
                "new_machines_after_mutation": 0,
                "human_parameters": {"model": "gawron"},
                "machine_parameters": {
                    "behavior": "selfish",
                    "observation_type": "previous_agents_plus_start_time",
                },
            },
            environment_parameters={"save_every": 1},
            simulator_parameters={
                "network_name": spec.network_name,
                "custom_network_folder": str(spec.network_directory),
                "sumo_type": "sumo",
                "simulation_timesteps": spec.simulation_timesteps,
            },
            plotter_parameters={
                "records_folder": str(spec.records_directory),
                "plots_folder": str(spec.records_directory / "plots"),
                "plot_choices": "none",
                "phases": [0, 1],
                "phase_names": ["Run"],
                "smooth_by": 5,
            },
            path_generation_parameters={
                "origins": list(spec.origins),
                "destinations": list(spec.destinations),
                "number_of_paths": _ROUTE_COUNT,
            },
        )
        self._patch_sumo_commands()
        self._started = False

    def reset(self) -> Mapping[str, Any]:
        if self._started:
            self._environment.simulator.reset()
        else:
            self._environment.start()
            self._started = True
        self._episode_count += 1
        self._active_vehicle_exposure = 0.0
        self._human_episode_actions = {}
        self._machine_episode_actions = {}
        self._depart_times_seconds = {}
        self._arrival_times_seconds = {}
        self._travel_times_seconds = {}
        self.restore_hdv_learning_state(self._hdv_learning_state)
        self._episode_rng = random.Random(int(self._spec.seed) + self._episode_count)
        self._capture_incident_lanes()
        self._add_all_demand_vehicles()
        return self.get_traffic_state()

    def get_traffic_state(self) -> Mapping[str, Any]:
        connection = self._connection()
        edge_ids = tuple(connection.edge.getIDList())
        vehicle_ids = tuple(connection.vehicle.getIDList())

        def edge_measure(method_name: str, edge_id: str, default: float = 0.0) -> float:
            method = getattr(connection.edge, method_name, None)
            if not callable(method):
                return float(default)
            try:
                return float(method(edge_id))
            except Exception:
                return float(default)

        edges = {
            str(edge_id): {
                "occupancy": edge_measure("getLastStepOccupancy", edge_id),
                "mean_speed": edge_measure("getLastStepMeanSpeed", edge_id),
                "vehicle_number": edge_measure(
                    "getLastStepVehicleNumber",
                    edge_id,
                ),
                "halting_number": edge_measure(
                    "getLastStepHaltingNumber",
                    edge_id,
                ),
            }
            for edge_id in edge_ids
            if not str(edge_id).startswith(":")
        }
        vehicles = {
            str(vehicle_id): {
                "edge": str(connection.vehicle.getRoadID(vehicle_id)),
                "speed": float(connection.vehicle.getSpeed(vehicle_id)),
                "waiting_time": float(connection.vehicle.getWaitingTime(vehicle_id)),
            }
            for vehicle_id in vehicle_ids
        }
        return {
            "simulation_time": float(connection.simulation.getTime()),
            "edges": edges,
            "vehicles": vehicles,
        }

    def set_vehicle_route(
        self,
        vehicle_id: str,
        route_index: int,
        route_edges: Sequence[str],
    ) -> None:
        if str(vehicle_id) not in self._machine_ids:
            raise RuntimeError(f"vehicle {vehicle_id!r} is not a controllable machine")
        selected_route_index = int(route_index)
        if selected_route_index < 0 or selected_route_index >= _ROUTE_COUNT:
            raise RuntimeError(
                f"route index must be in [0, {_ROUTE_COUNT}), got {route_index}"
            )
        route = tuple(str(edge) for edge in route_edges if str(edge))
        if not route:
            raise RuntimeError("route must be non-empty")
        self._machine_episode_actions[str(vehicle_id)] = selected_route_index
        if not self._route_is_safe_for_current_lane(str(vehicle_id), route):
            raise RuntimeError(f"unsafe reroute for {vehicle_id!r}")
        self._connection().vehicle.setRoute(str(vehicle_id), list(route))

    def advance(self, steps: int) -> None:
        active_vehicle_exposure = 0.0
        connection = self._connection()
        try:
            previous_time = float(connection.simulation.getTime())
        except Exception:
            previous_time = None
        for _ in range(max(int(steps), 0)):
            self._apply_incident_for_current_time()
            connection.simulationStep()
            current_time = float(connection.simulation.getTime())
            delta = (
                float(current_time - previous_time)
                if previous_time is not None
                else 1.0
            )
            if delta <= 0.0 or delta != delta:
                delta = 1.0
            previous_time = current_time
            arrived_ids = tuple(connection.simulation.getArrivedIDList())
            active_vehicle_count = int(connection.vehicle.getIDCount()) + len(arrived_ids)
            active_vehicle_exposure += float(active_vehicle_count) * float(delta)
            for vehicle_id in arrived_ids:
                vehicle_id_str = str(vehicle_id)
                if vehicle_id_str not in self._arrival_times_seconds:
                    self._arrival_times_seconds[vehicle_id_str] = float(current_time)
                    depart = self._depart_times_seconds.get(vehicle_id_str)
                    if depart is not None:
                        self._travel_times_seconds[vehicle_id_str] = float(
                            current_time - depart
                        )
            self._apply_incident_for_current_time()
            if int(connection.simulation.getMinExpectedNumber()) <= 0:
                break
        self._active_vehicle_exposure = active_vehicle_exposure

    def get_active_vehicle_exposure(self) -> float:
        return float(self._active_vehicle_exposure)

    def is_done(self) -> bool:
        return int(self._connection().simulation.getMinExpectedNumber()) <= 0

    def travel_times_seconds(self) -> Mapping[str, float]:
        return dict(self._travel_times_seconds)

    def hdv_route_choices(self) -> Mapping[str, int]:
        return dict(self._human_episode_actions)

    def travel_time_route_choices(self) -> Mapping[str, int]:
        return {
            **self._human_episode_actions,
            **self._machine_episode_actions,
        }

    def hdv_learning_state(self) -> Mapping[str, Any]:
        return dict(self._snapshot_hdv_learning_state())

    def restore_hdv_learning_state(self, state: Mapping[str, Any]) -> None:
        self._hdv_learning_state = self._normalized_hdv_learning_state(state)
        self._route_policy_lookup = self._route_policy_by_od()

    def update_hdv_learning(
        self, travel_times_seconds: Mapping[str, float] | None = None
    ) -> int:
        travel_times = (
            dict(self._travel_times_seconds)
            if travel_times_seconds is None
            else {
                str(vehicle_id): float(seconds)
                for vehicle_id, seconds in travel_times_seconds.items()
            }
        )
        route_observations: dict[tuple[int, int], list[list[float]]] = {}
        demand_by_id = {demand.vehicle_id: demand for demand in self._demand}
        updates = 0
        for vehicle_id, action in self.travel_time_route_choices().items():
            demand = demand_by_id.get(str(vehicle_id))
            if demand is None or str(vehicle_id) not in travel_times:
                continue
            travel_time = float(travel_times[str(vehicle_id)])
            if not math.isfinite(travel_time) or travel_time < 0.0:
                raise EnvironmentContractError(
                    f"route-feedback travel time for {vehicle_id!r} must be "
                    "finite and non-negative"
                )
            od_pair = (demand.origin, demand.destination)
            route_count = len(self._routes_by_od[od_pair])
            route_index = int(action)
            if route_index < 0 or route_index >= route_count:
                raise EnvironmentContractError(
                    f"route-feedback index {route_index} is invalid for OD {od_pair}"
                )
            per_route = route_observations.setdefault(
                od_pair,
                [[] for _ in range(route_count)],
            )
            per_route[route_index].append(travel_time)
            updates += 1
        if route_observations:
            policy = self._route_policy_by_od()
            for od_pair, per_route_times in route_observations.items():
                row = policy[od_pair]
                current_costs = [float(value) for value in row["costs"]]
                observed_means = [
                    (
                        sum(values) / float(len(values))
                        if values
                        else math.nan
                    )
                    for values in per_route_times
                ]
                observed = [value for value in observed_means if math.isfinite(value)]
                if not observed:
                    continue
                fill_value = max(observed) * 1.5
                filled_times = [
                    value if math.isfinite(value) else fill_value
                    for value in observed_means
                ]
                current_probs = _softmin_route_probabilities(current_costs)
                updated_costs = [
                    cost
                    + _HDV_GAWRON_ALPHA
                    * probability
                    * (travel_time - cost)
                    for cost, probability, travel_time in zip(
                        current_costs,
                        current_probs,
                        filled_times,
                        strict=True,
                    )
                ]
                row["costs"] = updated_costs
                row["probs"] = _softmin_route_probabilities(updated_costs)
            self._hdv_learning_state = {
                "version": 1,
                "route_policy": [
                    policy[od_pair]
                    for od_pair in sorted(policy)
                ],
            }
            self._route_policy_lookup = self._route_policy_by_od()
        return updates

    def close(self) -> None:
        try:
            self._restore_incident_lanes()
        finally:
            connection = getattr(self._environment.simulator, "sumo_connection", None)
            if connection is not None:
                try:
                    connection.close()
                finally:
                    self._environment.simulator.sumo_connection = None
            self._started = False

    def _connection(self) -> Any:
        connection = getattr(self._environment.simulator, "sumo_connection", None)
        if connection is None:
            raise RuntimeError("SUMO connection is not available")
        return connection

    def _patch_sumo_commands(self) -> None:
        simulator = self._environment.simulator

        def _start(sim_self: Any) -> None:
            sim_self.runs += 1
            detailed_stats = os.path.join(
                sim_self.sumo_save_path, f"detailed_sumo_stats_{sim_self.runs}.xml"
            )
            combined_stats = os.path.join(
                sim_self.sumo_save_path, f"sumo_stats_{sim_self.runs}.xml"
            )
            sumo_cmd = self._sumo_command(
                sim_self,
                detailed_stats_file=detailed_stats,
                combined_stats_file=combined_stats,
                include_binary=True,
            )
            self._traci.start(sumo_cmd, label=sim_self.sumo_id)
            sim_self.sumo_connection = self._traci.getConnection(sim_self.sumo_id)

        def _reset(sim_self: Any) -> dict[str, float]:
            det_dict = {name: 0.0 for name in getattr(sim_self, "detectors_name", ())}
            if getattr(sim_self, "save_detectors_info", False):
                for det_name in getattr(sim_self, "detectors_name", ()):
                    det_dict[det_name] = float(
                        sim_self.sumo_connection.lanearea.getIntervalVehicleNumber(
                            f"{det_name}_det"
                        )
                    )
            sim_self.runs += 1
            detailed_stats = os.path.join(
                sim_self.sumo_save_path, f"detailed_sumo_stats_{sim_self.runs}.xml"
            )
            combined_stats = os.path.join(
                sim_self.sumo_save_path, f"sumo_stats_{sim_self.runs}.xml"
            )
            sumo_cmd = self._sumo_command(
                sim_self,
                detailed_stats_file=detailed_stats,
                combined_stats_file=combined_stats,
                include_binary=False,
            )
            sim_self.sumo_connection.load(sumo_cmd)
            sim_self.timestep = 0
            sim_self.waiting_vehicles = {}
            return det_dict

        simulator.start = types.MethodType(_start, simulator)
        simulator.reset = types.MethodType(_reset, simulator)

    def _sumo_command(
        self,
        simulator: Any,
        *,
        detailed_stats_file: str,
        combined_stats_file: str,
        include_binary: bool,
    ) -> list[str]:
        command = []
        if include_binary:
            command.append(str(simulator.sumo_type))
        command.extend(
            [
                "--seed",
                str(simulator.seed),
                "--net-file",
                str(simulator.network_file_path),
                "--additional-files",
                f"{simulator.det_xml_save_path},{simulator.rou_xml_save_path}",
                "--no-step-log",
                "true",
                "--time-to-teleport",
                "-1",
                "--statistic-output",
                combined_stats_file,
                "--tripinfo-output",
                detailed_stats_file,
            ]
        )
        command.extend(str(value) for value in self._spec.sumo_extra_args)
        return command

    def _stage_routerl_artifacts(self) -> _StagedRouteRLArtifacts:
        self._spec.records_directory.mkdir(parents=True, exist_ok=True)
        staged_agents = self._spec.records_directory / "agents.csv"
        staged_routes = self._spec.records_directory / "routes.csv"
        staged_route_xml = self._spec.records_directory / "route.rou.xml"
        rows = self._read_agents_csv(self._spec.agents_path)
        known_ids = {row["id"] for row in rows}
        fixed_ids = self._machine_ids_or_human_ids()
        if known_ids != fixed_ids:
            missing = sorted(fixed_ids - known_ids)
            extra = sorted(known_ids - fixed_ids)
            details: list[str] = []
            if missing:
                details.append(f"missing fixed ids: {', '.join(missing[:10])}")
            if extra:
                details.append(f"unstaged demand ids: {', '.join(extra[:10])}")
            raise EnvironmentContractError("; ".join(details))
        for row in rows:
            row["kind"] = "AV" if row["id"] in self._machine_ids else "Human"
        with staged_agents.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["id", "origin", "destination", "start_time", "kind"],
            )
            writer.writeheader()
            writer.writerows(rows)
        shutil.copyfile(self._spec.routes_path, staged_routes)
        shutil.copyfile(self._spec.route_xml_path, staged_route_xml)
        return _StagedRouteRLArtifacts(
            agents_csv=staged_agents,
            routes_csv=staged_routes,
            route_xml=staged_route_xml,
        )

    def _machine_ids_or_human_ids(self) -> set[str]:
        return set(self._spec.fixed_machine_ids) | set(self._spec.fixed_human_ids)

    def _read_agents_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"id", "origin", "destination", "start_time", "kind"}
            if reader.fieldnames is None or set(reader.fieldnames) != required:
                raise EnvironmentContractError(
                    "agents.csv must contain exactly id, origin, destination, start_time, kind"
                )
            rows: list[dict[str, str]] = []
            for row in reader:
                rows.append(
                    {
                        "id": str(row["id"]).strip(),
                        "origin": str(row["origin"]).strip(),
                        "destination": str(row["destination"]).strip(),
                        "start_time": str(row["start_time"]).strip(),
                        "kind": str(row["kind"]).strip(),
                    }
                )
        if not rows:
            raise EnvironmentContractError("agents.csv must contain at least one demand row")
        return rows

    def _load_routes_by_od(
        self, routes_csv_path: Path
    ) -> dict[tuple[int, int], tuple[tuple[str, ...], ...]]:
        routes_by_od: dict[tuple[int, int], list[tuple[str, ...]]] = {}
        with routes_csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"origins", "destinations", "path", "free_flow_time"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise EnvironmentContractError(
                    "routes.csv must contain origins, destinations, path, free_flow_time"
                )
            for row in reader:
                origin = int(str(row["origins"]).strip())
                destination = int(str(row["destinations"]).strip())
                path = tuple(
                    edge for edge in str(row["path"]).strip().split() if edge
                )
                if not path:
                    raise EnvironmentContractError("routes.csv cannot contain empty paths")
                routes_by_od.setdefault((origin, destination), []).append(path)
        if not routes_by_od:
            raise EnvironmentContractError("routes.csv must contain at least one route")
        return {
            key: tuple(value)
            for key, value in routes_by_od.items()
        }

    def _load_demand(self, agents_csv_path: Path) -> tuple[_DemandVehicle, ...]:
        rows = self._read_agents_csv(agents_csv_path)
        demand: list[_DemandVehicle] = []
        for row in rows:
            demand.append(
                _DemandVehicle(
                    vehicle_id=str(row["id"]),
                    origin=int(str(row["origin"])),
                    destination=int(str(row["destination"])),
                    depart_seconds=float(str(row["start_time"])),
                    kind=str(row["kind"]),
                )
            )
        return tuple(demand)

    def _normalized_hdv_learning_state(
        self, state: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not state:
            raise EnvironmentContractError("HDV route state is required")
        normalized = to_plain_state(state)
        if not isinstance(normalized, dict):
            raise EnvironmentContractError("hdv learning state must be a mapping")
        if normalized.get("version") != 1:
            raise EnvironmentContractError("HDV route state version must be 1")
        rows = normalized.get("route_policy")
        if not isinstance(rows, list):
            raise EnvironmentContractError("HDV route state requires route_policy")
        expected_ods = set(self._routes_by_od)
        policy: dict[tuple[int, int], dict[str, Any]] = {}
        for raw_row in rows:
            if not isinstance(raw_row, Mapping) or set(raw_row) != {
                "origin",
                "destination",
                "probs",
                "costs",
            }:
                raise EnvironmentContractError(
                    "HDV route_policy rows must contain origin,destination,probs,costs"
                )
            od_pair = (int(raw_row["origin"]), int(raw_row["destination"]))
            if od_pair in policy:
                raise EnvironmentContractError(
                    f"duplicate HDV route_policy row for OD {od_pair}"
                )
            if od_pair not in expected_ods:
                raise EnvironmentContractError(
                    f"HDV route_policy contains unknown OD {od_pair}"
                )
            route_count = len(self._routes_by_od[od_pair])
            if route_count != _ROUTE_COUNT:
                raise EnvironmentContractError(
                    f"mainline requires four routes for OD {od_pair}"
                )
            probs = [float(value) for value in raw_row["probs"]]
            costs = [float(value) for value in raw_row["costs"]]
            if len(probs) != route_count or len(costs) != route_count:
                raise EnvironmentContractError(
                    f"HDV route_policy width mismatch for OD {od_pair}"
                )
            if (
                any(not math.isfinite(value) or value < 0.0 for value in probs)
                or sum(probs) <= 0.0
            ):
                raise EnvironmentContractError(
                    f"HDV route probabilities are invalid for OD {od_pair}"
                )
            if any(not math.isfinite(value) or value < 0.0 for value in costs):
                raise EnvironmentContractError(
                    f"HDV route costs are invalid for OD {od_pair}"
                )
            total = sum(probs)
            policy[od_pair] = {
                "origin": od_pair[0],
                "destination": od_pair[1],
                "probs": [value / total for value in probs],
                "costs": costs,
            }
        missing = expected_ods - set(policy)
        if missing:
            raise EnvironmentContractError(
                "HDV route_policy is missing OD rows: "
                + ", ".join(str(value) for value in sorted(missing)[:10])
            )
        return {
            "version": 1,
            "route_policy": [
                policy[od_pair]
                for od_pair in sorted(policy)
            ],
        }

    def _snapshot_hdv_learning_state(self) -> dict[str, Any]:
        return dict(to_plain_state(self._hdv_learning_state))

    def _route_policy_by_od(self) -> dict[tuple[int, int], dict[str, Any]]:
        return {
            (int(row["origin"]), int(row["destination"])): {
                "origin": int(row["origin"]),
                "destination": int(row["destination"]),
                "probs": [float(value) for value in row["probs"]],
                "costs": [float(value) for value in row["costs"]],
            }
            for row in self._hdv_learning_state["route_policy"]
        }

    def _add_all_demand_vehicles(self) -> None:
        connection = self._connection()
        for demand in self._demand:
            route_choices = self._routes_by_od.get((demand.origin, demand.destination))
            if not route_choices:
                raise EnvironmentContractError(
                    f"missing route choices for OD ({demand.origin}, {demand.destination})"
                )
            route_index = self._select_initial_route_index(demand, len(route_choices))
            route_id = f"{demand.origin}_{demand.destination}_{route_index}"
            connection.vehicle.add(
                vehID=demand.vehicle_id,
                routeID=route_id,
                depart=str(int(demand.depart_seconds)),
                typeID=demand.kind,
            )
            self._depart_times_seconds[demand.vehicle_id] = float(demand.depart_seconds)

    def _select_initial_route_index(self, demand: _DemandVehicle, route_count: int) -> int:
        row = self._route_policy_lookup[(demand.origin, demand.destination)]
        route_index = int(
            self._episode_rng.choices(
                range(route_count),
                weights=row["probs"],
                k=1,
            )[0]
        )
        if demand.vehicle_id in self._human_ids:
            self._human_episode_actions[demand.vehicle_id] = int(route_index)
        return int(route_index)

    def _incident_edges(self) -> tuple[str, ...]:
        return tuple(
            part.strip()
            for part in str(self._spec.incident_edge).split(",")
            if part.strip()
        )

    def _incident_enabled(self) -> bool:
        return (
            bool(self._incident_edges())
            and float(self._spec.incident_end_seconds)
            > float(self._spec.incident_start_seconds)
            and float(self._spec.incident_speed_mps) > 0.0
        )

    def _capture_incident_lanes(self) -> None:
        self._incident_lane_speeds = {}
        self._incident_active = False
        if not self._incident_enabled():
            return
        connection = self._connection()
        for edge_id in self._incident_edges():
            lane_count = int(connection.edge.getLaneNumber(edge_id))
            if lane_count <= 0:
                raise EnvironmentContractError(f"incident edge {edge_id!r} has no lanes")
            for lane_index in range(lane_count):
                lane_id = f"{edge_id}_{lane_index}"
                self._incident_lane_speeds[lane_id] = float(
                    connection.lane.getMaxSpeed(lane_id)
                )

    def _set_incident_lane_speed(self, speed_mps: float) -> None:
        connection = self._connection()
        for lane_id in self._incident_lane_speeds:
            connection.lane.setMaxSpeed(lane_id, float(speed_mps))

    def _restore_incident_lanes(self) -> None:
        connection = getattr(self._environment.simulator, "sumo_connection", None)
        if connection is None:
            return
        for lane_id, speed in self._incident_lane_speeds.items():
            try:
                connection.lane.setMaxSpeed(lane_id, float(speed))
            except Exception:
                continue
        self._incident_active = False

    def _apply_incident_for_current_time(self) -> None:
        if not self._incident_lane_speeds:
            return
        sim_time = float(self._connection().simulation.getTime())
        should_activate = (
            float(self._spec.incident_start_seconds)
            <= sim_time
            < float(self._spec.incident_end_seconds)
        )
        if should_activate and not self._incident_active:
            self._set_incident_lane_speed(float(self._spec.incident_speed_mps))
            self._incident_active = True
        elif self._incident_active and not should_activate:
            self._restore_incident_lanes()

    def _lane_links_to_edge(self, lane_id: str, next_edge: str) -> bool:
        links = self._connection().lane.getLinks(lane_id)
        for link in links:
            next_lane = ""
            if isinstance(link, Sequence) and link:
                next_lane = str(link[0])
            else:
                next_lane = str(link)
            if next_lane and next_lane.split("_", 1)[0] == str(next_edge):
                return True
        return False

    def _edge_has_lane_link_to_edge(self, edge_id: str, next_edge: str) -> bool:
        lane_count = int(self._connection().edge.getLaneNumber(edge_id))
        for lane_index in range(lane_count):
            if self._lane_links_to_edge(f"{edge_id}_{lane_index}", next_edge):
                return True
        return False

    def _lane_remaining_distance(self, vehicle_id: str) -> float:
        connection = self._connection()
        lane_id = str(connection.vehicle.getLaneID(vehicle_id))
        lane_length = float(connection.lane.getLength(lane_id))
        lane_position = float(connection.vehicle.getLanePosition(vehicle_id))
        return float(lane_length - lane_position)

    def _route_is_safe_for_current_lane(
        self, vehicle_id: str, route: Sequence[str]
    ) -> bool:
        if not route:
            return False
        connection = self._connection()
        current_edge = str(connection.vehicle.getRoadID(vehicle_id))
        if route[0] != current_edge:
            return False
        if len(route) < 2:
            return True
        lane_id = str(connection.vehicle.getLaneID(vehicle_id))
        next_edge = str(route[1])
        if self._lane_links_to_edge(lane_id, next_edge):
            return True
        if self._lane_remaining_distance(vehicle_id) < 35.0:
            return False
        return self._edge_has_lane_link_to_edge(current_edge, next_edge)


def build_live_session_spec(
    *,
    config: Any,
    prepared_inputs: Any,
    topology: Any,
    initial_hdv_learning_state: Mapping[str, Any],
    episode_counter: int,
) -> RouteRLSessionSpec:
    if int(topology.route_count) != _ROUTE_COUNT:
        raise EnvironmentContractError(
            f"mainline topology must provide exactly {_ROUTE_COUNT} routes"
        )
    post_state = _load_json_mapping(Path(prepared_inputs.post_mutation_state))
    machine_ids = tuple(str(value) for value in post_state["machine_agent_ids"])
    human_ids = tuple(str(value) for value in post_state["human_agent_ids"])
    if not machine_ids or not human_ids:
        raise EnvironmentContractError(
            "post-mutation state must define both machine_agent_ids and human_agent_ids"
        )
    decision_horizon_seconds = (
        int(config.simulation.decision_intervals_per_episode)
        * _DECISION_INTERVAL_SECONDS
    )
    warmup_horizon_seconds = (
        int(config.simulation.warmup_horizon_seconds)
        + int(config.simulation.warmup_completion_drain_seconds)
        + int(config.simulation.warmup_completion_patience_seconds)
    )
    simulation_timesteps = max(
        decision_horizon_seconds,
        warmup_horizon_seconds,
        decision_horizon_seconds
        + int(config.simulation.training_completion_buffer_seconds),
        int(config.simulation.evaluation_horizon_seconds),
    )
    return RouteRLSessionSpec(
        network_name=str(config.inputs.network_name),
        network_directory=Path(prepared_inputs.network_xml).parent,
        records_directory=Path(config.run.output_dir) / "routerl_records",
        agents_path=Path(prepared_inputs.route_agents_csv),
        routes_path=Path(prepared_inputs.routes_csv),
        route_xml_path=Path(prepared_inputs.route_xml),
        fixed_machine_ids=machine_ids,
        fixed_human_ids=human_ids,
        simulation_timesteps=simulation_timesteps,
        seed=int(config.run.seed),
        origins=tuple(str(value) for value in topology.origins),
        destinations=tuple(str(value) for value in topology.destinations),
        initial_hdv_learning_state=dict(initial_hdv_learning_state),
        sumo_extra_args=tuple(str(value) for value in config.simulation.sumo_extra_args),
        incident_edge=str(config.simulation.incident_edge),
        incident_start_seconds=float(config.simulation.incident_start_seconds),
        incident_end_seconds=float(config.simulation.incident_end_seconds),
        incident_speed_mps=float(config.simulation.incident_speed_mps),
        episode_counter=int(episode_counter),
    )


def build_session_factory(
    *,
    config: Any,
    prepared_inputs: Any,
    topology: Any,
) -> Callable[..., RouteRLSumoSession]:
    def factory(
        *,
        initial_hdv_learning_state: Mapping[str, Any],
        episode_counter: int,
    ) -> RouteRLSumoSession:
        return RouteRLSumoSession(
            edge_ids=topology.network.edge_ids,
            spec=build_live_session_spec(
                config=config,
                prepared_inputs=prepared_inputs,
                topology=topology,
                initial_hdv_learning_state=initial_hdv_learning_state,
                episode_counter=episode_counter,
            )
        )

    return factory


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentContractError(f"cannot read JSON mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EnvironmentContractError(f"{path} must contain a JSON object")
    return value


__all__ = [
    "ActionDecision",
    "CandidatePathGenerator",
    "EnvironmentContractError",
    "RouteRLSessionSpec",
    "RouteRLSumoSession",
    "SessionClosedError",
    "build_live_session_spec",
    "build_session_factory",
]
