from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..workflow.config import MainlineConfig
from ..traffic.environment import (
    ActionDecision,
    CandidatePathGenerator,
    build_session_factory,
)
from ..models.hdv import EpisodeObservation, HdvOdRouteResponsePredictor
from .observations import (
    aggregate_planned_edge_flow,
    advance_vehicle_on_path,
    build_actor_observations,
)
from ..models.policy import (
    CentralizedLatentValueCritic,
    CentralizedVehicleValueCritic,
    VehicleRouteActor,
    capped_route_reward_matrix,
    lambda_return,
    lambda_returns_by_vehicle_ids,
)
from .replay import (
    OnlineReplayBuffer,
    OnlineReplayTransition,
    VehicleSnapshot,
    WarmupReplayBuffer,
    WarmupReplayRecord,
)
from ..traffic.routing import PreparedInputs
from ..traffic.topology import TrafficTopology
from ..models.world_model import (
    FLOW_LATENT_DIM,
    UrbanDreamerWorldModel,
    cfm_loss,
    cfm_predict_one_timestep,
    decoder_bpr_supervision_loss,
    decoder_reconstruction_loss,
    episode_t_all_supervision_loss,
    transition_decoded_supervision_loss,
)


DECISION_INTERVAL_SECONDS = 30
WARMUP_ROUTE_EPSILON = 0.4
ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE = 4
IMAGINED_ROLLOUTS_PER_PASS = 8
IMAGINED_PASSES_PER_EPISODE = 16
IMAGINED_HORIZON = 8
IMAGINATION_BATCH_SIZE = 4
ACTOR_ENTROPY_COEF = 0.002
WORLD_MODEL_INITIAL_LR = 1e-4
WORLD_MODEL_INITIAL_WEIGHT_DECAY = 1e-2
WORLD_MODEL_ONLINE_LR = 1e-5
ACTIVE_READOUT_LR = 1e-3
ACTOR_LR = 2e-4
CRITIC_LR = 3e-4
GRAD_CLIP = 10.0
IMAGINATION_GLOBAL_REWARD_SCALE = DECISION_INTERVAL_SECONDS / 60.0


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _to_plain_float_dict(value: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(item) for key, item in value.items()}


def _to_tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_to_tuple_tree(item) for item in value)
    return value


def _route_probabilities_from_state(
    topology: TrafficTopology,
    state: Mapping[str, Any],
) -> np.ndarray:
    if state.get("version") != 1:
        raise ValueError("HDV route state version must be 1")
    route_policy = state.get("route_policy")
    if not isinstance(route_policy, list):
        raise ValueError("HDV route state must contain route_policy")
    route_count = int(topology.route_count)
    probs_by_od: dict[tuple[int, int], np.ndarray] = {}
    for row in route_policy:
        if not isinstance(row, Mapping) or set(row) != {
            "origin",
            "destination",
            "probs",
            "costs",
        }:
            raise ValueError(
                "HDV route_policy rows must contain "
                "origin,destination,probs,costs"
            )
        od_pair = (int(row["origin"]), int(row["destination"]))
        if od_pair in probs_by_od:
            raise ValueError(f"duplicate HDV route_policy row for OD {od_pair}")
        probs = np.asarray(row["probs"], dtype=np.float32).reshape(-1)
        costs = np.asarray(row["costs"], dtype=np.float32).reshape(-1)
        if probs.shape != (route_count,) or costs.shape != (route_count,):
            raise ValueError(f"HDV route_policy width mismatch for OD {od_pair}")
        if (
            not np.isfinite(probs).all()
            or np.any(probs < 0.0)
            or float(probs.sum()) <= 0.0
        ):
            raise ValueError(f"invalid HDV route probabilities for OD {od_pair}")
        if not np.isfinite(costs).all() or np.any(costs < 0.0):
            raise ValueError(f"invalid HDV route costs for OD {od_pair}")
        probs_by_od[od_pair] = probs
    probability_rows: list[np.ndarray] = []
    for od_pair in topology.od_pairs:
        key = (int(od_pair[0]), int(od_pair[1]))
        if key not in probs_by_od:
            raise ValueError(f"HDV route_policy is missing OD {key}")
        row = probs_by_od[key]
        probability_rows.append((row / float(row.sum())).astype(np.float32))
    if len(probs_by_od) != len(topology.od_pairs):
        raise ValueError("HDV route_policy contains rows outside the topology")
    return np.stack(probability_rows, axis=0)


def _edge_distribution_from_route_probabilities(
    topology: TrafficTopology,
    route_probabilities: np.ndarray,
) -> np.ndarray:
    route_probabilities = np.asarray(route_probabilities, dtype=np.float32)
    expected_shape = (len(topology.od_pairs), int(topology.route_count))
    if route_probabilities.shape != expected_shape:
        raise ValueError(
            "HDV route probability shape mismatch: "
            f"{route_probabilities.shape} vs {expected_shape}"
        )
    n_edges = topology.network.edge_count
    edge_mass = np.zeros((n_edges,), dtype=np.float32)
    total_humans = float(np.maximum(topology.od_human_counts, 0.0).sum())
    if total_humans <= 0.0:
        raise ValueError("mainline topology must contain human vehicles")
    for od_index, route_row in enumerate(topology.route_paths):
        od_weight = float(max(topology.od_human_counts[od_index], 0.0)) / total_humans
        for route_index, route in enumerate(route_row):
            mass = od_weight * float(route_probabilities[od_index, route_index])
            if mass <= 0.0:
                continue
            np.add.at(edge_mass, np.asarray(route, dtype=np.int64), mass)
    total = float(edge_mass.sum())
    if total <= 1e-12:
        raise ValueError("HDV route policy produced an empty edge distribution")
    return (edge_mass / total).astype(np.float32)


def _edge_distribution_from_route_choices(
    topology: TrafficTopology,
    route_choices: Mapping[str, int],
) -> np.ndarray:
    n_edges = topology.network.edge_count
    edge_mass = np.zeros((n_edges,), dtype=np.float32)
    human_by_id = {
        str(vehicle.vehicle_id): vehicle
        for vehicle in topology.vehicles
        if not vehicle.is_automated
    }
    if set(map(str, route_choices)) != set(human_by_id):
        raise ValueError("HDV route choices must exactly cover all human vehicles")
    for vehicle_id, vehicle in human_by_id.items():
        route_index = int(route_choices[vehicle_id])
        od_index = topology.od_to_index[(int(vehicle.origin_index), int(vehicle.destination_index))]
        route_row = topology.route_paths[od_index]
        if not 0 <= route_index < len(route_row):
            raise ValueError(
                f"HDV route choice {route_index} is invalid for {vehicle_id!r}"
            )
        np.add.at(
            edge_mass,
            np.asarray(route_row[route_index], dtype=np.int64),
            1.0,
        )
    total = float(edge_mass.sum())
    if total <= 0.0:
        raise ValueError("HDV route choices produced an empty edge distribution")
    return (edge_mass / total).astype(np.float32)


def _route_cost_seconds(candidate_paths: Sequence[Sequence[int]], edge_travel_seconds: np.ndarray) -> np.ndarray:
    costs = np.zeros((len(candidate_paths),), dtype=np.float32)
    for route_index, path in enumerate(candidate_paths):
        indices = np.asarray(path, dtype=np.int64)
        costs[route_index] = float(
            np.maximum(edge_travel_seconds[indices], 0.0).sum()
        )
    return costs


def _normalize_edge_vector(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    total = float(np.maximum(array, 0.0).sum())
    if total <= 1e-12:
        return np.zeros_like(array)
    return (np.maximum(array, 0.0) / total).astype(np.float32)


def _stack_metric_dict(metric_dicts: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    result: dict[str, list[float]] = {}
    for metric_dict in metric_dicts:
        for key, value in metric_dict.items():
            result.setdefault(str(key), []).append(float(value))
    return {key: float(np.mean(values)) for key, values in result.items()}


def _merge_float_metrics(base: dict[str, float], updates: Mapping[str, float]) -> None:
    for key, value in updates.items():
        base[str(key)] = float(value)


class _IncompleteWarmupAttempt(RuntimeError):
    """Signals a discarded warmup attempt without aborting collection."""


@dataclass(frozen=True)
class _CollectedEpisode:
    episode: int
    global_episode: int
    total_travel_minutes: float
    warmup_records: tuple[WarmupReplayRecord, ...]
    online_transitions: tuple[OnlineReplayTransition, ...]
    hdv_observation: EpisodeObservation | None
    metrics: dict[str, float]
    final_hdv_state: dict[str, Any]


@dataclass(frozen=True)
class _VehicleSeed:
    vehicle_id: str
    current_edge: int
    destination_edge: int
    departure_step: int


class MainlineTrainer:
    def __init__(
        self,
        *,
        config: MainlineConfig,
        prepared_inputs: PreparedInputs,
        topology: TrafficTopology,
        _session_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.prepared_inputs = prepared_inputs
        self.topology = topology
        if int(self.topology.route_count) != 4:
            raise ValueError("MainlineTrainer requires exactly four candidate routes")
        if int(self.config.resources.world_model_initialization_sequence_length) < 2:
            raise ValueError(
                "world_model_initialization_sequence_length must be >= 2 for the mainline warmup replay"
            )

        self.device = self._resolve_device(self.config.run.device)
        self._seed_everything(int(self.config.run.seed))
        self.rng = np.random.default_rng(int(self.config.run.seed))

        network = self.topology.network
        self.edge_ids = tuple(str(edge_id) for edge_id in network.edge_ids)
        self.edge_to_index = dict(network.edge_to_index)
        self.edge_count = int(network.edge_count)
        self.total_vehicle_count = int(len(self.topology.vehicles))
        self.total_automated = int(sum(1 for vehicle in self.topology.vehicles if vehicle.is_automated))
        self.total_human = int(sum(1 for vehicle in self.topology.vehicles if not vehicle.is_automated))
        self._vehicle_by_id = {str(vehicle.vehicle_id): vehicle for vehicle in self.topology.vehicles}
        self._machine_vehicles = tuple(vehicle for vehicle in self.topology.vehicles if vehicle.is_automated)
        self._human_vehicles = tuple(vehicle for vehicle in self.topology.vehicles if not vehicle.is_automated)

        adjacency = torch.as_tensor(network.adjacency, dtype=torch.float32, device=self.device)
        static_features = torch.as_tensor(network.static_features, dtype=torch.float32, device=self.device)
        self._lengths = torch.as_tensor(network.lengths_m, dtype=torch.float32, device=self.device)
        self._speed_limits = torch.as_tensor(network.speed_limits_mps, dtype=torch.float32, device=self.device)
        self._capacities = torch.as_tensor(network.capacities, dtype=torch.float32, device=self.device)
        self._free_flow_seconds = np.asarray(network.free_flow_travel_seconds, dtype=np.float32).reshape(-1)

        self.world_model = UrbanDreamerWorldModel(adjacency=adjacency, static_features=static_features).to(self.device)
        self.actor = VehicleRouteActor().to(self.device)
        self.vehicle_critic = CentralizedVehicleValueCritic().to(self.device)
        self.latent_critic = CentralizedLatentValueCritic().to(self.device)

        self.world_model_optimizer_initial = torch.optim.AdamW(
            list(self.world_model.flow_projector.parameters()) + list(self.world_model.dynamics.parameters()),
            lr=WORLD_MODEL_INITIAL_LR,
            weight_decay=WORLD_MODEL_INITIAL_WEIGHT_DECAY,
        )
        self.world_model_optimizer_online = torch.optim.Adam(
            list(self.world_model.flow_projector.parameters()) + list(self.world_model.dynamics.parameters()),
            lr=WORLD_MODEL_ONLINE_LR,
        )
        self.active_fraction_optimizer = torch.optim.Adam(
            self.world_model.active_vehicle_fraction_readout.parameters(),
            lr=ACTIVE_READOUT_LR,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.vehicle_critic_optimizer = torch.optim.Adam(self.vehicle_critic.parameters(), lr=CRITIC_LR)
        self.latent_critic_optimizer = torch.optim.Adam(self.latent_critic.parameters(), lr=CRITIC_LR)

        self.warmup_replay = WarmupReplayBuffer(
            sequence_length=int(self.config.resources.world_model_initialization_sequence_length)
        )
        self.online_replay = OnlineReplayBuffer()

        self.path_generator = CandidatePathGenerator(
            adjacency=self.topology.network.adjacency_list,
            free_flow_costs=self.topology.network.free_flow_travel_seconds,
        )
        self._current_hdv_learning_state = _read_json_mapping(Path(self.prepared_inputs.initial_hdv_state))
        self.hdv_predictor = HdvOdRouteResponsePredictor(
            checkpoint_path=self.prepared_inputs.hdv_response_checkpoint,
            topology=self.topology,
            initial_hdv_state_path=self.prepared_inputs.initial_hdv_state,
            device=self.device,
        )
        self._session_factory = (
            _session_factory
            if _session_factory is not None
            else build_session_factory(
                config=self.config,
                prepared_inputs=self.prepared_inputs,
                topology=self.topology,
            )
        )
        self._next_environment_episode = int(self.prepared_inputs.mutation_episode_index)
        self.metrics: dict[str, float] = {
            "warmup_episodes_completed": 0.0,
            "warmup_attempts_discarded": 0.0,
            "world_model_initialization_steps_completed": 0.0,
            "online_episodes_completed": 0.0,
            "evaluation_episodes_completed": 0.0,
            "evaluation_t_all_sum": 0.0,
            "evaluation_t_all_mean": 0.0,
            "world_model_loss_last": 0.0,
            "active_fraction_loss_last": 0.0,
            "actor_loss_last": 0.0,
            "vehicle_critic_loss_last": 0.0,
            "latent_critic_loss_last": 0.0,
            "t_all_last": 0.0,
        }

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return device

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def collect_warmup_episode(self, episode_index: int) -> dict[str, Any]:
        while True:
            try:
                collected = self._run_episode(
                    mode="warmup",
                    episode_index=int(episode_index),
                )
                break
            except _IncompleteWarmupAttempt:
                self.metrics["warmup_attempts_discarded"] = (
                    self.metrics.get("warmup_attempts_discarded", 0.0) + 1.0
                )
        if not self.warmup_replay.append_episode(
            collected.warmup_records,
            total_travel_minutes=collected.total_travel_minutes,
        ):
            raise RuntimeError("warmup episode did not satisfy fixed replay completeness")
        self.metrics["warmup_episodes_completed"] += 1.0
        self.metrics["t_all_last"] = float(collected.total_travel_minutes)
        _merge_float_metrics(self.metrics, collected.metrics)
        return {
            "complete": True,
            "episode": collected.episode,
            "global_episode": collected.global_episode,
            "warmup_records": len(collected.warmup_records),
            "t_all": float(collected.total_travel_minutes),
            **collected.metrics,
        }

    def initialize_world_model_step(self, step_index: int) -> dict[str, Any]:
        self.world_model.train()
        self.world_model.set_initial_chunk_profile()
        losses: list[torch.Tensor] = []
        metric_rows: list[dict[str, float]] = []
        configured_accumulation_steps = int(
            self.config.resources.world_model_gradient_accumulation_steps
        )
        completed_microbatches = int(step_index) * configured_accumulation_steps
        remaining_microbatches = (
            int(self.config.run.world_model_initialization_steps)
            - completed_microbatches
        )
        if remaining_microbatches <= 0:
            raise IndexError("world-model initialization step exceeds configured budget")
        accumulation_steps = min(
            configured_accumulation_steps,
            remaining_microbatches,
        )
        self.world_model_optimizer_initial.zero_grad(set_to_none=True)
        for _ in range(accumulation_steps):
            batch = self.warmup_replay.sample_batch(
                batch_size=int(self.config.resources.world_model_initialization_batch_size),
                rng=self.rng,
            )
            batch_metrics = self._initial_world_model_loss(batch)
            loss = batch_metrics.pop("loss_tensor")
            losses.append(loss.detach())
            metric_rows.append({key: float(value) for key, value in batch_metrics.items()})
            (loss / float(accumulation_steps)).backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.world_model.flow_projector.parameters()) + list(self.world_model.dynamics.parameters()),
            GRAD_CLIP,
        )
        self.world_model_optimizer_initial.step()
        mean_metrics = _stack_metric_dict(metric_rows)
        mean_loss = float(torch.stack(losses).mean().cpu().item())
        self.metrics["world_model_initialization_steps_completed"] += float(
            accumulation_steps
        )
        self.metrics["world_model_loss_last"] = mean_loss
        _merge_float_metrics(self.metrics, mean_metrics)
        return {
            "loss": mean_loss,
            "microbatches": accumulation_steps,
            **mean_metrics,
        }

    def train_online_episode(self, episode_index: int) -> dict[str, Any]:
        collected = self._run_episode(mode="online", episode_index=int(episode_index))
        if not self.online_replay.append_episode(
            collected.online_transitions,
            total_travel_minutes=collected.total_travel_minutes,
        ):
            raise RuntimeError("online episode did not satisfy fixed replay completeness")

        wm_metric_rows: list[dict[str, float]] = []
        for _ in range(ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE):
            wm_metric_rows.append(self._online_world_model_update())
        imagination_rows = [self._train_imagination_pass() for _ in range(IMAGINED_PASSES_PER_EPISODE)]
        merged = {}
        _merge_float_metrics(merged, collected.metrics)
        _merge_float_metrics(merged, _stack_metric_dict(wm_metric_rows))
        _merge_float_metrics(merged, _stack_metric_dict(imagination_rows))
        self.metrics["online_episodes_completed"] += 1.0
        self.metrics["t_all_last"] = float(collected.total_travel_minutes)
        _merge_float_metrics(self.metrics, merged)
        return {
            "complete": True,
            "environment_complete": bool(
                collected.metrics.get("online_completion_rate", 1.0) >= 1.0
            ),
            "episode": collected.episode,
            "global_episode": collected.global_episode,
            "online_transitions": len(collected.online_transitions),
            "t_all": float(collected.total_travel_minutes),
            **merged,
        }

    def evaluate_episode(self, episode_index: int) -> dict[str, Any]:
        collected = self._run_episode(mode="evaluation", episode_index=int(episode_index))
        completed = self.metrics["evaluation_episodes_completed"] + 1.0
        total = (
            self.metrics["evaluation_t_all_sum"]
            + float(collected.total_travel_minutes)
        )
        self.metrics["evaluation_episodes_completed"] = completed
        self.metrics["evaluation_t_all_sum"] = total
        self.metrics["evaluation_t_all_mean"] = total / completed
        self.metrics["t_all_last"] = float(collected.total_travel_minutes)
        _merge_float_metrics(self.metrics, collected.metrics)
        return {
            "complete": True,
            "episode": collected.episode,
            "global_episode": collected.global_episode,
            "t_all": float(collected.total_travel_minutes),
            **collected.metrics,
        }

    def _run_episode(self, *, mode: str, episode_index: int) -> _CollectedEpisode:
        if mode not in {"warmup", "online", "evaluation"}:
            raise ValueError(f"unsupported collection mode {mode!r}")
        global_episode = self._next_environment_episode
        session = self._session_factory(
            initial_hdv_learning_state=self._current_hdv_learning_state,
            episode_counter=global_episode,
        )
        wall_started_at = time.monotonic()
        try:
            state = dict(session.reset())
            pre_hdv_state = copy.deepcopy(dict(session.hdv_learning_state()))
            predicted_hdv_edge = self._initial_hdv_edge_context(
                mode=mode,
                global_episode=global_episode,
                pre_hdv_state=pre_hdv_state,
            )

            warmup_frames: list[dict[str, Any]] = []
            online_frames: list[dict[str, Any]] = []
            decision_metric_rows: list[dict[str, float]] = []
            warmup_assignments: dict[str, ActionDecision] = {}
            step_limit = self._episode_step_limit(mode)
            for step_index in range(step_limit):
                if (
                    time.monotonic() - wall_started_at
                    > float(self.config.simulation.episode_wall_timeout_seconds)
                ):
                    raise TimeoutError(
                        f"{mode} episode exceeded "
                        "episode_wall_timeout_seconds"
                    )
                if session.is_done():
                    break
                flow = self._extract_flow(state)
                if mode == "warmup":
                    (
                        action_decisions,
                        route_updates,
                    ) = self._select_warmup_actions(
                        state,
                        flow,
                        assignments=warmup_assignments,
                    )
                else:
                    action_decisions, decision_metrics = self._select_policy_actions(
                        state=state,
                        flow=flow,
                        predicted_hdv_edge=predicted_hdv_edge,
                        step_index=step_index,
                        deterministic=mode == "evaluation",
                    )
                    decision_metric_rows.append(decision_metrics)
                    route_updates = action_decisions
                if route_updates:
                    session.set_routes_safely(route_updates)
                departure = self._departure_edge_vector(global_step=step_index)
                context = self._step_context(state=state, step_index=step_index)
                machine_snapshot = self._build_vehicle_snapshot(state=state)
                planned_action = aggregate_planned_edge_flow(
                    list(action_decisions.values()),
                    edge_count=self.edge_count,
                )

                session.advance(DECISION_INTERVAL_SECONDS)
                next_state = dict(session.get_traffic_state())
                terminal = session.is_done()
                next_flow = self._extract_flow(next_state)
                if mode == "warmup":
                    warmup_frames.append(
                        {
                            "episode": int(episode_index),
                            "step_index": step_index,
                            "flow": flow,
                            "planned_action": planned_action,
                            "departure": departure,
                            "context": context,
                            "terminal": terminal,
                        }
                    )
                else:
                    online_frames.append(
                        {
                            "episode": int(episode_index),
                            "step_index": step_index,
                            "flow": flow,
                            "action": planned_action,
                            "next_flow": next_flow,
                            "departure": departure,
                            "context": context,
                            "vehicles": machine_snapshot,
                            "active_vehicle_seconds": float(
                                session.last_active_vehicle_exposure
                            ),
                            "transition_seconds": self._transition_seconds(
                                state,
                                next_state,
                            ),
                            "terminal": terminal,
                        }
                    )
                if terminal:
                    state = next_state
                    break
                state = next_state
            episode_complete = bool(session.is_done())
            if not episode_complete:
                if mode == "warmup":
                    self._next_environment_episode += 1
                    raise _IncompleteWarmupAttempt(
                        "warmup attempt exceeded its fixed completion budget"
                    )
                if mode == "evaluation":
                    raise RuntimeError(
                        "evaluation episode exceeded its fixed completion budget"
                    )
                if not online_frames:
                    raise RuntimeError(
                        "online episode produced no transitions within its fixed budget"
                    )

            travel_times = _to_plain_float_dict(session.travel_times_seconds())
            arrived_vehicle_count = len(travel_times)
            if episode_complete and arrived_vehicle_count != self.total_vehicle_count:
                raise RuntimeError(
                    f"{mode} episode finished without complete travel times: "
                    f"{arrived_vehicle_count} / {self.total_vehicle_count}"
                )
            if arrived_vehicle_count > self.total_vehicle_count:
                raise RuntimeError(
                    f"{mode} episode reported too many travel times: "
                    f"{arrived_vehicle_count} / {self.total_vehicle_count}"
                )
            if episode_complete:
                total_travel_minutes = (
                    float(np.mean(list(travel_times.values()))) / 60.0
                )
            else:
                simulation_time = float(state.get("simulation_time", 0.0))
                horizon_penalty_seconds = max(
                    simulation_time,
                    float(step_limit * DECISION_INTERVAL_SECONDS),
                    1.0,
                )
                unfinished_vehicle_count = (
                    self.total_vehicle_count - arrived_vehicle_count
                )
                total_travel_minutes = (
                    float(sum(travel_times.values()))
                    + float(unfinished_vehicle_count) * horizon_penalty_seconds
                ) / float(self.total_vehicle_count) / 60.0
            hdv_route_choices = session.hdv_route_choices()
            travel_time_route_choices = {
                str(vehicle_id): int(route_index)
                for vehicle_id, route_index in (
                    session.travel_time_route_choices().items()
                )
                if str(vehicle_id) in travel_times
            }
            actual_hdv_edge = self._final_hdv_edge_distribution(
                route_choices=hdv_route_choices,
            )

            post_hdv_state = copy.deepcopy(pre_hdv_state)
            hdv_metrics: dict[str, float] = {}
            if mode != "warmup":
                self._update_session_hdv_learning(session, travel_times)
                post_hdv_state = copy.deepcopy(dict(session.hdv_learning_state()))
                observation = EpisodeObservation(
                    episode=global_episode,
                    hdv_route_choices={
                        str(key): int(value)
                        for key, value in hdv_route_choices.items()
                    },
                    travel_time_route_choices={
                        str(key): int(value)
                        for key, value in travel_time_route_choices.items()
                    },
                    travel_times_seconds=travel_times,
                    pre_state=pre_hdv_state,
                    post_state=post_hdv_state,
                )
                hdv_metrics = self.hdv_predictor.observe_episode(
                    observation,
                    collect_online_sample=(mode == "online"),
                )
                if mode == "online":
                    _merge_float_metrics(hdv_metrics, self.hdv_predictor.update_online())
                hdv_observation: EpisodeObservation | None = observation
            else:
                hdv_observation = None

            warmup_records = tuple(
                WarmupReplayRecord(
                    **frame,
                    hdv_context=actual_hdv_edge,
                    hdv_target=actual_hdv_edge,
                )
                for frame in warmup_frames
            )
            online_transitions = tuple(
                OnlineReplayTransition(
                    **frame,
                    hdv_context=predicted_hdv_edge,
                    hdv_target=actual_hdv_edge,
                )
                for frame in online_frames
            )
            self._current_hdv_learning_state = post_hdv_state
            self._next_environment_episode += 1
            phase_metrics = {
                f"{mode}_t_all": float(total_travel_minutes),
                f"{mode}_steps": float(len(warmup_records) if mode == "warmup" else len(online_transitions)),
                f"{mode}_completion_rate": (
                    float(arrived_vehicle_count)
                    / max(float(self.total_vehicle_count), 1.0)
                ),
                f"{mode}_arrived_vehicles": float(arrived_vehicle_count),
            }
            _merge_float_metrics(
                phase_metrics,
                _stack_metric_dict(decision_metric_rows),
            )
            _merge_float_metrics(phase_metrics, hdv_metrics)
            return _CollectedEpisode(
                episode=int(episode_index),
                global_episode=int(global_episode),
                total_travel_minutes=float(total_travel_minutes),
                warmup_records=warmup_records,
                online_transitions=online_transitions,
                hdv_observation=hdv_observation,
                metrics=phase_metrics,
                final_hdv_state=copy.deepcopy(post_hdv_state),
            )
        finally:
            session.close()

    def _initial_hdv_edge_context(
        self,
        *,
        mode: str,
        global_episode: int,
        pre_hdv_state: Mapping[str, Any],
    ) -> np.ndarray:
        if mode == "warmup":
            return self._edge_distribution_from_hdv_state(pre_hdv_state)
        predicted, metrics = self.hdv_predictor.predict_for_next_episode(global_episode)
        _merge_float_metrics(self.metrics, metrics)
        return np.asarray(predicted, dtype=np.float32).reshape(self.edge_count)

    def _edge_distribution_from_hdv_state(self, state: Mapping[str, Any]) -> np.ndarray:
        route_probabilities = _route_probabilities_from_state(
            self.topology,
            state,
        )
        return _edge_distribution_from_route_probabilities(
            self.topology,
            route_probabilities,
        )

    def _final_hdv_edge_distribution(
        self,
        *,
        route_choices: Mapping[str, int],
    ) -> np.ndarray:
        return _edge_distribution_from_route_choices(
            self.topology,
            route_choices,
        )

    def _update_session_hdv_learning(
        self,
        session: Any,
        travel_times: Mapping[str, float],
    ) -> None:
        session.update_hdv_learning(travel_times)

    def _episode_step_limit(self, mode: str) -> int:
        if mode == "warmup":
            seconds = (
                int(self.config.simulation.warmup_horizon_seconds)
                + int(self.config.simulation.warmup_completion_drain_seconds)
                + int(self.config.simulation.warmup_completion_patience_seconds)
            )
        elif mode == "online":
            seconds = (
                int(self.config.simulation.decision_intervals_per_episode)
                * DECISION_INTERVAL_SECONDS
                + int(self.config.simulation.training_completion_buffer_seconds)
            )
        else:
            seconds = int(self.config.simulation.evaluation_horizon_seconds)
        return max(int(math.ceil(seconds / float(DECISION_INTERVAL_SECONDS))), 1)

    def _extract_flow(self, state: Mapping[str, Any]) -> np.ndarray:
        edges = state.get("edges")
        if not isinstance(edges, Mapping):
            raise RuntimeError("traffic state is missing edge measurements")
        return self.topology.network.extract_flow(edges)

    @staticmethod
    def _transition_seconds(
        state: Mapping[str, Any],
        next_state: Mapping[str, Any],
    ) -> float:
        try:
            duration = float(next_state["simulation_time"]) - float(
                state["simulation_time"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "traffic states must contain numeric simulation_time values"
            ) from exc
        if not math.isfinite(duration) or duration <= 0.0:
            raise RuntimeError(
                "traffic simulation_time must increase across a transition"
            )
        return duration

    def _episode_progress(self, step_index: int) -> float:
        return float(
            min(
                step_index
                / max(
                    int(self.config.simulation.decision_intervals_per_episode),
                    1,
                ),
                1.0,
            )
        )

    def _build_vehicle_snapshot(
        self,
        *,
        state: Mapping[str, Any],
    ) -> VehicleSnapshot:
        vehicles_state = state.get("vehicles", {})
        if not isinstance(vehicles_state, Mapping):
            raise RuntimeError("traffic state vehicles must be a mapping")
        try:
            simulation_time = float(state["simulation_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "traffic state must contain numeric simulation_time"
            ) from exc
        vehicle_ids: list[str] = []
        current_edges: list[int] = []
        destination_edges: list[int] = []
        departure_steps: list[int] = []
        for vehicle in self._machine_vehicles:
            runtime = vehicles_state.get(str(vehicle.vehicle_id))
            if isinstance(runtime, Mapping):
                current_edge = self.edge_to_index.get(
                    str(runtime.get("edge", ""))
                )
                if current_edge is None:
                    continue
            elif float(vehicle.departure_second) >= simulation_time:
                current_edge = self.edge_to_index[
                    str(self.topology.origins[int(vehicle.origin_index)])
                ]
            else:
                continue
            vehicle_ids.append(str(vehicle.vehicle_id))
            destination_edge = self.edge_to_index[str(self.topology.destinations[int(vehicle.destination_index)])]
            current_edges.append(int(current_edge))
            destination_edges.append(int(destination_edge))
            departure_steps.append(int(vehicle.departure_second // DECISION_INTERVAL_SECONDS))
        return VehicleSnapshot(
            vehicle_ids=tuple(vehicle_ids),
            current_edges=torch.tensor(current_edges, dtype=torch.int64),
            destination_edges=torch.tensor(destination_edges, dtype=torch.int64),
            departure_steps=torch.tensor(departure_steps, dtype=torch.int64),
        )

    def _departure_edge_vector(self, *, global_step: int) -> np.ndarray:
        vector = np.zeros((self.edge_count,), dtype=np.float32)
        start_second = global_step * DECISION_INTERVAL_SECONDS
        end_second = start_second + DECISION_INTERVAL_SECONDS
        for vehicle in self.topology.vehicles:
            if start_second <= int(vehicle.departure_second) < end_second:
                origin_edge = self.edge_to_index[str(self.topology.origins[int(vehicle.origin_index)])]
                vector[origin_edge] += 1.0
        return _normalize_edge_vector(vector)

    def _step_context(self, *, state: Mapping[str, Any], step_index: int) -> np.ndarray:
        vehicles = state.get("vehicles", {})
        if not isinstance(vehicles, Mapping):
            vehicles = {}
        active_av = 0
        active_hdv = 0
        for vehicle_id in vehicles:
            vehicle = self._vehicle_by_id.get(str(vehicle_id))
            if vehicle is None:
                continue
            if vehicle.is_automated:
                active_av += 1
            else:
                active_hdv += 1
        return np.asarray(
            [
                self._episode_progress(step_index),
                float(active_av) / max(float(self.total_automated), 1.0),
                float(active_hdv) / max(float(self.total_human), 1.0),
            ],
            dtype=np.float32,
        )

    def _select_warmup_actions(
        self,
        state: Mapping[str, Any],
        flow: np.ndarray,
        *,
        assignments: dict[str, ActionDecision],
    ) -> tuple[dict[str, ActionDecision], dict[str, ActionDecision]]:
        vehicles = self._active_machine_vehicle_ids(state)
        decisions: dict[str, ActionDecision] = {}
        new_assignments: dict[str, ActionDecision] = {}
        speed_ratio = np.clip(
            np.asarray(flow[:, 1], dtype=np.float32),
            0.05,
            1.5,
        )
        edge_travel_seconds = self._free_flow_seconds / speed_ratio
        for vehicle_id in vehicles:
            assigned = assignments.get(vehicle_id)
            if assigned is not None:
                decisions[vehicle_id] = assigned
                continue
            candidate_paths = self._candidate_paths_for_vehicle(vehicle_id=vehicle_id, state=state)
            route_costs = _route_cost_seconds(
                candidate_paths,
                edge_travel_seconds,
            )
            if float(self.rng.random()) < WARMUP_ROUTE_EPSILON:
                selected_route_index = int(
                    self.rng.integers(0, len(candidate_paths))
                )
            else:
                selected_route_index = int(route_costs.argmin())
            decision = ActionDecision.build(
                route_index=selected_route_index,
                candidate_paths=candidate_paths,
                edge_order=tuple(range(self.edge_count)),
                edge_travel_times=edge_travel_seconds,
                decision_horizon_seconds=DECISION_INTERVAL_SECONDS,
            )
            assignments[vehicle_id] = decision
            decisions[vehicle_id] = decision
            new_assignments[vehicle_id] = decision
        return decisions, new_assignments

    def _select_policy_actions(
        self,
        *,
        state: Mapping[str, Any],
        flow: np.ndarray,
        predicted_hdv_edge: np.ndarray,
        step_index: int,
        deterministic: bool = False,
    ) -> tuple[dict[str, ActionDecision], dict[str, float]]:
        with torch.no_grad():
            flow_tensor = torch.as_tensor(flow, dtype=torch.float32, device=self.device).unsqueeze(0)
            latent = self.world_model.encode(flow_tensor).squeeze(0)
            decoded = self.world_model.decode(latent).cpu().numpy()
        volume_norm = np.asarray(decoded[:, 2], dtype=np.float32)
        speed_ratio = np.clip(np.asarray(decoded[:, 1], dtype=np.float32), 0.05, 1.5)
        edge_travel_seconds = self._free_flow_seconds / speed_ratio
        bpr_minutes = (
            self._free_flow_seconds
            * (1.0 + 0.15 * np.power(np.maximum(volume_norm, 0.0), 4.0))
            / 60.0
        ).astype(np.float32)
        policy_edge_features = np.stack([volume_norm, bpr_minutes], axis=-1).astype(np.float32)

        vehicles = self._active_machine_vehicle_ids(state)
        assigned_route_counts = np.zeros((4,), dtype=np.float32)
        assigned_edge_pressure = np.zeros((self.edge_count,), dtype=np.float32)
        decisions: dict[str, ActionDecision] = {}
        route_delta_sum = 0.0
        entropy_sum = 0.0
        context_progress = self._episode_progress(step_index)
        active_fraction = float(len(vehicles)) / max(float(self.total_automated), 1.0)
        for vehicle_id in vehicles:
            candidate_paths = self._candidate_paths_for_vehicle(vehicle_id=vehicle_id, state=state)
            observations = build_actor_observations(
                policy_edge_features=policy_edge_features,
                candidate_paths=[candidate_paths],
                edge_travel_seconds=edge_travel_seconds,
                assigned_route_counts=assigned_route_counts,
                assigned_edge_pressure=assigned_edge_pressure,
                episode_progress=context_progress,
                active_av_fraction=active_fraction,
            )
            obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            selection = self.actor.sample_action(
                obs_tensor,
                deterministic=deterministic,
            )
            selected_route_index = int(selection.route_index.squeeze(0).item())
            entropy_sum += float(selection.entropy.mean().detach().cpu().item())
            decision = ActionDecision.build(
                route_index=selected_route_index,
                candidate_paths=candidate_paths,
                edge_order=tuple(range(self.edge_count)),
                edge_travel_times=edge_travel_seconds,
                decision_horizon_seconds=DECISION_INTERVAL_SECONDS,
            )
            decisions[vehicle_id] = decision
            assigned_route_counts[selected_route_index] += 1.0
            assigned_edge_pressure += np.asarray(decision.planned_edge_flow, dtype=np.float32)
            route_costs = _route_cost_seconds(candidate_paths, edge_travel_seconds)
            route_delta_sum += float(
                route_costs[selected_route_index] - route_costs.min()
            )
        metrics = {
            "policy_route_delta_seconds": route_delta_sum / max(float(len(vehicles)), 1.0),
            "policy_entropy": entropy_sum / max(float(len(vehicles)), 1.0),
            "predicted_hdv_edge_mass": float(np.maximum(predicted_hdv_edge, 0.0).sum()),
        }
        return decisions, metrics

    def _active_machine_vehicle_ids(self, state: Mapping[str, Any]) -> list[str]:
        vehicles = state.get("vehicles", {})
        if not isinstance(vehicles, Mapping):
            return []
        active: list[str] = []
        for vehicle_id in vehicles:
            vehicle = self._vehicle_by_id.get(str(vehicle_id))
            if vehicle is None or not vehicle.is_automated:
                continue
            edge_id = str(vehicles[vehicle_id].get("edge", "")) if isinstance(vehicles[vehicle_id], Mapping) else ""
            if edge_id in self.edge_to_index:
                active.append(str(vehicle_id))
        return active

    def _candidate_paths_for_vehicle(
        self,
        *,
        vehicle_id: str,
        state: Mapping[str, Any],
    ) -> tuple[tuple[int, ...], ...]:
        vehicle = self._vehicle_by_id[str(vehicle_id)]
        vehicles = state.get("vehicles", {})
        runtime = vehicles.get(str(vehicle_id), {}) if isinstance(vehicles, Mapping) else {}
        current_edge_id = (
            str(runtime.get("edge", ""))
            if isinstance(runtime, Mapping)
            else ""
        )
        if current_edge_id not in self.edge_to_index:
            current_edge_id = str(self.topology.origins[int(vehicle.origin_index)])
        destination_edge_id = str(self.topology.destinations[int(vehicle.destination_index)])
        return self.path_generator.generate_paths(
            self.edge_to_index[current_edge_id],
            self.edge_to_index[destination_edge_id],
        )

    def _initial_world_model_loss(self, batch: Any) -> dict[str, Any]:
        flow = batch.flow.to(device=self.device, dtype=torch.float32)
        z_sequence = self.world_model.encode(flow.view(-1, self.edge_count, flow.shape[-1])).view(
            flow.shape[0],
            flow.shape[1],
            self.edge_count,
            FLOW_LATENT_DIM,
        )
        cfm_value, cfm_metrics = cfm_loss(
            self.world_model.dynamics,
            z_sequence=z_sequence,
            action=batch.action[:, 1:].to(device=self.device, dtype=torch.float32),
            depart=batch.departure[:, 1:].to(device=self.device, dtype=torch.float32),
            context=batch.context[:, 1:].to(device=self.device, dtype=torch.float32),
            hdv_context=batch.hdv_context[:, 1:].to(device=self.device, dtype=torch.float32),
            hdv_target=batch.hdv_target.to(device=self.device, dtype=torch.float32),
        )
        decoded = self.world_model.decode(z_sequence)
        recon_value, recon_metrics = decoder_reconstruction_loss(
            decoded_dyn=decoded,
            flow_targets=flow,
        )
        bpr_value, bpr_metrics = decoder_bpr_supervision_loss(
            decoded_dyn=decoded,
            flow_targets=flow,
            lengths=self._lengths,
            speed_limits=self._speed_limits,
            capacities=self._capacities,
        )
        source_latents = z_sequence[:, :-1]
        transition_action = batch.action[:, 1:].to(device=self.device, dtype=torch.float32)
        transition_depart = batch.departure[:, 1:].to(device=self.device, dtype=torch.float32)
        transition_context = batch.context[:, 1:].to(device=self.device, dtype=torch.float32)
        transition_hdv_context = batch.hdv_context[:, 1:].to(device=self.device, dtype=torch.float32)
        predicted_velocity = self.world_model.dynamics(
            transition_action,
            depart=transition_depart,
            context=transition_context,
            packed_latents=source_latents,
            hdv_context=transition_hdv_context,
            flow_time=torch.zeros(
                (source_latents.shape[0], source_latents.shape[1]),
                device=self.device,
                dtype=torch.float32,
            ),
        ).latent_velocity
        predicted_next = source_latents + predicted_velocity
        transition_value, transition_metrics = transition_decoded_supervision_loss(
            z_pred_next=predicted_next,
            flow_projector=self.world_model.flow_projector,
            flow_target_next=flow[:, 1:],
        )
        t_all_value, t_all_metrics = episode_t_all_supervision_loss(
            flow_projector=self.world_model.flow_projector,
            z_tokens=z_sequence,
            reward_targets=-batch.total_travel_minutes.to(device=self.device, dtype=torch.float32).unsqueeze(-1),
        )
        total = cfm_value + recon_value + bpr_value + transition_value + t_all_value
        metrics: dict[str, Any] = {
            "loss_tensor": total,
            "cfm_loss": float(cfm_value.detach().cpu().item()),
            "decoder_reconstruction_loss": float(recon_value.detach().cpu().item()),
            "bpr_supervision_loss": float(bpr_value.detach().cpu().item()),
            "transition_decoded_loss": float(transition_value.detach().cpu().item()),
            "t_all_supervision_loss": float(t_all_value.detach().cpu().item()),
        }
        for source in (cfm_metrics, recon_metrics, bpr_metrics, transition_metrics, t_all_metrics):
            for key, value in source.items():
                metrics[str(key)] = float(value.detach().cpu().item())
        return metrics

    def _online_world_model_update(self) -> dict[str, float]:
        self.world_model.train()
        self.world_model.set_online_chunk_profile()
        batch = self.online_replay.sample_batch(
            batch_size=int(self.config.resources.online_world_model_batch_size),
            rng=self.rng,
        )
        flow = batch.flow.to(device=self.device, dtype=torch.float32)
        next_flow = batch.next_flow.to(device=self.device, dtype=torch.float32)
        z_prev = self.world_model.encode(flow)
        z_next = self.world_model.encode(next_flow)
        z_sequence = torch.stack([z_prev, z_next], dim=1)
        cfm_value, cfm_metrics = cfm_loss(
            self.world_model.dynamics,
            z_sequence=z_sequence,
            action=batch.action.to(device=self.device, dtype=torch.float32).unsqueeze(1),
            depart=batch.departure.to(device=self.device, dtype=torch.float32).unsqueeze(1),
            context=batch.context.to(device=self.device, dtype=torch.float32).unsqueeze(1),
            hdv_context=batch.hdv_context.to(device=self.device, dtype=torch.float32).unsqueeze(1),
            hdv_target=batch.hdv_target.to(device=self.device, dtype=torch.float32).unsqueeze(1),
        )
        predicted_next = cfm_predict_one_timestep(
            self.world_model.dynamics,
            past_latents=z_prev.unsqueeze(1),
            action=batch.action.to(device=self.device, dtype=torch.float32).unsqueeze(1),
            depart=batch.departure.to(device=self.device, dtype=torch.float32).unsqueeze(1),
            context=batch.context.to(device=self.device, dtype=torch.float32).unsqueeze(1),
            hdv_context=batch.hdv_context.to(device=self.device, dtype=torch.float32).unsqueeze(1),
        ).squeeze(1)
        transition_value, transition_metrics = transition_decoded_supervision_loss(
            z_pred_next=predicted_next,
            flow_projector=self.world_model.flow_projector,
            flow_target_next=next_flow,
        )
        t_all_value, t_all_metrics = episode_t_all_supervision_loss(
            flow_projector=self.world_model.flow_projector,
            z_tokens=predicted_next.unsqueeze(1),
            reward_targets=-batch.total_travel_minutes.to(device=self.device, dtype=torch.float32).unsqueeze(-1),
        )
        world_model_loss = cfm_value + transition_value + t_all_value
        self.world_model_optimizer_online.zero_grad(set_to_none=True)
        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.world_model.flow_projector.parameters()) + list(self.world_model.dynamics.parameters()),
            GRAD_CLIP,
        )
        self.world_model_optimizer_online.step()

        with torch.no_grad():
            decoded_next = self.world_model.decode(predicted_next)
            raw_decoder_fraction = decoded_next[..., 2].mean(dim=-1)
        target_fraction = (
            batch.active_vehicle_seconds.to(device=self.device, dtype=torch.float32)
            / batch.transition_seconds.to(device=self.device, dtype=torch.float32).clamp_min(1.0)
            / max(float(self.total_vehicle_count), 1.0)
        ).clamp(0.0, 1.0)
        predicted_fraction = self.world_model.active_vehicle_fraction_readout(
            predicted_next.detach(),
            z_prev=z_prev.detach(),
            raw_decoder_fraction=raw_decoder_fraction.detach(),
            transition_seconds=batch.transition_seconds.to(device=self.device, dtype=torch.float32),
        )
        active_loss = F.mse_loss(predicted_fraction, target_fraction)
        self.active_fraction_optimizer.zero_grad(set_to_none=True)
        active_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.world_model.active_vehicle_fraction_readout.parameters(),
            GRAD_CLIP,
        )
        self.active_fraction_optimizer.step()
        with torch.no_grad():
            self.world_model.active_vehicle_fraction_readout.calibration_updates.add_(
                1
            )

        metrics: dict[str, float] = {
            "world_model_loss": float(world_model_loss.detach().cpu().item()),
            "active_fraction_loss": float(active_loss.detach().cpu().item()),
        }
        for source in (cfm_metrics, transition_metrics, t_all_metrics):
            for key, value in source.items():
                metrics[str(key)] = float(value.detach().cpu().item())
        self.metrics["world_model_loss_last"] = metrics["world_model_loss"]
        self.metrics["active_fraction_loss_last"] = metrics[
            "active_fraction_loss"
        ]
        return metrics

    def _train_imagination_pass(self) -> dict[str, float]:
        if not self.online_replay.episodes:
            return {
                "actor_loss": 0.0,
                "vehicle_critic_loss": 0.0,
                "latent_critic_loss": 0.0,
                "imagined_steps": 0.0,
            }

        local_rollout_rows: list[dict[str, Any]] = []
        latent_critic_losses: list[torch.Tensor] = []
        total_steps = 0
        for rollout_start in range(
            0,
            IMAGINED_ROLLOUTS_PER_PASS,
            IMAGINATION_BATCH_SIZE,
        ):
            batch = self.online_replay.sample_batch(
                batch_size=min(
                    IMAGINATION_BATCH_SIZE,
                    IMAGINED_ROLLOUTS_PER_PASS - rollout_start,
                ),
                rng=self.rng,
            )
            for batch_index in range(len(batch.vehicles)):
                rollout_rows, latent_critic_loss = self._imagine_from_seed(
                    batch,
                    batch_index,
                    rollout_index=rollout_start + batch_index,
                )
                local_rollout_rows.extend(rollout_rows)
                if latent_critic_loss is not None:
                    latent_critic_losses.append(latent_critic_loss)
                total_steps += len(rollout_rows)

        if not local_rollout_rows:
            return {
                "actor_loss": 0.0,
                "vehicle_critic_loss": 0.0,
                "latent_critic_loss": 0.0,
                "imagined_steps": 0.0,
            }

        obs = torch.stack([row["obs"] for row in local_rollout_rows], dim=0).to(self.device)
        next_obs = torch.stack([row["next_obs"] for row in local_rollout_rows], dim=0).to(self.device)
        log_prob = torch.stack([row["log_prob"] for row in local_rollout_rows], dim=0).to(self.device)
        entropy = torch.stack([row["entropy"] for row in local_rollout_rows], dim=0).to(self.device)
        rewards = torch.stack([row["reward"] for row in local_rollout_rows], dim=0).to(self.device)
        terminals = torch.tensor(
            [row["terminal"] for row in local_rollout_rows],
            dtype=torch.bool,
            device=self.device,
        )
        next_valid = torch.tensor(
            [row["next_valid"] for row in local_rollout_rows],
            dtype=torch.bool,
            device=self.device,
        )
        trajectory_vehicle_keys = torch.tensor(
            [
                row["trajectory_vehicle_key"]
                for row in local_rollout_rows
            ],
            dtype=torch.long,
            device=self.device,
        )
        global_adv = torch.stack([row["global_advantage"] for row in local_rollout_rows], dim=0).to(self.device)

        current_values = self.vehicle_critic(obs).squeeze(-1)
        with torch.no_grad():
            next_values = self.vehicle_critic(next_obs).squeeze(-1)
        local_returns = lambda_returns_by_vehicle_ids(
            rewards,
            next_values,
            next_valid,
            trajectory_vehicle_keys,
            terminal=terminals,
        )
        local_advantage = local_returns - current_values
        combined_advantage = local_advantage + global_adv
        combined_advantage = (
            combined_advantage - combined_advantage.mean()
        ) / combined_advantage.std(unbiased=False).clamp_min(1e-6)
        combined_advantage = combined_advantage.clamp(min=-2.5, max=2.5)

        actor_loss = -(log_prob * combined_advantage.detach() + ACTOR_ENTROPY_COEF * entropy).mean()
        vehicle_critic_loss = F.mse_loss(current_values, local_returns.detach())
        latent_critic_loss = (
            torch.stack(latent_critic_losses).mean()
            if latent_critic_losses
            else torch.zeros((), device=self.device)
        )

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.actor_optimizer.step()

        self.vehicle_critic_optimizer.zero_grad(set_to_none=True)
        vehicle_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vehicle_critic.parameters(), GRAD_CLIP)
        self.vehicle_critic_optimizer.step()

        if latent_critic_losses:
            self.latent_critic_optimizer.zero_grad(set_to_none=True)
            latent_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.latent_critic.parameters(), GRAD_CLIP)
            self.latent_critic_optimizer.step()

        metrics = {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "vehicle_critic_loss": float(vehicle_critic_loss.detach().cpu().item()),
            "latent_critic_loss": float(latent_critic_loss.detach().cpu().item()),
            "imagined_steps": float(total_steps),
        }
        self.metrics["actor_loss_last"] = metrics["actor_loss"]
        self.metrics["vehicle_critic_loss_last"] = metrics["vehicle_critic_loss"]
        self.metrics["latent_critic_loss_last"] = metrics["latent_critic_loss"]
        return metrics

    def _imagine_from_seed(
        self,
        batch: Any,
        batch_index: int,
        *,
        rollout_index: int,
    ) -> tuple[list[dict[str, Any]], torch.Tensor | None]:
        seed_flow = batch.flow[batch_index].to(self.device, dtype=torch.float32)
        seed_context_row = batch.context[batch_index]
        seed_context_signal = float(seed_context_row[2].item())
        seed_hdv_context = batch.hdv_context[batch_index].to(self.device, dtype=torch.float32)
        seed_step = int(batch.step_index[batch_index].item())
        vehicle_snapshot = batch.vehicles[batch_index]
        with torch.no_grad():
            current_latent = self.world_model.encode(
                seed_flow.unsqueeze(0)
            ).squeeze(0)

        vehicle_states: list[dict[str, Any]] = []
        for idx, vehicle_id in enumerate(vehicle_snapshot.vehicle_ids):
            vehicle_states.append(
                {
                    "vehicle_id": str(vehicle_id),
                    "vehicle_index": idx,
                    "current_edge": int(vehicle_snapshot.current_edges[idx].item()),
                    "destination_edge": int(vehicle_snapshot.destination_edges[idx].item()),
                    "departure_step": int(vehicle_snapshot.departure_steps[idx].item()),
                    "remaining_edge_seconds": float(self._free_flow_seconds[int(vehicle_snapshot.current_edges[idx].item())]),
                    "arrived": False,
                }
            )

        rollout_rows: list[dict[str, Any]] = []
        latent_values: list[torch.Tensor] = []
        latent_next_values: list[torch.Tensor] = []
        latent_rewards: list[torch.Tensor] = []
        latent_terminals: list[bool] = []
        step_row_counts: list[int] = []
        edge_order = tuple(range(self.edge_count))
        zero_route_counts = np.zeros((4,), dtype=np.float32)
        zero_edge_pressure = np.zeros((self.edge_count,), dtype=np.float32)
        transition_seconds = torch.tensor(
            [DECISION_INTERVAL_SECONDS],
            dtype=torch.float32,
            device=self.device,
        )
        decoded_cache: np.ndarray | None = None
        for horizon_step in range(IMAGINED_HORIZON):
            if decoded_cache is None:
                with torch.no_grad():
                    decoded_cache = (
                        self.world_model.decode(current_latent.unsqueeze(0))
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
            decoded = decoded_cache
            volume_norm = np.asarray(decoded[:, 2], dtype=np.float32)
            speed_ratio = np.clip(np.asarray(decoded[:, 1], dtype=np.float32), 0.05, 1.5)
            edge_travel_seconds = self._free_flow_seconds / speed_ratio
            bpr_minutes = (
                self._free_flow_seconds
                * (1.0 + 0.15 * np.power(np.maximum(volume_norm, 0.0), 4.0))
                / 60.0
            ).astype(np.float32)
            policy_edge_features = np.stack([volume_norm, bpr_minutes], axis=-1).astype(np.float32)

            active_states = [
                state
                for state in vehicle_states
                if not bool(state["arrived"]) and int(state["departure_step"]) <= seed_step + horizon_step
            ]
            if not active_states:
                break
            candidate_path_batch = [
                self.path_generator.generate_paths(
                    int(vehicle_state["current_edge"]),
                    int(vehicle_state["destination_edge"]),
                )
                for vehicle_state in active_states
            ]
            action_decisions: list[ActionDecision] = []
            selection_rows: list[dict[str, Any]] = []
            context_progress = self._episode_progress(seed_step + horizon_step)
            active_fraction = float(len(active_states)) / max(float(self.total_automated), 1.0)
            observations = build_actor_observations(
                policy_edge_features=policy_edge_features,
                candidate_paths=candidate_path_batch,
                edge_travel_seconds=edge_travel_seconds,
                assigned_route_counts=zero_route_counts,
                assigned_edge_pressure=zero_edge_pressure,
                episode_progress=context_progress,
                active_av_fraction=active_fraction,
            )
            obs_tensor = torch.as_tensor(
                observations,
                dtype=torch.float32,
                device=self.device,
            )
            selection = self.actor.sample_action(obs_tensor)
            selected_route_indices = (
                selection.route_index.detach().cpu().numpy()
            )
            route_costs = np.stack(
                [
                    _route_cost_seconds(paths, edge_travel_seconds)
                    for paths in candidate_path_batch
                ],
                axis=0,
            )
            route_cost_tensor = torch.as_tensor(
                route_costs,
                dtype=torch.float32,
                device=self.device,
            )
            reward_matrix = capped_route_reward_matrix(
                route_cost_tensor,
                torch.isfinite(route_cost_tensor),
            )
            for vehicle_index, (
                vehicle_state,
                candidate_paths,
                selected_route_index,
            ) in enumerate(
                zip(
                    active_states,
                    candidate_path_batch,
                    selected_route_indices,
                )
            ):
                selected_route_index = int(selected_route_index)
                decision = ActionDecision.build(
                    route_index=selected_route_index,
                    candidate_paths=candidate_paths,
                    edge_order=edge_order,
                    edge_travel_times=edge_travel_seconds,
                    decision_horizon_seconds=DECISION_INTERVAL_SECONDS,
                )
                action_decisions.append(decision)
                selection_rows.append(
                    {
                        "vehicle_state": vehicle_state,
                        "decision": decision,
                        "obs": obs_tensor[vehicle_index],
                        "log_prob": selection.log_probability[vehicle_index],
                        "entropy": selection.entropy[vehicle_index],
                        "reward": reward_matrix[
                            vehicle_index,
                            selected_route_index,
                        ],
                    }
                )
            step_row_counts.append(len(selection_rows))

            planned_action = aggregate_planned_edge_flow(
                action_decisions,
                edge_count=self.edge_count,
            )
            departure = self._departure_vector_from_vehicle_states(vehicle_states, seed_step + horizon_step)
            context = torch.as_tensor(
                [context_progress, active_fraction, seed_context_signal],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                next_latent = cfm_predict_one_timestep(
                    self.world_model.dynamics,
                    past_latents=current_latent.unsqueeze(0).unsqueeze(0),
                    action=torch.as_tensor(
                        planned_action,
                        dtype=torch.float32,
                        device=self.device,
                    ).view(1, 1, -1),
                    depart=torch.as_tensor(
                        departure,
                        dtype=torch.float32,
                        device=self.device,
                    ).view(1, 1, -1),
                    context=context.view(1, 1, -1),
                    hdv_context=seed_hdv_context.view(1, 1, -1),
                ).squeeze(0).squeeze(0)
                next_decoded = self.world_model.decode(next_latent.unsqueeze(0)).squeeze(0)
                global_fraction = self.world_model.active_vehicle_fraction_readout(
                    next_latent.unsqueeze(0),
                    z_prev=current_latent.unsqueeze(0),
                    raw_decoder_fraction=next_decoded[..., 2].mean().unsqueeze(0),
                    transition_seconds=transition_seconds,
                ).squeeze(0)
            global_reward = -global_fraction * IMAGINATION_GLOBAL_REWARD_SCALE
            latent_value = self.latent_critic(current_latent.unsqueeze(0), context.unsqueeze(0)).squeeze(0).squeeze(-1)

            for row in selection_rows:
                vehicle_state = row["vehicle_state"]
                decision = row["decision"]
                advance_vehicle_on_path(
                    vehicle_state,
                    selected_path=decision.selected_path,
                    edge_travel_seconds=edge_travel_seconds,
                    interval_seconds=DECISION_INTERVAL_SECONDS,
                    edge_count=self.edge_count,
                )

            all_done = all(bool(state["arrived"]) for state in vehicle_states)
            latent_terminals.append(bool(all_done))

            next_decoded_np = next_decoded.detach().cpu().numpy()
            next_speed_ratio = np.clip(np.asarray(next_decoded_np[:, 1], dtype=np.float32), 0.05, 1.5)
            next_edge_travel_seconds = self._free_flow_seconds / next_speed_ratio
            next_policy_features = np.stack(
                [
                    np.asarray(next_decoded_np[:, 2], dtype=np.float32),
                    (
                        self._free_flow_seconds
                        * (1.0 + 0.15 * np.power(np.maximum(np.asarray(next_decoded_np[:, 2], dtype=np.float32), 0.0), 4.0))
                        / 60.0
                    ).astype(np.float32),
                ],
                axis=-1,
            )
            next_observations = torch.zeros(
                (len(selection_rows), 33),
                dtype=torch.float32,
                device=self.device,
            )
            next_active_row_indices: list[int] = []
            next_valid_flags = [False] * len(selection_rows)
            next_candidate_path_batch: list[tuple[tuple[int, ...], ...]] = []
            for row_index, row in enumerate(selection_rows):
                vehicle_state = row["vehicle_state"]
                still_active = (not bool(vehicle_state["arrived"])) and int(vehicle_state["departure_step"]) <= seed_step + horizon_step + 1
                if still_active:
                    next_active_row_indices.append(row_index)
                    next_valid_flags[row_index] = True
                    next_candidate_path_batch.append(
                        self.path_generator.generate_paths(
                            int(vehicle_state["current_edge"]),
                            int(vehicle_state["destination_edge"]),
                        )
                    )
            next_progress = self._episode_progress(
                seed_step + horizon_step + 1
            )
            next_active_fraction = float(
                len(next_candidate_path_batch)
            ) / max(float(self.total_automated), 1.0)
            next_context = context.clone()
            next_context[0] = next_progress
            next_context[1] = next_active_fraction
            next_value = self.latent_critic(
                next_latent.unsqueeze(0),
                next_context.unsqueeze(0),
            ).squeeze(0).squeeze(-1).detach()
            latent_values.append(latent_value)
            latent_next_values.append(next_value)
            latent_rewards.append(global_reward)
            if next_candidate_path_batch:
                next_observation_batch = build_actor_observations(
                    policy_edge_features=next_policy_features,
                    candidate_paths=next_candidate_path_batch,
                    edge_travel_seconds=next_edge_travel_seconds,
                    assigned_route_counts=zero_route_counts,
                    assigned_edge_pressure=zero_edge_pressure,
                    episode_progress=next_progress,
                    active_av_fraction=next_active_fraction,
                )
                next_observations[next_active_row_indices] = torch.as_tensor(
                    next_observation_batch,
                    dtype=torch.float32,
                    device=self.device,
                )
            for row_index, row in enumerate(selection_rows):
                vehicle_state = row["vehicle_state"]
                still_active = next_valid_flags[row_index]
                rollout_rows.append(
                    {
                        "obs": row["obs"],
                        "next_obs": next_observations[row_index],
                        "next_valid": bool(still_active),
                        "log_prob": row["log_prob"],
                        "entropy": row["entropy"],
                        "reward": row["reward"],
                        "terminal": bool(all_done or vehicle_state["arrived"]),
                        "trajectory_vehicle_key": int(
                            rollout_index * max(self.total_automated, 1)
                            + row["vehicle_state"]["vehicle_index"]
                        ),
                        "global_advantage": None,
                    }
                )
            current_latent = next_latent
            decoded_cache = next_decoded_np

        if not latent_rewards:
            return rollout_rows, None
        latent_reward_tensor = torch.stack(latent_rewards, dim=0)
        latent_next_value_tensor = torch.stack(latent_next_values, dim=0)
        latent_terminal_tensor = torch.tensor(
            latent_terminals,
            dtype=torch.bool,
            device=self.device,
        )
        latent_return_tensor = lambda_return(
            latent_reward_tensor,
            latent_next_value_tensor,
            terminal=latent_terminal_tensor,
        )
        latent_value_tensor = torch.stack(latent_values, dim=0)
        latent_advantage = (latent_return_tensor - latent_value_tensor).detach()
        offset = 0
        for step_index, step_reward in enumerate(latent_advantage):
            row_count = step_row_counts[step_index]
            for row_index in range(offset, offset + row_count):
                rollout_rows[row_index]["global_advantage"] = step_reward.detach()
            offset += row_count
        latent_loss = F.mse_loss(latent_value_tensor, latent_return_tensor.detach())
        return rollout_rows, latent_loss

    def _departure_vector_from_vehicle_states(
        self,
        vehicle_states: Sequence[Mapping[str, Any]],
        global_step: int,
    ) -> np.ndarray:
        vector = np.zeros((self.edge_count,), dtype=np.float32)
        for vehicle_state in vehicle_states:
            if int(vehicle_state["departure_step"]) == int(global_step):
                vector[int(vehicle_state["current_edge"])] += 1.0
        return _normalize_edge_vector(vector)

    def state_dict(self) -> dict[str, Any]:
        numpy_state = np.random.get_state()
        rng_state = {
            "python_random": random.getstate(),
            "numpy_global": {
                "algorithm": str(numpy_state[0]),
                "keys": np.asarray(numpy_state[1], dtype=np.uint32),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "numpy_generator": copy.deepcopy(self.rng.bit_generator.state),
            "torch_cpu": torch.get_rng_state().cpu(),
            "torch_cuda": (
                [state.cpu() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else []
            ),
        }
        return {
            "models": {
                "world_model": self.world_model.state_dict(),
                "actor": self.actor.state_dict(),
                "vehicle_critic": self.vehicle_critic.state_dict(),
                "latent_critic": self.latent_critic.state_dict(),
            },
            "optimizers": {
                "world_model_initial": self.world_model_optimizer_initial.state_dict(),
                "world_model_online": self.world_model_optimizer_online.state_dict(),
                "active_fraction": self.active_fraction_optimizer.state_dict(),
                "actor": self.actor_optimizer.state_dict(),
                "vehicle_critic": self.vehicle_critic_optimizer.state_dict(),
                "latent_critic": self.latent_critic_optimizer.state_dict(),
            },
            "replay": {
                "warmup": self.warmup_replay.state_dict(),
                "online": self.online_replay.state_dict(),
            },
            "hdv": {
                "predictor": self.hdv_predictor.state_dict(),
                "true_state": copy.deepcopy(self._current_hdv_learning_state),
            },
            "rng": rng_state,
            "metrics": dict(self.metrics),
            "caches": {
                "candidate_path_generator": self.path_generator.state_dict(),
                "next_environment_episode": int(self._next_environment_episode),
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {"models", "optimizers", "replay", "hdv", "rng", "metrics", "caches"}
        if set(state) != expected:
            raise ValueError("MainlineTrainer checkpoint state has an invalid schema")
        models = state["models"]
        optimizers = state["optimizers"]
        replay = state["replay"]
        hdv = state["hdv"]
        rng = state["rng"]
        metrics = state["metrics"]
        caches = state["caches"]
        if set(models) != {"world_model", "actor", "vehicle_critic", "latent_critic"}:
            raise ValueError("MainlineTrainer models state has an invalid schema")
        if set(optimizers) != {
            "world_model_initial",
            "world_model_online",
            "active_fraction",
            "actor",
            "vehicle_critic",
            "latent_critic",
        }:
            raise ValueError("MainlineTrainer optimizers state has an invalid schema")
        if set(replay) != {"warmup", "online"}:
            raise ValueError("MainlineTrainer replay state has an invalid schema")
        if set(hdv) != {"predictor", "true_state"}:
            raise ValueError("MainlineTrainer hdv state has an invalid schema")
        if set(rng) != {"python_random", "numpy_global", "numpy_generator", "torch_cpu", "torch_cuda"}:
            raise ValueError("MainlineTrainer rng state has an invalid schema")
        if set(caches) != {"candidate_path_generator", "next_environment_episode"}:
            raise ValueError("MainlineTrainer caches state has an invalid schema")

        self.world_model.load_state_dict(models["world_model"], strict=True)
        self.actor.load_state_dict(models["actor"], strict=True)
        self.vehicle_critic.load_state_dict(models["vehicle_critic"], strict=True)
        self.latent_critic.load_state_dict(models["latent_critic"], strict=True)

        self.world_model_optimizer_initial.load_state_dict(optimizers["world_model_initial"])
        self.world_model_optimizer_online.load_state_dict(optimizers["world_model_online"])
        self.active_fraction_optimizer.load_state_dict(
            optimizers["active_fraction"]
        )
        self.actor_optimizer.load_state_dict(optimizers["actor"])
        self.vehicle_critic_optimizer.load_state_dict(optimizers["vehicle_critic"])
        self.latent_critic_optimizer.load_state_dict(optimizers["latent_critic"])

        self.warmup_replay.load_state_dict(replay["warmup"])
        self.online_replay.load_state_dict(replay["online"])
        self.hdv_predictor.load_state_dict(hdv["predictor"])
        self._current_hdv_learning_state = copy.deepcopy(dict(hdv["true_state"]))

        self.path_generator.load_state_dict(caches["candidate_path_generator"])
        self._next_environment_episode = int(caches["next_environment_episode"])
        self.metrics = {str(key): float(value) for key, value in metrics.items()}
        self.metrics.setdefault("warmup_attempts_discarded", 0.0)

        random.setstate(_to_tuple_tree(rng["python_random"]))
        numpy_global = rng["numpy_global"]
        np.random.set_state(
            (
                str(numpy_global["algorithm"]),
                np.asarray(numpy_global["keys"], dtype=np.uint32),
                int(numpy_global["position"]),
                int(numpy_global["has_gauss"]),
                float(numpy_global["cached_gaussian"]),
            )
        )
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(dict(rng["numpy_generator"]))
        torch.set_rng_state(torch.as_tensor(rng["torch_cpu"], dtype=torch.uint8).cpu())
        if torch.cuda.is_available():
            cuda_states = [torch.as_tensor(state, dtype=torch.uint8, device="cpu") for state in rng["torch_cuda"]]
            if cuda_states:
                torch.cuda.set_rng_state_all(cuda_states)

    def summary(self) -> dict[str, Any]:
        return {
            **self.metrics,
            "warmup_replay_episodes": len(self.warmup_replay.episodes),
            "online_replay_episodes": len(self.online_replay.episodes),
            "next_environment_episode": int(self._next_environment_episode),
        }

    def close(self) -> None:
        pass


__all__ = ["MainlineTrainer"]
