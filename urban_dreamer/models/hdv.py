from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ..traffic.topology import TrafficTopology


HDV_RESPONSE_CHECKPOINT_FORMAT = (
    "causal_persistence_residual_hdv_od_route_predictor_v4"
)
HDV_RESPONSE_ANCHOR_FLOOR = 1e-3
HDV_RESPONSE_K = 4
HDV_RESPONSE_FEATURE_DIM = 43
DEFAULT_HDV_RESPONSE_HIDDEN_DIM = 128
DEFAULT_HDV_RESPONSE_DEPTH = 2
ONLINE_UPDATE_LR = 1e-4
ONLINE_UPDATE_WEIGHT_DECAY = 1e-4
ONLINE_UPDATE_EPOCHS = 2
ONLINE_UPDATE_BUFFER_EPISODES = 16
ONLINE_UPDATE_MIN_PAIRS = 2
ONLINE_UPDATE_L2 = 1e-4
ONLINE_UPDATE_GRAD_CLIP = 1.0


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).clip(min=0.0)
    if array.ndim != 2:
        raise ValueError(f"expected a rank-2 route matrix, got {array.shape}")
    denominator = array.sum(axis=-1, keepdims=True)
    uniform = np.full_like(array, 1.0 / max(int(array.shape[-1]), 1))
    return np.where(denominator > 1e-12, array / np.maximum(denominator, 1e-12), uniform)


def _normalise_vector(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1).clip(min=0.0)
    total = float(array.sum())
    if total <= 1e-12:
        return np.zeros_like(array)
    return array / total


def apply_persistence_anchored_route_residual(
    persistence: np.ndarray,
    raw_residual: np.ndarray,
    *,
    scale: float = 1.0,
    anchor_floor: float = HDV_RESPONSE_ANCHOR_FLOOR,
) -> np.ndarray:
    previous = _normalise_rows(persistence)
    residual = np.asarray(raw_residual, dtype=np.float64)
    if residual.shape != previous.shape:
        raise ValueError(
            f"route residual shape mismatch: {residual.shape} vs {previous.shape}"
        )
    n_routes = int(previous.shape[-1])
    floor = float(anchor_floor)
    if floor < 0.0 or floor >= 1.0 / max(n_routes, 1):
        raise ValueError(f"anchor_floor must be in [0, 1 / n_routes), got {floor}")
    anchor = (1.0 - float(n_routes) * floor) * previous + floor
    logits = np.log(np.clip(anchor, 1e-12, 1.0)) + float(scale) * residual
    logits = logits - logits.max(axis=-1, keepdims=True)
    response = _normalise_rows(np.exp(logits))
    prediction = np.clip(previous + response - anchor, 0.0, None)
    return _normalise_rows(prediction)


def apply_persistence_anchored_route_residual_torch(
    persistence: torch.Tensor,
    raw_residual: torch.Tensor,
    *,
    scale: float = 1.0,
    anchor_floor: float = HDV_RESPONSE_ANCHOR_FLOOR,
) -> torch.Tensor:
    previous = persistence.float().clamp_min(0.0)
    previous = previous / previous.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if raw_residual.shape != previous.shape:
        raise ValueError(
            "route residual shape mismatch: "
            f"{tuple(raw_residual.shape)} vs {tuple(previous.shape)}"
        )
    n_routes = int(previous.shape[-1])
    floor = float(anchor_floor)
    if floor < 0.0 or floor >= 1.0 / max(n_routes, 1):
        raise ValueError(f"anchor_floor must be in [0, 1 / n_routes), got {floor}")
    anchor = (1.0 - float(n_routes) * floor) * previous + floor
    response = torch.softmax(
        torch.log(anchor.clamp_min(1e-12)) + float(scale) * raw_residual.float(),
        dim=-1,
    )
    prediction = (previous + response - anchor).clamp_min(0.0)
    return prediction / prediction.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class PersistenceAnchoredOdRouteResidual(nn.Module):
    """Fixed v4 residual MLP for four-route OD response prediction."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = HDV_RESPONSE_FEATURE_DIM
        for _ in range(DEFAULT_HDV_RESPONSE_DEPTH):
            layers.extend(
                [
                    nn.Linear(width, DEFAULT_HDV_RESPONSE_HIDDEN_DIM),
                    nn.GELU(),
                    nn.LayerNorm(DEFAULT_HDV_RESPONSE_HIDDEN_DIM),
                ]
            )
            width = DEFAULT_HDV_RESPONSE_HIDDEN_DIM
        output = nn.Linear(width, HDV_RESPONSE_K)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass(frozen=True)
class EpisodeObservation:
    episode: int
    hdv_route_choices: dict[str, int]
    travel_time_route_choices: dict[str, int]
    travel_times_seconds: dict[str, float]
    pre_state: dict[str, Any]
    post_state: dict[str, Any]

    def state_dict(self) -> dict[str, Any]:
        return {
            "episode": int(self.episode),
            "hdv_route_choices": {
                str(vehicle_id): int(route_index)
                for vehicle_id, route_index in self.hdv_route_choices.items()
            },
            "travel_time_route_choices": {
                str(vehicle_id): int(route_index)
                for vehicle_id, route_index in self.travel_time_route_choices.items()
            },
            "travel_times_seconds": {
                str(vehicle_id): float(seconds)
                for vehicle_id, seconds in self.travel_times_seconds.items()
            },
            "pre_state": copy.deepcopy(self.pre_state),
            "post_state": copy.deepcopy(self.post_state),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "EpisodeObservation":
        expected_keys = {
            "episode",
            "hdv_route_choices",
            "travel_time_route_choices",
            "travel_times_seconds",
            "pre_state",
            "post_state",
        }
        if set(state) != expected_keys:
            raise ValueError(
                "episode observation state has invalid keys: "
                f"{sorted(state)} != {sorted(expected_keys)}"
            )
        hdv_route_choices = state["hdv_route_choices"]
        travel_time_route_choices = state["travel_time_route_choices"]
        travel_times_seconds = state["travel_times_seconds"]
        pre_state = state["pre_state"]
        post_state = state["post_state"]
        if not isinstance(hdv_route_choices, Mapping):
            raise ValueError(
                "episode observation hdv_route_choices must be a mapping"
            )
        if not isinstance(travel_time_route_choices, Mapping):
            raise ValueError(
                "episode observation travel_time_route_choices must be a mapping"
            )
        if not isinstance(travel_times_seconds, Mapping):
            raise ValueError(
                "episode observation travel_times_seconds must be a mapping"
            )
        if not isinstance(pre_state, Mapping) or not isinstance(post_state, Mapping):
            raise ValueError("episode observation pre_state/post_state must be mappings")
        return cls(
            episode=int(state["episode"]),
            hdv_route_choices={
                str(vehicle_id): int(route_index)
                for vehicle_id, route_index in hdv_route_choices.items()
            },
            travel_time_route_choices={
                str(vehicle_id): int(route_index)
                for vehicle_id, route_index in travel_time_route_choices.items()
            },
            travel_times_seconds={
                str(vehicle_id): float(seconds)
                for vehicle_id, seconds in travel_times_seconds.items()
            },
            pre_state=copy.deepcopy(dict(pre_state)),
            post_state=copy.deepcopy(dict(post_state)),
        )


@dataclass(frozen=True)
class _ProcessedEpisode:
    episode: int
    route_counts: np.ndarray
    actual_route: np.ndarray
    route_before: np.ndarray
    route_after: np.ndarray
    route_cost_before: np.ndarray
    filled_route_tt: np.ndarray
    route_tt_observed: np.ndarray
    route_tt_sample_counts: np.ndarray
    edge_distribution: np.ndarray
    od_weight: np.ndarray


class HdvOdRouteResponsePredictor:
    """Strict v4 HDV response predictor bound to the current topology and HDV state."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        topology: TrafficTopology,
        initial_hdv_state_path: str | Path,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.initial_hdv_state_path = Path(initial_hdv_state_path)
        self.device = torch.device(device)
        self.anchor_floor = HDV_RESPONSE_ANCHOR_FLOOR
        self.topology = topology
        self._validate_topology(topology)
        self.ods = tuple((int(origin), int(destination)) for origin, destination in topology.od_pairs)
        self.edge_ids = tuple(str(edge_id) for edge_id in topology.network.edge_ids)
        self.route_edge_indices = tuple(
            tuple(tuple(int(edge_index) for edge_index in route) for route in od_routes)
            for od_routes in topology.route_paths
        )
        self.n_edges = len(self.edge_ids)
        self.n_od = len(self.ods)
        self.route_free_flow_costs = self._build_route_free_flow_costs()
        self.human_od_counts = np.asarray(topology.od_human_counts, dtype=np.float32).reshape(-1)
        if self.human_od_counts.shape != (self.n_od,):
            raise ValueError(
                "topology.od_human_counts must have one value per OD row, got "
                f"{self.human_od_counts.shape}"
            )
        self.initial_route_policy = self._load_initial_route_policy(
            self.initial_hdv_state_path
        )
        self.human_vehicle_to_od = self._build_human_vehicle_lookup()
        self.vehicle_to_od = self._build_vehicle_lookup()
        self.model = PersistenceAnchoredOdRouteResidual().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=ONLINE_UPDATE_LR,
            weight_decay=ONLINE_UPDATE_WEIGHT_DECAY,
        )
        self._reset_runtime()
        deployment = self._load_deployment_checkpoint(self.checkpoint_path)
        self._load_deployment_state(deployment)

    def _reset_runtime(self) -> None:
        self.last_observed_route = np.asarray(
            self.initial_route_policy,
            dtype=np.float32,
        ).copy()
        self.previous_observation: EpisodeObservation | None = None
        self.pending_predicted_route: np.ndarray | None = None
        self.pending_prediction_episode: int | None = None
        self.pending_training_sample: dict[str, np.ndarray] | None = None
        self.online_training_samples: list[dict[str, np.ndarray]] = []
        self.trained_pairs = 0
        self.online_update_steps = 0

    def _validate_topology(self, topology: TrafficTopology) -> None:
        if int(topology.route_count) != HDV_RESPONSE_K:
            raise ValueError(
                f"v4 predictor requires topology.route_count={HDV_RESPONSE_K}, got "
                f"{topology.route_count}"
            )
        if len(topology.od_pairs) != len(topology.route_paths):
            raise ValueError("topology OD rows and route paths must align one-to-one")
        for od_index, route_row in enumerate(topology.route_paths):
            if len(route_row) != HDV_RESPONSE_K:
                raise ValueError(
                    f"topology OD row {od_index} must have {HDV_RESPONSE_K} routes, "
                    f"got {len(route_row)}"
                )

    def _build_human_vehicle_lookup(self) -> dict[str, int]:
        vehicle_to_od: dict[str, int] = {}
        od_to_index = {od_pair: od_index for od_index, od_pair in enumerate(self.ods)}
        for vehicle in self.topology.vehicles:
            if bool(vehicle.is_automated):
                continue
            od_pair = (int(vehicle.origin_index), int(vehicle.destination_index))
            if od_pair not in od_to_index:
                raise ValueError(
                    f"human vehicle {vehicle.vehicle_id!r} has unknown OD pair {od_pair}"
                )
            vehicle_to_od[str(vehicle.vehicle_id)] = od_to_index[od_pair]
        return vehicle_to_od

    def _build_vehicle_lookup(self) -> dict[str, int]:
        vehicle_to_od: dict[str, int] = {}
        od_to_index = {
            od_pair: od_index for od_index, od_pair in enumerate(self.ods)
        }
        for vehicle in self.topology.vehicles:
            od_pair = (
                int(vehicle.origin_index),
                int(vehicle.destination_index),
            )
            if od_pair not in od_to_index:
                raise ValueError(
                    f"vehicle {vehicle.vehicle_id!r} has unknown OD pair {od_pair}"
                )
            vehicle_to_od[str(vehicle.vehicle_id)] = od_to_index[od_pair]
        return vehicle_to_od

    def _build_route_free_flow_costs(self) -> np.ndarray:
        costs = np.zeros((self.n_od, HDV_RESPONSE_K), dtype=np.float32)
        free_flow = np.asarray(
            self.topology.network.free_flow_travel_seconds,
            dtype=np.float32,
        ).reshape(-1)
        for od_index, route_row in enumerate(self.topology.route_paths):
            for route_index, route in enumerate(route_row):
                if not route:
                    raise ValueError(
                        f"topology route_paths[{od_index}][{route_index}] must be non-empty"
                    )
                route_cost = float(sum(float(free_flow[int(edge_index)]) for edge_index in route))
                costs[od_index, route_index] = max(route_cost, 1e-3)
        return costs

    def _load_initial_route_policy(
        self,
        path: Path,
    ) -> np.ndarray:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load initial HDV state {path}: {exc}") from exc
        if not isinstance(state, dict):
            raise ValueError("initial HDV state must be a JSON object")
        if state.get("version") != 1:
            raise ValueError("initial HDV route state must use version=1")
        route_policy, _route_costs = self._route_policy_rows_to_arrays(
            state,
            label="initial HDV state",
        )
        return route_policy

    def _route_policy_rows_to_arrays(
        self,
        state: Mapping[str, Any],
        *,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(state, Mapping):
            raise ValueError(f"{label} must be a mapping")
        route_policy = state.get("route_policy")
        if not isinstance(route_policy, list):
            raise ValueError(f"{label} route_policy must be an array")
        if len(route_policy) != self.n_od:
            raise ValueError(
                f"{label} route_policy row count mismatch: {len(route_policy)} vs {self.n_od}"
            )
        probability_by_od: dict[tuple[int, int], np.ndarray] = {}
        cost_by_od: dict[tuple[int, int], np.ndarray] = {}
        for row in route_policy:
            if not isinstance(row, Mapping) or set(row) != {
                "origin",
                "destination",
                "probs",
                "costs",
            }:
                raise ValueError(
                    f"{label} route_policy rows must be "
                    "{origin,destination,probs,costs} mappings"
                )
            od_pair = (int(row["origin"]), int(row["destination"]))
            if od_pair in probability_by_od:
                raise ValueError(f"{label} route_policy has duplicate OD row {od_pair}")
            probs = np.asarray(row["probs"], dtype=np.float32).reshape(-1)
            costs = np.asarray(row["costs"], dtype=np.float32).reshape(-1)
            if probs.size != HDV_RESPONSE_K:
                raise ValueError(
                    f"{label} route_policy probs width mismatch for {od_pair}: "
                    f"{probs.size} vs {HDV_RESPONSE_K}"
                )
            if costs.size != HDV_RESPONSE_K:
                raise ValueError(
                    f"{label} route_policy costs width mismatch for {od_pair}: "
                    f"{costs.size} vs {HDV_RESPONSE_K}"
                )
            if not np.isfinite(probs).all() or np.any(probs < 0.0):
                raise ValueError(f"{label} route_policy probs must be finite and non-negative")
            if float(probs.sum()) <= 1e-12:
                raise ValueError(f"{label} route_policy probs rows must have positive mass")
            if not np.isfinite(costs).all() or np.any(costs < 0.0):
                raise ValueError(f"{label} route_policy costs must be finite and non-negative")
            probability_by_od[od_pair] = probs
            cost_by_od[od_pair] = costs
        probs_rows: list[np.ndarray] = []
        cost_rows: list[np.ndarray] = []
        for od_pair in self.ods:
            if od_pair not in probability_by_od:
                raise ValueError(f"{label} route_policy is missing OD row {od_pair}")
            probs_rows.append(probability_by_od[od_pair])
            cost_rows.append(cost_by_od[od_pair])
        return (
            _normalise_rows(np.stack(probs_rows, axis=0)).astype(np.float32),
            np.stack(cost_rows, axis=0).astype(np.float32),
        )

    def _load_deployment_checkpoint(self, path: Path) -> Mapping[str, Any]:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(state, Mapping):
            raise ValueError("HDV response deployment checkpoint must be a mapping")
        required_keys = {
            "format",
            "deployment_mode",
            "parameterization",
            "selection_split",
            "route_target",
            "travel_time_feedback",
            "selected_residual_scale",
            "anchor_floor",
            "n_routes",
            "ods",
            "edge_ids",
            "hidden_dim",
            "depth",
            "progress_denominator",
            "trained_pairs",
            "model_state_dict",
            "initial_hdv_state",
        }
        missing = sorted(required_keys - set(state))
        if missing:
            raise ValueError(
                "HDV response deployment checkpoint is missing keys: "
                + ", ".join(missing)
            )
        return state

    def _load_deployment_state(self, state: Mapping[str, Any]) -> None:
        if state["format"] != HDV_RESPONSE_CHECKPOINT_FORMAT:
            raise ValueError("HDV response checkpoint must use the strict v4 format")
        deployment_mode = str(state["deployment_mode"])
        if deployment_mode != "persistence_anchored_residual_od_route":
            raise ValueError(
                "HDV response deployment_mode must be "
                "'persistence_anchored_residual_od_route'"
            )
        if str(state["parameterization"]) != (
            "empirical_persistence_anchored_route_residual"
        ):
            raise ValueError(
                "HDV response parameterization does not match the v4 mainline"
            )
        if str(state["selection_split"]) != "train":
            raise ValueError("HDV response selection_split must be 'train'")
        if str(state["route_target"]) != "seen_hdv_exact_hdv_action":
            raise ValueError("HDV response route_target does not match the mainline")
        if str(state["travel_time_feedback"]) != (
            "seen AV policy_action plus seen HDV hdv_action"
        ):
            raise ValueError(
                "HDV response travel_time_feedback does not match the mainline"
            )
        if int(state["n_routes"]) != HDV_RESPONSE_K:
            raise ValueError(
                f"HDV deployment checkpoint n_routes must be {HDV_RESPONSE_K}"
            )
        if int(state["hidden_dim"]) != DEFAULT_HDV_RESPONSE_HIDDEN_DIM:
            raise ValueError(
                "HDV deployment checkpoint hidden_dim must be "
                f"{DEFAULT_HDV_RESPONSE_HIDDEN_DIM}"
            )
        if int(state["depth"]) != DEFAULT_HDV_RESPONSE_DEPTH:
            raise ValueError(
                f"HDV deployment checkpoint depth must be {DEFAULT_HDV_RESPONSE_DEPTH}"
            )
        self.residual_scale = float(state["selected_residual_scale"])
        if not np.isfinite(self.residual_scale) or self.residual_scale <= 0.0:
            raise ValueError(
                "HDV deployment selected_residual_scale must be finite and positive"
            )
        self.anchor_floor = float(state["anchor_floor"])
        if (
            not np.isfinite(self.anchor_floor)
            or self.anchor_floor < 0.0
            or self.anchor_floor >= 1.0 / HDV_RESPONSE_K
        ):
            raise ValueError(
                "HDV deployment anchor_floor must be in [0, 1 / n_routes)"
            )
        self.progress_denominator = int(state["progress_denominator"])
        if self.progress_denominator <= 0:
            raise ValueError(
                "HDV deployment progress_denominator must be a positive integer"
            )
        checkpoint_ods = tuple((int(row[0]), int(row[1])) for row in state["ods"])
        checkpoint_edge_ids = tuple(str(edge_id) for edge_id in state["edge_ids"])
        if checkpoint_ods != self.ods:
            raise ValueError(
                f"HDV checkpoint ODs do not match topology: {checkpoint_ods} vs {self.ods}"
            )
        if checkpoint_edge_ids != self.edge_ids:
            raise ValueError("HDV checkpoint edge_ids do not match topology")
        deployment_initial_state = state["initial_hdv_state"]
        if not isinstance(deployment_initial_state, str) or not deployment_initial_state:
            raise ValueError(
                "HDV deployment initial_hdv_state provenance must be a "
                "non-empty path string"
            )
        self.deployment_initial_hdv_state = deployment_initial_state
        self.deployment_mode = deployment_mode
        self.deployment_training_pairs = int(state["trained_pairs"])
        if self.deployment_training_pairs < 0:
            raise ValueError("HDV deployment trained_pairs must be non-negative")
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.model.eval()

    def _od_weight_from_counts(self, route_counts: np.ndarray) -> np.ndarray:
        totals = np.asarray(route_counts, dtype=np.float32).sum(axis=-1)
        if float(totals.sum()) > 1e-12:
            return (totals / totals.sum()).astype(np.float32)
        fallback = self.human_od_counts.clip(min=0.0)
        if float(fallback.sum()) > 1e-12:
            return (fallback / fallback.sum()).astype(np.float32)
        return np.full((self.n_od,), 1.0 / max(self.n_od, 1), dtype=np.float32)

    def _project_route_to_edges(
        self,
        route_probs: np.ndarray,
        od_weight: np.ndarray,
    ) -> np.ndarray:
        route_distribution = _normalise_rows(route_probs)
        weight = np.asarray(od_weight, dtype=np.float64).reshape(-1)
        if weight.shape != (self.n_od,):
            raise ValueError(
                f"OD projection weight mismatch: {weight.shape} vs {(self.n_od,)}"
            )
        edge_distribution = np.zeros((self.n_edges,), dtype=np.float64)
        for od_index, route_row in enumerate(self.route_edge_indices):
            od_mass = max(float(weight[od_index]), 0.0)
            if od_mass <= 0.0:
                continue
            for route_index, edge_indices in enumerate(route_row):
                edge_mass = od_mass * float(route_distribution[od_index, route_index])
                if edge_mass <= 0.0:
                    continue
                np.add.at(
                    edge_distribution,
                    np.asarray(edge_indices, dtype=np.int64),
                    edge_mass,
                )
        return _normalise_vector(edge_distribution).astype(np.float32)

    def _progress_feature(self, episode: int) -> float:
        episode_index = max(int(episode), 0)
        return float(np.clip(episode_index / self.progress_denominator, 0.0, 1.0))

    def _build_response_features(
        self,
        *,
        empirical_route: np.ndarray,
        route_before: np.ndarray,
        route_after: np.ndarray,
        route_cost_before: np.ndarray,
        filled_route_tt: np.ndarray,
        route_tt_observed: np.ndarray,
        route_counts: np.ndarray,
        route_tt_sample_counts: np.ndarray,
        od_weight: np.ndarray,
        progress: float,
    ) -> np.ndarray:
        empirical = _normalise_rows(empirical_route).astype(np.float32)
        before = _normalise_rows(route_before).astype(np.float32)
        after = _normalise_rows(route_after).astype(np.float32)
        cost_before = np.asarray(route_cost_before, dtype=np.float32)
        travel_time = np.asarray(filled_route_tt, dtype=np.float32)
        observed_mask = np.asarray(route_tt_observed, dtype=np.float32)
        route_count = np.asarray(route_counts, dtype=np.float32).clip(min=0.0)
        tt_count = np.asarray(route_tt_sample_counts, dtype=np.float32).clip(min=0.0)
        expected_shape = (self.n_od, HDV_RESPONSE_K)
        for name, value in (
            ("route_before", before),
            ("route_after", after),
            ("route_cost_before", cost_before),
            ("filled_route_tt", travel_time),
            ("route_tt_observed", observed_mask),
            ("route_counts", route_count),
            ("route_tt_sample_counts", tt_count),
        ):
            if value.shape != expected_shape:
                raise ValueError(f"{name} shape mismatch: {value.shape} vs {expected_shape}")
        observed_count = np.maximum(observed_mask.sum(axis=-1, keepdims=True), 1.0)
        travel_scale = np.maximum(
            (travel_time * observed_mask).sum(axis=-1, keepdims=True) / observed_count,
            1.0,
        )
        cost_gap = cost_before - np.nanmin(cost_before, axis=-1, keepdims=True)
        cost_feature = np.nan_to_num(
            cost_gap / travel_scale,
            nan=0.0,
            posinf=10.0,
            neginf=0.0,
        )
        tt_feature = np.nan_to_num(
            travel_time / travel_scale,
            nan=0.0,
            posinf=10.0,
            neginf=0.0,
        )
        route_delta = after - before
        branch_disagreement = 0.5 * np.abs(empirical - after).sum(axis=-1, keepdims=True)
        latent_response = 0.5 * np.abs(before - after).sum(axis=-1, keepdims=True)
        empirical_entropy = -(
            empirical * np.log(np.clip(empirical, 1e-12, 1.0))
        ).sum(axis=-1, keepdims=True) / max(np.log(float(HDV_RESPONSE_K)), 1e-12)
        route_count_total = route_count.sum(axis=-1, keepdims=True)
        tt_count_total = tt_count.sum(axis=-1, keepdims=True)
        weight_column = np.asarray(od_weight, dtype=np.float32).reshape(self.n_od, 1)
        progress_column = np.full(
            (self.n_od, 1),
            float(np.clip(progress, 0.0, 1.0)),
            dtype=np.float32,
        )
        features = np.concatenate(
            [
                empirical,
                before,
                after,
                route_delta,
                np.clip(cost_feature, 0.0, 10.0).astype(np.float32),
                np.clip(tt_feature, 0.0, 10.0).astype(np.float32),
                observed_mask,
                np.log1p(route_count).astype(np.float32),
                np.log1p(tt_count).astype(np.float32),
                branch_disagreement.astype(np.float32),
                latent_response.astype(np.float32),
                empirical_entropy.astype(np.float32),
                np.log1p(route_count_total).astype(np.float32),
                np.log1p(tt_count_total).astype(np.float32),
                weight_column,
                progress_column,
            ],
            axis=-1,
        )
        if features.shape != (self.n_od, HDV_RESPONSE_FEATURE_DIM):
            raise AssertionError(
                f"v4 feature shape mismatch: {features.shape} vs "
                f"{(self.n_od, HDV_RESPONSE_FEATURE_DIM)}"
            )
        return features.astype(np.float32)

    def _process_episode(self, observation: EpisodeObservation) -> _ProcessedEpisode:
        route_before, route_cost_before = self._route_policy_rows_to_arrays(
            observation.pre_state,
            label="pre_state",
        )
        route_after, _route_cost_after = self._route_policy_rows_to_arrays(
            observation.post_state,
            label="post_state",
        )
        route_counts = np.zeros((self.n_od, HDV_RESPONSE_K), dtype=np.float32)
        route_tt_sum = np.zeros((self.n_od, HDV_RESPONSE_K), dtype=np.float32)
        route_tt_sample_counts = np.zeros((self.n_od, HDV_RESPONSE_K), dtype=np.float32)
        travel_times = {
            str(vehicle_id): float(seconds)
            for vehicle_id, seconds in observation.travel_times_seconds.items()
        }
        if set(observation.hdv_route_choices) != set(self.human_vehicle_to_od):
            raise ValueError(
                "HDV route choices must cover exactly the topology's human vehicles"
            )
        for vehicle_id, route_index in observation.hdv_route_choices.items():
            route_slot = int(route_index)
            if route_slot < 0 or route_slot >= HDV_RESPONSE_K:
                raise ValueError(
                    f"route choice for vehicle {vehicle_id!r} is out of range: {route_slot}"
                )
            od_index = self.human_vehicle_to_od[str(vehicle_id)]
            route_counts[od_index, route_slot] += 1.0
        for vehicle_id, route_index in observation.travel_time_route_choices.items():
            vehicle_key = str(vehicle_id)
            if vehicle_key not in self.vehicle_to_od:
                raise ValueError(
                    f"travel-time route choice references unknown vehicle "
                    f"{vehicle_id!r}"
                )
            route_slot = int(route_index)
            if route_slot < 0 or route_slot >= HDV_RESPONSE_K:
                raise ValueError(
                    f"travel-time route choice for vehicle {vehicle_id!r} "
                    f"is out of range: {route_slot}"
                )
            if vehicle_key not in travel_times:
                raise ValueError(
                    f"travel time is missing for route-labeled vehicle "
                    f"{vehicle_id!r}"
                )
            travel_time = float(travel_times[vehicle_key])
            if not np.isfinite(travel_time) or travel_time < 0.0:
                raise ValueError(
                    f"travel time for vehicle {vehicle_id!r} must be "
                    "finite and non-negative"
                )
            od_index = self.vehicle_to_od[vehicle_key]
            route_tt_sum[od_index, route_slot] += travel_time
            route_tt_sample_counts[od_index, route_slot] += 1.0
        actual_route = route_before.copy()
        observed_rows = route_counts.sum(axis=-1) > 1e-12
        if bool(np.any(observed_rows)):
            actual_route[observed_rows] = _normalise_rows(route_counts[observed_rows])
        route_tt_observed = (route_tt_sample_counts > 0.0).astype(np.float32)
        filled_route_tt = np.zeros_like(route_cost_before, dtype=np.float32)
        observed_cells = route_tt_sample_counts > 0.0
        for od_index in range(self.n_od):
            observed_routes = observed_cells[od_index]
            if not bool(np.any(observed_routes)):
                continue
            observed_means = (
                route_tt_sum[od_index, observed_routes]
                / route_tt_sample_counts[od_index, observed_routes]
            )
            filled_route_tt[od_index] = float(observed_means.max()) * 1.5
            filled_route_tt[od_index, observed_routes] = observed_means
        od_weight = self._od_weight_from_counts(route_counts)
        edge_distribution = self._project_route_to_edges(actual_route, od_weight)
        return _ProcessedEpisode(
            episode=int(observation.episode),
            route_counts=route_counts,
            actual_route=actual_route.astype(np.float32),
            route_before=route_before.astype(np.float32),
            route_after=route_after.astype(np.float32),
            route_cost_before=route_cost_before.astype(np.float32),
            filled_route_tt=filled_route_tt.astype(np.float32),
            route_tt_observed=route_tt_observed.astype(np.float32),
            route_tt_sample_counts=route_tt_sample_counts.astype(np.float32),
            edge_distribution=edge_distribution.astype(np.float32),
            od_weight=od_weight.astype(np.float32),
        )

    def predict_for_next_episode(
        self,
        episode: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if self.previous_observation is None:
            self.pending_predicted_route = self.last_observed_route.copy()
            self.pending_prediction_episode = int(episode)
            self.pending_training_sample = None
            return self._project_route_to_edges(
                self.last_observed_route,
                self._od_weight_from_counts(np.zeros((self.n_od, HDV_RESPONSE_K), dtype=np.float32)),
            ), {
                "hdv_response_pred_enabled": 1.0,
                "hdv_response_pred_source_code": 1.0,
                "hdv_response_pred_route_delta_tv": 0.0,
            }

        processed = self._process_episode(self.previous_observation)
        features = self._build_response_features(
            empirical_route=self.last_observed_route,
            route_before=processed.route_before,
            route_after=processed.route_after,
            route_cost_before=processed.route_cost_before,
            filled_route_tt=processed.filled_route_tt,
            route_tt_observed=processed.route_tt_observed,
            route_counts=processed.route_counts,
            route_tt_sample_counts=processed.route_tt_sample_counts,
            od_weight=processed.od_weight,
            progress=self._progress_feature(int(episode)),
        )
        with torch.no_grad():
            residual = self.model(
                torch.from_numpy(features).to(device=self.device, dtype=torch.float32)
            ).cpu().numpy()
        persistence = _normalise_rows(self.last_observed_route)
        predicted_route = apply_persistence_anchored_route_residual(
            persistence,
            residual,
            scale=self.residual_scale,
            anchor_floor=self.anchor_floor,
        ).astype(np.float32)
        projected_persistence = self._project_route_to_edges(persistence, processed.od_weight)
        projected_prediction = self._project_route_to_edges(
            predicted_route,
            processed.od_weight,
        )
        predicted_edge = _normalise_vector(
            np.clip(
                np.asarray(processed.edge_distribution, dtype=np.float64)
                + np.asarray(projected_prediction, dtype=np.float64)
                - np.asarray(projected_persistence, dtype=np.float64),
                0.0,
                None,
            )
        ).astype(np.float32)
        route_delta = 0.5 * np.abs(predicted_route - persistence).sum(axis=-1)
        self.pending_predicted_route = predicted_route.copy()
        self.pending_prediction_episode = int(episode)
        self.pending_training_sample = {
            "features": features.astype(np.float32),
            "persistence": persistence.astype(np.float32),
        }
        return predicted_edge, {
            "hdv_response_pred_enabled": 1.0,
            "hdv_response_pred_source_code": 4.0,
            "hdv_response_pred_residual_l2": float(
                (
                    np.linalg.norm(np.asarray(residual, dtype=np.float64), axis=-1)
                    * np.asarray(processed.od_weight, dtype=np.float64)
                ).sum()
            ),
            "hdv_response_pred_route_delta_tv": float(
                (route_delta * np.asarray(processed.od_weight, dtype=np.float64)).sum()
            ),
        }

    def observe_episode(
        self,
        observation: EpisodeObservation,
        collect_online_sample: bool = False,
    ) -> dict[str, float]:
        processed = self._process_episode(observation)
        metrics: dict[str, float] = {}
        if (
            self.pending_predicted_route is not None
            and self.pending_prediction_episode == int(observation.episode)
        ):
            actual_route = np.asarray(self.last_observed_route, dtype=np.float64).copy()
            observed_rows = processed.route_counts.sum(axis=-1) > 1e-12
            if bool(np.any(observed_rows)):
                actual_route[observed_rows] = processed.actual_route[observed_rows]
            route_tv = 0.5 * np.abs(
                _normalise_rows(self.pending_predicted_route) - _normalise_rows(actual_route)
            ).sum(axis=-1)
            observed_fraction = float(np.mean(observed_rows.astype(np.float64)))
            weight = np.asarray(processed.od_weight, dtype=np.float64)
            metrics = {
                "hdv_response_pred_route_tv": float((route_tv * weight).sum()),
                "hdv_response_pred_route_observed_od_fraction": observed_fraction,
            }
            if collect_online_sample and self.pending_training_sample is not None:
                self.online_training_samples.append(
                    {
                        "features": np.asarray(
                            self.pending_training_sample["features"],
                            dtype=np.float32,
                        ),
                        "persistence": np.asarray(
                            self.pending_training_sample["persistence"],
                            dtype=np.float32,
                        ),
                        "target": np.asarray(processed.actual_route, dtype=np.float32),
                        "od_weight": np.asarray(processed.od_weight, dtype=np.float32),
                    }
                )
                if len(self.online_training_samples) > ONLINE_UPDATE_BUFFER_EPISODES:
                    self.online_training_samples = self.online_training_samples[
                        -ONLINE_UPDATE_BUFFER_EPISODES:
                    ]
                self.trained_pairs += 1
        self.last_observed_route = np.asarray(processed.actual_route, dtype=np.float32)
        self.previous_observation = EpisodeObservation.from_state_dict(
            observation.state_dict()
        )
        self.pending_predicted_route = None
        self.pending_prediction_episode = None
        self.pending_training_sample = None
        return metrics

    def update_online(self) -> dict[str, float]:
        if len(self.online_training_samples) < ONLINE_UPDATE_MIN_PAIRS:
            return {
                "hdv_response_loss": 0.0,
                "hdv_response_updates": 0.0,
                "hdv_response_online_pairs": float(len(self.online_training_samples)),
                "hdv_response_online_train_tv": 0.0,
                "hdv_response_online_update_steps": float(self.online_update_steps),
            }
        samples = self.online_training_samples[-ONLINE_UPDATE_BUFFER_EPISODES:]
        features = torch.from_numpy(
            np.stack([sample["features"] for sample in samples], axis=0)
        ).to(device=self.device, dtype=torch.float32)
        persistence = torch.from_numpy(
            np.stack([sample["persistence"] for sample in samples], axis=0)
        ).to(device=self.device, dtype=torch.float32)
        target = torch.from_numpy(
            np.stack([sample["target"] for sample in samples], axis=0)
        ).to(device=self.device, dtype=torch.float32)
        od_weight = torch.from_numpy(
            np.stack([sample["od_weight"] for sample in samples], axis=0)
        ).to(device=self.device, dtype=torch.float32)
        self.model.train()
        losses: list[float] = []
        for _ in range(ONLINE_UPDATE_EPOCHS):
            residual = self.model(features)
            prediction = apply_persistence_anchored_route_residual_torch(
                persistence,
                residual,
                scale=self.residual_scale,
                anchor_floor=self.anchor_floor,
            )
            route_tv = 0.5 * torch.abs(prediction - target).sum(dim=-1)
            route_loss = (route_tv * od_weight).sum(dim=-1).mean()
            l2_penalty = residual.square().mean()
            loss = route_loss + ONLINE_UPDATE_L2 * l2_penalty
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                ONLINE_UPDATE_GRAD_CLIP,
            )
            self.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        self.model.eval()
        self.online_update_steps += ONLINE_UPDATE_EPOCHS
        with torch.no_grad():
            residual = self.model(features)
            prediction = apply_persistence_anchored_route_residual_torch(
                persistence,
                residual,
                scale=self.residual_scale,
                anchor_floor=self.anchor_floor,
            )
            route_tv = 0.5 * torch.abs(prediction - target).sum(dim=-1)
            train_tv = (route_tv * od_weight).sum(dim=-1).mean()
        return {
            "hdv_response_loss": float(np.mean(losses)),
            "hdv_response_updates": float(ONLINE_UPDATE_EPOCHS),
            "hdv_response_online_pairs": float(len(samples)),
            "hdv_response_online_train_tv": float(train_tv.detach().cpu().item()),
            "hdv_response_online_update_steps": float(self.online_update_steps),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": HDV_RESPONSE_CHECKPOINT_FORMAT,
            "architecture": {
                "n_routes": HDV_RESPONSE_K,
                "input_dim": HDV_RESPONSE_FEATURE_DIM,
                "hidden_dim": DEFAULT_HDV_RESPONSE_HIDDEN_DIM,
                "depth": DEFAULT_HDV_RESPONSE_DEPTH,
                "anchor_floor": float(self.anchor_floor),
            },
            "deployment": {
                "deployment_mode": self.deployment_mode,
                "selected_residual_scale": float(self.residual_scale),
                "progress_denominator": int(self.progress_denominator),
                "deployment_training_pairs": int(
                    self.deployment_training_pairs
                ),
            },
            "topology": {
                "ods": [list(od_pair) for od_pair in self.ods],
                "edge_ids": list(self.edge_ids),
                "route_edge_indices": [
                    [list(route) for route in od_routes]
                    for od_routes in self.route_edge_indices
                ],
            },
            "initial_route_policy": self.initial_route_policy.astype(np.float32),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "runtime": {
                "last_observed_route": self.last_observed_route.astype(np.float32),
                "previous_observation": (
                    None
                    if self.previous_observation is None
                    else self.previous_observation.state_dict()
                ),
                "pending_predicted_route": (
                    None
                    if self.pending_predicted_route is None
                    else self.pending_predicted_route.astype(np.float32)
                ),
                "pending_prediction_episode": self.pending_prediction_episode,
                "pending_training_sample": (
                    None
                    if self.pending_training_sample is None
                    else {
                        "features": np.asarray(
                            self.pending_training_sample["features"],
                            dtype=np.float32,
                        ),
                        "persistence": np.asarray(
                            self.pending_training_sample["persistence"],
                            dtype=np.float32,
                        ),
                    }
                ),
                "online_training_samples": [
                    {
                        "features": np.asarray(sample["features"], dtype=np.float32),
                        "persistence": np.asarray(
                            sample["persistence"],
                            dtype=np.float32,
                        ),
                        "target": np.asarray(sample["target"], dtype=np.float32),
                        "od_weight": np.asarray(sample["od_weight"], dtype=np.float32),
                    }
                    for sample in self.online_training_samples
                ],
            },
            "counters": {
                "trained_pairs": int(self.trained_pairs),
                "online_update_steps": int(self.online_update_steps),
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected_keys = {
            "format",
            "architecture",
            "deployment",
            "topology",
            "initial_route_policy",
            "model_state_dict",
            "optimizer_state_dict",
            "runtime",
            "counters",
        }
        if not isinstance(state, Mapping) or set(state) != expected_keys:
            raise ValueError("HDV response checkpoint has an invalid schema")
        if state["format"] != HDV_RESPONSE_CHECKPOINT_FORMAT:
            raise ValueError("HDV response checkpoint must use the strict v4 format")
        architecture = state["architecture"]
        deployment = state["deployment"]
        topology_state = state["topology"]
        runtime = state["runtime"]
        counters = state["counters"]
        if not isinstance(architecture, Mapping):
            raise ValueError("HDV checkpoint architecture must be a mapping")
        if not isinstance(deployment, Mapping):
            raise ValueError("HDV checkpoint deployment must be a mapping")
        if not isinstance(topology_state, Mapping):
            raise ValueError("HDV checkpoint topology must be a mapping")
        if not isinstance(runtime, Mapping):
            raise ValueError("HDV checkpoint runtime must be a mapping")
        if not isinstance(counters, Mapping):
            raise ValueError("HDV checkpoint counters must be a mapping")
        expected_architecture = {
            "n_routes": HDV_RESPONSE_K,
            "input_dim": HDV_RESPONSE_FEATURE_DIM,
            "hidden_dim": DEFAULT_HDV_RESPONSE_HIDDEN_DIM,
            "depth": DEFAULT_HDV_RESPONSE_DEPTH,
            "anchor_floor": HDV_RESPONSE_ANCHOR_FLOOR,
        }
        if set(architecture) != set(expected_architecture):
            raise ValueError("HDV checkpoint architecture has invalid keys")
        for key, expected_value in expected_architecture.items():
            value = architecture[key]
            if isinstance(expected_value, float):
                if not np.isclose(float(value), float(expected_value)):
                    raise ValueError(
                        f"HDV checkpoint architecture mismatch for {key}: "
                        f"{value} vs {expected_value}"
                    )
            elif int(value) != int(expected_value):
                raise ValueError(
                    f"HDV checkpoint architecture mismatch for {key}: "
                    f"{value} vs {expected_value}"
                )
        expected_deployment_keys = {
            "deployment_mode",
            "selected_residual_scale",
            "progress_denominator",
            "deployment_training_pairs",
        }
        if set(deployment) != expected_deployment_keys:
            raise ValueError("HDV checkpoint deployment has invalid keys")
        if str(deployment["deployment_mode"]) != self.deployment_mode:
            raise ValueError("HDV checkpoint deployment_mode does not match the constructor payload")
        if not np.isclose(float(deployment["selected_residual_scale"]), self.residual_scale):
            raise ValueError("HDV checkpoint selected_residual_scale does not match the constructor payload")
        if int(deployment["progress_denominator"]) != self.progress_denominator:
            raise ValueError("HDV checkpoint progress_denominator does not match the constructor payload")
        if (
            int(deployment["deployment_training_pairs"])
            != self.deployment_training_pairs
        ):
            raise ValueError(
                "HDV checkpoint deployment_training_pairs does not match "
                "the constructor payload"
            )
        if set(topology_state) != {"ods", "edge_ids", "route_edge_indices"}:
            raise ValueError("HDV checkpoint topology has invalid keys")
        checkpoint_ods = tuple(
            (int(row[0]), int(row[1])) for row in topology_state["ods"]
        )
        checkpoint_edge_ids = tuple(str(edge_id) for edge_id in topology_state["edge_ids"])
        checkpoint_routes = tuple(
            tuple(tuple(int(edge_index) for edge_index in route) for route in od_routes)
            for od_routes in topology_state["route_edge_indices"]
        )
        if checkpoint_ods != self.ods:
            raise ValueError(
                f"HDV checkpoint ODs do not match topology: {checkpoint_ods} vs {self.ods}"
            )
        if checkpoint_edge_ids != self.edge_ids:
            raise ValueError("HDV checkpoint edge_ids do not match topology")
        if checkpoint_routes != self.route_edge_indices:
            raise ValueError("HDV checkpoint route_edge_indices do not match topology")
        initial_route_policy = _normalise_rows(
            np.asarray(state["initial_route_policy"], dtype=np.float32)
        ).astype(np.float32)
        if initial_route_policy.shape != self.initial_route_policy.shape:
            raise ValueError(
                "HDV checkpoint initial_route_policy shape mismatch: "
                f"{initial_route_policy.shape} vs {self.initial_route_policy.shape}"
            )
        if not np.allclose(initial_route_policy, self.initial_route_policy):
            raise ValueError(
                "HDV checkpoint initial_route_policy does not match the current "
                "initial HDV state route_policy"
            )
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        for group in self.optimizer.param_groups:
            if not np.isclose(float(group["lr"]), ONLINE_UPDATE_LR):
                raise ValueError("HDV checkpoint optimizer lr does not match v4 mainline")
            if not np.isclose(
                float(group["weight_decay"]),
                ONLINE_UPDATE_WEIGHT_DECAY,
            ):
                raise ValueError(
                    "HDV checkpoint optimizer weight_decay does not match v4 mainline"
                )
        if set(runtime) != {
            "last_observed_route",
            "previous_observation",
            "pending_predicted_route",
            "pending_prediction_episode",
            "pending_training_sample",
            "online_training_samples",
        }:
            raise ValueError("HDV checkpoint runtime has invalid keys")
        if set(counters) != {"trained_pairs", "online_update_steps"}:
            raise ValueError("HDV checkpoint counters have invalid keys")
        self._reset_runtime()
        last_observed_route = np.asarray(runtime["last_observed_route"], dtype=np.float32)
        if last_observed_route.shape != (self.n_od, HDV_RESPONSE_K):
            raise ValueError(
                "HDV checkpoint last_observed_route shape mismatch: "
                f"{last_observed_route.shape} vs {(self.n_od, HDV_RESPONSE_K)}"
            )
        self.last_observed_route = _normalise_rows(last_observed_route).astype(np.float32)
        previous_observation = runtime["previous_observation"]
        self.previous_observation = (
            None
            if previous_observation is None
            else EpisodeObservation.from_state_dict(previous_observation)
        )
        pending_predicted_route = runtime["pending_predicted_route"]
        pending_prediction_episode = runtime["pending_prediction_episode"]
        if pending_predicted_route is None:
            self.pending_predicted_route = None
            if pending_prediction_episode is not None:
                raise ValueError(
                    "HDV checkpoint pending_prediction_episode requires a "
                    "pending_predicted_route"
                )
        else:
            restored_pending = np.asarray(pending_predicted_route, dtype=np.float32)
            if restored_pending.shape != (self.n_od, HDV_RESPONSE_K):
                raise ValueError(
                    "HDV checkpoint pending_predicted_route shape mismatch: "
                    f"{restored_pending.shape} vs {(self.n_od, HDV_RESPONSE_K)}"
                )
            self.pending_predicted_route = _normalise_rows(restored_pending).astype(
                np.float32
            )
            if pending_prediction_episode is None:
                raise ValueError(
                    "HDV checkpoint pending_predicted_route requires "
                    "pending_prediction_episode"
                )
        self.pending_prediction_episode = (
            None if pending_prediction_episode is None else int(pending_prediction_episode)
        )
        pending_training_sample = runtime["pending_training_sample"]
        if pending_training_sample is None:
            self.pending_training_sample = None
        else:
            if not isinstance(pending_training_sample, Mapping) or set(
                pending_training_sample
            ) != {"features", "persistence"}:
                raise ValueError("HDV checkpoint pending_training_sample has invalid keys")
            features = np.asarray(
                pending_training_sample["features"],
                dtype=np.float32,
            )
            persistence = np.asarray(
                pending_training_sample["persistence"],
                dtype=np.float32,
            )
            if features.shape != (self.n_od, HDV_RESPONSE_FEATURE_DIM):
                raise ValueError(
                    "HDV checkpoint pending features shape mismatch: "
                    f"{features.shape} vs {(self.n_od, HDV_RESPONSE_FEATURE_DIM)}"
                )
            if persistence.shape != (self.n_od, HDV_RESPONSE_K):
                raise ValueError(
                    "HDV checkpoint pending persistence shape mismatch: "
                    f"{persistence.shape} vs {(self.n_od, HDV_RESPONSE_K)}"
                )
            self.pending_training_sample = {
                "features": features.astype(np.float32),
                "persistence": _normalise_rows(persistence).astype(np.float32),
            }
        restored_samples: list[dict[str, np.ndarray]] = []
        online_training_samples = runtime["online_training_samples"]
        if not isinstance(online_training_samples, list):
            raise ValueError("HDV checkpoint online_training_samples must be a list")
        for sample in online_training_samples:
            if not isinstance(sample, Mapping) or set(sample) != {
                "features",
                "persistence",
                "target",
                "od_weight",
            }:
                raise ValueError("HDV checkpoint online training sample has invalid keys")
            features = np.asarray(sample["features"], dtype=np.float32)
            persistence = np.asarray(sample["persistence"], dtype=np.float32)
            target = np.asarray(sample["target"], dtype=np.float32)
            od_weight = np.asarray(sample["od_weight"], dtype=np.float32).reshape(-1)
            if features.shape != (self.n_od, HDV_RESPONSE_FEATURE_DIM):
                raise ValueError("HDV checkpoint online sample features shape mismatch")
            if persistence.shape != (self.n_od, HDV_RESPONSE_K):
                raise ValueError(
                    "HDV checkpoint online sample persistence shape mismatch"
                )
            if target.shape != (self.n_od, HDV_RESPONSE_K):
                raise ValueError("HDV checkpoint online sample target shape mismatch")
            if od_weight.shape != (self.n_od,):
                raise ValueError("HDV checkpoint online sample od_weight shape mismatch")
            restored_samples.append(
                {
                    "features": features.astype(np.float32),
                    "persistence": _normalise_rows(persistence).astype(np.float32),
                    "target": _normalise_rows(target).astype(np.float32),
                    "od_weight": od_weight.astype(np.float32),
                }
            )
        self.online_training_samples = restored_samples[-ONLINE_UPDATE_BUFFER_EPISODES:]
        self.trained_pairs = int(counters["trained_pairs"])
        self.online_update_steps = int(counters["online_update_steps"])
        self.model.eval()


__all__ = [
    "HDV_RESPONSE_CHECKPOINT_FORMAT",
    "EpisodeObservation",
    "HdvOdRouteResponsePredictor",
]
