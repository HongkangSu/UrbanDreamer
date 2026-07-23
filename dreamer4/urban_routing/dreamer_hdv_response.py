"""HDV background and response helpers for Dreamer vehicle imagination.

The functions here keep episode-level HDV response logic out of
``train_marl.py``.  HDV response is treated as a static episode background
whose next OD-route distribution is inferred from the preceding episode's
realized route choices, costs, and travel-time feedback, not as a per-step
controllable policy.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


logger = logging.getLogger(__name__)

HDV_OD_ROUTE_PREDICTOR_FORMAT = "causal_persistence_gated_hdv_od_route_predictor_v3"
HDV_OD_ROUTE_RESIDUAL_PREDICTOR_FORMAT = (
    "causal_persistence_residual_hdv_od_route_predictor_v4"
)
HDV_OD_ROUTE_PREDICTOR_FORMATS = {
    HDV_OD_ROUTE_PREDICTOR_FORMAT,
    HDV_OD_ROUTE_RESIDUAL_PREDICTOR_FORMAT,
}
HDV_OD_ROUTE_RESIDUAL_ANCHOR_FLOOR = 1e-3


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).clip(min=0.0)
    if arr.ndim < 1:
        raise ValueError(f"route probabilities must have at least one dimension, got {arr.shape}")
    denom = arr.sum(axis=-1, keepdims=True)
    uniform = np.full_like(arr, 1.0 / max(int(arr.shape[-1]), 1))
    return np.where(denom > 1e-12, arr / np.maximum(denom, 1e-12), uniform)


def _normalise_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1).clip(min=0.0)
    total = float(arr.sum())
    return arr / total if total > 1e-12 else arr


def apply_persistence_anchored_route_residual(
    persistence: np.ndarray,
    raw_residual: np.ndarray,
    *,
    scale: float,
    anchor_floor: float = HDV_OD_ROUTE_RESIDUAL_ANCHOR_FLOOR,
) -> np.ndarray:
    """Apply a route residual while zero residual remains exact persistence."""
    previous = _normalise_rows(persistence)
    raw = np.asarray(raw_residual, dtype=np.float64)
    if raw.shape != previous.shape:
        raise ValueError(
            f"route residual shape mismatch: {raw.shape} vs {previous.shape}"
        )
    n_routes = int(previous.shape[-1])
    floor = float(anchor_floor)
    if floor < 0.0 or floor >= 1.0 / max(n_routes, 1):
        raise ValueError(
            f"anchor_floor must be in [0, 1 / n_routes), got {floor}"
        )
    anchor = (1.0 - float(n_routes) * floor) * previous + floor
    logits = np.log(np.clip(anchor, 1e-12, 1.0)) + float(scale) * raw
    logits = logits - logits.max(axis=-1, keepdims=True)
    response = _normalise_rows(np.exp(logits))
    return _normalise_rows(np.clip(previous + response - anchor, 0.0, None))


def apply_persistence_anchored_route_residual_torch(
    persistence: torch.Tensor,
    raw_residual: torch.Tensor,
    *,
    scale: float,
    anchor_floor: float = HDV_OD_ROUTE_RESIDUAL_ANCHOR_FLOOR,
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
        raise ValueError(
            f"anchor_floor must be in [0, 1 / n_routes), got {floor}"
        )
    anchor = (1.0 - float(n_routes) * floor) * previous + floor
    response = torch.softmax(
        torch.log(anchor.clamp_min(1e-12))
        + float(scale) * raw_residual.float(),
        dim=-1,
    )
    candidate = previous + response - anchor
    # Forward values stay on the simplex, while the straight-through clamp
    # keeps a usable gradient at persistence entries that are exactly zero.
    clipped = candidate.clamp_min(0.0)
    prediction = candidate + (clipped - candidate).detach()
    return prediction / prediction.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class PersistenceAnchoredOdRouteGate(nn.Module):
    """Choose how much of the exact RouteRL proposal to trust for each OD."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(int(depth), 1)):
            layers.extend(
                [
                    nn.Linear(dim, int(hidden_dim)),
                    nn.GELU(),
                    nn.LayerNorm(int(hidden_dim)),
                ]
            )
            dim = int(hidden_dim)
        output = nn.Linear(dim, 1)
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, -4.0)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))


class PersistenceAnchoredOdRouteResidual(nn.Module):
    """Predict an unrestricted route-logit response around persistence."""

    def __init__(
        self,
        input_dim: int,
        n_routes: int,
        hidden_dim: int,
        depth: int,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(int(depth), 1)):
            layers.extend(
                [
                    nn.Linear(dim, int(hidden_dim)),
                    nn.GELU(),
                    nn.LayerNorm(int(hidden_dim)),
                ]
            )
            dim = int(hidden_dim)
        output = nn.Linear(dim, int(n_routes))
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def build_persistence_gate_features(
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
    """Build the exact OD-local feature contract shared by v3 and v4."""
    empirical = _normalise_rows(empirical_route).astype(np.float32)
    latent_before = _normalise_rows(route_before).astype(np.float32)
    gawron_proposal = _normalise_rows(route_after).astype(np.float32)
    previous_cost = np.asarray(route_cost_before, dtype=np.float32)
    previous_tt = np.asarray(filled_route_tt, dtype=np.float32)
    tt_mask = np.asarray(route_tt_observed, dtype=np.float32)
    previous_counts = np.asarray(route_counts, dtype=np.float32).clip(min=0.0)
    tt_counts = np.asarray(route_tt_sample_counts, dtype=np.float32).clip(min=0.0)
    shape = empirical.shape
    for label, values in (
        ("route_before", latent_before),
        ("route_after", gawron_proposal),
        ("route_cost_before", previous_cost),
        ("filled_route_tt", previous_tt),
        ("route_tt_observed", tt_mask),
        ("route_counts", previous_counts),
        ("route_tt_sample_counts", tt_counts),
    ):
        if values.shape != shape:
            raise ValueError(f"{label} shape mismatch: {values.shape} vs {shape}")
    if empirical.ndim != 2:
        raise ValueError(f"OD-route arrays must be rank 2, got {empirical.shape}")

    observed_count = np.maximum(tt_mask.sum(axis=-1, keepdims=True), 1.0)
    tt_scale = np.maximum(
        (previous_tt * tt_mask).sum(axis=-1, keepdims=True) / observed_count,
        1.0,
    )
    cost_gap = previous_cost - np.nanmin(previous_cost, axis=-1, keepdims=True)
    cost_feature = np.nan_to_num(
        cost_gap / tt_scale,
        nan=0.0,
        posinf=10.0,
        neginf=0.0,
    )
    tt_feature = np.nan_to_num(
        previous_tt / tt_scale,
        nan=0.0,
        posinf=10.0,
        neginf=0.0,
    )
    latent_delta = gawron_proposal - latent_before
    branch_disagreement = 0.5 * np.abs(empirical - gawron_proposal).sum(
        axis=-1,
        keepdims=True,
    )
    latent_response = 0.5 * np.abs(latent_before - gawron_proposal).sum(
        axis=-1,
        keepdims=True,
    )
    n_routes = int(shape[-1])
    empirical_entropy = -(
        empirical * np.log(np.clip(empirical, 1e-12, 1.0))
    ).sum(axis=-1, keepdims=True) / max(np.log(max(n_routes, 2)), 1e-12)
    route_count_total = previous_counts.sum(axis=-1, keepdims=True)
    tt_count_total = tt_counts.sum(axis=-1, keepdims=True)
    weight = np.asarray(od_weight, dtype=np.float32).reshape(-1, 1)
    if weight.shape[0] != shape[0]:
        raise ValueError(f"OD weight shape mismatch: {weight.shape} vs {shape[:-1]}")
    progress_column = np.full(
        (shape[0], 1),
        float(np.clip(progress, 0.0, 1.0)),
        dtype=np.float32,
    )
    return np.concatenate(
        [
            empirical,
            latent_before,
            gawron_proposal,
            latent_delta,
            np.clip(cost_feature, 0.0, 10.0).astype(np.float32),
            np.clip(tt_feature, 0.0, 10.0).astype(np.float32),
            tt_mask,
            np.log1p(previous_counts).astype(np.float32),
            np.log1p(tt_counts).astype(np.float32),
            branch_disagreement.astype(np.float32),
            latent_response.astype(np.float32),
            empirical_entropy.astype(np.float32),
            np.log1p(route_count_total).astype(np.float32),
            np.log1p(tt_count_total).astype(np.float32),
            weight,
            progress_column,
        ],
        axis=-1,
    )


@dataclass
class HdvOdRouteEpisodeObservation:
    episode: int
    route_counts: np.ndarray
    route_before: np.ndarray
    route_cost_before: np.ndarray
    filled_route_tt: np.ndarray
    route_tt_observed: np.ndarray
    route_tt_sample_counts: np.ndarray
    route_after: np.ndarray
    edge_distribution: np.ndarray

    def state_dict(self) -> Dict[str, Any]:
        return {
            "episode": int(self.episode),
            "route_counts": np.asarray(self.route_counts, dtype=np.float32),
            "route_before": np.asarray(self.route_before, dtype=np.float32),
            "route_cost_before": np.asarray(self.route_cost_before, dtype=np.float32),
            "filled_route_tt": np.asarray(self.filled_route_tt, dtype=np.float32),
            "route_tt_observed": np.asarray(self.route_tt_observed, dtype=np.float32),
            "route_tt_sample_counts": np.asarray(
                self.route_tt_sample_counts,
                dtype=np.float32,
            ),
            "route_after": np.asarray(self.route_after, dtype=np.float32),
            "edge_distribution": np.asarray(self.edge_distribution, dtype=np.float32),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "HdvOdRouteEpisodeObservation":
        return cls(
            episode=int(state["episode"]),
            route_counts=np.asarray(state["route_counts"], dtype=np.float32),
            route_before=np.asarray(state["route_before"], dtype=np.float32),
            route_cost_before=np.asarray(state["route_cost_before"], dtype=np.float32),
            filled_route_tt=np.asarray(state["filled_route_tt"], dtype=np.float32),
            route_tt_observed=np.asarray(state["route_tt_observed"], dtype=np.float32),
            route_tt_sample_counts=np.asarray(
                state["route_tt_sample_counts"],
                dtype=np.float32,
            ),
            route_after=np.asarray(state["route_after"], dtype=np.float32),
            edge_distribution=np.asarray(state["edge_distribution"], dtype=np.float32),
        )


class PretrainedHdvOdRouteResponsePredictor:
    """Runtime adapter for persistence-anchored v3/v4 OD-route checkpoints."""

    def __init__(self, checkpoint: Mapping[str, Any], *, device: torch.device):
        self.format = str(checkpoint.get("format", ""))
        if self.format not in HDV_OD_ROUTE_PREDICTOR_FORMATS:
            raise ValueError(
                "unsupported HDV OD-route predictor format: "
                f"{checkpoint.get('format')!r}"
            )
        self.device = device
        self.n_routes = int(checkpoint["n_routes"])
        self.ods = tuple((int(row[0]), int(row[1])) for row in checkpoint["ods"])
        self.edge_ids = tuple(str(value) for value in checkpoint["edge_ids"])
        self.hidden_dim = int(checkpoint["hidden_dim"])
        self.depth = int(checkpoint["depth"])
        self.parameterization = (
            "persistence_residual"
            if self.format == HDV_OD_ROUTE_RESIDUAL_PREDICTOR_FORMAT
            else "persistence_gawron_gate"
        )
        anchor_floor = checkpoint.get("anchor_floor")
        if anchor_floor is None:
            anchor_floor = HDV_OD_ROUTE_RESIDUAL_ANCHOR_FLOOR
        self.anchor_floor = float(anchor_floor)
        selected_scale = checkpoint.get("selected_gate_scale")
        if selected_scale is None:
            selected_scale = checkpoint.get("selected_residual_scale", 0.0)
        self.selected_gate_scale = float(selected_scale)
        self.deployment_mode = str(checkpoint.get("deployment_mode", "persistence"))
        self.selection_split = str(checkpoint.get("selection_split", "validation"))
        self.pretrained_pairs = int(
            checkpoint.get(
                "pretrained_pairs",
                checkpoint.get("trained_pairs", 0),
            )
            or 0
        )
        self.progress_denominator = max(int(checkpoint.get("progress_denominator", 1)), 1)
        input_dim = 9 * int(self.n_routes) + 7
        if self.parameterization == "persistence_residual":
            self.model = PersistenceAnchoredOdRouteResidual(
                input_dim=input_dim,
                n_routes=self.n_routes,
                hidden_dim=self.hidden_dim,
                depth=self.depth,
            ).to(device)
        else:
            self.model = PersistenceAnchoredOdRouteGate(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                depth=self.depth,
            ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.route_edge_indices: Optional[Tuple[Tuple[Tuple[int, ...], ...], ...]] = None
        self.n_edges = len(self.edge_ids)
        self.last_observed_route: Optional[np.ndarray] = None
        self.previous_observation: Optional[HdvOdRouteEpisodeObservation] = None
        self.initial_edge_distribution: Optional[np.ndarray] = None
        self.pending_predicted_route: Optional[np.ndarray] = None
        self.pending_prediction_episode: Optional[int] = None
        self.pending_training_input: Optional[Dict[str, Any]] = None
        self.online_training_samples: List[Dict[str, np.ndarray]] = []
        self.online_trained_pairs = int(
            checkpoint.get("online_trained_pairs", 0) or 0
        )
        self.online_update_steps = int(
            checkpoint.get("online_update_steps", 0) or 0
        )
        self._pending_runtime_state: Optional[Mapping[str, Any]] = None

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "deployment_mode": self.deployment_mode,
            "parameterization": self.parameterization,
            "selection_split": self.selection_split,
            "trained_pairs": int(self.pretrained_pairs),
            "pretrained_pairs": int(self.pretrained_pairs),
            "selected_gate_scale": float(self.selected_gate_scale),
            "selected_residual_scale": (
                float(self.selected_gate_scale)
                if self.parameterization == "persistence_residual"
                else None
            ),
            "anchor_floor": float(self.anchor_floor),
            "online_trained_pairs": int(self.online_trained_pairs),
            "online_update_steps": int(self.online_update_steps),
            "n_routes": int(self.n_routes),
            "ods": [list(value) for value in self.ods],
            "edge_ids": list(self.edge_ids),
            "hidden_dim": int(self.hidden_dim),
            "depth": int(self.depth),
            "progress_denominator": int(self.progress_denominator),
            "model_state_dict": self.model.state_dict(),
        }

    @staticmethod
    def _od_rows(route_registry: Any) -> Tuple[Tuple[int, int], ...]:
        if hasattr(route_registry, "iter_od_rows"):
            return tuple(
                (int(key[0]), int(key[1]))
                for _idx, key in route_registry.iter_od_rows()
            )
        rows = getattr(route_registry, "od_pairs", None)
        if rows is None:
            raise ValueError("online route registry has no explicit OD row mapping")
        return tuple((int(value[0]), int(value[1])) for value in rows)

    def bind_route_registry(
        self,
        *,
        route_registry: Any,
        edge_ids: Sequence[str],
        initial_route_probs: Mapping[Tuple[int, int], np.ndarray],
        initial_od_weights: Optional[np.ndarray] = None,
    ) -> None:
        runtime_edges = tuple(str(value) for value in edge_ids)
        if runtime_edges != self.edge_ids:
            raise ValueError(
                "HDV OD-route predictor edge registry mismatch: "
                f"checkpoint={len(self.edge_ids)} runtime={len(runtime_edges)}"
            )
        runtime_ods = self._od_rows(route_registry)
        if runtime_ods != self.ods:
            raise ValueError(
                "HDV OD-route predictor OD registry mismatch: "
                f"checkpoint={len(self.ods)} runtime={len(runtime_ods)}"
            )
        rows: List[Tuple[Tuple[int, ...], ...]] = []
        initial = np.zeros((len(self.ods), self.n_routes), dtype=np.float32)
        for od_idx, od_key in enumerate(self.ods):
            routes = list(route_registry.route_edge_indices[od_idx])
            if len(routes) != self.n_routes:
                raise ValueError(
                    f"HDV OD-route predictor route-count mismatch for {od_key}: "
                    f"checkpoint={self.n_routes} runtime={len(routes)}"
                )
            route_row: List[Tuple[int, ...]] = []
            for route in routes:
                edge_row = tuple(int(value) for value in route)
                if any(value < 0 or value >= self.n_edges for value in edge_row):
                    raise ValueError(f"route {od_key} contains an out-of-range edge index")
                route_row.append(edge_row)
            rows.append(tuple(route_row))
            probs = np.asarray(
                initial_route_probs.get(od_key, np.ones(self.n_routes)),
                dtype=np.float32,
            ).reshape(-1)
            if probs.size != self.n_routes:
                raise ValueError(
                    f"initial HDV route state mismatch for {od_key}: "
                    f"expected {self.n_routes}, got {probs.size}"
                )
            initial[od_idx] = probs
        self.route_edge_indices = tuple(rows)
        self.last_observed_route = _normalise_rows(initial).astype(np.float32)
        initial_weights = np.asarray(
            (
                initial_od_weights
                if initial_od_weights is not None
                else getattr(
                    route_registry,
                    "od_hdv_counts",
                    np.ones(len(self.ods)),
                )
            ),
            dtype=np.float32,
        ).reshape(-1)
        if initial_weights.size != len(self.ods):
            raise ValueError(
                "initial HDV OD-weight mismatch: "
                f"{initial_weights.size} vs {len(self.ods)}"
            )
        if not bool(np.isfinite(initial_weights).all()) or bool(
            np.any(initial_weights < 0.0)
        ):
            raise ValueError("initial HDV OD weights must be finite and non-negative")
        self.initial_edge_distribution = self._project_route_to_edges(
            self.last_observed_route,
            initial_weights,
        ).astype(np.float32)
        if self._pending_runtime_state is not None:
            pending = self._pending_runtime_state
            self._pending_runtime_state = None
            self.load_runtime_state_dict(pending)

    def _project_route_to_edges(
        self,
        route_probs: np.ndarray,
        od_weights: np.ndarray,
    ) -> np.ndarray:
        if self.route_edge_indices is None:
            raise RuntimeError("HDV OD-route predictor has not been bound to a route registry")
        probs = _normalise_rows(route_probs)
        weights = np.asarray(od_weights, dtype=np.float64).reshape(-1).clip(min=0.0)
        if weights.size != len(self.ods):
            raise ValueError(f"OD projection weight mismatch: {weights.size} vs {len(self.ods)}")
        out = np.zeros((self.n_edges,), dtype=np.float64)
        for od_idx, route_row in enumerate(self.route_edge_indices):
            weight = float(weights[od_idx])
            if weight <= 0.0:
                continue
            for route_idx, edges in enumerate(route_row):
                mass = weight * float(probs[od_idx, route_idx])
                if mass > 0.0 and edges:
                    np.add.at(out, np.asarray(edges, dtype=np.int64), mass)
        return _normalise_vector(out)

    def observe_episode(
        self,
        observation: HdvOdRouteEpisodeObservation,
        *,
        collect_online_sample: bool = False,
    ) -> Dict[str, float]:
        if self.last_observed_route is None:
            raise RuntimeError("HDV OD-route predictor has not been bound to a route registry")
        expected = (len(self.ods), self.n_routes)
        for label in (
            "route_counts",
            "route_before",
            "route_cost_before",
            "filled_route_tt",
            "route_tt_observed",
            "route_tt_sample_counts",
            "route_after",
        ):
            values = np.asarray(getattr(observation, label))
            if values.shape != expected:
                raise ValueError(f"{label} shape mismatch: {values.shape} vs {expected}")
        edge = np.asarray(observation.edge_distribution).reshape(-1)
        if edge.size != self.n_edges:
            raise ValueError(
                f"edge distribution shape mismatch: {edge.size} vs {self.n_edges}"
            )
        counts = np.asarray(observation.route_counts, dtype=np.float64).clip(min=0.0)
        totals = counts.sum(axis=-1)
        observed = totals > 1e-12
        metrics: Dict[str, float] = {}
        if (
            self.pending_predicted_route is not None
            and self.pending_prediction_episode == int(observation.episode)
            and bool(np.any(observed))
        ):
            actual_route = np.asarray(self.last_observed_route, dtype=np.float64).copy()
            actual_route[observed] = counts[observed] / totals[observed, None]
            route_tv = 0.5 * np.abs(
                _normalise_rows(self.pending_predicted_route)
                - _normalise_rows(actual_route)
            ).sum(axis=-1)
            metrics = {
                "hdv_response_pred_route_tv": float(
                    (route_tv[observed] * totals[observed]).sum()
                    / max(float(totals[observed].sum()), 1e-12)
                ),
                "hdv_response_pred_route_observed_od_fraction": float(
                    np.mean(observed.astype(np.float64))
                ),
            }
            if bool(collect_online_sample) and self.pending_training_input is not None:
                target_weight = totals.clip(min=0.0)
                target_weight = target_weight / max(
                    float(target_weight.sum()),
                    1e-12,
                )
                self.online_training_samples.append(
                    {
                        "features": np.asarray(
                            self.pending_training_input["features"],
                            dtype=np.float32,
                        ),
                        "persistence": np.asarray(
                            self.pending_training_input["persistence"],
                            dtype=np.float32,
                        ),
                        "proposal": np.asarray(
                            self.pending_training_input["proposal"],
                            dtype=np.float32,
                        ),
                        "target": actual_route.astype(np.float32),
                        "od_weight": target_weight.astype(np.float32),
                    }
                )
                self.online_trained_pairs += 1
        if bool(np.any(observed)):
            self.last_observed_route[observed] = (
                counts[observed] / totals[observed, None]
            ).astype(np.float32)
        self.previous_observation = observation
        self.pending_predicted_route = None
        self.pending_prediction_episode = None
        self.pending_training_input = None
        return metrics

    def predict_for_next_episode(
        self,
        *,
        episode: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if self.last_observed_route is None or self.initial_edge_distribution is None:
            raise RuntimeError("HDV OD-route predictor has not been bound to a route registry")
        if self.previous_observation is None:
            self.pending_predicted_route = self.last_observed_route.copy()
            self.pending_prediction_episode = int(episode)
            self.pending_training_input = None
            return self.initial_edge_distribution.copy(), {
                "hdv_response_pred_enabled": 1.0,
                "hdv_response_pred_source_code": 1.75,
                "hdv_response_pred_gate_scale": float(self.selected_gate_scale),
                "hdv_response_pred_gate_mean": 0.0,
                "hdv_response_pred_route_delta_tv": 0.0,
            }
        observation = self.previous_observation
        od_totals = np.asarray(observation.route_counts, dtype=np.float32).sum(axis=-1)
        od_total = float(od_totals.sum())
        od_weight = (
            od_totals / od_total
            if od_total > 1e-12
            else np.full_like(od_totals, 1.0 / max(int(od_totals.size), 1))
        )
        features = build_persistence_gate_features(
            empirical_route=self.last_observed_route,
            route_before=observation.route_before,
            route_after=observation.route_after,
            route_cost_before=observation.route_cost_before,
            filled_route_tt=observation.filled_route_tt,
            route_tt_observed=observation.route_tt_observed,
            route_counts=observation.route_counts,
            route_tt_sample_counts=observation.route_tt_sample_counts,
            od_weight=od_weight,
            progress=float(episode) / float(self.progress_denominator),
        )
        with torch.no_grad():
            raw_response = (
                self.model(
                    torch.from_numpy(features).to(
                        device=self.device,
                        dtype=torch.float32,
                    )
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        persistence = _normalise_rows(self.last_observed_route)
        proposal = _normalise_rows(observation.route_after)
        if self.parameterization == "persistence_residual":
            predicted_route = apply_persistence_anchored_route_residual(
                persistence,
                raw_response,
                scale=float(self.selected_gate_scale),
                anchor_floor=float(self.anchor_floor),
            )
            weighted_response_mean = 0.0
            weighted_residual_l2 = float(
                (
                    np.sqrt(np.square(raw_response).sum(axis=-1))
                    * od_weight
                ).sum()
            )
            source_code = 4.0
        else:
            effective_gate = np.clip(
                float(self.selected_gate_scale) * raw_response,
                0.0,
                1.0,
            )
            predicted_route = _normalise_rows(
                (1.0 - effective_gate) * persistence
                + effective_gate * proposal
            )
            weighted_response_mean = float(
                (raw_response.reshape(-1) * od_weight).sum()
            )
            weighted_residual_l2 = 0.0
            source_code = 3.0
        self.pending_predicted_route = predicted_route.astype(np.float32)
        self.pending_prediction_episode = int(episode)
        self.pending_training_input = {
            "episode": int(episode),
            "features": features.astype(np.float32),
            "persistence": persistence.astype(np.float32),
            "proposal": proposal.astype(np.float32),
        }
        if float(self.selected_gate_scale) == 0.0:
            predicted_edge = _normalise_vector(observation.edge_distribution)
        else:
            projected_pred = self._project_route_to_edges(predicted_route, od_weight)
            projected_persistence = self._project_route_to_edges(persistence, od_weight)
            predicted_edge = _normalise_vector(
                np.clip(
                    _normalise_vector(observation.edge_distribution)
                    + projected_pred
                    - projected_persistence,
                    0.0,
                    None,
                )
            )
        route_delta = 0.5 * np.abs(predicted_route - persistence).sum(axis=-1)
        return predicted_edge.astype(np.float32), {
            "hdv_response_pred_enabled": 1.0,
            "hdv_response_pred_source_code": source_code,
            "hdv_response_pred_gate_scale": float(self.selected_gate_scale),
            "hdv_response_pred_gate_mean": weighted_response_mean,
            "hdv_response_pred_residual_l2": weighted_residual_l2,
            "hdv_response_pred_route_delta_tv": float(
                (route_delta * od_weight).sum()
            ),
        }

    def update_online(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        buffer_episodes: int,
        gate_penalty: float = 1e-4,
        grad_clip: float = 1.0,
    ) -> Dict[str, float]:
        capacity = max(int(buffer_episodes), 1)
        if len(self.online_training_samples) > capacity:
            self.online_training_samples = self.online_training_samples[-capacity:]
        if not self.online_training_samples:
            return {
                "hdv_response_loss": 0.0,
                "hdv_response_updates": 0.0,
                "hdv_response_online_pairs": 0.0,
                "hdv_response_online_train_tv": 0.0,
            }
        if float(self.selected_gate_scale) <= 0.0:
            return {
                "hdv_response_loss": 0.0,
                "hdv_response_updates": 0.0,
                "hdv_response_online_pairs": float(
                    len(self.online_training_samples)
                ),
                "hdv_response_online_train_tv": 0.0,
                "hdv_response_online_zero_scale": 1.0,
            }

        samples = self.online_training_samples
        features = torch.from_numpy(
            np.stack([sample["features"] for sample in samples], axis=0)
        ).to(device=self.device, dtype=torch.float32)
        persistence = torch.from_numpy(
            np.stack([sample["persistence"] for sample in samples], axis=0)
        ).to(device=self.device, dtype=torch.float32)
        proposal = torch.from_numpy(
            np.stack([sample["proposal"] for sample in samples], axis=0)
        ).to(device=self.device, dtype=torch.float32)
        target = torch.from_numpy(
            np.stack([sample["target"] for sample in samples], axis=0)
        ).to(device=self.device, dtype=torch.float32)
        od_weight = torch.from_numpy(
            np.stack([sample["od_weight"] for sample in samples], axis=0)
        ).to(device=self.device, dtype=torch.float32)

        self.model.train()
        losses: List[float] = []
        update_epochs = max(int(epochs), 1)
        for _ in range(update_epochs):
            raw_response = self.model(features)
            if self.parameterization == "persistence_residual":
                prediction = apply_persistence_anchored_route_residual_torch(
                    persistence,
                    raw_response,
                    scale=float(self.selected_gate_scale),
                    anchor_floor=float(self.anchor_floor),
                )
                route_tv = 0.5 * torch.abs(
                    prediction - target
                ).sum(dim=-1)
                route_loss = (route_tv * od_weight).sum(dim=-1).mean()
                regularizer = raw_response.square().mean()
            else:
                effective_gate = (
                    float(self.selected_gate_scale) * raw_response
                ).clamp(0.0, 1.0)
                prediction = (
                    (1.0 - effective_gate) * persistence
                    + effective_gate * proposal
                )
                prediction = prediction / prediction.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
                route_tv = 0.5 * torch.abs(prediction - target).sum(dim=-1)
                route_loss = (route_tv * od_weight).sum(dim=-1).mean()
                regularizer = (
                    effective_gate.squeeze(-1) * od_weight
                ).sum(dim=-1).mean()
            loss = route_loss + float(gate_penalty) * regularizer
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(grad_clip),
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        self.model.eval()
        self.online_update_steps += update_epochs
        with torch.no_grad():
            raw_response = self.model(features)
            if self.parameterization == "persistence_residual":
                prediction = apply_persistence_anchored_route_residual_torch(
                    persistence,
                    raw_response,
                    scale=float(self.selected_gate_scale),
                    anchor_floor=float(self.anchor_floor),
                )
                response_mean = (
                    torch.linalg.vector_norm(raw_response, dim=-1) * od_weight
                ).sum(dim=-1).mean()
            else:
                effective_gate = (
                    float(self.selected_gate_scale) * raw_response
                ).clamp(0.0, 1.0)
                prediction = (
                    (1.0 - effective_gate) * persistence
                    + effective_gate * proposal
                )
                prediction = prediction / prediction.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
                response_mean = (
                    effective_gate.squeeze(-1) * od_weight
                ).sum(dim=-1).mean()
            route_tv = 0.5 * torch.abs(prediction - target).sum(dim=-1)
            train_tv = (route_tv * od_weight).sum(dim=-1).mean()
        return {
            "hdv_response_loss": float(np.mean(losses)),
            "hdv_response_updates": float(update_epochs),
            "hdv_response_online_pairs": float(len(samples)),
            "hdv_response_online_train_tv": float(
                train_tv.detach().cpu().item()
            ),
            "hdv_response_online_gate_mean": float(
                response_mean.detach().cpu().item()
            ),
            "hdv_response_online_update_steps": float(self.online_update_steps),
        }

    def runtime_state_dict(self) -> Dict[str, Any]:
        return {
            "last_observed_route": (
                None
                if self.last_observed_route is None
                else np.asarray(self.last_observed_route, dtype=np.float32)
            ),
            "previous_observation": (
                None
                if self.previous_observation is None
                else self.previous_observation.state_dict()
            ),
            "pending_predicted_route": (
                None
                if self.pending_predicted_route is None
                else np.asarray(self.pending_predicted_route, dtype=np.float32)
            ),
            "pending_prediction_episode": self.pending_prediction_episode,
            "pending_training_input": self.pending_training_input,
            "online_training_samples": [
                {
                    key: np.asarray(value, dtype=np.float32)
                    for key, value in sample.items()
                }
                for sample in self.online_training_samples
            ],
            "online_trained_pairs": int(self.online_trained_pairs),
            "online_update_steps": int(self.online_update_steps),
        }

    def load_runtime_state_dict(self, state: Mapping[str, Any]) -> None:
        if self.last_observed_route is None:
            self._pending_runtime_state = dict(state)
            return
        route = state.get("last_observed_route")
        if route is not None:
            restored = np.asarray(route, dtype=np.float32)
            if restored.shape != self.last_observed_route.shape:
                raise ValueError(
                    "HDV OD-route runtime persistence shape mismatch: "
                    f"{restored.shape} vs {self.last_observed_route.shape}"
                )
            self.last_observed_route = _normalise_rows(restored).astype(np.float32)
        observation = state.get("previous_observation")
        self.previous_observation = (
            None
            if observation is None
            else HdvOdRouteEpisodeObservation.from_state_dict(observation)
        )
        pending_route = state.get("pending_predicted_route")
        if pending_route is not None:
            restored_pending = np.asarray(pending_route, dtype=np.float32)
            if restored_pending.shape != self.last_observed_route.shape:
                raise ValueError(
                    "HDV OD-route pending prediction shape mismatch: "
                    f"{restored_pending.shape} vs {self.last_observed_route.shape}"
                )
            self.pending_predicted_route = _normalise_rows(
                restored_pending
            ).astype(np.float32)
        else:
            self.pending_predicted_route = None
        pending_episode = state.get("pending_prediction_episode")
        self.pending_prediction_episode = (
            None if pending_episode is None else int(pending_episode)
        )
        pending_training_input = state.get("pending_training_input")
        self.pending_training_input = (
            None
            if not isinstance(pending_training_input, Mapping)
            else {
                "episode": int(
                    pending_training_input.get(
                        "episode",
                        self.pending_prediction_episode or 0,
                    )
                ),
                "features": np.asarray(
                    pending_training_input["features"],
                    dtype=np.float32,
                ),
                "persistence": np.asarray(
                    pending_training_input["persistence"],
                    dtype=np.float32,
                ),
                "proposal": np.asarray(
                    pending_training_input["proposal"],
                    dtype=np.float32,
                ),
            }
        )
        restored_samples: List[Dict[str, np.ndarray]] = []
        for sample in state.get("online_training_samples", []):
            if not isinstance(sample, Mapping):
                continue
            restored = {
                key: np.asarray(sample[key], dtype=np.float32)
                for key in (
                    "features",
                    "persistence",
                    "proposal",
                    "target",
                    "od_weight",
                )
            }
            if (
                restored["features"].shape[:1] != (len(self.ods),)
                or restored["persistence"].shape
                != (len(self.ods), self.n_routes)
                or restored["proposal"].shape
                != (len(self.ods), self.n_routes)
                or restored["target"].shape
                != (len(self.ods), self.n_routes)
                or restored["od_weight"].shape != (len(self.ods),)
            ):
                raise ValueError("HDV OD-route online sample shape mismatch")
            restored_samples.append(restored)
        self.online_training_samples = restored_samples
        self.online_trained_pairs = int(
            state.get("online_trained_pairs", self.online_trained_pairs) or 0
        )
        self.online_update_steps = int(
            state.get("online_update_steps", self.online_update_steps) or 0
        )


def load_hdv_route_fracs_from_state(
    state_path: str,
    route_registry,
    *,
    n_routes: int,
) -> Optional[np.ndarray]:
    """Load a static episode-level HDV route distribution from RouteRL state JSON.

    The serialized ``route_policy`` is a dense OD table, not a time-varying set
    of policies.  Only ODs with nonzero HDV counts affect the fixed edge
    background injected into the world-model dynamics.
    """
    path = str(state_path or "").strip()
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load HDV background state %s: %s", path, exc)
        return None

    n_od = int(route_registry.n_od)
    k_eff = int(n_routes)
    out = np.ones((n_od, k_eff), dtype=np.float32) / float(max(k_eff, 1))
    route_policy = state.get("route_policy") if isinstance(state, dict) else None
    if not isinstance(route_policy, list):
        logger.warning("HDV background state has no route_policy list: %s", path)
        return out

    loaded = 0
    loaded_active = 0
    active_od_mask = np.asarray(route_registry.od_hdv_counts, dtype=np.float32).reshape(-1) > 0.0
    active_od_total = int(active_od_mask[:n_od].sum())
    for row in route_policy:
        if not isinstance(row, dict):
            continue
        try:
            oi = int(row.get("origin", -1))
            di = int(row.get("destination", -1))
        except (TypeError, ValueError):
            continue
        if hasattr(route_registry, "od_index"):
            od_idx = int(route_registry.od_index(oi, di))
        else:
            n_dest = max(1, len(route_registry.destinations))
            od_idx = oi * n_dest + di
        if not (0 <= od_idx < n_od):
            continue
        probs = np.asarray(row.get("probs", []), dtype=np.float32).reshape(-1)
        if probs.size <= 0:
            continue
        row_probs = np.zeros((k_eff,), dtype=np.float32)
        take = min(k_eff, int(probs.size))
        row_probs[:take] = np.clip(probs[:take], 0.0, None)
        total = float(row_probs.sum())
        if total <= 1e-8:
            row_probs[:] = 1.0 / float(max(k_eff, 1))
        else:
            row_probs /= total
        out[od_idx] = row_probs
        loaded += 1
        if bool(active_od_mask[od_idx]):
            loaded_active += 1

    logger.info(
        "HDV static route distribution loaded: dense_od_rows=%d/%d active_hdv_od_rows=%d/%d <- %s",
        loaded,
        n_od,
        loaded_active,
        active_od_total,
        path,
    )
    return out


def build_hdv_background_edge_action(
    route_registry,
    hdv_route_fracs: np.ndarray,
    *,
    n_edges: int,
    edge_travel_times: Optional[np.ndarray] = None,
    horizon_seconds: Optional[float] = None,
) -> np.ndarray:
    """Project static episode-level HDV route distribution to one fixed edge background."""
    n_edges = int(n_edges)
    out = np.zeros((n_edges,), dtype=np.float32)
    if n_edges <= 0:
        return out
    hdv_counts = np.asarray(route_registry.od_hdv_counts, dtype=np.float32).reshape(-1)
    probs_all = np.asarray(hdv_route_fracs, dtype=np.float32)
    if probs_all.ndim != 2:
        return out

    use_horizon = edge_travel_times is not None and horizon_seconds is not None
    edge_tt = None
    if use_horizon:
        edge_tt = np.asarray(edge_travel_times, dtype=np.float32).reshape(-1)
        use_horizon = int(edge_tt.size) >= n_edges
    horizon = max(float(horizon_seconds or 0.0), 1.0)

    total_hdv = 0.0
    for od_idx, count in enumerate(hdv_counts[: int(route_registry.n_od)]):
        c = float(max(count, 0.0))
        if c <= 0.0:
            continue
        routes = route_registry.route_edge_indices[od_idx]
        if not routes:
            continue
        probs = probs_all[od_idx, : min(int(route_registry.K), probs_all.shape[1])].astype(np.float32)
        probs = np.clip(probs, 0.0, None)
        ps = float(probs.sum())
        if ps <= 1e-8:
            probs = np.ones((min(int(route_registry.K), len(routes)),), dtype=np.float32)
            probs /= float(max(int(probs.size), 1))
        else:
            probs /= ps
        total_hdv += c
        for k, prob in enumerate(probs[: len(routes)]):
            p = float(prob)
            if p <= 0.0:
                continue
            raw_idxs = [int(e) for e in routes[k] if 0 <= int(e) < n_edges]
            if not raw_idxs:
                continue
            if use_horizon:
                eta = 0.0
                keep: List[int] = []
                for edge_idx in raw_idxs:
                    if eta >= horizon:
                        break
                    keep.append(edge_idx)
                    eta += float(edge_tt[edge_idx])
                raw_idxs = keep
            if raw_idxs:
                np.add.at(out, np.asarray(raw_idxs, dtype=np.int64), c * p)

    if total_hdv > 1e-8:
        out /= float(total_hdv)
    return out.astype(np.float32)


def episode_av_pressure_from_action_edges(
    action_edges: List[torch.Tensor],
    *,
    n_edges: int,
) -> np.ndarray:
    """Aggregate a rollout's AV edge actions into one episode-level pressure field."""
    if not action_edges:
        return np.zeros((int(n_edges),), dtype=np.float32)
    arrs: List[np.ndarray] = []
    for action in action_edges:
        if action is None:
            continue
        a = action.detach().float().cpu().numpy().reshape(-1)
        if a.size < int(n_edges):
            padded = np.zeros((int(n_edges),), dtype=np.float32)
            padded[: a.size] = a.astype(np.float32)
            a = padded
        arrs.append(np.clip(a[: int(n_edges)], 0.0, None).astype(np.float32))
    if not arrs:
        return np.zeros((int(n_edges),), dtype=np.float32)
    return np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float32)


def build_hdv_route_fracs_from_episode_av_pressure(
    route_registry,
    episode_av_pressure: np.ndarray,
    *,
    base_edge_tt: np.ndarray,
    n_routes: int,
    pressure_weight: float,
    temperature: float,
    prior_route_fracs: Optional[np.ndarray] = None,
    prior_mix: float = 0.0,
) -> np.ndarray:
    """Approximate converged HDV route response to episode-level AV pressure.

    This is an explicit Wardrop/Gawron-style response model, not a per-step HDV
    action.  Each HDV OD receives route costs from current WM edge travel times
    plus a penalty for overlap with full-episode AV planned pressure, then
    converts those route costs into a static route distribution.
    """
    n_od = int(route_registry.n_od)
    k_eff = int(n_routes)
    edge_tt = np.asarray(base_edge_tt, dtype=np.float32).reshape(-1)
    pressure = np.asarray(episode_av_pressure, dtype=np.float32).reshape(-1)
    n_edges = int(route_registry.n_edges)
    if edge_tt.size < n_edges:
        fill = float(edge_tt.mean() if edge_tt.size else 1.0)
        edge_tt = np.pad(edge_tt, (0, n_edges - edge_tt.size), constant_values=fill)
    if pressure.size < n_edges:
        pressure = np.pad(pressure, (0, n_edges - pressure.size), constant_values=0.0)
    edge_cost = edge_tt[:n_edges] + float(pressure_weight) * np.clip(pressure[:n_edges], 0.0, None)
    temp = max(float(temperature), 1e-6)
    out = np.ones((n_od, k_eff), dtype=np.float32) / float(max(k_eff, 1))
    prior = None
    if prior_route_fracs is not None:
        prior = np.asarray(prior_route_fracs, dtype=np.float32)
        if prior.shape[0] != n_od:
            prior = None
    mix = float(np.clip(float(prior_mix), 0.0, 1.0))

    for od_idx in range(n_od):
        routes = route_registry.route_edge_indices[od_idx]
        costs = np.full((k_eff,), np.inf, dtype=np.float32)
        for k in range(min(k_eff, len(routes))):
            idxs = [int(e) for e in routes[k] if 0 <= int(e) < n_edges]
            if idxs:
                costs[k] = float(np.sum(edge_cost[np.asarray(idxs, dtype=np.int64)]))
        valid = np.isfinite(costs)
        if not bool(valid.any()):
            if prior is not None and prior.shape[1] >= k_eff:
                row = np.clip(prior[od_idx, :k_eff], 0.0, None)
                total = float(row.sum())
                out[od_idx] = row / total if total > 1e-8 else out[od_idx]
            continue
        min_cost = float(np.min(costs[valid]))
        logits = np.zeros((k_eff,), dtype=np.float32)
        logits[valid] = -(costs[valid] - min_cost) / temp
        logits[~valid] = -80.0
        logits = logits - float(np.max(logits))
        probs = np.exp(logits).astype(np.float32)
        probs[~valid] = 0.0
        total = float(probs.sum())
        probs = probs / total if total > 1e-8 else out[od_idx]
        if prior is not None and prior.shape[1] >= k_eff and mix > 0.0:
            prior_row = np.clip(prior[od_idx, :k_eff], 0.0, None)
            prior_total = float(prior_row.sum())
            if prior_total > 1e-8:
                prior_row = prior_row / prior_total
                probs = (1.0 - mix) * probs + mix * prior_row
                probs = probs / float(max(float(probs.sum()), 1e-8))
        out[od_idx] = probs.astype(np.float32)
    return out


def combine_av_hdv_edge_background(
    av_action_edge: torch.Tensor,
    hdv_background_edge: Optional[torch.Tensor],
    *,
    n_active_av: int,
    n_total_hdv: int,
    mode: str,
    weight: float,
) -> torch.Tensor:
    """Combine controllable AV edge action with fixed episode-level HDV background."""
    if hdv_background_edge is None or float(weight) == 0.0:
        return av_action_edge
    hdv = hdv_background_edge.to(device=av_action_edge.device, dtype=av_action_edge.dtype)
    mode_norm = str(mode or "additive").strip().lower()
    w = float(weight)
    if mode_norm == "population":
        av_count = float(max(int(n_active_av), 0))
        hdv_count = float(max(int(n_total_hdv), 0)) * w
        denom = max(av_count + hdv_count, 1e-8)
        return (av_action_edge * av_count + hdv * hdv_count) / denom
    if mode_norm != "additive":
        raise ValueError(f"Unsupported dreamer_hdv_background_combine={mode!r}")
    return av_action_edge + w * hdv
