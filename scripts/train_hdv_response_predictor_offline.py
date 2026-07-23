#!/usr/bin/env python3
"""Train and compare causal episode-level HDV response predictors offline.

Every target episode is predicted only from the immediately preceding episode.
The final test block is never used for fitting, early stopping, residual-scale
selection, or model-family selection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablation.online_wm_dreamer_actor_critic import (  # noqa: E402
    CausalHdvResponseModel,
    OnlineWmDreamerConfig,
)
from dreamer4.urban_routing.gawron import gawron_step_from_route_tt  # noqa: E402
from dreamer4.urban_routing.dreamer_hdv_response import (  # noqa: E402
    HDV_OD_ROUTE_PREDICTOR_FORMAT,
    HDV_OD_ROUTE_RESIDUAL_ANCHOR_FLOOR,
    HDV_OD_ROUTE_RESIDUAL_PREDICTOR_FORMAT,
    PersistenceAnchoredOdRouteResidual,
    apply_persistence_anchored_route_residual,
    apply_persistence_anchored_route_residual_torch,
)
from scripts.evaluate_hdv_static_response import (  # noqa: E402
    StaticHdvDataset,
    load_static_hdv_dataset,
)


@dataclass(frozen=True)
class CausalEpisodePairs:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class ResidualScaleSelection:
    scale: float
    candidate_tv: float
    persistence_tv: float


@dataclass(frozen=True)
class GawronRouteRollout:
    """Latent HDV route state before and after every recorded episode."""

    route_before_episode: np.ndarray
    cost_before_episode: np.ndarray
    route_after_episode: np.ndarray
    cost_after_episode: np.ndarray
    filled_route_tt: np.ndarray
    route_tt_observed: np.ndarray


@dataclass(frozen=True)
class GawronCalibration:
    rollout: GawronRouteRollout
    alpha_zero: float
    validation_route_tv: float
    validation_edge_tv: float
    candidates: Tuple[Dict[str, float], ...]


@dataclass(frozen=True)
class SparseRouteEdgeProjection:
    """Compact route-to-edge incidence used by repeated offline projections."""

    n_od: int
    n_routes: int
    n_edges: int
    flat_route_index: np.ndarray
    edge_index: np.ndarray
    value: np.ndarray

    @classmethod
    def from_dense(cls, route_to_edge: np.ndarray) -> "SparseRouteEdgeProjection":
        dense = np.asarray(route_to_edge)
        if dense.ndim != 3:
            raise ValueError(
                f"route-to-edge projection must be rank 3, got {dense.shape}"
            )
        od_index, route_index, edge_index = np.nonzero(dense)
        n_od, n_routes, n_edges = (int(value) for value in dense.shape)
        return cls(
            n_od=n_od,
            n_routes=n_routes,
            n_edges=n_edges,
            flat_route_index=(
                od_index.astype(np.int64) * n_routes + route_index.astype(np.int64)
            ),
            edge_index=edge_index.astype(np.int64),
            value=dense[od_index, route_index, edge_index].astype(
                np.float64,
                copy=False,
            ),
        )


def _pair_array(rows: Sequence[Tuple[int, int]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def build_causal_episode_pairs(
    episodes: np.ndarray,
    *,
    train_end_episode: int,
    validation_end_episode: int,
    test_start_episode: int,
) -> CausalEpisodePairs:
    """Return `(previous_index, target_index)` pairs without crossing gaps."""
    eps = np.asarray(episodes, dtype=np.int64).reshape(-1)
    if int(validation_end_episode) < int(train_end_episode):
        raise ValueError("validation_end_episode must be >= train_end_episode")
    if int(test_start_episode) <= int(validation_end_episode):
        raise ValueError("test_start_episode must be > validation_end_episode")
    train: List[Tuple[int, int]] = []
    validation: List[Tuple[int, int]] = []
    test: List[Tuple[int, int]] = []
    for current_idx in range(1, int(eps.size)):
        previous_idx = current_idx - 1
        current_episode = int(eps[current_idx])
        if current_episode - int(eps[previous_idx]) != 1:
            continue
        pair = (previous_idx, current_idx)
        if current_episode <= int(train_end_episode):
            train.append(pair)
        elif current_episode <= int(validation_end_episode):
            validation.append(pair)
        elif current_episode >= int(test_start_episode):
            test.append(pair)
    result = CausalEpisodePairs(
        train=_pair_array(train),
        validation=_pair_array(validation),
        test=_pair_array(test),
    )
    for label, values in (
        ("train", result.train),
        ("validation", result.validation),
        ("test", result.test),
    ):
        if int(values.shape[0]) == 0:
            raise RuntimeError(
                f"chronological split produced no {label} adjacent episode pairs"
            )
    return result


def _normalise_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).clip(min=0.0)
    denom = arr.sum(axis=-1, keepdims=True)
    uniform = np.full_like(arr, 1.0 / max(int(arr.shape[-1]), 1))
    return np.where(denom > eps, arr / np.maximum(denom, eps), uniform)


def smooth_empirical_route_counts(
    route_counts: np.ndarray,
    latent_prior: np.ndarray,
    *,
    prior_strength: float,
) -> np.ndarray:
    """Return a count-aware empirical route posterior with a latent prior."""
    strength = float(prior_strength)
    if strength < 0.0:
        raise ValueError("prior_strength must be non-negative")
    counts = np.asarray(route_counts, dtype=np.float64).clip(min=0.0)
    prior = _normalise_rows(latent_prior)
    if counts.shape != prior.shape:
        raise ValueError(
            f"route count/prior shape mismatch: {counts.shape} vs {prior.shape}"
        )
    total = counts.sum(axis=-1, keepdims=True)
    denominator = total + strength
    posterior = np.divide(
        counts + strength * prior,
        np.maximum(denominator, 1e-12),
    )
    return np.where(denominator > 1e-12, posterior, prior)


def causal_last_observed_route_counts(
    route_counts: np.ndarray,
    initial_route: np.ndarray,
) -> np.ndarray:
    """Carry each OD's latest observed route histogram forward causally.

    Empty OD rows in the CSV loader are represented as uniform probabilities,
    which is a missing-value placeholder rather than an observation.  This
    filter keeps the last actual histogram instead and uses the segment-start
    latent route state only until the first observation arrives.
    """
    counts = np.asarray(route_counts, dtype=np.float64).clip(min=0.0)
    if counts.ndim != 3:
        raise ValueError(f"route counts must be rank 3, got {counts.shape}")
    current = _normalise_rows(np.asarray(initial_route, dtype=np.float64))
    if current.shape != counts.shape[1:]:
        raise ValueError(
            "initial route/count shape mismatch: "
            f"{current.shape} vs {counts.shape[1:]}"
        )
    carried = np.empty_like(counts, dtype=np.float64)
    for episode_idx in range(int(counts.shape[0])):
        episode_counts = counts[episode_idx]
        totals = episode_counts.sum(axis=-1)
        observed = totals > 1e-12
        if bool(np.any(observed)):
            current[observed] = episode_counts[observed] / totals[observed, None]
        carried[episode_idx] = current
    return carried


def blend_route_anchors(
    persistence: np.ndarray,
    gawron: np.ndarray,
    *,
    gate: Any,
) -> np.ndarray:
    """Convexly blend empirical persistence with the Gawron proposal."""
    persistence_n = _normalise_rows(persistence)
    gawron_n = _normalise_rows(gawron)
    if persistence_n.shape != gawron_n.shape:
        raise ValueError(
            "route anchor shape mismatch: " f"{persistence_n.shape} vs {gawron_n.shape}"
        )
    gate_arr = np.asarray(gate, dtype=np.float64)
    while gate_arr.ndim < persistence_n.ndim:
        gate_arr = np.expand_dims(gate_arr, axis=-1)
    gate_arr = np.clip(gate_arr, 0.0, 1.0)
    return _normalise_rows((1.0 - gate_arr) * persistence_n + gate_arr * gawron_n)


def project_route_probs_to_edges(
    route_probs: np.ndarray,
    hdv_counts: np.ndarray,
    route_to_edge: Any,
) -> np.ndarray:
    """Project OD-route probabilities with OD population weighting."""
    probs = np.asarray(route_probs, dtype=np.float64)
    if probs.ndim == 2:
        probs = probs[None, ...]
    counts = np.asarray(hdv_counts, dtype=np.float64)
    if counts.ndim == 1:
        counts = np.broadcast_to(
            counts[None, :], (int(probs.shape[0]), int(counts.size))
        )
    if probs.shape[:2] != counts.shape:
        raise ValueError(
            f"route/count shape mismatch: route={probs.shape}, counts={counts.shape}"
        )
    if isinstance(route_to_edge, SparseRouteEdgeProjection):
        projection_shape = (
            route_to_edge.n_od,
            route_to_edge.n_routes,
            route_to_edge.n_edges,
        )
    else:
        projection = np.asarray(route_to_edge)
        projection_shape = projection.shape
    if projection_shape[:2] != probs.shape[1:]:
        raise ValueError(
            "route/projection shape mismatch: "
            f"route={probs.shape}, projection={projection_shape}"
        )
    weighted = _normalise_rows(probs) * np.clip(counts, 0.0, None)[..., None]
    if isinstance(route_to_edge, SparseRouteEdgeProjection):
        flat_weighted = weighted.reshape(int(weighted.shape[0]), -1)
        edge = np.stack(
            [
                np.bincount(
                    route_to_edge.edge_index,
                    weights=(
                        flat_weighted[batch_idx, route_to_edge.flat_route_index]
                        * route_to_edge.value
                    ),
                    minlength=route_to_edge.n_edges,
                )
                for batch_idx in range(int(weighted.shape[0]))
            ],
            axis=0,
        )
    else:
        edge = np.einsum("bok,oke->be", weighted, projection, optimize=True)
    return _normalise_rows(edge)


def project_route_response_to_edges(
    *,
    predicted_route: np.ndarray,
    previous_route: np.ndarray,
    hdv_counts: np.ndarray,
    route_to_edge: Any,
    previous_edge: np.ndarray,
) -> np.ndarray:
    """Apply the structured route change to the exact previous edge context."""
    projected_pred = project_route_probs_to_edges(
        predicted_route,
        hdv_counts,
        route_to_edge,
    )
    projected_previous = project_route_probs_to_edges(
        previous_route,
        hdv_counts,
        route_to_edge,
    )
    response = projected_pred - projected_previous
    return _normalise_rows(np.clip(np.asarray(previous_edge) + response, 0.0, None))


def apply_residual_logits(
    persistence: np.ndarray,
    residual_logits: np.ndarray,
    scale: float,
) -> np.ndarray:
    anchor = _normalise_rows(persistence)
    if float(scale) == 0.0:
        return anchor
    residual = np.asarray(residual_logits, dtype=np.float64)
    logits = np.log(np.clip(anchor, 1e-12, None)) + float(scale) * residual
    logits = logits - logits.max(axis=-1, keepdims=True)
    return _normalise_rows(np.exp(logits))


def _tv_rows(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(_normalise_rows(target) - _normalise_rows(pred)).sum(axis=-1)


def select_residual_scale(
    *,
    target: np.ndarray,
    persistence: np.ndarray,
    residual_logits: np.ndarray,
    scales: Iterable[float],
) -> ResidualScaleSelection:
    target_n = _normalise_rows(target)
    persistence_n = _normalise_rows(persistence)
    persistence_tv = float(_tv_rows(target_n, persistence_n).mean())
    candidates = sorted({float(scale) for scale in scales})
    if not candidates or candidates[0] < 0.0:
        raise ValueError("residual scales must contain one or more non-negative values")
    best_scale = candidates[0]
    best_tv = float("inf")
    for scale in candidates:
        pred = apply_residual_logits(persistence_n, residual_logits, scale)
        score = float(_tv_rows(target_n, pred).mean())
        if score < best_tv - 1e-12 or (
            abs(score - best_tv) <= 1e-12 and scale < best_scale
        ):
            best_scale = float(scale)
            best_tv = score
    return ResidualScaleSelection(
        scale=float(best_scale),
        candidate_tv=float(best_tv),
        persistence_tv=float(persistence_tv),
    )


def _safe_route(row: Dict[str, str], role: str) -> int:
    key = "policy_action" if role == "av" else "hdv_action"
    raw = row.get(key)
    if raw is None or str(raw).strip() == "":
        return -1
    try:
        route = int(float(raw))
    except ValueError:
        return -1
    return route if route >= 0 else -1


def load_route_travel_time_features(
    *,
    dataset: StaticHdvDataset,
    episodes_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the actual per-OD-route feedback used by the real Gawron update.

    The online collector pools arrivals from both AVs and HDVs for a given
    OD-route key.  Unobserved alternatives are filled with 1.5 times the
    slowest observed route for that OD, matching
    ``DataCollector._update_hdv_gawron_actual``.
    """
    od_to_idx = {tuple(od): idx for idx, od in enumerate(dataset.ods)}
    n_episode = int(dataset.episodes.size)
    sums = np.zeros(
        (n_episode, len(dataset.ods), int(dataset.n_routes)), dtype=np.float64
    )
    counts = np.zeros_like(sums)
    for episode_idx, episode in enumerate(dataset.episodes.tolist()):
        path = episodes_dir / f"ep{int(episode)}.csv"
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                role = str(row.get("vType", "")).strip().lower()
                if role not in {"av", "human"}:
                    continue
                seen = str(row.get("seen_in_episode", "1")).strip().lower()
                if seen in {"0", "false", "no", "off", ""}:
                    continue
                try:
                    od = (int(float(row["origin"])), int(float(row["destination"])))
                    travel_time = float(row["travel_time"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(travel_time) or travel_time < 0.0:
                    continue
                od_idx = od_to_idx.get(od)
                route = _safe_route(row, role)
                if od_idx is None or route < 0 or route >= int(dataset.n_routes):
                    continue
                sums[episode_idx, od_idx, route] += travel_time
                counts[episode_idx, od_idx, route] += 1.0
    observed = counts > 0.0
    means = np.divide(sums, np.maximum(counts, 1.0), out=np.zeros_like(sums))
    filled = np.zeros_like(means)
    for episode_idx in range(n_episode):
        for od_idx in range(len(dataset.ods)):
            mask = observed[episode_idx, od_idx]
            if not np.any(mask):
                continue
            observed_values = means[episode_idx, od_idx, mask]
            fill = float(observed_values.max()) * 1.5
            filled[episode_idx, od_idx] = np.where(
                mask,
                means[episode_idx, od_idx],
                fill,
            )
    return (
        filled.astype(np.float32),
        observed.astype(np.float32),
        counts.astype(np.float32),
    )


def _resolve_relocated_repo_path(raw_path: str, *, run_dir: Path) -> Path:
    """Resolve archived absolute paths after a repository relocation."""
    value = str(raw_path or "").strip()
    if not value:
        raise RuntimeError("no HDV route-state path was provided")
    direct = Path(value).expanduser()
    try:
        if direct.exists():
            return direct.resolve()
    except OSError:
        # Archived configs can contain server-only absolute paths (for
        # example /root/UrbanDreamer-main/...). A permission error while
        # probing one must not prevent relocation through a repo marker.
        pass
    for marker in ("networks/", "results/", "data/", "logs/"):
        marker_idx = value.find(marker)
        if marker_idx >= 0:
            candidate = REPO_ROOT / value[marker_idx:]
            try:
                if candidate.exists():
                    return candidate.resolve()
            except OSError:
                continue
    local_candidate = run_dir / value
    try:
        if local_candidate.exists():
            return local_candidate.resolve()
    except OSError:
        pass
    raise RuntimeError(
        "HDV route-state path does not exist locally and could not be relocated: "
        f"{value}"
    )


def load_gawron_config(
    *,
    run_dir: Path,
    initial_hdv_state: str,
) -> Tuple[Path, Dict[str, Any]]:
    """Load the exact route-update settings used by an online run."""
    config_path = run_dir / "online_wm_dreamer_config.json"
    if not config_path.exists():
        raise RuntimeError(f"missing online run config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise RuntimeError(f"online run config is not an object: {config_path}")
    state_path = _resolve_relocated_repo_path(
        initial_hdv_state or str(raw.get("hdv_state_in", "")),
        run_dir=run_dir,
    )
    settings: Dict[str, Any] = {
        "mode": str(raw.get("hdv_gawron_mode", "replicator")).strip().lower(),
        "lr": float(raw.get("hdv_gawron_lr", 0.1) or 0.1),
        "alpha_zero": float(raw.get("hdv_gawron_alpha_zero", 0.2) or 0.2),
        "beta": float(raw.get("hdv_gawron_beta", -1.5) or -1.5),
        "update_mode": str(raw.get("hdv_gawron_update_mode", "expected") or "expected")
        .strip()
        .lower(),
        "min_prob": float(raw.get("hdv_min_prob", 0.0) or 0.0),
    }
    if settings["mode"] not in {"replicator", "routerl"}:
        raise RuntimeError(
            "offline OD-route replay requires aggregate Gawron state, but the run "
            f"uses hdv_gawron_mode={settings['mode']!r}"
        )
    if settings["update_mode"] == "sample":
        raise RuntimeError(
            "sample-mode Gawron replay is stochastic and cannot be reconstructed "
            "without the original RNG state; use an expected-mode run"
        )
    return state_path, settings


def load_initial_hdv_route_state(
    *,
    state_path: Path,
    dataset: StaticHdvDataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map a serialized collector route state onto the dataset OD order."""
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    route_policy = state.get("route_policy") if isinstance(state, dict) else None
    if not isinstance(route_policy, list):
        raise RuntimeError(f"HDV state has no route_policy list: {state_path}")
    n_od = len(dataset.ods)
    n_routes = int(dataset.n_routes)
    route = np.full(
        (n_od, n_routes),
        1.0 / float(max(n_routes, 1)),
        dtype=np.float32,
    )
    cost = np.full((n_od, n_routes), np.nan, dtype=np.float32)
    od_to_idx = {tuple(od): idx for idx, od in enumerate(dataset.ods)}
    restored_route = np.zeros((n_od,), dtype=bool)
    restored_cost = np.zeros((n_od,), dtype=bool)
    for entry in route_policy:
        if not isinstance(entry, dict):
            continue
        try:
            od = (int(entry["origin"]), int(entry["destination"]))
        except (KeyError, TypeError, ValueError):
            continue
        od_idx = od_to_idx.get(od)
        if od_idx is None:
            continue
        probs = np.asarray(entry.get("probs", []), dtype=np.float32).reshape(-1)
        if int(probs.size) != n_routes:
            continue
        route[od_idx] = _normalise_rows(probs[None, :])[0].astype(np.float32)
        restored_route[od_idx] = True
        costs = np.asarray(entry.get("costs", []), dtype=np.float32).reshape(-1)
        if int(costs.size) == n_routes and bool(np.isfinite(costs).all()):
            cost[od_idx] = costs
            restored_cost[od_idx] = True
    active = np.asarray(dataset.hdv_od_weight, dtype=np.float64).sum(axis=0) > 0.0
    missing_route = np.flatnonzero(active & ~restored_route)
    missing_cost = np.flatnonzero(active & ~restored_cost)
    if int(missing_route.size) > 0:
        raise RuntimeError(
            f"initial HDV state is missing {missing_route.size} active OD route rows: "
            f"{state_path}"
        )
    if int(missing_cost.size) > 0:
        raise RuntimeError(
            f"initial HDV state is missing {missing_cost.size} active OD cost rows: "
            f"{state_path}"
        )
    cost = np.nan_to_num(cost, nan=0.0, posinf=0.0, neginf=0.0)
    return route, cost


def replay_gawron_route_state(
    *,
    initial_route: np.ndarray,
    initial_cost: np.ndarray,
    route_tt: np.ndarray,
    route_tt_observed: np.ndarray,
    mode: str,
    lr: float,
    alpha_zero: float,
    beta: float,
    update_mode: str,
    min_prob: float,
) -> GawronRouteRollout:
    """Replay the collector's causal OD-route state transition episode by episode."""
    filled_tt = np.asarray(route_tt, dtype=np.float32)
    observed = np.asarray(route_tt_observed, dtype=np.float32) > 0.0
    if filled_tt.shape != observed.shape:
        raise ValueError(
            f"route travel-time/mask mismatch: {filled_tt.shape} vs {observed.shape}"
        )
    n_episode, n_od, n_routes = filled_tt.shape
    current_route = np.asarray(initial_route, dtype=np.float32).copy()
    current_cost = np.asarray(initial_cost, dtype=np.float32).copy()
    expected_shape = (n_od, n_routes)
    if current_route.shape != expected_shape or current_cost.shape != expected_shape:
        raise ValueError(
            "initial route-state shape mismatch: "
            f"route={current_route.shape}, cost={current_cost.shape}, "
            f"expected={expected_shape}"
        )
    route_before = np.zeros_like(filled_tt)
    cost_before = np.zeros_like(filled_tt)
    route_after = np.zeros_like(filled_tt)
    cost_after = np.zeros_like(filled_tt)
    for episode_idx in range(n_episode):
        route_before[episode_idx] = current_route
        cost_before[episode_idx] = current_cost
        for od_idx in range(n_od):
            if not bool(observed[episode_idx, od_idx].any()):
                continue
            new_route, new_cost = gawron_step_from_route_tt(
                probs=current_route[od_idx],
                route_tt=filled_tt[episode_idx, od_idx],
                mode=str(mode),
                lr=float(lr),
                cost=(
                    current_cost[od_idx]
                    if str(mode).strip().lower() == "routerl"
                    else None
                ),
                alpha_zero=float(alpha_zero),
                beta=float(beta),
                update_mode=str(update_mode),
                min_prob=float(min_prob),
            )
            current_route[od_idx] = new_route
            if new_cost is not None:
                current_cost[od_idx] = new_cost
        route_after[episode_idx] = current_route
        cost_after[episode_idx] = current_cost
    return GawronRouteRollout(
        route_before_episode=route_before,
        cost_before_episode=cost_before,
        route_after_episode=route_after,
        cost_after_episode=cost_after,
        filled_route_tt=filled_tt,
        route_tt_observed=observed.astype(np.float32),
    )


def calibrate_gawron_alpha(
    *,
    dataset: StaticHdvDataset,
    pairs: CausalEpisodePairs,
    initial_route: np.ndarray,
    initial_cost: np.ndarray,
    route_tt: np.ndarray,
    route_tt_observed: np.ndarray,
    settings: Dict[str, Any],
    alpha_candidates: Sequence[float],
    min_improvement: float,
    min_relative_improvement: float = 0.0,
) -> GawronCalibration:
    """Select RouteRL's response rate on the chronological validation block.

    OD-route TV is primary.  Edge TV is a tie-breaker and is computed by
    applying the predicted route *response* to the exact previous edge
    context, rather than rebuilding that context from the route registry.
    """
    configured_alpha = float(settings["alpha_zero"])
    candidates = sorted(
        {
            configured_alpha,
            *(
                float(value)
                for value in alpha_candidates
                if math.isfinite(float(value)) and float(value) > 0.0
            ),
        }
    )
    if not candidates:
        raise ValueError("Gawron alpha calibration requires a positive candidate")
    previous_idx = pairs.validation[:, 0]
    current_idx = pairs.validation[:, 1]
    target_route = dataset.hdv_route[current_idx]
    target_weight = dataset.hdv_od_weight[current_idx]
    target_edge = dataset.hdv_edge[current_idx]
    previous_edge = dataset.hdv_edge[previous_idx]
    counts = dataset.hdv_od_weight[previous_idx]
    empirical_state = causal_last_observed_route_counts(
        dataset.hdv_route_counts,
        initial_route,
    )
    scored: List[Tuple[float, float, float, GawronRouteRollout]] = []
    records: List[Dict[str, float]] = []
    for alpha_zero in candidates:
        rollout = replay_gawron_route_state(
            initial_route=initial_route,
            initial_cost=initial_cost,
            route_tt=route_tt,
            route_tt_observed=route_tt_observed,
            mode=str(settings["mode"]),
            lr=float(settings["lr"]),
            alpha_zero=float(alpha_zero),
            beta=float(settings["beta"]),
            update_mode=str(settings["update_mode"]),
            min_prob=float(settings["min_prob"]),
        )
        previous_route = empirical_state[previous_idx]
        predicted_route = rollout.route_after_episode[previous_idx]
        route_tv = float(
            _active_route_metrics(
                target_route,
                predicted_route,
                target_weight,
            )["active_od_route_tv"]
        )
        predicted_edge = project_route_response_to_edges(
            predicted_route=predicted_route,
            previous_route=previous_route,
            hdv_counts=counts,
            route_to_edge=dataset.route_to_edge,
            previous_edge=previous_edge,
        )
        edge_tv = float(_tv_rows(target_edge, predicted_edge).mean())
        records.append(
            {
                "alpha_zero": float(alpha_zero),
                "validation_route_tv": route_tv,
                "validation_edge_tv": edge_tv,
            }
        )
        scored.append((route_tv, edge_tv, float(alpha_zero), rollout))
    best_route_tv, best_edge_tv, best_alpha, best_rollout = min(
        scored,
        key=lambda item: (
            item[0],
            item[1],
            abs(item[2] - configured_alpha),
            item[2],
        ),
    )
    (
        configured_route_tv,
        configured_edge_tv,
        _,
        configured_rollout,
    ) = min(
        scored,
        key=lambda item: abs(item[2] - configured_alpha),
    )
    required_gain = max(
        float(max(min_improvement, 0.0)),
        float(max(min_relative_improvement, 0.0)) * float(configured_route_tv),
    )
    if float(best_route_tv) >= float(configured_route_tv) - required_gain:
        best_route_tv = float(configured_route_tv)
        best_edge_tv = float(configured_edge_tv)
        best_alpha = float(configured_alpha)
        best_rollout = configured_rollout
    return GawronCalibration(
        rollout=best_rollout,
        alpha_zero=float(best_alpha),
        validation_route_tv=float(best_route_tv),
        validation_edge_tv=float(best_edge_tv),
        candidates=tuple(records),
    )


def load_edge_trace_targets(
    *,
    run_dir: Path,
    dataset: StaticHdvDataset,
) -> str:
    """Replace CSV reconstructions with the exact edge tensors used online."""
    trace_dir = run_dir / "hdv_edge_pressure_traces"
    if not trace_dir.exists():
        raise RuntimeError(f"no online HDV edge traces under {trace_dir}")
    av_rows: List[np.ndarray] = []
    hdv_rows: List[np.ndarray] = []
    target_source = ""
    for episode in dataset.episodes.tolist():
        path = trace_dir / f"episode_{int(episode):05d}.npz"
        if not path.exists():
            raise RuntimeError(f"missing HDV edge trace for episode {episode}: {path}")
        with np.load(path, allow_pickle=False) as data:
            edge_ids = [
                str(value)
                for value in np.asarray(data["edge_ids"]).reshape(-1).tolist()
            ]
            if edge_ids != list(dataset.edge_ids):
                raise RuntimeError(f"edge-id order mismatch between dataset and {path}")
            av_rows.append(
                _normalise_rows(
                    np.asarray(data["av_action"], dtype=np.float64).mean(axis=0)[
                        None, :
                    ]
                )[0]
            )
            if "hdv_static_edge_demand_mean" in data:
                hdv = np.asarray(data["hdv_static_edge_demand_mean"], dtype=np.float64)
            elif "hdv_target" in data:
                hdv = np.asarray(data["hdv_target"], dtype=np.float64)
            else:
                hdv = np.asarray(data["hdv_static_edge_demand"], dtype=np.float64).mean(
                    axis=0
                )
            hdv_rows.append(_normalise_rows(hdv[None, :])[0])
            if "hdv_target_source" in data:
                source = str(np.asarray(data["hdv_target_source"]).reshape(-1)[0])
                if target_source and source != target_source:
                    raise RuntimeError(
                        f"HDV target semantics change at {path}: {target_source} -> {source}"
                    )
                target_source = source
    dataset.av_edge = np.stack(av_rows, axis=0)
    dataset.hdv_edge = np.stack(hdv_rows, axis=0)
    return target_source or "unknown"


def _edge_features(
    dataset: StaticHdvDataset,
    pairs: np.ndarray,
    *,
    progress_denominator: int,
) -> np.ndarray:
    previous_idx = pairs[:, 0]
    target_idx = pairs[:, 1]
    progress = np.clip(
        dataset.episodes[target_idx].astype(np.float32)
        / float(max(int(progress_denominator), 1)),
        0.0,
        1.0,
    ).reshape(-1, 1)
    return np.concatenate(
        [
            dataset.av_edge[previous_idx].astype(np.float32),
            dataset.hdv_edge[previous_idx].astype(np.float32),
            progress,
        ],
        axis=1,
    )


def _clone_state(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }


def _soft_cross_entropy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log(pred.clamp_min(1e-12))).sum(dim=-1).mean()


def train_edge_predictor(
    *,
    dataset: StaticHdvDataset,
    pairs: CausalEpisodePairs,
    parameterization: str,
    scales: Sequence[float],
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    residual_penalty: float,
    hidden_dim: int,
    depth: int,
    seed: int,
    device: torch.device,
    progress_denominator: int,
) -> Tuple[CausalHdvResponseModel, Dict[str, Any], Dict[str, np.ndarray]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    cfg = OnlineWmDreamerConfig(
        train_episodes=int(progress_denominator),
        hdv_response_predictor_enabled=True,
        hdv_response_predictor_lr=float(lr),
        hdv_response_predictor_weight_decay=float(weight_decay),
        hdv_response_predictor_hidden_dim=int(hidden_dim),
        hdv_response_predictor_depth=int(depth),
        hdv_response_predictor_progress_denominator=int(progress_denominator),
        hdv_response_predictor_parameterization=str(parameterization),
        hdv_response_predictor_residual_scale=1.0,
    )
    response = CausalHdvResponseModel(
        n_edges=len(dataset.edge_ids),
        cfg=cfg,
        device=device,
    )
    x_train = torch.from_numpy(
        _edge_features(dataset, pairs.train, progress_denominator=progress_denominator)
    ).to(device)
    y_train = torch.from_numpy(
        dataset.hdv_edge[pairs.train[:, 1]].astype(np.float32)
    ).to(device)
    x_val = torch.from_numpy(
        _edge_features(
            dataset, pairs.validation, progress_denominator=progress_denominator
        )
    ).to(device)
    y_val_np = dataset.hdv_edge[pairs.validation[:, 1]]
    persistence_val = dataset.hdv_edge[pairs.validation[:, 0]]
    optimizer = response.optimizer
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_scale = 1.0
    best_val_tv = float("inf")
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    def evaluate_validation() -> Tuple[float, float]:
        response.model.eval()
        with torch.no_grad():
            raw_np = response.model(x_val).detach().cpu().numpy()
            if response.parameterization == "absolute_logits":
                pred_np = torch.softmax(torch.from_numpy(raw_np), dim=-1).numpy()
                return float(_tv_rows(y_val_np, pred_np).mean()), 1.0
            positive_scales = tuple(
                float(scale) for scale in scales if float(scale) > 0.0
            )
            if not positive_scales:
                raise ValueError(
                    "residual training requires at least one positive scale"
                )
            selection = select_residual_scale(
                target=y_val_np,
                persistence=persistence_val,
                residual_logits=raw_np,
                scales=positive_scales,
            )
            return float(selection.candidate_tv), float(selection.scale)

    for epoch in range(1, max(int(epochs), 1) + 1):
        response.model.train()
        pred = response.predict_distribution(x_train, residual_scale=1.0)
        loss = _soft_cross_entropy(pred, y_train)
        if (
            response.parameterization == "residual_logit"
            and float(residual_penalty) > 0.0
        ):
            residual = response.model(x_train)
            loss = loss + float(residual_penalty) * residual.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(response.model.parameters(), 10.0)
        optimizer.step()
        val_tv, selected_scale = evaluate_validation()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(loss.detach().cpu().item()),
                "validation_edge_tv": float(val_tv),
                "residual_scale": float(selected_scale),
            }
        )
        if val_tv < best_val_tv - 1e-9:
            best_val_tv = float(val_tv)
            best_scale = float(selected_scale)
            best_epoch = int(epoch)
            best_state = _clone_state(response.model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= max(int(patience), 1):
            break
    if best_state is None:
        raise RuntimeError("edge predictor training did not produce a validation state")
    response.model.load_state_dict(best_state, strict=True)
    response.parameterization = str(parameterization)
    response.residual_scale = float(best_scale)
    response.trained_pairs = int(pairs.train.shape[0])
    response.pretrained_pairs = int(pairs.train.shape[0])
    response.model.eval()

    predictions: Dict[str, np.ndarray] = {}
    for split_name, split_pairs in (
        ("train", pairs.train),
        ("validation", pairs.validation),
        ("test", pairs.test),
    ):
        x = torch.from_numpy(
            _edge_features(
                dataset, split_pairs, progress_denominator=progress_denominator
            )
        ).to(device)
        with torch.no_grad():
            predictions[split_name] = (
                response.predict_distribution(x)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    metadata: Dict[str, Any] = {
        "parameterization": str(parameterization),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "best_validation_edge_tv": float(best_val_tv),
        "selected_residual_scale": float(best_scale),
        "validation_persistence_edge_tv": float(
            _tv_rows(y_val_np, persistence_val).mean()
        ),
        "history_tail": history[-10:],
    }
    return response, metadata, predictions


class SharedOdRouteResidualPredictor(nn.Module):
    def __init__(self, n_routes: int, hidden_dim: int, depth: int):
        super().__init__()
        self.n_routes = int(n_routes)
        # Previous AV route distribution, latent HDV route probabilities,
        # RouteRL cost memory, actual route travel time, observation mask,
        # OD population weight, and episode progress.
        input_dim = 5 * self.n_routes + 2
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(max(int(depth), 1)):
            layers.extend(
                [
                    nn.Linear(dim, int(hidden_dim)),
                    nn.GELU(),
                    nn.LayerNorm(int(hidden_dim)),
                ]
            )
            dim = int(hidden_dim)
        output = nn.Linear(dim, self.n_routes)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class PersistenceAnchoredOdRouteGate(nn.Module):
    """Predict how much of the exact Gawron proposal to trust for each OD."""

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


def _route_features(
    dataset: StaticHdvDataset,
    gawron: GawronRouteRollout,
    pairs: np.ndarray,
    *,
    progress_denominator: int,
) -> np.ndarray:
    previous_idx = pairs[:, 0]
    target_idx = pairs[:, 1]
    batch = int(pairs.shape[0])
    n_od = len(dataset.ods)
    progress = np.clip(
        dataset.episodes[target_idx].astype(np.float32)
        / float(max(int(progress_denominator), 1)),
        0.0,
        1.0,
    )
    progress = np.broadcast_to(progress[:, None, None], (batch, n_od, 1))
    od_weight = dataset.hdv_od_weight[previous_idx].astype(np.float32)[..., None]
    previous_cost = gawron.cost_before_episode[previous_idx].astype(np.float32)
    previous_tt = gawron.filled_route_tt[previous_idx].astype(np.float32)
    tt_mask = gawron.route_tt_observed[previous_idx].astype(np.float32)
    observed_count = np.maximum(tt_mask.sum(axis=-1, keepdims=True), 1.0)
    tt_scale = np.maximum(
        (previous_tt * tt_mask).sum(axis=-1, keepdims=True) / observed_count,
        1.0,
    )
    cost_gap = previous_cost - np.nanmin(
        previous_cost,
        axis=-1,
        keepdims=True,
    )
    cost_feature = np.nan_to_num(cost_gap / tt_scale, nan=0.0, posinf=10.0, neginf=0.0)
    tt_feature = np.nan_to_num(previous_tt / tt_scale, nan=0.0, posinf=10.0, neginf=0.0)
    return np.concatenate(
        [
            dataset.av_route[previous_idx].astype(np.float32),
            gawron.route_before_episode[previous_idx].astype(np.float32),
            np.clip(cost_feature, 0.0, 10.0).astype(np.float32),
            np.clip(tt_feature, 0.0, 10.0).astype(np.float32),
            tt_mask,
            od_weight,
            progress.astype(np.float32),
        ],
        axis=-1,
    )


def _persistence_gate_features(
    dataset: StaticHdvDataset,
    gawron: GawronRouteRollout,
    route_tt_sample_counts: np.ndarray,
    pairs: np.ndarray,
    *,
    progress_denominator: int,
) -> np.ndarray:
    """Build OD-local gate features using only the previous episode."""
    previous_idx = pairs[:, 0]
    target_idx = pairs[:, 1]
    batch = int(pairs.shape[0])
    n_od = len(dataset.ods)
    n_routes = int(dataset.n_routes)
    previous_counts = np.asarray(
        dataset.hdv_route_counts[previous_idx],
        dtype=np.float32,
    )
    latent_before = gawron.route_before_episode[previous_idx].astype(np.float32)
    empirical_state = causal_last_observed_route_counts(
        dataset.hdv_route_counts,
        gawron.route_before_episode[0],
    )
    empirical = empirical_state[previous_idx].astype(np.float32)
    gawron_proposal = gawron.route_after_episode[previous_idx].astype(np.float32)
    latent_delta = gawron_proposal - latent_before
    previous_cost = gawron.cost_before_episode[previous_idx].astype(np.float32)
    previous_tt = gawron.filled_route_tt[previous_idx].astype(np.float32)
    tt_mask = gawron.route_tt_observed[previous_idx].astype(np.float32)
    tt_counts = np.asarray(
        route_tt_sample_counts[previous_idx],
        dtype=np.float32,
    )
    observed_count = np.maximum(tt_mask.sum(axis=-1, keepdims=True), 1.0)
    tt_scale = np.maximum(
        (previous_tt * tt_mask).sum(axis=-1, keepdims=True) / observed_count,
        1.0,
    )
    cost_gap = previous_cost - np.nanmin(
        previous_cost,
        axis=-1,
        keepdims=True,
    )
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
    branch_disagreement = 0.5 * np.abs(empirical - gawron_proposal).sum(
        axis=-1, keepdims=True
    )
    latent_response = 0.5 * np.abs(latent_before - gawron_proposal).sum(
        axis=-1, keepdims=True
    )
    empirical_entropy = -(empirical * np.log(np.clip(empirical, 1e-12, 1.0))).sum(
        axis=-1, keepdims=True
    ) / max(math.log(max(n_routes, 2)), 1e-12)
    route_count_total = previous_counts.sum(axis=-1, keepdims=True)
    tt_count_total = tt_counts.sum(axis=-1, keepdims=True)
    progress = np.clip(
        dataset.episodes[target_idx].astype(np.float32)
        / float(max(int(progress_denominator), 1)),
        0.0,
        1.0,
    )
    progress = np.broadcast_to(progress[:, None, None], (batch, n_od, 1))
    od_weight = dataset.hdv_od_weight[previous_idx].astype(np.float32)[..., None]
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
            od_weight,
            progress.astype(np.float32),
        ],
        axis=-1,
    )


def _blend_route_anchors_torch(
    persistence: torch.Tensor,
    gawron: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    persistence_n = persistence.float().clamp_min(0.0)
    persistence_n = persistence_n / persistence_n.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)
    gawron_n = gawron.float().clamp_min(0.0)
    gawron_n = gawron_n / gawron_n.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    gate_n = gate.float().clamp(0.0, 1.0)
    return (1.0 - gate_n) * persistence_n + gate_n * gawron_n


def _route_distribution(
    *,
    model: SharedOdRouteResidualPredictor,
    features: torch.Tensor,
    gawron_anchor: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    residual = model(features)
    anchor = gawron_anchor.float().clamp_min(0.0)
    anchor = anchor / anchor.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if float(scale) == 0.0:
        return anchor
    return torch.softmax(
        torch.log(anchor.clamp_min(1e-12)) + float(scale) * residual.float(),
        dim=-1,
    )


def _active_route_metrics(
    target: np.ndarray,
    pred: np.ndarray,
    od_weight: np.ndarray,
) -> Dict[str, float]:
    target_n = _normalise_rows(target)
    pred_n = _normalise_rows(pred)
    weight = np.asarray(od_weight, dtype=np.float64).clip(min=0.0)
    weight = weight / np.maximum(weight.sum(axis=-1, keepdims=True), 1e-12)
    ce = -(target_n * np.log(np.clip(pred_n, 1e-12, 1.0))).sum(axis=-1)
    tv = 0.5 * np.abs(target_n - pred_n).sum(axis=-1)
    top1 = (np.argmax(target_n, axis=-1) == np.argmax(pred_n, axis=-1)).astype(
        np.float64
    )
    return {
        "active_od_route_ce": float((ce * weight).sum(axis=-1).mean()),
        "active_od_route_tv": float((tv * weight).sum(axis=-1).mean()),
        "active_od_route_top1": float((top1 * weight).sum(axis=-1).mean()),
    }


def paired_route_metrics(
    *,
    target: np.ndarray,
    persistence: np.ndarray,
    pred: np.ndarray,
    od_weight: np.ndarray,
    block_size: int,
    bootstrap_samples: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Return population-weighted route metrics paired against persistence."""
    target_n = _normalise_rows(target)
    persistence_n = _normalise_rows(persistence)
    pred_n = _normalise_rows(pred)
    if target_n.shape != persistence_n.shape or target_n.shape != pred_n.shape:
        raise ValueError(
            "route metric shape mismatch: "
            f"target={target_n.shape}, persistence={persistence_n.shape}, "
            f"pred={pred_n.shape}"
        )
    weight = np.asarray(od_weight, dtype=np.float64).clip(min=0.0)
    if weight.shape != target_n.shape[:-1]:
        raise ValueError(
            f"route metric OD-weight shape mismatch: {weight.shape} vs "
            f"{target_n.shape[:-1]}"
        )
    weight = weight / np.maximum(weight.sum(axis=-1, keepdims=True), 1e-12)
    route_tv = (0.5 * np.abs(target_n - pred_n).sum(axis=-1) * weight).sum(axis=-1)
    persistence_tv = (0.5 * np.abs(target_n - persistence_n).sum(axis=-1) * weight).sum(
        axis=-1
    )
    route_ce = (
        -(target_n * np.log(np.clip(pred_n, 1e-12, 1.0))).sum(axis=-1) * weight
    ).sum(axis=-1)
    route_top1 = (
        (np.argmax(target_n, axis=-1) == np.argmax(pred_n, axis=-1)).astype(np.float64)
        * weight
    ).sum(axis=-1)
    delta = route_tv - persistence_tv
    ci_low, ci_high = _moving_block_bootstrap_ci(
        delta,
        block_size=block_size,
        samples=bootstrap_samples,
        seed=seed,
    )
    metrics: Dict[str, Any] = {
        "active_od_route_ce": float(route_ce.mean()),
        "active_od_route_tv": float(route_tv.mean()),
        "active_od_route_top1": float(route_top1.mean()),
        "persistence_active_od_route_tv": float(persistence_tv.mean()),
        "route_tv_delta_vs_persistence_mean": float(delta.mean()),
        "route_tv_delta_vs_persistence_ci95": [float(ci_low), float(ci_high)],
        "route_win_fraction_vs_persistence": float(np.mean(route_tv < persistence_tv)),
        "route_tie_fraction_vs_persistence": float(
            np.mean(np.isclose(route_tv, persistence_tv, atol=1e-12))
        ),
    }
    rows = {
        "route_tv": route_tv,
        "persistence_route_tv": persistence_tv,
        "route_tv_delta_vs_persistence": delta,
        "route_ce": route_ce,
        "route_top1": route_top1,
    }
    return metrics, rows


def route_candidate_acceptance(
    *,
    route_metrics: Dict[str, Any],
    candidate_edge_tv: float,
    persistence_edge_tv: float,
    min_absolute_route_improvement: float,
    min_relative_route_improvement: float,
    max_absolute_edge_degradation: float,
    max_relative_edge_degradation: float,
    min_route_win_fraction: float,
) -> Dict[str, Any]:
    """Pre-registered validation-only guard for a route response candidate."""
    candidate_route_tv = float(route_metrics["active_od_route_tv"])
    persistence_route_tv = float(route_metrics["persistence_active_od_route_tv"])
    route_gain = persistence_route_tv - candidate_route_tv
    required_route_gain = max(
        float(max(min_absolute_route_improvement, 0.0)),
        float(max(min_relative_route_improvement, 0.0)) * persistence_route_tv,
    )
    edge_delta = float(candidate_edge_tv) - float(persistence_edge_tv)
    allowed_edge_delta = max(
        float(max(max_absolute_edge_degradation, 0.0)),
        float(max(max_relative_edge_degradation, 0.0)) * float(persistence_edge_tv),
    )
    win_fraction = float(route_metrics["route_win_fraction_vs_persistence"])
    accepted = bool(
        route_gain > required_route_gain
        and edge_delta <= allowed_edge_delta
        and win_fraction >= float(min_route_win_fraction)
    )
    return {
        "accepted": accepted,
        "route_gain": float(route_gain),
        "required_route_gain": float(required_route_gain),
        "edge_delta": float(edge_delta),
        "allowed_edge_delta": float(allowed_edge_delta),
        "route_win_fraction": float(win_fraction),
        "required_route_win_fraction": float(min_route_win_fraction),
    }


def train_route_predictor(
    *,
    dataset: StaticHdvDataset,
    gawron: GawronRouteRollout,
    pairs: CausalEpisodePairs,
    scales: Sequence[float],
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    residual_penalty: float,
    hidden_dim: int,
    depth: int,
    seed: int,
    device: torch.device,
    progress_denominator: int,
) -> Tuple[SharedOdRouteResidualPredictor, Dict[str, Any], Dict[str, np.ndarray]]:
    torch.manual_seed(int(seed))
    model = SharedOdRouteResidualPredictor(
        n_routes=int(dataset.n_routes),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    x_train = torch.from_numpy(
        _route_features(
            dataset,
            gawron,
            pairs.train,
            progress_denominator=progress_denominator,
        )
    ).to(device)
    anchor_train = torch.from_numpy(
        gawron.route_after_episode[pairs.train[:, 0]].astype(np.float32)
    ).to(device)
    target_train = torch.from_numpy(
        dataset.hdv_route[pairs.train[:, 1]].astype(np.float32)
    ).to(device)
    weight_train = torch.from_numpy(
        dataset.hdv_od_weight[pairs.train[:, 1]].astype(np.float32)
    ).to(device)
    x_val = torch.from_numpy(
        _route_features(
            dataset,
            gawron,
            pairs.validation,
            progress_denominator=progress_denominator,
        )
    ).to(device)
    anchor_val_np = gawron.route_after_episode[pairs.validation[:, 0]]
    anchor_val = torch.from_numpy(anchor_val_np.astype(np.float32)).to(device)
    target_val_edge = dataset.hdv_edge[pairs.validation[:, 1]]
    projection_weight_val = dataset.hdv_od_weight[pairs.validation[:, 0]]
    gawron_val_edge = project_route_probs_to_edges(
        anchor_val_np,
        projection_weight_val,
        dataset.route_to_edge,
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_scale = 1.0
    best_val_tv = float("inf")
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, max(int(epochs), 1) + 1):
        model.train()
        pred = _route_distribution(
            model=model,
            features=x_train,
            gawron_anchor=anchor_train,
            scale=1.0,
        )
        ce = -(target_train * torch.log(pred.clamp_min(1e-12))).sum(dim=-1)
        loss = (ce * weight_train).sum(dim=-1).mean()
        if float(residual_penalty) > 0.0:
            loss = loss + float(residual_penalty) * model(x_train).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            residual_val = model(x_val).detach().cpu().numpy()
        candidate_scales = tuple(
            float(scale) for scale in scales if float(scale) >= 0.0
        )
        if not candidate_scales:
            raise ValueError("route residual training requires a non-negative scale")
        anchor_log = np.log(np.clip(_normalise_rows(anchor_val_np), 1e-12, None))
        route_candidates = [
            _normalise_rows(
                np.exp(
                    anchor_log
                    + float(scale) * residual_val
                    - (anchor_log + float(scale) * residual_val).max(
                        axis=-1,
                        keepdims=True,
                    )
                )
            )
            for scale in candidate_scales
        ]
        scores = [
            float(
                _tv_rows(
                    target_val_edge,
                    project_route_probs_to_edges(
                        route_pred,
                        hdv_counts=projection_weight_val,
                        route_to_edge=dataset.route_to_edge,
                    ),
                ).mean()
            )
            for route_pred in route_candidates
        ]
        best_idx = min(
            range(len(scores)),
            key=lambda idx: (scores[idx], float(candidate_scales[idx])),
        )
        val_tv = float(scores[best_idx])
        selected_scale = float(candidate_scales[best_idx])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(loss.detach().cpu().item()),
                "validation_edge_tv": val_tv,
                "residual_scale": selected_scale,
            }
        )
        if val_tv < best_val_tv - 1e-9:
            best_val_tv = val_tv
            best_scale = selected_scale
            best_epoch = int(epoch)
            best_state = _clone_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= max(int(patience), 1):
            break
    if best_state is None:
        raise RuntimeError(
            "route predictor training did not produce a validation state"
        )
    model.load_state_dict(best_state, strict=True)
    model.eval()

    predictions: Dict[str, np.ndarray] = {}
    for split_name, split_pairs in (
        ("train", pairs.train),
        ("validation", pairs.validation),
        ("test", pairs.test),
    ):
        features = torch.from_numpy(
            _route_features(
                dataset,
                gawron,
                split_pairs,
                progress_denominator=progress_denominator,
            )
        ).to(device)
        anchor = torch.from_numpy(
            gawron.route_after_episode[split_pairs[:, 0]].astype(np.float32)
        ).to(device)
        with torch.no_grad():
            predictions[split_name] = (
                _route_distribution(
                    model=model,
                    features=features,
                    gawron_anchor=anchor,
                    scale=float(best_scale),
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    metadata: Dict[str, Any] = {
        "parameterization": "gawron_anchored_od_route_residual_logit",
        "features": [
            "previous_av_route",
            "previous_hdv_route_probability",
            "previous_hdv_route_cost_memory",
            "previous_route_travel_time",
            "previous_route_travel_time_observed",
            "previous_hdv_od_weight",
            "episode_progress",
        ],
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "best_validation_edge_tv": float(best_val_tv),
        "selected_residual_scale": float(best_scale),
        "validation_persistence_edge_tv": float(
            _tv_rows(
                target_val_edge,
                dataset.hdv_edge[pairs.validation[:, 0]],
            ).mean()
        ),
        "validation_gawron_edge_tv": float(
            _tv_rows(target_val_edge, gawron_val_edge).mean()
        ),
        "history_tail": history[-10:],
    }
    return model, metadata, predictions


def train_persistence_anchored_gate(
    *,
    dataset: StaticHdvDataset,
    gawron: GawronRouteRollout,
    route_tt_sample_counts: np.ndarray,
    pairs: CausalEpisodePairs,
    scales: Sequence[float],
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    gate_penalty: float,
    hidden_dim: int,
    depth: int,
    seed: int,
    device: torch.device,
    progress_denominator: int,
    selection_split: str = "validation",
    response_mode: str = "gate",
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, np.ndarray]]:
    """Train a persistence-anchored scalar gate or unrestricted route residual."""
    selection_name = str(selection_split).strip().lower()
    if selection_name not in {"train", "validation"}:
        raise ValueError("selection_split must be either 'train' or 'validation'")
    parameterization = str(response_mode).strip().lower()
    if parameterization not in {"gate", "residual"}:
        raise ValueError("response_mode must be either 'gate' or 'residual'")
    torch.manual_seed(int(seed))
    x_train_np = _persistence_gate_features(
        dataset,
        gawron,
        route_tt_sample_counts,
        pairs.train,
        progress_denominator=progress_denominator,
    )
    if parameterization == "residual":
        model = PersistenceAnchoredOdRouteResidual(
            input_dim=int(x_train_np.shape[-1]),
            n_routes=int(dataset.n_routes),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
        ).to(device)
    else:
        model = PersistenceAnchoredOdRouteGate(
            input_dim=int(x_train_np.shape[-1]),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    def split_arrays(
        split_pairs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        previous_idx = split_pairs[:, 0]
        current_idx = split_pairs[:, 1]
        persistence = causal_last_observed_route_counts(
            dataset.hdv_route_counts,
            gawron.route_before_episode[0],
        )[previous_idx]
        return (
            _persistence_gate_features(
                dataset,
                gawron,
                route_tt_sample_counts,
                split_pairs,
                progress_denominator=progress_denominator,
            ),
            persistence,
            gawron.route_after_episode[previous_idx].astype(np.float64),
            dataset.hdv_route[current_idx].astype(np.float64),
            dataset.hdv_od_weight[current_idx].astype(np.float64),
        )

    split_pairs_by_name = {
        "train": pairs.train,
        "validation": pairs.validation,
        "test": pairs.test,
    }
    split_data = {
        name: split_arrays(split_pairs)
        for name, split_pairs in split_pairs_by_name.items()
    }
    (
        train_x,
        train_persistence,
        train_gawron,
        train_target,
        train_weight,
    ) = split_data["train"]
    (
        val_x,
        val_persistence,
        val_gawron,
        val_target,
        val_weight,
    ) = split_data["validation"]
    x_train = torch.from_numpy(train_x).to(device=device, dtype=torch.float32)
    persistence_train = torch.from_numpy(train_persistence).to(
        device=device,
        dtype=torch.float32,
    )
    gawron_train = torch.from_numpy(train_gawron).to(
        device=device,
        dtype=torch.float32,
    )
    target_train = torch.from_numpy(train_target).to(
        device=device,
        dtype=torch.float32,
    )
    weight_train = torch.from_numpy(train_weight).to(
        device=device,
        dtype=torch.float32,
    )
    (
        selection_x,
        selection_persistence,
        selection_gawron,
        selection_target,
        selection_weight,
    ) = split_data[selection_name]
    x_selection = torch.from_numpy(selection_x).to(
        device=device,
        dtype=torch.float32,
    )
    candidate_scales = tuple(
        sorted({float(scale) for scale in scales if float(scale) >= 0.0})
    )
    if not candidate_scales or candidate_scales[0] != 0.0:
        raise ValueError("persistence gate scales must include zero")

    selection_pairs = split_pairs_by_name[selection_name]
    selection_previous_idx = selection_pairs[:, 0]
    selection_current_idx = selection_pairs[:, 1]
    selection_edge_target = dataset.hdv_edge[selection_current_idx]
    selection_edge_persistence = dataset.hdv_edge[selection_previous_idx]
    selection_persistence_metrics, _ = paired_route_metrics(
        target=selection_target,
        persistence=selection_persistence,
        pred=selection_persistence,
        od_weight=selection_weight,
        block_size=1,
        bootstrap_samples=1,
        seed=int(seed),
    )
    selection_persistence_route_tv = float(
        selection_persistence_metrics["active_od_route_tv"]
    )
    selection_persistence_edge_tv = float(
        _tv_rows(selection_edge_target, selection_edge_persistence).mean()
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_scale = 0.0
    best_selection_route_tv = float("inf")
    best_selection_edge_tv = float("inf")
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, max(int(epochs), 1) + 1):
        model.train()
        raw_response = model(x_train)
        if parameterization == "residual":
            pred = apply_persistence_anchored_route_residual_torch(
                persistence_train,
                raw_response,
                scale=1.0,
            )
        else:
            pred = _blend_route_anchors_torch(
                persistence_train,
                gawron_train,
                raw_response,
            )
        route_tv_od = 0.5 * torch.abs(pred - target_train).sum(dim=-1)
        route_tv_loss = (route_tv_od * weight_train).sum(dim=-1).mean()
        if parameterization == "residual":
            # Optimize the same population-weighted OD-route TV used for
            # checkpoint selection. Cross-entropy makes the already tiny
            # response gradient vanish behind the persistence distribution on
            # high-count real networks.
            fit_loss = route_tv_loss
            regularizer = raw_response.square().mean()
        else:
            fit_loss = route_tv_loss
            regularizer = (
                raw_response.squeeze(-1) * weight_train
            ).sum(dim=-1).mean()
        loss = fit_loss + float(gate_penalty) * regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            raw_response = (
                model(x_selection).detach().cpu().numpy().astype(np.float64)
            )
        epoch_candidates: List[Tuple[float, float, float]] = []
        for scale in candidate_scales:
            if parameterization == "residual":
                route_pred = apply_persistence_anchored_route_residual(
                    selection_persistence,
                    raw_response,
                    scale=float(scale),
                )
            else:
                route_pred = blend_route_anchors(
                    selection_persistence,
                    selection_gawron,
                    gate=float(scale) * raw_response,
                )
            route_metrics, _ = paired_route_metrics(
                target=selection_target,
                persistence=selection_persistence,
                pred=route_pred,
                od_weight=selection_weight,
                block_size=1,
                bootstrap_samples=1,
                seed=int(seed),
            )
            edge_pred = project_route_response_to_edges(
                predicted_route=route_pred,
                previous_route=selection_persistence,
                hdv_counts=dataset.hdv_od_weight[selection_previous_idx],
                route_to_edge=dataset.route_to_edge,
                previous_edge=selection_edge_persistence,
            )
            epoch_candidates.append(
                (
                    float(scale),
                    float(route_metrics["active_od_route_tv"]),
                    float(_tv_rows(selection_edge_target, edge_pred).mean()),
                )
            )
        selected_scale, selection_route_tv, selection_edge_tv = min(
            epoch_candidates,
            key=lambda row: (row[1], row[2], row[0]),
        )
        history_row = {
            "epoch": float(epoch),
            "train_loss": float(loss.detach().cpu().item()),
            "train_route_tv": float(route_tv_loss.detach().cpu().item()),
            "selection_route_tv": float(selection_route_tv),
            "selection_edge_tv": float(selection_edge_tv),
            "gate_scale": float(selected_scale),
            "mean_raw_response": float(raw_response.mean()),
        }
        if parameterization == "gate":
            history_row["mean_raw_gate"] = float(raw_response.mean())
        history_row[f"{selection_name}_route_tv"] = float(selection_route_tv)
        history_row[f"{selection_name}_edge_tv"] = float(selection_edge_tv)
        history.append(history_row)
        if selection_route_tv < best_selection_route_tv - 1e-9 or (
            abs(selection_route_tv - best_selection_route_tv) <= 1e-9
            and (
                selection_edge_tv < best_selection_edge_tv - 1e-9
                or (
                    abs(selection_edge_tv - best_selection_edge_tv) <= 1e-9
                    and selected_scale < best_scale
                )
            )
        ):
            best_state = _clone_state(model)
            best_epoch = int(epoch)
            best_scale = float(selected_scale)
            best_selection_route_tv = float(selection_route_tv)
            best_selection_edge_tv = float(selection_edge_tv)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= max(int(patience), 1):
            break
    if best_state is None:
        raise RuntimeError(
            "persistence-anchored gate produced no "
            f"{selection_name} selection state"
        )
    model.load_state_dict(best_state, strict=True)
    model.eval()

    predictions: Dict[str, np.ndarray] = {}
    gate_predictions: Dict[str, np.ndarray] = {}
    for split_name, split_pairs in split_pairs_by_name.items():
        features, persistence, gawron_proposal, _, _ = split_data[split_name]
        with torch.no_grad():
            raw_response = (
                model(
                    torch.from_numpy(features).to(
                        device=device,
                        dtype=torch.float32,
                    )
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        gate_predictions[split_name] = raw_response
        if parameterization == "residual":
            predictions[split_name] = apply_persistence_anchored_route_residual(
                persistence,
                raw_response,
                scale=float(best_scale),
            )
        else:
            predictions[split_name] = blend_route_anchors(
                persistence,
                gawron_proposal,
                gate=float(best_scale) * raw_response,
            )
    final_split_metrics: Dict[str, Dict[str, float]] = {}
    for split_name, split_pairs in split_pairs_by_name.items():
        _, persistence, _, target, weight = split_data[split_name]
        route_metrics, _ = paired_route_metrics(
            target=target,
            persistence=persistence,
            pred=predictions[split_name],
            od_weight=weight,
            block_size=1,
            bootstrap_samples=1,
            seed=int(seed),
        )
        persistence_metrics, _ = paired_route_metrics(
            target=target,
            persistence=persistence,
            pred=persistence,
            od_weight=weight,
            block_size=1,
            bootstrap_samples=1,
            seed=int(seed),
        )
        previous_idx = split_pairs[:, 0]
        current_idx = split_pairs[:, 1]
        predicted_edge = project_route_response_to_edges(
            predicted_route=predictions[split_name],
            previous_route=persistence,
            hdv_counts=dataset.hdv_od_weight[previous_idx],
            route_to_edge=dataset.route_to_edge,
            previous_edge=dataset.hdv_edge[previous_idx],
        )
        final_split_metrics[split_name] = {
            "route_tv": float(route_metrics["active_od_route_tv"]),
            "persistence_route_tv": float(
                persistence_metrics["active_od_route_tv"]
            ),
            "edge_tv": float(
                _tv_rows(dataset.hdv_edge[current_idx], predicted_edge).mean()
            ),
            "persistence_edge_tv": float(
                _tv_rows(
                    dataset.hdv_edge[current_idx],
                    dataset.hdv_edge[previous_idx],
                ).mean()
            ),
        }
    metadata: Dict[str, Any] = {
        "parameterization": (
            "empirical_persistence_anchored_route_residual"
            if parameterization == "residual"
            else "empirical_persistence_anchored_gawron_gate"
        ),
        "response_mode": parameterization,
        "features": [
            "previous_causal_last_observed_hdv_route",
            "previous_latent_hdv_route",
            "gawron_next_route",
            "gawron_route_delta",
            "previous_route_cost_gap",
            "previous_route_travel_time",
            "previous_route_travel_time_observed",
            "previous_hdv_route_sample_counts",
            "previous_route_travel_time_sample_counts",
            "empirical_gawron_disagreement",
            "gawron_response_magnitude",
            "empirical_route_entropy",
            "previous_hdv_od_sample_count",
            "previous_travel_time_sample_count",
            "previous_hdv_od_weight",
            "episode_progress",
        ],
        "selection_split": selection_name,
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "selected_gate_scale": float(best_scale),
        "selected_residual_scale": (
            float(best_scale) if parameterization == "residual" else None
        ),
        "anchor_floor": (
            float(HDV_OD_ROUTE_RESIDUAL_ANCHOR_FLOOR)
            if parameterization == "residual"
            else None
        ),
        "best_selection_route_tv": float(best_selection_route_tv),
        "best_selection_edge_tv": float(best_selection_edge_tv),
        "selection_persistence_route_tv": float(
            selection_persistence_route_tv
        ),
        "selection_persistence_edge_tv": float(
            selection_persistence_edge_tv
        ),
        "best_validation_route_tv": float(
            final_split_metrics["validation"]["route_tv"]
        ),
        "best_validation_edge_tv": float(
            final_split_metrics["validation"]["edge_tv"]
        ),
        "validation_persistence_route_tv": float(
            final_split_metrics["validation"]["persistence_route_tv"]
        ),
        "validation_persistence_edge_tv": float(
            final_split_metrics["validation"]["persistence_edge_tv"]
        ),
        "train_route_tv": float(final_split_metrics["train"]["route_tv"]),
        "train_edge_tv": float(final_split_metrics["train"]["edge_tv"]),
        "train_persistence_route_tv": float(
            final_split_metrics["train"]["persistence_route_tv"]
        ),
        "train_persistence_edge_tv": float(
            final_split_metrics["train"]["persistence_edge_tv"]
        ),
        "validation_mean_raw_response": float(
            gate_predictions["validation"].mean()
        ),
        "train_mean_raw_response": float(gate_predictions["train"].mean()),
        "history_tail": history[-10:],
    }
    if parameterization == "gate":
        metadata["validation_mean_raw_gate"] = float(
            gate_predictions["validation"].mean()
        )
        metadata["train_mean_raw_gate"] = float(
            gate_predictions["train"].mean()
        )
    return model, metadata, predictions


def _load_legacy_predictor(
    *,
    checkpoint: Path,
    n_edges: int,
    device: torch.device,
    config_path: Optional[Path] = None,
) -> CausalHdvResponseModel:
    state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError(f"legacy checkpoint is not a dict: {checkpoint}")
    cfg_state = dict(state.get("cfg")) if isinstance(state.get("cfg"), dict) else {}
    if config_path is not None and config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config_raw = json.load(handle)
        if isinstance(config_raw, dict):
            cfg_state.update({str(key): value for key, value in config_raw.items()})
    hidden_dim = int(
        state.get(
            "hidden_dim",
            cfg_state.get("hdv_response_predictor_hidden_dim", 128),
        )
    )
    depth = int(
        state.get(
            "depth",
            cfg_state.get("hdv_response_predictor_depth", 2),
        )
    )
    progress_denominator = int(
        state.get("progress_denominator")
        or cfg_state.get("hdv_response_predictor_progress_denominator")
        or cfg_state.get("train_episodes")
        or 1
    )
    parameterization = str(
        state.get(
            "parameterization",
            state.get("hdv_response_parameterization", "absolute_logits"),
        )
    )
    residual_scale_raw = state.get(
        "residual_scale",
        state.get("hdv_response_residual_scale", 1.0),
    )
    residual_scale = float(1.0 if residual_scale_raw is None else residual_scale_raw)
    cfg = OnlineWmDreamerConfig(
        train_episodes=progress_denominator,
        hdv_response_predictor_enabled=True,
        hdv_response_predictor_hidden_dim=hidden_dim,
        hdv_response_predictor_depth=depth,
        hdv_response_predictor_progress_denominator=progress_denominator,
        hdv_response_predictor_parameterization=parameterization,
        hdv_response_predictor_residual_scale=residual_scale,
    )
    model = CausalHdvResponseModel(n_edges=n_edges, cfg=cfg, device=device)
    model_state = state.get("model_state_dict") or state.get("model")
    if model_state is None:
        model_state = state.get("hdv_response_predictor")
    if model_state is None:
        raise RuntimeError(f"no HDV response predictor state in {checkpoint}")
    model.parameterization = model._normalise_parameterization(parameterization)
    model.residual_scale = residual_scale
    model.model.load_state_dict(model_state, strict=True)
    model.trained_pairs = int(
        state.get("trained_pairs", state.get("hdv_response_trained_pairs", 0)) or 0
    )
    model.model.eval()
    return model


def predict_legacy(
    *,
    model: CausalHdvResponseModel,
    dataset: StaticHdvDataset,
    pairs: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    features = torch.from_numpy(
        _edge_features(
            dataset,
            pairs,
            progress_denominator=int(model.progress_denominator),
        )
    ).to(device)
    with torch.no_grad():
        return (
            model.predict_distribution(features)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )


def _mean_finite(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def _moving_block_bootstrap_ci(
    values: np.ndarray,
    *,
    block_size: int,
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if int(arr.size) == 0:
        return float("nan"), float("nan")
    if int(arr.size) == 1:
        return float(arr[0]), float(arr[0])
    block = max(1, min(int(block_size), int(arr.size)))
    starts = np.arange(0, int(arr.size) - block + 1, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty((max(int(samples), 1),), dtype=np.float64)
    blocks_needed = int(math.ceil(float(arr.size) / float(block)))
    for sample_idx in range(estimates.size):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        draw = np.concatenate([arr[start : start + block] for start in chosen])[
            : arr.size
        ]
        estimates[sample_idx] = float(draw.mean())
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _candidate_metrics(
    *,
    target: np.ndarray,
    persistence: np.ndarray,
    pred: np.ndarray,
    block_size: int,
    bootstrap_samples: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    target_n = _normalise_rows(target)
    persistence_n = _normalise_rows(persistence)
    pred_n = _normalise_rows(pred)
    tv = _tv_rows(target_n, pred_n)
    persistence_tv = _tv_rows(target_n, persistence_n)
    delta = tv - persistence_tv
    ci_low, ci_high = _moving_block_bootstrap_ci(
        delta,
        block_size=block_size,
        samples=bootstrap_samples,
        seed=seed,
    )
    ce = -(target_n * np.log(np.clip(pred_n, 1e-12, 1.0))).sum(axis=-1)
    target_entropy = -(target_n * np.log(np.clip(target_n, 1e-12, 1.0))).sum(axis=-1)
    observed_response = target_n - persistence_n
    predicted_response = pred_n - persistence_n
    observed_magnitude = 0.5 * np.abs(observed_response).sum(axis=-1)
    predicted_magnitude = 0.5 * np.abs(predicted_response).sum(axis=-1)
    numerator = (observed_response * predicted_response).sum(axis=-1)
    denominator = np.linalg.norm(observed_response, axis=-1) * np.linalg.norm(
        predicted_response, axis=-1
    )
    direction = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-12,
    )
    metrics: Dict[str, Any] = {
        "pairs": int(target_n.shape[0]),
        "edge_tv_mean": float(tv.mean()),
        "edge_ce_mean": float(ce.mean()),
        "edge_kl_mean": float((ce - target_entropy).mean()),
        "persistence_edge_tv_mean": float(persistence_tv.mean()),
        "edge_tv_delta_vs_persistence_mean": float(delta.mean()),
        "edge_tv_delta_vs_persistence_ci95": [ci_low, ci_high],
        "win_fraction_vs_persistence": float(np.mean(tv < persistence_tv)),
        "tie_fraction_vs_persistence": float(np.mean(np.isclose(tv, persistence_tv))),
        "observed_response_tv_mean": float(observed_magnitude.mean()),
        "predicted_response_tv_mean": float(predicted_magnitude.mean()),
        "response_magnitude_abs_error_mean": float(
            np.abs(predicted_magnitude - observed_magnitude).mean()
        ),
        "response_direction_cosine_mean": _mean_finite(direction),
    }
    rows = {
        "edge_tv": tv,
        "persistence_edge_tv": persistence_tv,
        "delta_vs_persistence": delta,
        "observed_response_tv": observed_magnitude,
        "predicted_response_tv": predicted_magnitude,
        "response_direction_cosine": direction,
    }
    return metrics, rows


def run_offline_comparison(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.set_num_threads(max(int(args.torch_threads), 1))
    device = torch.device(str(args.device))
    run_dir = Path(args.run_dir).resolve()
    episodes_dir = (
        Path(args.episodes_dir).resolve()
        if args.episodes_dir
        else run_dir / "collector_runtime" / "episodes"
    )
    route_xml = (
        Path(args.route_xml).resolve()
        if args.route_xml
        else run_dir / "collector_runtime" / "route.rou.xml"
    )
    edge_registry = Path(args.edge_registry).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_static_hdv_dataset(
        run_dir=run_dir,
        route_xml=route_xml,
        edge_registry=edge_registry,
        episodes_dir=episodes_dir,
        av_action_col="initial_route_action",
        hdv_action_col="hdv_action",
        require_seen_in_episode=True,
        strict_route_actions=True,
    )
    # City-scale route incidence is extremely sparse. Converting it once avoids
    # repeatedly multiplying a multi-hundred-megabyte dense tensor during
    # validation-scale search (Fontenay: 2,018 ODs x 4 routes x 7,585 edges).
    dataset.route_to_edge = SparseRouteEdgeProjection.from_dense(dataset.route_to_edge)
    target_source = load_edge_trace_targets(run_dir=run_dir, dataset=dataset)
    reconstructed_strict_edge = project_route_probs_to_edges(
        dataset.hdv_route,
        dataset.hdv_od_weight,
        dataset.route_to_edge,
    )
    target_alignment_tv = _tv_rows(
        dataset.hdv_edge,
        reconstructed_strict_edge,
    )
    if float(target_alignment_tv.max()) > float(args.target_alignment_tolerance):
        raise RuntimeError(
            "strict hdv_action route target does not reproduce the saved edge "
            "target: max TV "
            f"{float(target_alignment_tv.max()):.6g} exceeds "
            f"{float(args.target_alignment_tolerance):.6g}"
        )
    pairs = build_causal_episode_pairs(
        dataset.episodes,
        train_end_episode=int(args.train_end_episode),
        validation_end_episode=int(args.validation_end_episode),
        test_start_episode=int(args.test_start_episode),
    )
    progress_denominator = int(
        args.progress_denominator
        if int(args.progress_denominator) > 0
        else int(dataset.episodes.max())
    )
    scales = tuple(float(value) for value in str(args.residual_scales).split(","))
    if 0.0 not in scales:
        raise ValueError(
            "--residual-scales must include 0 so persistence remains selectable"
        )
    scales = tuple(sorted(set(scales)))
    target_test = dataset.hdv_edge[pairs.test[:, 1]]
    persistence_test = dataset.hdv_edge[pairs.test[:, 0]]
    target_val = dataset.hdv_edge[pairs.validation[:, 1]]
    persistence_val = dataset.hdv_edge[pairs.validation[:, 0]]

    residual_model: Optional[CausalHdvResponseModel] = None
    residual_train: Optional[Dict[str, Any]] = None
    residual_predictions: Optional[Dict[str, np.ndarray]] = None
    absolute_train: Optional[Dict[str, Any]] = None
    absolute_predictions: Optional[Dict[str, np.ndarray]] = None
    if bool(args.train_edge_baselines):
        residual_model, residual_train, residual_predictions = train_edge_predictor(
            dataset=dataset,
            pairs=pairs,
            parameterization="residual_logit",
            scales=scales,
            epochs=int(args.epochs),
            patience=int(args.patience),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            residual_penalty=float(args.residual_penalty),
            hidden_dim=int(args.hidden_dim),
            depth=int(args.depth),
            seed=int(args.seed),
            device=device,
            progress_denominator=progress_denominator,
        )
        _, absolute_train, absolute_predictions = train_edge_predictor(
            dataset=dataset,
            pairs=pairs,
            parameterization="absolute_logits",
            scales=(1.0,),
            epochs=int(args.epochs),
            patience=int(args.patience),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            residual_penalty=0.0,
            hidden_dim=int(args.hidden_dim),
            depth=int(args.depth),
            seed=int(args.seed),
            device=device,
            progress_denominator=progress_denominator,
        )

    route_tt, route_tt_mask, route_tt_sample_counts = load_route_travel_time_features(
        dataset=dataset,
        episodes_dir=episodes_dir,
    )
    initial_state_path, gawron_settings = load_gawron_config(
        run_dir=run_dir,
        initial_hdv_state=str(args.initial_hdv_state),
    )
    initial_route, initial_cost = load_initial_hdv_route_state(
        state_path=initial_state_path,
        dataset=dataset,
    )
    configured_gawron_rollout = replay_gawron_route_state(
        initial_route=initial_route,
        initial_cost=initial_cost,
        route_tt=route_tt,
        route_tt_observed=route_tt_mask,
        mode=str(gawron_settings["mode"]),
        lr=float(gawron_settings["lr"]),
        alpha_zero=float(gawron_settings["alpha_zero"]),
        beta=float(gawron_settings["beta"]),
        update_mode=str(gawron_settings["update_mode"]),
        min_prob=float(gawron_settings["min_prob"]),
    )
    gawron_alpha_candidates = tuple(
        float(value)
        for value in str(args.gawron_alpha_grid).split(",")
        if str(value).strip()
    )
    gawron_calibration = calibrate_gawron_alpha(
        dataset=dataset,
        pairs=pairs,
        initial_route=initial_route,
        initial_cost=initial_cost,
        route_tt=route_tt,
        route_tt_observed=route_tt_mask,
        settings=gawron_settings,
        alpha_candidates=gawron_alpha_candidates,
        min_improvement=float(args.validation_min_improvement),
        min_relative_improvement=float(args.validation_min_relative_improvement),
    )
    gawron_rollout = gawron_calibration.rollout
    route_model, route_train, route_predictions = train_route_predictor(
        dataset=dataset,
        gawron=gawron_rollout,
        pairs=pairs,
        scales=scales,
        epochs=int(args.route_epochs),
        patience=int(args.patience),
        lr=float(args.route_lr),
        weight_decay=float(args.weight_decay),
        residual_penalty=float(args.residual_penalty),
        hidden_dim=int(args.route_hidden_dim),
        depth=int(args.route_depth),
        seed=int(args.seed),
        device=device,
        progress_denominator=progress_denominator,
    )
    gate_model, gate_train, gate_route_predictions = train_persistence_anchored_gate(
        dataset=dataset,
        gawron=gawron_rollout,
        route_tt_sample_counts=route_tt_sample_counts,
        pairs=pairs,
        scales=scales,
        epochs=int(args.route_epochs),
        patience=int(args.patience),
        lr=float(args.route_lr),
        weight_decay=float(args.weight_decay),
        gate_penalty=float(args.gate_penalty),
        hidden_dim=int(args.route_hidden_dim),
        depth=int(args.route_depth),
        seed=int(args.seed),
        device=device,
        progress_denominator=progress_denominator,
        selection_split=str(args.route_selection_split),
        response_mode=str(args.route_response_model),
    )
    learned_route_name = (
        "persistence_anchored_residual_od_route"
        if str(args.route_response_model) == "residual"
        else "persistence_anchored_gate_od_route"
    )
    causal_empirical_route_state = causal_last_observed_route_counts(
        dataset.hdv_route_counts,
        gawron_rollout.route_before_episode[0],
    )
    empirical_route_by_split = {
        split_name: causal_empirical_route_state[split_pairs[:, 0]]
        for split_name, split_pairs in (
            ("train", pairs.train),
            ("validation", pairs.validation),
            ("test", pairs.test),
        )
    }
    configured_gawron_train_route = configured_gawron_rollout.route_after_episode[
        pairs.train[:, 0]
    ]
    configured_gawron_val_route = configured_gawron_rollout.route_after_episode[
        pairs.validation[:, 0]
    ]
    configured_gawron_test_route = configured_gawron_rollout.route_after_episode[
        pairs.test[:, 0]
    ]
    gawron_train_route = gawron_rollout.route_after_episode[pairs.train[:, 0]]
    gawron_val_route = gawron_rollout.route_after_episode[pairs.validation[:, 0]]
    gawron_test_route = gawron_rollout.route_after_episode[pairs.test[:, 0]]

    route_train_predictions: Dict[str, np.ndarray] = {
        "persistence": empirical_route_by_split["train"],
        "configured_gawron_od_route": configured_gawron_train_route,
        "gawron_od_route": gawron_train_route,
        "gawron_residual_od_route": route_predictions["train"],
        learned_route_name: gate_route_predictions["train"],
    }
    route_validation_predictions: Dict[str, np.ndarray] = {
        "persistence": empirical_route_by_split["validation"],
        "configured_gawron_od_route": configured_gawron_val_route,
        "gawron_od_route": gawron_val_route,
        "gawron_residual_od_route": route_predictions["validation"],
        learned_route_name: gate_route_predictions["validation"],
    }
    route_test_predictions: Dict[str, np.ndarray] = {
        "persistence": empirical_route_by_split["test"],
        "configured_gawron_od_route": configured_gawron_test_route,
        "gawron_od_route": gawron_test_route,
        "gawron_residual_od_route": route_predictions["test"],
        learned_route_name: gate_route_predictions["test"],
    }

    def project_split_route_response(
        route_prediction: np.ndarray,
        split_pairs: np.ndarray,
        previous_route: np.ndarray,
    ) -> np.ndarray:
        previous_idx = split_pairs[:, 0]
        return project_route_response_to_edges(
            predicted_route=route_prediction,
            previous_route=previous_route,
            hdv_counts=dataset.hdv_od_weight[previous_idx],
            route_to_edge=dataset.route_to_edge,
            previous_edge=dataset.hdv_edge[previous_idx],
        )

    route_validation_edges = {
        name: project_split_route_response(
            route_prediction,
            pairs.validation,
            empirical_route_by_split["validation"],
        )
        for name, route_prediction in route_validation_predictions.items()
    }
    route_train_edges = {
        name: project_split_route_response(
            route_prediction,
            pairs.train,
            empirical_route_by_split["train"],
        )
        for name, route_prediction in route_train_predictions.items()
    }
    route_test_edges = {
        name: project_split_route_response(
            route_prediction,
            pairs.test,
            empirical_route_by_split["test"],
        )
        for name, route_prediction in route_test_predictions.items()
    }
    validation_persistence_tv = float(_tv_rows(target_val, persistence_val).mean())
    training_route_metrics: Dict[str, Dict[str, Any]] = {}
    training_tv: Dict[str, float] = {}
    target_train_edge = dataset.hdv_edge[pairs.train[:, 1]]
    for candidate_idx, name in enumerate(route_train_predictions):
        metrics, _ = paired_route_metrics(
            target=dataset.hdv_route[pairs.train[:, 1]],
            persistence=empirical_route_by_split["train"],
            pred=route_train_predictions[name],
            od_weight=dataset.hdv_od_weight[pairs.train[:, 1]],
            block_size=1,
            bootstrap_samples=1,
            seed=int(args.seed) + 1000 + candidate_idx,
        )
        training_route_metrics[name] = metrics
        training_tv[name] = float(
            _tv_rows(target_train_edge, route_train_edges[name]).mean()
        )
    validation_route_metrics: Dict[str, Dict[str, Any]] = {}
    validation_tv: Dict[str, float] = {}
    for candidate_idx, name in enumerate(route_validation_predictions):
        metrics, _ = paired_route_metrics(
            target=dataset.hdv_route[pairs.validation[:, 1]],
            persistence=empirical_route_by_split["validation"],
            pred=route_validation_predictions[name],
            od_weight=dataset.hdv_od_weight[pairs.validation[:, 1]],
            block_size=1,
            bootstrap_samples=1,
            seed=int(args.seed) + candidate_idx,
        )
        validation_route_metrics[name] = metrics
        validation_tv[name] = float(
            _tv_rows(target_val, route_validation_edges[name]).mean()
        )

    validation_acceptance: Dict[str, Dict[str, Any]] = {}
    for name in route_validation_predictions:
        if name == "persistence":
            continue
        validation_acceptance[name] = route_candidate_acceptance(
            route_metrics=validation_route_metrics[name],
            candidate_edge_tv=validation_tv[name],
            persistence_edge_tv=validation_persistence_tv,
            min_absolute_route_improvement=float(args.validation_min_improvement),
            min_relative_route_improvement=float(
                args.validation_min_relative_improvement
            ),
            max_absolute_edge_degradation=float(args.validation_max_edge_degradation),
            max_relative_edge_degradation=float(
                args.validation_max_relative_edge_degradation
            ),
            min_route_win_fraction=float(args.validation_min_route_win_fraction),
        )
    eligible_route_candidates = [
        "persistence",
        *[
            name
            for name, decision in validation_acceptance.items()
            if bool(decision["accepted"])
        ],
    ]
    validation_selected_route_source = min(
        eligible_route_candidates,
        key=lambda name: (
            float(validation_route_metrics[name]["active_od_route_tv"]),
            float(validation_tv[name]),
            name,
        ),
    )
    selection_split = str(args.route_selection_split).strip().lower()
    if selection_split == "train":
        gate_name = learned_route_name
        gate_beats_train_persistence = (
            float(training_route_metrics[gate_name]["active_od_route_tv"])
            < float(training_route_metrics["persistence"]["active_od_route_tv"])
            - 1e-12
        )
        selected_route_source = (
            gate_name
            if gate_beats_train_persistence
            and float(gate_train["selected_gate_scale"]) > 0.0
            else "persistence"
        )
    else:
        selected_route_source = validation_selected_route_source
    route_train_predictions["selected_od_route_predictor"] = (
        route_train_predictions[selected_route_source]
    )
    route_validation_predictions["selected_od_route_predictor"] = (
        route_validation_predictions[selected_route_source]
    )
    route_test_predictions["selected_od_route_predictor"] = route_test_predictions[
        selected_route_source
    ]
    route_train_edges["selected_od_route_predictor"] = route_train_edges[
        selected_route_source
    ]
    route_validation_edges["selected_od_route_predictor"] = route_validation_edges[
        selected_route_source
    ]
    route_test_edges["selected_od_route_predictor"] = route_test_edges[
        selected_route_source
    ]
    validation_route_metrics["selected_od_route_predictor"] = validation_route_metrics[
        selected_route_source
    ]
    validation_tv["selected_od_route_predictor"] = validation_tv[selected_route_source]
    training_route_metrics["selected_od_route_predictor"] = training_route_metrics[
        selected_route_source
    ]
    training_tv["selected_od_route_predictor"] = training_tv[selected_route_source]
    route_residual_validation_gate = bool(
        validation_acceptance["gawron_residual_od_route"]["accepted"]
    )
    persistence_gate_validation_gate = bool(
        validation_acceptance[learned_route_name]["accepted"]
    )

    predictions: Dict[str, np.ndarray] = {
        name: route_test_edges[name]
        for name in (
            "persistence",
            "configured_gawron_od_route",
            "gawron_od_route",
            "gawron_residual_od_route",
            learned_route_name,
            "selected_od_route_predictor",
        )
    }
    residual_validation_gate = False
    if residual_predictions is not None and absolute_predictions is not None:
        residual_validation_gate = bool(
            float(_tv_rows(target_val, residual_predictions["validation"]).mean())
            < validation_persistence_tv - float(args.validation_min_improvement)
        )
        predictions.update(
            {
                "offline_absolute_edge_mlp": absolute_predictions["test"],
                "residual_edge_mlp": residual_predictions["test"],
                "gated_residual_edge_mlp": (
                    residual_predictions["test"]
                    if residual_validation_gate
                    else persistence_test
                ),
            }
        )
        validation_tv.update(
            {
                "offline_absolute_edge_mlp": float(
                    _tv_rows(
                        target_val,
                        absolute_predictions["validation"],
                    ).mean()
                ),
                "residual_edge_mlp": float(
                    _tv_rows(
                        target_val,
                        residual_predictions["validation"],
                    ).mean()
                ),
                "gated_residual_edge_mlp": (
                    float(
                        _tv_rows(
                            target_val,
                            residual_predictions["validation"],
                        ).mean()
                    )
                    if residual_validation_gate
                    else validation_persistence_tv
                ),
            }
        )
    legacy_checkpoint = (
        Path(args.legacy_checkpoint).resolve() if args.legacy_checkpoint else None
    )
    legacy_model: Optional[CausalHdvResponseModel] = None
    if legacy_checkpoint is not None:
        legacy_model = _load_legacy_predictor(
            checkpoint=legacy_checkpoint,
            n_edges=len(dataset.edge_ids),
            device=device,
            config_path=run_dir / "online_wm_dreamer_config.json",
        )
        predictions["deployed_legacy_absolute_mlp"] = predict_legacy(
            model=legacy_model,
            dataset=dataset,
            pairs=pairs.test,
            device=device,
        )
        legacy_val = predict_legacy(
            model=legacy_model,
            dataset=dataset,
            pairs=pairs.validation,
            device=device,
        )
        validation_tv["deployed_legacy_absolute_mlp"] = float(
            _tv_rows(target_val, legacy_val).mean()
        )

    candidate_metrics: Dict[str, Any] = {}
    per_candidate_rows: Dict[str, Dict[str, np.ndarray]] = {}
    for candidate_idx, (name, pred) in enumerate(predictions.items()):
        metrics, metric_rows = _candidate_metrics(
            target=target_test,
            persistence=persistence_test,
            pred=pred,
            block_size=int(args.bootstrap_block_size),
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.seed) + candidate_idx,
        )
        metrics["validation_edge_tv"] = float(validation_tv[name])
        candidate_metrics[name] = metrics
        per_candidate_rows[name] = metric_rows

    per_candidate_route_rows: Dict[str, Dict[str, np.ndarray]] = {}
    for candidate_idx, (name, route_prediction) in enumerate(
        route_test_predictions.items()
    ):
        route_metrics, route_rows = paired_route_metrics(
            target=dataset.hdv_route[pairs.test[:, 1]],
            persistence=empirical_route_by_split["test"],
            pred=route_prediction,
            od_weight=dataset.hdv_od_weight[pairs.test[:, 1]],
            block_size=int(args.bootstrap_block_size),
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.seed) + 100 + candidate_idx,
        )
        route_metrics["validation_active_od_route_tv"] = float(
            validation_route_metrics[name]["active_od_route_tv"]
        )
        route_metrics["training_active_od_route_tv"] = float(
            training_route_metrics[name]["active_od_route_tv"]
        )
        route_metrics["training_edge_tv"] = float(training_tv[name])
        if name in validation_acceptance:
            route_metrics["validation_acceptance"] = validation_acceptance[name]
        elif name == "selected_od_route_predictor":
            route_metrics["validation_acceptance"] = {
                "accepted": selected_route_source != "persistence",
                "selected_source": selected_route_source,
            }
        candidate_metrics[name].update(route_metrics)
        per_candidate_route_rows[name] = route_rows
    selected_by_validation = validation_selected_route_source
    residual_checkpoint: Optional[Path] = None
    residual_deployment_gate = False
    if residual_model is not None and residual_train is not None:
        residual_deployment_gate = bool(
            residual_validation_gate
            and float(residual_train["selected_residual_scale"]) > 0.0
        )
        residual_model.residual_scale = (
            float(residual_train["selected_residual_scale"])
            if residual_deployment_gate
            else 0.0
        )
        residual_checkpoint = output_dir / "residual_edge_predictor.pt"
        residual_model.save_checkpoint(
            residual_checkpoint,
            metrics={
                "split": {
                    "train_end_episode": int(args.train_end_episode),
                    "validation_end_episode": int(args.validation_end_episode),
                    "test_start_episode": int(args.test_start_episode),
                },
                "training": residual_train,
                "test": candidate_metrics["residual_edge_mlp"],
                "offline_deployment_gate_passed": residual_deployment_gate,
                "validation_gate_passed": residual_validation_gate,
                "trained_residual_scale": float(
                    residual_train["selected_residual_scale"]
                ),
            },
        )
    route_checkpoint = output_dir / "persistence_anchored_od_route_predictor.pt"
    selected_gawron_settings = dict(gawron_settings)
    selected_gawron_settings["alpha_zero"] = float(gawron_calibration.alpha_zero)
    route_checkpoint_format = (
        HDV_OD_ROUTE_RESIDUAL_PREDICTOR_FORMAT
        if str(args.route_response_model) == "residual"
        else HDV_OD_ROUTE_PREDICTOR_FORMAT
    )
    selected_response_scale = (
        float(gate_train["selected_gate_scale"])
        if selected_route_source == learned_route_name
        else 0.0
    )
    torch.save(
        {
            "format": route_checkpoint_format,
            "model_state_dict": gate_model.state_dict(),
            "diagnostic_gawron_residual_state_dict": route_model.state_dict(),
            "n_routes": int(dataset.n_routes),
            "hidden_dim": int(args.route_hidden_dim),
            "depth": int(args.route_depth),
            "progress_denominator": int(progress_denominator),
            "deployment_mode": selected_route_source,
            "selection_split": selection_split,
            "trained_pairs": int(pairs.train.shape[0]),
            "parameterization": str(gate_train["parameterization"]),
            "selected_gate_scale": selected_response_scale,
            "selected_residual_scale": (
                selected_response_scale
                if str(args.route_response_model) == "residual"
                else None
            ),
            "anchor_floor": (
                float(HDV_OD_ROUTE_RESIDUAL_ANCHOR_FLOOR)
                if str(args.route_response_model) == "residual"
                else None
            ),
            "persistence_anchor": "per_od_causal_last_observed_route_histogram",
            "route_target": "seen_hdv_exact_hdv_action",
            "travel_time_feedback": ("seen AV policy_action plus seen HDV hdv_action"),
            "edge_projection": "exact_previous_edge_plus_projected_route_delta",
            "initial_hdv_state": str(initial_state_path),
            "gawron": selected_gawron_settings,
            "gawron_configured": dict(gawron_settings),
            "gawron_alpha_calibration": list(gawron_calibration.candidates),
            "edge_ids": list(dataset.edge_ids),
            "ods": [list(od) for od in dataset.ods],
            "training": {
                "route_metrics": training_route_metrics,
                "edge_tv": {
                    name: training_tv[name] for name in route_train_predictions
                },
                "selected_source": (
                    selected_route_source
                    if selection_split == "train"
                    else ""
                ),
                "persistence_anchored_response": gate_train,
                "persistence_anchored_gate": (
                    gate_train
                    if str(args.route_response_model) == "gate"
                    else None
                ),
                "diagnostic_gawron_residual": route_train,
            },
            "validation": {
                "route_metrics": validation_route_metrics,
                "edge_tv": {
                    name: validation_tv[name] for name in route_validation_predictions
                },
                "acceptance": validation_acceptance,
                "selected_source": validation_selected_route_source,
            },
            "deployment": {
                "selection_split": selection_split,
                "selected_source": selected_route_source,
            },
        },
        route_checkpoint,
    )

    per_episode_path = output_dir / "per_episode_metrics.csv"
    fieldnames = ["episode"]
    for name in predictions:
        fieldnames.extend(
            [
                f"{name}_edge_tv",
                f"{name}_delta_vs_persistence",
                f"{name}_predicted_response_tv",
                f"{name}_response_direction_cosine",
            ]
        )
        if name in per_candidate_route_rows:
            fieldnames.extend(
                [
                    f"{name}_route_tv",
                    f"{name}_route_delta_vs_persistence",
                    f"{name}_route_top1",
                ]
            )
    with per_episode_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, episode_idx in enumerate(pairs.test[:, 1]):
            row: Dict[str, Any] = {"episode": int(dataset.episodes[episode_idx])}
            for name in predictions:
                values = per_candidate_rows[name]
                row[f"{name}_edge_tv"] = float(values["edge_tv"][row_idx])
                row[f"{name}_delta_vs_persistence"] = float(
                    values["delta_vs_persistence"][row_idx]
                )
                row[f"{name}_predicted_response_tv"] = float(
                    values["predicted_response_tv"][row_idx]
                )
                row[f"{name}_response_direction_cosine"] = float(
                    values["response_direction_cosine"][row_idx]
                )
                if name in per_candidate_route_rows:
                    route_values = per_candidate_route_rows[name]
                    row[f"{name}_route_tv"] = float(route_values["route_tv"][row_idx])
                    row[f"{name}_route_delta_vs_persistence"] = float(
                        route_values["route_tv_delta_vs_persistence"][row_idx]
                    )
                    row[f"{name}_route_top1"] = float(
                        route_values["route_top1"][row_idx]
                    )
            writer.writerow(row)

    summary: Dict[str, Any] = {
        "format": "hdv_response_offline_comparison_v2",
        "run_dir": str(run_dir),
        "episodes_dir": str(episodes_dir),
        "route_xml": str(route_xml),
        "edge_registry": str(edge_registry),
        "legacy_checkpoint": str(legacy_checkpoint) if legacy_checkpoint else "",
        "initial_hdv_state": str(initial_state_path),
        "causal_protocol": (
            "target episode e uses only episode e-1 information; real seen HDV "
            "hdv_action values form the OD-route target, seen AV policy_action "
            "and HDV hdv_action travel times drive the exact Gawron proposal, "
            "and a per-OD causal last-observed histogram is the persistence "
            "anchor. Predictor checkpoint selection uses the "
            f"{selection_split} split; validation and test remain separately "
            "reported and are not used for gradient fitting"
        ),
        "target": (
            "seen and successfully route-matched HDV choices from exact "
            "hdv_action, with predicted route response applied to the exact "
            "previous edge context"
        ),
        "target_scope_limitation": (
            "hdv_action is conditioned on successful collector registration "
            "and route matching; it is not the full scheduled-population "
            "initial_route_action target"
        ),
        "target_source": str(target_source),
        "target_alignment": {
            "strict_route_projection_edge_tv_mean": float(target_alignment_tv.mean()),
            "strict_route_projection_edge_tv_max": float(target_alignment_tv.max()),
            "tolerance": float(args.target_alignment_tolerance),
        },
        "primary_metric": (
            "population-weighted OD-route total-variation distance versus "
            "causal last-observed persistence"
        ),
        "secondary_metric": (
            "edge total-variation distance after route-response delta projection"
        ),
        "episodes": {
            "available": int(dataset.episodes.size),
            "min": int(dataset.episodes.min()),
            "max": int(dataset.episodes.max()),
            "train_pairs": int(pairs.train.shape[0]),
            "validation_pairs": int(pairs.validation.shape[0]),
            "test_pairs": int(pairs.test.shape[0]),
            "train_end": int(args.train_end_episode),
            "validation_end": int(args.validation_end_episode),
            "test_start": int(args.test_start_episode),
        },
        "dimensions": {
            "n_edges": int(len(dataset.edge_ids)),
            "n_od": int(len(dataset.ods)),
            "n_routes": int(dataset.n_routes),
        },
        "hyperparameters": {
            "seed": int(args.seed),
            "route_selection_split": selection_split,
            "route_response_model": str(args.route_response_model),
            "train_edge_baselines": bool(args.train_edge_baselines),
            "progress_denominator": int(progress_denominator),
            "gawron_alpha_grid": list(gawron_alpha_candidates),
            "residual_scales": list(scales),
            "epochs": int(args.epochs),
            "route_epochs": int(args.route_epochs),
            "patience": int(args.patience),
            "lr": float(args.lr),
            "route_lr": float(args.route_lr),
            "weight_decay": float(args.weight_decay),
            "residual_penalty": float(args.residual_penalty),
            "gate_penalty": float(args.gate_penalty),
            "validation_min_improvement": float(args.validation_min_improvement),
            "validation_min_relative_improvement": float(
                args.validation_min_relative_improvement
            ),
            "validation_max_edge_degradation": float(
                args.validation_max_edge_degradation
            ),
            "validation_max_relative_edge_degradation": float(
                args.validation_max_relative_edge_degradation
            ),
            "validation_min_route_win_fraction": float(
                args.validation_min_route_win_fraction
            ),
            "bootstrap_block_size": int(args.bootstrap_block_size),
            "bootstrap_samples": int(args.bootstrap_samples),
        },
        "training": {
            "offline_absolute_edge_mlp": absolute_train,
            "residual_edge_mlp": residual_train,
            "gawron_residual_od_route": route_train,
            learned_route_name: gate_train,
        },
        "gawron": {
            "configured": dict(gawron_settings),
            "selected": selected_gawron_settings,
            "alpha_calibration": list(gawron_calibration.candidates),
        },
        "validation_edge_tv": validation_tv,
        "validation_route_metrics": validation_route_metrics,
        "validation_acceptance": validation_acceptance,
        "training_edge_tv": training_tv,
        "training_route_metrics": training_route_metrics,
        "selected_by_validation": selected_by_validation,
        "selected_by_route_selection": selected_route_source,
        "candidates": candidate_metrics,
        "deployment": {
            "candidate": "selected_od_route_predictor",
            "selected_source": selected_route_source,
            "selection_split": selection_split,
            "persistence_gate_validation_gate_passed": (
                persistence_gate_validation_gate
            ),
            "route_residual_validation_gate_passed": route_residual_validation_gate,
            "route_checkpoint_gate_scale": (
                selected_response_scale
            ),
            "route_checkpoint_response_scale": selected_response_scale,
            "policy": (
                "training route TV selects a non-zero learned route response for online "
                "fine-tuning; validation and test are diagnostic only"
                if selection_split == "train"
                else "route TV is primary; a non-persistence candidate is "
                "deployable only when chronological validation clears the "
                "pre-registered route-gain, route-win, and edge-degradation guards"
            ),
        },
        "artifacts": {
            "residual_edge_checkpoint": (
                str(residual_checkpoint) if residual_checkpoint is not None else ""
            ),
            "persistence_anchored_od_route_checkpoint": str(route_checkpoint),
            "per_episode_metrics": str(per_episode_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    if not bool(getattr(args, "quiet", False)):
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))
    else:
        compact = {
            name: {
                "edge_tv_mean": values["edge_tv_mean"],
                "edge_delta_vs_persistence": values[
                    "edge_tv_delta_vs_persistence_mean"
                ],
                "edge_ci95": values["edge_tv_delta_vs_persistence_ci95"],
                "route_tv_mean": values.get("active_od_route_tv"),
                "route_delta_vs_persistence": values.get(
                    "route_tv_delta_vs_persistence_mean"
                ),
                "route_ci95": values.get("route_tv_delta_vs_persistence_ci95"),
            }
            for name, values in candidate_metrics.items()
        }
        print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=True))
    print(f"[saved] {summary_path}")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare causal HDV response predictors on offline episode records."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episodes-dir", default="")
    parser.add_argument("--route-xml", default="")
    parser.add_argument("--edge-registry", required=True)
    parser.add_argument(
        "--initial-hdv-state",
        default="",
        help=(
            "Collector route-policy JSON immediately before the first recorded "
            "episode. Defaults to hdv_state_in from online_wm_dreamer_config.json."
        ),
    )
    parser.add_argument("--legacy-checkpoint", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-end-episode", type=int, required=True)
    parser.add_argument("--validation-end-episode", type=int, required=True)
    parser.add_argument("--test-start-episode", type=int, required=True)
    parser.add_argument("--progress-denominator", type=int, default=0)
    parser.add_argument(
        "--route-selection-split",
        choices=["validation", "train"],
        default="validation",
        help=(
            "Split used for OD-route gate early stopping and scale selection. "
            "Use train only for a deliberately train-fit checkpoint intended "
            "for subsequent causal online fine-tuning."
        ),
    )
    parser.add_argument(
        "--route-response-model",
        choices=["gate", "residual"],
        default="gate",
        help=(
            "OD-route response parameterization. 'gate' is the constrained v3 "
            "blend between persistence and the Gawron proposal; 'residual' is "
            "the v4 unrestricted K-route logit residual around persistence."
        ),
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument(
        "--train-edge-baselines",
        action="store_true",
        help=(
            "Also fit fresh edge-level MLP diagnostics. The repaired predictor "
            "itself is always the OD-route Gawron model."
        ),
    )
    parser.add_argument("--route-epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--route-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--residual-penalty", type=float, default=1e-3)
    parser.add_argument("--gate-penalty", type=float, default=1e-4)
    parser.add_argument("--validation-min-improvement", type=float, default=5e-5)
    parser.add_argument(
        "--validation-min-relative-improvement",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--validation-max-edge-degradation",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--validation-max-relative-edge-degradation",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--validation-min-route-win-fraction",
        type=float,
        default=0.52,
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--route-hidden-dim", type=int, default=64)
    parser.add_argument("--route-depth", type=int, default=2)
    parser.add_argument(
        "--gawron-alpha-grid",
        default="0.01,0.025,0.05,0.1,0.2",
        help=(
            "Positive RouteRL cost-update rates considered on the chronological "
            "validation block; the run's configured value is always included."
        ),
    )
    parser.add_argument(
        "--residual-scales",
        default="0,0.0625,0.125,0.25,0.5,1",
    )
    parser.add_argument("--bootstrap-block-size", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--target-alignment-tolerance", type=float, default=1e-5)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_offline_comparison(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
