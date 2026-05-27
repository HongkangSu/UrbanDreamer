"""HDV background and response helpers for Dreamer vehicle imagination.

The functions here keep episode-level HDV response logic out of
``train_marl.py``.  HDV response is treated as a static episode background
conditioned on the whole episode's AV pressure, not as a per-step controllable
policy.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)


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

    n_dest = max(1, len(route_registry.destinations))
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
