#!/usr/bin/env python3
"""Compare static HDV edge-demand predictors on saved online episodes.

The target is the episode-level static HDV demand projected to edge space from
the route chosen by each human vehicle at episode start.  This script keeps the
comparison offline: fit/calibrate on early episodes, evaluate on later episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.train_hdv_response_predictor_ablation import (  # noqa: E402
    _ce_np,
    _load_route_edges,
    _normalise_np,
    _soft_ce,
    _tv_np,
    load_hdv_decision_dataset,
    train_edge_linear,
)
from dreamer4.urban_routing import dreamer_hdv_response  # noqa: E402
from dreamer4.urban_routing.urban_world_model import UrbanWorldModel  # noqa: E402


Od = Tuple[int, int]


@dataclass
class RouteRegistryLite:
    n_od: int
    K: int
    n_edges: int
    od_hdv_counts: np.ndarray
    route_edge_indices: List[List[List[int]]]


class EdgeMLPPredictor(nn.Module):
    def __init__(self, input_dim: int, n_edges: int, hidden_dim: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(int(depth), 1)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(int(hidden_dim)))
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, int(n_edges)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StackelbergEdgePredictor(nn.Module):
    """Batch version of the repo's edge-choice Stackelberg response layer."""

    def __init__(
        self,
        *,
        free_flow_time: torch.Tensor,
        capacities: torch.Tensor,
        total_av_count: float,
        total_hdv_count: float,
        bg_volume: torch.Tensor,
        n_iters: int,
        damping: float,
        alpha_init: float,
        beta_init: float,
        tau_init: float,
    ):
        super().__init__()
        self.register_buffer("free_flow_time", free_flow_time.float())
        self.register_buffer("capacities", capacities.float().clamp_min(1.0))
        self.register_buffer("bg_volume", bg_volume.float())
        self.total_av_count = float(total_av_count)
        self.total_hdv_count = float(total_hdv_count)
        self.n_iters = int(n_iters)
        self.damping = float(damping)
        self._alpha_raw = nn.Parameter(_inv_softplus(float(alpha_init)))
        self._beta_raw = nn.Parameter(_inv_softplus(float(beta_init)))
        self._tau_raw = nn.Parameter(_inv_softplus(float(tau_init)))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self._alpha_raw).clamp_min(1e-6)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self._beta_raw).clamp_min(1e-6)

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self._tau_raw).clamp_min(1e-3)

    def _bpr(self, volume: torch.Tensor) -> torch.Tensor:
        return self.free_flow_time.unsqueeze(0) * (
            1.0
            + self.alpha
            * (volume / self.capacities.unsqueeze(0).clamp_min(1.0)).clamp_min(0.0).pow(self.beta)
        )

    def forward(self, av_edge: torch.Tensor) -> torch.Tensor:
        av = av_edge.float().clamp_min(0.0)
        av = av / av.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        av_volume = av * float(self.total_av_count)
        bg = self.bg_volume.to(device=av.device, dtype=av.dtype).unsqueeze(0)
        h0 = torch.softmax(-self.free_flow_time.to(av.device, av.dtype) / self.tau, dim=-1)
        h = h0.unsqueeze(0).expand_as(av).clone()
        for _ in range(max(int(self.n_iters), 1)):
            volume = bg + av_volume + h * float(self.total_hdv_count)
            travel_time = self._bpr(volume)
            h_new = torch.softmax(-travel_time / self.tau, dim=-1)
            h = (1.0 - float(self.damping)) * h + float(self.damping) * h_new
        return h / h.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class OdRouteStackelbergPredictor(nn.Module):
    """Stackelberg response constrained to each OD's candidate routes."""

    def __init__(
        self,
        *,
        free_flow_time: torch.Tensor,
        capacities: torch.Tensor,
        route_edge_indices: Sequence[Sequence[Sequence[int]]],
        total_av_count: float,
        bg_volume: torch.Tensor,
        n_iters: int,
        damping: float,
        alpha_init: float,
        beta_init: float,
        tau_init: float,
    ):
        super().__init__()
        self.register_buffer("free_flow_time", free_flow_time.float())
        self.register_buffer("capacities", capacities.float().clamp_min(1.0))
        self.register_buffer("bg_volume", bg_volume.float())
        self.total_av_count = float(total_av_count)
        self.n_iters = int(n_iters)
        self.damping = float(damping)
        self._alpha_raw = nn.Parameter(_inv_softplus(float(alpha_init)))
        self._beta_raw = nn.Parameter(_inv_softplus(float(beta_init)))
        self._tau_raw = nn.Parameter(_inv_softplus(float(tau_init)))

        n_od = len(route_edge_indices)
        n_routes = max((len(routes) for routes in route_edge_indices), default=0)
        n_edges = int(free_flow_time.numel())
        incidence = torch.zeros((n_od, n_routes, n_edges), dtype=torch.float32)
        mask = torch.zeros((n_od, n_routes), dtype=torch.bool)
        for od_i, od_routes in enumerate(route_edge_indices):
            for route_i, edges in enumerate(od_routes):
                valid_edges = [int(edge) for edge in edges if 0 <= int(edge) < n_edges]
                if not valid_edges:
                    continue
                incidence[od_i, route_i, torch.as_tensor(valid_edges, dtype=torch.long)] = 1.0
                mask[od_i, route_i] = True
        self.register_buffer("route_incidence", incidence)
        self.register_buffer("route_mask", mask)

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self._alpha_raw).clamp_min(1e-6)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self._beta_raw).clamp_min(1e-6)

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self._tau_raw).clamp_min(1e-3)

    def _bpr(self, volume: torch.Tensor) -> torch.Tensor:
        return self.free_flow_time.unsqueeze(0) * (
            1.0
            + self.alpha
            * (volume / self.capacities.unsqueeze(0).clamp_min(1.0)).clamp_min(0.0).pow(self.beta)
        )

    def _masked_route_softmax(self, costs: torch.Tensor) -> torch.Tensor:
        mask = self.route_mask.to(device=costs.device).unsqueeze(0)
        logits = -costs / self.tau
        logits = logits.masked_fill(~mask, -1e9)
        probs = torch.softmax(logits, dim=-1)
        uniform = mask.float() / mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.where(mask.any(dim=-1, keepdim=True), probs, uniform)

    def _route_to_edge_volume(self, route_probs: torch.Tensor, hdv_counts: torch.Tensor) -> torch.Tensor:
        route_volume = route_probs * hdv_counts.float().clamp_min(0.0).unsqueeze(-1)
        return torch.einsum("bok,oke->be", route_volume, self.route_incidence.to(route_probs.device))

    def forward(self, av_edge: torch.Tensor, hdv_counts: torch.Tensor) -> torch.Tensor:
        av = av_edge.float().clamp_min(0.0)
        av = av / av.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        av_volume = av * float(self.total_av_count)
        bg = self.bg_volume.to(device=av.device, dtype=av.dtype).unsqueeze(0)

        free_flow_route_cost = torch.einsum(
            "e,oke->ok",
            self.free_flow_time.to(device=av.device, dtype=av.dtype),
            self.route_incidence.to(device=av.device, dtype=av.dtype),
        )
        route_probs = self._masked_route_softmax(
            free_flow_route_cost.unsqueeze(0).expand(av.shape[0], -1, -1)
        )
        for _ in range(max(int(self.n_iters), 1)):
            hdv_edge_volume = self._route_to_edge_volume(route_probs, hdv_counts.to(device=av.device, dtype=av.dtype))
            volume = bg + av_volume + hdv_edge_volume
            edge_tt = self._bpr(volume)
            route_costs = torch.einsum(
                "be,oke->bok",
                edge_tt,
                self.route_incidence.to(device=av.device, dtype=av.dtype),
            )
            route_new = self._masked_route_softmax(route_costs)
            route_probs = (1.0 - float(self.damping)) * route_probs + float(self.damping) * route_new
        edge_volume = self._route_to_edge_volume(route_probs, hdv_counts.to(device=av.device, dtype=av.dtype))
        return edge_volume / edge_volume.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _inv_softplus(value: float) -> torch.Tensor:
    value = max(float(value), 1e-6)
    if value > 20.0:
        return torch.tensor(value, dtype=torch.float32)
    return torch.tensor(math.log(math.expm1(value)), dtype=torch.float32)


def _parse_grid(raw: str) -> List[float]:
    return [float(part) for part in str(raw).split(",") if str(part).strip()]


def _edge_metrics(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ce = np.asarray([_ce_np(t, p) for t, p in zip(target, pred)], dtype=np.float64)
    tv = np.asarray([_tv_np(t, p) for t, p in zip(target, pred)], dtype=np.float64)
    entropy = np.asarray([_ce_np(t, t) for t in target], dtype=np.float64)
    return {
        "edge_ce": float(ce.mean()),
        "edge_kl": float((ce - entropy).mean()),
        "edge_tv": float(tv.mean()),
    }


def _episode_num(path: Path) -> int:
    stem = path.stem
    return int(stem[2:]) if stem.startswith("ep") and stem[2:].isdigit() else -1


def _load_episode_od_counts(run_dir: Path, episodes: Sequence[int], ods: Sequence[Od]) -> Tuple[np.ndarray, np.ndarray]:
    od_to_idx = {tuple(od): i for i, od in enumerate(ods)}
    hdv_counts = np.zeros((len(episodes), len(ods)), dtype=np.float64)
    av_counts = np.zeros((len(episodes), len(ods)), dtype=np.float64)
    ep_dir = run_dir / "collector_runtime" / "episodes"
    for row_i, episode in enumerate(episodes):
        path = ep_dir / f"ep{int(episode)}.csv"
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                role = str(row.get("vType", "")).strip().lower()
                try:
                    od = (int(float(row["origin"])), int(float(row["destination"])))
                except (KeyError, TypeError, ValueError):
                    continue
                od_i = od_to_idx.get(od)
                if od_i is None:
                    continue
                if role == "human":
                    hdv_counts[row_i, od_i] += 1.0
                elif role == "av":
                    av_counts[row_i, od_i] += 1.0
    return hdv_counts, av_counts


def _make_response_features(dataset, episodes: np.ndarray, *, mode: str) -> Tuple[torch.Tensor, np.ndarray]:
    mode_l = str(mode or "current_oracle").strip().lower()
    av_route = np.asarray(dataset.av_route, dtype=np.float32)
    hdv_route = np.asarray(dataset.hdv_route, dtype=np.float32)
    av_edge = np.asarray(dataset.av_edge, dtype=np.float32)
    hdv_edge = np.asarray(dataset.hdv_edge, dtype=np.float32)

    prev_av_route = np.zeros_like(av_route)
    prev_hdv_route = np.zeros_like(hdv_route)
    prev_av_edge = np.zeros_like(av_edge)
    prev_hdv_edge = np.zeros_like(hdv_edge)
    prev_av_route[0] = av_route[0]
    prev_hdv_route[0] = hdv_route[0]
    prev_av_edge[0] = av_edge[0]
    prev_hdv_edge[0] = hdv_edge[0]
    prev_av_route[1:] = av_route[:-1]
    prev_hdv_route[1:] = hdv_route[:-1]
    prev_av_edge[1:] = av_edge[:-1]
    prev_hdv_edge[1:] = hdv_edge[:-1]

    if mode_l in {"current_oracle", "oracle", "current"}:
        feature_parts = [
            av_route.reshape(len(episodes), -1),
            av_edge,
            prev_hdv_route.reshape(len(episodes), -1),
            prev_hdv_edge,
        ]
        condition_av_edge = av_edge
    elif mode_l in {"prev_episode", "previous_episode", "version_a", "causal"}:
        feature_parts = [
            prev_av_route.reshape(len(episodes), -1),
            prev_av_edge,
            prev_hdv_route.reshape(len(episodes), -1),
            prev_hdv_edge,
        ]
        condition_av_edge = prev_av_edge
    else:
        raise ValueError(
            f"Unsupported feature mode {mode!r}; expected current_oracle or prev_episode"
        )

    ep = np.asarray(episodes, dtype=np.float32)
    denom = max(float(ep.max() - ep.min()), 1.0)
    ep_norm = ((ep - float(ep.min())) / denom).reshape(-1, 1).astype(np.float32)
    feature_parts.append(ep_norm)
    return torch.from_numpy(np.concatenate(feature_parts, axis=1)).float(), condition_av_edge.astype(np.float32)


def _build_route_registry(
    *,
    ods: Sequence[Od],
    n_routes: int,
    n_edges: int,
    route_xml: Path,
    edge_ids: Sequence[str],
    od_hdv_counts: np.ndarray,
) -> RouteRegistryLite:
    _, _, route_edges = _load_route_edges(str(route_xml), edge_ids)
    route_edge_indices: List[List[List[int]]] = []
    for origin, destination in ods:
        rows: List[List[int]] = []
        for route in range(int(n_routes)):
            rows.append(list(route_edges.get((int(origin), int(destination), int(route)), [])))
        route_edge_indices.append(rows)
    return RouteRegistryLite(
        n_od=len(ods),
        K=int(n_routes),
        n_edges=int(n_edges),
        od_hdv_counts=np.asarray(od_hdv_counts, dtype=np.float32).reshape(-1),
        route_edge_indices=route_edge_indices,
    )


def _project_route_probs_to_edge(
    route_probs: np.ndarray,
    hdv_counts: np.ndarray,
    route_edge_indices: Sequence[Sequence[Sequence[int]]],
    n_edges: int,
) -> np.ndarray:
    probs = np.asarray(route_probs, dtype=np.float64)
    if probs.ndim == 2:
        probs = probs[None, ...]
    counts = np.asarray(hdv_counts, dtype=np.float64)
    if counts.ndim == 1:
        counts = np.broadcast_to(counts[None, :], (probs.shape[0], counts.shape[0]))
    out = np.zeros((probs.shape[0], int(n_edges)), dtype=np.float64)
    for row_i in range(probs.shape[0]):
        for od_i, od_routes in enumerate(route_edge_indices):
            count = float(max(counts[row_i, od_i], 0.0))
            if count <= 0.0:
                continue
            row = np.clip(probs[row_i, od_i], 0.0, None)
            total = float(row.sum())
            row = row / total if total > 1e-12 else np.full_like(row, 1.0 / max(int(row.size), 1))
            for route_i, prob in enumerate(row[: len(od_routes)]):
                if float(prob) <= 0.0:
                    continue
                for edge_i in od_routes[route_i]:
                    if 0 <= int(edge_i) < int(n_edges):
                        out[row_i, int(edge_i)] += count * float(prob)
    denom = out.sum(axis=1, keepdims=True)
    uniform = np.full_like(out, 1.0 / float(max(int(n_edges), 1)))
    return np.where(denom > 1e-12, out / np.maximum(denom, 1e-12), uniform)


def _load_edge_tt(edge_registry: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with edge_registry.open("r", encoding="utf-8") as f:
        data = json.load(f)
    lengths = np.asarray(data.get("lengths", []), dtype=np.float64)
    speeds = np.asarray(data.get("speed_limits", []), dtype=np.float64)
    capacities = np.asarray(data.get("capacities", []), dtype=np.float64)
    if lengths.size == 0 or speeds.size == 0 or capacities.size == 0:
        raise RuntimeError(f"edge registry missing lengths/speed_limits/capacities: {edge_registry}")
    free_flow_time = lengths / np.maximum(speeds, 0.1)
    return free_flow_time.astype(np.float32), capacities.astype(np.float32), lengths.astype(np.float32)


def _train_edge_mlp(
    *,
    x: torch.Tensor,
    target_edge: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    depth: int,
    seed: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    torch.manual_seed(int(seed))
    model = EdgeMLPPredictor(int(x.shape[1]), int(target_edge.shape[1]), int(hidden_dim), int(depth))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = None
    best_loss = float("inf")
    for _ in range(int(epochs)):
        logits = model(x[train_idx])
        loss = _soft_ce(logits, target_edge[train_idx], dim=-1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        pred = torch.softmax(model(x[test_idx]), dim=-1).cpu().numpy()
    return {"train_ce": float(best_loss)}, pred


def _wardrop_predict(
    *,
    dataset,
    condition_av_edge: np.ndarray,
    row_idx: np.ndarray,
    hdv_counts: np.ndarray,
    route_registry: RouteRegistryLite,
    base_edge_tt: np.ndarray,
    pressure_weight: float,
    temperature: float,
    prior_route_fracs: Optional[np.ndarray],
    prior_mix: float,
) -> np.ndarray:
    route_preds: List[np.ndarray] = []
    for idx in row_idx:
        route_pred = dreamer_hdv_response.build_hdv_route_fracs_from_episode_av_pressure(
            route_registry,
            condition_av_edge[int(idx)],
            base_edge_tt=base_edge_tt,
            n_routes=int(dataset.n_routes),
            pressure_weight=float(pressure_weight),
            temperature=float(temperature),
            prior_route_fracs=prior_route_fracs,
            prior_mix=float(prior_mix),
        )
        route_preds.append(route_pred)
    return _project_route_probs_to_edge(
        np.stack(route_preds, axis=0),
        hdv_counts[row_idx],
        route_registry.route_edge_indices,
        int(dataset.n_edges),
    )


def _calibrate_wardrop(
    *,
    dataset,
    condition_av_edge: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    hdv_counts: np.ndarray,
    route_registry: RouteRegistryLite,
    base_edge_tt: np.ndarray,
    pressure_grid: Sequence[float],
    temperature_grid: Sequence[float],
    prior_mix_grid: Sequence[float],
) -> Tuple[Dict[str, object], np.ndarray]:
    prior_route = np.asarray(dataset.hdv_route[train_idx], dtype=np.float64).mean(axis=0)
    for od_i in range(prior_route.shape[0]):
        prior_route[od_i] = _normalise_np(prior_route[od_i])
    best: Optional[Dict[str, object]] = None
    for pressure_weight in pressure_grid:
        for temperature in temperature_grid:
            for prior_mix in prior_mix_grid:
                train_pred = _wardrop_predict(
                    dataset=dataset,
                    condition_av_edge=condition_av_edge,
                    row_idx=train_idx,
                    hdv_counts=hdv_counts,
                    route_registry=route_registry,
                    base_edge_tt=base_edge_tt,
                    pressure_weight=float(pressure_weight),
                    temperature=float(temperature),
                    prior_route_fracs=prior_route,
                    prior_mix=float(prior_mix),
                )
                train_metrics = _edge_metrics(dataset.hdv_edge[train_idx], train_pred)
                candidate = {
                    "pressure_weight": float(pressure_weight),
                    "temperature": float(temperature),
                    "prior_mix": float(prior_mix),
                    "train": train_metrics,
                }
                if best is None or float(train_metrics["edge_tv"]) < float(best["train"]["edge_tv"]):
                    best = candidate
    assert best is not None
    test_pred = _wardrop_predict(
        dataset=dataset,
        condition_av_edge=condition_av_edge,
        row_idx=test_idx,
        hdv_counts=hdv_counts,
        route_registry=route_registry,
        base_edge_tt=base_edge_tt,
        pressure_weight=float(best["pressure_weight"]),
        temperature=float(best["temperature"]),
        prior_route_fracs=prior_route,
        prior_mix=float(best["prior_mix"]),
    )
    return best, test_pred


def _train_stackelberg(
    *,
    av_edge: torch.Tensor,
    target_edge: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    free_flow_time: np.ndarray,
    capacities: np.ndarray,
    train_mean_hdv: np.ndarray,
    total_av_count: float,
    total_hdv_count: float,
    bg_mode: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    n_iters: int,
    damping: float,
    seed: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    torch.manual_seed(int(seed))
    device = av_edge.device
    if str(bg_mode).strip().lower() == "train_mean_hdv":
        bg_volume = torch.as_tensor(train_mean_hdv, dtype=torch.float32, device=device) * float(total_hdv_count)
    else:
        bg_volume = torch.zeros((int(av_edge.shape[1]),), dtype=torch.float32, device=device)
    model = StackelbergEdgePredictor(
        free_flow_time=torch.as_tensor(free_flow_time, dtype=torch.float32, device=device),
        capacities=torch.as_tensor(capacities, dtype=torch.float32, device=device),
        total_av_count=float(total_av_count),
        total_hdv_count=float(total_hdv_count),
        bg_volume=bg_volume,
        n_iters=int(n_iters),
        damping=float(damping),
        alpha_init=0.15,
        beta_init=4.0,
        tau_init=30.0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = None
    best_loss = float("inf")
    for _ in range(int(epochs)):
        pred = model(av_edge[train_idx])
        loss = -(target_edge[train_idx] * pred.clamp_min(1e-8).log()).sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        pred = model(av_edge[test_idx]).cpu().numpy()
    stats = {
        "train_ce": float(best_loss),
        "alpha": float(model.alpha.detach().cpu()),
        "beta": float(model.beta.detach().cpu()),
        "tau": float(model.tau.detach().cpu()),
    }
    return stats, pred


def _train_od_route_stackelberg(
    *,
    av_edge: torch.Tensor,
    hdv_counts: torch.Tensor,
    target_edge: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    free_flow_time: np.ndarray,
    capacities: np.ndarray,
    route_edge_indices: Sequence[Sequence[Sequence[int]]],
    train_mean_hdv: np.ndarray,
    total_av_count: float,
    total_hdv_count: float,
    bg_mode: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    n_iters: int,
    damping: float,
    seed: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    torch.manual_seed(int(seed))
    device = av_edge.device
    if str(bg_mode).strip().lower() == "train_mean_hdv":
        bg_volume = torch.as_tensor(train_mean_hdv, dtype=torch.float32, device=device) * float(total_hdv_count)
    else:
        bg_volume = torch.zeros((int(av_edge.shape[1]),), dtype=torch.float32, device=device)
    model = OdRouteStackelbergPredictor(
        free_flow_time=torch.as_tensor(free_flow_time, dtype=torch.float32, device=device),
        capacities=torch.as_tensor(capacities, dtype=torch.float32, device=device),
        route_edge_indices=route_edge_indices,
        total_av_count=float(total_av_count),
        bg_volume=bg_volume,
        n_iters=int(n_iters),
        damping=float(damping),
        alpha_init=0.15,
        beta_init=4.0,
        tau_init=30.0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = None
    best_loss = float("inf")
    for _ in range(int(epochs)):
        pred = model(av_edge[train_idx], hdv_counts[train_idx])
        loss = -(target_edge[train_idx] * pred.clamp_min(1e-8).log()).sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        pred = model(av_edge[test_idx], hdv_counts[test_idx]).cpu().numpy()
    stats = {
        "train_ce": float(best_loss),
        "alpha": float(model.alpha.detach().cpu()),
        "beta": float(model.beta.detach().cpu()),
        "tau": float(model.tau.detach().cpu()),
    }
    return stats, pred


def _checkpoint_wm_paths(checkpoint: Path) -> Tuple[str, str, Dict[str, object]]:
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = state.get("cfg", {})
    if not isinstance(cfg, dict):
        cfg = vars(cfg) if hasattr(cfg, "__dict__") else {}
    projector = state.get("wm_projector_ckpt") or cfg.get("wm_projector_ckpt") or cfg.get("projector_ckpt")
    dynamics = state.get("wm_dynamics_ckpt") or cfg.get("wm_dynamics_ckpt") or cfg.get("dynamics_ckpt")
    if not projector or not dynamics:
        raise RuntimeError(f"checkpoint has no wm_projector_ckpt/wm_dynamics_ckpt: {checkpoint}")
    return str(projector), str(dynamics), dict(cfg)


def _evaluate_wm_hdv_head(
    *,
    checkpoint: Path,
    edge_registry: Path,
    dataset,
    condition_av_edge: np.ndarray,
    test_idx: np.ndarray,
    device: str,
) -> Dict[str, object]:
    projector_ckpt, dynamics_ckpt, cfg = _checkpoint_wm_paths(checkpoint)
    wm = UrbanWorldModel(
        flow_projector_ckpt=str(projector_ckpt),
        dynamics_ckpt=str(dynamics_ckpt),
        edge_registry_path=str(edge_registry),
        device=str(device),
        use_amp=False,
        cfm_init_mode=str(cfg.get("wm_cfm_init_mode", "current") or "current"),
        policy_obs_source=str(cfg.get("wm_policy_obs_source", "obs_proj") or "obs_proj"),
    )
    preds: List[np.ndarray] = []
    for idx in test_idx:
        pred = wm.predict_episode_hdv_response(condition_av_edge[int(idx)], normalize=True)
        if pred is None:
            raise RuntimeError(f"WM checkpoint has no usable hdv_policy_head: {checkpoint}")
        preds.append(pred.detach().cpu().numpy().astype(np.float64))
    pred_np = np.stack(preds, axis=0)
    return {
        "params": {
            "checkpoint": str(checkpoint),
            "projector_ckpt": str(projector_ckpt),
            "dynamics_ckpt": str(dynamics_ckpt),
            "device": str(device),
        },
        "test": _edge_metrics(dataset.hdv_edge[test_idx], pred_np),
    }


def _select_split(episodes: np.ndarray, train_end_episode: int, test_start_episode: int) -> Tuple[np.ndarray, np.ndarray]:
    train_idx = np.flatnonzero(episodes <= int(train_end_episode)).astype(np.int64)
    test_idx = np.flatnonzero(episodes >= int(test_start_episode)).astype(np.int64)
    if train_idx.size == 0 or test_idx.size == 0:
        raise RuntimeError(
            f"empty split: train_end_episode={train_end_episode}, test_start_episode={test_start_episode}, "
            f"train={train_idx.size}, test={test_idx.size}"
        )
    return train_idx, test_idx


def _rank_predictors(predictors: Dict[str, Dict[str, object]], metric: str) -> List[Dict[str, object]]:
    rows = []
    for name, info in predictors.items():
        test = info.get("test")
        if not isinstance(test, dict) or metric not in test:
            continue
        rows.append({"name": name, metric: float(test[metric])})
    return sorted(rows, key=lambda item: float(item[metric]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--route-xml", required=True)
    parser.add_argument("--edge-registry", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--train-end-episode", type=int, default=300)
    parser.add_argument("--test-start-episode", type=int, default=301)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--stackelberg-epochs", type=int, default=1500)
    parser.add_argument("--stackelberg-lr", type=float, default=5e-2)
    parser.add_argument("--stackelberg-iters", type=int, default=10)
    parser.add_argument("--stackelberg-damping", type=float, default=0.5)
    parser.add_argument("--stackelberg-bg-mode", choices=["zero", "train_mean_hdv"], default="zero")
    parser.add_argument("--feature-mode", choices=["current_oracle", "prev_episode"], default="current_oracle")
    parser.add_argument("--wm-head-checkpoint", default="")
    parser.add_argument("--wm-head-device", default="cpu")
    parser.add_argument("--pressure-grid", default="0,5,10,20,40,60,100,200")
    parser.add_argument("--temperature-grid", default="1,5,10,20,40,60,100,200")
    parser.add_argument("--prior-mix-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    torch.set_num_threads(1)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    run_dir = Path(args.run_dir).resolve()
    route_xml = Path(args.route_xml).resolve()
    edge_registry = Path(args.edge_registry).resolve()
    dataset = load_hdv_decision_dataset(
        run_dir=str(run_dir),
        route_xml=str(route_xml),
        edge_registry=str(edge_registry),
    )
    train_idx_np, test_idx_np = _select_split(
        dataset.episodes,
        train_end_episode=int(args.train_end_episode),
        test_start_episode=int(args.test_start_episode),
    )
    hdv_counts, av_counts = _load_episode_od_counts(run_dir, dataset.episodes, dataset.ods)
    free_flow_time, capacities, _lengths = _load_edge_tt(edge_registry)
    train_mean_hdv_counts = hdv_counts[train_idx_np].mean(axis=0)
    route_registry = _build_route_registry(
        ods=dataset.ods,
        n_routes=int(dataset.n_routes),
        n_edges=int(dataset.n_edges),
        route_xml=route_xml,
        edge_ids=dataset.edge_ids,
        od_hdv_counts=train_mean_hdv_counts,
    )

    x, condition_av_edge = _make_response_features(
        dataset,
        dataset.episodes,
        mode=str(args.feature_mode),
    )
    target_edge = torch.from_numpy(dataset.hdv_edge).float()
    hdv_counts_t = torch.from_numpy(hdv_counts).float()
    train_idx = torch.from_numpy(train_idx_np).long()
    test_idx = torch.from_numpy(test_idx_np).long()
    predictors: Dict[str, Dict[str, object]] = {}

    prior_edge = _normalise_np(dataset.hdv_edge[train_idx_np].sum(axis=0))
    prior_pred = np.stack([prior_edge for _ in test_idx_np], axis=0)
    predictors["global_mean_prior"] = {
        "test": _edge_metrics(dataset.hdv_edge[test_idx_np], prior_pred),
    }

    linear_train, linear_pred = train_edge_linear(
        x=x,
        target_edge=target_edge,
        train_idx=train_idx,
        test_idx=test_idx,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    predictors["edge_linear"] = {
        "train": linear_train,
        "test": _edge_metrics(dataset.hdv_edge[test_idx_np], linear_pred),
    }

    mlp_train, mlp_pred = _train_edge_mlp(
        x=x,
        target_edge=target_edge,
        train_idx=train_idx,
        test_idx=test_idx,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        seed=int(args.seed),
    )
    predictors["edge_mlp"] = {
        "train": mlp_train,
        "test": _edge_metrics(dataset.hdv_edge[test_idx_np], mlp_pred),
    }

    wardrop_best, wardrop_pred = _calibrate_wardrop(
        dataset=dataset,
        condition_av_edge=condition_av_edge,
        train_idx=train_idx_np,
        test_idx=test_idx_np,
        hdv_counts=hdv_counts,
        route_registry=route_registry,
        base_edge_tt=free_flow_time,
        pressure_grid=_parse_grid(args.pressure_grid),
        temperature_grid=_parse_grid(args.temperature_grid),
        prior_mix_grid=_parse_grid(args.prior_mix_grid),
    )
    predictors["wardrop_analytic"] = {
        "params": {
            "pressure_weight": wardrop_best["pressure_weight"],
            "temperature": wardrop_best["temperature"],
            "prior_mix": wardrop_best["prior_mix"],
        },
        "train": wardrop_best["train"],
        "test": _edge_metrics(dataset.hdv_edge[test_idx_np], wardrop_pred),
    }

    av_total = float(av_counts[train_idx_np].sum(axis=1).mean())
    hdv_total = float(hdv_counts[train_idx_np].sum(axis=1).mean())
    stack_train, stack_pred = _train_stackelberg(
        av_edge=torch.from_numpy(condition_av_edge).float(),
        target_edge=target_edge,
        train_idx=train_idx,
        test_idx=test_idx,
        free_flow_time=free_flow_time,
        capacities=capacities,
        train_mean_hdv=prior_edge,
        total_av_count=av_total,
        total_hdv_count=hdv_total,
        bg_mode=str(args.stackelberg_bg_mode),
        epochs=int(args.stackelberg_epochs),
        lr=float(args.stackelberg_lr),
        weight_decay=float(args.weight_decay),
        n_iters=int(args.stackelberg_iters),
        damping=float(args.stackelberg_damping),
        seed=int(args.seed),
    )
    predictors["stackelberg_equilibrium"] = {
        "params": {
            "alpha": stack_train.pop("alpha"),
            "beta": stack_train.pop("beta"),
            "tau": stack_train.pop("tau"),
            "bg_mode": str(args.stackelberg_bg_mode),
            "n_iters": int(args.stackelberg_iters),
            "damping": float(args.stackelberg_damping),
        },
        "train": stack_train,
        "test": _edge_metrics(dataset.hdv_edge[test_idx_np], stack_pred),
    }
    od_stack_train, od_stack_pred = _train_od_route_stackelberg(
        av_edge=torch.from_numpy(condition_av_edge).float(),
        hdv_counts=hdv_counts_t,
        target_edge=target_edge,
        train_idx=train_idx,
        test_idx=test_idx,
        free_flow_time=free_flow_time,
        capacities=capacities,
        route_edge_indices=route_registry.route_edge_indices,
        train_mean_hdv=prior_edge,
        total_av_count=av_total,
        total_hdv_count=hdv_total,
        bg_mode=str(args.stackelberg_bg_mode),
        epochs=int(args.stackelberg_epochs),
        lr=float(args.stackelberg_lr),
        weight_decay=float(args.weight_decay),
        n_iters=int(args.stackelberg_iters),
        damping=float(args.stackelberg_damping),
        seed=int(args.seed),
    )
    predictors["od_route_stackelberg_equilibrium"] = {
        "params": {
            "alpha": od_stack_train.pop("alpha"),
            "beta": od_stack_train.pop("beta"),
            "tau": od_stack_train.pop("tau"),
            "bg_mode": str(args.stackelberg_bg_mode),
            "n_iters": int(args.stackelberg_iters),
            "damping": float(args.stackelberg_damping),
        },
        "train": od_stack_train,
        "test": _edge_metrics(dataset.hdv_edge[test_idx_np], od_stack_pred),
    }
    if str(args.wm_head_checkpoint).strip():
        predictors["wm_hdv_policy_head"] = _evaluate_wm_hdv_head(
            checkpoint=Path(args.wm_head_checkpoint).resolve(),
            edge_registry=edge_registry,
            dataset=dataset,
            condition_av_edge=condition_av_edge,
            test_idx=test_idx_np,
            device=str(args.wm_head_device),
        )

    summary: Dict[str, object] = {
        "run_dir": str(run_dir),
        "route_xml": str(route_xml),
        "edge_registry": str(edge_registry),
        "episodes": int(len(dataset.episodes)),
        "train_end_episode": int(args.train_end_episode),
        "test_start_episode": int(args.test_start_episode),
        "train_episodes": int(train_idx_np.size),
        "test_episodes": int(test_idx_np.size),
        "test_episode_min": int(dataset.episodes[test_idx_np].min()),
        "test_episode_max": int(dataset.episodes[test_idx_np].max()),
        "n_od": int(dataset.n_od),
        "n_routes": int(dataset.n_routes),
        "n_edges": int(dataset.n_edges),
        "feature_dim": int(x.shape[1]),
        "feature_mode": str(args.feature_mode),
        "mean_train_av_vehicles": av_total,
        "mean_train_hdv_vehicles": hdv_total,
        "predictors": predictors,
        "ranking_by_edge_tv": _rank_predictors(predictors, "edge_tv"),
        "ranking_by_edge_kl": _rank_predictors(predictors, "edge_kl"),
    }
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[saved] {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
