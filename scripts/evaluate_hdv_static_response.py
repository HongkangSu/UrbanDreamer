#!/usr/bin/env python3
"""Diagnose episode-level static HDV route/demand response predictability.

This is deliberately separate from HDV current-occupancy diagnostics.  The
target here is the static HDV decision made at episode start, represented as an
OD-local route distribution and its projected full-path edge demand.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Od = Tuple[int, int]
RouteKey = Tuple[int, int, int]


def _episode_num(path: str) -> int:
    match = re.search(r"ep(\d+)\.csv$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def _normalise_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1).clip(min=0.0)
    total = float(arr.sum())
    if total <= eps:
        return np.zeros((int(arr.size),), dtype=np.float64)
    return arr / total


def _normalise_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).clip(min=0.0)
    denom = arr.sum(axis=-1, keepdims=True)
    return np.divide(arr, np.maximum(denom, eps), out=np.zeros_like(arr, dtype=np.float64))


def _normalise_od_routes(route_counts: np.ndarray) -> np.ndarray:
    out = np.asarray(route_counts, dtype=np.float64).copy()
    for i in range(int(out.shape[0])):
        row = _normalise_np(out[i])
        if float(row.sum()) <= 0.0:
            row = np.full((int(out.shape[1]),), 1.0 / max(int(out.shape[1]), 1), dtype=np.float64)
        out[i] = row
    return out


def _safe_route(row: Dict[str, str], preferred: str, fallback: str = "initial_route_action") -> int:
    for key in (preferred, fallback, "policy_action", "action", "hdv_action"):
        value = row.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            route = int(float(value))
        except ValueError:
            continue
        if route >= 0:
            return route
    return -1


def _load_edge_ids(edge_registry: Path) -> List[str]:
    with edge_registry.open("r", encoding="utf-8") as f:
        data = json.load(f)
    edge_ids = data.get("edge_ids")
    if not isinstance(edge_ids, list) or not edge_ids:
        raise RuntimeError(f"edge registry has no edge_ids: {edge_registry}")
    return [str(edge) for edge in edge_ids]


def _load_route_edges(route_xml: Path, edge_ids: Sequence[str]) -> Tuple[List[Od], int, Dict[RouteKey, List[int]]]:
    edge_to_idx = {edge: i for i, edge in enumerate(edge_ids)}
    route_edges: Dict[RouteKey, List[int]] = {}
    ods = set()
    max_route = -1
    tree = ET.parse(route_xml)
    for elem in tree.getroot().iter("route"):
        route_id = str(elem.attrib.get("id", ""))
        parts = route_id.split("_")
        if len(parts) != 3:
            continue
        try:
            origin, destination, route = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        edge_indices = [
            edge_to_idx[edge]
            for edge in str(elem.attrib.get("edges", "")).split()
            if edge in edge_to_idx
        ]
        if not edge_indices:
            continue
        ods.add((origin, destination))
        max_route = max(max_route, route)
        route_edges[(origin, destination, route)] = edge_indices
    if not ods or max_route < 0:
        raise RuntimeError(f"no usable routes in {route_xml}")
    return sorted(ods), max_route + 1, route_edges


@dataclass
class StaticHdvDataset:
    episodes: np.ndarray
    av_route: np.ndarray
    av_edge: np.ndarray
    hdv_route: np.ndarray
    hdv_edge: np.ndarray
    hdv_od_weight: np.ndarray
    ods: List[Od]
    n_routes: int
    edge_ids: List[str]
    route_to_edge: np.ndarray


def _route_projection_matrix(
    *,
    ods: Sequence[Od],
    n_routes: int,
    edge_ids: Sequence[str],
    route_edges: Dict[RouteKey, List[int]],
) -> np.ndarray:
    mat = np.zeros((len(ods), int(n_routes), len(edge_ids)), dtype=np.float64)
    for od_idx, (origin, destination) in enumerate(ods):
        for route in range(int(n_routes)):
            for edge_idx in route_edges.get((origin, destination, route), []):
                mat[int(od_idx), int(route), int(edge_idx)] += 1.0
    return mat


def _project_route_to_edge(route_probs: np.ndarray, od_weights: np.ndarray, route_to_edge: np.ndarray) -> np.ndarray:
    route_probs = np.asarray(route_probs, dtype=np.float64)
    od_weights = np.asarray(od_weights, dtype=np.float64)
    if route_probs.ndim == 2:
        edge = (route_probs[:, :, None] * od_weights[:, None, None] * route_to_edge).sum(axis=(0, 1))
        return _normalise_np(edge)
    edge = (route_probs[:, :, :, None] * od_weights[:, :, None, None] * route_to_edge[None, :, :, :]).sum(axis=(1, 2))
    return _normalise_rows(edge)


def load_static_hdv_dataset(
    *,
    run_dir: Path,
    route_xml: Path,
    edge_registry: Path,
    episodes_dir: Optional[Path],
    av_action_col: str = "policy_action",
    hdv_action_col: str = "initial_route_action",
) -> StaticHdvDataset:
    edge_ids = _load_edge_ids(edge_registry)
    ods, n_routes, route_edges = _load_route_edges(route_xml, edge_ids)
    route_to_edge = _route_projection_matrix(
        ods=ods,
        n_routes=n_routes,
        edge_ids=edge_ids,
        route_edges=route_edges,
    )
    od_to_idx = {od: i for i, od in enumerate(ods)}
    ep_dir = episodes_dir or (run_dir / "collector_runtime" / "episodes")
    paths = sorted(glob.glob(str(ep_dir / "ep*.csv")), key=_episode_num)
    if not paths:
        raise RuntimeError(f"no episode CSV files under {ep_dir}")

    episodes: List[int] = []
    av_route_rows: List[np.ndarray] = []
    hdv_route_rows: List[np.ndarray] = []
    av_edge_rows: List[np.ndarray] = []
    hdv_edge_rows: List[np.ndarray] = []
    hdv_od_weight_rows: List[np.ndarray] = []
    missing_routes = 0
    for path in paths:
        av_route = np.zeros((len(ods), n_routes), dtype=np.float64)
        hdv_route = np.zeros((len(ods), n_routes), dtype=np.float64)
        av_edge = np.zeros((len(edge_ids),), dtype=np.float64)
        hdv_edge = np.zeros((len(edge_ids),), dtype=np.float64)
        hdv_od_counts = np.zeros((len(ods),), dtype=np.float64)
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                role = str(row.get("vType", "")).strip().lower()
                if role not in {"av", "human"}:
                    continue
                try:
                    od = (int(float(row["origin"])), int(float(row["destination"])))
                except (KeyError, TypeError, ValueError):
                    continue
                od_idx = od_to_idx.get(od)
                if od_idx is None:
                    continue
                route = _safe_route(
                    row,
                    av_action_col if role == "av" else hdv_action_col,
                    fallback="initial_route_action",
                )
                if route < 0 or route >= n_routes:
                    continue
                edge_indices = route_edges.get((od[0], od[1], route))
                if not edge_indices:
                    missing_routes += 1
                    continue
                if role == "av":
                    av_route[od_idx, route] += 1.0
                    np.add.at(av_edge, np.asarray(edge_indices, dtype=np.int64), 1.0)
                else:
                    hdv_route[od_idx, route] += 1.0
                    hdv_od_counts[od_idx] += 1.0
                    np.add.at(hdv_edge, np.asarray(edge_indices, dtype=np.int64), 1.0)
        episodes.append(_episode_num(path))
        av_route_rows.append(_normalise_od_routes(av_route))
        hdv_route_rows.append(_normalise_od_routes(hdv_route))
        av_edge_rows.append(_normalise_np(av_edge))
        hdv_edge_rows.append(_normalise_np(hdv_edge))
        od_weight = _normalise_np(hdv_od_counts)
        if float(od_weight.sum()) <= 0.0:
            od_weight = np.full((len(ods),), 1.0 / max(len(ods), 1), dtype=np.float64)
        hdv_od_weight_rows.append(od_weight)

    if missing_routes:
        print(f"[warn] skipped {missing_routes} rows with no route edge mapping", file=sys.stderr)
    return StaticHdvDataset(
        episodes=np.asarray(episodes, dtype=np.int64),
        av_route=np.stack(av_route_rows, axis=0),
        av_edge=np.stack(av_edge_rows, axis=0),
        hdv_route=np.stack(hdv_route_rows, axis=0),
        hdv_edge=np.stack(hdv_edge_rows, axis=0),
        hdv_od_weight=np.stack(hdv_od_weight_rows, axis=0),
        ods=ods,
        n_routes=int(n_routes),
        edge_ids=edge_ids,
        route_to_edge=route_to_edge,
    )


def _make_split(
    episodes: np.ndarray,
    *,
    split: str,
    test_start_episode: int,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(episodes.size)
    if n < 2:
        raise RuntimeError("need at least two episodes")
    if split == "chronological":
        train = np.nonzero(episodes < int(test_start_episode))[0]
        test = np.nonzero(episodes >= int(test_start_episode))[0]
        if int(train.size) > 0 and int(test.size) > 0:
            return train, test
        cut = max(1, min(n - 1, int(round(n * (1.0 - float(test_fraction))))))
        return np.arange(cut, dtype=np.int64), np.arange(cut, n, dtype=np.int64)
    indices = np.arange(n, dtype=np.int64)
    shuffled = indices.tolist()
    random.Random(int(seed)).shuffle(shuffled)
    n_test = max(1, min(n - 1, int(round(n * float(test_fraction)))))
    test = np.asarray(sorted(shuffled[:n_test]), dtype=np.int64)
    train = np.asarray(sorted(shuffled[n_test:]), dtype=np.int64)
    return train, test


def _edge_metrics(target: np.ndarray, pred: np.ndarray, prefix: str) -> Dict[str, float]:
    target_n = _normalise_rows(target)
    pred_n = _normalise_rows(pred)
    eps = 1e-12
    ce = -(target_n * np.log(np.clip(pred_n, eps, 1.0))).sum(axis=-1)
    entropy = -(target_n * np.log(np.clip(target_n, eps, 1.0))).sum(axis=-1)
    l1 = np.abs(target_n - pred_n).sum(axis=-1)
    tv = 0.5 * l1
    top_k = min(5, int(target_n.shape[-1]))
    overlaps: List[float] = []
    for t, p in zip(target_n, pred_n):
        overlaps.append(float(len(set(np.argsort(t)[-top_k:]) & set(np.argsort(p)[-top_k:]))) / float(max(top_k, 1)))
    return {
        f"{prefix}_edge_ce": float(np.mean(ce)),
        f"{prefix}_edge_kl": float(np.mean(ce - entropy)),
        f"{prefix}_edge_l1": float(np.mean(l1)),
        f"{prefix}_edge_tv": float(np.mean(tv)),
        f"{prefix}_edge_top5_overlap": float(np.mean(overlaps)),
    }


def _route_metrics(target: np.ndarray, pred: np.ndarray, prefix: str) -> Dict[str, float]:
    target_n = np.asarray(target, dtype=np.float64).clip(min=0.0)
    pred_n = np.asarray(pred, dtype=np.float64).clip(min=0.0)
    target_n = target_n / np.maximum(target_n.sum(axis=-1, keepdims=True), 1e-12)
    pred_n = pred_n / np.maximum(pred_n.sum(axis=-1, keepdims=True), 1e-12)
    eps = 1e-12
    ce = -(target_n * np.log(np.clip(pred_n, eps, 1.0))).sum(axis=-1)
    entropy = -(target_n * np.log(np.clip(target_n, eps, 1.0))).sum(axis=-1)
    tv = 0.5 * np.abs(target_n - pred_n).sum(axis=-1)
    top1 = np.argmax(target_n, axis=-1) == np.argmax(pred_n, axis=-1)
    return {
        f"{prefix}_route_ce": float(np.mean(ce)),
        f"{prefix}_route_kl": float(np.mean(ce - entropy)),
        f"{prefix}_route_tv": float(np.mean(tv)),
        f"{prefix}_route_top1": float(np.mean(top1.astype(np.float64))),
    }


def _corr_np(x: np.ndarray, y: np.ndarray) -> float:
    xs = np.asarray(x, dtype=np.float64).reshape(-1)
    ys = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if int(finite.sum()) < 2:
        return float("nan")
    xs = xs[finite]
    ys = ys[finite]
    if float(xs.std()) <= 1e-12 or float(ys.std()) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _route_tv_series(route: np.ndarray) -> np.ndarray:
    arr = _normalise_route_tensor(route)
    if int(arr.shape[0]) < 2:
        return np.zeros((0,), dtype=np.float64)
    tv = 0.5 * np.abs(arr[1:] - arr[:-1]).sum(axis=-1)
    return tv.mean(axis=-1)


def _edge_tv_series(edge: np.ndarray) -> np.ndarray:
    arr = _normalise_rows(edge)
    if int(arr.shape[0]) < 2:
        return np.zeros((0,), dtype=np.float64)
    return 0.5 * np.abs(arr[1:] - arr[:-1]).sum(axis=-1)


def _window_stats(values: np.ndarray, window_size: int) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if int(arr.size) == 0:
        return {
            "mean": float("nan"),
            "max": float("nan"),
            "first_window_mean": float("nan"),
            "last_window_mean": float("nan"),
            "last_over_first": float("nan"),
        }
    window = max(1, min(int(window_size), int(arr.size)))
    first = arr[:window]
    last = arr[-window:]
    first_mean = float(first.mean())
    last_mean = float(last.mean())
    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "first_window_mean": first_mean,
        "last_window_mean": last_mean,
        "last_over_first": float(last_mean / max(first_mean, 1e-12)),
    }


def _add_convergence_metrics(
    summary: Dict[str, object],
    dataset: StaticHdvDataset,
    *,
    window_size: int,
    stable_threshold: float,
) -> None:
    av_route_delta = _route_tv_series(dataset.av_route)
    av_edge_delta = _edge_tv_series(dataset.av_edge)
    hdv_route_delta = _route_tv_series(dataset.hdv_route)
    hdv_edge_delta = _edge_tv_series(dataset.hdv_edge)

    summary["convergence_window_size"] = int(max(1, window_size))
    summary["convergence_stable_threshold"] = float(stable_threshold)
    for name, values in (
        ("av_route_delta", av_route_delta),
        ("av_edge_delta", av_edge_delta),
        ("hdv_route_delta", hdv_route_delta),
        ("hdv_edge_delta", hdv_edge_delta),
    ):
        for key, value in _window_stats(values, int(window_size)).items():
            summary[f"{name}_{key}"] = float(value)
        summary[f"{name}_stable_frac"] = float(np.mean(values <= float(stable_threshold))) if int(values.size) else float("nan")

    summary["av_hdv_edge_delta_corr_same_episode"] = _corr_np(av_edge_delta, hdv_edge_delta)
    summary["av_hdv_route_delta_corr_same_episode"] = _corr_np(av_route_delta, hdv_route_delta)
    if int(av_edge_delta.size) > 1 and int(hdv_edge_delta.size) > 1:
        summary["av_edge_delta_to_next_hdv_edge_delta_corr"] = _corr_np(av_edge_delta[:-1], hdv_edge_delta[1:])
        summary["av_route_delta_to_next_hdv_route_delta_corr"] = _corr_np(av_route_delta[:-1], hdv_route_delta[1:])
        summary["hdv_edge_delta_autocorr_lag1"] = _corr_np(hdv_edge_delta[:-1], hdv_edge_delta[1:])
        summary["hdv_route_delta_autocorr_lag1"] = _corr_np(hdv_route_delta[:-1], hdv_route_delta[1:])
    else:
        summary["av_edge_delta_to_next_hdv_edge_delta_corr"] = float("nan")
        summary["av_route_delta_to_next_hdv_route_delta_corr"] = float("nan")
        summary["hdv_edge_delta_autocorr_lag1"] = float("nan")
        summary["hdv_route_delta_autocorr_lag1"] = float("nan")


def _ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_shape = tuple(np.asarray(y_train).shape[1:])
    y_train_2d = np.asarray(y_train, dtype=np.float64).reshape(int(x_train.shape[0]), -1)
    x_test = np.asarray(x_test, dtype=np.float64)
    xt = np.concatenate([x_train, np.ones((int(x_train.shape[0]), 1), dtype=np.float64)], axis=1)
    xv = np.concatenate([x_test, np.ones((int(x_test.shape[0]), 1), dtype=np.float64)], axis=1)
    reg = float(ridge) * np.eye(int(xt.shape[1]), dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(xt.T @ xt + reg, xt.T @ y_train_2d)
    pred = np.clip(xv @ weights, 0.0, None).reshape((int(x_test.shape[0]),) + y_shape)
    if len(y_shape) == 2:
        return _normalise_route_tensor(pred)
    return _normalise_rows(pred)


def _normalise_route_tensor(route: np.ndarray) -> np.ndarray:
    arr = np.asarray(route, dtype=np.float64).clip(min=0.0)
    denom = arr.sum(axis=-1, keepdims=True)
    out = np.divide(arr, np.maximum(denom, 1e-12), out=np.zeros_like(arr, dtype=np.float64))
    empty = denom.reshape(-1) <= 1e-12
    if np.any(empty):
        flat = out.reshape(-1, int(out.shape[-1]))
        flat[empty] = 1.0 / max(int(out.shape[-1]), 1)
    return out


def _episode_dict(episodes: Iterable[int], values: np.ndarray) -> Dict[int, np.ndarray]:
    return {int(ep): np.asarray(value, dtype=np.float64) for ep, value in zip(episodes, values)}


def _online_previous(
    *,
    dataset: StaticHdvDataset,
    test_idx: np.ndarray,
    fallback_route: np.ndarray,
    fallback_edge: np.ndarray,
    fallback_od_weight: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    route_by_ep = _episode_dict(dataset.episodes, dataset.hdv_route)
    edge_by_ep = _episode_dict(dataset.episodes, dataset.hdv_edge)
    od_by_ep = _episode_dict(dataset.episodes, dataset.hdv_od_weight)
    available = sorted(int(ep) for ep in dataset.episodes.tolist())
    routes: List[np.ndarray] = []
    edges: List[np.ndarray] = []
    od_weights: List[np.ndarray] = []
    for ep in dataset.episodes[test_idx].tolist():
        previous = [value for value in available if value < int(ep)]
        if previous:
            prev = previous[-1]
            routes.append(route_by_ep[prev])
            edges.append(edge_by_ep[prev])
            od_weights.append(od_by_ep[prev])
        else:
            routes.append(fallback_route)
            edges.append(fallback_edge)
            od_weights.append(fallback_od_weight)
    return np.stack(routes, axis=0), np.stack(edges, axis=0), np.stack(od_weights, axis=0)


def _online_ema(
    *,
    dataset: StaticHdvDataset,
    test_idx: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_set = set(int(i) for i in np.asarray(test_idx, dtype=np.int64).tolist())
    route_state: Optional[np.ndarray] = None
    edge_state: Optional[np.ndarray] = None
    od_state: Optional[np.ndarray] = None
    route_preds: Dict[int, np.ndarray] = {}
    edge_preds: Dict[int, np.ndarray] = {}
    od_preds: Dict[int, np.ndarray] = {}
    for i, _ep in enumerate(dataset.episodes.tolist()):
        if route_state is not None and i in test_set:
            route_preds[i] = route_state.copy()
            edge_preds[i] = edge_state.copy() if edge_state is not None else dataset.hdv_edge[i]
            od_preds[i] = od_state.copy() if od_state is not None else dataset.hdv_od_weight[i]
        if route_state is None:
            route_state = dataset.hdv_route[i].copy()
            edge_state = dataset.hdv_edge[i].copy()
            od_state = dataset.hdv_od_weight[i].copy()
        else:
            route_state = (1.0 - float(alpha)) * route_state + float(alpha) * dataset.hdv_route[i]
            edge_state = _normalise_np((1.0 - float(alpha)) * edge_state + float(alpha) * dataset.hdv_edge[i])
            od_state = _normalise_np((1.0 - float(alpha)) * od_state + float(alpha) * dataset.hdv_od_weight[i])
    route_fallback = dataset.hdv_route[np.asarray(test_idx, dtype=np.int64)[0]]
    edge_fallback = dataset.hdv_edge[np.asarray(test_idx, dtype=np.int64)[0]]
    od_fallback = dataset.hdv_od_weight[np.asarray(test_idx, dtype=np.int64)[0]]
    return (
        np.stack([route_preds.get(int(i), route_fallback) for i in test_idx], axis=0),
        np.stack([edge_preds.get(int(i), edge_fallback) for i in test_idx], axis=0),
        np.stack([od_preds.get(int(i), od_fallback) for i in test_idx], axis=0),
    )


def _features(dataset: StaticHdvDataset, indices: np.ndarray, *, include_prev: bool) -> np.ndarray:
    episode_norm = dataset.episodes.astype(np.float64)
    denom = max(float(episode_norm.max() - episode_norm.min()), 1.0)
    episode_norm = ((episode_norm - float(episode_norm.min())) / denom).reshape(-1, 1)
    parts = [
        dataset.av_route.reshape(int(dataset.episodes.size), -1),
        dataset.av_edge,
        episode_norm,
    ]
    if include_prev:
        prev_route = np.zeros_like(dataset.hdv_route)
        prev_edge = np.zeros_like(dataset.hdv_edge)
        prev_od = np.zeros_like(dataset.hdv_od_weight)
        prev_route[0] = dataset.hdv_route[0]
        prev_edge[0] = dataset.hdv_edge[0]
        prev_od[0] = dataset.hdv_od_weight[0]
        prev_route[1:] = dataset.hdv_route[:-1]
        prev_edge[1:] = dataset.hdv_edge[:-1]
        prev_od[1:] = dataset.hdv_od_weight[:-1]
        parts.extend([prev_route.reshape(int(dataset.episodes.size), -1), prev_edge, prev_od])
    x = np.concatenate(parts, axis=1)
    return x[np.asarray(indices, dtype=np.int64)]


def _add_metrics(
    summary: Dict[str, object],
    dataset: StaticHdvDataset,
    test_idx: np.ndarray,
    *,
    route_pred: np.ndarray,
    edge_pred: Optional[np.ndarray],
    od_weight_pred: np.ndarray,
    prefix: str,
) -> None:
    summary.update(_route_metrics(dataset.hdv_route[test_idx], route_pred, prefix))
    projected = _project_route_to_edge(route_pred, od_weight_pred, dataset.route_to_edge)
    summary.update(_edge_metrics(dataset.hdv_edge[test_idx], projected, f"{prefix}_projected"))
    if edge_pred is not None:
        summary.update(_edge_metrics(dataset.hdv_edge[test_idx], edge_pred, prefix))


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    run_dir = Path(args.run_dir).resolve()
    edge_registry = Path(args.edge_registry).resolve()
    route_xml = Path(args.route_xml).resolve() if args.route_xml else run_dir / "collector_runtime" / "route.rou.xml"
    episodes_dir = Path(args.episodes_dir).resolve() if args.episodes_dir else None
    dataset = load_static_hdv_dataset(
        run_dir=run_dir,
        route_xml=route_xml,
        edge_registry=edge_registry,
        episodes_dir=episodes_dir,
    )
    train_idx, test_idx = _make_split(
        dataset.episodes,
        split=str(args.split),
        test_start_episode=int(args.test_start_episode),
        test_fraction=float(args.test_fraction),
        seed=int(args.seed),
    )

    route_global = _normalise_route_tensor(dataset.hdv_route[train_idx].mean(axis=0, keepdims=True))[0]
    edge_global = _normalise_np(dataset.hdv_edge[train_idx].mean(axis=0))
    od_global = _normalise_np(dataset.hdv_od_weight[train_idx].mean(axis=0))
    route_global_pred = np.stack([route_global for _ in test_idx], axis=0)
    edge_global_pred = np.stack([edge_global for _ in test_idx], axis=0)
    od_global_pred = np.stack([od_global for _ in test_idx], axis=0)

    summary: Dict[str, object] = {
        "run_dir": str(run_dir),
        "episodes_dir": str(episodes_dir or (run_dir / "collector_runtime" / "episodes")),
        "route_xml": str(route_xml),
        "edge_registry": str(edge_registry),
        "target_semantics": "episode_static_hdv_initial_route_response_and_projected_route_demand",
        "split": str(args.split),
        "episodes": int(dataset.episodes.size),
        "train_episodes": int(train_idx.size),
        "test_episodes": int(test_idx.size),
        "test_episode_min": int(dataset.episodes[test_idx].min()),
        "test_episode_max": int(dataset.episodes[test_idx].max()),
        "n_od": int(len(dataset.ods)),
        "n_routes": int(dataset.n_routes),
        "n_edges": int(len(dataset.edge_ids)),
        "ema_alpha": float(args.ema_alpha),
        "ridge": float(args.ridge),
    }
    _add_convergence_metrics(
        summary,
        dataset,
        window_size=int(args.convergence_window_size),
        stable_threshold=float(args.stable_threshold),
    )

    _add_metrics(
        summary,
        dataset,
        test_idx,
        route_pred=route_global_pred,
        edge_pred=edge_global_pred,
        od_weight_pred=od_global_pred,
        prefix="global_mean",
    )

    prev_route, prev_edge, prev_od = _online_previous(
        dataset=dataset,
        test_idx=test_idx,
        fallback_route=route_global,
        fallback_edge=edge_global,
        fallback_od_weight=od_global,
    )
    _add_metrics(
        summary,
        dataset,
        test_idx,
        route_pred=prev_route,
        edge_pred=prev_edge,
        od_weight_pred=prev_od,
        prefix="previous_episode_online",
    )

    ema_route, ema_edge, ema_od = _online_ema(dataset=dataset, test_idx=test_idx, alpha=float(args.ema_alpha))
    _add_metrics(
        summary,
        dataset,
        test_idx,
        route_pred=ema_route,
        edge_pred=ema_edge,
        od_weight_pred=ema_od,
        prefix="ema_online",
    )

    for include_prev, label in ((False, "ridge_av"), (True, "ridge_av_prev")):
        x_train = _features(dataset, train_idx, include_prev=include_prev)
        x_test = _features(dataset, test_idx, include_prev=include_prev)
        route_pred = _ridge_fit_predict(x_train, dataset.hdv_route[train_idx], x_test, ridge=float(args.ridge))
        edge_pred = _ridge_fit_predict(x_train, dataset.hdv_edge[train_idx], x_test, ridge=float(args.ridge))
        od_pred = _ridge_fit_predict(x_train, dataset.hdv_od_weight[train_idx], x_test, ridge=float(args.ridge))
        _add_metrics(
            summary,
            dataset,
            test_idx,
            route_pred=route_pred,
            edge_pred=edge_pred,
            od_weight_pred=od_pred,
            prefix=label,
        )

    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--edge-registry", required=True)
    parser.add_argument("--route-xml", default=None)
    parser.add_argument("--episodes-dir", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--split", choices=["chronological", "random"], default="chronological")
    parser.add_argument("--test-start-episode", type=int, default=301)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-alpha", type=float, default=0.1)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--convergence-window-size", type=int, default=50)
    parser.add_argument("--stable-threshold", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = evaluate(args)
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    output_json = (
        Path(args.output_json)
        if args.output_json
        else Path(args.run_dir) / "hdv_static_response_diagnostic.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[saved] {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
