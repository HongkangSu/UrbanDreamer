#!/usr/bin/env python3
"""Evaluate whether HDV edge pressure matches the WM-consumed target.

The online WM consumes an edge-level HDV context whose training target is the
episode mean of true per-decision HDV current-edge occupancy.  This diagnostic
therefore evaluates against saved ``hdv_edge_pressure_traces/*.npz`` files, not
against route-label accuracy.
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
import torch


RouteKey = Tuple[int, int, int]


def _episode_num(path: str) -> int:
    name = os.path.basename(path)
    for pattern in (r"episode_(\d+)\.npz$", r"ep(\d+)\.csv$"):
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return -1


def _normalise_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1).clip(min=0.0)
    total = float(arr.sum())
    if total <= eps:
        return np.zeros((int(arr.size),), dtype=np.float64)
    return arr / total


def _normalise_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return _normalise_np(arr, eps=eps)
    arr = arr.clip(min=0.0)
    denom = arr.sum(axis=-1, keepdims=True)
    out = np.divide(arr, np.maximum(denom, eps), out=np.zeros_like(arr, dtype=np.float64))
    return out


def _safe_route(row: Dict[str, str], preferred: str = "initial_route_action") -> int:
    for key in (preferred, "initial_route_action", "hdv_action", "action", "policy_action"):
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


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _resolve_optional_path(value: Optional[str], run_dir: Path, *fallback_parts: str) -> Optional[Path]:
    if value:
        path = Path(value)
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    if fallback_parts:
        fallback = run_dir.joinpath(*fallback_parts)
        return fallback if fallback.exists() else None
    return None


def _load_edge_ids(edge_registry: Optional[Path], trace_edge_ids: Sequence[str]) -> List[str]:
    if trace_edge_ids:
        return [str(edge) for edge in trace_edge_ids]
    if edge_registry is None:
        raise RuntimeError("edge ids were not saved in traces and --edge-registry was not provided")
    with edge_registry.open("r", encoding="utf-8") as f:
        data = json.load(f)
    edge_ids = data.get("edge_ids")
    if not isinstance(edge_ids, list) or not edge_ids:
        raise RuntimeError(f"edge registry has no edge_ids: {edge_registry}")
    return [str(edge) for edge in edge_ids]


def _edge_ids_or_generic(edge_registry: Optional[Path], n_edges: int) -> List[str]:
    if edge_registry is not None and edge_registry.exists():
        return _load_edge_ids(edge_registry, [])
    return [f"edge_{i}" for i in range(int(n_edges))]


@dataclass
class HdvTraceDataset:
    episodes: np.ndarray
    episode_mean: np.ndarray
    step_episode: np.ndarray
    step_hdv: np.ndarray
    step_decision_idx: np.ndarray
    step_sim_step: np.ndarray
    edge_ids: List[str]
    episode_av_mean: Optional[np.ndarray] = None
    step_av: Optional[np.ndarray] = None
    step_context: Optional[np.ndarray] = None


def _load_trace_dataset(trace_dir: Path, edge_registry: Optional[Path]) -> HdvTraceDataset:
    paths = sorted(glob.glob(str(trace_dir / "episode_*.npz")), key=_episode_num)
    if not paths:
        raise RuntimeError(
            f"no HDV pressure traces found under {trace_dir}. "
            "Run online WM training with wm_hdv_pressure_trace_enabled=true first."
        )

    episodes: List[int] = []
    episode_mean: List[np.ndarray] = []
    step_episode: List[int] = []
    step_hdv: List[np.ndarray] = []
    step_decision_idx: List[int] = []
    step_sim_step: List[int] = []
    trace_edge_ids: List[str] = []

    for path_str in paths:
        path = Path(path_str)
        with np.load(path, allow_pickle=False) as data:
            ep_arr = (
                np.asarray(data["episode"]).reshape(-1)
                if "episode" in data
                else np.asarray([_episode_num(path_str)], dtype=np.int64)
            )
            episode = int(ep_arr[0]) if ep_arr.size else int(_episode_num(path_str))
            hdv = _normalise_rows(np.asarray(data["hdv_action"], dtype=np.float64))
            if "hdv_episode_mean" in data:
                mean = _normalise_np(np.asarray(data["hdv_episode_mean"], dtype=np.float64))
            else:
                mean = _normalise_np(hdv.mean(axis=0))
            decision_idx = (
                np.asarray(data["decision_idx"], dtype=np.int64).reshape(-1)
                if "decision_idx" in data
                else np.arange(hdv.shape[0], dtype=np.int64)
            )
            sim_step = (
                np.asarray(data["sim_step"], dtype=np.int64).reshape(-1)
                if "sim_step" in data
                else np.full((hdv.shape[0],), -1, dtype=np.int64)
            )
            if not trace_edge_ids and "edge_ids" in data:
                trace_edge_ids = [str(v) for v in np.asarray(data["edge_ids"]).reshape(-1).tolist()]

        episodes.append(episode)
        episode_mean.append(mean)
        for i in range(int(hdv.shape[0])):
            step_episode.append(episode)
            step_hdv.append(hdv[i])
            step_decision_idx.append(int(decision_idx[i]) if i < int(decision_idx.size) else i)
            step_sim_step.append(int(sim_step[i]) if i < int(sim_step.size) else -1)

    edge_ids = _load_edge_ids(edge_registry, trace_edge_ids)
    order = np.argsort(np.asarray(episodes, dtype=np.int64))
    episodes_np = np.asarray(episodes, dtype=np.int64)[order]
    episode_mean_np = np.stack(episode_mean, axis=0)[order]
    if int(episode_mean_np.shape[-1]) != len(edge_ids):
        raise RuntimeError(
            f"trace edge dimension {episode_mean_np.shape[-1]} does not match edge ids {len(edge_ids)}"
        )
    return HdvTraceDataset(
        episodes=episodes_np,
        episode_mean=episode_mean_np,
        step_episode=np.asarray(step_episode, dtype=np.int64),
        step_hdv=np.stack(step_hdv, axis=0) if step_hdv else np.zeros((0, len(edge_ids)), dtype=np.float64),
        step_decision_idx=np.asarray(step_decision_idx, dtype=np.int64),
        step_sim_step=np.asarray(step_sim_step, dtype=np.int64),
        edge_ids=edge_ids,
    )


def _as_numpy_tensor(data: Dict[str, object], key: str) -> Optional[np.ndarray]:
    value = data.get(key)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_pt_dataset(dataset_pt: Path, edge_registry: Optional[Path]) -> HdvTraceDataset:
    raw = torch.load(str(dataset_pt), map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise RuntimeError(f"dataset is not a dict: {dataset_pt}")
    episodes_raw = _as_numpy_tensor(raw, "episode")
    hdv_raw = _as_numpy_tensor(raw, "hdv_edge_action")
    if episodes_raw is None or hdv_raw is None:
        raise RuntimeError(f"{dataset_pt} must contain 'episode' and 'hdv_edge_action'")
    step_episode = np.asarray(episodes_raw, dtype=np.int64).reshape(-1)
    step_hdv = _normalise_rows(np.asarray(hdv_raw, dtype=np.float64))
    if int(step_episode.shape[0]) != int(step_hdv.shape[0]):
        raise RuntimeError(
            f"episode length {step_episode.shape[0]} does not match hdv_edge_action rows {step_hdv.shape[0]}"
        )
    edge_ids = _edge_ids_or_generic(edge_registry, int(step_hdv.shape[-1]))
    if len(edge_ids) != int(step_hdv.shape[-1]):
        raise RuntimeError(f"edge ids {len(edge_ids)} do not match hdv_edge_action width {step_hdv.shape[-1]}")

    av_raw = _as_numpy_tensor(raw, "av_planned_edge_flow")
    if av_raw is None:
        av_raw = _as_numpy_tensor(raw, "av_edge_action")
    step_av = _normalise_rows(np.asarray(av_raw, dtype=np.float64)) if av_raw is not None else None
    context_raw = _as_numpy_tensor(raw, "context")
    step_context = np.asarray(context_raw, dtype=np.float64) if context_raw is not None else None

    episodes = np.asarray(sorted(int(ep) for ep in np.unique(step_episode)), dtype=np.int64)
    episode_mean: List[np.ndarray] = []
    episode_av_mean: List[np.ndarray] = []
    step_decision_idx = np.zeros_like(step_episode, dtype=np.int64)
    for ep in episodes.tolist():
        mask = step_episode == int(ep)
        idx = np.nonzero(mask)[0]
        episode_mean.append(_normalise_np(step_hdv[mask].mean(axis=0)))
        if step_av is not None:
            episode_av_mean.append(_normalise_np(step_av[mask].mean(axis=0)))
        step_decision_idx[idx] = np.arange(int(idx.size), dtype=np.int64)

    return HdvTraceDataset(
        episodes=episodes,
        episode_mean=np.stack(episode_mean, axis=0),
        step_episode=step_episode,
        step_hdv=step_hdv,
        step_decision_idx=step_decision_idx,
        step_sim_step=np.full_like(step_episode, -1, dtype=np.int64),
        edge_ids=edge_ids,
        episode_av_mean=np.stack(episode_av_mean, axis=0) if episode_av_mean else None,
        step_av=step_av,
        step_context=step_context,
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
        raise RuntimeError("need at least two traced episodes to form train/test diagnostics")
    if split == "chronological":
        train = np.nonzero(episodes < int(test_start_episode))[0]
        test = np.nonzero(episodes >= int(test_start_episode))[0]
        if int(train.size) > 0 and int(test.size) > 0:
            return train, test
        cut = max(1, min(n - 1, int(round(n * (1.0 - float(test_fraction))))))
        return np.arange(cut, dtype=np.int64), np.arange(cut, n, dtype=np.int64)

    indices = np.arange(n, dtype=np.int64)
    rng = random.Random(int(seed))
    shuffled = indices.tolist()
    rng.shuffle(shuffled)
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
    busy_l1: List[float] = []
    for t, p in zip(target_n, pred_n):
        target_top = set(np.argsort(t)[-top_k:].tolist())
        pred_top = set(np.argsort(p)[-top_k:].tolist())
        overlaps.append(float(len(target_top & pred_top)) / float(max(top_k, 1)))
        idx = np.asarray(sorted(target_top), dtype=np.int64)
        busy_l1.append(float(np.abs(t[idx] - p[idx]).sum()))

    return {
        f"{prefix}_edge_ce": float(np.mean(ce)),
        f"{prefix}_edge_kl": float(np.mean(ce - entropy)),
        f"{prefix}_edge_l1": float(np.mean(l1)),
        f"{prefix}_edge_tv": float(np.mean(tv)),
        f"{prefix}_top5_overlap": float(np.mean(overlaps)),
        f"{prefix}_target_top5_l1": float(np.mean(busy_l1)),
    }


def _prediction_rows_for_steps(
    *,
    step_episode: np.ndarray,
    episode_pred: Dict[int, np.ndarray],
    fallback: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [episode_pred.get(int(ep), fallback) for ep in np.asarray(step_episode, dtype=np.int64)],
        axis=0,
    )


def _ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    ones_train = np.ones((int(x_train.shape[0]), 1), dtype=np.float64)
    ones_test = np.ones((int(x_test.shape[0]), 1), dtype=np.float64)
    xt = np.concatenate([x_train, ones_train], axis=1)
    xv = np.concatenate([x_test, ones_test], axis=1)
    reg = float(ridge) * np.eye(int(xt.shape[1]), dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(xt.T @ xt + reg, xt.T @ y_train)
    pred = xv @ weights
    return _normalise_rows(np.clip(pred, 0.0, None))


def _features_with_episode_mean(
    *,
    step_episode: np.ndarray,
    step_av: np.ndarray,
    episode_av_by_ep: Dict[int, np.ndarray],
    context: Optional[np.ndarray],
) -> np.ndarray:
    episode_features = np.stack(
        [episode_av_by_ep[int(ep)] for ep in np.asarray(step_episode, dtype=np.int64)],
        axis=0,
    )
    parts = [np.asarray(step_av, dtype=np.float64), episode_features]
    if context is not None:
        parts.append(np.asarray(context, dtype=np.float64))
    return np.concatenate(parts, axis=1)


def _load_route_edges(route_xml: Path, edge_ids: Sequence[str]) -> Dict[RouteKey, List[int]]:
    edge_to_idx = {edge: i for i, edge in enumerate(edge_ids)}
    route_edges: Dict[RouteKey, List[int]] = {}
    tree = ET.parse(route_xml)
    for elem in tree.getroot().iter("route"):
        route_id = str(elem.attrib.get("id", ""))
        parts = route_id.split("_")
        if len(parts) != 3:
            continue
        try:
            origin, destination, route_idx = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        indices = [
            edge_to_idx[edge]
            for edge in str(elem.attrib.get("edges", "")).split()
            if edge in edge_to_idx
        ]
        if indices:
            route_edges[(origin, destination, route_idx)] = indices
    return route_edges


def _load_route_projected_hdv_demand(
    *,
    episodes_dir: Path,
    route_xml: Path,
    edge_ids: Sequence[str],
) -> Dict[int, np.ndarray]:
    route_edges = _load_route_edges(route_xml, edge_ids)
    if not route_edges:
        raise RuntimeError(f"no route edges could be parsed from {route_xml}")
    out: Dict[int, np.ndarray] = {}
    for path_str in sorted(glob.glob(str(episodes_dir / "ep*.csv")), key=_episode_num):
        episode = _episode_num(path_str)
        edge_counts = np.zeros((len(edge_ids),), dtype=np.float64)
        with open(path_str, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("vType", "")).strip().lower() != "human":
                    continue
                try:
                    origin = int(float(row["origin"]))
                    destination = int(float(row["destination"]))
                except (KeyError, TypeError, ValueError):
                    continue
                route_idx = _safe_route(row, "initial_route_action")
                indices = route_edges.get((origin, destination, route_idx))
                if not indices:
                    continue
                np.add.at(edge_counts, np.asarray(indices, dtype=np.int64), 1.0)
        out[int(episode)] = _normalise_np(edge_counts)
    return out


def _episode_pred_dict(episodes: Iterable[int], preds: np.ndarray) -> Dict[int, np.ndarray]:
    return {int(ep): _normalise_np(pred) for ep, pred in zip(episodes, preds)}


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    run_dir = Path(args.run_dir).resolve()
    config = _load_json(run_dir / "online_wm_dreamer_config.json")
    edge_registry = _resolve_optional_path(
        args.edge_registry or str(config.get("edge_registry", "") or ""),
        run_dir,
    )
    dataset_pt = _resolve_optional_path(args.dataset_pt, run_dir) if args.dataset_pt else None
    if dataset_pt is not None:
        dataset = _load_pt_dataset(dataset_pt, edge_registry)
        source_type = "offline_pt"
        source_path = dataset_pt
    else:
        trace_dir = _resolve_optional_path(args.trace_dir, run_dir, "hdv_edge_pressure_traces")
        if trace_dir is None:
            trace_dir = run_dir / "hdv_edge_pressure_traces"
        dataset = _load_trace_dataset(trace_dir, edge_registry)
        source_type = "online_trace_npz"
        source_path = trace_dir
    train_idx, test_idx = _make_split(
        dataset.episodes,
        split=str(args.split),
        test_start_episode=int(args.test_start_episode),
        test_fraction=float(args.test_fraction),
        seed=int(args.seed),
    )

    target_episode = dataset.episode_mean[test_idx]
    train_mean = _normalise_np(dataset.episode_mean[train_idx].mean(axis=0))
    global_pred = np.stack([train_mean for _ in test_idx], axis=0)

    ep_to_target = _episode_pred_dict(dataset.episodes, dataset.episode_mean)
    previous_preds: List[np.ndarray] = []
    available_eps = sorted(int(ep) for ep in dataset.episodes.tolist())
    for ep in dataset.episodes[test_idx].tolist():
        previous = [x for x in available_eps if x < int(ep)]
        previous_preds.append(ep_to_target[previous[-1]] if previous else train_mean)
    previous_pred = np.stack(previous_preds, axis=0)

    summary: Dict[str, object] = {
        "run_dir": str(run_dir),
        "source_type": source_type,
        "source_path": str(source_path),
        "target_semantics": "true_hdv_current_edge_occupancy_episode_mean",
        "split": str(args.split),
        "episodes": int(dataset.episodes.size),
        "train_episodes": int(train_idx.size),
        "test_episodes": int(test_idx.size),
        "test_episode_min": int(dataset.episodes[test_idx].min()),
        "test_episode_max": int(dataset.episodes[test_idx].max()),
        "n_edges": int(len(dataset.edge_ids)),
    }
    summary.update(_edge_metrics(target_episode, global_pred, "episode_global_mean"))
    summary.update(_edge_metrics(target_episode, previous_pred, "episode_previous_mean"))
    if dataset.episode_av_mean is not None:
        ridge_pred = _ridge_fit_predict(
            dataset.episode_av_mean[train_idx],
            dataset.episode_mean[train_idx],
            dataset.episode_av_mean[test_idx],
            ridge=float(args.ridge),
        )
        summary.update(_edge_metrics(target_episode, ridge_pred, "episode_av_ridge"))

    episode_pred_for_steps = _episode_pred_dict(dataset.episodes[test_idx], target_episode)
    test_episode_set = set(int(ep) for ep in dataset.episodes[test_idx].tolist())
    train_episode_set = set(int(ep) for ep in dataset.episodes[train_idx].tolist())
    train_step_mask = np.asarray([int(ep) in train_episode_set for ep in dataset.step_episode], dtype=bool)
    step_mask = np.asarray([int(ep) in test_episode_set for ep in dataset.step_episode], dtype=bool)
    step_target = dataset.step_hdv[step_mask]
    step_eps = dataset.step_episode[step_mask]
    if int(step_target.shape[0]) > 0:
        summary["test_steps"] = int(step_target.shape[0])
        summary.update(_edge_metrics(step_target, _prediction_rows_for_steps(
            step_episode=step_eps,
            episode_pred={int(ep): train_mean for ep in test_episode_set},
            fallback=train_mean,
        ), "step_global_mean"))
        summary.update(_edge_metrics(step_target, _prediction_rows_for_steps(
            step_episode=step_eps,
            episode_pred={int(ep): pred for ep, pred in zip(dataset.episodes[test_idx], previous_pred)},
            fallback=train_mean,
        ), "step_previous_mean"))
        summary.update(_edge_metrics(step_target, _prediction_rows_for_steps(
            step_episode=step_eps,
            episode_pred=episode_pred_for_steps,
            fallback=train_mean,
        ), "step_episode_mean_oracle"))
        if dataset.step_av is not None and dataset.episode_av_mean is not None:
            episode_av_by_ep = _episode_pred_dict(dataset.episodes, dataset.episode_av_mean)
            train_features = _features_with_episode_mean(
                step_episode=dataset.step_episode[train_step_mask],
                step_av=dataset.step_av[train_step_mask],
                episode_av_by_ep=episode_av_by_ep,
                context=dataset.step_context[train_step_mask] if dataset.step_context is not None else None,
            )
            test_features = _features_with_episode_mean(
                step_episode=dataset.step_episode[step_mask],
                step_av=dataset.step_av[step_mask],
                episode_av_by_ep=episode_av_by_ep,
                context=dataset.step_context[step_mask] if dataset.step_context is not None else None,
            )
            step_ridge = _ridge_fit_predict(
                train_features,
                dataset.step_hdv[train_step_mask],
                test_features,
                ridge=float(args.ridge),
            )
            summary.update(_edge_metrics(step_target, step_ridge, "step_av_context_ridge"))

    route_xml = _resolve_optional_path(args.route_xml, run_dir, "collector_runtime", "route.rou.xml")
    episodes_dir = _resolve_optional_path(args.episodes_dir, run_dir, "collector_runtime", "episodes")
    if route_xml is not None and episodes_dir is not None and route_xml.exists() and episodes_dir.exists():
        route_demand = _load_route_projected_hdv_demand(
            episodes_dir=episodes_dir,
            route_xml=route_xml,
            edge_ids=dataset.edge_ids,
        )
        route_episode_preds = [route_demand.get(int(ep), train_mean) for ep in dataset.episodes[test_idx].tolist()]
        route_episode_pred = np.stack(route_episode_preds, axis=0)
        summary["route_projection_semantics"] = "oracle_hdv_initial_route_decision_projected_to_full_path_edge_demand"
        summary.update(_edge_metrics(target_episode, route_episode_pred, "episode_route_projection_oracle"))
        if int(step_target.shape[0]) > 0:
            route_pred_by_ep = {
                int(ep): pred for ep, pred in zip(dataset.episodes[test_idx].tolist(), route_episode_pred)
            }
            summary.update(_edge_metrics(step_target, _prediction_rows_for_steps(
                step_episode=step_eps,
                episode_pred=route_pred_by_ep,
                fallback=train_mean,
            ), "step_route_projection_oracle"))
    else:
        summary["route_projection_skipped"] = "missing route xml or episode csv directory"

    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--dataset-pt", default=None)
    parser.add_argument("--edge-registry", default=None)
    parser.add_argument("--route-xml", default=None)
    parser.add_argument("--episodes-dir", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--split", choices=["chronological", "random"], default="chronological")
    parser.add_argument("--test-start-episode", type=int, default=301)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = evaluate(args)
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    output_json = Path(args.output_json) if args.output_json else Path(args.run_dir) / "hdv_edge_pressure_diagnostic.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[saved] {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
