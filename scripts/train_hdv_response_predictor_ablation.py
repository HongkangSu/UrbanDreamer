#!/usr/bin/env python3
"""Train/test HDV static-decision predictors on saved online-run episodes.

This script is intentionally offline.  It compares the current edge-level
HDV-response-head semantics against a structured OD-route predictor before
changing the online WM training path.
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
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Od = Tuple[int, int]
RouteKey = Tuple[int, int, int]


def _episode_num(path: str) -> int:
    match = re.search(r"ep(\d+)\.csv$", os.path.basename(path))
    return int(match.group(1)) if match else -1


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


def _normalise_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    total = float(x.sum())
    if total <= eps:
        return np.full_like(x, 1.0 / max(int(x.size), 1), dtype=np.float64)
    return x / total


def _ce_np(target: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    target = _normalise_np(target)
    pred = _normalise_np(pred)
    return float(-(target * np.log(np.clip(pred, eps, 1.0))).sum())


def _tv_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(_normalise_np(a) - _normalise_np(b)).sum())


def _load_edge_ids(edge_registry: str) -> List[str]:
    with open(edge_registry, "r", encoding="utf-8") as f:
        data = json.load(f)
    edge_ids = data.get("edge_ids")
    if not isinstance(edge_ids, list) or not edge_ids:
        raise RuntimeError(f"edge registry has no edge_ids: {edge_registry}")
    return [str(edge) for edge in edge_ids]


def _load_route_edges(route_xml: str, edge_ids: Sequence[str]) -> Tuple[List[Od], int, Dict[RouteKey, List[int]]]:
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
class HdvDecisionDataset:
    episodes: np.ndarray
    av_route: np.ndarray
    hdv_route: np.ndarray
    av_edge: np.ndarray
    hdv_edge: np.ndarray
    ods: List[Od]
    n_routes: int
    edge_ids: List[str]

    @property
    def n_od(self) -> int:
        return len(self.ods)

    @property
    def n_edges(self) -> int:
        return len(self.edge_ids)


def load_hdv_decision_dataset(
    *,
    run_dir: str,
    route_xml: str,
    edge_registry: str,
    av_action_col: str = "policy_action",
    hdv_action_col: str = "initial_route_action",
) -> HdvDecisionDataset:
    edge_ids = _load_edge_ids(edge_registry)
    ods, n_routes, route_edges = _load_route_edges(route_xml, edge_ids)
    od_to_idx = {od: i for i, od in enumerate(ods)}
    episode_dir = os.path.join(run_dir, "collector_runtime", "episodes")
    paths = sorted(glob.glob(os.path.join(episode_dir, "ep*.csv")), key=_episode_num)
    if not paths:
        raise RuntimeError(f"no episode CSV files under {episode_dir}")

    episodes: List[int] = []
    av_route_rows: List[np.ndarray] = []
    hdv_route_rows: List[np.ndarray] = []
    av_edge_rows: List[np.ndarray] = []
    hdv_edge_rows: List[np.ndarray] = []
    missing_routes = 0
    for path in paths:
        av_route = np.zeros((len(ods), n_routes), dtype=np.float64)
        hdv_route = np.zeros((len(ods), n_routes), dtype=np.float64)
        av_edge = np.zeros((len(edge_ids),), dtype=np.float64)
        hdv_edge = np.zeros((len(edge_ids),), dtype=np.float64)
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
                    np.add.at(hdv_edge, np.asarray(edge_indices, dtype=np.int64), 1.0)
        episodes.append(_episode_num(path))
        av_route_rows.append(_normalise_od_routes(av_route))
        hdv_route_rows.append(_normalise_od_routes(hdv_route))
        av_edge_rows.append(_normalise_np(av_edge))
        hdv_edge_rows.append(_normalise_np(hdv_edge))

    if missing_routes:
        print(f"[warn] skipped {missing_routes} rows with no route edge mapping", file=sys.stderr)
    return HdvDecisionDataset(
        episodes=np.asarray(episodes, dtype=np.int64),
        av_route=np.stack(av_route_rows, axis=0),
        hdv_route=np.stack(hdv_route_rows, axis=0),
        av_edge=np.stack(av_edge_rows, axis=0),
        hdv_edge=np.stack(hdv_edge_rows, axis=0),
        ods=ods,
        n_routes=n_routes,
        edge_ids=edge_ids,
    )


def _normalise_od_routes(route_counts: np.ndarray) -> np.ndarray:
    out = np.asarray(route_counts, dtype=np.float64).copy()
    for i in range(out.shape[0]):
        out[i] = _normalise_np(out[i])
    return out


def route_to_edge_projection_matrix(
    *,
    ods: Sequence[Od],
    n_routes: int,
    edge_ids: Sequence[str],
    route_xml: str,
) -> torch.Tensor:
    _, _, route_edges = _load_route_edges(route_xml, edge_ids)
    mat = np.zeros((len(ods) * int(n_routes), len(edge_ids)), dtype=np.float32)
    for od_i, (origin, destination) in enumerate(ods):
        for route in range(int(n_routes)):
            row = od_i * int(n_routes) + route
            edges = route_edges.get((origin, destination, route), [])
            if edges:
                mat[row, np.asarray(edges, dtype=np.int64)] = 1.0
    return torch.from_numpy(mat)


def make_features(dataset: HdvDecisionDataset, *, include_prev: bool, include_episode: bool) -> torch.Tensor:
    av_route = torch.from_numpy(dataset.av_route.reshape(len(dataset.episodes), -1)).float()
    av_edge = torch.from_numpy(dataset.av_edge).float()
    parts = [av_route, av_edge]
    if include_prev:
        prev_route = np.zeros_like(dataset.hdv_route)
        prev_edge = np.zeros_like(dataset.hdv_edge)
        prev_route[0] = dataset.hdv_route[0]
        prev_edge[0] = dataset.hdv_edge[0]
        prev_route[1:] = dataset.hdv_route[:-1]
        prev_edge[1:] = dataset.hdv_edge[:-1]
        parts.append(torch.from_numpy(prev_route.reshape(len(dataset.episodes), -1)).float())
        parts.append(torch.from_numpy(prev_edge).float())
    if include_episode:
        ep = dataset.episodes.astype(np.float32)
        denom = max(float(ep.max() - ep.min()), 1.0)
        ep_norm = ((ep - float(ep.min())) / denom).reshape(-1, 1)
        parts.append(torch.from_numpy(ep_norm).float())
    return torch.cat(parts, dim=1)


class EdgeLinearPredictor(nn.Module):
    def __init__(self, input_dim: int, n_edges: int):
        super().__init__()
        self.out = nn.Linear(input_dim, n_edges)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x)


class OdRoutePredictor(nn.Module):
    def __init__(self, input_dim: int, n_od: int, n_routes: int, hidden_dim: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(max(int(depth), 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            dim = hidden_dim
        layers.append(nn.Linear(dim, int(n_od) * int(n_routes)))
        self.net = nn.Sequential(*layers)
        self.n_od = int(n_od)
        self.n_routes = int(n_routes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.shape[0], self.n_od, self.n_routes)


def _soft_ce(logits: torch.Tensor, target_probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=dim)).sum(dim=dim).mean()


def _edge_metrics(target: np.ndarray, pred: np.ndarray, prefix: str) -> Dict[str, float]:
    ce = np.asarray([_ce_np(t, p) for t, p in zip(target, pred)], dtype=np.float64)
    tv = np.asarray([_tv_np(t, p) for t, p in zip(target, pred)], dtype=np.float64)
    ent = np.asarray([_ce_np(t, t) for t in target], dtype=np.float64)
    return {
        f"{prefix}_edge_ce": float(ce.mean()),
        f"{prefix}_edge_tv": float(tv.mean()),
        f"{prefix}_edge_kl": float((ce - ent).mean()),
        f"{prefix}_pred_episode_tv": float(np.mean([_tv_np(p, _normalise_np(pred.sum(axis=0))) for p in pred])),
    }


def _route_metrics(target: np.ndarray, pred: np.ndarray, prefix: str) -> Dict[str, float]:
    ces = []
    accs = []
    tvs = []
    for t, p in zip(target, pred):
        for od_i in range(t.shape[0]):
            ces.append(_ce_np(t[od_i], p[od_i]))
            tvs.append(_tv_np(t[od_i], p[od_i]))
            accs.append(float(int(np.argmax(t[od_i]) == np.argmax(p[od_i]))))
    return {
        f"{prefix}_route_ce": float(np.mean(ces)),
        f"{prefix}_route_tv": float(np.mean(tvs)),
        f"{prefix}_route_top1": float(np.mean(accs)),
    }


def _baseline_metrics(dataset: HdvDecisionDataset, test_idx: np.ndarray) -> Dict[str, float]:
    test_set = set(test_idx.tolist())
    train_idx = np.asarray([i for i in range(len(dataset.episodes)) if i not in test_set], dtype=np.int64)
    hdv_edge_train = _normalise_np(dataset.hdv_edge[train_idx].sum(axis=0))
    hdv_edge_test = dataset.hdv_edge[test_idx]
    edge_global = np.stack([hdv_edge_train for _ in test_idx], axis=0)
    out = _edge_metrics(hdv_edge_test, edge_global, "global_mean")

    hdv_route_train = dataset.hdv_route[train_idx]
    route_global = np.zeros((dataset.n_od, dataset.n_routes), dtype=np.float64)
    for od_i in range(dataset.n_od):
        route_global[od_i] = _normalise_np(hdv_route_train[:, od_i, :].sum(axis=0))
    route_pred = np.stack([route_global for _ in test_idx], axis=0)
    out.update(_route_metrics(dataset.hdv_route[test_idx], route_pred, "global_mean"))
    return out


def make_split(
    dataset: HdvDecisionDataset,
    *,
    split: str,
    test_start_episode: int,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    mode = str(split).strip().lower()
    if mode == "chronological":
        test_idx = np.flatnonzero(dataset.episodes >= int(test_start_episode)).astype(np.int64)
        train_idx = np.flatnonzero(dataset.episodes < int(test_start_episode)).astype(np.int64)
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        n = len(dataset.episodes)
        n_test = max(1, min(n - 1, int(round(float(test_fraction) * float(n)))))
        perm = rng.permutation(n)
        test_idx = np.sort(perm[:n_test]).astype(np.int64)
        train_idx = np.sort(perm[n_test:]).astype(np.int64)
    else:
        raise ValueError(f"Unsupported split={split!r}; expected chronological or random")
    if train_idx.size == 0 or test_idx.size == 0:
        raise RuntimeError(
            f"invalid {mode} split: train={train_idx.size}, test={test_idx.size}, "
            f"test_start_episode={test_start_episode}, test_fraction={test_fraction}"
        )
    return train_idx, test_idx


def train_edge_linear(
    *,
    x: torch.Tensor,
    target_edge: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    torch.manual_seed(seed)
    model = EdgeLinearPredictor(int(x.shape[1]), int(target_edge.shape[1]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = None
    best_loss = float("inf")
    for _ in range(int(epochs)):
        model.train()
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
    return {"edge_linear_train_ce": float(best_loss)}, pred


def train_route_mlp(
    *,
    x: torch.Tensor,
    target_route: torch.Tensor,
    route_to_edge: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    depth: int,
    seed: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    model = OdRoutePredictor(
        int(x.shape[1]),
        int(target_route.shape[1]),
        int(target_route.shape[2]),
        int(hidden_dim),
        int(depth),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = None
    best_loss = float("inf")
    for _ in range(int(epochs)):
        model.train()
        logits = model(x[train_idx])
        loss = _soft_ce(logits, target_route[train_idx], dim=-1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        route_pred = torch.softmax(model(x[test_idx]), dim=-1)
        flat_route = route_pred.reshape(route_pred.shape[0], -1)
        edge = flat_route @ route_to_edge.float()
        edge = edge / edge.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return {"route_mlp_train_ce": float(best_loss)}, route_pred.cpu().numpy(), edge.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--route-xml", required=True)
    parser.add_argument("--edge-registry", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["chronological", "random"], default="chronological")
    parser.add_argument("--test-start-episode", type=int, default=301)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-prev", action="store_true")
    parser.add_argument("--no-episode-feature", action="store_true")
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.set_num_threads(1)

    dataset = load_hdv_decision_dataset(
        run_dir=str(args.run_dir),
        route_xml=str(args.route_xml),
        edge_registry=str(args.edge_registry),
    )
    train_idx_np, test_idx_np = make_split(
        dataset,
        split=str(args.split),
        test_start_episode=int(args.test_start_episode),
        test_fraction=float(args.test_fraction),
        seed=int(args.seed),
    )

    x = make_features(
        dataset,
        include_prev=not bool(args.no_prev),
        include_episode=not bool(args.no_episode_feature),
    )
    target_edge = torch.from_numpy(dataset.hdv_edge).float()
    target_route = torch.from_numpy(dataset.hdv_route).float()
    train_idx = torch.from_numpy(train_idx_np).long()
    test_idx = torch.from_numpy(test_idx_np).long()
    route_to_edge = route_to_edge_projection_matrix(
        ods=dataset.ods,
        n_routes=dataset.n_routes,
        edge_ids=dataset.edge_ids,
        route_xml=str(args.route_xml),
    )

    summary: Dict[str, object] = {
        "run_dir": str(args.run_dir),
        "episodes": int(len(dataset.episodes)),
        "train_episodes": int(train_idx_np.size),
        "test_episodes": int(test_idx_np.size),
        "split": str(args.split),
        "test_start_episode": int(args.test_start_episode),
        "test_fraction": float(args.test_fraction),
        "test_episode_min": int(dataset.episodes[test_idx_np].min()),
        "test_episode_max": int(dataset.episodes[test_idx_np].max()),
        "n_od": int(dataset.n_od),
        "n_routes": int(dataset.n_routes),
        "n_edges": int(dataset.n_edges),
        "feature_dim": int(x.shape[1]),
        "include_prev": not bool(args.no_prev),
        "include_episode_feature": not bool(args.no_episode_feature),
    }
    summary.update(_baseline_metrics(dataset, test_idx_np))

    edge_train, edge_pred = train_edge_linear(
        x=x,
        target_edge=target_edge,
        train_idx=train_idx,
        test_idx=test_idx,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    summary.update(edge_train)
    summary.update(_edge_metrics(dataset.hdv_edge[test_idx_np], edge_pred, "edge_linear"))

    route_train, route_pred, route_edge_pred = train_route_mlp(
        x=x,
        target_route=target_route,
        route_to_edge=route_to_edge,
        train_idx=train_idx,
        test_idx=test_idx,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        seed=int(args.seed),
    )
    summary.update(route_train)
    summary.update(_route_metrics(dataset.hdv_route[test_idx_np], route_pred, "route_mlp"))
    summary.update(_edge_metrics(dataset.hdv_edge[test_idx_np], route_edge_pred, "route_mlp"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
