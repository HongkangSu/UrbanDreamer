#!/usr/bin/env python3
"""Train the causal HDV response predictor from warmup/post-mutation records.

The checkpoint produced here is consumed by
``ablation.online_wm_dreamer_actor_critic.CausalHdvResponseModel``.  Prefer
per-episode CSV records because they preserve episode-start HDV route choices
and therefore train the static-edge-demand target.  Offline ``*.pt`` flow
datasets are supported as a fallback, but those only contain dynamic HDV edge
occupancy and are marked as such in the checkpoint metadata.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablation.online_wm_dreamer_actor_critic import (  # noqa: E402
    CausalHdvResponseModel,
    OnlineHdvEpisodeSummary,
    OnlineWmDreamerConfig,
)
from scripts.train_hdv_response_predictor_ablation import _load_route_edges  # noqa: E402


def _episode_num(path: str | Path) -> int:
    match = re.search(r"ep(\d+)\.csv$", os.path.basename(str(path)))
    return int(match.group(1)) if match else -1


def _normalise_np(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1).clip(min=0.0)
    total = float(arr.sum())
    if total <= eps:
        return np.zeros((int(arr.size),), dtype=np.float64)
    return arr / total


def _normalise_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).clip(min=0.0)
    if arr.ndim == 1:
        return _normalise_np(arr, eps=eps)
    denom = arr.sum(axis=-1, keepdims=True)
    return np.divide(arr, np.maximum(denom, eps), out=np.zeros_like(arr, dtype=np.float64))


def _safe_route(row: Dict[str, str], preferred: str, fallback: str = "initial_route_action") -> int:
    for key in (preferred, fallback, "policy_action", "action", "hdv_action"):
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            route = int(float(raw))
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


def _resolve_episode_dir(records_dir: Path) -> Optional[Path]:
    for candidate in (
        records_dir / "episodes",
        records_dir / "collector_runtime" / "episodes",
        records_dir,
    ):
        if list(candidate.glob("ep*.csv")):
            return candidate
    return None


def _as_numpy(raw: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class WarmupHdvDataset:
    episodes: np.ndarray
    av_edge: np.ndarray
    hdv_edge: np.ndarray
    edge_ids: List[str]
    target_source: str
    source_path: str

    @property
    def n_edges(self) -> int:
        return int(self.hdv_edge.shape[1])


def _load_csv_dataset(
    *,
    records_dir: Path,
    route_xml: Path,
    edge_registry: Path,
) -> WarmupHdvDataset:
    episode_dir = _resolve_episode_dir(records_dir)
    if episode_dir is None:
        raise RuntimeError(f"no ep*.csv files found under {records_dir}")
    edge_ids = _load_edge_ids(edge_registry)
    _ods, _n_routes, route_edges = _load_route_edges(str(route_xml), edge_ids)

    episodes: List[int] = []
    av_edges: List[np.ndarray] = []
    hdv_edges: List[np.ndarray] = []
    missing_routes = 0
    for csv_path in sorted(glob.glob(str(episode_dir / "ep*.csv")), key=_episode_num):
        av_edge = np.zeros((len(edge_ids),), dtype=np.float64)
        hdv_edge = np.zeros((len(edge_ids),), dtype=np.float64)
        with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                role = str(row.get("vType", "")).strip().lower()
                if role not in {"av", "human"}:
                    continue
                try:
                    od = (int(float(row["origin"])), int(float(row["destination"])))
                except (KeyError, TypeError, ValueError):
                    continue
                if role == "av":
                    route = _safe_route(row, "policy_action", fallback="initial_route_action")
                else:
                    route = _safe_route(row, "initial_route_action", fallback="hdv_action")
                edge_indices = route_edges.get((od[0], od[1], route), [])
                if not edge_indices:
                    missing_routes += 1
                    continue
                if role == "av":
                    np.add.at(av_edge, np.asarray(edge_indices, dtype=np.int64), 1.0)
                else:
                    np.add.at(hdv_edge, np.asarray(edge_indices, dtype=np.int64), 1.0)
        episodes.append(_episode_num(csv_path))
        av_edges.append(_normalise_np(av_edge))
        hdv_edges.append(_normalise_np(hdv_edge))

    if missing_routes:
        print(f"[warn] skipped {missing_routes} vehicle rows with no route-edge mapping", file=sys.stderr)
    if len(episodes) < 2:
        raise RuntimeError(f"need at least two episode CSVs, found {len(episodes)} in {episode_dir}")
    return WarmupHdvDataset(
        episodes=np.asarray(episodes, dtype=np.int64),
        av_edge=np.stack(av_edges, axis=0),
        hdv_edge=np.stack(hdv_edges, axis=0),
        edge_ids=edge_ids,
        target_source="static_edge_demand",
        source_path=str(episode_dir),
    )


def _load_pt_dataset(*, dataset_pt: Path, edge_registry: Optional[Path]) -> WarmupHdvDataset:
    raw = torch.load(str(dataset_pt), map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise RuntimeError(f"expected dict dataset: {dataset_pt}")
    episode_raw = _as_numpy(raw, "episode")
    hdv_raw = _as_numpy(raw, "hdv_edge_action")
    if episode_raw is None or hdv_raw is None:
        raise RuntimeError(f"{dataset_pt} must contain episode and hdv_edge_action")
    episode_step = np.asarray(episode_raw, dtype=np.int64).reshape(-1)
    hdv_step = _normalise_rows(np.asarray(hdv_raw, dtype=np.float64))
    av_raw = _as_numpy(raw, "av_planned_edge_flow")
    if av_raw is None:
        av_raw = _as_numpy(raw, "av_edge_action")
    if av_raw is None:
        av_step = np.zeros_like(hdv_step, dtype=np.float64)
    else:
        av_step = _normalise_rows(np.asarray(av_raw, dtype=np.float64))
    n = min(int(episode_step.shape[0]), int(hdv_step.shape[0]), int(av_step.shape[0]))
    episode_step = episode_step[:n]
    hdv_step = hdv_step[:n]
    av_step = av_step[:n]
    if edge_registry is not None and edge_registry.exists():
        edge_ids = _load_edge_ids(edge_registry)
    else:
        edge_ids = [f"edge_{i}" for i in range(int(hdv_step.shape[1]))]
    if len(edge_ids) != int(hdv_step.shape[1]):
        raise RuntimeError(f"edge ids {len(edge_ids)} do not match hdv_edge_action width {hdv_step.shape[1]}")

    episodes = np.asarray(sorted(int(ep) for ep in np.unique(episode_step)), dtype=np.int64)
    av_edges: List[np.ndarray] = []
    hdv_edges: List[np.ndarray] = []
    for ep in episodes.tolist():
        mask = episode_step == int(ep)
        av_edges.append(_normalise_np(av_step[mask].mean(axis=0)))
        hdv_edges.append(_normalise_np(hdv_step[mask].mean(axis=0)))
    if len(episodes) < 2:
        raise RuntimeError(f"need at least two episodes in {dataset_pt}, found {len(episodes)}")
    return WarmupHdvDataset(
        episodes=episodes,
        av_edge=np.stack(av_edges, axis=0),
        hdv_edge=np.stack(hdv_edges, axis=0),
        edge_ids=edge_ids,
        target_source="episode_mean_current_occupancy",
        source_path=str(dataset_pt),
    )


def _load_dataset(
    *,
    records_dir: Optional[Path],
    dataset_pt: Optional[Path],
    route_xml: Optional[Path],
    edge_registry: Optional[Path],
    target_source: str,
) -> WarmupHdvDataset:
    mode = str(target_source or "auto").strip().lower()
    if mode in {"auto", "static_edge_demand"} and records_dir is not None:
        route = route_xml or (records_dir / "route.rou.xml")
        if edge_registry is None:
            raise RuntimeError("--edge-registry is required for static_edge_demand CSV training")
        if route.exists() and _resolve_episode_dir(records_dir) is not None:
            return _load_csv_dataset(records_dir=records_dir, route_xml=route, edge_registry=edge_registry)
        if mode == "static_edge_demand":
            raise RuntimeError(
                f"static_edge_demand requested but no episode CSVs/route XML are available under {records_dir}"
            )
    if mode in {"auto", "episode_mean_current_occupancy"} and dataset_pt is not None:
        return _load_pt_dataset(dataset_pt=dataset_pt, edge_registry=edge_registry)
    raise RuntimeError(
        "could not load warmup HDV response dataset; provide --records-dir with episodes/ep*.csv "
        "or --dataset-pt for episode_mean_current_occupancy"
    )


def _summaries_from_dataset(dataset: WarmupHdvDataset) -> List[OnlineHdvEpisodeSummary]:
    return [
        OnlineHdvEpisodeSummary(
            episode=int(ep),
            av_edge_pressure=np.asarray(av, dtype=np.float32),
            hdv_static_edge_demand=np.asarray(hdv, dtype=np.float32),
            hdv_target=np.asarray(hdv, dtype=np.float32),
        )
        for ep, av, hdv in zip(dataset.episodes.tolist(), dataset.av_edge, dataset.hdv_edge)
    ]


def _edge_metrics(targets: Sequence[np.ndarray], preds: Sequence[np.ndarray]) -> Dict[str, float]:
    tvs: List[float] = []
    ces: List[float] = []
    ents: List[float] = []
    for target, pred in zip(targets, preds):
        t = _normalise_np(np.asarray(target, dtype=np.float64))
        p = _normalise_np(np.asarray(pred, dtype=np.float64))
        tvs.append(float(0.5 * np.abs(t - p).sum()))
        ces.append(float(-(t * np.log(np.clip(p, 1e-12, 1.0))).sum()))
        ents.append(float(-(t * np.log(np.clip(t, 1e-12, 1.0))).sum()))
    return {
        "edge_tv": float(np.mean(tvs)) if tvs else 0.0,
        "edge_ce": float(np.mean(ces)) if ces else 0.0,
        "edge_kl": float(np.mean(np.asarray(ces) - np.asarray(ents))) if ces else 0.0,
    }


def _chronological_split(summaries: Sequence[OnlineHdvEpisodeSummary], test_fraction: float) -> Tuple[List[OnlineHdvEpisodeSummary], List[OnlineHdvEpisodeSummary]]:
    ordered = sorted(list(summaries), key=lambda s: int(s.episode))
    n = len(ordered)
    split_idx = min(max(int(round(float(n) * (1.0 - float(test_fraction)))), 2), max(n - 1, 2))
    return ordered[:split_idx], ordered[split_idx:]


def _train_model(
    *,
    summaries: Sequence[OnlineHdvEpisodeSummary],
    n_edges: int,
    train_episodes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    depth: int,
    min_pairs: int,
    seed: int,
    device: str,
) -> Tuple[CausalHdvResponseModel, Dict[str, float]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    cfg = OnlineWmDreamerConfig(
        train_episodes=int(train_episodes),
        hdv_response_predictor_enabled=True,
        hdv_response_predictor_lr=float(lr),
        hdv_response_predictor_weight_decay=float(weight_decay),
        hdv_response_predictor_hidden_dim=int(hidden_dim),
        hdv_response_predictor_depth=int(depth),
        hdv_response_predictor_update_epochs=int(epochs),
        hdv_response_predictor_min_pairs=int(min_pairs),
        hdv_response_predictor_progress_denominator=int(train_episodes),
    )
    model = CausalHdvResponseModel(n_edges=int(n_edges), cfg=cfg, device=torch.device(device))
    metrics = model.update(list(summaries))
    return model, metrics


def _evaluate_pretrained_model(
    model: CausalHdvResponseModel,
    *,
    history: Sequence[OnlineHdvEpisodeSummary],
    eval_summaries: Sequence[OnlineHdvEpisodeSummary],
) -> Dict[str, float]:
    known = sorted(list(history), key=lambda s: int(s.episode))
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    prev_preds: List[np.ndarray] = []
    for current in sorted(list(eval_summaries), key=lambda s: int(s.episode)):
        pred, _metrics = model.predict_for_next_episode(known, episode=int(current.episode))
        if pred is not None:
            preds.append(pred)
            targets.append(current.hdv_static_edge_demand)
        if known:
            prev_preds.append(known[-1].hdv_static_edge_demand)
        known.append(current)
    out: Dict[str, float] = {}
    if preds:
        out.update({f"pretrained_{k}": v for k, v in _edge_metrics(targets, preds).items()})
    if prev_preds and len(prev_preds) == len(targets):
        out.update({f"prev_hdv_{k}": v for k, v in _edge_metrics(targets, prev_preds).items()})
    if preds and prev_preds and len(prev_preds) == len(preds):
        tv_model = [
            float(0.5 * np.abs(_normalise_np(t) - _normalise_np(p)).sum())
            for t, p in zip(targets, preds)
        ]
        tv_prev = [
            float(0.5 * np.abs(_normalise_np(t) - _normalise_np(p)).sum())
            for t, p in zip(targets, prev_preds)
        ]
        out["pretrained_beats_prev_frac"] = float(np.mean(np.asarray(tv_model) < np.asarray(tv_prev)))
    out["eval_pairs"] = float(len(preds))
    return out


def train_and_save_hdv_response_predictor(
    *,
    records_dir: Optional[Path],
    dataset_pt: Optional[Path],
    route_xml: Optional[Path],
    edge_registry: Optional[Path],
    output_ckpt: Path,
    target_source: str = "auto",
    epochs: int = 1500,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    depth: int = 2,
    min_pairs: int = 2,
    test_fraction: float = 0.25,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    dataset = _load_dataset(
        records_dir=records_dir,
        dataset_pt=dataset_pt,
        route_xml=route_xml,
        edge_registry=edge_registry,
        target_source=target_source,
    )
    summaries = _summaries_from_dataset(dataset)
    train_summaries, eval_summaries = _chronological_split(summaries, float(test_fraction))
    progress_denominator = int(max(dataset.episodes.max(), len(dataset.episodes), 1))
    eval_model, train_metrics = _train_model(
        summaries=train_summaries,
        n_edges=int(dataset.n_edges),
        train_episodes=progress_denominator,
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        min_pairs=int(min_pairs),
        seed=int(seed),
        device=str(device),
    )
    eval_metrics = _evaluate_pretrained_model(
        eval_model,
        history=train_summaries,
        eval_summaries=eval_summaries,
    )
    final_model, final_train_metrics = _train_model(
        summaries=summaries,
        n_edges=int(dataset.n_edges),
        train_episodes=progress_denominator,
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        min_pairs=int(min_pairs),
        seed=int(seed),
        device=str(device),
    )
    metadata: Dict[str, Any] = {
        "target_source": str(dataset.target_source),
        "source_path": str(dataset.source_path),
        "episodes": int(len(dataset.episodes)),
        "episode_min": int(dataset.episodes.min()),
        "episode_max": int(dataset.episodes.max()),
        "train_episodes_for_eval": int(len(train_summaries)),
        "eval_episodes": int(len(eval_summaries)),
        "n_edges": int(dataset.n_edges),
        "progress_denominator": int(progress_denominator),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "hidden_dim": int(hidden_dim),
        "depth": int(depth),
        "min_pairs": int(min_pairs),
        "seed": int(seed),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "final_train_metrics": final_train_metrics,
    }
    ckpt_path = final_model.save_checkpoint(output_ckpt, metrics=metadata)
    summary_path = ckpt_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {"checkpoint": str(ckpt_path), "summary": str(summary_path), **metadata}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train causal HDV response predictor from warmup records.")
    parser.add_argument("--records-dir", default="", help="Warmup/post-mutation records dir with episodes/ep*.csv.")
    parser.add_argument("--dataset-pt", default="", help="Offline collector .pt dataset fallback.")
    parser.add_argument("--route-xml", default="", help="Route XML. Defaults to records-dir/route.rou.xml.")
    parser.add_argument("--edge-registry", default="", help="Edge registry JSON.")
    parser.add_argument("--output-ckpt", required=True)
    parser.add_argument(
        "--target-source",
        choices=["auto", "static_edge_demand", "episode_mean_current_occupancy"],
        default="auto",
    )
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--min-pairs", type=int, default=2)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    records_dir = Path(args.records_dir).resolve() if args.records_dir else None
    dataset_pt = Path(args.dataset_pt).resolve() if args.dataset_pt else None
    route_xml = Path(args.route_xml).resolve() if args.route_xml else None
    edge_registry = Path(args.edge_registry).resolve() if args.edge_registry else None
    result = train_and_save_hdv_response_predictor(
        records_dir=records_dir,
        dataset_pt=dataset_pt,
        route_xml=route_xml,
        edge_registry=edge_registry,
        output_ckpt=Path(args.output_ckpt).resolve(),
        target_source=str(args.target_source),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        min_pairs=int(args.min_pairs),
        test_fraction=float(args.test_fraction),
        seed=int(args.seed),
        device=str(args.device),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[saved] {result['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
