#!/usr/bin/env python3
"""Directly validate the episode-level HDV response predictor.

The evaluator freezes an online checkpoint at episode ``e`` and evaluates its
embedded HDV response predictor on later trace files.  Each prediction uses
only the previous episode's observed CAV pressure and HDV demand, matching the
training-time temporal and leakage-control semantics.

Besides held-out accuracy, the script writes an episode-to-episode response
curve, a true-versus-predicted road-segment comparison, and a frozen-state
counterfactual input test.  The counterfactual uses another *observed* CAV
pressure vector as the intervention while holding previous HDV demand and
episode progress fixed.  Its matched observed contrast is diagnostic rather
than a randomized causal estimate; that limitation is recorded in the output.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablation.online_wm_dreamer_actor_critic import CausalHdvResponsePredictor  # noqa: E402


METRIC_NAMES = ("ce", "kl", "js", "tv", "topk_recall")
PREDICTOR_NAMES = (
    "predictor",
    "persistence",
    "global_mean",
    "uniform_train_support",
    "av_mean_ablation",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    run_dir: Path
    checkpoint: Path


@dataclass(frozen=True)
class EvaluationConfig:
    top_k: int = 20
    max_test_episodes: int = 0
    match_pool_size: int = 10
    progress_match_weight: float = 1.0
    bootstrap_replicates: int = 2000
    bootstrap_block_size: int = 5
    seed: int = 42
    hash_checkpoint: bool = True


@dataclass(frozen=True)
class TraceRecord:
    episode: int
    av_pressure: np.ndarray
    hdv_demand: np.ndarray


@dataclass
class CaseResult:
    name: str
    summary: Dict[str, Any]
    episode_rows: List[Dict[str, Any]]
    counterfactual_rows: List[Dict[str, Any]]
    representative_rows: List[Dict[str, Any]]


def _normalise_distribution(values: np.ndarray, *, uniform_if_empty: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1).clip(min=0.0)
    total = float(arr.sum())
    if total > 1e-12:
        return arr / total
    if uniform_if_empty and int(arr.size) > 0:
        return np.full((int(arr.size),), 1.0 / float(int(arr.size)), dtype=np.float64)
    return np.zeros_like(arr, dtype=np.float64)


def _tv(first: np.ndarray, second: np.ndarray) -> float:
    a = _normalise_distribution(first)
    b = _normalise_distribution(second)
    return float(0.5 * np.abs(a - b).sum())


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-15:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _pearson(first: Sequence[float], second: Sequence[float]) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 2:
        return float("nan")
    a = a[finite]
    b = b[finite]
    if float(a.std()) <= 1e-15 or float(b.std()) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _distribution_metrics(target: np.ndarray, pred: np.ndarray, *, top_k: int) -> Dict[str, float]:
    target_n = _normalise_distribution(target)
    pred_n = _normalise_distribution(pred)
    eps = 1e-12
    ce = float(-(target_n * np.log(np.clip(pred_n, eps, 1.0))).sum())
    entropy = float(-(target_n * np.log(np.clip(target_n, eps, 1.0))).sum())
    midpoint = 0.5 * (target_n + pred_n)
    js = 0.5 * float((target_n * np.log(np.clip(target_n / np.clip(midpoint, eps, None), eps, None))).sum())
    js += 0.5 * float((pred_n * np.log(np.clip(pred_n / np.clip(midpoint, eps, None), eps, None))).sum())
    k = min(max(int(top_k), 1), int(target_n.size))
    target_top = set(np.argsort(target_n, kind="stable")[-k:].tolist())
    pred_top = set(np.argsort(pred_n, kind="stable")[-k:].tolist())
    return {
        "ce": ce,
        "kl": float(ce - entropy),
        "js": float(js),
        "tv": float(0.5 * np.abs(target_n - pred_n).sum()),
        "topk_recall": float(len(target_top & pred_top)) / float(k),
    }


def _mean_std(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if int(array.size) == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if int(array.size) > 1 else 0.0,
        "n": int(array.size),
    }


def _moving_block_bootstrap_ci(
    differences: Sequence[float],
    *,
    replicates: int,
    block_size: int,
    seed: int,
) -> Dict[str, Any]:
    values = np.asarray([float(v) for v in differences if np.isfinite(float(v))], dtype=np.float64)
    if int(values.size) == 0:
        return {"mean": None, "ci95": [None, None], "replicates": 0, "block_size": 0}
    if int(values.size) == 1 or int(replicates) <= 0:
        mean = float(values.mean())
        return {"mean": mean, "ci95": [mean, mean], "replicates": 0, "block_size": 1}

    n = int(values.size)
    block = min(max(int(block_size), 1), n)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty((int(replicates),), dtype=np.float64)
    for bootstrap_i in range(int(replicates)):
        sampled: List[float] = []
        while len(sampled) < n:
            start = int(rng.integers(0, n))
            sampled.extend(values[(start + offset) % n] for offset in range(block))
        estimates[bootstrap_i] = float(np.mean(sampled[:n]))
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "replicates": int(replicates),
        "block_size": int(block),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    return raw if isinstance(raw, dict) else {}


def _load_trace_records(run_dir: Path) -> Tuple[List[TraceRecord], List[str], str]:
    trace_dir = run_dir / "hdv_edge_pressure_traces"
    paths = sorted(trace_dir.glob("episode_*.npz"))
    if not paths:
        raise RuntimeError(f"no episode_*.npz traces under {trace_dir}")

    records: List[TraceRecord] = []
    edge_ids: List[str] = []
    target_source = ""
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            episode = int(np.asarray(data["episode"]).reshape(-1)[0])
            av_raw = np.asarray(data["av_action"], dtype=np.float64)
            av_pressure = _normalise_distribution(av_raw.mean(axis=0))
            if "hdv_static_edge_demand_mean" in data:
                hdv_demand = _normalise_distribution(
                    np.asarray(data["hdv_static_edge_demand_mean"], dtype=np.float64)
                )
            elif "hdv_target" in data:
                hdv_demand = _normalise_distribution(np.asarray(data["hdv_target"], dtype=np.float64))
            else:
                static = np.asarray(data["hdv_static_edge_demand"], dtype=np.float64)
                hdv_demand = _normalise_distribution(static.mean(axis=0))
            current_edge_ids = [str(value) for value in np.asarray(data["edge_ids"]).reshape(-1).tolist()]
            if not edge_ids:
                edge_ids = current_edge_ids
            elif current_edge_ids != edge_ids:
                raise RuntimeError(f"edge-id order changes at {path}")
            if "hdv_target_source" in data:
                source = str(np.asarray(data["hdv_target_source"]).reshape(-1)[0])
                if target_source and source != target_source:
                    raise RuntimeError(f"HDV target semantics change at {path}: {target_source} -> {source}")
                target_source = source
        if int(av_pressure.size) != len(edge_ids) or int(hdv_demand.size) != len(edge_ids):
            raise RuntimeError(f"edge dimension mismatch at {path}")
        records.append(TraceRecord(episode=episode, av_pressure=av_pressure, hdv_demand=hdv_demand))

    episodes = [record.episode for record in records]
    if len(set(episodes)) != len(episodes):
        raise RuntimeError(f"duplicate trace episodes under {trace_dir}")
    return sorted(records, key=lambda item: item.episode), edge_ids, target_source


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        state = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint is not a dictionary: {path}")
    return state


def _infer_predictor_shape(model_state: Mapping[str, torch.Tensor]) -> Tuple[int, int, int]:
    linear_weights = [
        (key, value)
        for key, value in model_state.items()
        if key.startswith("net.") and key.endswith(".weight") and int(value.ndim) == 2
    ]
    linear_weights.sort(key=lambda item: int(item[0].split(".")[1]))
    if len(linear_weights) < 2:
        raise RuntimeError("HDV predictor state has no recognizable hidden/output linear layers")
    hidden_dim = int(linear_weights[0][1].shape[0])
    n_edges = int(linear_weights[-1][1].shape[0])
    depth = int(len(linear_weights) - 1)
    expected_input = 2 * n_edges + 1
    if int(linear_weights[0][1].shape[1]) != expected_input:
        raise RuntimeError(
            f"HDV predictor input width {linear_weights[0][1].shape[1]} != 2*n_edges+1 ({expected_input})"
        )
    return n_edges, hidden_dim, depth


class _PredictorDistributionAdapter(torch.nn.Module):
    def __init__(
        self,
        predictor: CausalHdvResponsePredictor,
        *,
        n_edges: int,
        parameterization: str,
        residual_scale: float,
    ):
        super().__init__()
        self.predictor = predictor
        self.n_edges = int(n_edges)
        self.parameterization = str(parameterization)
        self.residual_scale = float(residual_scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.predictor(features)
        if self.parameterization == "absolute_logits":
            return torch.softmax(raw.float(), dim=-1)
        previous = features[..., self.n_edges : 2 * self.n_edges].float().clamp_min(0.0)
        previous_sum = previous.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(previous, 1.0 / max(self.n_edges, 1))
        previous = torch.where(
            previous_sum > 1e-12,
            previous / previous_sum.clamp_min(1e-12),
            uniform,
        )
        if self.residual_scale == 0.0:
            return previous
        return torch.softmax(
            torch.log(previous.clamp_min(1e-12)) + self.residual_scale * raw.float(),
            dim=-1,
        )


def _load_predictor(
    spec: CaseSpec,
) -> Tuple[_PredictorDistributionAdapter, Dict[str, Any]]:
    state = _torch_load(spec.checkpoint)
    model_state = state.get("hdv_response_predictor") or state.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise RuntimeError(f"checkpoint has no embedded HDV response predictor: {spec.checkpoint}")
    n_edges, inferred_hidden, inferred_depth = _infer_predictor_shape(model_state)
    config = _load_json(spec.run_dir / "online_wm_dreamer_config.json")
    hidden_dim = int(config.get("hdv_response_predictor_hidden_dim", inferred_hidden) or inferred_hidden)
    depth = int(config.get("hdv_response_predictor_depth", inferred_depth) or inferred_depth)
    if hidden_dim != inferred_hidden or depth != inferred_depth:
        raise RuntimeError(
            "predictor architecture metadata does not match checkpoint state: "
            f"config=({hidden_dim}, {depth}) state=({inferred_hidden}, {inferred_depth})"
        )
    predictor = CausalHdvResponsePredictor(
        n_edges=n_edges,
        hidden_dim=hidden_dim,
        depth=depth,
    ).cpu()
    predictor.load_state_dict(dict(model_state), strict=True)
    parameterization = str(
        state.get("hdv_response_parameterization")
        or state.get("parameterization")
        or "absolute_logits"
    )
    if parameterization not in {"absolute_logits", "residual_logit"}:
        raise RuntimeError(
            f"unsupported HDV response parameterization in checkpoint: {parameterization!r}"
        )
    residual_scale_raw = state.get(
        "hdv_response_residual_scale",
        state.get("residual_scale", 1.0),
    )
    residual_scale = float(1.0 if residual_scale_raw is None else residual_scale_raw)
    model = _PredictorDistributionAdapter(
        predictor,
        n_edges=n_edges,
        parameterization=parameterization,
        residual_scale=residual_scale,
    ).cpu()
    model.eval()

    checkpoint_episode = int(state.get("episode", 0) or 0)
    if checkpoint_episode <= 0:
        raise RuntimeError(f"checkpoint does not record a positive episode: {spec.checkpoint}")
    progress_denominator = int(
        config.get("hdv_response_predictor_progress_denominator", 0)
        or config.get("train_episodes", 0)
        or checkpoint_episode
    )
    metadata = {
        "checkpoint_episode": checkpoint_episode,
        "trained_pairs": int(state.get("hdv_response_trained_pairs", state.get("trained_pairs", 0)) or 0),
        "n_edges": n_edges,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "progress_denominator": progress_denominator,
        "parameterization": parameterization,
        "residual_scale": residual_scale,
    }
    del state
    gc.collect()
    return model, metadata


def _feature(av_pressure: np.ndarray, previous_hdv: np.ndarray, episode: int, progress_denominator: int) -> np.ndarray:
    av = _normalise_distribution(av_pressure)
    hdv = _normalise_distribution(previous_hdv)
    progress = float(np.clip(float(max(int(episode), 1)) / float(max(int(progress_denominator), 1)), 0.0, 1.0))
    return np.concatenate([av, hdv, np.asarray([progress], dtype=np.float64)], axis=0).astype(np.float32)


def _predict_batch(model: _PredictorDistributionAdapter, features: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    with torch.inference_mode():
        return model(tensor).cpu().numpy().astype(np.float64)


def _aggregate_accuracy(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for predictor in PREDICTOR_NAMES:
        output[predictor] = {
            metric: _mean_std(float(row[f"{predictor}_{metric}"]) for row in rows)
            for metric in METRIC_NAMES
        }
    return output


def _paired_accuracy(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    cfg: EvaluationConfig,
    seed_offset: int,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for metric_i, metric in enumerate(METRIC_NAMES):
        differences = [
            float(row[f"predictor_{metric}"]) - float(row[f"{baseline}_{metric}"])
            for row in rows
        ]
        output[metric] = _moving_block_bootstrap_ci(
            differences,
            replicates=int(cfg.bootstrap_replicates),
            block_size=int(cfg.bootstrap_block_size),
            seed=int(cfg.seed) + int(seed_offset) + metric_i,
        )
        output[metric]["direction"] = "lower_is_better" if metric != "topk_recall" else "higher_is_better"
        if metric == "topk_recall":
            output[metric]["predictor_win_fraction"] = float(np.mean(np.asarray(differences) > 0.0))
        else:
            output[metric]["predictor_win_fraction"] = float(np.mean(np.asarray(differences) < 0.0))
    return output


def _attach_prequential_accuracy(
    run_dir: Path,
    episode_rows: Sequence[Dict[str, Any]],
    *,
    cfg: EvaluationConfig,
) -> Dict[str, Any]:
    """Attach leakage-free online prediction errors already recorded during training."""
    log_path = run_dir / "online_wm_train_episodes.csv"
    if not log_path.exists():
        return {
            "available": False,
            "reason": f"missing {log_path.name}",
            "pairs": 0,
        }

    rows_by_episode = {int(row["episode"]): row for row in episode_rows}
    matched: List[Dict[str, Any]] = []
    with log_path.open("r", newline="", encoding="utf-8") as stream:
        for log_row in csv.DictReader(stream):
            try:
                episode = int(log_row["episode"])
                predictor_tv = float(log_row["hdv_response_pred_tv"])
                source_code = float(log_row["hdv_response_pred_source_code"])
                replay_accepted = int(float(log_row["wm_replay_accepted"]))
            except (KeyError, TypeError, ValueError):
                continue
            episode_row = rows_by_episode.get(episode)
            if episode_row is None or replay_accepted != 1 or source_code != 2.0:
                continue
            if not math.isfinite(predictor_tv):
                continue
            episode_row["online_prequential_predictor_tv"] = predictor_tv
            episode_row["online_prequential_source_code"] = source_code
            matched.append(episode_row)

    if not matched:
        return {
            "available": False,
            "reason": "no accepted episode-response predictions overlap the frozen holdout",
            "pairs": 0,
        }

    differences = [
        float(row["online_prequential_predictor_tv"]) - float(row["persistence_tv"])
        for row in matched
    ]
    paired = _moving_block_bootstrap_ci(
        differences,
        replicates=int(cfg.bootstrap_replicates),
        block_size=int(cfg.bootstrap_block_size),
        seed=int(cfg.seed) + 500,
    )
    paired["direction"] = "lower_is_better"
    paired["predictor_win_fraction"] = float(np.mean(np.asarray(differences) < 0.0))
    return {
        "available": True,
        "protocol": (
            "prediction logged after collecting the target episode but before appending that "
            "episode to replay or updating the response predictor"
        ),
        "source": str(log_path.resolve()),
        "pairs": int(len(matched)),
        "episode_min": min(int(row["episode"]) for row in matched),
        "episode_max": max(int(row["episode"]) for row in matched),
        "predictor_tv": _mean_std(float(row["online_prequential_predictor_tv"]) for row in matched),
        "persistence_tv": _mean_std(float(row["persistence_tv"]) for row in matched),
        "paired_predictor_minus_persistence_tv": paired,
    }


def _adaptation_event_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    cfg: EvaluationConfig,
) -> Dict[str, Any]:
    changed = [row for row in rows if float(row["observed_response_tv"]) > float(threshold)]
    unchanged = [row for row in rows if float(row["observed_response_tv"]) <= float(threshold)]
    output: Dict[str, Any] = {
        "threshold_tv": float(threshold),
        "changed_pairs": int(len(changed)),
        "unchanged_pairs": int(len(unchanged)),
        "changed_fraction": float(len(changed)) / float(max(len(rows), 1)),
        "unchanged_predicted_response_tv": _mean_std(
            float(row["predicted_response_tv"]) for row in unchanged
        ),
    }
    if changed:
        output["changed_accuracy"] = _aggregate_accuracy(changed)
        output["changed_predictor_minus_persistence"] = _paired_accuracy(
            changed,
            baseline="persistence",
            cfg=cfg,
            seed_offset=300,
        )
        output["changed_response_direction_cosine"] = _mean_std(
            float(row["response_direction_cosine"]) for row in changed
        )
    else:
        output["changed_accuracy"] = None
        output["changed_predictor_minus_persistence"] = None
        output["changed_response_direction_cosine"] = {"mean": None, "std": None, "n": 0}
    return output


def _counterfactual_rows(
    *,
    model: _PredictorDistributionAdapter,
    episodes: np.ndarray,
    previous_av: np.ndarray,
    previous_hdv: np.ndarray,
    targets: np.ndarray,
    factual_predictions: np.ndarray,
    progress_denominator: int,
    cfg: EvaluationConfig,
) -> List[Dict[str, Any]]:
    n = int(episodes.size)
    if n < 2:
        return []
    progress = np.clip(episodes.astype(np.float64) / float(max(progress_denominator, 1)), 0.0, 1.0)
    donors: List[int] = []
    for row_i in range(n):
        candidates = [candidate for candidate in range(n) if candidate != row_i]
        candidates.sort(
            key=lambda candidate: (
                _tv(previous_hdv[row_i], previous_hdv[candidate])
                + float(cfg.progress_match_weight) * abs(float(progress[row_i] - progress[candidate])),
                int(episodes[candidate]),
            )
        )
        pool = candidates[: min(max(int(cfg.match_pool_size), 1), len(candidates))]
        donor = max(pool, key=lambda candidate: (_tv(previous_av[row_i], previous_av[candidate]), -int(episodes[candidate])))
        donors.append(int(donor))

    counterfactual_features = np.stack(
        [
            _feature(previous_av[donor], previous_hdv[row_i], int(episodes[row_i]), progress_denominator)
            for row_i, donor in enumerate(donors)
        ],
        axis=0,
    )
    counterfactual_predictions = _predict_batch(model, counterfactual_features)
    rows: List[Dict[str, Any]] = []
    for row_i, donor in enumerate(donors):
        av_effect = previous_av[donor] - previous_av[row_i]
        predicted_effect = counterfactual_predictions[row_i] - factual_predictions[row_i]
        observed_effect = targets[donor] - targets[row_i]
        top_k = min(max(int(cfg.top_k), 1), int(observed_effect.size))
        top_effect = np.argsort(np.abs(observed_effect), kind="stable")[-top_k:]
        nonzero = top_effect[np.abs(observed_effect[top_effect]) > 1e-12]
        sign_agreement = (
            float(np.mean(np.sign(predicted_effect[nonzero]) == np.sign(observed_effect[nonzero])))
            if int(nonzero.size) > 0
            else float("nan")
        )
        rows.append(
            {
                "episode": int(episodes[row_i]),
                "donor_episode": int(episodes[donor]),
                "previous_hdv_match_tv": _tv(previous_hdv[row_i], previous_hdv[donor]),
                "progress_distance": float(abs(progress[row_i] - progress[donor])),
                "av_treatment_tv": _tv(previous_av[row_i], previous_av[donor]),
                "predicted_response_tv": _tv(factual_predictions[row_i], counterfactual_predictions[row_i]),
                "observed_matched_response_tv": _tv(targets[row_i], targets[donor]),
                "predicted_avoidance_alignment": -_cosine(av_effect, predicted_effect),
                "observed_avoidance_alignment": -_cosine(av_effect, observed_effect),
                "effect_direction_cosine": _cosine(predicted_effect, observed_effect),
                "topk_effect_sign_agreement": sign_agreement,
            }
        )
    return rows


def _counterfactual_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"pairs": 0}
    keys = (
        "previous_hdv_match_tv",
        "progress_distance",
        "av_treatment_tv",
        "predicted_response_tv",
        "observed_matched_response_tv",
        "predicted_avoidance_alignment",
        "observed_avoidance_alignment",
        "effect_direction_cosine",
        "topk_effect_sign_agreement",
    )
    output: Dict[str, Any] = {"pairs": int(len(rows))}
    for key in keys:
        output[key] = _mean_std(float(row[key]) for row in rows)
    output["nonzero_predicted_response_fraction"] = float(
        np.mean([float(row["predicted_response_tv"]) > 1e-8 for row in rows])
    )
    finite_avoidance = [
        float(row["predicted_avoidance_alignment"])
        for row in rows
        if np.isfinite(float(row["predicted_avoidance_alignment"]))
    ]
    output["positive_predicted_avoidance_fraction"] = (
        float(np.mean(np.asarray(finite_avoidance) > 0.0)) if finite_avoidance else None
    )
    output["response_magnitude_correlation"] = _pearson(
        [float(row["predicted_response_tv"]) for row in rows],
        [float(row["observed_matched_response_tv"]) for row in rows],
    )
    output["interpretation_boundary"] = (
        "Frozen-state model intervention with a feasible pressure vector from another held-out episode. "
        "The observed donor contrast is matched on previous HDV state but is not a randomized simulator intervention."
    )
    return output


def _representative_field_rows(
    *,
    edge_ids: Sequence[str],
    episode_rows: Sequence[Mapping[str, Any]],
    targets: np.ndarray,
    predictions: np.ndarray,
    persistence: np.ndarray,
    top_k: int,
) -> Tuple[int, List[Dict[str, Any]]]:
    errors = np.asarray([float(row["predictor_tv"]) for row in episode_rows], dtype=np.float64)
    median_error = float(np.median(errors))
    representative_i = int(np.argmin(np.abs(errors - median_error)))
    target = targets[representative_i]
    pred = predictions[representative_i]
    previous = persistence[representative_i]
    k = min(max(int(top_k), 1), int(target.size))
    indices = np.argsort(target, kind="stable")[-k:][::-1]
    rows = [
        {
            "rank": rank,
            "edge_index": int(edge_i),
            "edge_id": str(edge_ids[int(edge_i)]),
            "true_mass": float(target[int(edge_i)]),
            "predicted_mass": float(pred[int(edge_i)]),
            "persistence_mass": float(previous[int(edge_i)]),
        }
        for rank, edge_i in enumerate(indices, start=1)
    ]
    return int(episode_rows[representative_i]["episode"]), rows


def evaluate_case(spec: CaseSpec, cfg: EvaluationConfig) -> CaseResult:
    if not spec.run_dir.exists():
        raise FileNotFoundError(spec.run_dir)
    if not spec.checkpoint.exists():
        raise FileNotFoundError(spec.checkpoint)

    records, edge_ids, target_source = _load_trace_records(spec.run_dir)
    model, checkpoint_meta = _load_predictor(spec)
    if int(checkpoint_meta["n_edges"]) != len(edge_ids):
        raise RuntimeError(
            f"checkpoint has {checkpoint_meta['n_edges']} edges but traces have {len(edge_ids)}"
        )

    by_episode = {record.episode: record for record in records}
    checkpoint_episode = int(checkpoint_meta["checkpoint_episode"])
    train_records = [record for record in records if record.episode <= checkpoint_episode]
    test_records = [
        record
        for record in records
        if record.episode > checkpoint_episode and (record.episode - 1) in by_episode
    ]
    if int(cfg.max_test_episodes) > 0:
        test_records = test_records[: int(cfg.max_test_episodes)]
    if not train_records:
        raise RuntimeError(f"no traces at or before checkpoint episode {checkpoint_episode}")
    if not test_records:
        raise RuntimeError(f"no future adjacent-episode traces after checkpoint episode {checkpoint_episode}")

    episodes = np.asarray([record.episode for record in test_records], dtype=np.int64)
    previous_records = [by_episode[int(record.episode) - 1] for record in test_records]
    previous_av = np.stack([record.av_pressure for record in previous_records], axis=0)
    previous_hdv = np.stack([record.hdv_demand for record in previous_records], axis=0)
    targets = np.stack([record.hdv_demand for record in test_records], axis=0)
    train_av_mean = _normalise_distribution(np.stack([record.av_pressure for record in train_records]).mean(axis=0))
    train_hdv_mean = _normalise_distribution(np.stack([record.hdv_demand for record in train_records]).mean(axis=0))
    train_support = np.stack([record.hdv_demand for record in train_records]).sum(axis=0) > 0.0
    uniform_support = np.zeros((len(edge_ids),), dtype=np.float64)
    uniform_support[train_support] = 1.0 / float(max(int(train_support.sum()), 1))

    progress_denominator = int(checkpoint_meta["progress_denominator"])
    factual_features = np.stack(
        [
            _feature(previous_av[row_i], previous_hdv[row_i], int(episode), progress_denominator)
            for row_i, episode in enumerate(episodes.tolist())
        ],
        axis=0,
    )
    av_mean_features = np.stack(
        [
            _feature(train_av_mean, previous_hdv[row_i], int(episode), progress_denominator)
            for row_i, episode in enumerate(episodes.tolist())
        ],
        axis=0,
    )
    start = time.perf_counter()
    factual_predictions = _predict_batch(model, factual_features)
    prediction_seconds = float(time.perf_counter() - start)
    av_mean_predictions = _predict_batch(model, av_mean_features)
    global_predictions = np.broadcast_to(train_hdv_mean, targets.shape).copy()
    uniform_predictions = np.broadcast_to(uniform_support, targets.shape).copy()

    episode_rows: List[Dict[str, Any]] = []
    for row_i, episode in enumerate(episodes.tolist()):
        row: Dict[str, Any] = {"episode": int(episode), "previous_episode": int(episode) - 1}
        predictions = {
            "predictor": factual_predictions[row_i],
            "persistence": previous_hdv[row_i],
            "global_mean": global_predictions[row_i],
            "uniform_train_support": uniform_predictions[row_i],
            "av_mean_ablation": av_mean_predictions[row_i],
        }
        for name, prediction in predictions.items():
            for metric, value in _distribution_metrics(targets[row_i], prediction, top_k=int(cfg.top_k)).items():
                row[f"{name}_{metric}"] = value
        true_effect = targets[row_i] - previous_hdv[row_i]
        predicted_effect = factual_predictions[row_i] - previous_hdv[row_i]
        row.update(
            {
                "observed_response_tv": _tv(targets[row_i], previous_hdv[row_i]),
                "predicted_response_tv": _tv(factual_predictions[row_i], previous_hdv[row_i]),
                "response_direction_cosine": _cosine(predicted_effect, true_effect),
                "response_magnitude_abs_error": abs(
                    _tv(factual_predictions[row_i], previous_hdv[row_i])
                    - _tv(targets[row_i], previous_hdv[row_i])
                ),
            }
        )
        episode_rows.append(row)

    counterfactual_rows = _counterfactual_rows(
        model=model,
        episodes=episodes,
        previous_av=previous_av,
        previous_hdv=previous_hdv,
        targets=targets,
        factual_predictions=factual_predictions,
        progress_denominator=progress_denominator,
        cfg=cfg,
    )
    representative_episode, representative_rows = _representative_field_rows(
        edge_ids=edge_ids,
        episode_rows=episode_rows,
        targets=targets,
        predictions=factual_predictions,
        persistence=previous_hdv,
        top_k=min(int(cfg.top_k), 20),
    )

    response_observed = [float(row["observed_response_tv"]) for row in episode_rows]
    response_predicted = [float(row["predicted_response_tv"]) for row in episode_rows]
    prequential_accuracy = _attach_prequential_accuracy(spec.run_dir, episode_rows, cfg=cfg)
    summary: Dict[str, Any] = {
        "case": spec.name,
        "provenance": {
            "run_dir": str(spec.run_dir.resolve()),
            "checkpoint": str(spec.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(spec.checkpoint) if cfg.hash_checkpoint else None,
            "checkpoint_episode": checkpoint_episode,
            "checkpoint_trained_pairs": int(checkpoint_meta["trained_pairs"]),
            "trace_episode_min": int(records[0].episode),
            "trace_episode_max": int(records[-1].episode),
            "target_semantics": target_source or "static_edge_demand",
            "n_edges": int(len(edge_ids)),
            "predictor_hidden_dim": int(checkpoint_meta["hidden_dim"]),
            "predictor_depth": int(checkpoint_meta["depth"]),
            "progress_denominator": progress_denominator,
        },
        "split": {
            "protocol": "frozen_checkpoint_one_step_rolling_holdout",
            "train_trace_episodes": int(len(train_records)),
            "test_pairs": int(len(test_records)),
            "test_episode_min": int(episodes.min()),
            "test_episode_max": int(episodes.max()),
            "teacher_forcing": "observed previous-episode HDV demand only; no checkpoint updates",
            "leakage_guard": "checkpoint episode precedes every evaluated target episode",
        },
        "accuracy": _aggregate_accuracy(episode_rows),
        "paired_predictor_minus_persistence": _paired_accuracy(
            episode_rows, baseline="persistence", cfg=cfg, seed_offset=100
        ),
        "paired_predictor_minus_av_mean_ablation": _paired_accuracy(
            episode_rows, baseline="av_mean_ablation", cfg=cfg, seed_offset=200
        ),
        "online_prequential_accuracy": prequential_accuracy,
        "response_curve": {
            "observed_response_tv": _mean_std(response_observed),
            "predicted_response_tv": _mean_std(response_predicted),
            "magnitude_pearson_r": _pearson(response_observed, response_predicted),
            "direction_cosine": _mean_std(float(row["response_direction_cosine"]) for row in episode_rows),
            "magnitude_absolute_error": _mean_std(
                float(row["response_magnitude_abs_error"]) for row in episode_rows
            ),
        },
        "adaptation_events": _adaptation_event_summary(
            episode_rows,
            threshold=1e-8,
            cfg=cfg,
        ),
        "counterfactual": _counterfactual_summary(counterfactual_rows),
        "representative_field_episode": representative_episode,
        "predictor_latency_ms_per_episode": 1000.0 * prediction_seconds / float(len(test_records)),
        "top_k": int(cfg.top_k),
    }
    return CaseResult(
        name=spec.name,
        summary=summary,
        episode_rows=episode_rows,
        counterfactual_rows=counterfactual_rows,
        representative_rows=representative_rows,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(dict(payload)), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if isinstance(value, float) and not math.isfinite(value) else value
                    for key, value in row.items()
                }
            )


def write_case_artifacts(result: CaseResult, output_dir: Path) -> None:
    case_dir = output_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", result.name)
    _write_json(case_dir / "summary.json", result.summary)
    _write_csv(case_dir / "episode_metrics.csv", result.episode_rows)
    _write_csv(case_dir / "counterfactual_interventions.csv", result.counterfactual_rows)
    _write_csv(case_dir / "representative_field.csv", result.representative_rows)


def _rolling_mean(values: Sequence[float], window: int = 10) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    out = np.empty_like(array)
    for index in range(int(array.size)):
        out[index] = float(array[max(0, index - int(window) + 1) : index + 1].mean())
    return out


def write_plots(results: Sequence[CaseResult], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    blue = "#2C7BB6"
    orange = "#D95F02"
    gray = "#666666"
    green = "#1B9E77"

    n_cases = len(results)
    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(2, max(n_cases, 2), height_ratios=[1.05, 0.95])
    for case_i, result in enumerate(results):
        axis = fig.add_subplot(grid[0, case_i])
        episodes = [int(row["episode"]) for row in result.episode_rows]
        observed = [float(row["observed_response_tv"]) for row in result.episode_rows]
        predicted = [float(row["predicted_response_tv"]) for row in result.episode_rows]
        axis.plot(episodes, observed, color=gray, alpha=0.18, linewidth=0.6)
        axis.plot(episodes, predicted, color=blue, alpha=0.18, linewidth=0.6)
        axis.plot(episodes, _rolling_mean(observed), color=gray, linewidth=1.5, label="Observed")
        axis.plot(episodes, _rolling_mean(predicted), color=blue, linewidth=1.5, label="Predicted")
        axis.set_title(result.name)
        axis.set_xlabel("Held-out episode")
        axis.set_ylabel(r"Response magnitude, $\mathrm{TV}(H_e,H_{e-1})$")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="upper right")

    bottom_grid = grid[1, :].subgridspec(1, 2)
    accuracy_axis = fig.add_subplot(bottom_grid[0, 0])
    labels = [
        "Gretz"
        if result.name.startswith("Gretz-")
        else "Fontenay"
        if result.name.startswith("Fontenay-")
        else result.name
        for result in results
    ]
    x = np.arange(len(results), dtype=np.float64)
    width = 0.25
    width = 0.2
    for offset, predictor, label, color in (
        (-1.5 * width, "predictor", "Frozen predictor", blue),
        (-0.5 * width, "online_prequential", "Online predictor", green),
        (0.5 * width, "persistence", "Persistence", gray),
        (1.5 * width, "av_mean_ablation", "Mean-CAV input", orange),
    ):
        if predictor == "online_prequential":
            means = [
                float(result.summary["online_prequential_accuracy"]["predictor_tv"]["mean"])
                if bool(result.summary["online_prequential_accuracy"].get("available"))
                else float("nan")
                for result in results
            ]
        else:
            means = [float(result.summary["accuracy"][predictor]["tv"]["mean"]) for result in results]
        accuracy_axis.bar(x + offset, means, width=width, label=label, color=color)
    accuracy_axis.set_xticks(x, labels)
    accuracy_axis.set_ylabel("One-step prediction TV error")
    accuracy_axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    accuracy_axis.spines[["top", "right"]].set_visible(False)
    accuracy_axis.legend(frameon=False)

    intervention_axis = fig.add_subplot(bottom_grid[0, 1])
    predicted_alignment = [
        float(result.summary["counterfactual"]["predicted_avoidance_alignment"]["mean"])
        for result in results
    ]
    observed_alignment = [
        float(result.summary["counterfactual"]["observed_avoidance_alignment"]["mean"])
        for result in results
    ]
    intervention_axis.bar(x - width / 2.0, predicted_alignment, width=width, label="Predicted", color=green)
    intervention_axis.bar(x + width / 2.0, observed_alignment, width=width, label="Matched observed", color=gray)
    intervention_axis.axhline(0.0, color="#333333", linewidth=0.7)
    intervention_axis.set_xticks(x, labels)
    intervention_axis.set_ylim(-1.0, 1.0)
    intervention_axis.set_ylabel("HDV avoidance alignment")
    intervention_axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    intervention_axis.spines[["top", "right"]].set_visible(False)
    intervention_axis.legend(frameon=False)

    for column in range(n_cases, max(n_cases, 2)):
        fig.add_subplot(grid[0, column]).axis("off")
    fig.savefig(output_dir / "hdv_response_validation.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "hdv_response_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    field_fig, axes = plt.subplots(
        1,
        n_cases,
        figsize=(7.2, 2.8),
        constrained_layout=True,
        squeeze=False,
    )
    for case_i, result in enumerate(results):
        axis = axes[0, case_i]
        rows = result.representative_rows[:12]
        positions = np.arange(len(rows), dtype=np.float64)
        axis.bar(positions - width, [float(row["true_mass"]) for row in rows], width, label="Observed", color=gray)
        axis.bar(positions, [float(row["predicted_mass"]) for row in rows], width, label="Predicted", color=blue)
        axis.bar(
            positions + width,
            [float(row["persistence_mass"]) for row in rows],
            width,
            label="Persistence",
            color=orange,
        )
        short_labels = [str(row["edge_id"])[-16:] for row in rows]
        axis.set_xticks(positions, short_labels, rotation=55, ha="right")
        axis.set_ylabel("HDV route-demand mass")
        axis.set_title(f"{result.name}, episode {result.summary['representative_field_episode']}")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    field_fig.savefig(output_dir / "hdv_true_vs_predicted_fields.pdf", bbox_inches="tight")
    field_fig.savefig(output_dir / "hdv_true_vs_predicted_fields.png", dpi=220, bbox_inches="tight")
    plt.close(field_fig)


def _aggregate_summary(results: Sequence[CaseResult], cfg: EvaluationConfig) -> Dict[str, Any]:
    table_rows: List[Dict[str, Any]] = []
    prequential_rows: List[Dict[str, Any]] = []
    for result in results:
        accuracy = result.summary["accuracy"]
        for predictor in PREDICTOR_NAMES:
            row = {"case": result.name, "predictor": predictor, "test_pairs": result.summary["split"]["test_pairs"]}
            for metric in METRIC_NAMES:
                row[f"{metric}_mean"] = accuracy[predictor][metric]["mean"]
                row[f"{metric}_std"] = accuracy[predictor][metric]["std"]
            table_rows.append(row)
        prequential = result.summary["online_prequential_accuracy"]
        if bool(prequential.get("available")):
            prequential_rows.append(
                {
                    "case": result.name,
                    "test_pairs": prequential["pairs"],
                    "predictor_tv_mean": prequential["predictor_tv"]["mean"],
                    "predictor_tv_std": prequential["predictor_tv"]["std"],
                    "persistence_tv_mean": prequential["persistence_tv"]["mean"],
                    "persistence_tv_std": prequential["persistence_tv"]["std"],
                    "predictor_minus_persistence_tv": prequential[
                        "paired_predictor_minus_persistence_tv"
                    ]["mean"],
                    "difference_ci95_low": prequential["paired_predictor_minus_persistence_tv"][
                        "ci95"
                    ][0],
                    "difference_ci95_high": prequential["paired_predictor_minus_persistence_tv"][
                        "ci95"
                    ][1],
                    "predictor_win_fraction": prequential[
                        "paired_predictor_minus_persistence_tv"
                    ]["predictor_win_fraction"],
                }
            )
    return {
        "schema_version": "hdv_response_validation_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "protocol": {
            "checkpoint_policy": "freeze at the declared checkpoint episode",
            "holdout": "evaluate only strictly later adjacent episode pairs",
            "baselines": list(PREDICTOR_NAMES[1:]),
            "metrics": list(METRIC_NAMES),
            "top_k": int(cfg.top_k),
            "serial_uncertainty": (
                f"paired circular moving-block bootstrap, block={cfg.bootstrap_block_size}, "
                f"replicates={cfg.bootstrap_replicates}"
            ),
        },
        "cases": {result.name: result.summary for result in results},
        "table_rows": table_rows,
        "prequential_rows": prequential_rows,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("NAME", "RUN_DIR", "CHECKPOINT"),
        required=True,
        help="Repeat for each network/checkpoint case.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-test-episodes", type=int, default=0)
    parser.add_argument("--match-pool-size", type=int, default=10)
    parser.add_argument("--progress-match-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-block-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    specs = [
        CaseSpec(name=str(name), run_dir=Path(run_dir).resolve(), checkpoint=Path(checkpoint).resolve())
        for name, run_dir, checkpoint in args.case
    ]
    cfg = EvaluationConfig(
        top_k=int(args.top_k),
        max_test_episodes=int(args.max_test_episodes),
        match_pool_size=int(args.match_pool_size),
        progress_match_weight=float(args.progress_match_weight),
        bootstrap_replicates=int(args.bootstrap_replicates),
        bootstrap_block_size=int(args.bootstrap_block_size),
        seed=int(args.seed),
        hash_checkpoint=not bool(args.skip_checkpoint_hash),
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[CaseResult] = []
    for spec in specs:
        print(f"[hdv-response] evaluating {spec.name}: {spec.checkpoint}", flush=True)
        result = evaluate_case(spec, cfg)
        write_case_artifacts(result, output_dir)
        results.append(result)
    aggregate = _aggregate_summary(results, cfg)
    _write_json(output_dir / "summary.json", aggregate)
    _write_csv(output_dir / "table_rows.csv", aggregate["table_rows"])
    _write_csv(output_dir / "prequential_rows.csv", aggregate["prequential_rows"])
    if not bool(args.no_plots):
        write_plots(results, output_dir)
    print(json.dumps(_json_safe(aggregate), indent=2, sort_keys=True))
    print(f"[saved] {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
