#!/usr/bin/env python3
"""Evaluate a paired SUMO intervention against the embedded HDV predictor.

The baseline and treatment branches must start from the same post-mutation
state, HDV learning state, checkpoint, and random seed.  The treatment branch
differs only by an AV routing intervention.  Vehicle-level episode CSVs are
used to reconstruct the static HDV edge-demand target, while one-episode
collector traces provide the exact planned AV edge-pressure input used by the
response predictor.

This is a paired intervention diagnostic.  With one initial state and one
post-intervention transition it does not provide training-seed uncertainty or
a population causal estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_hdv_response_validation import (  # noqa: E402
    CaseSpec,
    _cosine,
    _distribution_metrics,
    _feature,
    _git_commit,
    _json_safe,
    _load_predictor,
    _normalise_distribution,
    _predict_batch,
    _sha256,
    _tv,
    _write_csv,
    _write_json,
)


def _load_edge_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if isinstance(raw, dict):
        for key in ("edge_ids", "edges"):
            values = raw.get(key)
            if isinstance(values, list):
                if values and isinstance(values[0], dict):
                    ids = [str(value.get("id", "")) for value in values]
                else:
                    ids = [str(value) for value in values]
                if ids and all(ids):
                    return ids
    raise RuntimeError(f"cannot read ordered edge IDs from {path}")


def _load_routes(path: Path) -> Dict[Tuple[int, int], List[List[str]]]:
    routes: Dict[Tuple[int, int], List[List[str]]] = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["origins"]), int(row["destinations"]))
            route = [edge for edge in str(row["path"]).split() if edge]
            routes.setdefault(key, []).append(route)
    if not routes:
        raise RuntimeError(f"no routes found in {path}")
    return routes


def _require_matching_artifact(
    baseline_dir: Path,
    treatment_dir: Path,
    relative_path: str,
) -> str:
    baseline_path = baseline_dir / relative_path
    treatment_path = treatment_dir / relative_path
    if not baseline_path.is_file() or not treatment_path.is_file():
        raise RuntimeError(f"missing paired control artifact: {relative_path}")
    baseline_hash = _sha256(baseline_path)
    treatment_hash = _sha256(treatment_path)
    if baseline_hash != treatment_hash:
        raise RuntimeError(f"baseline and treatment {relative_path} differ")
    return baseline_hash


def _episode_hdv_distribution(
    episode_csv: Path,
    *,
    routes: Mapping[Tuple[int, int], Sequence[Sequence[str]]],
    edge_to_index: Mapping[str, int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    counts = np.zeros((len(edge_to_index),), dtype=np.float64)
    route_counts: Dict[int, int] = {}
    n_hdv = 0
    missing_routes = 0
    missing_edges = 0
    with episode_csv.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("vType", "")).strip().lower() not in {"human", "hdv"}:
                continue
            if str(row.get("seen_in_episode", "1")).strip() in {"0", "false", "False"}:
                continue
            n_hdv += 1
            route_index = int(row.get("hdv_action", row.get("action", -1)))
            key = (int(row["origin"]), int(row["destination"]))
            candidates = routes.get(key, ())
            if route_index < 0 or route_index >= len(candidates):
                missing_routes += 1
                continue
            route_counts[route_index] = route_counts.get(route_index, 0) + 1
            for edge_id in (str(edge) for edge in candidates[route_index]):
                edge_index = edge_to_index.get(edge_id)
                if edge_index is None:
                    missing_edges += 1
                    continue
                counts[int(edge_index)] += 1.0
    if n_hdv <= 0 or float(counts.sum()) <= 0.0:
        raise RuntimeError(f"no usable HDV route demand in {episode_csv}")
    return _normalise_distribution(counts), {
        "hdv_vehicles": int(n_hdv),
        "missing_route_assignments": int(missing_routes),
        "missing_edge_references": int(missing_edges),
        "route_counts": {str(key): int(value) for key, value in sorted(route_counts.items())},
    }


def _load_av_pressure(trace_path: Path, *, episode: int, n_edges: int) -> Tuple[np.ndarray, int]:
    state = torch.load(trace_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError(f"unexpected trace payload in {trace_path}")
    episodes = torch.as_tensor(state["episode"]).reshape(-1)
    action = state.get("av_planned_edge_flow", state.get("av_edge_action"))
    if action is None:
        raise RuntimeError(f"missing planned AV edge action in {trace_path}")
    action_tensor = torch.as_tensor(action, dtype=torch.float32)
    mask = episodes.to(dtype=torch.long) == int(episode)
    if int(mask.sum().item()) <= 0:
        raise RuntimeError(f"episode {episode} absent from {trace_path}")
    selected = action_tensor[mask]
    if selected.ndim != 2 or int(selected.shape[1]) != int(n_edges):
        raise RuntimeError(
            f"AV trace shape {tuple(selected.shape)} does not match n_edges={n_edges}"
        )
    pressure = selected.mean(dim=0).cpu().numpy().astype(np.float64)
    return _normalise_distribution(pressure), int(selected.shape[0])


def _decision_summary(path: Path, *, episode: int) -> Dict[str, Any]:
    active = 0
    applied = 0
    selected = np.zeros((4,), dtype=np.int64)
    modes: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["episode"]) != int(episode):
                continue
            active += int(float(row.get("active_av", 0) or 0))
            applied += int(float(row.get("counterfactual_applied", 0) or 0))
            modes.add(str(row.get("counterfactual_mode", "")))
            for route_index in range(4):
                selected[route_index] += int(
                    float(row.get(f"selected_count_route{route_index}", 0) or 0)
                )
    selected_total = int(selected.sum())
    return {
        "active_av_decisions": int(active),
        "counterfactual_applied": int(applied),
        "counterfactual_application_fraction": (
            float(applied) / float(active) if active > 0 else None
        ),
        "counterfactual_modes": sorted(mode for mode in modes if mode),
        "selected_route_counts": {
            f"route{route_index}": int(selected[route_index])
            for route_index in range(4)
        },
        "selected_route_fractions": {
            f"route{route_index}": (
                float(selected[route_index]) / float(selected_total)
                if selected_total > 0
                else None
            )
            for route_index in range(4)
        },
    }


def _signed_effect_error(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(predicted) - np.asarray(observed)).sum())


def _top_effect_rows(
    edge_ids: Sequence[str],
    *,
    observed_effect: np.ndarray,
    predicted_effect: np.ndarray,
    baseline_before: np.ndarray,
    baseline_after: np.ndarray,
    treatment_after: np.ndarray,
    top_k: int,
) -> List[Dict[str, Any]]:
    k = min(max(int(top_k), 1), len(edge_ids))
    ranking_magnitude = np.maximum(
        np.abs(np.asarray(observed_effect)),
        np.abs(np.asarray(predicted_effect)),
    )
    indices = np.argsort(ranking_magnitude, kind="stable")[-k:][::-1]
    return [
        {
            "rank": rank,
            "ranking_abs_max_effect": float(ranking_magnitude[int(edge_index)]),
            "edge_index": int(edge_index),
            "edge_id": str(edge_ids[int(edge_index)]),
            "observed_treatment_minus_baseline": float(observed_effect[int(edge_index)]),
            "predicted_treatment_minus_baseline": float(predicted_effect[int(edge_index)]),
            "baseline_before": float(baseline_before[int(edge_index)]),
            "baseline_after": float(baseline_after[int(edge_index)]),
            "treatment_after": float(treatment_after[int(edge_index)]),
        }
        for rank, edge_index in enumerate(indices, start=1)
    ]


def evaluate_intervention(
    *,
    name: str,
    run_dir: Path,
    checkpoint: Path,
    edge_registry: Path,
    baseline_dir: Path,
    treatment_dir: Path,
    baseline_trace: Path,
    treatment_trace: Path,
    before_episode: int,
    after_episode: int,
    top_k: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    edge_ids = _load_edge_ids(edge_registry)
    edge_to_index = {edge_id: index for index, edge_id in enumerate(edge_ids)}
    baseline_routes_path = baseline_dir / "routes.csv"
    treatment_routes_path = treatment_dir / "routes.csv"
    routes = _load_routes(baseline_routes_path)
    routes_hash = _require_matching_artifact(
        baseline_dir,
        treatment_dir,
        "routes.csv",
    )
    agents_hash = _require_matching_artifact(
        baseline_dir,
        treatment_dir,
        "agents.csv",
    )
    initial_hdv_state_hash = _require_matching_artifact(
        baseline_dir,
        treatment_dir,
        "hdv_learning_state_initial_copy.json",
    )

    distributions: Dict[str, np.ndarray] = {}
    route_diagnostics: Dict[str, Any] = {}
    for branch, directory in (("baseline", baseline_dir), ("treatment", treatment_dir)):
        for label, episode in (("before", before_episode), ("after", after_episode)):
            key = f"{branch}_{label}"
            distribution, diagnostic = _episode_hdv_distribution(
                directory / "episodes" / f"ep{int(episode)}.csv",
                routes=routes,
                edge_to_index=edge_to_index,
            )
            distributions[key] = distribution
            route_diagnostics[key] = diagnostic

    baseline_av, baseline_steps = _load_av_pressure(
        baseline_trace, episode=before_episode, n_edges=len(edge_ids)
    )
    treatment_av, treatment_steps = _load_av_pressure(
        treatment_trace, episode=before_episode, n_edges=len(edge_ids)
    )

    spec = CaseSpec(name=name, run_dir=run_dir, checkpoint=checkpoint)
    model, checkpoint_meta = _load_predictor(spec)
    if int(checkpoint_meta["n_edges"]) != len(edge_ids):
        raise RuntimeError("checkpoint and edge registry sizes differ")
    progress_denominator = int(checkpoint_meta["progress_denominator"])
    features = np.stack(
        [
            _feature(
                baseline_av,
                distributions["baseline_before"],
                int(after_episode),
                progress_denominator,
            ),
            _feature(
                treatment_av,
                distributions["treatment_before"],
                int(after_episode),
                progress_denominator,
            ),
        ],
        axis=0,
    )
    baseline_prediction, treatment_prediction = _predict_batch(model, features)

    observed_effect = distributions["treatment_after"] - distributions["baseline_after"]
    predicted_effect = treatment_prediction - baseline_prediction
    baseline_observed_response = (
        distributions["baseline_after"] - distributions["baseline_before"]
    )
    treatment_observed_response = (
        distributions["treatment_after"] - distributions["treatment_before"]
    )
    baseline_predicted_response = (
        baseline_prediction - distributions["baseline_before"]
    )
    treatment_predicted_response = (
        treatment_prediction - distributions["treatment_before"]
    )
    av_effect = treatment_av - baseline_av
    baseline_metrics = _distribution_metrics(
        distributions["baseline_after"],
        baseline_prediction,
        top_k=top_k,
    )
    baseline_persistence_metrics = _distribution_metrics(
        distributions["baseline_after"],
        distributions["baseline_before"],
        top_k=top_k,
    )
    treatment_metrics = _distribution_metrics(
        distributions["treatment_after"],
        treatment_prediction,
        top_k=top_k,
    )
    treatment_persistence_metrics = _distribution_metrics(
        distributions["treatment_after"],
        distributions["treatment_before"],
        top_k=top_k,
    )

    nonzero_effect = np.flatnonzero(np.abs(observed_effect) > 1e-12)
    top_sign_agreement = None
    if int(nonzero_effect.size) > 0:
        strongest = nonzero_effect[
            np.argsort(np.abs(observed_effect[nonzero_effect]), kind="stable")[
                -min(int(top_k), int(nonzero_effect.size)) :
            ]
        ]
        top_sign_agreement = float(
            np.mean(
                np.sign(predicted_effect[strongest])
                == np.sign(observed_effect[strongest])
            )
        )

    summary: Dict[str, Any] = {
        "schema_version": "hdv_response_paired_intervention_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "case": name,
        "protocol": {
            "design": "paired persistent AV-route intervention",
            "before_episode": int(before_episode),
            "after_episode": int(after_episode),
            "branch_controls": (
                "same checkpoint, seed, post-mutation state, initial HDV learning "
                "state, demand, and route artifacts"
            ),
            "human_learning_after_mutation": True,
            "unit": "one paired post-intervention episode transition",
        },
        "provenance": {
            "run_dir": str(run_dir.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_episode": int(checkpoint_meta["checkpoint_episode"]),
            "checkpoint_trained_pairs": int(checkpoint_meta["trained_pairs"]),
            "baseline_dir": str(baseline_dir.resolve()),
            "treatment_dir": str(treatment_dir.resolve()),
            "baseline_trace": str(baseline_trace.resolve()),
            "baseline_trace_sha256": _sha256(baseline_trace),
            "treatment_trace": str(treatment_trace.resolve()),
            "treatment_trace_sha256": _sha256(treatment_trace),
            "edge_registry": str(edge_registry.resolve()),
            "edge_registry_sha256": _sha256(edge_registry),
            "routes_sha256": routes_hash,
            "agents_sha256": agents_hash,
            "initial_hdv_state_sha256": initial_hdv_state_hash,
            "n_edges": int(len(edge_ids)),
            "progress_denominator": progress_denominator,
        },
        "balance_checks": {
            "paired_agents_identical": True,
            "paired_route_table_identical": True,
            "paired_initial_hdv_state_identical": True,
            "initial_hdv_branch_tv": _tv(
                distributions["baseline_before"],
                distributions["treatment_before"],
            ),
            "baseline_trace_steps": int(baseline_steps),
            "treatment_trace_steps": int(treatment_steps),
            "route_diagnostics": route_diagnostics,
        },
        "intervention": {
            "av_pressure_tv": _tv(baseline_av, treatment_av),
            "baseline_decisions": _decision_summary(
                baseline_dir / "dynamic_av_decisions.csv",
                episode=before_episode,
            ),
            "treatment_decisions": _decision_summary(
                treatment_dir / "dynamic_av_decisions.csv",
                episode=before_episode,
            ),
        },
        "observed_response": {
            "baseline_episode_to_episode_tv": _tv(
                distributions["baseline_after"],
                distributions["baseline_before"],
            ),
            "treatment_episode_to_episode_tv": _tv(
                distributions["treatment_after"],
                distributions["treatment_before"],
            ),
            "post_intervention_branch_tv": _tv(
                distributions["treatment_after"],
                distributions["baseline_after"],
            ),
            "post_intervention_changed_edges": int(nonzero_effect.size),
        },
        "predictor_accuracy": {
            "baseline": baseline_metrics,
            "baseline_persistence": baseline_persistence_metrics,
            "baseline_predictor_minus_persistence_tv": float(
                baseline_metrics["tv"] - baseline_persistence_metrics["tv"]
            ),
            "treatment": treatment_metrics,
            "treatment_persistence": treatment_persistence_metrics,
            "treatment_predictor_minus_persistence_tv": float(
                treatment_metrics["tv"] - treatment_persistence_metrics["tv"]
            ),
        },
        "predicted_vs_observed_response": {
            "baseline_predicted_response_tv": _tv(
                baseline_prediction,
                distributions["baseline_before"],
            ),
            "baseline_direction_cosine": _cosine(
                baseline_predicted_response,
                baseline_observed_response,
            ),
            "treatment_predicted_response_tv": _tv(
                treatment_prediction,
                distributions["treatment_before"],
            ),
            "treatment_direction_cosine": _cosine(
                treatment_predicted_response,
                treatment_observed_response,
            ),
        },
        "counterfactual_contrast": {
            "predicted_branch_tv": _tv(
                treatment_prediction,
                baseline_prediction,
            ),
            "observed_branch_tv": _tv(
                distributions["treatment_after"],
                distributions["baseline_after"],
            ),
            "signed_effect_half_l1_error": _signed_effect_error(
                predicted_effect,
                observed_effect,
            ),
            "effect_direction_cosine": _cosine(predicted_effect, observed_effect),
            "top_effect_sign_agreement": top_sign_agreement,
            "predicted_avoidance_alignment": -_cosine(av_effect, predicted_effect),
            "observed_avoidance_alignment": -_cosine(av_effect, observed_effect),
        },
        "interpretation_boundary": (
            "This paired smoke verifies branch balance and intervention application, "
            "then measures one-step realized HDV response and predictor sensitivity "
            "for one seed and one initial state. It is not a multi-seed causal effect "
            "estimate and carries no inferential confidence interval."
        ),
    }
    edge_rows = _top_effect_rows(
        edge_ids,
        observed_effect=observed_effect,
        predicted_effect=predicted_effect,
        baseline_before=distributions["baseline_before"],
        baseline_after=distributions["baseline_after"],
        treatment_after=distributions["treatment_after"],
        top_k=top_k,
    )
    return summary, edge_rows


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="Ingolstadt")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--edge-registry", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--treatment-dir", required=True)
    parser.add_argument("--baseline-trace", required=True)
    parser.add_argument("--treatment-trace", required=True)
    parser.add_argument("--before-episode", type=int, required=True)
    parser.add_argument("--after-episode", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary, edge_rows = evaluate_intervention(
        name=str(args.case),
        run_dir=Path(args.run_dir).resolve(),
        checkpoint=Path(args.checkpoint).resolve(),
        edge_registry=Path(args.edge_registry).resolve(),
        baseline_dir=Path(args.baseline_dir).resolve(),
        treatment_dir=Path(args.treatment_dir).resolve(),
        baseline_trace=Path(args.baseline_trace).resolve(),
        treatment_trace=Path(args.treatment_trace).resolve(),
        before_episode=int(args.before_episode),
        after_episode=int(args.after_episode),
        top_k=int(args.top_k),
    )
    output_dir = Path(args.output_dir).resolve()
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "top_edge_contrasts.csv", edge_rows)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    print(f"[saved] {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
