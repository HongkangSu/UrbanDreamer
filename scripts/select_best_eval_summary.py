#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class CheckpointDecisionConfig:
    metric: str = "t_all"
    metric_mode: str = "min"
    min_delta: float = 0.0
    tie_rule: str = "keep_incumbent_earliest"
    non_improvement_rule: str = "abs(delta) <= min_delta"
    patience: int = 5
    patience_reset_rule: str = "strict_improvement_only"


def resolve_summary_path(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if path.is_dir():
        path = path / "collector_eval_summary.json"
    return path


def load_summary(raw: str) -> Dict[str, Any]:
    path = resolve_summary_path(raw)
    if not path.exists():
        raise FileNotFoundError(f"summary not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["_summary_path"] = str(path)
    data["_results_dir"] = str(path.parent)
    return data


def _metric_value(item: Dict[str, Any], metric: str) -> float:
    return float(item.get(metric, float("nan")))


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        fval = float(value)
    except Exception:
        return None
    if not math.isfinite(fval):
        return None
    return fval


def _delta_vs_incumbent(
    *,
    challenger_value: float,
    incumbent_value: float,
    metric_mode: str,
) -> float:
    mode = str(metric_mode).strip().lower()
    if mode == "min":
        return float(challenger_value - incumbent_value)
    if mode == "max":
        return float(incumbent_value - challenger_value)
    raise ValueError(f"unsupported metric_mode={metric_mode!r}")


def build_summary_row(item: Dict[str, Any], *, metric: str) -> Dict[str, Any]:
    value = _metric_value(item, metric)
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric {metric} in {item.get('_summary_path', '<unknown>')}")
    return {
        "metric": metric,
        "metric_value": value,
        "policy_ckpt": item.get("policy_ckpt", ""),
        "results_dir": item.get("_results_dir", ""),
        "summary_path": item.get("_summary_path", ""),
        "test_eps": int(item.get("test_eps", 0)),
        "ratio_machines": _finite_or_none(item.get("ratio_machines", float("nan"))),
        "t_CAV": _finite_or_none(item.get("t_CAV", float("nan"))),
        "t_HDV": _finite_or_none(item.get("t_HDV", float("nan"))),
        "t_all": _finite_or_none(item.get("t_all", float("nan"))),
    }


def apply_checkpoint_decision(
    *,
    incumbent: Optional[Dict[str, Any]],
    challenger: Dict[str, Any],
    config: CheckpointDecisionConfig,
    patience_before: int,
) -> Dict[str, Any]:
    challenger_row = build_summary_row(challenger, metric=config.metric)
    cfg = asdict(config)

    if incumbent is None:
        return {
            **challenger_row,
            **cfg,
            "decision": "select_best",
            "decision_reason": "initial_best",
            "challenger_checkpoint": challenger_row["policy_ckpt"],
            "incumbent_best_checkpoint": "",
            "challenger_metric_value": challenger_row["metric_value"],
            "incumbent_metric_value": None,
            "delta_vs_best": None,
            "patience_before": int(patience_before),
            "patience_after": 0,
            "selected_checkpoint_after_decision": challenger_row["policy_ckpt"],
        }

    incumbent_row = dict(incumbent)
    challenger_value = float(challenger_row["metric_value"])
    incumbent_value = float(incumbent_row["metric_value"])
    delta = _delta_vs_incumbent(
        challenger_value=challenger_value,
        incumbent_value=incumbent_value,
        metric_mode=config.metric_mode,
    )

    if delta < -float(config.min_delta):
        return {
            **challenger_row,
            **cfg,
            "decision": "select_best",
            "decision_reason": "strict_improvement",
            "challenger_checkpoint": challenger_row["policy_ckpt"],
            "incumbent_best_checkpoint": incumbent_row["policy_ckpt"],
            "challenger_metric_value": challenger_value,
            "incumbent_metric_value": incumbent_value,
            "delta_vs_best": delta,
            "patience_before": int(patience_before),
            "patience_after": 0,
            "selected_checkpoint_after_decision": challenger_row["policy_ckpt"],
        }

    if abs(delta) <= float(config.min_delta):
        selected = incumbent_row["policy_ckpt"]
        if str(config.tie_rule).strip().lower() == "keep_latest":
            selected = challenger_row["policy_ckpt"]
        return {
            **challenger_row,
            **cfg,
            "decision": "keep_best",
            "decision_reason": "tie_within_min_delta",
            "challenger_checkpoint": challenger_row["policy_ckpt"],
            "incumbent_best_checkpoint": incumbent_row["policy_ckpt"],
            "challenger_metric_value": challenger_value,
            "incumbent_metric_value": incumbent_value,
            "delta_vs_best": delta,
            "patience_before": int(patience_before),
            "patience_after": int(patience_before) + 1,
            "selected_checkpoint_after_decision": selected,
        }

    return {
        **challenger_row,
        **cfg,
        "decision": "keep_best",
        "decision_reason": "worse_than_best",
        "challenger_checkpoint": challenger_row["policy_ckpt"],
        "incumbent_best_checkpoint": incumbent_row["policy_ckpt"],
        "challenger_metric_value": challenger_value,
        "incumbent_metric_value": incumbent_value,
        "delta_vs_best": delta,
        "patience_before": int(patience_before),
        "patience_after": int(patience_before) + 1,
        "selected_checkpoint_after_decision": incumbent_row["policy_ckpt"],
    }


def rank_summaries(
    summaries: Iterable[Dict[str, Any]],
    *,
    metric: str = "t_all",
    metric_mode: str = "min",
) -> List[Dict[str, Any]]:
    rows = [build_summary_row(item, metric=metric) for item in summaries]
    reverse = str(metric_mode).strip().lower() == "max"
    rows.sort(key=lambda row: row["metric_value"], reverse=reverse)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def build_selection_payload(
    inputs: Iterable[str],
    *,
    metric: str = "t_all",
    metric_mode: str = "min",
    min_delta: float = 0.0,
    tie_rule: str = "keep_incumbent_earliest",
    patience: int = 5,
    patience_reset_rule: str = "strict_improvement_only",
) -> Dict[str, Any]:
    config = CheckpointDecisionConfig(
        metric=str(metric),
        metric_mode=str(metric_mode),
        min_delta=float(min_delta),
        tie_rule=str(tie_rule),
        non_improvement_rule="abs(delta) <= min_delta",
        patience=int(patience),
        patience_reset_rule=str(patience_reset_rule),
    )
    loaded = [load_summary(x) for x in inputs]
    ranked = rank_summaries(loaded, metric=config.metric, metric_mode=config.metric_mode)
    if not ranked:
        raise RuntimeError("no valid summaries found to rank")

    decisions: List[Dict[str, Any]] = []
    incumbent: Optional[Dict[str, Any]] = None
    patience_state = 0
    for item in loaded:
        decision = apply_checkpoint_decision(
            incumbent=incumbent,
            challenger=item,
            config=config,
            patience_before=patience_state,
        )
        decisions.append(decision)
        if decision["decision"] == "select_best":
            incumbent = build_summary_row(item, metric=config.metric)
            patience_state = 0
        else:
            patience_state = int(decision["patience_after"])

    best = ranked[0]
    return {
        "config": asdict(config),
        "best": best,
        "ranked": ranked,
        "decisions": decisions,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Select the best SUMO eval summary by metric.")
    p.add_argument("inputs", nargs="+", help="Results directories or collector_eval_summary.json paths.")
    p.add_argument("--metric", default="t_all", help="Metric to optimize. Default: t_all")
    p.add_argument("--metric-mode", default="min", choices=["min", "max"])
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--tie-rule", default="keep_incumbent_earliest", choices=["keep_incumbent_earliest", "keep_latest"])
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--patience-reset-rule", default="strict_improvement_only", choices=["strict_improvement_only"])
    p.add_argument("--output", default="", help="Optional JSON output path.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = build_selection_payload(
        args.inputs,
        metric=str(args.metric),
        metric_mode=str(args.metric_mode),
        min_delta=float(args.min_delta),
        tie_rule=str(args.tie_rule),
        patience=int(args.patience),
        patience_reset_rule=str(args.patience_reset_rule),
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
