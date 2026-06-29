#!/usr/bin/env python3
"""Diagnose WM route-cost predictions on active vehicle decision traces.

The input is a ``dynamic_av_agent_decisions.csv`` file written by
``scripts/eval_sumo.py``.  Each row is one active AV at one decision tick and
contains candidate route paths plus WM/live route-cost estimates.

This diagnostic intentionally answers a narrow question: whether the world
model's current active-vehicle candidate route cost ranking agrees with the
live/SUMO-derived route-cost ranking recorded at the same decision.  It does
not claim to validate full multi-step arrival-time prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    mu = sum(vals) / len(vals)
    return float(math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    mx = sum(x_vals) / len(x_vals)
    my = sum(y_vals) / len(y_vals)
    vx = sum((x - mx) ** 2 for x in x_vals)
    vy = sum((y - my) ** 2 for y in y_vals)
    if vx <= 1e-12 or vy <= 1e-12:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return float(cov / math.sqrt(vx * vy))


def _rank(values: Sequence[float]) -> List[float]:
    indexed = sorted((float(v), i) for i, v in enumerate(values))
    ranks = [0.0 for _ in values]
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and abs(indexed[end][0] - indexed[pos][0]) <= 1e-9:
            end += 1
        avg_rank = (pos + end - 1) / 2.0
        for _, idx in indexed[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    return _pearson(_rank([p[0] for p in pairs]), _rank([p[1] for p in pairs]))


def _kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    n = len(pairs)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pairs[i][0] - pairs[j][0]
            dy = pairs[i][1] - pairs[j][1]
            if abs(dx) <= 1e-9 or abs(dy) <= 1e-9:
                continue
            product = dx * dy
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    denom = concordant + discordant
    return float((concordant - discordant) / denom) if denom else float("nan")


def _parse_path(raw: str) -> Tuple[str, ...]:
    text = str(raw or "").strip()
    if not text:
        return tuple()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return tuple()
    if not isinstance(parsed, list):
        return tuple()
    return tuple(str(x) for x in parsed)


def _best_index(costs: Sequence[float]) -> Optional[int]:
    finite = [(idx, float(v)) for idx, v in enumerate(costs) if math.isfinite(float(v))]
    if not finite:
        return None
    return min(finite, key=lambda item: (item[1], item[0]))[0]


def _best_path(costs: Sequence[float], paths: Sequence[Tuple[str, ...]]) -> Optional[Tuple[str, ...]]:
    idx = _best_index(costs)
    if idx is None or idx >= len(paths):
        return None
    return paths[idx]


def _topk_indices(costs: Sequence[float], k: int) -> List[int]:
    finite = [(idx, float(v)) for idx, v in enumerate(costs) if math.isfinite(float(v))]
    finite.sort(key=lambda item: (item[1], item[0]))
    return [idx for idx, _ in finite[: max(int(k), 0)]]


def _topk_paths(costs: Sequence[float], paths: Sequence[Tuple[str, ...]], k: int) -> set[Tuple[str, ...]]:
    return {paths[idx] for idx in _topk_indices(costs, k) if idx < len(paths)}


@dataclass
class RowDiagnostic:
    episode: int
    decision_idx: int
    sim_step: int
    veh_id: str
    current_edge: str
    dest_edge: str
    selected_route: int
    wm_best_route: int
    live_best_route: int
    wm_live_best_match: int
    wm_live_best_path_match: int
    top2_path_overlap: int
    n_valid_routes: int
    n_unique_paths: int
    wm_live_mae: float
    wm_live_rmse: float
    wm_live_bias: float
    wm_live_pearson: float
    wm_live_spearman: float
    wm_live_kendall: float
    selected_live_regret: float
    selected_wm_regret: float
    wm_best_live_regret: float
    live_best_gap: float
    wm_best_gap: float
    route0_eq_route3: int

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _diagnose_row(row: Dict[str, str], n_routes: int) -> Optional[RowDiagnostic]:
    wm_costs = [_safe_float(row.get(f"wm_cost_route{k}")) for k in range(n_routes)]
    live_costs = [_safe_float(row.get(f"live_cost_route{k}")) for k in range(n_routes)]
    paths = [_parse_path(row.get(f"path_edges_route{k}", "")) for k in range(n_routes)]
    valid = [k for k in range(n_routes) if math.isfinite(wm_costs[k]) and math.isfinite(live_costs[k]) and paths[k]]
    if not valid:
        return None
    wm_valid = [wm_costs[k] for k in valid]
    live_valid = [live_costs[k] for k in valid]
    errors = [wm_costs[k] - live_costs[k] for k in valid]
    wm_best = _best_index(wm_costs)
    live_best = _best_index(live_costs)
    wm_best_path = _best_path(wm_costs, paths)
    live_best_path = _best_path(live_costs, paths)
    selected = _safe_int(row.get("selected_route"))
    live_best_cost = min(live_valid)
    wm_best_cost = min(wm_valid)
    selected_live_regret = (
        live_costs[selected] - live_best_cost
        if 0 <= selected < n_routes and math.isfinite(live_costs[selected])
        else float("nan")
    )
    selected_wm_regret = (
        wm_costs[selected] - wm_best_cost
        if 0 <= selected < n_routes and math.isfinite(wm_costs[selected])
        else float("nan")
    )
    wm_best_live_regret = (
        live_costs[wm_best] - live_best_cost
        if wm_best is not None and math.isfinite(live_costs[wm_best])
        else float("nan")
    )
    live_order = sorted(live_valid)
    wm_order = sorted(wm_valid)
    return RowDiagnostic(
        episode=_safe_int(row.get("episode")),
        decision_idx=_safe_int(row.get("decision_idx")),
        sim_step=_safe_int(row.get("sim_step")),
        veh_id=str(row.get("veh_id", "")),
        current_edge=str(row.get("current_edge", "")),
        dest_edge=str(row.get("dest_edge", "")),
        selected_route=selected,
        wm_best_route=int(wm_best) if wm_best is not None else -1,
        live_best_route=int(live_best) if live_best is not None else -1,
        wm_live_best_match=int(wm_best is not None and live_best is not None and wm_best == live_best),
        wm_live_best_path_match=int(wm_best_path is not None and live_best_path is not None and wm_best_path == live_best_path),
        top2_path_overlap=len(_topk_paths(wm_costs, paths, 2) & _topk_paths(live_costs, paths, 2)),
        n_valid_routes=len(valid),
        n_unique_paths=len(set(paths[k] for k in valid)),
        wm_live_mae=_mean(abs(e) for e in errors),
        wm_live_rmse=math.sqrt(_mean(e * e for e in errors)),
        wm_live_bias=_mean(errors),
        wm_live_pearson=_pearson(wm_valid, live_valid),
        wm_live_spearman=_spearman(wm_valid, live_valid),
        wm_live_kendall=_kendall_tau(wm_valid, live_valid),
        selected_live_regret=float(selected_live_regret),
        selected_wm_regret=float(selected_wm_regret),
        wm_best_live_regret=float(wm_best_live_regret),
        live_best_gap=(live_order[1] - live_order[0]) if len(live_order) >= 2 else float("nan"),
        wm_best_gap=(wm_order[1] - wm_order[0]) if len(wm_order) >= 2 else float("nan"),
        route0_eq_route3=int(n_routes > 3 and paths[0] and paths[0] == paths[3]),
    )


def diagnose(input_csv: Path, n_routes: int) -> Tuple[List[RowDiagnostic], Dict[str, Any]]:
    with input_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    diagnostics = [diag for row in rows if (diag := _diagnose_row(row, n_routes)) is not None]
    total = len(diagnostics)
    summary: Dict[str, Any] = {
        "input_csv": str(input_csv),
        "n_routes": int(n_routes),
        "rows_total": len(rows),
        "rows_with_valid_costs": total,
    }
    if not diagnostics:
        return diagnostics, summary

    def vals(name: str) -> List[float]:
        return [float(getattr(d, name)) for d in diagnostics]

    def rate(name: str) -> float:
        return _mean(vals(name))

    summary.update(
        {
            "wm_live_best_match_rate_index": rate("wm_live_best_match"),
            "wm_live_best_match_rate_path": rate("wm_live_best_path_match"),
            "top2_path_overlap_mean": _mean(vals("top2_path_overlap")),
            "n_unique_paths_mean": _mean(vals("n_unique_paths")),
            "route0_eq_route3_frac": rate("route0_eq_route3"),
            "wm_live_mae_mean": _mean(vals("wm_live_mae")),
            "wm_live_mae_std": _std(vals("wm_live_mae")),
            "wm_live_rmse_mean": _mean(vals("wm_live_rmse")),
            "wm_live_bias_mean": _mean(vals("wm_live_bias")),
            "wm_live_pearson_mean": _mean(vals("wm_live_pearson")),
            "wm_live_spearman_mean": _mean(vals("wm_live_spearman")),
            "wm_live_kendall_mean": _mean(vals("wm_live_kendall")),
            "selected_live_regret_mean": _mean(vals("selected_live_regret")),
            "selected_wm_regret_mean": _mean(vals("selected_wm_regret")),
            "wm_best_live_regret_mean": _mean(vals("wm_best_live_regret")),
            "live_best_gap_mean": _mean(vals("live_best_gap")),
            "wm_best_gap_mean": _mean(vals("wm_best_gap")),
        }
    )
    by_edge: Dict[str, List[RowDiagnostic]] = {}
    for diag in diagnostics:
        by_edge.setdefault(diag.current_edge, []).append(diag)
    summary["by_current_edge"] = {
        edge: {
            "rows": len(items),
            "wm_live_best_match_rate_path": _mean(d.wm_live_best_path_match for d in items),
            "wm_live_mae_mean": _mean(d.wm_live_mae for d in items),
            "wm_best_live_regret_mean": _mean(d.wm_best_live_regret for d in items),
            "selected_live_regret_mean": _mean(d.selected_live_regret for d in items),
        }
        for edge, items in sorted(by_edge.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    }
    return diagnostics, summary


def _write_markdown(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Active Vehicle WM Cost Diagnostic",
        "",
        f"- Input: `{summary.get('input_csv', '')}`",
        f"- Rows with valid route costs: {summary.get('rows_with_valid_costs', 0)} / {summary.get('rows_total', 0)}",
        f"- Index top-1 match: {summary.get('wm_live_best_match_rate_index', float('nan')):.4f}",
        f"- Path-equivalent top-1 match: {summary.get('wm_live_best_match_rate_path', float('nan')):.4f}",
        f"- Top-2 path overlap mean: {summary.get('top2_path_overlap_mean', float('nan')):.4f}",
        f"- WM/live MAE mean: {summary.get('wm_live_mae_mean', float('nan')):.4f}",
        f"- WM best live regret mean: {summary.get('wm_best_live_regret_mean', float('nan')):.4f}",
        f"- Selected live regret mean: {summary.get('selected_live_regret_mean', float('nan')):.4f}",
        f"- Spearman mean: {summary.get('wm_live_spearman_mean', float('nan')):.4f}",
        f"- Kendall mean: {summary.get('wm_live_kendall_mean', float('nan')):.4f}",
        "",
        "## By Current Edge",
        "",
        "| edge | rows | path top-1 | MAE | wm-best live regret | selected live regret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for edge, row in summary.get("by_current_edge", {}).items():
        lines.append(
            "| {edge} | {rows} | {top1:.4f} | {mae:.4f} | {wm_reg:.4f} | {sel_reg:.4f} |".format(
                edge=edge,
                rows=int(row.get("rows", 0)),
                top1=float(row.get("wm_live_best_match_rate_path", float("nan"))),
                mae=float(row.get("wm_live_mae_mean", float("nan"))),
                wm_reg=float(row.get("wm_best_live_regret_mean", float("nan"))),
                sel_reg=float(row.get("selected_live_regret_mean", float("nan"))),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--n-routes", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_csv = Path(args.input_csv)
    diagnostics, summary = diagnose(input_csv, int(args.n_routes))
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(RowDiagnostic.__annotations__.keys()))
            writer.writeheader()
            for row in diagnostics:
                writer.writerow(row.to_dict())
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(summary, out_md)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
