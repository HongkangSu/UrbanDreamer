#!/usr/bin/env python3
"""Visualize dynamic current-to-destination route choices from SUMO eval logs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize


ROOT_TAG = "ingolstadt_custom_current_to_dest_incident_20260528"
DEFAULT_RUNS = {
    "policy_ep400": Path(
        f"results/{ROOT_TAG}/eval_ep400_current_shortest_freeflow_exact"
    ),
    "freeflow_greedy": Path(
        f"results/{ROOT_TAG}/baseline_ep400_ref_freeflow_candidates_freeflow_greedy"
    ),
    "live_current_greedy": Path(
        f"results/{ROOT_TAG}/baseline_ep400_ref_freeflow_candidates_live_current_greedy"
    ),
}


def _parse_shape(shape: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for token in str(shape or "").split():
        if "," not in token:
            continue
        x_s, y_s = token.split(",", 1)
        try:
            pts.append((float(x_s), float(y_s)))
        except ValueError:
            continue
    return pts


def load_edge_shapes(net_xml: Path) -> Dict[str, List[Tuple[float, float]]]:
    root = ET.parse(net_xml).getroot()
    shapes: Dict[str, List[Tuple[float, float]]] = {}
    for edge in root.findall("edge"):
        edge_id = str(edge.get("id") or "")
        if not edge_id or edge_id.startswith(":"):
            continue
        lane_shape: List[Tuple[float, float]] = []
        for lane in edge.findall("lane"):
            lane_shape = _parse_shape(str(lane.get("shape") or ""))
            if lane_shape:
                break
        if not lane_shape:
            lane_shape = _parse_shape(str(edge.get("shape") or ""))
        if len(lane_shape) >= 2:
            shapes[edge_id] = lane_shape
    return shapes


def _parse_path_edges(value: str) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [x for x in text.split() if x]


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except Exception:
        return default


def load_decision_rows(run_dir: Path) -> List[dict]:
    decisions_path = run_dir / "dynamic_av_agent_decisions.csv"
    if not decisions_path.exists():
        raise FileNotFoundError(f"Missing {decisions_path}")
    rows: List[dict] = []
    with decisions_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            selected = _safe_int(row.get("selected_route"), 0)
            row["_selected_route"] = selected
            row["_decision_idx"] = _safe_int(row.get("decision_idx"), 0)
            row["_veh_id"] = str(row.get("veh_id") or "")
            row["_selected_path_edges"] = _parse_path_edges(
                row.get(f"path_edges_route{selected}", "")
            )
            rows.append(row)
    return rows


def edge_usage_from_rows(rows: Sequence[dict]) -> Counter:
    usage: Counter = Counter()
    for row in rows:
        for edge_id in row.get("_selected_path_edges", []):
            usage[str(edge_id)] += 1
    return usage


def route_counts_by_tick(rows: Sequence[dict], n_routes: int = 4) -> Dict[int, List[int]]:
    by_tick: Dict[int, List[int]] = defaultdict(lambda: [0] * n_routes)
    for row in rows:
        tick = int(row.get("_decision_idx", 0))
        route = int(row.get("_selected_route", 0))
        if 0 <= route < n_routes:
            by_tick[tick][route] += 1
    return dict(sorted(by_tick.items()))


def summarize_run(run_name: str, run_dir: Path, rows: Sequence[dict], usage: Counter) -> dict:
    summary_path = run_dir / "collector_eval_summary.json"
    eval_summary = {}
    if summary_path.exists():
        eval_summary = json.loads(summary_path.read_text())
    selected_counts = Counter(int(row.get("_selected_route", 0)) for row in rows)
    nonzero_rows = [row for row in rows if int(row.get("_selected_route", 0)) != 0]
    top_edges = usage.most_common(20)
    return {
        "run": run_name,
        "run_dir": str(run_dir),
        "t_all": eval_summary.get("t_all"),
        "t_CAV": eval_summary.get("t_CAV"),
        "t_HDV": eval_summary.get("t_HDV"),
        "completion_rate": eval_summary.get("completion_rate"),
        "av_candidate_source": eval_summary.get("av_candidate_source"),
        "av_route_cost_policy": eval_summary.get("av_route_cost_policy"),
        "wm_candidate_edge_cost_source": eval_summary.get("wm_candidate_edge_cost_source"),
        "decision_rows": len(rows),
        "unique_vehicles": len({str(row.get("_veh_id", "")) for row in rows}),
        "selected_route_counts": {f"route{k}": int(selected_counts.get(k, 0)) for k in range(4)},
        "nonzero_decision_rows": len(nonzero_rows),
        "top_selected_edges": [{"edge": edge, "count": int(count)} for edge, count in top_edges],
    }


def _build_segments(
    edge_shapes: Mapping[str, Sequence[Tuple[float, float]]],
    values: Mapping[str, float],
    *,
    min_abs: float = 0.0,
) -> Tuple[List[List[Tuple[float, float]]], List[float]]:
    segments: List[List[Tuple[float, float]]] = []
    weights: List[float] = []
    for edge_id, value in values.items():
        val = float(value)
        if abs(val) < float(min_abs):
            continue
        shape = edge_shapes.get(str(edge_id))
        if shape and len(shape) >= 2:
            segments.append(list(shape))
            weights.append(val)
    return segments, weights


def _plot_base_map(ax: plt.Axes, edge_shapes: Mapping[str, Sequence[Tuple[float, float]]]) -> None:
    base_segments = [list(shape) for shape in edge_shapes.values() if len(shape) >= 2]
    base = LineCollection(base_segments, colors="#d7d7d7", linewidths=0.25, alpha=0.45)
    ax.add_collection(base)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def plot_usage_map(
    *,
    edge_shapes: Mapping[str, Sequence[Tuple[float, float]]],
    usage: Mapping[str, int],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 10), dpi=180)
    _plot_base_map(ax, edge_shapes)
    segments, weights = _build_segments(edge_shapes, {k: float(v) for k, v in usage.items()})
    if segments:
        max_w = max(weights)
        norm = Normalize(vmin=1.0, vmax=max(1.0, max_w))
        lc = LineCollection(
            segments,
            cmap="magma",
            norm=norm,
            linewidths=[0.35 + 2.6 * (math.sqrt(w) / math.sqrt(max_w)) for w in weights],
            alpha=0.92,
        )
        lc.set_array(weights)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.030, pad=0.01)
        cbar.set_label("selected path edge count")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_diff_map(
    *,
    edge_shapes: Mapping[str, Sequence[Tuple[float, float]]],
    left_usage: Mapping[str, int],
    right_usage: Mapping[str, int],
    title: str,
    out_path: Path,
    min_abs: float = 2.0,
) -> None:
    all_edges = set(left_usage) | set(right_usage)
    diff = {edge: float(left_usage.get(edge, 0) - right_usage.get(edge, 0)) for edge in all_edges}
    fig, ax = plt.subplots(figsize=(13, 10), dpi=180)
    _plot_base_map(ax, edge_shapes)
    segments, weights = _build_segments(edge_shapes, diff, min_abs=min_abs)
    if segments:
        max_abs = max(abs(w) for w in weights)
        cmap = LinearSegmentedColormap.from_list("policy_diff", ["#2563eb", "#f7f7f7", "#dc2626"])
        norm = Normalize(vmin=-max_abs, vmax=max_abs)
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=[0.35 + 2.4 * (math.sqrt(abs(w)) / math.sqrt(max_abs)) for w in weights],
            alpha=0.95,
        )
        lc.set_array(weights)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.030, pad=0.01)
        cbar.set_label("edge count difference: first minus second")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_route_timeline(
    *,
    per_run_counts: Mapping[str, Mapping[int, Sequence[int]]],
    out_path: Path,
) -> None:
    colors = ["#525252", "#2563eb", "#16a34a", "#dc2626"]
    fig, axes = plt.subplots(
        len(per_run_counts),
        1,
        figsize=(13, max(3, 2.8 * len(per_run_counts))),
        dpi=180,
        sharex=True,
    )
    if len(per_run_counts) == 1:
        axes = [axes]
    for ax, (run_name, counts_by_tick) in zip(axes, per_run_counts.items()):
        ticks = sorted(counts_by_tick)
        bottoms = [0] * len(ticks)
        for route_idx in range(4):
            vals = [int(counts_by_tick[t][route_idx]) for t in ticks]
            ax.bar(
                ticks,
                vals,
                bottom=bottoms,
                color=colors[route_idx],
                width=0.9,
                label=f"route{route_idx}",
                linewidth=0,
            )
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_ylabel(run_name)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(bottoms) * 1.08 if bottoms else 1)
    axes[0].legend(ncol=4, loc="upper right", frameon=False)
    axes[-1].set_xlabel("decision_idx")
    fig.suptitle("Selected Route Slot Counts By Decision Tick", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_nonzero_examples(
    *,
    rows_by_run: Mapping[str, Sequence[dict]],
    out_path: Path,
    max_rows: int,
) -> None:
    fields = [
        "run",
        "episode",
        "decision_idx",
        "sim_step",
        "veh_id",
        "current_edge",
        "dest_edge",
        "selected_route",
        "wm_best_route",
        "live_best_route",
        "prob_selected",
        "wm_cost_selected",
        "live_cost_selected",
        "selected_path_edges",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run_name, rows in rows_by_run.items():
            written = 0
            for row in rows:
                selected = int(row.get("_selected_route", 0))
                if selected == 0:
                    continue
                writer.writerow(
                    {
                        "run": run_name,
                        "episode": row.get("episode", ""),
                        "decision_idx": row.get("decision_idx", ""),
                        "sim_step": row.get("sim_step", ""),
                        "veh_id": row.get("veh_id", ""),
                        "current_edge": row.get("current_edge", ""),
                        "dest_edge": row.get("dest_edge", ""),
                        "selected_route": selected,
                        "wm_best_route": row.get("wm_best_route", ""),
                        "live_best_route": row.get("live_best_route", ""),
                        "prob_selected": _safe_float(row.get(f"prob_route{selected}", "")),
                        "wm_cost_selected": _safe_float(row.get(f"wm_cost_route{selected}", "")),
                        "live_cost_selected": _safe_float(row.get(f"live_cost_route{selected}", "")),
                        "selected_path_edges": " ".join(row.get("_selected_path_edges", [])),
                    }
                )
                written += 1
                if written >= int(max_rows):
                    break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--net",
        type=Path,
        default=Path("networks/ingolstadt_custom/ingolstadt_custom.net.xml"),
    )
    parser.add_argument("--out", type=Path, default=Path("visualization/current_to_dest_choices"))
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Optional NAME=RUN_DIR override. May be repeated.",
    )
    parser.add_argument("--max-example-rows", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs: MutableMapping[str, Path] = dict(DEFAULT_RUNS)
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"--run must be NAME=RUN_DIR, got {item!r}")
        name, path = item.split("=", 1)
        runs[str(name)] = Path(path)

    args.out.mkdir(parents=True, exist_ok=True)
    edge_shapes = load_edge_shapes(args.net)

    rows_by_run: Dict[str, List[dict]] = {}
    usage_by_run: Dict[str, Counter] = {}
    route_timeline_by_run: Dict[str, Dict[int, List[int]]] = {}
    summaries: List[dict] = []

    for run_name, run_dir in runs.items():
        rows = load_decision_rows(run_dir)
        usage = edge_usage_from_rows(rows)
        rows_by_run[run_name] = rows
        usage_by_run[run_name] = usage
        route_timeline_by_run[run_name] = route_counts_by_tick(rows)
        summaries.append(summarize_run(run_name, run_dir, rows, usage))
        plot_usage_map(
            edge_shapes=edge_shapes,
            usage=usage,
            title=f"{run_name}: selected dynamic route paths",
            out_path=args.out / f"{run_name}_selected_edge_usage.png",
        )

    if "policy_ep400" in usage_by_run and "freeflow_greedy" in usage_by_run:
        plot_diff_map(
            edge_shapes=edge_shapes,
            left_usage=usage_by_run["policy_ep400"],
            right_usage=usage_by_run["freeflow_greedy"],
            title="policy_ep400 minus freeflow_greedy selected edge usage",
            out_path=args.out / "policy_ep400_minus_freeflow_greedy.png",
        )
    if "policy_ep400" in usage_by_run and "live_current_greedy" in usage_by_run:
        plot_diff_map(
            edge_shapes=edge_shapes,
            left_usage=usage_by_run["policy_ep400"],
            right_usage=usage_by_run["live_current_greedy"],
            title="policy_ep400 minus live_current_greedy selected edge usage",
            out_path=args.out / "policy_ep400_minus_live_current_greedy.png",
        )
    if "live_current_greedy" in usage_by_run and "freeflow_greedy" in usage_by_run:
        plot_diff_map(
            edge_shapes=edge_shapes,
            left_usage=usage_by_run["live_current_greedy"],
            right_usage=usage_by_run["freeflow_greedy"],
            title="live_current_greedy minus freeflow_greedy selected edge usage",
            out_path=args.out / "live_current_minus_freeflow_greedy.png",
        )

    plot_route_timeline(
        per_run_counts=route_timeline_by_run,
        out_path=args.out / "selected_route_counts_by_tick.png",
    )
    write_nonzero_examples(
        rows_by_run=rows_by_run,
        out_path=args.out / "nonzero_route_examples.csv",
        max_rows=int(args.max_example_rows),
    )
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))

    report_lines = [
        "# Current-to-Destination Route Choice Visualization",
        "",
        "All three runs use free-flow generated current-to-destination candidate paths.",
        "The maps color selected candidate path edge usage, not all live vehicle trajectories.",
        "",
        "## Runs",
        "",
    ]
    for item in summaries:
        counts = item["selected_route_counts"]
        report_lines.append(
            f"- `{item['run']}`: t_all={item['t_all']:.4f}, "
            f"rows={item['decision_rows']}, routes={counts}"
        )
    report_lines.extend(
        [
            "",
            "## Files",
            "",
            "- `*_selected_edge_usage.png`: selected path usage heatmap per run.",
            "- `policy_ep400_minus_freeflow_greedy.png`: red means more policy usage; blue means more free-flow greedy usage.",
            "- `policy_ep400_minus_live_current_greedy.png`: red means more policy usage; blue means more live-current greedy usage.",
            "- `live_current_minus_freeflow_greedy.png`: red means more live-current greedy usage; blue means more free-flow greedy usage.",
            "- `selected_route_counts_by_tick.png`: stacked selected route slot counts over decision ticks.",
            "- `nonzero_route_examples.csv`: sample non-route0 decisions with selected path edges.",
            "- `summary.json`: machine-readable metrics and top selected edges.",
            "",
        ]
    )
    (args.out / "README.md").write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
