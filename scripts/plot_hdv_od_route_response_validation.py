#!/usr/bin/env python3
"""Plot observed and OD-route-predicted episode-level HDV responses."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _parse_case(raw: str) -> Tuple[str, Path]:
    label, separator, path = str(raw).partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "--case must use LABEL=/path/to/per_episode_metrics.csv"
        )
    return label.strip(), Path(path).expanduser().resolve()


def _load_curve(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"no per-episode rows in {path}")

    def values(name: str) -> np.ndarray:
        try:
            return np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        except KeyError as exc:
            raise RuntimeError(f"{path} has no column {name!r}") from exc

    return {
        "episode": values("episode"),
        # Persistence error is exactly the magnitude of the observed change
        # H_e - H_{e-1}.
        "observed": values("persistence_edge_tv"),
        "gawron": values("gawron_od_route_predicted_response_tv"),
        "selected": values("selected_od_route_predictor_predicted_response_tv"),
    }


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    width = max(1, min(int(window), int(arr.size)))
    numerator = np.convolve(arr, np.ones(width, dtype=np.float64), mode="full")
    denominator = np.convolve(
        np.ones_like(arr),
        np.ones(width, dtype=np.float64),
        mode="full",
    )
    return (numerator / denominator)[: arr.size]


def plot_cases(
    cases: Sequence[Tuple[str, Path]],
    *,
    output_dir: Path,
    window: int,
) -> Tuple[Path, Path]:
    if not cases:
        raise ValueError("at least one --case is required")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        1,
        len(cases),
        figsize=(7.2, 2.35),
        squeeze=False,
    )
    observed_color = "#0077BB"
    gawron_color = "#CC3311"
    selected_color = "#009988"
    for axis, (label, path) in zip(axes[0], cases):
        curve = _load_curve(path)
        episode = curve["episode"]
        observed = 1e3 * curve["observed"]
        gawron = 1e3 * curve["gawron"]
        selected = 1e3 * curve["selected"]
        axis.plot(
            episode,
            observed,
            color=observed_color,
            linewidth=0.55,
            alpha=0.28,
        )
        axis.plot(
            episode,
            gawron,
            color=gawron_color,
            linewidth=0.55,
            linestyle="--",
            alpha=0.22,
        )
        axis.plot(
            episode,
            selected,
            color=selected_color,
            linewidth=0.55,
            linestyle=":",
            alpha=0.22,
        )
        axis.plot(
            episode,
            _trailing_mean(observed, window),
            color=observed_color,
            linewidth=1.4,
            label="Observed",
        )
        axis.plot(
            episode,
            _trailing_mean(gawron, window),
            color=gawron_color,
            linewidth=1.4,
            linestyle="--",
            label="Exact Gawron",
        )
        axis.plot(
            episode,
            _trailing_mean(selected, window),
            color=selected_color,
            linewidth=1.4,
            linestyle=":",
            label="Validation-selected",
        )
        axis.set_title(label)
        axis.set_xlabel("Episode")
        axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.45)
    axes[0, 0].set_ylabel(r"Response TV ($\times 10^3$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        frameon=False,
        ncol=3,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "hdv_od_route_response_curves.pdf"
    png_path = output_dir / "hdv_od_route_response_curves.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path)
    plt.close(figure)
    return pdf_path, png_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot observed versus causal OD-route HDV response magnitudes from "
            "offline comparison artifacts."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        required=True,
        help="Repeat as LABEL=/path/to/per_episode_metrics.csv.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    pdf_path, png_path = plot_cases(
        args.case,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        window=int(args.window),
    )
    print(f"[saved] {pdf_path}")
    print(f"[saved] {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
