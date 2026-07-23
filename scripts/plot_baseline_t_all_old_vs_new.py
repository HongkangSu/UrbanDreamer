#!/usr/bin/env python3
"""Plot episode-level baseline t_all curves for old and corrected runs.

The old series follow the checkpoints used by the formal native-executor
evaluation.  The corrected series are snapshots parsed from still-running
``train.log`` files, so they end at the last fully logged episode and are never
extrapolated.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ALGORITHMS = ("ippo", "mappo", "iql")
CITY_LABELS = {
    "gretz": "Gretz-Armainvilliers",
    "ingolstadt": "Ingolstadt",
}
ALGORITHM_LABELS = {name: name.upper() for name in ALGORITHMS}
OLD_COLOR = "#D97706"
NEW_COLOR = "#2563EB"
INCOMPLETE_COLOR = "#DC2626"

LOG_EPISODE_RE = re.compile(
    r"^\[toy-(?P<algorithm>ippo|mappo|iql)-torchrl\]\s+"
    r"ep=(?P<episode>\d+)/(?P<target>\d+).*?"
    r"\bt_all=(?P<t_all>[-+0-9.eE]+).*?"
    r"\bcompletion=(?P<completion>[-+0-9.eE]+)"
)


@dataclass(frozen=True)
class SeriesSource:
    city: str
    algorithm: str
    version: str
    paths: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for PNG figures and the frozen curve-data CSV.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=20,
        help="Trailing rolling-mean window in episodes (default: 20).",
    )
    return parser.parse_args()


def _read_episode_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["episode", "t_all", "completion_rate"])
    frame = frame.rename(columns={"completion_rate": "completion"})
    return frame[["episode", "t_all", "completion"]]


def _read_episode_log(path: Path, *, algorithm: str) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LOG_EPISODE_RE.match(line)
            if match is None or match.group("algorithm") != algorithm:
                continue
            records.append(
                {
                    "episode": int(match.group("episode")),
                    "t_all": float(match.group("t_all")),
                    "completion": float(match.group("completion")),
                }
            )
    if not records:
        raise ValueError(f"No completed {algorithm} episodes found in {path}")
    return pd.DataFrame.from_records(records)


def _normalize_series(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["episode"] = pd.to_numeric(result["episode"], errors="raise").astype(int)
    result["t_all"] = pd.to_numeric(result["t_all"], errors="raise").astype(float)
    result["completion"] = pd.to_numeric(result["completion"], errors="raise").astype(float)
    result = result.drop_duplicates(subset=["episode"], keep="last").sort_values("episode")
    result = result.reset_index(drop=True)
    if result.empty:
        raise ValueError("Episode series is empty")
    if not np.isfinite(result["t_all"]).all() or bool((result["t_all"] <= 0).any()):
        raise ValueError("t_all must contain only finite positive values")
    if not result["completion"].between(0.0, 1.0).all():
        raise ValueError("completion must be within [0, 1]")
    return result


def _assert_contiguous(frame: pd.DataFrame, *, expected_start: int, expected_end: int, label: str) -> None:
    episodes = frame["episode"].to_numpy(dtype=int)
    expected = np.arange(expected_start, expected_end + 1, dtype=int)
    if not np.array_equal(episodes, expected):
        missing = sorted(set(expected.tolist()) - set(episodes.tolist()))
        extra = sorted(set(episodes.tolist()) - set(expected.tolist()))
        raise ValueError(f"{label} is not contiguous; missing={missing[:10]} extra={extra[:10]}")


def _source_paths(repo_root: Path) -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str], Path]]:
    gretz_root = repo_root / (
        "results/gretz_armainvilliers_current_to_dest_incident_completion100_"
        "stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1"
    )
    ing_root = repo_root / "results/ingolstadt_custom_actionfix_incident_20260516_185810"
    corrected_tag = "baselines_torchrl_corrected_500ep_local_cpu6_fullgrad_replayfix_20260718_074805"

    current = {
        ("gretz", algorithm): gretz_root / corrected_tag / algorithm / "train.log"
        for algorithm in ALGORITHMS
    }
    current.update(
        {
            ("ingolstadt", algorithm): ing_root / corrected_tag / algorithm / "train.log"
            for algorithm in ALGORITHMS
        }
    )

    old = {
        ("gretz", "ippo"): gretz_root
        / "baselines_torchrl_500ep_hdvlearn_20260701_174743/ippo/online_train_episodes.csv",
        ("gretz", "mappo"): gretz_root
        / "baselines_torchrl_500ep_hdvlearn_20260701_174743/mappo/online_train_episodes.csv",
        ("gretz", "iql"): gretz_root
        / "baselines_torchrl_500ep_hdvlearn_memfix_20260702_052900/iql/online_train_episodes.csv",
        ("ingolstadt", "mappo"): ing_root
        / "baselines_torchrl_500ep_hdvlearn_20260701_174743/mappo/online_train_episodes.csv",
        ("ingolstadt", "iql"): ing_root
        / "baselines_torchrl_500ep_hdvlearn_memfix_20260702_052900/iql/online_train_episodes.csv",
    }
    return old, current


def load_all_series(repo_root: Path) -> tuple[pd.DataFrame, list[SeriesSource]]:
    old_paths, current_paths = _source_paths(repo_root)
    frames: list[pd.DataFrame] = []
    sources: list[SeriesSource] = []

    for city in CITY_LABELS:
        for algorithm in ALGORITHMS:
            if (city, algorithm) == ("ingolstadt", "ippo"):
                initial_log = repo_root / (
                    "results/ingolstadt_custom_actionfix_incident_20260516_185810/"
                    "baselines_torchrl_500ep_hdvlearn_20260701_174743/ippo/train.log"
                )
                resumed_csv = repo_root / (
                    "results/ingolstadt_custom_actionfix_incident_20260516_185810/"
                    "baselines_torchrl_500ep_hdvlearn_memfix_20260702_051813/ippo/"
                    "online_train_episodes.csv"
                )
                initial = _read_episode_log(initial_log, algorithm="ippo")
                initial = initial.loc[initial["episode"] <= 200]
                resumed = _read_episode_csv(resumed_csv)
                resumed = resumed.loc[resumed["episode"] >= 201]
                old_frame = _normalize_series(pd.concat([initial, resumed], ignore_index=True))
                old_source_paths = (initial_log, resumed_csv)
            else:
                old_path = old_paths[(city, algorithm)]
                old_frame = _normalize_series(_read_episode_csv(old_path))
                old_source_paths = (old_path,)

            _assert_contiguous(
                old_frame,
                expected_start=1,
                expected_end=500,
                label=f"old/{city}/{algorithm}",
            )
            old_frame = old_frame.assign(city=city, algorithm=algorithm, version="old")
            frames.append(old_frame)
            sources.append(SeriesSource(city, algorithm, "old", old_source_paths))

            current_path = current_paths[(city, algorithm)]
            current_frame = _normalize_series(_read_episode_log(current_path, algorithm=algorithm))
            current_end = int(current_frame["episode"].max())
            _assert_contiguous(
                current_frame,
                expected_start=1,
                expected_end=current_end,
                label=f"new/{city}/{algorithm}",
            )
            current_frame = current_frame.assign(city=city, algorithm=algorithm, version="new")
            frames.append(current_frame)
            sources.append(SeriesSource(city, algorithm, "new", (current_path,)))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[["city", "algorithm", "version", "episode", "t_all", "completion"]]
    return combined, sources


def _city_y_limits(data: pd.DataFrame) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for city in CITY_LABELS:
        values = data.loc[data["city"] == city, "t_all"].to_numpy(dtype=float)
        low = float(values.min())
        high = float(values.max())
        padding = max(0.05 * (high - low), 0.1)
        limits[city] = (np.floor((low - padding) * 10.0) / 10.0, np.ceil((high + padding) * 10.0) / 10.0)
    return limits


def _rolling(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame["t_all"].rolling(window=window, min_periods=max(2, window // 4)).mean()


def _plot_version(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    version: str,
    rolling_window: int,
    y_limits: tuple[float, float],
) -> None:
    color = NEW_COLOR if version == "new" else OLD_COLOR
    linestyle = "-" if version == "new" else "--"
    ax.plot(
        frame["episode"],
        frame["t_all"],
        color=color,
        alpha=0.16,
        linewidth=0.65,
        linestyle=linestyle,
        zorder=1,
    )
    smooth = _rolling(frame, rolling_window)
    ax.plot(
        frame["episode"],
        smooth,
        color=color,
        linewidth=2.2,
        linestyle=linestyle,
        zorder=3,
    )
    valid_smooth = smooth.dropna()
    if version == "new" and not valid_smooth.empty:
        endpoint = int(valid_smooth.index[-1])
        ax.scatter(
            [frame.loc[endpoint, "episode"]],
            [valid_smooth.iloc[-1]],
            color=color,
            s=24,
            zorder=4,
        )
    incomplete = frame.loc[frame["completion"] < 0.95, "episode"]
    if not incomplete.empty:
        rug_y = y_limits[0] + 0.015 * (y_limits[1] - y_limits[0])
        ax.scatter(
            incomplete,
            np.full(len(incomplete), rug_y),
            marker="|",
            s=20,
            linewidths=0.7,
            color=INCOMPLETE_COLOR,
            alpha=0.35,
            zorder=2,
        )


def _base_axes(
    data: pd.DataFrame,
    *,
    title: str,
    snapshot_label: str,
    y_limits: dict[str, tuple[float, float]],
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.8), sharey="row")
    fig.suptitle(f"{title}\nSnapshot: {snapshot_label}", fontsize=15, fontweight="bold", y=0.985)
    for row, city in enumerate(CITY_LABELS):
        for column, algorithm in enumerate(ALGORITHMS):
            ax = axes[row, column]
            ax.set_ylim(*y_limits[city])
            ax.grid(True, color="#CBD5E1", alpha=0.45, linewidth=0.65)
            ax.set_axisbelow(True)
            ax.set_title(f"{CITY_LABELS[city]} — {ALGORITHM_LABELS[algorithm]}", fontsize=11)
            ax.set_xlabel("Training episode")
            if column == 0:
                ax.set_ylabel("Episode t_all (minutes; arrived vehicles)")
    return fig, axes


def _figure_legend(fig: plt.Figure, *, include_old: bool, include_new: bool, rolling_window: int) -> None:
    handles: list[Line2D] = []
    if include_old:
        handles.append(
            Line2D([0], [0], color=OLD_COLOR, linestyle="--", linewidth=2.2, label="Old rolling mean")
        )
    if include_new:
        handles.append(
            Line2D([0], [0], color=NEW_COLOR, linestyle="-", linewidth=2.2, label="New rolling mean")
        )
    handles.extend(
        [
            Line2D([0], [0], color="#64748B", alpha=0.3, linewidth=0.8, label="Raw episode value"),
            Line2D(
                [0],
                [0],
                color=INCOMPLETE_COLOR,
                marker="|",
                linestyle="None",
                markersize=8,
                label="Completion < 95%",
            ),
        ]
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.045),
    )
    fig.text(
        0.5,
        0.014,
        (
            f"Thick lines: trailing {rolling_window}-episode mean. "
            "t_all excludes unfinished vehicles, so low-completion episodes may look artificially low. "
            "One seed; no confidence band."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0.025, 0.095, 0.995, 0.94))


def plot_comparison(
    data: pd.DataFrame,
    *,
    output_path: Path,
    rolling_window: int,
    y_limits: dict[str, tuple[float, float]],
    snapshot_label: str,
    matched_horizon: bool,
) -> None:
    horizon_label = "Matched training horizon" if matched_horizon else "Full 500-episode horizon"
    fig, axes = _base_axes(
        data,
        title=f"Baseline t_all: corrected vs old ({horizon_label})",
        snapshot_label=snapshot_label,
        y_limits=y_limits,
    )
    for row, city in enumerate(CITY_LABELS):
        for column, algorithm in enumerate(ALGORITHMS):
            ax = axes[row, column]
            new = data.loc[
                (data["city"] == city) & (data["algorithm"] == algorithm) & (data["version"] == "new")
            ].reset_index(drop=True)
            old = data.loc[
                (data["city"] == city) & (data["algorithm"] == algorithm) & (data["version"] == "old")
            ].reset_index(drop=True)
            current_end = int(new["episode"].max())
            old_visible = old.loc[old["episode"] <= current_end].reset_index(drop=True) if matched_horizon else old
            _plot_version(
                ax,
                old_visible,
                version="old",
                rolling_window=rolling_window,
                y_limits=y_limits[city],
            )
            _plot_version(
                ax,
                new,
                version="new",
                rolling_window=rolling_window,
                y_limits=y_limits[city],
            )
            ax.set_xlim(1, current_end if matched_horizon else 500)
            ax.text(
                0.98,
                0.96,
                f"old n={len(old_visible)} | new n={len(new)}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color="#475569",
            )
    _figure_legend(fig, include_old=True, include_new=True, rolling_window=rolling_window)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_single_version(
    data: pd.DataFrame,
    *,
    version: str,
    output_path: Path,
    rolling_window: int,
    y_limits: dict[str, tuple[float, float]],
    snapshot_label: str,
) -> None:
    title_version = "Corrected/new snapshot" if version == "new" else "Old formal-eval checkpoint runs"
    fig, axes = _base_axes(
        data,
        title=f"Baseline t_all: {title_version}",
        snapshot_label=snapshot_label,
        y_limits=y_limits,
    )
    for row, city in enumerate(CITY_LABELS):
        for column, algorithm in enumerate(ALGORITHMS):
            ax = axes[row, column]
            frame = data.loc[
                (data["city"] == city)
                & (data["algorithm"] == algorithm)
                & (data["version"] == version)
            ].reset_index(drop=True)
            _plot_version(
                ax,
                frame,
                version=version,
                rolling_window=rolling_window,
                y_limits=y_limits[city],
            )
            ax.set_xlim(1, int(frame["episode"].max()))
            ax.text(
                0.98,
                0.96,
                f"n={len(frame)}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color="#475569",
            )
    _figure_legend(
        fig,
        include_old=version == "old",
        include_new=version == "new",
        rolling_window=rolling_window,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_completion_comparison(
    data: pd.DataFrame,
    *,
    output_path: Path,
    rolling_window: int,
    snapshot_label: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.5), sharey=True)
    fig.suptitle(
        f"Completion diagnostic: corrected vs old (matched horizon)\nSnapshot: {snapshot_label}",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    for row, city in enumerate(CITY_LABELS):
        for column, algorithm in enumerate(ALGORITHMS):
            ax = axes[row, column]
            new = data.loc[
                (data["city"] == city) & (data["algorithm"] == algorithm) & (data["version"] == "new")
            ].reset_index(drop=True)
            current_end = int(new["episode"].max())
            old = data.loc[
                (data["city"] == city)
                & (data["algorithm"] == algorithm)
                & (data["version"] == "old")
                & (data["episode"] <= current_end)
            ].reset_index(drop=True)
            for frame, color, linestyle, label in (
                (old, OLD_COLOR, "--", "Old"),
                (new, NEW_COLOR, "-", "New"),
            ):
                smooth = frame["completion"].rolling(
                    window=rolling_window,
                    min_periods=max(2, rolling_window // 4),
                ).mean()
                ax.plot(frame["episode"], frame["completion"], color=color, alpha=0.12, linewidth=0.6)
                ax.plot(frame["episode"], smooth, color=color, linestyle=linestyle, linewidth=2.1, label=label)
            ax.axhline(0.95, color=INCOMPLETE_COLOR, linestyle=":", linewidth=1.0, alpha=0.8)
            ax.set_xlim(1, current_end)
            ax.set_ylim(0.6, 1.01)
            ax.grid(True, color="#CBD5E1", alpha=0.45, linewidth=0.65)
            ax.set_title(f"{CITY_LABELS[city]} — {ALGORITHM_LABELS[algorithm]}", fontsize=11)
            ax.set_xlabel("Training episode")
            if column == 0:
                ax.set_ylabel("Completion rate")
    fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.text(
        0.5,
        0.012,
        f"Raw episode completion plus trailing {rolling_window}-episode mean; red dotted line = 95%.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0.025, 0.08, 0.995, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _relative_sources(sources: Iterable[SeriesSource], repo_root: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for source in sources:
        for path in source.paths:
            try:
                display_path = str(path.relative_to(repo_root))
            except ValueError:
                display_path = str(path)
            records.append(
                {
                    "city": source.city,
                    "algorithm": source.algorithm,
                    "version": source.version,
                    "source_path": display_path,
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    args = parse_args()
    if args.rolling_window <= 1:
        raise ValueError("--rolling-window must be greater than 1")
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = datetime.now(timezone.utc)
    snapshot_label = snapshot.strftime("%Y-%m-%d %H:%M:%S UTC")

    data, sources = load_all_series(repo_root)
    data = data.assign(snapshot_utc=snapshot.isoformat())
    data.to_csv(output_dir / "curve_data_snapshot.csv", index=False)
    _relative_sources(sources, repo_root).to_csv(output_dir / "source_manifest.csv", index=False)
    y_limits = _city_y_limits(data)

    plot_comparison(
        data,
        output_path=output_dir / "t_all_old_vs_new_matched_horizon.png",
        rolling_window=args.rolling_window,
        y_limits=y_limits,
        snapshot_label=snapshot_label,
        matched_horizon=True,
    )
    plot_comparison(
        data,
        output_path=output_dir / "t_all_old_vs_new_full_500ep.png",
        rolling_window=args.rolling_window,
        y_limits=y_limits,
        snapshot_label=snapshot_label,
        matched_horizon=False,
    )
    plot_single_version(
        data,
        version="old",
        output_path=output_dir / "t_all_old_only.png",
        rolling_window=args.rolling_window,
        y_limits=y_limits,
        snapshot_label=snapshot_label,
    )
    plot_single_version(
        data,
        version="new",
        output_path=output_dir / "t_all_new_only_snapshot.png",
        rolling_window=args.rolling_window,
        y_limits=y_limits,
        snapshot_label=snapshot_label,
    )
    plot_completion_comparison(
        data,
        output_path=output_dir / "completion_old_vs_new_matched_horizon.png",
        rolling_window=args.rolling_window,
        snapshot_label=snapshot_label,
    )

    counts = (
        data.groupby(["city", "algorithm", "version"], as_index=False)
        .agg(first_episode=("episode", "min"), last_episode=("episode", "max"), rows=("episode", "size"))
        .sort_values(["city", "algorithm", "version"])
    )
    print(f"snapshot={snapshot_label}")
    print(f"output_dir={output_dir}")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
