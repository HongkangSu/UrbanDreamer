"""File, artifact, XML, and CSV metric helpers for SUMO evaluation."""

from __future__ import annotations

import os
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


_AV_KIND_VALUES  = {"av", "machine", "cav"}
_HDV_KIND_VALUES = {"human", "hdv"}


def _copy_artifact_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() and os.path.samefile(src, dst):
            return
    except Exception:
        pass
    shutil.copy2(src, dst)


def _reuse_route_artifacts(src_dir: Path, dst_dir: Path) -> Dict[str, Path]:
    required = {
        "agents_csv": src_dir / "agents.csv",
        "routes_csv": src_dir / "routes.csv",
        "route_rou_xml": src_dir / "route.rou.xml",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "reuse_artifacts_dir is missing required files: " + ", ".join(missing)
        )
    copied = {
        "agents_csv": dst_dir / "agents.csv",
        "routes_csv": dst_dir / "routes.csv",
        "route_rou_xml": dst_dir / "route.rou.xml",
    }
    for key, src in required.items():
        _copy_artifact_if_needed(src, copied[key])
    for optional_name in ("routes_meta.json", "det.add.xml"):
        src = src_dir / optional_name
        if src.exists():
            _copy_artifact_if_needed(src, dst_dir / optional_name)
    return copied


def _reuse_route_artifacts_from_refs(
    artifact_refs: Dict[str, Any],
    dst_dir: Path,
) -> Dict[str, Path]:
    required = {
        "agents_csv": Path(str(artifact_refs["agents_csv"])).resolve(),
        "routes_csv": Path(str(artifact_refs["routes_csv"])).resolve(),
        "route_rou_xml": Path(str(artifact_refs["route_rou_xml"])).resolve(),
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "post-mutation state references missing route artifacts: " + ", ".join(missing)
        )
    copied = {
        "agents_csv": dst_dir / "agents.csv",
        "routes_csv": dst_dir / "routes.csv",
        "route_rou_xml": dst_dir / "route.rou.xml",
    }
    for key, src in required.items():
        _copy_artifact_if_needed(src, copied[key])
    return copied


def _detect_kind_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("vType", "kind", "type", "agent_type"):
        if col in df.columns:
            return col
    return None


def compute_metrics_from_csvs(
    episodes_folder: str,
    test_start_ep:   int,
    test_end_ep:     int,
) -> Dict:
    tt_av  = []
    tt_hdv = []
    tt_all = []

    if not os.path.isdir(episodes_folder):
        print(f"  WARNING: episodes folder not found: {episodes_folder}")
        return {"t_CAV": float("nan"), "t_HDV": float("nan"), "t_all": float("nan"),
                "n_AV_trips": 0, "n_HDV_trips": 0, "n_total_trips": 0}

    for ep in range(test_start_ep, test_end_ep + 1):
        csv_path = os.path.join(episodes_folder, f"ep{ep}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  WARNING: could not read {csv_path}: {e}")
            continue

        if df.empty or "travel_time" not in df.columns:
            continue

        df["travel_time"] = pd.to_numeric(df["travel_time"], errors="coerce")
        df = df.dropna(subset=["travel_time"])

        kind_col = _detect_kind_column(df)
        if kind_col is not None:
            kinds = df[kind_col].astype(str).str.lower()
            av_mask  = kinds.isin(_AV_KIND_VALUES)
            hdv_mask = kinds.isin(_HDV_KIND_VALUES)
            tt_av.extend(df.loc[av_mask,  "travel_time"].tolist())
            tt_hdv.extend(df.loc[hdv_mask, "travel_time"].tolist())
        else:
            print(f"  NOTE: ep{ep}.csv has no kind/vType column – treating all as 'all'")

        tt_all.extend(df["travel_time"].tolist())

    def _safe_mean_minutes(lst):
        return float(np.mean(lst)) if lst else float("nan")

    return {
        "t_CAV":         _safe_mean_minutes(tt_av),
        "t_HDV":         _safe_mean_minutes(tt_hdv),
        "t_all":         _safe_mean_minutes(tt_all),
        "n_AV_trips":    len(tt_av),
        "n_HDV_trips":   len(tt_hdv),
        "n_total_trips": len(tt_all),
    }


def _summarize_test_episode_metrics(test_df: pd.DataFrame) -> Dict[str, Any]:
    """Use the final test episode as the headline metric.

    The per-episode CSV still records every rollout. The summary keeps the old
    mean values under explicit *_mean keys, while t_CAV/t_HDV/t_all are the
    final test episode values used for judging.
    """
    if len(test_df) == 0:
        return {
            "metric_aggregation": "last_test_episode",
            "summary_episode": None,
            "t_CAV": float("nan"),
            "t_HDV": float("nan"),
            "t_all": float("nan"),
            "t_CAV_mean": float("nan"),
            "t_HDV_mean": float("nan"),
            "t_all_mean": float("nan"),
            "arrived_all": 0,
            "total_agents": 0,
            "unfinished_all": 0,
            "completion_rate": float("nan"),
            "completion_rate_mean": float("nan"),
        }

    ordered = test_df.copy()
    if "episode" in ordered.columns:
        ordered["_episode_sort"] = pd.to_numeric(ordered["episode"], errors="coerce")
        ordered = ordered.sort_values("_episode_sort", kind="stable")

    last = ordered.iloc[-1]
    summary_episode: Any = None
    if "episode" in last.index and pd.notna(last["episode"]):
        try:
            summary_episode = int(last["episode"])
        except (TypeError, ValueError):
            summary_episode = str(last["episode"])

    def _float_col_mean(col: str) -> float:
        if col not in test_df.columns:
            return float("nan")
        vals = pd.to_numeric(test_df[col], errors="coerce")
        return float(vals.mean()) if vals.notna().any() else float("nan")

    def _int_last(col: str) -> int:
        if col not in last.index or pd.isna(last[col]):
            return 0
        try:
            return int(last[col])
        except (TypeError, ValueError):
            return 0

    return {
        "metric_aggregation": "last_test_episode",
        "summary_episode": summary_episode,
        "t_CAV": float(last["t_CAV"]),
        "t_HDV": float(last["t_HDV"]),
        "t_all": float(last["t_all"]),
        "t_CAV_mean": float(test_df["t_CAV"].mean()),
        "t_HDV_mean": float(test_df["t_HDV"].mean()),
        "t_all_mean": float(test_df["t_all"].mean()),
        "arrived_all": _int_last("arrived_all"),
        "total_agents": _int_last("total_agents"),
        "unfinished_all": _int_last("unfinished_all"),
        "completion_rate": float(last["completion_rate"]) if "completion_rate" in last.index and pd.notna(last["completion_rate"]) else float("nan"),
        "completion_rate_mean": _float_col_mean("completion_rate"),
    }


def _episode_complete_for_dataset(*, n_arrived: int, total_agents: int, n_steps: int) -> bool:
    """Return True only when an episode is valid to save as rollout data."""
    return int(total_agents) > 0 and int(n_arrived) >= int(total_agents) and int(n_steps) > 0


def _resolve_protocol_episode_indices(
    *,
    post_mutation_state: Optional[Dict[str, Any]],
    effective_human_eps: int,
    requested_human_eps: int,
) -> Tuple[int, int]:
    if post_mutation_state is None:
        return int(effective_human_eps), int(effective_human_eps)

    protocol_human_eps = int(
        post_mutation_state.get("human_learning_episodes", requested_human_eps)
    )
    mutation_episode_index = int(
        post_mutation_state.get("mutation_episode_index", protocol_human_eps)
    )
    return protocol_human_eps, mutation_episode_index


def _metrics_from_csv_set(csv_paths: set) -> Optional[Dict[str, float]]:
    """Compute mean travel-time metrics from a set of episode CSV files."""
    tt_av, tt_hdv, tt_all = [], [], []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "travel_time" not in df.columns:
            continue
        work = df.copy()
        work["travel_time"] = pd.to_numeric(work["travel_time"], errors="coerce")
        work = work.dropna(subset=["travel_time"])
        if work.empty:
            continue
        kind_col = next((c for c in ("vType", "kind", "type") if c in work.columns), None)
        if kind_col:
            kinds = work[kind_col].astype(str).str.lower()
            av_mask = kinds.isin({"av", "machine", "cav"})
            hdv_mask = kinds.isin({"human", "hdv"})
            tt_av.extend(work.loc[av_mask, "travel_time"].tolist())
            tt_hdv.extend(work.loc[hdv_mask, "travel_time"].tolist())
        tt_all.extend(work["travel_time"].tolist())
    if not tt_all:
        return None
    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")
    return {"t_CAV": _mean(tt_av), "t_HDV": _mean(tt_hdv), "t_all": _mean(tt_all)}


def _snapshot_episode_csvs(episodes_folder: str) -> Dict[str, int]:
    if not os.path.isdir(episodes_folder):
        return {}
    out: Dict[str, int] = {}
    for f in os.listdir(episodes_folder):
        if not (f.startswith("ep") and f.endswith(".csv")):
            continue
        p = os.path.join(episodes_folder, f)
        try:
            out[p] = os.stat(p).st_mtime_ns
        except OSError:
            continue
    return out


def _new_csvs_since(snapshot_before: Dict[str, int], episodes_folder: str, fallback_latest: int = 0) -> set[str]:
    latest = _snapshot_episode_csvs(episodes_folder)
    changed = {
        p for p, m in latest.items()
        if (p not in snapshot_before) or (m > snapshot_before[p])
    }
    if (not changed) and int(fallback_latest) > 0:
        ordered = sorted(latest.items(), key=lambda kv: kv[1], reverse=True)
        changed = {p for p, _ in ordered[: int(fallback_latest)]}
    return changed


def _snapshot_sumo_xmls(sumo_output_folder: str) -> Dict[str, int]:
    if not os.path.isdir(sumo_output_folder):
        return {}
    out: Dict[str, int] = {}
    for f in os.listdir(sumo_output_folder):
        if not f.endswith(".xml"):
            continue
        p = os.path.join(sumo_output_folder, f)
        try:
            out[p] = os.stat(p).st_mtime_ns
        except OSError:
            continue
    return out


def _new_sumo_xmls_since(snapshot_before: Dict[str, int], sumo_output_folder: str, fallback_latest: int = 0) -> set[str]:
    latest = _snapshot_sumo_xmls(sumo_output_folder)
    changed = {
        p for p, m in latest.items()
        if (p not in snapshot_before) or (m > snapshot_before[p])
    }
    if (not changed) and int(fallback_latest) > 0:
        ordered = sorted(latest.items(), key=lambda kv: kv[1], reverse=True)
        changed = {p for p, _ in ordered[: int(fallback_latest)]}
    return changed


def _wait_for_parseable_xml(path: Path, timeout_s: float = 3.0) -> bool:
    deadline = time.time() + max(float(timeout_s), 0.0)
    while time.time() < deadline:
        try:
            ET.parse(path)
            return True
        except ET.ParseError:
            time.sleep(0.1)
        except Exception:
            return False
    return False


def _expected_xml_root_tag(path: Path) -> Optional[str]:
    name = path.name
    if name.startswith("detailed_sumo_stats_"):
        return "tripinfos"
    if name.startswith("sumo_stats_"):
        return "statistics"
    return None


def _wait_for_expected_sumo_xml(path: Path, timeout_s: float = 5.0) -> bool:
    expected_root = _expected_xml_root_tag(path)
    if expected_root is None:
        return _wait_for_parseable_xml(path, timeout_s=timeout_s)
    deadline = time.time() + max(float(timeout_s), 0.0)
    while time.time() < deadline:
        try:
            root = ET.parse(path).getroot()
            tag = str(root.tag).split("}")[-1]
            if tag == expected_root:
                return True
        except ET.ParseError:
            pass
        except Exception:
            return False
        time.sleep(0.1)
    return False


def _copy_protocol_xml_aliases(xml_paths: set[str], protocol_episode_id: int) -> List[Path]:
    aliased: List[Path] = []
    for src_str in sorted(xml_paths):
        src = Path(src_str)
        stem = src.stem
        prefix, sep, suffix = stem.rpartition("_")
        if (not sep) or (not suffix.isdigit()):
            continue
        dst = src.with_name(f"{prefix}_{int(protocol_episode_id)}{src.suffix}")
        _wait_for_parseable_xml(src)
        _copy_artifact_if_needed(src, dst)
        aliased.append(dst)
    return aliased
