#!/usr/bin/env python3
"""
URB-fair collection wrapper.

This script aligns world-model data collection to the same environment phase as
URB-fair evaluation:

1. Reuse a shared route/agent artifact bundle.
2. Run human adaptation episodes.
3. Derive and fix the post-mutation machine/human set.
4. Continue collecting episodes in that post-mutation regime.

The collection policy itself may still use exploration (random / epsilon-greedy
or DAgger); the fairness target here is the environment protocol, not the
policy-generation scheme.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SETTINGS = REPO_ROOT / "configs" / "experiments" / "ingolstadt_custom_urb_config1_eval.env"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dreamer4.urban_routing.data_collector import DataCollector
from dreamer4.urban_routing.experiment_settings import load_settings_file
from dreamer4.urban_routing.urb_fair_protocol import (
    build_post_mutation_state,
    collect_route_artifact_contract,
    derive_fixed_roles,
    load_post_mutation_state,
    verify_post_mutation_state,
    write_post_mutation_state,
)
from scripts.eval_sumo import (
    _load_hdv_learning_state,
    _merge_hdv_learning_state,
    _resolve_optional_path,
    _reuse_route_artifacts,
    _reuse_route_artifacts_from_refs,
    _save_hdv_learning_state,
    _snapshot_hdv_learning_state,
    _snapshot_hdv_route_state,
)


def _default_output_dir(cfg: Dict[str, str]) -> str:
    network = cfg.get("NETWORK_NAME", "network")
    return f"data/{network}/collect_urb_fair"


def _default_task_name(cfg: Dict[str, str]) -> str:
    network = cfg.get("NETWORK_NAME", "network")
    return f"{network}_urb_fair"


def _default_records_dir(cfg: Dict[str, str]) -> str:
    network = cfg.get("NETWORK_NAME", "network")
    return f"results/{network}/collect_urb_fair_records"


def _log_tt_message(collector: DataCollector) -> str:
    n_arrived = len(collector._last_ep_travel_times)
    reroute_msg = (
        f"safe_reroutes={int(getattr(collector, '_last_ep_safe_reroutes', 0))} "
        f"kept_unsafe={int(getattr(collector, '_last_ep_unsafe_reroute_keeps', 0))} "
        f"drain={int(getattr(collector, '_last_ep_completion_drain_steps', 0))} "
        f"stale={int(getattr(collector, '_last_ep_completion_drain_stale_steps', 0))}"
    )
    if n_arrived > 0:
        avg_tt = float(sum(collector._last_ep_travel_times) / n_arrived)
        return f"arrived={n_arrived} | avg_tt={avg_tt:.1f}s ({avg_tt/60.0:.2f}min) | {reroute_msg}"
    return f"arrived=0 (no vehicles completed their route) | {reroute_msg}"


def _episode_selection_metrics(
    collector: DataCollector,
    ep_frames: List[Any],
    *,
    busy_threshold: float = 0.05,
) -> Dict[str, float]:
    avg_tt_sec = float("nan")
    if collector._last_ep_travel_times:
        avg_tt_sec = float(sum(collector._last_ep_travel_times) / max(len(collector._last_ep_travel_times), 1))
    elif ep_frames:
        avg_tt_sec = float(collector._global_avg_travel_time(ep_frames[-1]))

    od_route_means: Dict[Tuple[int, int], List[float]] = {}
    for (oi, di, _k), tts in collector._last_ep_route_tts.items():
        if not tts:
            continue
        od_route_means.setdefault((oi, di), []).append(float(np.mean(tts)))

    route_tt_gap_vals = [
        float(max(route_means) - min(route_means))
        for route_means in od_route_means.values()
        if len(route_means) > 1
    ]

    busy_counts: List[float] = []
    busy_mass: List[float] = []
    busy_mean_volume: List[float] = []
    threshold = max(0.0, float(busy_threshold))
    for frame in ep_frames:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        vol = np.maximum(arr[:, 2], 0.0)
        mask = vol > threshold
        busy_counts.append(float(mask.sum()))
        busy_mass.append(float(vol[mask].sum()) if bool(mask.any()) else 0.0)
        busy_mean_volume.append(float(vol[mask].mean()) if bool(mask.any()) else 0.0)

    return {
        "n_arrived": float(len(collector._last_ep_travel_times)),
        "avg_tt_sec": avg_tt_sec,
        "avg_tt_min": avg_tt_sec / 60.0 if np.isfinite(avg_tt_sec) else float("nan"),
        "route_tt_key_count": float(len(collector._last_ep_route_tts)),
        "route_tt_od_count": float(len(od_route_means)),
        "route_tt_gap_mean_sec": float(np.mean(route_tt_gap_vals)) if route_tt_gap_vals else 0.0,
        "busy_edge_count_mean": float(np.mean(busy_counts)) if busy_counts else 0.0,
        "busy_edge_count_max": float(np.max(busy_counts)) if busy_counts else 0.0,
        "busy_edge_mass_mean": float(np.mean(busy_mass)) if busy_mass else 0.0,
        "busy_edge_mean_volume": float(np.mean(busy_mean_volume)) if busy_mean_volume else 0.0,
    }


def _episode_matches_selection(
    metrics: Dict[str, float],
    *,
    min_avg_tt_min: Optional[float],
    max_avg_tt_min: Optional[float],
    min_route_tt_od_count: int,
    min_route_tt_gap_sec: float,
    max_route_tt_gap_sec: Optional[float],
    min_busy_edge_count_mean: float,
    max_busy_edge_count_mean: Optional[float],
    min_busy_edge_mass_mean: float,
) -> bool:
    avg_tt_min = float(metrics.get("avg_tt_min", float("nan")))
    if min_avg_tt_min is not None:
        if (not np.isfinite(avg_tt_min)) or avg_tt_min < float(min_avg_tt_min):
            return False
    if max_avg_tt_min is not None:
        if (not np.isfinite(avg_tt_min)) or avg_tt_min > float(max_avg_tt_min):
            return False
    if float(metrics.get("route_tt_od_count", 0.0)) < float(max(0, int(min_route_tt_od_count))):
        return False
    if float(metrics.get("route_tt_gap_mean_sec", 0.0)) < float(max(0.0, float(min_route_tt_gap_sec))):
        return False
    if max_route_tt_gap_sec is not None:
        if float(metrics.get("route_tt_gap_mean_sec", 0.0)) > float(max_route_tt_gap_sec):
            return False
    if float(metrics.get("busy_edge_count_mean", 0.0)) < float(max(0.0, float(min_busy_edge_count_mean))):
        return False
    if max_busy_edge_count_mean is not None:
        if float(metrics.get("busy_edge_count_mean", 0.0)) > float(max_busy_edge_count_mean):
            return False
    if float(metrics.get("busy_edge_mass_mean", 0.0)) < float(max(0.0, float(min_busy_edge_mass_mean))):
        return False
    return True


def _episode_all_arrived(collector: DataCollector, metrics: Dict[str, float]) -> bool:
    total_agents = int(getattr(collector, "_num_agents", 0) or 0)
    n_arrived = int(metrics.get("n_arrived", 0.0))
    return total_agents > 0 and n_arrived >= total_agents


def _score_similarity(
    value: float,
    *,
    target: Optional[float],
    tolerance: float,
) -> float:
    if target is None or not np.isfinite(float(value)):
        return 0.0
    tol = max(float(tolerance), 1e-8)
    delta = abs(float(value) - float(target))
    return max(0.0, 1.0 - (delta / tol))


def _episode_selection_score(
    metrics: Dict[str, float],
    *,
    target_avg_tt_min: Optional[float],
    target_route_tt_gap_sec: Optional[float],
    target_busy_edge_count_mean: Optional[float],
    target_busy_edge_mass_mean: Optional[float],
    avg_tt_tolerance_min: float,
    route_tt_gap_tolerance_sec: float,
    busy_edge_count_tolerance: float,
    busy_edge_mass_tolerance: float,
    avg_tt_weight: float,
    route_tt_gap_weight: float,
    busy_edge_count_weight: float,
    busy_edge_mass_weight: float,
) -> float:
    weighted_scores = []
    for value, target, tol, weight in (
        (metrics.get("avg_tt_min", float("nan")), target_avg_tt_min, avg_tt_tolerance_min, avg_tt_weight),
        (metrics.get("route_tt_gap_mean_sec", 0.0), target_route_tt_gap_sec, route_tt_gap_tolerance_sec, route_tt_gap_weight),
        (metrics.get("busy_edge_count_mean", 0.0), target_busy_edge_count_mean, busy_edge_count_tolerance, busy_edge_count_weight),
        (metrics.get("busy_edge_mass_mean", 0.0), target_busy_edge_mass_mean, busy_edge_mass_tolerance, busy_edge_mass_weight),
    ):
        w = max(float(weight), 0.0)
        if w <= 0.0 or target is None:
            continue
        weighted_scores.append((w, _score_similarity(float(value), target=target, tolerance=float(tol))))
    if not weighted_scores:
        return 0.0
    total_weight = sum(w for w, _ in weighted_scores)
    return float(sum(w * s for w, s in weighted_scores) / max(total_weight, 1e-8))


def _update_hdv_after_episode(collector: DataCollector, ep_frames: List[Any]) -> None:
    if collector._hdv_route_probs is not None and collector._route_registry is not None:
        if collector._last_ep_route_tts:
            collector._update_hdv_gawron_actual()
        elif len(ep_frames) > 0:
            collector._update_hdv_gawron(ep_frames[-1])


def _refresh_loaded_hdv_state(
    collector: DataCollector,
    base_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    env = getattr(collector.connector, "env", None)
    snapshot = _snapshot_hdv_learning_state(env) if env is not None else {}
    latest = _merge_hdv_learning_state(
        base_state,
        snapshot,
        route_state=_snapshot_hdv_route_state(collector),
    )
    collector._loaded_hdv_state = latest
    return latest


def _append_saved_episode(
    *,
    saved_episode_id: int,
    ep_ids: List[int],
    ep_av_edge_acts: List[Any],
    ep_hdv_edge_acts: List[Any],
    ep_joint_edge_acts: List[Any],
    ep_rews: List[float],
    ep_frames: List[Any],
    ep_departs: List[Any],
    ep_contexts: List[Any],
    all_episodes: List[int],
    all_av_edge_actions: List[Any],
    all_hdv_edge_actions: List[Any],
    all_joint_edge_actions: List[Any],
    all_rewards: List[float],
    all_flows: List[Any],
    all_departs: List[Any],
    all_contexts: List[Any],
) -> None:
    all_episodes += [saved_episode_id] * len(ep_ids)
    all_av_edge_actions += ep_av_edge_acts
    all_hdv_edge_actions += ep_hdv_edge_acts
    all_joint_edge_actions += ep_joint_edge_acts
    all_rewards += ep_rews
    all_flows += ep_frames
    all_departs += ep_departs
    all_contexts += ep_contexts


def _write_collection_manifest(
    *,
    records_dir: Path,
    settings_path: str,
    output_dir: Path,
    task_name: str,
    num_episodes: int,
    warmup_episodes: int,
    post_mutation_state_path: Optional[Path],
    hdv_state_in_path: Optional[Path],
    hdv_state_out_path: Optional[Path],
    partial: bool = False,
    episodes_saved: Optional[int] = None,
) -> Path:
    cfg = load_settings_file(settings_path)
    manifest = {
        "schema_version": 2,
        "action_schema": "av_planned_edge_flow_pre_advance_v1",
        "experiment_config": str(Path(settings_path).resolve()),
        "network": cfg.get("NETWORK_NAME", ""),
        "ratio_machines": float(cfg.get("RATIO_MACHINES", "0") or 0.0),
        "human_model": "gawron",
        "should_humans_adapt": False,
        "warmup_episodes": int(warmup_episodes),
        "num_episodes": int(num_episodes),
        "episodes_saved": int(episodes_saved) if episodes_saved is not None else int(num_episodes),
        "partial": bool(partial),
        "output_dir": str(output_dir.resolve()),
        "task_name": task_name,
        "records_dir": str(records_dir.resolve()),
        "post_mutation_state": str(post_mutation_state_path.resolve()) if post_mutation_state_path else "",
        "hdv_state_in": str(hdv_state_in_path.resolve()) if hdv_state_in_path else "",
        "hdv_state_out": str(hdv_state_out_path.resolve()) if hdv_state_out_path else "",
        "incident": {
            "edge": cfg.get("INCIDENT_EDGE", ""),
            "start": float(cfg.get("INCIDENT_START", "0") or 0.0),
            "end": float(cfg.get("INCIDENT_END", "0") or 0.0),
            "speed": float(cfg.get("INCIDENT_SPEED", "0") or 0.0),
        },
    }
    manifest.update(collect_route_artifact_contract(records_dir))
    out_path = records_dir / "urb_fair_collection_manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _flush_partial_collection(
    *,
    collector: DataCollector,
    task_name: str,
    records_dir: Path,
    settings_path: str,
    output_dir: Path,
    requested_num_episodes: int,
    warmup_episodes: int,
    post_mutation_state_path: Optional[Path],
    hdv_state_in_path: Optional[Path],
    hdv_state_out_path: Optional[Path],
    loaded_hdv_state: Optional[Dict[str, Any]],
    all_episodes: List[int],
    all_av_edge_actions: List[Any],
    all_hdv_edge_actions: List[Any],
    all_joint_edge_actions: List[Any],
    all_rewards: List[float],
    all_flows: List[Any],
    all_departs: List[Any],
    all_contexts: List[Any],
) -> Optional[Path]:
    if not all_episodes:
        return None
    if hdv_state_out_path is not None:
        _save_hdv_learning_state(hdv_state_out_path, loaded_hdv_state)
    collector._save_intermediate(
        task_name,
        all_episodes,
        all_av_edge_actions,
        all_hdv_edge_actions,
        all_joint_edge_actions,
        all_rewards,
        all_flows,
        all_departs,
        all_contexts,
    )
    return _write_collection_manifest(
        records_dir=records_dir,
        settings_path=settings_path,
        output_dir=output_dir,
        task_name=task_name,
        num_episodes=int(requested_num_episodes),
        warmup_episodes=int(warmup_episodes),
        post_mutation_state_path=post_mutation_state_path,
        hdv_state_in_path=hdv_state_in_path,
        hdv_state_out_path=hdv_state_out_path,
        partial=True,
        episodes_saved=len(set(all_episodes)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect URB-fair world-model data in the post-mutation regime.")
    p.add_argument("--settings", type=str, default=str(DEFAULT_SETTINGS))
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--task_name", type=str, default="")
    p.add_argument("--records_dir", type=str, default="")
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--warmup_episodes", type=int, default=-1,
                   help="Human adaptation episodes before mutation. Default: HDV_ADAPT_EP from settings unless post_mutation_state_in is provided.")
    p.add_argument("--max_steps", type=int, default=3600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--incident_edge", type=str, default="",
                   help="Optional edge id whose lanes are slowed during [incident_start, incident_end).")
    p.add_argument("--incident_start", type=float, default=None,
                   help="SUMO time in seconds when the optional incident starts.")
    p.add_argument("--incident_end", type=float, default=None,
                   help="SUMO time in seconds when the optional incident ends.")
    p.add_argument("--incident_speed", type=float, default=None,
                   help="Temporary incident lane max speed in m/s. <=0 disables the incident.")
    p.add_argument("--post_mutation_state_in", type=str, default="")
    p.add_argument("--post_mutation_state_out", type=str, default="")
    p.add_argument("--hdv_state_in", type=str, default="")
    p.add_argument("--hdv_state_out", type=str, default="")
    p.add_argument("--post_mutation_hdv_state_out", type=str, default="",
                   help="Path to save the HDV learning state immediately after warmup/mutation.")
    p.add_argument("--policy_ckpt", type=str, default="",
                   help="Optional MARL policy checkpoint for DAgger collection.")
    p.add_argument("--random_av_fraction", type=float, default=0.3)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--safe_reroute_min_remaining_m", type=float, default=35.0,
                   help="Minimum lane distance remaining for reroutes that require a lane change. Set <0 to disable.")
    p.add_argument("--completion_drain_steps", type=int, default=0,
                   help="Extra SUMO ticks after the recorded horizon used only to observe late arrivals.")
    p.add_argument("--completion_drain_patience_steps", type=int, default=0,
                   help="Stop completion drain after this many consecutive ticks without new arrivals. <=0 disables.")
    p.add_argument("--sumo_time_to_teleport", type=str, default="",
                   help="Override RouteRL's SUMO --time-to-teleport. Empty preserves the route artifact default.")
    p.add_argument("--dagger_beta", type=float, default=0.5)
    p.add_argument("--dagger_noise", type=float, default=0.3)
    p.add_argument("--checkpoint_every", type=int, default=10,
                   help="Save an intermediate demo+flow snapshot every N collected episodes. 0 disables intermediate checkpoints.")
    p.add_argument("--max_runtime_episodes", type=int, default=0,
                   help="Optional cap on total runtime episodes attempted. 0 means equal to num_episodes.")
    p.add_argument("--allow_incomplete_episodes", action="store_true",
                   help="Save episodes that meet --min_arrival_fraction even when not all vehicles arrive.")
    p.add_argument("--min_arrival_fraction", type=float, default=1.0,
                   help="Minimum arrived/total fraction required when --allow_incomplete_episodes is set.")
    p.add_argument("--min_avg_tt_min", type=float, default=None,
                   help="If set, only save episodes whose average travel time is at least this many minutes.")
    p.add_argument("--max_avg_tt_min", type=float, default=None,
                   help="If set, only save episodes whose average travel time is at most this many minutes.")
    p.add_argument("--min_route_tt_od_count", type=int, default=0,
                   help="If >0, only save episodes with at least this many OD pairs having observed route travel times.")
    p.add_argument("--min_route_tt_gap_sec", type=float, default=0.0,
                   help="If >0, only save episodes whose mean per-OD route-TT gap is at least this many seconds.")
    p.add_argument("--max_route_tt_gap_sec", type=float, default=None,
                   help="If set, only save episodes whose mean per-OD route-TT gap is at most this many seconds.")
    p.add_argument("--busy_threshold", type=float, default=0.05,
                   help="Busy-edge threshold on the decoded volume-ratio channel when computing episode-level busy-edge metrics.")
    p.add_argument("--min_busy_edge_count_mean", type=float, default=0.0,
                   help="If >0, only save episodes whose mean busy-edge count across frames is at least this value.")
    p.add_argument("--max_busy_edge_count_mean", type=float, default=None,
                   help="If set, only save episodes whose mean busy-edge count across frames is at most this value.")
    p.add_argument("--min_busy_edge_mass_mean", type=float, default=0.0,
                   help="If >0, only save episodes whose mean sum of busy-edge volume ratios across frames is at least this value.")
    p.add_argument("--selection_mode", type=str, default="threshold", choices=["threshold", "hybrid_score"],
                   help="Episode selection rule. 'threshold' uses hard bounds only; 'hybrid_score' first applies hard guardrails, then accepts by weighted score.")
    p.add_argument("--selection_score_threshold", type=float, default=0.0,
                   help="Minimum weighted score required when --selection_mode hybrid_score.")
    p.add_argument("--target_avg_tt_min", type=float, default=None,
                   help="Target avg_tt_min used by the hybrid score.")
    p.add_argument("--target_route_tt_gap_sec", type=float, default=None,
                   help="Target route-TT gap used by the hybrid score.")
    p.add_argument("--target_busy_edge_count_mean", type=float, default=None,
                   help="Target busy-edge count mean used by the hybrid score.")
    p.add_argument("--target_busy_edge_mass_mean", type=float, default=None,
                   help="Target busy-edge mass mean used by the hybrid score.")
    p.add_argument("--score_avg_tt_tolerance_min", type=float, default=1.5,
                   help="Tolerance band around target_avg_tt_min for the hybrid score.")
    p.add_argument("--score_route_tt_gap_tolerance_sec", type=float, default=120.0,
                   help="Tolerance band around target_route_tt_gap_sec for the hybrid score.")
    p.add_argument("--score_busy_edge_count_tolerance", type=float, default=4.0,
                   help="Tolerance band around target_busy_edge_count_mean for the hybrid score.")
    p.add_argument("--score_busy_edge_mass_tolerance", type=float, default=1.5,
                   help="Tolerance band around target_busy_edge_mass_mean for the hybrid score.")
    p.add_argument("--score_avg_tt_weight", type=float, default=1.0,
                   help="Weight for avg_tt component in the hybrid score.")
    p.add_argument("--score_route_tt_gap_weight", type=float, default=1.0,
                   help="Weight for route-gap component in the hybrid score.")
    p.add_argument("--score_busy_edge_count_weight", type=float, default=1.0,
                   help="Weight for busy-edge-count component in the hybrid score.")
    p.add_argument("--score_busy_edge_mass_weight", type=float, default=1.0,
                   help="Weight for busy-edge-mass component in the hybrid score.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings_path = str(Path(args.settings).resolve())
    cfg = load_settings_file(settings_path)
    output_dir = Path(args.output_dir or _default_output_dir(cfg)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_name = args.task_name or _default_task_name(cfg)
    records_dir = Path(args.records_dir or _default_records_dir(cfg)).resolve()
    records_dir.mkdir(parents=True, exist_ok=True)

    post_mutation_state_in_path = _resolve_optional_path(args.post_mutation_state_in)
    post_mutation_state_out_path = _resolve_optional_path(
        args.post_mutation_state_out or str(records_dir / "post_mutation_state.json")
    )
    post_mutation_hdv_state_out_path = _resolve_optional_path(
        args.post_mutation_hdv_state_out or str(records_dir / "post_mutation_hdv_learning_state.json")
    )
    hdv_state_in_path = _resolve_optional_path(args.hdv_state_in)
    hdv_state_out_path = _resolve_optional_path(
        args.hdv_state_out or str(records_dir / "hdv_learning_state_final.json")
    )

    print("=" * 70)
    print("  URB-fair Data Collection")
    print("=" * 70)
    print(f"  Settings:      {settings_path}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Records dir:   {records_dir}")
    print(f"  Task name:     {task_name}")
    print(f"  Num episodes:  {args.num_episodes}")
    print(f"  Seed:          {args.seed}")

    loaded_hdv_state = _load_hdv_learning_state(hdv_state_in_path)
    post_mutation_state = None
    reused_artifacts = None
    if post_mutation_state_in_path is not None and post_mutation_state_in_path.exists():
        post_mutation_state = load_post_mutation_state(post_mutation_state_in_path)
        reused_artifacts = _reuse_route_artifacts_from_refs(
            post_mutation_state.get("artifact_refs", {}),
            records_dir,
        )
        verify_post_mutation_state(
            results_dir=records_dir,
            expected_network=cfg.get("NETWORK_NAME", ""),
            expected_ratio_machines=float(cfg.get("RATIO_MACHINES", "0") or 0.0),
            expected_human_model="gawron",
            expected_should_humans_adapt=False,
            state=post_mutation_state,
        )
        if loaded_hdv_state is None:
            state_ref = post_mutation_state.get("human_learning_state_ref")
            if state_ref:
                loaded_hdv_state = _load_hdv_learning_state(Path(str(state_ref)))
    else:
        reuse_dir = cfg.get("ROUTE_ARTIFACTS_DIR", "")
        if reuse_dir:
            reused_artifacts = _reuse_route_artifacts(Path(reuse_dir).resolve(), records_dir)
        else:
            raise ValueError("URB-fair collection requires ROUTE_ARTIFACTS_DIR or --post_mutation_state_in")

    agents_csv = str(reused_artifacts["agents_csv"])
    warmup_episodes = 0 if post_mutation_state is not None else (
        int(cfg.get("HDV_ADAPT_EP", "0")) if args.warmup_episodes < 0 else int(args.warmup_episodes)
    )

    print(f"  Warmup episodes: {warmup_episodes}")
    incident_edge = str(args.incident_edge or cfg.get("INCIDENT_EDGE", "") or "").strip()
    incident_start = float(args.incident_start if args.incident_start is not None else (cfg.get("INCIDENT_START", "0") or 0.0))
    incident_end = float(args.incident_end if args.incident_end is not None else (cfg.get("INCIDENT_END", "0") or 0.0))
    incident_speed = float(args.incident_speed if args.incident_speed is not None else (cfg.get("INCIDENT_SPEED", "0") or 0.0))
    if incident_edge and incident_end > incident_start and incident_speed > 0.0:
        print(
            f"  Incident: edge={incident_edge} "
            f"window=[{incident_start:.1f}, {incident_end:.1f}) "
            f"speed={incident_speed:.2f}m/s"
        )
    if post_mutation_state_in_path is not None:
        print(f"  Post-mutation state in:  {post_mutation_state_in_path}")
    if post_mutation_state_out_path is not None:
        print(f"  Post-mutation state out: {post_mutation_state_out_path}")
    if hdv_state_in_path is not None:
        print(f"  HDV state in:            {hdv_state_in_path}")
    if hdv_state_out_path is not None:
        print(f"  HDV state out:           {hdv_state_out_path}")
    print("  Human learning after mutation: disabled")

    collector = DataCollector(
        output_dir=str(output_dir),
        shard_size=2048,
        seed=int(args.seed),
        backend="routerl",
        network_name=cfg.get("NETWORK_NAME", ""),
        network_folder=cfg.get("NETWORK_DIR", ""),
        origins=None,
        destinations=None,
        number_of_paths=int(cfg.get("N_ROUTES", "4") or 4),
        path_gen_beta=float(cfg.get("PATH_GEN_BETA", "-5.0") or -5.0),
        num_samples=int(cfg.get("PATH_NUM_SAMPLES", "10") or 10),
        simulation_timesteps=int(args.max_steps),
        records_folder=str(records_dir),
        av_fraction=float(cfg.get("RATIO_MACHINES", "0") or 0.0),
        random_av_fraction=float(args.random_av_fraction),
        epsilon=float(args.epsilon),
        decision_steps=int(cfg.get("EPISODE_STEPS", "60") or 60),
        av_candidate_source=str(cfg.get("AV_CANDIDATE_SOURCE", "current_to_dest") or "current_to_dest"),
        safe_reroute_min_remaining_m=float(args.safe_reroute_min_remaining_m),
        completion_drain_steps=int(args.completion_drain_steps),
        completion_drain_patience_steps=int(args.completion_drain_patience_steps),
        sumo_time_to_teleport=str(args.sumo_time_to_teleport or ""),
        sim_buffer_after_last_depart=int(cfg.get("SIM_BUFFER_AFTER_LAST_DEPART", "1000") or 1000),
        incident_edge=incident_edge,
        incident_start=incident_start,
        incident_end=incident_end,
        incident_speed=incident_speed,
        agents_csv=agents_csv,
        policy_ckpt=args.policy_ckpt,
        dagger_beta=float(args.dagger_beta),
        dagger_noise=float(args.dagger_noise),
        edge_registry_path=cfg.get("EDGE_REGISTRY", ""),
        hdv_gawron_lr=0.1,
        hdv_gawron_mode=str(cfg.get("HDV_GAWRON_MODE", "routerl") or "routerl"),
        hdv_gawron_alpha_zero=float(cfg.get("HDV_GAWRON_ALPHA_ZERO", "0.2") or 0.2),
        hdv_gawron_beta=float(cfg.get("HDV_GAWRON_BETA", "-1.5") or -1.5),
        hdv_gawron_update_mode=str(cfg.get("HDV_GAWRON_UPDATE_MODE", "expected") or "expected"),
        od_file=cfg.get("OD_FILE", ""),
        route_registry_path=cfg.get("ROUTE_REGISTRY", ""),
        skip_hdv_route_registry=False,
        use_light_hdv_route_registry=True,
        loaded_hdv_state=loaded_hdv_state,
    )

    all_episodes: List[int] = []
    all_av_edge_actions: List[Any] = []
    all_hdv_edge_actions: List[Any] = []
    all_joint_edge_actions: List[Any] = []
    all_rewards: List[float] = []
    all_flows: List[Any] = []
    all_departs: List[Any] = []
    all_contexts: List[Any] = []

    if post_mutation_state is None and warmup_episodes > 0:
        print(f"\n=== Phase 1: Human adaptation ({warmup_episodes} episodes) ===")
        collector.av_fraction = 0.0
        for ep in range(int(warmup_episodes)):
            (
                _ep_ids,
                _ep_av_edge_acts,
                _ep_hdv_edge_acts,
                _ep_joint_edge_acts,
                _ep_rews,
                ep_frames,
                _ep_departs,
                _ep_contexts,
            ) = collector._collect_single_episode(episode_id=ep, max_steps=int(args.max_steps))
            _update_hdv_after_episode(collector, ep_frames)
            loaded_hdv_state = _refresh_loaded_hdv_state(collector, loaded_hdv_state)
            print(f"  warmup {ep+1:3d}/{warmup_episodes:3d} | {_log_tt_message(collector)}", flush=True)

        if post_mutation_hdv_state_out_path is not None:
            _save_hdv_learning_state(post_mutation_hdv_state_out_path, loaded_hdv_state)

        machine_agent_ids, human_agent_ids = derive_fixed_roles(
            agents_csv=Path(agents_csv),
            ratio_machines=float(cfg.get("RATIO_MACHINES", "0") or 0.0),
            seed=int(args.seed),
        )
        post_mutation_state = build_post_mutation_state(
            settings_path=settings_path,
            forwarded_argv=[
                "collect_urb_fair.py",
                "--settings", settings_path,
                "--results_dir", str(records_dir),
                "--route_generator", str(cfg.get("ROUTE_GENERATOR", "dijkstra") or "dijkstra"),
                "--reuse_artifacts_dir", str(records_dir),
                "--human_learning_episodes", str(warmup_episodes),
                "--disable_human_learning_after_mutation",
            ],
            mutation_episode_index=int(warmup_episodes),
            machine_agent_ids=machine_agent_ids,
            human_agent_ids=human_agent_ids,
            human_learning_state_ref=str(post_mutation_hdv_state_out_path) if post_mutation_hdv_state_out_path is not None else None,
        )
        if post_mutation_state_out_path is not None:
            write_post_mutation_state(post_mutation_state_out_path, post_mutation_state)
    elif post_mutation_state is not None:
        print("\n=== Phase 1: Reusing post-mutation bundle ===")

    if post_mutation_state is None:
        raise RuntimeError("Failed to initialize or load post-mutation state for URB-fair collection.")

    collector.set_fixed_roles(
        machine_ids=[str(x) for x in post_mutation_state.get("machine_agent_ids", [])],
        human_ids=[str(x) for x in post_mutation_state.get("human_agent_ids", [])],
    )
    collector.av_fraction = float(cfg.get("RATIO_MACHINES", "0") or 0.0)

    requested_num_episodes = int(args.num_episodes)
    max_runtime_episodes = int(args.max_runtime_episodes) if int(args.max_runtime_episodes) > 0 else requested_num_episodes
    print(
        f"\n=== Phase 2: Post-mutation collection ({requested_num_episodes} saved episodes target"
        + (f", runtime cap {max_runtime_episodes}" if max_runtime_episodes != requested_num_episodes else "")
        + ") ==="
    )
    interrupted: Optional[BaseException] = None

    def _interrupt_handler(signum, frame):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt(f"signal {signum}")

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _interrupt_handler)
    signal.signal(signal.SIGTERM, _interrupt_handler)
    try:
        runtime_attempt_id = 0
        saved_episode_id = 0
        while saved_episode_id < requested_num_episodes and runtime_attempt_id < max_runtime_episodes:
            runtime_episode_id = int(post_mutation_state.get("mutation_episode_index", warmup_episodes)) + 1 + runtime_attempt_id
            (
                ep_ids,
                ep_av_edge_acts,
                ep_hdv_edge_acts,
                ep_joint_edge_acts,
                ep_rews,
                ep_frames,
                ep_departs,
                ep_contexts,
            ) = collector._collect_single_episode(episode_id=runtime_episode_id, max_steps=int(args.max_steps))
            selection_metrics = _episode_selection_metrics(
                collector,
                ep_frames,
                busy_threshold=float(args.busy_threshold),
            )
            passes_guardrails = _episode_matches_selection(
                selection_metrics,
                min_avg_tt_min=args.min_avg_tt_min,
                max_avg_tt_min=args.max_avg_tt_min,
                min_route_tt_od_count=int(args.min_route_tt_od_count),
                min_route_tt_gap_sec=float(args.min_route_tt_gap_sec),
                max_route_tt_gap_sec=args.max_route_tt_gap_sec,
                min_busy_edge_count_mean=float(args.min_busy_edge_count_mean),
                max_busy_edge_count_mean=args.max_busy_edge_count_mean,
                min_busy_edge_mass_mean=float(args.min_busy_edge_mass_mean),
            )
            selection_score = _episode_selection_score(
                selection_metrics,
                target_avg_tt_min=args.target_avg_tt_min,
                target_route_tt_gap_sec=args.target_route_tt_gap_sec,
                target_busy_edge_count_mean=args.target_busy_edge_count_mean,
                target_busy_edge_mass_mean=args.target_busy_edge_mass_mean,
                avg_tt_tolerance_min=float(args.score_avg_tt_tolerance_min),
                route_tt_gap_tolerance_sec=float(args.score_route_tt_gap_tolerance_sec),
                busy_edge_count_tolerance=float(args.score_busy_edge_count_tolerance),
                busy_edge_mass_tolerance=float(args.score_busy_edge_mass_tolerance),
                avg_tt_weight=float(args.score_avg_tt_weight),
                route_tt_gap_weight=float(args.score_route_tt_gap_weight),
                busy_edge_count_weight=float(args.score_busy_edge_count_weight),
                busy_edge_mass_weight=float(args.score_busy_edge_mass_weight),
            )
            accept = passes_guardrails
            if str(args.selection_mode) == "hybrid_score":
                accept = passes_guardrails and (selection_score >= float(args.selection_score_threshold))
            total_agents = int(getattr(collector, "_num_agents", 0) or 0)
            n_arrived = int(selection_metrics.get("n_arrived", 0.0))
            complete_episode = _episode_all_arrived(collector, selection_metrics)
            arrival_fraction = float(n_arrived) / max(float(total_agents), 1.0)
            enough_arrivals = arrival_fraction >= float(args.min_arrival_fraction)
            if not complete_episode and not bool(args.allow_incomplete_episodes):
                accept = False
            if bool(args.allow_incomplete_episodes) and not enough_arrivals:
                accept = False
            if accept:
                _append_saved_episode(
                    saved_episode_id=saved_episode_id,
                    ep_ids=ep_ids,
                    ep_av_edge_acts=ep_av_edge_acts,
                    ep_hdv_edge_acts=ep_hdv_edge_acts,
                    ep_joint_edge_acts=ep_joint_edge_acts,
                    ep_rews=ep_rews,
                    ep_frames=ep_frames,
                    ep_departs=ep_departs,
                    ep_contexts=ep_contexts,
                    all_episodes=all_episodes,
                    all_av_edge_actions=all_av_edge_actions,
                    all_hdv_edge_actions=all_hdv_edge_actions,
                    all_joint_edge_actions=all_joint_edge_actions,
                    all_rewards=all_rewards,
                    all_flows=all_flows,
                    all_departs=all_departs,
                    all_contexts=all_contexts,
                )
                action = "collect"
                saved_episode_id += 1
            else:
                action = "skip"
            print(
                f"  {action} saved={saved_episode_id:3d}/{requested_num_episodes-1:3d} "
                f"runtime={runtime_attempt_id:3d}/{max_runtime_episodes-1:3d} | "
                f"{_log_tt_message(collector)} | complete={int(complete_episode)} "
                f"({n_arrived}/{total_agents}, frac={arrival_fraction:.3f}) | "
                f"route_ods={int(selection_metrics['route_tt_od_count'])} "
                f"gap={float(selection_metrics['route_tt_gap_mean_sec']):.1f}s "
                f"busy_n={float(selection_metrics['busy_edge_count_mean']):.1f} "
                f"busy_mass={float(selection_metrics['busy_edge_mass_mean']):.3f} "
                f"score={selection_score:.3f}",
                flush=True,
            )
            checkpoint_every = max(0, int(args.checkpoint_every))
            if checkpoint_every > 0 and saved_episode_id > 0 and saved_episode_id % checkpoint_every == 0 and all_episodes:
                collector._save_intermediate(
                    task_name,
                    all_episodes,
                    all_av_edge_actions,
                    all_hdv_edge_actions,
                    all_joint_edge_actions,
                    all_rewards,
                    all_flows,
                    all_departs,
                    all_contexts,
                )
            runtime_attempt_id += 1
    except BaseException as exc:
        interrupted = exc
        print(f"\n[interrupt] collection stopped early: {exc}", flush=True)
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    if interrupted is not None:
        manifest_path = _flush_partial_collection(
            collector=collector,
            task_name=task_name,
            records_dir=records_dir,
            settings_path=settings_path,
            output_dir=output_dir,
            requested_num_episodes=int(args.num_episodes),
            warmup_episodes=int(warmup_episodes),
            post_mutation_state_path=post_mutation_state_out_path if post_mutation_state_in_path is None else post_mutation_state_in_path,
            hdv_state_in_path=hdv_state_in_path,
            hdv_state_out_path=hdv_state_out_path,
            loaded_hdv_state=loaded_hdv_state,
            all_episodes=all_episodes,
            all_av_edge_actions=all_av_edge_actions,
            all_hdv_edge_actions=all_hdv_edge_actions,
            all_joint_edge_actions=all_joint_edge_actions,
            all_rewards=all_rewards,
            all_flows=all_flows,
            all_departs=all_departs,
            all_contexts=all_contexts,
        )
        if manifest_path is not None:
            print(f"\nURB-fair partial collection manifest: {manifest_path}")
        return 130

    if hdv_state_out_path is not None:
        _save_hdv_learning_state(hdv_state_out_path, loaded_hdv_state)

    if not all_episodes:
        manifest_path = _write_collection_manifest(
            records_dir=records_dir,
            settings_path=settings_path,
            output_dir=output_dir,
            task_name=task_name,
            num_episodes=requested_num_episodes,
            warmup_episodes=int(warmup_episodes),
            post_mutation_state_path=post_mutation_state_out_path if post_mutation_state_in_path is None else post_mutation_state_in_path,
            hdv_state_in_path=hdv_state_in_path,
            hdv_state_out_path=hdv_state_out_path,
            partial=True,
            episodes_saved=0,
        )
        raise RuntimeError(
            "No complete post-mutation episodes were collected; "
            f"not writing an empty dataset. Manifest: {manifest_path}"
        )

    collector._save_final_data(
        task_name,
        all_episodes,
        all_av_edge_actions,
        all_hdv_edge_actions,
        all_joint_edge_actions,
        all_rewards,
        all_flows,
        all_departs,
        all_contexts,
    )
    manifest_path = _write_collection_manifest(
        records_dir=records_dir,
        settings_path=settings_path,
        output_dir=output_dir,
        task_name=task_name,
        num_episodes=requested_num_episodes,
        warmup_episodes=int(warmup_episodes),
        post_mutation_state_path=post_mutation_state_out_path if post_mutation_state_in_path is None else post_mutation_state_in_path,
        hdv_state_in_path=hdv_state_in_path,
        hdv_state_out_path=hdv_state_out_path,
        partial=(saved_episode_id < requested_num_episodes),
        episodes_saved=saved_episode_id,
    )
    print(f"\nURB-fair collection manifest: {manifest_path}")
    if saved_episode_id < requested_num_episodes:
        print(
            f"[warn] Saved only {saved_episode_id}/{requested_num_episodes} episodes "
            f"before hitting runtime cap {max_runtime_episodes}.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
