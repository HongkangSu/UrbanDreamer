#!/usr/bin/env python3
"""Add training-cost and tail-latency metrics to the three-city main table.

The existing JSON report is the source of truth.  This script augments every
city/method row with completion-adjusted training cost relative to a matched
pre-CAV reference and with pooled checkpoint-eval tail metrics, then rewrites
the CSV and Markdown mirrors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
REPORT_STEM = "three_city_torchrl_baselines_unified_eval_20260713"
EPISODE_RE = re.compile(r"(?:ep|detailed_sumo_stats_)(\d+)\.(?:csv|xml)$")
CHECKPOINT_RE = re.compile(r"checkpoint_ep(\d+)\.pt$")


@dataclass(frozen=True)
class CitySpec:
    key: str
    role_state: Path
    reference_records: Path
    reference_episode_ids: tuple[int, ...] | None
    reference_complete_count: int
    reference_shard_records: tuple[Path, ...]
    mainline_records: Path
    mainline_min_episode: int
    mainline_complete_count: int
    mainline_per_episode_roles: bool


@dataclass(frozen=True)
class Roles:
    cav: frozenset[str]
    hdv: frozenset[str]

    @property
    def all(self) -> frozenset[str]:
        return self.cav | self.hdv

    def groups(self) -> dict[str, frozenset[str]]:
        return {"all": self.all, "CAV": self.cav, "HDV": self.hdv}


@dataclass(frozen=True)
class EpisodeData:
    episode_id: int
    path: Path
    seen_ids: frozenset[str]
    travel_time_seconds: Mapping[str, float]
    cav_ids: frozenset[str] | None = None
    hdv_ids: frozenset[str] | None = None

    def groups(self, default_roles: Roles) -> dict[str, frozenset[str]]:
        if self.cav_ids is None and self.hdv_ids is None:
            return default_roles.groups()
        if self.cav_ids is None or self.hdv_ids is None:
            raise ValueError(f"partial episode role metadata: {self.path}")
        if self.cav_ids & self.hdv_ids:
            raise ValueError(f"overlapping episode roles: {self.path}")
        if self.cav_ids | self.hdv_ids != default_roles.all:
            raise ValueError(f"episode role cohort mismatch: {self.path}")
        return {"all": default_roles.all, "CAV": self.cav_ids, "HDV": self.hdv_ids}


def _city_specs(root: Path) -> dict[str, CitySpec]:
    gretz_roles = root / (
        "results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/"
        "collect_urb_fair_eval_postmutation_records/post_mutation_state.json"
    )
    gretz_mainline = root / (
        "results/gretz_armainvilliers_current_to_dest_incident_completion100_"
        "stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_"
        "S0b_s500_e2500_v0p1/collect_urb_fair_eval_postmutation_records"
    )
    ing_mainline = root / (
        "results/ingolstadt_custom_actionfix_incident_20260516_185810/"
        "collect_urb_fair_eval_postmutation_records"
    )
    font_mainline = root / (
        "results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_"
        "fastcap_20260609/collect_urb_fair_eval_postmutation_records"
    )
    gretz_reference = root / "results/gretz_strong_incident_matched_prehdv50_20260722"
    gretz_shards = tuple(
        gretz_reference
        / "shards"
        / f"shard_{index:02d}"
        / "collect_urb_fair_eval_postmutation_records"
        for index in range(10)
    )
    return {
        "gretz": CitySpec(
            key="gretz",
            role_state=gretz_roles,
            reference_records=gretz_reference,
            reference_episode_ids=None,
            reference_complete_count=50,
            reference_shard_records=gretz_shards,
            mainline_records=gretz_mainline,
            mainline_min_episode=1,
            mainline_complete_count=50,
            mainline_per_episode_roles=True,
        ),
        "ingolstadt": CitySpec(
            key="ingolstadt",
            role_state=ing_mainline / "post_mutation_state.json",
            reference_records=root
            / "results/ingolstadt_custom_incident_full_20260515T182240Z/collect_urb_fair_eval_postmutation_records",
            reference_episode_ids=tuple(range(151, 201)),
            reference_complete_count=50,
            reference_shard_records=(),
            mainline_records=ing_mainline,
            mainline_min_episode=1,
            mainline_complete_count=50,
            mainline_per_episode_roles=False,
        ),
        "fontenay": CitySpec(
            key="fontenay",
            role_state=font_mainline / "post_mutation_state.json",
            reference_records=font_mainline,
            reference_episode_ids=tuple(range(1, 51)),
            reference_complete_count=50,
            reference_shard_records=(),
            mainline_records=font_mainline,
            mainline_min_episode=51,
            mainline_complete_count=50,
            mainline_per_episode_roles=False,
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_id(path: Path) -> int:
    match = EPISODE_RE.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse episode id from {path}")
    return int(match.group(1))


def _load_roles(path: Path) -> tuple[Roles, dict[str, Any]]:
    state = _load_json(path)
    cav = frozenset(str(value) for value in state.get("machine_agent_ids", []))
    hdv = frozenset(str(value) for value in state.get("human_agent_ids", []))
    if not cav or not hdv or cav & hdv:
        raise ValueError(f"invalid mixed post-mutation roles: {path}")
    return Roles(cav=cav, hdv=hdv), state


def _validate_artifact_hashes(records: Path, state: Mapping[str, Any]) -> None:
    hashes = state.get("artifact_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("post-mutation state has no artifact_hashes")
    names = {
        "agents_csv": "agents.csv",
        "routes_csv": "routes.csv",
        "route_rou_xml": "route.rou.xml",
    }
    for key, filename in names.items():
        path = records / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != str(hashes.get(key, "")):
            raise ValueError(f"artifact hash mismatch for {key}: {records}")


def _validate_pure_hdv_reference_state(records: Path, roles: Roles) -> None:
    state_path = records / "post_mutation_state.json"
    state = _load_json(state_path)
    machine_ids = {str(value) for value in state.get("machine_agent_ids", [])}
    human_ids = {str(value) for value in state.get("human_agent_ids", [])}
    if float(state.get("ratio_machines", -1.0)) != 0.0:
        raise ValueError(f"pre-CAV reference ratio_machines is not zero: {state_path}")
    if machine_ids:
        raise ValueError(f"pre-CAV reference contains machine agents: {state_path}")
    if human_ids != set(roles.all):
        raise ValueError(f"pre-CAV reference human IDs do not match the evaluation cohort: {state_path}")
    _validate_artifact_hashes(records, state)


def _finite_nonnegative(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def _read_xml_episode(
    path: Path,
    roles: Roles,
    *,
    pure_hdv: bool,
    enforce_fixed_roles: bool = True,
) -> EpisodeData:
    root = ET.parse(path).getroot()
    seen: set[str] = set()
    times: dict[str, float] = {}
    observed_cav: set[str] = set()
    observed_hdv: set[str] = set()
    for trip in root.iter("tripinfo"):
        vehicle_id = str(trip.attrib.get("id", ""))
        if not vehicle_id or vehicle_id in seen:
            raise ValueError(f"missing/duplicate tripinfo id in {path}: {vehicle_id!r}")
        if vehicle_id not in roles.all:
            raise ValueError(f"unexpected vehicle id {vehicle_id!r} in {path}")
        seen.add(vehicle_id)
        vehicle_type = str(trip.attrib.get("vType", ""))
        if pure_hdv:
            if vehicle_type.lower() not in {"human", "hdv"}:
                raise ValueError(f"pre-CAV reference contains {vehicle_type!r}: {path}")
        else:
            if vehicle_type == "AV":
                observed_cav.add(vehicle_id)
            elif vehicle_type == "Human":
                observed_hdv.add(vehicle_id)
            else:
                raise ValueError(f"unexpected vehicle type {vehicle_type!r}: {path}")
            expected_type = "AV" if vehicle_id in roles.cav else "Human"
            if enforce_fixed_roles and vehicle_type != expected_type:
                raise ValueError(
                    f"role mismatch for {vehicle_id} in {path}: {vehicle_type!r} != {expected_type!r}"
                )
        arrival = _finite_nonnegative(trip.attrib.get("arrival"))
        duration = _finite_nonnegative(trip.attrib.get("duration"))
        if arrival is not None and duration is not None:
            times[vehicle_id] = duration
    return EpisodeData(
        episode_id=_episode_id(path),
        path=path,
        seen_ids=frozenset(seen),
        travel_time_seconds=times,
        cav_ids=None if pure_hdv else frozenset(observed_cav),
        hdv_ids=None if pure_hdv else frozenset(observed_hdv),
    )


def _read_csv_episode(path: Path, roles: Roles) -> EpisodeData:
    seen: set[str] = set()
    times: dict[str, float] = {}
    observed_cav: set[str] = set()
    observed_hdv: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "travel_time", "vType"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} missing columns {sorted(missing)}")
        for row in reader:
            vehicle_id = str(row["id"])
            if vehicle_id in seen:
                raise ValueError(f"duplicate vehicle id {vehicle_id!r}: {path}")
            if vehicle_id not in roles.all:
                raise ValueError(f"unexpected vehicle id {vehicle_id!r}: {path}")
            seen.add(vehicle_id)
            expected_type = "AV" if vehicle_id in roles.cav else "Human"
            if row["vType"] != expected_type:
                raise ValueError(
                    f"role mismatch for {vehicle_id} in {path}: {row['vType']!r} != {expected_type!r}"
                )
            if row["vType"] == "AV":
                observed_cav.add(vehicle_id)
            else:
                observed_hdv.add(vehicle_id)
            travel_time = _finite_nonnegative(row["travel_time"])
            if travel_time is not None:
                times[vehicle_id] = travel_time
    return EpisodeData(
        episode_id=_episode_id(path),
        path=path,
        seen_ids=frozenset(seen),
        travel_time_seconds=times,
        cav_ids=frozenset(observed_cav),
        hdv_ids=frozenset(observed_hdv),
    )


def _xml_path(records: Path, episode_id: int) -> Path:
    return records / "SUMO_output" / f"detailed_sumo_stats_{episode_id}.xml"


def _read_explicit_reference(spec: CitySpec, roles: Roles) -> list[EpisodeData]:
    if spec.reference_episode_ids is None:
        raise ValueError("explicit reference ids are not configured")
    episodes = [
        _read_xml_episode(_xml_path(spec.reference_records, episode_id), roles, pure_hdv=True)
        for episode_id in spec.reference_episode_ids
    ]
    if len(episodes) != spec.reference_complete_count:
        raise ValueError(
            f"pre-CAV reference: found {len(episodes)} episodes, "
            f"expected {spec.reference_complete_count}"
        )
    return episodes


def _read_complete_xml_episodes(
    records: Path,
    roles: Roles,
    *,
    minimum_episode: int,
    count: int,
    pure_hdv: bool,
    enforce_fixed_roles: bool = True,
) -> list[EpisodeData]:
    paths = sorted(
        (records / "SUMO_output").glob("detailed_sumo_stats_*.xml"),
        key=_episode_id,
    )
    selected: list[EpisodeData] = []
    for path in paths:
        if _episode_id(path) < minimum_episode:
            continue
        try:
            episode = _read_xml_episode(
                path,
                roles,
                pure_hdv=pure_hdv,
                enforce_fixed_roles=enforce_fixed_roles,
            )
        except ET.ParseError:
            # A concurrently running SUMO process may still be writing its XML.
            continue
        if len(episode.travel_time_seconds) == len(roles.all):
            selected.append(episode)
            if len(selected) == count:
                break
    _require_complete(selected, roles, count, "complete XML episode selection")
    return selected


def _read_sharded_reference(spec: CitySpec, roles: Roles) -> list[EpisodeData]:
    selected: list[EpisodeData] = []
    per_shard = spec.reference_complete_count // len(spec.reference_shard_records)
    if per_shard * len(spec.reference_shard_records) != spec.reference_complete_count:
        raise ValueError("reference shard count does not divide the requested episodes")
    for shard_index, records in enumerate(spec.reference_shard_records):
        paths = sorted(
            (records / "SUMO_output").glob("detailed_sumo_stats_*.xml"),
            key=_episode_id,
        )
        accepted = 0
        for path in paths:
            try:
                episode = _read_xml_episode(path, roles, pure_hdv=True)
            except ET.ParseError:
                continue
            completion = len(episode.travel_time_seconds) / len(roles.all)
            if completion < 0.90:
                continue
            selected.append(
                EpisodeData(
                    episode_id=shard_index * per_shard + accepted + 1,
                    path=episode.path,
                    seen_ids=episode.seen_ids,
                    travel_time_seconds=episode.travel_time_seconds,
                )
            )
            accepted += 1
            if accepted == per_shard:
                break
        if accepted != per_shard:
            raise ValueError(
                f"reference shard {shard_index:02d}: found {accepted} accepted episodes, "
                f"expected {per_shard}"
            )
    return selected


def _require_complete(
    episodes: Sequence[EpisodeData], roles: Roles, expected_count: int, label: str
) -> None:
    if len(episodes) != expected_count:
        raise ValueError(f"{label}: found {len(episodes)} episodes, expected {expected_count}")
    for episode in episodes:
        if episode.seen_ids != roles.all:
            raise ValueError(f"{label}: ID coverage mismatch in {episode.path}")
        if set(episode.travel_time_seconds) != set(roles.all):
            raise ValueError(f"{label}: incomplete arrivals in {episode.path}")


def _reference_group_stats(
    episodes: Sequence[EpisodeData], groups: Mapping[str, frozenset[str]]
) -> tuple[dict[str, float], dict[str, float]]:
    adjusted_means: dict[str, list[float]] = {group: [] for group in groups}
    completions: dict[str, list[float]] = {group: [] for group in groups}
    for episode in episodes:
        for group, ids in groups.items():
            if not ids:
                raise ValueError(f"empty reference group {group}: {episode.path}")
            arrived = [
                episode.travel_time_seconds[vehicle_id]
                for vehicle_id in ids
                if vehicle_id in episode.travel_time_seconds
            ]
            completion = len(arrived) / len(ids)
            if not arrived or completion <= 0.0:
                raise ValueError(f"no arrived reference vehicles for {group}: {episode.path}")
            adjusted_means[group].append(statistics.fmean(arrived) / completion)
            completions[group].append(completion)
    means = {
        group: statistics.fmean(values) for group, values in adjusted_means.items()
    }
    mean_completions = {
        group: statistics.fmean(values) for group, values in completions.items()
    }
    return means, mean_completions


def _reference_stats(
    episodes: Sequence[EpisodeData], roles: Roles
) -> tuple[dict[str, float], dict[str, Any]]:
    means, mean_completions = _reference_group_stats(episodes, roles.groups())
    expected_cells = len(episodes) * len(roles.all)
    seen_cells = sum(len(episode.seen_ids & roles.all) for episode in episodes)
    arrived_cells = sum(len(set(episode.travel_time_seconds) & roles.all) for episode in episodes)
    audit = {
        "vehicle_episode_coverage": seen_cells / expected_cells,
        "arrival_completion_rate": arrived_cells / expected_cells,
        "mean_completion_rate": mean_completions,
        "all_reference_episodes_complete": all(
            set(episode.travel_time_seconds) == set(roles.all) for episode in episodes
        ),
    }
    return means, audit


def _training_cost(
    episodes: Sequence[EpisodeData],
    roles: Roles,
    reference_seconds: Mapping[str, float],
    *,
    reference_episodes: Sequence[EpisodeData] | None = None,
) -> dict[str, Any]:
    fixed_groups = roles.groups()
    costs: dict[str, list[float]] = {group: [] for group in fixed_groups}
    completions: dict[str, list[float]] = {group: [] for group in fixed_groups}
    references_used: dict[str, list[float]] = {group: [] for group in fixed_groups}
    fixed_key = (roles.cav, roles.hdv)
    reference_cache: dict[
        tuple[frozenset[str], frozenset[str]], dict[str, float]
    ] = {fixed_key: {group: float(value) for group, value in reference_seconds.items()}}
    bundle_keys: set[tuple[frozenset[str], frozenset[str]]] = set()
    cav_bundle_sizes: list[int] = []
    hdv_bundle_sizes: list[int] = []
    consistency_gaps: list[float] = []
    expected_cells = len(episodes) * len(roles.all)
    seen_cells = sum(len(episode.seen_ids & roles.all) for episode in episodes)
    arrived_cells = sum(len(set(episode.travel_time_seconds) & roles.all) for episode in episodes)
    for episode in episodes:
        groups = episode.groups(roles)
        bundle_key = (groups["CAV"], groups["HDV"])
        bundle_keys.add(bundle_key)
        cav_bundle_sizes.append(len(groups["CAV"]))
        hdv_bundle_sizes.append(len(groups["HDV"]))
        if bundle_key not in reference_cache:
            if reference_episodes is None:
                raise ValueError(
                    f"episode-specific role bundle needs matched reference episodes: {episode.path}"
                )
            bundle_reference, _ = _reference_group_stats(reference_episodes, groups)
            reference_cache[bundle_key] = bundle_reference
        bundle_reference = reference_cache[bundle_key]
        episode_cost: dict[str, float] = {}
        for group, ids in groups.items():
            if not ids:
                raise ValueError(f"empty training group {group}: {episode.path}")
            arrived = [
                episode.travel_time_seconds[vehicle_id]
                for vehicle_id in ids
                if vehicle_id in episode.travel_time_seconds
            ]
            completion = len(arrived) / len(ids)
            if not arrived or completion <= 0.0:
                raise ValueError(f"no arrived vehicles for {group} in {episode.path}")
            adjusted_mean = statistics.fmean(arrived) / completion
            value = adjusted_mean - float(bundle_reference[group])
            costs[group].append(value)
            episode_cost[group] = value
            completions[group].append(completion)
            references_used[group].append(float(bundle_reference[group]))
        weighted_group_cost = (
            len(groups["CAV"]) * episode_cost["CAV"]
            + len(groups["HDV"]) * episode_cost["HDV"]
        ) / len(groups["all"])
        consistency_gaps.append(episode_cost["all"] - weighted_group_cost)
    mean_cost_seconds = {
        group: statistics.fmean(values) for group, values in costs.items()
    }
    per_episode_alignment = any(bundle_key != fixed_key for bundle_key in bundle_keys)
    return {
        "c_seconds": mean_cost_seconds,
        "c_minutes": {group: value / 60.0 for group, value in mean_cost_seconds.items()},
        "mean_completion_rate": {
            group: statistics.fmean(values) for group, values in completions.items()
        },
        "episodes": len(episodes),
        "episode_ids": [episode.episode_id for episode in episodes],
        "vehicle_episode_coverage": seen_cells / expected_cells,
        "arrival_completion_rate": arrived_cells / expected_cells,
        "role_alignment": (
            "per_episode_vType_bundle" if per_episode_alignment else "fixed_post_mutation_state"
        ),
        "role_bundles": len(bundle_keys),
        "role_bundle_sizes": {
            "CAV": {"min": min(cav_bundle_sizes), "max": max(cav_bundle_sizes)},
            "HDV": {"min": min(hdv_bundle_sizes), "max": max(hdv_bundle_sizes)},
        },
        "reference_t_pre_minutes": {
            group: statistics.fmean(values) / 60.0
            for group, values in references_used.items()
        },
        "reference_t_pre_range_minutes": {
            group: {"min": min(values) / 60.0, "max": max(values) / 60.0}
            for group, values in references_used.items()
        },
        "weighted_group_consistency_gap_minutes": statistics.fmean(consistency_gaps)
        / 60.0,
    }


def _linear_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty data")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _tail_metrics(episodes: Sequence[EpisodeData], roles: Roles) -> dict[str, Any]:
    values = [
        travel_time
        for episode in episodes
        for vehicle_id, travel_time in episode.travel_time_seconds.items()
        if vehicle_id in roles.all
    ]
    expected_rows = len(episodes) * len(roles.all)
    return {
        "p95_minutes": _linear_percentile(values, 95.0) / 60.0,
        "tt_max_minutes": max(values) / 60.0,
        "arrived_rows": len(values),
        "expected_rows": expected_rows,
        "scope": "arrived_only" if len(values) < expected_rows else "all_agents",
        "right_censored": len(values) < expected_rows,
        "episode_ids": [episode.episode_id for episode in episodes],
    }


def _eval_episodes(row: Mapping[str, Any], roles: Roles, root: Path) -> list[EpisodeData]:
    raw_dir = root / str(row["source"]["raw_episode_dir"])
    episodes = [
        _read_csv_episode(raw_dir / f"ep{int(episode_id)}.csv", roles)
        for episode_id in row["episode_ids"]
    ]
    actual = [episode.episode_id for episode in episodes]
    expected = [int(value) for value in row["episode_ids"]]
    if actual != expected:
        raise ValueError(f"eval episode id mismatch for {row['city']}/{row['method']}")
    return episodes


def _torchrl_training_episodes(
    row: Mapping[str, Any], roles: Roles, root: Path
) -> tuple[list[EpisodeData], Path, list[Path]]:
    source = row["source"]
    config_path = Path(str(source.get("config", "")))
    if not config_path.is_absolute():
        config_path = root / config_path
    config = _load_json(config_path)
    checkpoint_text = str(config.get("resume_checkpoint") or source.get("checkpoint") or "")
    if not checkpoint_text:
        raise ValueError(f"missing evaluated checkpoint for {row['city']}/{row['method']}")
    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    match = CHECKPOINT_RE.search(checkpoint.name)
    if match is None:
        raise ValueError(f"cannot infer training horizon from {checkpoint}")
    last_episode = int(match.group(1))
    method = str(row["method"])

    def load_chain(current_checkpoint: Path) -> tuple[list[EpisodeData], list[Path]]:
        current_match = CHECKPOINT_RE.search(current_checkpoint.name)
        if current_match is None:
            raise ValueError(f"cannot infer training horizon from {current_checkpoint}")
        current_last = int(current_match.group(1))
        method_root = current_checkpoint.parent.parent
        episode_dir = method_root / "collector_runtime" / "episodes"
        available = sorted(
            (
                _episode_id(path)
                for path in episode_dir.glob("ep*.csv")
                if _episode_id(path) <= current_last
            )
        )
        if not available:
            raise ValueError(f"no training episodes for checkpoint {current_checkpoint}")
        local_first = available[0]
        expected_local = list(range(local_first, current_last + 1))
        if available != expected_local:
            raise ValueError(f"non-contiguous local training episodes in {episode_dir}")

        prefix: list[EpisodeData] = []
        source_dirs: list[Path] = []
        if local_first > 1:
            train_config = _load_json(method_root / f"toy_{method}_torchrl_config.json")
            resume_text = str(train_config.get("resume_checkpoint") or "")
            if not resume_text:
                raise ValueError(f"missing resume checkpoint for segment starting at {local_first}")
            resume_checkpoint = Path(resume_text)
            if not resume_checkpoint.is_absolute():
                resume_checkpoint = root / resume_checkpoint
            resume_match = CHECKPOINT_RE.search(resume_checkpoint.name)
            if resume_match is None or int(resume_match.group(1)) != local_first - 1:
                raise ValueError(
                    f"resume chain mismatch for {method_root}: expected ep{local_first - 1}"
                )
            prefix, source_dirs = load_chain(resume_checkpoint)

        local = [
            _read_csv_episode(episode_dir / f"ep{episode_id}.csv", roles)
            for episode_id in expected_local
        ]
        return prefix + local, source_dirs + [episode_dir]

    episodes, source_dirs = load_chain(checkpoint)
    actual_ids = [episode.episode_id for episode in episodes]
    if actual_ids != list(range(1, last_episode + 1)):
        raise ValueError(f"incomplete resume chain for {checkpoint}")
    return episodes, checkpoint, source_dirs


def _augment(payload: dict[str, Any], root: Path) -> None:
    specs = _city_specs(root)
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 18:
        raise ValueError("main table must contain exactly 18 result rows")

    reference_audit: dict[str, Any] = {}
    city_context: dict[str, tuple[Roles, CitySpec, dict[str, float], list[EpisodeData]]] = {}
    for city, spec in specs.items():
        roles, role_state = _load_roles(spec.role_state)
        reference_records = spec.reference_shard_records or (spec.reference_records,)
        if spec.reference_shard_records:
            for records in reference_records:
                _validate_pure_hdv_reference_state(records, roles)
        else:
            for records in reference_records:
                _validate_artifact_hashes(records, role_state)
        _validate_artifact_hashes(spec.mainline_records, role_state)
        if spec.reference_shard_records:
            reference_episodes = _read_sharded_reference(spec, roles)
        elif spec.reference_episode_ids is not None:
            reference_episodes = _read_explicit_reference(spec, roles)
        else:
            reference_episodes = _read_complete_xml_episodes(
                spec.reference_records,
                roles,
                minimum_episode=1,
                count=spec.reference_complete_count,
                pure_hdv=True,
            )
        reference_seconds, reference_stats = _reference_stats(reference_episodes, roles)
        city_context[city] = (roles, spec, reference_seconds, reference_episodes)
        reference_audit[city] = {
            "definition": "matched_pre_CAV_all_HDV",
            "records": [_relative(records, root) for records in reference_records],
            "role_state": _relative(spec.role_state, root),
            "episode_ids": [episode.episode_id for episode in reference_episodes],
            "source_episode_ids": [_episode_id(episode.path) for episode in reference_episodes],
            "episode_files": [_relative(episode.path, root) for episode in reference_episodes],
            "episodes": len(reference_episodes),
            "expected_agents": len(roles.all),
            "future_CAV_ids": len(roles.cav),
            "future_HDV_ids": len(roles.hdv),
            "t_pre_all_minutes": reference_seconds["all"] / 60.0,
            "t_pre_CAV_minutes": reference_seconds["CAV"] / 60.0,
            "t_pre_HDV_minutes": reference_seconds["HDV"] / 60.0,
            "artifact_hashes_verified": True,
            "pure_HDV_role_state_verified": bool(spec.reference_shard_records),
            "pure_HDV_tripinfo_vType_verified": True,
            **reference_stats,
        }

    seen_rows: set[tuple[str, str]] = set()
    for row in results:
        city = str(row["city"])
        method = str(row["method"])
        identity = (city, method)
        if identity in seen_rows:
            raise ValueError(f"duplicate result row: {identity}")
        seen_rows.add(identity)
        roles, spec, reference_seconds, reference_episodes = city_context[city]
        eval_episodes = _eval_episodes(row, roles, root)
        tail = _tail_metrics(eval_episodes, roles)

        if method == "mainline":
            training_episodes = _read_complete_xml_episodes(
                spec.mainline_records,
                roles,
                minimum_episode=spec.mainline_min_episode,
                count=spec.mainline_complete_count,
                pure_hdv=False,
                enforce_fixed_roles=not spec.mainline_per_episode_roles,
            )
            source_phase = "real_environment_collection"
            source_paths = [
                _relative(spec.mainline_records / "SUMO_output", root),
                _relative(spec.mainline_records / "urb_fair_collection_manifest.json", root),
            ]
            source_checkpoint = None
        elif method in {"ippo", "mappo", "iql"}:
            training_episodes, checkpoint, episode_dirs = _torchrl_training_episodes(
                row, roles, root
            )
            source_phase = "online_training"
            source_paths = [_relative(episode_dir, root) for episode_dir in episode_dirs]
            source_checkpoint = _relative(checkpoint, root)
        else:
            training_episodes = eval_episodes
            source_phase = "fixed_policy_eval_proxy"
            proxy_path = Path(str(row["source"]["raw_episode_dir"]))
            if not proxy_path.is_absolute():
                proxy_path = root / proxy_path
            source_paths = [_relative(proxy_path, root)]
            source_checkpoint = None

        cost = _training_cost(
            training_episodes,
            roles,
            reference_seconds,
            reference_episodes=reference_episodes,
        )
        per_episode_roles = cost["role_alignment"] == "per_episode_vType_bundle"
        cost.update(
            {
                "definition": "completion_adjusted_excess_vs_matched_pre_CAV",
                "formula": (
                    "mean_episode(mean_arrived_travel_time_g / completion_rate_g "
                    "- matched_t_pre_g_for_episode_role_bundle)"
                ),
                "unit": "minutes",
                "source_phase": source_phase,
                "source_paths": source_paths,
                "source_checkpoint": source_checkpoint,
                "reference_city": city,
                "reference_alignment": (
                    "per_episode_vehicle_ID_bundle"
                    if per_episode_roles
                    else "fixed_post_mutation_vehicle_ID_groups"
                ),
                "role_state": _relative(spec.role_state, root),
            }
        )
        tail.update(
            {
                "definition": "pooled_vehicle_episode_checkpoint_eval",
                "unit": "minutes",
                "source_path": str(row["source"]["raw_episode_dir"]),
                "missing_travel_time_handling": "right_censored_excluded_not_zero",
            }
        )
        c_minutes = cost["c_minutes"]
        row["c_all"] = c_minutes["all"]
        row["c_CAV"] = c_minutes["CAV"]
        row["c_HDV"] = c_minutes["HDV"]
        row["p95"] = tail["p95_minutes"]
        row["tt_max"] = tail["tt_max_minutes"]
        row["training_cost"] = cost
        row["tail_metrics"] = tail

    expected_identities = {
        (city, method)
        for city in specs
        for method in ("mainline", "ippo", "mappo", "iql", "freeflow_greedy", "live_current_tt")
    }
    if seen_rows != expected_identities:
        raise ValueError(f"unexpected result identities: {seen_rows ^ expected_identities}")

    generated_at = datetime.now(timezone.utc).isoformat()
    payload["generated_at"] = generated_at
    payload["schema_version"] = max(int(payload.get("schema_version", 1)), 2)
    payload["extended_metrics"] = {
        "generated_at": generated_at,
        "training_cost": {
            "reported_fields": ["c_all", "c_CAV", "c_HDV"],
            "unit": "minutes",
            "completion_adjusted": True,
            "group_specific_completion_rates": True,
            "role_alignment": (
                "fixed post-mutation roles, except legacy Gretz UrbanDreamer collection "
                "which uses its observed per-episode vType bundle"
            ),
            "reference": reference_audit,
        },
        "tail_metrics": {
            "reported_fields": ["p95", "tt_max"],
            "unit": "minutes",
            "aggregation": "pooled finite per-vehicle travel times over five checkpoint-eval episodes",
            "incomplete_episode_scope": "arrived_only/right_censored",
            "completion_adjusted": False,
        },
    }
    protocol = payload.setdefault("protocol", {})
    protocol["training_cost_definition"] = (
        "completion-adjusted episode-macro excess travel time versus role-bundle-matched "
        "pre-CAV all-HDV reference"
    )
    protocol["tail_metric_definition"] = (
        "pooled per-vehicle P95 and global maximum over five checkpoint-eval episodes"
    )


CSV_METRIC_FIELDS = [
    "c_all",
    "c_CAV",
    "c_HDV",
    "p95",
    "tt_max",
    "c_source_phase",
    "c_source_episodes",
    "c_vehicle_episode_coverage",
    "c_role_alignment",
    "c_role_bundles",
    "tail_metric_scope",
]


def _write_csv_from_template(
    path: Path, payload: Mapping[str, Any], original_fields: list[str], templates: Mapping[tuple[str, str], dict[str, str]]
) -> None:
    fields = list(original_fields)
    insert_at = fields.index("source_attempt") if "source_attempt" in fields else len(fields)
    for field in reversed(CSV_METRIC_FIELDS):
        if field not in fields:
            fields.insert(insert_at, field)
    lines: list[dict[str, Any]] = []
    for row in payload["results"]:
        identity = (str(row["city"]), str(row["method"]))
        if identity not in templates:
            raise ValueError(f"missing CSV template row: {identity}")
        output = dict(templates[identity])
        for field in ("c_all", "c_CAV", "c_HDV", "p95", "tt_max"):
            output[field] = f"{float(row[field]):.4f}"
        output["c_source_phase"] = row["training_cost"]["source_phase"]
        output["c_source_episodes"] = row["training_cost"]["episodes"]
        output["c_vehicle_episode_coverage"] = (
            f"{float(row['training_cost']['vehicle_episode_coverage']):.4f}"
        )
        output["c_role_alignment"] = row["training_cost"]["role_alignment"]
        output["c_role_bundles"] = row["training_cost"]["role_bundles"]
        output["tail_metric_scope"] = row["tail_metrics"]["scope"]
        lines.append(output)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(lines)
    os.replace(temporary, path)


def _policy_label(row: Mapping[str, Any]) -> str:
    method = row["method"]
    if method == "mainline":
        return "deterministic argmax"
    if method == "freeflow_greedy":
        return "deterministic freeflow"
    if method == "live_current_tt":
        return "deterministic live-TT"
    if method == "iql":
        return "epsilon-greedy (ε=0.05), first_decision_per_vehicle"
    if method == "ippo" and row["city"] == "fontenay":
        return "deterministic argmax"
    return "categorical sample"


def _metric_cell(row: Mapping[str, Any], metric: str) -> str:
    value = row["metrics"][metric]
    marker = "†" if row["metric_scope"] != "all_agents" else ""
    return f"{float(value['mean']):.4f} ± {float(value['std']):.4f}{marker}"


def _completion_cell(row: Mapping[str, Any]) -> str:
    completion = row["completion_rate"]
    return (
        f"{100.0 * float(completion['mean']):.2f}% ± {100.0 * float(completion['std']):.2f}% "
        f"(min {100.0 * float(completion['min']):.2f}%; "
        f"{row['complete_episodes']}/{row['n_episodes']} 完整; "
        f"{row['arrived_all_sum']}/{row['total_agents_sum']} 到达)"
    )


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# 三城市 TorchRL baseline 统一评测主表",
        "",
        f"最近更新：{payload['generated_at']}。",
        "",
        (
            "统一协议：每个方法 5 个 checkpoint-eval episode；HDV post-mutation 状态冻结；"
            "`sim_buffer_after_last_depart=7200`；`max_steps_per_episode=9000`；"
            "`sumo_time_to_teleport=-1`。"
        ),
        "",
        (
            "MAPPO 使用训练时 valid-mask categorical sampling；IQL 使用终局 ε=0.05 的 "
            "valid-mask epsilon-greedy，并保持 `first_decision_per_vehicle` 探索范围。"
        ),
        "",
        (
            "Fontenay 训练使用 blocker60 以避免 junction blocker，但本主表评测仍使用与另外两城相同的"
            "原统一协议，即只设置 `collision.action=warn`，不启用 `ignore-junction-blocker`。"
            "Fontenay MAPPO/IQL 为 200ep checkpoint；Gretz 与 Ingolstadt 为 500ep checkpoint。"
        ),
        "",
        (
            "数值单位均为分钟。t 指标为 episode 等权宏平均 ± 总体标准差（ddof=0）。"
            "P95 为 5 个评测 episode 的逐车 pooled P95，TT max 为全局逐车最大旅行时间。"
        ),
        "",
        (
            "训练代价采用完成率校正代理："
            "`c_g = mean_ep(mean_arrived_tt_g / completion_g - matched_t_pre,g)`，"
            "其中每一组使用自己的完成率，`t_pre,g` 来自同需求、同 incident 的 matched pure-HDV reference。"
        ),
        "",
    ]
    city_order = ["gretz", "ingolstadt", "fontenay"]
    rows_by_city = {
        city: [row for row in payload["results"] if row["city"] == city]
        for city in city_order
    }
    for city in city_order:
        rows = rows_by_city[city]
        lines.extend(
            [
                f"## {rows[0]['city_name']}",
                "",
                "| 方法 | 策略执行 | t_all | t_CAV | t_HDV | c_all (adj.) | c_CAV (adj.) | c_HDV (adj.) | P95 | TT max | Completion | Δt_all vs 主线 | 口径 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            delta = row["metrics"]["t_all"].get("delta_vs_mainline")
            delta_text = "—" if delta is None else f"{float(delta):+.4f}"
            tail_marker = "†" if row["tail_metrics"]["right_censored"] else ""
            scope = "全体车辆" if row["metric_scope"] == "all_agents" else "仅到达车辆†"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["method_name"]),
                        _policy_label(row),
                        _metric_cell(row, "t_all"),
                        _metric_cell(row, "t_CAV"),
                        _metric_cell(row, "t_HDV"),
                        f"{float(row['c_all']):.4f}",
                        f"{float(row['c_CAV']):.4f}",
                        f"{float(row['c_HDV']):.4f}",
                        f"{float(row['p95']):.4f}{tail_marker}",
                        f"{float(row['tt_max']):.4f}{tail_marker}",
                        _completion_cell(row),
                        delta_text,
                        scope,
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## 审计与解释",
            "",
            (
                "- `c_all/c_CAV/c_HDV` 是 completion-adjusted proxy，不等同于把未到达车辆旅行时间设为 0；"
                "CAV 与 HDV 分别使用各自的完成率。"
            ),
            (
                "- UrbanDreamer 的 c 使用真实环境 collection；IPPO/MAPPO/IQL 使用所评 checkpoint 的在线训练轨迹；"
                "两个 Greedy 无学习轨迹，使用统一 5-episode fixed-policy eval proxy。"
            ),
            (
                "- Gretz 的 legacy UrbanDreamer collection 实际按 episode 改变 `vType` 角色 bundle；"
                "该行按每个 episode 的实际 CAV/HDV ID 分组，并用 matched pure-HDV reference 中同一组车辆作基准。"
                "其他行使用 checkpoint 对应的固定 post-mutation 角色。"
            ),
            (
                "- `†` 表示至少一个评测 episode 未达到 100% completion。t、P95 与 TT max 只基于已到达车辆，"
                "属于 right-censored/arrived-only 指标；缺失旅行时间没有当作 0。"
            ),
            "- Δt_all = 方法 t_all − 当前主线 t_all；负值更好。不完整评测存在幸存者偏差，因此不报告 delta。",
            "- P95 与 TT max 不做完成率除法；该操作对分位数与极值没有可解释的统计含义。",
            "- JSON 中保留每行训练 source phase、checkpoint/path、episode IDs、coverage、group completion 和 reference 审计。",
            "",
            "## 选用的 attempt",
            "",
        ]
    )
    for row in payload["results"]:
        lines.append(
            f"- {row['city_name']} / {row['method_name']}：`{row['source']['attempt']}`"
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("experiment_result") / f"{REPORT_STEM}.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    json_path = args.report_json
    if not json_path.is_absolute():
        json_path = root / json_path
    csv_path = json_path.with_suffix(".csv")
    markdown_path = json_path.with_suffix(".md")

    payload = _load_json(json_path)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        original_fields = list(reader.fieldnames or ())
        templates = {
            (str(row["city"]), str(row["method"])): row for row in reader
        }
    if len(templates) != 18:
        raise ValueError("CSV mirror must contain exactly 18 unique rows")

    _augment(payload, root)
    _write_csv_from_template(csv_path, payload, original_fields, templates)
    _atomic_write_text(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    _atomic_write_text(markdown_path, _markdown(payload))
    print(f"updated {json_path}")
    print(f"updated {csv_path}")
    print(f"updated {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
