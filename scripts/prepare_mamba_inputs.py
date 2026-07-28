#!/usr/bin/env python3
"""Prepare and validate portable inputs for the three-city MAMBA baseline.

The historical settings and post-mutation states contain absolute paths from
the host that produced them.  This helper validates the canonical, checked-in
post-mutation bundles and writes run-local copies whose references point at the
current checkout.  It also provides the launcher's post-smoke validation so a
failed or non-updating smoke run can never promote itself to the 500-episode
run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CitySpec:
    key: str
    network_name: str
    run_tag: str
    settings_rel: str
    bundle_rel: str
    gpu: str
    max_steps: int | None = None
    sumo_extra_args: str = ""


CITY_SPECS: Mapping[str, CitySpec] = {
    "gretz": CitySpec(
        key="gretz",
        network_name="gretz_armainvilliers",
        run_tag=(
            "gretz_armainvilliers_current_to_dest_incident_completion100_"
            "stronglivebad_collisionwarn_drain7200_20260601_"
            "neg_986733117_1_S0b_s500_e2500_v0p1"
        ),
        settings_rel=(
            ".urbandreamer/generated/"
            "gretz_armainvilliers_current_to_dest_incident_completion100_"
            "stronglivebad_collisionwarn_drain7200_20260601_"
            "neg_986733117_1_S0b_s500_e2500_v0p1/urb_fair_eval.env"
        ),
        bundle_rel="networks/gretz_armainvilliers/post_mutation",
        gpu="1",
        sumo_extra_args="--collision.action warn",
    ),
    "ingolstadt": CitySpec(
        key="ingolstadt",
        network_name="ingolstadt_custom",
        run_tag="ingolstadt_custom_current_to_dest_incident_20260528",
        settings_rel=(
            ".urbandreamer/generated/"
            "ingolstadt_custom_current_to_dest_incident_20260528/urb_fair_eval.env"
        ),
        bundle_rel="networks/ingolstadt_custom/post_mutation",
        gpu="1",
    ),
    "fontenay": CitySpec(
        key="fontenay",
        network_name="fontenay_en_parisis",
        run_tag=(
            "fontenay_en_parisis_current_to_dest_both494_incident_"
            "warmup50_fastcap_20260609"
        ),
        settings_rel=(
            ".urbandreamer/generated/"
            "fontenay_en_parisis_current_to_dest_both494_incident_"
            "warmup50_fastcap_20260609/urb_fair_eval.env"
        ),
        bundle_rel="networks/fontenay_en_parisis",
        gpu="0",
        max_steps=3600,
        sumo_extra_args=(
            "--collision.action warn --ignore-junction-blocker 60 "
            "--tripinfo-output.write-unfinished"
        ),
    ),
}

ARTIFACT_FILES: Mapping[str, str] = {
    "agents_csv": "agents.csv",
    "routes_csv": "routes.csv",
    "route_rou_xml": "route.rou.xml",
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw = stripped.split("=", 1)
        key = key.strip()
        try:
            parsed = shlex.split(raw, comments=False, posix=True)
            values[key] = parsed[0] if parsed else ""
        except ValueError:
            values[key] = raw.strip().strip('"').strip("'")
    return values


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def _parse_xml(path: Path, *, expected_root: str, counted_tag: str) -> int:
    """Parse an XML file to EOF and require at least one runtime element."""

    root_tag = ""
    count = 0
    try:
        for event, element in ET.iterparse(path, events=("start", "end")):
            tag = element.tag.rsplit("}", 1)[-1]
            if event == "start" and not root_tag:
                root_tag = tag
            if event == "end":
                if tag == counted_tag:
                    count += 1
                element.clear()
    except ET.ParseError as exc:
        raise ValueError(f"invalid or truncated XML {path}: {exc}") from exc
    if root_tag != expected_root:
        raise ValueError(
            f"unexpected XML root in {path}: {root_tag!r}, expected {expected_root!r}"
        )
    if count <= 0:
        raise ValueError(f"XML {path} contains no <{counted_tag}> elements")
    return count


def _agent_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(f"agents CSV has no id column: {path}")
        ids: list[str] = []
        for row_number, row in enumerate(reader, start=2):
            agent_id = str(row.get("id", "")).strip()
            if not agent_id:
                raise ValueError(f"empty agent id at {path}:{row_number}")
            ids.append(agent_id)
    if not ids:
        raise ValueError(f"agents CSV is empty: {path}")
    unique = set(ids)
    if len(unique) != len(ids):
        raise ValueError(f"agents CSV contains duplicate ids: {path}")
    return unique


def _integer_setting(settings: Mapping[str, str], key: str, settings_path: Path) -> int:
    raw = settings.get(key, "")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key}={raw!r} in {settings_path}") from exc
    if value <= 0:
        raise ValueError(f"{key} must be positive in {settings_path}, got {value}")
    return value


def validate_city_inputs(root: Path, spec: CitySpec) -> dict[str, Any]:
    source_settings = root / spec.settings_rel
    bundle = root / spec.bundle_rel
    network_dir = root / "networks" / spec.network_name
    post_state_path = bundle / "post_mutation_state.json"
    hdv_state_path = bundle / "post_mutation_hdv_learning_state.json"
    net_xml = network_dir / f"{spec.network_name}.net.xml"
    edge_registry = network_dir / "edge_registry.json"
    od_file = network_dir / f"od_{spec.network_name}.txt"

    required = {
        "source settings": source_settings,
        "post-mutation state": post_state_path,
        "HDV learning state": hdv_state_path,
        "network XML": net_xml,
        "edge registry": edge_registry,
        "OD file": od_file,
        "network agents CSV": network_dir / "agents.csv",
        **{f"artifact {key}": bundle / name for key, name in ARTIFACT_FILES.items()},
    }
    for label, path in required.items():
        _require_file(path, label)

    settings = read_env(source_settings)
    if settings.get("NETWORK_NAME") != spec.network_name:
        raise ValueError(
            f"settings network mismatch for {spec.key}: "
            f"{settings.get('NETWORK_NAME')!r} != {spec.network_name!r}"
        )
    state = json.loads(post_state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"post-mutation state must be an object: {post_state_path}")
    if str(state.get("network", "")) != spec.network_name:
        raise ValueError(
            f"post-mutation network mismatch for {spec.key}: {state.get('network')!r}"
        )
    if bool(state.get("should_humans_adapt", False)):
        raise ValueError(f"canonical post-mutation state must be frozen: {post_state_path}")

    machine_list = [str(value) for value in state.get("machine_agent_ids", [])]
    human_list = [str(value) for value in state.get("human_agent_ids", [])]
    machine_ids = set(machine_list)
    human_ids = set(human_list)
    if not machine_ids or not human_ids:
        raise ValueError(f"post-mutation roles must both be non-empty: {post_state_path}")
    if len(machine_ids) != len(machine_list) or len(human_ids) != len(human_list):
        raise ValueError(f"post-mutation role lists contain duplicate IDs: {post_state_path}")
    overlap = machine_ids & human_ids
    if overlap:
        raise ValueError(
            f"machine/human roles overlap in {post_state_path}: {sorted(overlap)[:5]}"
        )

    artifact_ids = _agent_ids(bundle / "agents.csv")
    network_ids = _agent_ids(network_dir / "agents.csv")
    role_ids = machine_ids | human_ids
    if artifact_ids != network_ids:
        raise ValueError(f"network and canonical artifact agent IDs differ for {spec.key}")
    if role_ids != artifact_ids:
        missing = sorted(artifact_ids - role_ids)[:5]
        extra = sorted(role_ids - artifact_ids)[:5]
        raise ValueError(
            f"fixed roles do not cover canonical agents for {spec.key}; "
            f"missing={missing}, extra={extra}"
        )

    n_agents = _integer_setting(settings, "N_AGENTS", source_settings)
    n_av = _integer_setting(settings, "N_AV", source_settings)
    n_hdv = _integer_setting(settings, "N_HDV", source_settings)
    if (n_agents, n_av, n_hdv) != (
        len(artifact_ids),
        len(machine_ids),
        len(human_ids),
    ):
        raise ValueError(
            f"settings/role count mismatch for {spec.key}: "
            f"settings={(n_agents, n_av, n_hdv)}, "
            f"bundle={(len(artifact_ids), len(machine_ids), len(human_ids))}"
        )

    try:
        settings_ratio = float(settings.get("RATIO_MACHINES", "nan"))
        state_ratio = float(state.get("ratio_machines", "nan"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid machine ratio for {spec.key}") from exc
    if not math.isfinite(settings_ratio) or not math.isclose(
        settings_ratio, state_ratio, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"settings/post-state machine ratio mismatch for {spec.key}: "
            f"{settings_ratio!r} != {state_ratio!r}"
        )

    expected_hashes = state.get("artifact_hashes")
    if not isinstance(expected_hashes, dict):
        raise ValueError(f"post-mutation state has no artifact_hashes: {post_state_path}")
    actual_hashes: dict[str, str] = {}
    for key, filename in ARTIFACT_FILES.items():
        artifact = bundle / filename
        actual = sha256(artifact)
        expected = str(expected_hashes.get(key, ""))
        if not expected or actual != expected:
            raise ValueError(
                f"canonical artifact hash mismatch for {spec.key}/{filename}: "
                f"expected={expected!r}, actual={actual!r}"
            )
        actual_hashes[key] = actual

    hdv_state = json.loads(hdv_state_path.read_text(encoding="utf-8"))
    hdv_agents_raw = hdv_state.get("agents") if isinstance(hdv_state, dict) else None
    if not isinstance(hdv_agents_raw, dict):
        raise ValueError(f"HDV learning state has no agents mapping: {hdv_state_path}")
    hdv_agent_ids = {str(value) for value in hdv_agents_raw}
    if not role_ids.issubset(hdv_agent_ids):
        raise ValueError(
            f"HDV learning state does not cover all fixed-role IDs for {spec.key}"
        )

    route_count = _parse_xml(
        bundle / "route.rou.xml", expected_root="routes", counted_tag="route"
    )
    edge_count = _parse_xml(net_xml, expected_root="net", counted_tag="edge")
    # Parse these JSON inputs eagerly as part of preflight, not on the first
    # expensive SUMO episode.
    json.loads(edge_registry.read_text(encoding="utf-8"))

    return {
        "source_settings": source_settings.resolve(),
        "bundle": bundle.resolve(),
        "network_dir": network_dir.resolve(),
        "post_state": post_state_path.resolve(),
        "hdv_state": hdv_state_path.resolve(),
        "state": state,
        "artifact_hashes": actual_hashes,
        "counts": {
            "agents": len(artifact_ids),
            "machines": len(machine_ids),
            "humans": len(human_ids),
            "routes_xml": route_count,
            "network_edges_xml": edge_count,
        },
    }


def _portable_settings_text(
    *, root: Path, spec: CitySpec, validated: Mapping[str, Any]
) -> str:
    source_path = Path(validated["source_settings"])
    source_text = source_path.read_text(encoding="utf-8")
    parsed = read_env(source_path)
    old_network_dir = Path(parsed.get("NETWORK_DIR", ""))
    if old_network_dir.is_absolute() and len(old_network_dir.parents) >= 2:
        old_root = old_network_dir.parents[1]
        source_text = source_text.replace(str(old_root), str(root.resolve()))

    network_dir = Path(validated["network_dir"])
    bundle = Path(validated["bundle"])
    overrides = {
        "NETWORK_DIR": str(network_dir),
        "NETWORK_NAME": spec.network_name,
        "EDGE_REGISTRY": str(network_dir / "edge_registry.json"),
        "AGENTS_CSV": str(network_dir / "agents.csv"),
        "OD_FILE": str(network_dir / f"od_{spec.network_name}.txt"),
        "ROUTE_ARTIFACTS_DIR": str(bundle),
        "AV_CANDIDATE_SOURCE": "current_to_dest",
    }
    pending = dict(overrides)
    output: list[str] = []
    assignment = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=")
    for line in source_text.splitlines():
        match = assignment.match(line)
        if match and match.group(1) in pending:
            key = match.group(1)
            output.append(f"{key}={json.dumps(pending.pop(key))}")
        else:
            output.append(line)
    for key, value in pending.items():
        output.append(f"{key}={json.dumps(value)}")
    return "\n".join(output) + "\n"


def prepare_inputs(
    root: Path, output_root: Path, city_keys: Iterable[str]
) -> dict[str, Any]:
    root = root.resolve()
    output_root = output_root.resolve()
    selected = tuple(city_keys)
    unknown = [key for key in selected if key not in CITY_SPECS]
    if not selected or unknown:
        raise ValueError(
            f"cities must be selected from {tuple(CITY_SPECS)}; unknown={unknown}"
        )

    # Validate every input before writing anything, so a bad city never leaves
    # a deceptively usable partial input set.
    validated = {key: validate_city_inputs(root, CITY_SPECS[key]) for key in selected}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "prepared_at": _now(),
        "repo_root": str(root),
        "cities": {},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for key in selected:
        spec = CITY_SPECS[key]
        info = validated[key]
        city_dir = output_root / key
        city_dir.mkdir(parents=True, exist_ok=True)
        settings_out = city_dir / "settings.env"
        post_state_out = city_dir / "post_mutation_state.json"

        settings_out.write_text(
            _portable_settings_text(root=root, spec=spec, validated=info),
            encoding="utf-8",
        )
        portable_state = dict(info["state"])
        bundle = Path(info["bundle"])
        portable_state["human_learning_state_ref"] = str(info["hdv_state"])
        portable_state["artifact_refs"] = {
            key_name: str((bundle / filename).resolve())
            for key_name, filename in ARTIFACT_FILES.items()
        }
        post_state_out.write_text(
            json.dumps(portable_state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["cities"][key] = {
            "network": spec.network_name,
            "run_tag": spec.run_tag,
            "gpu": spec.gpu,
            "source_settings": str(info["source_settings"]),
            "source_settings_sha256": sha256(Path(info["source_settings"])),
            "source_post_state": str(info["post_state"]),
            "source_post_state_sha256": sha256(Path(info["post_state"])),
            "source_hdv_state": str(info["hdv_state"]),
            "source_hdv_state_sha256": sha256(Path(info["hdv_state"])),
            "canonical_bundle": str(bundle),
            "portable_settings": str(settings_out),
            "portable_post_state": str(post_state_out),
            "artifact_hashes": info["artifact_hashes"],
            "counts": info["counts"],
            "xml_validated_to_eof": True,
            "fixed_role_ids_validated": True,
        }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _read_csv(path: Path, label: str) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty {label}: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or not rows:
        raise ValueError(f"{label} has no data rows: {path}")
    return rows, fields


def _finite_loss_columns(
    rows: Sequence[Mapping[str, str]], fields: Sequence[str], token: str
) -> list[str]:
    matching = [
        field
        for field in fields
        if token in field.lower() and "loss" in field.lower()
    ]
    finite: list[str] = []
    for field in matching:
        for row in rows:
            try:
                value = float(row.get(field, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                finite.append(field)
                break
    return finite


def validate_smoke_outputs(
    outputs: Mapping[str, Path],
    *,
    telemetry_path: Path,
    expected_episodes: int,
    required_gpus: Sequence[str] = ("0", "1"),
    max_gpu_memory_mib: int | None = 20000,
) -> dict[str, Any]:
    errors: list[str] = []
    city_metrics: dict[str, Any] = {}
    for city, raw_output in outputs.items():
        output = raw_output.resolve()
        metrics: dict[str, Any] = {"output_dir": str(output)}
        try:
            episode_rows, _ = _read_csv(
                output / "online_train_episodes.csv", f"{city} episode CSV"
            )
            if len(episode_rows) < int(expected_episodes):
                raise ValueError(
                    f"{city} completed {len(episode_rows)} episodes, "
                    f"expected at least {expected_episodes}"
                )
            metrics["episode_rows"] = len(episode_rows)
        except ValueError as exc:
            errors.append(str(exc))

        try:
            update_rows, update_fields = _read_csv(
                output / "online_train_updates.csv", f"{city} update CSV"
            )
            active_rows: list[dict[str, str]] = []
            for row in update_rows:
                try:
                    skipped = float(row.get("skipped", "0") or 0)
                    pending = float(row.get("pending_update", "0") or 0)
                except (TypeError, ValueError):
                    continue
                if skipped < 0.5 and pending < 0.5:
                    active_rows.append(row)
            if not active_rows:
                errors.append(f"{city} smoke has no non-skipped learner update")
            accepted_actor_rows: list[dict[str, str]] = []
            for row in active_rows:
                try:
                    actor_updated = float(row.get("actor_updated", "0") or 0)
                    actor_skipped = float(row.get("actor_skipped", "1") or 1)
                    accepted_epochs = float(row.get("ppo_epochs_completed", "0") or 0)
                except (TypeError, ValueError):
                    continue
                if (
                    actor_updated > 0.5
                    and actor_skipped < 0.5
                    and accepted_epochs >= 1.0
                ):
                    accepted_actor_rows.append(row)
            if not accepted_actor_rows:
                errors.append(
                    f"{city} smoke has no accepted PPO actor epoch "
                    "(actor_updated=1, actor_skipped=0, ppo_epochs_completed>=1)"
                )
            model_losses = _finite_loss_columns(active_rows, update_fields, "model")
            actor_losses = _finite_loss_columns(
                accepted_actor_rows, update_fields, "actor",
            )
            critic_losses = _finite_loss_columns(active_rows, update_fields, "critic")
            if not model_losses:
                errors.append(f"{city} smoke has no finite world-model loss")
            if not actor_losses:
                errors.append(f"{city} smoke has no finite actor loss")
            if not critic_losses:
                errors.append(f"{city} smoke has no finite critic loss")
            metrics.update(
                {
                    "update_rows": len(update_rows),
                    "active_update_rows": len(active_rows),
                    "accepted_actor_update_rows": len(accepted_actor_rows),
                    "model_loss_columns": sorted(set(model_losses)),
                    "actor_loss_columns": sorted(set(actor_losses)),
                    "critic_loss_columns": sorted(set(critic_losses)),
                }
            )
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))

        required_files = [
            output / "online_train_summary.json",
            output / "online_train_decisions.csv",
            output / "checkpoints" / f"checkpoint_ep{int(expected_episodes):05d}.pt",
        ]
        hdv_candidates = [
            output / "hdv_learning_state_after_online_train.json",
            output / "hdv_learning_state_after_online_wm_train.json",
        ]
        for path in required_files:
            if not path.is_file() or path.stat().st_size <= 0:
                errors.append(f"missing or empty {city} smoke artifact: {path}")
        if not any(path.is_file() and path.stat().st_size > 0 for path in hdv_candidates):
            errors.append(
                f"missing {city} post-training HDV state; expected one of "
                + ", ".join(str(path) for path in hdv_candidates)
            )
        summary_path = output / "online_train_summary.json"
        if summary_path.is_file() and summary_path.stat().st_size > 0:
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                family = str(
                    summary.get("model_family", summary.get("algorithm", "mamba"))
                ).lower()
                if "mamba" not in family:
                    errors.append(f"{city} smoke summary is not MAMBA: {family!r}")
            except (json.JSONDecodeError, AttributeError) as exc:
                errors.append(f"invalid {city} smoke summary JSON: {exc}")
        city_metrics[city] = metrics

    gpu_peaks: dict[str, int] = {}
    try:
        rows, fields = _read_csv(telemetry_path, "GPU telemetry")
        if not {"gpu", "memory_used_mib"}.issubset(fields):
            raise ValueError(
                f"GPU telemetry lacks gpu/memory_used_mib columns: {telemetry_path}"
            )
        for gpu in required_gpus:
            values: list[int] = []
            for row in rows:
                if str(row.get("gpu", "")).strip() != str(gpu):
                    continue
                try:
                    values.append(int(float(row.get("memory_used_mib", ""))))
                except (TypeError, ValueError):
                    continue
            if not values:
                errors.append(f"GPU {gpu} has no valid smoke telemetry samples")
                continue
            peak = max(values)
            gpu_peaks[str(gpu)] = peak
            if max_gpu_memory_mib is not None and peak > int(max_gpu_memory_mib):
                errors.append(
                    f"GPU {gpu} aggregate memory peak {peak} MiB exceeds "
                    f"{int(max_gpu_memory_mib)} MiB"
                )
    except ValueError as exc:
        errors.append(str(exc))

    result = {
        "schema_version": 1,
        "validated_at": _now(),
        "ok": not errors,
        "expected_episodes": int(expected_episodes),
        "max_gpu_memory_mib": (
            int(max_gpu_memory_mib) if max_gpu_memory_mib is not None else None
        ),
        "gpu_peak_memory_mib": gpu_peaks,
        "cities": city_metrics,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("smoke validation failed:\n  - " + "\n  - ".join(errors))
    return result


def _city_keys(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return tuple(CITY_SPECS) if not values or values == ("all",) else values


def _output_mapping(raw_values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"--output must use CITY=PATH, got {raw!r}")
        city, path = raw.split("=", 1)
        city = city.strip()
        if not city or not path.strip() or city in result:
            raise ValueError(f"invalid or duplicate --output value: {raw!r}")
        result[city] = Path(path).expanduser()
    if not result:
        raise ValueError("at least one --output CITY=PATH is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="validate and write portable inputs")
    prepare.add_argument("--repo-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--cities", default="all")

    verify = subparsers.add_parser(
        "verify-smoke", help="validate smoke outputs and aggregate GPU telemetry"
    )
    verify.add_argument("--output", action="append", default=[], metavar="CITY=PATH")
    verify.add_argument("--telemetry", type=Path, required=True)
    verify.add_argument("--expected-episodes", type=int, default=2)
    verify.add_argument("--required-gpus", default="0,1")
    verify.add_argument("--max-gpu-memory-mib", type=int, default=20000)
    verify.add_argument(
        "--skip-memory-limit",
        action="store_true",
        help="still record per-GPU peaks but do not apply a promotion threshold",
    )
    verify.add_argument("--result-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_inputs(
                args.repo_root, args.output_root, _city_keys(str(args.cities))
            )
        else:
            result = validate_smoke_outputs(
                _output_mapping(args.output),
                telemetry_path=args.telemetry,
                expected_episodes=args.expected_episodes,
                required_gpus=tuple(
                    value.strip()
                    for value in str(args.required_gpus).split(",")
                    if value.strip()
                ),
                max_gpu_memory_mib=(
                    None if args.skip_memory_limit else args.max_gpu_memory_mib
                ),
            )
            if args.result_json is not None:
                args.result_json.parent.mkdir(parents=True, exist_ok=True)
                args.result_json.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
