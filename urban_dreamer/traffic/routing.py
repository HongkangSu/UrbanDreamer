from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..workflow.config import MainlineConfig


HDV_RESPONSE_CHECKPOINT_FORMAT = (
    "causal_persistence_residual_hdv_od_route_predictor_v4"
)
MAINLINE_ROUTE_COUNT = 4


class InputValidationError(ValueError):
    """Raised when the prepared mainline input bundle is inconsistent."""


@dataclass(frozen=True)
class PreparedInputs:
    network_xml: Path
    network_agents_csv: Path
    od_file: Path
    edge_registry: Path
    route_agents_csv: Path
    routes_csv: Path
    route_xml: Path
    post_mutation_state: Path
    initial_hdv_state: Path
    hdv_response_checkpoint: Path
    automated_vehicle_count: int
    human_vehicle_count: int
    automated_vehicle_fraction: float
    route_count: int
    mutation_episode_index: int
    input_hashes: dict[str, str]


def validate_prepared_inputs(config: MainlineConfig) -> PreparedInputs:
    network_name = config.inputs.network_name
    network_dir = config.inputs.network_dir
    route_dir = config.inputs.route_artifacts_dir
    paths = {
        "network_xml": network_dir / f"{network_name}.net.xml",
        "network_agents": network_dir / "agents.csv",
        "od_file": network_dir / f"od_{network_name}.txt",
        "edge_registry": network_dir / "edge_registry.json",
        "route_agents": route_dir / "agents.csv",
        "routes": route_dir / "routes.csv",
        "route_xml": route_dir / "route.rou.xml",
        "post_mutation_state": config.inputs.post_mutation_state,
        "initial_hdv_state": config.inputs.initial_hdv_state,
        "hdv_response_checkpoint": config.inputs.hdv_response_checkpoint,
    }
    _require_files(paths)

    network_agent_ids = _read_agent_ids(paths["network_agents"])
    route_agent_ids = _read_agent_ids(paths["route_agents"])
    if network_agent_ids != route_agent_ids:
        raise InputValidationError(
            "network and route-artifact agents.csv must contain the same agent IDs"
        )

    post_state = _read_json_object(paths["post_mutation_state"])
    (
        automated_vehicle_count,
        human_vehicle_count,
        automated_vehicle_fraction,
        mutation_episode_index,
    ) = _validate_post_state(
        post_state,
        expected_network=network_name,
        expected_agent_ids=route_agent_ids,
        artifact_paths={
            "agents_csv": paths["route_agents"],
            "routes_csv": paths["routes"],
            "route_rou_xml": paths["route_xml"],
        },
    )
    _validate_initial_hdv_state(paths["initial_hdv_state"])
    _validate_edge_registry(paths["edge_registry"], network_name)
    route_count = _derive_route_count(paths["routes"])
    _validate_hdv_response_checkpoint(paths["hdv_response_checkpoint"])

    input_hashes = {
        name: _sha256(path)
        for name, path in paths.items()
    }
    return PreparedInputs(
        network_xml=paths["network_xml"],
        network_agents_csv=paths["network_agents"],
        od_file=paths["od_file"],
        edge_registry=paths["edge_registry"],
        route_agents_csv=paths["route_agents"],
        routes_csv=paths["routes"],
        route_xml=paths["route_xml"],
        post_mutation_state=paths["post_mutation_state"],
        initial_hdv_state=paths["initial_hdv_state"],
        hdv_response_checkpoint=paths["hdv_response_checkpoint"],
        automated_vehicle_count=automated_vehicle_count,
        human_vehicle_count=human_vehicle_count,
        automated_vehicle_fraction=automated_vehicle_fraction,
        route_count=route_count,
        mutation_episode_index=mutation_episode_index,
        input_hashes=input_hashes,
    )


def _require_files(paths: dict[str, Path]) -> None:
    missing = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise InputValidationError(
            "prepared input bundle is incomplete: " + ", ".join(missing)
        )


def _read_agent_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise InputValidationError(f"{path} must contain an id column")
        agent_ids = [str(row["id"]).strip() for row in reader]
    if not agent_ids or any(not agent_id for agent_id in agent_ids):
        raise InputValidationError(f"{path} must contain non-empty agent IDs")
    if len(agent_ids) != len(set(agent_ids)):
        raise InputValidationError(f"{path} contains duplicate agent IDs")
    return set(agent_ids)


def _validate_post_state(
    state: dict[str, Any],
    *,
    expected_network: str,
    expected_agent_ids: set[str],
    artifact_paths: dict[str, Path],
) -> tuple[int, int, float, int]:
    if state.get("schema_version") != 1:
        raise InputValidationError("post-mutation state schema_version must be 1")
    if state.get("network") != expected_network:
        raise InputValidationError(
            "post-mutation state network does not match [inputs].network_name"
        )
    machine_ids = _string_set(state.get("machine_agent_ids"), "machine_agent_ids")
    human_ids = _string_set(state.get("human_agent_ids"), "human_agent_ids")
    overlap = machine_ids & human_ids
    if overlap:
        raise InputValidationError(
            "post-mutation machine/human agent IDs overlap: "
            + ", ".join(sorted(overlap))
        )
    if machine_ids | human_ids != expected_agent_ids:
        raise InputValidationError(
            "post-mutation machine/human IDs must exactly cover agents.csv"
        )
    raw_fraction = state.get("ratio_machines")
    if isinstance(raw_fraction, bool) or not isinstance(raw_fraction, (int, float)):
        raise InputValidationError("post-mutation ratio_machines must be numeric")
    automated_vehicle_fraction = float(raw_fraction)
    if not 0.0 < automated_vehicle_fraction < 1.0:
        raise InputValidationError("post-mutation ratio_machines must be in (0, 1)")
    expected_machine_count = int(
        len(expected_agent_ids) * automated_vehicle_fraction
    )
    if len(machine_ids) != expected_machine_count:
        raise InputValidationError(
            "post-mutation machine_agent_ids count is inconsistent with ratio_machines"
        )
    if len(human_ids) != len(expected_agent_ids) - expected_machine_count:
        raise InputValidationError(
            "post-mutation human_agent_ids count is inconsistent with ratio_machines"
        )
    mutation_episode_index = state.get("mutation_episode_index")
    if (
        isinstance(mutation_episode_index, bool)
        or not isinstance(mutation_episode_index, int)
        or mutation_episode_index < 0
    ):
        raise InputValidationError(
            "post-mutation mutation_episode_index must be a non-negative integer"
        )
    _validate_artifact_contract(state, artifact_paths)
    return (
        len(machine_ids),
        len(human_ids),
        automated_vehicle_fraction,
        mutation_episode_index,
    )


def _validate_artifact_contract(
    state: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> None:
    refs = state.get("artifact_refs")
    hashes = state.get("artifact_hashes")
    if not isinstance(refs, dict) or set(refs) != set(artifact_paths):
        raise InputValidationError(
            "post-mutation artifact_refs must identify agents_csv,routes_csv,route_rou_xml"
        )
    if not isinstance(hashes, dict) or set(hashes) != set(artifact_paths):
        raise InputValidationError(
            "post-mutation artifact_hashes must identify agents_csv,routes_csv,route_rou_xml"
        )
    for name, artifact_path in artifact_paths.items():
        if not isinstance(refs.get(name), str) or not str(refs[name]).strip():
            raise InputValidationError(
                f"post-mutation artifact_refs.{name} must be a non-empty string"
            )
        if str(hashes.get(name, "")) != _sha256(artifact_path):
            raise InputValidationError(
                f"post-mutation artifact hash mismatch for {name}"
            )


def _validate_initial_hdv_state(path: Path) -> None:
    state = _read_json_object(path)
    if state.get("version") != 1:
        raise InputValidationError(
            "initial HDV route state must use version=1"
        )
    route_policy = state.get("route_policy")
    if not isinstance(route_policy, list) or not route_policy:
        raise InputValidationError(
            "initial HDV route state must contain a non-empty route_policy array"
        )
    seen_ods: set[tuple[int, int]] = set()
    for row in route_policy:
        if not isinstance(row, dict) or set(row) != {
            "origin",
            "destination",
            "probs",
            "costs",
        }:
            raise InputValidationError(
                "initial HDV route_policy rows must contain "
                "origin,destination,probs,costs"
            )
        try:
            od_pair = (int(row["origin"]), int(row["destination"]))
            probabilities = [float(value) for value in row["probs"]]
            costs = [float(value) for value in row["costs"]]
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                "initial HDV route_policy values must be numeric"
            ) from exc
        if od_pair in seen_ods:
            raise InputValidationError(
                f"initial HDV route_policy contains duplicate OD {od_pair}"
            )
        seen_ods.add(od_pair)
        if len(probabilities) != MAINLINE_ROUTE_COUNT or len(costs) != MAINLINE_ROUTE_COUNT:
            raise InputValidationError(
                "initial HDV route_policy must contain exactly four routes per OD"
            )
        if (
            any(not math.isfinite(value) or value < 0.0 for value in probabilities)
            or sum(probabilities) <= 0.0
        ):
            raise InputValidationError(
                "initial HDV route probabilities must be finite, non-negative, "
                "and have positive mass"
            )
        if any(not math.isfinite(value) or value < 0.0 for value in costs):
            raise InputValidationError(
                "initial HDV route costs must be finite and non-negative"
            )


def _string_set(value: Any, label: str) -> set[str]:
    if not isinstance(value, list) or not value:
        raise InputValidationError(f"post-mutation {label} must be a non-empty array")
    values = [str(item).strip() for item in value]
    if any(not item for item in values) or len(values) != len(set(values)):
        raise InputValidationError(
            f"post-mutation {label} must contain unique non-empty IDs"
        )
    return set(values)


def _validate_edge_registry(path: Path, network_name: str) -> None:
    registry = _read_json_object(path)
    edge_ids = registry.get("edge_ids")
    if not isinstance(edge_ids, list) or not edge_ids:
        raise InputValidationError("edge_registry.json must contain non-empty edge_ids")
    if len(edge_ids) != len(set(map(str, edge_ids))):
        raise InputValidationError("edge_registry.json contains duplicate edge_ids")
    if registry.get("network") != network_name:
        raise InputValidationError(
            "edge_registry.json network does not match [inputs].network_name"
        )


def _derive_route_count(path: Path) -> int:
    counts: Counter[tuple[str, str]] = Counter()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"origins", "destinations", "path", "free_flow_time"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise InputValidationError(
                "routes.csv must contain origins,destinations,path,free_flow_time"
            )
        for row in reader:
            if not str(row["path"]).strip():
                raise InputValidationError("routes.csv contains an empty path")
            counts[(str(row["origins"]), str(row["destinations"]))] += 1
    if not counts:
        raise InputValidationError("routes.csv contains no routes")
    route_counts = set(counts.values())
    if len(route_counts) != 1:
        raise InputValidationError(
            "every OD pair in routes.csv must have the same route count"
        )
    route_count = next(iter(route_counts))
    if route_count != MAINLINE_ROUTE_COUNT:
        raise InputValidationError(
            "the online-WM mainline requires exactly four routes per OD pair"
        )
    return route_count


def _validate_hdv_response_checkpoint(path: Path) -> None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise InputValidationError(
            f"cannot load v4 HDV response checkpoint {path}: {exc}"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise InputValidationError("v4 HDV response checkpoint must be a mapping")
    if checkpoint.get("format") != HDV_RESPONSE_CHECKPOINT_FORMAT:
        raise InputValidationError(
            "HDV response checkpoint must use the current v4 residual format"
        )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InputValidationError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
