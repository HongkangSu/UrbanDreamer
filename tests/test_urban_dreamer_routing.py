from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import (
    HDV_RESPONSE_CHECKPOINT_FORMAT,
    InputValidationError,
    validate_prepared_inputs,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_mainline_config(path: Path) -> Path:
    path.write_text(
        """
[inputs]
network_name = "toy_network"
network_dir = "network"
route_artifacts_dir = "routes"
post_mutation_state = "routes/post_mutation_state.json"
initial_hdv_state = "routes/post_mutation_hdv_learning_state.json"
hdv_response_checkpoint = "predictor.pt"

[run]
output_dir = "output"
device = "cpu"
seed = 42
warmup_episodes = 2
world_model_initialization_steps = 3
online_training_episodes = 4
evaluation_episodes = 5

[simulation]
decision_intervals_per_episode = 60
warmup_horizon_seconds = 3600
warmup_completion_drain_seconds = 7200
warmup_completion_patience_seconds = 600
episode_wall_timeout_seconds = 3600.0
training_completion_buffer_seconds = 2000
evaluation_horizon_seconds = 9000
incident_edge = "edge-1"
incident_start_seconds = 500.0
incident_end_seconds = 2500.0
incident_speed_mps = 0.1
sumo_extra_args = ["--collision.action", "warn"]

[resources]
world_model_initialization_batch_size = 1
world_model_initialization_sequence_length = 16
world_model_gradient_accumulation_steps = 32
online_world_model_batch_size = 4
""",
        encoding="utf-8",
    )
    return path


def _write_prepared_inputs(tmp_path: Path) -> Path:
    network_dir = tmp_path / "network"
    route_dir = tmp_path / "routes"
    network_dir.mkdir()
    route_dir.mkdir()
    agents = [
        {"id": "1", "origin": 0, "destination": 1, "start_time": 0, "kind": "Human"},
        {"id": "2", "origin": 0, "destination": 1, "start_time": 1, "kind": "Human"},
    ]
    agent_fields = ["id", "origin", "destination", "start_time", "kind"]
    _write_csv(network_dir / "agents.csv", agent_fields, agents)
    _write_csv(route_dir / "agents.csv", agent_fields, agents)
    _write_csv(
        route_dir / "routes.csv",
        ["origins", "destinations", "path", "free_flow_time"],
        [
            {"origins": 0, "destinations": 1, "path": "a b", "free_flow_time": 1.0},
            {"origins": 0, "destinations": 1, "path": "a c", "free_flow_time": 1.2},
            {"origins": 0, "destinations": 1, "path": "a b", "free_flow_time": 1.0},
            {"origins": 0, "destinations": 1, "path": "a c", "free_flow_time": 1.2},
        ],
    )
    (network_dir / "toy_network.net.xml").write_text("<net/>", encoding="utf-8")
    (network_dir / "od_toy_network.txt").write_text("0 1\n", encoding="utf-8")
    (network_dir / "edge_registry.json").write_text(
        json.dumps(
            {
                "network": "toy_network",
                "edge_ids": ["a", "b", "c"],
            }
        ),
        encoding="utf-8",
    )
    (route_dir / "route.rou.xml").write_text("<routes/>", encoding="utf-8")
    route_artifacts = {
        "agents_csv": route_dir / "agents.csv",
        "routes_csv": route_dir / "routes.csv",
        "route_rou_xml": route_dir / "route.rou.xml",
    }
    (route_dir / "post_mutation_state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "network": "toy_network",
                "ratio_machines": 0.5,
                "mutation_episode_index": 10,
                "machine_agent_ids": ["1"],
                "human_agent_ids": ["2"],
                "artifact_refs": {
                    key: str(value)
                    for key, value in route_artifacts.items()
                },
                "artifact_hashes": {
                    key: hashlib.sha256(value.read_bytes()).hexdigest()
                    for key, value in route_artifacts.items()
                },
            }
        ),
        encoding="utf-8",
    )
    (route_dir / "post_mutation_hdv_learning_state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "route_policy": [
                    {
                        "origin": 0,
                        "destination": 1,
                        "probs": [0.25, 0.25, 0.25, 0.25],
                        "costs": [20.0, 21.0, 22.0, 23.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "format": HDV_RESPONSE_CHECKPOINT_FORMAT,
            "route_model": {},
            "metadata": {},
        },
        tmp_path / "predictor.pt",
    )
    return _write_mainline_config(tmp_path / "mainline.toml")


def test_validate_prepared_inputs_derives_counts_and_hashes(tmp_path: Path) -> None:
    config = load_config(_write_prepared_inputs(tmp_path))

    prepared = validate_prepared_inputs(config)

    assert prepared.automated_vehicle_count == 1
    assert prepared.human_vehicle_count == 1
    assert prepared.automated_vehicle_fraction == 0.5
    assert prepared.route_count == 4
    assert prepared.mutation_episode_index == 10
    assert set(prepared.input_hashes) == {
        "network_xml",
        "network_agents",
        "od_file",
        "edge_registry",
        "route_agents",
        "routes",
        "route_xml",
        "post_mutation_state",
        "initial_hdv_state",
        "hdv_response_checkpoint",
    }
    assert all(len(digest) == 64 for digest in prepared.input_hashes.values())


def test_validate_prepared_inputs_rejects_old_hdv_predictor_format(
    tmp_path: Path,
) -> None:
    config_path = _write_prepared_inputs(tmp_path)
    torch.save(
        {"format": "causal_persistence_gated_hdv_od_route_predictor_v3"},
        tmp_path / "predictor.pt",
    )

    with pytest.raises(InputValidationError, match="v4"):
        validate_prepared_inputs(load_config(config_path))


def test_validate_prepared_inputs_rejects_inconsistent_fixed_roles(
    tmp_path: Path,
) -> None:
    config_path = _write_prepared_inputs(tmp_path)
    state_path = tmp_path / "routes" / "post_mutation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["human_agent_ids"] = ["1", "2"]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(InputValidationError, match="overlap"):
        validate_prepared_inputs(load_config(config_path))


def test_validate_prepared_inputs_rejects_nonuniform_route_count(
    tmp_path: Path,
) -> None:
    config_path = _write_prepared_inputs(tmp_path)
    routes_path = tmp_path / "routes" / "routes.csv"
    with routes_path.open("a", encoding="utf-8") as stream:
        stream.write("1,0,d e,1.0\n")
    state_path = tmp_path / "routes" / "post_mutation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["artifact_hashes"]["routes_csv"] = hashlib.sha256(
        routes_path.read_bytes()
    ).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(InputValidationError, match="same route count"):
        validate_prepared_inputs(load_config(config_path))


def test_validate_prepared_inputs_rejects_uniform_non_k4_routes(
    tmp_path: Path,
) -> None:
    config_path = _write_prepared_inputs(tmp_path)
    routes_path = tmp_path / "routes" / "routes.csv"
    with routes_path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _write_csv(
        routes_path,
        ["origins", "destinations", "path", "free_flow_time"],
        rows[:2],
    )
    state_path = tmp_path / "routes" / "post_mutation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["artifact_hashes"]["routes_csv"] = hashlib.sha256(
        routes_path.read_bytes()
    ).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(InputValidationError, match="exactly four"):
        validate_prepared_inputs(load_config(config_path))


def test_validate_prepared_inputs_rejects_mutated_route_artifact(
    tmp_path: Path,
) -> None:
    config_path = _write_prepared_inputs(tmp_path)
    with (tmp_path / "routes" / "routes.csv").open("a", encoding="utf-8") as stream:
        stream.write("0,1,a d,9.0\n")

    with pytest.raises(InputValidationError, match="artifact hash mismatch.*routes_csv"):
        validate_prepared_inputs(load_config(config_path))


def test_validate_prepared_inputs_rejects_invalid_hdv_learning_state(
    tmp_path: Path,
) -> None:
    config_path = _write_prepared_inputs(tmp_path)
    (tmp_path / "routes" / "post_mutation_hdv_learning_state.json").write_text(
        json.dumps({"version": 2, "agents": []}),
        encoding="utf-8",
    )

    with pytest.raises(InputValidationError, match="HDV route state"):
        validate_prepared_inputs(load_config(config_path))
