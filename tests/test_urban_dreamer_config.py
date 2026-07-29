from __future__ import annotations

from pathlib import Path

import pytest

from urban_dreamer.workflow.config import ConfigError, load_config


def _write_config(path: Path, *, extra: str = "") -> Path:
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
"""
        + extra,
        encoding="utf-8",
    )
    return path


def test_load_config_resolves_paths_relative_to_toml(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path / "mainline.toml"))

    assert config.inputs.network_name == "toy_network"
    assert config.inputs.network_dir == (tmp_path / "network").resolve()
    assert config.inputs.route_artifacts_dir == (tmp_path / "routes").resolve()
    assert config.run.output_dir == (tmp_path / "output").resolve()
    assert config.run.device == "cpu"
    assert config.run.online_training_episodes == 4
    assert config.simulation.sumo_extra_args == (
        "--collision.action",
        "warn",
    )


def test_semantic_fingerprint_ignores_paths_output_and_device(tmp_path: Path) -> None:
    first_path = _write_config(tmp_path / "first.toml")
    second_dir = tmp_path / "relocated"
    second_dir.mkdir()
    second_path = _write_config(second_dir / "second.toml")
    second_path.write_text(
        second_path.read_text(encoding="utf-8")
        .replace('output_dir = "output"', 'output_dir = "another-output"')
        .replace('device = "cpu"', 'device = "cuda"'),
        encoding="utf-8",
    )

    first = load_config(first_path)
    second = load_config(second_path)

    assert first.semantic_fingerprint == second.semantic_fingerprint


def test_semantic_fingerprint_changes_with_training_semantics(tmp_path: Path) -> None:
    first = load_config(_write_config(tmp_path / "first.toml"))
    second_path = _write_config(tmp_path / "second.toml")
    second_path.write_text(
        second_path.read_text(encoding="utf-8").replace(
            "online_training_episodes = 4",
            "online_training_episodes = 5",
        ),
        encoding="utf-8",
    )

    second = load_config(second_path)

    assert first.semantic_fingerprint != second.semantic_fingerprint


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "mainline.toml", extra="\n[experiment]\nmode = 'ablation'\n")

    with pytest.raises(ConfigError, match="unknown top-level section"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("warmup_episodes = 2", "warmup_episodes = 0", "warmup_episodes"),
        (
            "incident_end_seconds = 2500.0",
            "incident_end_seconds = 400.0",
            "incident_end_seconds",
        ),
        (
            "world_model_initialization_batch_size = 1",
            "world_model_initialization_batch_size = 0",
            "world_model_initialization_batch_size",
        ),
    ],
)
def test_load_config_rejects_invalid_ranges(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    path = _write_config(tmp_path / "mainline.toml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(field, replacement),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=message):
        load_config(path)
