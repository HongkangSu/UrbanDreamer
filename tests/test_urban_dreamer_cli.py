from __future__ import annotations

import json
from pathlib import Path

import urban_dreamer.__main__ as cli
from urban_dreamer.workflow.config import MainlineConfig


def _write_config(path: Path) -> Path:
    path.write_text(
        """
[inputs]
network_name = "toy"
network_dir = "network"

[run]
output_dir = "output"
device = "cpu"
seed = 42
warmup_episodes = 1
world_model_initialization_steps = 1
online_training_episodes = 1
evaluation_episodes = 1

[simulation]
vehicle_max_speed_mps = 40.0
world_model_intervals_per_episode = 60
warmup_horizon_seconds = 100
warmup_completion_drain_seconds = 20
warmup_completion_patience_seconds = 10
episode_wall_timeout_seconds = 30
training_completion_buffer_seconds = 5
evaluation_horizon_seconds = 100
incident_edge = "edge"
incident_start_seconds = 10
incident_end_seconds = 20
incident_speed_mps = 0.1
sumo_extra_args = []

[resources]
world_model_initialization_batch_size = 1
world_model_initialization_sequence_length = 2
world_model_gradient_accumulation_steps = 1
online_world_model_batch_size = 1
""",
        encoding="utf-8",
    )
    return path


def test_cli_applies_only_output_and_device_overrides(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured: list[MainlineConfig] = []

    def _fake_run(config: MainlineConfig) -> dict[str, object]:
        captured.append(config)
        return {"status": "completed", "episodes": 1}

    monkeypatch.setattr(cli, "run_mainline", _fake_run)
    output_override = tmp_path / "custom-output"

    exit_code = cli.main(
        [
            "--config",
            str(_write_config(tmp_path / "mainline.toml")),
            "--output-dir",
            str(output_override),
            "--device",
            "cuda:1",
        ]
    )

    assert exit_code == 0
    assert len(captured) == 1
    assert captured[0].run.output_dir == output_override.resolve()
    assert captured[0].run.device == "cuda:1"
    assert captured[0].run.seed == 42
    assert json.loads(capsys.readouterr().out) == {
        "episodes": 1,
        "status": "completed",
    }
