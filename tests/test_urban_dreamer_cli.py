from __future__ import annotations

import json
from pathlib import Path

import pytest
import urban_dreamer.__main__ as cli
from urban_dreamer.workflow.config import ConfigError, MainlineConfig


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
world_model_initialization_epochs = 1
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

    def _fake_run(config: MainlineConfig, *, pretrain_only=False) -> dict[str, object]:
        assert not pretrain_only
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


def test_cli_missing_offline_roots_fails_before_network_or_sumo_initialization(tmp_path, monkeypatch):
    import urban_dreamer.workflow.pipeline as pipeline
    monkeypatch.setattr(pipeline, 'validate_prepared_inputs',
                        lambda config: pytest.fail('missing data must be rejected before network loading'))
    monkeypatch.setattr(pipeline, 'load_traffic_topology',
                        lambda prepared: pytest.fail('missing data must be rejected before topology loading'))
    with pytest.raises(ConfigError, match=r'offline pretraining.*archive_roots'):
        cli.main(['--config', str(_write_config(tmp_path / 'mainline.toml'))])
    assert not (tmp_path / 'output').exists()


def test_cli_pretrain_only_flag_stops_before_online(tmp_path, monkeypatch, capsys):
    captured = []
    def run(config, *, pretrain_only=False):
        captured.append(pretrain_only)
        return {'status': 'pretrained'}
    monkeypatch.setattr(cli, 'run_mainline', run)
    assert cli.main(['--config', str(_write_config(tmp_path / 'mainline.toml')),
                     '--pretrain-only']) == 0
    assert captured == [True]
    assert json.loads(capsys.readouterr().out) == {'status': 'pretrained'}


def test_cli_accepts_prepared_cache_as_offline_source_without_archive_roots(tmp_path, monkeypatch):
    import urban_dreamer.workflow.pipeline as pipeline
    path = _write_config(tmp_path / 'mainline.toml')
    path.write_text(path.read_text() + '\n[data]\npretrain_cache_manifest="cache/manifest.json"\n')
    def capture(config):
        assert config.data.archive_roots == ()
        assert config.data.pretrain_cache_manifest == tmp_path / 'cache/manifest.json'
        raise RuntimeError('network validation reached after source configuration check')
    monkeypatch.setattr(pipeline, 'validate_prepared_inputs', capture)
    with pytest.raises(RuntimeError, match='network validation reached'):
        cli.main(['--config', str(path), '--pretrain-only'])
