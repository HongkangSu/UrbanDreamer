from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from urban_dreamer.workflow.checkpoint import (
    CheckpointPhase,
    load_checkpoint,
    save_checkpoint,
)
from urban_dreamer.workflow.config import (
    InputConfig,
    MainlineConfig,
    ResourceConfig,
    RunConfig,
    SimulationConfig,
)
from urban_dreamer.workflow.pipeline import MainlinePipeline, MainlineRunError


def _config(tmp_path: Path) -> MainlineConfig:
    return MainlineConfig(
        inputs=InputConfig(
            network_name="toy",
            network_dir=tmp_path / "network",
        ),
        run=RunConfig(
            output_dir=tmp_path / "output",
            device="cpu",
            seed=7,
            warmup_episodes=2,
            world_model_initialization_epochs=3,
            online_training_episodes=3,
            evaluation_episodes=2,
        ),
        simulation=SimulationConfig(
            world_model_intervals_per_episode=60,
            vehicle_max_speed_mps=40.,
            warmup_horizon_seconds=100,
            warmup_completion_drain_seconds=20,
            warmup_completion_patience_seconds=10,
            episode_wall_timeout_seconds=30.0,
            training_completion_buffer_seconds=5,
            evaluation_horizon_seconds=100,
            incident_edge="edge",
            incident_start_seconds=10.0,
            incident_end_seconds=20.0,
            incident_speed_mps=0.1,
            sumo_extra_args=(),
        ),
        resources=ResourceConfig(
            world_model_initialization_batch_size=1,
            world_model_initialization_sequence_length=2,
            world_model_gradient_accumulation_steps=1,
            online_world_model_batch_size=1,
        ),
        semantic_fingerprint="config-fingerprint",
        source_path=tmp_path / "mainline.toml",
    )


@dataclass
class _FakeTrainer:
    events: list[str] = field(default_factory=list)
    fail_on: str | None = None
    closed: bool = False

    def collect_warmup_episode(self, episode_index: int) -> dict[str, Any]:
        return self._record(f"warmup:{episode_index}")

    def initialize_world_model_epoch(self, step_index: int) -> dict[str, Any]:
        return self._record(f"world_model_initialization:{step_index}")

    def train_online_episode(self, episode_index: int) -> dict[str, Any]:
        return self._record(f"online_training:{episode_index}")

    def evaluate_episode(self, episode_index: int) -> dict[str, Any]:
        return self._record(f"final_evaluation:{episode_index}")

    def state_dict(self) -> dict[str, Any]:
        return {
            "reward_contract": "global_physical_individual_timeout_v1",
            "models": {},
            "optimizers": {},
            "replay": {},
            "rng": {},
            "metrics": {"events": list(self.events)},
            "next_environment_episode": 0,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.events = list(state["metrics"]["events"])

    def summary(self) -> dict[str, Any]:
        return {"events": list(self.events)}

    def close(self) -> None:
        self.closed = True

    def _record(self, event: str) -> dict[str, Any]:
        if self.fail_on == event:
            raise RuntimeError(f"failed at {event}")
        self.events.append(event)
        return {"event": event}


def test_fresh_pipeline_runs_the_only_phase_order_and_marks_completed(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    trainer = _FakeTrainer()

    result = MainlinePipeline(
        config=config,
        input_hash="input-hash",
        trainer_factory=lambda: trainer,
    ).run()

    assert trainer.events == [
        "warmup:0",
        "warmup:1",
        "world_model_initialization:0",
        "world_model_initialization:1",
        "world_model_initialization:2",
        "online_training:0",
        "online_training:1",
        "online_training:2",
        "final_evaluation:0",
        "final_evaluation:1",
    ]
    assert trainer.closed is True
    assert result["status"] == "completed"
    assert result["events"] == trainer.events

    checkpoint = load_checkpoint(
        output_dir=config.run.output_dir,
        expected_config_fingerprint=config.semantic_fingerprint,
        expected_input_hash="input-hash",
    )
    assert checkpoint.phase is CheckpointPhase.COMPLETED
    assert checkpoint.next_unit == 0
    assert checkpoint.training_state["metrics"]["events"] == trainer.events
    assert (config.run.output_dir / "resolved_config.json").is_file()
    assert (config.run.output_dir / "summary.json").is_file()


def test_world_model_initialization_epochs_are_independent_of_accumulation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = replace(
        config,
        run=replace(config.run, world_model_initialization_epochs=8),
        resources=replace(
            config.resources,
            world_model_gradient_accumulation_steps=4,
        ),
    )
    trainer = _FakeTrainer()

    MainlinePipeline(
        config=config,
        input_hash="input-hash",
        trainer_factory=lambda: trainer,
    ).run()

    assert [
        event
        for event in trainer.events
        if event.startswith("world_model_initialization:")
    ] == [f"world_model_initialization:{index}" for index in range(8)]


def test_resume_uses_only_latest_cursor_and_does_not_repeat_finished_work(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    prior = _FakeTrainer(
        events=[
            "warmup:0",
            "warmup:1",
            "world_model_initialization:0",
            "world_model_initialization:1",
            "world_model_initialization:2",
            "online_training:0",
            "online_training:1",
        ]
    )
    save_checkpoint(
        output_dir=config.run.output_dir,
        config_fingerprint=config.semantic_fingerprint,
        input_hash="input-hash",
        phase=CheckpointPhase.ONLINE_TRAINING,
        next_unit=2,
        training_state=prior.state_dict(),
    )
    resumed = _FakeTrainer()

    MainlinePipeline(
        config=config,
        input_hash="input-hash",
        trainer_factory=lambda: resumed,
    ).run()

    assert resumed.events == prior.events + [
        "online_training:2",
        "final_evaluation:0",
        "final_evaluation:1",
    ]
    assert resumed.closed is True


def test_completed_checkpoint_returns_without_constructing_runtime(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    completed = _FakeTrainer(events=["already-complete"])
    save_checkpoint(
        output_dir=config.run.output_dir,
        config_fingerprint=config.semantic_fingerprint,
        input_hash="input-hash",
        phase=CheckpointPhase.COMPLETED,
        next_unit=0,
        training_state=completed.state_dict(),
    )

    result = MainlinePipeline(
        config=config,
        input_hash="input-hash",
        trainer_factory=lambda: pytest.fail("trainer must not be constructed"),
    ).run()

    assert result == {
        "status": "completed",
        "events": ["already-complete"],
    }


def test_nonempty_output_without_valid_latest_checkpoint_fails_closed(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.run.output_dir.mkdir(parents=True)
    (config.run.output_dir / "partial.csv").write_text("partial", encoding="utf-8")

    with pytest.raises(MainlineRunError, match="non-empty"):
        MainlinePipeline(
            config=config,
            input_hash="input-hash",
            trainer_factory=lambda: _FakeTrainer(),
        ).run()


def test_failed_unit_keeps_checkpoint_at_last_safe_boundary(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    trainer = _FakeTrainer(fail_on="online_training:1")

    with pytest.raises(RuntimeError, match="online_training:1"):
        MainlinePipeline(
            config=config,
            input_hash="input-hash",
            trainer_factory=lambda: trainer,
        ).run()

    checkpoint = load_checkpoint(
        output_dir=config.run.output_dir,
        expected_config_fingerprint=config.semantic_fingerprint,
        expected_input_hash="input-hash",
    )
    # Online checkpoints are sparse (every 100 episodes), so the last safe
    # boundary before the failure is the start of the online phase.
    assert checkpoint.phase is CheckpointPhase.ONLINE_TRAINING
    assert checkpoint.next_unit == 0
    assert checkpoint.training_state["metrics"]["events"][-1] == "world_model_initialization:2"
    assert trainer.closed is True


def test_world_model_initialization_checkpoints_each_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    trainer = _FakeTrainer()
    boundaries = []
    original = MainlinePipeline._save_boundary

    def spy(self, trainer, *, phase, next_unit):
        boundaries.append((phase, next_unit))
        original(self, trainer, phase=phase, next_unit=next_unit)

    monkeypatch.setattr(MainlinePipeline, "_save_boundary", spy)
    MainlinePipeline(
        config=config,
        input_hash="input-hash",
        trainer_factory=lambda: trainer,
    ).run()

    init_boundaries = [unit for phase, unit in boundaries
                       if phase is CheckpointPhase.WORLD_MODEL_INITIALIZATION]
    # 0 is the phase-transition boundary; each finished epoch is durable.
    assert init_boundaries == list(range(config.run.world_model_initialization_epochs + 1))


def test_online_training_checkpoints_every_100_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config = replace(config, run=replace(config.run, online_training_episodes=101))
    trainer = _FakeTrainer()
    boundaries = []
    original = MainlinePipeline._save_boundary

    def spy(self, trainer, *, phase, next_unit):
        boundaries.append((phase, next_unit))
        original(self, trainer, phase=phase, next_unit=next_unit)

    monkeypatch.setattr(MainlinePipeline, "_save_boundary", spy)
    MainlinePipeline(
        config=config,
        input_hash="input-hash",
        trainer_factory=lambda: trainer,
    ).run()

    online_boundaries = [unit for phase, unit in boundaries
                         if phase is CheckpointPhase.ONLINE_TRAINING]
    # 0 is the phase-transition boundary; then every 100 episodes plus the last.
    assert online_boundaries == [0, 100, 101]


def test_interrupted_initialization_resumes_from_last_completed_epoch(tmp_path):
    config = _config(tmp_path)
    first = _FakeTrainer(fail_on='world_model_initialization:1')
    with pytest.raises(RuntimeError, match='world_model_initialization:1'):
        MainlinePipeline(config=config, input_hash='input-hash', trainer_factory=lambda: first).run()
    checkpoint = load_checkpoint(output_dir=config.run.output_dir,
        expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash='input-hash')
    assert checkpoint.phase is CheckpointPhase.WORLD_MODEL_INITIALIZATION
    assert checkpoint.next_unit == 1
    resumed = _FakeTrainer()
    MainlinePipeline(config=config, input_hash='input-hash', trainer_factory=lambda: resumed).run()
    events = [event for event in resumed.events if event.startswith('world_model_initialization:')]
    assert events == [f'world_model_initialization:{index}' for index in range(3)]
