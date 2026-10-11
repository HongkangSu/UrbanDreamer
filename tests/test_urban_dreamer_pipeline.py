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
            codec_pretraining_epochs=2,
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
    validation_calls: int = 0
    codec_ready_after: int | None = 2

    def validate_pretraining_data(self) -> None:
        self.validation_calls += 1

    def initialize_codec_epoch(self, epoch_index: int) -> dict[str, Any]:
        return self._record(f'codec:{epoch_index}')

    def codec_ready(self) -> bool:
        return (self.codec_ready_after is not None
                and sum(event.startswith('codec:') for event in self.events) >= self.codec_ready_after)

    def validate_codec_ready(self) -> None:
        if not self.codec_ready():
            raise ValueError('codec held-out quality thresholds have not been met')

    def initialize_world_model_epoch(self, step_index: int) -> dict[str, Any]:
        return self._record(f"pretrain:{step_index}")

    def train_online_episode(self, episode_index: int) -> dict[str, Any]:
        return self._record(f"online_training:{episode_index}")

    def evaluate_episode(self, episode_index: int) -> dict[str, Any]:
        return self._record(f"final_evaluation:{episode_index}")

    def state_dict(self) -> dict[str, Any]:
        return {
            "reward_contract": "event_q_counterfactual_global_individual_timeout_v2",
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
        'codec:0',
        'codec:1',
        "pretrain:0",
        "pretrain:1",
        "pretrain:2",
        "online_training:0",
        "online_training:1",
        "online_training:2",
        "final_evaluation:0",
        "final_evaluation:1",
    ]
    assert trainer.closed is True
    assert trainer.validation_calls == 1
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


def test_missing_offline_data_rejects_before_initial_checkpoint_or_collection(tmp_path):
    config = _config(tmp_path)
    trainer = _FakeTrainer()
    def reject():
        raise ValueError('offline pretraining requires compatible archives')
    trainer.validate_pretraining_data = reject
    with pytest.raises(ValueError, match='offline pretraining'):
        MainlinePipeline(config=config, input_hash='input-hash',
                         trainer_factory=lambda: trainer).run()
    assert not trainer.events
    assert trainer.closed
    assert not (config.run.output_dir / 'checkpoints/latest.pt').exists()


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
        if event.startswith("pretrain:")
    ] == [f"pretrain:{index}" for index in range(8)]


def test_resume_uses_only_latest_cursor_and_does_not_repeat_finished_work(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    prior = _FakeTrainer(
        events=[
            'codec:0',
            'codec:1',
            "pretrain:0",
            "pretrain:1",
            "pretrain:2",
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
    assert checkpoint.training_state["metrics"]["events"][-1] == "pretrain:2"
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
                       if phase is CheckpointPhase.PRETRAIN]
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
    first = _FakeTrainer(fail_on='pretrain:1')
    with pytest.raises(RuntimeError, match='pretrain:1'):
        MainlinePipeline(config=config, input_hash='input-hash', trainer_factory=lambda: first).run()
    checkpoint = load_checkpoint(output_dir=config.run.output_dir,
        expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash='input-hash')
    assert checkpoint.phase is CheckpointPhase.PRETRAIN
    assert checkpoint.next_unit == 1
    resumed = _FakeTrainer()
    MainlinePipeline(config=config, input_hash='input-hash', trainer_factory=lambda: resumed).run()
    events = [event for event in resumed.events if event.startswith('pretrain:')]
    assert events == [f'pretrain:{index}' for index in range(3)]


def test_codec_must_pass_quality_gate_before_dynamics_can_train(tmp_path):
    config = _config(tmp_path)
    trainer = _FakeTrainer(codec_ready_after=None)
    with pytest.raises(ValueError, match='codec held-out quality'):
        MainlinePipeline(config=config, input_hash='input-hash',
                         trainer_factory=lambda: trainer).run()
    assert trainer.events == ['codec:0', 'codec:1']
    checkpoint = load_checkpoint(output_dir=config.run.output_dir,
        expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash='input-hash')
    assert checkpoint.phase is CheckpointPhase.CODEC_PRETRAIN
    assert checkpoint.next_unit == 2
    assert trainer.closed
    assert not (config.run.output_dir / 'summary.json').exists()


def test_codec_passes_early_without_consuming_remaining_epoch_budget(tmp_path):
    config = _config(tmp_path)
    config = replace(config, run=replace(config.run, codec_pretraining_epochs=10))
    trainer = _FakeTrainer(codec_ready_after=1)
    result = MainlinePipeline(config=config, input_hash='input-hash',
                             trainer_factory=lambda: trainer).run()
    assert result['status'] == 'completed'
    assert [event for event in trainer.events if event.startswith('codec:')] == ['codec:0']
    assert trainer.events.index('codec:0') < trainer.events.index('pretrain:0')


def test_interrupted_codec_resumes_without_repeating_finished_epoch(tmp_path):
    config = _config(tmp_path)
    first = _FakeTrainer(fail_on='codec:1')
    with pytest.raises(RuntimeError, match='codec:1'):
        MainlinePipeline(config=config, input_hash='input-hash', trainer_factory=lambda: first).run()
    checkpoint = load_checkpoint(output_dir=config.run.output_dir,
        expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash='input-hash')
    assert checkpoint.phase is CheckpointPhase.CODEC_PRETRAIN and checkpoint.next_unit == 1
    resumed = _FakeTrainer()
    MainlinePipeline(config=config, input_hash='input-hash', trainer_factory=lambda: resumed).run()
    assert [event for event in resumed.events if event.startswith('codec:')] == ['codec:0', 'codec:1']


def test_pretrain_only_preserves_online_start_and_normal_resume_runs_online(tmp_path):
    config = _config(tmp_path)
    first = _FakeTrainer()
    result = MainlinePipeline(config=config, input_hash='input-hash', trainer_factory=lambda: first,
                             pretrain_only=True).run()
    expected_pretrain_events = ['codec:0', 'codec:1', 'pretrain:0', 'pretrain:1', 'pretrain:2']
    assert result == {'status': 'pretrained', 'events': expected_pretrain_events}
    checkpoint = load_checkpoint(output_dir=config.run.output_dir,
        expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash='input-hash')
    assert checkpoint.phase is CheckpointPhase.ONLINE_TRAINING and checkpoint.next_unit == 0
    assert (config.run.output_dir / 'pretrain_summary.json').exists()
    assert not (config.run.output_dir / 'summary.json').exists()
    original_bytes = checkpoint.path.read_bytes()
    repeated = _FakeTrainer()
    assert MainlinePipeline(config=config, input_hash='input-hash', trainer_factory=lambda: repeated,
                            pretrain_only=True).run() == result
    assert repeated.events == expected_pretrain_events
    assert checkpoint.path.read_bytes() == original_bytes
    resumed = _FakeTrainer()
    completed = MainlinePipeline(config=config, input_hash='input-hash',
                                trainer_factory=lambda: resumed).run()
    assert completed['status'] == 'completed'
    assert resumed.events == expected_pretrain_events + [
        'online_training:0', 'online_training:1', 'online_training:2',
        'final_evaluation:0', 'final_evaluation:1']


def test_pretrain_resume_checks_codec_readiness_before_first_fm_update(tmp_path):
    config = _config(tmp_path)
    prior = _FakeTrainer()
    save_checkpoint(output_dir=config.run.output_dir, config_fingerprint=config.semantic_fingerprint,
        input_hash='input-hash', phase=CheckpointPhase.PRETRAIN, next_unit=0,
        training_state=prior.state_dict())
    resumed = _FakeTrainer()
    with pytest.raises(ValueError, match='codec held-out quality'):
        MainlinePipeline(config=config, input_hash='input-hash', trainer_factory=lambda: resumed).run()
    assert resumed.events == []


def test_pretrain_only_cannot_relabel_partially_online_run(tmp_path):
    config = _config(tmp_path)
    prior = _FakeTrainer(events=['codec:0', 'codec:1', 'pretrain:0', 'pretrain:1',
                                'pretrain:2', 'online_training:0'])
    save_checkpoint(output_dir=config.run.output_dir, config_fingerprint=config.semantic_fingerprint,
        input_hash='input-hash', phase=CheckpointPhase.ONLINE_TRAINING, next_unit=1,
        training_state=prior.state_dict())
    resumed = _FakeTrainer()
    with pytest.raises(MainlineRunError, match='already|started online training'):
        MainlinePipeline(config=config, input_hash='input-hash', trainer_factory=lambda: resumed,
                         pretrain_only=True).run()
    assert resumed.events == prior.events
