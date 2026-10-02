from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Protocol

from .checkpoint import (
    CheckpointPhase,
    OutputDirectoryDecision,
    decide_output_directory,
    save_checkpoint,
)
from .config import MainlineConfig
from ..traffic.routing import PreparedInputs, validate_prepared_inputs
from ..traffic.topology import TrafficTopology, load_traffic_topology


class MainlineRunError(RuntimeError):
    """Raised when the single supported run cannot safely start or resume."""


# Units between full boundary checkpoints per phase; None saves only at phase end.
_CHECKPOINT_SAVE_INTERVALS: dict[CheckpointPhase, int | None] = {
    CheckpointPhase.WARMUP: 1,
    CheckpointPhase.WORLD_MODEL_INITIALIZATION: 1,
    CheckpointPhase.ONLINE_TRAINING: 100,
    CheckpointPhase.FINAL_EVALUATION: 1,
}


class MainlineTrainerProtocol(Protocol):
    def collect_warmup_episode(self, episode_index: int) -> dict[str, Any]: ...

    def initialize_world_model_epoch(self, epoch_index: int) -> dict[str, Any]: ...

    def train_online_episode(self, episode_index: int) -> dict[str, Any]: ...

    def evaluate_episode(self, episode_index: int) -> dict[str, Any]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def summary(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class MainlinePipeline:
    """Strict phase coordinator for the only supported training lifecycle."""

    def __init__(
        self,
        *,
        config: MainlineConfig,
        input_hash: str,
        trainer_factory: Callable[[], MainlineTrainerProtocol],
    ) -> None:
        self.config = config
        self.input_hash = str(input_hash)
        self._trainer_factory = trainer_factory

    def run(self) -> dict[str, Any]:
        status = decide_output_directory(
            output_dir=self.config.run.output_dir,
            expected_config_fingerprint=self.config.semantic_fingerprint,
            expected_input_hash=self.input_hash,
        )
        if status.decision is OutputDirectoryDecision.NONEMPTY_INVALID:
            raise MainlineRunError(status.reason or "output directory cannot be resumed")
        if status.decision is OutputDirectoryDecision.COMPLETED:
            assert status.checkpoint is not None
            return _completed_summary(
                self.config.run.output_dir,
                status.checkpoint.training_state,
            )

        trainer = self._trainer_factory()
        try:
            if status.decision is OutputDirectoryDecision.VALID_RESUME:
                assert status.checkpoint is not None
                phase = status.checkpoint.phase
                next_unit = status.checkpoint.next_unit
                trainer.load_state_dict(status.checkpoint.training_state)
            else:
                phase = CheckpointPhase.WARMUP
                next_unit = 0
                self._save_boundary(trainer, phase=phase, next_unit=next_unit)
                _atomic_write_json(
                    self.config.run.output_dir / "resolved_config.json",
                    self.config.resolved_snapshot(),
                )

            phase, next_unit = self._run_remaining_phases(
                trainer,
                phase=phase,
                next_unit=next_unit,
            )
            if phase is not CheckpointPhase.COMPLETED or next_unit != 0:
                raise AssertionError("phase coordinator did not reach completed")

            summary = {"status": "completed", **trainer.summary()}
            self._save_boundary(
                trainer,
                phase=CheckpointPhase.COMPLETED,
                next_unit=0,
            )
            _atomic_write_json(
                self.config.run.output_dir / "summary.json",
                summary,
            )
            return summary
        finally:
            trainer.close()

    def _run_remaining_phases(
        self,
        trainer: MainlineTrainerProtocol,
        *,
        phase: CheckpointPhase,
        next_unit: int,
    ) -> tuple[CheckpointPhase, int]:
        stages: tuple[
            tuple[CheckpointPhase, int, Callable[[int], dict[str, Any]]],
            ...,
        ] = (
            (
                CheckpointPhase.WARMUP,
                self.config.run.warmup_episodes,
                trainer.collect_warmup_episode,
            ),
            (
                CheckpointPhase.WORLD_MODEL_INITIALIZATION,
                self.config.run.world_model_initialization_epochs,
                trainer.initialize_world_model_epoch,
            ),
            (
                CheckpointPhase.ONLINE_TRAINING,
                self.config.run.online_training_episodes,
                trainer.train_online_episode,
            ),
            (
                CheckpointPhase.FINAL_EVALUATION,
                self.config.run.evaluation_episodes,
                trainer.evaluate_episode,
            ),
        )
        stage_index = _phase_index(phase)
        if stage_index == len(stages):
            return CheckpointPhase.COMPLETED, 0

        for index in range(stage_index, len(stages)):
            current_phase, target_units, execute_unit = stages[index]
            cursor = next_unit if index == stage_index else 0
            if cursor > target_units:
                raise MainlineRunError(
                    f"checkpoint cursor {cursor} exceeds {current_phase.value} "
                    f"target {target_units}"
                )
            save_interval = _CHECKPOINT_SAVE_INTERVALS[current_phase]
            for unit_index in range(cursor, target_units):
                execute_unit(unit_index)
                # A full boundary checkpoint serializes the entire replay and
                # is saved after each initialization epoch. Online training
                # checkpoints sparsely; the final unit of every
                # phase is always saved (followed by the next-phase boundary).
                if (save_interval is not None and (unit_index + 1) % save_interval == 0) \
                        or unit_index + 1 == target_units:
                    self._save_boundary(
                        trainer,
                        phase=current_phase,
                        next_unit=unit_index + 1,
                    )
            next_phase = (
                stages[index + 1][0]
                if index + 1 < len(stages)
                else CheckpointPhase.COMPLETED
            )
            self._save_boundary(
                trainer,
                phase=next_phase,
                next_unit=0,
            )
            next_unit = 0
        return CheckpointPhase.COMPLETED, 0

    def _save_boundary(
        self,
        trainer: MainlineTrainerProtocol,
        *,
        phase: CheckpointPhase,
        next_unit: int,
    ) -> None:
        save_checkpoint(
            output_dir=self.config.run.output_dir,
            config_fingerprint=self.config.semantic_fingerprint,
            input_hash=self.input_hash,
            phase=phase,
            next_unit=next_unit,
            training_state=trainer.state_dict(),
        )


def run_mainline(config: MainlineConfig) -> dict[str, Any]:
    """Validate prepared inputs and execute or strictly resume the sole mainline."""
    if not isinstance(config, MainlineConfig):
        raise TypeError("run_mainline expects a MainlineConfig from load_config()")
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    input_hash = combined_input_hash(prepared)
    return MainlinePipeline(
        config=config,
        input_hash=input_hash,
        trainer_factory=lambda: _build_trainer(config, prepared, topology),
    ).run()


def combined_input_hash(prepared: PreparedInputs) -> str:
    encoded = json.dumps(
        sorted(prepared.input_hashes.items()),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_trainer(
    config: MainlineConfig,
    prepared: PreparedInputs,
    topology: TrafficTopology,
) -> MainlineTrainerProtocol:
    from ..learning.training import MainlineTrainer

    return MainlineTrainer(
        config=config,
        prepared_inputs=prepared,
        topology=topology,
    )


def _phase_index(phase: CheckpointPhase) -> int:
    ordered = (
        CheckpointPhase.WARMUP,
        CheckpointPhase.WORLD_MODEL_INITIALIZATION,
        CheckpointPhase.ONLINE_TRAINING,
        CheckpointPhase.FINAL_EVALUATION,
        CheckpointPhase.COMPLETED,
    )
    try:
        return ordered.index(phase)
    except ValueError as exc:  # pragma: no cover - enum makes this defensive
        raise MainlineRunError(f"unsupported phase {phase!r}") from exc


def _completed_summary(
    output_dir: Path,
    training_state: dict[str, Any],
) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        try:
            value = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MainlineRunError(f"cannot read completed summary: {exc}") from exc
        if not isinstance(value, dict):
            raise MainlineRunError("completed summary must be a JSON object")
        return value
    metrics = training_state.get("metrics", {})
    if not isinstance(metrics, dict):
        raise MainlineRunError("completed checkpoint metrics must be an object")
    return {"status": "completed", **metrics}


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.tmp.",
        suffix=path.suffix,
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
