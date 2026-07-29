from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import pytest
import numpy as np
import torch

from urban_dreamer.workflow.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    CheckpointPhase,
    OutputDirectoryDecision,
    decide_output_directory,
    latest_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)


def _training_state() -> dict[str, object]:
    return {
        "models": {"world_model": {"weights": torch.tensor([1.0, 2.0])}},
        "optimizers": {"world_model": {"lr": 1e-3}},
        "replay": {"episodes": [1, 2, 3]},
        "hdv": {"route_ids": ("a", "b")},
        "rng": {"torch_state": [1, 2, 3]},
        "metrics": {"losses": (1.0, 0.5)},
        "caches": {"latent": {"shape": [4, 8]}},
    }


def _write_raw_checkpoint(tmp_path: Path, payload: object) -> Path:
    path = latest_checkpoint_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def test_save_and_load_checkpoint_round_trip(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    path = save_checkpoint(
        output_dir=output_dir,
        config_fingerprint="cfg-1",
        input_hash="input-1",
        phase=CheckpointPhase.WARMUP,
        next_unit=3,
        training_state=_training_state(),
    )

    assert path == output_dir / "checkpoints" / "latest.pt"
    assert path.is_file()

    loaded = load_checkpoint(
        output_dir=output_dir,
        expected_config_fingerprint="cfg-1",
        expected_input_hash="input-1",
    )

    assert loaded.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert loaded.phase is CheckpointPhase.WARMUP
    assert loaded.next_unit == 3
    assert loaded.training_state["hdv"]["route_ids"] == ["a", "b"]
    assert loaded.training_state["metrics"]["losses"] == [1.0, 0.5]
    assert loaded.training_state["models"]["world_model"]["weights"].device.type == "cpu"
    assert torch.equal(
        loaded.training_state["models"]["world_model"]["weights"],
        torch.tensor([1.0, 2.0]),
    )

    status = decide_output_directory(
        output_dir=output_dir,
        expected_config_fingerprint="cfg-1",
        expected_input_hash="input-1",
    )
    assert status.decision is OutputDirectoryDecision.VALID_RESUME
    assert status.checkpoint is not None
    assert status.checkpoint.path == loaded.path
    assert status.checkpoint.phase is loaded.phase
    assert status.checkpoint.next_unit == loaded.next_unit


def test_checkpoint_normalizes_numpy_and_preserves_optimizer_integer_keys(
    tmp_path: Path,
) -> None:
    state = _training_state()
    state["replay"] = {
        "flow": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "count": np.int64(1),
    }
    state["optimizers"] = {
        "world_model": {
            "state": {0: {"step": torch.tensor(2), "exp_avg": torch.ones(2)}},
            "param_groups": [{"params": [0], "lr": 1e-3}],
        }
    }

    save_checkpoint(
        output_dir=tmp_path,
        config_fingerprint="cfg-1",
        input_hash="input-1",
        phase=CheckpointPhase.ONLINE_TRAINING,
        next_unit=1,
        training_state=state,
    )
    loaded = load_checkpoint(
        output_dir=tmp_path,
        expected_config_fingerprint="cfg-1",
        expected_input_hash="input-1",
    )

    assert torch.equal(
        loaded.training_state["replay"]["flow"],
        torch.tensor([[1.0, 2.0]]),
    )
    assert loaded.training_state["replay"]["count"] == 1
    assert 0 in loaded.training_state["optimizers"]["world_model"]["state"]


def test_decide_output_directory_reports_fresh_completed_and_nonempty_invalid(
    tmp_path: Path,
) -> None:
    fresh = decide_output_directory(
        output_dir=tmp_path / "fresh",
        expected_config_fingerprint="cfg-1",
        expected_input_hash="input-1",
    )
    assert fresh.decision is OutputDirectoryDecision.FRESH

    completed_dir = tmp_path / "completed"
    save_checkpoint(
        output_dir=completed_dir,
        config_fingerprint="cfg-1",
        input_hash="input-1",
        phase=CheckpointPhase.COMPLETED,
        next_unit=0,
        training_state=_training_state(),
    )
    completed = decide_output_directory(
        output_dir=completed_dir,
        expected_config_fingerprint="cfg-1",
        expected_input_hash="input-1",
    )
    assert completed.decision is OutputDirectoryDecision.COMPLETED
    assert completed.checkpoint is not None
    assert completed.checkpoint.phase is CheckpointPhase.COMPLETED

    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    (invalid_dir / "metrics.csv").write_text("header\n", encoding="utf-8")
    invalid = decide_output_directory(
        output_dir=invalid_dir,
        expected_config_fingerprint="cfg-1",
        expected_input_hash="input-1",
    )
    assert invalid.decision is OutputDirectoryDecision.NONEMPTY_INVALID
    assert invalid.checkpoint is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("input_hash"), "missing key"),
        (lambda payload: payload.__setitem__("unexpected", 1), "unexpected key"),
        (lambda payload: payload.__setitem__("schema_version", 999), "unsupported checkpoint schema version"),
        (lambda payload: payload.__setitem__("phase", "unknown"), "invalid checkpoint phase"),
        (lambda payload: payload.__setitem__("next_unit", -1), "next_unit must be a non-negative integer"),
        (
            lambda payload: payload["training_state"].pop("models"),
            "training_state is missing key",
        ),
        (
            lambda payload: payload["training_state"].__setitem__("extra", {}),
            "training_state has unexpected key",
        ),
    ],
)
def test_load_rejects_strict_schema_and_cursor_mismatches(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_fingerprint": "cfg-1",
        "input_hash": "input-1",
        "phase": CheckpointPhase.ONLINE_TRAINING.value,
        "next_unit": 7,
        "training_state": _training_state(),
    }
    mutate(payload)
    _write_raw_checkpoint(tmp_path, payload)

    with pytest.raises(CheckpointError, match=message):
        load_checkpoint(
            output_dir=tmp_path,
            expected_config_fingerprint="cfg-1",
            expected_input_hash="input-1",
        )


def test_load_rejects_fingerprint_and_input_hash_mismatch(tmp_path: Path) -> None:
    save_checkpoint(
        output_dir=tmp_path,
        config_fingerprint="cfg-1",
        input_hash="input-1",
        phase=CheckpointPhase.WORLD_MODEL_INITIALIZATION,
        next_unit=2,
        training_state=_training_state(),
    )

    with pytest.raises(CheckpointError, match="config fingerprint does not match"):
        load_checkpoint(
            output_dir=tmp_path,
            expected_config_fingerprint="cfg-2",
            expected_input_hash="input-1",
        )

    with pytest.raises(CheckpointError, match="input hash does not match"):
        load_checkpoint(
            output_dir=tmp_path,
            expected_config_fingerprint="cfg-1",
            expected_input_hash="input-2",
        )


def test_load_rejects_corrupt_payload(tmp_path: Path) -> None:
    path = latest_checkpoint_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a checkpoint")

    with pytest.raises(CheckpointError, match="cannot load checkpoint"):
        load_checkpoint(
            output_dir=tmp_path,
            expected_config_fingerprint="cfg-1",
            expected_input_hash="input-1",
        )


def test_save_rejects_non_plain_objects(tmp_path: Path) -> None:
    @dataclass
    class Unsupported:
        value: int

    state = _training_state()
    state["caches"] = {"path": Path("artifact.pt")}
    with pytest.raises(CheckpointError, match="must contain only arrays/CPU tensors"):
        save_checkpoint(
            output_dir=tmp_path / "path",
            config_fingerprint="cfg-1",
            input_hash="input-1",
            phase=CheckpointPhase.FINAL_EVALUATION,
            next_unit=1,
            training_state=state,
        )

    state = _training_state()
    state["caches"] = {"custom": Unsupported(1)}
    with pytest.raises(CheckpointError, match="must contain only arrays/CPU tensors"):
        save_checkpoint(
            output_dir=tmp_path / "dataclass",
            config_fingerprint="cfg-1",
            input_hash="input-1",
            phase=CheckpointPhase.FINAL_EVALUATION,
            next_unit=1,
            training_state=state,
        )

    state = _training_state()
    state["caches"] = {"module": torch}
    with pytest.raises(CheckpointError, match="must contain only arrays/CPU tensors"):
        save_checkpoint(
            output_dir=tmp_path / "module",
            config_fingerprint="cfg-1",
            input_hash="input-1",
            phase=CheckpointPhase.FINAL_EVALUATION,
            next_unit=1,
            training_state=state,
        )


def test_atomic_failure_leaves_old_latest_checkpoint_intact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    save_checkpoint(
        output_dir=output_dir,
        config_fingerprint="cfg-1",
        input_hash="input-1",
        phase=CheckpointPhase.WARMUP,
        next_unit=4,
        training_state=_training_state(),
    )

    import urban_dreamer.workflow.checkpoint as checkpoint_module

    def _boom(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(checkpoint_module.os, "replace", _boom)
    with pytest.raises(OSError, match="replace failed"):
        save_checkpoint(
            output_dir=output_dir,
            config_fingerprint="cfg-1",
            input_hash="input-1",
            phase=CheckpointPhase.ONLINE_TRAINING,
            next_unit=99,
            training_state=_training_state(),
        )

    loaded = load_checkpoint(
        output_dir=output_dir,
        expected_config_fingerprint="cfg-1",
        expected_input_hash="input-1",
    )
    assert loaded.phase is CheckpointPhase.WARMUP
    assert loaded.next_unit == 4
    assert list((output_dir / "checkpoints").glob("*.tmp.*")) == []
