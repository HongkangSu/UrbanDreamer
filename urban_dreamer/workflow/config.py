from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


NONARRIVAL_PENALTY = -5000.0


class ConfigError(ValueError):
    """Raised when the mainline TOML contract is invalid."""


@dataclass(frozen=True)
class InputConfig:
    network_name: str
    network_dir: Path


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path
    device: str
    seed: int
    warmup_episodes: int
    world_model_initialization_steps: int
    online_training_episodes: int
    evaluation_episodes: int


@dataclass(frozen=True)
class SimulationConfig:
    vehicle_max_speed_mps: float
    world_model_intervals_per_episode: int
    warmup_horizon_seconds: int
    warmup_completion_drain_seconds: int
    warmup_completion_patience_seconds: int
    episode_wall_timeout_seconds: float
    training_completion_buffer_seconds: int
    evaluation_horizon_seconds: int
    incident_edge: str
    incident_start_seconds: float
    incident_end_seconds: float
    incident_speed_mps: float
    sumo_extra_args: tuple[str, ...]


@dataclass(frozen=True)
class ResourceConfig:
    world_model_initialization_batch_size: int
    world_model_initialization_sequence_length: int
    world_model_gradient_accumulation_steps: int
    online_world_model_batch_size: int


@dataclass(frozen=True)
class MainlineConfig:
    inputs: InputConfig
    run: RunConfig
    simulation: SimulationConfig
    resources: ResourceConfig
    semantic_fingerprint: str
    source_path: Path

    def resolved_snapshot(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("semantic_fingerprint")
        payload.pop("source_path")
        return _json_compatible(payload)


_SECTION_FIELDS: dict[str, frozenset[str]] = {
    "inputs": frozenset(InputConfig.__dataclass_fields__),
    "run": frozenset(RunConfig.__dataclass_fields__),
    "simulation": frozenset(SimulationConfig.__dataclass_fields__),
    "resources": frozenset(ResourceConfig.__dataclass_fields__),
}


def load_config(path: str | Path) -> MainlineConfig:
    source_path = Path(path).expanduser().resolve()
    try:
        with source_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read TOML config {source_path}: {exc}") from exc

    _validate_exact_schema(raw)
    base_dir = source_path.parent
    inputs = _parse_inputs(raw["inputs"], base_dir)
    run = _parse_run(raw["run"], base_dir)
    simulation = _parse_simulation(raw["simulation"])
    resources = _parse_resources(raw["resources"])
    semantic_fingerprint = _semantic_fingerprint(
        inputs=inputs,
        run=run,
        simulation=simulation,
        resources=resources,
    )
    return MainlineConfig(
        inputs=inputs,
        run=run,
        simulation=simulation,
        resources=resources,
        semantic_fingerprint=semantic_fingerprint,
        source_path=source_path,
    )


def _validate_exact_schema(raw: Mapping[str, Any]) -> None:
    unknown_sections = set(raw) - set(_SECTION_FIELDS)
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ConfigError(f"unknown top-level section(s): {names}")
    missing_sections = set(_SECTION_FIELDS) - set(raw)
    if missing_sections:
        names = ", ".join(sorted(missing_sections))
        raise ConfigError(f"missing top-level section(s): {names}")
    for section_name, expected_fields in _SECTION_FIELDS.items():
        section = raw[section_name]
        if not isinstance(section, dict):
            raise ConfigError(f"[{section_name}] must be a TOML table")
        unknown_fields = set(section) - set(expected_fields)
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise ConfigError(f"unknown [{section_name}] field(s): {names}")
        missing_fields = set(expected_fields) - set(section)
        if missing_fields:
            names = ", ".join(sorted(missing_fields))
            raise ConfigError(f"missing [{section_name}] field(s): {names}")


def _parse_inputs(raw: Mapping[str, Any], base_dir: Path) -> InputConfig:
    return InputConfig(
        network_name=_nonempty_string(raw, "network_name", "inputs"),
        network_dir=_resolved_path(raw, "network_dir", "inputs", base_dir),

    )


def _parse_run(raw: Mapping[str, Any], base_dir: Path) -> RunConfig:
    return RunConfig(
        output_dir=_resolved_path(raw, "output_dir", "run", base_dir),
        device=_nonempty_string(raw, "device", "run"),
        seed=_integer(raw, "seed", "run"),
        warmup_episodes=_positive_integer(raw, "warmup_episodes", "run"),
        world_model_initialization_steps=_positive_integer(
            raw, "world_model_initialization_steps", "run"
        ),
        online_training_episodes=_positive_integer(
            raw, "online_training_episodes", "run"
        ),
        evaluation_episodes=_positive_integer(
            raw, "evaluation_episodes", "run"
        ),
    )


def _parse_simulation(raw: Mapping[str, Any]) -> SimulationConfig:
    incident_start_seconds = _nonnegative_number(
        raw, "incident_start_seconds", "simulation"
    )
    incident_end_seconds = _positive_number(
        raw, "incident_end_seconds", "simulation"
    )
    if incident_end_seconds <= incident_start_seconds:
        raise ConfigError(
            "[simulation].incident_end_seconds must be greater than "
            "incident_start_seconds"
        )
    sumo_extra_args = raw["sumo_extra_args"]
    if not isinstance(sumo_extra_args, list) or not all(
        isinstance(value, str) and value for value in sumo_extra_args
    ):
        raise ConfigError("[simulation].sumo_extra_args must be an array of strings")
    reserved = {
        "-c", "--configuration-file", "--configuration", "-n", "--net-file",
        "-r", "--route-files", "--routes", "-a", "--additional-files", "--additional",
        "--step-length", "--begin", "-b", "--end", "-e", "--seed", "--random",
        "--default.speedfactor", "--default.speeddev", "--speed-factor",
        "--default.action-step-length", "--default.departspeed", "--default.departpos",
        "--load-state", "--route-steps", "--scale", "--device.rerouting.probability",
        "--device.rerouting.explicit", "--remote-port", "--num-clients",
    }
    for argument in sumo_extra_args:
        option = argument.split("=", 1)[0]
        if option in reserved:
            raise ConfigError(
                f"[simulation].sumo_extra_args contains reserved SUMO option {option}"
            )
    vehicle_max_speed = _positive_number(raw, "vehicle_max_speed_mps", "simulation")
    incident_speed = _positive_number(raw, "incident_speed_mps", "simulation")
    if incident_speed > vehicle_max_speed:
        raise ConfigError(
            "[simulation].incident_speed_mps must not exceed vehicle_max_speed_mps"
        )
    return SimulationConfig(
        vehicle_max_speed_mps=vehicle_max_speed,
        world_model_intervals_per_episode=_positive_integer(
            raw, "world_model_intervals_per_episode", "simulation"
        ),
        warmup_horizon_seconds=_positive_integer(
            raw, "warmup_horizon_seconds", "simulation"
        ),
        warmup_completion_drain_seconds=_nonnegative_integer(
            raw, "warmup_completion_drain_seconds", "simulation"
        ),
        warmup_completion_patience_seconds=_nonnegative_integer(
            raw, "warmup_completion_patience_seconds", "simulation"
        ),
        episode_wall_timeout_seconds=_positive_number(
            raw, "episode_wall_timeout_seconds", "simulation"
        ),
        training_completion_buffer_seconds=_nonnegative_integer(
            raw, "training_completion_buffer_seconds", "simulation"
        ),
        evaluation_horizon_seconds=_positive_integer(
            raw, "evaluation_horizon_seconds", "simulation"
        ),
        incident_edge=_nonempty_string(raw, "incident_edge", "simulation"),
        incident_start_seconds=incident_start_seconds,
        incident_end_seconds=incident_end_seconds,
        incident_speed_mps=incident_speed,
        sumo_extra_args=tuple(sumo_extra_args),
    )


def _parse_resources(raw: Mapping[str, Any]) -> ResourceConfig:
    section = "resources"
    sequence_length = _positive_integer(
        raw,
        "world_model_initialization_sequence_length",
        section,
    )
    return ResourceConfig(
        world_model_initialization_batch_size=_positive_integer(
            raw, "world_model_initialization_batch_size", section
        ),
        world_model_initialization_sequence_length=sequence_length,
        world_model_gradient_accumulation_steps=_positive_integer(
            raw, "world_model_gradient_accumulation_steps", section
        ),
        online_world_model_batch_size=_positive_integer(
            raw, "online_world_model_batch_size", section
        ),
    )


def _semantic_fingerprint(
    *,
    inputs: InputConfig,
    run: RunConfig,
    simulation: SimulationConfig,
    resources: ResourceConfig,
) -> str:
    run_values = asdict(run)
    run_values.pop("output_dir")
    run_values.pop("device")
    payload = {
        "decoder_speed_target": "forward_endpoint_mean_v1",
        "world_model_conditions": "action_context_only_v1",
        "imagination_horizon": "eight_windows_full_episode_sampling_v1",
        "nonarrival_penalty": NONARRIVAL_PENALTY,
        "network_name": inputs.network_name,
        "run": run_values,
        "simulation": asdict(simulation),
        "resources": asdict(resources),
    }
    encoded = json.dumps(
        _json_compatible(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_path(
    raw: Mapping[str, Any],
    key: str,
    section: str,
    base_dir: Path,
) -> Path:
    value = _nonempty_string(raw, key, section)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _nonempty_string(raw: Mapping[str, Any], key: str, section: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"[{section}].{key} must be a non-empty string")
    return value.strip()


def _integer(raw: Mapping[str, Any], key: str, section: str) -> int:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"[{section}].{key} must be an integer")
    return value


def _positive_integer(raw: Mapping[str, Any], key: str, section: str) -> int:
    value = _integer(raw, key, section)
    if value <= 0:
        raise ConfigError(f"[{section}].{key} must be positive")
    return value


def _nonnegative_integer(raw: Mapping[str, Any], key: str, section: str) -> int:
    value = _integer(raw, key, section)
    if value < 0:
        raise ConfigError(f"[{section}].{key} must be non-negative")
    return value


def _number(raw: Mapping[str, Any], key: str, section: str) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"[{section}].{key} must be a number")
    if not math.isfinite(float(value)):
        raise ConfigError(f"[{section}].{key} must be finite")
    return float(value)


def _positive_number(raw: Mapping[str, Any], key: str, section: str) -> float:
    value = _number(raw, key, section)
    if value <= 0.0:
        raise ConfigError(f"[{section}].{key} must be positive")
    return value


def _nonnegative_number(raw: Mapping[str, Any], key: str, section: str) -> float:
    value = _number(raw, key, section)
    if value < 0.0:
        raise ConfigError(f"[{section}].{key} must be non-negative")
    return value


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    return value
