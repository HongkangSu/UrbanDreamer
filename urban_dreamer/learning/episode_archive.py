"""Compressed real SUMO episodes, independent of bounded training replay.

All JSON is UTF-8 bytes and all numerical arrays have concrete dtypes: archives
can be read with ``allow_pickle=False``. Boundary vehicle snapshots retain
inactive vehicles as well as active ones, including final arrival status.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from ..traffic.environment import WindowResult, to_plain_state


SCHEMA_VERSION = 1
ACTION_SEMANTICS = {"event": "branch-only", "microstep": "all-next-hop"}


def _json_array(value: Any) -> np.ndarray:
    encoded = json.dumps(to_plain_state(value), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return np.frombuffer(encoded, dtype=np.uint8)


def _decode(value: np.ndarray) -> Any:
    return json.loads(value.tobytes().decode("utf-8"))


def _flow(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy()
    if (result.ndim != 2 or result.shape[0] < 1 or result.shape[1] != 8
            or not np.isfinite(result).all() or (shape is not None and result.shape != shape)):
        raise ValueError("flow must have finite shape (E, 8), constant throughout an episode")
    return result


def _counts(value: Any, edges: int) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if (values.shape != (edges,) or not np.isfinite(values).all()
            or (values < 0).any() or (values != np.floor(values)).any()
            or (values > np.iinfo(np.uint32).max).any()):
        raise ValueError("counts must be nonnegative integral values with shape (E,)")
    return values.astype(np.uint32)


class EpisodeArchive:
    """Collect one real episode; append only successfully observed windows.

    Metadata must identify city, method, seed, episode, phase, input_hashes,
    total_vehicle_count and session_protocol. Extra fields (including the full
    realized demand) are preserved. Existing paths may only be replaced when
    *all* metadata matches, allowing atomic replay of a checkpointed episode.
    """

    def __init__(self, metadata: dict[str, Any]):
        self.metadata = _decode(_json_array(metadata))
        required = {"city", "method", "seed", "episode", "phase", "input_hashes",
                    "total_vehicle_count", "session_protocol"}
        if not required <= self.metadata.keys():
            raise ValueError(f"missing archive metadata: {sorted(required - self.metadata.keys())}")
        population = self.metadata["total_vehicle_count"]
        if isinstance(population, bool) or not isinstance(population, int) or population < 0:
            raise ValueError("total_vehicle_count must be a nonnegative integer")
        protocol = self.metadata["session_protocol"]
        if protocol not in ACTION_SEMANTICS:
            raise ValueError("session_protocol must be event or microstep")
        semantics = ACTION_SEMANTICS[protocol]
        if self.metadata.get("action_semantics", semantics) != semantics:
            raise ValueError("action_semantics does not match session_protocol")
        self.metadata["action_semantics"] = semantics
        self._states: list[dict[str, Any]] = []
        self._flows: list[np.ndarray] = []
        self._windows: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []

    def _state(self, state: dict[str, Any], shape: tuple[int, ...] | None = None):
        flow = _flow(state["flow"], shape)
        plain = deepcopy({key: value for key, value in state.items() if key != "flow"})
        plain = _decode(_json_array(plain))
        timestamp = float(plain["time"])
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("state time must be finite and nonnegative")
        ids = [row["vehicle_id"] for row in plain["vehicles"]]
        if len(ids) != len(set(ids)) or len(ids) != self.metadata["total_vehicle_count"]:
            raise ValueError("boundary snapshots must contain each episode vehicle exactly once")
        return plain, flow

    def start(self, state: dict[str, Any]) -> None:
        if self._states:
            raise ValueError("archive has already started")
        plain, flow = self._state(state)
        self._states.append(plain)
        self._flows.append(flow)

    def append(self, result: WindowResult) -> None:
        if not self._states:
            raise ValueError("start must precede append")
        if self._windows and self._states[-1]["done"]:
            raise ValueError("cannot append after physical termination")
        start, end = float(result.start_time), float(result.end_time)
        if (not np.isfinite([start, end]).all()
                or not np.isclose(start, self._states[-1]["time"], rtol=0, atol=1e-7)
                or not np.isclose(end - start, 30., rtol=0, atol=1e-7)):
            raise ValueError("windows must have continuous physical 30 second boundaries")
        state, flow = self._state(result.next_state, self._flows[0].shape)
        if not np.isclose(state["time"], end, rtol=0, atol=1e-7):
            raise ValueError("next state must coincide with window end")
        if {v["vehicle_id"] for v in state["vehicles"]} != {v["vehicle_id"] for v in self._states[0]["vehicles"]}:
            raise ValueError("vehicle identities changed within episode")
        edges = flow.shape[0]
        actions, departures = _counts(result.action_counts, edges), _counts(result.departure_counts, edges)
        exposure = float(result.active_vehicle_seconds)
        if not np.isfinite(exposure) or not 0 <= exposure <= self.metadata["total_vehicle_count"] * 30 + 1e-5:
            raise ValueError("invalid active vehicle exposure")
        events, reconstructed = [], np.zeros(edges, dtype=np.uint32)
        vehicle_ids = {v["vehicle_id"] for v in state["vehicles"]}
        for decision in result.decisions:
            row = to_plain_state(decision)
            if (row["vehicle_id"] not in vehicle_ids or not 0 <= row["current_edge"] < edges
                    or not 0 <= row["next_edge"] < edges
                    or not start <= row["decision_time"] < end):
                raise ValueError("decision must belong to a known vehicle, edge and half-open window")
            row["window_index"] = len(self._windows)
            events.append(row)
            reconstructed[row["next_edge"]] += 1
        if not np.array_equal(actions, reconstructed):
            raise ValueError("action counts do not match successful decisions")
        self._windows.append({"start": start, "end": end, "actions": actions,
                              "departures": departures, "exposure": exposure})
        self._events.extend(events)
        self._states.append(state)
        self._flows.append(flow)

    def save(self, path: Path, *, metrics: dict, travel_times: dict,
             completed: bool, deadline_seconds: float) -> None:
        if not self._windows:
            raise ValueError("episode needs at least one window, including empty episodes")
        if not isinstance(completed, bool) or completed != bool(self._states[-1]["done"]):
            raise ValueError("completed must match physical terminal state, not time limit")
        if not np.isfinite(deadline_seconds) or deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive and finite")
        ids = {row["vehicle_id"] for row in self._states[-1]["vehicles"]}
        arrived = {row["vehicle_id"] for row in self._states[-1]["vehicles"] if row["arrived"]}
        if set(travel_times) != arrived or (completed and arrived != ids):
            raise ValueError("travel times must identify arrived episode vehicles")
        if any(not np.isfinite(value) or value < 0 for value in travel_times.values()):
            raise ValueError("travel times must be finite nonnegative seconds")
        summary = {"schema_version": SCHEMA_VERSION, "metadata": self.metadata,
                   "metrics": metrics, "travel_times": travel_times, "completed": completed,
                   "episode_truncated": not completed, "deadline_seconds": float(deadline_seconds),
                   "source": "real_sumo"}
        denominator = max(self.metadata["total_vehicle_count"], 1)
        actions = np.stack([row["actions"] for row in self._windows])
        departures = np.stack([row["departures"] for row in self._windows])
        terminal = np.asarray([bool(state["done"]) for state in self._states[1:]], dtype=np.bool_)
        truncated = np.zeros(len(self._windows), dtype=np.bool_)
        truncated[-1] = not completed
        arrays = dict(summary_json=_json_array(summary), states_json=_json_array(self._states),
                      decisions_json=_json_array(self._events), flow=np.stack(self._flows),
                      action_counts=actions, departure_counts=departures,
                      action=actions.astype(np.float32) / denominator,
                      departure=departures.astype(np.float32) / denominator,
                      time=np.asarray([state["time"] for state in self._states], dtype=np.float64),
                      active_vehicle_seconds=np.asarray([row["exposure"] for row in self._windows], dtype=np.float64),
                      terminated=terminal, truncated=truncated)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as existing:
                old = _decode(existing["summary_json"])
            if old.get("schema_version") != SCHEMA_VERSION or old.get("metadata") != self.metadata:
                raise ValueError("refusing to overwrite an episode with different provenance")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def read_episode_archive(path: Path, *, expected_session_protocol: str | None = None) -> dict[str, Any]:
    """Read without pickle; optionally reject data from a different action protocol."""
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files if not key.endswith("_json")}
        result.update(_decode(archive["summary_json"]))
        result["states"] = _decode(archive["states_json"])
        result["decisions"] = _decode(archive["decisions_json"])
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported episode archive version")
    protocol = result["metadata"]["session_protocol"]
    if protocol not in ACTION_SEMANTICS or result["metadata"]["action_semantics"] != ACTION_SEMANTICS[protocol]:
        raise ValueError("invalid archived action semantics")
    if expected_session_protocol is not None and expected_session_protocol != protocol:
        raise ValueError(f"session protocol mismatch: expected {expected_session_protocol}, found {protocol}")
    return result


def to_replay_episode(path: Path, *, expected_session_protocol: str) -> dict[str, Any]:
    """Convert to replay only after explicitly selecting the compatible protocol.

    Timeout/censoring never becomes ``MacroTransition.terminal``. Original N
    and observed exposure are preserved, including zero-population episodes.
    """
    from .replay import MacroTransition
    if expected_session_protocol not in ACTION_SEMANTICS:
        raise ValueError("an explicit event or microstep session protocol is required")
    data = read_episode_archive(path, expected_session_protocol=expected_session_protocol)
    population = data["metadata"]["total_vehicle_count"]
    denominator = max(population, 1)
    rows = []
    for index, state in enumerate(data["states"][:-1]):
        vehicles = tuple(v for v in state["vehicles"] if v["departed"] and not v["arrived"])
        rows.append(MacroTransition(flow=data["flow"][index], next_flow=data["flow"][index + 1],
            action=data["action"][index], departure=data["departure"][index],
            context=[min(state["time"] / data["deadline_seconds"], 1.), len(vehicles) / denominator],
            active_vehicle_fraction=float(np.clip(data["active_vehicle_seconds"][index] / (30 * denominator), 0, 1)),
            vehicles=vehicles, time=state["time"], terminal=bool(data["terminated"][index]),
            episode=data["metadata"]["episode"], step_index=index))
    seconds = (sum(data["travel_times"].values()) if data["completed"]
               else float(data["active_vehicle_seconds"].sum()))
    return {"transitions": tuple(rows), "total_vehicle_count": population,
            "total_travel_minutes": seconds / (60 * denominator), "completed": data["completed"]}
