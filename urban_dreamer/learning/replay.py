from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch


MAX_ONLINE_REPLAY_EPISODES = 8


def _tensor(
    value: Any,
    *,
    name: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    try:
        result = torch.as_tensor(value, dtype=dtype).detach().cpu().clone()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be tensor-like") from exc
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _finite_float(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class VehicleSnapshot:
    """Minimal real-state seed required by imagined vehicle rollouts."""

    vehicle_ids: tuple[str, ...]
    current_edges: torch.Tensor
    destination_edges: torch.Tensor
    departure_steps: torch.Tensor

    def __post_init__(self) -> None:
        ids = tuple(str(value) for value in self.vehicle_ids)
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("vehicle_ids must contain unique non-empty strings")
        current = _tensor(
            self.current_edges,
            name="vehicle_snapshot.current_edges",
            dtype=torch.int64,
        ).reshape(-1)
        destinations = _tensor(
            self.destination_edges,
            name="vehicle_snapshot.destination_edges",
            dtype=torch.int64,
        ).reshape(-1)
        departures = _tensor(
            self.departure_steps,
            name="vehicle_snapshot.departure_steps",
            dtype=torch.int64,
        ).reshape(-1)
        expected = len(ids)
        if not (
            int(current.numel())
            == int(destinations.numel())
            == int(departures.numel())
            == expected
        ):
            raise ValueError("vehicle snapshot arrays must align with vehicle_ids")
        if bool((current < 0).any().item()) or bool((destinations < 0).any().item()):
            raise ValueError("vehicle snapshot edge indices must be non-negative")
        if bool((departures < 0).any().item()):
            raise ValueError("vehicle snapshot departure steps must be non-negative")
        object.__setattr__(self, "vehicle_ids", ids)
        object.__setattr__(self, "current_edges", current)
        object.__setattr__(self, "destination_edges", destinations)
        object.__setattr__(self, "departure_steps", departures)

    def state_dict(self) -> dict[str, Any]:
        return {
            "vehicle_ids": list(self.vehicle_ids),
            "current_edges": self.current_edges.clone(),
            "destination_edges": self.destination_edges.clone(),
            "departure_steps": self.departure_steps.clone(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> VehicleSnapshot:
        if set(state) != {
            "vehicle_ids",
            "current_edges",
            "destination_edges",
            "departure_steps",
        }:
            raise ValueError("vehicle snapshot state has an invalid schema")
        raw_ids = state["vehicle_ids"]
        if not isinstance(raw_ids, list):
            raise TypeError("vehicle snapshot vehicle_ids must be a list")
        return cls(
            vehicle_ids=tuple(str(value) for value in raw_ids),
            current_edges=state["current_edges"],
            destination_edges=state["destination_edges"],
            departure_steps=state["departure_steps"],
        )


@dataclass(frozen=True)
class WarmupReplayRecord:
    episode: int
    step_index: int
    flow: torch.Tensor
    planned_action: torch.Tensor
    departure: torch.Tensor
    context: torch.Tensor
    hdv_context: torch.Tensor
    hdv_target: torch.Tensor
    terminal: bool = False

    def __post_init__(self) -> None:
        episode = _integer(self.episode, name="warmup.episode")
        step_index = _integer(self.step_index, name="warmup.step_index")
        if episode < 0 or step_index < 0:
            raise ValueError("warmup episode and step_index must be non-negative")
        flow = _tensor(self.flow, name="warmup.flow")
        if flow.dim() != 2 or int(flow.shape[-1]) != 8:
            raise ValueError("warmup.flow must have shape (E, 8)")
        edge_count = int(flow.shape[0])
        action = _edge_vector(
            self.planned_action,
            edge_count=edge_count,
            name="warmup.planned_action",
        )
        departure = _edge_vector(
            self.departure,
            edge_count=edge_count,
            name="warmup.departure",
        )
        hdv_context = _edge_vector(
            self.hdv_context,
            edge_count=edge_count,
            name="warmup.hdv_context",
        )
        hdv_target = _edge_vector(
            self.hdv_target,
            edge_count=edge_count,
            name="warmup.hdv_target",
        )
        context = _tensor(self.context, name="warmup.context").reshape(-1)
        if tuple(context.shape) != (3,):
            raise ValueError("warmup.context must have shape (3,)")
        if not isinstance(self.terminal, bool):
            raise TypeError("warmup.terminal must be a bool")
        object.__setattr__(self, "episode", episode)
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "flow", flow)
        object.__setattr__(self, "planned_action", action)
        object.__setattr__(self, "departure", departure)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "hdv_context", hdv_context)
        object.__setattr__(self, "hdv_target", hdv_target)

    def state_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "step_index": self.step_index,
            "flow": self.flow.clone(),
            "planned_action": self.planned_action.clone(),
            "departure": self.departure.clone(),
            "context": self.context.clone(),
            "hdv_context": self.hdv_context.clone(),
            "hdv_target": self.hdv_target.clone(),
            "terminal": self.terminal,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> WarmupReplayRecord:
        return cls(**dict(state))


@dataclass(frozen=True)
class OnlineReplayTransition:
    episode: int
    step_index: int
    flow: torch.Tensor
    action: torch.Tensor
    next_flow: torch.Tensor
    departure: torch.Tensor
    context: torch.Tensor
    hdv_context: torch.Tensor
    hdv_target: torch.Tensor
    vehicles: VehicleSnapshot
    active_vehicle_seconds: float
    transition_seconds: float
    terminal: bool = False

    def __post_init__(self) -> None:
        episode = _integer(self.episode, name="online.episode")
        step_index = _integer(self.step_index, name="online.step_index")
        if episode < 0 or step_index < 0:
            raise ValueError("online episode and step_index must be non-negative")
        flow = _tensor(self.flow, name="online.flow")
        next_flow = _tensor(self.next_flow, name="online.next_flow")
        if flow.dim() != 2 or int(flow.shape[-1]) != 8:
            raise ValueError("online.flow must have shape (E, 8)")
        if tuple(next_flow.shape) != tuple(flow.shape):
            raise ValueError("online.next_flow must match online.flow")
        edge_count = int(flow.shape[0])
        action = _edge_vector(self.action, edge_count=edge_count, name="online.action")
        departure = _edge_vector(
            self.departure,
            edge_count=edge_count,
            name="online.departure",
        )
        hdv_context = _edge_vector(
            self.hdv_context,
            edge_count=edge_count,
            name="online.hdv_context",
        )
        hdv_target = _edge_vector(
            self.hdv_target,
            edge_count=edge_count,
            name="online.hdv_target",
        )
        context = _tensor(self.context, name="online.context").reshape(-1)
        if tuple(context.shape) != (3,):
            raise ValueError("online.context must have shape (3,)")
        vehicles = (
            self.vehicles
            if isinstance(self.vehicles, VehicleSnapshot)
            else VehicleSnapshot.from_state_dict(self.vehicles)
        )
        active_seconds = _finite_float(
            self.active_vehicle_seconds,
            name="online.active_vehicle_seconds",
        )
        transition_seconds = _finite_float(
            self.transition_seconds,
            name="online.transition_seconds",
            positive=True,
        )
        if active_seconds < 0.0:
            raise ValueError("online.active_vehicle_seconds must be non-negative")
        if not isinstance(self.terminal, bool):
            raise TypeError("online.terminal must be a bool")
        object.__setattr__(self, "episode", episode)
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "flow", flow)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "next_flow", next_flow)
        object.__setattr__(self, "departure", departure)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "hdv_context", hdv_context)
        object.__setattr__(self, "hdv_target", hdv_target)
        object.__setattr__(self, "vehicles", vehicles)
        object.__setattr__(self, "active_vehicle_seconds", active_seconds)
        object.__setattr__(self, "transition_seconds", transition_seconds)

    def state_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "step_index": self.step_index,
            "flow": self.flow.clone(),
            "action": self.action.clone(),
            "next_flow": self.next_flow.clone(),
            "departure": self.departure.clone(),
            "context": self.context.clone(),
            "hdv_context": self.hdv_context.clone(),
            "hdv_target": self.hdv_target.clone(),
            "vehicles": self.vehicles.state_dict(),
            "active_vehicle_seconds": self.active_vehicle_seconds,
            "transition_seconds": self.transition_seconds,
            "terminal": self.terminal,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> OnlineReplayTransition:
        values = dict(state)
        values["vehicles"] = VehicleSnapshot.from_state_dict(values["vehicles"])
        return cls(**values)


@dataclass(frozen=True)
class WarmupSequenceBatch:
    flow: torch.Tensor
    action: torch.Tensor
    departure: torch.Tensor
    context: torch.Tensor
    hdv_context: torch.Tensor
    hdv_target: torch.Tensor
    terminal: torch.Tensor
    total_travel_minutes: torch.Tensor
    episode: torch.Tensor
    start_index: torch.Tensor


@dataclass(frozen=True)
class OnlineTransitionBatch:
    flow: torch.Tensor
    action: torch.Tensor
    next_flow: torch.Tensor
    departure: torch.Tensor
    context: torch.Tensor
    hdv_context: torch.Tensor
    hdv_target: torch.Tensor
    vehicles: tuple[VehicleSnapshot, ...]
    active_vehicle_seconds: torch.Tensor
    transition_seconds: torch.Tensor
    total_travel_minutes: torch.Tensor
    terminal: torch.Tensor
    episode: torch.Tensor
    step_index: torch.Tensor


@dataclass(frozen=True)
class _WarmupEpisode:
    storage_id: int
    records: tuple[WarmupReplayRecord, ...]
    total_travel_minutes: float

    def state_dict(self) -> dict[str, Any]:
        return {
            "storage_id": self.storage_id,
            "records": [record.state_dict() for record in self.records],
            "total_travel_minutes": self.total_travel_minutes,
        }


@dataclass(frozen=True)
class _OnlineEpisode:
    storage_id: int
    transitions: tuple[OnlineReplayTransition, ...]
    total_travel_minutes: float

    def state_dict(self) -> dict[str, Any]:
        return {
            "storage_id": self.storage_id,
            "transitions": [
                transition.state_dict() for transition in self.transitions
            ],
            "total_travel_minutes": self.total_travel_minutes,
        }


class WarmupReplayBuffer:
    """Complete warmup episodes sampled as once-shifted causal state windows."""

    def __init__(self, *, sequence_length: int) -> None:
        self.sequence_length = _positive_integer(sequence_length, "sequence_length")
        self._episodes: list[_WarmupEpisode] = []
        self._sample_order: list[tuple[int, int]] = []
        self._sample_cursor = 0
        self._next_storage_id = 0

    @property
    def episodes(self) -> tuple[tuple[WarmupReplayRecord, ...], ...]:
        return tuple(episode.records for episode in self._episodes)

    def append_episode(
        self,
        records: Sequence[WarmupReplayRecord],
        *,
        total_travel_minutes: float,
    ) -> bool:
        normalized = tuple(
            record
            if isinstance(record, WarmupReplayRecord)
            else WarmupReplayRecord(**record)
            for record in records
        )
        if not _complete_warmup_episode(normalized, self.sequence_length):
            return False
        _validate_consistent_warmup_shapes(normalized)
        episode = _WarmupEpisode(
            storage_id=self._next_storage_id,
            records=normalized,
            total_travel_minutes=_finite_float(
                total_travel_minutes,
                name="warmup.total_travel_minutes",
                positive=True,
            ),
        )
        self._next_storage_id += 1
        self._episodes.append(episode)
        self._reset_sampler()
        return True

    def sample_batch(
        self,
        *,
        batch_size: int,
        rng: np.random.Generator,
    ) -> WarmupSequenceBatch:
        picks = self._sample_keys(
            self._candidates(),
            batch_size=_positive_integer(batch_size, "batch_size"),
            rng=rng,
        )
        windows: list[tuple[_WarmupEpisode, tuple[WarmupReplayRecord, ...], int]] = []
        for storage_id, start in picks:
            episode = self._episode(storage_id)
            records = episode.records[start : start + self.sequence_length]
            windows.append((episode, records, start))

        flow = _stack_nested([[record.flow for record in rows] for _, rows, _ in windows])
        action = _stack_nested(
            [
                _shift_once([record.planned_action for record in rows])
                for _, rows, _ in windows
            ]
        )
        departure = _stack_nested(
            [
                _shift_once([record.departure for record in rows])
                for _, rows, _ in windows
            ]
        )
        context = _stack_nested(
            [
                _shift_once([record.context for record in rows])
                for _, rows, _ in windows
            ]
        )
        hdv_context = _stack_nested(
            [
                _shift_once([record.hdv_context for record in rows])
                for _, rows, _ in windows
            ]
        )
        hdv_target = _stack_nested(
            [
                [record.hdv_target for record in rows[1:]]
                for _, rows, _ in windows
            ]
        )
        terminal = torch.stack(
            [
                torch.tensor(
                    [record.terminal for record in rows[1:]],
                    dtype=torch.bool,
                )
                for _, rows, _ in windows
            ],
            dim=0,
        )
        return WarmupSequenceBatch(
            flow=flow,
            action=action,
            departure=departure,
            context=context,
            hdv_context=hdv_context,
            hdv_target=hdv_target,
            terminal=terminal,
            total_travel_minutes=torch.tensor(
                [episode.total_travel_minutes for episode, _, _ in windows],
                dtype=torch.float32,
            ),
            episode=torch.tensor(
                [rows[0].episode for _, rows, _ in windows],
                dtype=torch.int64,
            ),
            start_index=torch.tensor(
                [start for _, _, start in windows],
                dtype=torch.int64,
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "warmup",
            "sequence_length": self.sequence_length,
            "episodes": [episode.state_dict() for episode in self._episodes],
            "sampler": _sampler_state(self._sample_order, self._sample_cursor),
            "next_storage_id": self._next_storage_id,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _require_keys(
            state,
            {"kind", "sequence_length", "episodes", "sampler", "next_storage_id"},
            "warmup replay",
        )
        if state["kind"] != "warmup":
            raise ValueError("invalid warmup replay kind")
        if _integer(state["sequence_length"], name="sequence_length") != self.sequence_length:
            raise ValueError("warmup replay sequence_length mismatch")
        raw_episodes = state["episodes"]
        if not isinstance(raw_episodes, list):
            raise TypeError("warmup replay episodes must be a list")
        episodes: list[_WarmupEpisode] = []
        for raw in raw_episodes:
            _require_keys(
                raw,
                {"storage_id", "records", "total_travel_minutes"},
                "warmup replay episode",
            )
            episodes.append(
                _WarmupEpisode(
                    storage_id=_integer(raw["storage_id"], name="storage_id"),
                    records=tuple(
                        WarmupReplayRecord.from_state_dict(record)
                        for record in raw["records"]
                    ),
                    total_travel_minutes=_finite_float(
                        raw["total_travel_minutes"],
                        name="total_travel_minutes",
                        positive=True,
                    ),
                )
            )
        for episode in episodes:
            if not _complete_warmup_episode(episode.records, self.sequence_length):
                raise ValueError("warmup replay contains an incomplete episode")
            _validate_consistent_warmup_shapes(episode.records)
        self._episodes = episodes
        self._sample_order, self._sample_cursor = _load_sampler(state["sampler"])
        self._next_storage_id = _integer(
            state["next_storage_id"],
            name="next_storage_id",
        )

    def _candidates(self) -> list[tuple[int, int]]:
        return [
            (episode.storage_id, start)
            for episode in self._episodes
            for start in range(len(episode.records) - self.sequence_length + 1)
        ]

    def _episode(self, storage_id: int) -> _WarmupEpisode:
        for episode in self._episodes:
            if episode.storage_id == storage_id:
                return episode
        raise KeyError(f"unknown warmup storage_id={storage_id}")

    def _sample_keys(
        self,
        candidates: Sequence[tuple[int, int]],
        *,
        batch_size: int,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        self._sample_order, self._sample_cursor, result = _sample_keys(
            candidates=candidates,
            order=self._sample_order,
            cursor=self._sample_cursor,
            batch_size=batch_size,
            rng=rng,
        )
        return result

    def _reset_sampler(self) -> None:
        self._sample_order = []
        self._sample_cursor = 0


class OnlineReplayBuffer:
    """Real episode trajectories with explicit, never-shifted transition pairs."""

    def __init__(self) -> None:
        self._episodes: list[_OnlineEpisode] = []
        self._sample_order: list[tuple[int, int]] = []
        self._sample_cursor = 0
        self._next_storage_id = 0

    @property
    def episodes(self) -> tuple[tuple[OnlineReplayTransition, ...], ...]:
        return tuple(episode.transitions for episode in self._episodes)

    def append_episode(
        self,
        transitions: Sequence[OnlineReplayTransition],
        *,
        total_travel_minutes: float,
    ) -> bool:
        normalized = tuple(
            transition
            if isinstance(transition, OnlineReplayTransition)
            else OnlineReplayTransition(**transition)
            for transition in transitions
        )
        if not _valid_online_trajectory(normalized):
            return False
        _validate_consistent_online_shapes(normalized)
        episode = _OnlineEpisode(
            storage_id=self._next_storage_id,
            transitions=normalized,
            total_travel_minutes=_finite_float(
                total_travel_minutes,
                name="online.total_travel_minutes",
                positive=True,
            ),
        )
        self._next_storage_id += 1
        self._episodes.append(episode)
        self._episodes = self._episodes[-MAX_ONLINE_REPLAY_EPISODES:]
        self._reset_sampler()
        return True

    def sample_batch(
        self,
        *,
        batch_size: int,
        rng: np.random.Generator,
    ) -> OnlineTransitionBatch:
        order, cursor, picks = _sample_keys(
            candidates=self._candidates(),
            order=self._sample_order,
            cursor=self._sample_cursor,
            batch_size=_positive_integer(batch_size, "batch_size"),
            rng=rng,
        )
        self._sample_order, self._sample_cursor = order, cursor
        rows: list[tuple[_OnlineEpisode, OnlineReplayTransition]] = []
        for storage_id, index in picks:
            episode = self._episode(storage_id)
            rows.append((episode, episode.transitions[index]))
        return OnlineTransitionBatch(
            flow=torch.stack([row.flow for _, row in rows]),
            action=torch.stack([row.action for _, row in rows]),
            next_flow=torch.stack([row.next_flow for _, row in rows]),
            departure=torch.stack([row.departure for _, row in rows]),
            context=torch.stack([row.context for _, row in rows]),
            hdv_context=torch.stack([row.hdv_context for _, row in rows]),
            hdv_target=torch.stack([row.hdv_target for _, row in rows]),
            vehicles=tuple(row.vehicles for _, row in rows),
            active_vehicle_seconds=torch.tensor(
                [row.active_vehicle_seconds for _, row in rows],
                dtype=torch.float32,
            ),
            transition_seconds=torch.tensor(
                [row.transition_seconds for _, row in rows],
                dtype=torch.float32,
            ),
            total_travel_minutes=torch.tensor(
                [episode.total_travel_minutes for episode, _ in rows],
                dtype=torch.float32,
            ),
            terminal=torch.tensor(
                [row.terminal for _, row in rows],
                dtype=torch.bool,
            ),
            episode=torch.tensor(
                [row.episode for _, row in rows],
                dtype=torch.int64,
            ),
            step_index=torch.tensor(
                [row.step_index for _, row in rows],
                dtype=torch.int64,
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "online",
            "episodes": [episode.state_dict() for episode in self._episodes],
            "sampler": _sampler_state(self._sample_order, self._sample_cursor),
            "next_storage_id": self._next_storage_id,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _require_keys(
            state,
            {"kind", "episodes", "sampler", "next_storage_id"},
            "online replay",
        )
        if state["kind"] != "online":
            raise ValueError("invalid online replay kind")
        raw_episodes = state["episodes"]
        if not isinstance(raw_episodes, list):
            raise TypeError("online replay episodes must be a list")
        episodes: list[_OnlineEpisode] = []
        for raw in raw_episodes:
            _require_keys(
                raw,
                {"storage_id", "transitions", "total_travel_minutes"},
                "online replay episode",
            )
            episodes.append(
                _OnlineEpisode(
                    storage_id=_integer(raw["storage_id"], name="storage_id"),
                    transitions=tuple(
                        OnlineReplayTransition.from_state_dict(item)
                        for item in raw["transitions"]
                    ),
                    total_travel_minutes=_finite_float(
                        raw["total_travel_minutes"],
                        name="total_travel_minutes",
                        positive=True,
                    ),
                )
            )
        if len(episodes) > MAX_ONLINE_REPLAY_EPISODES:
            raise ValueError("online replay exceeds fixed episode capacity")
        for episode in episodes:
            if not _valid_online_trajectory(episode.transitions):
                raise ValueError("online replay contains an invalid trajectory")
            _validate_consistent_online_shapes(episode.transitions)
        self._episodes = episodes
        self._sample_order, self._sample_cursor = _load_sampler(state["sampler"])
        self._next_storage_id = _integer(
            state["next_storage_id"],
            name="next_storage_id",
        )

    def _candidates(self) -> list[tuple[int, int]]:
        return [
            (episode.storage_id, index)
            for episode in self._episodes
            for index in range(len(episode.transitions))
        ]

    def _episode(self, storage_id: int) -> _OnlineEpisode:
        for episode in self._episodes:
            if episode.storage_id == storage_id:
                return episode
        raise KeyError(f"unknown online storage_id={storage_id}")

    def _reset_sampler(self) -> None:
        self._sample_order = []
        self._sample_cursor = 0


def _edge_vector(value: Any, *, edge_count: int, name: str) -> torch.Tensor:
    result = _tensor(value, name=name).reshape(-1)
    if tuple(result.shape) != (edge_count,):
        raise ValueError(f"{name} must have shape ({edge_count},)")
    return result


def _positive_integer(value: Any, name: str) -> int:
    result = _integer(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _complete_warmup_episode(
    records: tuple[WarmupReplayRecord, ...],
    sequence_length: int,
) -> bool:
    if len(records) < sequence_length or not records[-1].terminal:
        return False
    episode = records[0].episode
    return all(
        record.episode == episode
        and record.step_index == index
        and record.terminal == (index == len(records) - 1)
        for index, record in enumerate(records)
    )


def _valid_online_trajectory(
    transitions: tuple[OnlineReplayTransition, ...],
) -> bool:
    if not transitions:
        return False
    episode = transitions[0].episode
    return all(
        transition.episode == episode
        and transition.step_index == index
        and (not transition.terminal or index == len(transitions) - 1)
        for index, transition in enumerate(transitions)
    )


def _validate_consistent_warmup_shapes(
    records: tuple[WarmupReplayRecord, ...],
) -> None:
    reference = (
        tuple(records[0].flow.shape),
        tuple(records[0].planned_action.shape),
        tuple(records[0].departure.shape),
        tuple(records[0].hdv_context.shape),
        tuple(records[0].hdv_target.shape),
    )
    for record in records[1:]:
        current = (
            tuple(record.flow.shape),
            tuple(record.planned_action.shape),
            tuple(record.departure.shape),
            tuple(record.hdv_context.shape),
            tuple(record.hdv_target.shape),
        )
        if current != reference:
            raise ValueError("warmup episode contains inconsistent tensor shapes")


def _validate_consistent_online_shapes(
    transitions: tuple[OnlineReplayTransition, ...],
) -> None:
    reference = (
        tuple(transitions[0].flow.shape),
        tuple(transitions[0].action.shape),
        tuple(transitions[0].departure.shape),
        tuple(transitions[0].hdv_context.shape),
        tuple(transitions[0].hdv_target.shape),
    )
    for transition in transitions[1:]:
        current = (
            tuple(transition.flow.shape),
            tuple(transition.action.shape),
            tuple(transition.departure.shape),
            tuple(transition.hdv_context.shape),
            tuple(transition.hdv_target.shape),
        )
        if current != reference:
            raise ValueError("online episode contains inconsistent tensor shapes")


def _shift_once(values: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    if not values:
        raise ValueError("cannot shift an empty conditioning sequence")
    return [torch.zeros_like(values[0]), *[value.clone() for value in values[:-1]]]


def _stack_nested(rows: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
    if not rows or not rows[0]:
        raise ValueError("cannot stack an empty tensor grid")
    return torch.stack(
        [torch.stack([value.clone() for value in row], dim=0) for row in rows],
        dim=0,
    )


def _sample_keys(
    *,
    candidates: Sequence[tuple[int, int]],
    order: list[tuple[int, int]],
    cursor: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], int, list[tuple[int, int]]]:
    if not candidates:
        raise RuntimeError("cannot sample from an empty replay")
    if set(order) != set(candidates):
        order = []
        cursor = 0
    result: list[tuple[int, int]] = []
    while len(result) < batch_size:
        if cursor >= len(order):
            permutation = rng.permutation(len(candidates)).tolist()
            order = [candidates[int(index)] for index in permutation]
            cursor = 0
        result.append(order[cursor])
        cursor += 1
    return order, cursor, result


def _sampler_state(
    order: Sequence[tuple[int, int]],
    cursor: int,
) -> dict[str, Any]:
    return {
        "order": [
            {"storage_id": storage_id, "index": index}
            for storage_id, index in order
        ],
        "cursor": cursor,
    }


def _load_sampler(
    state: Any,
) -> tuple[list[tuple[int, int]], int]:
    if not isinstance(state, Mapping):
        raise TypeError("replay sampler must be an object")
    _require_keys(state, {"order", "cursor"}, "replay sampler")
    raw_order = state["order"]
    if not isinstance(raw_order, list):
        raise TypeError("replay sampler order must be a list")
    order: list[tuple[int, int]] = []
    for item in raw_order:
        _require_keys(item, {"storage_id", "index"}, "replay sampler item")
        order.append(
            (
                _integer(item["storage_id"], name="storage_id"),
                _integer(item["index"], name="index"),
            )
        )
    cursor = _integer(state["cursor"], name="cursor")
    if cursor < 0 or cursor > len(order):
        raise ValueError("replay sampler cursor is out of range")
    return order, cursor


def _require_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} has an invalid schema")


__all__ = [
    "MAX_ONLINE_REPLAY_EPISODES",
    "OnlineReplayBuffer",
    "OnlineReplayTransition",
    "OnlineTransitionBatch",
    "VehicleSnapshot",
    "WarmupReplayBuffer",
    "WarmupReplayRecord",
    "WarmupSequenceBatch",
]
