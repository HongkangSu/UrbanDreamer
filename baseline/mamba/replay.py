"""Sparse episode replay with fixed global agent slots.

Only agents represented by a frame are stored.  Dense ``[T, B, A, ...]``
tensors are materialized for a sampled, at-most-``max_agents_per_chunk``
communication chunk, avoiding ``capacity * total_agents`` preallocation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def _array(value: Any, dtype: np.dtype[Any], shape_tail: tuple[int, ...] = ()) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.size == 0:
        return np.empty((0, *shape_tail), dtype=dtype)
    return result


def _unique_slots(values: Sequence[int], *, name: str) -> np.ndarray:
    slots = np.asarray(values, dtype=np.int64).reshape(-1)
    if len(slots) != len(np.unique(slots)):
        raise ValueError(f"{name} contains duplicate global slots")
    if (slots < 0).any():
        raise ValueError(f"{name} contains a negative global slot")
    return slots


@dataclass(slots=True)
class SparseFrame:
    presence_slots: np.ndarray
    obs_slots: np.ndarray
    decision_slots: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    old_log_probs: np.ndarray
    action_valid: np.ndarray
    terminal_slots: np.ndarray
    truncated: bool = False
    episode: int = 0
    decision_idx: int = 0
    sim_step: int = 0

    @classmethod
    def from_mapping(
        cls,
        frame: Mapping[str, Any],
        *,
        obs_dim: int,
        action_dim: int,
    ) -> "SparseFrame":
        presence = _unique_slots(frame.get("presence_slots", ()), name="presence_slots")
        obs_slots = _unique_slots(
            frame.get("obs_slots", frame.get("decision_slots", ())), name="obs_slots",
        )
        decision_slots = _unique_slots(
            frame.get("decision_slots", frame.get("obs_slots", ())), name="decision_slots",
        )
        terminal = _unique_slots(frame.get("terminated_slots", ()), name="terminated_slots")

        observations = _array(frame.get("obs", ()), np.dtype(np.float32), (obs_dim,))
        observations = observations.reshape(-1, obs_dim)
        actions = _array(frame.get("actions", ()), np.dtype(np.int64)).reshape(-1)
        rewards = _array(frame.get("rewards", np.zeros(len(actions))), np.dtype(np.float32)).reshape(-1)
        old_log_probs = _array(
            frame.get("old_log_probs", np.zeros(len(actions))), np.dtype(np.float32),
        ).reshape(-1)
        action_valid = _array(
            frame.get("action_valid", np.ones((len(actions), action_dim))),
            np.dtype(np.bool_),
            (action_dim,),
        ).reshape(-1, action_dim)

        if observations.shape[0] != len(obs_slots):
            raise ValueError("obs row count must equal obs_slots length")
        for name, value in (
            ("actions", actions), ("rewards", rewards),
            ("old_log_probs", old_log_probs), ("action_valid", action_valid),
        ):
            if value.shape[0] != len(decision_slots):
                raise ValueError(f"{name} row count must equal decision_slots length")
        if len(actions) and ((actions < 0) | (actions >= action_dim)).any():
            raise ValueError("actions contains an out-of-range action")
        if len(actions) and (~action_valid[np.arange(len(actions)), actions]).any():
            raise ValueError("recorded action is invalid under action_valid")
        if not set(obs_slots).issubset(set(presence)):
            raise ValueError("obs_slots must be a subset of presence_slots")
        if not set(decision_slots).issubset(set(obs_slots)):
            raise ValueError("decision_slots must be a subset of obs_slots")

        return cls(
            presence_slots=presence,
            obs_slots=obs_slots,
            decision_slots=decision_slots,
            observations=observations.copy(),
            actions=actions.copy(),
            rewards=rewards.copy(),
            old_log_probs=old_log_probs.copy(),
            action_valid=action_valid.copy(),
            terminal_slots=terminal,
            truncated=bool(frame.get("truncated", False)),
            episode=int(frame.get("episode", 0)),
            decision_idx=int(frame.get("decision_idx", 0)),
            sim_step=int(frame.get("sim_step", 0)),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "presence_slots": self.presence_slots,
            "obs_slots": self.obs_slots,
            "decision_slots": self.decision_slots,
            "obs": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "old_log_probs": self.old_log_probs,
            "action_valid": self.action_valid,
            "terminated_slots": self.terminal_slots,
            "truncated": self.truncated,
            "episode": self.episode,
            "decision_idx": self.decision_idx,
            "sim_step": self.sim_step,
        }


@dataclass(slots=True)
class SparseEpisode:
    episode: int = 0
    frames: list[SparseFrame] = field(default_factory=list)

    def append(self, frame: SparseFrame) -> None:
        self.frames.append(frame)

    def __len__(self) -> int:
        return len(self.frames)


@dataclass(slots=True)
class SparseBatch:
    observation: Tensor
    action: Tensor
    reward: Tensor
    old_log_prob: Tensor
    presence: Tensor
    obs_valid: Tensor
    decision: Tensor
    terminal: Tensor
    action_valid: Tensor
    truncated: Tensor
    loss_mask: Tensor
    valid_step: Tensor
    global_slots: Tensor

    def to(self, device: torch.device | str) -> "SparseBatch":
        return SparseBatch(**{
            name: getattr(self, name).to(device)
            for name in self.__dataclass_fields__
        })

    def as_dict(self) -> dict[str, Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def rotating_agent_chunks(
    slots: Iterable[int], chunk_size: int, rotation: int = 0
) -> list[np.ndarray]:
    """Return disjoint chunks after a cyclic rotation; every slot appears once."""
    ordered = np.asarray(sorted(set(int(slot) for slot in slots)), dtype=np.int64)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not len(ordered):
        return [np.empty((0,), dtype=np.int64)]
    shift = int(rotation) % len(ordered)
    rotated = np.concatenate((ordered[shift:], ordered[:shift]))
    return [rotated[index:index + chunk_size] for index in range(0, len(rotated), chunk_size)]


def fixed_slot_agent_chunks(
    slots: Iterable[int], chunk_size: int,
) -> list[np.ndarray]:
    """Group fixed global identities into stable, non-overlapping slot blocks."""
    ordered = np.asarray(sorted(set(int(slot) for slot in slots)), dtype=np.int64)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not len(ordered):
        return [np.empty((0,), dtype=np.int64)]
    block_ids = ordered // int(chunk_size)
    return [ordered[block_ids == block] for block in np.unique(block_ids)]


def _slots_in_window(frames: Sequence[SparseFrame]) -> set[int]:
    slots: set[int] = set()
    for frame in frames:
        slots.update(int(value) for value in frame.presence_slots)
        slots.update(int(value) for value in frame.terminal_slots)
    return slots


def collate_sparse_windows(
    windows: Sequence[Sequence[SparseFrame]],
    *,
    obs_dim: int,
    action_dim: int,
    sequence_length: int,
    burn_in: int,
    max_agents_per_chunk: int,
    chunk_indices: Sequence[int] | None = None,
    rotations: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
) -> SparseBatch:
    if not windows:
        raise ValueError("cannot collate an empty window list")
    batch_size = len(windows)
    chunk_indices = list(chunk_indices or [0] * batch_size)
    rotations = list(rotations or [0] * batch_size)
    if len(chunk_indices) != batch_size or len(rotations) != batch_size:
        raise ValueError("chunk_indices and rotations must match window count")

    selected: list[np.ndarray] = []
    for frames, chunk_index, _rotation in zip(windows, chunk_indices, rotations, strict=True):
        chunks = fixed_slot_agent_chunks(
            _slots_in_window(frames), max_agents_per_chunk,
        )
        selected.append(chunks[int(chunk_index) % len(chunks)])
    agents = max(1, max(len(item) for item in selected))
    shape = (sequence_length, batch_size, agents)
    observation = torch.zeros((*shape, obs_dim), dtype=torch.float32)
    action_index = torch.zeros(shape, dtype=torch.long)
    reward = torch.zeros((*shape, 1), dtype=torch.float32)
    old_log_prob = torch.zeros(shape, dtype=torch.float32)
    presence = torch.zeros(shape, dtype=torch.bool)
    obs_valid = torch.zeros(shape, dtype=torch.bool)
    decision = torch.zeros(shape, dtype=torch.bool)
    terminal = torch.zeros(shape, dtype=torch.bool)
    action_valid = torch.zeros((*shape, action_dim), dtype=torch.bool)
    truncated = torch.zeros((sequence_length, batch_size), dtype=torch.bool)
    valid_step = torch.zeros((sequence_length, batch_size), dtype=torch.bool)
    global_slots = torch.full((batch_size, agents), -1, dtype=torch.long)

    for batch_index, (frames, slots) in enumerate(zip(windows, selected, strict=True)):
        positions = {int(slot): index for index, slot in enumerate(slots)}
        if len(slots):
            global_slots[batch_index, :len(slots)] = torch.from_numpy(slots)
        for time_index, frame in enumerate(frames[:sequence_length]):
            valid_step[time_index, batch_index] = True
            truncated[time_index, batch_index] = frame.truncated
            for slot in frame.presence_slots:
                if int(slot) in positions:
                    presence[time_index, batch_index, positions[int(slot)]] = True
            for row, slot in enumerate(frame.obs_slots):
                if int(slot) in positions:
                    pos = positions[int(slot)]
                    observation[time_index, batch_index, pos] = torch.from_numpy(frame.observations[row])
                    obs_valid[time_index, batch_index, pos] = True
            for row, slot in enumerate(frame.decision_slots):
                if int(slot) in positions:
                    pos = positions[int(slot)]
                    action_index[time_index, batch_index, pos] = int(frame.actions[row])
                    reward[time_index, batch_index, pos, 0] = float(frame.rewards[row])
                    old_log_prob[time_index, batch_index, pos] = float(frame.old_log_probs[row])
                    action_valid[time_index, batch_index, pos] = torch.from_numpy(frame.action_valid[row])
                    decision[time_index, batch_index, pos] = True
            for slot in frame.terminal_slots:
                if int(slot) in positions:
                    terminal[time_index, batch_index, positions[int(slot)]] = True

    action = F.one_hot(action_index, action_dim).to(torch.float32)
    action *= decision.unsqueeze(-1)
    temporal = torch.arange(sequence_length).view(-1, 1, 1) >= int(burn_in)
    loss_mask = temporal & valid_step.unsqueeze(-1) & presence
    return SparseBatch(
        observation=observation.to(device),
        action=action.to(device),
        reward=reward.to(device),
        old_log_prob=old_log_prob.to(device),
        presence=presence.to(device),
        obs_valid=obs_valid.to(device),
        decision=decision.to(device),
        terminal=terminal.to(device),
        action_valid=action_valid.to(device),
        truncated=truncated.to(device),
        loss_mask=loss_mask.to(device),
        valid_step=valid_step.to(device),
        global_slots=global_slots.to(device),
    )


class SparseReplayBuffer:
    def __init__(
        self,
        *,
        capacity_episodes: int,
        obs_dim: int,
        action_dim: int,
        seed: int = 42,
    ) -> None:
        if capacity_episodes <= 0:
            raise ValueError("capacity_episodes must be positive")
        self.capacity_episodes = int(capacity_episodes)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.episodes: deque[SparseEpisode] = deque(maxlen=self.capacity_episodes)
        self.rng = np.random.default_rng(seed)
        self._chunk_cursor = 0

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def n_frames(self) -> int:
        return sum(len(episode) for episode in self.episodes)

    def add_episode(self, episode: SparseEpisode | Sequence[Mapping[str, Any]]) -> None:
        if not isinstance(episode, SparseEpisode):
            frames = [
                SparseFrame.from_mapping(item, obs_dim=self.obs_dim, action_dim=self.action_dim)
                for item in episode
            ]
            number = frames[0].episode if frames else 0
            episode = SparseEpisode(number, frames)
        if not episode.frames:
            return
        self.episodes.append(episode)

    def iter_window_chunks(
        self, frames: Sequence[SparseFrame], *, sequence_length: int, chunk_size: int
    ) -> Iterator[tuple[list[SparseFrame], np.ndarray]]:
        window = list(frames[:sequence_length])
        for chunk in fixed_slot_agent_chunks(_slots_in_window(window), chunk_size):
            yield window, chunk

    def sample(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        burn_in: int,
        max_agents_per_chunk: int,
        device: torch.device | str,
    ) -> SparseBatch:
        candidates = [episode for episode in self.episodes if episode.frames]
        if not candidates:
            raise RuntimeError("cannot sample an empty replay buffer")
        windows: list[list[SparseFrame]] = []
        rotations: list[int] = []
        chunk_indices: list[int] = []
        for _ in range(batch_size):
            episode = candidates[int(self.rng.integers(len(candidates)))]
            # Include padded tail windows instead of sampling only full-length
            # windows.  A latest admissible start leaves ``burn_in`` context
            # frames plus one transition with an available next frame.  This is
            # especially important for continuation learning: with full windows
            # only, an episode-final terminal can occur at a supervised offset
            # for exactly one start, severely under-sampling absorbing events.
            max_start = max(0, len(episode.frames) - burn_in - 2)
            high = max_start + 1
            start = int(self.rng.integers(high))
            frames = episode.frames[start:start + sequence_length]
            slots = _slots_in_window(frames)
            # Walk stable global-slot blocks in order.  This matches online
            # controller communication even when earlier vehicles disappear;
            # packing currently present slots would otherwise change peers.
            rotation = 0
            n_chunks = len(fixed_slot_agent_chunks(slots, max_agents_per_chunk))
            chunk_index = self._chunk_cursor % n_chunks
            self._chunk_cursor += 1
            windows.append(frames)
            rotations.append(rotation)
            chunk_indices.append(chunk_index)
        return collate_sparse_windows(
            windows,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            sequence_length=sequence_length,
            burn_in=burn_in,
            max_agents_per_chunk=max_agents_per_chunk,
            chunk_indices=chunk_indices,
            rotations=rotations,
            device=device,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity_episodes": self.capacity_episodes,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "episodes": [
                {
                    "episode": episode.episode,
                    "frames": [frame.state_dict() for frame in episode.frames],
                }
                for episode in self.episodes
            ],
            "rng_state": self.rng.bit_generator.state,
            "chunk_cursor": self._chunk_cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["obs_dim"]) != self.obs_dim or int(state["action_dim"]) != self.action_dim:
            raise ValueError("replay dimensions do not match learner configuration")
        self.episodes.clear()
        for item in state.get("episodes", ()):
            frames = [
                SparseFrame.from_mapping(
                    frame, obs_dim=self.obs_dim, action_dim=self.action_dim,
                )
                for frame in item["frames"]
            ]
            self.episodes.append(SparseEpisode(int(item["episode"]), frames))
        self.rng.bit_generator.state = state["rng_state"]
        self._chunk_cursor = int(state.get("chunk_cursor", 0))
