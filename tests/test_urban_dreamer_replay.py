from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch

from urban_dreamer.learning.replay import (
    MAX_ONLINE_REPLAY_EPISODES,
    OnlineReplayBuffer,
    OnlineReplayTransition,
    VehicleSnapshot,
    WarmupReplayBuffer,
    WarmupReplayRecord,
)


def _flow(value: float) -> torch.Tensor:
    result = torch.zeros((2, 8), dtype=torch.float32)
    result[:, 0] = float(value)
    result[:, 1] = float(value) + 0.5
    return result


def _warmup_record(
    *,
    episode: int,
    step: int,
    terminal: bool = False,
) -> WarmupReplayRecord:
    return WarmupReplayRecord(
        episode=episode,
        step_index=step,
        flow=_flow(float(step)),
        planned_action=torch.tensor([float(step + 1), float(step + 2)]),
        departure=torch.tensor([float(step + 10), float(step + 20)]),
        context=torch.tensor(
            [float(step + 100), float(step + 110), float(step + 120)]
        ),
        hdv_context=torch.tensor([float(step + 30), float(step + 40)]),
        hdv_target=torch.tensor([float(step + 50), float(step + 60)]),
        terminal=terminal,
    )


def _vehicles(step: int) -> VehicleSnapshot:
    return VehicleSnapshot(
        vehicle_ids=(f"veh-{step}",),
        current_edges=torch.tensor([step % 2]),
        destination_edges=torch.tensor([1]),
        departure_steps=torch.tensor([step]),
    )


def _online_transition(
    *,
    episode: int,
    step: int,
    terminal: bool = False,
) -> OnlineReplayTransition:
    return OnlineReplayTransition(
        episode=episode,
        step_index=step,
        flow=_flow(float(step)),
        action=torch.tensor([float(step + 1), float(step + 2)]),
        next_flow=_flow(float(step + 1)),
        departure=torch.tensor([float(step + 10), float(step + 20)]),
        context=torch.tensor(
            [float(step + 100), float(step + 110), float(step + 120)]
        ),
        hdv_context=torch.tensor([float(step + 30), float(step + 40)]),
        hdv_target=torch.tensor([float(step + 50), float(step + 60)]),
        vehicles=_vehicles(step),
        active_vehicle_seconds=float(step + 12),
        transition_seconds=30.0,
        terminal=terminal,
    )


def _assert_plain(value: Any) -> None:
    if value is None or isinstance(value, (str, int, float, bool, torch.Tensor)):
        return
    if isinstance(value, list):
        for item in value:
            _assert_plain(item)
        return
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value)
        for item in value.values():
            _assert_plain(item)
        return
    raise AssertionError(f"non-plain replay state: {type(value)!r}")


def _assert_equal(left: Any, right: Any) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right):
            _assert_equal(lhs, rhs)
    else:
        assert left == right


def _assert_batch_equal(left: Any, right: Any) -> None:
    for field_name in left.__dataclass_fields__:
        lhs = getattr(left, field_name)
        rhs = getattr(right, field_name)
        if isinstance(lhs, tuple) and lhs and isinstance(lhs[0], VehicleSnapshot):
            assert len(lhs) == len(rhs)
            for lhs_snapshot, rhs_snapshot in zip(lhs, rhs):
                _assert_equal(
                    lhs_snapshot.state_dict(),
                    rhs_snapshot.state_dict(),
                )
        else:
            _assert_equal(lhs, rhs)


def test_warmup_sequence_shifts_conditioning_once_and_keeps_state_order() -> None:
    replay = WarmupReplayBuffer(sequence_length=3)
    assert replay.append_episode(
        [
            _warmup_record(episode=7, step=0),
            _warmup_record(episode=7, step=1),
            _warmup_record(episode=7, step=2),
            _warmup_record(episode=7, step=3, terminal=True),
        ],
        total_travel_minutes=42.0,
    )

    batch = replay.sample_batch(batch_size=1, rng=np.random.default_rng(0))

    assert tuple(batch.flow.shape) == (1, 3, 2, 8)
    torch.testing.assert_close(batch.flow[0, :, 0, 0], torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(batch.action[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(
        batch.departure[0, :, 0],
        torch.tensor([0.0, 10.0, 11.0]),
    )
    torch.testing.assert_close(
        batch.context[0, :, 0],
        torch.tensor([0.0, 100.0, 101.0]),
    )
    torch.testing.assert_close(
        batch.hdv_context[0, :, 0],
        torch.tensor([0.0, 30.0, 31.0]),
    )
    torch.testing.assert_close(
        batch.hdv_target[0, :, 0],
        torch.tensor([51.0, 52.0]),
    )
    assert batch.terminal.shape == (1, 2)
    assert batch.total_travel_minutes.tolist() == [42.0]


def test_online_replay_preserves_explicit_pairs_and_vehicle_seed_without_shift() -> None:
    replay = OnlineReplayBuffer()
    assert replay.append_episode(
        [
            _online_transition(episode=5, step=0),
            _online_transition(episode=5, step=1, terminal=True),
        ],
        total_travel_minutes=12.5,
    )

    batch = replay.sample_batch(batch_size=2, rng=np.random.default_rng(0))

    sampled = {
        (
            int(batch.step_index[index]),
            float(batch.action[index, 0]),
            float(batch.flow[index, 0, 0]),
            float(batch.next_flow[index, 0, 0]),
            batch.vehicles[index].vehicle_ids,
            float(batch.total_travel_minutes[index]),
        )
        for index in range(2)
    }
    assert sampled == {
        (0, 1.0, 0.0, 1.0, ("veh-0",), 12.5),
        (1, 2.0, 1.0, 2.0, ("veh-1",), 12.5),
    }


def test_warmup_rejects_incomplete_episode_but_online_keeps_valid_prefix() -> None:
    warmup = WarmupReplayBuffer(sequence_length=2)
    online = OnlineReplayBuffer()

    assert not warmup.append_episode(
        [
            _warmup_record(episode=1, step=0),
            _warmup_record(episode=1, step=1),
        ],
        total_travel_minutes=1.0,
    )
    assert online.append_episode(
        [
            _online_transition(episode=1, step=0),
            _online_transition(episode=1, step=1),
        ],
        total_travel_minutes=1.0,
    )
    assert warmup.episodes == ()
    assert len(online.episodes) == 1
    assert [transition.terminal for transition in online.episodes[0]] == [
        False,
        False,
    ]

    restored = OnlineReplayBuffer()
    restored.load_state_dict(online.state_dict())

    assert len(restored.episodes) == 1
    assert [transition.terminal for transition in restored.episodes[0]] == [
        False,
        False,
    ]


def test_warmup_replay_keeps_all_complete_episodes_and_online_replay_rolls() -> None:
    warmup = WarmupReplayBuffer(sequence_length=2)
    online = OnlineReplayBuffer()

    for episode in range(MAX_ONLINE_REPLAY_EPISODES + 1):
        warmup.append_episode(
            [
                _warmup_record(episode=episode, step=0),
                _warmup_record(episode=episode, step=1, terminal=True),
            ],
            total_travel_minutes=float(episode + 1),
        )
        online.append_episode(
            [_online_transition(episode=episode, step=0, terminal=True)],
            total_travel_minutes=float(episode + 1),
        )

    assert len(warmup.episodes) == MAX_ONLINE_REPLAY_EPISODES + 1
    assert len(online.episodes) == MAX_ONLINE_REPLAY_EPISODES
    assert [episode[0].episode for episode in warmup.episodes] == list(range(9))
    assert [episode[0].episode for episode in online.episodes] == list(range(1, 9))


def test_replay_state_roundtrip_preserves_sampling_cursor_and_vehicle_state() -> None:
    warmup = WarmupReplayBuffer(sequence_length=2)
    online = OnlineReplayBuffer()
    for episode in (10, 11):
        warmup.append_episode(
            [
                _warmup_record(episode=episode, step=0),
                _warmup_record(episode=episode, step=1),
                _warmup_record(episode=episode, step=2, terminal=True),
            ],
            total_travel_minutes=float(episode),
        )
        online.append_episode(
            [
                _online_transition(episode=episode, step=0),
                _online_transition(episode=episode, step=1, terminal=True),
            ],
            total_travel_minutes=float(episode),
        )

    rng = np.random.default_rng(123)
    warmup.sample_batch(batch_size=1, rng=rng)
    online.sample_batch(batch_size=1, rng=rng)
    warmup_state = warmup.state_dict()
    online_state = online.state_dict()
    _assert_plain(warmup_state)
    _assert_plain(online_state)

    restored_warmup = WarmupReplayBuffer(sequence_length=2)
    restored_online = OnlineReplayBuffer()
    restored_warmup.load_state_dict(deepcopy(warmup_state))
    restored_online.load_state_dict(deepcopy(online_state))
    _assert_equal(restored_warmup.state_dict(), warmup_state)
    _assert_equal(restored_online.state_dict(), online_state)

    rng_state = deepcopy(rng.bit_generator.state)
    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = deepcopy(rng_state)
    _assert_batch_equal(
        warmup.sample_batch(batch_size=3, rng=rng),
        restored_warmup.sample_batch(batch_size=3, rng=restored_rng),
    )
    _assert_batch_equal(
        online.sample_batch(batch_size=3, rng=rng),
        restored_online.sample_batch(batch_size=3, rng=restored_rng),
    )
