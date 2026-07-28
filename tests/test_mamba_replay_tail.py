from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from baseline.mamba import SparseReplayBuffer


OBS_DIM = 2
ACTION_DIM = 2


class _ScriptedRng:
    """Minimal integer RNG that also verifies each sampler upper bound."""

    def __init__(self, draws: Iterable[tuple[int, int]]) -> None:
        self._draws = iter(draws)

    def integers(self, high: int) -> int:
        expected_high, value = next(self._draws)
        assert high == expected_high
        assert 0 <= value < high
        return value


def _frame(step: int, *, terminal: bool = False, present: bool = True) -> dict:
    slots = [0] if present else []
    return {
        "episode": 1,
        "decision_idx": step,
        "sim_step": step,
        "presence_slots": slots,
        "obs_slots": slots,
        "decision_slots": slots,
        "obs": np.full((len(slots), OBS_DIM), step + 1, dtype=np.float32),
        "actions": np.zeros(len(slots), dtype=np.int64),
        "rewards": np.zeros(len(slots), dtype=np.float32),
        "old_log_probs": np.zeros(len(slots), dtype=np.float32),
        "action_valid": np.ones((len(slots), ACTION_DIM), dtype=bool),
        "terminated_slots": [0] if terminal else [],
        "truncated": False,
    }


def _buffer(*, final_terminal: bool = False) -> SparseReplayBuffer:
    replay = SparseReplayBuffer(
        capacity_episodes=1,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        seed=0,
    )
    frames = [_frame(step) for step in range(10)]
    if final_terminal:
        frames[-1] = _frame(9, terminal=True, present=False)
    replay.add_episode(frames)
    return replay


def _sample_at(replay: SparseReplayBuffer, start: int):
    # One candidate episode, then a start from 0..6 inclusive: ten frames minus
    # burn-in two minus the two frames required for a supervised transition.
    replay.rng = _ScriptedRng([(1, 0), (7, start)])  # type: ignore[assignment]
    return replay.sample(
        batch_size=1,
        sequence_length=6,
        burn_in=2,
        max_agents_per_chunk=8,
        device="cpu",
    )


def test_sampler_can_select_a_padded_tail_window() -> None:
    batch = _sample_at(_buffer(), start=6)

    assert batch.valid_step[:, 0].tolist() == [True, True, True, True, False, False]
    assert batch.observation[:4, 0, 0, 0].tolist() == [7.0, 8.0, 9.0, 10.0]


def test_final_terminal_has_multiple_supervised_start_offsets() -> None:
    supervised_offsets = []
    for start in (4, 5, 6):
        batch = _sample_at(_buffer(final_terminal=True), start=start)
        next_terminal = torch.zeros_like(batch.terminal)
        next_terminal[:-1] = batch.terminal[1:]
        has_next = torch.zeros_like(batch.valid_step)
        has_next[:-1] = batch.valid_step[1:]
        temporal = torch.arange(6).view(-1, 1, 1) >= 2
        continuation_mask = temporal & has_next.unsqueeze(-1) & batch.presence
        supervised = torch.nonzero(
            continuation_mask & next_terminal, as_tuple=False,
        )

        assert supervised.shape[0] == 1
        supervised_offsets.append(int(supervised[0, 0]))

    # Under full-window-only sampling the final terminal was reachable only at
    # offset five (start four).  Padded tails additionally expose offsets 4/3.
    assert supervised_offsets == [4, 3, 2]


def test_tail_padding_and_burn_in_do_not_leak_into_loss_mask() -> None:
    batch = _sample_at(_buffer(), start=6)

    assert not batch.loss_mask[:2].any()
    assert batch.loss_mask[2:4, 0, 0].all()
    assert not batch.loss_mask[4:].any()
    assert not batch.presence[4:].any()
