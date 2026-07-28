from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from baseline.mamba import (
    AgentAttentionEncoder,
    MambaConfig,
    MambaLearner,
    SparseFrame,
    collate_sparse_windows,
    fixed_slot_agent_chunks,
    masked_world_model_loss,
    rotating_agent_chunks,
)


def _config(**overrides) -> MambaConfig:
    values = dict(
        obs_dim=6,
        action_dim=4,
        n_slots=40,
        device="cpu",
        seed=7,
        hidden_size=32,
        model_hidden_size=32,
        embedding_size=32,
        deterministic_size=32,
        n_categoricals=4,
        n_classes=4,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        replay_capacity_episodes=2,
        sequence_length=6,
        burn_in=2,
        max_agents_per_chunk=8,
        sequence_minibatch_size=1,
        min_model_replay_frames=1,
        min_agent_replay_frames=1,
        model_epochs=1,
        agent_epochs=1,
        ppo_epochs=2,
        critic_epochs=2,
        critic_warmup_updates=0,
        target_tau=1.0,
        imagination_horizon=3,
        free_nats=0.0,
    )
    values.update(overrides)
    return MambaConfig(**values)


def _frame(step: int, slots=(0, 3, 9), *, terminal=()) -> dict:
    slots = list(slots)
    rng = np.random.default_rng(step)
    return {
        "episode": 1,
        "decision_idx": step,
        "sim_step": step * 30,
        "presence_slots": slots,
        "obs_slots": slots,
        "decision_slots": slots,
        "obs": rng.normal(size=(len(slots), 6)).astype(np.float32),
        "actions": np.asarray([step % 4] * len(slots), dtype=np.int64),
        "rewards": rng.normal(size=len(slots)).astype(np.float32),
        "old_log_probs": np.full(len(slots), -math.log(4.0), dtype=np.float32),
        "action_valid": np.ones((len(slots), 4), dtype=bool),
        "terminated_slots": list(terminal),
        "truncated": False,
    }


def test_attention_has_no_agent_limit_and_is_permutation_equivariant():
    torch.manual_seed(0)
    encoder = AgentAttentionEncoder(32, layers=2, heads=4, dropout=0.0).eval()
    inputs = torch.randn(2, 57, 32)
    presence = torch.ones(2, 57, dtype=torch.bool)
    permutation = torch.randperm(57)
    inverse = torch.argsort(permutation)

    expected = encoder(inputs, presence)
    actual = encoder(inputs[:, permutation], presence[:, permutation])[:, inverse]

    assert expected.shape == (2, 57, 32)
    assert torch.allclose(expected, actual, atol=2e-5, rtol=2e-5)


def test_attention_all_padding_is_finite_and_zero():
    encoder = AgentAttentionEncoder(16, layers=1, heads=4)
    output = encoder(torch.randn(3, 5, 16), torch.zeros(3, 5, dtype=torch.bool))
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0


def test_sparse_collate_preserves_global_slots_and_all_masks():
    frames = [SparseFrame.from_mapping(_frame(index), obs_dim=6, action_dim=4) for index in range(6)]
    frames[3] = SparseFrame.from_mapping(
        {
            **_frame(3),
            "obs_slots": [0, 9],
            "decision_slots": [0],
            "obs": _frame(3)["obs"][[0, 2]],
            "actions": [3],
            "rewards": [1.0],
            "old_log_probs": [-1.0],
            "action_valid": [[True, True, True, True]],
        },
        obs_dim=6,
        action_dim=4,
    )
    batch = collate_sparse_windows(
        [frames], obs_dim=6, action_dim=4, sequence_length=6, burn_in=2,
        max_agents_per_chunk=16,
    )

    assert batch.observation.shape == (6, 1, 3, 6)
    assert batch.global_slots[0].tolist() == [0, 3, 9]
    assert batch.presence.sum().item() == 18
    assert batch.obs_valid[3].sum().item() == 2
    assert batch.decision[3].sum().item() == 1
    assert not batch.loss_mask[:2].any()
    assert batch.loss_mask[2:].all()


def test_rotating_chunks_cover_fontenay_sized_slot_set_once():
    chunks = rotating_agent_chunks(range(827), 32, rotation=19)
    flattened = np.concatenate(chunks)
    assert max(len(chunk) for chunk in chunks) == 32
    assert sorted(flattened.tolist()) == list(range(827))
    assert len(np.unique(flattened)) == 827


def test_fixed_slot_chunks_keep_peers_stable_when_slots_disappear():
    chunks = fixed_slot_agent_chunks([0, 31, 32, 63, 64], 32)
    assert [chunk.tolist() for chunk in chunks] == [[0, 31], [32, 63], [64]]

    without_early_slot = fixed_slot_agent_chunks([31, 32, 63, 64], 32)
    assert [chunk.tolist() for chunk in without_early_slot] == [
        [31], [32, 63], [64],
    ]


def test_masked_world_model_loss_is_finite_and_backpropagates():
    cfg = _config()
    learner = MambaLearner(cfg, [str(index) for index in range(cfg.n_slots)])
    frames = [SparseFrame.from_mapping(_frame(index), obs_dim=6, action_dim=4) for index in range(6)]
    batch = collate_sparse_windows(
        [frames], obs_dim=6, action_dim=4, sequence_length=6, burn_in=2,
        max_agents_per_chunk=8,
    )

    loss, metrics, _ = masked_world_model_loss(cfg, learner.world_model, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in learner.world_model.parameters()
    )


def test_empty_decision_frames_do_not_create_nan_losses():
    cfg = _config()
    learner = MambaLearner(cfg, [str(index) for index in range(cfg.n_slots)])
    empty = {
        **_frame(0, slots=()),
        "obs": np.empty((0, cfg.obs_dim), dtype=np.float32),
        "action_valid": np.empty((0, cfg.action_dim), dtype=bool),
    }
    frames = [
        SparseFrame.from_mapping({**empty, "decision_idx": index}, obs_dim=6, action_dim=4)
        for index in range(cfg.sequence_length)
    ]
    batch = collate_sparse_windows(
        [frames], obs_dim=6, action_dim=4, sequence_length=cfg.sequence_length,
        burn_in=cfg.burn_in, max_agents_per_chunk=cfg.max_agents_per_chunk,
    )

    loss, metrics, _ = masked_world_model_loss(cfg, learner.world_model, batch)

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert metrics["model_observations"].item() == 0
    assert metrics["model_decisions"].item() == 0


def test_disappearance_terminal_is_kept_without_marking_truncation():
    frames = [
        SparseFrame.from_mapping(_frame(0, slots=(0, 3)), obs_dim=6, action_dim=4),
        SparseFrame.from_mapping(
            _frame(1, slots=(0,), terminal=(3,)), obs_dim=6, action_dim=4,
        ),
    ]
    batch = collate_sparse_windows(
        [frames], obs_dim=6, action_dim=4, sequence_length=3, burn_in=0,
        max_agents_per_chunk=8,
    )
    slot_three = batch.global_slots[0].tolist().index(3)

    assert batch.presence[0, 0, slot_three]
    assert not batch.presence[1, 0, slot_three]
    assert batch.terminal[1, 0, slot_three]
    assert not batch.truncated.any()


def test_online_controller_enforces_one_begin_and_action_mask():
    cfg = _config(n_slots=4)
    learner = MambaLearner(cfg, ["a", "b", "c", "d"])
    learner.reset_episode()
    learner.begin_decision([0, 2])
    result = learner.act(
        2,
        np.ones(cfg.obs_dim, dtype=np.float32),
        np.asarray([False, False, True, False]),
    )
    assert result["action"] == 2
    assert math.isfinite(result["log_prob"])
    with pytest.raises(RuntimeError):
        learner.begin_decision([0])
    frame = _frame(0, slots=(0, 2))
    frame["actions"] = np.asarray([0, 2])
    learner.end_decision(frame)
    learner.end_episode()
    assert len(learner.replay) == 1


def test_update_runs_model_and_imagined_ppo_with_finite_metrics():
    cfg = _config(n_slots=10, ppo_epochs=1, imagination_horizon=2)
    learner = MambaLearner(cfg, [str(index) for index in range(10)])
    learner.replay.add_episode([_frame(index) for index in range(7)])

    metrics = learner.update()

    assert metrics["skipped"] == 0.0
    assert metrics["agent_skipped"] == 0.0
    for value in metrics.values():
        assert math.isfinite(value)


def test_checkpoint_round_trip_restores_replay_optimizers_rng_and_roles(tmp_path: Path):
    cfg = _config(n_slots=10, ppo_epochs=1, imagination_horizon=2)
    ids = [str(index) for index in range(10)]
    learner = MambaLearner(cfg, ids)
    learner.replay.add_episode([_frame(index) for index in range(7)])
    learner.update()
    path = tmp_path / "mamba.pt"
    learner.save_checkpoint(
        path, episode=11, metrics={"reward": 2.5}, role_source="roles.json", role_hash="abc",
    )

    restored = MambaLearner(cfg, ids)
    episode = restored.load_checkpoint(
        path, role_source="roles.json", role_hash="abc",
    )

    assert episode == 11
    assert restored.update_count == learner.update_count
    assert restored.replay.n_frames == learner.replay.n_frames
    assert restored.replay.state_dict()["chunk_cursor"] == learner.replay.state_dict()["chunk_cursor"]
    for expected, actual in zip(learner.actor.parameters(), restored.actor.parameters(), strict=True):
        assert torch.equal(expected, actual)

    wrong_roles = MambaLearner(cfg, list(reversed(ids)))
    with pytest.raises(ValueError, match="role mapping"):
        wrong_roles.load_checkpoint(path)
