from __future__ import annotations

import numpy as np
import pytest
import torch
from copy import deepcopy
from types import MethodType
from torch.nn import functional as F

from baseline.mamba import MambaConfig, MambaLearner, SparseFrame, collate_sparse_windows
from baseline.mamba.learner import _sample_decision_starts, masked_world_model_loss, observe_sparse_batch


def _config(**overrides) -> MambaConfig:
    values = dict(
        obs_dim=3,
        action_dim=2,
        n_slots=3,
        device="cpu",
        seed=19,
        hidden_size=16,
        model_hidden_size=16,
        embedding_size=16,
        deterministic_size=8,
        n_categoricals=2,
        n_classes=2,
        transformer_layers=1,
        transformer_heads=2,
        transformer_dropout=0.0,
        replay_capacity_episodes=2,
        sequence_length=4,
        burn_in=1,
        max_agents_per_chunk=3,
        sequence_minibatch_size=1,
        min_model_replay_frames=1,
        min_agent_replay_frames=1,
        model_epochs=1,
        agent_epochs=1,
        ppo_epochs=1,
        critic_epochs=1,
        critic_warmup_updates=0,
        target_tau=1.0,
        imagination_horizon=2,
        max_imagination_starts=3,
        use_bfloat16=False,
    )
    values.update(overrides)
    return MambaConfig(**values)


def _frame(
    step: int,
    *,
    slots: tuple[int, ...] = (0, 1, 2),
    terminal: tuple[int, ...] = (),
    truncated: bool = False,
) -> dict:
    rng = np.random.default_rng(step + 100)
    return {
        "episode": 1,
        "decision_idx": step,
        "sim_step": step,
        "presence_slots": list(slots),
        "obs_slots": list(slots),
        "decision_slots": list(slots),
        "obs": rng.normal(size=(len(slots), 3)).astype(np.float32),
        "actions": np.asarray([step % 2] * len(slots), dtype=np.int64),
        "rewards": np.full(len(slots), -0.5, dtype=np.float32),
        "old_log_probs": np.full(len(slots), -np.log(2.0), dtype=np.float32),
        "action_valid": np.ones((len(slots), 2), dtype=bool),
        "terminated_slots": list(terminal),
        "truncated": truncated,
    }


def _batch(cfg: MambaConfig, frames: list[dict]):
    sparse = [
        SparseFrame.from_mapping(frame, obs_dim=cfg.obs_dim, action_dim=cfg.action_dim)
        for frame in frames
    ]
    return collate_sparse_windows(
        [sparse],
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        sequence_length=cfg.sequence_length,
        burn_in=cfg.burn_in,
        max_agents_per_chunk=cfg.max_agents_per_chunk,
    )


class _ConstantHead(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (*inputs.shape[:-1], 1), self.value,
            device=inputs.device, dtype=inputs.dtype,
        )


class _ZeroCritic(torch.nn.Module):
    def forward(
        self, features: torch.Tensor, presence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del presence
        return torch.zeros(
            (*features.shape[:-1], 1), device=features.device, dtype=features.dtype,
        )


class _ConstantVectorHead(torch.nn.Module):
    def __init__(self, values: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.values.to(dtype=inputs.dtype).expand(*inputs.shape[:-1], -1)


class _PerAgentRewardHead(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = -torch.arange(
            1,
            inputs.shape[-2] + 1,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        return values.view(*([1] * (inputs.ndim - 2)), -1, 1).expand(
            *inputs.shape[:-1], 1,
        )


def test_production_defaults_do_not_disable_kl_or_leave_a_stale_target() -> None:
    cfg = MambaConfig(obs_dim=3)

    assert cfg.free_nats == 0.0
    assert cfg.target_tau == pytest.approx(0.05)
    assert cfg.normalize_advantages is True
    assert cfg.advantage_clip == pytest.approx(2.5)
    assert cfg.critic_warmup_updates == 20
    assert cfg.imagination_reward_scope == "individual"
    assert cfg.min_model_replay_frames == 100
    assert cfg.min_agent_replay_frames == 2500
    assert cfg.action_information_coef == 1.0
    assert cfg.availability_coef == 1.0
    assert cfg.target_policy_kl == pytest.approx(0.01)


def test_continuation_metrics_include_disappearance_and_censor_truncation() -> None:
    cfg = _config(burn_in=0)
    learner = MambaLearner(cfg, ["a", "b", "c"])
    with torch.no_grad():
        for parameter in learner.world_model.continuation_model.parameters():
            parameter.zero_()
        learner.world_model.continuation_model[-1].bias.fill_(2.0)
    batch = _batch(
        cfg,
        [
            _frame(0, slots=(0, 1)),
            _frame(1, slots=(0,), terminal=(1,)),
            _frame(2, slots=(0,), truncated=True),
            _frame(3, slots=()),
        ],
    )

    loss, metrics, _ = masked_world_model_loss(cfg, learner.world_model, batch)

    assert torch.isfinite(loss)
    assert metrics["continuation_terminal_count"].item() == pytest.approx(1.0)
    assert metrics["continuation_truncated_count"].item() == pytest.approx(1.0)
    assert metrics["continuation_truncated_excluded_count"].item() == pytest.approx(1.0)
    assert 0.0 < metrics["continuation_terminal_fraction"].item() <= 1.0
    n_continue = metrics["continuation_continue_count"]
    n_terminal = metrics["continuation_terminal_count"]
    expected = (
        n_continue * F.softplus(torch.tensor(-2.0))
        + n_terminal * F.softplus(torch.tensor(2.0))
    ) / (n_continue + n_terminal)
    assert metrics["continuation_loss"].item() == pytest.approx(expected.item())
    assert metrics["action_information_loss"].item() > 0.0
    assert metrics["availability_loss"].item() > 0.0


def test_imagination_starts_exclude_burn_in_and_preserve_individual_credit() -> None:
    cfg = _config(burn_in=2, max_imagination_starts=8)
    learner = MambaLearner(cfg, ["a", "b", "c"])
    learner.world_model.reward_model = _PerAgentRewardHead().to(learner.device)
    batch = _batch(cfg, [_frame(step) for step in range(4)])
    _, posterior = observe_sparse_batch(
        learner.world_model, batch, burn_in=cfg.burn_in, sample=False,
    )

    _, active, _ = _sample_decision_starts(
        posterior, batch, max_starts=cfg.max_imagination_starts,
    )
    rollout = learner._imagination_rollout(batch)

    assert active.shape[0] == 2
    assert rollout is not None
    assert rollout["imagined_starts"].item() == pytest.approx(2.0)
    torch.testing.assert_close(
        rollout["imagined_reward"][0, 0],
        torch.tensor([-1.0, -2.0, -3.0]),
    )

    team_cfg = _config(
        burn_in=2,
        max_imagination_starts=8,
        imagination_reward_scope="team",
    )
    team_learner = MambaLearner(team_cfg, ["a", "b", "c"])
    team_learner.world_model.reward_model = _PerAgentRewardHead().to(team_learner.device)
    team_rollout = team_learner._imagination_rollout(_batch(team_cfg, [_frame(step) for step in range(4)]))
    assert team_rollout is not None
    torch.testing.assert_close(
        team_rollout["imagined_reward"][0, 0],
        torch.full((3,), -2.0),
    )


def test_imagination_removes_arrivals_without_double_discounting() -> None:
    cfg = _config(
        n_slots=2,
        burn_in=0,
        sequence_length=4,
        imagination_horizon=3,
        max_imagination_starts=1,
        imagination_alive_threshold=0.5,
        gamma=1.0,
        gae_lambda=1.0,
    )
    learner = MambaLearner(cfg, ["a", "b"])
    learner.world_model.reward_model = _ConstantHead(-1.0).to(learner.device)
    # sigmoid(0) = 0.5: both agents survive the first threshold, then their
    # expected survival mass falls below it and they leave later imagination.
    learner.world_model.continuation_model = _ConstantHead(0.0).to(learner.device)
    learner.target_critic = _ZeroCritic().to(learner.device)
    batch = _batch(cfg, [_frame(step, slots=(0, 1)) for step in range(4)])

    rollout = learner._imagination_rollout(batch)

    assert rollout is not None
    reward = rollout["imagined_reward"][:, 0, 0]
    assert torch.allclose(reward, torch.tensor([-1.0, -1.0, 0.0]), atol=1.0e-6)
    assert rollout["returns"][0, 0, 0].item() == pytest.approx(-1.5)
    assert rollout["survival"][:, 0].sum().item() == pytest.approx(3.0)
    assert rollout["presence"][:, 0].sum().item() == pytest.approx(4.0)
    # The public active mask remains the start mask; dynamic presence is
    # separately exposed for critic/loss masking.
    assert rollout["active"][:, 0].all()


def test_imagination_predicts_future_action_availability() -> None:
    cfg = _config(
        n_slots=2,
        burn_in=0,
        sequence_length=4,
        imagination_horizon=2,
        max_imagination_starts=1,
    )
    learner = MambaLearner(cfg, ["a", "b"])
    learner.world_model.continuation_model = _ConstantHead(10.0).to(learner.device)
    learner.world_model.availability_model = _ConstantVectorHead((-10.0, 10.0)).to(
        learner.device
    )
    frames = []
    for step in range(4):
        frame = _frame(step, slots=(0, 1))
        frame["actions"] = np.zeros(2, dtype=np.int64)
        frame["action_valid"] = np.tile(
            np.asarray([True, False]), (2, 1),
        )
        frames.append(frame)
    rollout = learner._imagination_rollout(_batch(cfg, frames))

    assert rollout is not None
    assert rollout["action_valid"][0, 0, :, 0].all()
    assert not rollout["action_valid"][0, 0, :, 1].any()
    # Candidate zero is the guaranteed real-decision safety anchor, so the
    # prediction is conservatively unioned with it.
    assert rollout["action_valid"][1, 0, :, 0].all()
    assert rollout["action_valid"][1, 0, :, 1].all()
    assert (rollout["actions"][0, 0] == 0).all()


def test_imagination_availability_fallback_uses_only_best_predicted_action() -> None:
    cfg = _config(
        n_slots=2,
        burn_in=0,
        sequence_length=4,
        imagination_horizon=2,
        max_imagination_starts=1,
    )
    learner = MambaLearner(cfg, ["a", "b"])
    learner.world_model.continuation_model = _ConstantHead(10.0).to(learner.device)
    learner.world_model.availability_model = _ConstantVectorHead((-3.0, -1.0)).to(
        learner.device
    )
    rollout = learner._imagination_rollout(
        _batch(cfg, [_frame(step, slots=(0, 1)) for step in range(4)]),
    )

    assert rollout is not None
    assert rollout["action_valid"][1, 0, :, 0].all()
    assert not rollout["action_valid"][1, 0, :, 1].any()
    assert (rollout["actions"][1, 0] == 0).all()


def test_world_model_updates_before_actor_and_target_hard_syncs_after_warmup() -> None:
    cfg = _config(
        min_model_replay_frames=5,
        min_agent_replay_frames=9,
        sequence_length=4,
        burn_in=1,
    )
    learner = MambaLearner(cfg, ["a", "b", "c"])
    learner.replay.add_episode([_frame(step) for step in range(6)])
    actor_before = deepcopy(learner.actor.state_dict())

    warmup = learner.update()

    assert warmup["model_ready"] == 1.0
    assert warmup["agent_ready"] == 0.0
    assert warmup["agent_skipped"] == 1.0
    assert warmup["critic_skipped"] == 0.0
    assert warmup["critic_update_count"] == 1.0
    for name, value in learner.actor.state_dict().items():
        assert torch.equal(value, actor_before[name])
    assert "model_loss" in warmup

    learner.replay.add_episode([{**_frame(step + 10), "episode": 2} for step in range(6)])
    trained = learner.update()

    assert trained["agent_ready"] == 1.0
    assert trained["agent_skipped"] == 0.0
    assert trained["imagined_starts"] > 0.0
    assert trained["policy_kl"] >= 0.0
    assert 0.0 <= trained["clip_fraction"] <= 1.0
    assert 1.0 <= trained["ppo_epochs_completed"] <= cfg.ppo_epochs
    for target, current in zip(
        learner.target_critic.parameters(), learner.critic.parameters(), strict=True,
    ):
        assert torch.equal(target, current)


def test_critic_warms_for_multiple_updates_before_actor_changes() -> None:
    cfg = _config(
        min_model_replay_frames=5,
        min_agent_replay_frames=5,
        critic_warmup_updates=2,
        target_tau=0.05,
    )
    learner = MambaLearner(cfg, ["a", "b", "c"])
    learner.replay.add_episode([_frame(step) for step in range(6)])
    actor_before = deepcopy(learner.actor.state_dict())

    first = learner.update()
    second = learner.update()

    assert first["agent_ready"] == 0.0
    assert second["agent_ready"] == 0.0
    assert second["critic_update_count"] == 2.0
    assert second["critic_warmup_complete"] == 1.0
    assert second["target_hard_sync"] == 1.0
    for name, value in learner.actor.state_dict().items():
        assert torch.equal(value, actor_before[name])
    for target, current in zip(
        learner.target_critic.parameters(), learner.critic.parameters(), strict=True,
    ):
        assert torch.equal(target, current)

    third = learner.update()
    assert third["agent_ready"] == 1.0
    assert third["actor_skipped"] == 0.0


def test_nonfinite_world_model_gradient_is_rejected_before_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config()
    learner = MambaLearner(cfg, ["a", "b", "c"])
    before = deepcopy(learner.world_model.state_dict())
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: torch.tensor(float("inf")),
    )

    with pytest.raises(FloatingPointError, match="world-model gradient"):
        learner._train_world_model(_batch(cfg, [_frame(step) for step in range(4)]))

    for name, value in learner.world_model.state_dict().items():
        assert torch.equal(value, before[name])


def test_nonfinite_critic_gradient_cannot_reach_target_critic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(n_slots=2, action_dim=2, critic_warmup_updates=1)
    learner = MambaLearner(cfg, ["a", "b"])
    horizon, batch_size, n_agents = 1, 4, 2
    features = torch.randn(horizon, batch_size, n_agents, cfg.feature_size)
    valid = torch.ones(
        horizon, batch_size, n_agents, cfg.action_dim, dtype=torch.bool,
    )
    with torch.no_grad():
        distribution = learner.actor.distribution(features, valid)
        actions = distribution.sample()
    active = torch.ones(horizon, batch_size, n_agents, dtype=torch.bool)
    rollout = {
        "features": features,
        "actions": actions,
        "old_log_probs": distribution.log_prob(actions),
        "old_logits": distribution.logits,
        "returns": torch.ones(horizon, batch_size, n_agents),
        "active": active,
        "presence": active,
        "initial_active": active,
        "survival": active.float(),
        "action_valid": valid,
        "imagined_reward": torch.ones(horizon, batch_size, n_agents),
        "imagined_starts": torch.tensor(float(batch_size)),
    }
    learner._imagination_rollout = MethodType(lambda self, batch: rollout, learner)
    critic_before = deepcopy(learner.critic.state_dict())
    target_before = deepcopy(learner.target_critic.state_dict())
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: torch.tensor(float("inf")),
    )

    with pytest.raises(FloatingPointError, match="critic gradient"):
        learner._train_agent(object(), update_actor=False)

    assert learner.critic_update_count == 0
    for name, value in learner.critic.state_dict().items():
        assert torch.equal(value, critic_before[name])
    for name, value in learner.target_critic.state_dict().items():
        assert torch.equal(value, target_before[name])


def test_rejected_ppo_epoch_restores_actor_and_adam_state() -> None:
    cfg = _config(
        n_slots=2,
        action_dim=2,
        ppo_epochs=1,
        critic_epochs=1,
        actor_lr=0.5,
        target_policy_kl=1.0e-12,
        normalize_advantages=False,
        advantage_clip=0.0,
    )
    learner = MambaLearner(cfg, ["a", "b"])
    horizon, batch_size, n_agents = 1, 16, 2
    features = torch.randn(
        horizon, batch_size, n_agents, cfg.feature_size,
    )
    valid = torch.ones(horizon, batch_size, n_agents, cfg.action_dim, dtype=torch.bool)
    with torch.no_grad():
        old_distribution = learner.actor.distribution(features, valid)
        actions = torch.zeros(horizon, batch_size, n_agents, dtype=torch.long)
        old_log_probs = old_distribution.log_prob(actions)
        old_logits = old_distribution.logits
    active = torch.ones(horizon, batch_size, n_agents, dtype=torch.bool)
    rollout = {
        "features": features,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "old_logits": old_logits,
        "returns": torch.ones(horizon, batch_size, n_agents),
        "active": active,
        "presence": active,
        "initial_active": active,
        "survival": active.float(),
        "action_valid": valid,
        "imagined_reward": torch.ones(horizon, batch_size, n_agents),
        "imagined_starts": torch.tensor(float(batch_size)),
    }
    learner._imagination_rollout = MethodType(lambda self, batch: rollout, learner)
    actor_before = deepcopy(learner.actor.state_dict())
    optimizer_before = deepcopy(learner.actor_optimizer.state_dict())

    metrics = learner._train_agent(object(), update_actor=True)

    assert metrics["ppo_epochs_completed"] == 0.0
    assert metrics["ppo_epochs_rejected"] == 1.0
    assert metrics["policy_kl_rejected"] > cfg.target_policy_kl
    for name, value in learner.actor.state_dict().items():
        assert torch.equal(value, actor_before[name])
    assert learner.actor_optimizer.state_dict() == optimizer_before


def test_actor_warmup_uses_cumulative_frames_not_bounded_replay_occupancy() -> None:
    cfg = _config(
        replay_capacity_episodes=2,
        min_model_replay_frames=5,
        min_agent_replay_frames=15,
        sequence_length=4,
        burn_in=1,
    )
    learner = MambaLearner(cfg, ["a", "b", "c"])
    for episode in range(3):
        learner.replay.add_episode([
            {**_frame(step + episode * 10), "episode": episode + 1}
            for step in range(6)
        ])

    assert learner.replay.n_frames == 12
    learner.collected_frames = 18

    trained = learner.update()

    assert trained["replay_frames"] == 12.0
    assert trained["collected_frames"] == 18.0
    assert trained["agent_ready"] == 1.0
    assert trained["agent_skipped"] == 0.0
    assert trained["imagined_starts"] > 0.0


def test_checkpoint_rejects_semantically_different_attention_heads() -> None:
    learner = MambaLearner(_config(transformer_heads=2), ["a", "b", "c"])
    incompatible = MambaLearner(_config(transformer_heads=4), ["a", "b", "c"])

    with pytest.raises(ValueError, match="transformer_heads"):
        incompatible.load_state_dict(learner.state_dict())
