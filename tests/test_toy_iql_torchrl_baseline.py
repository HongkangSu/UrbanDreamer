import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import baseline.run_toy_cross_bottleneck_iql_torchrl as iql_module  # noqa: E402
from baseline.online_actor_critic import OnlineDecisionSample  # noqa: E402
from baseline.online_actor_critic import _append_vehicle_context_features  # noqa: E402
from baseline.run_toy_cross_bottleneck_iql_torchrl import (  # noqa: E402
    IqlTransitionReplayBuffer,
    TorchRLIqlQNetwork,
    ToyIqlTorchRLConfig,
    ToyIqlTorchRLTrainer,
    _materialize_iql_replay_transitions,
    build_config,
    build_plan,
    build_torchrl_iql_batch,
    build_torchrl_iql_loss,
    parse_args,
)
from dreamer4.urban_routing.vehicle_imagination import build_vehicle_obs_vectorized  # noqa: E402


def _sample(
    *,
    episode: int,
    decision_idx: int,
    veh_id: str,
    obs: np.ndarray,
    action: int,
    reward: float,
) -> OnlineDecisionSample:
    return OnlineDecisionSample(
        episode=episode,
        decision_idx=decision_idx,
        sim_step=30 * decision_idx,
        veh_id=veh_id,
        obs=obs.astype(np.float32),
        action=action,
        old_log_prob=-math.log(4.0),
        reward=reward,
        valid_mask=np.ones((4,), dtype=bool),
        route_costs=np.ones((4,), dtype=np.float32),
        elapsed_time=0.0,
    )


def _q_network(obs_dim: int = 10) -> TorchRLIqlQNetwork:
    cfg = ToyIqlTorchRLConfig(
        device="cpu",
        n_routes=4,
        q_network_depth=1,
        q_network_num_cells=16,
        eps_init=0.2,
        eps_end=0.05,
    )
    return TorchRLIqlQNetwork(
        obs_dim=obs_dim,
        n_routes=4,
        machine_ids=["0", "2", "4"],
        cfg=cfg,
        device=torch.device("cpu"),
    )


def test_torchrl_iql_q_network_uses_fixed_vehicle_slots_and_epsilon_logits():
    q_network = _q_network(obs_dim=7)
    obs = torch.randn(2, 7)

    q_values = q_network.forward_action_values_for_active_ids(obs, ["2", "0"])
    logits = q_network.forward_logits_for_active_ids(obs, ["2", "0"])

    assert q_values.shape == (2, 4)
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
    assert q_network.slot_for_vehicle("0") == 0
    assert q_network.slot_for_vehicle("2") == 1
    assert q_network.slot_for_vehicle("4") == 2


def test_iql_forward_logits_distributes_epsilon_only_over_valid_actions():
    q_network = _q_network(obs_dim=5)
    q_network.train()
    q_network.set_epsilon(1.0)
    obs = torch.zeros((2, 5), dtype=torch.float32)
    action_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, True],
        ],
        dtype=torch.bool,
    )

    logits = q_network.forward_logits_for_active_ids(obs, ["0", "2"], action_mask=action_mask)
    probs = torch.softmax(logits, dim=-1)

    torch.testing.assert_close(probs[0], torch.tensor([0.5, 0.0, 0.5, 0.0]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(probs[1], torch.tensor([0.0, 0.5, 0.0, 0.5]), atol=1e-6, rtol=1e-6)


def test_iql_forward_logits_masks_invalid_routes_from_obs_route_cost_block():
    q_network = _q_network(obs_dim=5)
    q_network.eval()
    edge_feats = np.ones((4, 2), dtype=np.float32)
    obs_base = build_vehicle_obs_vectorized(
        edge_feats_dyn=edge_feats,
        path_batch=[[[0], [1], [], [3]]],
        include_global=True,
        route_stats="mean",
        edge_cost=np.ones((4,), dtype=np.float32),
        include_route_costs=True,
        include_fleet_pressure=True,
        fleet_route_counts=np.zeros((4,), dtype=np.float32),
        fleet_edge_pressure=np.zeros((4,), dtype=np.float32),
    )
    obs = torch.from_numpy(
        _append_vehicle_context_features(
            obs_base,
            step_idx=0,
            episode_steps=60,
            n_active_av=1,
            n_total_av=3,
        )
    )
    q_network = _q_network(obs_dim=int(obs.shape[1]))

    logits = q_network.forward_logits_for_active_ids(obs, ["0"])

    assert logits.shape == (1, 4)
    assert torch.softmax(logits, dim=-1)[0, 2].item() < 1e-6
    assert torch.isfinite(logits[0, [0, 1, 3]]).all()


def test_iql_eval_can_reuse_final_training_epsilon_greedy_policy():
    args = parse_args(
        [
            "--eval-only",
            "--eval-policy-route-selection",
            "training",
            "--eps-end",
            "0.05",
            "--device",
            "cpu",
        ]
    )

    cfg = build_config(args)

    assert cfg.real_policy_route_selection == "sample"
    assert cfg.eps_end == 0.05
    assert cfg.disable_human_learning_after_mutation is True


def test_build_torchrl_iql_batch_links_next_decisions_per_vehicle():
    torch.manual_seed(0)
    q_network = _q_network(obs_dim=6)
    samples = [
        _sample(episode=1, decision_idx=0, veh_id="0", obs=np.ones(6), action=0, reward=1.0),
        _sample(episode=1, decision_idx=1, veh_id="0", obs=np.ones(6) * 2.0, action=1, reward=2.0),
        _sample(episode=1, decision_idx=0, veh_id="4", obs=np.ones(6) * 3.0, action=2, reward=-1.0),
    ]

    td = build_torchrl_iql_batch(
        samples=samples,
        q_network=q_network,
        device=torch.device("cpu"),
    )

    assert td.get(("agents", "observation")).shape == (3, 3, 6)
    assert td.get(("next", "agents", "observation")).shape == (3, 3, 6)
    assert td.get(("agents", "action")).shape == (3, 3)
    assert int(td.get(("agents", "active_mask")).sum().item()) == 3
    assert bool(td.get(("agents", "active_mask"))[0, 0].item())
    assert bool(td.get(("agents", "active_mask"))[2, 2].item())
    assert not bool(td.get(("next", "agents", "done"))[0, 0, 0].item())
    assert bool(td.get(("next", "agents", "done"))[1, 0, 0].item())
    assert torch.allclose(td.get(("next", "agents", "observation"))[0, 0], torch.ones(6) * 2.0)


def test_torchrl_dqn_update_runs_on_masked_active_samples():
    torch.manual_seed(1)
    q_network = _q_network(obs_dim=5)
    cfg = ToyIqlTorchRLConfig(
        device="cpu",
        n_routes=4,
        lr=1e-3,
        max_grad_norm=3.0,
        gamma=0.99,
        num_epochs=1,
        minibatch_size=2,
        min_samples_per_update=1,
        q_network_depth=1,
        q_network_num_cells=16,
    )
    loss_module = build_torchrl_iql_loss(q_network=q_network, cfg=cfg)
    trainer = object.__new__(ToyIqlTorchRLTrainer)
    trainer.cfg = cfg
    trainer.actor = q_network
    trainer.loss_module = loss_module
    trainer.target_net_updater = torch.nn.Identity()
    trainer.target_net_updater.step = lambda: None
    trainer.optimizer = torch.optim.Adam(loss_module.parameters(), lr=float(cfg.lr))
    trainer.device = torch.device("cpu")
    samples = [
        _sample(episode=1, decision_idx=0, veh_id="0", obs=np.ones(5), action=0, reward=1.0),
        _sample(episode=1, decision_idx=1, veh_id="0", obs=np.ones(5) * 1.5, action=1, reward=0.5),
        _sample(episode=1, decision_idx=0, veh_id="2", obs=np.ones(5) * 2.0, action=2, reward=-0.25),
    ]

    metrics = ToyIqlTorchRLTrainer.update_from_replay(trainer, [samples])

    assert metrics["n_samples"] == 3.0
    assert metrics["new_samples"] == 3.0
    assert metrics["frames_per_batch"] == 3.0
    assert metrics["updates_per_epoch"] == 1.0
    assert metrics["replay_size"] == 3.0
    assert metrics["sampled_transitions"] == 2.0
    assert metrics["pending_episodes"] == 0.0
    assert metrics["update_epochs"] == 1.0
    assert metrics["skipped"] == 0.0
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["chosen_q_mean"])

    next_metrics = ToyIqlTorchRLTrainer.update_from_replay(
        trainer,
        [[_sample(episode=2, decision_idx=0, veh_id="4", obs=np.ones(5) * 3.0, action=3, reward=0.75)]],
    )

    assert next_metrics["new_samples"] == 1.0
    assert next_metrics["replay_size"] == 4.0
    assert next_metrics["replay_episodes"] == 2.0
    assert next_metrics["sampled_transitions"] == 2.0


def test_iql_update_uses_original_budget_and_backprops_sampler_tail(monkeypatch):
    torch.manual_seed(2)
    np.random.seed(2)
    q_network = _q_network(obs_dim=5)
    cfg = ToyIqlTorchRLConfig(
        device="cpu",
        n_routes=4,
        lr=1e-3,
        max_grad_norm=3.0,
        gamma=0.99,
        agent_frames_per_batch=2,
        num_epochs=2,
        minibatch_size=2,
        memory_size=5,
        min_samples_per_update=1,
        q_network_depth=1,
        q_network_num_cells=16,
    )
    loss_module = build_torchrl_iql_loss(q_network=q_network, cfg=cfg)
    trainer = object.__new__(ToyIqlTorchRLTrainer)
    trainer.cfg = cfg
    trainer.actor = q_network
    trainer.loss_module = loss_module
    trainer.target_net_updater = torch.nn.Identity()
    trainer.target_net_updater.step = lambda: None
    trainer.optimizer = torch.optim.Adam(loss_module.parameters(), lr=float(cfg.lr))
    trainer.device = torch.device("cpu")
    samples = [
        _sample(episode=1, decision_idx=0, veh_id="0", obs=np.ones(5), action=0, reward=1.0),
        _sample(episode=1, decision_idx=1, veh_id="0", obs=np.ones(5) * 1.5, action=1, reward=0.5),
        _sample(episode=1, decision_idx=0, veh_id="2", obs=np.ones(5) * 2.0, action=2, reward=-0.25),
        _sample(episode=1, decision_idx=1, veh_id="2", obs=np.ones(5) * 2.5, action=3, reward=0.25),
        _sample(episode=1, decision_idx=0, veh_id="4", obs=np.ones(5) * 3.0, action=0, reward=-0.5),
    ]

    scheduled_batches = []
    original_build_minibatch = iql_module._build_iql_minibatch_td

    def record_build_minibatch(**kwargs):
        scheduled_batches.append(tuple(int(idx) for idx in kwargs["indices"]))
        return original_build_minibatch(**kwargs)

    optimizer_steps = 0
    original_optimizer_step = trainer.optimizer.step

    def record_optimizer_step(*args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_optimizer_step(*args, **kwargs)

    monkeypatch.setattr(iql_module, "_build_iql_minibatch_td", record_build_minibatch)
    monkeypatch.setattr(trainer.optimizer, "step", record_optimizer_step)

    metrics = ToyIqlTorchRLTrainer.update_from_replay(trainer, [samples])

    assert metrics["new_samples"] == 5.0
    assert metrics["replay_size"] == 5.0
    assert metrics["sampled_transitions"] == 10.0
    assert metrics["frames_per_batch"] == 6.0
    assert metrics["updates_per_epoch"] == 3.0
    assert optimizer_steps == 6
    assert len(scheduled_batches) == 6
    for epoch in range(2):
        epoch_batches = scheduled_batches[epoch * 3 : (epoch + 1) * 3]
        assert [len(batch) for batch in epoch_batches] == [2, 2, 1]
        assert sorted(idx for batch in epoch_batches for idx in batch) == list(range(5))


def test_iql_original_budget_does_not_force_an_update_below_one_minibatch():
    q_network = _q_network(obs_dim=5)
    cfg = ToyIqlTorchRLConfig(
        device="cpu",
        n_routes=4,
        agent_frames_per_batch=1,
        num_epochs=1,
        minibatch_size=4,
        memory_size=5,
        min_samples_per_update=1,
        q_network_depth=1,
        q_network_num_cells=16,
    )
    loss_module = build_torchrl_iql_loss(q_network=q_network, cfg=cfg)
    trainer = object.__new__(ToyIqlTorchRLTrainer)
    trainer.cfg = cfg
    trainer.actor = q_network
    trainer.loss_module = loss_module
    trainer.target_net_updater = torch.nn.Identity()
    trainer.target_net_updater.step = lambda: None
    trainer.optimizer = torch.optim.Adam(loss_module.parameters(), lr=float(cfg.lr))
    trainer.device = torch.device("cpu")

    metrics = ToyIqlTorchRLTrainer.update_from_replay(
        trainer,
        [[_sample(episode=1, decision_idx=0, veh_id="0", obs=np.ones(5), action=0, reward=1.0)]],
    )

    assert metrics["frames_per_batch"] == 3.0
    assert metrics["updates_per_epoch"] == 0.0
    assert metrics["new_samples"] == 1.0
    assert metrics["replay_size"] == 1.0
    assert metrics["sampled_transitions"] == 0.0
    assert metrics["n_samples"] == 1.0
    assert metrics["skipped"] == 1.0


def test_iql_pending_metrics_keep_fresh_replay_and_pending_counts_distinct():
    metrics = ToyIqlTorchRLTrainer._pending_update_metrics(
        n_samples=7,
        pending_episodes=2,
        replay_size=11,
        replay_episodes=3,
        epsilon=0.25,
    )

    assert metrics["n_samples"] == 7.0
    assert metrics["new_samples"] == 7.0
    assert metrics["sampled_transitions"] == 0.0
    assert metrics["replay_size"] == 11.0
    assert metrics["replay_episodes"] == 3.0
    assert metrics["pending_episodes"] == 2.0


def test_iql_rejects_non_positive_memory_size():
    with pytest.raises(ValueError, match="memory_size must be positive"):
        IqlTransitionReplayBuffer(memory_size=0, minibatch_size=2)


def test_iql_transition_replay_buffer_persists_and_evicts_as_original_ring():
    replay = IqlTransitionReplayBuffer(memory_size=5, minibatch_size=2)
    first = [
        _sample(episode=1, decision_idx=idx, veh_id="0", obs=np.ones(5), action=0, reward=float(idx))
        for idx in range(3)
    ]
    second = [
        _sample(episode=2, decision_idx=idx, veh_id="0", obs=np.ones(5), action=0, reward=float(10 + idx))
        for idx in range(4)
    ]

    first_transitions = _materialize_iql_replay_transitions(first, n_routes=4)
    expected_successor_obs = np.asarray(first[1].obs).copy()
    first[1].obs[:] = 99.0
    np.testing.assert_array_equal(first_transitions[0].next_obs, expected_successor_obs)
    assert first_transitions[0].done is False

    replay.extend(first_transitions)
    assert len(replay) == 3
    assert {(sample.episode, sample.decision_idx) for sample in replay.snapshot()} == {
        (1, 0),
        (1, 1),
        (1, 2),
    }

    replay.extend(_materialize_iql_replay_transitions(second, n_routes=4))
    assert len(replay) == 5
    assert replay.memory_size == 5
    assert replay.sampler_name == "SamplerWithoutReplacement"
    assert {(sample.episode, sample.decision_idx) for sample in replay.snapshot()} == {
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
    }


def test_iql_torchrl_dry_run_plan_matches_no_cfm_toy_interface():
    args = parse_args(
        [
            "--dry-run",
            "--train-episodes",
            "3",
            "--checkpoint-episodes",
            "1,3",
            "--device",
            "cpu",
            "--av-execution-mode",
            "first_decision_only",
        ]
    )

    cfg = build_config(args)
    plan = build_plan(cfg)

    assert cfg.train_episodes == 3
    assert cfg.checkpoint_episodes == (1, 3)
    assert plan["baseline"] == "toy_cross_bottleneck_iql_torchrl"
    assert plan["setting_match"]["av_execution_mode"] == "first_decision_only"
    assert plan["setting_match"]["reward_mode"] == "capped_step_cost"
    assert plan["iql_torchrl"]["share_parameters_policy"] is False
    assert plan["iql_torchrl"]["centralized_q_network"] is False
    assert plan["iql_torchrl"]["memory_size"] == 20_000
    assert plan["iql_torchrl"]["replay_sampler"] == "SamplerWithoutReplacement"


def test_iql_torchrl_default_checkpoint_interval_is_every_100_episodes():
    args = parse_args(["--dry-run", "--train-episodes", "400", "--device", "cpu"])

    cfg = build_config(args)
    plan = build_plan(cfg)

    assert cfg.checkpoint_interval == 100
    assert cfg.checkpoint_episodes == (100, 200, 300, 400)
    assert plan["iql_torchrl"]["checkpoint_interval"] == 100
    assert plan["iql_torchrl"]["checkpoint_episodes"] == [100, 200, 300, 400]
