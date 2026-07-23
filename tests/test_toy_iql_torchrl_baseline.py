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
from baseline.run_toy_cross_bottleneck_online_ac import ToyOnlineAcCollector  # noqa: E402
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
    sim_step: int | None = None,
    elapsed_time: float = 0.0,
    arrival_remaining_time: float = 30.0,
) -> OnlineDecisionSample:
    return OnlineDecisionSample(
        episode=episode,
        decision_idx=decision_idx,
        sim_step=30 * decision_idx if sim_step is None else int(sim_step),
        veh_id=veh_id,
        obs=obs.astype(np.float32),
        action=action,
        old_log_prob=-math.log(4.0),
        reward=reward,
        valid_mask=np.ones((4,), dtype=bool),
        route_costs=np.ones((4,), dtype=np.float32),
        elapsed_time=float(elapsed_time),
        arrival_remaining_time=float(arrival_remaining_time),
    )


def _q_network(
    obs_dim: int = 10,
    *,
    epsilon_scope: str = "per_decision",
) -> TorchRLIqlQNetwork:
    cfg = ToyIqlTorchRLConfig(
        device="cpu",
        n_routes=4,
        q_network_depth=1,
        q_network_num_cells=16,
        eps_init=0.2,
        eps_end=0.05,
        epsilon_scope=epsilon_scope,
    )
    return TorchRLIqlQNetwork(
        obs_dim=obs_dim,
        n_routes=4,
        machine_ids=["0", "2", "4"],
        cfg=cfg,
        device=torch.device("cpu"),
    )


def test_iql_resume_rejects_misaligned_optimizer_state_shapes():
    current_model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Linear(4, 2),
    )
    saved_model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Linear(4, 2),
    )
    current_optimizer = torch.optim.Adam(current_model.parameters())
    saved_optimizer = torch.optim.Adam(saved_model.parameters())
    sum(parameter.sum() for parameter in saved_model.parameters()).backward()
    saved_optimizer.step()
    optimizer_state = saved_optimizer.state_dict()
    saved_params = optimizer_state["param_groups"][0]["params"]
    saved_params[0], saved_params[2] = saved_params[2], saved_params[0]

    trainer = object.__new__(ToyIqlTorchRLTrainer)
    trainer.optimizer = current_optimizer

    assert trainer._optimizer_state_shapes_match(optimizer_state) is False


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

    logits = q_network.forward_logits_for_active_ids(
        obs,
        ["0", "2"],
        action_mask=action_mask,
        epsilon_by_row=torch.ones(2),
    )
    probs = torch.softmax(logits, dim=-1)

    torch.testing.assert_close(probs[0], torch.tensor([0.5, 0.0, 0.5, 0.0]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(probs[1], torch.tensor([0.0, 0.5, 0.0, 0.5]), atol=1e-6, rtol=1e-6)


def test_iql_first_decision_epsilon_scope_resets_each_episode():
    q_network = _q_network(
        obs_dim=5,
        epsilon_scope="first_decision_per_vehicle",
    )
    q_network.train()
    q_network.set_epsilon(1.0)
    collector = object.__new__(ToyOnlineAcCollector)
    collector._policy_eval = q_network
    collector._reset_episode_epsilon_scope()
    obs = torch.zeros((2, 5), dtype=torch.float32)
    active_ids = ["0", "2"]

    first_probs = torch.softmax(
        collector._policy_logits_for_active_ids(obs, active_ids),
        dim=-1,
    )
    collector._commit_episode_epsilon_scope(active_ids)
    later_probs = torch.softmax(
        collector._policy_logits_for_active_ids(obs, active_ids),
        dim=-1,
    )
    collector._commit_episode_epsilon_scope(active_ids)

    torch.testing.assert_close(first_probs, torch.full_like(first_probs, 0.25))
    torch.testing.assert_close(
        later_probs.max(dim=-1).values,
        torch.ones(2),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.all((later_probs > 1e-6).sum(dim=-1) == 1)
    assert collector.episode_epsilon_scope_stats() == {
        "epsilon_policy_rows": 4.0,
        "epsilon_eligible_rows": 2.0,
        "effective_epsilon_mean": 0.5,
    }

    collector._reset_episode_epsilon_scope()
    reset_probs = torch.softmax(
        collector._policy_logits_for_active_ids(obs, active_ids),
        dim=-1,
    )
    torch.testing.assert_close(reset_probs, torch.full_like(reset_probs, 0.25))


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


def test_iql_smdp_transition_counts_skipped_decision_intervals():
    cfg = ToyIqlTorchRLConfig(
        device="cpu",
        n_routes=4,
        gamma=0.99,
        reward_mode="capped_step_cost",
        reward_temperature=60.0,
        step_interval=30.0,
    )
    samples = [
        _sample(
            episode=1,
            decision_idx=0,
            sim_step=0,
            veh_id="0",
            obs=np.ones(5),
            action=0,
            reward=-0.5,
        ),
        _sample(
            episode=1,
            decision_idx=3,
            sim_step=90,
            veh_id="0",
            obs=np.ones(5) * 2.0,
            action=1,
            reward=-0.5,
            arrival_remaining_time=30.0,
        ),
    ]

    transitions = _materialize_iql_replay_transitions(samples, n_routes=4, cfg=cfg)

    assert len(transitions) == 2
    assert transitions[0].done is False
    assert transitions[0].steps_to_next_obs == 3
    assert transitions[0].reward == pytest.approx(-0.5 * (1.0 + 0.99 + 0.99**2))
    assert transitions[1].done is True
    assert transitions[1].steps_to_next_obs == 1
    assert transitions[1].reward == pytest.approx(-0.5)


def test_iql_smdp_terminal_transition_includes_actual_arrival_tail():
    cfg = ToyIqlTorchRLConfig(
        device="cpu",
        n_routes=4,
        gamma=0.99,
        reward_mode="capped_step_cost",
        reward_temperature=60.0,
        step_interval=30.0,
    )
    sample = _sample(
        episode=1,
        decision_idx=59,
        sim_step=1800,
        veh_id="0",
        obs=np.ones(5),
        action=0,
        reward=-0.5,
        arrival_remaining_time=75.0,
    )

    transitions = _materialize_iql_replay_transitions([sample], n_routes=4, cfg=cfg)

    assert len(transitions) == 1
    assert transitions[0].done is True
    assert transitions[0].steps_to_next_obs == 3
    assert transitions[0].reward == pytest.approx(-1.240025)


def test_iql_smdp_drops_censored_terminal_without_arrival_target():
    cfg = ToyIqlTorchRLConfig(device="cpu", n_routes=4)
    sample = _sample(
        episode=1,
        decision_idx=59,
        veh_id="0",
        obs=np.ones(5),
        action=0,
        reward=-0.5,
        arrival_remaining_time=float("nan"),
    )

    assert _materialize_iql_replay_transitions([sample], n_routes=4, cfg=cfg) == []


def test_iql_batch_exposes_smdp_steps_to_torchrl_td0():
    q_network = _q_network(obs_dim=5)
    cfg = ToyIqlTorchRLConfig(
        device="cpu",
        n_routes=4,
        gamma=0.99,
        reward_mode="capped_step_cost",
        reward_temperature=60.0,
        step_interval=30.0,
    )
    samples = [
        _sample(
            episode=1,
            decision_idx=0,
            sim_step=0,
            veh_id="0",
            obs=np.ones(5),
            action=0,
            reward=-0.5,
        ),
        _sample(
            episode=1,
            decision_idx=2,
            sim_step=60,
            veh_id="0",
            obs=np.ones(5) * 2.0,
            action=1,
            reward=-0.5,
            arrival_remaining_time=30.0,
        ),
    ]

    td = build_torchrl_iql_batch(
        samples=samples,
        q_network=q_network,
        device=torch.device("cpu"),
        cfg=cfg,
    )

    steps = td.get("steps_to_next_obs")
    assert int(steps[0, 0, 0].item()) == 2
    assert int(steps[1, 0, 0].item()) == 1


def test_iql_torchrl_td0_discounts_bootstrap_by_smdp_duration():
    q_network = _q_network(obs_dim=5)
    with torch.no_grad():
        for parameter in q_network.q_net.parameters():
            parameter.zero_()
        final_bias = dict(q_network.q_net.named_parameters())["params.2.bias"]
        final_bias.fill_(1.0)
    cfg = ToyIqlTorchRLConfig(device="cpu", n_routes=4, gamma=0.99)
    loss_module = build_torchrl_iql_loss(q_network=q_network, cfg=cfg)
    transition = iql_module.IqlReplayTransition(
        episode=1,
        decision_idx=0,
        veh_id="0",
        obs=np.zeros(5, dtype=np.float32),
        action=0,
        action_mask=np.ones(4, dtype=bool),
        reward=0.0,
        next_obs=np.zeros(5, dtype=np.float32),
        next_action_mask=np.ones(4, dtype=bool),
        done=False,
        steps_to_next_obs=3,
    )
    td = iql_module._build_iql_minibatch_td(
        transitions=[transition],
        indices=[0],
        q_network=q_network,
        device=torch.device("cpu"),
    )

    loss_values = loss_module(td).get("loss")
    active_loss = loss_values[0, 0]

    assert float(active_loss.item()) == pytest.approx((1.0 - 0.99**3) ** 2, rel=1e-5)


def test_iql_soft_update_interpolates_target_parameters_by_configured_eps():
    q_network = _q_network(obs_dim=5)
    cfg = ToyIqlTorchRLConfig(device="cpu", n_routes=4)
    loss_module = build_torchrl_iql_loss(q_network=q_network, cfg=cfg)
    updater = iql_module.SoftUpdate(loss_module, eps=0.75)

    with torch.no_grad():
        for parameter in loss_module.value_network_params.values(True, True):
            parameter.fill_(2.0)
        for parameter in loss_module.target_value_network_params.values(True, True):
            parameter.zero_()

    updater.step()

    for parameter in loss_module.target_value_network_params.values(True, True):
        torch.testing.assert_close(parameter, torch.full_like(parameter, 0.5))
    assert iql_module._online_target_relative_l2(loss_module) == pytest.approx(0.75)


def test_iql_arrival_annotation_uses_observed_trip_time():
    sample = _sample(
        episode=1,
        decision_idx=10,
        veh_id="0",
        obs=np.ones(5),
        action=0,
        reward=-0.5,
        elapsed_time=100.0,
        arrival_remaining_time=float("nan"),
    )

    iql_module._annotate_iql_samples_with_arrival_remaining([sample], {"0": 176.0})

    assert sample.arrival_remaining_time == pytest.approx(75.0)


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
    assert metrics["fresh_sampled_transitions"] == 0.0
    assert metrics["fresh_coverage"] == 0.0
    assert metrics["replay_sampled_transitions"] == 2.0
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


def test_iql_ratio_budget_samples_exact_fresh_transition_target(monkeypatch):
    torch.manual_seed(3)
    q_network = _q_network(obs_dim=5)
    cfg = ToyIqlTorchRLConfig(
        device="cpu",
        n_routes=4,
        lr=1e-3,
        max_grad_norm=3.0,
        gamma=0.99,
        agent_frames_per_batch=99,
        replay_sample_ratio=0.6,
        num_epochs=2,
        minibatch_size=2,
        memory_size=10,
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
        _sample(
            episode=1,
            decision_idx=idx,
            veh_id=["0", "2", "4"][idx % 3],
            obs=np.ones(5) * (idx + 1),
            action=idx % 4,
            reward=float(idx),
        )
        for idx in range(5)
    ]

    scheduled_batches = []
    original_build_minibatch = iql_module._build_iql_minibatch_td

    def record_build_minibatch(**kwargs):
        scheduled_batches.append(tuple(int(idx) for idx in kwargs["indices"]))
        return original_build_minibatch(**kwargs)

    monkeypatch.setattr(iql_module, "_build_iql_minibatch_td", record_build_minibatch)

    metrics = ToyIqlTorchRLTrainer.update_from_replay(trainer, [samples])

    assert [len(batch) for batch in scheduled_batches] == [2, 1, 2, 1]
    assert metrics["new_transitions"] == 5.0
    assert metrics["frames_per_batch"] == 3.0
    assert metrics["sample_budget_per_epoch"] == 3.0
    assert metrics["target_sampled_transitions"] == 6.0
    assert metrics["sampled_transitions"] == 6.0
    assert metrics["unique_sampled_transitions"] == 6.0
    assert metrics["sampled_to_new_ratio"] == pytest.approx(1.2)
    assert metrics["replay_sample_ratio"] == pytest.approx(0.6)
    assert metrics["updates_per_epoch"] == 2.0
    assert metrics["optimizer_steps"] == 4.0
    assert metrics["sampler_resets"] == 2.0
    assert metrics["fresh_evicted_before_update"] == 0.0


def test_iql_fresh_transition_epochs_cover_every_new_transition_before_replay(monkeypatch):
    torch.manual_seed(31)
    q_network = _q_network(obs_dim=5)
    cfg = ToyIqlTorchRLConfig(
        device="cpu",
        n_routes=4,
        lr=1e-3,
        max_grad_norm=3.0,
        gamma=0.99,
        fresh_transition_epochs=2,
        replay_sample_ratio=0.4,
        num_epochs=2,
        minibatch_size=2,
        memory_size=3,
        min_samples_per_update=1,
        q_network_depth=1,
        q_network_num_cells=16,
    )
    loss_module = build_torchrl_iql_loss(q_network=q_network, cfg=cfg)
    trainer = object.__new__(ToyIqlTorchRLTrainer)
    trainer.cfg = cfg
    trainer.actor = q_network
    trainer.loss_module = loss_module

    target_updates = 0

    class _TargetUpdater:
        def step(self):
            nonlocal target_updates
            target_updates += 1

    trainer.target_net_updater = _TargetUpdater()
    trainer.optimizer = torch.optim.Adam(loss_module.parameters(), lr=float(cfg.lr))
    trainer.device = torch.device("cpu")
    samples = [
        _sample(
            episode=1,
            decision_idx=idx,
            veh_id=["0", "2", "4"][idx % 3],
            obs=np.ones(5) * (idx + 1),
            action=idx % 4,
            reward=float(idx),
        )
        for idx in range(5)
    ]

    scheduled_batches = []
    original_build_minibatch = iql_module._build_iql_minibatch_td

    def record_build_minibatch(**kwargs):
        scheduled_batches.append(
            (
                len(kwargs["transitions"]),
                tuple(int(idx) for idx in kwargs["indices"]),
            )
        )
        return original_build_minibatch(**kwargs)

    monkeypatch.setattr(iql_module, "_build_iql_minibatch_td", record_build_minibatch)

    metrics = ToyIqlTorchRLTrainer.update_from_replay(trainer, [samples])

    fresh_batches = [indices for population, indices in scheduled_batches if population == 5]
    replay_batches = [indices for population, indices in scheduled_batches if population == 3]
    assert [len(batch) for batch in fresh_batches] == [2, 2, 1, 2, 2, 1]
    assert sorted(idx for batch in fresh_batches[:3] for idx in batch) == list(range(5))
    assert sorted(idx for batch in fresh_batches[3:] for idx in batch) == list(range(5))
    assert [len(batch) for batch in replay_batches] == [2, 2]
    assert metrics["new_transitions"] == 5.0
    assert metrics["fresh_sampled_transitions"] == 10.0
    assert metrics["fresh_unique_transitions"] == 5.0
    assert metrics["fresh_coverage"] == 1.0
    assert metrics["replay_sampled_transitions"] == 4.0
    assert metrics["target_sampled_transitions"] == 14.0
    assert metrics["sampled_transitions"] == 14.0
    assert metrics["optimizer_steps"] == 8.0
    assert metrics["sampler_resets"] == 2.0
    assert metrics["combined_update_epochs"] == 2.0
    assert target_updates == 8
    assert metrics["target_update_steps"] == 8.0


def test_iql_fresh_transition_epochs_cover_evicted_fresh_transitions_without_replay():
    q_network = _q_network(obs_dim=5)
    cfg = ToyIqlTorchRLConfig(
        device="cpu",
        n_routes=4,
        fresh_transition_epochs=1,
        replay_sample_ratio=0.0,
        num_epochs=7,
        minibatch_size=4,
        memory_size=2,
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
        _sample(
            episode=1,
            decision_idx=idx,
            veh_id=["0", "2", "4"][idx % 3],
            obs=np.ones(5) * (idx + 1),
            action=idx % 4,
            reward=float(idx),
        )
        for idx in range(5)
    ]

    metrics = ToyIqlTorchRLTrainer.update_from_replay(trainer, [samples])

    assert metrics["fresh_evicted_before_update"] == 3.0
    assert metrics["fresh_sampled_transitions"] == 5.0
    assert metrics["fresh_unique_transitions"] == 5.0
    assert metrics["fresh_coverage"] == 1.0
    assert metrics["replay_sampled_transitions"] == 0.0
    assert metrics["target_sampled_transitions"] == 5.0
    assert metrics["sampled_transitions"] == 5.0
    assert metrics["optimizer_steps"] == 2.0
    assert metrics["combined_update_epochs"] == 1.0


def test_iql_ratio_budget_updates_when_target_is_smaller_than_minibatch():
    q_network = _q_network(obs_dim=5)
    cfg = ToyIqlTorchRLConfig(
        device="cpu",
        n_routes=4,
        replay_sample_ratio=1.0,
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

    assert metrics["sample_budget_per_epoch"] == 1.0
    assert metrics["sampled_transitions"] == 1.0
    assert metrics["updates_per_epoch"] == 1.0
    assert metrics["optimizer_steps"] == 1.0
    assert metrics["skipped"] == 0.0


def test_iql_ratio_budget_reports_fresh_transitions_evicted_before_update():
    q_network = _q_network(obs_dim=5)
    cfg = ToyIqlTorchRLConfig(
        device="cpu",
        n_routes=4,
        replay_sample_ratio=1.0,
        num_epochs=1,
        minibatch_size=2,
        memory_size=3,
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
        _sample(
            episode=1,
            decision_idx=idx,
            veh_id=["0", "2", "4"][idx % 3],
            obs=np.ones(5) * (idx + 1),
            action=idx % 4,
            reward=float(idx),
        )
        for idx in range(5)
    ]

    metrics = ToyIqlTorchRLTrainer.update_from_replay(trainer, [samples])

    assert metrics["fresh_retained_transitions"] == 3.0
    assert metrics["fresh_evicted_before_update"] == 2.0
    assert metrics["target_sampled_transitions"] == 5.0
    assert metrics["sampled_transitions"] == 5.0
    assert metrics["unique_sampled_transitions"] == 3.0


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
        replay_sample_ratio=0.25,
        fresh_transition_epochs=1,
    )

    assert metrics["n_samples"] == 7.0
    assert metrics["new_samples"] == 7.0
    assert metrics["sampled_transitions"] == 0.0
    assert metrics["fresh_sampled_transitions"] == 0.0
    assert metrics["fresh_unique_transitions"] == 0.0
    assert metrics["fresh_coverage"] == 0.0
    assert metrics["replay_sampled_transitions"] == 0.0
    assert metrics["fresh_transition_epochs"] == 1.0
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


def test_iql_transition_replay_buffer_reset_starts_a_fresh_without_replacement_cycle():
    torch.manual_seed(7)
    replay = IqlTransitionReplayBuffer(memory_size=5, minibatch_size=2)
    first = [
        _sample(episode=1, decision_idx=idx, veh_id="0", obs=np.ones(5), action=0, reward=float(idx))
        for idx in range(5)
    ]
    second = [
        _sample(episode=2, decision_idx=idx, veh_id="0", obs=np.ones(5), action=0, reward=float(10 + idx))
        for idx in range(5)
    ]
    replay.extend(_materialize_iql_replay_transitions(first, n_routes=4))
    replay.sample_indices(batch_size=2)
    replay.extend(_materialize_iql_replay_transitions(second, n_routes=4))

    replay.reset_sampling_cycle()
    sampled = np.concatenate(
        [
            replay.sample_indices(batch_size=2),
            replay.sample_indices(batch_size=2),
            replay.sample_indices(batch_size=2),
        ]
    )

    assert sampled.shape == (5,)
    assert sorted(int(idx) for idx in sampled) == list(range(5))


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
    assert plan["iql_torchrl"]["replay_sample_ratio"] == 0.0
    assert plan["iql_torchrl"]["fresh_transition_epochs"] == 0
    assert plan["iql_torchrl"]["update_budget_mode"] == "legacy_agent_frames"
    assert plan["iql_torchrl"]["fresh_transition_sampling"] == "disabled"
    assert plan["iql_torchrl"]["replay_sampler"] == "SamplerWithoutReplacement"
    assert plan["iql_torchrl"]["replay_sampler_reset_each_epoch"] is False
    assert plan["iql_torchrl"]["eps_init"] == pytest.approx(0.99)
    assert plan["iql_torchrl"]["exploration_fraction"] == pytest.approx(0.4)
    assert plan["iql_torchrl"]["epsilon_scope"] == "per_decision"
    assert plan["iql_torchrl"]["target_update_cadence"] == "once_per_optimizer_step"


def test_iql_torchrl_plan_exposes_ratio_budget_configuration():
    args = parse_args(
        [
            "--dry-run",
            "--device",
            "cpu",
            "--replay-sample-ratio",
            "0.25",
            "--epsilon-scope",
            "first_decision_per_vehicle",
        ]
    )

    cfg = build_config(args)
    plan = build_plan(cfg)

    assert cfg.replay_sample_ratio == pytest.approx(0.25)
    assert plan["iql_torchrl"]["replay_sample_ratio"] == pytest.approx(0.25)
    assert plan["iql_torchrl"]["update_budget_mode"] == "fresh_transition_ratio"
    assert plan["iql_torchrl"]["replay_sampler_reset_each_epoch"] is True
    assert plan["iql_torchrl"]["epsilon_scope"] == "first_decision_per_vehicle"


def test_iql_torchrl_plan_exposes_fresh_transition_coverage_configuration():
    args = parse_args(
        [
            "--dry-run",
            "--device",
            "cpu",
            "--fresh-transition-epochs",
            "1",
            "--replay-sample-ratio",
            "0.25",
        ]
    )

    cfg = build_config(args)
    plan = build_plan(cfg)

    assert cfg.fresh_transition_epochs == 1
    assert plan["iql_torchrl"]["fresh_transition_epochs"] == 1
    assert plan["iql_torchrl"]["update_budget_mode"] == "fresh_full_coverage_plus_replay"
    assert plan["iql_torchrl"]["fresh_transition_sampling"] == "random_without_replacement"
    assert plan["iql_torchrl"]["target_update_cadence"] == "once_per_optimizer_step"


def test_iql_training_drain_override_wins_over_settings(tmp_path):
    settings = tmp_path / "iql.env"
    settings.write_text("SIM_BUFFER_AFTER_LAST_DEPART=1000\n", encoding="utf-8")
    args = parse_args(
        [
            "--settings",
            str(settings),
            "--output-dir",
            str(tmp_path / "out"),
            "--train-sim-buffer-after-last-depart",
            "3000",
            "--device",
            "cpu",
        ]
    )

    cfg = build_config(args)

    assert cfg.sim_buffer_after_last_depart == 3000


def test_iql_rejects_negative_training_drain_override():
    args = parse_args(["--train-sim-buffer-after-last-depart", "-1"])

    with pytest.raises(
        ValueError,
        match="train_sim_buffer_after_last_depart must be non-negative",
    ):
        build_config(args)


def test_iql_rejects_unknown_epsilon_scope():
    with pytest.raises(ValueError, match="epsilon_scope must be"):
        ToyIqlTorchRLConfig(epsilon_scope="per_trajectory").resolve()


@pytest.mark.parametrize("invalid_ratio", [-0.1, float("nan"), float("inf")])
def test_iql_rejects_invalid_replay_sample_ratio(invalid_ratio):
    with pytest.raises(ValueError, match="replay_sample_ratio must be finite and non-negative"):
        ToyIqlTorchRLConfig(replay_sample_ratio=invalid_ratio).resolve()


def test_iql_rejects_negative_fresh_transition_epochs():
    with pytest.raises(ValueError, match="fresh_transition_epochs must be non-negative"):
        ToyIqlTorchRLConfig(fresh_transition_epochs=-1).resolve()


def test_iql_torchrl_default_checkpoint_interval_is_every_100_episodes():
    args = parse_args(["--dry-run", "--train-episodes", "400", "--device", "cpu"])

    cfg = build_config(args)
    plan = build_plan(cfg)

    assert cfg.checkpoint_interval == 100
    assert cfg.checkpoint_episodes == (100, 200, 300, 400)
    assert plan["iql_torchrl"]["checkpoint_interval"] == 100
    assert plan["iql_torchrl"]["checkpoint_episodes"] == [100, 200, 300, 400]
