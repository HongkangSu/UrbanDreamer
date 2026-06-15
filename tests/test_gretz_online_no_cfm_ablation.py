import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablation.online_no_cfm_actor_critic import (  # noqa: E402
    OnlineDecisionSample,
    OnlineNoCfmConfig,
    build_eval_command,
    build_gretz_ablation_plan,
    compute_a2c_loss,
    discounted_vehicle_returns,
    save_policy_checkpoint,
    samples_to_tensors,
    selected_progress_fraction_rewards,
    selected_vehicle_route_rewards,
)
from dreamer4.urban_routing.dreamer_actor_critic import (  # noqa: E402
    CentralizedVehicleValueCritic,
    VehicleRouteActor,
)
from scripts.eval_sumo_policy import load_policy  # noqa: E402


def test_gretz_ablation_plan_resolves_defaults():
    cfg = OnlineNoCfmConfig(train_episodes=100, eval_episodes=3).resolve()

    plan = build_gretz_ablation_plan(cfg)

    assert plan["network"] == "gretz_armainvilliers"
    assert plan["training"]["episodes"] == 100
    assert plan["evaluation"]["episodes_per_checkpoint"] == 3
    assert plan["world_model_usage"]["cfm_transition_in_training"] is False
    assert plan["reward_semantics"]["mode"] == "progress_fraction"
    assert set(plan["training"]["checkpoint_paths"]) == {25, 50, 100}
    assert "completion100_stronglivebad" in plan["tag"]
    assert "20260530_odcompact" in plan["paths"]["post_mutation_state_in"]
    assert "20260530_odcompact" in plan["paths"]["hdv_state_in"]
    assert cfg.incident_edge == "-986733117#1"
    assert cfg.incident_start == 500.0
    assert cfg.incident_end == 2500.0
    assert cfg.incident_speed == 0.1


def test_gretz_ablation_plan_exposes_resume_settings(tmp_path):
    ckpt = tmp_path / "checkpoint_ep00100.pt"
    ckpt.write_bytes(b"placeholder")
    cfg = OnlineNoCfmConfig(
        train_episodes=400,
        checkpoint_episodes=(200, 300, 400),
        resume_checkpoint=str(ckpt),
        resume_start_episode=100,
    ).resolve()

    plan = build_gretz_ablation_plan(cfg)

    assert plan["training"]["episodes"] == 400
    assert plan["training"]["resume_checkpoint"] == str(ckpt)
    assert plan["training"]["resume_start_episode"] == 100
    assert set(plan["training"]["checkpoint_paths"]) == {200, 300, 400}


def test_gretz_ablation_plan_exposes_replay_settings(tmp_path):
    cfg = OnlineNoCfmConfig(
        output_dir=str(tmp_path),
        replay_buffer_episodes=8,
        replay_update_epochs=2,
        replay_actor_max_age_episodes=6,
        return_horizon_steps=8,
    ).resolve()

    plan = build_gretz_ablation_plan(cfg)

    assert plan["training"]["replay_buffer_episodes"] == 8
    assert plan["training"]["replay_update_epochs"] == 2
    assert plan["training"]["replay_actor_max_age_episodes"] == 6
    assert plan["training"]["return_horizon_steps"] == 8
    assert plan["world_model_usage"]["cfm_transition_in_training"] is False
    assert plan["world_model_usage"]["wm_rollout_or_rerank_in_training"] is False


def test_gretz_ablation_plan_exposes_capped_step_cost_reward(tmp_path):
    cfg = OnlineNoCfmConfig(
        output_dir=str(tmp_path),
        reward_mode="capped_step_cost",
        reward_temperature=60.0,
        return_horizon_steps=0,
    ).resolve()

    plan = build_gretz_ablation_plan(cfg)

    assert plan["reward_semantics"]["mode"] == "capped_step_cost"
    assert plan["reward_semantics"]["temperature"] == 60.0
    assert plan["training"]["return_horizon_steps"] == 0


def test_progress_fraction_reward_matches_offline_semantics():
    rewards = selected_progress_fraction_rewards(
        route_costs=np.asarray([[100.0, 10.0], [20.0, 40.0]], dtype=np.float32),
        route_valid=np.asarray([[True, True], [False, True]]),
        selected_idx=np.asarray([1, 0]),
        step_interval=30.0,
        elapsed_time=np.asarray([20.0, 0.0], dtype=np.float32),
        progress_cap=1.0,
    )

    torch.testing.assert_close(rewards, torch.tensor([10.0 / 30.0, 0.0]))


def test_progress_fraction_reward_caps_and_uses_elapsed_denominator():
    rewards = selected_progress_fraction_rewards(
        route_costs=torch.tensor([[100.0, 5.0]]),
        route_valid=torch.tensor([[True, True]]),
        selected_idx=torch.tensor([0]),
        step_interval=30.0,
        elapsed_time=torch.tensor([20.0]),
        progress_cap=0.2,
    )

    torch.testing.assert_close(rewards, torch.tensor([0.2]))


def test_capped_step_cost_reward_is_negative_scaled_cost():
    rewards = selected_vehicle_route_rewards(
        route_costs=torch.tensor([[100.0, 10.0], [20.0, 40.0]]),
        route_valid=torch.tensor([[True, True], [False, True]]),
        selected_idx=torch.tensor([1, 0]),
        reward_mode="capped_step_cost",
        step_interval=30.0,
        reward_temperature=60.0,
    )

    torch.testing.assert_close(rewards, torch.tensor([-10.0 / 60.0, 0.0]))


def test_a2c_loss_is_finite_on_dummy_batch():
    torch.manual_seed(0)
    obs_dim = 8
    n_routes = 3
    actor = VehicleRouteActor(obs_dim, n_routes, hidden_dim=16, n_hidden_layers=1)
    critic = CentralizedVehicleValueCritic(
        obs_dim,
        hidden_dim=16,
        n_hidden_layers=1,
        architecture="mlp",
    )
    out = compute_a2c_loss(
        actor=actor,
        critic=critic,
        obs=torch.randn(5, obs_dim),
        actions=torch.tensor([0, 1, 2, 1, 0]),
        returns=torch.randn(5),
        valid_mask=torch.ones(5, n_routes, dtype=torch.bool),
        entropy_coef=0.01,
        value_coef=1.0,
    )

    assert torch.isfinite(out["loss"])
    assert int(out["n_samples"]) == 5
    assert torch.isfinite(out["adv_std"])
    assert torch.isfinite(out["combined_adv_std"])


def test_a2c_loss_filters_invalid_selected_actions():
    torch.manual_seed(1)
    obs_dim = 6
    n_routes = 3
    actor = VehicleRouteActor(obs_dim, n_routes, hidden_dim=12, n_hidden_layers=1)
    critic = CentralizedVehicleValueCritic(
        obs_dim,
        hidden_dim=12,
        n_hidden_layers=1,
        architecture="mlp",
    )

    out = compute_a2c_loss(
        actor=actor,
        critic=critic,
        obs=torch.randn(3, obs_dim),
        actions=torch.tensor([0, 1, 2]),
        returns=torch.tensor([1.0, 2.0, 3.0]),
        valid_mask=torch.tensor(
            [
                [True, False, False],
                [True, False, True],
                [False, False, True],
            ],
            dtype=torch.bool,
        ),
        entropy_coef=0.0,
        value_coef=1.0,
        normalize_advantages=True,
        advantage_clip=2.5,
        ppo_clip_ratio=0.075,
        old_log_probs=torch.zeros(3),
    )

    assert torch.isfinite(out["loss"])
    assert int(out["n_samples"]) == 2
    torch.testing.assert_close(out["selected_valid_frac"], torch.tensor(2.0 / 3.0))


def test_a2c_loss_supports_offline_style_advantage_controls():
    torch.manual_seed(2)
    obs_dim = 7
    n_routes = 4
    actor = VehicleRouteActor(obs_dim, n_routes, hidden_dim=16, n_hidden_layers=1)
    critic = CentralizedVehicleValueCritic(
        obs_dim,
        hidden_dim=16,
        n_hidden_layers=1,
        architecture="mlp",
    )

    out = compute_a2c_loss(
        actor=actor,
        critic=critic,
        obs=torch.randn(8, obs_dim),
        actions=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
        returns=torch.linspace(-5.0, 5.0, 8),
        valid_mask=torch.ones(8, n_routes, dtype=torch.bool),
        old_log_probs=torch.full((8,), -1.0),
        entropy_coef=0.002,
        value_coef=1.0,
        normalize_advantages=True,
        advantage_clip=0.5,
        ppo_clip_ratio=0.075,
        entropy_target_frac=0.65,
        entropy_floor_coef=0.02,
    )

    assert torch.isfinite(out["loss"])
    assert int(out["n_samples"]) == 8
    assert float(out["combined_adv_std"]) <= 0.5 + 1e-6
    assert float(out["entropy_floor"]) >= 0.0


def test_replay_group_index_separates_same_step_across_episodes():
    obs = np.zeros((4,), dtype=np.float32)
    mask = np.ones((2,), dtype=bool)
    samples = [
        OnlineDecisionSample(episode=1, decision_idx=0, sim_step=0, veh_id="a", obs=obs, action=0, old_log_prob=0.0, reward=1.0, valid_mask=mask, route_costs=np.ones(2), elapsed_time=0.0),
        OnlineDecisionSample(episode=1, decision_idx=0, sim_step=0, veh_id="b", obs=obs, action=1, old_log_prob=0.0, reward=1.0, valid_mask=mask, route_costs=np.ones(2), elapsed_time=0.0),
        OnlineDecisionSample(episode=2, decision_idx=0, sim_step=0, veh_id="c", obs=obs, action=0, old_log_prob=0.0, reward=1.0, valid_mask=mask, route_costs=np.ones(2), elapsed_time=0.0),
        OnlineDecisionSample(episode=2, decision_idx=1, sim_step=30, veh_id="d", obs=obs, action=1, old_log_prob=0.0, reward=1.0, valid_mask=mask, route_costs=np.ones(2), elapsed_time=30.0),
    ]

    tensors = samples_to_tensors(samples, device=torch.device("cpu"), gamma=1.0)

    assert tensors["group_index"].tolist() == [0, 0, 1, 2]


def test_discounted_vehicle_returns_can_limit_real_rollout_horizon():
    obs = np.zeros((1,), dtype=np.float32)
    mask = np.ones((1,), dtype=bool)
    samples = [
        OnlineDecisionSample(episode=1, decision_idx=i, sim_step=i * 30, veh_id="veh", obs=obs, action=0, old_log_prob=0.0, reward=float(i + 1), valid_mask=mask, route_costs=np.ones(1), elapsed_time=float(i * 30))
        for i in range(4)
    ]

    full_returns = discounted_vehicle_returns(samples, gamma=1.0, horizon_steps=0)
    horizon_returns = discounted_vehicle_returns(samples, gamma=1.0, horizon_steps=2)
    discounted_horizon_returns = discounted_vehicle_returns(samples, gamma=0.5, horizon_steps=3)

    np.testing.assert_allclose(full_returns, np.asarray([10.0, 9.0, 7.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(horizon_returns, np.asarray([3.0, 5.0, 7.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(
        discounted_horizon_returns,
        np.asarray([2.75, 4.5, 5.0, 4.0], dtype=np.float32),
    )


def test_samples_to_tensors_uses_return_horizon_steps():
    obs = np.zeros((2,), dtype=np.float32)
    mask = np.ones((2,), dtype=bool)
    samples = [
        OnlineDecisionSample(episode=1, decision_idx=i, sim_step=i * 30, veh_id="veh", obs=obs, action=0, old_log_prob=0.0, reward=float(i + 1), valid_mask=mask, route_costs=np.ones(2), elapsed_time=float(i * 30))
        for i in range(4)
    ]

    tensors = samples_to_tensors(
        samples,
        device=torch.device("cpu"),
        gamma=1.0,
        return_horizon_steps=2,
    )

    torch.testing.assert_close(tensors["returns"], torch.tensor([3.0, 5.0, 7.0, 4.0]))


def test_a2c_loss_skips_empty_batch():
    actor = VehicleRouteActor(4, 2, hidden_dim=8, n_hidden_layers=1)
    critic = CentralizedVehicleValueCritic(4, hidden_dim=8, n_hidden_layers=1, architecture="mlp")

    out = compute_a2c_loss(
        actor=actor,
        critic=critic,
        obs=torch.zeros(0, 4),
        actions=torch.zeros(0, dtype=torch.long),
        returns=torch.zeros(0),
        valid_mask=torch.zeros(0, 2, dtype=torch.bool),
    )

    assert int(out["n_samples"]) == 0
    assert torch.isfinite(out["loss"])


def test_checkpoint_writer_is_eval_loader_compatible(tmp_path):
    obs_dim = 15
    n_routes = 3
    cfg = OnlineNoCfmConfig(
        n_routes=n_routes,
        hidden_dim=16,
        n_hidden_layers=1,
        include_global=True,
        include_route_costs=False,
        include_fleet_pressure=False,
        actor_architecture="mlp",
        replay_buffer_episodes=8,
        replay_update_epochs=2,
        replay_actor_max_age_episodes=6,
        return_horizon_steps=8,
    )
    actor = VehicleRouteActor(
        obs_dim,
        n_routes,
        hidden_dim=16,
        n_hidden_layers=1,
        architecture="mlp",
        include_global=True,
        include_route_costs=False,
        include_fleet_pressure=False,
    )
    critic = CentralizedVehicleValueCritic(obs_dim, hidden_dim=16, n_hidden_layers=1, architecture="mlp")
    ckpt = save_policy_checkpoint(
        path=tmp_path / "ckpt.pt",
        actor=actor,
        critic=critic,
        cfg=cfg,
        episode=25,
        obs_dim=obs_dim,
    )

    loaded = load_policy(str(ckpt), obs_dim, n_routes, "cpu")

    assert isinstance(loaded, VehicleRouteActor)
    assert loaded.obs_dim == obs_dim
    assert loaded.n_routes == n_routes
    state = torch.load(ckpt, map_location="cpu")
    assert state["cfg"]["online_replay_buffer_episodes"] == 8
    assert state["cfg"]["online_replay_update_epochs"] == 2
    assert state["cfg"]["online_replay_actor_max_age_episodes"] == 6
    assert state["cfg"]["online_return_horizon_steps"] == 8


def test_eval_command_forces_no_rerank_and_policy_only(tmp_path):
    cfg = OnlineNoCfmConfig(output_dir=str(tmp_path), eval_python="python").resolve()

    cmd = build_eval_command(cfg=cfg, checkpoint_path=tmp_path / "ckpt.pt", results_dir=tmp_path / "eval")

    assert "--mpc_horizon" in cmd
    assert cmd[cmd.index("--mpc_horizon") + 1] == "0"
    assert "--av_route_cost_policy" in cmd
    assert cmd[cmd.index("--av_route_cost_policy") + 1] == "none"
    assert "--policy_ckpt" in cmd


def test_training_module_static_guard_has_no_forbidden_cfm_calls():
    text = (ROOT / "ablation" / "online_no_cfm_actor_critic.py").read_text(encoding="utf-8")
    forbidden = [
        "predict_next_state",
        "predict_next_state_differentiable",
        "_cfm_sample_one_step",
        "rollout_candidate_metrics",
        "rerank_topk_candidates",
    ]

    for token in forbidden:
        assert token not in text
