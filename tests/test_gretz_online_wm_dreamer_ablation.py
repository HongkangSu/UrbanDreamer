import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablation.online_wm_dreamer_actor_critic import (  # noqa: E402
    OnlineWmDreamerConfig,
    OnlineWorldModelUpdater,
    _causal_pair,
    _advance_imagined_route_progress,
    _current_to_dest_candidate_paths,
    _discounted_returns_by_vehicle_ids,
    build_eval_command,
    build_gretz_online_wm_dreamer_plan,
    capped_step_cost_reward_matrix,
)


def test_online_wm_plan_marks_cfm_and_online_wm_training(tmp_path):
    cfg = OnlineWmDreamerConfig(
        output_dir=str(tmp_path),
        train_episodes=100,
        wm_replay_buffer_episodes=8,
        wm_update_epochs=1,
        wm_update_every_episodes=5,
        wm_batch_size=16,
        imag_horizon=8,
    ).resolve()

    plan = build_gretz_online_wm_dreamer_plan(cfg)

    assert plan["ablation"] == "gretz_online_wm_dreamer_actor_critic"
    assert plan["world_model_usage"]["cfm_transition_in_training"] is True
    assert plan["world_model_usage"]["world_model_online_update"] is True
    assert plan["world_model_usage"]["direct_real_a2c_update"] is False
    assert plan["world_model_usage"]["updated_modules"] == ["flow_projector", "cfm_dynamics"]
    assert plan["world_model_usage"]["wm_grad_enabled"] is True
    assert plan["training"]["wm_replay_buffer_episodes"] == 8
    assert plan["training"]["wm_update_epochs"] == 1
    assert plan["training"]["wm_update_every_episodes"] == 5
    assert plan["training"]["wm_replay_require_completion"] is True
    assert plan["training"]["wm_replay_completion_threshold"] == 1.0
    assert plan["training"]["wm_batch_size"] == 16
    assert plan["training"]["wm_hdv_policy_target_mode"] == "episode_mean"
    assert plan["training"]["wm_hdv_policy_action_mode"] == "window_mean"
    assert plan["training"]["wm_transition_hdv_context_mode"] == "episode_response"
    assert plan["training"]["imag_horizon"] == 8
    assert plan["evaluation"]["uses_updated_world_model_checkpoints"] is True


def test_online_wm_updater_unfreezes_world_model_parameters():
    class DummyWm:
        def __init__(self):
            self.flow_projector = torch.nn.Linear(2, 2)
            self.dynamics = torch.nn.Linear(2, 2)
            for param in self.flow_projector.parameters():
                param.requires_grad_(False)
            for param in self.dynamics.parameters():
                param.requires_grad_(False)

    wm = DummyWm()
    OnlineWorldModelUpdater(wm, OnlineWmDreamerConfig(wm_lr=1e-5), torch.device("cpu"))

    assert all(param.requires_grad for param in wm.flow_projector.parameters())
    assert all(param.requires_grad for param in wm.dynamics.parameters())


def test_online_wm_causal_pair_aligns_current_action_to_next_target():
    current = torch.tensor([[1.0, 2.0, 3.0]])

    pair = _causal_pair(current)

    torch.testing.assert_close(pair[:, 0], torch.zeros_like(current))
    torch.testing.assert_close(pair[:, 1], current)


def test_online_wm_trainer_filters_incomplete_wm_replay_without_init():
    trainer = object.__new__(__import__(
        "ablation.online_wm_dreamer_actor_critic",
        fromlist=["OnlineWmDreamerTrainer"],
    ).OnlineWmDreamerTrainer)
    trainer.cfg = OnlineWmDreamerConfig(
        wm_replay_require_completion=True,
        wm_replay_completion_threshold=1.0,
    )

    assert trainer._episode_allowed_for_wm_replay({"completion_rate": 1.0}) is True
    assert trainer._episode_allowed_for_wm_replay({"completion_rate": 0.998428}) is False

    trainer.cfg = OnlineWmDreamerConfig(wm_replay_require_completion=False)
    assert trainer._episode_allowed_for_wm_replay({"completion_rate": 0.0}) is True


def test_online_wm_trainer_respects_wm_update_interval_without_init():
    trainer = object.__new__(__import__(
        "ablation.online_wm_dreamer_actor_critic",
        fromlist=["OnlineWmDreamerTrainer"],
    ).OnlineWmDreamerTrainer)
    trainer.cfg = OnlineWmDreamerConfig(wm_update_every_episodes=5)

    assert trainer._wm_update_due(1) is False
    assert trainer._wm_update_due(4) is False
    assert trainer._wm_update_due(5) is True
    assert trainer._wm_update_due(10) is True


def test_online_wm_reward_is_capped_negative_step_cost():
    route_costs = torch.tensor([[100.0, 10.0, 40.0], [20.0, 40.0, 90.0]])
    route_valid = torch.tensor([[True, True, False], [False, True, True]])

    rewards = capped_step_cost_reward_matrix(
        route_costs,
        route_valid,
        step_interval=30.0,
        reward_temperature=60.0,
    )

    expected = torch.tensor(
        [
            [-30.0 / 60.0, -10.0 / 60.0, 0.0],
            [0.0, -30.0 / 60.0, -30.0 / 60.0],
        ]
    )
    torch.testing.assert_close(rewards, expected)


def test_imagined_route_progress_reward_stops_after_arrival():
    edge_tt = torch.tensor([10.0, 8.0, 20.0]).numpy()

    first = _advance_imagined_route_progress(
        path=[0, 1, 2],
        edge_tt=edge_tt,
        current_edge=0,
        current_edge_remaining=10.0,
        step_interval=15.0,
    )
    second = _advance_imagined_route_progress(
        path=first.remaining_path,
        edge_tt=edge_tt,
        current_edge=first.current_edge,
        current_edge_remaining=first.current_edge_remaining,
        step_interval=15.0,
    )
    third = _advance_imagined_route_progress(
        path=second.remaining_path,
        edge_tt=edge_tt,
        current_edge=second.current_edge,
        current_edge_remaining=second.current_edge_remaining,
        step_interval=15.0,
    )

    assert first.elapsed_cost == 15.0
    assert first.current_edge == 1
    assert first.current_edge_remaining == 3.0
    assert first.arrived is False
    assert second.elapsed_cost == 15.0
    assert second.current_edge == 2
    assert second.current_edge_remaining == 8.0
    assert second.arrived is False
    assert third.elapsed_cost == 8.0
    assert third.arrived is True


def test_imagined_returns_accumulate_per_vehicle_until_arrival():
    rewards = torch.tensor([-0.5, -0.5, -0.2, -0.5])
    vehicle_ids = torch.tensor([0, 1, 0, 1])

    returns = _discounted_returns_by_vehicle_ids(rewards, vehicle_ids, gamma=1.0)

    expected = torch.tensor([-0.7, -1.0, -0.2, -0.5])
    torch.testing.assert_close(returns, expected)


def test_current_to_dest_candidate_paths_regenerate_from_imagined_edge():
    adj = [[1, 2], [3], [3], []]
    edge_tt = torch.tensor([1.0, 10.0, 2.0, 1.0]).numpy()

    paths = _current_to_dest_candidate_paths(
        adj_list=adj,
        edge_tt=edge_tt,
        current_edge=0,
        dest_edge=3,
        n_routes=2,
        n_edges=4,
        penalty_factor=3.0,
        fallback_paths=[[0, 1, 3]],
    )

    assert paths[0][0] == 0
    assert paths[0][-1] == 3
    assert [0, 2, 3] in paths


def test_online_wm_eval_command_uses_checkpoint_world_model_paths(tmp_path):
    cfg = OnlineWmDreamerConfig(
        output_dir=str(tmp_path),
        eval_python="python",
        projector_ckpt=str(tmp_path / "original_projector.pt"),
        dynamics_ckpt=str(tmp_path / "original_dynamics.pt"),
    ).resolve()
    updated_projector = tmp_path / "wm_projector_ep00100.pt"
    updated_dynamics = tmp_path / "wm_dynamics_ep00100.pt"
    ckpt = tmp_path / "checkpoint_ep00100.pt"
    torch.save(
        {
            "model_family": "dreamer_ctde",
            "cfg": {"training_family": "dreamer_ctde", "agent_level": "vehicle"},
            "actor": {},
            "wm_projector_ckpt": str(updated_projector),
            "wm_dynamics_ckpt": str(updated_dynamics),
        },
        ckpt,
    )

    cmd = build_eval_command(cfg=cfg, checkpoint_path=ckpt, results_dir=tmp_path / "eval")

    assert "--projector_ckpt" in cmd
    assert cmd[cmd.index("--projector_ckpt") + 1] == str(updated_projector)
    assert "--dynamics_ckpt" in cmd
    assert cmd[cmd.index("--dynamics_ckpt") + 1] == str(updated_dynamics)
    assert str(cfg.projector_ckpt) not in {
        cmd[cmd.index("--projector_ckpt") + 1],
        cmd[cmd.index("--dynamics_ckpt") + 1],
    }


def test_online_wm_runner_dry_run_allows_missing(tmp_path):
    from ablation.run_gretz_online_wm_dreamer_ablation import build_config, parse_args

    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--train-episodes",
            "25",
            "--checkpoint-episodes",
            "25",
            "--wm-replay-buffer-episodes",
            "4",
            "--wm-update-every-episodes",
            "5",
            "--sim-buffer-after-last-depart",
            "7200",
            "--imag-horizon",
            "8",
            "--dry-run",
            "--allow-missing",
        ]
    )
    cfg = build_config(args)

    assert cfg.train_episodes == 25
    assert cfg.checkpoint_episodes == (25,)
    assert cfg.wm_replay_buffer_episodes == 4
    assert cfg.wm_update_every_episodes == 5
    assert cfg.wm_replay_require_completion is True
    assert cfg.wm_replay_completion_threshold == 1.0
    assert cfg.sim_buffer_after_last_depart == 7200
    assert cfg.imag_horizon == 8
    assert cfg.reward_mode == "capped_step_cost"
