import math
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablation.online_no_cfm_actor_critic import OnlineDecisionSample  # noqa: E402
from ablation.online_no_cfm_actor_critic import _append_vehicle_context_features  # noqa: E402
from dreamer4.urban_routing.vehicle_imagination import build_vehicle_obs_vectorized  # noqa: E402
from baseline.run_toy_cross_bottleneck_mappo_torchrl import (  # noqa: E402
    TorchRLMappoActorCritic,
    ToyMappoTorchRLConfig,
    ToyMappoTorchRLTrainer,
    build_config,
    build_plan,
    build_torchrl_mappo_batch,
    build_torchrl_mappo_loss,
    parse_args,
)


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


def _actor_critic(obs_dim: int = 10) -> TorchRLMappoActorCritic:
    cfg = ToyMappoTorchRLConfig(
        device="cpu",
        n_routes=4,
        policy_network_depth=1,
        policy_network_num_cells=16,
        critic_network_depth=1,
        critic_network_num_cells=16,
    )
    return TorchRLMappoActorCritic(
        obs_dim=obs_dim,
        n_routes=4,
        machine_ids=["0", "2", "4"],
        cfg=cfg,
        device=torch.device("cpu"),
    )


def test_torchrl_mappo_actor_critic_uses_fixed_vehicle_slots():
    actor_critic = _actor_critic(obs_dim=7)
    obs = torch.randn(2, 7)

    logits = actor_critic.forward_logits_for_active_ids(obs, ["2", "0"])
    values = actor_critic.critic_net(torch.randn(2, 3, 7))

    assert logits.shape == (2, 4)
    assert values.shape == (2, 3, 1)
    assert actor_critic.slot_for_vehicle("0") == 0
    assert actor_critic.slot_for_vehicle("2") == 1
    assert actor_critic.slot_for_vehicle("4") == 2


def test_mappo_torchrl_actor_respects_action_mask_when_sampling():
    torch.manual_seed(2)
    actor_critic = _actor_critic(obs_dim=5)
    td = torch.zeros((32, 3, 5), dtype=torch.float32)
    action_mask = torch.zeros((32, 3, 4), dtype=torch.bool)
    action_mask[..., 1] = True

    out = actor_critic.policy(
        TensorDict({
            ("agents", "observation"): td,
            ("agents", "action_mask"): action_mask,
        }, batch_size=[32])
    )

    assert torch.equal(out[("agents", "action")], torch.ones((32, 3), dtype=torch.long))


def test_mappo_forward_logits_masks_invalid_routes_from_obs_route_cost_block():
    edge_feats = np.ones((4, 2), dtype=np.float32)
    obs_base = build_vehicle_obs_vectorized(
        edge_feats_dyn=edge_feats,
        path_batch=[[[0], [], [2], [3]]],
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
    actor_critic = _actor_critic(obs_dim=int(obs.shape[1]))

    logits = actor_critic.forward_logits_for_active_ids(obs, ["0"])

    assert logits.shape == (1, 4)
    assert logits[0, 1].item() < -1.0e8
    assert torch.isfinite(logits[0, [0, 2, 3]]).all()


def test_build_torchrl_mappo_batch_preserves_active_slots_and_advantages():
    torch.manual_seed(0)
    actor_critic = _actor_critic(obs_dim=6)
    samples = [
        _sample(episode=1, decision_idx=0, veh_id="0", obs=np.ones(6), action=0, reward=1.0),
        _sample(episode=1, decision_idx=1, veh_id="0", obs=np.ones(6) * 2.0, action=1, reward=2.0),
        _sample(episode=1, decision_idx=0, veh_id="4", obs=np.ones(6) * 3.0, action=2, reward=-1.0),
    ]

    td = build_torchrl_mappo_batch(
        samples=samples,
        actor_critic=actor_critic,
        gamma=0.99,
        lmbda=0.9,
        normalize_advantage=True,
        device=torch.device("cpu"),
    )

    assert td.get(("agents", "observation")).shape == (3, 3, 6)
    assert td.get(("agents", "action")).shape == (3, 3)
    assert int(td.get(("agents", "active_mask")).sum().item()) == 3
    assert bool(td.get(("agents", "active_mask"))[0, 0].item())
    assert bool(td.get(("agents", "active_mask"))[2, 2].item())
    assert torch.isfinite(td.get(("agents", "advantage"))).all()
    assert torch.isfinite(td.get(("agents", "value_target"))).all()


def test_torchrl_clip_ppo_update_runs_on_masked_active_samples():
    torch.manual_seed(1)
    actor_critic = _actor_critic(obs_dim=5)
    cfg = ToyMappoTorchRLConfig(
        device="cpu",
        n_routes=4,
        lr=1e-3,
        max_grad_norm=3.0,
        clip_epsilon=0.2,
        gamma=0.99,
        lmbda=0.9,
        entropy_eps=1e-4,
        normalize_advantage=False,
        num_epochs=1,
        minibatch_size=2,
        min_samples_per_update=1,
    )
    loss_module = build_torchrl_mappo_loss(actor_critic=actor_critic, cfg=cfg)
    trainer = object.__new__(ToyMappoTorchRLTrainer)
    trainer.cfg = cfg
    trainer.actor = actor_critic
    trainer.loss_module = loss_module
    trainer.optimizer = torch.optim.Adam(loss_module.parameters(), lr=float(cfg.lr))
    trainer.device = torch.device("cpu")
    samples = [
        _sample(episode=1, decision_idx=0, veh_id="0", obs=np.ones(5), action=0, reward=1.0),
        _sample(episode=1, decision_idx=1, veh_id="0", obs=np.ones(5) * 1.5, action=1, reward=0.5),
        _sample(episode=1, decision_idx=0, veh_id="2", obs=np.ones(5) * 2.0, action=2, reward=-0.25),
    ]

    metrics = ToyMappoTorchRLTrainer.update_from_replay(trainer, [samples])

    assert metrics["n_samples"] == 3.0
    assert metrics["update_epochs"] == 1.0
    assert metrics["skipped"] == 0.0
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["loss_objective"])
    assert math.isfinite(metrics["loss_critic"])


def test_mappo_torchrl_dry_run_plan_matches_no_cfm_toy_interface():
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
    assert plan["baseline"] == "toy_cross_bottleneck_mappo_torchrl"
    assert plan["setting_match"]["av_execution_mode"] == "first_decision_only"
    assert plan["setting_match"]["reward_mode"] == "capped_step_cost"
    assert plan["mappo_torchrl"]["share_parameters_policy"] is False
    assert plan["mappo_torchrl"]["share_parameters_critic"] is True
    assert plan["mappo_torchrl"]["centralized_critic"] is True


def test_mappo_torchrl_default_checkpoint_interval_is_every_100_episodes():
    args = parse_args(["--dry-run", "--train-episodes", "400", "--device", "cpu"])

    cfg = build_config(args)
    plan = build_plan(cfg)

    assert cfg.checkpoint_interval == 100
    assert cfg.checkpoint_episodes == (100, 200, 300, 400)
    assert plan["mappo_torchrl"]["checkpoint_interval"] == 100
    assert plan["mappo_torchrl"]["checkpoint_episodes"] == [100, 200, 300, 400]
