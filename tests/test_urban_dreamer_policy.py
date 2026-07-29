from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_policy_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "urban_dreamer"
        / "models"
        / "policy.py"
    )
    spec = importlib.util.spec_from_file_location("urban_dreamer_policy_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


policy_module = _load_policy_module()
CentralizedLatentValueCritic = policy_module.CentralizedLatentValueCritic
CentralizedVehicleValueCritic = policy_module.CentralizedVehicleValueCritic
VehicleRouteActor = policy_module.VehicleRouteActor
capped_route_reward_matrix = policy_module.capped_route_reward_matrix
lambda_return = policy_module.lambda_return
lambda_returns_by_vehicle_ids = policy_module.lambda_returns_by_vehicle_ids


def _actor_obs(batch: int = 1) -> torch.Tensor:
    return torch.zeros((batch, 33), dtype=torch.float32)


def test_route_actor_is_fixed_to_route_shared_4x33_layout_and_masks_invalid_routes() -> None:
    actor = VehicleRouteActor()
    with torch.no_grad():
        for param in actor.parameters():
            param.zero_()

    obs = _actor_obs()
    obs[0, 18:22] = torch.tensor([1.0, 0.0, 1.0, 1.0])
    logits = actor.forward_logits(obs)

    assert actor.route_scorer[0].in_features == 12
    assert actor.route_scorer[0].out_features == 256
    assert actor.route_scorer[2].in_features == 256
    assert actor.route_scorer[2].out_features == 256
    torch.testing.assert_close(logits, torch.tensor([[0.0, -20.0, 0.0, 0.0]]))


def test_action_sampling_respects_masks_and_returns_exact_one_hot() -> None:
    torch.manual_seed(0)
    actor = VehicleRouteActor()
    with torch.no_grad():
        for param in actor.parameters():
            param.zero_()

    obs = _actor_obs(batch=512)
    obs[:, 18:22] = torch.tensor([1.0, 0.0, 1.0, 1.0])
    selection = actor.sample_action(obs)

    assert bool(selection.valid_route_mask[:, 1].eq(False).all())
    assert bool(selection.valid_route_mask[:, 0].all())
    assert bool(selection.valid_route_mask[:, 2].all())
    assert bool(selection.valid_route_mask[:, 3].all())
    assert bool(
        selection.valid_route_mask.gather(
            1, selection.route_index.unsqueeze(1)
        ).all()
    )
    torch.testing.assert_close(
        selection.route_one_hot.argmax(dim=-1), selection.route_index
    )
    torch.testing.assert_close(
        selection.route_one_hot.sum(dim=-1), torch.ones(512)
    )
    assert bool((selection.route_index == 0).any())
    assert bool((selection.route_index == 2).any())
    assert torch.isfinite(selection.log_probability).all()
    assert torch.isfinite(selection.entropy).all()


def test_deterministic_action_uses_the_highest_scoring_valid_route() -> None:
    actor = VehicleRouteActor()
    with torch.no_grad():
        for param in actor.parameters():
            param.zero_()

    obs = _actor_obs(batch=4)
    obs[:, 18:22] = torch.tensor([0.0, 1.0, 1.0, 1.0])
    selection = actor.sample_action(obs, deterministic=True)

    torch.testing.assert_close(
        selection.route_index,
        torch.ones(4, dtype=torch.long),
    )
    assert bool(
        selection.valid_route_mask.gather(
            1,
            selection.route_index.unsqueeze(1),
        ).all()
    )


def test_critics_have_fixed_input_shapes_and_roundtrip_state_dicts() -> None:
    torch.manual_seed(123)
    vehicle_critic = CentralizedVehicleValueCritic()
    latent_critic = CentralizedLatentValueCritic()

    assert vehicle_critic.encoder[0].in_features == 33
    assert vehicle_critic.encoder[0].out_features == 256
    assert latent_critic.trunk[0].in_features == 67
    assert latent_critic.trunk[0].out_features == 256

    vehicle_obs = torch.randn(5, 33)
    latent_z = torch.randn(2, 4, 64)
    latent_context = torch.randn(2, 3)

    vehicle_clone = CentralizedVehicleValueCritic()
    latent_clone = CentralizedLatentValueCritic()
    vehicle_status = vehicle_clone.load_state_dict(vehicle_critic.state_dict(), strict=True)
    latent_status = latent_clone.load_state_dict(latent_critic.state_dict(), strict=True)

    assert vehicle_status.missing_keys == []
    assert vehicle_status.unexpected_keys == []
    assert latent_status.missing_keys == []
    assert latent_status.unexpected_keys == []
    torch.testing.assert_close(vehicle_critic(vehicle_obs), vehicle_clone(vehicle_obs))
    torch.testing.assert_close(
        latent_critic(latent_z, context=latent_context),
        latent_clone(latent_z, context=latent_context),
    )


def test_capped_route_reward_matrix_matches_fixed_mainline_reward() -> None:
    route_costs = torch.tensor([[100.0, 10.0, 40.0, 5.0], [20.0, 40.0, 90.0, 15.0]])
    route_valid = torch.tensor([[True, True, False, True], [False, True, True, True]])

    rewards = capped_route_reward_matrix(
        route_costs,
        route_valid,
    )

    expected = torch.tensor(
        [
            [-1.0, -10.0 / 30.0, 0.0, -5.0 / 30.0],
            [0.0, -1.0, -1.0, -15.0 / 30.0],
        ]
    )
    torch.testing.assert_close(rewards, expected)


def test_lambda_return_bootstraps_the_last_nonterminal_step() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0])
    next_values = torch.tensor([10.0, 20.0, 30.0])
    terminal = torch.tensor([False, True, False])

    returns = lambda_return(rewards, next_values, terminal=terminal)

    torch.testing.assert_close(returns, torch.tensor([3.4, 2.0, 33.0]))


def test_lambda_returns_by_vehicle_ids_mask_invalid_next_bootstrap() -> None:
    rewards = torch.tensor([1.0, 100.0, 2.0, 3.0, 200.0])
    next_values = torch.tensor([5.0, 50.0, 6.0, 7.0, 80.0])
    next_valid = torch.tensor([True, False, True, True, False])
    vehicle_ids = torch.tensor([0, 1, 0, 0, 1])
    terminal = torch.tensor([False, True, False, True, True])

    returns = lambda_returns_by_vehicle_ids(
        rewards,
        next_values,
        next_valid,
        vehicle_ids,
        terminal=terminal,
    )

    torch.testing.assert_close(returns, torch.tensor([6.1425, 100.0, 5.15, 3.0, 200.0]))


def test_policy_module_exports_only_fixed_surface() -> None:
    assert set(policy_module.__all__) == {
        "ActionSelection",
        "CentralizedLatentValueCritic",
        "CentralizedVehicleValueCritic",
        "VehicleRouteActor",
        "capped_route_reward_matrix",
        "lambda_return",
        "lambda_returns_by_vehicle_ids",
    }
