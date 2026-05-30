import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dreamer4", "urban_routing"))
from departure_schedule import compute_avg_route_steps  # noqa: E402
from train_marl import (  # noqa: E402
    ActionHorizonProxyReward,
    OfflineInitDataset,
    TrainConfig,
    CentralizedLatentValueCritic,
    _dreamer_action_route_probs,
    _dreamer_step_reward,
    _combined_grpo_policy_loss,
    _expected_cost_policy_loss,
    _expected_route_regret_from_paths,
    _ensure_active_vehicle_current_edges,
    _grpo_route_cost_policy_loss,
    _marginal_av_pressure_cost_matrix,
    _marginal_pressure_reference_probs,
    _od_suffix_path_batch,
    _planned_av_pressure_cost_matrix,
    _planned_av_pressure_edge_weights,
    _route_share_penalty_cost_matrix,
    _route_cost_policy_aux_loss,
    _restore_dreamer_extra_state,
    _rollout_baseline_relative_advantages,
    _rollout_group_relative_advantages,
    _sequential_planned_pressure_greedy_action_probs,
    _sequential_marginal_av_pressure_cost_matrix,
    _standardize_with_detached_stats,
    build_vehicle_fleet_pressure_obs_features,
    aggregate_vehicle_actions_to_edge_distribution,
    build_vehicle_obs_vectorized,
    build_vehicle_route_cost_obs_features,
    choose_vehicle_next_edge,
    capture_rng_state,
    count_fresh_pending_trajs,
    dreamer_update,
    edge_tt_from_latent_state,
    load_dreamer_checkpoint,
    normalize_stats_history_defaults,
    maybe_resume_dreamer_checkpoint,
    resolve_wm_candidate_edge_cost_source,
    restore_rng_state,
    save_dreamer_checkpoint,
    uses_explicit_dynamic_candidate_edge_cost_source,
    vehicle_mode_action_supported,
    vehicle_mode_obs_dim,
    vehicle_route_feature_dim,
)
from dreamer4.urban_routing.dreamer_actor_critic import VehicleRouteActor  # noqa: E402
from dreamer4.urban_routing.dreamer_rollout import aggregate_route_probs_to_action_edge  # noqa: E402
from dreamer4.urban_routing.wm_reranker import rerank_topk_candidates  # noqa: E402


def test_offline_init_dataset_can_sample_step_aligned_frames(tmp_path):
    flows = torch.arange(6, dtype=torch.float32).view(6, 1, 1).expand(6, 2, 3).clone()
    torch.save({"flows": flows}, tmp_path / "flows.pt")
    torch.save({"episode": torch.tensor([0, 0, 0, 1, 1, 1])}, tmp_path / "meta.pt")

    dataset = OfflineInitDataset(
        data_dir=str(tmp_path),
        meta_file="meta.pt",
        flow_glob="flows.pt",
        K=4,
    )

    rng = np.random.default_rng(0)
    seen = {float(dataset.sample_at_step(rng, 1)[0, 0].item()) for _ in range(20)}
    assert seen == {1.0, 4.0}

    rng = np.random.default_rng(1)
    clamped = {float(dataset.sample_at_step(rng, 99)[0, 0].item()) for _ in range(20)}
    assert clamped == {2.0, 5.0}


def test_vehicle_mode_action_supported_accepts_raw_edge_action():
    assert vehicle_mode_action_supported(
        n_edges=853,
        action_dim=67,
        raw_action_dim=853,
    ) is True


def test_compute_avg_route_steps_accepts_route_masks_tensor_like_input():
    route_masks = np.zeros((2, 2, 5), dtype=np.float32)
    route_masks[0, 0, [0, 1]] = 0.5
    route_masks[0, 1, [1, 2, 3]] = 1.0 / 3.0
    route_masks[1, 0, [2, 3]] = 0.5
    route_masks[1, 1, [3, 4]] = 0.5

    steps = compute_avg_route_steps(route_masks, avg_edge_time=8.0, step_interval=30.0)
    assert isinstance(steps, int)
    assert steps >= 1


def test_vehicle_mode_obs_dim_includes_three_context_features():
    assert vehicle_mode_obs_dim(n_routes=4, d_obs=16, include_global=True) == 83
    assert vehicle_mode_obs_dim(n_routes=4, d_obs=16, include_global=False) == 67
    assert vehicle_mode_obs_dim(
        n_routes=4,
        d_obs=16,
        include_global=True,
        include_route_costs=True,
    ) == 95
    assert vehicle_mode_obs_dim(
        n_routes=4,
        d_obs=16,
        include_global=True,
        include_route_costs=True,
        include_fleet_pressure=True,
    ) == 103


def test_vehicle_fleet_pressure_obs_features_use_selected_prefix_context():
    features = build_vehicle_fleet_pressure_obs_features(
        path_batch=[[[0, 1], [2]], [[0], [3]]],
        n_edges=4,
        n_routes=2,
        selected_route_counts=np.asarray([2.0, 1.0], dtype=np.float32),
        selected_edge_pressure=np.asarray([3.0, 1.0, 0.0, 2.0], dtype=np.float32),
    )

    np.testing.assert_allclose(features[:, :2], np.asarray([[2.0 / 3.0, 1.0 / 3.0]] * 2), atol=1e-6)
    np.testing.assert_allclose(features[0, 2:], np.asarray([4.0 / 3.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(features[1, 2:], np.asarray([1.0, 2.0 / 3.0]), atol=1e-6)


def test_wm_candidate_edge_cost_source_auto_uses_freeflow_candidate_paths():
    cfg = TrainConfig(wm_edge_cost_source="route_head")
    assert resolve_wm_candidate_edge_cost_source(cfg) == "freeflow"
    assert uses_explicit_dynamic_candidate_edge_cost_source(cfg) is False


def test_wm_candidate_edge_cost_source_can_decouple_candidate_generation():
    cfg = TrainConfig(
        wm_edge_cost_source="route_head",
        wm_candidate_edge_cost_source="decoded_speed",
    )
    assert resolve_wm_candidate_edge_cost_source(cfg) == "decoded_speed"
    assert uses_explicit_dynamic_candidate_edge_cost_source(cfg) is True


def test_vehicle_action_aggregation_uses_planned_horizon_flow_not_full_route_distribution():
    path_batch = [
        [[0, 1, 2], [0, 3, 4]],
        [[5, 6, 2], [5, 7, 8]],
    ]
    probs_np = np.asarray(
        [
            [1.0, 0.0],
            [0.25, 0.75],
        ],
        dtype=np.float32,
    )
    edge_tt = np.ones((9,), dtype=np.float32) * 10.0

    action_np = aggregate_vehicle_actions_to_edge_distribution(
        probs_np,
        path_batch,
        n_edges=9,
        n_routes=2,
        current_edges=np.asarray([0, 5], dtype=np.int64),
        edge_travel_times=edge_tt,
        horizon_seconds=15.0,
    )

    expected = np.zeros((9,), dtype=np.float32)
    expected[1] = 0.5
    expected[6] = 0.125
    expected[7] = 0.375
    np.testing.assert_allclose(action_np, expected, atol=1e-6)

    probs_t = torch.tensor(probs_np, requires_grad=True)
    action_t = aggregate_route_probs_to_action_edge(
        probs_t,
        path_batch,
        n_edges=9,
        n_routes=2,
        current_edges=torch.tensor([0, 5], dtype=torch.long),
        edge_travel_times=torch.ones(9) * 10.0,
        horizon_seconds=15.0,
    )
    torch.testing.assert_close(action_t, torch.from_numpy(expected))
    loss = action_t[1] + 2.0 * action_t[7]
    loss.backward()
    assert probs_t.grad is not None
    assert probs_t.grad[0, 0] > 0
    assert probs_t.grad[1, 1] > 0


def test_dreamer_action_route_probs_expected_is_softmax():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]], requires_grad=True)

    probs = _dreamer_action_route_probs(logits, mode="expected", temperature=1.0)

    assert torch.allclose(probs, torch.softmax(logits, dim=-1))
    assert probs.requires_grad


def test_dreamer_action_route_probs_straight_through_argmax_forward_is_one_hot_with_grad():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]], requires_grad=True)

    probs = _dreamer_action_route_probs(logits, mode="straight_through_argmax", temperature=1.0)

    assert torch.equal(probs.detach(), torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert probs.requires_grad
    (probs[:, 0] * torch.tensor([1.0, 2.0])).sum().backward()
    assert logits.grad is not None


def test_route_cost_aux_hard_target_preserves_original_route_slots():
    logits = torch.tensor([[4.0, 0.0, 1.0, -1.0]], requires_grad=True)
    route_costs = torch.tensor([[10.0, 3.0, 1.0, 5.0]])
    route_valid = torch.tensor([[True, True, True, True]])

    loss, stats = _route_cost_policy_aux_loss(
        logits=logits,
        route_costs=route_costs,
        route_valid=route_valid,
        temperature=30.0,
        hard_target=True,
    )

    assert loss is not None
    assert stats["route_cost_aux_acc"] == 0.0
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([2]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None


def test_route_cost_aux_nonzero_margin_penalizes_route0_when_nonzero_slot_is_best():
    logits = torch.tensor([[2.0, 0.0, 1.0, -1.0]], requires_grad=True)
    route_costs = torch.tensor([[10.0, 3.0, 1.0, 5.0]])
    route_valid = torch.tensor([[True, True, True, True]])

    base, _ = _route_cost_policy_aux_loss(
        logits=logits,
        route_costs=route_costs,
        route_valid=route_valid,
        temperature=30.0,
        hard_target=True,
        nonzero_margin=0.0,
    )
    margin_loss, stats = _route_cost_policy_aux_loss(
        logits=logits,
        route_costs=route_costs,
        route_valid=route_valid,
        temperature=30.0,
        hard_target=True,
        nonzero_margin=0.5,
    )

    assert base is not None
    assert margin_loss is not None
    assert stats["route_cost_aux_nonzero_best_frac"] == 1.0
    torch.testing.assert_close(margin_loss, base + torch.tensor(1.5))


def test_expected_cost_policy_loss_is_not_distillation_and_preserves_slots():
    logits = torch.tensor([[2.0, 0.0, 1.0, -1.0]], requires_grad=True)
    route_costs = torch.tensor([[10.0, 3.0, 1.0, 5.0]])
    route_valid = torch.tensor([[True, True, True, True]])

    loss, stats = _expected_cost_policy_loss(
        logits=logits,
        route_costs=route_costs,
        route_valid=route_valid,
        temperature=1.0,
        detach_cost=True,
    )

    assert loss is not None
    probs = torch.softmax(logits, dim=-1)
    expected = (probs * torch.tensor([[9.0, 2.0, 0.0, 4.0]])).sum(dim=-1).mean()
    torch.testing.assert_close(loss, expected)
    assert stats["expected_cost_best_prob"] == probs[0, 2].item()
    assert stats["expected_cost_best_route0_frac"] == 0.0
    assert stats["expected_cost_nonzero_best_frac"] == 1.0
    assert stats["expected_cost_margin"] == 2.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 2] < 0.0


def test_expected_cost_policy_loss_can_balance_min_cost_slots_without_ce_targets():
    logits = torch.zeros((3, 2), requires_grad=True)
    route_costs = torch.tensor(
        [
            [0.0, 10.0],
            [10.0, 0.0],
            [10.0, 0.0],
        ]
    )
    route_valid = torch.ones_like(route_costs, dtype=torch.bool)

    unbalanced, _ = _expected_cost_policy_loss(
        logits=logits,
        route_costs=route_costs,
        route_valid=route_valid,
        temperature=1.0,
        detach_cost=True,
        balance_best_classes=False,
    )
    balanced, _ = _expected_cost_policy_loss(
        logits=logits,
        route_costs=route_costs,
        route_valid=route_valid,
        temperature=1.0,
        detach_cost=True,
        balance_best_classes=True,
    )

    assert unbalanced is not None
    assert balanced is not None
    torch.testing.assert_close(unbalanced, balanced)
    balanced.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] < 0.0
    assert logits.grad[1, 1] < 0.0


def test_expected_cost_route0_margin_pushes_cost_better_nonzero_over_route0():
    logits = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
    route_costs = torch.tensor([[10.0, 1.0, 20.0]])
    route_valid = torch.tensor([[True, True, True]])

    loss, stats = _expected_cost_policy_loss(
        logits=logits,
        route_costs=route_costs,
        route_valid=route_valid,
        temperature=10.0,
        detach_cost=True,
        route0_margin_weight=1.0,
        route0_margin=0.5,
        route0_margin_min_gap=5.0,
    )

    assert loss is not None
    assert stats["expected_cost_route0_margin_loss"] > 0.0
    assert stats["expected_cost_route0_margin_frac"] == 1.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] > 0.0
    assert logits.grad[0, 1] < 0.0


def test_grpo_route_cost_policy_loss_pushes_low_cost_nonzero_slot():
    logits = torch.tensor([[2.0, 0.0, 1.0, -1.0]], requires_grad=True)
    route_costs = torch.tensor([[10.0, 3.0, 1.0, 5.0]])
    route_valid = torch.tensor([[True, True, True, True]])

    loss, stats = _grpo_route_cost_policy_loss(
        logits=logits,
        route_costs=route_costs,
        route_valid=route_valid,
        temperature=1.0,
        normalize_advantages=False,
        detach_cost=True,
    )

    assert loss is not None
    assert stats["grpo_route_cost_best_route0_frac"] == 0.0
    assert stats["grpo_route_cost_nonzero_best_frac"] == 1.0
    assert stats["grpo_route_cost_margin"] == 2.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 2] < 0.0
    assert logits.grad[0, 0] > 0.0


def test_grpo_route_cost_policy_loss_masks_invalid_routes():
    logits = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    route_costs = torch.tensor([[1.0, 100.0, 3.0]])
    route_valid = torch.tensor([[True, False, True]])

    loss, stats = _grpo_route_cost_policy_loss(
        logits=logits,
        route_costs=route_costs,
        route_valid=route_valid,
        temperature=1.0,
        normalize_advantages=False,
        detach_cost=True,
    )

    assert loss is not None
    assert stats["grpo_route_cost_best_route0_frac"] == 1.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] < 0.0
    assert logits.grad[0, 1] == 0.0
    assert logits.grad[0, 2] > 0.0


def test_grpo_route_cost_pairwise_term_ranks_lower_cost_logit_higher():
    logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
    route_costs = torch.tensor([[1.0, 5.0, 9.0]])
    route_valid = torch.tensor([[True, True, True]])

    loss, stats = _grpo_route_cost_policy_loss(
        logits=logits,
        route_costs=route_costs,
        route_valid=route_valid,
        temperature=1.0,
        normalize_advantages=False,
        detach_cost=True,
        pairwise_weight=1.0,
        pairwise_margin=0.0,
    )

    assert loss is not None
    assert stats["grpo_route_cost_pairwise_loss"] > 0.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] < 0.0
    assert logits.grad[0, 1] > 0.0


def test_combined_grpo_policy_loss_uses_selected_advantage_with_clip_and_kl():
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    selected = torch.tensor([1])
    valid = torch.tensor([[True, True]])
    old_log_probs = torch.tensor([np.log(0.5)], dtype=torch.float32)
    route_adv = torch.tensor([1.0])

    loss, stats = _combined_grpo_policy_loss(
        logits=logits,
        selected_idx=selected,
        old_log_probs=old_log_probs,
        route_valid=valid,
        route_advantages=route_adv,
        rollout_advantage=0.5,
        route_weight=1.0,
        rollout_weight=1.0,
        temperature=1.0,
        policy_clip_ratio=0.2,
        advantage_clip=5.0,
        kl_coef=0.01,
    )

    assert loss is not None
    assert stats["grpo_combined_adv_mean"] == 1.5
    assert stats["grpo_combined_clip_frac"] == 0.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 1] < 0.0
    assert logits.grad[0, 0] > 0.0


def test_route_share_penalty_cost_matrix_penalizes_over_cap_slots():
    route_probs = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
        ]
    )
    route_valid = torch.ones_like(route_probs, dtype=torch.bool)

    costs = _route_share_penalty_cost_matrix(
        route_probs=route_probs,
        route_valid=route_valid,
        cap=0.5,
        detach_share=True,
    )

    assert costs is not None
    assert costs.shape == route_probs.shape
    assert costs[0, 0] > 0.0
    assert costs[0, 1] == 0.0
    assert costs[0, 2] == 0.0


def test_rollout_group_relative_advantages_use_same_group_baseline():
    advantages = _rollout_group_relative_advantages(
        [10.0, 12.0, 100.0, 104.0],
        [7, 7, 8, 8],
        normalize=False,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([1.0, -1.0, 2.0, -2.0]),
    )


def test_rollout_group_relative_advantages_fall_back_to_batch_when_no_groups():
    advantages = _rollout_group_relative_advantages(
        [10.0, 12.0, 100.0],
        [7, 8, 9],
        normalize=False,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([30.666666, 28.666666, -59.333332]),
    )


def test_rollout_baseline_relative_advantages_compare_to_counterfactual_cost():
    advantages = _rollout_baseline_relative_advantages(
        [9.0, 11.0, 20.0],
        [10.0, 10.0, 18.0],
        [1, 1, 2],
        normalize=False,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([1.0, -1.0, -2.0]),
    )


def test_sequential_planned_pressure_greedy_accounts_for_prior_choices():
    probs = _sequential_planned_pressure_greedy_action_probs(
        route_costs=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        route_valid=torch.tensor([[True, True], [True, True]]),
        path_batch=[[[0], [1]], [[0], [2]]],
        n_edges=3,
        n_routes=2,
        pressure_weight=3.0,
    )

    torch.testing.assert_close(
        probs,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )


def test_expected_route_regret_from_paths_normalizes_per_agent_costs():
    route_probs = torch.tensor([[0.25, 0.75], [0.50, 0.50]], requires_grad=True)

    regret = _expected_route_regret_from_paths(
        route_probs=route_probs,
        path_batch=[[[0], [1]], [[0, 1], [1]]],
        edge_cost=torch.tensor([10.0, 20.0]),
        n_routes=2,
        invalid_route_cost=torch.tensor(100.0),
        temperature=10.0,
    )

    assert regret is not None
    assert torch.isclose(regret, torch.tensor(0.625), atol=1e-6)
    regret.backward()
    assert route_probs.grad is not None
    assert route_probs.grad[0, 1] > route_probs.grad[0, 0]


def test_planned_av_pressure_cost_matrix_keeps_original_route_slots():
    route_probs = torch.tensor(
        [
            [0.75, 0.25],
            [0.10, 0.90],
        ],
        requires_grad=True,
    )
    path_batch = [
        [[0, 1, 2], [0, 3]],
        [[4, 1], [4, 3, 5]],
    ]
    route_valid = torch.tensor([[True, True], [True, True]])

    pressure = _planned_av_pressure_cost_matrix(
        route_probs=route_probs,
        path_batch=path_batch,
        route_valid=route_valid,
        n_edges=6,
        n_routes=2,
        detach_pressure=True,
    )

    assert pressure is not None
    assert pressure.shape == (2, 2)
    assert pressure[0, 0] > pressure[0, 1]
    assert pressure[1, 1] > pressure[1, 0]


def test_planned_av_pressure_cost_matrix_can_use_edge_weights():
    route_probs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    path_batch = [
        [[0], [1]],
        [[0], [2]],
    ]
    route_valid = torch.tensor([[True, True], [True, True]])

    pressure = _planned_av_pressure_cost_matrix(
        route_probs=route_probs,
        path_batch=path_batch,
        route_valid=route_valid,
        n_edges=3,
        n_routes=2,
        edge_weights=torch.tensor([0.0, 10.0, 0.0]),
        detach_pressure=True,
    )

    assert pressure is not None
    assert torch.isclose(pressure[0, 0], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(pressure[0, 1], torch.tensor(0.0), atol=1e-6)

    pressure = _planned_av_pressure_cost_matrix(
        route_probs=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        path_batch=path_batch,
        route_valid=route_valid,
        n_edges=3,
        n_routes=2,
        edge_weights=torch.tensor([0.0, 10.0, 0.0]),
        detach_pressure=True,
    )

    assert pressure is not None
    assert pressure[0, 1] > pressure[0, 0]


def test_planned_av_pressure_edge_weights_can_clip_or_threshold_delay():
    class _Registry:
        lengths = np.ones(4, dtype=np.float32)
        speed_limits = np.ones(4, dtype=np.float32)

    class _WM:
        _registry = _Registry()

    edge_tt = torch.tensor([1.0, 1.2, 2.0, 10.0])
    v_total = torch.zeros(4)
    capacities = torch.ones(4)

    clipped = _planned_av_pressure_edge_weights(
        wm=_WM(),
        edge_tt=edge_tt,
        v_total=v_total,
        capacities=capacities,
        mode="clipped_excess_tt_ratio",
    )
    thresholded = _planned_av_pressure_edge_weights(
        wm=_WM(),
        edge_tt=edge_tt,
        v_total=v_total,
        capacities=capacities,
        mode="threshold_excess_tt_ratio",
    )

    torch.testing.assert_close(clipped, torch.tensor([0.0, 0.2, 1.0, 3.0]))
    torch.testing.assert_close(thresholded, torch.tensor([0.0, 0.0, 0.75, 3.0]))


def test_marginal_av_pressure_cost_matrix_removes_self_pressure():
    route_probs = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        requires_grad=True,
    )
    path_batch = [
        [[0], [1]],
        [[0], [2]],
    ]
    route_valid = torch.tensor([[True, True], [True, True]])

    marginal = _marginal_av_pressure_cost_matrix(
        route_probs=route_probs,
        path_batch=path_batch,
        route_valid=route_valid,
        n_edges=3,
        n_routes=2,
        detach_pressure=True,
    )

    assert marginal is not None
    torch.testing.assert_close(marginal[:, 0], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(marginal[:, 1], torch.tensor([0.0, 0.0]))

    single = _marginal_av_pressure_cost_matrix(
        route_probs=torch.tensor([[1.0, 0.0]]),
        path_batch=[[[0], [1]]],
        route_valid=torch.tensor([[True, True]]),
        n_edges=2,
        n_routes=2,
        detach_pressure=True,
    )

    assert single is not None
    torch.testing.assert_close(single, torch.zeros_like(single))


def test_marginal_av_pressure_cost_matrix_can_use_argmax_reference():
    route_probs = torch.tensor(
        [
            [0.45, 0.40, 0.15],
            [0.45, 0.40, 0.15],
        ]
    )
    route_valid = torch.ones_like(route_probs, dtype=torch.bool)
    ref = _marginal_pressure_reference_probs(
        route_probs,
        route_valid,
        reference="argmax",
    )

    torch.testing.assert_close(
        ref,
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )

    marginal = _marginal_av_pressure_cost_matrix(
        route_probs=route_probs,
        pressure_route_probs=ref,
        path_batch=[[[0], [1], [2]], [[0], [3], [4]]],
        route_valid=route_valid,
        n_edges=5,
        n_routes=3,
        detach_pressure=True,
    )

    assert marginal is not None
    torch.testing.assert_close(marginal[:, 0], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(marginal[:, 1:], torch.zeros_like(marginal[:, 1:]))


def test_sequential_marginal_av_pressure_cost_matrix_matches_previous_choices():
    route_probs = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.4, 0.6],
        ]
    )
    route_valid = torch.ones_like(route_probs, dtype=torch.bool)

    costs = _sequential_marginal_av_pressure_cost_matrix(
        route_probs=route_probs,
        path_batch=[
            [[0], [1]],
            [[0], [2]],
            [[0], [2]],
        ],
        route_valid=route_valid,
        n_edges=3,
        n_routes=2,
    )

    assert costs is not None
    torch.testing.assert_close(costs[0], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(costs[1], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(costs[2], torch.tensor([1.0, 0.0]))


def test_dreamer_step_reward_can_penalize_planned_av_pressure_as_reward():
    class _Projector:
        def decode(self, z):
            return torch.zeros(3, 4, dtype=z.dtype, device=z.device)

    class _BPR:
        capacities = torch.ones(3, dtype=torch.float32)

        def __call__(self, v_total):
            return torch.ones_like(v_total)

    class _Registry:
        lengths = np.ones(3, dtype=np.float32)
        speed_limits = np.ones(3, dtype=np.float32)

    class _WM:
        flow_projector = _Projector()
        bpr = _BPR()
        _registry = _Registry()
        n_edges = 3

    reward, _t_all_min, _t_cav_min = _dreamer_step_reward(
        wm=_WM(),
        z_next=torch.zeros(3, 4),
        reward_scale=1.0,
        all_reward_weight=0.0,
        route_probs=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        path_batch=[[[0], [1]], [[0], [2]]],
        route_cost_reward_weight=1.0,
        route_cost_reward_normalize_regret=True,
        route_cost_reward_temperature=10.0,
        route_cost_reward_planned_av_pressure_weight=10.0,
    )

    assert torch.isclose(reward, torch.tensor(-1.0), atol=1e-6)


def test_dreamer_step_reward_can_weight_pressure_by_wm_excess_tt():
    class _Projector:
        def decode(self, z):
            return torch.tensor(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=z.dtype,
                device=z.device,
            )

    class _BPR:
        capacities = torch.ones(3, dtype=torch.float32)

        def __call__(self, v_total):
            return torch.ones_like(v_total)

    class _Registry:
        lengths = np.ones(3, dtype=np.float32)
        speed_limits = np.ones(3, dtype=np.float32)

    class _WM:
        flow_projector = _Projector()
        bpr = _BPR()
        _registry = _Registry()
        n_edges = 3

    reward, _t_all_min, _t_cav_min = _dreamer_step_reward(
        wm=_WM(),
        z_next=torch.zeros(3, 4),
        reward_scale=1.0,
        all_reward_weight=0.0,
        route_probs=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        path_batch=[[[0], [1]], [[0], [2]]],
        route_cost_reward_weight=1.0,
        route_cost_reward_normalize_regret=True,
        route_cost_reward_temperature=10.0,
        route_cost_reward_planned_av_pressure_weight=10.0,
        route_cost_reward_planned_av_pressure_mode="excess_tt_ratio",
    )

    assert torch.isclose(reward, torch.tensor(-0.3), atol=1e-6)


def test_detached_stats_standardization_preserves_value_gradient():
    values = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
    normalized = _standardize_with_detached_stats(values, center=True)

    assert torch.isclose(normalized.detach().mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(normalized.detach().std(unbiased=False), torch.tensor(1.0), atol=1e-6)
    normalized.mean().backward()
    assert values.grad is not None
    assert torch.all(values.grad > 0)


def test_dreamer_step_reward_penalizes_route_max_share_above_cap():
    class _Projector:
        def decode(self, z):
            return torch.zeros(2, 4, dtype=z.dtype, device=z.device)

    class _BPR:
        capacities = torch.ones(2, dtype=torch.float32)

        def __call__(self, v_total):
            return torch.ones_like(v_total)

    class _Registry:
        lengths = np.ones(2, dtype=np.float32)
        speed_limits = np.ones(2, dtype=np.float32)

    class _WM:
        flow_projector = _Projector()
        bpr = _BPR()
        _registry = _Registry()
        n_edges = 2

    reward, _t_all_min, _t_cav_min = _dreamer_step_reward(
        wm=_WM(),
        z_next=torch.zeros(2, 4),
        reward_scale=1.0,
        all_reward_weight=0.0,
        route_balance_probs=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        route_max_share_weight=10.0,
        route_max_share_cap=0.6,
    )

    assert torch.isclose(reward, torch.tensor(-1.6), atol=1e-6)


def test_vehicle_bottleneck_obs_dim_adds_route_stats_without_changing_context():
    assert vehicle_route_feature_dim(2, "bottleneck") == 9
    assert vehicle_mode_obs_dim(
        n_routes=4,
        d_obs=2,
        include_global=True,
        route_stats="bottleneck",
    ) == 48


def test_build_vehicle_obs_vectorized_bottleneck_keeps_route_cost_signal():
    edge_feats = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [4.0, 40.0],
        ],
        dtype=np.float32,
    )
    obs = build_vehicle_obs_vectorized(
        edge_feats_dyn=edge_feats,
        path_batch=[[[0, 1], [2]]],
        include_global=False,
        route_stats="bottleneck",
    )

    assert obs.shape == (1, 18)
    route0 = obs[0, :9]
    route1 = obs[0, 9:18]
    assert np.allclose(route0[:2], [1.5, 15.0])
    assert np.allclose(route0[2:4], [2.0, 20.0])
    assert np.allclose(route0[6:8], [3.0, 30.0])
    assert np.allclose(route1[:2], [4.0, 40.0])
    assert np.allclose(route1[6:8], [4.0, 40.0])


def test_vehicle_route_cost_obs_features_are_current_state_only():
    edge_cost = np.array([10.0, 20.0, 100.0], dtype=np.float32)
    path_batch = [
        [[0, 1], [2], []],
    ]

    features = build_vehicle_route_cost_obs_features(
        path_batch=path_batch,
        edge_cost=edge_cost,
        n_routes=3,
    )
    obs = build_vehicle_obs_vectorized(
        edge_feats_dyn=np.ones((3, 2), dtype=np.float32),
        path_batch=path_batch,
        include_global=False,
        route_stats="mean",
        edge_cost=edge_cost,
        include_route_costs=True,
    )

    assert features.shape == (1, 9)
    assert obs.shape == (1, 15)
    assert np.argmin(features[0, :3]) == 0
    assert np.allclose(features[0, 6:9], [1.0, 1.0, 0.0])


def test_vehicle_actor_route_cost_logit_bias_uses_relative_gap_features():
    actor = VehicleRouteActor(
        obs_dim=11,
        n_routes=2,
        hidden_dim=4,
        n_hidden_layers=1,
        route_cost_logit_bias=3.0,
        include_global=False,
        include_route_costs=True,
    )
    with torch.no_grad():
        for param in actor.parameters():
            param.zero_()

    # Layout: per-route features, route-cost [rel_gap0, rel_gap1,
    # log0, log1, valid0, valid1], then three context features.
    obs = torch.tensor([[0.0, 0.0, 1.5, 0.0, 0.2, 0.1, 1.0, 1.0, 0.0, 0.0, 0.0]])

    logits = actor.forward_logits(obs)

    torch.testing.assert_close(logits, torch.tensor([[-4.5, 0.0]]))


def test_vehicle_obs_route_cost_precedes_fleet_pressure_block():
    edge_cost = np.array([10.0, 20.0, 100.0], dtype=np.float32)
    path_batch = [[[0, 1], [2]]]
    obs = build_vehicle_obs_vectorized(
        edge_feats_dyn=np.ones((3, 1), dtype=np.float32),
        path_batch=path_batch,
        include_global=False,
        route_stats="mean",
        edge_cost=edge_cost,
        include_route_costs=True,
        include_fleet_pressure=True,
        fleet_route_counts=np.array([3.0, 1.0], dtype=np.float32),
        fleet_edge_pressure=np.array([2.0, 0.0, 1.0], dtype=np.float32),
    )

    base_dim = 2
    route_cost = obs[0, base_dim:base_dim + 6]
    fleet = obs[0, base_dim + 6:base_dim + 10]

    assert np.allclose(route_cost[4:6], [1.0, 1.0])
    assert np.allclose(fleet[:2], [0.75, 0.25])


def test_route_shared_actor_reads_cost_before_fleet_pressure():
    actor = VehicleRouteActor(
        obs_dim=15,
        n_routes=2,
        hidden_dim=4,
        n_hidden_layers=1,
        architecture="route_shared",
        include_global=False,
        include_route_costs=True,
        include_fleet_pressure=True,
    )
    with torch.no_grad():
        for param in actor.parameters():
            param.zero_()

    # Layout: route features, route-cost block, fleet-pressure block, context.
    # The invalid route mask lives in the route-cost block, not the following
    # fleet-pressure values.
    obs = torch.tensor([[
        0.0, 0.0,
        0.0, 1.0, 0.1, 0.2, 1.0, 0.0,
        0.0, 1.0, 0.5, 0.5,
        0.0, 0.0, 0.0,
    ]])

    logits = actor.forward_logits(obs)

    torch.testing.assert_close(logits, torch.tensor([[0.0, -20.0]]))


def test_reconstructed_active_vehicle_current_edge_uses_route_suffix_position():
    class _Registry:
        n_od = 1
        route_edge_indices = [[[10, 11, 12, 13]]]

    active = [
        {
            "od_idx": 0,
            "origin_edge": 10,
            "dest_edge": 13,
            "depart_step": 0,
        }
    ]

    _ensure_active_vehicle_current_edges(
        active_avs=active,
        current_step=2,
        avg_edge_time=30.0,
        step_interval=30.0,
        route_registry=_Registry(),
    )

    assert active[0]["current_edge"] == 12


def test_od_suffix_path_batch_preserves_original_route_slots():
    class _Registry:
        n_od = 1
        route_edge_indices = [
            [
                [10, 11, 12, 13],
                [10, 21, 22, 13],
                [10, 31, 32, 13],
                [10, 41, 42, 13],
            ]
        ]

    batch = _od_suffix_path_batch(
        route_registry=_Registry(),
        active_avs=[{"od_idx": 0, "dest_edge": 13}],
        current_edges=np.asarray([21], dtype=np.int64),
        done=np.asarray([False]),
        K=4,
    )

    assert batch == [[[], [21, 22, 13], [], []]]
    assert choose_vehicle_next_edge(21, batch[0][1]) == 22


def test_edge_tt_from_latent_state_can_use_decoded_speed_costs():
    class _Projector:
        def decode(self, z):
            return torch.tensor(
                [
                    [0.0, 1.0, 0.1, 0.0],
                    [0.0, 0.5, 0.2, 0.0],
                    [0.0, 2.0, 0.3, 0.0],
                ],
                dtype=torch.float32,
            )

    class _Registry:
        lengths = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        speed_limits = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        capacities = np.array([5.0, 5.0, 5.0], dtype=np.float32)

    class _WM:
        flow_projector = _Projector()
        _registry = _Registry()
        n_edges = 3

    _edge_dyn, v_e, free_flow, t_e = edge_tt_from_latent_state(
        wm=_WM(),
        z=torch.zeros(3, 4),
        bpr_alpha=0.15,
        bpr_beta=4.0,
        edge_cost_source="decoded_speed",
        decoded_speed_min_ratio=0.05,
        decoded_speed_max_ratio=1.5,
    )

    assert np.allclose(v_e, [0.5, 1.0, 1.5])
    assert np.allclose(free_flow, [1.0, 2.0, 3.0])
    assert np.allclose(t_e, [1.0, 4.0, 2.0])

    _edge_dyn, _v_e, _free_flow, scaled_t_e = edge_tt_from_latent_state(
        wm=_WM(),
        z=torch.zeros(3, 4),
        bpr_alpha=0.15,
        bpr_beta=4.0,
        edge_cost_source="decoded_speed",
        decoded_speed_min_ratio=0.05,
        decoded_speed_max_ratio=1.5,
        decoded_speed_excess_scale=0.1,
    )

    assert np.allclose(scaled_t_e, [1.0, 2.2, 2.9], atol=1e-6)


def test_edge_tt_from_latent_state_can_use_route_cost_head():
    class _Projector:
        def decode(self, z):
            return torch.tensor(
                [
                    [0.0, 1.0, 0.1, 0.0],
                    [0.0, 1.0, 0.2, 0.0],
                    [0.0, 1.0, 0.3, 0.0],
                ],
                dtype=torch.float32,
            )

        def predict_edge_route_cost(self, z, free_flow):
            return free_flow.to(dtype=torch.float32) * torch.tensor([1.0, 3.0, 0.5])

    class _Registry:
        lengths = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        speed_limits = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        capacities = np.array([5.0, 5.0, 5.0], dtype=np.float32)

    class _WM:
        flow_projector = _Projector()
        _registry = _Registry()
        n_edges = 3

    _edge_dyn, _v_e, free_flow, t_e = edge_tt_from_latent_state(
        wm=_WM(),
        z=torch.zeros(3, 4),
        bpr_alpha=0.15,
        bpr_beta=4.0,
        edge_cost_source="route_head",
    )

    assert np.allclose(free_flow, [1.0, 2.0, 3.0])
    assert np.allclose(t_e, [1.0, 6.0, 1.5], atol=1e-6)


def test_dreamer_step_reward_can_use_route_cost_head():
    class _Projector:
        def decode(self, z):
            return torch.tensor(
                [
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                ],
                dtype=z.dtype,
                device=z.device,
            )

        def predict_edge_route_cost(self, z, free_flow):
            return free_flow.to(device=z.device, dtype=z.dtype) * torch.tensor(
                [10.0, 20.0],
                dtype=z.dtype,
                device=z.device,
            )

    class _BPR:
        capacities = torch.ones(2, dtype=torch.float32)

        def __call__(self, v_total):
            return torch.ones_like(v_total)

    class _Registry:
        lengths = np.array([1.0, 1.0], dtype=np.float32)
        speed_limits = np.array([1.0, 1.0], dtype=np.float32)

    class _WM:
        flow_projector = _Projector()
        bpr = _BPR()
        _registry = _Registry()
        n_edges = 2

    reward, t_all_min, t_cav_min = _dreamer_step_reward(
        wm=_WM(),
        z_next=torch.zeros(2, 4),
        reward_scale=1.0,
        all_reward_weight=1.0,
        edge_cost_source="route_head",
    )

    assert torch.isclose(reward, torch.tensor(-15.0))
    assert np.isclose(t_all_min, 15.0 / 60.0)
    assert np.isclose(t_cav_min, 15.0 / 60.0)


def test_dreamer_step_reward_can_use_pred_t_cav_proxy_reward():
    class _Projector:
        def decode(self, z):
            return torch.zeros(2, 4, dtype=z.dtype, device=z.device)

        def predict_edge_route_cost(self, z, free_flow):
            return torch.tensor([10.0, 20.0], dtype=z.dtype, device=z.device)

    class _BPR:
        capacities = torch.ones(2, dtype=torch.float32)

        def __call__(self, v_total):
            return torch.ones_like(v_total)

    class _Registry:
        lengths = np.ones(2, dtype=np.float32)
        speed_limits = np.ones(2, dtype=np.float32)

    class _WM:
        flow_projector = _Projector()
        bpr = _BPR()
        _registry = _Registry()
        n_edges = 2

    route_probs = torch.tensor([[0.25, 0.75]], requires_grad=True)
    reward, _t_all_min, t_cav_min = _dreamer_step_reward(
        wm=_WM(),
        z_next=torch.zeros(2, 4),
        reward_scale=1.0,
        all_reward_weight=0.0,
        pred_t_cav_proxy_reward_weight=1.0,
        route_probs=route_probs,
        path_batch=[[[0], [1]]],
        edge_cost_source="route_head",
    )

    assert torch.isclose(reward, torch.tensor(-17.5))
    assert np.isclose(t_cav_min, 17.5 / 60.0)
    reward.backward()
    assert route_probs.grad is not None
    assert route_probs.grad[0, 1] < route_probs.grad[0, 0]


def test_wm_reranker_uses_requested_route_head_cost_source():
    class _Projector:
        def decode(self, z):
            return torch.tensor(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=z.dtype,
                device=z.device,
            )

        def predict_edge_route_cost(self, z, free_flow):
            return torch.tensor([100.0, 1.0], dtype=z.dtype, device=z.device)

    class _Registry:
        lengths = np.ones(2, dtype=np.float32)
        speed_limits = np.ones(2, dtype=np.float32)
        capacities = np.ones(2, dtype=np.float32)

    class _WM:
        device = torch.device("cpu")
        n_edges = 2
        d_spatial = 4
        flow_projector = _Projector()
        _registry = _Registry()

        def predict_next_state(self, buffer, **_kwargs):
            return buffer.get_latest()

        def predict_t_all_from_latent(self, _z):
            return None

    rerank = rerank_topk_candidates(
        wm=_WM(),
        z_current=torch.zeros(2, 4),
        route_probs=np.asarray([[0.51, 0.49]], dtype=np.float32),
        baseline_route_indices=np.asarray([0], dtype=np.int64),
        path_batch=[[[0], [1]]],
        horizon=1,
        top_k=2,
        discount=1.0,
        depart_edge_np=None,
        context_np=None,
        hdv_edge_np=None,
        total_av_count=1.0,
        total_hdv_count=0.0,
        t_all_weight=0.0,
        t_cav_weight=1.0,
        edge_cost_source="route_head",
    )

    assert rerank["selected_idx"] == 1
    assert rerank["selected"].route_indices.tolist() == [1]


def test_dreamer_step_reward_can_use_action_horizon_proxy():
    class _Projector:
        def decode(self, z):
            return torch.tensor(
                [
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                ],
                dtype=z.dtype,
                device=z.device,
            )

    class _BPR:
        capacities = torch.ones(2, dtype=torch.float32)

        def __call__(self, v_total):
            return torch.ones_like(v_total)

    class _Registry:
        lengths = np.array([1.0, 1.0], dtype=np.float32)
        speed_limits = np.array([1.0, 1.0], dtype=np.float32)
        static_features = np.zeros((2, 4), dtype=np.float32)

    class _WM:
        flow_projector = _Projector()
        bpr = _BPR()
        _registry = _Registry()
        n_edges = 2

    proxy = ActionHorizonProxyReward(input_dim=27, hidden_dim=4, target_mean=2.0, target_std=1.0)
    for param in proxy.parameters():
        torch.nn.init.zeros_(param)

    reward, _t_all_min, _t_cav_min = _dreamer_step_reward(
        wm=_WM(),
        z_next=torch.zeros(2, 4),
        z_current=torch.zeros(2, 4),
        action_edge=torch.tensor([0.75, 0.25]),
        reward_scale=1.0,
        all_reward_weight=1.0,
        horizon_proxy=proxy,
        horizon_proxy_reward_weight=0.5,
    )

    assert torch.isclose(reward, torch.tensor(-(1.0 + 0.5 * 120.0)))


def test_dreamer_checkpoint_roundtrips_extra_state(tmp_path):
    cfg = TrainConfig(checkpoint_dir=str(tmp_path))
    actor = VehicleRouteActor(obs_dim=5, n_routes=2, hidden_dim=8, n_hidden_layers=1)
    critic = CentralizedLatentValueCritic(d_spatial=5, hidden_dim=8)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    save_dreamer_checkpoint(
        cfg,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        episode=3,
        stats={"ep_reward": [1.0]},
        extra_state={
            "update_id": 7,
            "pending_trajs": [
                {
                    "traj_id": 1,
                    "rewards": torch.tensor([1.0]),
                    "z_states": [torch.tensor([0.0, 1.0])],
                }
            ],
        },
    )
    ckpt_path = tmp_path / "checkpoint_ep00003.pt"
    loaded_episode, loaded_stats, loaded_extra_state = load_dreamer_checkpoint(
        str(ckpt_path),
        actor,
        critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )

    assert loaded_episode == 3
    assert loaded_stats["ep_reward"] == [1.0]
    assert loaded_extra_state["update_id"] == 7
    assert loaded_extra_state["pending_trajs"][0]["traj_id"] == 1


def test_restore_dreamer_extra_state_restores_pending_trajs_regardless_of_flag():
    pending_trajs, update_id = _restore_dreamer_extra_state(
        {
            "update_id": 3,
            "pending_trajs": [
                {
                    "traj_id": 9,
                    "rewards": torch.tensor([1.0]),
                }
            ],
        },
        device=torch.device("cpu"),
    )
    assert update_id == 3
    assert pending_trajs[0]["traj_id"] == 9
    assert pending_trajs[0]["restored_from_checkpoint"] is True
    assert torch.is_tensor(pending_trajs[0]["rewards"])


def test_count_fresh_pending_trajs_ignores_restored_entries():
    pending_trajs = [
        {"traj_id": 1, "restored_from_checkpoint": True},
        {"traj_id": 2, "restored_from_checkpoint": False},
        {"traj_id": 3},
    ]
    assert count_fresh_pending_trajs(pending_trajs) == 2


def test_maybe_resume_dreamer_checkpoint_falls_back_on_value_error(monkeypatch):
    actor = VehicleRouteActor(obs_dim=5, n_routes=2, hidden_dim=8, n_hidden_layers=1)
    critic = CentralizedLatentValueCritic(d_spatial=5, hidden_dim=8)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    monkeypatch.setattr("train_marl.os.path.exists", lambda path: True)
    monkeypatch.setattr("train_marl.torch.load", lambda *args, **kwargs: {"model_family": "dreamer_ctde"})

    actor_weight_before = actor.trunk[0].weight.detach().clone()

    def _raise(*args, **kwargs):
        with torch.no_grad():
            actor.trunk[0].weight.add_(1.0)
        raise ValueError("optimizer mismatch")

    monkeypatch.setattr("train_marl.load_dreamer_checkpoint", _raise)
    resume_state = maybe_resume_dreamer_checkpoint(
        resume_ckpt="/tmp/bad.pt",
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        device=torch.device("cpu"),
    )
    assert resume_state is None
    assert torch.equal(actor.trunk[0].weight, actor_weight_before)


def test_maybe_resume_dreamer_checkpoint_returns_none_for_non_dreamer_family(monkeypatch):
    actor = VehicleRouteActor(obs_dim=5, n_routes=2, hidden_dim=8, n_hidden_layers=1)
    critic = CentralizedLatentValueCritic(d_spatial=5, hidden_dim=8)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    monkeypatch.setattr("train_marl.os.path.exists", lambda path: True)
    monkeypatch.setattr("train_marl.torch.load", lambda *args, **kwargs: {"model_family": "external_baseline"})

    resume_state = maybe_resume_dreamer_checkpoint(
        resume_ckpt="/tmp/non_dreamer.pt",
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        device=torch.device("cpu"),
    )
    assert resume_state is None


def test_normalize_stats_history_defaults_backfills_missing_keys():
    history = normalize_stats_history_defaults({"ep_reward": [1.0]})
    assert history["ep_reward"] == [1.0]
    assert history["invalid_route_rate"] == []
    assert history["traj_diagnostics"] == []
    assert history["update_diagnostics"] == []


def test_rng_state_roundtrip_restores_next_numpy_draw():
    rng = np.random.default_rng(42)
    first = int(rng.integers(0, 1000))
    state = capture_rng_state(rng=rng)
    expected_next = int(rng.integers(0, 1000))

    restored_rng = restore_rng_state(state)
    assert restored_rng is not None
    assert int(restored_rng.integers(0, 1000)) == expected_next
    assert first != expected_next


def test_rng_state_roundtrip_restores_python_and_torch_draws():
    original_py_state = random.getstate()
    original_torch_state = torch.random.get_rng_state()
    try:
        random.seed(123)
        torch.manual_seed(456)
        rng = np.random.default_rng(42)
        _ = int(rng.integers(0, 1000))
        _ = random.random()
        _ = torch.rand(1)

        state = capture_rng_state(rng=rng)
        expected_np = rng.integers(0, 1000, size=3)
        expected_py = [random.random() for _ in range(3)]
        expected_torch = torch.rand(3)

        random.seed(999)
        torch.manual_seed(999)
        _ = np.random.default_rng(999).integers(0, 1000, size=3)

        restored_rng = restore_rng_state(state)
        assert restored_rng is not None
        assert restored_rng.integers(0, 1000, size=3).tolist() == expected_np.tolist()
        assert [random.random() for _ in range(3)] == expected_py
        assert torch.equal(torch.rand(3), expected_torch)
    finally:
        random.setstate(original_py_state)
        torch.random.set_rng_state(original_torch_state)


def test_maybe_resume_dreamer_checkpoint_restores_rng_and_pending_trajs(tmp_path):
    original_py_state = random.getstate()
    original_torch_state = torch.random.get_rng_state()
    try:
        cfg = TrainConfig(checkpoint_dir=str(tmp_path))
        actor = VehicleRouteActor(obs_dim=5, n_routes=2, hidden_dim=8, n_hidden_layers=1)
        critic = CentralizedLatentValueCritic(d_spatial=5, hidden_dim=8)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

        random.seed(123)
        torch.manual_seed(456)
        rng = np.random.default_rng(42)
        _ = int(rng.integers(0, 1000))
        _ = random.random()
        _ = torch.rand(1)

        extra_state = {
            "update_id": 11,
            "pending_trajs": [
                {
                    "traj_id": 5,
                    "rewards": torch.tensor([1.0]),
                    "z_states": [torch.tensor([0.0, 1.0])],
                }
            ],
            **capture_rng_state(rng=rng),
        }
        expected_np = rng.integers(0, 1000, size=3)
        expected_py = [random.random() for _ in range(3)]
        expected_torch = torch.rand(3)

        save_dreamer_checkpoint(
            cfg,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            episode=3,
            stats={"ep_reward": [1.0]},
            extra_state=extra_state,
        )

        random.seed(999)
        torch.manual_seed(999)

        resume_state = maybe_resume_dreamer_checkpoint(
            resume_ckpt=str(tmp_path / "checkpoint_ep00003.pt"),
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            device=torch.device("cpu"),
        )

        assert resume_state is not None
        assert resume_state["start_episode"] == 3
        assert resume_state["update_id"] == 11
        assert resume_state["pending_trajs"][0]["traj_id"] == 5
        assert resume_state["pending_trajs"][0]["restored_from_checkpoint"] is True
        assert torch.is_tensor(resume_state["pending_trajs"][0]["rewards"])

        restored_rng = resume_state["restored_rng"]
        assert restored_rng is not None
        assert restored_rng.integers(0, 1000, size=3).tolist() == expected_np.tolist()
        assert [random.random() for _ in range(3)] == expected_py
        assert torch.equal(torch.rand(3), expected_torch)
        assert count_fresh_pending_trajs(resume_state["pending_trajs"]) == 0
    finally:
        random.setstate(original_py_state)
        torch.random.set_rng_state(original_torch_state)


def test_dreamer_update_reports_restored_traj_no_grad_graph():
    cfg = TrainConfig(
        dreamer_value_coef=1.0,
        dreamer_entropy_coef=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
    )
    actor = VehicleRouteActor(obs_dim=5, n_routes=2, hidden_dim=8, n_hidden_layers=1)
    critic = CentralizedLatentValueCritic(d_spatial=2, hidden_dim=8)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    stats = dreamer_update(
        cfg,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        trajectories=[
            {
                "traj_id": 7,
                "z_states": [torch.zeros(2), torch.ones(2)],
                "rewards": torch.tensor([0.0]),
                "entropies": torch.tensor([0.0]),
                "restored_from_checkpoint": True,
                "n_actor_steps": 1.0,
                "n_valid_path_agents": 1.0,
                "reward_grad_steps": 1.0,
                "entropy_grad_steps": 1.0,
                "invalid_route_rate": 0.0,
                "route_churn": 0.0,
            }
        ],
        update_id=3,
    )
    assert stats["skip_reason_counts"] == {"restored_traj_no_grad_graph": 1}
    assert stats["skipped_traj_ids"] == [7]
    assert stats["used_traj_ids"] == []
    assert stats["consumed_fresh_traj_count"] == 0
    assert stats["consumed_restored_traj_count"] == 1
    assert stats["used_fresh_traj_count"] == 0
    assert stats["skipped_restored_traj_count"] == 1
