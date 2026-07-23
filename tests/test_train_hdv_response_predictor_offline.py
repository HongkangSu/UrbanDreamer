import sys
from csv import DictWriter
from json import dumps
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_hdv_response_predictor_offline import (  # noqa: E402
    CausalEpisodePairs,
    GawronRouteRollout,
    SparseRouteEdgeProjection,
    _resolve_relocated_repo_path,
    blend_route_anchors,
    build_causal_episode_pairs,
    causal_last_observed_route_counts,
    load_route_travel_time_features,
    paired_route_metrics,
    project_route_probs_to_edges,
    project_route_response_to_edges,
    replay_gawron_route_state,
    route_candidate_acceptance,
    select_residual_scale,
    smooth_empirical_route_counts,
    train_persistence_anchored_gate,
)
from dreamer4.urban_routing.gawron import gawron_step_from_route_tt  # noqa: E402
from scripts.evaluate_hdv_static_response import (  # noqa: E402
    load_static_hdv_dataset,
)


def _write_route_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    edge_registry = tmp_path / "edge_registry.json"
    edge_registry.write_text(dumps({"edge_ids": ["edge_0", "edge_1"]}))
    route_xml = tmp_path / "route.rou.xml"
    route_xml.write_text("""
<routes>
  <route id="0_1_0" edges="edge_0"/>
  <route id="0_1_1" edges="edge_1"/>
</routes>
""".strip())
    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir()
    fields = [
        "id",
        "travel_time",
        "action",
        "initial_route_action",
        "policy_action",
        "hdv_action",
        "seen_in_episode",
        "origin",
        "destination",
        "vType",
    ]
    rows = [
        {
            "id": 1,
            "travel_time": 10,
            "action": 1,
            "initial_route_action": 0,
            "policy_action": -1,
            "hdv_action": 1,
            "seen_in_episode": 1,
            "origin": 0,
            "destination": 1,
            "vType": "Human",
        },
        {
            "id": 2,
            "travel_time": 5,
            "action": 0,
            "initial_route_action": 0,
            "policy_action": -1,
            "hdv_action": -1,
            "seen_in_episode": 0,
            "origin": 0,
            "destination": 1,
            "vType": "Human",
        },
        {
            "id": 3,
            "travel_time": 20,
            "action": 1,
            "initial_route_action": 0,
            "policy_action": 1,
            "hdv_action": -1,
            "seen_in_episode": 1,
            "origin": 0,
            "destination": 1,
            "vType": "AV",
        },
        {
            "id": 4,
            "travel_time": 7,
            "action": 0,
            "initial_route_action": 0,
            "policy_action": -1,
            "hdv_action": -1,
            "seen_in_episode": 0,
            "origin": 0,
            "destination": 1,
            "vType": "AV",
        },
    ]
    with (episodes_dir / "ep1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return edge_registry, route_xml, episodes_dir


def test_build_causal_episode_pairs_excludes_gaps_and_future_features():
    episodes = np.asarray([1, 2, 4, 5, 6], dtype=np.int64)

    pairs = build_causal_episode_pairs(
        episodes,
        train_end_episode=4,
        validation_end_episode=5,
        test_start_episode=6,
    )

    assert pairs.train.tolist() == [[0, 1]]
    assert pairs.validation.tolist() == [[2, 3]]
    assert pairs.test.tolist() == [[3, 4]]


def test_relocated_repo_path_ignores_inaccessible_archived_prefix(
    monkeypatch,
    tmp_path,
):
    relocated = ROOT / "networks" / "fontenay_en_parisis" / "edge_registry.json"
    original_exists = Path.exists

    def guarded_exists(path):
        if str(path).startswith("/root/UrbanDreamer-main/"):
            raise PermissionError("archived server root is inaccessible")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", guarded_exists)

    resolved = _resolve_relocated_repo_path(
        "/root/UrbanDreamer-main/networks/fontenay_en_parisis/edge_registry.json",
        run_dir=tmp_path,
    )

    assert resolved == relocated.resolve()


def test_static_route_loader_can_match_online_seen_vehicle_action_semantics(tmp_path):
    edge_registry, route_xml, episodes_dir = _write_route_fixture(tmp_path)

    dataset = load_static_hdv_dataset(
        run_dir=tmp_path,
        route_xml=route_xml,
        edge_registry=edge_registry,
        episodes_dir=episodes_dir,
        hdv_action_col="hdv_action",
        require_seen_in_episode=True,
        strict_route_actions=True,
    )

    np.testing.assert_allclose(dataset.hdv_route, [[[0.0, 1.0]]])
    np.testing.assert_allclose(dataset.hdv_route_counts, [[[0.0, 1.0]]])
    np.testing.assert_allclose(dataset.hdv_edge, [[0.0, 1.0]])
    np.testing.assert_allclose(dataset.av_route, [[[0.0, 1.0]]])
    np.testing.assert_allclose(dataset.av_route_counts, [[[0.0, 1.0]]])
    np.testing.assert_allclose(dataset.av_edge, [[0.0, 1.0]])


def test_route_tt_loader_uses_seen_final_route_actions_and_returns_sample_counts(
    tmp_path,
):
    edge_registry, route_xml, episodes_dir = _write_route_fixture(tmp_path)
    dataset = load_static_hdv_dataset(
        run_dir=tmp_path,
        route_xml=route_xml,
        edge_registry=edge_registry,
        episodes_dir=episodes_dir,
        hdv_action_col="hdv_action",
        require_seen_in_episode=True,
        strict_route_actions=True,
    )

    route_tt, observed, sample_counts = load_route_travel_time_features(
        dataset=dataset,
        episodes_dir=episodes_dir,
    )

    np.testing.assert_allclose(route_tt, [[[22.5, 15.0]]])
    np.testing.assert_allclose(observed, [[[0.0, 1.0]]])
    np.testing.assert_allclose(sample_counts, [[[0.0, 2.0]]])


def test_route_count_smoothing_uses_latent_prior_and_preserves_zero_strength():
    counts = np.asarray([[9.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    prior = np.asarray([[0.5, 0.5], [0.25, 0.75]], dtype=np.float64)

    unsmoothed = smooth_empirical_route_counts(
        counts,
        prior,
        prior_strength=0.0,
    )
    smoothed = smooth_empirical_route_counts(
        counts,
        prior,
        prior_strength=2.0,
    )

    np.testing.assert_allclose(unsmoothed, [[0.9, 0.1], [0.25, 0.75]])
    np.testing.assert_allclose(smoothed, [[10.0 / 12.0, 2.0 / 12.0], [0.25, 0.75]])


def test_causal_last_observed_route_counts_never_turns_missing_od_uniform():
    counts = np.asarray(
        [
            [[0.0, 0.0], [0.0, 3.0]],
            [[4.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 2.0]],
        ],
        dtype=np.float64,
    )
    initial = np.asarray([[0.9, 0.1], [0.25, 0.75]], dtype=np.float64)

    carried = causal_last_observed_route_counts(counts, initial)

    np.testing.assert_allclose(
        carried,
        [
            [[0.9, 0.1], [0.0, 1.0]],
            [[0.8, 0.2], [0.0, 1.0]],
            [[0.8, 0.2], [0.5, 0.5]],
        ],
    )


def test_route_anchor_blend_has_exact_persistence_and_gawron_endpoints():
    persistence = np.asarray([[[0.8, 0.2]]], dtype=np.float64)
    gawron = np.asarray([[[0.3, 0.7]]], dtype=np.float64)

    at_zero = blend_route_anchors(persistence, gawron, gate=0.0)
    at_one = blend_route_anchors(persistence, gawron, gate=1.0)
    halfway = blend_route_anchors(persistence, gawron, gate=0.5)

    np.testing.assert_allclose(at_zero, persistence)
    np.testing.assert_allclose(at_one, gawron)
    np.testing.assert_allclose(halfway, [[[0.55, 0.45]]])


def test_paired_route_metrics_compare_against_empirical_persistence():
    target = np.asarray(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    persistence = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    predictor = np.asarray(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    target_od_weight = np.asarray([[0.75, 0.25], [0.25, 0.75]])

    metrics, rows = paired_route_metrics(
        target=target,
        persistence=persistence,
        pred=predictor,
        od_weight=target_od_weight,
        block_size=1,
        bootstrap_samples=100,
        seed=7,
    )

    np.testing.assert_allclose(rows["route_tv"], [0.25, 0.25])
    np.testing.assert_allclose(rows["persistence_route_tv"], [0.75, 0.75])
    np.testing.assert_allclose(rows["route_tv_delta_vs_persistence"], [-0.5, -0.5])
    assert metrics["active_od_route_tv"] == 0.25
    assert metrics["route_tv_delta_vs_persistence_mean"] == -0.5
    assert metrics["route_tv_delta_vs_persistence_ci95"] == [-0.5, -0.5]


def test_route_candidate_acceptance_requires_route_gain_and_edge_guard():
    route_metrics = {
        "active_od_route_tv": 0.007,
        "persistence_active_od_route_tv": 0.010,
        "route_win_fraction_vs_persistence": 0.60,
    }

    accepted = route_candidate_acceptance(
        route_metrics=route_metrics,
        candidate_edge_tv=0.0101,
        persistence_edge_tv=0.0100,
        min_absolute_route_improvement=5e-5,
        min_relative_route_improvement=0.02,
        max_absolute_edge_degradation=5e-5,
        max_relative_edge_degradation=0.02,
        min_route_win_fraction=0.52,
    )
    rejected_for_edge = route_candidate_acceptance(
        route_metrics=route_metrics,
        candidate_edge_tv=0.0103,
        persistence_edge_tv=0.0100,
        min_absolute_route_improvement=5e-5,
        min_relative_route_improvement=0.02,
        max_absolute_edge_degradation=5e-5,
        max_relative_edge_degradation=0.02,
        min_route_win_fraction=0.52,
    )

    assert accepted["accepted"] is True
    assert accepted["required_route_gain"] == 0.0002
    assert accepted["allowed_edge_delta"] == 0.0002
    assert rejected_for_edge["accepted"] is False


def test_route_candidate_acceptance_rejects_tiny_or_unreliable_route_gain():
    tiny_gain = {
        "active_od_route_tv": 0.0099,
        "persistence_active_od_route_tv": 0.0100,
        "route_win_fraction_vs_persistence": 0.70,
    }
    low_win_fraction = {
        "active_od_route_tv": 0.0070,
        "persistence_active_od_route_tv": 0.0100,
        "route_win_fraction_vs_persistence": 0.50,
    }
    common = {
        "candidate_edge_tv": 0.0100,
        "persistence_edge_tv": 0.0100,
        "min_absolute_route_improvement": 5e-5,
        "min_relative_route_improvement": 0.02,
        "max_absolute_edge_degradation": 5e-5,
        "max_relative_edge_degradation": 0.02,
        "min_route_win_fraction": 0.52,
    }

    assert (
        route_candidate_acceptance(route_metrics=tiny_gain, **common)["accepted"]
        is False
    )
    assert (
        route_candidate_acceptance(
            route_metrics=low_win_fraction,
            **common,
        )["accepted"]
        is False
    )


def test_project_route_probs_to_edges_weights_each_od_by_hdv_population():
    route_probs = np.asarray(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )
    hdv_counts = np.asarray([[3.0, 1.0]], dtype=np.float64)
    route_to_edge = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )

    projected = project_route_probs_to_edges(route_probs, hdv_counts, route_to_edge)

    np.testing.assert_allclose(projected, np.asarray([[0.75, 0.25]]))


def test_sparse_route_projection_matches_dense_projection():
    route_probs = np.asarray(
        [
            [[0.25, 0.75], [0.6, 0.4]],
            [[0.8, 0.2], [0.1, 0.9]],
        ],
        dtype=np.float64,
    )
    hdv_counts = np.asarray([[3.0, 1.0], [1.0, 4.0]], dtype=np.float64)
    dense = np.asarray(
        [
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float64,
    )

    expected = project_route_probs_to_edges(route_probs, hdv_counts, dense)
    actual = project_route_probs_to_edges(
        route_probs,
        hdv_counts,
        SparseRouteEdgeProjection.from_dense(dense),
    )

    np.testing.assert_allclose(actual, expected)


def test_route_response_projection_is_exact_persistence_for_zero_route_change():
    previous_route = np.asarray([[[0.75, 0.25]]], dtype=np.float64)
    previous_edge = np.asarray([[0.6, 0.4]], dtype=np.float64)
    route_to_edge = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)

    projected = project_route_response_to_edges(
        predicted_route=previous_route,
        previous_route=previous_route,
        hdv_counts=np.asarray([[1.0]], dtype=np.float64),
        route_to_edge=route_to_edge,
        previous_edge=previous_edge,
    )

    np.testing.assert_allclose(projected, previous_edge)


def test_select_residual_scale_falls_back_to_persistence_when_residual_is_worse():
    target = np.asarray([[0.8, 0.2], [0.7, 0.3]], dtype=np.float64)
    persistence = np.asarray([[0.8, 0.2], [0.7, 0.3]], dtype=np.float64)
    residual_logits = np.asarray([[0.0, 2.0], [0.0, 2.0]], dtype=np.float64)

    selected = select_residual_scale(
        target=target,
        persistence=persistence,
        residual_logits=residual_logits,
        scales=(0.0, 0.25, 0.5, 1.0),
    )

    assert selected.scale == 0.0
    assert selected.candidate_tv == selected.persistence_tv


def test_gawron_replay_uses_previous_episode_feedback_for_next_route_distribution():
    initial_route = np.asarray([[0.6, 0.4]], dtype=np.float32)
    initial_cost = np.asarray([[10.0, 12.0]], dtype=np.float32)
    route_tt = np.asarray(
        [
            [[20.0, 8.0]],
            [[9.0, 18.0]],
        ],
        dtype=np.float32,
    )
    observed = np.ones_like(route_tt, dtype=np.float32)

    rollout = replay_gawron_route_state(
        initial_route=initial_route,
        initial_cost=initial_cost,
        route_tt=route_tt,
        route_tt_observed=observed,
        mode="routerl",
        lr=0.1,
        alpha_zero=0.2,
        beta=-0.5,
        update_mode="expected",
        min_prob=0.0,
    )

    expected_route_1, expected_cost_1 = gawron_step_from_route_tt(
        probs=initial_route[0],
        route_tt=route_tt[0, 0],
        mode="routerl",
        cost=initial_cost[0],
        alpha_zero=0.2,
        beta=-0.5,
        update_mode="expected",
    )
    expected_route_2, expected_cost_2 = gawron_step_from_route_tt(
        probs=expected_route_1,
        route_tt=route_tt[1, 0],
        mode="routerl",
        cost=expected_cost_1,
        alpha_zero=0.2,
        beta=-0.5,
        update_mode="expected",
    )

    np.testing.assert_allclose(rollout.route_before_episode[0, 0], initial_route[0])
    np.testing.assert_allclose(rollout.route_after_episode[0, 0], expected_route_1)
    np.testing.assert_allclose(rollout.cost_after_episode[0, 0], expected_cost_1)
    np.testing.assert_allclose(rollout.route_before_episode[1, 0], expected_route_1)
    np.testing.assert_allclose(rollout.route_after_episode[1, 0], expected_route_2)
    np.testing.assert_allclose(rollout.cost_after_episode[1, 0], expected_cost_2)


def test_gawron_replay_keeps_od_state_when_no_route_travel_time_was_observed():
    initial_route = np.asarray([[0.25, 0.75]], dtype=np.float32)
    initial_cost = np.asarray([[8.0, 9.0]], dtype=np.float32)

    rollout = replay_gawron_route_state(
        initial_route=initial_route,
        initial_cost=initial_cost,
        route_tt=np.zeros((1, 1, 2), dtype=np.float32),
        route_tt_observed=np.zeros((1, 1, 2), dtype=np.float32),
        mode="routerl",
        lr=0.1,
        alpha_zero=0.2,
        beta=-1.5,
        update_mode="expected",
        min_prob=0.0,
    )

    np.testing.assert_allclose(rollout.route_after_episode[0, 0], initial_route[0])
    np.testing.assert_allclose(rollout.cost_after_episode[0, 0], initial_cost[0])


def test_train_selected_gate_uses_train_for_early_stopping_and_scale_selection():
    episodes = np.arange(1, 7, dtype=np.int64)
    route = np.asarray(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    dataset = type(
        "SyntheticDataset",
        (),
        {
            "episodes": episodes,
            "av_route": np.zeros_like(route),
            "av_route_counts": np.zeros_like(route),
            "hdv_route": route,
            "hdv_route_counts": route * 10.0,
            "av_edge": np.zeros((6, 2), dtype=np.float32),
            "hdv_edge": route[:, 0],
            "hdv_od_weight": np.ones((6, 1), dtype=np.float32),
            "ods": [(0, 1)],
            "n_routes": 2,
            "edge_ids": ["e0", "e1"],
            "route_to_edge": np.asarray(
                [[[1.0, 0.0], [0.0, 1.0]]],
                dtype=np.float32,
            ),
        },
    )()
    proposal = np.asarray(
        [
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    gawron = GawronRouteRollout(
        route_before_episode=np.full_like(route, 0.5),
        cost_before_episode=np.ones_like(route),
        route_after_episode=proposal,
        cost_after_episode=np.ones_like(route),
        filled_route_tt=np.ones_like(route),
        route_tt_observed=np.ones_like(route),
    )
    pairs = CausalEpisodePairs(
        train=np.asarray([[0, 1]], dtype=np.int64),
        validation=np.asarray([[2, 3]], dtype=np.int64),
        test=np.asarray([[4, 5]], dtype=np.int64),
    )

    _, metadata, predictions = train_persistence_anchored_gate(
        dataset=dataset,
        gawron=gawron,
        route_tt_sample_counts=np.ones_like(route),
        pairs=pairs,
        scales=(0.0, 1.0),
        epochs=40,
        patience=10,
        lr=5e-2,
        weight_decay=0.0,
        gate_penalty=0.0,
        hidden_dim=4,
        depth=1,
        seed=7,
        device=torch.device("cpu"),
        progress_denominator=6,
        selection_split="train",
    )

    assert metadata["selection_split"] == "train"
    assert metadata["selected_gate_scale"] == 1.0
    assert metadata["best_selection_route_tv"] < metadata[
        "selection_persistence_route_tv"
    ]
    assert "train" in predictions
    np.testing.assert_allclose(predictions["train"], route[1:2], atol=1e-3)


def test_train_selected_residual_learns_response_opposite_to_gawron_direction():
    episodes = np.arange(1, 7, dtype=np.int64)
    route = np.asarray(
        [
            [[0.9, 0.1]],
            [[0.1, 0.9]],
            [[0.1, 0.9]],
            [[0.1, 0.9]],
            [[0.1, 0.9]],
            [[0.1, 0.9]],
        ],
        dtype=np.float32,
    )
    dataset = type(
        "SyntheticDataset",
        (),
        {
            "episodes": episodes,
            "av_route": np.zeros_like(route),
            "av_route_counts": np.zeros_like(route),
            "hdv_route": route,
            "hdv_route_counts": route * 10.0,
            "av_edge": np.zeros((6, 2), dtype=np.float32),
            "hdv_edge": route[:, 0],
            "hdv_od_weight": np.ones((6, 1), dtype=np.float32),
            "ods": [(0, 1)],
            "n_routes": 2,
            "edge_ids": ["e0", "e1"],
            "route_to_edge": np.asarray(
                [[[1.0, 0.0], [0.0, 1.0]]],
                dtype=np.float32,
            ),
        },
    )()
    # The constrained v3 gate can only move toward route 0 here, while the
    # observed response moves sharply toward route 1.
    proposal = np.asarray([[[1.0, 0.0]]] * 6, dtype=np.float32)
    gawron = GawronRouteRollout(
        route_before_episode=np.full_like(route, 0.5),
        cost_before_episode=np.ones_like(route),
        route_after_episode=proposal,
        cost_after_episode=np.ones_like(route),
        filled_route_tt=np.ones_like(route),
        route_tt_observed=np.ones_like(route),
    )
    pairs = CausalEpisodePairs(
        train=np.asarray([[0, 1]], dtype=np.int64),
        validation=np.asarray([[2, 3]], dtype=np.int64),
        test=np.asarray([[4, 5]], dtype=np.int64),
    )

    _, metadata, predictions = train_persistence_anchored_gate(
        dataset=dataset,
        gawron=gawron,
        route_tt_sample_counts=np.ones_like(route),
        pairs=pairs,
        scales=(0.0, 1.0),
        epochs=80,
        patience=20,
        lr=5e-2,
        weight_decay=0.0,
        gate_penalty=0.0,
        hidden_dim=4,
        depth=1,
        seed=7,
        device=torch.device("cpu"),
        progress_denominator=6,
        selection_split="train",
        response_mode="residual",
    )

    assert metadata["parameterization"] == (
        "empirical_persistence_anchored_route_residual"
    )
    assert metadata["selected_residual_scale"] == 1.0
    assert metadata["train_route_tv"] < metadata["train_persistence_route_tv"]
    assert predictions["train"][0, 0, 1] > 0.85
