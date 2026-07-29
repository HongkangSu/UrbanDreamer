from __future__ import annotations

import numpy as np

from urban_dreamer.traffic.environment import ActionDecision
from urban_dreamer.learning.observations import (
    ACTOR_OBSERVATION_DIM,
    advance_vehicle_on_path,
    aggregate_planned_edge_flow,
    build_actor_observations,
)


def test_fixed_actor_observation_has_the_k4_33_feature_layout() -> None:
    edge_features = np.asarray(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        dtype=np.float32,
    )
    candidate_paths = [
        [[0, 1], [1, 2], [2], [0, 2]],
        [[2], [1], [0], [0, 1]],
    ]

    observation = build_actor_observations(
        policy_edge_features=edge_features,
        candidate_paths=candidate_paths,
        edge_travel_seconds=np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
        assigned_route_counts=np.asarray(
            [1.0, 2.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        assigned_edge_pressure=np.asarray(
            [0.5, 1.0, 0.0],
            dtype=np.float32,
        ),
        episode_progress=0.25,
        active_av_fraction=0.5,
    )

    assert ACTOR_OBSERVATION_DIM == 33
    assert observation.shape == (2, ACTOR_OBSERVATION_DIM)
    np.testing.assert_allclose(
        observation[0, -3:],
        [0.25, 0.5, 0.0],
        atol=1e-6,
    )


def test_actor_observation_rejects_non_mainline_route_count() -> None:
    try:
        build_actor_observations(
            policy_edge_features=np.ones((2, 2), dtype=np.float32),
            candidate_paths=[[[0], [1]]],
            edge_travel_seconds=np.ones((2,), dtype=np.float32),
            assigned_route_counts=np.ones((4,), dtype=np.float32),
            assigned_edge_pressure=np.ones((2,), dtype=np.float32),
            episode_progress=0.0,
            active_av_fraction=1.0,
        )
    except ValueError as exc:
        assert "exactly four" in str(exc)
    else:  # pragma: no cover - protects the fixed mainline contract
        raise AssertionError("non-K4 candidate paths were accepted")


def test_aggregate_planned_edge_flow_means_over_controlled_vehicles() -> None:
    decisions = [
        ActionDecision.build(
            route_index=0,
            candidate_paths=[[0, 1, 2], [0, 2], [0, 1], [0, 2]],
            edge_order=[0, 1, 2],
            edge_travel_times=[40.0, 20.0, 20.0],
            decision_horizon_seconds=30.0,
        ),
        ActionDecision.build(
            route_index=1,
            candidate_paths=[[1, 2], [1, 0], [1, 2], [1, 0]],
            edge_order=[0, 1, 2],
            edge_travel_times=[40.0, 20.0, 20.0],
            decision_horizon_seconds=30.0,
        ),
    ]

    aggregated = aggregate_planned_edge_flow(decisions, edge_count=3)

    np.testing.assert_allclose(aggregated, [0.5, 0.0, 0.0], atol=1e-6)


def test_advance_vehicle_on_path_preserves_remaining_edge_time() -> None:
    vehicle = {
        "current_edge": 0,
        "remaining_edge_seconds": 5.0,
        "arrived": False,
    }

    advance_vehicle_on_path(
        vehicle,
        selected_path=[0, 1, 2],
        edge_travel_seconds=np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
        interval_seconds=15.0,
        edge_count=3,
    )

    assert vehicle == {
        "current_edge": 1,
        "remaining_edge_seconds": 10.0,
        "arrived": False,
    }
