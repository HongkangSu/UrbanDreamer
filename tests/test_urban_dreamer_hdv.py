from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from urban_dreamer.models.hdv import (
    DEFAULT_HDV_RESPONSE_DEPTH,
    DEFAULT_HDV_RESPONSE_HIDDEN_DIM,
    EpisodeObservation,
    HDV_RESPONSE_CHECKPOINT_FORMAT,
    HDV_RESPONSE_K,
    HdvOdRouteResponsePredictor,
    PersistenceAnchoredOdRouteResidual,
)
from urban_dreamer.traffic.topology import EdgeNetwork, TrafficTopology, VehicleDemand


def _topology() -> TrafficTopology:
    network = EdgeNetwork(
        edge_ids=("e0", "e1", "e2", "e3", "e4"),
        edge_to_index={"e0": 0, "e1": 1, "e2": 2, "e3": 3, "e4": 4},
        static_features=np.ones((5, 4), dtype=np.float32),
        speed_limits_mps=np.asarray([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32),
        capacities=np.asarray([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32),
        lengths_m=np.asarray([100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32),
        adjacency=np.asarray(
            [
                [0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        adjacency_list=((1, 2, 3, 4), (), (), (), ()),
        passenger_allowed=np.asarray([True, True, True, True, True]),
        free_flow_travel_seconds=np.asarray(
            [10.0, 10.0, 20.0, 30.0, 40.0],
            dtype=np.float32,
        ),
    )
    return TrafficTopology(
        network=network,
        origins=("e0",),
        destinations=("e1",),
        vehicles=(
            VehicleDemand("av0", 0, 0, 0, True),
            VehicleDemand("h0", 0, 0, 0, False),
            VehicleDemand("h1", 0, 0, 0, False),
        ),
        od_pairs=((0, 0),),
        route_paths=(((0, 1), (0, 2), (0, 3), (0, 4)),),
        od_automated_counts=np.asarray([1.0], dtype=np.float32),
        od_human_counts=np.asarray([2.0], dtype=np.float32),
    )


def _initial_state(path: Path, route_policy: list[list[float]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "route_policy": [
                    {
                        "origin": 0,
                        "destination": 0,
                        "probs": route_policy[0],
                        "costs": [10.0, 20.0, 30.0, 40.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _checkpoint(
    path: Path,
    *,
    topology: TrafficTopology,
    initial_state_path: Path,
    bias: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> Path:
    model = PersistenceAnchoredOdRouteResidual()
    with torch.no_grad():
        output = model.net[-1]
        assert isinstance(output, torch.nn.Linear)
        output.weight.zero_()
        output.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    torch.save(
        {
            "format": HDV_RESPONSE_CHECKPOINT_FORMAT,
            "deployment_mode": "persistence_anchored_residual_od_route",
            "parameterization": "empirical_persistence_anchored_route_residual",
            "selection_split": "train",
            "route_target": "seen_hdv_exact_hdv_action",
            "travel_time_feedback": (
                "seen AV policy_action plus seen HDV hdv_action"
            ),
            "selected_residual_scale": 1.0,
            "anchor_floor": 1e-3,
            "n_routes": HDV_RESPONSE_K,
            "ods": [list(od) for od in topology.od_pairs],
            "edge_ids": list(topology.network.edge_ids),
            "hidden_dim": DEFAULT_HDV_RESPONSE_HIDDEN_DIM,
            "depth": DEFAULT_HDV_RESPONSE_DEPTH,
            "progress_denominator": 10,
            "model_state_dict": model.state_dict(),
            "trained_pairs": 0,
            "initial_hdv_state": str(initial_state_path),
        },
        path,
    )
    return path


def _episode(
    *,
    episode: int,
    pre_policy: list[float],
    post_policy: list[float],
    hdv_route_choices: dict[str, int],
    travel_times: dict[str, float],
    travel_time_route_choices: dict[str, int] | None = None,
) -> EpisodeObservation:
    pre_state = {
        "version": 1,
        "route_policy": [
            {
                "origin": 0,
                "destination": 0,
                "probs": pre_policy,
                "costs": [10.0, 20.0, 30.0, 40.0],
            }
        ],
    }
    post_state = {
        "version": 1,
        "route_policy": [
            {
                "origin": 0,
                "destination": 0,
                "probs": post_policy,
                "costs": [9.0, 18.0, 27.0, 36.0],
            }
        ],
    }
    return EpisodeObservation(
        episode=episode,
        hdv_route_choices=hdv_route_choices,
        travel_time_route_choices=(
            dict(hdv_route_choices)
            if travel_time_route_choices is None
            else dict(travel_time_route_choices)
        ),
        travel_times_seconds=travel_times,
        pre_state=pre_state,
        post_state=post_state,
    )


def _predictor(
    tmp_path: Path,
    *,
    initial_route_policy: list[list[float]] | None = None,
    bias: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> tuple[HdvOdRouteResponsePredictor, TrafficTopology, Path, Path]:
    topology = _topology()
    route_policy = initial_route_policy or [[0.7, 0.2, 0.05, 0.05]]
    state_path = _initial_state(tmp_path / "hdv.json", route_policy)
    checkpoint_path = _checkpoint(
        tmp_path / "predictor.pt",
        topology=topology,
        initial_state_path=state_path,
        bias=bias,
    )
    predictor = HdvOdRouteResponsePredictor(
        checkpoint_path,
        topology,
        state_path,
        device="cpu",
    )
    return predictor, topology, checkpoint_path, state_path


def test_strict_v4_loading_requires_current_format(tmp_path: Path) -> None:
    predictor, topology, checkpoint_path, state_path = _predictor(tmp_path)
    del predictor
    broken = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    broken["format"] = "causal_persistence_residual_hdv_od_route_predictor_v3"
    torch.save(broken, checkpoint_path)

    with pytest.raises(ValueError, match="strict v4 format"):
        HdvOdRouteResponsePredictor(checkpoint_path, topology, state_path, device="cpu")


def test_constructor_rejects_topology_and_initial_state_mismatches(tmp_path: Path) -> None:
    topology = _topology()
    deployment_state_path = _initial_state(
        tmp_path / "deployment_hdv.json",
        [[0.7, 0.2, 0.05, 0.05]],
    )
    state_path = _initial_state(tmp_path / "hdv.json", [[0.7, 0.2, 0.1]])
    checkpoint_path = _checkpoint(
        tmp_path / "predictor.pt",
        topology=topology,
        initial_state_path=deployment_state_path,
    )
    with pytest.raises(ValueError, match="route_policy probs width mismatch"):
        HdvOdRouteResponsePredictor(checkpoint_path, topology, state_path, device="cpu")

    state_path = _initial_state(tmp_path / "hdv_ok.json", [[0.7, 0.2, 0.05, 0.05]])
    broken = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    broken["edge_ids"] = ["x0", "x1", "x2", "x3", "x4"]
    torch.save(broken, checkpoint_path)
    with pytest.raises(ValueError, match="edge_ids do not match topology"):
        HdvOdRouteResponsePredictor(checkpoint_path, topology, state_path, device="cpu")


def test_route_target_is_hdv_only_but_travel_time_feedback_includes_av(
    tmp_path: Path,
) -> None:
    predictor, _, _, _ = _predictor(tmp_path)
    observation = _episode(
        episode=1,
        pre_policy=[0.7, 0.2, 0.05, 0.05],
        post_policy=[0.6, 0.25, 0.1, 0.05],
        hdv_route_choices={"h0": 0, "h1": 1},
        travel_time_route_choices={"h0": 0, "h1": 1, "av0": 2},
        travel_times={"h0": 10.0, "h1": 20.0, "av0": 50.0},
    )

    processed = predictor._process_episode(observation)

    np.testing.assert_array_equal(
        processed.route_counts,
        np.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        processed.route_tt_sample_counts,
        np.asarray([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        processed.filled_route_tt,
        np.asarray([[10.0, 20.0, 50.0, 75.0]], dtype=np.float32),
        rtol=0.0,
        atol=1e-7,
    )


def test_causal_predict_observe_update_flow(tmp_path: Path) -> None:
    predictor, _, _, _ = _predictor(tmp_path, bias=(0.0, 0.0, 0.0, 0.0))
    episode1 = _episode(
        episode=1,
        pre_policy=[0.7, 0.2, 0.05, 0.05],
        post_policy=[0.1, 0.1, 0.1, 0.7],
        hdv_route_choices={"h0": 0, "h1": 0},
        travel_times={"h0": 10.0, "h1": 11.0},
    )
    episode2 = _episode(
        episode=2,
        pre_policy=[0.1, 0.1, 0.1, 0.7],
        post_policy=[0.05, 0.05, 0.1, 0.8],
        hdv_route_choices={"h0": 3, "h1": 3},
        travel_times={"h0": 40.0, "h1": 41.0},
    )
    episode3 = _episode(
        episode=3,
        pre_policy=[0.05, 0.05, 0.1, 0.8],
        post_policy=[0.05, 0.05, 0.05, 0.85],
        hdv_route_choices={"h0": 3, "h1": 3},
        travel_times={"h0": 39.0, "h1": 40.0},
    )

    assert predictor.observe_episode(episode1) == {}

    before_update_edge, before_update_metrics = predictor.predict_for_next_episode(2)
    accuracy2 = predictor.observe_episode(episode2, collect_online_sample=True)
    predictor.predict_for_next_episode(3)
    predictor.observe_episode(episode3, collect_online_sample=True)
    before_train_edge, _ = predictor.predict_for_next_episode(4)
    update_metrics = predictor.update_online()
    after_train_edge, _ = predictor.predict_for_next_episode(4)

    assert before_update_edge.shape == (5,)
    assert before_update_metrics["hdv_response_pred_source_code"] == 4.0
    assert accuracy2["hdv_response_pred_route_tv"] > 0.0
    assert update_metrics["hdv_response_updates"] == 2.0
    assert update_metrics["hdv_response_online_pairs"] == 2.0
    assert update_metrics["hdv_response_online_train_tv"] >= 0.0
    assert predictor.online_update_steps == 2
    assert predictor.optimizer.state_dict()["state"]
    np.testing.assert_allclose(after_train_edge, before_train_edge, rtol=0.0, atol=1e-7)


def test_state_dict_roundtrip_restores_optimizer_runtime_and_pending_prediction(
    tmp_path: Path,
) -> None:
    predictor, topology, checkpoint_path, state_path = _predictor(tmp_path)
    predictor.observe_episode(
        _episode(
            episode=1,
            pre_policy=[0.7, 0.2, 0.05, 0.05],
            post_policy=[0.1, 0.1, 0.1, 0.7],
            hdv_route_choices={"h0": 0, "h1": 1},
            travel_times={"h0": 10.0, "h1": 20.0},
        )
    )
    predictor.predict_for_next_episode(2)
    predictor.observe_episode(
        _episode(
            episode=2,
            pre_policy=[0.1, 0.1, 0.1, 0.7],
            post_policy=[0.05, 0.05, 0.1, 0.8],
            hdv_route_choices={"h0": 3, "h1": 3},
            travel_times={"h0": 39.0, "h1": 40.0},
        ),
        collect_online_sample=True,
    )
    predictor.predict_for_next_episode(3)
    predictor.observe_episode(
        _episode(
            episode=3,
            pre_policy=[0.05, 0.05, 0.1, 0.8],
            post_policy=[0.05, 0.05, 0.05, 0.85],
            hdv_route_choices={"h0": 3, "h1": 3},
            travel_times={"h0": 38.0, "h1": 39.0},
        ),
        collect_online_sample=True,
    )
    predictor.update_online()
    predictor.predict_for_next_episode(4)

    saved = predictor.state_dict()
    restored = HdvOdRouteResponsePredictor(
        checkpoint_path,
        topology,
        state_path,
        device="cpu",
    )
    restored.load_state_dict(saved)

    assert restored.trained_pairs == predictor.trained_pairs
    assert restored.online_update_steps == predictor.online_update_steps
    assert restored.pending_prediction_episode == predictor.pending_prediction_episode
    assert len(restored.online_training_samples) == len(predictor.online_training_samples)
    np.testing.assert_allclose(
        restored.last_observed_route,
        predictor.last_observed_route,
        rtol=0.0,
        atol=1e-7,
    )
    assert restored.pending_predicted_route is not None
    assert predictor.pending_predicted_route is not None
    np.testing.assert_allclose(
        restored.pending_predicted_route,
        predictor.pending_predicted_route,
        rtol=0.0,
        atol=1e-7,
    )
    assert restored.optimizer.state_dict()["state"]
