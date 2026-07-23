from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablation.online_wm_dreamer_actor_critic import (  # noqa: E402
    CausalHdvResponseModel,
    OnlineWmDreamerCollector,
    OnlineWmDreamerConfig,
    OnlineWmDreamerTrainer,
)
from dreamer4.urban_routing.dreamer_hdv_response import (  # noqa: E402
    HDV_OD_ROUTE_PREDICTOR_FORMAT,
    HDV_OD_ROUTE_RESIDUAL_PREDICTOR_FORMAT,
    HdvOdRouteEpisodeObservation,
    PersistenceAnchoredOdRouteGate,
    PersistenceAnchoredOdRouteResidual,
    PretrainedHdvOdRouteResponsePredictor,
    apply_persistence_anchored_route_residual,
    apply_persistence_anchored_route_residual_torch,
    build_persistence_gate_features,
)
from scripts.evaluate_hdv_static_response import StaticHdvDataset  # noqa: E402
from scripts.train_hdv_response_predictor_offline import (  # noqa: E402
    GawronRouteRollout,
    _persistence_gate_features,
)


class _SyntheticRouteRegistry:
    n_od = 1
    K = 2
    od_pairs = [(0, 0)]
    route_edge_indices = [[[0, 1], [2]]]
    od_hdv_counts = np.asarray([10.0], dtype=np.float32)

    def iter_od_rows(self):
        yield 0, (0, 0)

    def od_index(self, origin: int, destination: int) -> int:
        return 0 if (int(origin), int(destination)) == (0, 0) else -1


def _checkpoint(*, gate_scale: float, gate_bias: float = 20.0):
    model = PersistenceAnchoredOdRouteGate(
        input_dim=9 * 2 + 7,
        hidden_dim=4,
        depth=1,
    )
    with torch.no_grad():
        output = model.net[-1]
        assert isinstance(output, torch.nn.Linear)
        output.weight.zero_()
        output.bias.fill_(float(gate_bias))
    return {
        "format": HDV_OD_ROUTE_PREDICTOR_FORMAT,
        "deployment_mode": (
            "persistence" if gate_scale == 0.0 else "persistence_anchored_gate_od_route"
        ),
        "selected_gate_scale": float(gate_scale),
        "n_routes": 2,
        "ods": [[0, 0]],
        "edge_ids": ["e0", "e1", "e2"],
        "hidden_dim": 4,
        "depth": 1,
        "progress_denominator": 10,
        "model_state_dict": model.state_dict(),
    }


def _residual_checkpoint(
    *,
    residual_scale: float,
    residual_bias: tuple[float, float] = (-1.0, 1.0),
):
    model = PersistenceAnchoredOdRouteResidual(
        input_dim=9 * 2 + 7,
        n_routes=2,
        hidden_dim=4,
        depth=1,
    )
    with torch.no_grad():
        output = model.net[-1]
        assert isinstance(output, torch.nn.Linear)
        output.weight.zero_()
        output.bias.copy_(torch.tensor(residual_bias, dtype=torch.float32))
    return {
        "format": HDV_OD_ROUTE_RESIDUAL_PREDICTOR_FORMAT,
        "deployment_mode": (
            "persistence"
            if residual_scale == 0.0
            else "persistence_anchored_residual_od_route"
        ),
        "selection_split": "train",
        "selected_residual_scale": float(residual_scale),
        "n_routes": 2,
        "ods": [[0, 0]],
        "edge_ids": ["e0", "e1", "e2"],
        "hidden_dim": 4,
        "depth": 1,
        "progress_denominator": 10,
        "model_state_dict": model.state_dict(),
    }


def _observation() -> HdvOdRouteEpisodeObservation:
    return HdvOdRouteEpisodeObservation(
        episode=1,
        route_counts=np.asarray([[8.0, 2.0]], dtype=np.float32),
        route_before=np.asarray([[0.5, 0.5]], dtype=np.float32),
        route_cost_before=np.asarray([[10.0, 20.0]], dtype=np.float32),
        filled_route_tt=np.asarray([[10.0, 30.0]], dtype=np.float32),
        route_tt_observed=np.asarray([[1.0, 1.0]], dtype=np.float32),
        route_tt_sample_counts=np.asarray([[8.0, 2.0]], dtype=np.float32),
        route_after=np.asarray([[0.1, 0.9]], dtype=np.float32),
        edge_distribution=np.asarray([4.0, 4.0, 1.0], dtype=np.float32) / 9.0,
    )


def _bound_predictor(*, gate_scale: float) -> PretrainedHdvOdRouteResponsePredictor:
    predictor = PretrainedHdvOdRouteResponsePredictor(
        _checkpoint(gate_scale=gate_scale),
        device=torch.device("cpu"),
    )
    predictor.bind_route_registry(
        route_registry=_SyntheticRouteRegistry(),
        edge_ids=["e0", "e1", "e2"],
        initial_route_probs={(0, 0): np.asarray([0.5, 0.5], dtype=np.float32)},
    )
    return predictor


def test_runtime_feature_contract_matches_offline_training_builder():
    dataset = StaticHdvDataset(
        episodes=np.asarray([1, 2], dtype=np.int64),
        av_route=np.zeros((2, 1, 2), dtype=np.float32),
        av_route_counts=np.zeros((2, 1, 2), dtype=np.float32),
        av_edge=np.zeros((2, 3), dtype=np.float32),
        hdv_route=np.asarray(
            [[[0.8, 0.2]], [[0.7, 0.3]]],
            dtype=np.float32,
        ),
        hdv_route_counts=np.asarray(
            [[[8.0, 2.0]], [[7.0, 3.0]]],
            dtype=np.float32,
        ),
        hdv_edge=np.zeros((2, 3), dtype=np.float32),
        hdv_od_weight=np.ones((2, 1), dtype=np.float32),
        ods=[(0, 0)],
        n_routes=2,
        edge_ids=["e0", "e1", "e2"],
        route_to_edge=np.zeros((1, 2, 3), dtype=np.float32),
    )
    gawron = GawronRouteRollout(
        route_before_episode=np.asarray(
            [[[0.5, 0.5]], [[0.4, 0.6]]],
            dtype=np.float32,
        ),
        cost_before_episode=np.asarray(
            [[[10.0, 20.0]], [[11.0, 19.0]]],
            dtype=np.float32,
        ),
        route_after_episode=np.asarray(
            [[[0.1, 0.9]], [[0.2, 0.8]]],
            dtype=np.float32,
        ),
        cost_after_episode=np.asarray(
            [[[11.0, 19.0]], [[12.0, 18.0]]],
            dtype=np.float32,
        ),
        filled_route_tt=np.asarray(
            [[[10.0, 30.0]], [[12.0, 25.0]]],
            dtype=np.float32,
        ),
        route_tt_observed=np.ones((2, 1, 2), dtype=np.float32),
    )
    tt_counts = np.asarray(
        [[[8.0, 2.0]], [[7.0, 3.0]]],
        dtype=np.float32,
    )
    offline = _persistence_gate_features(
        dataset,
        gawron,
        tt_counts,
        np.asarray([[0, 1]], dtype=np.int64),
        progress_denominator=10,
    )
    runtime = build_persistence_gate_features(
        empirical_route=np.asarray([[0.8, 0.2]], dtype=np.float32),
        route_before=gawron.route_before_episode[0],
        route_after=gawron.route_after_episode[0],
        route_cost_before=gawron.cost_before_episode[0],
        filled_route_tt=gawron.filled_route_tt[0],
        route_tt_observed=gawron.route_tt_observed[0],
        route_counts=dataset.hdv_route_counts[0],
        route_tt_sample_counts=tt_counts[0],
        od_weight=dataset.hdv_od_weight[0],
        progress=0.2,
    )

    np.testing.assert_allclose(runtime, offline[0], rtol=0.0, atol=1e-7)


def test_validation_selected_zero_scale_is_exact_edge_persistence():
    predictor = _bound_predictor(gate_scale=0.0)
    observation = _observation()
    predictor.observe_episode(observation)

    prediction, metrics = predictor.predict_for_next_episode(episode=2)

    np.testing.assert_allclose(
        prediction,
        observation.edge_distribution,
        rtol=0.0,
        atol=1e-7,
    )
    assert metrics["hdv_response_pred_source_code"] == 3.0
    assert metrics["hdv_response_pred_gate_scale"] == 0.0
    assert metrics["hdv_response_pred_route_delta_tv"] == pytest.approx(0.0)


def test_v3_checkpoint_ignores_null_v4_anchor_floor():
    state = _checkpoint(gate_scale=1.0)
    state["anchor_floor"] = None

    predictor = PretrainedHdvOdRouteResponsePredictor(
        state,
        device=torch.device("cpu"),
    )

    assert predictor.parameterization == "persistence_gawron_gate"


def test_selected_gate_projects_predicted_od_route_delta_to_edge_context():
    predictor = _bound_predictor(gate_scale=1.0)
    predictor.observe_episode(_observation())

    prediction, metrics = predictor.predict_for_next_episode(episode=2)
    target = _observation()
    target.episode = 2
    target.route_counts = np.asarray([[1.0, 9.0]], dtype=np.float32)
    accuracy = predictor.observe_episode(target)

    expected = np.asarray([0.1, 0.1, 0.9], dtype=np.float64)
    expected /= expected.sum()
    np.testing.assert_allclose(prediction, expected, rtol=0.0, atol=1e-6)
    assert metrics["hdv_response_pred_gate_mean"] > 0.999
    assert metrics["hdv_response_pred_route_delta_tv"] > 0.69
    assert accuracy["hdv_response_pred_route_tv"] < 1e-6
    assert accuracy["hdv_response_pred_route_observed_od_fraction"] == 1.0


def test_v4_residual_predicts_unrestricted_route_distribution_and_projects_edges():
    predictor = PretrainedHdvOdRouteResponsePredictor(
        _residual_checkpoint(residual_scale=1.0),
        device=torch.device("cpu"),
    )
    predictor.bind_route_registry(
        route_registry=_SyntheticRouteRegistry(),
        edge_ids=["e0", "e1", "e2"],
        initial_route_probs={(0, 0): np.asarray([0.5, 0.5], dtype=np.float32)},
    )
    predictor.observe_episode(_observation())

    prediction, metrics = predictor.predict_for_next_episode(episode=2)

    expected_route = apply_persistence_anchored_route_residual(
        np.asarray([[0.8, 0.2]]),
        np.asarray([[-1.0, 1.0]]),
        scale=1.0,
    )[0]
    expected_edge = np.asarray(
        [expected_route[0], expected_route[0], expected_route[1]],
        dtype=np.float64,
    )
    expected_edge /= expected_edge.sum()
    np.testing.assert_allclose(
        predictor.pending_predicted_route,
        expected_route[None, :],
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(prediction, expected_edge, rtol=0.0, atol=1e-6)
    assert metrics["hdv_response_pred_source_code"] == 4.0
    assert metrics["hdv_response_pred_residual_l2"] > 1.0
    assert metrics["hdv_response_pred_route_delta_tv"] > 0.4


def test_v4_residual_has_gradient_from_zero_probability_route():
    persistence = torch.tensor([[1.0, 0.0]])
    raw_residual = torch.zeros((1, 2), requires_grad=True)
    target = torch.tensor([[0.0, 1.0]])

    prediction = apply_persistence_anchored_route_residual_torch(
        persistence,
        raw_residual,
        scale=1.0,
    )
    loss = 0.5 * torch.abs(prediction - target).sum()
    loss.backward()

    torch.testing.assert_close(prediction, persistence)
    assert raw_residual.grad is not None
    assert float(raw_residual.grad[0, 1]) < 0.0
    assert float(raw_residual.grad.abs().sum()) > 0.0


def test_runtime_state_can_be_restored_before_registry_binding():
    predictor = _bound_predictor(gate_scale=1.0)
    predictor.observe_episode(_observation())
    expected, _ = predictor.predict_for_next_episode(episode=2)

    restored = PretrainedHdvOdRouteResponsePredictor(
        _checkpoint(gate_scale=1.0),
        device=torch.device("cpu"),
    )
    restored.load_runtime_state_dict(predictor.runtime_state_dict())
    restored.bind_route_registry(
        route_registry=_SyntheticRouteRegistry(),
        edge_ids=["e0", "e1", "e2"],
        initial_route_probs={(0, 0): np.asarray([0.5, 0.5], dtype=np.float32)},
    )
    actual, _ = restored.predict_for_next_episode(episode=2)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)


def test_runtime_adapter_rejects_edge_registry_mismatch():
    predictor = PretrainedHdvOdRouteResponsePredictor(
        _checkpoint(gate_scale=1.0),
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="edge registry mismatch"):
        predictor.bind_route_registry(
            route_registry=_SyntheticRouteRegistry(),
            edge_ids=["e1", "e0", "e2"],
            initial_route_probs={(0, 0): np.asarray([0.5, 0.5], dtype=np.float32)},
        )


def test_initial_edge_context_uses_exact_fixed_hdv_od_weights():
    class TwoOdRegistry:
        n_od = 2
        K = 2
        od_pairs = [(0, 0), (1, 0)]
        route_edge_indices = [
            [[0], [0]],
            [[1], [1]],
        ]
        # Deliberately different from the fixed-role counts passed below.
        od_hdv_counts = np.asarray([1.0, 9.0], dtype=np.float32)

        def iter_od_rows(self):
            yield 0, (0, 0)
            yield 1, (1, 0)

    state = _checkpoint(gate_scale=1.0)
    state["ods"] = [[0, 0], [1, 0]]
    state["edge_ids"] = ["e0", "e1"]
    predictor = PretrainedHdvOdRouteResponsePredictor(
        state,
        device=torch.device("cpu"),
    )
    predictor.bind_route_registry(
        route_registry=TwoOdRegistry(),
        edge_ids=["e0", "e1"],
        initial_route_probs={
            (0, 0): np.asarray([0.5, 0.5], dtype=np.float32),
            (1, 0): np.asarray([0.5, 0.5], dtype=np.float32),
        },
        initial_od_weights=np.asarray([9.0, 1.0], dtype=np.float32),
    )

    prediction, metrics = predictor.predict_for_next_episode(episode=1)

    np.testing.assert_allclose(prediction, [0.9, 0.1], rtol=0.0, atol=1e-7)
    assert metrics["hdv_response_pred_source_code"] == 1.75


def test_online_response_model_loads_v3_checkpoint_and_stays_frozen_by_default(
    tmp_path,
):
    checkpoint = tmp_path / "hdv_response_v3.pt"
    torch.save(_checkpoint(gate_scale=1.0), checkpoint)
    cfg = OnlineWmDreamerConfig(
        hdv_response_predictor_enabled=True,
        hdv_response_predictor_ckpt=str(checkpoint),
    )
    model = CausalHdvResponseModel(
        n_edges=3,
        cfg=cfg,
        device=torch.device("cpu"),
    )

    model.load_checkpoint(checkpoint)
    model.bind_od_route_runtime(
        route_registry=_SyntheticRouteRegistry(),
        edge_ids=["e0", "e1", "e2"],
        initial_route_probs={(0, 0): np.asarray([0.5, 0.5], dtype=np.float32)},
    )
    model.observe_od_route_episode(_observation())
    prediction, metrics = model.predict_for_next_episode([], episode=2)
    update = model.update([])

    assert prediction is not None
    assert metrics["hdv_response_pred_source_code"] == 3.0
    assert model.parameterization == "persistence_gated_od_route_v3"
    assert update["hdv_response_updates"] == 0.0
    assert update["hdv_response_frozen_pretrained"] == 1.0


def test_online_response_model_finetunes_v3_after_target_episode(tmp_path):
    checkpoint = tmp_path / "hdv_response_v3.pt"
    torch.save(_checkpoint(gate_scale=1.0, gate_bias=-4.0), checkpoint)
    cfg = OnlineWmDreamerConfig(
        hdv_response_predictor_enabled=True,
        hdv_response_predictor_ckpt=str(checkpoint),
        hdv_response_predictor_online_update_enabled=True,
        hdv_response_predictor_online_lr=5e-2,
        hdv_response_predictor_online_update_epochs=20,
        hdv_response_predictor_online_buffer_episodes=4,
    )
    model = CausalHdvResponseModel(
        n_edges=3,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    model.load_checkpoint(checkpoint)
    model.bind_od_route_runtime(
        route_registry=_SyntheticRouteRegistry(),
        edge_ids=["e0", "e1", "e2"],
        initial_route_probs={(0, 0): np.asarray([0.5, 0.5], dtype=np.float32)},
    )
    model.observe_od_route_episode(_observation())
    prediction_before_update, _ = model.predict_for_next_episode([], episode=2)
    parameters_before_update = {
        name: value.detach().clone()
        for name, value in model.model.state_dict().items()
    }

    target = _observation()
    target.episode = 2
    target.route_counts = np.asarray([[1.0, 9.0]], dtype=np.float32)
    accuracy = model.observe_od_route_episode(target)
    update = model.update([])

    assert prediction_before_update is not None
    assert accuracy["hdv_response_pred_route_tv"] > 0.5
    assert update["hdv_response_updates"] == 20.0
    assert update["hdv_response_online_pairs"] == 1.0
    assert update["hdv_response_frozen_pretrained"] == 0.0
    assert update["hdv_response_online_train_tv"] < accuracy[
        "hdv_response_pred_route_tv"
    ]
    assert any(
        not torch.equal(parameters_before_update[name], value)
        for name, value in model.model.state_dict().items()
    )


def test_online_response_model_finetunes_v4_after_target_episode(tmp_path):
    checkpoint = tmp_path / "hdv_response_v4.pt"
    torch.save(
        _residual_checkpoint(
            residual_scale=1.0,
            residual_bias=(0.0, 0.0),
        ),
        checkpoint,
    )
    cfg = OnlineWmDreamerConfig(
        hdv_response_predictor_enabled=True,
        hdv_response_predictor_ckpt=str(checkpoint),
        hdv_response_predictor_online_update_enabled=True,
        hdv_response_predictor_online_lr=5e-2,
        hdv_response_predictor_online_update_epochs=20,
        hdv_response_predictor_online_buffer_episodes=4,
    )
    model = CausalHdvResponseModel(
        n_edges=3,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    model.load_checkpoint(checkpoint)
    model.bind_od_route_runtime(
        route_registry=_SyntheticRouteRegistry(),
        edge_ids=["e0", "e1", "e2"],
        initial_route_probs={(0, 0): np.asarray([0.5, 0.5], dtype=np.float32)},
    )
    model.observe_od_route_episode(_observation())
    prediction_before_update, metrics = model.predict_for_next_episode([], episode=2)

    target = _observation()
    target.episode = 2
    target.route_counts = np.asarray([[1.0, 9.0]], dtype=np.float32)
    accuracy = model.observe_od_route_episode(target)
    update = model.update([])

    assert prediction_before_update is not None
    assert metrics["hdv_response_pred_source_code"] == 4.0
    assert model.parameterization == "persistence_residual_od_route_v4"
    assert accuracy["hdv_response_pred_route_tv"] > 0.6
    assert update["hdv_response_updates"] == 20.0
    assert update["hdv_response_online_pairs"] == 1.0
    assert update["hdv_response_online_train_tv"] < accuracy[
        "hdv_response_pred_route_tv"
    ]


def test_online_response_model_restores_v3_optimizer_and_training_buffer(tmp_path):
    checkpoint = tmp_path / "hdv_response_v3.pt"
    torch.save(_checkpoint(gate_scale=1.0, gate_bias=-4.0), checkpoint)
    cfg = OnlineWmDreamerConfig(
        hdv_response_predictor_enabled=True,
        hdv_response_predictor_ckpt=str(checkpoint),
        hdv_response_predictor_online_update_enabled=True,
        hdv_response_predictor_online_lr=5e-2,
        hdv_response_predictor_online_update_epochs=1,
        hdv_response_predictor_online_buffer_episodes=4,
    )
    model = CausalHdvResponseModel(
        n_edges=3,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    model.load_checkpoint(checkpoint)
    model.bind_od_route_runtime(
        route_registry=_SyntheticRouteRegistry(),
        edge_ids=["e0", "e1", "e2"],
        initial_route_probs={(0, 0): np.asarray([0.5, 0.5], dtype=np.float32)},
    )
    model.observe_od_route_episode(_observation())
    model.predict_for_next_episode([], episode=2)
    target = _observation()
    target.episode = 2
    target.route_counts = np.asarray([[1.0, 9.0]], dtype=np.float32)
    model.observe_od_route_episode(target)
    model.update([])
    saved = tmp_path / "online_v3.pt"
    model.save_checkpoint(saved)

    restored = CausalHdvResponseModel(
        n_edges=3,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    restored.load_checkpoint(saved, load_optimizer=True)
    restored.bind_od_route_runtime(
        route_registry=_SyntheticRouteRegistry(),
        edge_ids=["e0", "e1", "e2"],
        initial_route_probs={(0, 0): np.asarray([0.5, 0.5], dtype=np.float32)},
    )

    assert restored.od_route_predictor is not None
    assert model.od_route_predictor is not None
    assert restored.trained_pairs == model.trained_pairs
    assert len(restored.od_route_predictor.online_training_samples) == 1
    for name, expected in model.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], expected)
    assert restored.optimizer.state_dict()["state"]


def test_online_response_model_restores_v4_optimizer_and_training_buffer(tmp_path):
    checkpoint = tmp_path / "hdv_response_v4.pt"
    torch.save(
        _residual_checkpoint(
            residual_scale=1.0,
            residual_bias=(0.0, 0.0),
        ),
        checkpoint,
    )
    cfg = OnlineWmDreamerConfig(
        hdv_response_predictor_enabled=True,
        hdv_response_predictor_ckpt=str(checkpoint),
        hdv_response_predictor_online_update_enabled=True,
        hdv_response_predictor_online_lr=5e-2,
        hdv_response_predictor_online_update_epochs=1,
        hdv_response_predictor_online_buffer_episodes=4,
    )
    model = CausalHdvResponseModel(
        n_edges=3,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    model.load_checkpoint(checkpoint)
    model.bind_od_route_runtime(
        route_registry=_SyntheticRouteRegistry(),
        edge_ids=["e0", "e1", "e2"],
        initial_route_probs={(0, 0): np.asarray([0.5, 0.5], dtype=np.float32)},
    )
    model.observe_od_route_episode(_observation())
    model.predict_for_next_episode([], episode=2)
    target = _observation()
    target.episode = 2
    target.route_counts = np.asarray([[1.0, 9.0]], dtype=np.float32)
    model.observe_od_route_episode(target)
    model.update([])
    saved = model.save_checkpoint(tmp_path / "online_v4.pt")

    restored = CausalHdvResponseModel(
        n_edges=3,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    state = restored.load_checkpoint(saved, load_optimizer=True)
    restored.bind_od_route_runtime(
        route_registry=_SyntheticRouteRegistry(),
        edge_ids=["e0", "e1", "e2"],
        initial_route_probs={(0, 0): np.asarray([0.5, 0.5], dtype=np.float32)},
    )

    assert state["format"] == HDV_OD_ROUTE_RESIDUAL_PREDICTOR_FORMAT
    assert restored.parameterization == "persistence_residual_od_route_v4"
    assert restored.od_route_predictor is not None
    assert len(restored.od_route_predictor.online_training_samples) == 1
    assert restored.optimizer.state_dict()["state"]
    for name, expected in model.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], expected)


def test_resume_keeps_explicit_v3_checkpoint_when_old_run_contains_edge_predictor(
    tmp_path,
):
    cfg = OnlineWmDreamerConfig(hdv_response_predictor_enabled=True)
    response_model = CausalHdvResponseModel(
        n_edges=3,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    response_model.load_od_route_checkpoint_state(_checkpoint(gate_scale=1.0))
    actor = torch.nn.Linear(2, 2)
    resume = tmp_path / "legacy_response_in_main_checkpoint.pt"
    torch.save(
        {
            "episode": 17,
            "actor": actor.state_dict(),
            "hdv_response_predictor": CausalHdvResponseModel(
                n_edges=3,
                cfg=cfg,
                device=torch.device("cpu"),
            ).model.state_dict(),
            "hdv_response_parameterization": "absolute_logits",
        },
        resume,
    )
    trainer = object.__new__(OnlineWmDreamerTrainer)
    trainer.cfg = OnlineWmDreamerConfig(resume_checkpoint=str(resume))
    trainer.device = torch.device("cpu")
    trainer.actor = torch.nn.Linear(2, 2)
    trainer.critic = torch.nn.Linear(2, 1)
    trainer.global_critic = torch.nn.Linear(2, 1)
    trainer.wm = type(
        "DummyWm",
        (),
        {
            "flow_projector": torch.nn.Linear(2, 2),
            "dynamics": torch.nn.Linear(2, 2),
        },
    )()
    trainer.hdv_response_model = response_model

    resumed_episode = trainer._load_resume_checkpoint()

    assert resumed_episode == 17
    assert response_model.od_route_predictor is not None
    assert response_model.parameterization == "persistence_gated_od_route_v3"


def test_collector_feedback_uses_hdv_routes_and_pooled_arrival_times():
    collector = object.__new__(OnlineWmDreamerCollector)
    collector._route_registry = _SyntheticRouteRegistry()
    collector.n_routes = 2
    collector._hdv_vehicles = {"h0", "h1"}
    collector._veh_od = {"h0": (0, 0), "h1": (0, 0)}
    collector._hdv_route_idx = {"h0": 0, "h1": 0}
    collector._veh_candidates = {
        "h0": [["e0", "e1"], ["e2"]],
        "h1": [["e0", "e1"], ["e2"]],
    }
    collector._last_ep_route_tts = {(0, 0, 0): [10.0, 14.0]}

    route_counts, filled_tt, observed, tt_counts = collector._hdv_od_route_feedback()

    np.testing.assert_array_equal(route_counts, [[2.0, 0.0]])
    np.testing.assert_allclose(filled_tt, [[12.0, 18.0]])
    np.testing.assert_array_equal(observed, [[1.0, 0.0]])
    np.testing.assert_array_equal(tt_counts, [[2.0, 0.0]])
