import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ablation.online_wm_dreamer_actor_critic as online_wm_module  # noqa: E402

from ablation.online_wm_dreamer_actor_critic import (  # noqa: E402
    ACTIVE_COUNT_READOUT_SEMANTICS,
    AggregateActiveVehicleReadout,
    CausalHdvResponseModel,
    GLOBAL_CRITIC_SEMANTICS,
    OnlineHdvEpisodeSummary,
    OnlineImaginedActorCritic,
    OnlineWmDreamerConfig,
    OnlineWmDreamerCollector,
    OnlineWmDreamerTrainer,
    OnlineWorldModelReplay,
    OnlineWorldModelStep,
    OnlineWorldModelUpdater,
    _causal_pair,
    _advance_imagined_route_progress,
    _active_vehicle_count_from_latent,
    _active_vehicle_count_readout_supervision_loss,
    _aggregate_selected_routes_to_action_edge,
    _decoded_active_vehicle_count,
    _combine_vehicle_global_advantages,
    _current_to_dest_candidate_paths,
    _sample_masked_actions,
    _discounted_returns_by_vehicle_ids,
    _global_active_vehicle_reward,
    _global_network_reward_from_latent,
    _imagined_global_context,
    _nominal_transition_active_vehicle_count,
    _real_global_transition_part,
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
        wm_gradient_checkpointing=True,
        wm_space_attention_batch_chunk_size=2,
        wm_mlp_token_chunk_size=64,
        wm_time_attention_batch_chunk_size=3,
        imag_horizon=8,
        imagined_rollout_batch_size=4,
    ).resolve()

    plan = build_gretz_online_wm_dreamer_plan(cfg)

    assert plan["ablation"] == "gretz_online_wm_dreamer_actor_critic"
    assert plan["world_model_usage"]["cfm_transition_in_training"] is True
    assert plan["world_model_usage"]["world_model_online_update"] is True
    assert plan["world_model_usage"]["direct_real_a2c_update"] is False
    assert plan["world_model_usage"]["updated_modules"] == [
        "flow_projector",
        "cfm_dynamics",
        "transition_exposure_readout",
    ]
    assert plan["world_model_usage"]["wm_grad_enabled"] is True
    assert plan["training"]["wm_replay_buffer_episodes"] == 8
    assert plan["training"]["wm_update_epochs"] == 1
    assert plan["training"]["wm_update_every_episodes"] == 5
    assert plan["training"]["wm_replay_require_completion"] is True
    assert plan["training"]["wm_replay_completion_threshold"] == 1.0
    assert plan["training"]["wm_batch_size"] == 16
    assert plan["training"]["wm_gradient_checkpointing"] is True
    assert plan["training"]["wm_space_attention_batch_chunk_size"] == 2
    assert plan["training"]["wm_mlp_token_chunk_size"] == 64
    assert plan["training"]["wm_time_attention_batch_chunk_size"] == 3
    assert plan["training"]["wm_hdv_target_source"] == "episode_mean_current_occupancy"
    assert plan["training"]["wm_hdv_policy_target_mode"] == "episode_mean"
    assert plan["training"]["wm_hdv_policy_action_mode"] == "window_mean"
    assert plan["training"]["wm_transition_hdv_context_mode"] == "episode_response"
    assert plan["training"]["imagined_hdv_context_source"] == "oracle_current_occupancy"
    assert plan["training"]["hdv_response_predictor_enabled"] is False
    assert plan["training"]["wm_hdv_pressure_trace_enabled"] is True
    assert plan["training"]["imag_horizon"] == 8
    assert plan["training"]["imagined_rollout_batch_size"] == 4
    assert plan["training"]["global_critic_enabled"] is True
    assert plan["training"]["global_critic_semantics"] == "state_value_v1"
    assert plan["training"]["ac_global_advantage_weight"] == 1.0
    assert plan["training"]["wm_t_all_supervision_weight"] == 0.2
    assert plan["reward_semantics"]["global_reward_semantics"] == (
        "active_vehicle_transition_exposure_readout_v4"
    )
    assert plan["reward_semantics"]["global_reward_enabled"] is True
    assert plan["reward_semantics"]["global_enablement"] == "enabled_by_default"
    assert plan["reward_semantics"]["global_source"] == (
        "real_sumo_vehicle_seconds_and_imagined_calibrated_exposure"
    )
    assert plan["reward_semantics"]["global_imagined_active_exposure_source"] == (
        "current_next_latent_transition_exposure_readout_with_scheduled_duration"
    )
    assert plan["reward_semantics"]["global_raw_decoder_sum_role"] == (
        "initialization_anchor_diagnostic_and_legacy_fallback"
    )
    assert plan["reward_semantics"]["global_unit"] == "mean_travel_time_minutes"
    assert plan["world_model_usage"]["imagined_action_mode"] == "stochastic_masked_sample_one_hot"
    assert plan["world_model_usage"]["real_collection_action_mode"] == "stochastic_policy_sample_one_hot"
    assert plan["training"]["real_policy_route_selection"] == "sample"
    assert plan["evaluation"]["action_mode"] == "stochastic_masked_sample"
    assert plan["evaluation"]["policy_route_selection"] == "sample"
    assert plan["evaluation"]["hdv_state_mode"] == "final_online_training_state"
    assert plan["evaluation"]["hdv_state_path"].endswith(
        "hdv_learning_state_after_online_wm_train.json"
    )
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
    updater = OnlineWorldModelUpdater(
        wm,
        OnlineWmDreamerConfig(
            wm_lr=1e-5,
            wm_active_count_readout_lr=1e-3,
        ),
        torch.device("cpu"),
    )

    assert all(param.requires_grad for param in wm.flow_projector.parameters())
    assert all(param.requires_grad for param in wm.dynamics.parameters())
    assert updater.active_count_readout_optimizer.param_groups[0]["lr"] == (
        pytest.approx(1e-3)
    )


def test_online_wm_batch_carries_completed_vehicle_episode_t_all():
    step = _wm_step(decision_idx=0, hdv_action=torch.tensor([0.25, 0.75]))
    step.episode_t_all_minutes = 4.75
    step.active_vehicle_seconds = 90.0
    step.transition_seconds = 30.0
    step.scheduled_transition_seconds = 60.0
    updater = object.__new__(OnlineWorldModelUpdater)
    updater.device = torch.device("cpu")

    batch = updater._batch_tensors([step])

    torch.testing.assert_close(batch["episode_t_all"], torch.tensor([4.75]))
    torch.testing.assert_close(
        batch["active_vehicle_seconds"],
        torch.tensor([90.0]),
    )
    torch.testing.assert_close(
        batch["transition_seconds"],
        torch.tensor([30.0]),
    )
    torch.testing.assert_close(
        batch["scheduled_transition_seconds"],
        torch.tensor([60.0]),
    )


def test_online_wm_causal_pair_aligns_current_action_to_next_target():
    current = torch.tensor([[1.0, 2.0, 3.0]])

    pair = _causal_pair(current)

    torch.testing.assert_close(pair[:, 0], torch.zeros_like(current))
    torch.testing.assert_close(pair[:, 1], current)


def _wm_step(
    *,
    decision_idx: int,
    hdv_action: torch.Tensor,
    hdv_static_edge_demand: torch.Tensor | None = None,
) -> OnlineWorldModelStep:
    zeros_flow = torch.zeros((2, 1), dtype=torch.float32).numpy()
    zeros_edge = torch.zeros_like(hdv_action).numpy()
    return OnlineWorldModelStep(
        episode=1,
        decision_idx=int(decision_idx),
        sim_step=int(decision_idx),
        flow=zeros_flow,
        next_flow=zeros_flow,
        av_action=zeros_edge,
        hdv_action=hdv_action.numpy(),
        depart=zeros_edge,
        context=torch.zeros((3,), dtype=torch.float32).numpy(),
        hdv_static_edge_demand=None
        if hdv_static_edge_demand is None
        else hdv_static_edge_demand.numpy(),
    )


def test_online_wm_replay_uses_static_hdv_edge_demand_when_configured():
    replay = OnlineWorldModelReplay(1, hdv_target_source="static_edge_demand")
    steps = [
        _wm_step(
            decision_idx=0,
            hdv_action=torch.tensor([1.0, 0.0, 0.0]),
            hdv_static_edge_demand=torch.tensor([0.0, 4.0, 0.0]),
        ),
        _wm_step(
            decision_idx=1,
            hdv_action=torch.tensor([0.0, 1.0, 0.0]),
            hdv_static_edge_demand=torch.tensor([0.0, 2.0, 0.0]),
        ),
    ]

    replay.append_episode(steps)

    for step in replay.all_steps():
        torch.testing.assert_close(torch.from_numpy(step.hdv_target), torch.tensor([0.0, 1.0, 0.0]))


def test_online_wm_replay_falls_back_to_current_occupancy_mean_without_static_demand():
    replay = OnlineWorldModelReplay(1, hdv_target_source="static_edge_demand")
    steps = [
        _wm_step(decision_idx=0, hdv_action=torch.tensor([1.0, 0.0])),
        _wm_step(decision_idx=1, hdv_action=torch.tensor([0.0, 3.0])),
    ]

    replay.append_episode(steps)

    for step in replay.all_steps():
        torch.testing.assert_close(torch.from_numpy(step.hdv_target), torch.tensor([0.25, 0.75]))


def test_online_wm_replay_stores_pre_episode_hdv_response_prediction():
    replay = OnlineWorldModelReplay(1, hdv_target_source="static_edge_demand")
    steps = [
        _wm_step(
            decision_idx=0,
            hdv_action=torch.tensor([1.0, 0.0, 0.0]),
            hdv_static_edge_demand=torch.tensor([0.0, 4.0, 0.0]),
        )
    ]

    replay.append_episode(
        steps,
        predicted_hdv_edge_demand=np.asarray([2.0, 0.0, 2.0], dtype=np.float32),
        prediction_source="fallback_prev_hdv",
    )

    expected = torch.tensor([0.5, 0.0, 0.5])
    for step in replay.all_steps():
        torch.testing.assert_close(torch.from_numpy(step.hdv_predicted_edge_demand), expected)
        assert step.hdv_prediction_source == "fallback_prev_hdv"
    assert len(replay.summaries) == 1
    torch.testing.assert_close(torch.from_numpy(replay.summaries[0].predicted_hdv_edge_demand), expected)


def test_causal_hdv_response_pairs_previous_episode_to_current_static_demand():
    cfg = OnlineWmDreamerConfig(
        train_episodes=10,
        hdv_response_predictor_enabled=True,
    )
    model = CausalHdvResponseModel(n_edges=2, cfg=cfg, device=torch.device("cpu"))
    summaries = [
        OnlineHdvEpisodeSummary(
            episode=1,
            av_edge_pressure=np.asarray([3.0, 1.0], dtype=np.float32),
            hdv_static_edge_demand=np.asarray([2.0, 0.0], dtype=np.float32),
            hdv_target=np.asarray([2.0, 0.0], dtype=np.float32),
        ),
        OnlineHdvEpisodeSummary(
            episode=2,
            av_edge_pressure=np.asarray([0.0, 1.0], dtype=np.float32),
            hdv_static_edge_demand=np.asarray([0.0, 5.0], dtype=np.float32),
            hdv_target=np.asarray([0.0, 5.0], dtype=np.float32),
        ),
    ]

    x, y = model._pairs_from_summaries(summaries)

    assert x is not None
    assert y is not None
    torch.testing.assert_close(x[0], torch.tensor([0.75, 0.25, 1.0, 0.0, 0.2]))
    torch.testing.assert_close(y[0], torch.tensor([0.0, 1.0]))


def test_residual_hdv_response_initialises_to_previous_episode_distribution():
    cfg = OnlineWmDreamerConfig(
        train_episodes=10,
        hdv_response_predictor_enabled=True,
        hdv_response_predictor_min_pairs=1,
        hdv_response_predictor_parameterization="residual_logit",
    )
    model = CausalHdvResponseModel(n_edges=3, cfg=cfg, device=torch.device("cpu"))
    model.trained_pairs = 1
    summaries = [
        OnlineHdvEpisodeSummary(
            episode=1,
            av_edge_pressure=np.asarray([3.0, 1.0, 0.0], dtype=np.float32),
            hdv_static_edge_demand=np.asarray([2.0, 1.0, 1.0], dtype=np.float32),
            hdv_target=np.asarray([2.0, 1.0, 1.0], dtype=np.float32),
        )
    ]

    pred, metrics = model.predict_for_next_episode(summaries, episode=2)

    assert pred is not None
    np.testing.assert_allclose(pred, np.asarray([0.5, 0.25, 0.25]), atol=1e-6)
    assert metrics["hdv_response_pred_source_code"] == 2.0


def test_causal_hdv_response_checkpoint_roundtrip_preserves_progress_denominator(tmp_path):
    cfg = OnlineWmDreamerConfig(
        train_episodes=10,
        hdv_response_predictor_enabled=True,
        hdv_response_predictor_progress_denominator=7,
        hdv_response_predictor_parameterization="residual_logit",
    )
    model = CausalHdvResponseModel(n_edges=2, cfg=cfg, device=torch.device("cpu"))
    model.trained_pairs = 3
    ckpt = model.save_checkpoint(tmp_path / "hdv_response.pt", metrics={"edge_tv": 0.1})

    loaded = CausalHdvResponseModel(
        n_edges=2,
        cfg=OnlineWmDreamerConfig(
            train_episodes=99,
            hdv_response_predictor_enabled=True,
            hdv_response_predictor_progress_denominator=99,
        ),
        device=torch.device("cpu"),
    )
    state = loaded.load_checkpoint(ckpt)

    assert state["metrics"]["edge_tv"] == 0.1
    assert loaded.trained_pairs == 3
    assert loaded.progress_denominator == 7
    assert loaded.parameterization == "residual_logit"


def test_legacy_hdv_response_checkpoint_keeps_absolute_logit_semantics(tmp_path):
    legacy_cfg = OnlineWmDreamerConfig(
        train_episodes=10,
        hdv_response_predictor_enabled=True,
        hdv_response_predictor_parameterization="absolute_logits",
    )
    legacy = CausalHdvResponseModel(n_edges=2, cfg=legacy_cfg, device=torch.device("cpu"))
    with torch.no_grad():
        for parameter in legacy.model.parameters():
            parameter.zero_()
    legacy_path = tmp_path / "legacy_hdv_response.pt"
    torch.save(
        {
            "format": "causal_hdv_response_predictor_v1",
            "model_state_dict": legacy.model.state_dict(),
            "trained_pairs": 4,
            "n_edges": 2,
            "hidden_dim": legacy.hidden_dim,
            "depth": legacy.depth,
        },
        legacy_path,
    )
    loaded = CausalHdvResponseModel(
        n_edges=2,
        cfg=OnlineWmDreamerConfig(
            train_episodes=10,
            hdv_response_predictor_enabled=True,
            hdv_response_predictor_parameterization="residual_logit",
        ),
        device=torch.device("cpu"),
    )

    loaded.load_checkpoint(legacy_path)

    assert loaded.parameterization == "absolute_logits"
    x = torch.tensor([[0.8, 0.2, 0.9, 0.1, 0.2]], dtype=torch.float32)
    torch.testing.assert_close(
        loaded.predict_distribution(x),
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
    )


def test_residual_hdv_response_checkpoint_roundtrip_preserves_prediction_scale(tmp_path):
    cfg = OnlineWmDreamerConfig(
        train_episodes=10,
        hdv_response_predictor_enabled=True,
        hdv_response_predictor_parameterization="residual_logit",
        hdv_response_predictor_residual_scale=0.25,
    )
    model = CausalHdvResponseModel(n_edges=2, cfg=cfg, device=torch.device("cpu"))
    ckpt = model.save_checkpoint(tmp_path / "residual_hdv_response.pt")
    loaded = CausalHdvResponseModel(
        n_edges=2,
        cfg=OnlineWmDreamerConfig(
            train_episodes=10,
            hdv_response_predictor_enabled=True,
            hdv_response_predictor_parameterization="absolute_logits",
        ),
        device=torch.device("cpu"),
    )

    state = loaded.load_checkpoint(ckpt)

    assert state["format"] == "causal_hdv_response_predictor_v2"
    assert loaded.parameterization == "residual_logit"
    assert loaded.residual_scale == pytest.approx(0.25)


def test_zero_scale_residual_checkpoint_can_continue_online_training():
    cfg = OnlineWmDreamerConfig(
        train_episodes=3,
        hdv_response_predictor_enabled=True,
        hdv_response_predictor_update_epochs=1,
        hdv_response_predictor_parameterization="residual_logit",
        hdv_response_predictor_residual_scale=0.0,
    )
    model = CausalHdvResponseModel(n_edges=2, cfg=cfg, device=torch.device("cpu"))
    summaries = [
        OnlineHdvEpisodeSummary(
            episode=1,
            av_edge_pressure=np.asarray([1.0, 0.0], dtype=np.float32),
            hdv_static_edge_demand=np.asarray([1.0, 0.0], dtype=np.float32),
            hdv_target=np.asarray([1.0, 0.0], dtype=np.float32),
        ),
        OnlineHdvEpisodeSummary(
            episode=2,
            av_edge_pressure=np.asarray([0.0, 1.0], dtype=np.float32),
            hdv_static_edge_demand=np.asarray([0.0, 1.0], dtype=np.float32),
            hdv_target=np.asarray([0.0, 1.0], dtype=np.float32),
        ),
    ]

    metrics = model.update(summaries)

    assert metrics["hdv_response_updates"] == 1.0
    assert np.isfinite(metrics["hdv_response_loss"])


def test_online_wm_resume_preserves_hdv_response_pairs_as_pretrained(tmp_path):
    cfg = OnlineWmDreamerConfig(
        resume_checkpoint=str(tmp_path / "checkpoint_ep00150.pt"),
        hdv_response_predictor_enabled=True,
    )
    actor = torch.nn.Linear(2, 2)
    response_model = CausalHdvResponseModel(
        n_edges=2,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    torch.save(
        {
            "episode": 150,
            "actor": actor.state_dict(),
            "hdv_response_predictor": response_model.model.state_dict(),
            "hdv_response_trained_pairs": 145,
        },
        cfg.resume_checkpoint,
    )
    trainer = object.__new__(OnlineWmDreamerTrainer)
    trainer.cfg = cfg
    trainer.device = torch.device("cpu")
    trainer.actor = torch.nn.Linear(2, 2)
    trainer.hdv_response_model = response_model

    resumed_episode = trainer._load_resume_checkpoint()

    assert resumed_episode == 150
    assert response_model.pretrained_pairs == 145
    assert response_model.trained_pairs == 145


def test_online_wm_resume_rejects_legacy_action_conditioned_global_critic(tmp_path):
    checkpoint = tmp_path / "legacy_global_critic.pt"
    actor = torch.nn.Linear(2, 2)
    torch.save(
        {
            "episode": 150,
            "actor": actor.state_dict(),
            "global_critic": {"legacy.weight": torch.ones(1)},
        },
        checkpoint,
    )
    trainer = object.__new__(OnlineWmDreamerTrainer)
    trainer.cfg = OnlineWmDreamerConfig(resume_checkpoint=str(checkpoint))
    trainer.device = torch.device("cpu")
    trainer.actor = torch.nn.Linear(2, 2)
    trainer.global_critic = torch.nn.Linear(2, 1)
    trainer.hdv_response_model = None

    with pytest.raises(RuntimeError, match="legacy action-conditioned global critic"):
        trainer._load_resume_checkpoint()


def test_online_wm_resume_resets_global_critic_when_reward_semantics_changed(
    tmp_path,
):
    checkpoint = tmp_path / "old_reward_global_critic.pt"
    actor = torch.nn.Linear(2, 2)
    saved_global_critic = torch.nn.Linear(2, 1)
    with torch.no_grad():
        saved_global_critic.weight.fill_(7.0)
        saved_global_critic.bias.fill_(7.0)
    torch.save(
        {
            "episode": 150,
            "actor": actor.state_dict(),
            "global_critic": saved_global_critic.state_dict(),
            "global_critic_semantics": GLOBAL_CRITIC_SEMANTICS,
            "global_reward_semantics": "t_all_potential_v1",
        },
        checkpoint,
    )
    trainer = object.__new__(OnlineWmDreamerTrainer)
    trainer.cfg = OnlineWmDreamerConfig(resume_checkpoint=str(checkpoint))
    trainer.device = torch.device("cpu")
    trainer.actor = torch.nn.Linear(2, 2)
    trainer.global_critic = torch.nn.Linear(2, 1)
    trainer.hdv_response_model = None
    with torch.no_grad():
        trainer.global_critic.weight.zero_()
        trainer.global_critic.bias.zero_()

    trainer._load_resume_checkpoint()

    torch.testing.assert_close(
        trainer.global_critic.weight,
        torch.zeros_like(trainer.global_critic.weight),
    )
    torch.testing.assert_close(
        trainer.global_critic.bias,
        torch.zeros_like(trainer.global_critic.bias),
    )


def test_online_wm_resume_restores_compatible_active_count_readout(tmp_path):
    checkpoint = tmp_path / "active_count_readout.pt"
    actor = torch.nn.Linear(2, 2)
    saved_readout = AggregateActiveVehicleReadout(
        d_spatial=3,
        hidden_dim=4,
    )
    with torch.no_grad():
        saved_readout.correction_head[-1].bias.fill_(1.25)
    torch.save(
        {
            "episode": 150,
            "actor": actor.state_dict(),
            "active_count_readout": saved_readout.state_dict(),
            "active_count_readout_semantics": (
                ACTIVE_COUNT_READOUT_SEMANTICS
            ),
        },
        checkpoint,
    )
    trainer = object.__new__(OnlineWmDreamerTrainer)
    trainer.cfg = OnlineWmDreamerConfig(resume_checkpoint=str(checkpoint))
    trainer.device = torch.device("cpu")
    trainer.actor = torch.nn.Linear(2, 2)
    trainer.hdv_response_model = None
    restored_readout = AggregateActiveVehicleReadout(
        d_spatial=3,
        hidden_dim=4,
    )
    trainer.wm_updater = type(
        "Updater",
        (),
        {"active_count_readout": restored_readout},
    )()

    resumed_episode = trainer._load_resume_checkpoint()

    assert resumed_episode == 150
    torch.testing.assert_close(
        restored_readout.correction_head[-1].bias,
        torch.tensor([1.25]),
    )


def test_global_critic_checkpoint_semantics_constant_is_state_value():
    assert GLOBAL_CRITIC_SEMANTICS == "state_value_v1"


def test_online_wm_collector_projects_vehicle_local_hdv_routes_to_static_edge_demand():
    class DummyRegistry:
        n_edges = 4
        _edge_to_idx = {"a": 0, "b": 1, "c": 2, "d": 3}

    collector = object.__new__(OnlineWmDreamerCollector)
    collector._edge_registry = DummyRegistry()
    collector._hdv_vehicles = {"h1", "h2"}
    collector._hdv_route_idx = {"h1": 1, "h2": 0}
    collector._veh_candidates = {
        "h1": [["a"], ["b", "c"]],
        "h2": [["c", "d"], ["a"]],
    }

    demand = collector._compute_hdv_static_edge_demand()

    torch.testing.assert_close(
        torch.from_numpy(demand),
        torch.tensor([0.0, 0.25, 0.5, 0.25]),
    )


def test_online_collector_reuses_injected_current_to_dest_path_cache_across_episode_reset():
    from dreamer4.urban_routing.vehicle_imagination import LazyVehiclePathCache

    class DummyRegistry:
        edge_ids = ["a", "b", "c", "d"]
        _edge_to_idx = {edge_id: idx for idx, edge_id in enumerate(edge_ids)}

    collector = object.__new__(OnlineWmDreamerCollector)
    collector._edge_registry = DummyRegistry()
    collector._av_candidate_adj_list = [[1, 2], [3], [3], []]
    collector._av_candidate_path_penalty_factor = 3.0
    collector._veh_dest = {"veh": "d"}
    collector._route_cache = {}
    collector.n_routes = 2

    shared_cache = LazyVehiclePathCache(
        adj_list=collector._av_candidate_adj_list,
        n_edges=4,
        K=2,
        penalty_factor=3.0,
    )
    compute_calls = 0
    original_compute = shared_cache._compute

    def counted_compute(src, dst, edge_tt):
        nonlocal compute_calls
        compute_calls += 1
        return original_compute(src, dst, edge_tt)

    shared_cache._compute = counted_compute
    collector._current_to_dest_path_cache = shared_cache
    edge_tt = np.asarray([1.0, 10.0, 2.0, 1.0], dtype=np.float32)

    shared_cache.begin_interval()
    first = collector._current_to_dest_route_candidates("veh", "a", edge_tt)
    cold_interval = shared_cache.interval_stats()
    collector._route_cache.clear()
    shared_cache.begin_interval()
    second = collector._current_to_dest_route_candidates("veh", "a", edge_tt)
    warm_interval = shared_cache.interval_stats()

    assert first == second
    assert compute_calls == 1
    assert len(shared_cache._cache) == 1
    assert shared_cache.stats() == {
        "lookups": 2.0,
        "hits": 1.0,
        "misses": 1.0,
        "entries": 1.0,
        "hit_rate": 0.5,
    }
    assert cold_interval == {
        "lookups": 1.0,
        "hits": 0.0,
        "misses": 1.0,
        "warm_hits": 0.0,
        "entries_before": 0.0,
        "entries_after": 1.0,
        "hit_rate": 0.0,
        "warm_hit_rate": 0.0,
    }
    assert warm_interval == {
        "lookups": 1.0,
        "hits": 1.0,
        "misses": 0.0,
        "warm_hits": 1.0,
        "entries_before": 1.0,
        "entries_after": 1.0,
        "hit_rate": 1.0,
        "warm_hit_rate": 1.0,
    }


def test_online_wm_trainer_shares_path_cache_only_for_static_freeflow_candidates():
    trainer = object.__new__(OnlineWmDreamerTrainer)
    shared_cache = object()
    trainer.imagined_ac = type("Imagined", (), {"_path_cache": shared_cache})()

    trainer.cfg = OnlineWmDreamerConfig(
        av_candidate_source="current_shortest_freeflow"
    )
    assert trainer._shared_current_to_dest_path_cache() is shared_cache

    trainer.cfg = OnlineWmDreamerConfig(av_candidate_source="current_shortest_live")
    assert trainer._shared_current_to_dest_path_cache() is None


def test_online_collector_rejects_injected_path_cache_for_dynamic_candidates():
    with pytest.raises(ValueError, match="only valid for static"):
        OnlineWmDreamerCollector(
            wm=object(),
            actor=object(),
            cfg=OnlineWmDreamerConfig(av_candidate_source="current_shortest_live"),
            current_to_dest_path_cache=object(),
        )


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


def test_imagined_action_samples_masked_policy_and_reuses_one_hot_for_wm():
    torch.manual_seed(0)
    logits = torch.tensor([[0.1, 3.0, 2.0]]).repeat(128, 1)
    valid = torch.tensor([[True, False, True]]).repeat(128, 1)

    selected, one_hot, log_prob, _entropy = _sample_masked_actions(
        logits,
        valid,
        temperature=1.0,
    )

    assert bool(valid.gather(1, selected.unsqueeze(1)).all())
    torch.testing.assert_close(one_hot.argmax(dim=-1), selected)
    torch.testing.assert_close(one_hot.sum(dim=-1), torch.ones(128))
    assert bool((selected == 0).any())
    assert bool((selected == 2).any())
    assert torch.isfinite(log_prob).all()


def test_sampled_route_aggregation_matches_generic_one_hot_aggregation():
    from dreamer4.urban_routing.dreamer_rollout import (
        aggregate_route_probs_to_action_edge,
    )

    paths = [
        [[0, 1, 2], [0, 3, 4]],
        [[5, 6, 7], [5, 8, 9]],
    ]
    selected = np.asarray([1, 0], dtype=np.int64)
    one_hot = torch.nn.functional.one_hot(
        torch.from_numpy(selected),
        num_classes=2,
    ).to(dtype=torch.float32)
    current = [0, 5]
    edge_tt = np.asarray([5.0] * 10, dtype=np.float32)

    expected = aggregate_route_probs_to_action_edge(
        one_hot,
        paths,
        n_edges=10,
        n_routes=2,
        current_edges=torch.tensor(current),
        edge_travel_times=torch.from_numpy(edge_tt),
        horizon_seconds=8.0,
    )
    actual = _aggregate_selected_routes_to_action_edge(
        selected,
        paths,
        current_edges=current,
        edge_travel_times=edge_tt,
        horizon_seconds=8.0,
        n_edges=10,
        n_routes=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(actual, expected)


def test_imagined_updates_batch_independent_rollouts_without_reducing_count():
    class DummyReplay:
        @staticmethod
        def all_steps():
            return [type("Step", (), {"samples": [object()]})()]

    imagined = object.__new__(OnlineImaginedActorCritic)
    imagined.actor = torch.nn.Linear(1, 1)
    imagined.critic = torch.nn.Linear(1, 1)
    imagined.global_critic = torch.nn.Linear(1, 1)
    imagined.device = torch.device("cpu")
    imagined.obs_dim = 1
    imagined.cfg = OnlineWmDreamerConfig(
        imagined_updates_per_episode=1,
        imagined_rollouts_per_update=8,
        imagined_rollout_batch_size=4,
    )
    batch_sizes = []

    def fake_imagine(starts):
        batch_sizes.append(len(starts))
        return [imagined._empty_imagined_rollout() for _ in starts]

    imagined.imagine_from_steps = fake_imagine

    metrics = imagined.update(DummyReplay(), np.random.default_rng(42))

    assert batch_sizes == [4, 4]
    assert sum(batch_sizes) == 8
    assert metrics["imag_skipped"] == 1.0


def test_world_model_batch_prediction_uses_each_rollout_latent_state():
    from dreamer4.urban_routing.urban_world_model import UrbanWorldModel

    class DummyWorldModel:
        device = torch.device("cpu")

        @staticmethod
        def _prepare_model_actions(**kwargs):
            return kwargs["action"]

        @staticmethod
        def _predict_one_step(past_packed, actions):
            return past_packed + actions[:, None, :, None]

    z = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ]
    )
    actions = torch.tensor([[0.5, 1.0], [2.0, 3.0]])

    predicted = UrbanWorldModel.predict_next_state_batch(
        DummyWorldModel(),
        z,
        actions,
    )

    expected = z + actions[:, :, None]
    torch.testing.assert_close(predicted, expected)


def test_global_step_advantage_is_broadcast_into_vehicle_actor_advantage():
    vehicle_advantage = torch.tensor([1.0, 2.0, 3.0, 4.0])
    global_advantage = torch.tensor([10.0, -2.0])
    group_index = torch.tensor([0, 0, 1, 1])

    combined = _combine_vehicle_global_advantages(
        vehicle_advantage,
        global_advantage,
        group_index,
        global_weight=0.5,
    )

    torch.testing.assert_close(combined, torch.tensor([6.0, 7.0, 2.0, 3.0]))


def test_global_network_reward_weights_edge_tt_by_all_vehicle_count():
    class DummyProjector:
        @staticmethod
        def decode(_z):
            return torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]])

    class DummyWm:
        flow_projector = DummyProjector()
        bpr = type("Bpr", (), {"capacities": torch.tensor([10.0, 10.0])})()

    reward = _global_network_reward_from_latent(
        DummyWm(),
        torch.zeros((2, 4)),
        np.asarray([2.0, 6.0], dtype=np.float32),
        reward_temperature=10.0,
    )

    torch.testing.assert_close(reward, torch.tensor(-0.5))


def test_global_active_vehicle_reward_sums_to_negative_mean_travel_time_minutes():
    active_counts = torch.tensor([3.0, 2.0, 1.0])

    rewards = _global_active_vehicle_reward(
        active_counts,
        total_vehicle_count=3,
        transition_seconds=60.0,
        reward_scale=1.0,
    )

    torch.testing.assert_close(
        rewards,
        torch.tensor([-1.0, -2.0 / 3.0, -1.0 / 3.0]),
    )
    torch.testing.assert_close(rewards.sum(), torch.tensor(-2.0))


def test_nominal_transition_count_preserves_truncated_vehicle_seconds():
    # Three vehicles active for a realized 30-second terminal transition:
    # 90 vehicle-seconds represented over a nominal 60-second imagined step.
    equivalent_count = _nominal_transition_active_vehicle_count(
        90.0,
        nominal_transition_seconds=60.0,
        total_vehicle_count=3,
    )
    reward = _global_active_vehicle_reward(
        equivalent_count,
        total_vehicle_count=3,
        transition_seconds=60.0,
        reward_scale=1.0,
    )

    torch.testing.assert_close(equivalent_count, torch.tensor(1.5))
    torch.testing.assert_close(reward, torch.tensor(-0.5))


def test_nominal_transition_count_does_not_hide_unrepresentable_exposure():
    equivalent_count = _nominal_transition_active_vehicle_count(
        600.0,
        nominal_transition_seconds=30.0,
        total_vehicle_count=10,
    )

    torch.testing.assert_close(equivalent_count, torch.tensor(20.0))


def test_decoded_active_vehicle_count_restores_capacity_scaled_total():
    class DummyProjector:
        @staticmethod
        def decode(_z):
            return torch.tensor(
                [[0.0, 1.0, 0.5, 0.0], [0.0, 1.0, 0.25, 0.0]],
            )

    class DummyWm:
        flow_projector = DummyProjector()
        bpr = type("Bpr", (), {"capacities": torch.tensor([10.0, 20.0])})()

    count = _decoded_active_vehicle_count(
        DummyWm(),
        torch.zeros((2, 3)),
        total_vehicle_count=20,
    )

    torch.testing.assert_close(count, torch.tensor(10.0))


def test_active_count_readout_starts_as_decoder_sum_residual_anchor():
    readout = AggregateActiveVehicleReadout(d_spatial=3, hidden_dim=4)
    z = torch.zeros((2, 3), dtype=torch.float32)

    predicted_fraction = readout(
        z,
        z_prev=z,
        raw_decoder_fraction=torch.tensor(0.25),
        transition_seconds=30.0,
    )

    torch.testing.assert_close(predicted_fraction, torch.tensor(0.25))


def test_active_count_readout_broadcasts_batch_over_singleton_time_axis():
    readout = AggregateActiveVehicleReadout(d_spatial=3, hidden_dim=4)
    z = torch.zeros((2, 1, 2, 3), dtype=torch.float32)

    predicted_fraction = readout(
        z,
        z_prev=z,
        raw_decoder_fraction=torch.tensor([0.25, 0.5]),
        transition_seconds=torch.tensor([30.0, 60.0]),
    )

    assert predicted_fraction.shape == (2, 1)
    torch.testing.assert_close(
        predicted_fraction[:, 0],
        torch.tensor([0.25, 0.5]),
    )


def test_active_count_readout_is_conditioned_on_both_transition_endpoints():
    readout = AggregateActiveVehicleReadout(d_spatial=2, hidden_dim=1)
    z_next = torch.zeros((2, 2), dtype=torch.float32)
    with torch.no_grad():
        readout.correction_head[0].weight.zero_()
        readout.correction_head[0].bias.zero_()
        readout.correction_head[0].weight[0, 0] = 1.0
        readout.correction_head[-1].weight.fill_(1.0)
        readout.correction_head[-1].bias.zero_()

    unchanged = readout(
        z_next,
        z_prev=torch.zeros_like(z_next),
        raw_decoder_fraction=0.5,
        transition_seconds=30.0,
    )
    changed = readout(
        z_next,
        z_prev=torch.ones_like(z_next),
        raw_decoder_fraction=0.5,
        transition_seconds=30.0,
    )

    assert float(changed.detach()) != pytest.approx(float(unchanged.detach()))


def test_active_count_readout_is_conditioned_on_scheduled_duration():
    readout = AggregateActiveVehicleReadout(d_spatial=2, hidden_dim=1)
    z = torch.zeros((2, 2), dtype=torch.float32)
    with torch.no_grad():
        readout.correction_head[0].weight.zero_()
        readout.correction_head[0].bias.zero_()
        readout.correction_head[0].weight[0, -1] = 1.0
        readout.correction_head[-1].weight.fill_(1.0)
        readout.correction_head[-1].bias.zero_()

    short = readout(
        z,
        z_prev=z,
        raw_decoder_fraction=0.5,
        transition_seconds=30.0,
    )
    long = readout(
        z,
        z_prev=z,
        raw_decoder_fraction=0.5,
        transition_seconds=300.0,
    )

    assert float(long.detach()) > float(short.detach())


def test_active_count_readout_can_correct_large_decoder_sum_error():
    readout = AggregateActiveVehicleReadout(d_spatial=3, hidden_dim=4)
    z = torch.zeros((2, 3), dtype=torch.float32)
    with torch.no_grad():
        readout.correction_head[-1].bias.fill_(
            torch.logit(torch.tensor(0.2))
            - torch.logit(torch.tensor(0.999))
        )

    predicted_fraction = readout(
        z,
        z_prev=z,
        raw_decoder_fraction=torch.tensor(4.0),
        transition_seconds=30.0,
    )

    torch.testing.assert_close(
        predicted_fraction,
        torch.tensor(0.2),
        atol=1e-5,
        rtol=1e-5,
    )


def test_active_count_readout_applies_real_transition_calibration_immediately():
    readout = AggregateActiveVehicleReadout(d_spatial=3, hidden_dim=4)
    z = torch.zeros((2, 3), dtype=torch.float32)

    readout.update_global_calibration(
        raw_decoder_fraction=torch.tensor([4.0]),
        target_fraction=torch.tensor([0.2]),
    )
    predicted_fraction = readout(
        z,
        z_prev=z,
        raw_decoder_fraction=torch.tensor(4.0),
        transition_seconds=30.0,
    )

    assert int(readout.calibration_updates) == 1
    torch.testing.assert_close(
        predicted_fraction,
        torch.tensor(0.2),
        atol=1e-5,
        rtol=1e-5,
    )


def test_active_vehicle_count_from_latent_uses_readout_not_raw_edge_sum():
    class DummyProjector:
        @staticmethod
        def decode(_z):
            return torch.tensor(
                [[0.0, 1.0, 2.0, 0.0], [0.0, 1.0, 2.0, 0.0]],
            )

    class DummyReadout(torch.nn.Module):
        def forward(
            self,
            _z,
            *,
            raw_decoder_fraction,
            z_prev,
            transition_seconds,
        ):
            assert float(raw_decoder_fraction) == pytest.approx(4.0)
            assert tuple(z_prev.shape) == (2, 3)
            assert float(transition_seconds) == pytest.approx(30.0)
            return torch.tensor(0.25)

    class DummyWm:
        flow_projector = DummyProjector()
        bpr = type("Bpr", (), {"capacities": torch.tensor([10.0, 10.0])})()

    predicted_count, raw_count = _active_vehicle_count_from_latent(
        DummyWm(),
        DummyReadout(),
        torch.zeros((2, 3)),
        z_prev=torch.ones((2, 3)),
        transition_seconds=30.0,
        total_vehicle_count=10,
    )

    torch.testing.assert_close(raw_count, torch.tensor(40.0))
    torch.testing.assert_close(predicted_count, torch.tensor(2.5))


def test_active_count_readout_supervision_uses_exact_sumo_fraction():
    class DummyProjector:
        @staticmethod
        def decode(_z):
            return torch.tensor(
                [
                    [[0.0, 1.0, 0.5, 0.0], [0.0, 1.0, 0.25, 0.0]],
                    [[0.0, 1.0, 0.2, 0.0], [0.0, 1.0, 0.1, 0.0]],
                ],
                requires_grad=True,
            )

    class SpyReadout(AggregateActiveVehicleReadout):
        def __init__(self):
            super().__init__(d_spatial=3, hidden_dim=4)
            self.seen_previous_latents = []

        def corrected_logits(self, z_next, **kwargs):
            self.seen_previous_latents.append(
                kwargs["z_prev"].detach().clone()
            )
            return super().corrected_logits(z_next, **kwargs)

    readout = SpyReadout()
    loss, diagnostics = _active_vehicle_count_readout_supervision_loss(
        z_current=torch.ones((2, 2, 3)),
        z_pred_next=torch.zeros((2, 2, 3)),
        z_encoded_next=torch.zeros((2, 2, 3)),
        flow_projector=DummyProjector(),
        active_count_readout=readout,
        capacities=torch.tensor([10.0, 20.0]),
        target_active_vehicle_count=torch.tensor([8.0, 5.0]),
        transition_seconds=torch.tensor([60.0, 60.0], dtype=torch.float32),
        raw_decoder_target_active_vehicle_count=torch.tensor([10.0, 4.0]),
        total_vehicle_count=torch.tensor([20.0, 20.0]),
        supervision_weight=1.0,
    )

    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
    torch.testing.assert_close(
        diagnostics["pred_next_active_vehicle_count_raw_sum_mae"],
        torch.tensor(0.0),
    )
    assert (
        float(
            diagnostics[
                "pred_next_active_vehicle_count_mae"
            ].detach()
        )
        < 1.5
    )
    assert len(readout.seen_previous_latents) == 2
    for z_previous in readout.seen_previous_latents:
        torch.testing.assert_close(
            z_previous,
            torch.ones((2, 2, 3)),
        )
    loss.backward()
    assert any(
        param.grad is not None and bool(torch.isfinite(param.grad).all())
        for param in readout.parameters()
    )


def test_active_count_readout_supervision_rejects_unrepresentable_fraction():
    class DummyProjector:
        @staticmethod
        def decode(z):
            decoded = torch.zeros((*z.shape[:-1], 4), dtype=z.dtype)
            decoded[..., 2] = 0.5
            return decoded

    readout = AggregateActiveVehicleReadout(d_spatial=3, hidden_dim=4)
    loss, diagnostics = _active_vehicle_count_readout_supervision_loss(
        z_current=torch.zeros((1, 2, 3)),
        z_pred_next=torch.zeros((1, 2, 3)),
        z_encoded_next=torch.zeros((1, 2, 3)),
        flow_projector=DummyProjector(),
        active_count_readout=readout,
        capacities=torch.tensor([10.0, 10.0]),
        target_active_vehicle_count=torch.tensor([12.0]),
        transition_seconds=torch.tensor([30.0]),
        raw_decoder_target_active_vehicle_count=torch.tensor([0.0]),
        total_vehicle_count=torch.tensor([10.0]),
        supervision_weight=1.0,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(
        diagnostics["pred_next_active_vehicle_count_valid_frac"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        diagnostics["pred_next_active_vehicle_count_unrepresentable_frac"],
        torch.tensor(1.0),
    )


def test_real_global_transition_part_uses_recorded_sumo_rewards():
    class IdentityProjector(torch.nn.Module):
        def forward(self, flow):
            return flow

    wm = type("Wm", (), {"flow_projector": IdentityProjector()})()
    first = _wm_step(decision_idx=0, hdv_action=torch.tensor([0.5, 0.5]))
    second = _wm_step(decision_idx=1, hdv_action=torch.tensor([0.5, 0.5]))
    first.flow = np.ones((2, 3), dtype=np.float32)
    first.next_flow = np.full((2, 3), 2.0, dtype=np.float32)
    first.next_context = np.asarray([0.1, 0.9, 0.8], dtype=np.float32)
    first.global_reward = -0.5
    first.transition_terminal = False
    second.flow = np.full((2, 3), 2.0, dtype=np.float32)
    second.next_flow = np.full((2, 3), 3.0, dtype=np.float32)
    second.next_context = np.asarray([0.2, 0.0, 0.0], dtype=np.float32)
    second.global_reward = -0.25
    second.transition_terminal = True

    part = _real_global_transition_part(wm, [first, second], torch.device("cpu"))

    torch.testing.assert_close(part["global_rewards"], torch.tensor([-0.5, -0.25]))
    torch.testing.assert_close(
        part["global_latent_states"],
        torch.stack([torch.from_numpy(first.flow), torch.from_numpy(second.flow)]),
    )
    torch.testing.assert_close(
        part["global_terminal"],
        torch.tensor([False, True]),
    )


def test_global_critic_is_state_only_when_building_lambda_advantage():
    class ConstantGlobalCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, z, action=None, context=None):
            assert action is None
            return self.value.expand(int(z.shape[0]), 1)

    imagined = object.__new__(OnlineImaginedActorCritic)
    imagined.global_critic = ConstantGlobalCritic()
    imagined.device = torch.device("cpu")
    imagined.cfg = OnlineWmDreamerConfig(
        gamma=0.0,
        ac_lambda=0.95,
        ac_global_advantage_weight=1.0,
    )
    part = {
        "global_latent_states": torch.zeros((2, 3, 4)),
        "global_next_latent_states": torch.zeros((2, 3, 4)),
        "global_contexts": torch.zeros((2, 3)),
        "global_rewards": torch.tensor([-1.0, -2.0]),
    }

    loss, advantage, stats = imagined._global_critic_terms([part])

    torch.testing.assert_close(loss, torch.tensor(1.25))
    torch.testing.assert_close(advantage, torch.tensor([-1.0, -2.0]))
    assert stats["imag_global_adv_mean"] == -1.5


def test_imagined_global_context_advances_time_and_active_cavs_only():
    initial = torch.tensor([0.25, 0.8, 0.6])

    current = _imagined_global_context(
        initial,
        horizon_idx=0,
        episode_steps=60,
        initial_active_av=10,
        active_av=8,
    )
    later = _imagined_global_context(
        initial,
        horizon_idx=6,
        episode_steps=60,
        initial_active_av=10,
        active_av=5,
    )

    torch.testing.assert_close(current, initial)
    torch.testing.assert_close(later, torch.tensor([0.35, 0.4, 0.6]))


def test_global_critic_bootstraps_with_explicit_next_context():
    class RecordingGlobalCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))
            self.context_calls = []

        def forward(self, z, action=None, context=None):
            assert action is None
            self.context_calls.append(context.detach().clone())
            return self.bias + context[:, :1]

    critic = RecordingGlobalCritic()
    imagined = object.__new__(OnlineImaginedActorCritic)
    imagined.global_critic = critic
    imagined.device = torch.device("cpu")
    imagined.cfg = OnlineWmDreamerConfig(
        gamma=0.9,
        ac_lambda=0.95,
        ac_global_advantage_weight=1.0,
    )
    current_context = torch.tensor(
        [[0.1, 1.0, 0.8], [0.2, 0.5, 0.8]],
    )
    next_context = torch.tensor(
        [[0.2, 0.5, 0.8], [0.3, 0.0, 0.8]],
    )
    part = {
        "global_latent_states": torch.zeros((2, 3, 4)),
        "global_next_latent_states": torch.zeros((2, 3, 4)),
        "global_contexts": current_context,
        "global_next_contexts": next_context,
        "global_rewards": torch.tensor([-1.0, -2.0]),
    }

    imagined._global_critic_terms([part])

    torch.testing.assert_close(critic.context_calls[0], current_context)
    torch.testing.assert_close(critic.context_calls[1], next_context)


def test_global_critic_zeroes_explicit_terminal_bootstrap():
    class ContextGlobalCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, z, action=None, context=None):
            assert action is None
            return self.bias + context[:, :1]

    imagined = object.__new__(OnlineImaginedActorCritic)
    imagined.global_critic = ContextGlobalCritic()
    imagined.device = torch.device("cpu")
    imagined.cfg = OnlineWmDreamerConfig(
        gamma=1.0,
        ac_lambda=1.0,
        ac_global_advantage_weight=1.0,
    )
    part = {
        "global_latent_states": torch.zeros((2, 3, 4)),
        "global_next_latent_states": torch.zeros((2, 3, 4)),
        "global_contexts": torch.zeros((2, 3)),
        "global_next_contexts": torch.tensor(
            [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ),
        "global_rewards": torch.tensor([-1.0, -2.0]),
        "global_terminal": torch.tensor([False, True]),
    }

    _loss, advantage, _stats = imagined._global_critic_terms(
        [part],
    )

    torch.testing.assert_close(advantage, torch.tensor([-3.0, -2.0]))


@pytest.mark.parametrize(
    ("scheduled_episode_terminal", "expected_terminal"),
    [(True, True), (False, False)],
)
def test_imagined_rollout_emits_only_known_scheduled_terminal(
    monkeypatch,
    scheduled_episode_terminal,
    expected_terminal,
):
    monkeypatch.setattr(
        online_wm_module,
        "_decoded_edge_state_numpy_from_z",
        lambda *_args, **_kwargs: (
            np.zeros((2, 4), dtype=np.float32),
            np.ones((2,), dtype=np.float32),
        ),
    )
    imagined = object.__new__(OnlineImaginedActorCritic)
    imagined.wm = object()
    imagined.device = torch.device("cpu")
    imagined.cfg = OnlineWmDreamerConfig(
        ac_global_advantage_weight=0.0,
    )
    start = _wm_step(
        decision_idx=59,
        hdv_action=torch.tensor([0.5, 0.5]),
    )
    start.scheduled_episode_terminal = scheduled_episode_terminal
    state = {
        "start": start,
        "samples": [],
        "vehicle_states": [],
        "z": torch.zeros((2, 3)),
        "initial_context": torch.tensor([0.9, 0.0, 0.2]),
        "initial_active_av": 0,
        "total_vehicle_count": 10,
        "next_step_cache": None,
        "done": False,
        "obs_steps": [],
        "action_steps": [],
        "old_logp_steps": [],
        "valid_steps": [],
        "reward_steps": [],
        "group_steps": [],
        "next_group_steps": [],
        "vehicle_id_steps": [],
        "next_obs_steps": [],
        "next_valid_steps": [],
        "entropies": [],
        "action_edges": [],
        "global_latent_steps": [],
        "global_next_latent_steps": [],
        "global_context_steps": [],
        "global_next_context_steps": [],
        "global_reward_steps": [],
        "global_active_vehicle_count_steps": [],
        "global_raw_decoder_vehicle_count_steps": [],
        "global_transition_seconds_steps": [],
        "global_terminal_steps": [],
    }
    prepared = {
        "state": state,
        "horizon_idx": 0,
        "active_indices": [],
        "obs": torch.zeros((1, 4)),
        "selected": torch.tensor([0]),
        "old_logp": torch.tensor([0.0]),
        "valid": torch.tensor([[True]]),
        "rewards": torch.tensor([0.0]),
        "action_edge": torch.zeros((2,)),
        "context": torch.tensor([0.9, 0.0, 0.2]),
        "entropy": torch.tensor(0.0),
        "transition_seconds": 30.0,
    }

    imagined._finish_imagined_horizon_step(
        prepared,
        torch.zeros((2, 3)),
    )

    assert bool(state["global_terminal_steps"][0]) is expected_terminal
    assert bool(state["done"]) is expected_terminal


def test_disabled_global_advantage_does_not_train_global_critic():
    class ExplodingGlobalCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, *_args, **_kwargs):
            raise AssertionError("disabled global critic must not be evaluated")

    imagined = object.__new__(OnlineImaginedActorCritic)
    imagined.global_critic = ExplodingGlobalCritic()
    imagined.device = torch.device("cpu")
    imagined.cfg = OnlineWmDreamerConfig(ac_global_advantage_weight=0.0)
    part = {"global_rewards": torch.tensor([1.0, 2.0])}

    loss, advantage, stats = imagined._global_critic_terms([part])

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(advantage, torch.zeros(2))
    assert stats["imag_global_enabled"] == 0.0


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


def test_imagined_freeflow_candidates_are_cached_across_dynamic_wm_states():
    from dreamer4.urban_routing.vehicle_imagination import LazyVehiclePathCache

    imagined = object.__new__(OnlineImaginedActorCritic)
    imagined.wm = type("Wm", (), {"n_edges": 4})()
    imagined.cfg = OnlineWmDreamerConfig(
        n_routes=2,
        av_candidate_path_penalty_factor=3.0,
    )
    imagined.adj_list = [[1, 2], [3], [3], []]
    imagined._candidate_edge_tt = np.ones((4,), dtype=np.float32)
    imagined._path_cache = LazyVehiclePathCache(
        adj_list=imagined.adj_list,
        n_edges=4,
        K=2,
        penalty_factor=3.0,
    )

    first = imagined._imagined_candidate_paths(
        edge_tt=np.asarray([100.0, 100.0, 1.0, 1.0], dtype=np.float32),
        current_edge=0,
        dest_edge=3,
        fallback_paths=[],
    )
    second = imagined._imagined_candidate_paths(
        edge_tt=np.asarray([1.0, 1.0, 100.0, 1.0], dtype=np.float32),
        current_edge=0,
        dest_edge=3,
        fallback_paths=[],
    )

    assert first == second
    assert len(imagined._path_cache._cache) == 1


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
    assert cmd[cmd.index("--av_policy_route_selection") + 1] == "sample"
    assert "--deterministic" in cmd
    assert str(cfg.projector_ckpt) not in {
        cmd[cmd.index("--projector_ckpt") + 1],
        cmd[cmd.index("--dynamics_ckpt") + 1],
    }


def test_online_wm_eval_command_keeps_argmax_as_explicit_ablation(tmp_path):
    cfg = OnlineWmDreamerConfig(
        output_dir=str(tmp_path),
        eval_python="python",
        eval_policy_route_selection="argmax",
    ).resolve()

    cmd = build_eval_command(
        cfg=cfg,
        checkpoint_path=tmp_path / "missing_checkpoint.pt",
        results_dir=tmp_path / "eval",
    )

    assert cmd[cmd.index("--av_policy_route_selection") + 1] == "argmax"


def test_online_wm_runner_evaluations_use_final_hdv_state(tmp_path, monkeypatch):
    from ablation import run_gretz_online_wm_dreamer_ablation as runner

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    initial_hdv_state = tmp_path / "initial_hdv_state.json"
    initial_hdv_state.write_text("{}", encoding="utf-8")
    final_hdv_state = output_dir / "hdv_learning_state_after_online_wm_train.json"
    final_hdv_state.write_text("{}", encoding="utf-8")
    checkpoint = output_dir / "checkpoints" / "checkpoint_ep00100.pt"
    cfg = OnlineWmDreamerConfig(
        output_dir=str(output_dir),
        eval_python="python",
        hdv_state_in=str(initial_hdv_state),
        checkpoint_episodes=(100,),
        eval_episodes=1,
    ).resolve()
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, check: calls.append((cmd, check)))

    outputs = runner.run_evaluations(
        cfg,
        {
            "checkpoint_paths": {100: str(checkpoint)},
            "final_hdv_state": str(final_hdv_state),
        },
    )

    assert len(calls) == 1
    cmd, check = calls[0]
    assert check is True
    assert cmd[cmd.index("--hdv_state_in") + 1] == str(final_hdv_state)
    assert cmd[cmd.index("--av_policy_route_selection") + 1] == "sample"
    assert str(initial_hdv_state) not in cmd
    assert outputs["100"]["hdv_state_in"] == str(final_hdv_state)


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
            "--sumo-extra-args",
            "--collision.action warn",
            "--imag-horizon",
            "8",
            "--imagined-rollout-batch-size",
            "4",
            "--global-advantage-weight",
            "0.75",
            "--global-reward-scale",
            "2.0",
            "--wm-active-count-readout-lr",
            "0.002",
            "--wm-t-all-supervision-weight",
            "0.15",
            "--real-policy-route-selection",
            "sample",
            "--eval-policy-route-selection",
            "sample",
            "--enable-hdv-response-predictor",
            "--imagined-hdv-context-source",
            "causal_response_predictor",
            "--hdv-response-predictor-ckpt",
            str(tmp_path / "hdv_response.pt"),
            "--hdv-response-predictor-progress-denominator",
            "50",
            "--enable-hdv-response-online-update",
            "--hdv-response-predictor-online-lr",
            "0.0002",
            "--hdv-response-predictor-online-update-epochs",
            "3",
            "--hdv-response-predictor-online-buffer-episodes",
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
    assert cfg.sumo_extra_args == "--collision.action warn"
    assert cfg.imag_horizon == 8
    assert cfg.imagined_rollout_batch_size == 4
    assert cfg.ac_global_advantage_weight == 0.75
    assert cfg.global_reward_scale == 2.0
    assert cfg.wm_active_count_readout_lr == pytest.approx(0.002)
    assert cfg.wm_t_all_supervision_weight == 0.15
    assert cfg.real_policy_route_selection == "sample"
    assert cfg.eval_policy_route_selection == "sample"
    assert cfg.reward_mode == "capped_step_cost"
    assert cfg.vehicle_arrival_head_enabled is False
    assert cfg.hdv_response_predictor_enabled is True
    assert cfg.imagined_hdv_context_source == "causal_response_predictor"
    assert cfg.hdv_response_predictor_ckpt == str(tmp_path / "hdv_response.pt")
    assert cfg.hdv_response_predictor_progress_denominator == 50
    assert cfg.hdv_response_predictor_online_update_enabled is True
    assert cfg.hdv_response_predictor_online_lr == 0.0002
    assert cfg.hdv_response_predictor_online_update_epochs == 3
    assert cfg.hdv_response_predictor_online_buffer_episodes == 8

    enabled_args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--allow-missing",
            "--enable-vehicle-arrival-head",
        ]
    )
    enabled_cfg = build_config(enabled_args)
    assert enabled_cfg.vehicle_arrival_head_enabled is True

    default_cfg = build_config(
        parse_args(
            [
                "--output-dir",
                str(tmp_path),
                "--dry-run",
                "--allow-missing",
            ]
        )
    )
    assert default_cfg.ac_global_advantage_weight == 1.0
    assert default_cfg.wm_active_count_supervision_weight == 1.0
    assert default_cfg.wm_active_count_readout_lr == pytest.approx(1e-3)
