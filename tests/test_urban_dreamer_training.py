from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
import torch

import urban_dreamer.learning.training as training_module
from urban_dreamer.workflow.config import (
    InputConfig,
    MainlineConfig,
    ResourceConfig,
    RunConfig,
    SimulationConfig,
)
from urban_dreamer.models.hdv import (
    HDV_RESPONSE_CHECKPOINT_FORMAT,
    PersistenceAnchoredOdRouteResidual,
)
from urban_dreamer.learning.replay import OnlineReplayTransition, VehicleSnapshot
from urban_dreamer.traffic.routing import PreparedInputs
from urban_dreamer.traffic.topology import EdgeNetwork, TrafficTopology, VehicleDemand
from urban_dreamer.learning.training import MainlineTrainer


INITIAL_ROUTE_STATE = {
    "version": 1,
    "route_policy": [
        {
            "origin": 0,
            "destination": 0,
            "probs": [0.25, 0.25, 0.25, 0.25],
            "costs": [20.0, 21.0, 22.0, 23.0],
        }
    ],
}


def _traffic_state(
    simulation_time: float,
    *,
    vehicles: Mapping[str, str],
) -> dict[str, Any]:
    edge_vehicle_counts = {
        edge_id: sum(current_edge == edge_id for current_edge in vehicles.values())
        for edge_id in ("a", "b", "c")
    }
    return {
        "simulation_time": float(simulation_time),
        "edges": {
            edge_id: {
                "occupancy": 0.1 if edge_vehicle_counts[edge_id] else 0.0,
                "mean_speed": 8.0 if edge_vehicle_counts[edge_id] else 10.0,
                "vehicle_number": float(edge_vehicle_counts[edge_id]),
                "halting_number": 0.0,
            }
            for edge_id in ("a", "b", "c")
        },
        "vehicles": {
            vehicle_id: {
                "edge": edge_id,
                "speed": 8.0,
                "waiting_time": 0.0,
            }
            for vehicle_id, edge_id in vehicles.items()
        },
    }


class _FakeSession:
    def __init__(self, route_state: Mapping[str, Any]) -> None:
        self._states = (
            _traffic_state(0.0, vehicles={"1": "a", "2": "a"}),
            _traffic_state(30.0, vehicles={"1": "b", "2": "b"}),
            _traffic_state(60.0, vehicles={}),
        )
        self._state_index = 0
        self._route_state = copy.deepcopy(dict(route_state))
        self.last_active_vehicle_exposure = 0.0
        self.route_assignment_calls: list[tuple[str, ...]] = []
        self.closed = False

    def reset(self) -> dict[str, Any]:
        self._state_index = 0
        self.last_active_vehicle_exposure = 0.0
        return copy.deepcopy(self._states[0])

    def get_traffic_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._states[self._state_index])

    def set_routes_safely(
        self,
        assignments: Mapping[str, Any],
    ) -> None:
        assert set(assignments) <= {"1"}
        self.route_assignment_calls.append(tuple(assignments))

    def advance(self, steps: int) -> None:
        assert steps == 30
        self._state_index = min(self._state_index + 1, 2)
        self.last_active_vehicle_exposure = (
            60.0 if self._state_index == 1 else 30.0
        )

    def is_done(self) -> bool:
        return self._state_index == 2

    def travel_times_seconds(self) -> dict[str, float]:
        return {"1": 60.0, "2": 60.0} if self.is_done() else {}

    def hdv_route_choices(self) -> dict[str, int]:
        return {"2": 1}

    def travel_time_route_choices(self) -> dict[str, int]:
        return {"1": 0, "2": 1}

    def hdv_learning_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._route_state)

    def update_hdv_learning(
        self,
        travel_times_seconds: Mapping[str, float] | None = None,
    ) -> int:
        assert travel_times_seconds == {"1": 60.0, "2": 60.0}
        self._route_state["route_policy"][0] = {
            "origin": 0,
            "destination": 0,
            "probs": [0.6, 0.3, 0.05, 0.05],
            "costs": [20.0, 25.0, 27.0, 28.0],
        }
        return 1

    def close(self) -> None:
        self.closed = True


class _IncompleteSession(_FakeSession):
    def is_done(self) -> bool:
        return False

    def travel_times_seconds(self) -> dict[str, float]:
        return {}


class _PartialOnlineSession(_IncompleteSession):
    def __init__(self, route_state: Mapping[str, Any]) -> None:
        super().__init__(route_state)
        self._states = (
            _traffic_state(0.0, vehicles={"1": "a", "2": "a"}),
            _traffic_state(30.0, vehicles={"1": "b", "2": "b"}),
            _traffic_state(60.0, vehicles={"2": "b"}),
        )
        self.hdv_update_inputs: list[dict[str, float]] = []

    def travel_times_seconds(self) -> dict[str, float]:
        return {"1": 60.0}

    def update_hdv_learning(
        self,
        travel_times_seconds: Mapping[str, float] | None = None,
    ) -> int:
        values = {
            str(vehicle_id): float(seconds)
            for vehicle_id, seconds in dict(travel_times_seconds or {}).items()
        }
        self.hdv_update_inputs.append(values)
        self._route_state["route_policy"][0] = {
            "origin": 0,
            "destination": 0,
            "probs": [0.6, 0.3, 0.05, 0.05],
            "costs": [20.0, 25.0, 27.0, 28.0],
        }
        return len(values)


def _topology() -> TrafficTopology:
    network = EdgeNetwork(
        edge_ids=("a", "b", "c"),
        edge_to_index={"a": 0, "b": 1, "c": 2},
        static_features=np.asarray(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5],
                [0.25, 0.25, 0.25, 0.25],
            ],
            dtype=np.float32,
        ),
        speed_limits_mps=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
        capacities=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
        lengths_m=np.asarray([100.0, 100.0, 100.0], dtype=np.float32),
        adjacency=np.asarray(
            [
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        adjacency_list=((1, 2), (), ()),
        passenger_allowed=np.asarray([True, True, True]),
        free_flow_travel_seconds=np.asarray(
            [10.0, 10.0, 10.0],
            dtype=np.float32,
        ),
    )
    return TrafficTopology(
        network=network,
        origins=("a",),
        destinations=("b",),
        vehicles=(
            VehicleDemand("1", 0, 0, 0, True),
            VehicleDemand("2", 0, 0, 0, False),
        ),
        od_pairs=((0, 0),),
        route_paths=(
            (
                (0, 1),
                (0, 2),
                (0, 1),
                (0, 2),
            ),
        ),
        od_automated_counts=np.asarray([1.0], dtype=np.float32),
        od_human_counts=np.asarray([1.0], dtype=np.float32),
    )


def _write_predictor(
    path: Path,
    topology: TrafficTopology,
    initial_hdv_state: Path,
) -> None:
    model = PersistenceAnchoredOdRouteResidual()
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
            "selected_residual_scale": 4.0,
            "anchor_floor": 1e-3,
            "n_routes": 4,
            "ods": [list(od_pair) for od_pair in topology.od_pairs],
            "edge_ids": list(topology.network.edge_ids),
            "hidden_dim": 128,
            "depth": 2,
            "progress_denominator": 450,
            "trained_pairs": 3,
            "model_state_dict": model.state_dict(),
            "initial_hdv_state": str(initial_hdv_state),
        },
        path,
    )


def _config(tmp_path: Path) -> MainlineConfig:
    return MainlineConfig(
        inputs=InputConfig(
            network_name="toy",
            network_dir=tmp_path,
            route_artifacts_dir=tmp_path,
            post_mutation_state=tmp_path / "post.json",
            initial_hdv_state=tmp_path / "hdv.json",
            hdv_response_checkpoint=tmp_path / "predictor.pt",
        ),
        run=RunConfig(
            output_dir=tmp_path / "output",
            device="cpu",
            seed=7,
            warmup_episodes=1,
            world_model_initialization_steps=1,
            online_training_episodes=1,
            evaluation_episodes=1,
        ),
        simulation=SimulationConfig(
            decision_intervals_per_episode=2,
            warmup_horizon_seconds=60,
            warmup_completion_drain_seconds=0,
            warmup_completion_patience_seconds=0,
            episode_wall_timeout_seconds=120.0,
            training_completion_buffer_seconds=0,
            evaluation_horizon_seconds=60,
            incident_edge="a",
            incident_start_seconds=0.0,
            incident_end_seconds=1.0,
            incident_speed_mps=1.0,
            sumo_extra_args=(),
        ),
        resources=ResourceConfig(
            world_model_initialization_batch_size=1,
            world_model_initialization_sequence_length=2,
            world_model_gradient_accumulation_steps=1,
            online_world_model_batch_size=1,
        ),
        semantic_fingerprint="fingerprint",
        source_path=tmp_path / "mainline.toml",
    )


def _prepared(tmp_path: Path) -> PreparedInputs:
    return PreparedInputs(
        network_xml=tmp_path / "toy.net.xml",
        network_agents_csv=tmp_path / "agents.csv",
        od_file=tmp_path / "od_toy.txt",
        edge_registry=tmp_path / "edge_registry.json",
        route_agents_csv=tmp_path / "agents.csv",
        routes_csv=tmp_path / "routes.csv",
        route_xml=tmp_path / "route.rou.xml",
        post_mutation_state=tmp_path / "post.json",
        initial_hdv_state=tmp_path / "hdv.json",
        hdv_response_checkpoint=tmp_path / "predictor.pt",
        automated_vehicle_count=1,
        human_vehicle_count=1,
        automated_vehicle_fraction=0.5,
        route_count=4,
        mutation_episode_index=10,
        input_hashes={},
    )


def _build_trainer(tmp_path: Path) -> MainlineTrainer:
    topology = _topology()
    config = _config(tmp_path)
    config.inputs.initial_hdv_state.write_text(
        json.dumps(INITIAL_ROUTE_STATE),
        encoding="utf-8",
    )
    _write_predictor(
        config.inputs.hdv_response_checkpoint,
        topology,
        config.inputs.initial_hdv_state,
    )

    def session_factory(**kwargs: Any) -> _FakeSession:
        return _FakeSession(kwargs["initial_hdv_learning_state"])

    return MainlineTrainer(
        config=config,
        prepared_inputs=_prepared(tmp_path),
        topology=topology,
        _session_factory=session_factory,
    )


def _online_transition_with_vehicles(
    *,
    vehicle_ids: tuple[str, ...],
    destination_edges: tuple[int, ...],
) -> OnlineReplayTransition:
    vehicle_count = len(vehicle_ids)
    return OnlineReplayTransition(
        episode=0,
        step_index=0,
        flow=torch.ones((3, 8), dtype=torch.float32),
        action=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        next_flow=torch.full((3, 8), 2.0, dtype=torch.float32),
        departure=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        context=torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32),
        hdv_context=torch.full((3,), 1.0 / 3.0, dtype=torch.float32),
        hdv_target=torch.full((3,), 1.0 / 3.0, dtype=torch.float32),
        vehicles=VehicleSnapshot(
            vehicle_ids=vehicle_ids,
            current_edges=torch.zeros(vehicle_count, dtype=torch.int64),
            destination_edges=torch.tensor(
                destination_edges,
                dtype=torch.int64,
            ),
            departure_steps=torch.zeros(vehicle_count, dtype=torch.int64),
        ),
        active_vehicle_seconds=30.0 * vehicle_count,
        transition_seconds=30.0,
        terminal=True,
    )


def test_fixed_online_schedule_constants() -> None:
    assert training_module.ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE == 4
    assert training_module.IMAGINED_PASSES_PER_EPISODE == 16
    assert training_module.IMAGINED_ROLLOUTS_PER_PASS == 8
    assert training_module.IMAGINED_HORIZON == 8
    assert training_module.IMAGINATION_BATCH_SIZE == 4


def test_imagination_selects_each_step_as_one_vehicle_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _build_trainer(tmp_path)
    trainer.total_automated = 3
    monkeypatch.setattr(training_module, "IMAGINED_HORIZON", 1)
    transition = _online_transition_with_vehicles(
        vehicle_ids=("av-1", "av-2", "av-3"),
        destination_edges=(1, 1, 1),
    )
    assert trainer.online_replay.append_episode(
        [transition],
        total_travel_minutes=3.0,
    )
    batch = trainer.online_replay.sample_batch(
        batch_size=1,
        rng=np.random.default_rng(3),
    )
    sampled_observations: list[torch.Tensor] = []
    original_sample_action = trainer.actor.sample_action

    def record_sample_action(
        observations: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> Any:
        sampled_observations.append(observations.detach().cpu().clone())
        return original_sample_action(
            observations,
            deterministic=deterministic,
        )

    monkeypatch.setattr(trainer.actor, "sample_action", record_sample_action)
    torch.manual_seed(11)
    rollout_rows, _ = trainer._imagine_from_seed(
        batch,
        0,
        rollout_index=0,
    )

    assert len(rollout_rows) == 3
    assert len(sampled_observations) == 1
    assert sampled_observations[0].shape == (3, 33)
    torch.testing.assert_close(
        sampled_observations[0][:, 22:30],
        torch.zeros((3, 8)),
    )


def test_imagination_keeps_rows_aligned_across_horizons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _build_trainer(tmp_path)
    trainer.total_automated = 2
    trainer._free_flow_seconds = np.asarray(
        [0.1, 0.1, 1000.0],
        dtype=np.float32,
    )
    monkeypatch.setattr(training_module, "IMAGINED_HORIZON", 2)
    transition = _online_transition_with_vehicles(
        vehicle_ids=("fast", "slow"),
        destination_edges=(1, 2),
    )
    assert trainer.online_replay.append_episode(
        [transition],
        total_travel_minutes=3.0,
    )
    batch = trainer.online_replay.sample_batch(
        batch_size=1,
        rng=np.random.default_rng(3),
    )
    sampled_observations: list[torch.Tensor] = []
    original_sample_action = trainer.actor.sample_action

    def record_sample_action(
        observations: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> Any:
        sampled_observations.append(observations.detach().cpu().clone())
        return original_sample_action(
            observations,
            deterministic=deterministic,
        )

    monkeypatch.setattr(trainer.actor, "sample_action", record_sample_action)
    torch.manual_seed(11)
    rollout_rows, _ = trainer._imagine_from_seed(
        batch,
        0,
        rollout_index=0,
    )

    assert [tuple(obs.shape) for obs in sampled_observations] == [
        (2, 33),
        (1, 33),
    ]
    assert [
        row["trajectory_vehicle_key"]
        for row in rollout_rows
    ] == [0, 1, 1]
    assert [
        row["next_valid"]
        for row in rollout_rows
    ] == [False, True, True]
    torch.testing.assert_close(
        rollout_rows[1]["next_obs"].detach().cpu(),
        sampled_observations[1][0],
    )


@pytest.mark.parametrize(
    "runtime_edge",
    (":junction_0_0", "", "ghost_edge"),
)
def test_vehicle_snapshot_ignores_unmodeled_runtime_edges(
    tmp_path: Path,
    runtime_edge: str,
) -> None:
    trainer = _build_trainer(tmp_path)

    snapshot = trainer._build_vehicle_snapshot(
        state=_traffic_state(
            30.0,
            vehicles={"1": runtime_edge},
        )
    )

    assert snapshot.vehicle_ids == ()
    assert snapshot.current_edges.numel() == 0
    assert snapshot.destination_edges.numel() == 0
    assert snapshot.departure_steps.numel() == 0


def test_online_episode_uses_fixed_update_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _build_trainer(tmp_path)
    transition = OnlineReplayTransition(
        episode=0,
        step_index=0,
        flow=torch.ones((3, 8), dtype=torch.float32),
        action=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        next_flow=torch.full((3, 8), 2.0, dtype=torch.float32),
        departure=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        context=torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32),
        hdv_context=torch.full((3,), 1.0 / 3.0, dtype=torch.float32),
        hdv_target=torch.full((3,), 1.0 / 3.0, dtype=torch.float32),
        vehicles=VehicleSnapshot(
            vehicle_ids=("1",),
            current_edges=torch.tensor([0], dtype=torch.int64),
            destination_edges=torch.tensor([1], dtype=torch.int64),
            departure_steps=torch.tensor([0], dtype=torch.int64),
        ),
        active_vehicle_seconds=30.0,
        transition_seconds=30.0,
        terminal=True,
    )
    collected = training_module._CollectedEpisode(
        episode=0,
        global_episode=0,
        total_travel_minutes=2.0,
        warmup_records=(),
        online_transitions=(transition,),
        hdv_observation=None,
        metrics={"online_steps": 1.0},
        final_hdv_state=copy.deepcopy(INITIAL_ROUTE_STATE),
    )
    world_model_updates: list[int] = []
    imagination_passes: list[int] = []

    monkeypatch.setattr(
        trainer,
        "_run_episode",
        lambda *, mode, episode_index: collected,
    )
    monkeypatch.setattr(
        trainer,
        "_online_world_model_update",
        lambda: world_model_updates.append(1) or {"wm_update": float(len(world_model_updates))},
    )
    monkeypatch.setattr(
        trainer,
        "_train_imagination_pass",
        lambda: imagination_passes.append(1) or {"imagined_steps": float(len(imagination_passes))},
    )

    result = trainer.train_online_episode(0)

    assert len(world_model_updates) == training_module.ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE
    assert len(imagination_passes) == training_module.IMAGINED_PASSES_PER_EPISODE
    assert result["complete"] is True
    assert result["online_transitions"] == 1
    assert result["wm_update"] == 2.5
    assert result["imagined_steps"] == 8.5


def test_mainline_trainer_runs_all_phases_and_strictly_restores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_module, "IMAGINED_PASSES_PER_EPISODE", 1)
    monkeypatch.setattr(training_module, "IMAGINED_ROLLOUTS_PER_PASS", 1)
    monkeypatch.setattr(training_module, "IMAGINED_HORIZON", 1)
    monkeypatch.setattr(training_module, "IMAGINATION_BATCH_SIZE", 1)
    trainer = _build_trainer(tmp_path)

    warmup = trainer.collect_warmup_episode(0)
    initialized = trainer.initialize_world_model_step(0)
    online = trainer.train_online_episode(0)
    evaluation = trainer.evaluate_episode(0)
    second_evaluation = trainer.evaluate_episode(1)
    saved_state = trainer.state_dict()

    restored = _build_trainer(tmp_path)
    restored.load_state_dict(saved_state)

    assert warmup["complete"] is True
    assert initialized["loss"] >= 0.0
    assert online["complete"] is True
    assert evaluation["complete"] is True
    assert second_evaluation["complete"] is True
    assert trainer.summary()["evaluation_episodes_completed"] == 2.0
    assert trainer.summary()["evaluation_t_all_mean"] == pytest.approx(
        (evaluation["t_all"] + second_evaluation["t_all"]) / 2.0
    )
    assert len(restored.warmup_replay.episodes) == 1
    assert len(restored.online_replay.episodes) == 1
    assert restored.summary() == trainer.summary()
    assert restored.state_dict()["caches"] == saved_state["caches"]

    invalid_state = copy.deepcopy(saved_state)
    invalid_state["legacy"] = {}
    with pytest.raises(ValueError, match="invalid schema"):
        restored.load_state_dict(invalid_state)


def test_world_model_initialization_uses_only_remaining_microbatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _build_trainer(tmp_path)
    trainer.config = replace(
        trainer.config,
        run=replace(
            trainer.config.run,
            world_model_initialization_steps=3,
        ),
        resources=replace(
            trainer.config.resources,
            world_model_gradient_accumulation_steps=2,
        ),
    )
    trainer.collect_warmup_episode(0)
    sample_batch = trainer.warmup_replay.sample_batch
    sampled_batch_sizes: list[int] = []

    def record_sample_batch(**kwargs: Any) -> Any:
        sampled_batch_sizes.append(int(kwargs["batch_size"]))
        return sample_batch(**kwargs)

    monkeypatch.setattr(
        trainer.warmup_replay,
        "sample_batch",
        record_sample_batch,
    )

    trainer.initialize_world_model_step(0)
    trainer.initialize_world_model_step(1)

    assert sampled_batch_sizes == [1, 1, 1]


def test_warmup_assigns_each_vehicle_route_only_once(
    tmp_path: Path,
) -> None:
    trainer = _build_trainer(tmp_path)
    session = _FakeSession(INITIAL_ROUTE_STATE)
    trainer._session_factory = lambda **_kwargs: session

    trainer.collect_warmup_episode(0)

    assert session.route_assignment_calls == [("1",)]


def test_warmup_discards_incomplete_attempt_and_retries_next_environment_episode(
    tmp_path: Path,
) -> None:
    trainer = _build_trainer(tmp_path)
    incomplete = _IncompleteSession(INITIAL_ROUTE_STATE)
    complete = _FakeSession(INITIAL_ROUTE_STATE)
    sessions = iter((incomplete, complete))
    environment_episodes: list[int] = []

    def session_factory(**kwargs: Any) -> _FakeSession:
        environment_episodes.append(int(kwargs["episode_counter"]))
        return next(sessions)

    trainer._session_factory = session_factory

    result = trainer.collect_warmup_episode(0)

    assert result["complete"] is True
    assert environment_episodes == [10, 11]
    assert incomplete.closed is True
    assert complete.closed is True
    assert len(trainer.warmup_replay.episodes) == 1
    assert trainer.summary()["warmup_episodes_completed"] == 1.0
    assert trainer.summary()["warmup_attempts_discarded"] == 1.0
    assert trainer.summary()["next_environment_episode"] == 12


def test_online_trains_on_partial_trajectory_when_completion_budget_expires(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _build_trainer(tmp_path)
    session = _PartialOnlineSession(INITIAL_ROUTE_STATE)
    trainer._session_factory = lambda **_kwargs: session
    world_model_updates: list[int] = []
    imagination_passes: list[int] = []
    monkeypatch.setattr(
        trainer,
        "_online_world_model_update",
        lambda: world_model_updates.append(1)
        or {"wm_update": float(len(world_model_updates))},
    )
    monkeypatch.setattr(
        trainer,
        "_train_imagination_pass",
        lambda: imagination_passes.append(1)
        or {"imagined_steps": float(len(imagination_passes))},
    )

    result = trainer.train_online_episode(0)

    assert result["complete"] is True
    assert result["environment_complete"] is False
    assert result["online_transitions"] == 2
    assert result["online_completion_rate"] == pytest.approx(0.5)
    assert result["t_all"] == pytest.approx(1.0)
    assert session.hdv_update_inputs == [{"1": 60.0}]
    assert len(trainer.online_replay.episodes) == 1
    assert [
        transition.terminal
        for transition in trainer.online_replay.episodes[0]
    ] == [False, False]
    assert len(world_model_updates) == (
        training_module.ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE
    )
    assert len(imagination_passes) == training_module.IMAGINED_PASSES_PER_EPISODE
    assert trainer.summary()["online_episodes_completed"] == 1.0


def test_evaluation_remains_strict_when_completion_budget_expires(
    tmp_path: Path,
) -> None:
    trainer = _build_trainer(tmp_path)
    trainer._session_factory = lambda **_kwargs: _IncompleteSession(
        INITIAL_ROUTE_STATE
    )

    with pytest.raises(
        RuntimeError,
        match="evaluation episode exceeded its fixed completion budget",
    ):
        trainer.evaluate_episode(0)
