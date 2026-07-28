import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.mamba import MambaConfig, MambaLearner  # noqa: E402
from baseline.online_actor_critic import OnlineDecisionSample  # noqa: E402
from baseline.run_toy_cross_bottleneck_mamba import (  # noqa: E402
    SparseDecisionBoundary,
    ToyMambaConfig,
    ToyMambaCollector,
    ToyMambaTrainer,
    _normalized_action_result,
    _pad_route_candidates,
    build_config,
    build_plan,
    fixed_role_hash,
    main,
    mamba_transition_diagnostics,
    parse_args,
)
from dreamer4.urban_routing.vehicle_imagination import (  # noqa: E402
    build_vehicle_route_cost_obs_features,
)


def _sample(
    veh_id: str,
    *,
    obs_dim: int = 5,
    episode: int = 1,
    decision_idx: int = 0,
    action: int = 1,
) -> OnlineDecisionSample:
    return OnlineDecisionSample(
        episode=episode,
        decision_idx=decision_idx,
        sim_step=decision_idx * 30,
        veh_id=veh_id,
        obs=np.arange(obs_dim, dtype=np.float32),
        action=action,
        old_log_prob=-math.log(4.0),
        reward=0.25,
        valid_mask=np.ones((4,), dtype=bool),
        route_costs=np.arange(4, dtype=np.float32),
        elapsed_time=3.0,
    )


class _FakeLearner:
    def __init__(self):
        self.reset_calls = 0
        self.begin_calls = []
        self.frames = []
        self.boundaries = []
        self.replay = []
        self.update_calls = 0

    def reset_episode(self):
        self.reset_calls += 1

    def begin_decision(self, slots):
        self.begin_calls.append(list(slots))

    def end_decision(self, frame):
        self.frames.append(frame)

    def end_episode(self, frame=None):
        self.boundaries.append(frame)
        self.replay.append(frame)

    def update(self):
        self.update_calls += 1
        return {"model_loss": 1.0 / self.update_calls, "skipped": 0.0}

    def checkpoint_state(self, **metadata):
        return {
            "schema_version": 1,
            "model_family": "mamba",
            "episode": metadata["episode"],
            "learner": {"fake": True},
        }

    def load_checkpoint_state(self, state):
        self.loaded_checkpoint = state
        return int(state["episode"])


def test_sparse_boundary_uses_fixed_slots_and_one_joint_begin_per_decision():
    learner = _FakeLearner()
    boundary = SparseDecisionBoundary(
        learner=learner,
        machine_ids=["veh-8", "veh-2", "veh-5"],
        obs_dim=5,
        n_routes=4,
    )

    boundary.reset_episode()
    boundary.begin_decision(
        episode=4,
        decision_idx=0,
        sim_step=100,
        presence_vehicle_ids=["veh-2", "veh-8"],
    )
    frame0 = boundary.end_decision([_sample("veh-8", episode=4)])
    boundary.begin_decision(
        episode=4,
        decision_idx=1,
        sim_step=130,
        presence_vehicle_ids=["veh-2", "veh-5"],
    )
    frame1 = boundary.end_decision([])
    final = boundary.end_episode(
        episode=4,
        decision_idx=2,
        sim_step=160,
        final_presence_vehicle_ids=["veh-5"],
        truncated=True,
    )

    assert learner.reset_calls == 1
    assert learner.begin_calls == [[0, 1], [1, 2]]
    assert len(learner.frames) == 2
    assert frame0["presence_slots"] == [0, 1]
    assert frame0["obs_slots"] == [0]
    assert frame0["decision_slots"] == [0]
    assert frame0["obs"].shape == (1, 5)
    assert frame1["presence_slots"] == [1, 2]
    assert frame1["terminated_slots"] == [0]
    assert frame1["obs"].shape == (0, 5)
    assert frame1["action_valid"].shape == (0, 4)
    assert final["presence_slots"] == [2]
    assert final["terminated_slots"] == [1]
    assert final["truncated"] is True


def test_sparse_boundary_does_not_mark_never_departed_agents_terminal():
    learner = _FakeLearner()
    boundary = SparseDecisionBoundary(
        learner=learner,
        machine_ids=["a", "b", "c"],
        obs_dim=3,
        n_routes=4,
    )
    boundary.reset_episode()
    boundary.begin_decision(
        episode=1,
        decision_idx=0,
        sim_step=0,
        presence_vehicle_ids=["b"],
    )
    frame = boundary.end_decision([])

    assert frame["presence_slots"] == [1]
    assert frame["terminated_slots"] == []


def test_action_result_preserves_learner_action_log_prob_and_masked_probs():
    action, log_prob, probs = _normalized_action_result(
        {
            "action": 2,
            "log_prob": -0.3,
            "logits": np.asarray([-1.0, 100.0, 2.0, 0.5], dtype=np.float32),
        },
        valid_mask=np.asarray([True, False, True, True]),
        n_routes=4,
    )

    assert action == 2
    assert log_prob == pytest.approx(-0.3)
    assert probs[1] == 0.0
    assert probs.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="invalid route"):
        _normalized_action_result(
            {"action": 1, "log_prob": 0.0},
            valid_mask=np.asarray([True, False, True, True]),
            n_routes=4,
        )


def test_padded_duplicate_routes_are_not_advertised_as_valid_actions():
    indexed, named, valid = _pad_route_candidates(
        [[1, 2], [1, 3], [1, 2]],
        [["a", "b"], ["a", "c"], ["a", "b"]],
        n_routes=4,
    )

    assert len(indexed) == len(named) == 4
    assert indexed[2:] == [[], []]
    assert named[2:] == [["a", "b"], ["a", "b"]]
    assert valid.tolist() == [True, True, False, False]

    cost_features = build_vehicle_route_cost_obs_features(
        [indexed],
        edge_cost=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        n_routes=4,
    )
    # The final K features are the observation's valid-route flags.  They must
    # agree exactly with the execution/replay mask for padded aliases.
    assert cost_features[0, -4:].astype(bool).tolist() == valid.tolist()


def test_mamba_collector_keeps_episode_samples_without_unbounded_run_history():
    class Boundary:
        slot_by_vehicle = {"veh": 0}

        def __init__(self):
            self.frames = []

        def begin_decision(self, **_kwargs):
            return None

        def end_decision(self, samples):
            self.frames.append(list(samples))

    boundary = Boundary()
    collector = SimpleNamespace(
        _av_decision_skip_stats={},
        _av_vehicles={"veh"},
        _mamba_boundary=boundary,
        _current_episode_id_eval=1,
        _current_decision_idx_eval=0,
        connector=SimpleNamespace(current_step=0),
        _edge_registry=SimpleNamespace(
            _edge_to_idx={"e0": 0, "e1": 1}, edge_ids=["e0", "e1"],
        ),
        _av_candidate_source="current_shortest_freeflow",
        n_routes=2,
        _vehicle_obs_route_stats="mean",
        _vehicle_obs_include_route_costs=True,
        _vehicle_obs_include_fleet_pressure=False,
        _cfg=SimpleNamespace(
            include_global=False,
            include_route_costs=True,
            real_policy_route_selection="sample",
            episode_steps=60,
            route_cost_length_norm="sum",
            reward_mode="capped_step_cost",
            reward_weight=1.0,
            reward_scale=1.0,
            step_interval=30.0,
            progress_cap=1.0,
            remaining_tt_eps=1.0,
            reward_temperature=60.0,
        ),
        _mamba=SimpleNamespace(
            act=lambda **_kwargs: {
                "action": 0,
                "log_prob": 0.0,
                "probs": np.asarray([1.0, 0.0], dtype=np.float32),
            },
        ),
        _av_route_idx={},
        _veh_dest={"veh": "e1"},
        _episode_samples=[],
        _all_training_samples=[],
        _dynamic_decision_rows=[],
        _dynamic_agent_decision_rows=[],
    )
    collector._live_edge_features = lambda _state: np.zeros((2, 1), dtype=np.float32)
    collector._live_current_edge_tt = lambda _state: np.asarray([5.0, 5.0], dtype=np.float32)
    collector._freeflow_edge_tt = lambda: np.asarray([5.0, 5.0], dtype=np.float32)
    collector._current_to_dest_route_candidates = (
        lambda _veh_id, _current_edge, _cost: [["e0", "e1"]]
    )
    collector._filter_safe_reroute_candidates = (
        lambda _veh_id, _current_edge, candidates: candidates
    )
    collector._candidate_route_suffixes = lambda _veh_id, _current_edge: []
    collector._vehicle_elapsed_seconds = (
        lambda veh_ids, _sim_step: np.zeros((len(veh_ids),), dtype=np.float32)
    )
    state = {"vehicles": {"veh": {"edge": "e0"}}}

    for episode in (1, 2):
        collector._current_episode_id_eval = episode
        collector._episode_samples = []
        actions = ToyMambaCollector._select_av_actions(
            collector, state, np.random.RandomState(episode),
        )

        assert actions == {"veh": ["e0", "e1"]}
        assert len(collector._episode_samples) == 1
        assert collector._episode_samples[0].episode == episode
        assert collector._episode_samples[0].reward == pytest.approx(-1.0 / 6.0)
        assert boundary.frames[-1] == collector._episode_samples
        assert collector._all_training_samples == []

    assert len(boundary.frames) == 2
    assert len(collector._dynamic_decision_rows) == 2
    assert len(collector._dynamic_agent_decision_rows) == 2


def test_runner_boundary_integrates_with_real_mamba_controller_and_sparse_replay():
    config = MambaConfig(
        obs_dim=5,
        action_dim=4,
        n_slots=3,
        device="cpu",
        hidden_size=16,
        model_hidden_size=16,
        embedding_size=16,
        deterministic_size=8,
        n_categoricals=2,
        n_classes=2,
        transformer_layers=1,
        transformer_heads=2,
        replay_capacity_episodes=2,
        sequence_length=2,
        burn_in=0,
        max_agents_per_chunk=3,
        sequence_minibatch_size=1,
        imagination_horizon=2,
        min_model_replay_frames=1,
        min_agent_replay_frames=1,
        max_imagination_starts=1,
        use_bfloat16=False,
    )
    learner = MambaLearner(config, ["a", "b", "c"])
    boundary = SparseDecisionBoundary(
        learner=learner,
        machine_ids=["a", "b", "c"],
        obs_dim=5,
        n_routes=4,
    )
    boundary.reset_episode()
    boundary.begin_decision(
        episode=1,
        decision_idx=0,
        sim_step=0,
        presence_vehicle_ids=["a", "c"],
    )
    result = learner.act(
        slot=0,
        obs=np.zeros((5,), dtype=np.float32),
        action_mask=np.ones((4,), dtype=bool),
    )
    sample = _sample("a", action=result["action"])
    sample.old_log_prob = result["log_prob"]
    boundary.end_decision([sample])
    boundary.end_episode(
        episode=1,
        decision_idx=1,
        sim_step=30,
        final_presence_vehicle_ids=[],
        truncated=False,
    )

    assert len(learner.replay) == 1
    assert learner.replay.n_frames == 2
    metrics = learner.update()
    assert metrics["skipped"] == 0.0
    assert math.isfinite(metrics["model_loss"])
    assert math.isfinite(metrics["actor_loss"])


def test_cli_defaults_match_three_city_training_protocol_and_flags():
    args = parse_args(
        [
            "--device",
            "cpu",
            "--train-episodes",
            "30",
            "--checkpoint-interval",
            "10",
            "--hidden-dim",
            "32",
            "--embed-dim",
            "24",
            "--deter-dim",
            "16",
            "--stoch-classes",
            "4",
            "--stoch-categories",
            "8",
            "--communication-chunk-size",
            "7",
            "--imag-horizon",
            "3",
            "--amp-dtype",
            "fp32",
        ]
    )
    cfg = build_config(args)
    plan = build_plan(cfg)

    assert cfg.train_episodes == 30
    assert cfg.checkpoint_episodes == (10, 20, 30)
    assert cfg.update_every_episodes == 1
    assert cfg.replay_episodes == 20
    assert cfg.min_model_replay_frames == 100
    assert cfg.min_agent_replay_frames == 2500
    assert cfg.max_imagination_starts == 16
    assert cfg.model_epochs == 4
    assert cfg.actor_lr == pytest.approx(2.0e-4)
    assert cfg.entropy_coef == pytest.approx(5.0e-3)
    assert cfg.target_policy_kl == pytest.approx(1.0e-2)
    assert cfg.critic_warmup_updates == 20
    assert cfg.critic_epochs == 5
    assert cfg.imagination_reward_scope == "individual"
    assert cfg.ac_normalize_advantages is True
    assert cfg.ac_advantage_clip == pytest.approx(2.5)
    assert cfg.ppo_clip == pytest.approx(0.075)
    assert cfg.hidden_dim == 32
    assert cfg.embed_dim == 24
    assert cfg.deter_dim == 16
    assert cfg.communication_chunk_size == 7
    assert cfg.imagination_horizon == 3
    assert plan["mamba"]["min_model_replay_frames"] == 100
    assert plan["mamba"]["min_agent_replay_frames"] == 2500
    assert plan["mamba"]["max_imagination_starts"] == 16
    assert plan["mamba"]["amp_dtype"] == "fp32"


def test_dry_run_does_not_require_artifacts_or_create_output(tmp_path, capsys):
    output = tmp_path / "not-created"
    assert main(["--dry-run", "--device", "cpu", "--output-dir", str(output)]) == 0
    parsed = json.loads(capsys.readouterr().out)

    assert parsed["algorithm"] == "mamba"
    assert parsed["mamba"]["train_episodes"] == 500
    assert not output.exists()


def test_checkpoint_is_atomic_and_round_trips_collector_protocol_state(tmp_path):
    trainer = object.__new__(ToyMambaTrainer)
    trainer.output_dir = tmp_path
    trainer.learner = _FakeLearner()
    trainer.cfg = ToyMambaConfig(
        settings="",
        output_dir=str(tmp_path),
        network_dir=str(tmp_path),
        edge_registry=str(tmp_path / "edge_registry.json"),
        route_artifacts_dir=str(tmp_path),
        post_mutation_state_in="",
        hdv_state_in="",
        device="cpu",
    )
    trainer.obs_dim = 9
    trainer.machine_ids = ["m0", "m2"]
    trainer.human_ids = ["h1"]
    trainer.role_source = "unit-test"
    trainer.role_hash = fixed_role_hash(trainer.machine_ids)
    trainer.collector = SimpleNamespace(
        connector=SimpleNamespace(
            _episode_count=37,
            env=SimpleNamespace(
                human_agents=[
                    SimpleNamespace(
                        id="h1",
                        model=SimpleNamespace(
                            cost=np.asarray([2.0, 3.0], dtype=np.float32),
                            memory=np.asarray([0.25, 0.75], dtype=np.float32),
                            beta=-1.5,
                        ),
                    )
                ]
            ),
        ),
        _loaded_hdv_state_eval={
            "version": 1,
            "agents": {"h0": {"model_class": "Legacy", "cost": [9.0]}},
        },
        _hdv_route_probs={(0, 1): np.asarray([0.25, 0.75], dtype=np.float32)},
        _hdv_route_costs={(0, 1): np.asarray([12.0, 18.0], dtype=np.float32)},
    )

    path = ToyMambaTrainer.save_checkpoint(trainer, 100, {"loss": 1.25})
    state = torch.load(path, map_location="cpu", weights_only=False)

    assert path.name == "checkpoint_ep00100.pt"
    assert state["schema_version"] == 3
    assert state["model_family"] == "mamba"
    assert state["machine_agent_ids"] == ["m0", "m2"]
    assert state["role_hash"] == trainer.role_hash
    assert state["runner_config"]["train_episodes"] == 500
    protocol_state = state["collector_protocol_state"]
    assert protocol_state["version"] == 1
    assert protocol_state["connector_episode_count"] == 37
    assert set(protocol_state["hdv_learning_state"]["agents"]) == {"h0", "h1"}
    assert protocol_state["hdv_learning_state"]["agents"]["h1"]["cost"] == [2.0, 3.0]
    assert protocol_state["hdv_learning_state"]["route_policy"] == [
        {
            "origin": 0,
            "destination": 1,
            "probs": [0.25, 0.75],
            "costs": [12.0, 18.0],
        }
    ]
    assert not list(path.parent.glob("*.tmp.*"))

    resumed = object.__new__(ToyMambaTrainer)
    resumed.cfg = SimpleNamespace(resume_checkpoint=str(path), n_routes=4)
    resumed.device = torch.device("cpu")
    resumed.machine_ids = list(trainer.machine_ids)
    resumed.obs_dim = trainer.obs_dim
    resumed.learner = _FakeLearner()
    resumed._resume_collector_protocol_state = None

    assert ToyMambaTrainer._load_resume_checkpoint(resumed) == 100
    restored_hdv_state = ToyMambaTrainer._resume_hdv_state(
        resumed,
        {"version": 1, "agents": {"wrong-source": {}}},
    )
    assert restored_hdv_state == protocol_state["hdv_learning_state"]
    restored_collector = SimpleNamespace(connector=SimpleNamespace(_episode_count=0))
    ToyMambaTrainer._initialize_connector_episode_count(
        resumed,
        restored_collector,
        {"mutation_episode_index": 5},
    )
    assert restored_collector.connector._episode_count == 37


def test_fixed_role_hash_depends_on_identity_order():
    assert fixed_role_hash(["a", "b"]) == fixed_role_hash(["a", "b"])
    assert fixed_role_hash(["a", "b"]) != fixed_role_hash(["b", "a"])


def test_mamba_transition_metrics_do_not_claim_urban_dreamer_cfm_usage():
    diagnostics = mamba_transition_diagnostics()

    assert diagnostics["cfm_transition_used"] == 0
    assert diagnostics["mamba_transition_used"] == 1
    assert diagnostics["wm_rollout_rerank_used"] == 0
