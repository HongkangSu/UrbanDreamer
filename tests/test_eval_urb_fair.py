import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_urb_fair import build_eval_urb_fair_argv  # noqa: E402
from scripts.eval_sumo import _episode_complete_for_dataset, _summarize_test_episode_metrics  # noqa: E402
from dreamer4.urban_routing.urb_fair_protocol import (  # noqa: E402
    build_run_manifest,
    derive_fixed_roles,
    load_post_mutation_state,
    verify_post_mutation_state,
    write_post_mutation_state,
    write_run_manifest,
)


def _write_settings(path: Path, results_dir: Path, artifacts_dir: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "NETWORK_NAME=toy_net",
                f"NETWORK_DIR={artifacts_dir}",
                f"EDGE_REGISTRY={artifacts_dir / 'edge_registry.json'}",
                f"AGENTS_CSV={artifacts_dir / 'agents.csv'}",
                f"ROUTE_ARTIFACTS_DIR={artifacts_dir}",
                "ROUTE_GENERATOR=dijkstra",
                "RATIO_MACHINES=0.4",
                "HDV_ADAPT_EP=200",
                "EVAL_TEST_EPISODES=100",
                "SIM_BUFFER_AFTER_LAST_DEPART=1000",
                "EPISODE_STEPS=60",
                "INCIDENT_EDGE=201201950#0",
                "INCIDENT_START=500",
                "INCIDENT_END=1500",
                "INCIDENT_SPEED=1.0",
                f"EVAL_RESULTS_DIR={results_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_artifacts(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "agents.csv").write_text(
        "id,start_time,origin,destination,kind\n0,7,0,0,Human\n1,11,0,0,Human\n",
        encoding="utf-8",
    )
    (path / "routes.csv").write_text("origins,destinations,path,free_flow_time\n0,0,a b,1.0\n", encoding="utf-8")
    (path / "route.rou.xml").write_text("<routes></routes>\n", encoding="utf-8")
    (path / "edge_registry.json").write_text("{}", encoding="utf-8")


def test_eval_urb_fair_injects_alignment_defaults(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    _write_artifacts(artifacts_dir)
    settings = _write_settings(tmp_path / "eval.env", tmp_path / "results", artifacts_dir)

    argv = build_eval_urb_fair_argv(
        settings_path=str(settings),
        policy_ckpt="logs/toy_net/marl/latest.pt",
        passthrough=[],
    )

    assert "--settings" in argv
    assert "--policy_ckpt" in argv
    assert "--route_generator" in argv
    assert argv[argv.index("--route_generator") + 1] == "dijkstra"
    assert "--reuse_artifacts_dir" in argv
    assert argv[argv.index("--reuse_artifacts_dir") + 1] == str(artifacts_dir)
    assert "--disable_human_learning_after_mutation" in argv
    assert "--keep_human_learning_after_mutation" not in argv
    assert argv[argv.index("--av_policy_route_selection") + 1] == "sample"
    assert "--hdv_state_out" in argv
    assert "--post_mutation_state_out" in argv


def test_eval_urb_fair_respects_explicit_override_flags(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    _write_artifacts(artifacts_dir)
    settings = _write_settings(tmp_path / "eval.env", tmp_path / "results", artifacts_dir)

    argv = build_eval_urb_fair_argv(
        settings_path=str(settings),
        policy_ckpt="logs/toy_net/marl/latest.pt",
        passthrough=[
            "--route_generator",
            "routerl",
            "--keep_human_learning_after_mutation",
            "--av_policy_route_selection",
            "argmax",
        ],
    )

    assert argv[argv.index("--route_generator") + 1] == "routerl"
    assert "--disable_human_learning_after_mutation" not in argv
    assert "--keep_human_learning_after_mutation" in argv
    assert argv[argv.index("--av_policy_route_selection") + 1] == "argmax"


def test_eval_urb_fair_post_mutation_state_follows_results_dir_override(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    _write_artifacts(artifacts_dir)
    settings = _write_settings(tmp_path / "eval.env", tmp_path / "results", artifacts_dir)

    argv = build_eval_urb_fair_argv(
        settings_path=str(settings),
        policy_ckpt="logs/toy_net/marl/latest.pt",
        passthrough=["--results_dir", str(tmp_path / "custom_eval")],
    )

    assert argv[argv.index("--post_mutation_state_out") + 1].endswith("custom_eval/post_mutation_state.json")
    assert argv[argv.index("--hdv_state_out") + 1].endswith("custom_eval/hdv_learning_state.json")


def test_eval_urb_fair_respects_explicit_post_mutation_state_in(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    _write_artifacts(artifacts_dir)
    settings = _write_settings(tmp_path / "eval.env", tmp_path / "results", artifacts_dir)

    argv = build_eval_urb_fair_argv(
        settings_path=str(settings),
        policy_ckpt="logs/toy_net/marl/latest.pt",
        passthrough=["--post_mutation_state_in", "state.json"],
    )

    assert "--post_mutation_state_in" in argv
    assert "--post_mutation_state_out" not in argv


def test_eval_summary_uses_last_test_episode_as_headline():
    import pandas as pd

    test_df = pd.DataFrame(
        [
            {
                "episode": 31,
                "phase": "test",
                "t_CAV": 10.0,
                "t_HDV": 20.0,
                "t_all": 15.0,
                "arrived_all": 50,
                "total_agents": 100,
                "unfinished_all": 50,
                "completion_rate": 0.5,
            },
            {
                "episode": 32,
                "phase": "test",
                "t_CAV": 4.0,
                "t_HDV": 8.0,
                "t_all": 6.0,
                "arrived_all": 80,
                "total_agents": 100,
                "unfinished_all": 20,
                "completion_rate": 0.8,
            },
        ]
    )

    summary = _summarize_test_episode_metrics(test_df)

    assert summary["metric_aggregation"] == "last_test_episode"
    assert summary["summary_episode"] == 32
    assert summary["t_CAV"] == 4.0
    assert summary["t_HDV"] == 8.0
    assert summary["t_all"] == 6.0
    assert summary["t_all_mean"] == 10.5
    assert summary["arrived_all"] == 80
    assert summary["total_agents"] == 100
    assert summary["unfinished_all"] == 20
    assert summary["completion_rate"] == 0.8
    assert summary["completion_rate_mean"] == pytest.approx(0.65)


def test_episode_complete_for_dataset_requires_all_agents_and_steps():
    assert _episode_complete_for_dataset(n_arrived=5, total_agents=5, n_steps=3)
    assert not _episode_complete_for_dataset(n_arrived=4, total_agents=5, n_steps=3)
    assert not _episode_complete_for_dataset(n_arrived=5, total_agents=5, n_steps=0)
    assert not _episode_complete_for_dataset(n_arrived=1, total_agents=0, n_steps=3)


def test_urb_fair_manifest_contains_artifact_contract(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_artifacts(artifacts_dir)
    for name in ("agents.csv", "routes.csv", "route.rou.xml"):
        (results_dir / name).write_text((artifacts_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
    settings = _write_settings(tmp_path / "eval.env", results_dir, artifacts_dir)

    argv = [
        "eval_sumo.py",
        "--settings",
        str(settings),
        "--policy_ckpt",
        "logs/example/latest.pt",
        "--results_dir",
        str(results_dir),
        "--route_generator",
        "dijkstra",
        "--reuse_artifacts_dir",
        str(results_dir),
    ]

    manifest = build_run_manifest(settings_path=str(settings), forwarded_argv=argv)
    assert manifest["artifact_refs"]["agents_csv"].endswith("agents.csv")
    assert manifest["artifact_hashes"]["routes_csv"]
    assert manifest["human_learning_episodes"] == 200
    assert manifest["test_eps"] == 100
    assert manifest["training_eps"] == 0
    assert manifest["mutation_episode_index"] == 200
    assert manifest["sim_buffer_after_last_depart"] == 1000
    assert manifest["incident"] == {
        "edge": "201201950#0",
        "start": 500.0,
        "end": 1500.0,
        "speed": 1.0,
    }
    path = write_run_manifest(results_dir, manifest)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["artifact_hashes"] == manifest["artifact_hashes"]


def test_derive_fixed_roles_is_deterministic(tmp_path):
    agents_csv = tmp_path / "agents.csv"
    agents_csv.write_text(
        "id,start_time,origin,destination,kind\n"
        "0,0,0,0,Human\n"
        "1,0,0,1,Human\n"
        "2,0,1,0,Human\n"
        "3,0,1,1,Human\n",
        encoding="utf-8",
    )

    first = derive_fixed_roles(agents_csv=agents_csv, ratio_machines=0.5, seed=7)
    second = derive_fixed_roles(agents_csv=agents_csv, ratio_machines=0.5, seed=7)

    assert first == second
    machine_ids, human_ids = first
    assert len(machine_ids) == 2
    assert len(human_ids) == 2


def test_post_mutation_state_roundtrip(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for name, content in (
        ("agents.csv", "id,start_time,origin,destination,kind\n0,7,0,0,Human\n1,11,0,0,Human\n"),
        ("routes.csv", "origins,destinations,path,free_flow_time\n0,0,a b,1.0\n"),
        ("route.rou.xml", "<routes></routes>\n"),
    ):
        (results_dir / name).write_text(content, encoding="utf-8")

    state = {
        "schema_version": 1,
        "network": "toy_net",
        "ratio_machines": 0.4,
        "human_model": "gawron",
        "should_humans_adapt": False,
        "human_learning_episodes": 200,
        "mutation_episode_index": 200,
        "mutation_start_percentile": -1,
        "machine_agent_ids": ["0"],
        "human_agent_ids": ["1"],
        "artifact_refs": {
            "agents_csv": str(results_dir / "agents.csv"),
            "routes_csv": str(results_dir / "routes.csv"),
            "route_rou_xml": str(results_dir / "route.rou.xml"),
        },
        "artifact_hashes": {
            "agents_csv": "",
            "routes_csv": "",
            "route_rou_xml": "",
        },
        "human_learning_state_ref": str(results_dir / "hdv_learning_state.json"),
    }
    from dreamer4.urban_routing.urb_fair_protocol import collect_route_artifact_contract

    state["artifact_hashes"] = collect_route_artifact_contract(results_dir)["artifact_hashes"]
    state_path = write_post_mutation_state(
        results_dir=results_dir,
        state=state,
    )
    state = load_post_mutation_state(state_path)

    verify_post_mutation_state(
        results_dir=results_dir,
        expected_network="toy_net",
        expected_ratio_machines=0.4,
        expected_human_model="gawron",
        expected_should_humans_adapt=False,
        state=state,
    )

    # Counterfactual pre-CAV references legitimately contain no machines.
    state["ratio_machines"] = 0.0
    state["machine_agent_ids"] = []
    state["human_agent_ids"] = ["0", "1"]
    verify_post_mutation_state(
        results_dir=results_dir,
        expected_network="toy_net",
        expected_ratio_machines=0.0,
        expected_human_model="gawron",
        expected_should_humans_adapt=False,
        state=state,
    )
