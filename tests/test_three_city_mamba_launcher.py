import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_mamba_inputs import (  # noqa: E402
    ARTIFACT_FILES,
    CITY_SPECS,
    prepare_inputs,
    validate_city_inputs,
    validate_smoke_outputs,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fake_city(root: Path, city: str = "ingolstadt") -> None:
    spec = CITY_SPECS[city]
    settings = root / spec.settings_rel
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        "\n".join(
            [
                '# generated on another host',
                f'NETWORK_DIR="/old/server/UrbanDreamer/networks/{spec.network_name}"',
                f'NETWORK_NAME="{spec.network_name}"',
                f'EDGE_REGISTRY="/old/server/UrbanDreamer/networks/{spec.network_name}/edge_registry.json"',
                f'AGENTS_CSV="/old/server/UrbanDreamer/networks/{spec.network_name}/agents.csv"',
                f'OD_FILE="/old/server/UrbanDreamer/networks/{spec.network_name}/od_{spec.network_name}.txt"',
                'ROUTE_ARTIFACTS_DIR="/old/server/UrbanDreamer/results/stale/records"',
                'FLOW_DATA_DIR="/old/server/UrbanDreamer/data/old"',
                'RATIO_MACHINES="0.4"',
                'N_AGENTS="5"',
                'N_AV="2"',
                'N_HDV="3"',
                'AV_CANDIDATE_SOURCE="current_to_dest"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    network = root / "networks" / spec.network_name
    bundle = root / spec.bundle_rel
    network.mkdir(parents=True, exist_ok=True)
    bundle.mkdir(parents=True, exist_ok=True)
    agents = (
        "id,origin,destination,start_time,kind\n"
        "0,0,0,0,Human\n"
        "1,0,0,1,Human\n"
        "2,0,0,2,Human\n"
        "3,0,0,3,Human\n"
        "4,0,0,4,Human\n"
    )
    (network / "agents.csv").write_text(agents, encoding="utf-8")
    (bundle / "agents.csv").write_text(agents, encoding="utf-8")
    (bundle / "routes.csv").write_text(
        "origins,destinations,path,free_flow_time\n0,0,edge0,1.0\n",
        encoding="utf-8",
    )
    (bundle / "route.rou.xml").write_text(
        '<routes><route id="r0" edges="edge0"/></routes>\n', encoding="utf-8"
    )
    (network / f"{spec.network_name}.net.xml").write_text(
        '<net><edge id="edge0"/></net>\n', encoding="utf-8"
    )
    (network / "edge_registry.json").write_text("{}\n", encoding="utf-8")
    (network / f"od_{spec.network_name}.txt").write_text("0 0 1\n", encoding="utf-8")
    (bundle / "post_mutation_hdv_learning_state.json").write_text(
        json.dumps({"version": 1, "agents": {str(i): {} for i in range(5)}}) + "\n",
        encoding="utf-8",
    )
    state = {
        "schema_version": 1,
        "network": spec.network_name,
        "ratio_machines": 0.4,
        "should_humans_adapt": False,
        "machine_agent_ids": ["1", "4"],
        "human_agent_ids": ["0", "2", "3"],
        "human_learning_state_ref": "/old/server/hdv.json",
        "artifact_refs": {
            key: f"/old/server/{filename}" for key, filename in ARTIFACT_FILES.items()
        },
        "artifact_hashes": {
            key: _hash(bundle / filename) for key, filename in ARTIFACT_FILES.items()
        },
    }
    (bundle / "post_mutation_state.json").write_text(
        json.dumps(state) + "\n", encoding="utf-8"
    )


def test_city_specs_use_canonical_bundles_and_fixed_gpu_mapping():
    assert CITY_SPECS["gretz"].bundle_rel == "networks/gretz_armainvilliers/post_mutation"
    assert CITY_SPECS["ingolstadt"].bundle_rel == "networks/ingolstadt_custom/post_mutation"
    assert CITY_SPECS["fontenay"].bundle_rel == "networks/fontenay_en_parisis"
    assert CITY_SPECS["fontenay"].gpu == "0"
    assert CITY_SPECS["gretz"].gpu == CITY_SPECS["ingolstadt"].gpu == "1"
    assert CITY_SPECS["ingolstadt"].run_tag == (
        "ingolstadt_custom_current_to_dest_incident_20260528"
    )


def test_prepare_writes_portable_settings_and_post_state(tmp_path):
    _write_fake_city(tmp_path)
    output = tmp_path / "run" / "portable_inputs"

    manifest = prepare_inputs(tmp_path, output, ["ingolstadt"])

    city = manifest["cities"]["ingolstadt"]
    portable_settings = Path(city["portable_settings"])
    settings_text = portable_settings.read_text(encoding="utf-8")
    bundle = (tmp_path / CITY_SPECS["ingolstadt"].bundle_rel).resolve()
    assert "/old/server/UrbanDreamer" not in settings_text
    assert f'ROUTE_ARTIFACTS_DIR="{bundle}"' in settings_text
    assert f'NETWORK_DIR="{tmp_path.resolve()}/networks/ingolstadt_custom"' in settings_text

    portable_state = json.loads(Path(city["portable_post_state"]).read_text())
    assert portable_state["human_learning_state_ref"] == str(
        bundle / "post_mutation_hdv_learning_state.json"
    )
    assert portable_state["artifact_refs"]["routes_csv"] == str(bundle / "routes.csv")
    assert city["counts"] == {
        "agents": 5,
        "machines": 2,
        "humans": 3,
        "routes_xml": 1,
        "network_edges_xml": 1,
    }
    assert city["xml_validated_to_eof"] is True
    assert (output / "manifest.json").is_file()


def test_prepare_rejects_hash_and_fixed_role_mismatches(tmp_path):
    _write_fake_city(tmp_path)
    spec = CITY_SPECS["ingolstadt"]
    bundle = tmp_path / spec.bundle_rel
    (bundle / "routes.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_city_inputs(tmp_path, spec)

    _write_fake_city(tmp_path)
    state_path = bundle / "post_mutation_state.json"
    state = json.loads(state_path.read_text())
    state["human_agent_ids"].remove("3")
    state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="do not cover canonical agents"):
        validate_city_inputs(tmp_path, spec)


def _write_smoke_output(path: Path, expected_episodes: int = 2) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "online_train_episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode", "reward_mean"])
        writer.writeheader()
        for episode in range(1, expected_episodes + 1):
            writer.writerow({"episode": episode, "reward_mean": -1.0})
    with (path / "online_train_updates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "episode",
                "model_loss",
                "actor_loss",
                "critic_loss",
                "skipped",
                "actor_updated",
                "actor_skipped",
                "ppo_epochs_completed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "episode": expected_episodes,
                "model_loss": 1.2,
                "actor_loss": -0.1,
                "critic_loss": 0.4,
                "skipped": 0,
                "actor_updated": 1,
                "actor_skipped": 0,
                "ppo_epochs_completed": 1,
            }
        )
    (path / "online_train_decisions.csv").write_text(
        "episode,veh_id,action\n1,1,0\n", encoding="utf-8"
    )
    (path / "online_train_summary.json").write_text(
        json.dumps({"algorithm": "mamba"}) + "\n", encoding="utf-8"
    )
    (path / "hdv_learning_state_after_online_train.json").write_text(
        "{}\n", encoding="utf-8"
    )
    checkpoints = path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / f"checkpoint_ep{expected_episodes:05d}.pt").write_bytes(b"checkpoint")


def test_smoke_validation_requires_real_updates_artifacts_and_memory_limit(tmp_path):
    outputs = {}
    for city in ("fontenay", "gretz", "ingolstadt"):
        output = tmp_path / city
        _write_smoke_output(output)
        outputs[city] = output
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "timestamp,phase,gpu,memory_used_mib,utilization_gpu_pct\n"
        "t,smoke,0,19000,80\n"
        "t,smoke,1,19800,75\n",
        encoding="utf-8",
    )

    result = validate_smoke_outputs(
        outputs,
        telemetry_path=telemetry,
        expected_episodes=2,
        max_gpu_memory_mib=20000,
    )

    assert result["ok"] is True
    assert result["gpu_peak_memory_mib"] == {"0": 19000, "1": 19800}
    assert result["cities"]["gretz"]["model_loss_columns"] == ["model_loss"]

    telemetry.write_text(
        telemetry.read_text(encoding="utf-8").replace("19800", "20001"),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="exceeds 20000 MiB"):
        validate_smoke_outputs(
            outputs,
            telemetry_path=telemetry,
            expected_episodes=2,
            max_gpu_memory_mib=20000,
        )


def test_smoke_validation_rejects_finite_but_fully_rolled_back_actor(tmp_path):
    outputs = {}
    for city in ("fontenay", "gretz", "ingolstadt"):
        output = tmp_path / city
        _write_smoke_output(output)
        outputs[city] = output
    failed = outputs["ingolstadt"] / "online_train_updates.csv"
    failed.write_text(
        failed.read_text(encoding="utf-8").replace(",1,0,1\n", ",0,1,0\n"),
        encoding="utf-8",
    )
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "timestamp,phase,gpu,memory_used_mib,utilization_gpu_pct\n"
        "t,smoke,0,1000,10\n"
        "t,smoke,1,1000,10\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="no accepted PPO actor epoch"):
        validate_smoke_outputs(
            outputs,
            telemetry_path=telemetry,
            expected_episodes=2,
            max_gpu_memory_mib=20000,
        )


def test_launcher_syntax_and_safety_contract():
    launcher = ROOT / "scripts" / "run_three_city_mamba_parallel3.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text(encoding="utf-8")
    assert 'IDLE_FREE_MIB="${IDLE_FREE_MIB:-23000}"' in text
    assert 'IDLE_MAX_UTIL="${IDLE_MAX_UTIL:-5}"' in text
    assert 'IDLE_CONFIRMATIONS="${IDLE_CONFIRMATIONS:-3}"' in text
    assert 'IDLE_POLL_SECONDS="${IDLE_POLL_SECONDS:-30}"' in text
    assert 'SMOKE_MAX_GPU_MEMORY_MIB="${SMOKE_MAX_GPU_MEMORY_MIB:-20000}"' in text
    assert 'SKIP_IDLE_GATE="${SKIP_IDLE_GATE:-0}"' in text
    assert 'SMOKE_ONLY="${SMOKE_ONLY:-0}"' in text
    assert 'if [[ "${SMOKE_ONLY}" == "1" ]]' in text
    assert 'explicit SKIP_IDLE_GATE=1' in text
    assert 'FORMAL_UPDATE_EVERY_EPISODES="${FORMAL_UPDATE_EVERY_EPISODES:-1}"' in text
    assert 'FORMAL_MIN_MODEL_REPLAY_FRAMES="${FORMAL_MIN_MODEL_REPLAY_FRAMES:-100}"' in text
    assert 'FORMAL_MIN_AGENT_REPLAY_FRAMES="${FORMAL_MIN_AGENT_REPLAY_FRAMES:-2500}"' in text
    assert 'FORMAL_MAX_IMAGINATION_STARTS="${FORMAL_MAX_IMAGINATION_STARTS:-16}"' in text
    assert 'FORMAL_CRITIC_WARMUP_UPDATES="${FORMAL_CRITIC_WARMUP_UPDATES:-20}"' in text
    assert '--min-model-replay-frames "${min_model_replay_frames}"' in text
    assert '--min-agent-replay-frames "${min_agent_replay_frames}"' in text
    assert '--max-imagination-starts "${FORMAL_MAX_IMAGINATION_STARTS}"' in text
    assert '--critic-warmup-updates "${critic_warmup_updates}"' in text
    assert '--imagination-reward-scope individual' in text
    assert '--advantage-clip 2.5' in text
    assert '--normalize-advantages' in text
    assert '--ppo-clip 7.5e-2' in text
    assert '--target-policy-kl 1e-2' in text
    assert 'min_model_replay_frames=1' in text
    assert 'min_agent_replay_frames=1' in text
    assert 'critic_warmup_updates=0' in text
    assert 'REUSE_SMOKE_VALIDATION="${REUSE_SMOKE_VALIDATION:-}"' in text
    assert 'previous update-every-episode smoke validation reused' in text
    assert 'GRETZ_GPU="${GRETZ_GPU:-1}"' in text
    assert 'INGOLSTADT_GPU="${INGOLSTADT_GPU:-1}"' in text
    assert "setsid \"${CMD[@]}\"" in text
    assert "pkill" not in text
    assert "killall" not in text
    assert "IDLE_GATE=SKIPPED_FOR_DRY_RUN" in text
    assert 'sysconfig.get_paths()["purelib"]' in text
    assert '"${PYTHON_PURELIB}/sumo"' in text
    assert '"${ENV_ROOT}/share/sumo"' in text
    assert 'export PATH="${SUMO_HOME}/bin:${PATH}"' in text
    purelib_pos = text.index('"${PYTHON_PURELIB}/sumo"')
    share_pos = text.index('"${ENV_ROOT}/share/sumo"')
    env_root_pos = text.index('      "${ENV_ROOT}"; do')
    assert purelib_pos < share_pos < env_root_pos


def test_current_python_purelib_sumo_binary_is_usable_when_installed():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sysconfig; print(sysconfig.get_paths()["purelib"])',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    binary = Path(completed.stdout.strip()) / "sumo" / "bin" / "sumo"
    if not binary.is_file():
        pytest.skip("the test Python does not provide pip SUMO binaries")
    version = subprocess.run(
        [str(binary), "--version"], check=True, capture_output=True, text=True
    )
    assert "Eclipse SUMO" in version.stdout
