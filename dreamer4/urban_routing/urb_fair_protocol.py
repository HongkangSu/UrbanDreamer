from __future__ import annotations

import hashlib
import json
import random
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .experiment_settings import load_settings_file


def _arg_value(argv: Iterable[str], flag: str) -> Optional[str]:
    seq = list(argv)
    for idx, item in enumerate(seq):
        if str(item) == flag and idx + 1 < len(seq):
            return str(seq[idx + 1])
    return None


def _has_flag(argv: Iterable[str], flag: str) -> bool:
    return any(str(x) == flag for x in argv)


def _load_json_if_exists(path_str: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_route_artifact_contract(results_dir: Path) -> Dict[str, Dict[str, str]]:
    required = {
        "agents_csv": results_dir / "agents.csv",
        "routes_csv": results_dir / "routes.csv",
        "route_rou_xml": results_dir / "route.rou.xml",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing URB-fair route artifacts: " + ", ".join(missing)
        )
    refs = {k: str(v.resolve()) for k, v in required.items()}
    hashes = {k: sha256_file(v) for k, v in required.items()}
    return {"artifact_refs": refs, "artifact_hashes": hashes}


def _max_start_time(agents_csv: Path) -> int:
    max_start = 0
    with agents_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                max_start = max(max_start, int(row["start_time"]))
            except Exception:
                continue
    return int(max_start)


def build_run_manifest(
    *,
    settings_path: str,
    forwarded_argv: List[str],
) -> Dict[str, object]:
    cfg = load_settings_file(settings_path)
    results_dir = Path(
        _arg_value(forwarded_argv, "--results_dir")
        or cfg.get("EVAL_RESULTS_DIR", "")
    ).resolve()
    route_generator = (
        _arg_value(forwarded_argv, "--route_generator")
        or cfg.get("ROUTE_GENERATOR", "")
    )
    reuse_artifacts_dir = _arg_value(forwarded_argv, "--reuse_artifacts_dir")
    policy_ckpt = _arg_value(forwarded_argv, "--policy_ckpt") or cfg.get("MARL_POLICY_CKPT", cfg.get("MARL_CKPT", ""))
    human_learning_episodes = int(
        _arg_value(forwarded_argv, "--human_learning_episodes")
        or cfg.get("HDV_ADAPT_EP", "0")
    )
    test_eps = int(
        _arg_value(forwarded_argv, "--test_eps")
        or cfg.get("EVAL_TEST_EPISODES", "0")
    )
    post_mutation_state_in = _arg_value(forwarded_argv, "--post_mutation_state_in")
    loaded_post_mutation_state = _load_json_if_exists(post_mutation_state_in)
    disable_human_learning_after_mutation = (
        _has_flag(forwarded_argv, "--disable_human_learning_after_mutation")
        or (
            not _has_flag(forwarded_argv, "--keep_human_learning_after_mutation")
            and str(cfg.get("DISABLE_HUMAN_LEARNING_AFTER_MUTATION", "")).strip().lower() in {"1", "true", "yes", "on"}
        )
    )
    if loaded_post_mutation_state is not None:
        human_learning_episodes = int(
            loaded_post_mutation_state.get("human_learning_episodes", human_learning_episodes)
        )
        mutation_episode_index = int(
            loaded_post_mutation_state.get("mutation_episode_index", human_learning_episodes)
        )
        human_model = str(loaded_post_mutation_state.get("human_model", "gawron") or "gawron")
        should_humans_adapt = bool(
            loaded_post_mutation_state.get("should_humans_adapt", not disable_human_learning_after_mutation)
        )
        effective_human_learning_episodes = 0
    else:
        mutation_episode_index = human_learning_episodes
        human_model = "gawron"
        should_humans_adapt = (not disable_human_learning_after_mutation)
        effective_human_learning_episodes = human_learning_episodes
    sim_buffer_after_last_depart = int(
        _arg_value(forwarded_argv, "--sim_buffer_after_last_depart")
        or cfg.get("SIM_BUFFER_AFTER_LAST_DEPART", "0")
        or 0
    )

    manifest: Dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_config": str(Path(settings_path).resolve()),
        "network": cfg.get("NETWORK_NAME", ""),
        "policy_ckpt": policy_ckpt,
        "ratio_machines": float(cfg.get("RATIO_MACHINES", "0") or 0.0),
        "human_model": human_model,
        "should_humans_adapt": should_humans_adapt,
        "human_learning_episodes": human_learning_episodes,
        "effective_human_learning_episodes": effective_human_learning_episodes,
        "training_eps": 0,
        "test_eps": test_eps,
        "mutation_episode_index": mutation_episode_index,
        "sim_buffer_after_last_depart": sim_buffer_after_last_depart,
        "episode_steps": int(cfg.get("EPISODE_STEPS", "0") or 0),
        "route_generator": route_generator,
        "reuse_artifacts_dir": str(Path(reuse_artifacts_dir).resolve()) if reuse_artifacts_dir else "",
        "post_mutation_state_in": str(Path(post_mutation_state_in).resolve()) if post_mutation_state_in else "",
        "incident": {
            "edge": _arg_value(forwarded_argv, "--incident_edge") or cfg.get("INCIDENT_EDGE", ""),
            "start": float(_arg_value(forwarded_argv, "--incident_start") or cfg.get("INCIDENT_START", "0") or 0.0),
            "end": float(_arg_value(forwarded_argv, "--incident_end") or cfg.get("INCIDENT_END", "0") or 0.0),
            "speed": float(_arg_value(forwarded_argv, "--incident_speed") or cfg.get("INCIDENT_SPEED", "0") or 0.0),
        },
        "results_dir": str(results_dir),
    }
    manifest.update(collect_route_artifact_contract(results_dir))
    agents_csv = Path(manifest["artifact_refs"]["agents_csv"])
    manifest["simulation_timesteps"] = int(_max_start_time(agents_csv) + sim_buffer_after_last_depart)
    return manifest


def write_run_manifest(results_dir: Path, manifest: Dict[str, object]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "urb_fair_run_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def derive_fixed_roles(
    *,
    agents_csv: Path,
    ratio_machines: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    import csv

    ids: List[str] = []
    with agents_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(str(row["id"]))
    rng = random.Random(int(seed))
    ordered = list(ids)
    rng.shuffle(ordered)
    n_av = max(0, min(len(ordered), int(round(len(ordered) * float(ratio_machines)))))
    machine_ids = sorted(ordered[:n_av], key=lambda x: int(x))
    human_ids = sorted(ordered[n_av:], key=lambda x: int(x))
    return machine_ids, human_ids


def build_post_mutation_state(
    *,
    settings_path: str,
    forwarded_argv: List[str],
    mutation_episode_index: int,
    machine_agent_ids: List[str],
    human_agent_ids: List[str],
    human_learning_state_ref: Optional[str],
) -> Dict[str, object]:
    manifest = build_run_manifest(settings_path=settings_path, forwarded_argv=forwarded_argv)
    state = {
        "schema_version": 1,
        "network": manifest["network"],
        "ratio_machines": manifest["ratio_machines"],
        "human_model": manifest["human_model"],
        "should_humans_adapt": manifest["should_humans_adapt"],
        "human_learning_episodes": manifest["human_learning_episodes"],
        "mutation_episode_index": int(mutation_episode_index),
        "mutation_start_percentile": -1,
        "machine_agent_ids": list(machine_agent_ids),
        "human_agent_ids": list(human_agent_ids),
        "artifact_refs": manifest["artifact_refs"],
        "artifact_hashes": manifest["artifact_hashes"],
        "human_learning_state_ref": human_learning_state_ref,
    }
    return state


def write_post_mutation_state(results_dir: Path, state: Dict[str, object]) -> Path:
    path = results_dir
    if path.suffix.lower() != ".json":
        path = path / "post_mutation_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_post_mutation_state(path: Path) -> Dict[str, object]:
    state = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "network",
        "ratio_machines",
        "human_model",
        "should_humans_adapt",
        "human_learning_episodes",
        "mutation_episode_index",
        "mutation_start_percentile",
        "machine_agent_ids",
        "human_agent_ids",
        "artifact_refs",
        "artifact_hashes",
        "human_learning_state_ref",
    }
    missing = required - set(state.keys())
    if missing:
        raise ValueError(f"post-mutation state missing keys: {sorted(missing)}")
    return state


def verify_post_mutation_state(
    *,
    results_dir: Path,
    expected_network: str,
    expected_ratio_machines: float,
    expected_human_model: str = "gawron",
    expected_should_humans_adapt: Optional[bool] = None,
    state: Dict[str, object],
) -> None:
    contract = collect_route_artifact_contract(results_dir)
    if str(state.get("network", "")) != str(expected_network):
        raise ValueError("post-mutation state network mismatch")
    if float(state.get("ratio_machines", -1.0)) != float(expected_ratio_machines):
        raise ValueError("post-mutation state ratio_machines mismatch")
    if str(state.get("human_model", "")).strip().lower() != str(expected_human_model).strip().lower():
        raise ValueError("post-mutation state human_model mismatch")
    if expected_should_humans_adapt is not None and bool(state.get("should_humans_adapt")) != bool(expected_should_humans_adapt):
        raise ValueError("post-mutation state should_humans_adapt mismatch")
    machine_ids = [str(x) for x in state.get("machine_agent_ids", [])]
    human_ids = [str(x) for x in state.get("human_agent_ids", [])]
    ratio_machines = float(expected_ratio_machines)
    if not machine_ids and ratio_machines > 0.0:
        raise ValueError("post-mutation state machine ids must be non-empty when ratio_machines > 0")
    if not human_ids and ratio_machines < 1.0:
        raise ValueError("post-mutation state human ids must be non-empty when ratio_machines < 1")
    if set(machine_ids) & set(human_ids):
        raise ValueError("post-mutation state machine/human ids overlap")
    state_hashes = state.get("artifact_hashes", {})
    if not isinstance(state_hashes, dict):
        raise ValueError("post-mutation state artifact_hashes missing")
    for key, cur_hash in contract["artifact_hashes"].items():
        if str(state_hashes.get(key, "")) != str(cur_hash):
            raise ValueError(f"post-mutation state artifact hash mismatch for {key}")
