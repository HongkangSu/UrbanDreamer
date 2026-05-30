#!/usr/bin/env python3
"""Smoke-test one candidate network without touching the protected Gretz run.

This temporary runner executes only the collection stage from the full-cycle
planner. It keeps the same mainline semantics as the active Gretz run:
current_to_dest dynamic candidates, free-flow candidate generation, and no
od_suffix path. The script also polls the known Gretz PIDs while the smoke test
runs; if any protected PID disappears, the candidate smoke process is stopped.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_urb_fair_full_cycle import (  # noqa: E402
    _bootstrap_route_artifacts,
    build_plan,
    parse_args,
    resolve_paths,
    write_plan_settings,
)


PYTHON_BIN = "/home/wangwenrui/opt/miniconda3/envs/URB/bin/python"
# Only protect the stable Gretz full-cycle runner and collector by default.
# The SUMO wrapper/binary PIDs can legitimately change between episodes.
DEFAULT_PROTECTED_PIDS = "3772964,3773358"
DEFAULT_TAG_DATE = "20260530"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _parse_protected_pids() -> list[int]:
    raw = _env("PROTECTED_GRETZ_PIDS", DEFAULT_PROTECTED_PIDS)
    pids: list[int] = []
    for item in raw.replace(" ", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            pids.append(int(item))
        except ValueError:
            raise ValueError(f"Invalid PROTECTED_GRETZ_PIDS item: {item!r}") from None
    return pids


def _missing_pids(pids: list[int]) -> list[int]:
    missing: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            missing.append(pid)
        except PermissionError:
            continue
    return missing


def _apply_nice() -> int:
    nice_level = int(_env("NICE_LEVEL", "19"))
    if nice_level <= 0:
        return 0
    try:
        return os.nice(nice_level)
    except OSError as exc:
        print(f"[candidate-smoke] warning: failed to apply nice={nice_level}: {exc}", file=sys.stderr)
        return 0


def _ensure_protected_alive(stage: str, pids: list[int]) -> None:
    if not pids:
        return
    missing = _missing_pids(pids)
    if missing:
        joined = ", ".join(str(pid) for pid in missing)
        raise RuntimeError(
            f"Protected Gretz PID(s) missing before/during {stage}: {joined}. "
            "Abort candidate smoke instead of continuing next to a broken Gretz run."
        )


def _network_dir_for(network_name: str) -> str:
    return _env("NETWORK_DIR", f"networks/{network_name}")


def _default_tag(network_name: str) -> str:
    incident = "incident" if _env("INCIDENT_EDGE", "").strip() else "noincident"
    return f"{network_name}_current_to_dest_{incident}_smoke_{DEFAULT_TAG_DATE}"


def _full_cycle_args(network_name: str, network_dir: str, tag: str) -> list[str]:
    incident_edge = _env("INCIDENT_EDGE", "").strip()
    incident_speed = _env("INCIDENT_SPEED", "1.0" if incident_edge else "0.0")
    argv = [
        "--network-dir",
        network_dir,
        "--network-name",
        network_name,
        "--route-generator",
        _env("ROUTE_GENERATOR", "dijkstra"),
        "--route-candidate-source",
        "current_to_dest",
        "--wm-candidate-edge-cost-source",
        "freeflow",
        "--eval-candidate-source",
        "current_shortest_freeflow",
        "--ratio-machines",
        _env("RATIO_MACHINES", "0.4"),
        "--no-require-incident",
        f"--incident-edge={incident_edge}",
        "--incident-start",
        _env("INCIDENT_START", "500"),
        "--incident-end",
        _env("INCIDENT_END", "1500"),
        "--incident-speed",
        incident_speed,
        "--output-root",
        f"data/{tag}",
        "--logs-root",
        f"logs/{tag}",
        "--results-root",
        f"results/{tag}",
        "--generated-config-dir",
        f".urbandreamer/generated/{tag}",
        "--collect-episodes",
        _env("COLLECT_EPISODES", "1"),
        "--warmup-episodes",
        _env("WARMUP_EPISODES", "3"),
        "--collect-max-steps",
        _env("COLLECT_MAX_STEPS", "3600"),
        "--collect-max-runtime-episodes",
        _env("COLLECT_MAX_RUNTIME_EPISODES", "20"),
        "--sim-buffer-after-last-depart",
        _env("SIM_BUFFER_AFTER_LAST_DEPART", "1000"),
        "--collect-sumo-time-to-teleport",
        _env("SUMO_TIME_TO_TELEPORT", "600"),
        "--hdv-gawron-mode",
        _env("HDV_GAWRON_MODE", "routerl"),
        "--hdv-gawron-lr",
        _env("HDV_GAWRON_LR", "0.02"),
        "--hdv-gawron-alpha-zero",
        _env("HDV_GAWRON_ALPHA_ZERO", "0.05"),
        "--hdv-gawron-beta",
        _env("HDV_GAWRON_BETA", "-1.5"),
        "--hdv-gawron-update-mode",
        _env("HDV_GAWRON_UPDATE_MODE", "expected"),
        "--hdv-min-prob",
        _env("HDV_MIN_PROB", "0.02"),
        "--python-bin",
        _env("PYTHON_BIN", PYTHON_BIN),
        "--device",
        _env("RUN_DEVICE", "cuda:0"),
    ]
    agents_csv = _env("AGENTS_CSV", "").strip()
    route_artifacts_dir = _env("ROUTE_ARTIFACTS_DIR", "").strip()
    if agents_csv:
        argv.extend(["--agents-csv", agents_csv])
    if route_artifacts_dir:
        argv.extend(["--route-artifacts-dir", route_artifacts_dir])
    return argv


def _run_collect(command: list[str], log_path: Path, protected_pids: list[int]) -> int:
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("CUDA_VISIBLE_DEVICES", _env("CUDA_VISIBLE_DEVICES", "1"))
    poll_seconds = max(1.0, float(_env("PROTECTED_GRETZ_POLL_SECONDS", "15")))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            while True:
                ret = proc.poll()
                if ret is not None:
                    return int(ret)
                try:
                    _ensure_protected_alive("candidate smoke", protected_pids)
                except Exception:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=30)
                    raise
                time.sleep(poll_seconds)
        finally:
            if proc.poll() is None:
                proc.terminate()


def main() -> int:
    effective_nice = _apply_nice()
    network_name = _env("NETWORK_NAME", "fontainebleau").strip()
    network_dir = _network_dir_for(network_name)
    tag = _env("TAG", _default_tag(network_name)).strip()
    protected_pids = _parse_protected_pids()
    _ensure_protected_alive("startup", protected_pids)

    args = parse_args(_full_cycle_args(network_name, network_dir, tag))
    plan = build_plan(args)
    write_plan_settings(args, plan)

    print(f"[candidate-smoke] network={network_name}")
    print(f"[candidate-smoke] tag={tag}")
    print(f"[candidate-smoke] protected_gretz_pids={','.join(str(pid) for pid in protected_pids)}")
    print(f"[candidate-smoke] effective_nice={effective_nice}")
    print(f"[candidate-smoke] command={' '.join(plan.collect_command)}")

    if _env("DRY_RUN", "0") == "1":
        print("[candidate-smoke] dry-run: not bootstrapping routes or starting SUMO")
        return 0

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    if args.sumo_home:
        env["SUMO_HOME"] = args.sumo_home
    paths = resolve_paths(args)
    _bootstrap_route_artifacts(args, paths, env)

    hdv_state_in = _env("HDV_STATE_IN", "").strip()
    if hdv_state_in:
        plan.collect_command.extend(["--hdv_state_in", hdv_state_in])

    log_path = Path(args.logs_root) / "collect_smoke.log"
    exit_code = _run_collect(plan.collect_command, log_path, protected_pids)
    _ensure_protected_alive("shutdown", protected_pids)
    print(f"[candidate-smoke] log={log_path}")
    print(f"[candidate-smoke] collect exit={exit_code}")
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
