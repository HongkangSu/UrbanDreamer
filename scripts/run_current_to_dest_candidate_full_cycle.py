#!/usr/bin/env python3
"""Run a protected current_to_dest/free-flow-candidate full cycle.

This is the stable entry point for second-network experiments. It keeps the
candidate-generation semantics aligned with the Gretz mainline while refusing
to start by default if the protected Gretz run is still marked running.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FULL_CYCLE_SCRIPT = "scripts/run_urb_fair_full_cycle.py"
DEFAULT_GRETZ_STATUS_FILE = (
    "logs/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/"
    "urb_fair_full_cycle.status.env"
)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _today_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _read_status_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _parse_protected_pids(raw: str) -> list[int]:
    pids: list[int] = []
    for item in raw.replace(" ", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            pids.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid protected Gretz PID: {item!r}") from exc
    return pids


def _missing_pids(pids: Sequence[int]) -> list[int]:
    missing: list[int] = []
    for pid in pids:
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            missing.append(int(pid))
        except PermissionError:
            continue
    return missing


def _protected_status_path(args: argparse.Namespace) -> Path:
    status_path = Path(args.protected_gretz_status_file)
    if not status_path.is_absolute():
        status_path = REPO_ROOT / status_path
    return status_path


def _raise_missing_status(status_path: Path) -> None:
    raise RuntimeError(
        f"Protected Gretz status file is missing or empty: {status_path}. "
        "Refusing to start another SUMO/full-cycle job unless "
        "--allow-missing-gretz-status is explicit."
    )


def _raise_missing_pids(pids: Sequence[int]) -> None:
    joined = ", ".join(str(pid) for pid in pids)
    raise RuntimeError(
        f"Protected Gretz PID(s) are missing: {joined}. Refusing to start a "
        "candidate job until the protected run state is understood."
    )


def _positive_float(value: object, name: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return parsed


def _nonnegative_float(value: object, name: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return parsed


def assert_protected_gretz_can_be_bypassed(args: argparse.Namespace) -> None:
    """Raise before any candidate run can compete with the protected Gretz job."""

    status_path = _protected_status_path(args)
    poll_seconds = _positive_float(args.gretz_wait_poll_seconds, "--gretz-wait-poll-seconds")
    timeout_seconds = _nonnegative_float(args.gretz_wait_timeout_seconds, "--gretz-wait-timeout-seconds")
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0.0 else None
    wait_notice_printed = False

    while True:
        status = _read_status_env(status_path)
        if not status:
            if bool(args.allow_missing_gretz_status):
                return
            _raise_missing_status(status_path)

        pids = _parse_protected_pids(str(args.protected_gretz_pids or ""))
        missing = _missing_pids(pids)
        if missing and status.get("STATE") == "running":
            _raise_missing_pids(missing)

        if status.get("STATE") != "running":
            return

        if bool(args.allow_while_gretz_running):
            return

        if not bool(args.wait_for_gretz):
            raise RuntimeError(
                "Protected Gretz is still running "
                f"(stage={status.get('STAGE', 'unknown')}, status={status_path}). "
                "Refusing to start another SUMO/full-cycle job. Use "
                "--wait-for-gretz to queue safely, or --allow-while-gretz-running "
                "only for an intentional override."
            )

        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for protected Gretz to leave STATE='running' "
                f"(stage={status.get('STAGE', 'unknown')}, status={status_path})."
            )

        if not wait_notice_printed:
            print(
                "Protected Gretz is still running "
                f"(stage={status.get('STAGE', 'unknown')}); waiting before "
                "starting candidate full-cycle.",
                file=sys.stderr,
                flush=True,
            )
            wait_notice_printed = True

        sleep_seconds = poll_seconds
        if deadline is not None:
            sleep_seconds = min(sleep_seconds, max(0.0, deadline - time.monotonic()))
        time.sleep(sleep_seconds)


def _default_tag(network_name: str, incident_edge: str) -> str:
    incident_part = "incident" if incident_edge.strip() else "noincident"
    return f"{network_name}_current_to_dest_{incident_part}_{_today_tag()}"


def build_full_cycle_argv(args: argparse.Namespace, passthrough: Sequence[str] = ()) -> list[str]:
    incident_edge = str(args.incident_edge or "").strip()
    incident_speed = str(args.incident_speed)
    if not incident_edge and args.incident_speed_was_default:
        incident_speed = "0.0"

    tag = str(args.tag or _default_tag(args.network_name, incident_edge)).strip()
    checkpoint_episodes = str(args.checkpoint_episodes).strip()

    argv = [
        "--network-dir",
        str(args.network_dir),
        "--network-name",
        str(args.network_name),
        "--route-generator",
        str(args.route_generator),
        "--route-candidate-source",
        "current_to_dest",
        "--wm-candidate-edge-cost-source",
        "freeflow",
        "--eval-candidate-source",
        "current_shortest_freeflow",
        "--ratio-machines",
        str(args.ratio_machines),
        f"--incident-edge={incident_edge}",
        "--incident-start",
        str(args.incident_start),
        "--incident-end",
        str(args.incident_end),
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
        str(args.collect_episodes),
        "--warmup-episodes",
        str(args.warmup_episodes),
        "--collect-max-steps",
        str(args.collect_max_steps),
        "--collect-max-runtime-episodes",
        str(args.collect_max_runtime_episodes),
        "--collect-episode-wall-timeout-seconds",
        str(args.collect_episode_wall_timeout_seconds),
        "--sim-buffer-after-last-depart",
        str(args.sim_buffer_after_last_depart),
        "--collect-sumo-time-to-teleport",
        str(args.collect_sumo_time_to_teleport),
        "--collect-completion-drain-steps",
        str(args.collect_completion_drain_steps),
        "--collect-completion-drain-patience-steps",
        str(args.collect_completion_drain_patience_steps),
        "--collect-sumo-extra-args",
        str(args.collect_sumo_extra_args),
        "--wm-max-steps",
        str(args.wm_max_steps),
        "--wm-save-every",
        str(args.wm_save_every),
        "--wm-eval-every",
        str(args.wm_eval_every),
        "--wm-batch-size",
        str(args.wm_batch_size),
        "--wm-seq-len",
        str(args.wm_seq_len),
        "--wm-grad-accum",
        str(args.wm_grad_accum),
        "--wm-d-model-dyn",
        str(args.wm_d_model_dyn),
        "--wm-dyn-depth",
        str(args.wm_dyn_depth),
        "--wm-n-heads",
        str(args.wm_n_heads),
        "--marl-actor-objective",
        "discrete_actor_critic",
        "--marl-episodes",
        str(args.marl_episodes),
        "--marl-batch-episodes",
        "1",
        "--marl-actor-lr",
        str(args.marl_actor_lr),
        "--marl-dreamer-actor-horizon",
        str(args.marl_dreamer_actor_horizon),
        "--marl-ac-step-advantage-weight",
        "1.0",
        "--marl-ac-route-advantage-weight",
        "0.0",
        "--marl-ac-route-credit-mode",
        "route_cost",
        "--marl-ac-route-credit-base-cost-mode",
        "progress_fraction",
        "--marl-ac-advantage-clip",
        str(args.marl_ac_advantage_clip),
        "--marl-ac-ppo-clip-ratio",
        str(args.marl_ac_ppo_clip_ratio),
        "--marl-ac-entropy-target-frac",
        str(args.marl_ac_entropy_target_frac),
        "--marl-ac-entropy-floor-coef",
        str(args.marl_ac_entropy_floor_coef),
        "--marl-ac-rollout-action-mode",
        "sample",
        "--marl-ac-rollout-argmax-final-mix",
        "1.0",
        "--marl-route-cost-reward-weight",
        "1.0",
        "--marl-route-cost-reward-scope",
        "vehicle_critic",
        "--marl-route-cost-reward-state",
        "current",
        "--marl-remaining-time-progress-weight",
        "0.0",
        "--marl-grpo-route-cost-weight",
        "0.0",
        "--marl-grpo-rollout-weight",
        "0.0",
        "--marl-grpo-rollout-group-size",
        "1",
        "--marl-rollout-cost-reward-weight",
        "0.0",
        "--marl-route-cost-aux-weight",
        "0.0",
        "--marl-actor-trust-region-kl",
        str(args.marl_actor_trust_region_kl),
        "--marl-actor-trust-region-max-argmax-change-frac",
        str(args.marl_actor_trust_region_max_argmax_change_frac),
        "--marl-actor-trust-region-max-route0-frac",
        str(args.marl_actor_trust_region_max_route0_frac),
        "--marl-actor-trust-region-max-retries",
        str(args.marl_actor_trust_region_max_retries),
        "--marl-actor-anchor-start-episode",
        str(args.marl_actor_anchor_start_episode),
        "--marl-actor-anchor-kl",
        str(args.marl_actor_anchor_kl),
        "--marl-actor-anchor-max-argmax-change-frac",
        "0.0",
        "--marl-actor-anchor-max-vehicle-argmax-change-frac",
        "0.0",
        "--marl-actor-anchor-max-route-frac",
        "0.0",
        "--marl-actor-anchor-max-route0-frac",
        str(args.marl_actor_anchor_max_route0_frac),
        "--marl-vehicle-obs-include-route-costs",
        "--marl-vehicle-obs-include-fleet-pressure",
        "--checkpoint-episodes",
        checkpoint_episodes,
        "--marl-checkpoint-every",
        str(args.marl_checkpoint_every),
        "--eval-test-eps",
        str(args.eval_test_eps),
        "--eval-sumo-time-to-teleport",
        str(args.eval_sumo_time_to_teleport),
        "--eval-sumo-extra-args",
        str(args.eval_sumo_extra_args),
        "--eval-av-route-cost-policy",
        "none",
        "--marl-seed",
        str(args.marl_seed),
        "--device",
        str(args.device),
        "--start-stage",
        str(args.start_stage),
        "--python-bin",
        str(args.python_bin),
    ]
    if not bool(args.require_incident):
        argv.append("--no-require-incident")
    if bool(args.dry_run):
        argv.append("--dry-run")
    return [*argv, *passthrough]


def validate_mainline_argv(argv: Sequence[str]) -> None:
    pairs = {
        "--route-candidate-source": "current_to_dest",
        "--wm-candidate-edge-cost-source": "freeflow",
        "--eval-candidate-source": "current_shortest_freeflow",
        "--marl-actor-objective": "discrete_actor_critic",
        "--marl-batch-episodes": "1",
        "--marl-ac-route-advantage-weight": "0.0",
        "--marl-route-cost-reward-scope": "vehicle_critic",
        "--marl-route-cost-reward-state": "current",
        "--marl-grpo-route-cost-weight": "0.0",
        "--marl-grpo-rollout-weight": "0.0",
        "--marl-rollout-cost-reward-weight": "0.0",
        "--marl-route-cost-aux-weight": "0.0",
        "--eval-av-route-cost-policy": "none",
    }
    for flag, expected in pairs.items():
        if flag not in argv:
            raise ValueError(f"Missing required mainline flag: {flag}")
        actual = argv[argv.index(flag) + 1]
        if actual != expected:
            raise ValueError(f"{flag}={actual!r}, expected {expected!r}")
    if "od_suffix" in argv:
        raise ValueError("Deprecated od_suffix candidate source is not allowed in this runner")


def _normalize_passthrough(passthrough: Sequence[str]) -> list[str]:
    values = list(passthrough)
    if values and values[0] == "--":
        return values[1:]
    return values


def parse_args(argv: Iterable[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    default_network = _env("NETWORK_NAME", "plaisir")
    default_network_dir = _env("NETWORK_DIR", f"networks/{default_network}")
    default_incident_speed = _env("INCIDENT_SPEED", "")
    parser.add_argument("--network-name", default=default_network)
    parser.add_argument("--network-dir", default=default_network_dir)
    parser.add_argument("--tag", default=_env("TAG", ""))
    parser.add_argument("--route-generator", default=_env("ROUTE_GENERATOR", "dijkstra"))
    parser.add_argument("--ratio-machines", default=_env("RATIO_MACHINES", "0.4"))
    parser.add_argument("--incident-edge", default=_env("INCIDENT_EDGE", ""))
    parser.add_argument("--incident-start", default=_env("INCIDENT_START", "500"))
    parser.add_argument("--incident-end", default=_env("INCIDENT_END", "1500"))
    parser.add_argument("--incident-speed", default=default_incident_speed or "1.0")
    parser.add_argument("--require-incident", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--collect-episodes", default=_env("COLLECT_EPISODES", "50"))
    parser.add_argument("--warmup-episodes", default=_env("WARMUP_EPISODES", "200"))
    parser.add_argument("--collect-max-steps", default=_env("COLLECT_MAX_STEPS", "3600"))
    parser.add_argument("--collect-max-runtime-episodes", default=_env("COLLECT_MAX_RUNTIME_EPISODES", "1000"))
    parser.add_argument("--collect-episode-wall-timeout-seconds", default=_env("COLLECT_EPISODE_WALL_TIMEOUT_SECONDS", "0"))
    parser.add_argument("--sim-buffer-after-last-depart", default=_env("SIM_BUFFER_AFTER_LAST_DEPART", "1000"))
    parser.add_argument("--collect-sumo-time-to-teleport", default=_env("SUMO_TIME_TO_TELEPORT", "-1"))
    parser.add_argument("--collect-completion-drain-steps", default=_env("COLLECT_COMPLETION_DRAIN_STEPS", "7200"))
    parser.add_argument(
        "--collect-completion-drain-patience-steps",
        default=_env("COLLECT_COMPLETION_DRAIN_PATIENCE_STEPS", "600"),
    )
    parser.add_argument("--collect-sumo-extra-args", default=_env("COLLECT_SUMO_EXTRA_ARGS", ""))
    parser.add_argument("--wm-max-steps", default=_env("WM_MAX_STEPS", "8000"))
    parser.add_argument("--wm-save-every", default=_env("WM_SAVE_EVERY", "1000"))
    parser.add_argument("--wm-eval-every", default=_env("WM_EVAL_EVERY", "1000"))
    parser.add_argument("--wm-batch-size", default=_env("WM_BATCH_SIZE", "1"))
    parser.add_argument("--wm-seq-len", default=_env("WM_SEQ_LEN", "8"))
    parser.add_argument("--wm-grad-accum", default=_env("WM_GRAD_ACCUM", "4"))
    parser.add_argument("--wm-d-model-dyn", default=_env("WM_D_MODEL_DYN", "128"))
    parser.add_argument("--wm-dyn-depth", default=_env("WM_DYN_DEPTH", "4"))
    parser.add_argument("--wm-n-heads", default=_env("WM_N_HEADS", "4"))
    parser.add_argument("--marl-episodes", default=_env("MARL_EPISODES", "400"))
    parser.add_argument("--marl-actor-lr", default=_env("MARL_ACTOR_LR", "2.0e-4"))
    parser.add_argument("--marl-dreamer-actor-horizon", default=_env("ACTOR_HORIZON", "8"))
    parser.add_argument("--marl-ac-advantage-clip", default=_env("AC_ADVANTAGE_CLIP", "2.5"))
    parser.add_argument("--marl-ac-ppo-clip-ratio", default=_env("AC_PPO_CLIP_RATIO", "0.075"))
    parser.add_argument("--marl-ac-entropy-target-frac", default=_env("AC_ENTROPY_TARGET_FRAC", "0.65"))
    parser.add_argument("--marl-ac-entropy-floor-coef", default=_env("AC_ENTROPY_FLOOR_COEF", "0.02"))
    parser.add_argument("--marl-actor-trust-region-kl", default=_env("ACTOR_TRUST_REGION_KL", "0.02"))
    parser.add_argument(
        "--marl-actor-trust-region-max-argmax-change-frac",
        default=_env("ACTOR_TRUST_REGION_MAX_ARGMAX_CHANGE_FRAC", "0.2"),
    )
    parser.add_argument(
        "--marl-actor-trust-region-max-route0-frac",
        default=_env("ACTOR_TRUST_REGION_MAX_ROUTE0_FRAC", "0.0"),
    )
    parser.add_argument("--marl-actor-trust-region-max-retries", default=_env("ACTOR_TRUST_REGION_MAX_RETRIES", "3"))
    parser.add_argument("--marl-actor-anchor-start-episode", default=_env("ACTOR_ANCHOR_START_EPISODE", "60"))
    parser.add_argument("--marl-actor-anchor-kl", default=_env("ACTOR_ANCHOR_KL", "0.05"))
    parser.add_argument("--marl-actor-anchor-max-route0-frac", default=_env("ACTOR_ANCHOR_MAX_ROUTE0_FRAC", "0.9"))
    parser.add_argument("--checkpoint-episodes", default=_env("CHECKPOINT_EPISODES", "100 200 400"))
    parser.add_argument("--marl-checkpoint-every", default=_env("MARL_CHECKPOINT_EVERY", "100"))
    parser.add_argument("--eval-test-eps", default=_env("EVAL_TEST_EPS", "1"))
    parser.add_argument("--eval-sumo-time-to-teleport", default=_env("EVAL_SUMO_TIME_TO_TELEPORT", ""))
    parser.add_argument("--eval-sumo-extra-args", default=_env("EVAL_SUMO_EXTRA_ARGS", ""))
    parser.add_argument("--marl-seed", default=_env("MARL_SEED", "98"))
    parser.add_argument("--device", default=_env("RUN_DEVICE", "cuda:0"))
    parser.add_argument("--start-stage", choices=["collect", "train_dynamics", "train_marl", "eval"], default=_env("START_STAGE", "collect"))
    parser.add_argument("--python-bin", default=_env("PYTHON_BIN", sys.executable))
    parser.add_argument("--cuda-visible-devices", default=_env("CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-command-json", action="store_true")
    parser.add_argument(
        "--protected-gretz-status-file",
        default=_env("GRETZ_STATUS_FILE", DEFAULT_GRETZ_STATUS_FILE),
    )
    parser.add_argument(
        "--protected-gretz-pids",
        default=_env("PROTECTED_GRETZ_PIDS", ""),
        help="Optional comma-separated protected runner PIDs to assert alive before starting.",
    )
    parser.add_argument(
        "--allow-while-gretz-running",
        action=argparse.BooleanOptionalAction,
        default=_env("ALLOW_WHILE_GRETZ_RUNNING", "0") == "1",
    )
    parser.add_argument(
        "--wait-for-gretz",
        action=argparse.BooleanOptionalAction,
        default=_env("WAIT_FOR_GRETZ", "0") == "1",
        help=(
            "Queue the candidate full-cycle by polling the protected Gretz status "
            "file until it is no longer STATE='running'. No SUMO or training "
            "process is started while waiting."
        ),
    )
    parser.add_argument(
        "--gretz-wait-poll-seconds",
        default=_env("GRETZ_WAIT_POLL_SECONDS", "60"),
        help="Polling interval used with --wait-for-gretz.",
    )
    parser.add_argument(
        "--gretz-wait-timeout-seconds",
        default=_env("GRETZ_WAIT_TIMEOUT_SECONDS", "0"),
        help="Optional timeout for --wait-for-gretz; 0 waits indefinitely.",
    )
    parser.add_argument(
        "--allow-missing-gretz-status",
        action=argparse.BooleanOptionalAction,
        default=_env("ALLOW_MISSING_GRETZ_STATUS", "0") == "1",
    )
    args, passthrough = parser.parse_known_args(list(argv) if argv is not None else None)
    args.incident_speed_was_default = not bool(default_incident_speed)
    return args, _normalize_passthrough(passthrough)


def main(argv: Iterable[str] | None = None) -> int:
    args, passthrough = parse_args(argv)
    assert_protected_gretz_can_be_bypassed(args)
    full_cycle_argv = build_full_cycle_argv(args, passthrough)
    validate_mainline_argv(full_cycle_argv)
    command = [str(args.python_bin), "-u", FULL_CYCLE_SCRIPT, *full_cycle_argv]

    if args.print_command_json:
        print(json.dumps({"command": command, "cwd": str(REPO_ROOT)}, indent=2))

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
