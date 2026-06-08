import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_current_to_dest_candidate_full_cycle import (  # noqa: E402
    assert_protected_gretz_can_be_bypassed,
    build_full_cycle_argv,
    parse_args,
    validate_mainline_argv,
)
import scripts.run_current_to_dest_candidate_full_cycle as candidate_runner  # noqa: E402


def _args(tmp_path: Path, *extra: str):
    status = tmp_path / "gretz.status.env"
    argv = [
        "--network-name",
        "plaisir",
        "--network-dir",
        "networks/plaisir",
        "--tag",
        "plaisir_current_to_dest_noincident_test",
        "--protected-gretz-status-file",
        str(status),
        *extra,
    ]
    args, passthrough = parse_args(argv)
    return args, passthrough, status


def test_candidate_runner_refuses_while_gretz_status_is_running(tmp_path):
    args, _passthrough, status = _args(tmp_path)
    status.write_text("STAGE='collect_urb_fair'\nSTATE='running'\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Protected Gretz is still running"):
        assert_protected_gretz_can_be_bypassed(args)


def test_candidate_runner_allows_non_running_gretz_status(tmp_path):
    args, _passthrough, status = _args(tmp_path)
    status.write_text("STAGE='complete'\nSTATE='succeeded'\n", encoding="utf-8")

    assert_protected_gretz_can_be_bypassed(args)


def test_candidate_runner_refuses_missing_gretz_status_by_default(tmp_path):
    args, _passthrough, _status = _args(tmp_path)

    with pytest.raises(RuntimeError, match="status file is missing"):
        assert_protected_gretz_can_be_bypassed(args)


def test_candidate_runner_wait_for_gretz_timeout_does_not_bypass_running_status(tmp_path):
    args, _passthrough, status = _args(
        tmp_path,
        "--wait-for-gretz",
        "--gretz-wait-poll-seconds",
        "0.001",
        "--gretz-wait-timeout-seconds",
        "0.001",
    )
    status.write_text("STAGE='collect_urb_fair'\nSTATE='running'\n", encoding="utf-8")

    with pytest.raises(TimeoutError, match="Timed out waiting"):
        assert_protected_gretz_can_be_bypassed(args)


def test_candidate_runner_wait_for_gretz_polls_until_status_finishes(tmp_path, monkeypatch):
    args, _passthrough, status = _args(
        tmp_path,
        "--wait-for-gretz",
        "--gretz-wait-poll-seconds",
        "1",
        "--gretz-wait-timeout-seconds",
        "5",
    )
    status.write_text("STAGE='collect_urb_fair'\nSTATE='running'\n", encoding="utf-8")
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        status.write_text("STAGE='complete'\nSTATE='succeeded'\n", encoding="utf-8")

    monkeypatch.setattr(candidate_runner.time, "sleep", fake_sleep)

    assert_protected_gretz_can_be_bypassed(args)

    assert sleeps


def test_candidate_runner_builds_current_to_dest_freeflow_mainline(tmp_path):
    args, passthrough, status = _args(tmp_path, "--dry-run", "--", "--marl-log-every", "5")
    status.write_text("STAGE='complete'\nSTATE='succeeded'\n", encoding="utf-8")

    argv = build_full_cycle_argv(args, passthrough)
    validate_mainline_argv(argv)

    assert argv[argv.index("--route-candidate-source") + 1] == "current_to_dest"
    assert argv[argv.index("--wm-candidate-edge-cost-source") + 1] == "freeflow"
    assert argv[argv.index("--eval-candidate-source") + 1] == "current_shortest_freeflow"
    assert argv[argv.index("--marl-route-cost-reward-scope") + 1] == "vehicle_critic"
    assert argv[argv.index("--marl-route-cost-reward-state") + 1] == "current"
    assert argv[argv.index("--marl-batch-episodes") + 1] == "1"
    assert argv[argv.index("--collect-sumo-time-to-teleport") + 1] == "-1"
    assert argv[argv.index("--eval-av-route-cost-policy") + 1] == "none"
    assert "--marl-vehicle-obs-include-route-costs" in argv
    assert "--marl-vehicle-obs-include-fleet-pressure" in argv
    assert "od_suffix" not in argv
    assert "--marl-log-every" in argv
    assert argv[argv.index("--marl-log-every") + 1] == "5"


def test_candidate_runner_defaults_to_no_incident_without_requiring_one(tmp_path):
    args, passthrough, _status = _args(tmp_path)

    argv = build_full_cycle_argv(args, passthrough)
    validate_mainline_argv(argv)

    assert "--no-require-incident" in argv
    assert "--incident-edge=" in argv
    assert argv[argv.index("--incident-speed") + 1] == "0.0"


def test_candidate_runner_passes_dash_prefixed_incident_edge_as_single_token(tmp_path):
    args, passthrough, _status = _args(
        tmp_path,
        "--incident-edge=-221452293#6",
        "--incident-speed",
        "1.0",
    )

    argv = build_full_cycle_argv(args, passthrough)
    validate_mainline_argv(argv)

    assert "--incident-edge=-221452293#6" in argv
    assert "-221452293#6" not in argv


def test_candidate_runner_parses_collect_drain_flags_before_passthrough(tmp_path):
    args, passthrough, _status = _args(
        tmp_path,
        "--collect-completion-drain-steps",
        "9000",
        "--collect-completion-drain-patience-steps",
        "7200",
        "--",
        "--collect-post-mutation-state-in",
        "old/post_mutation_state.json",
    )

    argv = build_full_cycle_argv(args, passthrough)
    validate_mainline_argv(argv)

    assert argv[argv.index("--collect-completion-drain-steps") + 1] == "9000"
    assert argv[argv.index("--collect-completion-drain-patience-steps") + 1] == "7200"
    assert "--" not in argv
    assert "--collect-post-mutation-state-in" in argv


def test_candidate_runner_forwards_sumo_extra_args(tmp_path):
    args, passthrough, _status = _args(
        tmp_path,
        "--collect-sumo-extra-args",
        "--collision.action warn",
        "--eval-sumo-time-to-teleport",
        "-1",
        "--eval-sumo-extra-args",
        "--collision.action warn",
    )

    argv = build_full_cycle_argv(args, passthrough)
    validate_mainline_argv(argv)

    assert argv[argv.index("--collect-sumo-extra-args") + 1] == "--collision.action warn"
    assert argv[argv.index("--eval-sumo-time-to-teleport") + 1] == "-1"
    assert argv[argv.index("--eval-sumo-extra-args") + 1] == "--collision.action warn"
