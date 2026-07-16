#!/usr/bin/env python3
"""
URB-fair evaluation wrapper.

This is a thin entrypoint that forwards into ``scripts/eval_sumo.py`` while
pinning the fairness-critical evaluation defaults needed to compare against
the URB-fair mainline protocol:

- use an explicit experiment settings file
- reuse a shared route/agent artifact directory when configured
- preserve the configured post-mutation human-learning regime
- sample learned CAV routes from the policy distribution in mainline evaluation

The wrapper does not redefine the underlying evaluator; it only makes the
intended fairness-facing settings explicit and reproducible.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SETTINGS = REPO_ROOT / "configs" / "experiments" / "ingolstadt_custom_urb_config1_eval.env"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dreamer4.urban_routing.experiment_settings import load_settings_file
from dreamer4.urban_routing.urb_fair_protocol import build_run_manifest, write_run_manifest


def _has_flag(argv: Iterable[str], flag: str) -> bool:
    return any(str(x) == flag for x in argv)


def _arg_value(argv: Iterable[str], flag: str) -> str | None:
    seq = list(argv)
    for idx, item in enumerate(seq):
        if str(item) == flag and idx + 1 < len(seq):
            return str(seq[idx + 1])
    return None


def _load_eval_sumo_module():
    eval_path = SCRIPT_DIR / "eval_sumo.py"
    spec = importlib.util.spec_from_file_location("eval_sumo_module", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load eval_sumo.py from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_eval_urb_fair_argv(
    *,
    settings_path: str,
    policy_ckpt: str,
    passthrough: Iterable[str] = (),
) -> List[str]:
    cfg = load_settings_file(settings_path)
    extra = list(passthrough)
    argv: List[str] = [
        "eval_sumo.py",
        "--settings",
        settings_path,
        "--policy_ckpt",
        policy_ckpt or cfg.get("MARL_POLICY_CKPT", cfg.get("MARL_CKPT", "")),
    ]

    defaults = {
        "--route_generator": cfg.get("ROUTE_GENERATOR", ""),
        "--reuse_artifacts_dir": cfg.get("ROUTE_ARTIFACTS_DIR", ""),
    }
    for flag, value in defaults.items():
        if value and (not _has_flag(extra, flag)):
            argv.extend([flag, value])

    if not _has_flag(extra, "--av_policy_route_selection"):
        argv.extend(["--av_policy_route_selection", "sample"])

    if not (
        _has_flag(extra, "--disable_human_learning_after_mutation")
        or _has_flag(extra, "--keep_human_learning_after_mutation")
    ):
        argv.append("--disable_human_learning_after_mutation")

    explicit_results_dir = _arg_value(extra, "--results_dir")
    results_dir = Path(explicit_results_dir or cfg.get("EVAL_RESULTS_DIR", "results/eval"))
    post_mutation_state = str(results_dir / "post_mutation_state.json")
    hdv_state_path = str(results_dir / "hdv_learning_state.json")
    if (not _has_flag(extra, "--hdv_state_in")) and (not _has_flag(extra, "--hdv_state_out")):
        argv.extend(["--hdv_state_out", hdv_state_path])
    if (not _has_flag(extra, "--post_mutation_state_in")) and (not _has_flag(extra, "--post_mutation_state_out")):
        argv.extend(["--post_mutation_state_out", post_mutation_state])

    argv.extend(extra)
    return argv


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(
        description="URB-fair wrapper around scripts/eval_sumo.py",
    )
    p.add_argument(
        "--settings",
        type=str,
        default=str(DEFAULT_SETTINGS),
        help="Experiment settings file. Defaults to the URB-config1-aligned eval shell.",
    )
    p.add_argument(
        "--policy_ckpt",
        type=str,
        default="",
        help="Policy checkpoint to evaluate. Defaults to MARL_POLICY_CKPT from settings.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the forwarded eval_sumo argv and exit.",
    )
    return p.parse_known_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args, extra = parse_args(argv)
    settings_path = str(Path(args.settings).resolve())
    forwarded = build_eval_urb_fair_argv(
        settings_path=settings_path,
        policy_ckpt=args.policy_ckpt,
        passthrough=extra,
    )
    if args.dry_run:
        print(" ".join(forwarded))
        return 0

    eval_sumo = _load_eval_sumo_module()
    prev_argv = sys.argv[:]
    try:
        sys.argv = forwarded
        eval_sumo.main()
    finally:
        sys.argv = prev_argv
    manifest = build_run_manifest(settings_path=settings_path, forwarded_argv=forwarded)
    results_dir = Path(manifest["results_dir"])
    manifest_path = write_run_manifest(results_dir, manifest)
    print(f"URB-fair run manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
