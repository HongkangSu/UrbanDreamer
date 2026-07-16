#!/usr/bin/env python3
"""Run the current online-WM mainline for Gretz, Ingolstadt, and Fontenay.

The pipeline starts from a precomputed post-mutation bundle and runs, in order:

1. frozen-HDV, all-CAV epsilon-greedy warmup collection;
2. 8k-step CFM-current world-model initialization;
3. online-WM actor-critic training with online HDV learning;
4. stochastic SUMO evaluation against both final and original HDV states;
5. separate best-checkpoint selection for each HDV-state protocol.

The source settings and post-mutation JSON files contain paths from the machine
that produced them.  This runner writes portable copies under the new run tag,
so the repository can live at a different absolute path on a rented server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("collect", "wm", "online", "eval")
EVAL_HDV_VARIANTS = ("final", "original")


@dataclass(frozen=True)
class CitySpec:
    key: str
    network_name: str
    settings_rel: str
    post_bundle_rel: str
    sumo_extra_args: str = "--collision.action warn"


CITY_SPECS: Mapping[str, CitySpec] = {
    "gretz_armainvilliers": CitySpec(
        key="gretz_armainvilliers",
        network_name="gretz_armainvilliers",
        settings_rel=(
            ".urbandreamer/generated/"
            "gretz_armainvilliers_current_to_dest_incident_completion100_"
            "stronglivebad_collisionwarn_drain7200_20260601_"
            "neg_986733117_1_S0b_s500_e2500_v0p1/urb_fair_eval.env"
        ),
        post_bundle_rel=(
            "results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/"
            "collect_urb_fair_eval_postmutation_records"
        ),
    ),
    "ingolstadt_custom": CitySpec(
        key="ingolstadt_custom",
        network_name="ingolstadt_custom",
        settings_rel=(
            ".urbandreamer/generated/ingolstadt_custom_current_to_dest_incident_20260528/"
            "urb_fair_eval.env"
        ),
        post_bundle_rel=(
            "results/ingolstadt_custom_current_to_dest_incident_20260528/"
            "collect_urb_fair_eval_postmutation_records"
        ),
    ),
    "fontenay_en_parisis": CitySpec(
        key="fontenay_en_parisis",
        network_name="fontenay_en_parisis",
        settings_rel=(
            ".urbandreamer/generated/"
            "fontenay_en_parisis_current_to_dest_both494_incident_warmup50_"
            "fastcap_20260609/urb_fair_eval.env"
        ),
        post_bundle_rel=(
            "results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_"
            "fastcap_20260609/collect_urb_fair_eval_postmutation_records"
        ),
        sumo_extra_args=(
            "--collision.action warn --ignore-junction-blocker 60 "
            "--tripinfo-output.write-unfinished"
        ),
    ),
}


@dataclass(frozen=True)
class CityPaths:
    run_tag: str
    source_settings: Path
    source_bundle: Path
    source_post_state: Path
    source_hdv_state: Path
    network_dir: Path
    edge_registry: Path
    generated_dir: Path
    portable_settings: Path
    portable_post_state: Path
    provenance_json: Path
    data_dir: Path
    records_dir: Path
    task_name: str
    wm_dir: Path
    online_dir: Path
    pipeline_log: Path
    status_json: Path


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_episodes(raw: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        values = [
            int(part)
            for part in raw.replace(";", ",").replace(" ", ",").split(",")
            if part.strip()
        ]
    else:
        values = [int(value) for value in raw]
    result = tuple(sorted(set(values)))
    if not result or any(value <= 0 for value in result):
        raise ValueError("checkpoint episodes must be positive")
    return result


def _selected_cities(raw: str) -> tuple[CitySpec, ...]:
    names = [part.strip() for part in str(raw).replace(";", ",").split(",") if part.strip()]
    if not names or names == ["all"]:
        names = list(CITY_SPECS)
    unknown = [name for name in names if name not in CITY_SPECS]
    if unknown:
        raise ValueError(f"unknown cities: {unknown}; expected one of {list(CITY_SPECS)}")
    return tuple(CITY_SPECS[name] for name in names)


def city_paths(root: Path, run_id: str, spec: CitySpec) -> CityPaths:
    run_tag = f"{spec.network_name}_{run_id}"
    generated_dir = root / ".urbandreamer" / "generated" / run_tag
    result_root = root / "results" / run_tag
    return CityPaths(
        run_tag=run_tag,
        source_settings=root / spec.settings_rel,
        source_bundle=root / spec.post_bundle_rel,
        source_post_state=root / spec.post_bundle_rel / "post_mutation_state.json",
        source_hdv_state=(
            root / spec.post_bundle_rel / "post_mutation_hdv_learning_state.json"
        ),
        network_dir=root / "networks" / spec.network_name,
        edge_registry=root / "networks" / spec.network_name / "edge_registry.json",
        generated_dir=generated_dir,
        portable_settings=generated_dir / "online_wm_mainline.env",
        portable_post_state=result_root / "portable_inputs" / "post_mutation_state.json",
        provenance_json=result_root / "portable_inputs" / "input_provenance.json",
        data_dir=root / "data" / run_tag / "collect_urb_fair_eval_postmutation",
        records_dir=result_root / "collect_urb_fair_eval_postmutation_records",
        task_name=f"{spec.network_name}_online_wm_mainline_warmup",
        wm_dir=root / "logs" / run_tag / "dynamics_cfm_current_8k",
        online_dir=result_root / "online_wm_mainline",
        pipeline_log=root / "logs" / run_tag / "full_pipeline.log",
        status_json=result_root / "pipeline_status.json",
    )


def required_inputs(root: Path, spec: CitySpec) -> tuple[Path, ...]:
    paths = city_paths(root, "required_input_probe", spec)
    return (
        paths.source_settings,
        paths.source_post_state,
        paths.source_hdv_state,
        paths.source_bundle / "agents.csv",
        paths.source_bundle / "routes.csv",
        paths.source_bundle / "route.rou.xml",
        paths.network_dir / "agents.csv",
        paths.network_dir / f"od_{spec.network_name}.txt",
        paths.network_dir / f"{spec.network_name}.net.xml",
        paths.edge_registry,
    )


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        try:
            parsed = shlex.split(raw_value, comments=False, posix=True)
            values[key] = parsed[0] if parsed else ""
        except ValueError:
            values[key] = raw_value.strip().strip('"').strip("'")
    return values


def _write_portable_settings(
    *,
    root: Path,
    spec: CitySpec,
    paths: CityPaths,
) -> None:
    source_text = paths.source_settings.read_text(encoding="utf-8")
    parsed = _read_env(paths.source_settings)
    old_network_dir = Path(parsed.get("NETWORK_DIR", ""))
    if old_network_dir.is_absolute() and len(old_network_dir.parents) >= 2:
        old_root = old_network_dir.parents[1]
        source_text = source_text.replace(str(old_root), str(root))

    overrides = {
        "NETWORK_DIR": paths.network_dir,
        "NETWORK_NAME": spec.network_name,
        "EDGE_REGISTRY": paths.edge_registry,
        "AGENTS_CSV": paths.network_dir / "agents.csv",
        "OD_FILE": paths.network_dir / f"od_{spec.network_name}.txt",
        "ROUTE_ARTIFACTS_DIR": paths.source_bundle,
        "FLOW_DATA_DIR": paths.data_dir,
        "FLOW_TASK_NAME": paths.task_name,
        "DYNAMICS_LOG_DIR": paths.wm_dir,
        "PROJECTOR_CKPT": paths.wm_dir / "flow_projector_latest.pt",
        "DYNAMICS_CKPT": paths.wm_dir / "latest.pt",
        "MARL_CKPT_DIR": paths.online_dir / "checkpoints",
        "EVAL_RESULTS_DIR": paths.online_dir / "eval",
        "WM_CFM_INIT_MODE": "current",
        "WM_POLICY_OBS_SOURCE": "decoder_volume_tt",
        "WM_EDGE_COST_SOURCE": "decoded_speed",
        "WM_CANDIDATE_EDGE_COST_SOURCE": "freeflow",
        "AV_CANDIDATE_SOURCE": "current_to_dest",
    }
    pending = {key: str(value) for key, value in overrides.items()}
    output_lines: list[str] = []
    assignment = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")
    for line in source_text.splitlines():
        match = assignment.match(line.strip())
        if match and match.group(1) in pending:
            key = match.group(1)
            output_lines.append(f"{key}={json.dumps(pending.pop(key))}")
        else:
            output_lines.append(line)
    for key, value in pending.items():
        output_lines.append(f"{key}={json.dumps(value)}")
    paths.portable_settings.parent.mkdir(parents=True, exist_ok=True)
    paths.portable_settings.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def prepare_city_inputs(root: Path, spec: CitySpec, paths: CityPaths) -> None:
    missing = [path for path in required_inputs(root, spec) if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"missing required inputs for {spec.key}:\n{formatted}")

    state = json.loads(paths.source_post_state.read_text(encoding="utf-8"))
    if str(state.get("network", "")) != spec.network_name:
        raise ValueError(
            f"post-mutation network mismatch for {spec.key}: {state.get('network')!r}"
        )
    if bool(state.get("should_humans_adapt", False)):
        raise ValueError(f"post-mutation state must be frozen for warmup: {paths.source_post_state}")
    state["human_learning_state_ref"] = str(paths.source_hdv_state.resolve())
    state["artifact_refs"] = {
        "agents_csv": str((paths.source_bundle / "agents.csv").resolve()),
        "routes_csv": str((paths.source_bundle / "routes.csv").resolve()),
        "route_rou_xml": str((paths.source_bundle / "route.rou.xml").resolve()),
    }
    paths.portable_post_state.parent.mkdir(parents=True, exist_ok=True)
    paths.portable_post_state.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_portable_settings(root=root, spec=spec, paths=paths)
    provenance = {
        "schema_version": 1,
        "prepared_at": _now(),
        "city": spec.key,
        "source_settings": str(paths.source_settings.resolve()),
        "source_post_mutation_state": str(paths.source_post_state.resolve()),
        "source_post_mutation_state_sha256": _sha256(paths.source_post_state),
        "source_hdv_state": str(paths.source_hdv_state.resolve()),
        "source_hdv_state_sha256": _sha256(paths.source_hdv_state),
        "source_bundle": str(paths.source_bundle.resolve()),
        "portable_settings": str(paths.portable_settings.resolve()),
        "portable_post_mutation_state": str(paths.portable_post_state.resolve()),
        "protocol": {
            "warmup": "all_fixed_cavs_epsilon_greedy",
            "warmup_epsilon": 0.4,
            "warmup_random_av_fraction": 0.0,
            "warmup_hdv_learning": False,
            "online_real_action": "masked_policy_sample_one_hot",
            "imagined_action": "masked_policy_sample_one_hot",
            "eval_action": "masked_policy_sample",
            "eval_hdv_variants": {
                "final": {
                    "state": "final_online_training_state",
                    "human_learning_after_mutation": True,
                },
                "original": {
                    "state": "source_post_mutation_state",
                    "human_learning_after_mutation": False,
                },
            },
            "train_sumo_extra_args": spec.sumo_extra_args,
            "eval_sumo_extra_args": spec.sumo_extra_args,
        },
    }
    paths.provenance_json.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_collect_command(
    *,
    python_bin: str,
    root: Path,
    spec: CitySpec,
    paths: CityPaths,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python_bin,
        "-u",
        str(root / "scripts" / "collect_urb_fair.py"),
        "--settings",
        str(paths.portable_settings),
        "--output_dir",
        str(paths.data_dir),
        "--records_dir",
        str(paths.records_dir),
        "--task_name",
        paths.task_name,
        "--reuse_artifacts_dir",
        str(paths.source_bundle),
        "--post_mutation_state_in",
        str(paths.portable_post_state),
        "--hdv_state_in",
        str(paths.source_hdv_state),
        "--num_episodes",
        str(args.collect_episodes),
        "--warmup_episodes",
        "0",
        "--max_steps",
        str(args.collect_max_steps),
        "--max_runtime_episodes",
        str(max(int(args.collect_episodes) * 20, int(args.collect_episodes))),
        "--episode_wall_timeout_seconds",
        str(args.episode_wall_timeout_seconds),
        "--random_av_fraction",
        "0.0",
        "--epsilon",
        "0.4",
        "--safe_reroute_min_remaining_m",
        "35",
        "--completion_drain_steps",
        str(args.collect_completion_drain_steps),
        "--completion_drain_patience_steps",
        str(args.collect_completion_drain_patience_steps),
        "--sumo_time_to_teleport",
        "-1",
        "--checkpoint_every",
        "10",
        "--disable_post_mutation_hdv_learning",
        "--seed",
        str(args.seed),
    ]
    if spec.sumo_extra_args:
        command.extend(["--sumo_extra_args", spec.sumo_extra_args])
    return command


def build_wm_command(
    *,
    python_bin: str,
    root: Path,
    paths: CityPaths,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python_bin,
        "-u",
        str(root / "dreamer4" / "train_dynamics.py"),
        "--settings",
        str(paths.portable_settings),
        "--edge_registry",
        str(paths.edge_registry),
        "--data_dirs",
        str(paths.data_dir),
        "--flow_task_name",
        paths.task_name,
        "--batch_size",
        str(args.wm_batch_size),
        "--seq_len",
        str(args.wm_seq_len),
        "--grad_accum",
        str(args.wm_grad_accum),
        "--num_workers",
        "0",
        "--d_model_dyn",
        "256",
        "--dyn_depth",
        "6",
        "--n_heads",
        "4",
        "--use_actions",
        "--joint_route_action",
        "--n_routes",
        "4",
        "--action_key",
        "av_planned_edge_flow",
        "--aux_action_key",
        "hdv_edge_action",
        "--action_encoder_type",
        "graph_cond",
        "--edge_action_injection",
        "--edge_action_value_norm",
        "active_count",
        "--hdv_policy_weight",
        "0.05",
        "--hdv_policy_target_mode",
        "episode_mean",
        "--hdv_policy_action_mode",
        "window_mean",
        "--transition_hdv_context_mode",
        "episode_response",
        "--transition_hdv_context_weight",
        "1.0",
        "--transition_hdv_context_scale",
        "1.0",
        "--graph_cond_edge_conditioning",
        "--graph_cond_edge_conditioning_scale",
        "1.0",
        "--decoder_recon_weight",
        "0.20",
        "--packing_factor",
        "1",
        "--k_max",
        "8",
        "--n_ode_steps",
        "1",
        "--max_steps",
        str(args.wm_steps),
        "--save_every",
        "1000",
        "--log_every",
        "200",
        "--eval_every",
        "1000",
        "--transition_decoded_supervision_weight",
        "1.0",
        "--bpr_supervision_weight",
        "0.10",
        "--t_all_supervision_weight",
        "0.20",
        "--decoder_speed_weight",
        "4.0",
        "--decoder_volume_weight",
        "2.0",
        "--transition_cfm_init_mode",
        "current",
        "--adm_cfm_init_mode",
        "current",
        "--wandb_mode",
        "offline",
        "--wandb_run_name",
        f"{paths.run_tag}_wm_cfm_current",
        "--ckpt_dir",
        str(paths.wm_dir),
    ]
    final_step = paths.wm_dir / f"step_{int(args.wm_steps):07d}.pt"
    latest = paths.wm_dir / "latest.pt"
    if latest.exists() and not final_step.exists():
        command.extend(["--resume", str(latest)])
    return command


def build_online_command(
    *,
    python_bin: str,
    root: Path,
    spec: CitySpec,
    paths: CityPaths,
    checkpoints: Sequence[int],
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python_bin,
        "-u",
        str(root / "ablation" / "run_gretz_online_wm_dreamer_ablation.py"),
        "--settings",
        str(paths.portable_settings),
        "--output-dir",
        str(paths.online_dir),
        "--tag",
        paths.run_tag,
        "--network-name",
        spec.network_name,
        "--network-dir",
        str(paths.network_dir),
        "--edge-registry",
        str(paths.edge_registry),
        "--route-artifacts-dir",
        str(paths.records_dir),
        "--post-mutation-state-in",
        str(paths.portable_post_state),
        "--hdv-state-in",
        str(paths.source_hdv_state),
        "--projector-ckpt",
        str(paths.wm_dir / "flow_projector_latest.pt"),
        "--dynamics-ckpt",
        str(paths.wm_dir / "latest.pt"),
        "--wm-cfm-init-mode",
        "current",
        "--wm-hdv-target-source",
        "static_edge_demand",
        "--imagined-hdv-context-source",
        "causal_response_predictor",
        "--enable-hdv-response-predictor",
        "--hdv-response-predictor-fallback",
        "prev_hdv",
        "--hdv-response-predictor-update-epochs",
        "64",
        "--hdv-response-predictor-min-pairs",
        "2",
        "--ratio-machines",
        "0.4",
        "--av-candidate-source",
        "current_shortest_freeflow",
        "--route-generator",
        "dijkstra",
        "--train-episodes",
        str(args.online_episodes),
        "--eval-episodes",
        str(args.eval_episodes),
        "--checkpoint-episodes",
        ",".join(str(value) for value in checkpoints),
        "--episode-steps",
        "60",
        "--sim-buffer-after-last-depart",
        str(args.train_sim_buffer_after_last_depart),
        "--reward-mode",
        "capped_step_cost",
        "--gamma",
        "1.0",
        "--real-policy-route-selection",
        "sample",
        "--eval-policy-route-selection",
        "sample",
        "--global-advantage-weight",
        "1.0",
        "--imag-horizon",
        "8",
        "--imagined-updates-per-episode",
        "1",
        "--imagined-update-passes-per-collect",
        "16",
        "--imagined-rollouts-per-update",
        "8",
        "--wm-replay-buffer-episodes",
        "8",
        "--wm-updates-per-collect",
        "4",
        "--wm-update-epochs",
        "1",
        "--wm-batch-size",
        "16",
        "--disable-vehicle-arrival-head",
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--eval-python",
        python_bin,
        "--keep-human-learning-after-mutation",
        "--skip-eval",
    ]
    if spec.sumo_extra_args:
        command.extend(["--sumo-extra-args", spec.sumo_extra_args])
    return command


def build_eval_command(
    *,
    python_bin: str,
    root: Path,
    spec: CitySpec,
    paths: CityPaths,
    episode: int,
    args: argparse.Namespace,
    hdv_variant: str = "final",
) -> list[str]:
    if hdv_variant not in EVAL_HDV_VARIANTS:
        raise ValueError(
            f"unknown eval HDV variant {hdv_variant!r}; expected {EVAL_HDV_VARIANTS}"
        )
    ep = int(episode)
    suffix = f"{ep:05d}"
    checkpoint = paths.online_dir / "checkpoints" / f"checkpoint_ep{suffix}.pt"
    wm_dir = paths.online_dir / "checkpoints" / f"wm_ep{suffix}"
    eval_dir_name = "eval" if hdv_variant == "final" else "eval_original_hdv"
    eval_root = paths.online_dir / eval_dir_name
    result_dir = eval_root / f"checkpoint_ep{suffix}"
    if hdv_variant == "final":
        hdv_state = paths.online_dir / "hdv_learning_state_after_online_wm_train.json"
        human_learning_flag = "--keep_human_learning_after_mutation"
    else:
        hdv_state = paths.source_hdv_state
        human_learning_flag = "--disable_human_learning_after_mutation"
    command = [
        python_bin,
        "-u",
        str(root / "scripts" / "eval_sumo.py"),
        "--settings",
        str(paths.portable_settings),
        "--policy_ckpt",
        str(checkpoint),
        "--projector_ckpt",
        str(wm_dir / f"wm_projector_ep{suffix}.pt"),
        "--dynamics_ckpt",
        str(wm_dir / f"wm_dynamics_ep{suffix}.pt"),
        "--edge_registry",
        str(paths.edge_registry),
        "--network",
        spec.network_name,
        "--network_folder",
        str(paths.network_dir),
        "--n_routes",
        "4",
        "--ratio_machines",
        "0.4",
        "--human_learning_episodes",
        "0",
        "--test_eps",
        str(args.eval_episodes),
        "--results_dir",
        str(result_dir),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--episode_steps",
        "60",
        "--sim_buffer_after_last_depart",
        str(args.eval_sim_buffer_after_last_depart),
        "--max_steps_per_episode",
        str(args.eval_max_steps_per_episode),
        "--reuse_artifacts_dir",
        str(paths.records_dir),
        "--post_mutation_state_in",
        str(paths.portable_post_state),
        "--hdv_state_in",
        str(hdv_state),
        "--av_candidate_source",
        "current_shortest_freeflow",
        "--av_policy_route_selection",
        "sample",
        "--wm_policy_obs_source",
        "decoder_volume_tt",
        "--wm_edge_cost_source",
        "decoded_speed",
        "--wm_candidate_edge_cost_source",
        "freeflow",
        "--wm_cfm_init_mode",
        "current",
        "--mpc_horizon",
        "0",
        "--av_route_cost_policy",
        "none",
        "--sumo_time_to_teleport",
        "-1",
        human_learning_flag,
        "--deterministic",
    ]
    if spec.sumo_extra_args:
        command.extend(["--sumo_extra_args", spec.sumo_extra_args])
    return command


def _command_text(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def run_command(
    command: Sequence[str],
    *,
    root: Path,
    env: Mapping[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    rendered = _command_text(command)
    print(f"[{_now()}] $ {rendered}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_now()}] $ {rendered}\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=str(root),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def _write_status(paths: CityPaths, *, stage: str, state: str, **extra: Any) -> None:
    payload = {
        "schema_version": 1,
        "updated_at": _now(),
        "run_tag": paths.run_tag,
        "stage": stage,
        "state": state,
        **extra,
    }
    paths.status_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.status_json.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(paths.status_json)


def _collection_complete(paths: CityPaths, expected_episodes: int) -> bool:
    manifest_path = paths.records_dir / "urb_fair_collection_manifest.json"
    dataset_path = paths.data_dir / f"{paths.task_name}.pt"
    if not manifest_path.exists() or not dataset_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        not bool(manifest.get("partial", True))
        and int(manifest.get("episodes_saved", 0)) >= int(expected_episodes)
    )


def _wm_complete(paths: CityPaths, expected_steps: int) -> bool:
    return all(
        path.exists()
        for path in (
            paths.wm_dir / f"step_{int(expected_steps):07d}.pt",
            paths.wm_dir / "latest.pt",
            paths.wm_dir / "flow_projector_latest.pt",
        )
    )


def _online_complete(paths: CityPaths, final_episode: int) -> bool:
    return all(
        path.exists()
        for path in (
            paths.online_dir / "checkpoints" / f"checkpoint_ep{int(final_episode):05d}.pt",
            paths.online_dir / "online_wm_train_summary.json",
            paths.online_dir / "online_wm_dreamer_final_summary.json",
            paths.online_dir / "hdv_learning_state_after_online_wm_train.json",
        )
    )


def _eval_root(paths: CityPaths, hdv_variant: str) -> Path:
    if hdv_variant == "final":
        return paths.online_dir / "eval"
    if hdv_variant == "original":
        return paths.online_dir / "eval_original_hdv"
    raise ValueError(f"unknown eval HDV variant {hdv_variant!r}")


def _eval_summary_path(
    paths: CityPaths,
    episode: int,
    hdv_variant: str = "final",
) -> Path:
    return (
        _eval_root(paths, hdv_variant)
        / f"checkpoint_ep{int(episode):05d}"
        / "collector_eval_summary.json"
    )


def _eval_complete(
    paths: CityPaths,
    episode: int,
    eval_episodes: int,
    hdv_variant: str = "final",
) -> bool:
    summary_path = _eval_summary_path(paths, episode, hdv_variant)
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return (
        str(summary.get("av_policy_route_selection", "")) == "sample"
        and int(summary.get("test_eps", 0)) == int(eval_episodes)
    )


def _write_eval_manifest(
    paths: CityPaths,
    checkpoints: Sequence[int],
    hdv_variant: str = "final",
) -> Path:
    if hdv_variant == "final":
        hdv_state = paths.online_dir / "hdv_learning_state_after_online_wm_train.json"
        human_learning_after_mutation = True
        output = paths.online_dir / "online_wm_eval_summary.json"
    elif hdv_variant == "original":
        hdv_state = paths.source_hdv_state
        human_learning_after_mutation = False
        output = paths.online_dir / "online_wm_eval_original_hdv_summary.json"
    else:
        raise ValueError(f"unknown eval HDV variant {hdv_variant!r}")
    payload = {
        str(int(episode)): {
            "checkpoint": str(
                paths.online_dir / "checkpoints" / f"checkpoint_ep{int(episode):05d}.pt"
            ),
            "hdv_state_variant": hdv_variant,
            "hdv_state_in": str(hdv_state),
            "human_learning_after_mutation": human_learning_after_mutation,
            "results_dir": str(
                _eval_root(paths, hdv_variant) / f"checkpoint_ep{int(episode):05d}"
            ),
            "summary_json": str(_eval_summary_path(paths, episode, hdv_variant)),
            "policy_route_selection": "sample",
        }
        for episode in checkpoints
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def _base_environment(args: argparse.Namespace, root: Path, python_bin: str) -> dict[str, str]:
    env = dict(os.environ)
    python_path = Path(python_bin).resolve()
    inferred_sumo_home = python_path.parent.parent
    sumo_home = Path(args.sumo_home).resolve() if args.sumo_home else inferred_sumo_home
    env["SUMO_HOME"] = str(sumo_home)
    env["PATH"] = f"{sumo_home / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{root}:{env.get('PYTHONPATH', '')}"
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-urb")
    env.setdefault("PYTHONHASHSEED", str(args.seed))
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    return env


def _resolve_python(raw: str) -> str:
    candidate = shutil.which(raw)
    if candidate:
        return str(Path(candidate).resolve())
    path = Path(raw).expanduser()
    if path.exists():
        return str(path.resolve())
    raise FileNotFoundError(f"Python executable not found: {raw}")


def runtime_preflight(
    *,
    root: Path,
    python_bin: str,
    env: Mapping[str, str],
    device: str,
) -> None:
    required_code = (
        root / "scripts" / "collect_urb_fair.py",
        root / "dreamer4" / "train_dynamics.py",
        root / "ablation" / "run_gretz_online_wm_dreamer_ablation.py",
        root / "scripts" / "eval_sumo.py",
        root / "scripts" / "select_best_eval_summary.py",
    )
    missing_code = [path for path in required_code if not path.exists()]
    if missing_code:
        raise FileNotFoundError(f"missing pipeline code: {missing_code}")
    sumo_binary = Path(env["SUMO_HOME"]) / "bin" / "sumo"
    if not sumo_binary.exists():
        raise FileNotFoundError(
            f"SUMO binary not found at {sumo_binary}; pass --sumo-home or set SUMO_HOME"
        )
    probe = (
        "import json, torch; "
        "print(json.dumps({'torch': torch.__version__, "
        "'cuda_available': torch.cuda.is_available(), "
        "'cuda_devices': torch.cuda.device_count()}))"
    )
    completed = subprocess.run(
        [python_bin, "-c", probe],
        cwd=str(root),
        env=dict(env),
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"Runtime: {completed.stdout.strip()}", flush=True)
    if str(device).lower().startswith("cuda"):
        runtime = json.loads(completed.stdout.strip().splitlines()[-1])
        if not bool(runtime.get("cuda_available")) or int(runtime.get("cuda_devices", 0)) <= 0:
            raise RuntimeError("CUDA device requested, but the selected Python environment sees no GPU")


def _stage_enabled(start_stage: str, stage: str) -> bool:
    return STAGES.index(stage) >= STAGES.index(start_stage)


def run_city(
    *,
    root: Path,
    python_bin: str,
    spec: CitySpec,
    checkpoints: Sequence[int],
    args: argparse.Namespace,
    env: Mapping[str, str],
) -> dict[str, Any]:
    paths = city_paths(root, args.run_id, spec)
    prepare_city_inputs(root, spec, paths)

    def write_status(*, stage: str, state: str, **extra: Any) -> None:
        if not args.dry_run:
            _write_status(paths, stage=stage, state=state, **extra)

    write_status(stage="preflight", state="complete")
    print(f"\n{'=' * 88}\n[{spec.key}] run_tag={paths.run_tag}\n{'=' * 88}", flush=True)

    if _stage_enabled(args.start_stage, "collect"):
        if _collection_complete(paths, args.collect_episodes):
            print(f"[{spec.key}] collection already complete; skipping", flush=True)
        else:
            if paths.data_dir.exists() and any(paths.data_dir.iterdir()):
                raise RuntimeError(
                    f"incomplete collection output exists at {paths.data_dir}; "
                    "use a new --run-id instead of overwriting it"
                )
            write_status(stage="collect", state="running")
            run_command(
                build_collect_command(
                    python_bin=python_bin,
                    root=root,
                    spec=spec,
                    paths=paths,
                    args=args,
                ),
                root=root,
                env=env,
                log_path=paths.pipeline_log,
                dry_run=args.dry_run,
            )
            if not args.dry_run and not _collection_complete(paths, args.collect_episodes):
                raise RuntimeError(f"collection did not produce a complete manifest: {paths.run_tag}")
            write_status(stage="collect", state="complete")

    if _stage_enabled(args.start_stage, "wm"):
        if _wm_complete(paths, args.wm_steps):
            print(f"[{spec.key}] world-model initialization already complete; skipping", flush=True)
        else:
            if not args.dry_run and not _collection_complete(paths, args.collect_episodes):
                raise RuntimeError(f"world-model stage requires complete collection: {paths.run_tag}")
            write_status(stage="wm", state="running")
            run_command(
                build_wm_command(
                    python_bin=python_bin,
                    root=root,
                    paths=paths,
                    args=args,
                ),
                root=root,
                env=env,
                log_path=paths.pipeline_log,
                dry_run=args.dry_run,
            )
            if not args.dry_run and not _wm_complete(paths, args.wm_steps):
                raise RuntimeError(f"world-model stage is incomplete: {paths.run_tag}")
            write_status(stage="wm", state="complete")

    if _stage_enabled(args.start_stage, "online"):
        if _online_complete(paths, args.online_episodes):
            print(f"[{spec.key}] online-WM training already complete; skipping", flush=True)
        else:
            if not args.dry_run and not _wm_complete(paths, args.wm_steps):
                raise RuntimeError(f"online stage requires complete world model: {paths.run_tag}")
            if paths.online_dir.exists() and any(paths.online_dir.iterdir()):
                raise RuntimeError(
                    f"incomplete online output exists at {paths.online_dir}; online HDV state "
                    "cannot be safely reconstructed from a policy checkpoint, so use a new --run-id"
                )
            write_status(stage="online", state="running")
            run_command(
                build_online_command(
                    python_bin=python_bin,
                    root=root,
                    spec=spec,
                    paths=paths,
                    checkpoints=checkpoints,
                    args=args,
                ),
                root=root,
                env=env,
                log_path=paths.pipeline_log,
                dry_run=args.dry_run,
            )
            if not args.dry_run and not _online_complete(paths, args.online_episodes):
                raise RuntimeError(f"online-WM training is incomplete: {paths.run_tag}")
            write_status(stage="online", state="complete")

    best_selections = {
        "final": paths.online_dir / "best_checkpoint_selection.json",
        "original": paths.online_dir / "best_checkpoint_selection_original_hdv.json",
    }
    if _stage_enabled(args.start_stage, "eval"):
        if not args.dry_run and not _online_complete(paths, args.online_episodes):
            raise RuntimeError(f"eval stage requires complete online training: {paths.run_tag}")
        write_status(stage="eval", state="running")
        for hdv_variant in EVAL_HDV_VARIANTS:
            print(f"[{spec.key}] eval HDV variant: {hdv_variant}", flush=True)
            for episode in checkpoints:
                if _eval_complete(
                    paths,
                    episode,
                    args.eval_episodes,
                    hdv_variant,
                ):
                    print(
                        f"[{spec.key}] {hdv_variant}-HDV eval checkpoint {episode} "
                        "already complete; skipping",
                        flush=True,
                    )
                    continue
                run_command(
                    build_eval_command(
                        python_bin=python_bin,
                        root=root,
                        spec=spec,
                        paths=paths,
                        episode=episode,
                        args=args,
                        hdv_variant=hdv_variant,
                    ),
                    root=root,
                    env=env,
                    log_path=paths.pipeline_log,
                    dry_run=args.dry_run,
                )
                if not args.dry_run and not _eval_complete(
                    paths,
                    episode,
                    args.eval_episodes,
                    hdv_variant,
                ):
                    raise RuntimeError(
                        f"stochastic {hdv_variant}-HDV eval did not produce the expected "
                        f"summary: {spec.key} ep{episode}"
                    )
            if not args.dry_run:
                _write_eval_manifest(paths, checkpoints, hdv_variant)
            selection_command = [
                python_bin,
                str(root / "scripts" / "select_best_eval_summary.py"),
                "--output",
                str(best_selections[hdv_variant]),
                *[
                    str(
                        _eval_root(paths, hdv_variant)
                        / f"checkpoint_ep{int(episode):05d}"
                    )
                    for episode in checkpoints
                ],
            ]
            run_command(
                selection_command,
                root=root,
                env=env,
                log_path=paths.pipeline_log,
                dry_run=args.dry_run,
            )
        write_status(
            stage="eval",
            state="complete",
            best_checkpoint_selections={
                variant: str(path) for variant, path in best_selections.items()
            },
        )

    result: dict[str, Any] = {
        "city": spec.key,
        "run_tag": paths.run_tag,
        "train_sumo_extra_args": spec.sumo_extra_args,
        "eval_sumo_extra_args": spec.sumo_extra_args,
        "status": str(paths.status_json),
        "collection_manifest": str(paths.records_dir / "urb_fair_collection_manifest.json"),
        "world_model": str(paths.wm_dir / "latest.pt"),
        "online_summary": str(paths.online_dir / "online_wm_train_summary.json"),
        "final_hdv_state": str(paths.online_dir / "hdv_learning_state_after_online_wm_train.json"),
        "eval_summary": str(paths.online_dir / "online_wm_eval_summary.json"),
        "eval_original_hdv_summary": str(
            paths.online_dir / "online_wm_eval_original_hdv_summary.json"
        ),
        "best_checkpoint_selection": str(best_selections["final"]),
        "best_checkpoint_selection_original_hdv": str(best_selections["original"]),
    }
    if best_selections["final"].exists():
        selection = json.loads(best_selections["final"].read_text(encoding="utf-8"))
        result["best"] = selection.get("best", {})
    if best_selections["original"].exists():
        selection = json.loads(best_selections["original"].read_text(encoding="utf-8"))
        result["best_original_hdv"] = selection.get("best", {})
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the current online-WM mainline and stochastic eval on three cities."
    )
    parser.add_argument(
        "--cities",
        default="all",
        help="Comma-separated city names or 'all'.",
    )
    parser.add_argument("--run-id", default="online_wm_mainline_20260716")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--sumo-home", default=os.environ.get("SUMO_HOME", ""))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collect-episodes", type=int, default=50)
    parser.add_argument("--collect-max-steps", type=int, default=3600)
    parser.add_argument("--collect-completion-drain-steps", type=int, default=7200)
    parser.add_argument("--collect-completion-drain-patience-steps", type=int, default=600)
    parser.add_argument("--episode-wall-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--wm-steps", type=int, default=8000)
    parser.add_argument("--wm-batch-size", type=int, default=2)
    parser.add_argument("--wm-seq-len", type=int, default=16)
    parser.add_argument("--wm-grad-accum", type=int, default=16)
    parser.add_argument("--online-episodes", type=int, default=400)
    parser.add_argument("--checkpoint-episodes", default="50,100,200,400")
    parser.add_argument("--train-sim-buffer-after-last-depart", type=int, default=3000)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Stochastic SUMO episodes per checkpoint and HDV-state variant.",
    )
    parser.add_argument("--eval-sim-buffer-after-last-depart", type=int, default=7200)
    parser.add_argument("--eval-max-steps-per-episode", type=int, default=9000)
    parser.add_argument("--start-stage", choices=STAGES, default="collect")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--print-required-inputs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = Path(args.repo_root).expanduser().resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(args.run_id)):
        raise ValueError("--run-id may contain only letters, digits, '.', '_' and '-'")
    cities = _selected_cities(args.cities)
    checkpoints = _checkpoint_episodes(args.checkpoint_episodes)
    if max(checkpoints) > int(args.online_episodes):
        raise ValueError("checkpoint episodes cannot exceed --online-episodes")
    if int(args.online_episodes) not in checkpoints:
        raise ValueError("--checkpoint-episodes must include the final online episode")

    if args.print_required_inputs:
        for spec in cities:
            print(f"[{spec.key}]")
            for path in required_inputs(root, spec):
                print(path)
        return 0

    python_bin = _resolve_python(args.python_bin)
    env = _base_environment(args, root, python_bin)
    runtime_preflight(
        root=root,
        python_bin=python_bin,
        env=env,
        device=args.device,
    )
    for spec in cities:
        paths = city_paths(root, args.run_id, spec)
        prepare_city_inputs(root, spec, paths)
        print(
            f"[{spec.key}] post_mutation={paths.source_post_state}\n"
            f"[{spec.key}] hdv_state={paths.source_hdv_state}\n"
            f"[{spec.key}] portable_post_mutation={paths.portable_post_state}",
            flush=True,
        )
    if args.preflight_only:
        print("Preflight complete: all required inputs exist and portable copies were written.")
        return 0

    results: list[dict[str, Any]] = []
    for spec in cities:
        paths = city_paths(root, args.run_id, spec)
        try:
            results.append(
                run_city(
                    root=root,
                    python_bin=python_bin,
                    spec=spec,
                    checkpoints=checkpoints,
                    args=args,
                    env=env,
                )
            )
        except BaseException as exc:
            if not args.dry_run:
                _write_status(paths, stage="pipeline", state="failed", error=repr(exc))
            raise

    aggregate = {
        "schema_version": 1,
        "completed_at": _now(),
        "run_id": args.run_id,
        "protocol": {
            "warmup": "all_fixed_cavs_epsilon_greedy_eps_0.4",
            "warmup_hdv_learning": False,
            "wm_init": "cfm_current_8k",
            "online_real_action": "masked_policy_sample_one_hot",
            "imagined_action": "masked_policy_sample_one_hot",
            "global_advantage_weight": 1.0,
            "online_hdv_learning": True,
            "eval_action": "masked_policy_sample",
            "eval_episodes_per_checkpoint": int(args.eval_episodes),
            "eval_hdv_variants": {
                "final": {
                    "state": "final_online_training_state",
                    "human_learning_after_mutation": True,
                },
                "original": {
                    "state": "source_post_mutation_state",
                    "human_learning_after_mutation": False,
                },
            },
        },
        "cities": results,
    }
    scope = "three_city" if len(cities) == len(CITY_SPECS) else "_".join(
        spec.key for spec in cities
    )
    aggregate_path = root / "results" / f"{scope}_{args.run_id}_summary.json"
    if not args.dry_run:
        aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(f"Three-city summary: {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
