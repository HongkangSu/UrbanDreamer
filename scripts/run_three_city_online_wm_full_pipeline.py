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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("collect", "wm", "online", "eval")
EVAL_HDV_VARIANTS = ("final", "original")
EVAL_HDV_PROTOCOLS: Mapping[str, Mapping[str, Any]] = {
    "final": {
        "state": "final_online_training_state",
        "human_learning_after_mutation": True,
    },
    "original": {
        "state": "source_post_mutation_state",
        "human_learning_after_mutation": False,
    },
}


@dataclass(frozen=True)
class CitySpec:
    key: str
    network_name: str
    settings_rel: str
    post_bundle_rel: str
    sumo_extra_args: str = "--collision.action warn"
    train_sim_buffer_after_last_depart: int | None = None


@dataclass(frozen=True)
class ExistingWarmupSpec:
    run_tag: str
    task_name: str
    random_av_fraction: float
    epsilon: float


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
        post_bundle_rel="networks/gretz_armainvilliers/post_mutation",
    ),
    "ingolstadt_custom": CitySpec(
        key="ingolstadt_custom",
        network_name="ingolstadt_custom",
        settings_rel=(
            ".urbandreamer/generated/ingolstadt_custom_current_to_dest_incident_20260528/"
            "urb_fair_eval.env"
        ),
        post_bundle_rel="networks/ingolstadt_custom/post_mutation",
    ),
    "fontenay_en_parisis": CitySpec(
        key="fontenay_en_parisis",
        network_name="fontenay_en_parisis",
        settings_rel=(
            ".urbandreamer/generated/"
            "fontenay_en_parisis_current_to_dest_both494_incident_warmup50_"
            "fastcap_20260609/urb_fair_eval.env"
        ),
        post_bundle_rel="networks/fontenay_en_parisis",
        sumo_extra_args=(
            "--collision.action warn --ignore-junction-blocker 60 "
            "--tripinfo-output.write-unfinished"
        ),
        train_sim_buffer_after_last_depart=1500,
    ),
}


EXISTING_WARMUP_SPECS: Mapping[str, ExistingWarmupSpec] = {
    "gretz_armainvilliers": ExistingWarmupSpec(
        run_tag=(
            "gretz_armainvilliers_current_to_dest_incident_completion100_"
            "stronglivebad_collisionwarn_drain7200_20260601_"
            "neg_986733117_1_S0b_s500_e2500_v0p1"
        ),
        task_name="gretz_armainvilliers_urb_fair_eval_postmutation",
        random_av_fraction=0.6,
        epsilon=0.4,
    ),
    "ingolstadt_custom": ExistingWarmupSpec(
        run_tag="ingolstadt_custom_current_to_dest_incident_20260528",
        task_name="ingolstadt_custom_urb_fair_eval_postmutation",
        random_av_fraction=0.6,
        epsilon=0.4,
    ),
    "fontenay_en_parisis": ExistingWarmupSpec(
        run_tag=(
            "fontenay_en_parisis_current_to_dest_both494_incident_"
            "warmup50_fastcap_20260609"
        ),
        task_name="fontenay_en_parisis_urb_fair_eval_postmutation",
        random_av_fraction=0.6,
        epsilon=0.4,
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


def _eval_hdv_variants(raw: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        values = [
            part.strip()
            for part in raw.replace(";", ",").replace(" ", ",").split(",")
            if part.strip()
        ]
    else:
        values = [str(value).strip() for value in raw if str(value).strip()]
    result = tuple(dict.fromkeys(values))
    unknown = [value for value in result if value not in EVAL_HDV_VARIANTS]
    if not result or unknown:
        expected = ",".join(EVAL_HDV_VARIANTS)
        details = f"; unknown values: {','.join(unknown)}" if unknown else ""
        raise argparse.ArgumentTypeError(
            f"eval HDV variants must be selected from {expected}{details}"
        )
    return result


def _selected_cities(raw: str) -> tuple[CitySpec, ...]:
    names = [part.strip() for part in str(raw).replace(";", ",").split(",") if part.strip()]
    if not names or names == ["all"]:
        names = list(CITY_SPECS)
    unknown = [name for name in names if name not in CITY_SPECS]
    if unknown:
        raise ValueError(f"unknown cities: {unknown}; expected one of {list(CITY_SPECS)}")
    return tuple(CITY_SPECS[name] for name in names)


def _city_gpu_ids(raw: str) -> tuple[str, ...]:
    gpu_ids = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--city-gpus must not contain duplicate GPU IDs")
    if any(not re.fullmatch(r"[0-9]+", gpu_id) for gpu_id in gpu_ids):
        raise ValueError("--city-gpus accepts comma-separated numeric CUDA device IDs")
    return gpu_ids


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


def prepare_city_inputs(
    root: Path,
    spec: CitySpec,
    paths: CityPaths,
    *,
    eval_hdv_variants: str | Iterable[str] = EVAL_HDV_VARIANTS,
) -> None:
    selected_eval_hdv_variants = _eval_hdv_variants(eval_hdv_variants)
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
                variant: EVAL_HDV_PROTOCOLS[variant]
                for variant in selected_eval_hdv_variants
            },
            "train_sumo_extra_args": spec.sumo_extra_args,
            "eval_sumo_extra_args": spec.sumo_extra_args,
        },
    }
    paths.provenance_json.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_warmup_paths(
    *,
    root: Path,
    spec: CitySpec,
    output_paths: CityPaths,
    expected_episodes: int,
    reuse_existing: bool,
) -> CityPaths:
    if not reuse_existing:
        return output_paths

    source = EXISTING_WARMUP_SPECS[spec.key]
    data_dir = root / "data" / source.run_tag / "collect_urb_fair_eval_postmutation"
    records_dir = (
        root / "results" / source.run_tag / "collect_urb_fair_eval_postmutation_records"
    )
    manifest_path = records_dir / "urb_fair_collection_manifest.json"
    dataset_path = data_dir / f"{source.task_name}.pt"
    missing = [path for path in (manifest_path, dataset_path) if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"existing warmup input is incomplete for {spec.key}:\n{formatted}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if str(manifest.get("network", "")) != spec.network_name:
        errors.append(f"network={manifest.get('network')!r}")
    if str(manifest.get("task_name", "")) != source.task_name:
        errors.append(f"task_name={manifest.get('task_name')!r}")
    if bool(manifest.get("partial", True)):
        errors.append("manifest is partial")
    episodes_saved = int(manifest.get("episodes_saved", 0))
    if episodes_saved < int(expected_episodes):
        errors.append(
            f"episodes_saved={episodes_saved} < expected={int(expected_episodes)}"
        )
    if bool(manifest.get("should_humans_adapt", True)):
        errors.append("warmup HDV learning was enabled")
    if str(manifest.get("action_schema", "")) != "av_planned_edge_flow_pre_advance_v1":
        errors.append(f"action_schema={manifest.get('action_schema')!r}")

    artifact_hashes = manifest.get("artifact_hashes", {})
    for key, filename in (
        ("agents_csv", "agents.csv"),
        ("routes_csv", "routes.csv"),
        ("route_rou_xml", "route.rou.xml"),
    ):
        expected_hash = str(artifact_hashes.get(key, ""))
        current_hash = _sha256(output_paths.source_bundle / filename)
        if expected_hash != current_hash:
            errors.append(
                f"{filename} hash mismatch: warmup={expected_hash!r}, "
                f"post-mutation={current_hash!r}"
            )
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise RuntimeError(
            f"existing warmup input failed validation for {spec.key}:\n{details}"
        )

    warmup_paths = replace(
        output_paths,
        data_dir=data_dir,
        records_dir=records_dir,
        task_name=source.task_name,
    )
    _write_portable_settings(root=root, spec=spec, paths=warmup_paths)
    provenance = json.loads(output_paths.provenance_json.read_text(encoding="utf-8"))
    provenance["protocol"].update(
        {
            "warmup": "legacy_fixed_cav_random_epsilon_split",
            "warmup_epsilon": source.epsilon,
            "warmup_random_av_fraction": source.random_av_fraction,
            "warmup_hdv_learning": False,
            "warmup_reused": True,
        }
    )
    provenance["warmup_input"] = {
        "run_tag": source.run_tag,
        "data_dir": str(data_dir.resolve()),
        "records_dir": str(records_dir.resolve()),
        "task_name": source.task_name,
        "manifest": str(manifest_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "episodes_saved": episodes_saved,
        "action_schema": manifest["action_schema"],
        "random_av_fraction": source.random_av_fraction,
        "epsilon": source.epsilon,
        "hdv_learning": False,
        "artifact_hashes_verified": True,
    }
    output_paths.provenance_json.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return warmup_paths


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
        "--flow_cache_mb",
        str(args.wm_flow_cache_mb),
        "--d_model_dyn",
        "256",
        "--dyn_depth",
        "6",
        "--gradient_checkpointing",
        "--space_attention_batch_chunk_size",
        str(args.wm_space_attention_batch_chunk_size),
        "--mlp_token_chunk_size",
        str(args.wm_mlp_token_chunk_size),
        "--time_attention_batch_chunk_size",
        str(args.wm_time_attention_batch_chunk_size),
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
        str(paths.source_bundle),
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
        str(
            spec.train_sim_buffer_after_last_depart
            if spec.train_sim_buffer_after_last_depart is not None
            else args.train_sim_buffer_after_last_depart
        ),
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
        "--imagined-rollout-batch-size",
        str(args.online_imagined_rollout_batch_size),
        "--wm-replay-buffer-episodes",
        "8",
        "--wm-updates-per-collect",
        "4",
        "--wm-update-epochs",
        "1",
        "--wm-batch-size",
        str(args.online_wm_batch_size),
        "--wm-gradient-checkpointing",
        "--wm-space-attention-batch-chunk-size",
        str(args.online_wm_space_attention_batch_chunk_size),
        "--wm-mlp-token-chunk-size",
        str(args.online_wm_mlp_token_chunk_size),
        "--wm-time-attention-batch-chunk-size",
        str(args.online_wm_time_attention_batch_chunk_size),
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
        str(paths.source_bundle),
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
    output_prefix: str = "",
) -> None:
    rendered = _command_text(command)
    print(f"{output_prefix}[{_now()}] $ {rendered}", flush=True)
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
            print(f"{output_prefix}{line}", end="", flush=True)
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


def resolve_world_model_paths(
    *,
    root: Path,
    spec: CitySpec,
    output_paths: CityPaths,
    expected_steps: int,
    reuse_run_id: str,
) -> CityPaths:
    source_run_id = str(reuse_run_id or "").strip()
    if not source_run_id:
        return output_paths
    source_paths = city_paths(root, source_run_id, spec)
    if not _wm_complete(source_paths, expected_steps):
        print(
            f"[{spec.key}] no complete {int(expected_steps)}-step WM in "
            f"run {source_run_id}; training it in the current run",
            flush=True,
        )
        return output_paths
    print(
        f"[{spec.key}] reusing complete WM from {source_paths.wm_dir}",
        flush=True,
    )
    return replace(output_paths, wm_dir=source_paths.wm_dir)


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
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        try:
            valid_thread_count = int(str(env.get(variable, "")).strip()) > 0
        except ValueError:
            valid_thread_count = False
        if not valid_thread_count:
            env[variable] = "1"
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
    probe = """
import importlib
import importlib.metadata
import inspect
import json

modules = {}
errors = {}
for name in ("torch", "routerl", "traci"):
    try:
        modules[name] = importlib.import_module(name)
        if name == "routerl":
            traffic_environment = getattr(modules[name], "TrafficEnvironment")
            routerlurb_version = importlib.metadata.version("routerlurb")
            if routerlurb_version != "1.0.0":
                raise RuntimeError(
                    f"routerlurb==1.0.0 is required, found {routerlurb_version}"
                )
            signature = inspect.signature(traffic_environment)
            if "save_detectors_info" not in signature.parameters:
                raise RuntimeError(
                    "incompatible TrafficEnvironment API: constructor does not expose "
                    "save_detectors_info; remove conflicting 'routerl' packages and "
                    "reinstall routerlurb==1.0.0"
                )
    except Exception as exc:
        errors[name] = f"{type(exc).__name__}: {exc}"
if errors:
    print(json.dumps({"errors": errors}))
    raise SystemExit(3)

torch = modules["torch"]
print(json.dumps({
    "torch": torch.__version__,
    "routerl": routerlurb_version,
    "traci": getattr(modules["traci"], "__version__", "available"),
    "cuda_available": torch.cuda.is_available(),
    "cuda_devices": torch.cuda.device_count(),
}))
"""
    completed = subprocess.run(
        [python_bin, "-c", probe],
        cwd=str(root),
        env=dict(env),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = "\n".join(
            part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
        )
        raise RuntimeError(
            "Runtime dependency preflight failed for the selected Python interpreter.\n"
            f"Interpreter: {python_bin}\n"
            f"Install command: {python_bin} -m pip install -r {root / 'requirements.txt'}\n"
            "No Conda environment name is required; every stage uses this selected "
            "interpreter.\n"
            "The RouteRL distribution is named 'routerlurb', while its Python import is "
            f"'routerl' and must expose TrafficEnvironment.\n"
            f"Probe output:\n{details or '(no output)'}"
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
    eval_hdv_variants = _eval_hdv_variants(args.eval_hdv_variants)
    output_paths = city_paths(root, args.run_id, spec)
    prepare_city_inputs(
        root,
        spec,
        output_paths,
        eval_hdv_variants=eval_hdv_variants,
    )
    paths = resolve_warmup_paths(
        root=root,
        spec=spec,
        output_paths=output_paths,
        expected_episodes=args.collect_episodes,
        reuse_existing=args.reuse_existing_warmup,
    )
    paths = resolve_world_model_paths(
        root=root,
        spec=spec,
        output_paths=paths,
        expected_steps=args.wm_steps,
        reuse_run_id=args.reuse_world_models_from_run_id,
    )

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
                output_prefix=f"[{spec.key}] ",
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
                output_prefix=f"[{spec.key}] ",
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
                output_prefix=f"[{spec.key}] ",
            )
            if not args.dry_run and not _online_complete(paths, args.online_episodes):
                raise RuntimeError(f"online-WM training is incomplete: {paths.run_tag}")
            write_status(stage="online", state="complete")

    all_best_selections = {
        "final": paths.online_dir / "best_checkpoint_selection.json",
        "original": paths.online_dir / "best_checkpoint_selection_original_hdv.json",
    }
    best_selections = {
        variant: all_best_selections[variant] for variant in eval_hdv_variants
    }
    if _stage_enabled(args.start_stage, "eval"):
        if not args.dry_run and not _online_complete(paths, args.online_episodes):
            raise RuntimeError(f"eval stage requires complete online training: {paths.run_tag}")
        write_status(stage="eval", state="running")
        for hdv_variant in eval_hdv_variants:
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
                    output_prefix=f"[{spec.key}] ",
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
                "--metric",
                "t_all_mean",
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
                output_prefix=f"[{spec.key}] ",
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
        "eval_hdv_variants": list(eval_hdv_variants),
    }
    if "final" in best_selections:
        result["eval_summary"] = str(paths.online_dir / "online_wm_eval_summary.json")
        result["best_checkpoint_selection"] = str(best_selections["final"])
        if best_selections["final"].exists():
            selection = json.loads(best_selections["final"].read_text(encoding="utf-8"))
            result["best"] = selection.get("best", {})
    if "original" in best_selections:
        result["eval_original_hdv_summary"] = str(
            paths.online_dir / "online_wm_eval_original_hdv_summary.json"
        )
        result["best_checkpoint_selection_original_hdv"] = str(
            best_selections["original"]
        )
        if best_selections["original"].exists():
            selection = json.loads(
                best_selections["original"].read_text(encoding="utf-8")
            )
            result["best_original_hdv"] = selection.get("best", {})
    return result


def run_cities(
    *,
    root: Path,
    python_bin: str,
    cities: Sequence[CitySpec],
    checkpoints: Sequence[int],
    args: argparse.Namespace,
    env: Mapping[str, str],
    city_workers: int,
) -> list[dict[str, Any]]:
    if city_workers <= 0:
        raise ValueError("--city-workers must be positive")
    if not cities:
        return []

    gpu_ids = _city_gpu_ids(getattr(args, "city_gpus", ""))
    if gpu_ids and not str(args.device).lower().startswith("cuda"):
        raise ValueError("--city-gpus requires a CUDA --device")
    worker_count = min(
        int(city_workers),
        len(cities),
        len(gpu_ids) if gpu_ids else len(cities),
    )
    slots: tuple[str | None, ...] = (
        tuple(gpu_ids[:worker_count]) if gpu_ids else (None,) * worker_count
    )
    gpu_text = f" on physical GPU slots {','.join(gpu_ids[:worker_count])}" if gpu_ids else ""
    print(
        f"Launching {len(cities)} city pipeline(s) with {worker_count} parallel "
        f"worker(s){gpu_text}.",
        flush=True,
    )
    results_by_city: dict[str, dict[str, Any]] = {}
    failures: list[tuple[CitySpec, Exception]] = []

    def submit_city(
        executor: ThreadPoolExecutor,
        spec: CitySpec,
        gpu_id: str | None,
    ) -> Future[dict[str, Any]]:
        city_args = argparse.Namespace(**vars(args))
        city_env = dict(env)
        if gpu_id is not None:
            city_env["CUDA_VISIBLE_DEVICES"] = gpu_id
            city_args.device = "cuda"
            print(f"[{spec.key}] assigned physical GPU {gpu_id}", flush=True)
        return executor.submit(
            run_city,
            root=root,
            python_bin=python_bin,
            spec=spec,
            checkpoints=checkpoints,
            args=city_args,
            env=city_env,
        )

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="city-pipeline",
    ) as executor:
        queued_cities = list(cities)
        reuse_run_id = str(args.reuse_world_models_from_run_id or "").strip()
        if reuse_run_id and _stage_enabled(args.start_stage, "wm"):
            queued_cities.sort(
                key=lambda spec: _wm_complete(
                    city_paths(root, reuse_run_id, spec), args.wm_steps
                )
            )
            print(
                "City launch order prioritizes missing reused WM checkpoints: "
                + ", ".join(spec.key for spec in queued_cities),
                flush=True,
            )
        pending_cities = iter(queued_cities)
        active: dict[Future[dict[str, Any]], tuple[CitySpec, str | None]] = {}
        for slot in slots:
            spec = next(pending_cities)
            active[submit_city(executor, spec, slot)] = (spec, slot)

        while active:
            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in completed:
                spec, slot = active.pop(future)
                try:
                    results_by_city[spec.key] = future.result()
                except Exception as exc:
                    paths = city_paths(root, args.run_id, spec)
                    if not args.dry_run:
                        _write_status(
                            paths,
                            stage="pipeline",
                            state="failed",
                            error=repr(exc),
                        )
                    failures.append((spec, exc))
                    print(
                        f"[{spec.key}] pipeline failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(f"[{spec.key}] pipeline complete", flush=True)

                next_spec = next(pending_cities, None)
                if next_spec is not None:
                    active[submit_city(executor, next_spec, slot)] = (next_spec, slot)

    if failures:
        failed_cities = ", ".join(spec.key for spec, _ in failures)
        first_error = failures[0][1]
        raise RuntimeError(
            f"{len(failures)} city pipeline(s) failed: {failed_cities}; "
            "inspect each city's full_pipeline.log and pipeline_status.json"
        ) from first_error

    return [results_by_city[spec.key] for spec in cities]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the current online-WM mainline and stochastic eval on three cities."
    )
    parser.add_argument(
        "--cities",
        default="all",
        help="Comma-separated city names or 'all'.",
    )
    parser.add_argument(
        "--city-workers",
        type=int,
        default=3,
        help=(
            "Number of city pipelines to run concurrently (default: 3). "
            "Use 1 for serial execution or when one GPU cannot hold concurrent jobs."
        ),
    )
    parser.add_argument(
        "--city-gpus",
        default="",
        help=(
            "Optional comma-separated physical CUDA device IDs. Each ID is an exclusive "
            "city worker slot; queued cities reuse the first available slot."
        ),
    )
    parser.add_argument("--run-id", default="online_wm_mainline_150ep_20260717")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help=(
            "Python executable used by every pipeline stage. Defaults to PYTHON_BIN "
            "or the interpreter running this script; no named Conda environment is required."
        ),
    )
    parser.add_argument("--sumo-home", default=os.environ.get("SUMO_HOME", ""))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collect-episodes", type=int, default=50)
    parser.add_argument(
        "--reuse-existing-warmup",
        action="store_true",
        help=(
            "Reuse each city's validated 50-episode legacy warmup dataset instead of "
            "collecting again. The legacy policy used random_av_fraction=0.6 and "
            "epsilon=0.4 with frozen HDVs."
        ),
    )
    parser.add_argument(
        "--reuse-world-models-from-run-id",
        default="",
        help=(
            "Reuse a city's complete WM checkpoint from this prior run ID. Cities "
            "without a complete checkpoint automatically train WM in the current run."
        ),
    )
    parser.add_argument("--collect-max-steps", type=int, default=3600)
    parser.add_argument("--collect-completion-drain-steps", type=int, default=7200)
    parser.add_argument("--collect-completion-drain-patience-steps", type=int, default=600)
    parser.add_argument("--episode-wall-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--wm-steps", type=int, default=8000)
    parser.add_argument("--wm-batch-size", type=int, default=1)
    parser.add_argument("--wm-seq-len", type=int, default=16)
    parser.add_argument("--wm-grad-accum", type=int, default=32)
    parser.add_argument("--wm-flow-cache-mb", type=int, default=512)
    parser.add_argument("--wm-space-attention-batch-chunk-size", type=int, default=1)
    parser.add_argument("--wm-mlp-token-chunk-size", type=int, default=4096)
    parser.add_argument("--wm-time-attention-batch-chunk-size", type=int, default=256)
    parser.add_argument("--online-wm-batch-size", type=int, default=4)
    parser.add_argument(
        "--online-wm-space-attention-batch-chunk-size",
        type=int,
        default=16,
    )
    parser.add_argument("--online-wm-mlp-token-chunk-size", type=int, default=16384)
    parser.add_argument(
        "--online-wm-time-attention-batch-chunk-size",
        type=int,
        default=32768,
    )
    parser.add_argument(
        "--online-imagined-rollout-batch-size",
        type=int,
        default=4,
        help="Independent rollout batch size; horizon steps remain causal and sequential.",
    )
    parser.add_argument("--online-episodes", type=int, default=150)
    parser.add_argument("--checkpoint-episodes", default="25,50,75,100,125,150")
    parser.add_argument("--train-sim-buffer-after-last-depart", type=int, default=2000)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Stochastic SUMO episodes per checkpoint and HDV-state variant.",
    )
    parser.add_argument(
        "--eval-hdv-variants",
        type=_eval_hdv_variants,
        default=EVAL_HDV_VARIANTS,
        help=(
            "Comma-separated evaluation HDV states: final, original, or both "
            "(default: final,original). 'original' loads the pre-online-training "
            "post-mutation HDV state and freezes HDV learning."
        ),
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
    if args.reuse_world_models_from_run_id and not re.fullmatch(
        r"[A-Za-z0-9_.-]+", str(args.reuse_world_models_from_run_id)
    ):
        raise ValueError(
            "--reuse-world-models-from-run-id may contain only letters, digits, '.', '_' and '-'"
        )
    cities = _selected_cities(args.cities)
    checkpoints = _checkpoint_episodes(args.checkpoint_episodes)
    if max(checkpoints) > int(args.online_episodes):
        raise ValueError("checkpoint episodes cannot exceed --online-episodes")
    if int(args.online_episodes) not in checkpoints:
        raise ValueError("--checkpoint-episodes must include the final online episode")
    if int(args.city_workers) <= 0:
        raise ValueError("--city-workers must be positive")
    city_gpu_ids = _city_gpu_ids(args.city_gpus)
    if city_gpu_ids and not str(args.device).lower().startswith("cuda"):
        raise ValueError("--city-gpus requires a CUDA --device")

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
        prepare_city_inputs(
            root,
            spec,
            paths,
            eval_hdv_variants=args.eval_hdv_variants,
        )
        warmup_paths = resolve_warmup_paths(
            root=root,
            spec=spec,
            output_paths=paths,
            expected_episodes=args.collect_episodes,
            reuse_existing=args.reuse_existing_warmup,
        )
        print(
            f"[{spec.key}] post_mutation={paths.source_post_state}\n"
            f"[{spec.key}] hdv_state={paths.source_hdv_state}\n"
            f"[{spec.key}] portable_post_mutation={paths.portable_post_state}\n"
            f"[{spec.key}] warmup_data={warmup_paths.data_dir}\n"
            f"[{spec.key}] warmup_records={warmup_paths.records_dir}",
            flush=True,
        )
    if args.preflight_only:
        print("Preflight complete: all required inputs exist and portable copies were written.")
        return 0

    results = run_cities(
        root=root,
        python_bin=python_bin,
        cities=cities,
        checkpoints=checkpoints,
        args=args,
        env=env,
        city_workers=args.city_workers,
    )

    aggregate = {
        "schema_version": 1,
        "completed_at": _now(),
        "run_id": args.run_id,
        "execution": {
            "city_workers": min(
                int(args.city_workers),
                len(cities),
                len(city_gpu_ids) if city_gpu_ids else len(cities),
            ),
            "city_gpus": list(city_gpu_ids),
        },
        "protocol": {
            "warmup": (
                "legacy_fixed_cav_random0.6_epsilon0.4_split_reused"
                if args.reuse_existing_warmup
                else "all_fixed_cavs_epsilon_greedy_eps_0.4"
            ),
            "warmup_reused": bool(args.reuse_existing_warmup),
            "warmup_hdv_learning": False,
            "wm_init": "cfm_current_8k",
            "online_real_action": "masked_policy_sample_one_hot",
            "imagined_action": "masked_policy_sample_one_hot",
            "global_advantage_weight": 1.0,
            "online_hdv_learning": True,
            "eval_action": "masked_policy_sample",
            "eval_episodes_per_checkpoint": int(args.eval_episodes),
            "eval_hdv_variants": {
                variant: EVAL_HDV_PROTOCOLS[variant]
                for variant in args.eval_hdv_variants
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
