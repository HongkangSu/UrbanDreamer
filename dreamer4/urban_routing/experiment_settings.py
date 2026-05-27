"""Shared experiment settings loader for urban-routing stages.

This module keeps shell scripts and Python entrypoints aligned around one
experiment config file. The config format is a simple `.env`-style file:

    export NETWORK_NAME=ingolstadt_custom
    NETWORK_DIR=/path/to/network
    RATIO_MACHINES=0.6

Variable references such as `${NETWORK_DIR}/edge_registry.json` are expanded
against previously parsed keys plus the ambient environment.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional


_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _expand_vars(value: str, scope: Dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in scope:
            return scope[key]
        return os.environ.get(key, match.group(0))

    return _VAR_RE.sub(repl, value)


def load_settings_file(path: str) -> Dict[str, str]:
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Settings file not found: {cfg_path}")

    values: Dict[str, str] = {}
    for raw_line in cfg_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        scope = {**os.environ, **values}
        values[key] = _expand_vars(value, scope)
    values["EXPERIMENT_CONFIG"] = str(cfg_path)
    return values


def add_settings_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--settings",
        type=str,
        default=os.environ.get("EXPERIMENT_CONFIG", ""),
        help="Shared experiment settings file (.env style).",
    )


def apply_stage_defaults(
    parser: argparse.ArgumentParser,
    settings_path: str,
    stage: str,
) -> Dict[str, str]:
    if not settings_path:
        return {}
    cfg = load_settings_file(settings_path)
    defaults = build_stage_defaults(cfg, stage)
    if defaults:
        parser.set_defaults(**defaults)
    return cfg


def parse_settings_arg(argv: Optional[Iterable[str]] = None) -> str:
    mini = argparse.ArgumentParser(add_help=False)
    add_settings_arg(mini)
    known, _ = mini.parse_known_args(list(argv) if argv is not None else None)
    return known.settings


def build_stage_defaults(cfg: Dict[str, str], stage: str) -> Dict[str, object]:
    common = {
        "edge_registry": cfg.get("EDGE_REGISTRY", ""),
        "n_routes": _maybe_int(cfg.get("N_ROUTES")),
    }
    if stage == "collect":
        return _drop_none({
            "network_name": cfg.get("NETWORK_NAME", ""),
            "network_folder": cfg.get("NETWORK_DIR", ""),
            "agents_csv": cfg.get("AGENTS_CSV", ""),
            "od_file": cfg.get("OD_FILE", ""),
            "route_registry": cfg.get("ROUTE_REGISTRY", ""),
            "edge_registry": cfg.get("EDGE_REGISTRY", ""),
            "av_fraction": _maybe_float(cfg.get("RATIO_MACHINES")),
            "number_of_paths": _maybe_int(cfg.get("N_ROUTES")),
            "decision_steps": _maybe_int(cfg.get("EPISODE_STEPS")),
            "sim_buffer_after_last_depart": _maybe_int(cfg.get("SIM_BUFFER_AFTER_LAST_DEPART")),
            "incident_edge": cfg.get("INCIDENT_EDGE", ""),
            "incident_start": _maybe_float(cfg.get("INCIDENT_START")),
            "incident_end": _maybe_float(cfg.get("INCIDENT_END")),
            "incident_speed": _maybe_float(cfg.get("INCIDENT_SPEED")),
            "route_generator": cfg.get("ROUTE_GENERATOR", ""),
            "path_gen_beta": _maybe_float(cfg.get("PATH_GEN_BETA")),
            "num_samples": _maybe_int(cfg.get("PATH_NUM_SAMPLES")),
            "hdv_gawron_mode": cfg.get("HDV_GAWRON_MODE", ""),
            "hdv_gawron_alpha_zero": _maybe_float(cfg.get("HDV_GAWRON_ALPHA_ZERO")),
            "hdv_gawron_beta": _maybe_float(cfg.get("HDV_GAWRON_BETA")),
            "hdv_gawron_update_mode": cfg.get("HDV_GAWRON_UPDATE_MODE", ""),
        })
    if stage == "dynamics":
        return _drop_none({
            "data_dirs": [cfg["FLOW_DATA_DIR"]] if cfg.get("FLOW_DATA_DIR") else None,
            "edge_registry": cfg.get("EDGE_REGISTRY", ""),
            "flow_task_name": cfg.get("FLOW_TASK_NAME", ""),
            "action_key": cfg.get("ACTION_KEY", ""),
            "aux_action_key": cfg.get("AUX_ACTION_KEY", ""),
            "hdv_policy_target_mode": cfg.get("HDV_POLICY_TARGET_MODE", ""),
            "hdv_policy_action_mode": cfg.get("HDV_POLICY_ACTION_MODE", ""),
            "transition_hdv_context_mode": cfg.get("TRANSITION_HDV_CONTEXT_MODE", ""),
            "transition_hdv_context_weight": _maybe_float(cfg.get("TRANSITION_HDV_CONTEXT_WEIGHT")),
            "transition_hdv_context_scale": _maybe_float(cfg.get("TRANSITION_HDV_CONTEXT_SCALE")),
            "graph_cond_edge_conditioning": _maybe_bool(cfg.get("GRAPH_COND_EDGE_CONDITIONING")),
            "graph_cond_edge_conditioning_scale": _maybe_float(cfg.get("GRAPH_COND_EDGE_CONDITIONING_SCALE")),
            "ckpt_dir": cfg.get("DYNAMICS_LOG_DIR", ""),
        })
    if stage == "marl":
        n_agents = _maybe_int(cfg.get("N_AGENTS"))
        n_av = _maybe_int(cfg.get("N_AV"))
        n_hdv = _maybe_int(cfg.get("N_HDV"))
        if (n_agents is None or n_av is None or n_hdv is None) and cfg.get("AGENTS_CSV") and cfg.get("RATIO_MACHINES"):
            derived = _derive_agent_counts(cfg["AGENTS_CSV"], float(cfg["RATIO_MACHINES"]))
            n_agents = n_agents if n_agents is not None else derived["n_agents"]
            n_av = n_av if n_av is not None else derived["n_av"]
            n_hdv = n_hdv if n_hdv is not None else derived["n_hdv"]
        return _drop_none({
            "data_dir": cfg.get("FLOW_DATA_DIR", ""),
            "meta_file": _task_meta_file(cfg),
            "projector_ckpt": cfg.get("PROJECTOR_CKPT", ""),
            "dynamics_ckpt": cfg.get("DYNAMICS_CKPT", ""),
            "edge_registry": cfg.get("EDGE_REGISTRY", ""),
            "agents_csv": cfg.get("AGENTS_CSV", ""),
            "ratio_machines": _maybe_float(cfg.get("RATIO_MACHINES")),
            "network_folder": cfg.get("NETWORK_DIR", ""),
            "network_name": cfg.get("NETWORK_NAME", ""),
            "vehicle_routes_csv": str(Path(cfg["ROUTE_ARTIFACTS_DIR"]) / "routes.csv") if cfg.get("ROUTE_ARTIFACTS_DIR") else "",
            "n_routes": _maybe_int(cfg.get("N_ROUTES")),
            "n_av": n_av,
            "n_hdv": n_hdv,
            "checkpoint_dir": cfg.get("MARL_CKPT_DIR", ""),
            "training_family": cfg.get("TRAINING_FAMILY", ""),
            "wm_cfm_init_mode": cfg.get("WM_CFM_INIT_MODE", ""),
            "wm_policy_obs_source": cfg.get("WM_POLICY_OBS_SOURCE", ""),
            "vehicle_reward_scale": _maybe_float(cfg.get("VEHICLE_REWARD_SCALE")),
            "dense_bpr_reward_weight": _maybe_float(cfg.get("DENSE_BPR_REWARD_WEIGHT")),
            "terminal_tall_reward_weight": _maybe_float(cfg.get("TERMINAL_TALL_REWARD_WEIGHT")),
            "dreamer_batch_episodes": _maybe_int(cfg.get("DREAMER_BATCH_EPISODES")),
            "dreamer_actor_horizon": _maybe_int(cfg.get("DREAMER_ACTOR_HORIZON")),
            "dreamer_value_coef": _maybe_float(cfg.get("DREAMER_VALUE_COEF")),
            "dreamer_entropy_coef": _maybe_float(cfg.get("DREAMER_ENTROPY_COEF")),
            "dreamer_critic_variant": cfg.get("DREAMER_CRITIC_VARIANT", ""),
            "dreamer_ac_route_credit_mode": cfg.get("DREAMER_AC_ROUTE_CREDIT_MODE", ""),
            "dreamer_ac_route_credit_pressure_weight": _maybe_float(cfg.get("DREAMER_AC_ROUTE_CREDIT_PRESSURE_WEIGHT")),
            "dreamer_ac_route_credit_pressure_mode": cfg.get("DREAMER_AC_ROUTE_CREDIT_PRESSURE_MODE", ""),
            "dreamer_ac_route_credit_pressure_reference": cfg.get("DREAMER_AC_ROUTE_CREDIT_PRESSURE_REFERENCE", ""),
            "lr": _maybe_float(cfg.get("MARL_LR")),
            "gamma": _maybe_float(cfg.get("MARL_GAMMA")),
            "gae_lambda": _maybe_float(cfg.get("MARL_GAE_LAMBDA")),
            "hidden_dim": _maybe_int(cfg.get("MARL_HIDDEN_DIM")),
            "n_hidden_layers": _maybe_int(cfg.get("MARL_N_HIDDEN_LAYERS")),
        })
    if stage == "eval":
        return _drop_none({
            "policy_ckpt": cfg.get("MARL_POLICY_CKPT", cfg.get("MARL_CKPT", "")),
            "projector_ckpt": cfg.get("PROJECTOR_CKPT", ""),
            "dynamics_ckpt": cfg.get("DYNAMICS_CKPT", ""),
            "edge_registry": cfg.get("EDGE_REGISTRY", ""),
            "network": cfg.get("NETWORK_NAME", ""),
            "network_folder": cfg.get("NETWORK_DIR", ""),
            "n_routes": _maybe_int(cfg.get("N_ROUTES")),
            "ratio_machines": _maybe_float(cfg.get("RATIO_MACHINES")),
            "human_learning_episodes": _maybe_int(cfg.get("HDV_ADAPT_EP")),
            "test_eps": _maybe_int(cfg.get("EVAL_TEST_EPISODES")),
            "results_dir": cfg.get("EVAL_RESULTS_DIR", ""),
            "episode_steps": _maybe_int(cfg.get("EPISODE_STEPS")),
            "sim_buffer_after_last_depart": _maybe_int(cfg.get("SIM_BUFFER_AFTER_LAST_DEPART")),
            "incident_edge": cfg.get("INCIDENT_EDGE", ""),
            "incident_start": _maybe_float(cfg.get("INCIDENT_START")),
            "incident_end": _maybe_float(cfg.get("INCIDENT_END")),
            "incident_speed": _maybe_float(cfg.get("INCIDENT_SPEED")),
            "route_generator": cfg.get("ROUTE_GENERATOR", ""),
            "path_gen_beta": _maybe_float(cfg.get("PATH_GEN_BETA")),
            "path_num_samples": _maybe_int(cfg.get("PATH_NUM_SAMPLES")),
            "reuse_artifacts_dir": cfg.get("ROUTE_ARTIFACTS_DIR", ""),
            "hdv_gawron_mode": cfg.get("HDV_GAWRON_MODE", ""),
            "hdv_gawron_alpha_zero": _maybe_float(cfg.get("HDV_GAWRON_ALPHA_ZERO")),
            "hdv_gawron_beta": _maybe_float(cfg.get("HDV_GAWRON_BETA")),
            "hdv_gawron_update_mode": cfg.get("HDV_GAWRON_UPDATE_MODE", ""),
            "disable_human_learning_after_mutation": _maybe_bool(cfg.get("DISABLE_HUMAN_LEARNING_AFTER_MUTATION")),
        })
    return _drop_none(common)


def _task_meta_file(cfg: Dict[str, str]) -> Optional[str]:
    task = cfg.get("FLOW_TASK_NAME", "")
    if not task:
        return None
    return f"{task}.pt"


def _drop_none(values: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in values.items() if v not in (None, "")}


def _maybe_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _maybe_int(value: Optional[str]) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def _maybe_bool(value: Optional[str]) -> Optional[bool]:
    if value in (None, ""):
        return None
    norm = str(value).strip().lower()
    if norm in {"1", "true", "yes", "y", "on"}:
        return True
    if norm in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def _derive_agent_counts(agents_csv: str, ratio_machines: float) -> Dict[str, int]:
    path = Path(agents_csv).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    n_agents = 0
    with path.open("r", encoding="utf-8") as f:
        next(f, None)
        for _ in f:
            n_agents += 1
    n_av = int(n_agents * float(ratio_machines))
    return {
        "n_agents": n_agents,
        "n_av": n_av,
        "n_hdv": n_agents - n_av,
    }
