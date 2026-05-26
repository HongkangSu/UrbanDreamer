#!/usr/bin/env python3
"""
Evaluate trained Dreamer4 MARL policy in SUMO via RouteRL TrafficEnvironment.

Defaults to the repo-local ingolstadt_custom network under ./networks:
  - 40% AVs by default
  - human-stabilisation episodes use RouteRL/Gawron
  - test episodes use the Dreamer4 policy controlling AVs
  - Metrics: t_CAV, t_HDV, t_all (final test-episode travel times in minutes)

Usage:
    conda activate URB
    cd UrbanDreamer
    python scripts/eval_sumo.py \
        --policy_ckpt       logs/marl/latest.pt \
        --projector_ckpt    logs/urban_dynamics/flow_projector_latest.pt \
        --dynamics_ckpt     logs/urban_dynamics/latest.pt \
        --edge_registry     networks/ingolstadt_custom/edge_registry.json
"""

import os
import sys
import copy
import json
import argparse
import socket
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional, Dict, Tuple, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_REPO_ROOT    = _SCRIPT_DIR.parent
_DREAMER4_DIR = _REPO_ROOT / "dreamer4"
_URBAN_DIR    = _DREAMER4_DIR / "urban_routing"
_DEFAULT_NETWORK_NAME = "ingolstadt_custom"
_DEFAULT_NETWORK_FOLDER = _REPO_ROOT / "networks" / _DEFAULT_NETWORK_NAME
_DEFAULT_EDGE_REGISTRY = _DEFAULT_NETWORK_FOLDER / "edge_registry.json"

for _p in [str(_DREAMER4_DIR), str(_URBAN_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _ensure_sumo_runtime_env() -> None:
    """Make sure both SUMO python tools and the `sumo` binary are discoverable."""
    sumo_home = os.environ.get("SUMO_HOME", "").strip()
    candidate_homes = [sumo_home] if sumo_home else []
    candidate_homes += ["/usr/share/sumo"]

    chosen_home = ""
    for home in candidate_homes:
        if home and os.path.isdir(home):
            chosen_home = home
            break

    if chosen_home and (not os.environ.get("SUMO_HOME")):
        os.environ["SUMO_HOME"] = chosen_home

    if chosen_home:
        tools_dir = os.path.join(chosen_home, "tools")
        bin_dir = os.path.join(chosen_home, "bin")
        if os.path.isdir(tools_dir) and tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
            print(f"[SUCCESS] Added module directory: {tools_dir}")
        path_parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
        if os.path.isdir(bin_dir) and bin_dir not in path_parts:
            os.environ["PATH"] = bin_dir + (os.pathsep + os.environ["PATH"] if os.environ.get("PATH") else "")
            print(f"[SUCCESS] Added binary directory to PATH: {bin_dir}")
        print("[CONFIRMED] Environment variable exists: SUMO_HOME")
    else:
        print("WARNING: SUMO_HOME not configured and no fallback installation was found.")


_ensure_sumo_runtime_env()

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from routerl import TrafficEnvironment
    ROUTERL_AVAILABLE = True
except ImportError:
    ROUTERL_AVAILABLE = False
    print("ERROR: routerl not installed. Activate the URB conda environment first.")

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False


def _patch_traci_start_port_fallback() -> None:
    """Work around environments where SUMO's dynamic free-port probe returns None."""
    if not TRACI_AVAILABLE:
        return
    try:
        from sumolib.miscutils import getFreeSocketPort
    except Exception:
        return

    try:
        probe = getFreeSocketPort()
    except Exception:
        probe = None

    if probe is not None:
        return

    try:
        import traci.main as traci_main
    except Exception:
        traci_main = None

    original_start = traci.start

    def _reserve_loopback_port() -> Optional[int]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        except OSError:
            return None
        finally:
            try:
                sock.close()
            except Exception:
                pass
        return port

    def _iter_fallback_ports() -> List[int]:
        ports: List[int] = []
        seen = set()
        # Prefer true loopback-ephemeral ports first to avoid collisions on busy boxes.
        for _ in range(16):
            cand = _reserve_loopback_port()
            if cand is None or cand in seen:
                continue
            seen.add(cand)
            ports.append(cand)
        # Keep the old deterministic band only as a last resort.
        base_port = 8800 + (os.getpid() % 100)
        for cand in range(base_port, base_port + 32):
            if cand in seen:
                continue
            seen.add(cand)
            ports.append(cand)
        return ports

    def _fallback_start(cmd, port=None, numRetries=60, label="default", verbose=False,
                        traceFile=None, traceGetters=True, stdout=None, doSwitch=True):
        if port is not None:
            return original_start(
                cmd,
                port=port,
                numRetries=numRetries,
                label=label,
                verbose=verbose,
                traceFile=traceFile,
                traceGetters=traceGetters,
                stdout=stdout,
                doSwitch=doSwitch,
            )

        last_exc = None
        for cand in _iter_fallback_ports():
            try:
                return original_start(
                    cmd,
                    port=cand,
                    numRetries=numRetries,
                    label=label,
                    verbose=verbose,
                    traceFile=traceFile,
                    traceGetters=traceGetters,
                    stdout=stdout,
                    doSwitch=doSwitch,
                )
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("failed to allocate any fallback TraCI port")

    traci.start = _fallback_start
    if traci_main is not None:
        traci_main.start = _fallback_start
    print("[INFO] Patched traci.start with fixed-port fallback because getFreeSocketPort() returned None.")


_patch_traci_start_port_fallback()


def _check_local_tcp_accept_support() -> Tuple[bool, str]:
    """Check whether the current runtime can open a local TCP listener for TraCI."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except OSError as exc:
        return False, str(exc)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return True, ""
    except OSError as exc:
        return False, str(exc)
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _copy_artifact_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() and os.path.samefile(src, dst):
            return
    except Exception:
        pass
    shutil.copy2(src, dst)


def _reuse_route_artifacts(src_dir: Path, dst_dir: Path) -> Dict[str, Path]:
    required = {
        "agents_csv": src_dir / "agents.csv",
        "routes_csv": src_dir / "routes.csv",
        "route_rou_xml": src_dir / "route.rou.xml",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "reuse_artifacts_dir is missing required files: " + ", ".join(missing)
        )
    copied = {
        "agents_csv": dst_dir / "agents.csv",
        "routes_csv": dst_dir / "routes.csv",
        "route_rou_xml": dst_dir / "route.rou.xml",
    }
    for key, src in required.items():
        _copy_artifact_if_needed(src, copied[key])
    for optional_name in ("routes_meta.json", "det.add.xml"):
        src = src_dir / optional_name
        if src.exists():
            _copy_artifact_if_needed(src, dst_dir / optional_name)
    return copied


def _reuse_route_artifacts_from_refs(
    artifact_refs: Dict[str, Any],
    dst_dir: Path,
) -> Dict[str, Path]:
    required = {
        "agents_csv": Path(str(artifact_refs["agents_csv"])).resolve(),
        "routes_csv": Path(str(artifact_refs["routes_csv"])).resolve(),
        "route_rou_xml": Path(str(artifact_refs["route_rou_xml"])).resolve(),
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "post-mutation state references missing route artifacts: " + ", ".join(missing)
        )
    copied = {
        "agents_csv": dst_dir / "agents.csv",
        "routes_csv": dst_dir / "routes.csv",
        "route_rou_xml": dst_dir / "route.rou.xml",
    }
    for key, src in required.items():
        _copy_artifact_if_needed(src, copied[key])
    return copied

try:
    from experiment_settings import add_settings_arg, apply_stage_defaults, parse_settings_arg
except ImportError:
    from dreamer4.urban_routing.experiment_settings import (
        add_settings_arg,
        apply_stage_defaults,
        parse_settings_arg,
    )

try:
    from routerl_shared import (
        load_od_data,
        load_agents_dataframe,
        compute_agent_counts,
        ensure_agents_csv_in_records,
        generate_dijkstra_routes,
        generate_routerl_routes,
        create_routerl_environment,
    )
except ImportError:
    from dreamer4.urban_routing.routerl_shared import (
        load_od_data,
        load_agents_dataframe,
        compute_agent_counts,
        ensure_agents_csv_in_records,
        generate_dijkstra_routes,
        generate_routerl_routes,
        create_routerl_environment,
    )

try:
    from data_collector import DataCollector
except ImportError:
    from dreamer4.urban_routing.data_collector import DataCollector
try:
    from data_collector import serialize_hdv_route_state
except ImportError:
    from dreamer4.urban_routing.data_collector import serialize_hdv_route_state

try:
    from train_marl import (
        LazyVehiclePathCache,
        VehiclePathCache,
        build_diverse_paths_current_to_dest,
        build_vehicle_fleet_pressure_obs_features,
        build_vehicle_obs_vectorized,
        build_vehicle_obs_from_path_batch,
        edge_tt_from_latent_state,
        vehicle_mode_obs_dim,
    )
except Exception:
    try:
        from dreamer4.urban_routing.train_marl import (
            LazyVehiclePathCache,
            VehiclePathCache,
            build_diverse_paths_current_to_dest,
            build_vehicle_fleet_pressure_obs_features,
            build_vehicle_obs_vectorized,
            build_vehicle_obs_from_path_batch,
            edge_tt_from_latent_state,
            vehicle_mode_obs_dim,
        )
    except Exception:
        LazyVehiclePathCache = None
        VehiclePathCache = None
        build_diverse_paths_current_to_dest = None
        build_vehicle_fleet_pressure_obs_features = None
        build_vehicle_obs_vectorized = None
        build_vehicle_obs_from_path_batch = None
        edge_tt_from_latent_state = None
        vehicle_mode_obs_dim = None

try:
    from wm_reranker import rerank_topk_candidates, rollout_candidate_metrics
except Exception:
    from dreamer4.urban_routing.wm_reranker import rerank_topk_candidates, rollout_candidate_metrics

try:
    from dreamer4.urban_routing.urb_fair_protocol import (
        build_post_mutation_state,
        derive_fixed_roles,
        load_post_mutation_state,
        verify_post_mutation_state,
        write_post_mutation_state,
    )
except Exception:
    try:
        from urb_fair_protocol import (
            build_post_mutation_state,
            derive_fixed_roles,
            load_post_mutation_state,
            verify_post_mutation_state,
            write_post_mutation_state,
        )
    except Exception:
        build_post_mutation_state = None
        derive_fixed_roles = None
        load_post_mutation_state = None
        verify_post_mutation_state = None
        write_post_mutation_state = None

try:
    from dreamer4.urban_routing.dreamer_actor_critic import VehicleRouteActor
except Exception:
    try:
        from dreamer_actor_critic import VehicleRouteActor
    except Exception:
        VehicleRouteActor = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # noqa: E731

def load_policy(
    ckpt_path: str,
    obs_dim:   int,
    n_routes:  int,
    device:    str,
) -> nn.Module:
    """Load trained policy from a Dreamer4 MARL checkpoint. Auto-detects policy type."""
    print(f"  Loading policy checkpoint: {ckpt_path}")
    state    = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = state.get("cfg", {})
    model_family = str(state.get("model_family", cfg_dict.get("training_family", ""))).strip().lower()
    if model_family != "dreamer_ctde":
        raise RuntimeError(f"Unsupported policy checkpoint family {model_family!r}; expected dreamer_ctde.")

    hidden_dim      = int(cfg_dict.get("hidden_dim",     256))
    n_hidden_layers = int(cfg_dict.get("n_hidden_layers", 2))

    if VehicleRouteActor is None:
        raise RuntimeError("VehicleRouteActor import unavailable for Dreamer checkpoint loading.")
    route_cost_logit_bias = (
        float(cfg_dict.get("vehicle_actor_route_cost_logit_bias", 0.0))
        if bool(cfg_dict.get("vehicle_obs_include_route_costs", False))
        else 0.0
    )
    policy = VehicleRouteActor(
        obs_dim,
        n_routes,
        hidden_dim,
        n_hidden_layers,
        route_cost_logit_bias=route_cost_logit_bias,
        architecture=str(cfg_dict.get("vehicle_actor_architecture", "mlp")),
        include_global=bool(cfg_dict.get("vehicle_obs_include_global", True)),
        include_route_costs=bool(cfg_dict.get("vehicle_obs_include_route_costs", False)),
        include_fleet_pressure=bool(cfg_dict.get("vehicle_obs_include_fleet_pressure", False)),
    ).to(device)
    actor_state = state.get("actor", state.get("policy"))
    if actor_state is None:
        raise RuntimeError(f"Dreamer checkpoint missing actor state: {ckpt_path}")
    policy.load_state_dict(actor_state)
    policy.eval()
    episode_saved = state.get("episode", "?")
    print(f"  VehicleRouteActor loaded (saved at episode {episode_saved})")
    return policy


def _append_vehicle_context_features(
    obs_vehicle_np: np.ndarray,
    *,
    step_idx: int,
    episode_steps: int,
    n_active_av: int,
    n_total_av: int,
) -> np.ndarray:
    """Match train_marl vehicle-mode obs shape by appending 3 context features."""
    if obs_vehicle_np.ndim != 2:
        return obs_vehicle_np
    context_features = np.array([
        float(step_idx) / max(int(episode_steps), 1),
        float(n_active_av) / max(int(n_total_av), 1),
        0.0,
    ], dtype=np.float32)
    if obs_vehicle_np.shape[0] == 0:
        return np.zeros((0, obs_vehicle_np.shape[1] + 3), dtype=np.float32)
    ctx_tiled = np.tile(context_features, (obs_vehicle_np.shape[0], 1))
    return np.concatenate([obs_vehicle_np, ctx_tiled], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Flow-based observation encoder
# ══════════════════════════════════════════════════════════════════════════════

class FlowObsEncoder:
    """
    Capture current SUMO traffic state via traci edge data,
    extract flow features via EdgeRegistry, and encode to latent.
    """

    def __init__(self, wm, device: str):
        self.wm     = wm
        self.device = torch.device(device)

    def _build_traci_edges_state(self) -> dict:
        """Extract edge-level data via traci."""
        edges_state = {}
        if not TRACI_AVAILABLE:
            return edges_state
        try:
            for edge_id in traci.edge.getIDList():
                if edge_id.startswith(":"):
                    continue
                try:
                    edges_state[edge_id] = {
                        "occupancy":      traci.edge.getLastStepOccupancy(edge_id),
                        "mean_speed":     traci.edge.getLastStepMeanSpeed(edge_id),
                        "vehicle_number": traci.edge.getLastStepVehicleNumber(edge_id),
                        "halting_number": traci.edge.getLastStepHaltingNumber(edge_id),
                    }
                except Exception:
                    pass
        except Exception:
            pass
        return edges_state

    def try_encode(self) -> Optional[torch.Tensor]:
        """
        Try to capture + encode current traffic state via flow features.

        Returns:
            Projected tokens (n_spatial, d_spatial) on success, None on failure.
        """
        edges_state = self._build_traci_edges_state()
        if not edges_state:
            return None
        try:
            state = {"edges": edges_state}
            z = self.wm.encode_observation(state)
            return z
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# Travel-time metrics from RouteRL episode CSVs
# ══════════════════════════════════════════════════════════════════════════════

_AV_KIND_VALUES  = {"av", "machine", "cav"}
_HDV_KIND_VALUES = {"human", "hdv"}

def _detect_kind_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("vType", "kind", "type", "agent_type"):
        if col in df.columns:
            return col
    return None


def compute_metrics_from_csvs(
    episodes_folder: str,
    test_start_ep:   int,
    test_end_ep:     int,
) -> Dict:
    tt_av  = []
    tt_hdv = []
    tt_all = []

    if not os.path.isdir(episodes_folder):
        print(f"  WARNING: episodes folder not found: {episodes_folder}")
        return {"t_CAV": float("nan"), "t_HDV": float("nan"), "t_all": float("nan"),
                "n_AV_trips": 0, "n_HDV_trips": 0, "n_total_trips": 0}

    for ep in range(test_start_ep, test_end_ep + 1):
        csv_path = os.path.join(episodes_folder, f"ep{ep}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  WARNING: could not read {csv_path}: {e}")
            continue

        if df.empty or "travel_time" not in df.columns:
            continue

        df["travel_time"] = pd.to_numeric(df["travel_time"], errors="coerce")
        df = df.dropna(subset=["travel_time"])

        kind_col = _detect_kind_column(df)
        if kind_col is not None:
            kinds = df[kind_col].astype(str).str.lower()
            av_mask  = kinds.isin(_AV_KIND_VALUES)
            hdv_mask = kinds.isin(_HDV_KIND_VALUES)
            tt_av.extend(df.loc[av_mask,  "travel_time"].tolist())
            tt_hdv.extend(df.loc[hdv_mask, "travel_time"].tolist())
        else:
            print(f"  NOTE: ep{ep}.csv has no kind/vType column – treating all as 'all'")

        tt_all.extend(df["travel_time"].tolist())

    def _safe_mean_minutes(lst):
        return float(np.mean(lst)) if lst else float("nan")

    return {
        "t_CAV":         _safe_mean_minutes(tt_av),
        "t_HDV":         _safe_mean_minutes(tt_hdv),
        "t_all":         _safe_mean_minutes(tt_all),
        "n_AV_trips":    len(tt_av),
        "n_HDV_trips":   len(tt_hdv),
        "n_total_trips": len(tt_all),
    }


def _summarize_test_episode_metrics(test_df: pd.DataFrame) -> Dict[str, Any]:
    """Use the final test episode as the headline metric.

    The per-episode CSV still records every rollout. The summary keeps the old
    mean values under explicit *_mean keys, while t_CAV/t_HDV/t_all are the
    final test episode values used for judging.
    """
    if len(test_df) == 0:
        return {
            "metric_aggregation": "last_test_episode",
            "summary_episode": None,
            "t_CAV": float("nan"),
            "t_HDV": float("nan"),
            "t_all": float("nan"),
            "t_CAV_mean": float("nan"),
            "t_HDV_mean": float("nan"),
            "t_all_mean": float("nan"),
            "arrived_all": 0,
            "total_agents": 0,
            "unfinished_all": 0,
            "completion_rate": float("nan"),
            "completion_rate_mean": float("nan"),
        }

    ordered = test_df.copy()
    if "episode" in ordered.columns:
        ordered["_episode_sort"] = pd.to_numeric(ordered["episode"], errors="coerce")
        ordered = ordered.sort_values("_episode_sort", kind="stable")

    last = ordered.iloc[-1]
    summary_episode: Any = None
    if "episode" in last.index and pd.notna(last["episode"]):
        try:
            summary_episode = int(last["episode"])
        except (TypeError, ValueError):
            summary_episode = str(last["episode"])

    def _float_col_mean(col: str) -> float:
        if col not in test_df.columns:
            return float("nan")
        vals = pd.to_numeric(test_df[col], errors="coerce")
        return float(vals.mean()) if vals.notna().any() else float("nan")

    def _int_last(col: str) -> int:
        if col not in last.index or pd.isna(last[col]):
            return 0
        try:
            return int(last[col])
        except (TypeError, ValueError):
            return 0

    return {
        "metric_aggregation": "last_test_episode",
        "summary_episode": summary_episode,
        "t_CAV": float(last["t_CAV"]),
        "t_HDV": float(last["t_HDV"]),
        "t_all": float(last["t_all"]),
        "t_CAV_mean": float(test_df["t_CAV"].mean()),
        "t_HDV_mean": float(test_df["t_HDV"].mean()),
        "t_all_mean": float(test_df["t_all"].mean()),
        "arrived_all": _int_last("arrived_all"),
        "total_agents": _int_last("total_agents"),
        "unfinished_all": _int_last("unfinished_all"),
        "completion_rate": float(last["completion_rate"]) if "completion_rate" in last.index and pd.notna(last["completion_rate"]) else float("nan"),
        "completion_rate_mean": _float_col_mean("completion_rate"),
    }


def _episode_complete_for_dataset(*, n_arrived: int, total_agents: int, n_steps: int) -> bool:
    """Return True only when an episode is valid to save as rollout data."""
    return int(total_agents) > 0 and int(n_arrived) >= int(total_agents) and int(n_steps) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Dreamer4 MARL policy in SUMO (repo-local network settings)"
    )
    add_settings_arg(p)
    # Checkpoints
    p.add_argument("--policy_ckpt",    default="logs/marl/latest.pt",
                   help="Trained MARL policy checkpoint")
    p.add_argument("--projector_ckpt", default="logs/urban_dynamics/flow_projector_latest.pt",
                   help="Trained FlowProjector checkpoint")
    p.add_argument("--dynamics_ckpt",  default="logs/urban_dynamics/latest.pt",
                   help="Trained dynamics checkpoint")
    p.add_argument("--edge_registry",  default=str(_DEFAULT_EDGE_REGISTRY),
                   help="Path to edge_registry.json")

    # Network / environment
    p.add_argument("--network",        default=_DEFAULT_NETWORK_NAME)
    p.add_argument("--network_folder", default=str(_DEFAULT_NETWORK_FOLDER))
    p.add_argument("--n_routes",       type=int,   default=4)
    p.add_argument("--bpr_alpha",      type=float, default=0.15)
    p.add_argument("--bpr_beta",       type=float, default=4.0)

    # OpenURB task settings
    p.add_argument("--ratio_machines", type=float, default=0.4)
    p.add_argument("--human_learning_episodes", type=int, default=300)
    p.add_argument("--test_eps",       type=int,   default=100)
    p.add_argument("--hdv_gawron_mode", type=str, default="routerl",
                   choices=["replicator", "routerl", "routerl_blackbox"],
                   help="HDV route update mode across episodes. HDV routes remain fixed within each episode.")
    p.add_argument("--hdv_gawron_alpha_zero", type=float, default=0.2,
                   help="RouteRL-compatible alpha_zero for HDV route adaptation (used when hdv_gawron_mode=routerl).")
    p.add_argument("--hdv_gawron_beta", type=float, default=-1.5,
                   help="RouteRL-compatible beta for HDV route choice (used when hdv_gawron_mode=routerl).")
    p.add_argument("--hdv_gawron_update_mode", type=str, default="expected",
                   choices=["expected", "sample"],
                   help="HDV route adaptation style across episodes when hdv_gawron_mode=routerl.")

    # Output
    p.add_argument("--results_dir",    default="results/eval_dreamer4")

    # Misc
    p.add_argument("--device",         default="cuda")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--deterministic",  action="store_true", default=True)
    p.add_argument("--stochastic",     dest="deterministic", action="store_false")
    p.add_argument("--explore_eps", type=float, default=0.0,
                   help="Epsilon-mixing with uniform distribution when sampling AV actions.")
    p.add_argument("--dynamic_routing", action="store_true", default=False,
                   help="Re-route active AV vehicles at each SUMO timestep using policy probabilities.")
    p.add_argument("--av_candidate_source", type=str, default="current_shortest_wm",
                   choices=["od_suffix", "current_shortest_live", "current_shortest_wm"],
                   help=(
                       "Candidate route source for dynamic AV decisions. 'od_suffix' preserves the "
                       "precomputed OD-route suffix behavior; 'current_shortest_live' and "
                       "'current_shortest_wm' regenerate K diverse current-edge-to-destination "
                       "routes from live SUMO edge travel times or WM/BPR edge travel times."
                   ))
    p.add_argument("--av_policy_route_selection", type=str, default="auto",
                   choices=["auto", "argmax", "sample", "quota"],
                   help=(
                       "How policy route probabilities become discrete AV routes. auto preserves "
                       "--deterministic/--stochastic behavior; quota deterministically realizes "
                       "fleet-level probability mass within each decision tick."
                   ))
    p.add_argument("--av_policy_sample_temperature", type=float, default=1.0,
                   help=(
                       "Temperature applied when policy probabilities are converted to sampled/quota "
                       "route choices. Values below 1 sharpen the distribution; argmax selection is unchanged."
                   ))
    p.add_argument("--av_policy_confidence_gate_source", type=str, default="none",
                   choices=["none", "wm_current_bpr", "live_current_tt", "freeflow_greedy"],
                   help=(
                       "Optional deterministic policy guardrail. For argmax route selection, if the policy's "
                       "top route is less than --av_policy_confidence_gate_margin more probable than the "
                       "selected fallback source's cost-best route, choose the fallback route instead. "
                       "This preserves candidate slot order and is disabled by default."
                   ))
    p.add_argument("--av_policy_confidence_gate_margin", type=float, default=0.0,
                   help="Probability margin used by --av_policy_confidence_gate_source.")
    p.add_argument("--av_policy_confidence_gate_trigger_share", type=float, default=0.0,
                   help=(
                       "If >0, apply the confidence gate only when the baseline policy assigns at least this "
                       "fraction of active AVs to one nonzero route slot in the current decision tick. "
                       "This is useful as an anti-collapse guardrail while leaving ordinary mixed policies intact."
                   ))
    p.add_argument("--av_policy_nonzero_route_share_cap", type=float, default=0.0,
                   help=(
                       "Optional deterministic fleet-share guardrail for policy argmax. If >0, each nonzero "
                       "candidate route slot can receive at most this fraction of active AVs in a decision tick; "
                       "excess low-confidence selections fall back to the selected cost-best source."
                   ))
    p.add_argument("--av_policy_share_cap_fallback_source", type=str, default="wm_current_bpr",
                   choices=["wm_current_bpr", "live_current_tt", "freeflow_greedy"],
                   help="Fallback route-cost source used by --av_policy_nonzero_route_share_cap.")
    p.add_argument("--av_od_route_override_json", type=str, default="",
                   help="Counterfactual eval: JSON mapping/list of OD route overrides, e.g. {'4,20': 3}.")
    p.add_argument("--av_route_cost_policy", type=str, default="none",
                   choices=["none", "freeflow_greedy", "wm_current_bpr", "live_current_tt", "wm_live_blend"],
                   help=(
                       "No-leak route-cost baseline. 'freeflow_greedy' chooses the lowest static "
                       "free-flow candidate suffix; 'wm_current_bpr' chooses the lowest-cost candidate "
                       "route using current WM-decoded edge travel times; 'live_current_tt' uses current "
                       "SUMO edge mean speeds; 'wm_live_blend' interpolates between live and WM route "
                       "costs. No OD-specific lookup is used."
                   ))
    p.add_argument("--av_route_cost_length_norm", type=str, default="sum",
                   choices=["sum", "mean"],
                   help="How to aggregate edge costs along each candidate route for --av_route_cost_policy.")
    p.add_argument("--av_route_cost_temperature", type=float, default=0.0,
                   help="If >0 and --stochastic is used, sample route-cost choices from softmax(-cost/temp). Units are seconds.")
    p.add_argument("--av_route_hdv_pressure_penalty", type=float, default=0.0,
                   help=(
                       "Optional system-aware penalty added to route-cost scores: "
                       "penalty_seconds * sum(current normalized HDV edge occupancy on the candidate route). "
                       "Use with --av_route_cost_policy to discourage rerouting AVs onto currently HDV-heavy corridors."
                   ))
    p.add_argument("--av_route_wm_live_blend_weight", type=float, default=0.5,
                   help="Blend weight for --av_route_cost_policy wm_live_blend: 0=live_current_tt, 1=WM route cost.")
    p.add_argument("--av_route_planned_av_pressure_penalty", type=float, default=0.0,
                   help=(
                       "Optional sequential same-tick AV externality penalty in seconds. During route-cost "
                       "selection, each already-selected AV path adds normalized planned edge pressure to "
                       "later AV route scores."
                   ))
    p.add_argument("--wm_mcc_diagnostic_penalties", type=str, default="",
                   help=(
                       "Comma/space separated planned-AV pressure penalties for WM-MCC diagnostic logging. "
                       "When set, eval writes wm_mcc_diagnostic.csv with WM rollout scores for each penalty."
                   ))
    p.add_argument("--wm_mcc_diagnostic_horizon", type=int, default=1,
                   help="WM rollout horizon for WM-MCC diagnostics.")
    p.add_argument("--mpc_horizon", type=int, default=0,
                   help="MPC lookahead steps (0=disabled, use policy directly).")
    p.add_argument("--mpc_candidates", type=int, default=5,
                   help="Number of action candidates to evaluate in MPC.")
    p.add_argument("--mpc_discount", type=float, default=0.99,
                   help="Discount factor for MPC cumulative reward.")
    p.add_argument("--wm_rerank_t_all_weight", type=float, default=1.0,
                   help="Weight on predicted t_all in WM-assisted reranking.")
    p.add_argument("--wm_rerank_t_cav_weight", type=float, default=2.0,
                   help="Weight on predicted AV travel-time proxy in WM-assisted reranking.")
    p.add_argument("--wm_rerank_cav_tolerance", type=float, default=0.0,
                   help="Hard-gate tolerance: candidates with predicted t_CAV worse than baseline+tolerance cannot override baseline.")
    p.add_argument("--wm_cfm_init_mode", choices=["randn", "zero"], default="zero",
                   help="CFM rollout source for the world model during SUMO evaluation.")
    p.add_argument("--wm_policy_obs_source", choices=["auto", "decoder_volume_tt", "decoder_volume_tt_route_head", "decoder_volume", "decoder", "obs_proj"], default="auto",
                   help="Policy observation source. auto uses policy checkpoint cfg when available.")
    p.add_argument("--wm_edge_cost_source", choices=["bpr_volume", "decoded_speed", "route_head"], default="decoded_speed",
                   help="Source for WM-derived edge travel times in current_shortest_wm and WM route-cost policy.")
    p.add_argument("--wm_candidate_edge_cost_source", choices=["auto", "bpr_volume", "decoded_speed", "route_head"], default="auto",
                   help="WM edge cost source used only to generate current_shortest_wm candidate paths. auto reuses --wm_edge_cost_source.")
    p.add_argument("--wm_decoded_speed_min_ratio", type=float, default=0.05,
                   help="Minimum decoded speed/speed_limit ratio when --wm_edge_cost_source=decoded_speed.")
    p.add_argument("--wm_decoded_speed_max_ratio", type=float, default=1.50,
                   help="Maximum decoded speed/speed_limit ratio when --wm_edge_cost_source=decoded_speed.")
    p.add_argument("--wm_decoded_speed_excess_scale", type=float, default=1.0,
                   help="Blend factor for decoded-speed excess delay: 1.0 gives free_flow/speed_ratio; lower values damp global low-speed bias.")
    p.add_argument("--vehicle_obs_route_stats", type=str, default="auto",
                   choices=["auto", "mean", "bottleneck"],
                   help="Vehicle actor route observation aggregation. auto uses policy checkpoint cfg.")
    p.add_argument("--vehicle_obs_include_route_costs", action="store_true",
                   help=(
                       "Force route-cost observation features for vehicle actor eval. "
                       "By default this is inferred from the policy checkpoint cfg."
                   ))
    p.add_argument("--wm_volume_floor", type=float, default=0.10,
                   help="Normalized decoded volume below this floor is treated as inactive for WM planner scoring.")
    p.add_argument("--greedy_routing", action="store_true", default=False,
                   help="Use BPR travel-time greedy routing instead of policy (for data collection).")
    p.add_argument("--episode_steps", type=int, default=60,
                   help="Number of decision/recording steps per episode. SUMO still runs at native tick resolution.")
    p.add_argument("--sim_buffer_after_last_depart", type=int, default=1000,
                   help="Extra SUMO ticks to run after the last departure time.")
    p.add_argument("--sumo_depart_lane", type=str, default="",
                   help="Optional traci.vehicle.add departLane value, e.g. best/free.")
    p.add_argument("--sumo_depart_pos", type=str, default="",
                   help="Optional traci.vehicle.add departPos value, e.g. free/random_free.")
    p.add_argument("--sumo_depart_speed", type=str, default="",
                   help="Optional traci.vehicle.add departSpeed value, e.g. max/desired.")
    p.add_argument("--sumo_time_to_teleport", type=str, default="",
                   help="Override RouteRL's SUMO --time-to-teleport value. Empty preserves RouteRL default -1.")
    p.add_argument("--sumo_eager_insert", action="store_true",
                   help="Pass SUMO --eager-insert for insertion diagnostics.")
    p.add_argument("--sumo_sloppy_insert", action="store_true",
                   help="Pass SUMO --sloppy-insert for insertion diagnostics.")
    p.add_argument("--sumo_emergency_insert", action="store_true",
                   help="Pass SUMO --emergency-insert for insertion diagnostics.")
    p.add_argument("--sumo_extra_args", type=str, default="",
                   help="Extra SUMO CLI args parsed with shlex, e.g. '--ignore-route-errors'.")
    p.add_argument("--incident_edge", type=str, default="",
                   help="Optional edge id whose lanes are slowed during [incident_start, incident_end).")
    p.add_argument("--incident_start", type=float, default=0.0,
                   help="SUMO time in seconds when the optional incident starts.")
    p.add_argument("--incident_end", type=float, default=0.0,
                   help="SUMO time in seconds when the optional incident ends.")
    p.add_argument("--incident_speed", type=float, default=0.0,
                   help="Temporary incident lane max speed in m/s. <=0 disables the incident.")
    p.add_argument("--max_steps_per_episode", type=int, default=0,
                   help="Override SUMO rollout length per episode; 0 uses the computed sim length.")
    p.add_argument("--route_generator", type=str, default="dijkstra",
                   choices=["dijkstra", "routerl"],
                   help="How to generate routes.csv/route.rou.xml once at startup.")
    p.add_argument("--reuse_artifacts_dir", type=str, default="",
                   help="If set, reuse agents.csv/routes.csv/route.rou.xml from this directory instead of regenerating route artifacts.")
    p.add_argument("--path_gen_beta", type=float, default=-5.0,
                   help="RouteRL path-generation beta when --route_generator=routerl.")
    p.add_argument("--path_num_samples", type=int, default=10,
                   help="RouteRL candidate path samples when --route_generator=routerl.")
    p.add_argument("--save_policy_ckpt", type=str, default="",
                   help="Optional path to save policy checkpoint after online PPO.")
    p.add_argument("--collect_dataset_dir", type=str, default="",
                   help="If set, collect SUMO rollout as flow dataset at this directory.")
    p.add_argument("--collect_task_name", type=str, default="flow_online",
                   help="Task name used in saved dataset file and flow shard folder.")
    p.add_argument("--collect_shard_size", type=int, default=2048)
    p.add_argument("--hdv_state_in", type=str, default="",
                   help="Optional JSON path to restore HDV learning state before rollout.")
    p.add_argument("--hdv_state_out", type=str, default="",
                   help="Optional JSON path to save HDV learning state after rollout.")
    p.add_argument("--post_mutation_state_in", type=str, default="",
                   help="Optional JSON path to load fixed post-mutation machine/human state for the test phase.")
    p.add_argument("--post_mutation_state_out", type=str, default="",
                   help="Optional JSON path to save fixed post-mutation machine/human state after adaptation.")
    p.add_argument("--episode_log_every", type=int, default=10,
                   help="Print per-episode t_all progress every N episodes during human adaptation and rollout.")
    p.add_argument("--save_every", type=int, default=1,
                   help="RouteRL save_every for episode CSVs. Use 1 for per-episode metrics.")
    p.add_argument("--hdv_adapt_episodes", type=int, default=0,
                   help="Number of HDV adaptation episodes to run before sampling HDV distribution (0=disabled).")
    p.add_argument("--hdv_sample_episodes", type=int, default=3,
                   help="Number of episodes to sample HDV distribution after adaptation.")
    p.add_argument("--hdv_adapt_interval", type=int, default=5,
                   help="Run HDV adaptation every N AV episodes during data collection.")
    mut_group = p.add_mutually_exclusive_group()
    mut_group.add_argument(
        "--disable_human_learning_after_mutation",
        dest="disable_human_learning_after_mutation",
        action="store_true",
        help="Freeze human (HDV) learning after mutation.",
    )
    mut_group.add_argument(
        "--keep_human_learning_after_mutation",
        dest="disable_human_learning_after_mutation",
        action="store_false",
        help="Keep human (HDV) Gawron learning active after mutation.",
    )
    p.set_defaults(disable_human_learning_after_mutation=True)
    settings_path = parse_settings_arg()
    apply_stage_defaults(p, settings_path, stage="eval")
    return p.parse_args()


def _load_av_od_route_overrides(path_value: str) -> Dict[Tuple[int, int], int]:
    if not str(path_value or "").strip():
        return {}
    path = Path(str(path_value))
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("overrides"), list):
        data = data["overrides"]

    overrides: Dict[Tuple[int, int], int] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            origin = item.get("origin", item.get("o"))
            destination = item.get("destination", item.get("dest", item.get("d")))
            action = item.get("action", item.get("route_index", item.get("route")))
            if origin is None or destination is None or action is None:
                continue
            overrides[(int(origin), int(destination))] = int(action)
        return overrides

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                action = value.get("action", value.get("route_index", value.get("route")))
            else:
                action = value
            if action is None:
                continue
            if isinstance(key, str):
                parts = key.replace(":", ",").split(",")
                if len(parts) != 2:
                    continue
                origin, destination = int(parts[0].strip()), int(parts[1].strip())
            elif isinstance(key, (list, tuple)) and len(key) == 2:
                origin, destination = int(key[0]), int(key[1])
            else:
                continue
            overrides[(origin, destination)] = int(action)
        return overrides

    raise ValueError(
        f"Unsupported --av_od_route_override_json format in {path}; expected list or object."
    )


def _parse_float_list(value: str) -> List[float]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in text.replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    vals = [float(p) for p in parts]
    if any(not np.isfinite(v) for v in vals):
        raise ValueError(f"Expected finite float list, got {value!r}")
    return vals


def _edge_route_cost_matrix(
    path_batch: List[List[List[int]]],
    edge_cost: np.ndarray,
    *,
    n_routes: int,
    length_norm: str,
) -> np.ndarray:
    """Compute per-vehicle, per-route costs from an edge-indexed cost vector."""
    edge_cost_arr = np.asarray(edge_cost, dtype=np.float32).reshape(-1)
    n_agents = int(len(path_batch))
    k = int(max(n_routes, 1))
    costs = np.full((n_agents, k), np.inf, dtype=np.float32)
    use_mean = str(length_norm).strip().lower() == "mean"

    for i, paths in enumerate(path_batch):
        for route_idx, path in enumerate(list(paths)[:k]):
            if not path:
                continue
            idxs = np.asarray(path, dtype=np.int64)
            idxs = idxs[(idxs >= 0) & (idxs < edge_cost_arr.shape[0])]
            if idxs.size == 0:
                continue
            vals = edge_cost_arr[idxs]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            costs[i, route_idx] = float(vals.mean() if use_mean else vals.sum())
    return costs


def _select_min_cost_route_indices(
    cost_matrix: np.ndarray,
    baseline_indices: np.ndarray,
    *,
    deterministic: bool,
    rng: np.random.RandomState,
    temperature: float,
) -> Tuple[np.ndarray, int]:
    """Select lowest-cost route per vehicle, falling back to baseline when costs are invalid."""
    costs = np.asarray(cost_matrix, dtype=np.float32)
    selected = np.asarray(baseline_indices, dtype=np.int64).copy()
    if costs.ndim != 2 or selected.shape[0] != costs.shape[0]:
        return selected, 0

    applied = 0
    temp = float(temperature)
    for i in range(costs.shape[0]):
        row = costs[i]
        finite = np.isfinite(row)
        if not bool(finite.any()):
            continue
        finite_indices = np.flatnonzero(finite).astype(np.int64)
        finite_costs = row[finite_indices].astype(np.float64)
        if (not deterministic) and temp > 0.0 and finite_costs.size > 1:
            logits = -(finite_costs - float(finite_costs.min())) / max(temp, 1e-6)
            logits = logits - float(logits.max())
            probs = np.exp(logits)
            denom = float(probs.sum())
            if np.isfinite(denom) and denom > 0.0:
                probs = probs / denom
                selected[i] = int(rng.choice(finite_indices, p=probs))
            else:
                selected[i] = int(finite_indices[int(np.argmin(finite_costs))])
        else:
            selected[i] = int(finite_indices[int(np.argmin(finite_costs))])
        applied += 1
    return selected, applied


def _select_min_cost_route_indices_with_planned_pressure(
    cost_matrix: np.ndarray,
    baseline_indices: np.ndarray,
    path_batch: Sequence[Sequence[Sequence[int]]],
    *,
    n_edges: int,
    deterministic: bool,
    rng: np.random.RandomState,
    temperature: float,
    pressure_penalty: float,
) -> Tuple[np.ndarray, int]:
    """Sequentially select routes with a same-tick planned-AV pressure penalty."""
    penalty = max(float(pressure_penalty), 0.0)
    if penalty <= 0.0:
        return _select_min_cost_route_indices(
            cost_matrix,
            baseline_indices,
            deterministic=deterministic,
            rng=rng,
            temperature=temperature,
        )
    costs = np.asarray(cost_matrix, dtype=np.float32)
    selected = np.asarray(baseline_indices, dtype=np.int64).copy()
    if costs.ndim != 2 or selected.shape[0] != costs.shape[0]:
        return selected, 0
    edge_pressure = np.zeros((max(int(n_edges), 0),), dtype=np.float32)
    norm = max(float(costs.shape[0]), 1.0)
    applied = 0
    temp = float(temperature)
    for i in range(costs.shape[0]):
        row = costs[i].astype(np.float64).copy()
        finite = np.isfinite(row)
        if not bool(finite.any()):
            continue
        routes = path_batch[i] if i < len(path_batch) else []
        for k in range(min(len(routes), row.shape[0])):
            idxs = [int(e) for e in routes[k] if 0 <= int(e) < edge_pressure.shape[0]]
            if idxs:
                row[k] += penalty * float(edge_pressure[idxs].sum()) / norm
        finite_indices = np.flatnonzero(finite).astype(np.int64)
        finite_costs = row[finite_indices].astype(np.float64)
        if (not deterministic) and temp > 0.0 and finite_costs.size > 1:
            logits = -(finite_costs - float(finite_costs.min())) / max(temp, 1e-6)
            logits = logits - float(logits.max())
            probs = np.exp(logits)
            denom = float(probs.sum())
            if np.isfinite(denom) and denom > 0.0:
                choice = int(rng.choice(finite_indices, p=probs / denom))
            else:
                choice = int(finite_indices[int(np.argmin(finite_costs))])
        else:
            choice = int(finite_indices[int(np.argmin(finite_costs))])
        selected[i] = choice
        if i < len(routes) and 0 <= choice < len(routes):
            for edge_idx in routes[choice]:
                e = int(edge_idx)
                if 0 <= e < edge_pressure.shape[0]:
                    edge_pressure[e] += 1.0
        applied += 1
    return selected, applied


def _select_quota_route_indices(
    route_probs: np.ndarray,
    baseline_indices: np.ndarray,
    *,
    n_routes: int,
) -> np.ndarray:
    """Deterministically realize fleet-level policy probabilities as route counts.

    Independent per-vehicle argmax collapses when one route is only slightly
    better for every vehicle. Quota selection preserves the actor's aggregate
    probability mass within the current decision tick without using route labels
    or future simulator outcomes.
    """
    probs = np.asarray(route_probs, dtype=np.float64)
    baseline = np.asarray(baseline_indices, dtype=np.int64).reshape(-1)
    n_agents = int(baseline.shape[0])
    k = max(int(n_routes), 1)
    if probs.ndim != 2 or probs.shape[0] != n_agents or n_agents == 0:
        return baseline.copy()

    probs = probs[:, :k].copy()
    probs[~np.isfinite(probs)] = 0.0
    probs = np.clip(probs, 0.0, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    valid_rows = row_sums[:, 0] > 0.0
    if not bool(valid_rows.any()):
        return baseline.copy()
    probs[valid_rows] = probs[valid_rows] / row_sums[valid_rows]
    probs[~valid_rows] = 0.0
    for i in np.flatnonzero(~valid_rows):
        b = int(np.clip(baseline[i], 0, k - 1))
        probs[i, b] = 1.0

    expected = probs.sum(axis=0)
    target = np.floor(expected).astype(np.int64)
    remainder = int(n_agents - int(target.sum()))
    if remainder > 0:
        frac_order = np.argsort(-(expected - target), kind="mergesort")
        for ridx in frac_order[:remainder]:
            target[int(ridx)] += 1
    elif remainder < 0:
        frac_order = np.argsort(expected - target, kind="mergesort")
        for ridx in frac_order[: -remainder]:
            if target[int(ridx)] > 0:
                target[int(ridx)] -= 1

    selected = np.full((n_agents,), -1, dtype=np.int64)
    remaining_agents = set(range(n_agents))
    route_order = np.argsort(-target, kind="mergesort")
    for ridx in route_order:
        quota = int(target[int(ridx)])
        if quota <= 0 or not remaining_agents:
            continue
        candidates = np.asarray(sorted(remaining_agents), dtype=np.int64)
        scores = probs[candidates, int(ridx)]
        order = np.argsort(-scores, kind="mergesort")
        take = candidates[order[: min(quota, candidates.size)]]
        selected[take] = int(ridx)
        for idx in take.tolist():
            remaining_agents.discard(int(idx))

    if remaining_agents:
        for idx in sorted(remaining_agents):
            selected[int(idx)] = int(np.argmax(probs[int(idx)]))
    invalid = selected < 0
    if bool(invalid.any()):
        selected[invalid] = baseline[invalid]
    return selected.clip(0, k - 1).astype(np.int64)


def _route_cost_matrix_summary(
    cost_matrix: Optional[np.ndarray],
    n_routes: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    means = np.full((int(n_routes),), np.nan, dtype=np.float32)
    best_counts = np.zeros((int(n_routes),), dtype=np.int64)
    if cost_matrix is None:
        return means, best_counts, 0
    costs = np.asarray(cost_matrix, dtype=np.float32)
    if costs.ndim != 2:
        return means, best_counts, 0
    k_eff = min(int(n_routes), int(costs.shape[1]))
    for k in range(k_eff):
        vals = costs[:, k]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            means[k] = float(vals.mean())
    total = 0
    for row in costs:
        row = np.asarray(row[:k_eff], dtype=np.float32)
        finite = np.isfinite(row)
        if not bool(finite.any()):
            continue
        finite_indices = np.flatnonzero(finite).astype(np.int64)
        finite_costs = row[finite_indices]
        best_idx = int(finite_indices[int(np.argmin(finite_costs))])
        if 0 <= best_idx < int(n_routes):
            best_counts[best_idx] += 1
            total += 1
    return means, best_counts, int(total)


def _build_edge_adj_list(adjacency: Optional[np.ndarray]) -> List[List[int]]:
    if adjacency is None:
        return []
    arr = np.asarray(adjacency)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return []
    adj: List[List[int]] = [[] for _ in range(int(arr.shape[0]))]
    rows, cols = np.nonzero(arr)
    for r, c in zip(rows.tolist(), cols.tolist()):
        ri = int(r)
        ci = int(c)
        if 0 <= ri < len(adj):
            adj[ri].append(ci)
    return adj


def _route_batch_uniqueness_summary(route_batch: List[List[List[str]]]) -> Tuple[float, int, int]:
    if not route_batch:
        return 0.0, 0, 0
    counts: List[int] = []
    for candidates in route_batch:
        unique = {tuple(route) for route in candidates if route}
        counts.append(int(len(unique)))
    if not counts:
        return 0.0, 0, 0
    arr = np.asarray(counts, dtype=np.float32)
    return float(arr.mean()), int(arr.min()), int(arr.max())


def _resolve_optional_path(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p


def _load_hdv_learning_state(state_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if state_path is None:
        return None
    if not state_path.exists():
        print(f"  [HDV state] input not found, skip restore: {state_path}")
        return None
    try:
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("top-level JSON is not an object")
        agents = state.get("agents", {})
        if not isinstance(agents, dict):
            raise ValueError("`agents` is not an object")
        print(f"  [HDV state] loaded {len(agents)} agent entries <- {state_path}")
        return state
    except Exception as exc:
        print(f"  [HDV state] WARNING: failed to load {state_path}: {exc}")
        return None


def _numeric_to_list(value: Any) -> Optional[list]:
    try:
        arr = np.asarray(value, dtype=np.float64)
        return arr.tolist()
    except Exception:
        return None


def _snapshot_hdv_learning_state(env) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for human in getattr(env, "human_agents", []):
        model = getattr(human, "model", None)
        if model is None:
            continue
        aid = str(getattr(human, "id", ""))
        if not aid:
            continue
        entry: Dict[str, Any] = {"model_class": model.__class__.__name__}
        if hasattr(model, "cost"):
            cost_list = _numeric_to_list(getattr(model, "cost"))
            if cost_list is not None:
                entry["cost"] = cost_list
        if hasattr(model, "memory"):
            mem_list = _numeric_to_list(getattr(model, "memory"))
            if mem_list is not None:
                entry["memory"] = mem_list
        if hasattr(model, "beta"):
            try:
                entry["beta"] = float(getattr(model, "beta"))
            except Exception:
                pass
        out[aid] = entry
    return out


def _merge_hdv_learning_state(
    base_state: Optional[Dict[str, Any]],
    update_agents: Dict[str, Dict[str, Any]],
    route_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged_agents: Dict[str, Dict[str, Any]] = {}
    if isinstance(base_state, dict):
        base_agents = base_state.get("agents", {})
        if isinstance(base_agents, dict):
            merged_agents.update(base_agents)
    merged_agents.update(update_agents)
    merged: Dict[str, Any] = {"version": 1, "agents": merged_agents}
    if isinstance(base_state, dict):
        if "route_policy" in base_state and isinstance(base_state["route_policy"], list):
            merged["route_policy"] = list(base_state["route_policy"])
    if route_state and isinstance(route_state.get("route_policy"), list):
        merged["route_policy"] = list(route_state["route_policy"])
    return merged


def _restore_hdv_learning_state(env, state: Optional[Dict[str, Any]], *, label: str) -> None:
    if not state:
        return
    agents_state = state.get("agents", {})
    if not isinstance(agents_state, dict):
        return

    humans = list(getattr(env, "human_agents", []))
    restored = 0
    missing = 0
    mismatched = 0
    no_model = 0

    for human in humans:
        aid = str(getattr(human, "id", ""))
        entry = agents_state.get(aid)
        if entry is None:
            missing += 1
            continue
        model = getattr(human, "model", None)
        if model is None:
            no_model += 1
            continue

        applied = False

        if "cost" in entry and hasattr(model, "cost"):
            try:
                cur = np.asarray(getattr(model, "cost"))
                new = np.asarray(entry["cost"], dtype=cur.dtype)
                if cur.shape == new.shape:
                    setattr(model, "cost", new.copy())
                    applied = True
                else:
                    mismatched += 1
            except Exception:
                mismatched += 1

        if "memory" in entry and hasattr(model, "memory"):
            try:
                cur_mem = np.asarray(getattr(model, "memory"))
                new_mem = np.asarray(entry["memory"], dtype=cur_mem.dtype)
                if cur_mem.shape == new_mem.shape:
                    setattr(model, "memory", new_mem.copy())
                    applied = True
                else:
                    mismatched += 1
            except Exception:
                mismatched += 1

        if "beta" in entry and hasattr(model, "beta"):
            try:
                setattr(model, "beta", float(entry["beta"]))
                applied = True
            except Exception:
                pass

        if applied:
            restored += 1

    print(
        f"  [HDV state/{label}] restored={restored}/{len(humans)} "
        f"missing={missing} mismatched={mismatched} no_model={no_model}",
        flush=True,
    )


def _save_hdv_learning_state(state_path: Optional[Path], state: Optional[Dict[str, Any]]) -> None:
    if state_path is None or not state:
        return
    agents = state.get("agents", {})
    if not isinstance(agents, dict):
        return
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(state, f)
        print(f"  [HDV state] saved {len(agents)} agent entries -> {state_path}")
    except Exception as exc:
        print(f"  [HDV state] WARNING: failed to save {state_path}: {exc}")


def _snapshot_hdv_route_state(collector) -> Optional[Dict[str, Any]]:
    return serialize_hdv_route_state(
        getattr(collector, "_hdv_route_probs", None),
        getattr(collector, "_hdv_route_costs", None),
    )


def _resolve_protocol_episode_indices(
    *,
    post_mutation_state: Optional[Dict[str, Any]],
    effective_human_eps: int,
    requested_human_eps: int,
) -> Tuple[int, int]:
    if post_mutation_state is None:
        return int(effective_human_eps), int(effective_human_eps)

    protocol_human_eps = int(
        post_mutation_state.get("human_learning_episodes", requested_human_eps)
    )
    mutation_episode_index = int(
        post_mutation_state.get("mutation_episode_index", protocol_human_eps)
    )
    return protocol_human_eps, mutation_episode_index


def _metrics_from_csv_set(csv_paths: set) -> Optional[Dict[str, float]]:
    """Compute mean travel-time metrics from a set of episode CSV files."""
    tt_av, tt_hdv, tt_all = [], [], []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "travel_time" not in df.columns:
            continue
        work = df.copy()
        work["travel_time"] = pd.to_numeric(work["travel_time"], errors="coerce")
        work = work.dropna(subset=["travel_time"])
        if work.empty:
            continue
        kind_col = next((c for c in ("vType", "kind", "type") if c in work.columns), None)
        if kind_col:
            kinds = work[kind_col].astype(str).str.lower()
            av_mask = kinds.isin({"av", "machine", "cav"})
            hdv_mask = kinds.isin({"human", "hdv"})
            tt_av.extend(work.loc[av_mask, "travel_time"].tolist())
            tt_hdv.extend(work.loc[hdv_mask, "travel_time"].tolist())
        tt_all.extend(work["travel_time"].tolist())
    if not tt_all:
        return None
    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")
    return {"t_CAV": _mean(tt_av), "t_HDV": _mean(tt_hdv), "t_all": _mean(tt_all)}


def _snapshot_episode_csvs(episodes_folder: str) -> Dict[str, int]:
    if not os.path.isdir(episodes_folder):
        return {}
    out: Dict[str, int] = {}
    for f in os.listdir(episodes_folder):
        if not (f.startswith("ep") and f.endswith(".csv")):
            continue
        p = os.path.join(episodes_folder, f)
        try:
            out[p] = os.stat(p).st_mtime_ns
        except OSError:
            continue
    return out


def _new_csvs_since(snapshot_before: Dict[str, int], episodes_folder: str, fallback_latest: int = 0) -> set[str]:
    latest = _snapshot_episode_csvs(episodes_folder)
    changed = {
        p for p, m in latest.items()
        if (p not in snapshot_before) or (m > snapshot_before[p])
    }
    if (not changed) and int(fallback_latest) > 0:
        ordered = sorted(latest.items(), key=lambda kv: kv[1], reverse=True)
        changed = {p for p, _ in ordered[: int(fallback_latest)]}
    return changed


def _snapshot_sumo_xmls(sumo_output_folder: str) -> Dict[str, int]:
    if not os.path.isdir(sumo_output_folder):
        return {}
    out: Dict[str, int] = {}
    for f in os.listdir(sumo_output_folder):
        if not f.endswith(".xml"):
            continue
        p = os.path.join(sumo_output_folder, f)
        try:
            out[p] = os.stat(p).st_mtime_ns
        except OSError:
            continue
    return out


def _new_sumo_xmls_since(snapshot_before: Dict[str, int], sumo_output_folder: str, fallback_latest: int = 0) -> set[str]:
    latest = _snapshot_sumo_xmls(sumo_output_folder)
    changed = {
        p for p, m in latest.items()
        if (p not in snapshot_before) or (m > snapshot_before[p])
    }
    if (not changed) and int(fallback_latest) > 0:
        ordered = sorted(latest.items(), key=lambda kv: kv[1], reverse=True)
        changed = {p for p, _ in ordered[: int(fallback_latest)]}
    return changed


def _wait_for_parseable_xml(path: Path, timeout_s: float = 3.0) -> bool:
    deadline = time.time() + max(float(timeout_s), 0.0)
    while time.time() < deadline:
        try:
            ET.parse(path)
            return True
        except ET.ParseError:
            time.sleep(0.1)
        except Exception:
            return False
    return False


def _expected_xml_root_tag(path: Path) -> Optional[str]:
    name = path.name
    if name.startswith("detailed_sumo_stats_"):
        return "tripinfos"
    if name.startswith("sumo_stats_"):
        return "statistics"
    return None


def _wait_for_expected_sumo_xml(path: Path, timeout_s: float = 5.0) -> bool:
    expected_root = _expected_xml_root_tag(path)
    if expected_root is None:
        return _wait_for_parseable_xml(path, timeout_s=timeout_s)
    deadline = time.time() + max(float(timeout_s), 0.0)
    while time.time() < deadline:
        try:
            root = ET.parse(path).getroot()
            tag = str(root.tag).split("}")[-1]
            if tag == expected_root:
                return True
        except ET.ParseError:
            pass
        except Exception:
            return False
        time.sleep(0.1)
    return False


def _copy_protocol_xml_aliases(xml_paths: set[str], protocol_episode_id: int) -> List[Path]:
    aliased: List[Path] = []
    for src_str in sorted(xml_paths):
        src = Path(src_str)
        stem = src.stem
        prefix, sep, suffix = stem.rpartition("_")
        if (not sep) or (not suffix.isdigit()):
            continue
        dst = src.with_name(f"{prefix}_{int(protocol_episode_id)}{src.suffix}")
        _wait_for_parseable_xml(src)
        _copy_artifact_if_needed(src, dst)
        aliased.append(dst)
    return aliased


def _aggregate_global_frac(fracs_od: np.ndarray, counts_od: np.ndarray, n_routes: int) -> np.ndarray:
    veh = fracs_od * counts_od[:, None]
    total = float(veh.sum())
    if total < 1e-8:
        return np.ones(n_routes, dtype=np.float32) / float(n_routes)
    out = veh.sum(axis=0).astype(np.float32)
    out = np.clip(out, 1e-8, None)
    out = out / out.sum()
    return out


def _normalize_prob_vector(vec: np.ndarray, n_routes: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] != int(n_routes):
        arr = np.ones(int(n_routes), dtype=np.float32)
    arr = np.clip(arr, 1e-8, None)
    arr = arr / arr.sum()
    return arr.astype(np.float32)


def _temperature_adjust_probs(vec: np.ndarray, temperature: float, n_routes: int) -> np.ndarray:
    probs = _normalize_prob_vector(vec, n_routes)
    temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0.0 or abs(temp - 1.0) < 1e-6:
        return probs
    logits = np.log(np.clip(probs.astype(np.float64), 1e-12, None)) / temp
    logits = logits - float(np.max(logits))
    out = np.exp(logits)
    out = out / max(float(out.sum()), 1e-12)
    return out.astype(np.float32)


def _safe_parse_agent_id(agent_id: Any) -> Optional[int]:
    try:
        return int(str(agent_id).split("_")[-1])
    except Exception:
        return None


def _build_av_agent_index(
    env,
    od_agent_map: Dict[int, int],
    n_od: int,
) -> Tuple[List[int], np.ndarray, Dict[int, int]]:
    """
    Build stable AV-agent index:
      - av_agent_ids: sorted integer AV IDs
      - av_od_indices: AV->OD index (for per-vehicle obs gathering)
      - av_id_to_idx: ID->row mapping in per-vehicle tensors
    """
    av_ids: List[int] = []
    for m in getattr(env, "machine_agents", []):
        aid = _safe_parse_agent_id(getattr(m, "id", ""))
        if aid is not None:
            av_ids.append(int(aid))
    av_ids = sorted(set(av_ids))

    if not av_ids:
        return [], np.zeros((0,), dtype=np.int64), {}

    n_od_safe = max(int(n_od), 1)
    av_od = np.zeros((len(av_ids),), dtype=np.int64)
    for i, aid in enumerate(av_ids):
        od_idx = int(od_agent_map.get(int(aid), 0))
        av_od[i] = int(np.clip(od_idx, 0, n_od_safe - 1))
    id_to_idx = {int(aid): int(i) for i, aid in enumerate(av_ids)}
    return av_ids, av_od, id_to_idx


def _aggregate_vehicle_fracs_to_od(
    av_vehicle_fracs: np.ndarray,  # (N_av, K)
    av_od_indices: np.ndarray,     # (N_av,)
    n_od: int,
    n_routes: int,
) -> np.ndarray:
    """
    Convert per-vehicle AV route distributions to per-OD route distributions.
    """
    n_od = max(int(n_od), 1)
    k = max(int(n_routes), 1)
    out = np.zeros((n_od, k), dtype=np.float32)
    cnt = np.zeros((n_od,), dtype=np.float32)

    if av_vehicle_fracs.size == 0 or av_od_indices.size == 0:
        return np.ones((n_od, k), dtype=np.float32) / float(k)

    for i in range(min(av_vehicle_fracs.shape[0], av_od_indices.shape[0])):
        od = int(av_od_indices[i])
        if od < 0 or od >= n_od:
            continue
        p = _normalize_prob_vector(av_vehicle_fracs[i], k)
        out[od] += p
        cnt[od] += 1.0

    uniform = np.ones((k,), dtype=np.float32) / float(k)
    for od in range(n_od):
        if cnt[od] > 0.0:
            out[od] /= float(cnt[od])
        else:
            out[od] = uniform
    return out


def _vehicle_action_edge_distribution(
    av_vehicle_fracs: np.ndarray,   # (N_av, K)
    av_od_indices: np.ndarray,      # (N_av,)
    route_incidence_np: Optional[np.ndarray],  # (N_od, K, E)
    n_edges: int,
    n_routes: int,
) -> np.ndarray:
    """
    Convert per-vehicle route distributions into expected edge-visit distribution.
    """
    out = np.zeros((int(n_edges),), dtype=np.float32)
    if (
        route_incidence_np is None
        or av_vehicle_fracs.size == 0
        or av_od_indices.size == 0
        or int(n_edges) <= 0
    ):
        return out

    n_od = int(route_incidence_np.shape[0])
    k = int(min(route_incidence_np.shape[1], int(n_routes)))
    for i in range(min(av_vehicle_fracs.shape[0], av_od_indices.shape[0])):
        od = int(av_od_indices[i])
        if od < 0 or od >= n_od:
            continue
        probs = _normalize_prob_vector(av_vehicle_fracs[i], k)
        out += np.einsum("k,ke->e", probs, route_incidence_np[od, :k], optimize=True).astype(np.float32)

    s = float(out.sum())
    if s > 1e-8:
        out /= s
    return out


def _vehicle_action_edge_distribution_from_path_batch(
    av_vehicle_fracs: np.ndarray,   # (N_av, K)
    path_batch: List[List[List[int]]],
    n_edges: int,
    n_routes: int,
) -> np.ndarray:
    """
    Convert per-vehicle candidate-path distributions into expected edge visitation.
    """
    out = np.zeros((int(n_edges),), dtype=np.float32)
    if av_vehicle_fracs.size == 0 or not path_batch or int(n_edges) <= 0:
        return out

    for i in range(min(int(av_vehicle_fracs.shape[0]), len(path_batch))):
        paths = path_batch[i]
        if not paths:
            continue
        k_eff = int(min(len(paths), int(n_routes), int(av_vehicle_fracs.shape[1])))
        if k_eff <= 0:
            continue
        probs = _normalize_prob_vector(av_vehicle_fracs[i][:k_eff], k_eff)
        for k in range(k_eff):
            path = paths[k]
            if not path:
                continue
            idxs = np.asarray(path, dtype=np.int64)
            idxs = idxs[(idxs >= 0) & (idxs < int(n_edges))]
            if idxs.size == 0:
                continue
            out[idxs] += float(probs[k])

    s = float(out.sum())
    if s > 1e-8:
        out /= s
    return out.astype(np.float32)


def _build_live_vehicle_path_batch(
    *,
    av_agent_ids: List[int],
    av_agent_name_map: Dict[int, str],
    av_od_indices: np.ndarray,
    route_registry,
    edge_to_idx: Dict[str, int],
    adj_list: List[List[int]],
    edge_tt: np.ndarray,
    n_routes: int,
    path_penalty_factor: float,
) -> List[List[List[int]]]:
    """
    Build current-edge -> destination candidate paths for each AV.
    Falls back to the OD origin edge if the vehicle is not yet active in TraCI.
    """
    path_batch: List[List[List[int]]] = []
    n_dest = max(1, len(route_registry.destinations))
    default_origin = route_registry.origins[0] if route_registry.origins else None
    active_vehicle_ids: set[str] = set()
    if TRACI_AVAILABLE:
        try:
            active_vehicle_ids = {str(v) for v in traci.vehicle.getIDList()}
        except Exception:
            active_vehicle_ids = set()

    for i, aid in enumerate(av_agent_ids):
        od = int(av_od_indices[i]) if i < len(av_od_indices) else 0
        oi = int(od // n_dest)
        di = int(od % n_dest)
        orig_id = route_registry.origins[oi] if oi < len(route_registry.origins) else default_origin
        dest_id = route_registry.destinations[di] if di < len(route_registry.destinations) else orig_id
        fallback_cur = int(edge_to_idx.get(orig_id, 0)) if orig_id is not None else 0
        dst_edge = int(edge_to_idx.get(dest_id, fallback_cur)) if dest_id is not None else fallback_cur
        cur_edge = fallback_cur

        if TRACI_AVAILABLE:
            raw_name = av_agent_name_map.get(int(aid), "")
            for cand in (raw_name, str(aid)):
                if (not cand) or (cand not in active_vehicle_ids):
                    continue
                try:
                    edge_id = str(traci.vehicle.getRoadID(cand))
                except Exception:
                    continue
                if edge_id and edge_id in edge_to_idx:
                    cur_edge = int(edge_to_idx[edge_id])
                    break

        if cur_edge == dst_edge:
            path_batch.append([[cur_edge] for _ in range(int(n_routes))])
            continue

        if build_diverse_paths_current_to_dest is None:
            path_batch.append([[cur_edge] for _ in range(int(n_routes))])
            continue

        paths = build_diverse_paths_current_to_dest(
            adj_list=adj_list,
            edge_tt=edge_tt,
            src_idx=cur_edge,
            dst_idx=dst_edge,
            K=int(n_routes),
            penalty_factor=float(path_penalty_factor),
        )
        if not paths:
            paths = [[cur_edge] for _ in range(int(n_routes))]
        while len(paths) < int(n_routes):
            paths.append(list(paths[0]))
        path_batch.append(paths[: int(n_routes)])

    return path_batch


def _build_live_vehicle_path_batch_cached(
    *,
    av_agent_ids: List[int],
    av_agent_name_map: Dict[str, str],
    av_od_indices: np.ndarray,
    route_registry,
    edge_to_idx: Dict[str, int],
    path_cache,   # VehiclePathCache
    n_routes: int,
    edge_tt: Optional[np.ndarray] = None,
) -> List[List[List[int]]]:
    """
    Cache-based version: lookup pre-computed paths instead of running Dijkstra.
    Still uses TraCI to get each vehicle's current edge for accurate routing.
    """
    path_batch: List[List[List[int]]] = []
    n_dest = max(1, len(route_registry.destinations))
    default_origin = route_registry.origins[0] if route_registry.origins else None

    active_vehicle_ids: set = set()
    if TRACI_AVAILABLE:
        try:
            active_vehicle_ids = {str(v) for v in traci.vehicle.getIDList()}
        except Exception:
            pass

    for i, aid in enumerate(av_agent_ids):
        od = int(av_od_indices[i]) if i < len(av_od_indices) else 0
        oi = int(od // n_dest)
        di = int(od % n_dest)
        orig_id = route_registry.origins[oi] if oi < len(route_registry.origins) else default_origin
        dest_id = route_registry.destinations[di] if di < len(route_registry.destinations) else orig_id
        fallback_cur = int(edge_to_idx.get(orig_id, 0)) if orig_id else 0
        dst_edge = int(edge_to_idx.get(dest_id, fallback_cur)) if dest_id else fallback_cur
        cur_edge = fallback_cur

        # TraCI current-edge lookup (same as original)
        if TRACI_AVAILABLE:
            raw_name = av_agent_name_map.get(int(aid), "")
            for cand in (raw_name, str(aid)):
                if (not cand) or (cand not in active_vehicle_ids):
                    continue
                try:
                    edge_id = str(traci.vehicle.getRoadID(cand))
                except Exception:
                    continue
                if edge_id and edge_id in edge_to_idx:
                    cur_edge = int(edge_to_idx[edge_id])
                    break

        # Cache lookup instead of Dijkstra. Lazy cache needs edge travel times.
        if edge_tt is not None:
            paths = path_cache.lookup(cur_edge, dst_edge, edge_tt)
        else:
            paths = path_cache.lookup(cur_edge, dst_edge)
        path_batch.append(paths)

    return path_batch


def _dynamic_reroute_avs(
    *,
    av_agent_ids: List[int],
    av_agent_name_map: Dict[int, str],
    av_vehicle_fracs_np: np.ndarray,   # (N_av, K)
    av_id_to_idx: Dict[int, int],
    path_batch: List[List[List[int]]],  # [N_av][K][edge_indices]
    edge_ids: List[str],               # idx -> edge name
    n_routes: int,
    deterministic: bool = False,
) -> Tuple[int, Dict[int, int]]:
    """Re-route all active AV vehicles via traci.vehicle.setRoute().

    Uses already-computed per-vehicle route probabilities and candidate paths
    from the per-step policy refresh.

    Returns:
        (n_rerouted, reroute_choices) where reroute_choices maps
        agent_int_id -> chosen route index, so that env.step() can reuse
        the same choice instead of sampling independently.
    """
    reroute_choices: Dict[int, int] = {}
    if not TRACI_AVAILABLE or av_vehicle_fracs_np is None or not path_batch:
        return 0, reroute_choices
    try:
        active_ids = {str(v) for v in traci.vehicle.getIDList()}
    except Exception:
        return 0, reroute_choices

    n_rerouted = 0
    n_av = min(len(av_agent_ids), av_vehicle_fracs_np.shape[0], len(path_batch))
    for i in range(n_av):
        aid = av_agent_ids[i]
        veh_name = av_agent_name_map.get(int(aid), "")
        if not veh_name or veh_name not in active_ids:
            continue
        try:
            cur_edge = traci.vehicle.getRoadID(veh_name)
        except Exception:
            continue
        if not cur_edge or cur_edge.startswith(":"):
            continue

        probs = _normalize_prob_vector(av_vehicle_fracs_np[i], n_routes)
        k = int(np.argmax(probs)) if deterministic else int(np.random.choice(n_routes, p=probs))

        chosen = path_batch[i][k] if k < len(path_batch[i]) else path_batch[i][0]
        if not chosen:
            continue
        edge_names = [edge_ids[idx] for idx in chosen if 0 <= idx < len(edge_ids)]
        if not edge_names or edge_names[0] != cur_edge:
            continue
        try:
            traci.vehicle.setRoute(veh_name, edge_names)
            n_rerouted += 1
            reroute_choices[int(aid)] = k
        except Exception:
            pass
    return n_rerouted, reroute_choices


def _mpc_select_action(
    *,
    wm,
    z_current: "torch.Tensor",
    policy,
    obs: "torch.Tensor",             # (N_av, obs_dim) from policy obs
    av_vehicle_fracs_np: np.ndarray,  # (N_av, K) baseline policy output
    path_batch: List[List[List[int]]],
    hdv_edge_np: Optional[np.ndarray],  # (E,) fixed HDV edge dist or None
    horizon: int,
    n_candidates: int,
    discount: float,
    n_routes: int,
    device: "torch.device",
    depart_edge_np: Optional[np.ndarray] = None,
    context_np: Optional[np.ndarray] = None,
    wm_volume_floor: float = 0.0,
) -> np.ndarray:
    """MPC lookahead: sample N candidate actions, imagine H steps, pick best.

    Returns:
        best_fracs: (N_av, K) route fractions from the best candidate.
    """
    import torch
    from dreamer4.urban_routing.rollout_buffer import RolloutBuffer

    n_av = av_vehicle_fracs_np.shape[0]
    n_edges = wm.n_edges
    K = n_routes
    capacities = np.asarray(wm._registry.capacities, dtype=np.float32)

    # Sample N candidate action sets from policy
    candidates = [av_vehicle_fracs_np.copy()]  # include baseline
    with torch.no_grad():
        for _ in range(max(0, n_candidates - 1)):
            fracs, _, _, _, _ = policy.get_action_and_value(obs)
            candidates.append(fracs.cpu().numpy().astype(np.float32))

    best_score = -float("inf")
    best_fracs = candidates[0]

    mpc_buf = RolloutBuffer(
        max_length=2, n_spatial=wm.n_edges,
        d_spatial=wm.d_spatial, device=str(device),
    )

    for cand_fracs in candidates:
        # Convert per-vehicle fracs to edge-level action for WM
        edge_action = np.zeros((n_edges,), dtype=np.float32)
        for i in range(min(n_av, len(path_batch))):
            probs = _normalize_prob_vector(cand_fracs[i], K)
            for k in range(min(K, len(path_batch[i]))):
                path = path_batch[i][k]
                if path:
                    idxs = np.asarray(path, dtype=np.int64)
                    idxs = idxs[(idxs >= 0) & (idxs < n_edges)]
                    if idxs.size > 0:
                        edge_action[idxs] += float(probs[k])
        s = float(edge_action.sum())
        if s > 1e-8:
            edge_action /= s

        # Imagine H steps and accumulate reward
        mpc_buf.buffer.clear()
        mpc_buf.add(z_current)
        z_h = z_current
        total_reward = 0.0
        action_t = torch.from_numpy(edge_action).to(device=device, dtype=torch.float32)
        depart_t = None if depart_edge_np is None else torch.from_numpy(depart_edge_np).to(device=device, dtype=torch.float32)
        context_t = None if context_np is None else torch.from_numpy(context_np).to(device=device, dtype=torch.float32)
        for h in range(horizon):
            z_h = wm.predict_next_state(
                mpc_buf,
                action=action_t,
                depart=depart_t,
                context=context_t,
                skip_stackelberg=True,
            )
            mpc_buf.add(z_h)

            # Decode and compute BPR reward
            decode_fn = getattr(wm.flow_projector, "decode", None)
            if decode_fn is None:
                decode_fn = wm.flow_projector.inverse
            edge_dyn_h = decode_fn(z_h).detach().cpu().numpy().astype(np.float32)
            v_norm_h = np.maximum(edge_dyn_h[:, 2], 0.0)
            if float(wm_volume_floor) > 0.0:
                v_norm_h = np.where(v_norm_h >= float(wm_volume_floor), v_norm_h, 0.0)
            v_h = v_norm_h * capacities
            t_e_h = wm.bpr.numpy_forward(v_h)
            step_tt = float(np.dot(v_h, t_e_h))
            n_veh = max(float(v_h.sum()), 1.0)
            total_reward += (discount ** h) * (-step_tt / n_veh)

        if total_reward > best_score:
            best_score = total_reward
            best_fracs = cand_fracs

    return best_fracs


def _extract_av_rewards_from_csvs(
    csv_paths: set,
    av_agent_ids: List[int],
    default_reward: float,
) -> np.ndarray:
    """
    Build per-AV rewards from episode CSVs using negative per-vehicle travel_time.
    Falls back to global default reward when an AV ID is missing.
    """
    n_av = int(len(av_agent_ids))
    if n_av <= 0:
        return np.zeros((0,), dtype=np.float32)

    rewards = np.full((n_av,), float(default_reward), dtype=np.float32)
    if not csv_paths:
        return rewards

    id_to_idx = {int(a): i for i, a in enumerate(av_agent_ids)}
    travel_time_acc: Dict[int, List[float]] = {}

    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or ("travel_time" not in df.columns) or ("id" not in df.columns):
            continue

        work = df.copy()
        work["travel_time"] = pd.to_numeric(work["travel_time"], errors="coerce")
        work = work.dropna(subset=["travel_time"])
        if work.empty:
            continue

        kind_col = _detect_kind_column(work)
        if kind_col is not None:
            kinds = work[kind_col].astype(str).str.lower()
            work = work.loc[kinds.isin(_AV_KIND_VALUES)]
            if work.empty:
                continue

        for _, row in work.iterrows():
            aid = _safe_parse_agent_id(row.get("id"))
            if aid is None or aid not in id_to_idx:
                continue
            tt = float(row["travel_time"])
            travel_time_acc.setdefault(int(aid), []).append(tt)

    for aid, vals in travel_time_acc.items():
        if not vals:
            continue
        idx = id_to_idx[int(aid)]
        rewards[idx] = -float(np.mean(vals))
    return rewards


def _infer_hdv_global_from_models(env, n_routes: int) -> np.ndarray:
    """
    Infer HDV global route fractions from current human policies.
    Priority per human:
      1) default_action (if fixed externally)
      2) model.act(None)
      3) last_action
    """
    counts = np.full(int(n_routes), 1e-6, dtype=np.float32)
    humans = list(getattr(env, "human_agents", []))
    for human in humans:
        action = None
        da = getattr(human, "default_action", None)
        if da is not None:
            try:
                action = int(da)
            except Exception:
                action = None
        if action is None:
            model = getattr(human, "model", None)
            if model is not None and hasattr(model, "act"):
                try:
                    action = int(model.act(None))
                except Exception:
                    action = None
        if action is None:
            try:
                action = int(getattr(human, "last_action"))
            except Exception:
                action = None
        if action is not None and 0 <= int(action) < int(n_routes):
            counts[int(action)] += 1.0
    return _normalize_prob_vector(counts, int(n_routes))


def _infer_hdv_global_from_episode_actions(
    env,
    n_routes: int,
    fallback: np.ndarray,
) -> np.ndarray:
    """
    Infer HDV global route fractions from RouteRL per-episode actions already
    committed in current simulation day (`env.episode_actions`).
    """
    counts = np.zeros(int(n_routes), dtype=np.float32)
    ep_actions = getattr(env, "episode_actions", None)
    if isinstance(ep_actions, dict):
        for _, rec in ep_actions.items():
            if not isinstance(rec, dict):
                continue
            kind = str(rec.get("kind", "")).strip().lower()
            if kind not in {"human", "hdv"}:
                continue
            try:
                a = int(rec.get("action"))
            except Exception:
                continue
            if 0 <= a < int(n_routes):
                counts[a] += 1.0
    if float(counts.sum()) <= 0.0:
        return _normalize_prob_vector(fallback, int(n_routes))
    return _normalize_prob_vector(counts, int(n_routes))


def _infer_hdv_edge_distribution(
    env,
    edge_to_idx: Dict[str, int],
) -> np.ndarray:
    """
    Infer HDV edge occupancy distribution p(edge) from current in-network vehicles.

    Returns:
        (E,) float32 vector summing to 1 when HDVs are present; otherwise zeros.
    """
    n_edges = int(len(edge_to_idx))
    out = np.zeros((n_edges,), dtype=np.float32)
    if n_edges <= 0 or (not TRACI_AVAILABLE):
        return out

    human_ids: set[str] = set()
    for h in getattr(env, "human_agents", []):
        try:
            hid = str(getattr(h, "id"))
        except Exception:
            hid = ""
        if hid:
            human_ids.add(hid)

    try:
        veh_ids = traci.vehicle.getIDList()
    except Exception:
        return out

    for vid in veh_ids:
        is_hdv = str(vid) in human_ids
        if not is_hdv:
            try:
                vtype = str(traci.vehicle.getTypeID(vid)).strip().lower()
            except Exception:
                vtype = ""
            if vtype in {"human", "hdv"}:
                is_hdv = True
            elif vtype in {"machine", "av", "cav"}:
                is_hdv = False
        if not is_hdv:
            continue

        try:
            edge_id = str(traci.vehicle.getRoadID(vid))
        except Exception:
            continue
        if (not edge_id) or edge_id.startswith(":"):
            continue
        ei = edge_to_idx.get(edge_id)
        if ei is None:
            continue
        out[int(ei)] += 1.0

    s = float(out.sum())
    if s > 0.0:
        out /= s
    return out


def _collect_hdv_equilibrium_distribution(
    env,
    edge_to_idx: Dict[str, int],
    n_adapt_eps: int = 10,
    n_sample_eps: int = 3,
) -> np.ndarray:
    """
    Collect HDV equilibrium edge distribution by running adaptation episodes.

    Fixes the current AV policy, lets HDV agents (Gawron) adapt over
    n_adapt_eps episodes, then samples the HDV edge distribution over
    n_sample_eps episodes and returns the average.

    Args:
        env: RouteRL PettingZoo environment with human_agents.
        edge_to_idx: edge name → index mapping.
        n_adapt_eps: number of adaptation episodes (HDV-only learning).
        n_sample_eps: number of sampling episodes after adaptation.

    Returns:
        (E,) float32 normalized HDV edge distribution.
    """
    n_edges = len(edge_to_idx)
    if n_edges <= 0 or not TRACI_AVAILABLE:
        return np.zeros((n_edges,), dtype=np.float32)

    # Adaptation: run episodes letting HDV learn (env.step handles Gawron)
    for _ in range(n_adapt_eps):
        try:
            env.reset()
            for agent_id in env.agent_iter():
                obs_env, reward, termination, truncation, info = env.last()
                if termination or truncation:
                    action = None
                else:
                    action = int(np.random.randint(0, 4))  # random AV action during adaptation
                env.step(action)
        except Exception:
            pass

    # Sampling: collect HDV edge distributions
    samples = []
    for _ in range(n_sample_eps):
        try:
            env.reset()
            ep_dist = np.zeros((n_edges,), dtype=np.float32)
            n_steps = 0
            for agent_id in env.agent_iter():
                obs_env, reward, termination, truncation, info = env.last()
                if termination or truncation:
                    action = None
                else:
                    action = int(np.random.randint(0, 4))
                env.step(action)
                step_dist = _infer_hdv_edge_distribution(env, edge_to_idx)
                if step_dist.sum() > 0:
                    ep_dist += step_dist
                    n_steps += 1
            if n_steps > 0:
                ep_dist /= float(n_steps)
                samples.append(ep_dist)
        except Exception:
            pass

    if not samples:
        return np.zeros((n_edges,), dtype=np.float32)

    avg = np.mean(samples, axis=0).astype(np.float32)
    s = float(avg.sum())
    if s > 0:
        avg /= s
    return avg


def _save_collected_flow_dataset(
    output_dir: str,
    task_name: str,
    shard_size: int,
    *,
    episodes: list[int],
    av_edge_actions: list[np.ndarray],
    hdv_edge_actions: list[np.ndarray],
    joint_edge_actions: list[np.ndarray],
    rewards: list[float],
    flows: list[np.ndarray],
    departs: list[np.ndarray] | None = None,
    contexts: list[np.ndarray] | None = None,
    decision_indices: list[int] | None = None,
    sim_steps: list[int] | None = None,
) -> None:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    av_edge_action_tensor = torch.from_numpy(np.stack(av_edge_actions, axis=0)).float()
    hdv_edge_action_tensor = torch.from_numpy(np.stack(hdv_edge_actions, axis=0)).float()
    joint_edge_action_tensor = torch.from_numpy(np.stack(joint_edge_actions, axis=0)).float()
    demo = {
        "episode": torch.tensor(episodes, dtype=torch.int64),
        "av_edge_action": av_edge_action_tensor,
        "av_planned_edge_flow": av_edge_action_tensor,
        "hdv_edge_action": hdv_edge_action_tensor,
        "joint_edge_action": joint_edge_action_tensor,
        "reward": torch.tensor(rewards, dtype=torch.float32),
        "action_schema_version": torch.tensor(2, dtype=torch.int64),
    }
    if decision_indices and len(decision_indices) == len(episodes):
        demo["decision_idx"] = torch.tensor(decision_indices, dtype=torch.int64)
    if sim_steps and len(sim_steps) == len(episodes):
        demo["sim_step"] = torch.tensor(sim_steps, dtype=torch.int64)
    if departs and len(departs) > 0:
        demo["depart"] = torch.from_numpy(np.stack(departs, axis=0)).float()
    if contexts and len(contexts) > 0:
        demo["context"] = torch.from_numpy(np.stack(contexts, axis=0)).float()
    demo_path = out_dir / f"{task_name}.pt"
    torch.save(demo, demo_path)

    flows_dir = out_dir / "flows" / task_name
    flows_dir.mkdir(parents=True, exist_ok=True)
    num_shards = (len(flows) + int(shard_size) - 1) // int(shard_size)
    for si in range(num_shards):
        s = si * int(shard_size)
        e = min((si + 1) * int(shard_size), len(flows))
        arr = np.stack(flows[s:e], axis=0)
        torch.save({"flows": torch.from_numpy(arr).float()}, flows_dir / f"shard_{si:04d}.pt")

    print(f"\n  Collected dataset saved:")
    print(f"    demo:  {demo_path}")
    print(f"    flows: {flows_dir} ({num_shards} shards)")
    print(f"    steps: {len(episodes)}")


class PolicyGuidedCollectorEval(DataCollector):
    """SUMO evaluator that reuses the DataCollector rollout loop."""

    def __init__(
        self,
        *,
        wm,
        policy,
        deterministic: bool,
        episode_steps: int,
        disable_human_learning_after_mutation: bool = True,
        mpc_horizon: int = 0,
        mpc_candidates: int = 5,
        mpc_discount: float = 0.99,
        rerank_t_all_weight: float = 1.0,
        rerank_t_cav_weight: float = 2.0,
        rerank_cav_tolerance: float = 0.0,
        wm_volume_floor: float = 0.10,
        loaded_hdv_state: Optional[Dict[str, Any]] = None,
        av_od_route_overrides: Optional[Dict[Tuple[int, int], int]] = None,
        av_policy_route_selection: str = "auto",
        av_policy_sample_temperature: float = 1.0,
        av_policy_confidence_gate_source: str = "none",
        av_policy_confidence_gate_margin: float = 0.0,
        av_policy_confidence_gate_trigger_share: float = 0.0,
        av_policy_nonzero_route_share_cap: float = 0.0,
        av_policy_share_cap_fallback_source: str = "wm_current_bpr",
        av_candidate_source: str = "current_shortest_wm",
        av_candidate_path_penalty_factor: float = 3.0,
        av_route_cost_policy: str = "none",
        av_route_cost_length_norm: str = "sum",
        av_route_cost_temperature: float = 0.0,
        av_route_hdv_pressure_penalty: float = 0.0,
        av_route_wm_live_blend_weight: float = 0.5,
        av_route_planned_av_pressure_penalty: float = 0.0,
        wm_mcc_diagnostic_penalties: Optional[Sequence[float]] = None,
        wm_mcc_diagnostic_horizon: int = 1,
        bpr_alpha: float = 0.15,
        bpr_beta: float = 4.0,
        wm_edge_cost_source: str = "decoded_speed",
        wm_candidate_edge_cost_source: str = "auto",
        wm_decoded_speed_min_ratio: float = 0.05,
        wm_decoded_speed_max_ratio: float = 1.50,
        wm_decoded_speed_excess_scale: float = 1.0,
        vehicle_obs_route_stats: str = "mean",
        vehicle_obs_include_route_costs: bool = False,
        vehicle_obs_include_fleet_pressure: bool = False,
        **collector_kwargs,
    ):
        collector_kwargs.setdefault("loaded_hdv_state", loaded_hdv_state)
        collector_kwargs.setdefault("av_candidate_source", av_candidate_source)
        super().__init__(**collector_kwargs)
        self._wm_eval = wm
        self._policy_eval = policy
        self._policy_eval_device = next(policy.parameters()).device
        self._deterministic_eval = bool(deterministic)
        self._episode_steps_eval = max(int(episode_steps), 1)
        requested_h = max(int(mpc_horizon), 0)
        self._mpc_horizon = requested_h
        self._mpc_horizon_requested = requested_h
        self._mpc_candidates = max(int(mpc_candidates), 1)
        self._mpc_discount = float(mpc_discount)
        self._rerank_t_all_weight = float(rerank_t_all_weight)
        self._rerank_t_cav_weight = float(rerank_t_cav_weight)
        self._rerank_cav_tolerance = float(rerank_cav_tolerance)
        self._wm_volume_floor = max(float(wm_volume_floor), 0.0)
        self._rerank_enabled = self._mpc_horizon > 0 and self._mpc_candidates > 1
        self._av_od_route_overrides = {
            (int(k[0]), int(k[1])): int(v)
            for k, v in dict(av_od_route_overrides or {}).items()
        }
        self._av_policy_route_selection = str(av_policy_route_selection or "auto").strip().lower()
        if self._av_policy_route_selection not in {"auto", "argmax", "sample", "quota"}:
            raise ValueError(f"Unsupported av_policy_route_selection={av_policy_route_selection!r}")
        self._av_policy_sample_temperature = max(float(av_policy_sample_temperature), 1e-6)
        self._av_policy_confidence_gate_source = str(
            av_policy_confidence_gate_source or "none"
        ).strip().lower()
        if self._av_policy_confidence_gate_source not in {"none", "wm_current_bpr", "live_current_tt", "freeflow_greedy"}:
            raise ValueError(
                f"Unsupported av_policy_confidence_gate_source={av_policy_confidence_gate_source!r}"
            )
        self._av_policy_confidence_gate_margin = max(float(av_policy_confidence_gate_margin), 0.0)
        self._av_policy_confidence_gate_trigger_share = float(
            np.clip(float(av_policy_confidence_gate_trigger_share), 0.0, 1.0)
        )
        self._av_policy_nonzero_route_share_cap = float(
            np.clip(float(av_policy_nonzero_route_share_cap), 0.0, 1.0)
        )
        self._av_policy_share_cap_fallback_source = str(
            av_policy_share_cap_fallback_source or "wm_current_bpr"
        ).strip().lower()
        if self._av_policy_share_cap_fallback_source not in {"wm_current_bpr", "live_current_tt", "freeflow_greedy"}:
            raise ValueError(
                f"Unsupported av_policy_share_cap_fallback_source={av_policy_share_cap_fallback_source!r}"
            )
        self._av_candidate_source = str(av_candidate_source or "current_shortest_wm").strip().lower()
        if self._av_candidate_source not in {"od_suffix", "current_shortest_live", "current_shortest_wm"}:
            raise ValueError(f"Unsupported av_candidate_source={av_candidate_source!r}")
        self._av_candidate_path_penalty_factor = max(float(av_candidate_path_penalty_factor), 1.0)
        edge_registry = getattr(self, "_edge_registry", None)
        self._av_candidate_adj_list = _build_edge_adj_list(
            getattr(edge_registry, "adjacency", None)
        )
        self._av_route_cost_policy = str(av_route_cost_policy or "none").strip().lower()
        if self._av_route_cost_policy not in {"none", "freeflow_greedy", "wm_current_bpr", "live_current_tt", "wm_live_blend"}:
            raise ValueError(f"Unsupported av_route_cost_policy={av_route_cost_policy!r}")
        self._av_route_cost_length_norm = str(av_route_cost_length_norm or "sum").strip().lower()
        if self._av_route_cost_length_norm not in {"sum", "mean"}:
            raise ValueError(f"Unsupported av_route_cost_length_norm={av_route_cost_length_norm!r}")
        self._av_route_cost_temperature = max(float(av_route_cost_temperature), 0.0)
        self._av_route_hdv_pressure_penalty = max(float(av_route_hdv_pressure_penalty), 0.0)
        self._av_route_wm_live_blend_weight = float(
            np.clip(float(av_route_wm_live_blend_weight), 0.0, 1.0)
        )
        self._av_route_planned_av_pressure_penalty = max(
            float(av_route_planned_av_pressure_penalty),
            0.0,
        )
        self._wm_mcc_diagnostic_penalties = [
            max(float(v), 0.0)
            for v in list(wm_mcc_diagnostic_penalties or [])
        ]
        self._wm_mcc_diagnostic_horizon = max(int(wm_mcc_diagnostic_horizon), 1)
        self._bpr_alpha_eval = float(bpr_alpha)
        self._bpr_beta_eval = float(bpr_beta)
        self._wm_edge_cost_source = str(wm_edge_cost_source or "decoded_speed").strip().lower()
        if self._wm_edge_cost_source not in {"bpr_volume", "decoded_speed", "route_head"}:
            raise ValueError(f"Unsupported wm_edge_cost_source={wm_edge_cost_source!r}")
        self._wm_candidate_edge_cost_source = str(wm_candidate_edge_cost_source or "auto").strip().lower()
        if self._wm_candidate_edge_cost_source == "auto":
            self._wm_candidate_edge_cost_source = self._wm_edge_cost_source
        if self._wm_candidate_edge_cost_source not in {"bpr_volume", "decoded_speed", "route_head"}:
            raise ValueError(f"Unsupported wm_candidate_edge_cost_source={wm_candidate_edge_cost_source!r}")
        self._wm_decoded_speed_min_ratio = max(float(wm_decoded_speed_min_ratio), 1e-4)
        self._wm_decoded_speed_max_ratio = max(
            float(wm_decoded_speed_max_ratio),
            self._wm_decoded_speed_min_ratio,
        )
        self._wm_decoded_speed_excess_scale = max(float(wm_decoded_speed_excess_scale), 0.0)
        self._vehicle_obs_route_stats = str(vehicle_obs_route_stats or "mean").strip().lower()
        self._vehicle_obs_include_route_costs = bool(vehicle_obs_include_route_costs)
        self._vehicle_obs_include_fleet_pressure = bool(vehicle_obs_include_fleet_pressure)
        self._loaded_hdv_state_eval = loaded_hdv_state
        self._use_policy_actions = True
        self._disable_human_learning_after_mutation_eval = bool(
            disable_human_learning_after_mutation
        )
        self._current_decision_idx_eval = 0
        self._current_episode_id_eval = -1
        self._last_ep_av_travel_times: List[float] = []
        self._last_ep_hdv_travel_times: List[float] = []
        self._last_ep_vehicle_tt: Dict[str, float] = {}
        self._protocol_xml_alias_map: Dict[int, int] = {}
        self._dynamic_decision_rows: List[Dict[str, Any]] = []
        self._dynamic_agent_decision_rows: List[Dict[str, Any]] = []
        self._wm_mcc_diagnostic_rows: List[Dict[str, Any]] = []

    def _should_update_hdv_after_episode(self, desc: str) -> bool:
        if self._hdv_route_probs is None or self._route_registry is None:
            return False
        if (
            bool(self._disable_human_learning_after_mutation_eval)
            and str(desc).strip().lower() == "test"
        ):
            return False
        return True

    def _write_urb_episode_csv(self, episode_id: int, episode_reward: float) -> Optional[Path]:
        agents_df = getattr(self.connector, "_agents_df", None)
        if agents_df is None or len(agents_df) == 0:
            return None

        records_folder = Path(getattr(self.connector, "records_folder", ""))
        if not records_folder:
            return None
        episodes_dir = records_folder / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        initial_route_choices = getattr(self.connector, "_episode_route_choices", {}) or {}
        fixed_machine_ids = set(getattr(self, "_fixed_machine_ids", set()) or set())
        fixed_human_ids = set(getattr(self, "_fixed_human_ids", set()) or set())
        for _, row in agents_df.iterrows():
            veh_id = str(row.get("id"))
            policy_route_idx = self._av_route_idx.get(veh_id, -1)
            hdv_route_idx = self._hdv_route_idx.get(veh_id, -1)
            route_idx = self._av_route_idx.get(veh_id, self._hdv_route_idx.get(veh_id, -1))
            initial_route_idx = int(initial_route_choices.get(veh_id, -1))
            if veh_id in fixed_machine_ids or veh_id in self._av_vehicles:
                vtype = "AV"
            elif veh_id in fixed_human_ids or veh_id in self._hdv_vehicles:
                vtype = "Human"
            else:
                vtype = str(row.get("kind", "Human"))
            rows.append(
                {
                    "id": int(float(row.get("id", -1))),
                    "travel_time": float(self._last_ep_vehicle_tt.get(veh_id, np.nan)),
                    "action": int(route_idx),
                    "initial_route_action": int(initial_route_idx),
                    "policy_action": int(policy_route_idx),
                    "hdv_action": int(hdv_route_idx),
                    "seen_in_episode": int(veh_id in self._av_vehicles or veh_id in self._hdv_vehicles),
                    "origin": int(float(row.get("origin", -1))),
                    "destination": int(float(row.get("destination", -1))),
                    "start_time": int(float(row.get("start_time", 0))),
                    "reward": float(episode_reward),
                    "vType": vtype,
                }
            )

        out_path = episodes_dir / f"ep{int(episode_id)}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        return out_path

    def _register_protocol_xml_aliases(self, xml_paths: set[str], episode_id: int) -> None:
        for src_str in sorted(xml_paths):
            stem = Path(src_str).stem
            _prefix, sep, suffix = stem.rpartition("_")
            if sep and suffix.isdigit():
                self._protocol_xml_alias_map[int(suffix)] = int(episode_id)

    def finalize_protocol_xml_aliases(self) -> None:
        records_folder = str(getattr(self.connector, "records_folder", ""))
        if not records_folder:
            return
        sumo_output_folder = Path(records_folder) / "SUMO_output"
        for src in sorted(sumo_output_folder.glob("*.xml")):
            stem = src.stem
            prefix, sep, suffix = stem.rpartition("_")
            if (not sep) or (not suffix.isdigit()):
                continue
            runtime_episode_id = int(suffix)
            protocol_episode_id = self._protocol_xml_alias_map.get(runtime_episode_id)
            if protocol_episode_id is None:
                continue
            _wait_for_expected_sumo_xml(src)
            dst = src.with_name(f"{prefix}_{protocol_episode_id}{src.suffix}")
            _copy_artifact_if_needed(src, dst)

    def _wm_current_edge_tt(self, z_obs: torch.Tensor, edge_cost_source: Optional[str] = None) -> Optional[np.ndarray]:
        if edge_tt_from_latent_state is None:
            return None
        try:
            _edge_dyn, _v_e, _free_flow, edge_tt = edge_tt_from_latent_state(
                wm=self._wm_eval,
                z=z_obs,
                bpr_alpha=float(self._bpr_alpha_eval),
                bpr_beta=float(self._bpr_beta_eval),
                edge_cost_source=str(edge_cost_source or self._wm_edge_cost_source),
                decoded_speed_min_ratio=self._wm_decoded_speed_min_ratio,
                decoded_speed_max_ratio=self._wm_decoded_speed_max_ratio,
                decoded_speed_excess_scale=self._wm_decoded_speed_excess_scale,
            )
        except Exception:
            return None
        out = np.asarray(edge_tt, dtype=np.float32).reshape(-1)
        return out if out.size > 0 else None

    def _live_current_edge_tt(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        edge_ids = list(getattr(self._edge_registry, "edge_ids", []))
        if not edge_ids:
            return None
        lengths = getattr(self._edge_registry, "lengths", None)
        if lengths is None:
            lengths_arr = np.ones((len(edge_ids),), dtype=np.float32) * 100.0
        else:
            lengths_arr = np.asarray(lengths, dtype=np.float32).reshape(-1)
        speed_limits = np.asarray(
            getattr(self._edge_registry, "speed_limits", np.ones((len(edge_ids),), dtype=np.float32) * 13.89),
            dtype=np.float32,
        ).reshape(-1)
        n = min(len(edge_ids), int(lengths_arr.shape[0]), int(speed_limits.shape[0]))
        if n <= 0:
            return None
        edge_state = state.get("edges", {}) if isinstance(state, dict) else {}
        out = np.zeros((n,), dtype=np.float32)
        for i, eid in enumerate(edge_ids[:n]):
            data = edge_state.get(eid, {}) if isinstance(edge_state, dict) else {}
            speed = float(data.get("mean_speed", 0.0) or 0.0) if isinstance(data, dict) else 0.0
            if not np.isfinite(speed) or speed <= 0.1:
                speed = float(speed_limits[i])
            speed = max(speed, 0.1)
            out[i] = float(lengths_arr[i]) / speed
        return out

    def _freeflow_edge_tt(self) -> Optional[np.ndarray]:
        edge_ids = list(getattr(self._edge_registry, "edge_ids", []))
        if not edge_ids:
            return None
        lengths = getattr(self._edge_registry, "lengths", None)
        speed_limits = getattr(self._edge_registry, "speed_limits", None)
        if lengths is None or speed_limits is None:
            return None
        lengths_arr = np.asarray(lengths, dtype=np.float32).reshape(-1)
        speed_limits_arr = np.asarray(speed_limits, dtype=np.float32).reshape(-1)
        n = min(len(edge_ids), int(lengths_arr.shape[0]), int(speed_limits_arr.shape[0]))
        if n <= 0:
            return None
        return lengths_arr[:n] / np.maximum(speed_limits_arr[:n], 0.1)

    def _current_to_dest_route_candidates(
        self,
        veh_id: str,
        current_edge: str,
        edge_cost: Optional[np.ndarray],
    ) -> List[List[str]]:
        if build_diverse_paths_current_to_dest is None:
            return []
        if not self._av_candidate_adj_list:
            return []
        if edge_cost is None:
            return []
        if (not current_edge) or current_edge.startswith(":"):
            return []

        dest_edge = str(self._veh_dest.get(veh_id, ""))
        if (not dest_edge) or dest_edge.startswith(":"):
            return []
        edge_to_idx = getattr(self._edge_registry, "_edge_to_idx", {})
        edge_ids = list(getattr(self._edge_registry, "edge_ids", []))
        if current_edge not in edge_to_idx or dest_edge not in edge_to_idx:
            return []
        n_edges = len(edge_ids)
        costs = np.asarray(edge_cost, dtype=np.float32).reshape(-1)
        if n_edges <= 0 or costs.shape[0] < n_edges:
            return []

        src_idx = int(edge_to_idx[current_edge])
        dst_idx = int(edge_to_idx[dest_edge])
        paths = build_diverse_paths_current_to_dest(
            adj_list=self._av_candidate_adj_list,
            edge_tt=costs[:n_edges],
            src_idx=src_idx,
            dst_idx=dst_idx,
            K=int(self.n_routes),
            penalty_factor=float(self._av_candidate_path_penalty_factor),
        )
        named_paths: List[List[str]] = []
        for path in paths[: int(self.n_routes)]:
            route = [
                edge_ids[int(edge_idx)]
                for edge_idx in path
                if 0 <= int(edge_idx) < n_edges
            ]
            if route and route[0] == current_edge:
                named_paths.append(route)
        if not named_paths:
            return []
        while len(named_paths) < int(self.n_routes):
            named_paths.append(list(named_paths[0]))
        return named_paths[: int(self.n_routes)]

    def _record_wm_mcc_diagnostics(
        self,
        *,
        z_obs: torch.Tensor,
        path_batch: List[List[List[int]]],
        baseline_route_indices: np.ndarray,
        live_route_cost_matrix: Optional[np.ndarray],
        depart_np: np.ndarray,
        context_np: np.ndarray,
        hdv_edge_np: np.ndarray,
        current_edges_np: np.ndarray,
        action_horizon_seconds: float,
        rng: np.random.RandomState,
    ) -> None:
        penalties = list(getattr(self, "_wm_mcc_diagnostic_penalties", []) or [])
        if not penalties or not path_batch:
            return

        def _append_metrics_row(
            *,
            candidate_kind: str,
            route_indices: np.ndarray,
            applied: int,
            penalty: float = float("nan"),
        ) -> None:
            metrics = rollout_candidate_metrics(
                wm=self._wm_eval,
                z_current=z_obs,
                path_batch=path_batch,
                route_indices=route_indices,
                horizon=int(self._wm_mcc_diagnostic_horizon),
                discount=float(self._mpc_discount),
                depart_edge_np=depart_np,
                context_np=context_np,
                hdv_edge_np=hdv_edge_np,
                total_av_count=max(float(len(self._av_vehicles)), 1.0),
                total_hdv_count=max(float(len(self._hdv_vehicles)), 1.0),
                volume_floor=float(self._wm_volume_floor),
                edge_cost_source=str(self._wm_edge_cost_source),
                current_edges=current_edges_np,
                action_horizon_seconds=float(action_horizon_seconds),
                decoded_speed_min_ratio=float(self._wm_decoded_speed_min_ratio),
                decoded_speed_max_ratio=float(self._wm_decoded_speed_max_ratio),
                decoded_speed_excess_scale=float(self._wm_decoded_speed_excess_scale),
            )
            counts = np.bincount(
                np.asarray(route_indices, dtype=np.int64).clip(0, int(self.n_routes) - 1),
                minlength=int(self.n_routes),
            )
            row: Dict[str, Any] = {
                "episode": int(self._current_episode_id_eval),
                "decision_idx": int(self._current_decision_idx_eval),
                "sim_step": int(getattr(self.connector, "current_step", -1)),
                "active_av": int(len(route_indices)),
                "candidate_kind": str(candidate_kind),
                "penalty": float(penalty),
                "applied": int(applied),
                "pred_t_all": float(metrics.get("pred_t_all", float("nan"))),
                "pred_t_all_pressure": float(metrics.get("pred_t_all_pressure", float("nan"))),
                "pred_t_all_head": float(metrics.get("pred_t_all_head", float("nan"))),
                "pred_t_all_decoded": float(metrics.get("pred_t_all_decoded", float("nan"))),
                "pred_t_cav_proxy": float(metrics.get("pred_t_cav_proxy", float("nan"))),
                "horizon": int(self._wm_mcc_diagnostic_horizon),
            }
            for k in range(int(self.n_routes)):
                row[f"selected_count_route{k}"] = int(counts[k])
            self._wm_mcc_diagnostic_rows.append(row)

        for penalty in penalties:
            try:
                if live_route_cost_matrix is None:
                    continue
                route_indices, applied = _select_min_cost_route_indices_with_planned_pressure(
                    live_route_cost_matrix,
                    baseline_route_indices,
                    path_batch,
                    n_edges=len(getattr(self._edge_registry, "edge_ids", [])),
                    deterministic=True,
                    rng=rng,
                    temperature=0.0,
                    pressure_penalty=float(penalty),
                )
                _append_metrics_row(
                    candidate_kind="penalty",
                    route_indices=route_indices,
                    applied=int(applied),
                    penalty=float(penalty),
                )
            except Exception as exc:
                self._wm_mcc_diagnostic_rows.append(
                    {
                        "episode": int(self._current_episode_id_eval),
                        "decision_idx": int(self._current_decision_idx_eval),
                        "sim_step": int(getattr(self.connector, "current_step", -1)),
                        "active_av": int(len(baseline_route_indices)),
                        "candidate_kind": "penalty",
                        "penalty": float(penalty),
                        "applied": 0,
                        "pred_t_all": float("nan"),
                        "pred_t_all_pressure": float("nan"),
                        "pred_t_all_head": float("nan"),
                        "pred_t_all_decoded": float("nan"),
                        "pred_t_cav_proxy": float("nan"),
                        "horizon": int(self._wm_mcc_diagnostic_horizon),
                        "error": str(exc),
                    }
                )
    def _select_av_actions(
        self, state: Dict[str, Any], rng: np.random.RandomState
    ) -> Dict[str, List[str]]:
        if not self._use_policy_actions:
            return {}

        in_network = state.get("vehicles", {})
        active_avs = sorted(self._av_vehicles & set(in_network.keys()))
        if not active_avs:
            return {}

        try:
            z_obs = self._wm_eval.encode_observation(state)
            edge_feats_dyn = (
                self._wm_eval.latent_edge_features(z_obs)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        except Exception:
            return {}

        edge_to_idx = getattr(self._edge_registry, "_edge_to_idx", {})
        wm_edge_cost = self._wm_current_edge_tt(z_obs)
        wm_candidate_edge_cost = (
            wm_edge_cost
            if self._wm_candidate_edge_cost_source == self._wm_edge_cost_source
            else self._wm_current_edge_tt(z_obs, edge_cost_source=self._wm_candidate_edge_cost_source)
        )
        live_edge_cost = self._live_current_edge_tt(state)
        freeflow_edge_cost = self._freeflow_edge_tt()
        candidate_edge_cost: Optional[np.ndarray] = None
        if self._av_candidate_source == "current_shortest_wm":
            candidate_edge_cost = wm_candidate_edge_cost
        elif self._av_candidate_source == "current_shortest_live":
            candidate_edge_cost = live_edge_cost
        path_batch: List[List[List[int]]] = []
        route_batch: List[List[List[str]]] = []
        active_ids: List[str] = []
        active_current_edges: List[str] = []
        candidate_dynamic_count = 0
        candidate_fallback_count = 0

        for veh_id in active_avs:
            veh_info = in_network.get(veh_id)
            if not isinstance(veh_info, dict):
                continue
            current_edge = str(veh_info.get("edge", ""))
            if (not current_edge) or current_edge.startswith(":"):
                continue
            candidates: List[List[str]] = []
            if self._av_candidate_source != "od_suffix":
                candidates = self._current_to_dest_route_candidates(
                    veh_id,
                    current_edge,
                    candidate_edge_cost,
                )
                if candidates:
                    candidate_dynamic_count += 1
                else:
                    candidate_fallback_count += 1
            if not candidates:
                candidates = self._candidate_route_suffixes(veh_id, current_edge)
            if not candidates:
                continue

            idx_paths: List[List[int]] = []
            named_paths: List[List[str]] = []
            for route in candidates[: self.n_routes]:
                idxs = [int(edge_to_idx[e]) for e in route if e in edge_to_idx]
                if idxs:
                    idx_paths.append(idxs)
                    named_paths.append(list(route))
            if not idx_paths:
                continue
            while len(idx_paths) < self.n_routes:
                idx_paths.append(list(idx_paths[0]))
                named_paths.append(list(named_paths[0]))
            path_batch.append(idx_paths[: self.n_routes])
            route_batch.append(named_paths[: self.n_routes])
            active_ids.append(veh_id)
            active_current_edges.append(current_edge)

        if not path_batch:
            return {}

        candidate_unique_mean, candidate_unique_min, candidate_unique_max = _route_batch_uniqueness_summary(
            route_batch
        )

        obs_vehicle_base_np = build_vehicle_obs_vectorized(
            edge_feats_dyn=edge_feats_dyn,
            path_batch=path_batch,
            include_global=True,
            route_stats=self._vehicle_obs_route_stats,
            edge_cost=wm_edge_cost,
            include_route_costs=self._vehicle_obs_include_route_costs,
            include_fleet_pressure=False,
        )
        if self._vehicle_obs_include_fleet_pressure:
            if build_vehicle_fleet_pressure_obs_features is None:
                return {}
            eval_selection_mode = str(self._av_policy_route_selection)
            if eval_selection_mode == "auto":
                eval_selection_mode = "argmax" if self._deterministic_eval else "sample"
            fleet_counts_np = np.zeros((int(self.n_routes),), dtype=np.float32)
            fleet_pressure_np = np.zeros((len(getattr(self._edge_registry, "edge_ids", [])),), dtype=np.float32)
            av_frac_rows: List[np.ndarray] = []
            for i in range(len(active_ids)):
                fleet_i = build_vehicle_fleet_pressure_obs_features(
                    path_batch=[path_batch[i]],
                    n_edges=len(getattr(self._edge_registry, "edge_ids", [])),
                    n_routes=int(self.n_routes),
                    selected_route_counts=fleet_counts_np,
                    selected_edge_pressure=fleet_pressure_np,
                )
                obs_i_np = _append_vehicle_context_features(
                    np.concatenate([obs_vehicle_base_np[i:i + 1], fleet_i], axis=1),
                    step_idx=int(getattr(self.connector, "current_step", 0)),
                    episode_steps=self._episode_steps_eval,
                    n_active_av=len(active_ids),
                    n_total_av=max(len(self._av_vehicles), 1),
                )
                obs_i = torch.from_numpy(obs_i_np).to(
                    device=self._policy_eval_device,
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    frac_i = self._policy_eval.get_av_frac(
                        obs_i,
                        deterministic=self._deterministic_eval,
                    ).detach().cpu().numpy().astype(np.float32)[0]
                probs_i = _temperature_adjust_probs(
                    frac_i,
                    self._av_policy_sample_temperature,
                    self.n_routes,
                )
                if eval_selection_mode == "sample":
                    selected_i = int(rng.choice(self.n_routes, p=probs_i))
                else:
                    selected_i = int(np.argmax(probs_i))
                selected_i = int(np.clip(selected_i, 0, len(route_batch[i]) - 1))
                av_frac_rows.append(frac_i)
                fleet_counts_np[selected_i] += 1.0
                if selected_i < len(path_batch[i]):
                    for edge_idx in path_batch[i][selected_i]:
                        e = int(edge_idx)
                        if 0 <= e < int(fleet_pressure_np.shape[0]):
                            fleet_pressure_np[e] += 1.0
            av_fracs_np = np.stack(av_frac_rows, axis=0).astype(np.float32)
        else:
            obs_vehicle_np = _append_vehicle_context_features(
                obs_vehicle_base_np,
                step_idx=int(getattr(self.connector, "current_step", 0)),
                episode_steps=self._episode_steps_eval,
                n_active_av=len(active_ids),
                n_total_av=max(len(self._av_vehicles), 1),
            )
            obs_online = torch.from_numpy(obs_vehicle_np).to(
                device=self._policy_eval_device,
                dtype=torch.float32,
            )
            with torch.no_grad():
                av_fracs = self._policy_eval.get_av_frac(
                    obs_online,
                    deterministic=self._deterministic_eval,
                )
            av_fracs_np = av_fracs.detach().cpu().numpy().astype(np.float32)

        baseline_route_indices = np.zeros((len(active_ids),), dtype=np.int64)
        selection_mode = str(self._av_policy_route_selection)
        if selection_mode == "auto":
            selection_mode = "argmax" if self._deterministic_eval else "sample"
        if selection_mode == "quota":
            av_fracs_for_selection = np.stack([
                _temperature_adjust_probs(row, self._av_policy_sample_temperature, self.n_routes)
                for row in av_fracs_np
            ], axis=0).astype(np.float32)
            argmax_baseline = np.zeros((len(active_ids),), dtype=np.int64)
            for i in range(len(active_ids)):
                probs = av_fracs_for_selection[i]
                argmax_baseline[i] = int(np.clip(int(np.argmax(probs)), 0, len(route_batch[i]) - 1))
            baseline_route_indices = _select_quota_route_indices(
                av_fracs_for_selection,
                argmax_baseline,
                n_routes=int(self.n_routes),
            )
            for i in range(len(active_ids)):
                baseline_route_indices[i] = int(np.clip(baseline_route_indices[i], 0, len(route_batch[i]) - 1))
        else:
            for i in range(len(active_ids)):
                probs = _temperature_adjust_probs(
                    av_fracs_np[i],
                    self._av_policy_sample_temperature,
                    self.n_routes,
                )
                if selection_mode == "sample":
                    ridx = int(rng.choice(self.n_routes, p=probs))
                else:
                    ridx = int(np.argmax(probs))
                baseline_route_indices[i] = int(np.clip(ridx, 0, len(route_batch[i]) - 1))

        selected_route_indices = baseline_route_indices.copy()
        counterfactual_mode = "none"
        counterfactual_applied = 0
        confidence_gate_applied = 0
        confidence_gate_triggered = False
        confidence_gate_max_nonzero_share = 0.0
        share_cap_applied = 0
        route_cost_matrix: Optional[np.ndarray] = None
        wm_route_cost_matrix: Optional[np.ndarray] = None
        live_route_cost_matrix: Optional[np.ndarray] = None
        freeflow_route_cost_matrix: Optional[np.ndarray] = None
        hdv_pressure_route_matrix: Optional[np.ndarray] = None
        route_cost_applied = 0
        rerank_trace: Optional[Dict[str, Any]] = None
        rerank_error = ""
        if wm_edge_cost is not None:
            wm_route_cost_matrix = _edge_route_cost_matrix(
                path_batch,
                wm_edge_cost,
                n_routes=int(self.n_routes),
                length_norm=self._av_route_cost_length_norm,
            )
        if live_edge_cost is not None:
            live_route_cost_matrix = _edge_route_cost_matrix(
                path_batch,
                live_edge_cost,
                n_routes=int(self.n_routes),
                length_norm=self._av_route_cost_length_norm,
            )
        if freeflow_edge_cost is not None:
            freeflow_route_cost_matrix = _edge_route_cost_matrix(
                path_batch,
                freeflow_edge_cost,
                n_routes=int(self.n_routes),
                length_norm=self._av_route_cost_length_norm,
            )
        if self._av_route_hdv_pressure_penalty > 0.0:
            hdv_edge_pressure = self._compute_hdv_edge_distribution(state)
            hdv_pressure_route_matrix = _edge_route_cost_matrix(
                path_batch,
                hdv_edge_pressure,
                n_routes=int(self.n_routes),
                length_norm="sum",
            )
        diagnostic_depart_np = self._decision_depart_edge(int(self._current_decision_idx_eval))
        diagnostic_context_np = np.array([
            float(self._current_decision_idx_eval) / max(int(self._episode_steps_eval), 1),
            float(len(active_ids)) / max(int(self._expected_n_av), 1),
            0.0,
        ], dtype=np.float32)
        diagnostic_hdv_edge_np = self._compute_hdv_edge_distribution(state)
        diagnostic_current_edge_idx = np.asarray(
            [int(edge_to_idx[e]) for e in active_current_edges if e in edge_to_idx],
            dtype=np.int64,
        )
        diagnostic_action_horizon_seconds = (
            float(getattr(self, "_decision_horizon_seconds", self._episode_steps_eval))
            / max(float(getattr(self, "_decision_steps", 1) or 1), 1.0)
        )
        self._record_wm_mcc_diagnostics(
            z_obs=z_obs,
            path_batch=path_batch,
            baseline_route_indices=baseline_route_indices,
            live_route_cost_matrix=live_route_cost_matrix,
            depart_np=diagnostic_depart_np,
            context_np=diagnostic_context_np,
            hdv_edge_np=diagnostic_hdv_edge_np,
            current_edges_np=diagnostic_current_edge_idx,
            action_horizon_seconds=float(diagnostic_action_horizon_seconds),
            rng=rng,
        )
        if len(active_ids) > 0:
            route_slots = np.clip(
                np.asarray(baseline_route_indices, dtype=np.int64),
                0,
                int(self.n_routes) - 1,
            )
            route_counts_for_gate = np.bincount(route_slots, minlength=int(self.n_routes))
            if int(self.n_routes) > 1:
                confidence_gate_max_nonzero_share = (
                    float(np.max(route_counts_for_gate[1:])) / float(len(active_ids))
                )
            trigger_share = float(self._av_policy_confidence_gate_trigger_share)
            confidence_gate_triggered = (
                trigger_share <= 0.0
                or confidence_gate_max_nonzero_share >= trigger_share
            )
        if self._av_od_route_overrides:
            counterfactual_mode = "od_override"
            selected_route_indices = baseline_route_indices.copy()
            for i, veh_id in enumerate(active_ids):
                od_key = self._veh_od.get(veh_id)
                if od_key is None:
                    continue
                override = self._av_od_route_overrides.get((int(od_key[0]), int(od_key[1])))
                if override is None:
                    continue
                selected_route_indices[i] = int(override)
                counterfactual_applied += 1
        elif (
            self._av_policy_confidence_gate_source != "none"
            and self._av_policy_confidence_gate_margin > 0.0
            and confidence_gate_triggered
            and selection_mode == "argmax"
        ):
            if self._av_policy_confidence_gate_source == "wm_current_bpr":
                gate_cost_matrix = wm_route_cost_matrix
            elif self._av_policy_confidence_gate_source == "live_current_tt":
                gate_cost_matrix = live_route_cost_matrix
            elif self._av_policy_confidence_gate_source == "freeflow_greedy":
                gate_cost_matrix = freeflow_route_cost_matrix
            else:
                gate_cost_matrix = None
            if gate_cost_matrix is not None:
                gated = selected_route_indices.copy()
                margin_threshold = float(self._av_policy_confidence_gate_margin)
                for i in range(min(len(active_ids), int(gate_cost_matrix.shape[0]))):
                    cost_row = np.asarray(gate_cost_matrix[i, : int(self.n_routes)], dtype=np.float32)
                    finite = np.isfinite(cost_row)
                    if not bool(finite.any()):
                        continue
                    masked_cost = np.where(finite, cost_row, np.inf)
                    fallback = int(np.argmin(masked_cost))
                    top = int(np.clip(int(gated[i]), 0, int(self.n_routes) - 1))
                    if fallback == top:
                        continue
                    probs = _temperature_adjust_probs(
                        av_fracs_np[i],
                        self._av_policy_sample_temperature,
                        self.n_routes,
                    )
                    if fallback >= int(probs.shape[0]) or top >= int(probs.shape[0]):
                        continue
                    policy_margin = float(probs[top] - probs[fallback])
                    if policy_margin < margin_threshold:
                        gated[i] = int(np.clip(fallback, 0, len(route_batch[i]) - 1))
                        confidence_gate_applied += 1
                selected_route_indices = gated
                if confidence_gate_applied > 0:
                    counterfactual_mode = f"policy_confidence_gate_{self._av_policy_confidence_gate_source}"
                    counterfactual_applied = int(confidence_gate_applied)
        elif self._av_route_cost_policy != "none":
            if self._av_route_cost_policy == "freeflow_greedy":
                route_cost_matrix = freeflow_route_cost_matrix
            elif self._av_route_cost_policy == "wm_current_bpr":
                route_cost_matrix = wm_route_cost_matrix
            elif self._av_route_cost_policy == "live_current_tt":
                route_cost_matrix = live_route_cost_matrix
            elif self._av_route_cost_policy == "wm_live_blend":
                if wm_route_cost_matrix is not None and live_route_cost_matrix is not None:
                    w = float(self._av_route_wm_live_blend_weight)
                    route_cost_matrix = (
                        (1.0 - w) * np.asarray(live_route_cost_matrix, dtype=np.float32)
                        + w * np.asarray(wm_route_cost_matrix, dtype=np.float32)
                    )
                else:
                    route_cost_matrix = None
            else:
                route_cost_matrix = None
            if (
                route_cost_matrix is not None
                and hdv_pressure_route_matrix is not None
                and self._av_route_hdv_pressure_penalty > 0.0
            ):
                route_cost_matrix = (
                    np.asarray(route_cost_matrix, dtype=np.float32)
                    + float(self._av_route_hdv_pressure_penalty)
                    * np.asarray(hdv_pressure_route_matrix, dtype=np.float32)
                )

            if route_cost_matrix is not None:
                selected_route_indices, route_cost_applied = _select_min_cost_route_indices_with_planned_pressure(
                    route_cost_matrix,
                    baseline_route_indices,
                    path_batch,
                    n_edges=len(getattr(self._edge_registry, "edge_ids", [])),
                    deterministic=bool(self._deterministic_eval),
                    rng=rng,
                    temperature=float(self._av_route_cost_temperature),
                    pressure_penalty=float(self._av_route_planned_av_pressure_penalty),
                )
                counterfactual_mode = f"route_cost_{self._av_route_cost_policy}"
                counterfactual_applied = int(route_cost_applied)

        if (
            self._av_policy_nonzero_route_share_cap > 0.0
            and counterfactual_mode in {"none", f"policy_confidence_gate_{self._av_policy_confidence_gate_source}"}
            and selection_mode == "argmax"
            and len(active_ids) > 0
        ):
            if self._av_policy_share_cap_fallback_source == "wm_current_bpr":
                cap_cost_matrix = wm_route_cost_matrix
            elif self._av_policy_share_cap_fallback_source == "live_current_tt":
                cap_cost_matrix = live_route_cost_matrix
            elif self._av_policy_share_cap_fallback_source == "freeflow_greedy":
                cap_cost_matrix = freeflow_route_cost_matrix
            else:
                cap_cost_matrix = None
            if cap_cost_matrix is not None:
                capped = selected_route_indices.copy()
                max_per_nonzero_route = int(
                    np.ceil(float(self._av_policy_nonzero_route_share_cap) * float(len(active_ids)))
                )
                max_per_nonzero_route = max(max_per_nonzero_route, 0)
                route_probs_for_cap = np.stack([
                    _temperature_adjust_probs(row, self._av_policy_sample_temperature, self.n_routes)
                    for row in av_fracs_np
                ], axis=0).astype(np.float32)
                for route_idx in range(1, int(self.n_routes)):
                    selected_for_route = np.where(capped == int(route_idx))[0]
                    if selected_for_route.size <= max_per_nonzero_route:
                        continue
                    candidates_to_rank: List[Tuple[float, int, int]] = []
                    mandatory_keep = 0
                    for i in selected_for_route.tolist():
                        if i >= int(cap_cost_matrix.shape[0]):
                            mandatory_keep += 1
                            continue
                        cost_row = np.asarray(cap_cost_matrix[i, : int(self.n_routes)], dtype=np.float32)
                        finite = np.isfinite(cost_row)
                        if not bool(finite.any()):
                            mandatory_keep += 1
                            continue
                        fallback = int(np.argmin(np.where(finite, cost_row, np.inf)))
                        if fallback == int(route_idx):
                            mandatory_keep += 1
                            continue
                        probs = route_probs_for_cap[i]
                        confidence = float(probs[int(route_idx)] - probs[int(fallback)])
                        candidates_to_rank.append((confidence, int(i), int(fallback)))
                    keep_budget = max(int(max_per_nonzero_route) - int(mandatory_keep), 0)
                    candidates_to_rank.sort(key=lambda x: x[0], reverse=True)
                    keep_ids = {int(i) for _conf, i, _fallback in candidates_to_rank[:keep_budget]}
                    for _conf, i, fallback in candidates_to_rank[keep_budget:]:
                        capped[int(i)] = int(np.clip(fallback, 0, len(route_batch[int(i)]) - 1))
                        share_cap_applied += 1
                if share_cap_applied > 0:
                    selected_route_indices = capped
                    if counterfactual_mode == "none":
                        counterfactual_mode = "policy_nonzero_share_cap"
                    else:
                        counterfactual_mode = f"{counterfactual_mode}+share_cap"
                    counterfactual_applied += int(share_cap_applied)

        if counterfactual_mode == "none" and self._rerank_enabled and len(active_ids) > 0:
            depart_np = self._decision_depart_edge(int(self._current_decision_idx_eval))
            context_np = np.array([
                float(self._current_decision_idx_eval) / max(int(self._episode_steps_eval), 1),
                float(len(active_ids)) / max(int(self._expected_n_av), 1),
                0.0,
            ], dtype=np.float32)
            hdv_edge_np = self._compute_hdv_edge_distribution(state)
            active_current_edge_idx = np.asarray(
                [int(edge_to_idx[e]) for e in active_current_edges if e in edge_to_idx],
                dtype=np.int64,
            )
            action_horizon_seconds = (
                float(getattr(self, "_decision_horizon_seconds", self._episode_steps_eval))
                / max(float(getattr(self, "_decision_steps", 1) or 1), 1.0)
            )
            try:
                rerank = rerank_topk_candidates(
                    wm=self._wm_eval,
                    z_current=z_obs,
                    route_probs=av_fracs_np,
                    baseline_route_indices=baseline_route_indices,
                    path_batch=path_batch,
                    horizon=self._mpc_horizon,
                    top_k=min(self._mpc_candidates, self.n_routes),
                    discount=self._mpc_discount,
                    depart_edge_np=depart_np,
                    context_np=context_np,
                    hdv_edge_np=hdv_edge_np,
                    total_av_count=float(max(int(self._expected_n_av), 1)),
                    total_hdv_count=float(max(int(self._expected_n_hdv), 1)),
                    t_all_weight=self._rerank_t_all_weight,
                    t_cav_weight=self._rerank_t_cav_weight,
                    cav_tolerance=self._rerank_cav_tolerance,
                    volume_floor=self._wm_volume_floor,
                    edge_cost_source=self._wm_edge_cost_source,
                    current_edges=active_current_edge_idx,
                    action_horizon_seconds=action_horizon_seconds,
                    decoded_speed_min_ratio=self._wm_decoded_speed_min_ratio,
                    decoded_speed_max_ratio=self._wm_decoded_speed_max_ratio,
                    decoded_speed_excess_scale=self._wm_decoded_speed_excess_scale,
                )
                rerank_trace = rerank
                selected_route_indices = np.asarray(rerank["selected"].route_indices, dtype=np.int64)
                cand_parts = []
                for d in rerank["decisions"]:
                    cand_parts.append(
                        f"c{d.candidate_id}:pressure={d.pred_t_all_pressure:.3f},"
                        f"head={d.pred_t_all_head:.3f},decoded={d.pred_t_all_decoded:.3f},"
                        f"cav_proxy={d.pred_t_cav_proxy:.3f},score={d.weighted_score:.3f},"
                        f"gate={int(d.gate_pass)}"
                    )
                print(
                    f"  [wm_rerank_local] step={int(self._current_decision_idx_eval):02d} "
                    f"selected={int(rerank['selected_idx'])} baseline=0 | " + " ; ".join(cand_parts)
                )
            except Exception as exc:
                rerank_error = str(exc)
                print(f"  [wm_rerank_local] step={int(self._current_decision_idx_eval):02d} fallback_to_baseline due to error: {exc}")
                selected_route_indices = baseline_route_indices

        actions: Dict[str, List[str]] = {}
        selected_counts = np.bincount(
            np.asarray(selected_route_indices, dtype=np.int64).clip(0, self.n_routes - 1),
            minlength=self.n_routes,
        )
        baseline_counts = np.bincount(
            np.asarray(baseline_route_indices, dtype=np.int64).clip(0, self.n_routes - 1),
            minlength=self.n_routes,
        )
        mean_probs = np.mean(av_fracs_np[:, : self.n_routes], axis=0) if av_fracs_np.size else np.zeros((self.n_routes,), dtype=np.float32)
        if route_cost_matrix is not None and route_cost_matrix.size:
            route_cost_mean = np.full((int(self.n_routes),), np.nan, dtype=np.float32)
            for k in range(int(self.n_routes)):
                vals = route_cost_matrix[:, k]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    route_cost_mean[k] = float(vals.mean())
        else:
            route_cost_mean = np.full((int(self.n_routes),), np.nan, dtype=np.float32)
        wm_route_cost_mean, wm_route_best_counts, wm_route_best_total = _route_cost_matrix_summary(
            wm_route_cost_matrix,
            int(self.n_routes),
        )
        live_route_cost_mean, live_route_best_counts, live_route_best_total = _route_cost_matrix_summary(
            live_route_cost_matrix,
            int(self.n_routes),
        )
        row: Dict[str, Any] = {
            "episode": int(self._current_episode_id_eval),
            "decision_idx": int(self._current_decision_idx_eval),
            "sim_step": int(getattr(self.connector, "current_step", -1)),
            "active_av": int(len(active_ids)),
            "deterministic": bool(self._deterministic_eval),
            "policy_route_selection": self._av_policy_route_selection,
            "rerank_enabled": bool(self._rerank_enabled),
            "counterfactual_mode": counterfactual_mode,
            "counterfactual_applied": int(counterfactual_applied),
            "av_candidate_source": self._av_candidate_source,
            "candidate_dynamic_count": int(candidate_dynamic_count),
            "candidate_fallback_count": int(candidate_fallback_count),
            "candidate_unique_mean": float(candidate_unique_mean),
            "candidate_unique_min": int(candidate_unique_min),
            "candidate_unique_max": int(candidate_unique_max),
            "route_cost_policy": self._av_route_cost_policy,
            "wm_edge_cost_source": self._wm_edge_cost_source,
            "wm_candidate_edge_cost_source": self._wm_candidate_edge_cost_source,
            "wm_decoded_speed_excess_scale": float(self._wm_decoded_speed_excess_scale),
            "route_cost_length_norm": self._av_route_cost_length_norm,
            "route_cost_temperature": float(self._av_route_cost_temperature),
            "route_hdv_pressure_penalty": float(self._av_route_hdv_pressure_penalty),
            "route_wm_live_blend_weight": float(self._av_route_wm_live_blend_weight),
            "route_planned_av_pressure_penalty": float(self._av_route_planned_av_pressure_penalty),
            "route_cost_applied": int(route_cost_applied),
            "policy_confidence_gate_source": self._av_policy_confidence_gate_source,
            "policy_confidence_gate_margin": float(self._av_policy_confidence_gate_margin),
            "policy_confidence_gate_trigger_share": float(self._av_policy_confidence_gate_trigger_share),
            "policy_confidence_gate_triggered": bool(confidence_gate_triggered),
            "policy_confidence_gate_max_nonzero_share": float(confidence_gate_max_nonzero_share),
            "policy_confidence_gate_applied": int(confidence_gate_applied),
            "policy_nonzero_route_share_cap": float(self._av_policy_nonzero_route_share_cap),
            "policy_share_cap_fallback_source": self._av_policy_share_cap_fallback_source,
            "policy_share_cap_applied": int(share_cap_applied),
            "wm_route_best_total": int(wm_route_best_total),
            "live_route_best_total": int(live_route_best_total),
            "rerank_selected_candidate": int(rerank_trace["selected_idx"]) if rerank_trace is not None else -1,
            "rerank_candidate_count": int(len(rerank_trace.get("decisions", []))) if rerank_trace is not None else 0,
            "rerank_baseline_t_cav_proxy": float(rerank_trace.get("baseline_t_cav_proxy", float("nan"))) if rerank_trace is not None else float("nan"),
            "rerank_error": rerank_error,
        }
        if rerank_trace is not None:
            for decision in rerank_trace.get("decisions", []):
                cid = int(decision.candidate_id)
                row[f"rerank_c{cid}_t_all"] = float(decision.pred_t_all)
                row[f"rerank_c{cid}_t_all_pressure"] = float(decision.pred_t_all_pressure)
                row[f"rerank_c{cid}_t_all_head"] = float(decision.pred_t_all_head)
                row[f"rerank_c{cid}_t_all_decoded"] = float(decision.pred_t_all_decoded)
                row[f"rerank_c{cid}_t_cav_proxy"] = float(decision.pred_t_cav_proxy)
                row[f"rerank_c{cid}_weighted_score"] = float(decision.weighted_score)
                row[f"rerank_c{cid}_gate_pass"] = int(bool(decision.gate_pass))
                row[f"rerank_c{cid}_gate_reason"] = str(decision.gate_reason)
        for k in range(self.n_routes):
            row[f"mean_prob_route{k}"] = float(mean_probs[k]) if k < len(mean_probs) else 0.0
            row[f"baseline_count_route{k}"] = int(baseline_counts[k])
            row[f"selected_count_route{k}"] = int(selected_counts[k])
            row[f"mean_route_cost_route{k}"] = float(route_cost_mean[k]) if k < len(route_cost_mean) and np.isfinite(route_cost_mean[k]) else float("nan")
            row[f"wm_mean_route_cost_route{k}"] = float(wm_route_cost_mean[k]) if k < len(wm_route_cost_mean) and np.isfinite(wm_route_cost_mean[k]) else float("nan")
            row[f"wm_best_count_route{k}"] = int(wm_route_best_counts[k]) if k < len(wm_route_best_counts) else 0
            row[f"live_mean_route_cost_route{k}"] = float(live_route_cost_mean[k]) if k < len(live_route_cost_mean) and np.isfinite(live_route_cost_mean[k]) else float("nan")
            row[f"live_best_count_route{k}"] = int(live_route_best_counts[k]) if k < len(live_route_best_counts) else 0
        self._dynamic_decision_rows.append(row)

        for i, veh_id in enumerate(active_ids):
            agent_row: Dict[str, Any] = {
                "episode": int(self._current_episode_id_eval),
                "decision_idx": int(self._current_decision_idx_eval),
                "sim_step": int(getattr(self.connector, "current_step", -1)),
                "veh_id": str(veh_id),
                "current_edge": active_current_edges[i] if i < len(active_current_edges) else "",
                "dest_edge": str(self._veh_dest.get(veh_id, "")),
                "deterministic": bool(self._deterministic_eval),
                "policy_route_selection": self._av_policy_route_selection,
                "av_candidate_source": self._av_candidate_source,
                "wm_candidate_edge_cost_source": self._wm_candidate_edge_cost_source,
                "route_cost_length_norm": self._av_route_cost_length_norm,
                "policy_confidence_gate_source": self._av_policy_confidence_gate_source,
                "policy_confidence_gate_margin": float(self._av_policy_confidence_gate_margin),
                "policy_confidence_gate_trigger_share": float(self._av_policy_confidence_gate_trigger_share),
                "policy_confidence_gate_triggered": bool(confidence_gate_triggered),
                "policy_confidence_gate_max_nonzero_share": float(confidence_gate_max_nonzero_share),
                "policy_nonzero_route_share_cap": float(self._av_policy_nonzero_route_share_cap),
                "policy_share_cap_fallback_source": self._av_policy_share_cap_fallback_source,
                "baseline_route": int(baseline_route_indices[i]) if i < len(baseline_route_indices) else -1,
                "selected_route": int(selected_route_indices[i]) if i < len(selected_route_indices) else -1,
                "wm_best_route": -1,
                "live_best_route": -1,
                "wm_live_best_match": 0,
            }
            wm_best_idx = -1
            live_best_idx = -1
            if wm_route_cost_matrix is not None and i < int(wm_route_cost_matrix.shape[0]):
                wm_row = np.asarray(wm_route_cost_matrix[i, : int(self.n_routes)], dtype=np.float32)
                finite = np.isfinite(wm_row)
                if bool(finite.any()):
                    finite_idx = np.flatnonzero(finite)
                    wm_best_idx = int(finite_idx[int(np.argmin(wm_row[finite]))])
            if live_route_cost_matrix is not None and i < int(live_route_cost_matrix.shape[0]):
                live_row = np.asarray(live_route_cost_matrix[i, : int(self.n_routes)], dtype=np.float32)
                finite = np.isfinite(live_row)
                if bool(finite.any()):
                    finite_idx = np.flatnonzero(finite)
                    live_best_idx = int(finite_idx[int(np.argmin(live_row[finite]))])
            agent_row["wm_best_route"] = int(wm_best_idx)
            agent_row["live_best_route"] = int(live_best_idx)
            agent_row["wm_live_best_match"] = int(wm_best_idx >= 0 and wm_best_idx == live_best_idx)
            for k in range(self.n_routes):
                agent_row[f"prob_route{k}"] = float(av_fracs_np[i, k]) if i < av_fracs_np.shape[0] and k < av_fracs_np.shape[1] else float("nan")
                agent_row[f"path_edges_route{k}"] = (
                    json.dumps(route_batch[i][k], separators=(",", ":"))
                    if i < len(route_batch) and k < len(route_batch[i])
                    else "[]"
                )
                agent_row[f"path_edge_indices_route{k}"] = (
                    json.dumps(path_batch[i][k], separators=(",", ":"))
                    if i < len(path_batch) and k < len(path_batch[i])
                    else "[]"
                )
                agent_row[f"wm_cost_route{k}"] = (
                    float(wm_route_cost_matrix[i, k])
                    if wm_route_cost_matrix is not None
                    and i < int(wm_route_cost_matrix.shape[0])
                    and k < int(wm_route_cost_matrix.shape[1])
                    and np.isfinite(wm_route_cost_matrix[i, k])
                    else float("nan")
                )
                agent_row[f"live_cost_route{k}"] = (
                    float(live_route_cost_matrix[i, k])
                    if live_route_cost_matrix is not None
                    and i < int(live_route_cost_matrix.shape[0])
                    and k < int(live_route_cost_matrix.shape[1])
                    and np.isfinite(live_route_cost_matrix[i, k])
                    else float("nan")
                )
            self._dynamic_agent_decision_rows.append(agent_row)

        for i, veh_id in enumerate(active_ids):
            ridx = int(np.clip(int(selected_route_indices[i]), 0, len(route_batch[i]) - 1))
            self._av_route_idx[veh_id] = ridx
            actions[veh_id] = route_batch[i][ridx]
        return actions

    def _collect_single_episode(
        self,
        episode_id: int,
        max_steps: int,
    ) -> Tuple[
        List[int],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[float],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[int],
        List[int],
    ]:
        rng = np.random.RandomState((self.seed or 0) + episode_id)

        self._av_vehicles.clear()
        self._hdv_vehicles.clear()
        self._random_avs.clear()
        self._greedy_avs.clear()
        self._av_route_idx.clear()
        self._hdv_route_idx.clear()
        self._veh_dest.clear()
        self._route_cache.clear()
        self._veh_candidates.clear()
        self._veh_od.clear()
        self._last_ep_travel_times.clear()
        self._last_ep_av_travel_times.clear()
        self._last_ep_hdv_travel_times.clear()
        self._last_ep_vehicle_tt.clear()
        self._last_ep_route_tts.clear()

        veh_depart: Dict[str, float] = {}
        veh_travel_times: Dict[str, float] = {}

        ep_ids: List[int] = []
        ep_av_edge_acts: List[np.ndarray] = []
        ep_hdv_edge_acts: List[np.ndarray] = []
        ep_joint_edge_acts: List[np.ndarray] = []
        ep_rews: List[float] = []
        ep_frames: List[np.ndarray] = []
        ep_departs: List[np.ndarray] = []
        ep_contexts: List[np.ndarray] = []
        ep_decision_indices: List[int] = []
        ep_sim_steps: List[int] = []

        try:
            state = self.connector.reset(
                route_probs=self._hdv_route_probs,
                human_learning_state=self._loaded_hdv_state_eval,
            )
        except TypeError:
            state = self.connector.reset(route_probs=self._hdv_route_probs)
            if self._loaded_hdv_state_eval is not None and hasattr(self.connector, "env"):
                _restore_hdv_learning_state(
                    self.connector.env,
                    self._loaded_hdv_state_eval,
                    label=f"collector-ep{episode_id}",
                )
        except Exception as exc:
            print(f"  [episode {episode_id}] reset failed: {exc}")
            return (
                ep_ids,
                ep_av_edge_acts,
                ep_hdv_edge_acts,
                ep_joint_edge_acts,
                ep_rews,
                ep_frames,
                ep_departs,
                ep_contexts,
                ep_decision_indices,
                ep_sim_steps,
            )

        self._register_vehicles(list(state.get("vehicles", {}).keys()), rng)
        self._record_depart_times(state, veh_depart)

        decision_ticks = sorted(self._decision_tick_map(max_steps).keys())
        if not decision_ticks:
            decision_ticks = list(range(max_steps))

        current_tick = 0
        for decision_idx, t in enumerate(decision_ticks):
            steps_to_boundary = max(int(t) - int(current_tick), 0)
            if steps_to_boundary > 0:
                arrival_events = self.connector.advance_steps(steps_to_boundary)
                current_tick += steps_to_boundary
                self._record_arrivals(
                    veh_depart,
                    veh_travel_times,
                    track_role_breakdown=True,
                    arrival_events=arrival_events,
                )
                state = self.connector.get_traffic_state()

            frame_t = self._collect_flow(state)

            self._register_vehicles(list(state.get("vehicles", {}).keys()), rng)
            self._record_depart_times(state, veh_depart)

            self._current_episode_id_eval = int(episode_id)
            self._current_decision_idx_eval = int(decision_idx)
            av_actions = self._select_av_actions(state, rng)
            next_tick = int(decision_ticks[decision_idx + 1]) if (decision_idx + 1) < len(decision_ticks) else int(max_steps)
            advance_steps = max(next_tick - int(t), 1)
            av_edge_frac = self._compute_planned_av_edge_flow(
                state,
                av_actions,
                horizon_seconds=float(advance_steps),
            )
            hdv_edge_frac = self._compute_hdv_edge_distribution(state)
            joint_edge_frac = self._compute_joint_edge_distribution(state)
            decision_depart = self._decision_depart_edge(int(decision_idx))
            decision_context = self._decision_context(state, int(t))
            if av_actions:
                self._apply_reroutings(av_actions)

            try:
                arrival_events = self.connector.advance_steps(advance_steps)
            except Exception as exc:
                print(f"  [episode {episode_id}, decision {decision_idx}] advance failed: {exc}")
                break
            current_tick = int(t) + int(advance_steps)
            self._record_arrivals(
                veh_depart,
                veh_travel_times,
                track_role_breakdown=True,
                arrival_events=arrival_events,
            )
            state = self.connector.get_traffic_state()

            reward = self.connector._calculate_reward(state)

            ep_ids.append(episode_id)
            ep_frames.append(frame_t)
            ep_av_edge_acts.append(av_edge_frac)
            ep_hdv_edge_acts.append(hdv_edge_frac)
            ep_joint_edge_acts.append(joint_edge_frac)
            ep_rews.append(float(reward))
            ep_departs.append(decision_depart)
            ep_contexts.append(decision_context)
            ep_decision_indices.append(int(decision_idx))
            ep_sim_steps.append(int(t))

            if self.connector.is_done() or (len(state.get("vehicles", {})) == 0 and current_tick > 10):
                break

        if self._use_routerl_hdv_blackbox and hasattr(self.connector, "apply_routerl_hdv_learning"):
            try:
                self.connector.apply_routerl_hdv_learning(veh_travel_times)
            except Exception as exc:
                print(f"  [episode {episode_id}] RouteRL black-box HDV learning failed: {exc}")

        if self._decision_steps <= 0:
            ep_frames.append(self._collect_flow(state))

        if self._decision_steps > 0 and ep_rews:
            ep_reward = self._completion_aware_episode_reward(
                current_tick=int(current_tick),
                max_steps=int(max_steps),
                fallback_frame=ep_frames[-1] if ep_frames else None,
            )
            ep_rews = [ep_reward] * len(ep_rews)

        return (
            ep_ids,
            ep_av_edge_acts,
            ep_hdv_edge_acts,
            ep_joint_edge_acts,
            ep_rews,
            ep_frames,
            ep_departs,
            ep_contexts,
            ep_decision_indices,
            ep_sim_steps,
        )

    def run_eval_episodes(
        self,
        *,
        start_episode_id: int,
        num_episodes: int,
        max_steps_per_episode: int,
        use_policy_actions: bool,
        desc: str,
        log_every: int,
        collect_dataset: bool,
    ) -> Dict[str, Any]:
        self._use_policy_actions = bool(use_policy_actions)
        episode_rows: List[Dict[str, Any]] = []

        all_episodes: List[int] = []
        all_av_edge_actions: List[np.ndarray] = []
        all_hdv_edge_actions: List[np.ndarray] = []
        all_joint_edge_actions: List[np.ndarray] = []
        all_rewards: List[float] = []
        all_flows: List[np.ndarray] = []
        all_departs: List[np.ndarray] = []
        all_contexts: List[np.ndarray] = []
        all_decision_indices: List[int] = []
        all_sim_steps: List[int] = []
        dataset_saved_episodes = 0
        dataset_skipped_incomplete_episodes = 0

        iterator = tqdm(range(num_episodes), desc=desc)
        for offset in iterator:
            episode_id = int(start_episode_id + offset)
            records_folder = str(getattr(self.connector, "records_folder", ""))
            sumo_output_folder = os.path.join(records_folder, "SUMO_output")
            xml_snapshot_before = _snapshot_sumo_xmls(sumo_output_folder)
            (
                ep_ids,
                ep_av_edge_acts,
                ep_hdv_edge_acts,
                ep_joint_edge_acts,
                ep_rews,
                ep_frames,
                ep_departs,
                ep_contexts,
                ep_decision_indices,
                ep_sim_steps,
            ) = self._collect_single_episode(episode_id=episode_id, max_steps=max_steps_per_episode)

            if self._should_update_hdv_after_episode(desc):
                if self._last_ep_route_tts:
                    self._update_hdv_gawron_actual()
                elif len(ep_frames) > 0:
                    self._update_hdv_gawron(ep_frames[-1])

            t_all = float(np.mean(self._last_ep_travel_times) / 60.0) if self._last_ep_travel_times else float("nan")
            t_cav = float(np.mean(self._last_ep_av_travel_times) / 60.0) if self._last_ep_av_travel_times else float("nan")
            t_hdv = float(np.mean(self._last_ep_hdv_travel_times) / 60.0) if self._last_ep_hdv_travel_times else float("nan")
            ep_reward = float(ep_rews[-1]) if ep_rews else 0.0
            self._write_urb_episode_csv(episode_id, ep_reward)
            new_xmls = _new_sumo_xmls_since(xml_snapshot_before, sumo_output_folder, fallback_latest=2)
            self._register_protocol_xml_aliases(new_xmls, episode_id)
            row = {
                "episode": episode_id,
                "phase": desc,
                "steps": len(ep_ids),
                "arrived_all": len(self._last_ep_travel_times),
                "arrived_av": len(self._last_ep_av_travel_times),
                "arrived_hdv": len(self._last_ep_hdv_travel_times),
                "total_agents": int(getattr(self, "_num_agents", 0) or 0),
                "unfinished_all": max(int(getattr(self, "_num_agents", 0) or 0) - len(self._last_ep_travel_times), 0),
                "completion_rate": (
                    float(len(self._last_ep_travel_times)) / float(max(int(getattr(self, "_num_agents", 0) or 0), 1))
                ),
                "t_all": t_all,
                "t_CAV": t_cav,
                "t_HDV": t_hdv,
            }
            total_agents = int(getattr(self, "_num_agents", 0) or 0)
            episode_complete = _episode_complete_for_dataset(
                n_arrived=len(self._last_ep_travel_times),
                total_agents=total_agents,
                n_steps=len(ep_ids),
            )
            row["dataset_collected"] = int(bool(collect_dataset and episode_complete))
            episode_rows.append(row)

            if collect_dataset and episode_complete and ep_ids:
                all_episodes.extend(ep_ids)
                all_av_edge_actions.extend(ep_av_edge_acts)
                all_hdv_edge_actions.extend(ep_hdv_edge_acts)
                all_joint_edge_actions.extend(ep_joint_edge_acts)
                all_rewards.extend(ep_rews)
                all_flows.extend(ep_frames)
                all_departs.extend(ep_departs)
                all_contexts.extend(ep_contexts)
                all_decision_indices.extend(ep_decision_indices)
                all_sim_steps.extend(ep_sim_steps)
                dataset_saved_episodes += 1
            elif collect_dataset:
                dataset_skipped_incomplete_episodes += 1

            should_log = ((offset + 1) % max(int(log_every), 1) == 0) or ((offset + 1) == num_episodes)
            if should_log:
                if np.isfinite(t_all):
                    print(
                        f"  [{desc}] ep={offset+1:04d}/{num_episodes:04d} "
                        f"t_CAV={t_cav:.2f} min t_HDV={t_hdv:.2f} min t_all={t_all:.2f} min",
                        flush=True,
                    )
                else:
                    print(
                        f"  [{desc}] ep={offset+1:04d}/{num_episodes:04d} t_all=N/A",
                        flush=True,
                    )

        dataset = None
        if collect_dataset and all_episodes:
            dataset = {
                "episodes": all_episodes,
                "av_edge_actions": all_av_edge_actions,
                "hdv_edge_actions": all_hdv_edge_actions,
                "joint_edge_actions": all_joint_edge_actions,
                "rewards": all_rewards,
                "flows": all_flows,
                "departs": all_departs,
                "contexts": all_contexts,
                "decision_indices": all_decision_indices,
                "sim_steps": all_sim_steps,
            }
        return {
            "rows": episode_rows,
            "dataset": dataset,
            "dataset_saved_episodes": dataset_saved_episodes,
            "dataset_skipped_incomplete_episodes": dataset_skipped_incomplete_episodes,
        }


def main():
    args = parse_args()

    if not ROUTERL_AVAILABLE:
        sys.exit("routerl is not installed. Please activate the URB conda environment.")

    tcp_ok, tcp_reason = _check_local_tcp_accept_support()
    if not tcp_ok:
        sys.exit(
            "Local TCP socket accept is not permitted in the current runtime, "
            f"so SUMO/TraCI evaluation cannot start here. Underlying error: {tcp_reason}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve paths relative to repo root
    network_folder = Path(args.network_folder)
    if not network_folder.is_absolute():
        network_folder = (_REPO_ROOT / network_folder).resolve()
    network_folder = str(network_folder)

    edge_registry_path = Path(args.edge_registry)
    if not edge_registry_path.is_absolute():
        edge_registry_path = (_REPO_ROOT / edge_registry_path).resolve()
    edge_registry_path = str(edge_registry_path)

    results_dir    = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (_REPO_ROOT / results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    episodes_folder = str(results_dir / "episodes")
    plots_folder    = str(results_dir / "plots")
    Path(episodes_folder).mkdir(parents=True, exist_ok=True)
    Path(plots_folder).mkdir(parents=True, exist_ok=True)

    hdv_state_in_path = _resolve_optional_path(args.hdv_state_in)
    hdv_state_out_path = _resolve_optional_path(args.hdv_state_out)
    post_mutation_state_in_path = _resolve_optional_path(args.post_mutation_state_in)
    post_mutation_state_out_path = _resolve_optional_path(args.post_mutation_state_out)
    loaded_hdv_state = _load_hdv_learning_state(hdv_state_in_path)
    av_od_route_overrides = _load_av_od_route_overrides(args.av_od_route_override_json)
    wm_mcc_diagnostic_penalties = _parse_float_list(args.wm_mcc_diagnostic_penalties)

    print("=" * 70)
    print("  Dreamer4 MARL Policy Evaluation in SUMO (Flow Mode)")
    print("=" * 70)
    print(f"  Network:     {args.network}  ({network_folder})")
    print(f"  Policy:      {args.policy_ckpt}")
    print(f"  Results dir: {results_dir}")
    print(f"  Device:      {args.device}")
    print(f"  Human eps:   {args.human_learning_episodes}")
    print(f"  Test eps:    {args.test_eps}")
    print(f"  Deterministic: {args.deterministic}")
    if hdv_state_in_path is not None:
        print(f"  HDV state in:  {hdv_state_in_path}")
    if hdv_state_out_path is not None:
        print(f"  HDV state out: {hdv_state_out_path}")
    if post_mutation_state_in_path is not None:
        print(f"  Post-mutation state in:  {post_mutation_state_in_path}")
    if post_mutation_state_out_path is not None:
        print(f"  Post-mutation state out: {post_mutation_state_out_path}")
    print(
        "  Human learning after mutation: "
        + ("disabled" if args.disable_human_learning_after_mutation else "enabled")
    )
    if av_od_route_overrides:
        print(f"  AV counterfactual: {len(av_od_route_overrides)} OD route overrides")
    print(f"  AV candidate source: {args.av_candidate_source}")
    if str(args.av_route_cost_policy).strip().lower() != "none":
        print(
            "  AV dynamic route-cost policy: "
            f"{args.av_route_cost_policy} "
            f"(agg={args.av_route_cost_length_norm}, temp={float(args.av_route_cost_temperature):.3f}, "
            f"hdv_penalty={float(args.av_route_hdv_pressure_penalty):.3f}, "
            f"wm_live_w={float(args.av_route_wm_live_blend_weight):.3f}, "
            f"planned_av_penalty={float(args.av_route_planned_av_pressure_penalty):.3f})"
        )
    print()

    # ── Load world model (FlowProjector + dynamics) ──────────────────────────
    print("=== Loading World Model ===")
    from urban_world_model import UrbanWorldModel
    wm = UrbanWorldModel(
        flow_projector_ckpt=args.projector_ckpt,
        dynamics_ckpt=args.dynamics_ckpt,
        edge_registry_path=edge_registry_path,
        device=args.device,
        use_amp=True,
        cfm_init_mode=args.wm_cfm_init_mode,
        policy_obs_source=("obs_proj" if args.wm_policy_obs_source == "auto" else args.wm_policy_obs_source),
    )
    wm.eval()

    # ── Detect policy type from checkpoint ────────────────────────────────────
    print("=== Loading Policy ===")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_state = torch.load(args.policy_ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ckpt_state.get("cfg", {})
    if args.wm_policy_obs_source == "auto":
        ckpt_obs_source = str(cfg_dict.get("wm_policy_obs_source", "")).strip().lower()
        if ckpt_obs_source in ("decoder_volume_tt", "decoder_volume_tt_route_head", "decoder_volume", "decoder", "obs_proj"):
            wm.set_policy_obs_source(ckpt_obs_source)
    model_family = str(ckpt_state.get("model_family", cfg_dict.get("training_family", ""))).strip().lower()
    if model_family != "dreamer_ctde":
        raise RuntimeError(f"Unsupported policy checkpoint family {model_family!r}; expected dreamer_ctde.")
    agent_level = str(cfg_dict.get("agent_level", "")).strip().lower()
    if agent_level != "vehicle":
        raise RuntimeError(f"Unsupported policy agent_level {agent_level!r}; expected vehicle.")
    vehicle_mode = True
    vehicle_obs_include_global = bool(cfg_dict.get("vehicle_obs_include_global", True))
    vehicle_path_penalty_factor = float(cfg_dict.get("vehicle_path_penalty_factor", 3.0))
    ckpt_route_stats = str(cfg_dict.get("vehicle_obs_route_stats", "mean")).strip().lower()
    vehicle_obs_route_stats = (
        ckpt_route_stats
        if str(args.vehicle_obs_route_stats).strip().lower() == "auto"
        else str(args.vehicle_obs_route_stats).strip().lower()
    )
    vehicle_obs_include_route_costs = bool(
        args.vehicle_obs_include_route_costs
        or cfg_dict.get("vehicle_obs_include_route_costs", False)
    )
    vehicle_obs_include_fleet_pressure = bool(cfg_dict.get("vehicle_obs_include_fleet_pressure", False))

    if vehicle_mode_obs_dim is not None:
        obs_dim = vehicle_mode_obs_dim(
            n_routes=args.n_routes,
            d_obs=wm.d_obs,
            include_global=vehicle_obs_include_global,
            route_stats=vehicle_obs_route_stats,
            include_route_costs=vehicle_obs_include_route_costs,
            include_fleet_pressure=vehicle_obs_include_fleet_pressure,
        )
    else:
        obs_dim = args.n_routes * wm.d_obs + (wm.d_obs if vehicle_obs_include_global else 0) + 3
        if vehicle_obs_include_route_costs:
            obs_dim += 3 * int(args.n_routes)
        if vehicle_obs_include_fleet_pressure:
            obs_dim += 2 * int(args.n_routes)
    print(
        f"  Vehicle mode: obs_dim={obs_dim}, global_obs={vehicle_obs_include_global}, "
        f"route_stats={vehicle_obs_route_stats}, route_cost_obs={vehicle_obs_include_route_costs}, "
        f"fleet_pressure_obs={vehicle_obs_include_fleet_pressure}"
    )
    print(f"  Human eps (effective): {args.human_learning_episodes} (requested {args.human_learning_episodes})")

    print(f"  obs_dim = {obs_dim}  (n_edges={wm.n_edges}, n_dynamic={wm.d_obs})\n")

    policy = load_policy(args.policy_ckpt, obs_dim, args.n_routes, str(device))
    print()

    gawron_mode = str(args.hdv_gawron_mode).strip().lower()
    gawron_mode_ckpt = str(cfg_dict.get("gawron_mode", gawron_mode)).strip().lower()
    if gawron_mode_ckpt != gawron_mode:
        print(
            f"  NOTE: collector-style eval overrides checkpoint gawron_mode={gawron_mode_ckpt} "
            f"-> {gawron_mode}."
        )
    if not args.dynamic_routing:
        print("  NOTE: collector-style SUMO eval still uses dynamic AV rerouting to match data collection.")

    collect_enabled = bool(args.collect_dataset_dir)
    if collect_enabled:
        print(
            f"  SUMO rollout collection enabled: dir={args.collect_dataset_dir}, "
            f"task={args.collect_task_name}, shard_size={args.collect_shard_size}"
        )

    od_data = load_od_data(network_folder, args.network)
    origins = od_data["origins"]
    destinations = od_data["destinations"]
    post_mutation_state = None
    if post_mutation_state_in_path is not None and post_mutation_state_in_path.exists() and load_post_mutation_state is not None:
        post_mutation_state = load_post_mutation_state(post_mutation_state_in_path)
    reused_artifacts = None
    if post_mutation_state is not None:
        reused_artifacts = _reuse_route_artifacts_from_refs(
            post_mutation_state.get("artifact_refs", {}),
            results_dir,
        )
        if verify_post_mutation_state is not None:
            expected_should_adapt = (
                False if bool(args.disable_human_learning_after_mutation) else None
            )
            verify_post_mutation_state(
                results_dir=results_dir,
                expected_network=args.network,
                expected_ratio_machines=float(args.ratio_machines),
                expected_human_model="gawron",
                expected_should_humans_adapt=expected_should_adapt,
                state=post_mutation_state,
            )
        if loaded_hdv_state is None:
            state_ref = post_mutation_state.get("human_learning_state_ref")
            if state_ref:
                loaded_hdv_state = _load_hdv_learning_state(Path(str(state_ref)))
        agents_df = pd.read_csv(reused_artifacts["agents_csv"])
    elif args.reuse_artifacts_dir:
        reused_artifacts = _reuse_route_artifacts(Path(args.reuse_artifacts_dir).resolve(), results_dir)
        agents_df = pd.read_csv(reused_artifacts["agents_csv"])
    else:
        agents_df = load_agents_dataframe(network_folder)
    num_agents, num_machines, max_start_time = compute_agent_counts(
        agents_df, args.ratio_machines
    )
    if reused_artifacts is None:
        ensure_agents_csv_in_records(network_folder, str(results_dir), overwrite=True)
        if str(args.route_generator).strip().lower() == "routerl":
            generate_routerl_routes(
                network_folder=network_folder,
                network_name=args.network,
                origins=origins,
                destinations=destinations,
                n_routes=args.n_routes,
                records_folder=str(results_dir),
                simulation_timesteps=max_start_time + int(args.sim_buffer_after_last_depart),
                num_agents=num_agents,
                beta=float(args.path_gen_beta),
                num_samples=int(args.path_num_samples),
            )
            ensure_agents_csv_in_records(network_folder, str(results_dir), overwrite=True)
        else:
            generate_dijkstra_routes(
                network_folder=network_folder,
                network_name=args.network,
                origins=origins,
                destinations=destinations,
                n_routes=args.n_routes,
                records_folder=str(results_dir),
            )
    sim_length = int(
        max(
            float(max_start_time),
            float(agents_df["start_time"].max()) if "start_time" in agents_df.columns else float(max_start_time),
        ) + float(args.sim_buffer_after_last_depart)
    )

    effective_human_eps = 0 if post_mutation_state is not None else int(args.human_learning_episodes)
    protocol_human_eps, mutation_episode_index = _resolve_protocol_episode_indices(
        post_mutation_state=post_mutation_state,
        effective_human_eps=int(effective_human_eps),
        requested_human_eps=int(args.human_learning_episodes),
    )
    print(f"\n  Agents: {num_agents} total, {num_machines} AVs ({args.ratio_machines*100:.0f}%), "
          f"{num_agents - num_machines} HDVs")
    print(f"  SUMO sim length: {sim_length} timesteps")
    rollout_max_steps = int(args.max_steps_per_episode) if int(args.max_steps_per_episode) > 0 else int(sim_length)
    print(f"  Rollout max steps: {rollout_max_steps} timesteps")
    if int(args.mpc_horizon) > 1:
        print(
            "  NOTE: current WM reranker uses a baseline-centered local candidate set; "
            f"requested mpc_horizon={args.mpc_horizon} is used by repeating the candidate "
            "planned-flow action across the rollout horizon."
        )
    print(f"  WM CFM init mode: {args.wm_cfm_init_mode}")
    print(f"  WM policy obs   : {wm.policy_obs_source}")
    print(f"  WM edge cost    : {args.wm_edge_cost_source}")
    print(f"  WM cand cost    : {args.wm_candidate_edge_cost_source}")
    print(f"  WM volume floor : {float(args.wm_volume_floor):.4f}")
    print(f"  AV route select : {args.av_policy_route_selection}")
    print(f"  Human eps (effective): {effective_human_eps}\n")
    if post_mutation_state is not None:
        print(f"  Mutation episode index (reused): {mutation_episode_index}")
        print(f"  Human eps (protocol): {protocol_human_eps}\n")

    collector_output_dir = args.collect_dataset_dir if collect_enabled else str(results_dir / "collector_runtime")
    collector = PolicyGuidedCollectorEval(
        wm=wm,
        policy=policy,
        deterministic=args.deterministic,
        episode_steps=args.episode_steps,
        disable_human_learning_after_mutation=bool(args.disable_human_learning_after_mutation),
        mpc_horizon=args.mpc_horizon,
        mpc_candidates=args.mpc_candidates,
        mpc_discount=args.mpc_discount,
        rerank_t_all_weight=args.wm_rerank_t_all_weight,
        rerank_t_cav_weight=args.wm_rerank_t_cav_weight,
        rerank_cav_tolerance=args.wm_rerank_cav_tolerance,
        wm_volume_floor=args.wm_volume_floor,
        loaded_hdv_state=loaded_hdv_state,
        av_od_route_overrides=av_od_route_overrides,
        av_policy_route_selection=args.av_policy_route_selection,
        av_policy_sample_temperature=args.av_policy_sample_temperature,
        av_policy_confidence_gate_source=args.av_policy_confidence_gate_source,
        av_policy_confidence_gate_margin=args.av_policy_confidence_gate_margin,
        av_policy_confidence_gate_trigger_share=args.av_policy_confidence_gate_trigger_share,
        av_policy_nonzero_route_share_cap=args.av_policy_nonzero_route_share_cap,
        av_policy_share_cap_fallback_source=args.av_policy_share_cap_fallback_source,
        av_candidate_source=args.av_candidate_source,
        av_candidate_path_penalty_factor=vehicle_path_penalty_factor,
        av_route_cost_policy=args.av_route_cost_policy,
        av_route_cost_length_norm=args.av_route_cost_length_norm,
        av_route_cost_temperature=args.av_route_cost_temperature,
        av_route_hdv_pressure_penalty=args.av_route_hdv_pressure_penalty,
        av_route_wm_live_blend_weight=args.av_route_wm_live_blend_weight,
        av_route_planned_av_pressure_penalty=args.av_route_planned_av_pressure_penalty,
        wm_mcc_diagnostic_penalties=wm_mcc_diagnostic_penalties,
        wm_mcc_diagnostic_horizon=args.wm_mcc_diagnostic_horizon,
        bpr_alpha=args.bpr_alpha,
        bpr_beta=args.bpr_beta,
        wm_edge_cost_source=args.wm_edge_cost_source,
        wm_candidate_edge_cost_source=args.wm_candidate_edge_cost_source,
        wm_decoded_speed_min_ratio=args.wm_decoded_speed_min_ratio,
        wm_decoded_speed_max_ratio=args.wm_decoded_speed_max_ratio,
        wm_decoded_speed_excess_scale=args.wm_decoded_speed_excess_scale,
        vehicle_obs_route_stats=vehicle_obs_route_stats,
        vehicle_obs_include_route_costs=vehicle_obs_include_route_costs,
        vehicle_obs_include_fleet_pressure=vehicle_obs_include_fleet_pressure,
        output_dir=collector_output_dir,
        shard_size=args.collect_shard_size,
        seed=args.seed,
        backend="routerl",
        network_name=args.network,
        network_folder=network_folder,
        origins=origins,
        destinations=destinations,
        number_of_paths=args.n_routes,
        num_agents=num_agents,
        simulation_timesteps=sim_length,
        decision_steps=args.episode_steps,
        sim_buffer_after_last_depart=args.sim_buffer_after_last_depart,
        sumo_depart_lane=args.sumo_depart_lane,
        sumo_depart_pos=args.sumo_depart_pos,
        sumo_depart_speed=args.sumo_depart_speed,
        sumo_time_to_teleport=args.sumo_time_to_teleport,
        sumo_eager_insert=args.sumo_eager_insert,
        sumo_sloppy_insert=args.sumo_sloppy_insert,
        sumo_emergency_insert=args.sumo_emergency_insert,
        sumo_extra_args=args.sumo_extra_args,
        incident_edge=args.incident_edge,
        incident_start=args.incident_start,
        incident_end=args.incident_end,
        incident_speed=args.incident_speed,
        records_folder=str(results_dir),
        av_fraction=float(args.ratio_machines),
        random_av_fraction=0.0,
        epsilon=0.0,
        agents_csv=str(reused_artifacts["agents_csv"]) if reused_artifacts is not None else os.path.join(network_folder, "agents.csv"),
        edge_registry_path=edge_registry_path,
        hdv_gawron_mode=gawron_mode,
        hdv_gawron_alpha_zero=float(args.hdv_gawron_alpha_zero),
        hdv_gawron_beta=float(args.hdv_gawron_beta),
        hdv_gawron_update_mode=str(args.hdv_gawron_update_mode),
        od_file=os.path.join(network_folder, f"od_{args.network}.txt"),
        route_registry_path=os.path.join(network_folder, "route_registry.json"),
        skip_hdv_route_registry=False,
        use_light_hdv_route_registry=True,
    )

    episode_rows: List[Dict[str, Any]] = []
    if effective_human_eps > 0:
        print(f"=== Phase 1: Human stabilisation ({effective_human_eps} episodes) ===")
        collector.av_fraction = 0.0
        adapt_result = collector.run_eval_episodes(
            start_episode_id=1,
            num_episodes=effective_human_eps,
            max_steps_per_episode=rollout_max_steps,
            use_policy_actions=False,
            desc="human_adapt",
            log_every=args.episode_log_every,
            collect_dataset=False,
        )
        episode_rows.extend(adapt_result["rows"])
    else:
        print("=== Phase 1: Human stabilisation skipped ===")

    latest_hdv_state = _merge_hdv_learning_state(
        loaded_hdv_state,
        _snapshot_hdv_learning_state(collector.connector.env) if hasattr(collector.connector, "env") else {},
        route_state=_snapshot_hdv_route_state(collector),
    )
    if hdv_state_out_path is not None:
        _save_hdv_learning_state(hdv_state_out_path, latest_hdv_state)

    if post_mutation_state is None and derive_fixed_roles is not None and build_post_mutation_state is not None:
        machine_agent_ids, human_agent_ids = derive_fixed_roles(
            agents_csv=Path(str(reused_artifacts["agents_csv"]) if reused_artifacts is not None else os.path.join(network_folder, "agents.csv")),
            ratio_machines=float(args.ratio_machines),
            seed=int(args.seed),
        )
        human_state_ref = None
        if hdv_state_out_path is not None:
            human_state_ref = str(hdv_state_out_path)
        post_mutation_state = build_post_mutation_state(
            settings_path=str(args.settings),
            forwarded_argv=sys.argv,
            mutation_episode_index=int(effective_human_eps),
            machine_agent_ids=machine_agent_ids,
            human_agent_ids=human_agent_ids,
            human_learning_state_ref=human_state_ref,
        )
        if post_mutation_state_out_path is not None and write_post_mutation_state is not None:
            write_post_mutation_state(post_mutation_state_out_path, post_mutation_state)

    if post_mutation_state is not None:
        collector.set_fixed_roles(
            machine_ids=[str(x) for x in post_mutation_state.get("machine_agent_ids", [])],
            human_ids=[str(x) for x in post_mutation_state.get("human_agent_ids", [])],
        )
        if latest_hdv_state:
            collector._loaded_hdv_state_eval = latest_hdv_state
        if effective_human_eps == 0 and hasattr(collector.connector, "_episode_count"):
            # Keep RouteRLConnector's per-episode RNG aligned with the full
            # protocol path. Otherwise a reused post-mutation bundle starts its
            # first test reset at internal episode 1 instead of mutation+1.
            collector.connector._episode_count = int(mutation_episode_index)

    print(f"\n=== Phase 2: Testing ({args.test_eps} episodes, deterministic={args.deterministic}) ===")
    collector.av_fraction = float(args.ratio_machines)
    test_result = collector.run_eval_episodes(
        start_episode_id=int(mutation_episode_index) + 1,
        num_episodes=int(args.test_eps),
        max_steps_per_episode=rollout_max_steps,
        use_policy_actions=True,
        desc="test",
        log_every=args.episode_log_every,
        collect_dataset=collect_enabled,
    )
    episode_rows.extend(test_result["rows"])
    collector.finalize_protocol_xml_aliases()

    decision_trace_df = pd.DataFrame(getattr(collector, "_dynamic_decision_rows", []))
    decision_trace_path = results_dir / "dynamic_av_decisions.csv"
    if len(decision_trace_df) > 0:
        decision_trace_df.to_csv(decision_trace_path, index=False)
    agent_decision_trace_df = pd.DataFrame(getattr(collector, "_dynamic_agent_decision_rows", []))
    agent_decision_trace_path = results_dir / "dynamic_av_agent_decisions.csv"
    if len(agent_decision_trace_df) > 0:
        agent_decision_trace_df.to_csv(agent_decision_trace_path, index=False)
    wm_mcc_diagnostic_df = pd.DataFrame(getattr(collector, "_wm_mcc_diagnostic_rows", []))
    wm_mcc_diagnostic_path = results_dir / "wm_mcc_diagnostic.csv"
    if len(wm_mcc_diagnostic_df) > 0:
        wm_mcc_diagnostic_df.to_csv(wm_mcc_diagnostic_path, index=False)

    metrics_df = pd.DataFrame(episode_rows)
    metrics_path = results_dir / "collector_eval_episodes.csv"
    metrics_df.to_csv(metrics_path, index=False)
    test_df = metrics_df.loc[metrics_df["phase"] == "test"].copy()
    dynamic_route_counts: Dict[str, int] = {}
    dynamic_mean_route_probs: Dict[str, float] = {}
    wm_route_best_counts: Dict[str, int] = {}
    live_route_best_counts: Dict[str, int] = {}
    wm_mean_route_costs: Dict[str, float] = {}
    live_mean_route_costs: Dict[str, float] = {}
    dynamic_total_decisions = 0
    dynamic_total_active_av = 0
    candidate_dynamic_total = 0
    candidate_fallback_total = 0
    policy_confidence_gate_applied_total = 0
    policy_confidence_gate_triggered_ticks = 0
    policy_confidence_gate_max_nonzero_share_mean = float("nan")
    policy_share_cap_applied_total = 0
    candidate_unique_mean = float("nan")
    candidate_unique_min = 0
    if len(decision_trace_df) > 0:
        dynamic_total_decisions = int(len(decision_trace_df))
        dynamic_total_active_av = int(decision_trace_df["active_av"].sum())
        if "candidate_dynamic_count" in decision_trace_df.columns:
            candidate_dynamic_total = int(decision_trace_df["candidate_dynamic_count"].sum())
        if "candidate_fallback_count" in decision_trace_df.columns:
            candidate_fallback_total = int(decision_trace_df["candidate_fallback_count"].sum())
        if "policy_confidence_gate_applied" in decision_trace_df.columns:
            policy_confidence_gate_applied_total = int(decision_trace_df["policy_confidence_gate_applied"].sum())
        if "policy_confidence_gate_triggered" in decision_trace_df.columns:
            policy_confidence_gate_triggered_ticks = int(
                pd.to_numeric(decision_trace_df["policy_confidence_gate_triggered"], errors="coerce")
                .fillna(0)
                .astype(int)
                .sum()
            )
        if "policy_confidence_gate_max_nonzero_share" in decision_trace_df.columns:
            vals = pd.to_numeric(decision_trace_df["policy_confidence_gate_max_nonzero_share"], errors="coerce")
            policy_confidence_gate_max_nonzero_share_mean = (
                float(vals.mean()) if vals.notna().any() else float("nan")
            )
        if "policy_share_cap_applied" in decision_trace_df.columns:
            policy_share_cap_applied_total = int(decision_trace_df["policy_share_cap_applied"].sum())
        if "candidate_unique_mean" in decision_trace_df.columns:
            vals = pd.to_numeric(decision_trace_df["candidate_unique_mean"], errors="coerce")
            candidate_unique_mean = float(vals.mean()) if vals.notna().any() else float("nan")
        if "candidate_unique_min" in decision_trace_df.columns:
            vals = pd.to_numeric(decision_trace_df["candidate_unique_min"], errors="coerce")
            candidate_unique_min = int(vals.min()) if vals.notna().any() else 0
        for k in range(int(args.n_routes)):
            count_col = f"selected_count_route{k}"
            prob_col = f"mean_prob_route{k}"
            if count_col in decision_trace_df.columns:
                dynamic_route_counts[f"route{k}"] = int(decision_trace_df[count_col].sum())
            if prob_col in decision_trace_df.columns:
                dynamic_mean_route_probs[f"route{k}"] = float(decision_trace_df[prob_col].mean())
            wm_best_col = f"wm_best_count_route{k}"
            live_best_col = f"live_best_count_route{k}"
            wm_cost_col = f"wm_mean_route_cost_route{k}"
            live_cost_col = f"live_mean_route_cost_route{k}"
            if wm_best_col in decision_trace_df.columns:
                wm_route_best_counts[f"route{k}"] = int(decision_trace_df[wm_best_col].sum())
            if live_best_col in decision_trace_df.columns:
                live_route_best_counts[f"route{k}"] = int(decision_trace_df[live_best_col].sum())
            if wm_cost_col in decision_trace_df.columns:
                vals = pd.to_numeric(decision_trace_df[wm_cost_col], errors="coerce")
                wm_mean_route_costs[f"route{k}"] = float(vals.mean()) if vals.notna().any() else float("nan")
            if live_cost_col in decision_trace_df.columns:
                vals = pd.to_numeric(decision_trace_df[live_cost_col], errors="coerce")
                live_mean_route_costs[f"route{k}"] = float(vals.mean()) if vals.notna().any() else float("nan")
    test_metric_summary = _summarize_test_episode_metrics(test_df)
    summary = {
        "network": args.network,
        "policy_ckpt": args.policy_ckpt,
        "gawron_mode": gawron_mode,
        "human_learning_episodes": int(protocol_human_eps),
        "effective_human_learning_episodes": int(effective_human_eps),
        "mutation_episode_index": int(mutation_episode_index),
        "test_eps": int(args.test_eps),
        "ratio_machines": float(args.ratio_machines),
        "dynamic_routing": True,
        "dynamic_decision_trace": str(decision_trace_path) if len(decision_trace_df) > 0 else "",
        "dynamic_agent_decision_trace": str(agent_decision_trace_path) if len(agent_decision_trace_df) > 0 else "",
        "wm_mcc_diagnostic_trace": str(wm_mcc_diagnostic_path) if len(wm_mcc_diagnostic_df) > 0 else "",
        "dynamic_total_decision_ticks": dynamic_total_decisions,
        "dynamic_total_active_av_decisions": dynamic_total_active_av,
        "dynamic_selected_route_counts": dynamic_route_counts,
        "dynamic_mean_route_probs": dynamic_mean_route_probs,
        "av_candidate_source": str(args.av_candidate_source),
        "candidate_dynamic_total": int(candidate_dynamic_total),
        "candidate_fallback_total": int(candidate_fallback_total),
        "candidate_unique_mean": float(candidate_unique_mean),
        "candidate_unique_min": int(candidate_unique_min),
        "wm_route_best_counts": wm_route_best_counts,
        "live_route_best_counts": live_route_best_counts,
        "wm_mean_route_costs": wm_mean_route_costs,
        "live_mean_route_costs": live_mean_route_costs,
        "av_policy_route_selection": str(args.av_policy_route_selection),
        "av_policy_sample_temperature": float(args.av_policy_sample_temperature),
        "av_policy_confidence_gate_source": str(args.av_policy_confidence_gate_source),
        "av_policy_confidence_gate_margin": float(args.av_policy_confidence_gate_margin),
        "av_policy_confidence_gate_trigger_share": float(args.av_policy_confidence_gate_trigger_share),
        "av_policy_confidence_gate_triggered_ticks": int(policy_confidence_gate_triggered_ticks),
        "av_policy_confidence_gate_max_nonzero_share_mean": float(policy_confidence_gate_max_nonzero_share_mean),
        "av_policy_confidence_gate_applied_total": int(policy_confidence_gate_applied_total),
        "av_policy_nonzero_route_share_cap": float(args.av_policy_nonzero_route_share_cap),
        "av_policy_share_cap_fallback_source": str(args.av_policy_share_cap_fallback_source),
        "av_policy_share_cap_applied_total": int(policy_share_cap_applied_total),
        "av_od_route_override_count": int(len(av_od_route_overrides)),
        "av_route_cost_policy": str(args.av_route_cost_policy),
        "wm_edge_cost_source": str(args.wm_edge_cost_source),
        "wm_candidate_edge_cost_source": str(args.wm_candidate_edge_cost_source),
        "wm_decoded_speed_excess_scale": float(args.wm_decoded_speed_excess_scale),
        "av_route_cost_length_norm": str(args.av_route_cost_length_norm),
        "av_route_cost_temperature": float(args.av_route_cost_temperature),
        "av_route_hdv_pressure_penalty": float(args.av_route_hdv_pressure_penalty),
        "av_route_wm_live_blend_weight": float(args.av_route_wm_live_blend_weight),
        "av_route_planned_av_pressure_penalty": float(args.av_route_planned_av_pressure_penalty),
        "vehicle_obs_route_stats": str(vehicle_obs_route_stats),
        "vehicle_obs_include_route_costs": bool(vehicle_obs_include_route_costs),
        "vehicle_obs_include_fleet_pressure": bool(vehicle_obs_include_fleet_pressure),
        "collect_dataset_saved_episodes": int(test_result.get("dataset_saved_episodes", 0)),
        "collect_dataset_skipped_incomplete_episodes": int(test_result.get("dataset_skipped_incomplete_episodes", 0)),
        **test_metric_summary,
    }
    summary_path = results_dir / "collector_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if collect_enabled and test_result.get("dataset") is not None:
        dataset = test_result["dataset"]
        _save_collected_flow_dataset(
            args.collect_dataset_dir,
            args.collect_task_name,
            args.collect_shard_size,
            episodes=dataset["episodes"],
            av_edge_actions=dataset["av_edge_actions"],
            hdv_edge_actions=dataset["hdv_edge_actions"],
            joint_edge_actions=dataset["joint_edge_actions"],
            rewards=dataset["rewards"],
            flows=dataset["flows"],
            departs=dataset["departs"],
            contexts=dataset["contexts"],
            decision_indices=dataset.get("decision_indices"),
            sim_steps=dataset.get("sim_steps"),
        )

    print("\n=== Summary ===")
    print(f"  t_CAV = {summary['t_CAV']:.4f} min")
    print(f"  t_HDV = {summary['t_HDV']:.4f} min")
    print(f"  t_all = {summary['t_all']:.4f} min")
    print(f"  Episode metrics: {metrics_path}")
    print(f"  Summary JSON:    {summary_path}")
    return




if __name__ == "__main__":
    main()
