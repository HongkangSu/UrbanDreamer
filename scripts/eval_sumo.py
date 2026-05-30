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
import json
import argparse
import socket
from pathlib import Path
from typing import Any, Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import torch

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


try:
    from scripts.eval_sumo_io import (
        _AV_KIND_VALUES,
        _HDV_KIND_VALUES,
        _copy_artifact_if_needed,
        _reuse_route_artifacts,
        _reuse_route_artifacts_from_refs,
        _detect_kind_column,
        compute_metrics_from_csvs,
        _summarize_test_episode_metrics,
        _episode_complete_for_dataset,
        _resolve_protocol_episode_indices,
        _metrics_from_csv_set,
        _snapshot_episode_csvs,
        _new_csvs_since,
        _snapshot_sumo_xmls,
        _new_sumo_xmls_since,
        _wait_for_parseable_xml,
        _expected_xml_root_tag,
        _wait_for_expected_sumo_xml,
        _copy_protocol_xml_aliases,
    )
    from scripts.eval_sumo_hdv_state import (
        _resolve_optional_path,
        _load_hdv_learning_state,
        _numeric_to_list,
        _snapshot_hdv_learning_state,
        _merge_hdv_learning_state,
        _restore_hdv_learning_state,
        _save_hdv_learning_state,
    )
    from scripts.eval_sumo_routing import (
        _load_av_od_route_overrides,
        _parse_float_list,
        _edge_route_cost_matrix,
        _select_min_cost_route_indices,
        _select_min_cost_route_indices_with_planned_pressure,
        _select_quota_route_indices,
        _route_cost_matrix_summary,
        _build_edge_adj_list,
        _route_batch_uniqueness_summary,
        _aggregate_global_frac,
        _normalize_prob_vector,
        _temperature_adjust_probs,
        _safe_parse_agent_id,
        _build_av_agent_index,
        _aggregate_vehicle_fracs_to_od,
        _vehicle_action_edge_distribution,
        _vehicle_action_edge_distribution_from_path_batch,
        _build_live_vehicle_path_batch,
        _build_live_vehicle_path_batch_cached,
        _dynamic_reroute_avs,
        _mpc_select_action,
        _infer_hdv_global_from_models,
        _infer_hdv_global_from_episode_actions,
        _infer_hdv_edge_distribution,
        _collect_hdv_equilibrium_distribution,
    )
except ImportError:
    from eval_sumo_io import (
        _AV_KIND_VALUES,
        _HDV_KIND_VALUES,
        _copy_artifact_if_needed,
        _reuse_route_artifacts,
        _reuse_route_artifacts_from_refs,
        _detect_kind_column,
        compute_metrics_from_csvs,
        _summarize_test_episode_metrics,
        _episode_complete_for_dataset,
        _resolve_protocol_episode_indices,
        _metrics_from_csv_set,
        _snapshot_episode_csvs,
        _new_csvs_since,
        _snapshot_sumo_xmls,
        _new_sumo_xmls_since,
        _wait_for_parseable_xml,
        _expected_xml_root_tag,
        _wait_for_expected_sumo_xml,
        _copy_protocol_xml_aliases,
    )
    from eval_sumo_hdv_state import (
        _resolve_optional_path,
        _load_hdv_learning_state,
        _numeric_to_list,
        _snapshot_hdv_learning_state,
        _merge_hdv_learning_state,
        _restore_hdv_learning_state,
        _save_hdv_learning_state,
    )
    from eval_sumo_routing import (
        _load_av_od_route_overrides,
        _parse_float_list,
        _edge_route_cost_matrix,
        _select_min_cost_route_indices,
        _select_min_cost_route_indices_with_planned_pressure,
        _select_quota_route_indices,
        _route_cost_matrix_summary,
        _build_edge_adj_list,
        _route_batch_uniqueness_summary,
        _aggregate_global_frac,
        _normalize_prob_vector,
        _temperature_adjust_probs,
        _safe_parse_agent_id,
        _build_av_agent_index,
        _aggregate_vehicle_fracs_to_od,
        _vehicle_action_edge_distribution,
        _vehicle_action_edge_distribution_from_path_batch,
        _build_live_vehicle_path_batch,
        _build_live_vehicle_path_batch_cached,
        _dynamic_reroute_avs,
        _mpc_select_action,
        _infer_hdv_global_from_models,
        _infer_hdv_global_from_episode_actions,
        _infer_hdv_edge_distribution,
        _collect_hdv_equilibrium_distribution,
    )

try:
    from scripts.eval_sumo_collector import PolicyGuidedCollectorEval
    from scripts.eval_sumo_dataset import _save_collected_flow_dataset
    from scripts.eval_sumo_policy import load_policy
except ImportError:
    from eval_sumo_collector import PolicyGuidedCollectorEval
    from eval_sumo_dataset import _save_collected_flow_dataset
    from eval_sumo_policy import load_policy


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
    from data_collector import serialize_hdv_route_state
except ImportError:
    from dreamer4.urban_routing.data_collector import serialize_hdv_route_state

try:
    from train_marl import vehicle_mode_obs_dim
except Exception:
    try:
        from dreamer4.urban_routing.train_marl import vehicle_mode_obs_dim
    except Exception:
        vehicle_mode_obs_dim = None

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
                   choices=["od_suffix", "current_shortest_live", "current_shortest_wm", "current_shortest_freeflow"],
                   help=(
                       "Candidate route source for dynamic AV decisions. 'od_suffix' is deprecated "
                       "legacy precomputed OD-route suffix behavior; 'current_shortest_live' and "
                       "'current_shortest_wm'/'current_shortest_freeflow' regenerate K diverse "
                       "current-edge-to-destination routes from live SUMO, explicit WM, or free-flow edge travel times."
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
    p.add_argument("--wm_candidate_edge_cost_source", choices=["auto", "freeflow", "bpr_volume", "decoded_speed", "route_head"], default="auto",
                   help="Edge cost source used only to generate current_shortest_wm candidate paths. auto/freeflow uses registry free-flow times; WM sources are explicit ablations.")
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


def _snapshot_hdv_route_state(collector) -> Optional[Dict[str, Any]]:
    return serialize_hdv_route_state(
        getattr(collector, "_hdv_route_probs", None),
        getattr(collector, "_hdv_route_costs", None),
    )








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
