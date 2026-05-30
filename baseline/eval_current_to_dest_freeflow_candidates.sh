#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wangwenrui/shk/UrbanDreamer"
PY="/home/wangwenrui/opt/miniconda3/envs/URB/bin/python"

POLICY="${POLICY:-logs/ingolstadt_custom_current_to_dest_incident_20260528/marl_ac_stability_refactor_current_to_dest_seed95_ep400_route0tr90_anchorveh25_anchorgchg01_progress_only_seed95_ep400_from_ep200/checkpoint_ep00400.pt}"
GPU="${GPU:-1}"
SEED="${SEED:-42}"
TEST_EPS="${TEST_EPS:-1}"
POLICY_NAME="${POLICY_NAME:-baseline_ref_policy}"

ROOT_TAG="ingolstadt_custom_current_to_dest_incident_20260528"
WM_DIR="logs/${ROOT_TAG}/dynamics_urb_fair_mainline_8k"
NETWORK_DIR="networks/ingolstadt_custom"
REUSE_DIR="results/${ROOT_TAG}/collect_urb_fair_eval_postmutation_records"

run_baseline() {
  local policy_name="$1"
  local route_policy="$2"
  local result_name="$3"

  env CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PY" scripts/eval_sumo.py \
      --settings ".urbandreamer/generated/${ROOT_TAG}/urb_fair_marl.env" \
      --policy_ckpt "$POLICY" \
      --projector_ckpt "$WM_DIR/flow_projector_latest.pt" \
      --dynamics_ckpt "$WM_DIR/latest.pt" \
      --edge_registry "$NETWORK_DIR/edge_registry.json" \
      --network ingolstadt_custom \
      --network_folder "$NETWORK_DIR" \
      --ratio_machines 0.4 \
      --human_learning_episodes 200 \
      --test_eps "$TEST_EPS" \
      --results_dir "results/${ROOT_TAG}/${result_name}" \
      --device cuda:0 \
      --seed "$SEED" \
      --deterministic \
      --dynamic_routing \
      --av_candidate_source current_shortest_freeflow \
      --av_policy_route_selection auto \
      --av_route_cost_policy "$route_policy" \
      --av_route_cost_length_norm sum \
      --av_route_cost_temperature 0.0 \
      --av_route_hdv_pressure_penalty 0.0 \
      --av_route_planned_av_pressure_penalty 0.0 \
      --wm_cfm_init_mode zero \
      --wm_policy_obs_source decoder_volume_tt \
      --wm_edge_cost_source decoded_speed \
      --wm_candidate_edge_cost_source freeflow \
      --vehicle_obs_include_route_costs \
      --reuse_artifacts_dir "$REUSE_DIR" \
      --hdv_state_in "$REUSE_DIR/hdv_learning_state_final.json" \
      --post_mutation_state_in "$REUSE_DIR/post_mutation_state.json"
}

cd "$ROOT"

run_baseline "$POLICY_NAME" freeflow_greedy "baseline_${POLICY_NAME}_freeflow_candidates_freeflow_greedy"
run_baseline "$POLICY_NAME" live_current_tt "baseline_${POLICY_NAME}_freeflow_candidates_live_current_greedy"
