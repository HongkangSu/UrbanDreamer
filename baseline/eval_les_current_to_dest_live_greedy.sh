#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wangwenrui/shk/UrbanDreamer"
PY="/home/wangwenrui/opt/miniconda3/envs/URB/bin/python"

ROOT_TAG="${ROOT_TAG:-les_mureaux_current_to_dest_noincident_earlystop_wu1_col12_20260531}"
NETWORK="${NETWORK:-les_mureaux}"
NETWORK_DIR="${NETWORK_DIR:-networks/les_mureaux}"
WM_DIR="${WM_DIR:-logs/${ROOT_TAG}/dynamics_urb_fair_mainline_8k}"
MARL_DIR="${MARL_DIR:-logs/${ROOT_TAG}/marl_urb_fair_mainline_500ep}"
REUSE_DIR="${REUSE_DIR:-results/${ROOT_TAG}/collect_urb_fair_eval_postmutation_records}"
STATUS_FILE="${STATUS_FILE:-logs/${ROOT_TAG}/urb_fair_full_cycle.status.env}"

POLICY="${POLICY:-${MARL_DIR}/checkpoint_ep00200.pt}"
RESULT_NAME="${RESULT_NAME:-baseline_live_current_greedy_freeflow_candidates}"
GPU="${GPU:-1}"
SEED="${SEED:-42}"
TEST_EPS="${TEST_EPS:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-120}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-0}"

cd "$ROOT"

deadline=0
if [[ "$WAIT_TIMEOUT_SECONDS" != "0" ]]; then
  deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
fi

wait_for_file() {
  local path="$1"
  local label="$2"
  while [[ ! -s "$path" ]]; do
    echo "[les-live-greedy] waiting for ${label}: ${path}" >&2
    if [[ "$deadline" != "0" && "$(date +%s)" -ge "$deadline" ]]; then
      echo "[les-live-greedy] timed out waiting for ${label}: ${path}" >&2
      exit 1
    fi
    sleep "$WAIT_POLL_SECONDS"
  done
}

wait_for_status_done() {
  while [[ -s "$STATUS_FILE" ]] && grep -q "STATE='running'" "$STATUS_FILE"; do
    echo "[les-live-greedy] waiting for Les full-cycle to leave running state: ${STATUS_FILE}" >&2
    if [[ "$deadline" != "0" && "$(date +%s)" -ge "$deadline" ]]; then
      echo "[les-live-greedy] timed out waiting for Les full-cycle status" >&2
      exit 1
    fi
    sleep "$WAIT_POLL_SECONDS"
  done
}

wait_for_status_done
wait_for_file ".urbandreamer/generated/${ROOT_TAG}/urb_fair_marl.env" "generated MARL env"
wait_for_file "${WM_DIR}/flow_projector_latest.pt" "flow projector checkpoint"
wait_for_file "${WM_DIR}/latest.pt" "world model checkpoint"
wait_for_file "${POLICY}" "policy checkpoint"
wait_for_file "${NETWORK_DIR}/edge_registry.json" "edge registry"
wait_for_file "${REUSE_DIR}/post_mutation_state.json" "post-mutation state"

if [[ -s "${REUSE_DIR}/hdv_learning_state_final.json" ]]; then
  HDV_STATE="${REUSE_DIR}/hdv_learning_state_final.json"
else
  HDV_STATE="${REUSE_DIR}/post_mutation_hdv_learning_state.json"
fi
wait_for_file "$HDV_STATE" "HDV learning state"

echo "[les-live-greedy] running live_current_tt baseline into results/${ROOT_TAG}/${RESULT_NAME}" >&2

env CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$PY" scripts/eval_sumo.py \
    --settings ".urbandreamer/generated/${ROOT_TAG}/urb_fair_marl.env" \
    --policy_ckpt "$POLICY" \
    --projector_ckpt "$WM_DIR/flow_projector_latest.pt" \
    --dynamics_ckpt "$WM_DIR/latest.pt" \
    --edge_registry "$NETWORK_DIR/edge_registry.json" \
    --network "$NETWORK" \
    --network_folder "$NETWORK_DIR" \
    --ratio_machines 0.4 \
    --human_learning_episodes 200 \
    --test_eps "$TEST_EPS" \
    --results_dir "results/${ROOT_TAG}/${RESULT_NAME}" \
    --device cuda:0 \
    --seed "$SEED" \
    --deterministic \
    --dynamic_routing \
    --av_candidate_source current_shortest_freeflow \
    --av_policy_route_selection auto \
    --av_route_cost_policy live_current_tt \
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
    --hdv_state_in "$HDV_STATE" \
    --post_mutation_state_in "$REUSE_DIR/post_mutation_state.json"
