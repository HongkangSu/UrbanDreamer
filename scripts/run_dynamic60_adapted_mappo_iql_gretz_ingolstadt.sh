#!/usr/bin/env bash
set -Eeuo pipefail

# Re-run the two baselines whose update budgets were still calibrated for the
# original one-decision-per-episode URB task.  The two city pipelines run in
# parallel (one per GPU). After the per-GPU memory gate opens, MAPPO and IQL
# also run together so both algorithms produce early checkpoints instead of
# making IQL wait for a complete 500-episode MAPPO run.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

LOCK_DIR="${REPO_ROOT}/tmp/locks"
LOCK_FILE="${LOCK_DIR}/dynamic60_adapted_mappo_iql.lock"
mkdir -p "${LOCK_DIR}"
exec 9<>"${LOCK_FILE}"
if ! flock -n 9; then
  printf 'another dynamic60 adapted MAPPO/IQL launcher already holds %s\n' "${LOCK_FILE}" >&2
  exit 3
fi
printf 'pid=%s started=%s\n' "$$" "$(date '+%Y-%m-%d %H:%M:%S UTC')" >"${LOCK_FILE}"

PYTHON="${PYTHON:-/home/wangwenrui/opt/miniconda3/envs/URB/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_EPISODES="${TRAIN_EPISODES:-500}"
CHECKPOINT_EPISODES="${CHECKPOINT_EPISODES:-20,50,100,200,300,400,500}"
SEED="${SEED:-42}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
IQL_SIM_BUFFER_AFTER_LAST_DEPART="${IQL_SIM_BUFFER_AFTER_LAST_DEPART:-3000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-4500}"
GPU_GRETZ="${GPU_GRETZ:-0}"
GPU_INGOLSTADT="${GPU_INGOLSTADT:-1}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"

if [[ ! "${IQL_SIM_BUFFER_AFTER_LAST_DEPART}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'IQL_SIM_BUFFER_AFTER_LAST_DEPART must be a positive integer, got: %s\n' \
    "${IQL_SIM_BUFFER_AFTER_LAST_DEPART}" >&2
  exit 2
fi

SETTINGS_ROOT="${SETTINGS_ROOT:-${REPO_ROOT}/tmp/logs/torchrl_corrected_cpu6_500ep_20260718_074805/settings}"
LOG_DIR="${REPO_ROOT}/tmp/logs/dynamic60_adapted_mappo_iql_${RUN_ID}"
STATUS_FILE="${LOG_DIR}/status.tsv"
if [[ "${ALLOW_EXISTING_OUTPUT}" != "1" && -e "${LOG_DIR}" ]]; then
  if [[ ! -d "${LOG_DIR}" || -n "$(ls -A "${LOG_DIR}" 2>/dev/null)" ]]; then
    printf 'refusing non-empty launcher log path for RUN_ID=%s: %s\n' "${RUN_ID}" "${LOG_DIR}" >&2
    exit 4
  fi
fi
mkdir -p "${LOG_DIR}"
printf 'time\tstate\tcity\talgorithm\tgpu\tpid\trc\tout_dir\n' >"${STATUS_FILE}"

GRETZ_BASE="${REPO_ROOT}/results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1"
INGOLSTADT_BASE="${REPO_ROOT}/results/ingolstadt_custom_actionfix_incident_20260516_185810"
GRETZ_PROTOCOL="${REPO_ROOT}/results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/collect_urb_fair_eval_postmutation_records"
INGOLSTADT_PROTOCOL="${REPO_ROOT}/results/ingolstadt_custom_actionfix_incident_20260516_185810/collect_urb_fair_eval_postmutation_records"
MAPPO_SOURCE="${REPO_ROOT}/baseline/run_toy_cross_bottleneck_mappo_torchrl.py"
IQL_SOURCE="${REPO_ROOT}/baseline/run_toy_cross_bottleneck_iql_torchrl.py"
LAUNCHER_SOURCE="${REPO_ROOT}/scripts/run_dynamic60_adapted_mappo_iql_gretz_ingolstadt.sh"
GRETZ_SETTINGS="${SETTINGS_ROOT}/gretz.env"
INGOLSTADT_SETTINGS="${SETTINGS_ROOT}/ingolstadt.env"
GRETZ_POST_STATE="${GRETZ_PROTOCOL}/post_mutation_state.json"
GRETZ_HDV_STATE="${GRETZ_PROTOCOL}/post_mutation_hdv_learning_state.json"
INGOLSTADT_POST_STATE="${INGOLSTADT_PROTOCOL}/post_mutation_state.json"
INGOLSTADT_HDV_STATE="${INGOLSTADT_PROTOCOL}/post_mutation_hdv_learning_state.json"

BASELINE_CREATE_NETWORK_SOURCE="${REPO_ROOT}/baseline/create_toy_cross_bottleneck_network.py"
BASELINE_ONLINE_ACTOR_CRITIC_SOURCE="${REPO_ROOT}/baseline/online_actor_critic.py"
BASELINE_ONLINE_AC_RUNNER_SOURCE="${REPO_ROOT}/baseline/run_toy_cross_bottleneck_online_ac.py"
BASELINE_DYNAMIC_EVAL_SOURCE="${REPO_ROOT}/baseline/eval_toy_online_ac_dynamic.py"
DREAMER_FLOW_PROJECTOR_SOURCE="${REPO_ROOT}/dreamer4/urban_routing/flow_projector.py"
DREAMER_TRAIN_MARL_SOURCE="${REPO_ROOT}/dreamer4/urban_routing/train_marl.py"
DREAMER_URB_FAIR_PROTOCOL_SOURCE="${REPO_ROOT}/dreamer4/urban_routing/urb_fair_protocol.py"
DREAMER_DATA_COLLECTOR_SOURCE="${REPO_ROOT}/dreamer4/urban_routing/data_collector.py"
EVAL_SUMO_IO_SOURCE="${REPO_ROOT}/scripts/eval_sumo_io.py"

artifact_ref_path() {
  "${PYTHON}" -c \
    'import json, sys; from pathlib import Path; state = json.loads(Path(sys.argv[1]).read_text()); print(Path(str(state["artifact_refs"][sys.argv[2]])).resolve())' \
    "$1" "$2"
}

# These are the route artifacts actually copied into each job runtime when a
# post-mutation state is supplied. Resolve them from the frozen state instead
# of assuming that similarly named files under networks/ are consumed.
GRETZ_PROTOCOL_AGENTS_CSV="$(artifact_ref_path "${GRETZ_POST_STATE}" agents_csv)"
GRETZ_PROTOCOL_ROUTES_CSV="$(artifact_ref_path "${GRETZ_POST_STATE}" routes_csv)"
GRETZ_PROTOCOL_ROUTE_ROU_XML="$(artifact_ref_path "${GRETZ_POST_STATE}" route_rou_xml)"
INGOLSTADT_PROTOCOL_AGENTS_CSV="$(artifact_ref_path "${INGOLSTADT_POST_STATE}" agents_csv)"
INGOLSTADT_PROTOCOL_ROUTES_CSV="$(artifact_ref_path "${INGOLSTADT_POST_STATE}" routes_csv)"
INGOLSTADT_PROTOCOL_ROUTE_ROU_XML="$(artifact_ref_path "${INGOLSTADT_POST_STATE}" route_rou_xml)"

GRETZ_NETWORK_DIR="${REPO_ROOT}/networks/gretz_armainvilliers"
INGOLSTADT_NETWORK_DIR="${REPO_ROOT}/networks/ingolstadt_custom"
GRETZ_NET_XML="${GRETZ_NETWORK_DIR}/gretz_armainvilliers.net.xml"
GRETZ_EDGE_REGISTRY="${GRETZ_NETWORK_DIR}/edge_registry.json"
GRETZ_ROUTE_REGISTRY="${GRETZ_NETWORK_DIR}/route_registry.json"
GRETZ_AGENTS_CSV="${GRETZ_NETWORK_DIR}/agents.csv"
GRETZ_OD_FILE="${GRETZ_NETWORK_DIR}/od_gretz_armainvilliers.txt"
INGOLSTADT_NET_XML="${INGOLSTADT_NETWORK_DIR}/ingolstadt_custom.net.xml"
INGOLSTADT_EDGE_REGISTRY="${INGOLSTADT_NETWORK_DIR}/edge_registry.json"
INGOLSTADT_ROUTE_REGISTRY="${INGOLSTADT_NETWORK_DIR}/route_registry.json"
INGOLSTADT_AGENTS_CSV="${INGOLSTADT_NETWORK_DIR}/agents.csv"
INGOLSTADT_OD_FILE="${INGOLSTADT_NETWORK_DIR}/od_ingolstadt_custom.txt"

file_sha256() {
  local value
  value="$(sha256sum "$1")"
  printf '%s' "${value%% *}"
}

file_sha256_or_missing() {
  if [[ ! -e "$1" ]]; then
    printf 'MISSING'
    return 0
  fi
  file_sha256 "$1"
}

EXPECTED_MAPPO_SHA="$(file_sha256 "${MAPPO_SOURCE}")"
EXPECTED_IQL_SHA="$(file_sha256 "${IQL_SOURCE}")"
EXPECTED_LAUNCHER_SHA="$(file_sha256 "${LAUNCHER_SOURCE}")"
EXPECTED_GRETZ_SETTINGS_SHA="$(file_sha256 "${GRETZ_SETTINGS}")"
EXPECTED_INGOLSTADT_SETTINGS_SHA="$(file_sha256 "${INGOLSTADT_SETTINGS}")"
EXPECTED_GRETZ_POST_SHA="$(file_sha256 "${GRETZ_POST_STATE}")"
EXPECTED_GRETZ_HDV_SHA="$(file_sha256 "${GRETZ_HDV_STATE}")"
EXPECTED_INGOLSTADT_POST_SHA="$(file_sha256 "${INGOLSTADT_POST_STATE}")"
EXPECTED_INGOLSTADT_HDV_SHA="$(file_sha256 "${INGOLSTADT_HDV_STATE}")"

EXTRA_REQUIRED_LABELS=(
  baseline_create_network
  baseline_online_actor_critic
  baseline_online_ac_runner
  baseline_dynamic_eval
  dreamer_flow_projector
  dreamer_train_marl
  dreamer_urb_fair_protocol
  dreamer_data_collector
  eval_sumo_io
  gretz_protocol_agents_csv
  gretz_protocol_routes_csv
  gretz_protocol_route_rou_xml
  ingolstadt_protocol_agents_csv
  ingolstadt_protocol_routes_csv
  ingolstadt_protocol_route_rou_xml
  gretz_net_xml
  gretz_edge_registry
  gretz_agents_csv
  gretz_od_file
  ingolstadt_net_xml
  ingolstadt_edge_registry
  ingolstadt_agents_csv
  ingolstadt_od_file
)
EXTRA_REQUIRED_PATHS=(
  "${BASELINE_CREATE_NETWORK_SOURCE}"
  "${BASELINE_ONLINE_ACTOR_CRITIC_SOURCE}"
  "${BASELINE_ONLINE_AC_RUNNER_SOURCE}"
  "${BASELINE_DYNAMIC_EVAL_SOURCE}"
  "${DREAMER_FLOW_PROJECTOR_SOURCE}"
  "${DREAMER_TRAIN_MARL_SOURCE}"
  "${DREAMER_URB_FAIR_PROTOCOL_SOURCE}"
  "${DREAMER_DATA_COLLECTOR_SOURCE}"
  "${EVAL_SUMO_IO_SOURCE}"
  "${GRETZ_PROTOCOL_AGENTS_CSV}"
  "${GRETZ_PROTOCOL_ROUTES_CSV}"
  "${GRETZ_PROTOCOL_ROUTE_ROU_XML}"
  "${INGOLSTADT_PROTOCOL_AGENTS_CSV}"
  "${INGOLSTADT_PROTOCOL_ROUTES_CSV}"
  "${INGOLSTADT_PROTOCOL_ROUTE_ROU_XML}"
  "${GRETZ_NET_XML}"
  "${GRETZ_EDGE_REGISTRY}"
  "${GRETZ_AGENTS_CSV}"
  "${GRETZ_OD_FILE}"
  "${INGOLSTADT_NET_XML}"
  "${INGOLSTADT_EDGE_REGISTRY}"
  "${INGOLSTADT_AGENTS_CSV}"
  "${INGOLSTADT_OD_FILE}"
)
EXPECTED_EXTRA_REQUIRED_SHAS=()
for index in "${!EXTRA_REQUIRED_LABELS[@]}"; do
  EXPECTED_EXTRA_REQUIRED_SHAS+=("$(file_sha256 "${EXTRA_REQUIRED_PATHS[index]}")")
done

ROUTE_REGISTRY_LABELS=(gretz_route_registry ingolstadt_route_registry)
ROUTE_REGISTRY_PATHS=("${GRETZ_ROUTE_REGISTRY}" "${INGOLSTADT_ROUTE_REGISTRY}")
EXPECTED_ROUTE_REGISTRY_STATES=()
for index in "${!ROUTE_REGISTRY_LABELS[@]}"; do
  EXPECTED_ROUTE_REGISTRY_STATES+=("$(file_sha256_or_missing "${ROUTE_REGISTRY_PATHS[index]}")")
done

{
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
  printf 'python=%s\n' "${PYTHON}"
  printf 'train_device=%s\n' "${TRAIN_DEVICE}"
  printf 'iql_sim_buffer_after_last_depart=%s\n' "${IQL_SIM_BUFFER_AFTER_LAST_DEPART}"
  printf 'train_episodes=%s\n' "${TRAIN_EPISODES}"
  printf 'checkpoint_episodes=%s\n' "${CHECKPOINT_EPISODES}"
  printf 'seed=%s\n' "${SEED}"
  printf 'gpu_gretz=%s\n' "${GPU_GRETZ}"
  printf 'gpu_ingolstadt=%s\n' "${GPU_INGOLSTADT}"
  printf 'min_free_mib=%s\n' "${MIN_FREE_MIB}"
  printf 'allow_existing_output=%s\n' "${ALLOW_EXISTING_OUTPUT}"
  printf 'scheduling=two_cities_parallel_and_mappo_iql_parallel_within_city\n'
  printf 'singleton_lock=%s\n' "${LOCK_FILE}"
  printf 'mappo=joint_frame,batch_episodes=1,epochs=5,minibatch_frames=64,lr=3e-4,entropy=1e-2,normalized_advantage=true,critic_cells=128\n'
  printf 'iql=smdp_arrival_tail=true,target_update=per_optimizer_step,fresh_transition_epochs=1,replay_sample_ratio=0.25,batch_episodes=1,replay_epochs=1,minibatch=128,memory_gretz=100000,memory_ingolstadt=250000,eps_init=0.99,eps_end=0.05,exploration_fraction=0.4,epsilon_scope=first_decision_per_vehicle\n'
  printf 'sumo_time_to_teleport=-1\n'
  printf 'sha256_mappo=%s\n' "${EXPECTED_MAPPO_SHA}"
  printf 'sha256_iql=%s\n' "${EXPECTED_IQL_SHA}"
  printf 'sha256_launcher=%s\n' "${EXPECTED_LAUNCHER_SHA}"
  printf 'sha256_gretz_settings=%s\n' "${EXPECTED_GRETZ_SETTINGS_SHA}"
  printf 'sha256_ingolstadt_settings=%s\n' "${EXPECTED_INGOLSTADT_SETTINGS_SHA}"
  printf 'sha256_gretz_post_state=%s\n' "${EXPECTED_GRETZ_POST_SHA}"
  printf 'sha256_gretz_hdv_state=%s\n' "${EXPECTED_GRETZ_HDV_SHA}"
  printf 'sha256_ingolstadt_post_state=%s\n' "${EXPECTED_INGOLSTADT_POST_SHA}"
  printf 'sha256_ingolstadt_hdv_state=%s\n' "${EXPECTED_INGOLSTADT_HDV_SHA}"
  for index in "${!EXTRA_REQUIRED_LABELS[@]}"; do
    printf 'sha256_%s=%s\n' "${EXTRA_REQUIRED_LABELS[index]}" "${EXPECTED_EXTRA_REQUIRED_SHAS[index]}"
    printf 'path_%s=%s\n' "${EXTRA_REQUIRED_LABELS[index]}" "${EXTRA_REQUIRED_PATHS[index]}"
  done
  for index in "${!ROUTE_REGISTRY_LABELS[@]}"; do
    printf 'state_%s=%s\n' "${ROUTE_REGISTRY_LABELS[index]}" "${EXPECTED_ROUTE_REGISTRY_STATES[index]}"
    printf 'path_%s=%s\n' "${ROUTE_REGISTRY_LABELS[index]}" "${ROUTE_REGISTRY_PATHS[index]}"
  done
  printf 'git_status_begin\n'
  git status --short
  printf 'git_status_end\n'
} >"${LOG_DIR}/manifest.txt"

verify_frozen_inputs() {
  local label="$1" path="$2" expected="$3" actual
  actual="$(file_sha256 "${path}")"
  if [[ "${actual}" != "${expected}" ]]; then
    printf '[%s] source/input drift: %s expected=%s actual=%s path=%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${label}" "${expected}" "${actual}" "${path}" | tee -a "${LOG_DIR}/queue.log" >&2
    return 1
  fi
}

verify_frozen_input_state() {
  local label="$1" path="$2" expected="$3" actual
  actual="$(file_sha256_or_missing "${path}")"
  if [[ "${actual}" != "${expected}" ]]; then
    printf '[%s] source/input state drift: %s expected=%s actual=%s path=%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${label}" "${expected}" "${actual}" "${path}" | tee -a "${LOG_DIR}/queue.log" >&2
    return 1
  fi
}

verify_all_frozen_inputs() {
  verify_frozen_inputs mappo "${MAPPO_SOURCE}" "${EXPECTED_MAPPO_SHA}" || return
  verify_frozen_inputs iql "${IQL_SOURCE}" "${EXPECTED_IQL_SHA}" || return
  verify_frozen_inputs launcher "${LAUNCHER_SOURCE}" "${EXPECTED_LAUNCHER_SHA}" || return
  verify_frozen_inputs gretz_settings "${GRETZ_SETTINGS}" "${EXPECTED_GRETZ_SETTINGS_SHA}" || return
  verify_frozen_inputs ingolstadt_settings "${INGOLSTADT_SETTINGS}" "${EXPECTED_INGOLSTADT_SETTINGS_SHA}" || return
  verify_frozen_inputs gretz_post_state "${GRETZ_POST_STATE}" "${EXPECTED_GRETZ_POST_SHA}" || return
  verify_frozen_inputs gretz_hdv_state "${GRETZ_HDV_STATE}" "${EXPECTED_GRETZ_HDV_SHA}" || return
  verify_frozen_inputs ingolstadt_post_state "${INGOLSTADT_POST_STATE}" "${EXPECTED_INGOLSTADT_POST_SHA}" || return
  verify_frozen_inputs ingolstadt_hdv_state "${INGOLSTADT_HDV_STATE}" "${EXPECTED_INGOLSTADT_HDV_SHA}" || return
  for index in "${!EXTRA_REQUIRED_LABELS[@]}"; do
    verify_frozen_inputs \
      "${EXTRA_REQUIRED_LABELS[index]}" \
      "${EXTRA_REQUIRED_PATHS[index]}" \
      "${EXPECTED_EXTRA_REQUIRED_SHAS[index]}" || return
  done
  for index in "${!ROUTE_REGISTRY_LABELS[@]}"; do
    verify_frozen_input_state \
      "${ROUTE_REGISTRY_LABELS[index]}" \
      "${ROUTE_REGISTRY_PATHS[index]}" \
      "${EXPECTED_ROUTE_REGISTRY_STATES[index]}" || return
  done
}

status() {
  local state="$1" city="$2" algorithm="$3" gpu="$4" pid="$5" rc="$6" out_dir="$7"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${state}" "${city}" "${algorithm}" \
    "${gpu}" "${pid}" "${rc}" "${out_dir}" >>"${STATUS_FILE}"
}

wait_for_gpu() {
  local gpu="$1" free_mib attempts=0
  if [[ "${TRAIN_DEVICE}" != cuda* || "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  while true; do
    free_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits)"
    free_mib="${free_mib%% *}"
    if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      printf '[%s] GPU %s ready: %s MiB free\n' "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${gpu}" "${free_mib}" | tee -a "${LOG_DIR}/queue.log"
      return 0
    fi
    if (( attempts % 10 == 0 )); then
      printf '[%s] waiting for GPU %s: %s MiB free, need %s MiB\n' \
        "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${gpu}" "${free_mib:-unknown}" "${MIN_FREE_MIB}" | tee -a "${LOG_DIR}/queue.log"
    fi
    attempts=$((attempts + 1))
    sleep 30
  done
}

run_one() {
  local city="$1" algorithm="$2" gpu="$3" gpu_ready="${4:-0}"
  local settings base protocol memory_size script out_dir rc child_pid="" started=0 command_line
  local -a cmd common algorithm_args sumo_args dry_args

  cancel_run_one() {
    local signal_name="$1" signal_rc="$2"
    trap - HUP INT TERM
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
      kill -TERM -- "-${child_pid}" 2>/dev/null || kill -TERM "${child_pid}" 2>/dev/null || true
      wait "${child_pid}" 2>/dev/null || true
    fi
    if [[ "${started}" == "1" ]]; then
      status CANCELLED "${city}" "${algorithm}" "${gpu}" "${BASHPID}" "${signal_rc}" "${out_dir}"
    fi
    printf '[%s] CANCELLED city=%s algorithm=%s signal=%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${city}" "${algorithm}" "${signal_name}" | tee -a "${LOG_DIR}/queue.log"
    exit "${signal_rc}"
  }
  trap 'cancel_run_one HUP 129' HUP
  trap 'cancel_run_one INT 130' INT
  trap 'cancel_run_one TERM 143' TERM

  case "${city}" in
    gretz)
      settings="${SETTINGS_ROOT}/gretz.env"
      base="${GRETZ_BASE}"
      protocol="${GRETZ_PROTOCOL}"
      memory_size=100000
      sumo_args=(--sumo-extra-args "--collision.action warn")
      ;;
    ingolstadt)
      settings="${SETTINGS_ROOT}/ingolstadt.env"
      base="${INGOLSTADT_BASE}"
      protocol="${INGOLSTADT_PROTOCOL}"
      memory_size=250000
      sumo_args=()
      ;;
    *)
      printf 'unknown city: %s\n' "${city}" >&2
      return 2
      ;;
  esac

  case "${algorithm}" in
    mappo)
      script="baseline/run_toy_cross_bottleneck_mappo_torchrl.py"
      algorithm_args=(
        --agent-frames-per-batch 1
        --num-epochs 5
        --minibatch-size 64
        --lr 3e-4
        --entropy-eps 1e-2
        --normalize-advantage
        --critic-network-depth 3
        --critic-network-num-cells 128
        --centralized-state-mode joint_frame
      )
      ;;
    iql)
      script="baseline/run_toy_cross_bottleneck_iql_torchrl.py"
      algorithm_args=(
        --agent-frames-per-batch 1
        --num-epochs 1
        --minibatch-size 128
        --memory-size "${memory_size}"
        --fresh-transition-epochs 1
        --replay-sample-ratio 0.25
        --eps-init 0.99
        --eps-end 0.05
        --exploration-fraction 0.4
        --epsilon-scope first_decision_per_vehicle
        --target-update-eps 0.995
        --train-sim-buffer-after-last-depart "${IQL_SIM_BUFFER_AFTER_LAST_DEPART}"
      )
      ;;
    *)
      printf 'unknown algorithm: %s\n' "${algorithm}" >&2
      return 2
      ;;
  esac

  out_dir="${base}/baselines_torchrl_dynamic60_adapted_${TRAIN_EPISODES}ep_${RUN_ID}/${algorithm}"
  if [[ "${ALLOW_EXISTING_OUTPUT}" != "1" && -d "${out_dir}" && -n "$(ls -A "${out_dir}")" ]]; then
    printf '[%s] refusing non-empty output directory: %s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${out_dir}" | tee -a "${LOG_DIR}/queue.log" >&2
    status REFUSED "${city}" "${algorithm}" "${gpu}" "${BASHPID}" 4 "${out_dir}"
    return 4
  fi
  mkdir -p "${out_dir}"
  common=(
    --settings "${settings}"
    --output-dir "${out_dir}"
    --post-mutation-state-in "${protocol}/post_mutation_state.json"
    --hdv-state-in "${protocol}/post_mutation_hdv_learning_state.json"
    --train-episodes "${TRAIN_EPISODES}"
    --checkpoint-episodes "${CHECKPOINT_EPISODES}"
    --sumo-time-to-teleport -1
    --device "$( [[ "${TRAIN_DEVICE}" == cuda* ]] && printf 'cuda:0' || printf '%s' "${TRAIN_DEVICE}" )"
    --seed "${SEED}"
  )
  dry_args=()
  if [[ "${DRY_RUN}" == "1" ]]; then
    dry_args=(--dry-run)
  fi
  cmd=("${PYTHON}" -u "${script}" "${common[@]}" "${algorithm_args[@]}" "${sumo_args[@]}" "${dry_args[@]}")

  if [[ "${gpu_ready}" != "1" ]]; then
    wait_for_gpu "${gpu}"
  fi
  if ! verify_all_frozen_inputs; then
    status REFUSED "${city}" "${algorithm}" "${gpu}" "${BASHPID}" 5 "${out_dir}"
    return 5
  fi
  printf -v command_line '%q ' "${cmd[@]}"
  if [[ "${TRAIN_DEVICE}" == cuda* ]]; then
    command_line="env CUDA_VISIBLE_DEVICES=$(printf '%q' "${gpu}") setsid ${command_line}"
  else
    command_line="setsid ${command_line}"
  fi
  printf '%s city=%s algorithm=%s gpu=%s command=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${city}" "${algorithm}" "${gpu}" "${command_line}" >>"${LOG_DIR}/commands.sh"
  status START "${city}" "${algorithm}" "${gpu}" "${BASHPID}" - "${out_dir}"
  started=1
  printf '[%s] START city=%s algorithm=%s gpu=%s out=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${city}" "${algorithm}" "${gpu}" "${out_dir}" | tee -a "${LOG_DIR}/queue.log"
  if [[ "${TRAIN_DEVICE}" == cuda* ]]; then
    env CUDA_VISIBLE_DEVICES="${gpu}" setsid "${cmd[@]}" >"${out_dir}/train.log" 2>&1 &
  else
    setsid "${cmd[@]}" >"${out_dir}/train.log" 2>&1 &
  fi
  child_pid=$!
  status RUNNING "${city}" "${algorithm}" "${gpu}" "${child_pid}" - "${out_dir}"
  if wait "${child_pid}"; then
    rc=0
  else
    rc=$?
  fi
  child_pid=""
  trap - HUP INT TERM
  if [[ "${rc}" == "0" ]]; then
    status DONE "${city}" "${algorithm}" "${gpu}" "${BASHPID}" "${rc}" "${out_dir}"
  else
    status FAIL "${city}" "${algorithm}" "${gpu}" "${BASHPID}" "${rc}" "${out_dir}"
  fi
  printf '[%s] DONE city=%s algorithm=%s gpu=%s rc=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${city}" "${algorithm}" "${gpu}" "${rc}" | tee -a "${LOG_DIR}/queue.log"
  return "${rc}"
}

run_city_pipeline() {
  local city="$1" gpu="$2" rc=0 job_rc mappo_pid="" iql_pid="" pid

  cancel_city_jobs() {
    local signal_name="$1" signal_rc="$2"
    trap - HUP INT TERM
    for pid in "${mappo_pid}" "${iql_pid}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}" 2>/dev/null || true
      fi
    done
    for pid in "${mappo_pid}" "${iql_pid}"; do
      if [[ -n "${pid}" ]]; then
        wait "${pid}" 2>/dev/null || true
      fi
    done
    printf '[%s] city pipeline cancelled city=%s signal=%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${city}" "${signal_name}" | tee -a "${LOG_DIR}/queue.log"
    exit "${signal_rc}"
  }
  trap 'cancel_city_jobs HUP 129' HUP
  trap 'cancel_city_jobs INT 130' INT
  trap 'cancel_city_jobs TERM 143' TERM

  wait_for_gpu "${gpu}"
  run_one "${city}" mappo "${gpu}" 1 &
  mappo_pid=$!
  run_one "${city}" iql "${gpu}" 1 &
  iql_pid=$!
  wait "${mappo_pid}" || { job_rc=$?; rc="${job_rc}"; }
  wait "${iql_pid}" || { job_rc=$?; rc="${job_rc}"; }
  trap - HUP INT TERM
  return "${rc}"
}

gretz_pid=""
ingolstadt_pid=""
cancel_pipelines() {
  local signal_name="$1" signal_rc="$2" pid
  trap - HUP INT TERM
  for pid in "${gretz_pid}" "${ingolstadt_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${gretz_pid}" "${ingolstadt_pid}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
  printf '[%s] launcher cancelled signal=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${signal_name}" | tee -a "${LOG_DIR}/queue.log"
  exit "${signal_rc}"
}
trap 'cancel_pipelines HUP 129' HUP
trap 'cancel_pipelines INT 130' INT
trap 'cancel_pipelines TERM 143' TERM

run_city_pipeline gretz "${GPU_GRETZ}" &
gretz_pid=$!
run_city_pipeline ingolstadt "${GPU_INGOLSTADT}" &
ingolstadt_pid=$!

overall_rc=0
wait "${gretz_pid}" || overall_rc=$?
wait "${ingolstadt_pid}" || overall_rc=$?
printf '[%s] all city pipelines finished rc=%s status=%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${overall_rc}" "${STATUS_FILE}" | tee -a "${LOG_DIR}/queue.log"
exit "${overall_rc}"
