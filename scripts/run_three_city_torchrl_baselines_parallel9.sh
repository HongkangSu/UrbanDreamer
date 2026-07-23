#!/usr/bin/env bash
set -uo pipefail

# One job per GPU (recommended):
#   GPU_LIST=0,1,2,3,4,5,6,7,8 \
#     scripts/run_three_city_torchrl_baselines_parallel9.sh
# Fewer GPUs are accepted only with explicit oversubscription:
#   GPU_LIST=0,1,2 ALLOW_GPU_OVERSUBSCRIBE=1 \
#     scripts/run_three_city_torchrl_baselines_parallel9.sh
# Select a subset of cities and/or use CPU workers when GPUs are unavailable:
#   CITIES=gretz,ingolstadt TRAIN_DEVICE=cpu \
#     scripts/run_three_city_torchrl_baselines_parallel9.sh
# Resume selected jobs into an explicitly existing output tree:
#   ALLOW_EXISTING_OUTPUT=1 RESUME_EPISODE_GRETZ_IPPO=100 \
#     scripts/run_three_city_torchrl_baselines_parallel9.sh
# This script trains nine checkpoints. Run the unified long-horizon evaluator
# separately after training so evaluation uses its own 9000/7200 protocol.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TRAIN_EPISODES="${TRAIN_EPISODES:-500}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
AGENT_FRAMES_PER_BATCH="${AGENT_FRAMES_PER_BATCH:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-32}"
IQL_MEMORY_SIZE="${IQL_MEMORY_SIZE:-20000}"
IQL_AGENT_FRAMES_PER_BATCH="${IQL_AGENT_FRAMES_PER_BATCH:-1}"
IQL_MINIBATCH_SIZE="${IQL_MINIBATCH_SIZE:-128}"
IQL_FRESH_TRANSITION_EPOCHS="${IQL_FRESH_TRANSITION_EPOCHS:-1}"
IQL_REPLAY_SAMPLE_RATIO="${IQL_REPLAY_SAMPLE_RATIO:-0.25}"
IQL_SIM_BUFFER_AFTER_LAST_DEPART="${IQL_SIM_BUFFER_AFTER_LAST_DEPART:-3000}"
SEED="${SEED:-42}"
STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
GPU_LIST="${GPU_LIST:-}"
ALLOW_GPU_OVERSUBSCRIBE="${ALLOW_GPU_OVERSUBSCRIBE:-0}"
CITIES="${CITIES-gretz,ingolstadt,fontenay}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-0}"

if [[ "${TRAIN_DEVICE}" != "cuda" && "${TRAIN_DEVICE}" != "cpu" ]]; then
  echo "TRAIN_DEVICE must be either cuda or cpu, got: ${TRAIN_DEVICE}" >&2
  exit 2
fi
if [[ "${ALLOW_GPU_OVERSUBSCRIBE}" != "0" && "${ALLOW_GPU_OVERSUBSCRIBE}" != "1" ]]; then
  echo "ALLOW_GPU_OVERSUBSCRIBE must be 0 or 1, got: ${ALLOW_GPU_OVERSUBSCRIBE}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1, got: ${DRY_RUN}" >&2
  exit 2
fi
if [[ "${ALLOW_EXISTING_OUTPUT}" != "0" && "${ALLOW_EXISTING_OUTPUT}" != "1" ]]; then
  echo "ALLOW_EXISTING_OUTPUT must be 0 or 1, got: ${ALLOW_EXISTING_OUTPUT}" >&2
  exit 2
fi
if [[ ! "${LAUNCH_STAGGER_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "LAUNCH_STAGGER_SECONDS must be a non-negative integer, got: ${LAUNCH_STAGGER_SECONDS}" >&2
  exit 2
fi
LAUNCH_STAGGER_SECONDS="$((10#${LAUNCH_STAGGER_SECONDS}))"
for integer_setting in \
  IQL_MEMORY_SIZE \
  IQL_AGENT_FRAMES_PER_BATCH \
  IQL_MINIBATCH_SIZE \
  IQL_SIM_BUFFER_AFTER_LAST_DEPART; do
  if [[ ! "${!integer_setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${integer_setting} must be a positive integer, got: ${!integer_setting}" >&2
    exit 2
  fi
done
if [[ ! "${IQL_FRESH_TRANSITION_EPOCHS}" =~ ^[0-9]+$ ]]; then
  echo "IQL_FRESH_TRANSITION_EPOCHS must be a non-negative integer, got: ${IQL_FRESH_TRANSITION_EPOCHS}" >&2
  exit 2
fi
if [[ -z "${CITIES}" || "${CITIES}" == ,* || "${CITIES}" == *, || "${CITIES}" == *,,* ]]; then
  echo "CITIES must be a non-empty comma-separated list without empty entries." >&2
  echo "Valid city names: gretz,ingolstadt,fontenay" >&2
  exit 2
fi

declare -A SEEN_CITIES=()
SELECTED_CITIES=()
IFS=',' read -r -a REQUESTED_CITIES <<< "${CITIES}"
for city in "${REQUESTED_CITIES[@]}"; do
  case "${city}" in
    gretz|ingolstadt|fontenay)
      ;;
    *)
      echo "Unknown city in CITIES: ${city}" >&2
      echo "Valid city names: gretz,ingolstadt,fontenay" >&2
      exit 2
      ;;
  esac
  if [[ -n "${SEEN_CITIES[${city}]+x}" ]]; then
    echo "Duplicate city in CITIES: ${city}" >&2
    exit 2
  fi
  SEEN_CITIES["${city}"]=1
  SELECTED_CITIES+=("${city}")
done
CITIES="$(IFS=,; printf '%s' "${SELECTED_CITIES[*]}")"

DEFAULT_PY="/home/wangwenrui/opt/miniconda3/envs/URB/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="${PYTHON_BIN}"
elif [[ -x "${DEFAULT_PY}" ]]; then
  PY="${DEFAULT_PY}"
else
  PY="$(command -v python || true)"
fi

if [[ -z "${PY}" || ! -x "${PY}" ]]; then
  echo "Python executable not found. Set PYTHON_BIN=/path/to/the/URB/python." >&2
  exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "setsid is required to clean up each Python/SUMO process group safely." >&2
  exit 2
fi

ENV_ROOT="$(cd "$(dirname "${PY}")/.." && pwd)"
export SUMO_HOME="${SUMO_HOME:-${ENV_ROOT}}"
export PATH="${SUMO_HOME}/bin:${PATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="${RUN_LOG_DIR:-${ROOT}/tmp/logs/torchrl_corrected_parallel9_${STAMP}}"
STATUS_FILE="${LOG_DIR}/status.tsv"
MASTER_LOG="${LOG_DIR}/master.log"
MANIFEST="${LOG_DIR}/manifest.txt"
MPLCONFIG_ROOT="${LOG_DIR}/matplotlib"
if ! mkdir -p "${LOG_DIR}" "${MPLCONFIG_ROOT}"; then
  echo "Failed to create run log directory: ${LOG_DIR}" >&2
  exit 2
fi

GRETZ_TAG="gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1"
GRETZ_POST_TAG="gretz_armainvilliers_current_to_dest_incident_20260530_odcompact"
ING_TAG="ingolstadt_custom_actionfix_incident_20260516_185810"
FONT_TAG="fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609"

GRETZ_SETTINGS_SOURCE=".urbandreamer/generated/${GRETZ_TAG}/urb_fair_eval.env"
GRETZ_POST="results/${GRETZ_POST_TAG}/collect_urb_fair_eval_postmutation_records/post_mutation_state.json"
GRETZ_HDV="results/${GRETZ_POST_TAG}/collect_urb_fair_eval_postmutation_records/post_mutation_hdv_learning_state.json"
ING_SETTINGS_SOURCE=".urbandreamer/generated/${ING_TAG}/urb_fair_eval.env"
ING_POST="results/${ING_TAG}/collect_urb_fair_eval_postmutation_records/post_mutation_state.json"
ING_HDV="results/${ING_TAG}/collect_urb_fair_eval_postmutation_records/post_mutation_hdv_learning_state.json"
FONT_SETTINGS_SOURCE=".urbandreamer/generated/${FONT_TAG}/urb_fair_eval.env"
FONT_POST="results/${FONT_TAG}/collect_urb_fair_eval_postmutation_records/post_mutation_state.json"
FONT_HDV="results/${FONT_TAG}/collect_urb_fair_eval_postmutation_records/post_mutation_hdv_learning_state.json"
FONT_BLOCKER60_SUMO_EXTRA="--collision.action warn --ignore-junction-blocker 60 --tripinfo-output.write-unfinished"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    return 1
  fi
}

settings_value() {
  local key="$1"
  local settings="$2"
  sed -n "s|^${key}=\"\(.*\)\"$|\1|p" "${settings}" | head -n 1
}

rebase_settings() {
  local city="$1"
  local source_settings="$2"
  local network_dir old_root rebased_settings
  require_file "${source_settings}" || return 1
  network_dir="$(settings_value NETWORK_DIR "${source_settings}")"
  old_root="${network_dir%%/networks/*}"
  if [[ -z "${network_dir}" || -z "${old_root}" || "${old_root}" == "${network_dir}" ]]; then
    echo "Cannot infer repository root from NETWORK_DIR in ${source_settings}" >&2
    return 1
  fi
  rebased_settings="${LOG_DIR}/settings/${city}.env"
  mkdir -p "$(dirname "${rebased_settings}")"
  "${PY}" -c \
    'from pathlib import Path; import sys; src, dst, old, new = sys.argv[1:]; Path(dst).write_text(Path(src).read_text().replace(old, new))' \
    "${source_settings}" "${rebased_settings}" "${old_root}" "${ROOT}"
  printf "%s\n" "${rebased_settings}"
}

# city|algorithm|entrypoint|tag|settings|post_state|hdv_state|max_steps|sumo_extra_args
TASKS=()
for city in "${SELECTED_CITIES[@]}"; do
  case "${city}" in
    gretz)
      if ! settings="$(rebase_settings gretz "${GRETZ_SETTINGS_SOURCE}")"; then
        exit 2
      fi
      tag="${GRETZ_TAG}"
      post_state="${GRETZ_POST}"
      hdv_state="${GRETZ_HDV}"
      max_steps=0
      sumo_extra="--collision.action warn"
      ;;
    ingolstadt)
      if ! settings="$(rebase_settings ingolstadt "${ING_SETTINGS_SOURCE}")"; then
        exit 2
      fi
      tag="${ING_TAG}"
      post_state="${ING_POST}"
      hdv_state="${ING_HDV}"
      max_steps=0
      sumo_extra=""
      ;;
    fontenay)
      if ! settings="$(rebase_settings fontenay "${FONT_SETTINGS_SOURCE}")"; then
        exit 2
      fi
      tag="${FONT_TAG}"
      post_state="${FONT_POST}"
      hdv_state="${FONT_HDV}"
      max_steps=3600
      sumo_extra="${FONT_BLOCKER60_SUMO_EXTRA}"
      ;;
  esac
  TASKS+=(
    "${city}|ippo|baseline/run_toy_cross_bottleneck_ippo_torchrl.py|${tag}|${settings}|${post_state}|${hdv_state}|${max_steps}|${sumo_extra}"
    "${city}|mappo|baseline/run_toy_cross_bottleneck_mappo_torchrl.py|${tag}|${settings}|${post_state}|${hdv_state}|${max_steps}|${sumo_extra}"
    "${city}|iql|baseline/run_toy_cross_bottleneck_iql_torchrl.py|${tag}|${settings}|${post_state}|${hdv_state}|${max_steps}|${sumo_extra}"
  )
done

resume_episode_for() {
  local city="$1"
  local algorithm="$2"
  local variable="RESUME_EPISODE_${city^^}_${algorithm^^}"
  local value="${!variable-}"
  if [[ -z "${value}" ]]; then
    value=0
  fi
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${variable} must be a non-negative integer, got: ${value}" >&2
    return 1
  fi
  value="$((10#${value}))"
  if (( value >= TRAIN_EPISODES && value != 0 )); then
    echo "${variable}=${value} must be smaller than TRAIN_EPISODES=${TRAIN_EPISODES}." >&2
    return 1
  fi
  printf '%s\n' "${value}"
}

resume_checkpoint_for() {
  local out_dir="$1"
  local episode="$2"
  printf '%s/checkpoints/checkpoint_ep%05d.pt\n' "${out_dir}" "${episode}"
}

validate_resume_checkpoint() {
  local checkpoint="$1"
  local expected_episode="$2"
  local algorithm="$3"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing or empty resume checkpoint: ${checkpoint}" >&2
    return 1
  fi
  "${PY}" -c '
import sys
import torch

path, expected_episode, algorithm = sys.argv[1], int(sys.argv[2]), sys.argv[3]
state = torch.load(path, map_location="cpu")
actual_episode = int(state.get("episode", -1))
expected_family = f"{algorithm}_torchrl"
actual_family = str(state.get("model_family", ""))
if actual_episode != expected_episode:
    raise SystemExit(
        f"{path}: checkpoint episode={actual_episode}, expected {expected_episode}"
    )
if actual_family != expected_family:
    raise SystemExit(
        f"{path}: model_family={actual_family!r}, expected {expected_family!r}"
    )
' "${checkpoint}" "${expected_episode}" "${algorithm}"
}

if [[ "${TRAIN_DEVICE}" == "cuda" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to validate CUDA GPU ids." >&2
    exit 2
  fi
  if ! available_gpu_output="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "nvidia-smi failed while enumerating CUDA GPUs." >&2
    exit 2
  fi
  mapfile -t AVAILABLE_GPUS <<< "${available_gpu_output}"
  declare -A AVAILABLE_GPU_IDS=()
  NORMALIZED_AVAILABLE_GPUS=()
  for gpu in "${AVAILABLE_GPUS[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    if [[ -n "${gpu}" ]]; then
      AVAILABLE_GPU_IDS["${gpu}"]=1
      NORMALIZED_AVAILABLE_GPUS+=("${gpu}")
    fi
  done
  if [[ "${#AVAILABLE_GPU_IDS[@]}" -eq 0 ]]; then
    echo "No CUDA GPUs detected by nvidia-smi." >&2
    exit 2
  fi
  if [[ -z "${GPU_LIST}" ]]; then
    GPU_LIST="$(IFS=,; printf '%s' "${NORMALIZED_AVAILABLE_GPUS[*]}")"
  fi
  if [[ "${GPU_LIST}" == ,* || "${GPU_LIST}" == *, || "${GPU_LIST}" == *,,* ]]; then
    echo "GPU_LIST must be a comma-separated list without empty entries: ${GPU_LIST}" >&2
    exit 2
  fi
  IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
  declare -A SEEN_GPUS=()
  for gpu in "${GPUS[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ || -z "${AVAILABLE_GPU_IDS[${gpu}]+x}" ]]; then
      echo "GPU_LIST contains unavailable CUDA GPU id: ${gpu}" >&2
      echo "Available GPU ids: $(IFS=,; printf '%s' "${NORMALIZED_AVAILABLE_GPUS[*]}")" >&2
      exit 2
    fi
    if [[ "${ALLOW_GPU_OVERSUBSCRIBE}" == "0" && -n "${SEEN_GPUS[${gpu}]+x}" ]]; then
      echo "GPU_LIST contains duplicate id ${gpu} without ALLOW_GPU_OVERSUBSCRIBE=1." >&2
      exit 2
    fi
    SEEN_GPUS["${gpu}"]=1
  done
  declare -A CUDA_PREFLIGHTED_GPUS=()
  for gpu in "${GPUS[@]}"; do
    if [[ -n "${CUDA_PREFLIGHTED_GPUS[${gpu}]+x}" ]]; then
      continue
    fi
    if ! CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -c '
import torch

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
probe = torch.empty(1, device="cuda")
torch.cuda.synchronize()
print(f"CUDA preflight ok: {torch.cuda.get_device_name(0)}")
'; then
      echo "PyTorch CUDA preflight failed for physical GPU ${gpu}." >&2
      exit 2
    fi
    CUDA_PREFLIGHTED_GPUS["${gpu}"]=1
  done
  if [[ "${#GPUS[@]}" -lt "${#TASKS[@]}" && "${ALLOW_GPU_OVERSUBSCRIBE}" != "1" ]]; then
    echo "Found ${#GPUS[@]} GPU slots for ${#TASKS[@]} simultaneous jobs." >&2
    echo "Provide one id per task, or set ALLOW_GPU_OVERSUBSCRIBE=1 to assign jobs round-robin." >&2
    exit 2
  fi
else
  GPU_LIST="cpu"
  GPUS=("cpu")
  available_threads="$(nproc)"
  if [[ ! "${available_threads}" =~ ^[1-9][0-9]*$ ]]; then
    echo "nproc returned an invalid CPU thread count: ${available_threads}" >&2
    exit 2
  fi
  if [[ -z "${CPU_THREADS_PER_JOB}" ]]; then
    CPU_THREADS_PER_JOB=$((available_threads / (${#TASKS[@]} + 2)))
    if [[ "${CPU_THREADS_PER_JOB}" -lt 1 ]]; then
      CPU_THREADS_PER_JOB=1
    elif [[ "${CPU_THREADS_PER_JOB}" -gt 8 ]]; then
      CPU_THREADS_PER_JOB=8
    fi
  elif [[ ! "${CPU_THREADS_PER_JOB}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CPU_THREADS_PER_JOB must be a positive integer, got: ${CPU_THREADS_PER_JOB}" >&2
    exit 2
  elif (( CPU_THREADS_PER_JOB > available_threads )); then
    echo "CPU_THREADS_PER_JOB=${CPU_THREADS_PER_JOB} exceeds available CPU threads (${available_threads})." >&2
    exit 2
  fi
fi

validate_settings_artifacts() {
  local settings="$1"
  local key path network_dir network_name route_artifacts_dir failed=0
  local -a required_runtime_files
  for key in NETWORK_DIR ROUTE_ARTIFACTS_DIR; do
    path="$(settings_value "${key}" "${settings}")"
    if [[ -z "${path}" || ! -d "${path}" ]]; then
      echo "Missing ${key} directory from ${settings}: ${path:-<unset>}" >&2
      failed=1
    fi
  done
  for key in EDGE_REGISTRY AGENTS_CSV OD_FILE; do
    path="$(settings_value "${key}" "${settings}")"
    if [[ -z "${path}" || ! -f "${path}" ]]; then
      echo "Missing ${key} file from ${settings}: ${path:-<unset>}" >&2
      failed=1
    fi
  done
  network_dir="$(settings_value NETWORK_DIR "${settings}")"
  network_name="$(settings_value NETWORK_NAME "${settings}")"
  route_artifacts_dir="$(settings_value ROUTE_ARTIFACTS_DIR "${settings}")"
  required_runtime_files=(
    "${network_dir}/${network_name}.net.xml"
    "${route_artifacts_dir}/agents.csv"
    "${route_artifacts_dir}/routes.csv"
    "${route_artifacts_dir}/route.rou.xml"
  )
  for path in "${required_runtime_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
      echo "Missing runtime artifact referenced by ${settings}: ${path}" >&2
      failed=1
    fi
  done
  return "${failed}"
}

preflight_failed=0
for task in "${TASKS[@]}"; do
  IFS='|' read -r city algorithm entrypoint tag settings post_state hdv_state max_steps sumo_extra <<< "${task}"
  validate_settings_artifacts "${settings}" || preflight_failed=1
  require_file "${entrypoint}" || preflight_failed=1
  require_file "${settings}" || preflight_failed=1
  require_file "${post_state}" || preflight_failed=1
  require_file "${hdv_state}" || preflight_failed=1
  out_dir="results/${tag}/baselines_torchrl_corrected_${TRAIN_EPISODES}ep_${STAMP}/${algorithm}"
  if ! resume_episode="$(resume_episode_for "${city}" "${algorithm}")"; then
    preflight_failed=1
    resume_episode=0
  fi
  if [[ -e "${out_dir}" && "${ALLOW_EXISTING_OUTPUT}" != "1" ]]; then
    echo "Output already exists: ${out_dir}" >&2
    preflight_failed=1
  fi
  if (( resume_episode > 0 )); then
    if [[ "${ALLOW_EXISTING_OUTPUT}" != "1" ]]; then
      echo "Resume requested for ${city}/${algorithm}, but ALLOW_EXISTING_OUTPUT is not 1." >&2
      preflight_failed=1
    fi
    resume_checkpoint="$(resume_checkpoint_for "${out_dir}" "${resume_episode}")"
    validate_resume_checkpoint "${resume_checkpoint}" "${resume_episode}" "${algorithm}" \
      || preflight_failed=1
  elif [[ -e "${out_dir}" && "${ALLOW_EXISTING_OUTPUT}" == "1" ]]; then
    echo "Existing output for ${city}/${algorithm} requires RESUME_EPISODE_${city^^}_${algorithm^^}." >&2
    preflight_failed=1
  fi
done
if [[ "${preflight_failed}" != "0" ]]; then
  exit 2
fi

write_status() {
  local state="$1"
  local city="$2"
  local algorithm="$3"
  local gpu="$4"
  local pid="$5"
  local rc="$6"
  local out_dir="$7"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${state}" "${city}" "${algorithm}" \
    "${gpu}" "${pid}" "${rc}" "${out_dir}" >> "${STATUS_FILE}"
}

worker_group_alive() {
  [[ -n "${WORKER_CHILD_PGID:-}" ]] && kill -0 -- "-${WORKER_CHILD_PGID}" 2>/dev/null
}

worker_cleanup() {
  local attempt
  if [[ "${WORKER_CLEANUP_DONE:-0}" == "1" ]]; then
    return "${WORKER_CLEANUP_RC:-0}"
  fi
  WORKER_CLEANUP_DONE=1
  WORKER_CLEANUP_RC=0
  trap '' HUP INT TERM

  # The signal can arrive in the tiny interval before setsid establishes the
  # new process group. Terminating the leader directly closes that race; the
  # group signal below handles the post-setsid Python/SUMO descendants.
  if [[ -n "${WORKER_CHILD_PID:-}" && "${WORKER_CHILD_REAPED:-0}" != "1" ]]; then
    kill -TERM "${WORKER_CHILD_PID}" 2>/dev/null || true
  fi
  if worker_group_alive; then
    kill -TERM -- "-${WORKER_CHILD_PGID}" 2>/dev/null || true
    for attempt in {1..10}; do
      if ! worker_group_alive; then
        break
      fi
      sleep 0.5
    done
  fi
  if worker_group_alive; then
    kill -KILL -- "-${WORKER_CHILD_PGID}" 2>/dev/null || true
  fi
  if [[ -n "${WORKER_CHILD_PID:-}" && "${WORKER_CHILD_REAPED:-0}" != "1" ]]; then
    wait "${WORKER_CHILD_PID}" 2>/dev/null || true
    WORKER_CHILD_REAPED=1
  fi
  for attempt in {1..10}; do
    if ! worker_group_alive; then
      break
    fi
    kill -KILL -- "-${WORKER_CHILD_PGID}" 2>/dev/null || true
    sleep 0.2
  done
  if worker_group_alive; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ERROR process group ${WORKER_CHILD_PGID} survived cleanup"
    WORKER_CLEANUP_RC=125
  fi
  return "${WORKER_CLEANUP_RC}"
}

worker_signal() {
  local signal_name="$1"
  local signal_rc="$2"
  if [[ "${WORKER_LAUNCHING:-0}" == "1" ]]; then
    if [[ -z "${WORKER_PENDING_SIGNAL_NAME:-}" ]]; then
      WORKER_PENDING_SIGNAL_NAME="${signal_name}"
      WORKER_PENDING_SIGNAL_RC="${signal_rc}"
    fi
    return
  fi
  if [[ "${WORKER_SIGNALLED:-0}" == "1" ]]; then
    return
  fi
  WORKER_SIGNALLED=1
  WORKER_SIGNAL_NAME="${signal_name}"
  WORKER_REQUESTED_RC="${signal_rc}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] CANCEL city=${WORKER_CITY} algorithm=${WORKER_ALGORITHM} signal=${signal_name}"
  exit "${signal_rc}"
}

worker_exit() {
  # Preserve the triggering status via the trap argument, then make EXIT
  # teardown non-interruptible before doing any other shell work.
  trap '' HUP INT TERM
  trap - EXIT
  if [[ "${WORKER_EXIT_ACTIVE:-0}" == "1" ]]; then
    return
  fi
  WORKER_EXIT_ACTIVE=1
  local original_rc="$1"
  local final_rc state
  final_rc="${WORKER_REQUESTED_RC:-${original_rc}}"

  if ! worker_cleanup; then
    final_rc=125
  fi

  # Bash <5.1 has no `wait -n -p`; notify the fallback supervisor before
  # publishing the terminal status so a broken notification cannot leave DONE.
  if [[ "${WAIT_HAS_P:-1}" == "0" ]]; then
    if ! printf '%s\n' "${BASHPID}" >&"${COMPLETION_FD}"; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ERROR failed to notify parent supervisor"
      final_rc=125
    fi
  fi

  if [[ "${WORKER_SIGNALLED:-0}" == "1" ]]; then
    state="CANCELLED"
  elif [[ "${final_rc}" == "0" ]]; then
    state="DONE"
  else
    state="FAIL"
  fi
  if ! write_status "${state}" "${WORKER_CITY}" "${WORKER_ALGORITHM}" \
      "${WORKER_GPU}" "${BASHPID}" "${final_rc}" "${WORKER_OUT_DIR}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ERROR failed to write terminal status=${state}"
    final_rc=125
  fi
  exit "${final_rc}"
}

run_one() {
  local city="$1"
  local algorithm="$2"
  local entrypoint="$3"
  local tag="$4"
  local settings="$5"
  local post_state="$6"
  local hdv_state="$7"
  local max_steps="$8"
  local sumo_extra="$9"
  local gpu="${10}"
  local out_dir="results/${tag}/baselines_torchrl_corrected_${TRAIN_EPISODES}ep_${STAMP}/${algorithm}"
  local resume_episode resume_checkpoint resume_label
  if ! resume_episode="$(resume_episode_for "${city}" "${algorithm}")"; then
    exit 125
  fi
  local log_file="${out_dir}/train.log"
  if (( resume_episode > 0 )); then
    printf -v resume_label '%05d' "${resume_episode}"
    log_file="${out_dir}/train_resume_ep${resume_label}_${TRAIN_DEVICE}.log"
  fi
  local mpl_dir="${MPLCONFIG_ROOT}/${city}_${algorithm}"
  local rc=0
  local child_pgid=""
  local -a cmd

  # These globals are process-local because run_one is launched as a background
  # shell job. Keeping them alive until EXIT makes early infrastructure failures
  # use the same cleanup/status path as training failures.
  WORKER_CITY="${city}"
  WORKER_ALGORITHM="${algorithm}"
  WORKER_GPU="${gpu}"
  WORKER_OUT_DIR="${out_dir}"
  WORKER_CHILD_PID=""
  WORKER_CHILD_PGID=""
  WORKER_CHILD_REAPED=0
  WORKER_CLEANUP_DONE=0
  WORKER_CLEANUP_RC=0
  WORKER_SIGNALLED=0
  WORKER_REQUESTED_RC=""
  WORKER_LAUNCHING=0
  WORKER_PENDING_SIGNAL_NAME=""
  WORKER_PENDING_SIGNAL_RC=""
  WORKER_EXIT_ACTIVE=0
  trap 'worker_signal HUP 129' HUP
  trap 'worker_signal INT 130' INT
  trap 'worker_signal TERM 143' TERM
  trap 'worker_exit "$?"' EXIT

  if ! mkdir -p "${out_dir}" "${mpl_dir}"; then
    echo "Failed to create output directory: ${out_dir}" >&2
    exit 125
  fi
  if ! exec > "${log_file}" 2>&1; then
    echo "Failed to open training log: ${log_file}" >&2
    exit 125
  fi
  if ! write_status "START" "${city}" "${algorithm}" "${gpu}" "${BASHPID}" "-" "${out_dir}"; then
    echo "Failed to write START status for ${city}/${algorithm}" >&2
    exit 125
  fi

  cmd=(
    "${PY}" -u "${entrypoint}"
    --settings "${settings}"
    --output-dir "${out_dir}"
    --post-mutation-state-in "${post_state}"
    --hdv-state-in "${hdv_state}"
    --train-episodes "${TRAIN_EPISODES}"
    --checkpoint-interval "${CHECKPOINT_INTERVAL}"
    --agent-frames-per-batch "${AGENT_FRAMES_PER_BATCH}"
    --num-epochs "${NUM_EPOCHS}"
    --minibatch-size "${MINIBATCH_SIZE}"
    --sumo-time-to-teleport -1
    --device "${TRAIN_DEVICE}"
    --seed "${SEED}"
  )
  if [[ "${max_steps}" != "0" ]]; then
    cmd+=(--max-steps-per-episode "${max_steps}")
  fi
  if [[ "${algorithm}" == "iql" ]]; then
    cmd+=(
      --memory-size "${IQL_MEMORY_SIZE}"
      --agent-frames-per-batch "${IQL_AGENT_FRAMES_PER_BATCH}"
      --num-epochs 1
      --minibatch-size "${IQL_MINIBATCH_SIZE}"
      --fresh-transition-epochs "${IQL_FRESH_TRANSITION_EPOCHS}"
      --replay-sample-ratio "${IQL_REPLAY_SAMPLE_RATIO}"
      --train-sim-buffer-after-last-depart "${IQL_SIM_BUFFER_AFTER_LAST_DEPART}"
      --eps-init 0.99
      --eps-end 0.05
      --exploration-fraction 0.4
      --epsilon-scope first_decision_per_vehicle
      --target-update-eps 0.995
    )
  fi
  if (( resume_episode > 0 )); then
    resume_checkpoint="$(resume_checkpoint_for "${out_dir}" "${resume_episode}")"
    cmd+=(--resume-checkpoint "${resume_checkpoint}" --resume-start-episode "${resume_episode}")
  fi
  if [[ -n "${sumo_extra}" ]]; then
    cmd+=(--sumo-extra-args "${sumo_extra}")
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] START city=${city} algorithm=${algorithm} device=${TRAIN_DEVICE} slot=${gpu}"
  echo "output_dir=${out_dir}"
  # Defer catchable signals until the just-forked child PID/PGID has been
  # registered. Otherwise a signal in the `command &` -> `$!` window could
  # make EXIT cleanup miss an already-running Python process.
  WORKER_LAUNCHING=1
  if [[ "${TRAIN_DEVICE}" == "cuda" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${gpu} MPLCONFIGDIR=${mpl_dir} ${cmd[*]}"
    CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${mpl_dir}" setsid "${cmd[@]}" &
  else
    echo "CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=${CPU_THREADS_PER_JOB} MPLCONFIGDIR=${mpl_dir} ${cmd[*]}"
    CUDA_VISIBLE_DEVICES="" \
      OMP_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
      MKL_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
      OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
      MPLCONFIGDIR="${mpl_dir}" \
      setsid "${cmd[@]}" &
  fi
  WORKER_CHILD_PID=$!
  WORKER_CHILD_PGID="${WORKER_CHILD_PID}"
  for _attempt in {1..100}; do
    child_pgid="$(ps -o pgid= -p "${WORKER_CHILD_PID}" 2>/dev/null | tr -d '[:space:]')"
    if [[ -z "${child_pgid}" || "${child_pgid}" == "${WORKER_CHILD_PGID}" ]]; then
      break
    fi
    sleep 0.01
  done
  if [[ -n "${child_pgid}" && "${child_pgid}" != "${WORKER_CHILD_PGID}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ERROR setsid child pid=${WORKER_CHILD_PID} has unexpected pgid=${child_pgid}"
    kill "${WORKER_CHILD_PID}" 2>/dev/null || true
    wait "${WORKER_CHILD_PID}" 2>/dev/null || true
    WORKER_CHILD_REAPED=1
    WORKER_CHILD_PGID=""
    WORKER_LAUNCHING=0
    exit 125
  fi
  WORKER_LAUNCHING=0
  if [[ -n "${WORKER_PENDING_SIGNAL_NAME}" ]]; then
    worker_signal "${WORKER_PENDING_SIGNAL_NAME}" "${WORKER_PENDING_SIGNAL_RC}"
  fi
  if wait "${WORKER_CHILD_PID}"; then
    rc=0
  else
    rc=$?
  fi
  WORKER_CHILD_REAPED=1
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] FINISH city=${city} algorithm=${algorithm} rc=${rc}"
  exit "${rc}"
}

if ! {
  echo "repo_root=${ROOT}"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python=${PY}"
  echo "sumo_home=${SUMO_HOME}"
  echo "stamp=${STAMP}"
  echo "cities=${CITIES}"
  echo "train_device=${TRAIN_DEVICE}"
  echo "gpu_list=${GPU_LIST}"
  echo "cpu_threads_per_job=${CPU_THREADS_PER_JOB:-not_applicable}"
  echo "train_episodes=${TRAIN_EPISODES}"
  echo "checkpoint_interval=${CHECKPOINT_INTERVAL}"
  echo "agent_frames_per_batch=${AGENT_FRAMES_PER_BATCH}"
  echo "num_epochs=${NUM_EPOCHS}"
  echo "minibatch_size=${MINIBATCH_SIZE}"
  echo "iql_memory_size=${IQL_MEMORY_SIZE}"
  echo "iql_agent_frames_per_batch=${IQL_AGENT_FRAMES_PER_BATCH}"
  echo "iql_minibatch_size=${IQL_MINIBATCH_SIZE}"
  echo "iql_fresh_transition_epochs=${IQL_FRESH_TRANSITION_EPOCHS}"
  echo "iql_replay_sample_ratio=${IQL_REPLAY_SAMPLE_RATIO}"
  echo "iql_sim_buffer_after_last_depart=${IQL_SIM_BUFFER_AFTER_LAST_DEPART}"
  echo "iql_epsilon_scope=first_decision_per_vehicle"
  echo "allow_existing_output=${ALLOW_EXISTING_OUTPUT}"
  echo "launch_stagger_seconds=${LAUNCH_STAGGER_SECONDS}"
  echo "seed=${SEED}"
  echo "sumo_time_to_teleport=-1"
  echo "fontenay_sumo_extra_args=${FONT_BLOCKER60_SUMO_EXTRA}"
  echo "ppo_gradient_coverage=all_actual_samples_once_per_epoch"
  echo "iql_update_semantics=persistent_transition_replay_without_replacement,fresh_full_coverage,smdp_arrival_tail,per_optimizer_target_update"
  for task in "${TASKS[@]}"; do
    IFS='|' read -r city algorithm _entrypoint _tag _settings _post_state _hdv_state _max_steps _sumo_extra <<< "${task}"
    echo "resume_episode_${city}_${algorithm}=$(resume_episode_for "${city}" "${algorithm}")"
  done
} > "${MANIFEST}"; then
  echo "Failed to initialize manifest: ${MANIFEST}" >&2
  exit 2
fi

if ! printf "time\tstate\tcity\talgorithm\tgpu\tpid\trc\tout_dir\n" > "${STATUS_FILE}"; then
  echo "Failed to initialize status file: ${STATUS_FILE}" >&2
  exit 2
fi
if ! echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] launching ${#TASKS[@]} jobs in parallel" > "${MASTER_LOG}"; then
  echo "Failed to initialize master log: ${MASTER_LOG}" >&2
  exit 2
fi
echo "stamp=${STAMP}"
echo "log_dir=${LOG_DIR}"
echo "status_file=${STATUS_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  for idx in "${!TASKS[@]}"; do
    IFS='|' read -r city algorithm entrypoint tag settings post_state hdv_state max_steps sumo_extra <<< "${TASKS[$idx]}"
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    memory_size="-"
    resume_episode="$(resume_episode_for "${city}" "${algorithm}")"
    if [[ "${algorithm}" == "iql" ]]; then
      memory_size="${IQL_MEMORY_SIZE}"
    fi
    printf "%s\t%s\tDEVICE=%s\tSLOT=%s\tTELEPORT=-1\tMEMORY_SIZE=%s\tRESUME_EPISODE=%s\tSUMO_EXTRA_ARGS=%q\tresults/%s/baselines_torchrl_corrected_%sep_%s/%s\n" \
      "${city}" "${algorithm}" "${TRAIN_DEVICE}" "${gpu}" "${memory_size}" "${resume_episode}" "${sumo_extra}" "${tag}" \
      "${TRAIN_EPISODES}" "${STAMP}" "${algorithm}"
  done
  exit 0
fi

WAIT_HAS_P=0
COMPLETION_FIFO=""
COMPLETION_FD=""
if (( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )); then
  WAIT_HAS_P=1
else
  COMPLETION_FIFO="${LOG_DIR}/completed.fifo"
  if ! mkfifo "${COMPLETION_FIFO}"; then
    echo "Failed to create Bash 5.0 completion FIFO: ${COMPLETION_FIFO}" >&2
    exit 2
  fi
  if ! exec {COMPLETION_FD}<>"${COMPLETION_FIFO}"; then
    echo "Failed to open Bash 5.0 completion FIFO: ${COMPLETION_FIFO}" >&2
    rm -f "${COMPLETION_FIFO}"
    exit 2
  fi
fi

declare -A ACTIVE_NAMES=()
ACTIVE_COUNT=0
PARENT_CLEANUP_DONE=0
PARENT_SIGNALLED=0
PARENT_EXIT_ACTIVE=0
PARENT_REGISTERING=0
PARENT_PENDING_SIGNAL_NAME=""
PARENT_PENDING_SIGNAL_RC=""

parent_cleanup() {
  local pid
  if [[ "${PARENT_CLEANUP_DONE}" == "1" ]]; then
    return
  fi
  PARENT_CLEANUP_DONE=1
  trap '' HUP INT TERM
  for pid in "${!ACTIVE_NAMES[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "${!ACTIVE_NAMES[@]}"; do
    wait "${pid}" 2>/dev/null || true
    unset 'ACTIVE_NAMES[$pid]'
  done
  ACTIVE_COUNT=0
  if [[ -n "${COMPLETION_FD}" ]]; then
    exec {COMPLETION_FD}>&-
    COMPLETION_FD=""
  fi
  if [[ -n "${COMPLETION_FIFO}" ]]; then
    rm -f "${COMPLETION_FIFO}"
  fi
}

parent_signal() {
  local signal_name="$1"
  local signal_rc="$2"
  if [[ "${PARENT_REGISTERING}" == "1" ]]; then
    if [[ -z "${PARENT_PENDING_SIGNAL_NAME}" ]]; then
      PARENT_PENDING_SIGNAL_NAME="${signal_name}"
      PARENT_PENDING_SIGNAL_RC="${signal_rc}"
    fi
    return
  fi
  if [[ "${PARENT_SIGNALLED}" == "1" ]]; then
    return
  fi
  PARENT_SIGNALLED=1
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] received ${signal_name}; terminating ${ACTIVE_COUNT} active jobs" >> "${MASTER_LOG}"
  exit "${signal_rc}"
}

parent_exit() {
  trap '' HUP INT TERM
  trap - EXIT
  if [[ "${PARENT_EXIT_ACTIVE}" == "1" ]]; then
    return
  fi
  PARENT_EXIT_ACTIVE=1
  local rc="$1"
  parent_cleanup
  exit "${rc}"
}

trap 'parent_signal HUP 129' HUP
trap 'parent_signal INT 130' INT
trap 'parent_signal TERM 143' TERM
trap 'parent_exit "$?"' EXIT

for idx in "${!TASKS[@]}"; do
  IFS='|' read -r city algorithm entrypoint tag settings post_state hdv_state max_steps sumo_extra <<< "${TASKS[$idx]}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  # As in the worker, defer signals across `run_one &` -> `$!` registration so
  # parent EXIT cleanup cannot miss a wrapper that has already been forked.
  PARENT_REGISTERING=1
  run_one "${city}" "${algorithm}" "${entrypoint}" "${tag}" "${settings}" \
    "${post_state}" "${hdv_state}" "${max_steps}" "${sumo_extra}" "${gpu}" &
  pid=$!
  ACTIVE_NAMES["${pid}"]="${city}/${algorithm}"
  ACTIVE_COUNT=$((ACTIVE_COUNT + 1))
  PARENT_REGISTERING=0
  if [[ -n "${PARENT_PENDING_SIGNAL_NAME}" ]]; then
    parent_signal "${PARENT_PENDING_SIGNAL_NAME}" "${PARENT_PENDING_SIGNAL_RC}"
  fi
  if (( LAUNCH_STAGGER_SECONDS > 0 && idx + 1 < ${#TASKS[@]} )); then
    sleep "${LAUNCH_STAGGER_SECONDS}"
  fi
done

failures=0
while (( ACTIVE_COUNT > 0 )); do
  finished_pid=""
  rc=0
  if [[ "${WAIT_HAS_P}" == "1" ]]; then
    active_pids=("${!ACTIVE_NAMES[@]}")
    if wait -n -p finished_pid "${active_pids[@]}"; then
      rc=0
    else
      rc=$?
    fi
  else
    while [[ -z "${finished_pid}" ]]; do
      candidate=""
      if IFS= read -r -t 5 candidate <&"${COMPLETION_FD}"; then
        if [[ "${candidate}" =~ ^[1-9][0-9]*$ && -n "${ACTIVE_NAMES[${candidate}]+x}" ]]; then
          finished_pid="${candidate}"
        fi
      else
        # A wrapper killed with SIGKILL cannot notify the FIFO. Bash retains
        # its wait status, so find the vanished active pid and reap it directly.
        for candidate in "${!ACTIVE_NAMES[@]}"; do
          if ! kill -0 "${candidate}" 2>/dev/null; then
            finished_pid="${candidate}"
            break
          fi
        done
      fi
    done
    if wait "${finished_pid}"; then
      rc=0
    else
      rc=$?
    fi
  fi

  if [[ -z "${finished_pid}" || -z "${ACTIVE_NAMES[${finished_pid}]+x}" ]]; then
    failures=$((failures + 1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] FAIL supervisor could not identify reaped pid rc=${rc}" >> "${MASTER_LOG}"
    exit 125
  fi
  name="${ACTIVE_NAMES[${finished_pid}]}"
  unset 'ACTIVE_NAMES[$finished_pid]'
  ACTIVE_COUNT=$((ACTIVE_COUNT - 1))
  if [[ "${rc}" == "0" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] DONE ${name}" >> "${MASTER_LOG}"
  else
    failures=$((failures + 1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] FAIL ${name} rc=${rc}" >> "${MASTER_LOG}"
  fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] all jobs finished failures=${failures}" >> "${MASTER_LOG}"
exit "${failures}"
