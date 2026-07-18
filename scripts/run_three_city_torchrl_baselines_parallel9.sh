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
SEED="${SEED:-42}"
STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
GPU_LIST="${GPU_LIST:-}"
ALLOW_GPU_OVERSUBSCRIBE="${ALLOW_GPU_OVERSUBSCRIBE:-0}"
CITIES="${CITIES:-gretz,ingolstadt,fontenay}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${TRAIN_DEVICE}" != "cuda" && "${TRAIN_DEVICE}" != "cpu" ]]; then
  echo "TRAIN_DEVICE must be either cuda or cpu, got: ${TRAIN_DEVICE}" >&2
  exit 2
fi

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
mkdir -p "${LOG_DIR}" "${MPLCONFIG_ROOT}"

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

if ! GRETZ_SETTINGS="$(rebase_settings gretz "${GRETZ_SETTINGS_SOURCE}")"; then
  exit 2
fi
if ! ING_SETTINGS="$(rebase_settings ingolstadt "${ING_SETTINGS_SOURCE}")"; then
  exit 2
fi
if ! FONT_SETTINGS="$(rebase_settings fontenay "${FONT_SETTINGS_SOURCE}")"; then
  exit 2
fi

# city|algorithm|entrypoint|tag|settings|post_state|hdv_state|max_steps|sumo_extra_args
TASKS=(
  "gretz|ippo|baseline/run_toy_cross_bottleneck_ippo_torchrl.py|${GRETZ_TAG}|${GRETZ_SETTINGS}|${GRETZ_POST}|${GRETZ_HDV}|0|--collision.action warn"
  "gretz|mappo|baseline/run_toy_cross_bottleneck_mappo_torchrl.py|${GRETZ_TAG}|${GRETZ_SETTINGS}|${GRETZ_POST}|${GRETZ_HDV}|0|--collision.action warn"
  "gretz|iql|baseline/run_toy_cross_bottleneck_iql_torchrl.py|${GRETZ_TAG}|${GRETZ_SETTINGS}|${GRETZ_POST}|${GRETZ_HDV}|0|--collision.action warn"
  "ingolstadt|ippo|baseline/run_toy_cross_bottleneck_ippo_torchrl.py|${ING_TAG}|${ING_SETTINGS}|${ING_POST}|${ING_HDV}|0|"
  "ingolstadt|mappo|baseline/run_toy_cross_bottleneck_mappo_torchrl.py|${ING_TAG}|${ING_SETTINGS}|${ING_POST}|${ING_HDV}|0|"
  "ingolstadt|iql|baseline/run_toy_cross_bottleneck_iql_torchrl.py|${ING_TAG}|${ING_SETTINGS}|${ING_POST}|${ING_HDV}|0|"
  "fontenay|ippo|baseline/run_toy_cross_bottleneck_ippo_torchrl.py|${FONT_TAG}|${FONT_SETTINGS}|${FONT_POST}|${FONT_HDV}|3600|${FONT_BLOCKER60_SUMO_EXTRA}"
  "fontenay|mappo|baseline/run_toy_cross_bottleneck_mappo_torchrl.py|${FONT_TAG}|${FONT_SETTINGS}|${FONT_POST}|${FONT_HDV}|3600|${FONT_BLOCKER60_SUMO_EXTRA}"
  "fontenay|iql|baseline/run_toy_cross_bottleneck_iql_torchrl.py|${FONT_TAG}|${FONT_SETTINGS}|${FONT_POST}|${FONT_HDV}|3600|${FONT_BLOCKER60_SUMO_EXTRA}"
)

IFS=',' read -r -a SELECTED_CITIES <<< "${CITIES}"
FILTERED_TASKS=()
for task in "${TASKS[@]}"; do
  IFS='|' read -r city _ <<< "${task}"
  for selected_city in "${SELECTED_CITIES[@]}"; do
    if [[ "${city}" == "${selected_city}" ]]; then
      FILTERED_TASKS+=("${task}")
      break
    fi
  done
done
TASKS=("${FILTERED_TASKS[@]}")
if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "CITIES selected no tasks: ${CITIES}" >&2
  echo "Valid city names: gretz,ingolstadt,fontenay" >&2
  exit 2
fi

if [[ "${TRAIN_DEVICE}" == "cuda" ]]; then
  if [[ -z "${GPU_LIST}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  fi
  if [[ -z "${GPU_LIST}" ]]; then
    echo "No GPUs detected. Set GPU_LIST to one GPU id per selected task." >&2
    exit 2
  fi
  IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
  if [[ "${#GPUS[@]}" -lt "${#TASKS[@]}" && "${ALLOW_GPU_OVERSUBSCRIBE}" != "1" ]]; then
    echo "Found ${#GPUS[@]} GPU slots for ${#TASKS[@]} simultaneous jobs." >&2
    echo "Provide one id per task, or set ALLOW_GPU_OVERSUBSCRIBE=1 to assign jobs round-robin." >&2
    exit 2
  fi
else
  GPU_LIST="cpu"
  GPUS=("cpu")
  if [[ -z "${CPU_THREADS_PER_JOB}" ]]; then
    available_threads="$(nproc)"
    CPU_THREADS_PER_JOB=$((available_threads / (${#TASKS[@]} + 2)))
    if [[ "${CPU_THREADS_PER_JOB}" -lt 1 ]]; then
      CPU_THREADS_PER_JOB=1
    elif [[ "${CPU_THREADS_PER_JOB}" -gt 8 ]]; then
      CPU_THREADS_PER_JOB=8
    fi
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
  if [[ -e "${out_dir}" && "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ]]; then
    echo "Output already exists: ${out_dir}" >&2
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
  local log_file="${out_dir}/train.log"
  local mpl_dir="${MPLCONFIG_ROOT}/${city}_${algorithm}"
  local rc=0
  local child_pid=""
  local -a cmd

  mkdir -p "${out_dir}" "${mpl_dir}"
  write_status "START" "${city}" "${algorithm}" "${gpu}" "${BASHPID}" "-" "${out_dir}"

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
    cmd+=(--memory-size "${IQL_MEMORY_SIZE}")
  fi
  if [[ -n "${sumo_extra}" ]]; then
    cmd+=(--sumo-extra-args "${sumo_extra}")
  fi

  {
    terminate_job_group() {
      echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] CANCEL city=${city} algorithm=${algorithm}"
      if [[ -n "${child_pid}" ]]; then
        kill -TERM -- "-${child_pid}" 2>/dev/null || true
        for _attempt in {1..10}; do
          if ! kill -0 -- "-${child_pid}" 2>/dev/null; then
            break
          fi
          sleep 1
        done
        kill -KILL -- "-${child_pid}" 2>/dev/null || true
        wait "${child_pid}" 2>/dev/null || true
      fi
      write_status "CANCELLED" "${city}" "${algorithm}" "${gpu}" "${BASHPID}" "143" "${out_dir}"
      exit 143
    }
    trap terminate_job_group INT TERM

    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] START city=${city} algorithm=${algorithm} device=${TRAIN_DEVICE} slot=${gpu}"
    echo "output_dir=${out_dir}"
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
    child_pid=$!
    if wait "${child_pid}"; then
      rc=0
    else
      rc=$?
    fi
    trap - INT TERM
    kill -TERM -- "-${child_pid}" 2>/dev/null || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] FINISH city=${city} algorithm=${algorithm} rc=${rc}"
  } > "${log_file}" 2>&1

  if [[ "${rc}" == "0" ]]; then
    write_status "DONE" "${city}" "${algorithm}" "${gpu}" "${BASHPID}" "${rc}" "${out_dir}"
  else
    write_status "FAIL" "${city}" "${algorithm}" "${gpu}" "${BASHPID}" "${rc}" "${out_dir}"
  fi
  return "${rc}"
}

{
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
  echo "seed=${SEED}"
  echo "sumo_time_to_teleport=-1"
  echo "fontenay_sumo_extra_args=${FONT_BLOCKER60_SUMO_EXTRA}"
  echo "ppo_gradient_coverage=all_actual_samples_once_per_epoch"
  echo "iql_update_semantics=persistent_transition_replay_without_replacement"
} > "${MANIFEST}"

printf "time\tstate\tcity\talgorithm\tgpu\tpid\trc\tout_dir\n" > "${STATUS_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] launching ${#TASKS[@]} jobs in parallel" > "${MASTER_LOG}"
echo "stamp=${STAMP}"
echo "log_dir=${LOG_DIR}"
echo "status_file=${STATUS_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  for idx in "${!TASKS[@]}"; do
    IFS='|' read -r city algorithm entrypoint tag settings post_state hdv_state max_steps sumo_extra <<< "${TASKS[$idx]}"
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    memory_size="-"
    if [[ "${algorithm}" == "iql" ]]; then
      memory_size="${IQL_MEMORY_SIZE}"
    fi
    printf "%s\t%s\tDEVICE=%s\tSLOT=%s\tTELEPORT=-1\tMEMORY_SIZE=%s\tSUMO_EXTRA_ARGS=%q\tresults/%s/baselines_torchrl_corrected_%sep_%s/%s\n" \
      "${city}" "${algorithm}" "${TRAIN_DEVICE}" "${gpu}" "${memory_size}" "${sumo_extra}" "${tag}" \
      "${TRAIN_EPISODES}" "${STAMP}" "${algorithm}"
  done
  exit 0
fi

PIDS=()
NAMES=()
terminate_children() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] received termination signal" >> "${MASTER_LOG}"
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap terminate_children INT TERM

for idx in "${!TASKS[@]}"; do
  IFS='|' read -r city algorithm entrypoint tag settings post_state hdv_state max_steps sumo_extra <<< "${TASKS[$idx]}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  run_one "${city}" "${algorithm}" "${entrypoint}" "${tag}" "${settings}" \
    "${post_state}" "${hdv_state}" "${max_steps}" "${sumo_extra}" "${gpu}" &
  PIDS+=("$!")
  NAMES+=("${city}/${algorithm}")
done

failures=0
for idx in "${!PIDS[@]}"; do
  if wait "${PIDS[$idx]}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] DONE ${NAMES[$idx]}" >> "${MASTER_LOG}"
  else
    rc=$?
    failures=$((failures + 1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] FAIL ${NAMES[$idx]} rc=${rc}" >> "${MASTER_LOG}"
  fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] all jobs finished failures=${failures}" >> "${MASTER_LOG}"
exit "${failures}"
