#!/usr/bin/env bash
set -uo pipefail

# Train the latest joint-frame MAPPO and SMDP IQL baselines concurrently on
# Gretz-Armainvilliers and Fontenay-en-Parisis.  The default CPU execution is
# intentional: the shared GPUs may be occupied, while SUMO collection remains
# mostly CPU-bound.  Override TRAIN_DEVICE only after checking GPU ownership.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/wangwenrui/opt/miniconda3/envs/URB/bin/python}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cpu}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
GRETZ_EPISODES="${GRETZ_EPISODES:-500}"
FONTENAY_EPISODES="${FONTENAY_EPISODES:-200}"
GRETZ_IQL_MEMORY_SIZE="${GRETZ_IQL_MEMORY_SIZE:-100000}"
FONTENAY_IQL_MEMORY_SIZE="${FONTENAY_IQL_MEMORY_SIZE:-500000}"
IQL_SIM_BUFFER_AFTER_LAST_DEPART="${IQL_SIM_BUFFER_AFTER_LAST_DEPART:-3000}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"

LOG_DIR="${RUN_LOG_DIR:-${ROOT}/tmp/logs/gretz_fontenay_latest_mappo_iql_${RUN_ID}}"
STATUS_FILE="${LOG_DIR}/status.tsv"
COMMANDS_FILE="${LOG_DIR}/commands.sh"
MANIFEST_FILE="${LOG_DIR}/manifest.txt"
MASTER_LOG="${LOG_DIR}/master.log"
SETTINGS_DIR="${LOG_DIR}/settings"
MPLCONFIG_ROOT="${LOG_DIR}/matplotlib"

GRETZ_TAG="gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1"
GRETZ_ROOT="${ROOT}/results/${GRETZ_TAG}"
GRETZ_SETTINGS_SOURCE="${ROOT}/.urbandreamer/generated/${GRETZ_TAG}/urb_fair_eval.env"
GRETZ_RECORDS="${ROOT}/results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/collect_urb_fair_eval_postmutation_records"

FONTENAY_TAG="fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609"
FONTENAY_ROOT="${ROOT}/results/${FONTENAY_TAG}"
FONTENAY_SETTINGS_SOURCE="${ROOT}/.urbandreamer/generated/${FONTENAY_TAG}/urb_fair_eval.env"
FONTENAY_RECORDS="${FONTENAY_ROOT}/collect_urb_fair_eval_postmutation_records"

MAPPO_SOURCE="${ROOT}/baseline/run_toy_cross_bottleneck_mappo_torchrl.py"
IQL_SOURCE="${ROOT}/baseline/run_toy_cross_bottleneck_iql_torchrl.py"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S UTC'
}

require_positive_integer() {
  local name="$1" value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer; got %s\n' "${name}" "${value}" >&2
    return 1
  fi
}

file_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    printf 'Missing or empty required file: %s\n' "${path}" >&2
    return 1
  fi
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python executable is unavailable: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi
if [[ "${TRAIN_DEVICE}" != "cpu" && "${TRAIN_DEVICE}" != cuda* ]]; then
  printf 'TRAIN_DEVICE must be cpu or a cuda device; got %s\n' "${TRAIN_DEVICE}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  printf 'DRY_RUN must be 0 or 1; got %s\n' "${DRY_RUN}" >&2
  exit 2
fi
if [[ "${ALLOW_EXISTING_OUTPUT}" != "0" && "${ALLOW_EXISTING_OUTPUT}" != "1" ]]; then
  printf 'ALLOW_EXISTING_OUTPUT must be 0 or 1; got %s\n' "${ALLOW_EXISTING_OUTPUT}" >&2
  exit 2
fi
for pair in \
  "CPU_THREADS_PER_JOB:${CPU_THREADS_PER_JOB}" \
  "GRETZ_EPISODES:${GRETZ_EPISODES}" \
  "FONTENAY_EPISODES:${FONTENAY_EPISODES}" \
  "GRETZ_IQL_MEMORY_SIZE:${GRETZ_IQL_MEMORY_SIZE}" \
  "FONTENAY_IQL_MEMORY_SIZE:${FONTENAY_IQL_MEMORY_SIZE}" \
  "IQL_SIM_BUFFER_AFTER_LAST_DEPART:${IQL_SIM_BUFFER_AFTER_LAST_DEPART}"; do
  IFS=: read -r name value <<<"${pair}"
  require_positive_integer "${name}" "${value}" || exit 2
done

for path in \
  "${MAPPO_SOURCE}" \
  "${IQL_SOURCE}" \
  "${GRETZ_SETTINGS_SOURCE}" \
  "${GRETZ_RECORDS}/post_mutation_state.json" \
  "${GRETZ_RECORDS}/post_mutation_hdv_learning_state.json" \
  "${FONTENAY_SETTINGS_SOURCE}" \
  "${FONTENAY_RECORDS}/post_mutation_state.json" \
  "${FONTENAY_RECORDS}/post_mutation_hdv_learning_state.json"; do
  require_file "${path}" || exit 2
done

if [[ -e "${LOG_DIR}" && "${ALLOW_EXISTING_OUTPUT}" != "1" ]]; then
  printf 'Refusing existing log directory: %s\n' "${LOG_DIR}" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}" "${SETTINGS_DIR}" "${MPLCONFIG_ROOT}"

# Keep an immutable copy of the settings used by all four processes.  The
# generated settings currently point at this workspace, but rebasing makes the
# launcher portable if the workspace is relocated before a resumed run.
rebase_settings() {
  local source="$1" destination="$2" old_root
  old_root="$(sed -n 's|^NETWORK_DIR="\(.*\)/networks/.*"$|\1|p' "${source}" | head -n 1)"
  if [[ -z "${old_root}" ]]; then
    printf 'Cannot infer generated-settings root from %s\n' "${source}" >&2
    return 1
  fi
  sed "s|${old_root}|${ROOT}|g" "${source}" >"${destination}"
}

GRETZ_SETTINGS="${SETTINGS_DIR}/gretz.env"
FONTENAY_SETTINGS="${SETTINGS_DIR}/fontenay.env"
rebase_settings "${GRETZ_SETTINGS_SOURCE}" "${GRETZ_SETTINGS}" || exit 2
rebase_settings "${FONTENAY_SETTINGS_SOURCE}" "${FONTENAY_SETTINGS}" || exit 2

export SUMO_HOME="${SUMO_HOME:-$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)}"
export PATH="${SUMO_HOME}/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

printf 'time\tstate\tcity\talgorithm\tdevice\tworker_pid\tchild_pid\trc\tout_dir\n' >"${STATUS_FILE}"
: >"${COMMANDS_FILE}"

{
  printf 'repo_root=%s\n' "${ROOT}"
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'train_device=%s\n' "${TRAIN_DEVICE}"
  printf 'cpu_threads_per_job=%s\n' "${CPU_THREADS_PER_JOB}"
  printf 'parallel_processes=4\n'
  printf 'seed=%s\n' "${SEED}"
  printf 'gretz_episodes=%s\n' "${GRETZ_EPISODES}"
  printf 'fontenay_episodes=%s\n' "${FONTENAY_EPISODES}"
  printf 'gretz_iql_memory_size=%s\n' "${GRETZ_IQL_MEMORY_SIZE}"
  printf 'fontenay_iql_memory_size=%s\n' "${FONTENAY_IQL_MEMORY_SIZE}"
  printf 'iql_sim_buffer_after_last_depart=%s\n' "${IQL_SIM_BUFFER_AFTER_LAST_DEPART}"
  printf 'mappo_semantics=joint_frame,batch_episodes=1,epochs=5,minibatch_frames=64,lr=3e-4,entropy=1e-2,normalized_advantage=true,critic_cells=128\n'
  printf 'iql_semantics=smdp_arrival_tail,target_update_per_optimizer_step,fresh_full_coverage,replay_ratio=0.25,batch_episodes=1,epsilon_scope=first_decision_per_vehicle\n'
  printf 'fontenay_training_protocol=max_steps_per_episode=3600,ignore_junction_blocker=60,collision_action=warn,write_unfinished=true\n'
  printf 'sha256_mappo_source=%s\n' "$(file_sha256 "${MAPPO_SOURCE}")"
  printf 'sha256_iql_source=%s\n' "$(file_sha256 "${IQL_SOURCE}")"
  printf 'sha256_gretz_settings=%s\n' "$(file_sha256 "${GRETZ_SETTINGS}")"
  printf 'sha256_fontenay_settings=%s\n' "$(file_sha256 "${FONTENAY_SETTINGS}")"
  printf 'sha256_gretz_post_state=%s\n' "$(file_sha256 "${GRETZ_RECORDS}/post_mutation_state.json")"
  printf 'sha256_gretz_hdv_state=%s\n' "$(file_sha256 "${GRETZ_RECORDS}/post_mutation_hdv_learning_state.json")"
  printf 'sha256_fontenay_post_state=%s\n' "$(file_sha256 "${FONTENAY_RECORDS}/post_mutation_state.json")"
  printf 'sha256_fontenay_hdv_state=%s\n' "$(file_sha256 "${FONTENAY_RECORDS}/post_mutation_hdv_learning_state.json")"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
  printf 'git_status_begin\n'
  git status --short 2>/dev/null || true
  printf 'git_status_end\n'
} >"${MANIFEST_FILE}"

status() {
  local state="$1" city="$2" algorithm="$3" worker_pid="$4" child_pid="$5" rc="$6" out_dir="$7"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(timestamp)" "${state}" "${city}" "${algorithm}" "${TRAIN_DEVICE}" \
    "${worker_pid}" "${child_pid}" "${rc}" "${out_dir}" >>"${STATUS_FILE}"
}

checkpoint_episodes() {
  local episodes="$1"
  if ((episodes >= 500)); then
    printf '20,50,100,200,300,400,500'
  elif ((episodes >= 200)); then
    printf '20,50,100,200'
  elif ((episodes >= 100)); then
    printf '20,50,100'
  elif ((episodes >= 50)); then
    printf '20,50'
  elif ((episodes >= 20)); then
    printf '20'
  else
    printf '%s' "${episodes}"
  fi
}

validate_output() {
  local out_dir="$1" algorithm="$2" episodes="$3"
  "${PYTHON_BIN}" - "${out_dir}" "${algorithm}" "${episodes}" <<'PY'
import csv
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
algorithm = sys.argv[2]
episodes = int(sys.argv[3])
csv_path = out_dir / "online_train_episodes.csv"
checkpoint = out_dir / "checkpoints" / f"checkpoint_ep{episodes:05d}.pt"
summary = out_dir / f"toy_{algorithm}_torchrl_final_summary.json"
if not csv_path.is_file() or not checkpoint.is_file() or not summary.is_file():
    raise SystemExit(1)
with csv_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
actual = [int(float(row["episode"])) for row in rows]
if actual != list(range(1, episodes + 1)):
    raise SystemExit(1)
PY
}

run_one() {
  local city="$1" algorithm="$2"
  local episodes settings records city_root memory_size max_steps checkpoint_list output_stem out_dir log_file
  local child_pid="" rc=0 started=0 device_arg
  local -a command algorithm_args sumo_args max_steps_args

  cancel_one() {
    local signal_name="$1" signal_rc="$2"
    trap - HUP INT TERM
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
      kill -TERM -- "-${child_pid}" 2>/dev/null || kill -TERM "${child_pid}" 2>/dev/null || true
      wait "${child_pid}" 2>/dev/null || true
    fi
    if [[ "${started}" == "1" ]]; then
      status CANCELLED "${city}" "${algorithm}" "${BASHPID}" "${child_pid:--}" "${signal_rc}" "${out_dir}"
    fi
    exit "${signal_rc}"
  }
  trap 'cancel_one HUP 129' HUP
  trap 'cancel_one INT 130' INT
  trap 'cancel_one TERM 143' TERM

  case "${city}" in
    gretz)
      episodes="${GRETZ_EPISODES}"
      settings="${GRETZ_SETTINGS}"
      records="${GRETZ_RECORDS}"
      city_root="${GRETZ_ROOT}"
      memory_size="${GRETZ_IQL_MEMORY_SIZE}"
      max_steps=0
      sumo_args=(--sumo-extra-args "--collision.action warn")
      ;;
    fontenay)
      episodes="${FONTENAY_EPISODES}"
      settings="${FONTENAY_SETTINGS}"
      records="${FONTENAY_RECORDS}"
      city_root="${FONTENAY_ROOT}"
      memory_size="${FONTENAY_IQL_MEMORY_SIZE}"
      max_steps=3600
      sumo_args=(--sumo-extra-args "--collision.action warn --ignore-junction-blocker 60 --tripinfo-output.write-unfinished")
      ;;
    *)
      printf 'Unknown city: %s\n' "${city}" >&2
      return 2
      ;;
  esac

  case "${algorithm}" in
    mappo)
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
      printf 'Unknown algorithm: %s\n' "${algorithm}" >&2
      return 2
      ;;
  esac

  checkpoint_list="$(checkpoint_episodes "${episodes}")"
  output_stem="baselines_torchrl_latest_jointframe_smdp_${episodes}ep_${RUN_ID}"
  out_dir="${city_root}/${output_stem}/${algorithm}"
  log_file="${LOG_DIR}/${city}_${algorithm}.log"
  if [[ -e "${out_dir}" && "${ALLOW_EXISTING_OUTPUT}" != "1" ]]; then
    printf 'Refusing existing output: %s\n' "${out_dir}" >&2
    status REFUSED "${city}" "${algorithm}" "${BASHPID}" - 4 "${out_dir}"
    return 4
  fi

  if [[ "${TRAIN_DEVICE}" == cuda* ]]; then
    device_arg="${TRAIN_DEVICE}"
  else
    device_arg=cpu
  fi
  max_steps_args=()
  if ((max_steps > 0)); then
    max_steps_args=(--max-steps-per-episode "${max_steps}")
  fi
  command=(
    "${PYTHON_BIN}" -u "${ROOT}/baseline/run_toy_cross_bottleneck_${algorithm}_torchrl.py"
    --settings "${settings}"
    --output-dir "${out_dir}"
    --post-mutation-state-in "${records}/post_mutation_state.json"
    --hdv-state-in "${records}/post_mutation_hdv_learning_state.json"
    --train-episodes "${episodes}"
    --checkpoint-episodes "${checkpoint_list}"
    --sumo-time-to-teleport -1
    --device "${device_arg}"
    --seed "${SEED}"
    "${algorithm_args[@]}"
    "${max_steps_args[@]}"
    "${sumo_args[@]}"
  )

  {
    printf '# %s city=%s algorithm=%s\n' "$(timestamp)" "${city}" "${algorithm}"
    printf 'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q MPLCONFIGDIR=%q ' \
      "${CPU_THREADS_PER_JOB}" "${CPU_THREADS_PER_JOB}" "${CPU_THREADS_PER_JOB}" "${MPLCONFIG_ROOT}/${city}_${algorithm}"
    printf '%q ' "${command[@]}"
    printf '\n'
  } >>"${COMMANDS_FILE}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN %s/%s -> %s\n' "${city}" "${algorithm}" "${out_dir}"
    return 0
  fi

  mkdir -p "${out_dir}" "${MPLCONFIG_ROOT}/${city}_${algorithm}"
  status START "${city}" "${algorithm}" "${BASHPID}" - - "${out_dir}"
  started=1
  CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
    MKL_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
    OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
    MPLCONFIGDIR="${MPLCONFIG_ROOT}/${city}_${algorithm}" \
    setsid "${command[@]}" >"${log_file}" 2>&1 &
  child_pid=$!
  status RUNNING "${city}" "${algorithm}" "${BASHPID}" "${child_pid}" - "${out_dir}"

  wait "${child_pid}" || rc=$?
  if [[ "${rc}" -eq 0 ]] && ! validate_output "${out_dir}" "${algorithm}" "${episodes}"; then
    rc=5
  fi
  if [[ "${rc}" -eq 0 ]]; then
    status DONE "${city}" "${algorithm}" "${BASHPID}" "${child_pid}" 0 "${out_dir}"
  else
    status FAIL "${city}" "${algorithm}" "${BASHPID}" "${child_pid}" "${rc}" "${out_dir}"
  fi
  return "${rc}"
}

TASKS=(
  "gretz mappo"
  "gretz iql"
  "fontenay mappo"
  "fontenay iql"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  for task in "${TASKS[@]}"; do
    read -r city algorithm <<<"${task}"
    run_one "${city}" "${algorithm}" || exit $?
  done
  printf 'DRY_RUN complete: four commands validated.\n'
  exit 0
fi

declare -a worker_pids=()
declare -A worker_names=()
parent_cleanup_done=0

cleanup_parent() {
  local signal_name="$1" signal_rc="$2" pid
  if [[ "${parent_cleanup_done}" == "1" ]]; then
    return
  fi
  parent_cleanup_done=1
  trap - HUP INT TERM
  for pid in "${worker_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${worker_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  printf '[%s] launcher cancelled signal=%s\n' "$(timestamp)" "${signal_name}" >>"${MASTER_LOG}"
  exit "${signal_rc}"
}
trap 'cleanup_parent HUP 129' HUP
trap 'cleanup_parent INT 130' INT
trap 'cleanup_parent TERM 143' TERM

printf '[%s] launching four parallel jobs\n' "$(timestamp)" >"${MASTER_LOG}"
for task in "${TASKS[@]}"; do
  read -r city algorithm <<<"${task}"
  run_one "${city}" "${algorithm}" &
  worker_pids+=("$!")
  worker_names["$!"]="${city}/${algorithm}"
done

failures=0
for pid in "${worker_pids[@]}"; do
  rc=0
  wait "${pid}" || rc=$?
  printf '[%s] worker=%s rc=%s\n' "$(timestamp)" "${worker_names[${pid}]}" "${rc}" >>"${MASTER_LOG}"
  if [[ "${rc}" -ne 0 ]]; then
    failures=$((failures + 1))
  fi
done

printf '[%s] all jobs finished failures=%s\n' "$(timestamp)" "${failures}" >>"${MASTER_LOG}"
if [[ "${failures}" -ne 0 ]]; then
  exit 1
fi
