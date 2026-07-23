#!/usr/bin/env bash
set -uo pipefail

# Evaluate the latest MAPPO/IQL checkpoints for all three cities under the
# shared long-horizon protocol.  The six evaluations run concurrently on CPU.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/wangwenrui/opt/miniconda3/envs/URB/bin/python}"
TRAIN_RUN_ID="${TRAIN_RUN_ID:-}"
EVAL_RUN_ID="${EVAL_RUN_ID:-${TRAIN_RUN_ID}}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-3}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
MAPPO_CRITIC_NETWORK_NUM_CELLS="${MAPPO_CRITIC_NETWORK_NUM_CELLS:-128}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_FOR_TRAINING="${WAIT_FOR_TRAINING:-0}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"

if [[ -z "${TRAIN_RUN_ID}" ]]; then
  echo "TRAIN_RUN_ID is required." >&2
  exit 2
fi
if [[ ! "${CPU_THREADS_PER_JOB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPU_THREADS_PER_JOB must be a positive integer." >&2
  exit 2
fi
if [[ ! "${EVAL_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_EPISODES must be a positive integer." >&2
  exit 2
fi
if [[ ! "${MAPPO_CRITIC_NETWORK_NUM_CELLS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAPPO_CRITIC_NETWORK_NUM_CELLS must be a positive integer." >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1." >&2
  exit 2
fi
if [[ "${WAIT_FOR_TRAINING}" != "0" && "${WAIT_FOR_TRAINING}" != "1" ]]; then
  echo "WAIT_FOR_TRAINING must be 0 or 1." >&2
  exit 2
fi
if [[ ! "${WAIT_POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WAIT_POLL_SECONDS must be a positive integer." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi

TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${ROOT}/tmp/logs/gretz_fontenay_latest_mappo_iql_${TRAIN_RUN_ID}}"
LOG_DIR="${EVAL_LOG_DIR:-${ROOT}/tmp/logs/latest_mappo_iql_unified_eval_${EVAL_RUN_ID}}"
STATUS_FILE="${LOG_DIR}/status.tsv"
COMMANDS_FILE="${LOG_DIR}/commands.sh"
MANIFEST_FILE="${LOG_DIR}/manifest.txt"
MASTER_LOG="${LOG_DIR}/master.log"
MPLCONFIG_ROOT="${LOG_DIR}/matplotlib"
AUDIT_JSON="${LOG_DIR}/audit.json"
EVAL_TAG="eval_torchrl_latest_mappo_iql_native_long9000_${EVAL_EPISODES}ep_${EVAL_RUN_ID}"

GRETZ_TAG="gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1"
GRETZ_ROOT="${ROOT}/results/${GRETZ_TAG}"
GRETZ_RECORDS="${ROOT}/results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/collect_urb_fair_eval_postmutation_records"
GRETZ_SETTINGS="${TRAIN_LOG_DIR}/settings/gretz.env"

INGOLSTADT_TAG="ingolstadt_custom_actionfix_incident_20260516_185810"
INGOLSTADT_ROOT="${ROOT}/results/${INGOLSTADT_TAG}"
INGOLSTADT_RECORDS="${INGOLSTADT_ROOT}/collect_urb_fair_eval_postmutation_records"
INGOLSTADT_SETTINGS="${ROOT}/.urbandreamer/generated/${INGOLSTADT_TAG}/urb_fair_eval.env"

FONTENAY_TAG="fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609"
FONTENAY_ROOT="${ROOT}/results/${FONTENAY_TAG}"
FONTENAY_RECORDS="${FONTENAY_ROOT}/collect_urb_fair_eval_postmutation_records"
FONTENAY_SETTINGS="${TRAIN_LOG_DIR}/settings/fontenay.env"

MAPPO_SOURCE="${ROOT}/baseline/run_toy_cross_bottleneck_mappo_torchrl.py"
IQL_SOURCE="${ROOT}/baseline/run_toy_cross_bottleneck_iql_torchrl.py"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S UTC'
}

file_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Missing or empty required file: $1" >&2
    return 1
  fi
}

if [[ -e "${LOG_DIR}" ]]; then
  echo "Refusing existing evaluation log directory: ${LOG_DIR}" >&2
  exit 2
fi

for path in \
  "${TRAIN_LOG_DIR}/manifest.txt" \
  "${TRAIN_LOG_DIR}/master.log" \
  "${GRETZ_SETTINGS}" \
  "${FONTENAY_SETTINGS}" \
  "${INGOLSTADT_SETTINGS}" \
  "${MAPPO_SOURCE}" \
  "${IQL_SOURCE}" \
  "${GRETZ_RECORDS}/post_mutation_state.json" \
  "${GRETZ_RECORDS}/post_mutation_hdv_learning_state.json" \
  "${INGOLSTADT_RECORDS}/post_mutation_state.json" \
  "${INGOLSTADT_RECORDS}/post_mutation_hdv_learning_state.json" \
  "${FONTENAY_RECORDS}/post_mutation_state.json" \
  "${FONTENAY_RECORDS}/post_mutation_hdv_learning_state.json"; do
  require_file "${path}" || exit 2
done

if [[ "${WAIT_FOR_TRAINING}" == "1" ]]; then
  printf '[%s] waiting for training run %s to finish\n' "$(timestamp)" "${TRAIN_RUN_ID}"
  while ! rg -q 'all jobs finished failures=0' "${TRAIN_LOG_DIR}/master.log"; do
    if rg -q 'all jobs finished failures=[1-9][0-9]*' "${TRAIN_LOG_DIR}/master.log"; then
      echo "Training reached a failed terminal state: ${TRAIN_LOG_DIR}/master.log" >&2
      exit 3
    fi
    sleep "${WAIT_POLL_SECONDS}"
  done
  printf '[%s] training finished successfully; starting unified evaluation\n' "$(timestamp)"
fi

if ! rg -q 'all jobs finished failures=0' "${TRAIN_LOG_DIR}/master.log"; then
  echo "Training has not reached a zero-failure terminal state: ${TRAIN_LOG_DIR}/master.log" >&2
  exit 3
fi

expected_mappo_hash="$(sed -n 's/^sha256_mappo_source=//p' "${TRAIN_LOG_DIR}/manifest.txt" | head -n 1)"
expected_iql_hash="$(sed -n 's/^sha256_iql_source=//p' "${TRAIN_LOG_DIR}/manifest.txt" | head -n 1)"
actual_mappo_hash="$(file_sha256 "${MAPPO_SOURCE}")"
actual_iql_hash="$(file_sha256 "${IQL_SOURCE}")"
if [[ -z "${expected_mappo_hash}" || "${actual_mappo_hash}" != "${expected_mappo_hash}" ]]; then
  echo "MAPPO source changed after training; refusing incompatible evaluation." >&2
  exit 4
fi
if [[ -z "${expected_iql_hash}" || "${actual_iql_hash}" != "${expected_iql_hash}" ]]; then
  echo "IQL source changed after training; refusing incompatible evaluation." >&2
  exit 4
fi

gretz_mappo_checkpoint="${GRETZ_ROOT}/baselines_torchrl_latest_jointframe_smdp_500ep_${TRAIN_RUN_ID}/mappo/checkpoints/checkpoint_ep00500.pt"
gretz_iql_checkpoint="${GRETZ_ROOT}/baselines_torchrl_latest_jointframe_smdp_500ep_${TRAIN_RUN_ID}/iql/checkpoints/checkpoint_ep00500.pt"
ingolstadt_mappo_checkpoint="${INGOLSTADT_ROOT}/baselines_torchrl_dynamic60_adapted_500ep_20260718_175330/mappo/checkpoints/checkpoint_ep00500.pt"
ingolstadt_iql_checkpoint="${INGOLSTADT_ROOT}/baselines_torchrl_iql_smdp_firsteps_500ep_20260719_160524/iql/checkpoints/checkpoint_ep00500.pt"
fontenay_mappo_checkpoint="${FONTENAY_ROOT}/baselines_torchrl_latest_jointframe_smdp_200ep_${TRAIN_RUN_ID}/mappo/checkpoints/checkpoint_ep00200.pt"
fontenay_iql_checkpoint="${FONTENAY_ROOT}/baselines_torchrl_latest_jointframe_smdp_200ep_${TRAIN_RUN_ID}/iql/checkpoints/checkpoint_ep00200.pt"

for checkpoint in \
  "${gretz_mappo_checkpoint}" \
  "${gretz_iql_checkpoint}" \
  "${ingolstadt_mappo_checkpoint}" \
  "${ingolstadt_iql_checkpoint}" \
  "${fontenay_mappo_checkpoint}" \
  "${fontenay_iql_checkpoint}"; do
  require_file "${checkpoint}" || exit 3
done

mkdir -p "${LOG_DIR}" "${MPLCONFIG_ROOT}"
printf 'time\tstate\tcity\talgorithm\tworker_pid\tchild_pid\trc\tout_dir\n' >"${STATUS_FILE}"
: >"${COMMANDS_FILE}"
{
  printf 'repo_root=%s\n' "${ROOT}"
  printf 'train_run_id=%s\n' "${TRAIN_RUN_ID}"
  printf 'eval_run_id=%s\n' "${EVAL_RUN_ID}"
  printf 'eval_tag=%s\n' "${EVAL_TAG}"
  printf 'eval_episodes=%s\n' "${EVAL_EPISODES}"
  printf 'parallel_processes=6\n'
  printf 'cpu_threads_per_job=%s\n' "${CPU_THREADS_PER_JOB}"
  printf 'protocol=max_steps_per_episode=9000,sim_buffer_after_last_depart=7200,sumo_time_to_teleport=-1,hdv_learning_frozen=true\n'
  printf 'mappo_executor=masked_categorical\n'
  printf 'mappo_critic_network_num_cells=%s\n' "${MAPPO_CRITIC_NETWORK_NUM_CELLS}"
  printf 'iql_executor=valid_mask_epsilon_greedy,epsilon=0.05,epsilon_scope=first_decision_per_vehicle\n'
  printf 'fontenay_unified_protocol=collision_action_warn_without_ignore_junction_blocker\n'
  printf 'sha256_mappo_source=%s\n' "${actual_mappo_hash}"
  printf 'sha256_iql_source=%s\n' "${actual_iql_hash}"
} >"${MANIFEST_FILE}"

export SUMO_HOME="${SUMO_HOME:-$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)}"
export PATH="${SUMO_HOME}/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${WANDB_MODE:-offline}"

status() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(timestamp)" "$1" "$2" "$3" "$4" "$5" "$6" "$7" >>"${STATUS_FILE}"
}

validate_eval() {
  local out_dir="$1" algorithm="$2" first_episode="$3"
  "${PYTHON_BIN}" - "${out_dir}" "${algorithm}" "${first_episode}" "${EVAL_EPISODES}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
algorithm = sys.argv[2]
first = int(sys.argv[3])
count = int(sys.argv[4])
episodes_path = out_dir / "dynamic_episodes.csv"
summary_path = out_dir / f"toy_{algorithm}_torchrl_eval_summary.json"
if not episodes_path.is_file() or not summary_path.is_file():
    raise SystemExit(1)
with episodes_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
actual = [int(float(row["episode"])) for row in rows]
if actual != list(range(first, first + count)):
    raise SystemExit(1)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if int(summary.get("eval_episodes", 0)) != count:
    raise SystemExit(1)
if summary.get("eval_policy_route_selection") != "sample":
    raise SystemExit(1)
if summary.get("hdv_updates_disabled") is not True:
    raise SystemExit(1)
expected_executor = "valid_mask_epsilon_greedy" if algorithm == "iql" else "masked_categorical"
if summary.get("eval_policy_executor") != expected_executor:
    raise SystemExit(1)
if algorithm == "iql" and not math.isclose(float(summary.get("eval_epsilon", -1)), 0.05, abs_tol=1e-12):
    raise SystemExit(1)
PY
}

run_eval() {
  local city="$1" algorithm="$2" settings="$3" records="$4" city_root="$5" checkpoint="$6" resume_start="$7" sumo_extra="$8"
  local method_root="${city_root}/${EVAL_TAG}/${algorithm}"
  local out_dir="${method_root}/attempt_001"
  local child_pid="" rc=0 started=0
  local -a command algorithm_args sumo_args

  cancel_one() {
    local signal_rc="$1"
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
  trap 'cancel_one 129' HUP
  trap 'cancel_one 130' INT
  trap 'cancel_one 143' TERM

  if [[ -e "${out_dir}" ]]; then
    echo "Refusing existing evaluation output: ${out_dir}" >&2
    status REFUSED "${city}" "${algorithm}" "${BASHPID}" - 4 "${out_dir}"
    return 4
  fi

  algorithm_args=()
  if [[ "${algorithm}" == "iql" ]]; then
    algorithm_args=(--epsilon-scope first_decision_per_vehicle)
  else
    algorithm_args=(--critic-network-num-cells "${MAPPO_CRITIC_NETWORK_NUM_CELLS}")
  fi
  sumo_args=()
  if [[ -n "${sumo_extra}" ]]; then
    sumo_args=(--sumo-extra-args "${sumo_extra}")
  fi
  command=(
    "${PYTHON_BIN}" -u "${ROOT}/baseline/run_toy_cross_bottleneck_${algorithm}_torchrl.py"
    --settings "${settings}"
    --output-dir "${out_dir}"
    --post-mutation-state-in "${records}/post_mutation_state.json"
    --hdv-state-in "${records}/post_mutation_hdv_learning_state.json"
    --resume-checkpoint "${checkpoint}"
    --resume-start-episode "${resume_start}"
    --eval-only
    --eval-policy-route-selection training
    --eval-episodes "${EVAL_EPISODES}"
    --agent-frames-per-batch 10
    --device cpu
    --seed 42
    --max-steps-per-episode 9000
    --eval-sim-buffer-after-last-depart 7200
    --sumo-time-to-teleport -1
    "${algorithm_args[@]}"
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
  printf 'running\n' >"${out_dir}/STATUS"
  status START "${city}" "${algorithm}" "${BASHPID}" - - "${out_dir}"
  started=1
  CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
    MKL_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
    OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
    MPLCONFIGDIR="${MPLCONFIG_ROOT}/${city}_${algorithm}" \
    setsid "${command[@]}" >"${out_dir}/eval.log" 2>&1 &
  child_pid=$!
  status RUNNING "${city}" "${algorithm}" "${BASHPID}" "${child_pid}" - "${out_dir}"
  wait "${child_pid}" || rc=$?
  if [[ "${rc}" -eq 0 ]] && ! validate_eval "${out_dir}" "${algorithm}" "$((resume_start + 1))"; then
    rc=5
  fi
  if [[ "${rc}" -eq 0 ]]; then
    printf 'complete\n' >"${out_dir}/STATUS"
    status DONE "${city}" "${algorithm}" "${BASHPID}" "${child_pid}" 0 "${out_dir}"
  else
    printf 'failed rc=%s\n' "${rc}" >"${out_dir}/STATUS"
    status FAIL "${city}" "${algorithm}" "${BASHPID}" "${child_pid}" "${rc}" "${out_dir}"
  fi
  return "${rc}"
}

# city|algorithm|settings|records|root|checkpoint|resume_start|sumo_extra
TASKS=(
  "gretz|mappo|${GRETZ_SETTINGS}|${GRETZ_RECORDS}|${GRETZ_ROOT}|${gretz_mappo_checkpoint}|200|--collision.action warn"
  "gretz|iql|${GRETZ_SETTINGS}|${GRETZ_RECORDS}|${GRETZ_ROOT}|${gretz_iql_checkpoint}|200|--collision.action warn"
  "ingolstadt|mappo|${INGOLSTADT_SETTINGS}|${INGOLSTADT_RECORDS}|${INGOLSTADT_ROOT}|${ingolstadt_mappo_checkpoint}|200|"
  "ingolstadt|iql|${INGOLSTADT_SETTINGS}|${INGOLSTADT_RECORDS}|${INGOLSTADT_ROOT}|${ingolstadt_iql_checkpoint}|200|"
  "fontenay|mappo|${FONTENAY_SETTINGS}|${FONTENAY_RECORDS}|${FONTENAY_ROOT}|${fontenay_mappo_checkpoint}|50|--collision.action warn"
  "fontenay|iql|${FONTENAY_SETTINGS}|${FONTENAY_RECORDS}|${FONTENAY_ROOT}|${fontenay_iql_checkpoint}|50|--collision.action warn"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  for task in "${TASKS[@]}"; do
    IFS='|' read -r city algorithm settings records city_root checkpoint resume_start sumo_extra <<<"${task}"
    run_eval "${city}" "${algorithm}" "${settings}" "${records}" "${city_root}" "${checkpoint}" "${resume_start}" "${sumo_extra}" || exit $?
  done
  echo "DRY_RUN complete: six evaluation commands validated."
  exit 0
fi

declare -a worker_pids=()
declare -A worker_names=()

cleanup_parent() {
  local signal_rc="$1" pid
  trap - HUP INT TERM
  for pid in "${worker_pids[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "${worker_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  exit "${signal_rc}"
}
trap 'cleanup_parent 129' HUP
trap 'cleanup_parent 130' INT
trap 'cleanup_parent 143' TERM

printf '[%s] launching six parallel unified evaluations\n' "$(timestamp)" >"${MASTER_LOG}"
for task in "${TASKS[@]}"; do
  IFS='|' read -r city algorithm settings records city_root checkpoint resume_start sumo_extra <<<"${task}"
  run_eval "${city}" "${algorithm}" "${settings}" "${records}" "${city_root}" "${checkpoint}" "${resume_start}" "${sumo_extra}" &
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
printf '[%s] all evaluations finished failures=%s\n' "$(timestamp)" "${failures}" >>"${MASTER_LOG}"
if [[ "${failures}" -ne 0 ]]; then
  exit 1
fi

"${PYTHON_BIN}" "${ROOT}/tmp/audit_torchrl_native_eval_20260716.py" \
  --protocol unified \
  --cities gretz,ingolstadt,fontenay \
  --algorithms mappo,iql \
  --iql-epsilon-scope first_decision_per_vehicle \
  --unified-tag "${EVAL_TAG}" \
  --output "${AUDIT_JSON}"

"${PYTHON_BIN}" "${ROOT}/tmp/update_completed_native_eval_reports_20260717.py" \
  --audit-path "${AUDIT_JSON}" \
  --cities gretz,ingolstadt,fontenay \
  --methods mappo,iql

printf '[%s] audit and main-table update complete\n' "$(timestamp)" >>"${MASTER_LOG}"
