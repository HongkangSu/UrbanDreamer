#!/usr/bin/env bash
set -Eeuo pipefail

# Run the corrected dynamic-routing IQL protocol on Ingolstadt only.  The
# launcher owns a singleton lock and keeps the Python process in its own
# process group so an interrupted launcher cannot leave SUMO children behind.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/wangwenrui/opt/miniconda3/envs/URB/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_EPISODES="${TRAIN_EPISODES:-500}"
CHECKPOINT_EPISODES="${CHECKPOINT_EPISODES:-1,5,20,50,100,200,300,400,500}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
SIM_BUFFER_AFTER_LAST_DEPART="${SIM_BUFFER_AFTER_LAST_DEPART:-3000}"
DRY_RUN="${DRY_RUN:-0}"

SETTINGS="${SETTINGS:-${REPO_ROOT}/tmp/logs/torchrl_corrected_cpu6_500ep_20260718_074805/settings/ingolstadt.env}"
PROTOCOL_DIR="${PROTOCOL_DIR:-${REPO_ROOT}/results/ingolstadt_custom_actionfix_incident_20260516_185810/collect_urb_fair_eval_postmutation_records}"
RESULTS_BASE="${RESULTS_BASE:-${REPO_ROOT}/results/ingolstadt_custom_actionfix_incident_20260516_185810}"
OUT_DIR="${OUT_DIR:-${RESULTS_BASE}/baselines_torchrl_iql_smdp_firsteps_${TRAIN_EPISODES}ep_${RUN_ID}/iql}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/tmp/logs/ingolstadt_iql_smdp_firsteps_${RUN_ID}}"
LOCK_DIR="${REPO_ROOT}/tmp/locks"
LOCK_FILE="${LOCK_DIR}/ingolstadt_iql_smdp_firsteps.lock"

if [[ ! "${TRAIN_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'TRAIN_EPISODES must be a positive integer, got %s\n' "${TRAIN_EPISODES}" >&2
  exit 2
fi
if [[ ! "${SIM_BUFFER_AFTER_LAST_DEPART}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'SIM_BUFFER_AFTER_LAST_DEPART must be a positive integer, got %s\n' \
    "${SIM_BUFFER_AFTER_LAST_DEPART}" >&2
  exit 2
fi

POST_MUTATION_STATE="${PROTOCOL_DIR}/post_mutation_state.json"
HDV_STATE="${PROTOCOL_DIR}/post_mutation_hdv_learning_state.json"
IQL_SOURCE="${REPO_ROOT}/baseline/run_toy_cross_bottleneck_iql_torchrl.py"
COLLECTOR_SOURCE="${REPO_ROOT}/baseline/run_toy_cross_bottleneck_online_ac.py"

for required in \
  "${PYTHON}" \
  "${SETTINGS}" \
  "${POST_MUTATION_STATE}" \
  "${HDV_STATE}" \
  "${IQL_SOURCE}" \
  "${COLLECTOR_SOURCE}"; do
  if [[ ! -e "${required}" ]]; then
    printf 'missing required input: %s\n' "${required}" >&2
    exit 2
  fi
done

mkdir -p "${LOCK_DIR}"
exec 9<>"${LOCK_FILE}"
if ! flock -n 9; then
  printf 'another corrected Ingolstadt IQL launcher holds %s\n' "${LOCK_FILE}" >&2
  exit 3
fi
printf 'launcher_pid=%s started=%s run_id=%s\n' \
  "$$" "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${RUN_ID}" >"${LOCK_FILE}"

if [[ -e "${OUT_DIR}" && ( ! -d "${OUT_DIR}" || -n "$(ls -A "${OUT_DIR}" 2>/dev/null)" ) ]]; then
  printf 'refusing non-empty output directory: %s\n' "${OUT_DIR}" >&2
  exit 4
fi
if [[ -e "${LOG_DIR}" && ( ! -d "${LOG_DIR}" || -n "$(ls -A "${LOG_DIR}" 2>/dev/null)" ) ]]; then
  printf 'refusing non-empty log directory: %s\n' "${LOG_DIR}" >&2
  exit 4
fi
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

cmd=(
  "${PYTHON}" -u "${IQL_SOURCE}"
  --settings "${SETTINGS}"
  --output-dir "${OUT_DIR}"
  --post-mutation-state-in "${POST_MUTATION_STATE}"
  --hdv-state-in "${HDV_STATE}"
  --train-episodes "${TRAIN_EPISODES}"
  --checkpoint-episodes "${CHECKPOINT_EPISODES}"
  --agent-frames-per-batch 1
  --num-epochs 1
  --minibatch-size 128
  --memory-size 250000
  --fresh-transition-epochs 1
  --replay-sample-ratio 0.25
  --eps-init 0.99
  --eps-end 0.05
  --exploration-fraction 0.4
  --epsilon-scope first_decision_per_vehicle
  --target-update-eps 0.995
  --train-sim-buffer-after-last-depart "${SIM_BUFFER_AFTER_LAST_DEPART}"
  --sumo-time-to-teleport -1
  --device cuda:0
  --seed "${SEED}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

file_sha256() {
  local value
  value="$(sha256sum "$1")"
  printf '%s' "${value%% *}"
}

{
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'started=%s\n' "$(date '+%Y-%m-%d %H:%M:%S UTC')"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
  printf 'train_episodes=%s\n' "${TRAIN_EPISODES}"
  printf 'checkpoint_episodes=%s\n' "${CHECKPOINT_EPISODES}"
  printf 'seed=%s\n' "${SEED}"
  printf 'physical_gpu=%s\n' "${GPU}"
  printf 'sumo_time_to_teleport=-1\n'
  printf 'sim_buffer_after_last_depart=%s\n' "${SIM_BUFFER_AFTER_LAST_DEPART}"
  printf 'epsilon_scope=first_decision_per_vehicle\n'
  printf 'sha256_iql_source=%s\n' "$(file_sha256 "${IQL_SOURCE}")"
  printf 'sha256_collector_source=%s\n' "$(file_sha256 "${COLLECTOR_SOURCE}")"
  printf 'sha256_settings=%s\n' "$(file_sha256 "${SETTINGS}")"
  printf 'sha256_post_mutation_state=%s\n' "$(file_sha256 "${POST_MUTATION_STATE}")"
  printf 'sha256_hdv_state=%s\n' "$(file_sha256 "${HDV_STATE}")"
  printf 'git_status_begin\n'
  git status --short
  printf 'git_status_end\n'
} >"${LOG_DIR}/manifest.txt"
printf -v command_line '%q ' "${cmd[@]}"
printf 'env CUDA_VISIBLE_DEVICES=%q setsid %s\n' "${GPU}" "${command_line}" >"${LOG_DIR}/command.sh"
printf 'time\tstate\tlauncher_pid\tchild_pid\trc\tout_dir\n' >"${LOG_DIR}/status.tsv"
printf '%s\tSTART\t%s\t-\t-\t%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S UTC')" "$$" "${OUT_DIR}" >>"${LOG_DIR}/status.tsv"

child_pid=""
cancel_run() {
  local signal_name="$1" signal_rc="$2"
  trap - HUP INT TERM
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill -TERM -- "-${child_pid}" 2>/dev/null || kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  printf '%s\tCANCELLED\t%s\t%s\t%s\t%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S UTC')" "$$" "${child_pid:--}" "${signal_rc}" "${OUT_DIR}" >>"${LOG_DIR}/status.tsv"
  printf 'cancelled by %s\n' "${signal_name}" >>"${OUT_DIR}/train.log"
  exit "${signal_rc}"
}
trap 'cancel_run HUP 129' HUP
trap 'cancel_run INT 130' INT
trap 'cancel_run TERM 143' TERM

env CUDA_VISIBLE_DEVICES="${GPU}" setsid "${cmd[@]}" >"${OUT_DIR}/train.log" 2>&1 &
child_pid=$!
printf '%s\tRUNNING\t%s\t%s\t-\t%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S UTC')" "$$" "${child_pid}" "${OUT_DIR}" >>"${LOG_DIR}/status.tsv"
printf 'RUNNING launcher_pid=%s child_pid=%s out=%s\n' "$$" "${child_pid}" "${OUT_DIR}"

if wait "${child_pid}"; then
  rc=0
  state=DONE
else
  rc=$?
  state=FAIL
fi
child_pid=""
trap - HUP INT TERM
printf '%s\t%s\t%s\t-\t%s\t%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S UTC')" "${state}" "$$" "${rc}" "${OUT_DIR}" >>"${LOG_DIR}/status.tsv"
printf '%s rc=%s out=%s\n' "${state}" "${rc}" "${OUT_DIR}"
exit "${rc}"
