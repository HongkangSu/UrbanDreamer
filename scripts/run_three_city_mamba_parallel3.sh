#!/usr/bin/env bash
set -uo pipefail

# Waiting supervisor for the three-city MAMBA baseline.
#
# Default behavior:
#   1. validate/rebase canonical inputs;
#   2. wait until physical GPUs 0 and 1 are genuinely idle;
#   3. run three concurrent 2-episode smoke jobs (Fontenay on GPU0,
#      Gretz + Ingolstadt on GPU1), including real MAMBA updates;
#   4. validate losses/artifacts and an aggregate <=20 GiB peak per GPU;
#   5. wait for idle again, then run all three 500-episode jobs.
#
# Inspect the complete plan without waiting for GPUs or launching SUMO:
#   DRY_RUN=1 scripts/run_three_city_mamba_parallel3.sh
#
# The script intentionally stays attached as a supervisor.  Run it under tmux,
# systemd, or nohup if detachment is desired; signals are forwarded only to
# process groups created by this invocation, never to pre-existing jobs.

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}" || exit 2

FORMAL_EPISODES="${TRAIN_EPISODES:-500}"
FORMAL_UPDATE_EVERY_EPISODES="${FORMAL_UPDATE_EVERY_EPISODES:-1}"
FORMAL_MIN_MODEL_REPLAY_FRAMES="${FORMAL_MIN_MODEL_REPLAY_FRAMES:-100}"
FORMAL_MIN_AGENT_REPLAY_FRAMES="${FORMAL_MIN_AGENT_REPLAY_FRAMES:-2500}"
FORMAL_MAX_IMAGINATION_STARTS="${FORMAL_MAX_IMAGINATION_STARTS:-16}"
FORMAL_CRITIC_WARMUP_UPDATES="${FORMAL_CRITIC_WARMUP_UPDATES:-20}"
SMOKE_EPISODES="${SMOKE_EPISODES:-2}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
SEED="${SEED:-42}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"
STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_IDLE_GATE="${SKIP_IDLE_GATE:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
IDLE_FREE_MIB="${IDLE_FREE_MIB:-23000}"
IDLE_MAX_UTIL="${IDLE_MAX_UTIL:-5}"
IDLE_CONFIRMATIONS="${IDLE_CONFIRMATIONS:-3}"
IDLE_POLL_SECONDS="${IDLE_POLL_SECONDS:-30}"
IDLE_MAX_POLLS="${IDLE_MAX_POLLS:-0}"
SMOKE_MAX_GPU_MEMORY_MIB="${SMOKE_MAX_GPU_MEMORY_MIB:-20000}"
TELEMETRY_POLL_SECONDS="${TELEMETRY_POLL_SECONDS:-5}"
MIN_FREE_DISK_KIB="${MIN_FREE_DISK_KIB:-10485760}"
MAMBA_EXTRA_ARGS="${MAMBA_EXTRA_ARGS:-}"
REUSE_SMOKE_VALIDATION="${REUSE_SMOKE_VALIDATION:-}"

DEFAULT_PY="/root/miniconda3/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="${PYTHON_BIN}"
elif [[ -x "${DEFAULT_PY}" ]]; then
  PY="${DEFAULT_PY}"
else
  PY="$(command -v python || true)"
fi

LOG_DIR="${RUN_LOG_DIR:-${ROOT}/tmp/logs/mamba_parallel3_${STAMP}}"
INPUT_ROOT="${LOG_DIR}/portable_inputs"
STATUS_FILE="${LOG_DIR}/status.tsv"
MASTER_LOG="${LOG_DIR}/master.log"
MANIFEST="${LOG_DIR}/manifest.txt"
SMOKE_TELEMETRY="${LOG_DIR}/smoke_gpu_telemetry.csv"
FORMAL_TELEMETRY="${LOG_DIR}/formal_gpu_telemetry.csv"
SMOKE_VALIDATION="${LOG_DIR}/smoke_validation.json"
FORMAL_VALIDATION="${LOG_DIR}/formal_validation.json"
MPLCONFIG_ROOT="${LOG_DIR}/matplotlib"
RUNNER="${ROOT}/baseline/run_toy_cross_bottleneck_mamba.py"
PREPARE_INPUTS="${ROOT}/scripts/prepare_mamba_inputs.py"

CITIES=(fontenay gretz ingolstadt)
REQUIRED_GPUS=(0 1)
FONTENAY_GPU="${FONTENAY_GPU:-0}"
GRETZ_GPU="${GRETZ_GPU:-1}"
INGOLSTADT_GPU="${INGOLSTADT_GPU:-1}"
declare -A GPU_BY_CITY=(
  [fontenay]="${FONTENAY_GPU}"
  [gretz]="${GRETZ_GPU}"
  [ingolstadt]="${INGOLSTADT_GPU}"
)
declare -A TAG_BY_CITY=(
  [gretz]="gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1"
  [ingolstadt]="ingolstadt_custom_current_to_dest_incident_20260528"
  [fontenay]="fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609"
)
declare -A HDV_STATE_BY_CITY=(
  [gretz]="${ROOT}/networks/gretz_armainvilliers/post_mutation/post_mutation_hdv_learning_state.json"
  [ingolstadt]="${ROOT}/networks/ingolstadt_custom/post_mutation/post_mutation_hdv_learning_state.json"
  [fontenay]="${ROOT}/networks/fontenay_en_parisis/post_mutation_hdv_learning_state.json"
)
declare -A MAX_STEPS_BY_CITY=(
  [gretz]="0"
  [ingolstadt]="0"
  [fontenay]="3600"
)
declare -A SUMO_EXTRA_BY_CITY=(
  [gretz]="--collision.action warn"
  [ingolstadt]=""
  [fontenay]="--collision.action warn --ignore-junction-blocker 60 --tripinfo-output.write-unfinished"
)
declare -A SMOKE_OUT=()
declare -A FORMAL_OUT=()
for city in "${CITIES[@]}"; do
  SMOKE_OUT["${city}"]="${LOG_DIR}/smoke_outputs/${city}"
  FORMAL_OUT["${city}"]="${ROOT}/results/${TAG_BY_CITY[${city}]}/baselines_mamba_${FORMAL_EPISODES}ep_${STAMP}/mamba"
done

die() {
  echo "ERROR: $*" >&2
  exit 2
}

positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer, got: ${value}"
}

nonnegative_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer, got: ${value}"
}

for pair in \
  "FORMAL_EPISODES:${FORMAL_EPISODES}" \
  "FORMAL_UPDATE_EVERY_EPISODES:${FORMAL_UPDATE_EVERY_EPISODES}" \
  "FORMAL_MIN_MODEL_REPLAY_FRAMES:${FORMAL_MIN_MODEL_REPLAY_FRAMES}" \
  "FORMAL_MIN_AGENT_REPLAY_FRAMES:${FORMAL_MIN_AGENT_REPLAY_FRAMES}" \
  "FORMAL_MAX_IMAGINATION_STARTS:${FORMAL_MAX_IMAGINATION_STARTS}" \
  "SMOKE_EPISODES:${SMOKE_EPISODES}" \
  "CHECKPOINT_INTERVAL:${CHECKPOINT_INTERVAL}" \
  "CPU_THREADS_PER_JOB:${CPU_THREADS_PER_JOB}" \
  "IDLE_FREE_MIB:${IDLE_FREE_MIB}" \
  "IDLE_MAX_UTIL:${IDLE_MAX_UTIL}" \
  "IDLE_CONFIRMATIONS:${IDLE_CONFIRMATIONS}" \
  "IDLE_POLL_SECONDS:${IDLE_POLL_SECONDS}" \
  "SMOKE_MAX_GPU_MEMORY_MIB:${SMOKE_MAX_GPU_MEMORY_MIB}" \
  "TELEMETRY_POLL_SECONDS:${TELEMETRY_POLL_SECONDS}" \
  "MIN_FREE_DISK_KIB:${MIN_FREE_DISK_KIB}"; do
  positive_integer "${pair%%:*}" "${pair#*:}"
done
nonnegative_integer "SEED" "${SEED}"
nonnegative_integer "IDLE_MAX_POLLS" "${IDLE_MAX_POLLS}"
nonnegative_integer "FONTENAY_GPU" "${FONTENAY_GPU}"
nonnegative_integer "GRETZ_GPU" "${GRETZ_GPU}"
nonnegative_integer "INGOLSTADT_GPU" "${INGOLSTADT_GPU}"
for mapped_gpu in "${FONTENAY_GPU}" "${GRETZ_GPU}" "${INGOLSTADT_GPU}"; do
  [[ "${mapped_gpu}" == "0" || "${mapped_gpu}" == "1" ]] || \
    die "city GPU mappings must use available physical GPU 0 or 1, got: ${mapped_gpu}"
done
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
[[ "${SKIP_IDLE_GATE}" == "0" || "${SKIP_IDLE_GATE}" == "1" ]] || \
  die "SKIP_IDLE_GATE must be 0 or 1"
[[ "${SMOKE_ONLY}" == "0" || "${SMOKE_ONLY}" == "1" ]] || \
  die "SMOKE_ONLY must be 0 or 1"
[[ "${ALLOW_EXISTING_OUTPUT}" == "0" || "${ALLOW_EXISTING_OUTPUT}" == "1" ]] || \
  die "ALLOW_EXISTING_OUTPUT must be 0 or 1"
[[ -n "${PY}" && -x "${PY}" ]] || die "Python not found; set PYTHON_BIN"
[[ -f "${RUNNER}" ]] || die "MAMBA runner is missing: ${RUNNER}"
[[ -f "${PREPARE_INPUTS}" ]] || die "input preparation helper is missing: ${PREPARE_INPUTS}"
command -v setsid >/dev/null 2>&1 || die "setsid is required"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"

ENV_ROOT="$(cd "$(dirname "${PY}")/.." && pwd)"
sumo_binary_works() {
  local candidate="$1"
  [[ -x "${candidate}/bin/sumo" ]] && \
    "${candidate}/bin/sumo" --version >/dev/null 2>&1
}
if [[ -n "${SUMO_HOME:-}" ]]; then
  # An explicit value is authoritative, but fail now instead of discovering a
  # broken wrapper only after the GPU idle gate and a long SUMO startup.
  sumo_binary_works "${SUMO_HOME}" || \
    die "SUMO_HOME does not contain a working bin/sumo: ${SUMO_HOME}"
else
  PYTHON_PURELIB="$("${PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
  SUMO_HOME=""
  for sumo_candidate in \
      "${PYTHON_PURELIB}/sumo" \
      "${ENV_ROOT}/share/sumo" \
      "${ENV_ROOT}"; do
    if [[ -n "${sumo_candidate}" ]] && sumo_binary_works "${sumo_candidate}"; then
      SUMO_HOME="${sumo_candidate}"
      break
    fi
  done
  [[ -n "${SUMO_HOME}" ]] || \
    die "could not find a working SUMO installation under Python purelib or ${ENV_ROOT}"
fi
export SUMO_HOME
# This must precede the Python environment's bin directory: pip's top-level
# `bin/sumo` may be a broken console wrapper while the bundled native binary
# under purelib/sumo/bin is valid.
export PATH="${SUMO_HOME}/bin:${PATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"

available_kib="$(df -Pk "${ROOT}" | awk 'NR == 2 {print $4}')"
[[ "${available_kib}" =~ ^[0-9]+$ ]] || die "could not determine free disk space for ${ROOT}"
(( available_kib >= MIN_FREE_DISK_KIB )) || \
  die "free disk space $((available_kib / 1024)) MiB is below required $((MIN_FREE_DISK_KIB / 1024)) MiB"

mkdir -p "${LOG_DIR}" "${MPLCONFIG_ROOT}" || die "cannot create ${LOG_DIR}"
printf "time\tstate\tphase\tcity\tgpu\tpid\trc\tout_dir\tdetail\n" > "${STATUS_FILE}" || \
  die "cannot initialize ${STATUS_FILE}"
: > "${MASTER_LOG}" || die "cannot initialize ${MASTER_LOG}"

write_status() {
  local state="$1"
  local phase="$2"
  local city="$3"
  local gpu="$4"
  local pid="$5"
  local rc="$6"
  local out_dir="$7"
  local detail="${8:-}"
  detail="${detail//$'\t'/ }"
  detail="${detail//$'\n'/ }"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${state}" "${phase}" "${city}" \
    "${gpu}" "${pid}" "${rc}" "${out_dir}" "${detail}" >> "${STATUS_FILE}"
}

log_master() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "${MASTER_LOG}"
}

write_status "PREPARE" "preflight" "-" "-" "$$" "-" "${INPUT_ROOT}" "validating canonical bundles"
if ! "${PY}" "${PREPARE_INPUTS}" prepare \
    --repo-root "${ROOT}" \
    --output-root "${INPUT_ROOT}" \
    --cities "gretz,ingolstadt,fontenay" \
    > "${LOG_DIR}/input_prepare_stdout.json"; then
  write_status "FAIL" "preflight" "-" "-" "$$" "2" "${INPUT_ROOT}" "canonical input validation failed"
  die "canonical input validation failed; see ${LOG_DIR}/input_prepare_stdout.json"
fi
write_status "DONE" "preflight" "-" "-" "$$" "0" "${INPUT_ROOT}" "hash/XML/fixed-role validation passed"

for city in "${CITIES[@]}"; do
  for path in "${SMOKE_OUT[${city}]}" "${FORMAL_OUT[${city}]}"; do
    if [[ -e "${path}" && "${ALLOW_EXISTING_OUTPUT}" != "1" ]]; then
      die "output already exists: ${path}; choose a new STAMP"
    fi
  done
done

if ! available_gpu_rows="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null)"; then
  die "nvidia-smi failed while enumerating GPUs"
fi
for required_gpu in "${REQUIRED_GPUS[@]}"; do
  if ! awk -v wanted="${required_gpu}" '{gsub(/[[:space:]]/, "", $0); if ($0 == wanted) found=1} END {exit !found}' <<< "${available_gpu_rows}"; then
    die "required physical GPU ${required_gpu} is unavailable"
  fi
done

if ! {
  echo "repo_root=${ROOT}"
  echo "python=${PY}"
  echo "sumo_home=${SUMO_HOME}"
  echo "sumo_binary=$(command -v sumo)"
  echo "stamp=${STAMP}"
  echo "runner=${RUNNER}"
  echo "formal_episodes=${FORMAL_EPISODES}"
  echo "formal_update_every_episodes=${FORMAL_UPDATE_EVERY_EPISODES}"
  echo "formal_min_model_replay_frames=${FORMAL_MIN_MODEL_REPLAY_FRAMES}"
  echo "formal_min_agent_replay_frames=${FORMAL_MIN_AGENT_REPLAY_FRAMES}"
  echo "formal_max_imagination_starts=${FORMAL_MAX_IMAGINATION_STARTS}"
  echo "formal_critic_warmup_updates=${FORMAL_CRITIC_WARMUP_UPDATES}"
  echo "smoke_episodes=${SMOKE_EPISODES}"
  echo "checkpoint_interval=${CHECKPOINT_INTERVAL}"
  echo "seed=${SEED}"
  echo "skip_idle_gate=${SKIP_IDLE_GATE}"
  echo "smoke_only=${SMOKE_ONLY}"
  echo "gpu_mapping=fontenay:${GPU_BY_CITY[fontenay]},gretz:${GPU_BY_CITY[gretz]},ingolstadt:${GPU_BY_CITY[ingolstadt]}"
  echo "cpu_threads_per_job=${CPU_THREADS_PER_JOB}"
  echo "pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
  echo "idle_gate=no_compute_pid,free_mib>=${IDLE_FREE_MIB},util<=${IDLE_MAX_UTIL},consecutive=${IDLE_CONFIRMATIONS},poll_seconds=${IDLE_POLL_SECONDS}"
  echo "smoke_max_aggregate_gpu_memory_mib=${SMOKE_MAX_GPU_MEMORY_MIB}"
  echo "disk_free_kib_at_preflight=${available_kib}"
  echo "canonical_gretz_bundle=${ROOT}/networks/gretz_armainvilliers/post_mutation"
  echo "canonical_ingolstadt_bundle=${ROOT}/networks/ingolstadt_custom/post_mutation"
  echo "canonical_fontenay_bundle=${ROOT}/networks/fontenay_en_parisis"
  echo "ingolstadt_tag=${TAG_BY_CITY[ingolstadt]}"
  echo "mamba_extra_args=${MAMBA_EXTRA_ARGS}"
  echo "reuse_smoke_validation=${REUSE_SMOKE_VALIDATION:-none}"
} > "${MANIFEST}"; then
  die "cannot write ${MANIFEST}"
fi

declare -a CMD=()
build_command() {
  local phase="$1"
  local city="$2"
  local output_dir="$3"
  local episodes="$4"
  local checkpoint_interval="$5"
  local update_every="${FORMAL_UPDATE_EVERY_EPISODES}"
  local min_model_replay_frames="${FORMAL_MIN_MODEL_REPLAY_FRAMES}"
  local min_agent_replay_frames="${FORMAL_MIN_AGENT_REPLAY_FRAMES}"
  local critic_warmup_updates="${FORMAL_CRITIC_WARMUP_UPDATES}"
  local -a extra_args=()
  if [[ "${phase}" == "smoke" ]]; then
    update_every=1
    min_model_replay_frames=1
    min_agent_replay_frames=1
    critic_warmup_updates=0
  fi

  CMD=(
    "${PY}" -u "${RUNNER}"
    --settings "${INPUT_ROOT}/${city}/settings.env"
    --output-dir "${output_dir}"
    --post-mutation-state-in "${INPUT_ROOT}/${city}/post_mutation_state.json"
    --hdv-state-in "${HDV_STATE_BY_CITY[${city}]}"
    --train-episodes "${episodes}"
    --checkpoint-interval "${checkpoint_interval}"
    --sumo-time-to-teleport -1
    --device cuda
    --seed "${SEED}"
    --hidden-dim 256
    --embed-dim 256
    --deter-dim 256
    --stoch-classes 32
    --stoch-categories 32
    --transformer-layers 3
    --attention-heads 8
    --imag-horizon 15
    --replay-episodes 20
    --sequence-length 20
    --burn-in 5
    --update-every-episodes "${update_every}"
    --min-model-replay-frames "${min_model_replay_frames}"
    --min-agent-replay-frames "${min_agent_replay_frames}"
    --max-imagination-starts "${FORMAL_MAX_IMAGINATION_STARTS}"
    --sequence-minibatch-size 4
    --model-epochs 4
    --agent-epochs 1
    --critic-epochs 5
    --ppo-epochs 5
    --critic-warmup-updates "${critic_warmup_updates}"
    --imagination-reward-scope individual
    --actor-lr 2e-4
    --critic-lr 3e-4
    --entropy-coef 5e-3
    --ppo-clip 7.5e-2
    --target-policy-kl 1e-2
    --critic-huber-delta 1.0
    --target-tau 5e-2
    --advantage-clip 2.5
    --normalize-advantages
    --max-grad-norm 10
    --communication-chunk-size 32
    --amp-dtype bf16
  )
  if [[ "${MAX_STEPS_BY_CITY[${city}]}" != "0" ]]; then
    CMD+=(--max-steps-per-episode "${MAX_STEPS_BY_CITY[${city}]}")
  fi
  if [[ -n "${SUMO_EXTRA_BY_CITY[${city}]}" ]]; then
    CMD+=(--sumo-extra-args "${SUMO_EXTRA_BY_CITY[${city}]}")
  fi
  if [[ -n "${MAMBA_EXTRA_ARGS}" ]]; then
    # Intended for simple argparse overrides.  Word splitting is deliberate;
    # multi-word SUMO values already have dedicated, safely quoted flags above.
    read -r -a extra_args <<< "${MAMBA_EXTRA_ARGS}"
    CMD+=("${extra_args[@]}")
  fi
}

render_command() {
  local item
  printf "CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q " \
    "$1" "${CPU_THREADS_PER_JOB}" "${CPU_THREADS_PER_JOB}" "${CPU_THREADS_PER_JOB}"
  shift
  for item in "$@"; do
    printf "%q " "${item}"
  done
  printf "\n"
}

echo "stamp=${STAMP}"
echo "log_dir=${LOG_DIR}"
echo "status_file=${STATUS_FILE}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "IDLE_GATE=SKIPPED_FOR_DRY_RUN"
  for phase in smoke formal; do
    if [[ "${phase}" == "smoke" ]]; then
      episodes="${SMOKE_EPISODES}"
      interval=1
    else
      episodes="${FORMAL_EPISODES}"
      interval="${CHECKPOINT_INTERVAL}"
    fi
    for city in "${CITIES[@]}"; do
      if [[ "${phase}" == "smoke" ]]; then
        output_dir="${SMOKE_OUT[${city}]}"
      else
        output_dir="${FORMAL_OUT[${city}]}"
      fi
      build_command "${phase}" "${city}" "${output_dir}" "${episodes}" "${interval}"
      printf "phase=%s city=%s physical_gpu=%s output=%s\n" \
        "${phase}" "${city}" "${GPU_BY_CITY[${city}]}" "${output_dir}"
      render_command "${GPU_BY_CITY[${city}]}" "${CMD[@]}"
      if [[ "${phase}" == "smoke" ]]; then
        # Exercise the real runner parser/config builder once per city.  This
        # catches launcher/runner CLI drift while remaining SUMO- and GPU-free.
        if ! CUDA_VISIBLE_DEVICES="${GPU_BY_CITY[${city}]}" \
            OMP_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
            MKL_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
            OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
            "${CMD[@]}" --dry-run > "${LOG_DIR}/runner_plan_${city}.json"; then
          die "runner dry-run validation failed for ${city}"
        fi
      fi
    done
  done
  write_status "DRY_RUN" "plan" "-" "0,1" "$$" "0" "${LOG_DIR}" "GPU idle wait and training skipped"
  exit 0
fi

gpu_idle_snapshot() {
  local stage="$1"
  local rows apps
  local gpu index uuid free util app_uuid app_pid
  local failed=0
  declare -A uuid_by_gpu=()
  declare -A free_by_gpu=()
  declare -A util_by_gpu=()
  declare -A compute_by_uuid=()

  if ! rows="$(nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    log_master "${stage} idle probe: nvidia-smi GPU query failed"
    return 1
  fi
  if ! apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null)"; then
    log_master "${stage} idle probe: nvidia-smi compute-process query failed"
    return 1
  fi
  while IFS=',' read -r index uuid free util; do
    index="${index//[[:space:]]/}"
    uuid="${uuid//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    [[ -n "${index}" ]] || continue
    uuid_by_gpu["${index}"]="${uuid}"
    free_by_gpu["${index}"]="${free}"
    util_by_gpu["${index}"]="${util}"
  done <<< "${rows}"
  while IFS=',' read -r app_uuid app_pid; do
    app_uuid="${app_uuid//[[:space:]]/}"
    app_pid="${app_pid//[[:space:]]/}"
    if [[ -n "${app_uuid}" && "${app_pid}" =~ ^[1-9][0-9]*$ ]]; then
      compute_by_uuid["${app_uuid}"]+="${app_pid},"
    fi
  done <<< "${apps}"

  for gpu in "${REQUIRED_GPUS[@]}"; do
    uuid="${uuid_by_gpu[${gpu}]:-}"
    free="${free_by_gpu[${gpu}]:-invalid}"
    util="${util_by_gpu[${gpu}]:-invalid}"
    if [[ -z "${uuid}" || ! "${free}" =~ ^[0-9]+$ || ! "${util}" =~ ^[0-9]+$ ]]; then
      failed=1
      log_master "${stage} idle probe GPU${gpu}: invalid nvidia-smi sample"
      continue
    fi
    app_pid="${compute_by_uuid[${uuid}]:-none}"
    log_master "${stage} idle probe GPU${gpu}: free=${free}MiB util=${util}% compute_pids=${app_pid}"
    if [[ "${app_pid}" != "none" ]] || (( free < IDLE_FREE_MIB )) || (( util > IDLE_MAX_UTIL )); then
      failed=1
    fi
  done
  return "${failed}"
}

wait_for_idle_gpus() {
  local stage="$1"
  local consecutive=0
  local polls=0
  if [[ "${SKIP_IDLE_GATE}" == "1" ]]; then
    write_status "GPU_IDLE_BYPASS" "${stage}" "-" "0,1" "$$" "0" "-" \
      "explicit SKIP_IDLE_GATE=1; existing GPU processes were not terminated"
    log_master "${stage} idle gate bypassed by explicit SKIP_IDLE_GATE=1"
    return 0
  fi
  write_status "WAIT_GPU_IDLE" "${stage}" "-" "0,1" "$$" "-" "-" \
    "need ${IDLE_CONFIRMATIONS} consecutive samples"
  while (( consecutive < IDLE_CONFIRMATIONS )); do
    polls=$((polls + 1))
    if gpu_idle_snapshot "${stage}"; then
      consecutive=$((consecutive + 1))
      log_master "${stage} idle confirmation ${consecutive}/${IDLE_CONFIRMATIONS}"
    else
      consecutive=0
      log_master "${stage} GPUs are busy; idle confirmation reset"
    fi
    if (( IDLE_MAX_POLLS > 0 && polls >= IDLE_MAX_POLLS && consecutive < IDLE_CONFIRMATIONS )); then
      write_status "FAIL" "${stage}" "-" "0,1" "$$" "124" "-" "idle gate poll limit reached"
      return 124
    fi
    if (( consecutive < IDLE_CONFIRMATIONS )); then
      sleep "${IDLE_POLL_SECONDS}"
    fi
  done
  write_status "GPU_IDLE" "${stage}" "-" "0,1" "$$" "0" "-" \
    "${IDLE_CONFIRMATIONS} consecutive samples passed"
  return 0
}

TELEMETRY_PID=""
monitor_gpu_memory() {
  local phase="$1"
  local output="$2"
  local rows now index used util
  printf "timestamp,phase,gpu,memory_used_mib,utilization_gpu_pct\n" > "${output}"
  while :; do
    now="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    if rows="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
      while IFS=',' read -r index used util; do
        index="${index//[[:space:]]/}"
        used="${used//[[:space:]]/}"
        util="${util//[[:space:]]/}"
        if [[ "${index}" == "0" || "${index}" == "1" ]]; then
          printf "%s,%s,%s,%s,%s\n" "${now}" "${phase}" "${index}" "${used}" "${util}" >> "${output}"
        fi
      done <<< "${rows}"
    fi
    sleep "${TELEMETRY_POLL_SECONDS}"
  done
}

start_telemetry() {
  local phase="$1"
  local output="$2"
  local _attempt lines
  monitor_gpu_memory "${phase}" "${output}" &
  TELEMETRY_PID=$!
  # Ensure both required GPUs have a baseline sample before training starts.
  # This closes the short-job race where the monitor could otherwise be
  # stopped before its first nvidia-smi query was scheduled.
  for _attempt in {1..100}; do
    lines="$(wc -l < "${output}" 2>/dev/null || echo 0)"
    if [[ "${lines}" =~ ^[0-9]+$ ]] && (( lines >= 3 )); then
      return 0
    fi
    sleep 0.05
  done
  return 1
}

stop_telemetry() {
  if [[ -n "${TELEMETRY_PID}" ]]; then
    kill -TERM "${TELEMETRY_PID}" 2>/dev/null || true
    wait "${TELEMETRY_PID}" 2>/dev/null || true
    TELEMETRY_PID=""
  fi
}

worker_exit() {
  local original_rc="$1"
  trap - EXIT HUP INT TERM
  local final_rc="${original_rc}"
  local state="DONE"
  if [[ -n "${WORKER_CHILD_PID:-}" ]]; then
    if [[ "${WORKER_CHILD_REAPED:-0}" != "1" ]]; then
      kill -TERM "${WORKER_CHILD_PID}" 2>/dev/null || true
    fi
    kill -TERM -- "-${WORKER_CHILD_PID}" 2>/dev/null || true
    for _attempt in {1..20}; do
      kill -0 -- "-${WORKER_CHILD_PID}" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 -- "-${WORKER_CHILD_PID}" 2>/dev/null; then
      kill -KILL -- "-${WORKER_CHILD_PID}" 2>/dev/null || true
    fi
    if [[ "${WORKER_CHILD_REAPED:-0}" != "1" ]]; then
      wait "${WORKER_CHILD_PID}" 2>/dev/null || true
    fi
    WORKER_CHILD_PID=""
  fi
  [[ "${final_rc}" == "0" ]] || state="FAIL"
  write_status "${state}" "${WORKER_PHASE}" "${WORKER_CITY}" "${WORKER_GPU}" \
    "${BASHPID}" "${final_rc}" "${WORKER_OUT}" "worker terminal status"
  exit "${final_rc}"
}

run_one() {
  local phase="$1"
  local city="$2"
  local output_dir="$3"
  local episodes="$4"
  local checkpoint_interval="$5"
  local gpu="${GPU_BY_CITY[${city}]}"
  local log_file="${output_dir}/train.log"
  local mpl_dir="${MPLCONFIG_ROOT}/${phase}_${city}"
  local rc=0

  WORKER_PHASE="${phase}"
  WORKER_CITY="${city}"
  WORKER_GPU="${gpu}"
  WORKER_OUT="${output_dir}"
  WORKER_CHILD_PID=""
  WORKER_CHILD_REAPED=0
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'worker_exit "$?"' EXIT

  mkdir -p "${output_dir}" "${mpl_dir}" || exit 125
  exec > "${log_file}" 2>&1 || exit 125
  build_command "${phase}" "${city}" "${output_dir}" "${episodes}" "${checkpoint_interval}"
  write_status "START" "${phase}" "${city}" "${gpu}" "${BASHPID}" "-" "${output_dir}" "training launched"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] START phase=${phase} city=${city} physical_gpu=${gpu}"
  render_command "${gpu}" "${CMD[@]}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
    MKL_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
    OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
    NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
    MPLCONFIGDIR="${mpl_dir}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
    setsid "${CMD[@]}" &
  WORKER_CHILD_PID=$!
  if wait "${WORKER_CHILD_PID}"; then
    rc=0
  else
    rc=$?
  fi
  WORKER_CHILD_REAPED=1
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] FINISH phase=${phase} city=${city} rc=${rc}"
  exit "${rc}"
}

declare -A ACTIVE_WORKERS=()
PARENT_CLEANUP_DONE=0
parent_cleanup() {
  local pid
  [[ "${PARENT_CLEANUP_DONE}" == "0" ]] || return
  PARENT_CLEANUP_DONE=1
  trap - HUP INT TERM
  stop_telemetry
  for pid in "${!ACTIVE_WORKERS[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "${!ACTIVE_WORKERS[@]}"; do
    wait "${pid}" 2>/dev/null || true
    unset 'ACTIVE_WORKERS[$pid]'
  done
}
parent_signal() {
  local rc="$1"
  log_master "received signal; stopping only this launcher's active workers"
  parent_cleanup
  exit "${rc}"
}
trap 'parent_signal 129' HUP
trap 'parent_signal 130' INT
trap 'parent_signal 143' TERM
trap 'rc=$?; parent_cleanup; exit "${rc}"' EXIT

run_phase() {
  local phase="$1"
  local episodes="$2"
  local checkpoint_interval="$3"
  local city pid rc
  local failures=0
  ACTIVE_WORKERS=()
  log_master "launching phase=${phase}: 3 concurrent jobs"
  for city in "${CITIES[@]}"; do
    if [[ "${phase}" == "smoke" ]]; then
      output_dir="${SMOKE_OUT[${city}]}"
    else
      output_dir="${FORMAL_OUT[${city}]}"
    fi
    run_one "${phase}" "${city}" "${output_dir}" "${episodes}" "${checkpoint_interval}" &
    pid=$!
    ACTIVE_WORKERS["${pid}"]="${city}"
  done
  # Waiting sequentially does not serialize execution: all wrappers above are
  # already running.  It works on Bash 5.0 as well as 5.1 and preserves each rc.
  for pid in "${!ACTIVE_WORKERS[@]}"; do
    city="${ACTIVE_WORKERS[${pid}]}"
    if wait "${pid}"; then
      rc=0
      log_master "phase=${phase} city=${city} wrapper done"
    else
      rc=$?
      failures=$((failures + 1))
      log_master "phase=${phase} city=${city} wrapper failed rc=${rc}"
    fi
    unset 'ACTIVE_WORKERS[$pid]'
  done
  return "${failures}"
}

if [[ -n "${REUSE_SMOKE_VALIDATION}" ]]; then
  if [[ ! -f "${REUSE_SMOKE_VALIDATION}" ]] || ! "${PY}" -c \
      'import json, sys; data=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if data.get("ok") is True else 1)' \
      "${REUSE_SMOKE_VALIDATION}"; then
    write_status "FAIL" "smoke_validation_reuse" "-" "0,1" "$$" "2" \
      "${REUSE_SMOKE_VALIDATION}" "reused smoke validation is missing, invalid, or not ok"
    die "REUSE_SMOKE_VALIDATION is not a successful smoke report: ${REUSE_SMOKE_VALIDATION}"
  fi
  cp "${REUSE_SMOKE_VALIDATION}" "${SMOKE_VALIDATION}" || \
    die "failed to copy reused smoke validation into this run"
  write_status "REUSED" "smoke_validation" "-" "0,1" "$$" "0" \
    "${SMOKE_VALIDATION}" "previous update-every-episode smoke validation reused"
  log_master "reusing successful smoke validation: ${REUSE_SMOKE_VALIDATION}"
else
  wait_for_idle_gpus "pre_smoke" || exit $?
  if ! start_telemetry "smoke" "${SMOKE_TELEMETRY}"; then
    write_status "FAIL" "smoke" "-" "0,1" "$$" "125" "${SMOKE_TELEMETRY}" \
      "GPU telemetry failed to initialize"
    stop_telemetry
    exit 125
  fi
  if run_phase "smoke" "${SMOKE_EPISODES}" 1; then
    smoke_rc=0
  else
    smoke_rc=$?
  fi
  stop_telemetry
  if (( smoke_rc != 0 )); then
    write_status "FORMAL_BLOCKED" "smoke" "-" "0,1" "$$" "${smoke_rc}" "-" "one or more smoke jobs failed"
    log_master "formal training blocked: smoke process failure count=${smoke_rc}"
    exit "${smoke_rc}"
  fi

  verify_cmd=(
    "${PY}" "${PREPARE_INPUTS}" verify-smoke
    --telemetry "${SMOKE_TELEMETRY}"
    --expected-episodes "${SMOKE_EPISODES}"
    --required-gpus 0,1
    --max-gpu-memory-mib "${SMOKE_MAX_GPU_MEMORY_MIB}"
    --result-json "${SMOKE_VALIDATION}"
  )
  for city in "${CITIES[@]}"; do
    verify_cmd+=(--output "${city}=${SMOKE_OUT[${city}]}")
  done
  if ! "${verify_cmd[@]}" >> "${MASTER_LOG}" 2>&1; then
    write_status "FORMAL_BLOCKED" "smoke_validation" "-" "0,1" "$$" "2" "-" \
      "updates/artifacts/GPU-memory validation failed"
    log_master "formal training blocked: smoke validation failed"
    exit 2
  fi
  write_status "VALIDATED" "smoke_validation" "-" "0,1" "$$" "0" "${SMOKE_VALIDATION}" \
    "finite model+actor+critic updates, artifacts, and GPU peaks passed"
fi

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  write_status "DONE" "smoke_only" "-" "0,1" "$$" "0" "${SMOKE_VALIDATION}" \
    "smoke validated; formal phase intentionally skipped"
  log_master "SMOKE_ONLY=1; formal phase intentionally skipped"
  exit 0
fi

wait_for_idle_gpus "pre_formal" || exit $?
if ! start_telemetry "formal" "${FORMAL_TELEMETRY}"; then
  write_status "FAIL" "formal" "-" "0,1" "$$" "125" "${FORMAL_TELEMETRY}" \
    "GPU telemetry failed to initialize"
  stop_telemetry
  exit 125
fi
if run_phase "formal" "${FORMAL_EPISODES}" "${CHECKPOINT_INTERVAL}"; then
  formal_rc=0
else
  formal_rc=$?
fi
stop_telemetry
if (( formal_rc != 0 )); then
  log_master "formal phase finished with ${formal_rc} failed job(s)"
  exit "${formal_rc}"
fi
formal_verify_cmd=(
  "${PY}" "${PREPARE_INPUTS}" verify-smoke
  --telemetry "${FORMAL_TELEMETRY}"
  --expected-episodes "${FORMAL_EPISODES}"
  --required-gpus 0,1
  --skip-memory-limit
  --result-json "${FORMAL_VALIDATION}"
)
for city in "${CITIES[@]}"; do
  formal_verify_cmd+=(--output "${city}=${FORMAL_OUT[${city}]}")
done
if ! "${formal_verify_cmd[@]}" >> "${MASTER_LOG}" 2>&1; then
  write_status "FAIL" "formal_validation" "-" "0,1" "$$" "2" "${FORMAL_VALIDATION}" \
    "ep${FORMAL_EPISODES} checkpoint/CSV/finite-update/artifact validation failed"
  log_master "formal jobs exited zero but terminal artifact validation failed"
  exit 2
fi
write_status "VALIDATED" "formal_validation" "-" "0,1" "$$" "0" "${FORMAL_VALIDATION}" \
  "ep${FORMAL_EPISODES} checkpoint/CSV/finite updates/artifacts passed"
write_status "ALL_DONE" "formal" "-" "0,1" "$$" "0" "-" "all three MAMBA trainings completed"
log_master "all three ${FORMAL_EPISODES}-episode MAMBA jobs completed"
exit 0
