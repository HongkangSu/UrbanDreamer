#!/usr/bin/env bash
set -Eeuo pipefail

# Reuse the validated 50-episode warmup, train the current 8k-step WM on a
# dedicated GPU, run 150 online episodes, then evaluate with the initial HDV state.
# Override PYTHON_BIN, CUDA_VISIBLE_DEVICES, or RUN_ID through the environment.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-online_wm_fontenay_reusewarmup_b4_150ep_server}"

# Respect an existing Slurm/Docker GPU mapping; otherwise use physical GPU 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi

CONTROLLER="$REPO_ROOT/scripts/run_three_city_online_wm_full_pipeline.py"
if [[ ! -f "$CONTROLLER" ]]; then
  echo "Pipeline controller not found: $CONTROLLER" >&2
  exit 2
fi

SUMO_ARGS=()
if [[ -n "${SUMO_HOME:-}" ]]; then
  SUMO_ARGS+=(--sumo-home "$SUMO_HOME")
fi

LOG_DIR="$REPO_ROOT/logs/fontenay_en_parisis_${RUN_ID}"
LOG_PATH="$LOG_DIR/launcher.log"
mkdir -p "$LOG_DIR"

echo "Repository: $REPO_ROOT"
echo "Python:     $PYTHON_BIN"
echo "GPU map:    $CUDA_VISIBLE_DEVICES"
echo "Run ID:     $RUN_ID"
echo "Log:        $LOG_PATH"

cd "$REPO_ROOT"

"$PYTHON_BIN" -u "$CONTROLLER" \
  --repo-root "$REPO_ROOT" \
  --python-bin "$PYTHON_BIN" \
  "${SUMO_ARGS[@]}" \
  --run-id "$RUN_ID" \
  --cities fontenay_en_parisis \
  --city-workers 1 \
  --reuse-existing-warmup \
  --start-stage wm \
  --collect-episodes 50 \
  --wm-steps 8000 \
  --wm-batch-size 1 \
  --wm-seq-len 16 \
  --wm-grad-accum 32 \
  --wm-flow-cache-mb 1024 \
  --wm-space-attention-batch-chunk-size 4 \
  --wm-mlp-token-chunk-size 16384 \
  --wm-time-attention-batch-chunk-size 8192 \
  --online-wm-batch-size 4 \
  --online-wm-space-attention-batch-chunk-size 16 \
  --online-wm-mlp-token-chunk-size 16384 \
  --online-wm-time-attention-batch-chunk-size 32768 \
  --online-imagined-rollout-batch-size 4 \
  --online-episodes 150 \
  --checkpoint-episodes 25,50,75,100,125,150 \
  --train-sim-buffer-after-last-depart 1500 \
  --eval-episodes 5 \
  --eval-hdv-variants original \
  --eval-sim-buffer-after-last-depart 7200 \
  --eval-max-steps-per-episode 9000 \
  "$@" \
  2>&1 | tee -a "$LOG_PATH"
