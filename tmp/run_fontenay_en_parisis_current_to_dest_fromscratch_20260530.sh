#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wangwenrui/shk/UrbanDreamer"
PY="/home/wangwenrui/opt/miniconda3/envs/URB/bin/python"
TAG="${TAG:-fontenay_en_parisis_current_to_dest_noincident_20260530}"
NETWORK_DIR="networks/fontenay_en_parisis"

cd "$ROOT"

env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" WANDB_MODE=offline MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" "$PY" -u scripts/run_urb_fair_full_cycle.py \
  --network-dir "$NETWORK_DIR" \
  --network-name fontenay_en_parisis \
  --route-generator dijkstra \
  --route-candidate-source current_to_dest \
  --wm-candidate-edge-cost-source freeflow \
  --eval-candidate-source current_shortest_freeflow \
  --ratio-machines "${RATIO_MACHINES:-0.4}" \
  "${REQUIRE_INCIDENT_FLAG:---no-require-incident}" \
  "--incident-edge=${INCIDENT_EDGE:-}" \
  --incident-start "${INCIDENT_START:-500}" \
  --incident-end "${INCIDENT_END:-1500}" \
  --incident-speed "${INCIDENT_SPEED:-0.0}" \
  --output-root "data/${TAG}" \
  --logs-root "logs/${TAG}" \
  --results-root "results/${TAG}" \
  --generated-config-dir ".urbandreamer/generated/${TAG}" \
  --collect-episodes "${COLLECT_EPISODES:-50}" \
  --warmup-episodes "${WARMUP_EPISODES:-200}" \
  --collect-max-steps "${COLLECT_MAX_STEPS:-3600}" \
  --sim-buffer-after-last-depart "${SIM_BUFFER_AFTER_LAST_DEPART:-1000}" \
  --wm-max-steps "${WM_MAX_STEPS:-8000}" \
  --wm-save-every "${WM_SAVE_EVERY:-1000}" \
  --wm-eval-every "${WM_EVAL_EVERY:-1000}" \
  --wm-batch-size "${WM_BATCH_SIZE:-1}" \
  --wm-seq-len "${WM_SEQ_LEN:-8}" \
  --wm-grad-accum "${WM_GRAD_ACCUM:-4}" \
  --wm-d-model-dyn "${WM_D_MODEL_DYN:-128}" \
  --wm-dyn-depth "${WM_DYN_DEPTH:-4}" \
  --wm-n-heads "${WM_N_HEADS:-4}" \
  --marl-actor-objective discrete_actor_critic \
  --marl-episodes "${MARL_EPISODES:-400}" \
  --marl-batch-episodes 1 \
  --marl-actor-lr "${MARL_ACTOR_LR:-2.0e-4}" \
  --marl-dreamer-actor-horizon "${ACTOR_HORIZON:-8}" \
  --marl-ac-step-advantage-weight 1.0 \
  --marl-ac-route-advantage-weight 0.0 \
  --marl-ac-route-credit-mode route_cost \
  --marl-ac-route-credit-base-cost-mode progress_fraction \
  --marl-ac-advantage-clip "${AC_ADVANTAGE_CLIP:-2.5}" \
  --marl-ac-ppo-clip-ratio "${AC_PPO_CLIP_RATIO:-0.075}" \
  --marl-ac-entropy-target-frac "${AC_ENTROPY_TARGET_FRAC:-0.65}" \
  --marl-ac-entropy-floor-coef "${AC_ENTROPY_FLOOR_COEF:-0.02}" \
  --marl-ac-rollout-action-mode sample \
  --marl-ac-rollout-argmax-final-mix 1.0 \
  --marl-route-cost-reward-weight 1.0 \
  --marl-route-cost-reward-scope vehicle_critic \
  --marl-route-cost-reward-state current \
  --marl-remaining-time-progress-weight 0.0 \
  --marl-grpo-route-cost-weight 0.0 \
  --marl-grpo-rollout-weight 0.0 \
  --marl-grpo-rollout-group-size 1 \
  --marl-rollout-cost-reward-weight 0.0 \
  --marl-route-cost-aux-weight 0.0 \
  --marl-actor-trust-region-kl "${ACTOR_TRUST_REGION_KL:-0.02}" \
  --marl-actor-trust-region-max-argmax-change-frac "${ACTOR_TRUST_REGION_MAX_ARGMAX_CHANGE_FRAC:-0.2}" \
  --marl-actor-trust-region-max-route0-frac "${ACTOR_TRUST_REGION_MAX_ROUTE0_FRAC:-0.0}" \
  --marl-actor-trust-region-max-retries "${ACTOR_TRUST_REGION_MAX_RETRIES:-3}" \
  --marl-actor-anchor-start-episode "${ACTOR_ANCHOR_START_EPISODE:-60}" \
  --marl-actor-anchor-kl "${ACTOR_ANCHOR_KL:-0.05}" \
  --marl-actor-anchor-max-argmax-change-frac 0.0 \
  --marl-actor-anchor-max-vehicle-argmax-change-frac 0.0 \
  --marl-actor-anchor-max-route-frac 0.0 \
  --marl-actor-anchor-max-route0-frac "${ACTOR_ANCHOR_MAX_ROUTE0_FRAC:-0.9}" \
  --marl-vehicle-obs-include-route-costs \
  --marl-vehicle-obs-include-fleet-pressure \
  --checkpoint-episodes "${CHECKPOINT_EPISODES:-100 200 400}" \
  --marl-checkpoint-every "${MARL_CHECKPOINT_EVERY:-100}" \
  --eval-test-eps "${EVAL_TEST_EPS:-1}" \
  --eval-av-route-cost-policy none \
  --marl-seed "${MARL_SEED:-98}" \
  --device cuda:0 \
  --start-stage "${START_STAGE:-collect}" \
  "$@"
