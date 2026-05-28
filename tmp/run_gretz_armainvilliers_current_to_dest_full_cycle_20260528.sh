#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wangwenrui/shk/UrbanDreamer"
PY="/home/wangwenrui/opt/miniconda3/envs/URB/bin/python"
TAG="gretz_armainvilliers_current_to_dest_incident_20260528"

cd "$ROOT"

env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" WANDB_MODE=offline "$PY" -u scripts/run_urb_fair_full_cycle.py \
  --network-dir networks/gretz_armainvilliers \
  --network-name gretz_armainvilliers \
  --route-generator dijkstra \
  --route-candidate-source current_to_dest \
  --ratio-machines 0.4 \
  --output-root "data/${TAG}" \
  --logs-root "logs/${TAG}" \
  --results-root "results/${TAG}" \
  --generated-config-dir ".urbandreamer/generated/${TAG}" \
  --collect-episodes "${COLLECT_EPISODES:-50}" \
  --warmup-episodes "${WARMUP_EPISODES:-200}" \
  --collect-max-steps 3600 \
  --sim-buffer-after-last-depart 1000 \
  --wm-max-steps "${WM_MAX_STEPS:-8000}" \
  --wm-save-every 1000 \
  --wm-eval-every 1000 \
  --marl-actor-objective discrete_actor_critic \
  --marl-episodes "${MARL_EPISODES:-200}" \
  --marl-dreamer-actor-horizon "${ACTOR_HORIZON:-8}" \
  --marl-ac-step-advantage-weight "${AC_STEP_WEIGHT:-0.1}" \
  --marl-ac-route-advantage-weight "${AC_ROUTE_WEIGHT:-1.0}" \
  --marl-ac-route-credit-mode "${AC_ROUTE_CREDIT_MODE:-route_cost}" \
  --marl-ac-route-credit-base-cost-mode "${AC_BASE_COST_MODE:-progress_fraction}" \
  --marl-grpo-route-cost-weight 0.0 \
  --marl-grpo-rollout-weight 0.0 \
  --marl-grpo-rollout-group-size 1 \
  --marl-vehicle-obs-include-route-costs \
  --marl-vehicle-obs-include-fleet-pressure \
  --checkpoint-episodes "${CHECKPOINT_EPISODES:-100 200}" \
  --eval-test-eps "${EVAL_TEST_EPS:-1}" \
  --marl-seed "${MARL_SEED:-48}" \
  --device cuda:0
