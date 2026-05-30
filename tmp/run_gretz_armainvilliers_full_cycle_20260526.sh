#!/usr/bin/env bash
set -euo pipefail

echo "[deprecated] tmp/run_gretz_armainvilliers_full_cycle_20260526.sh uses the legacy od_suffix/GRPO validation path." >&2
echo "[deprecated] Use tmp/run_gretz_armainvilliers_current_to_dest_fromscratch_20260530.sh for the current per-vehicle-critic mainline." >&2
exit 2

ROOT="/home/wangwenrui/shk/UrbanDreamer"
PY="/home/wangwenrui/opt/miniconda3/envs/URB/bin/python"

cd "$ROOT"

env CUDA_VISIBLE_DEVICES=1 "$PY" -u scripts/run_urb_fair_full_cycle.py \
  --network-dir networks/gretz_armainvilliers \
  --network-name gretz_armainvilliers \
  --route-artifacts-dir results/gretz_armainvilliers/collect_urb_fair_eval_postmutation_records \
  --route-generator dijkstra \
  --route-candidate-source od_suffix \
  --ratio-machines 0.4 \
  --n-av 30 \
  --n-hdv 70 \
  --incident-edge=-78047947#0 \
  --incident-start=500 \
  --incident-end=1500 \
  --incident-speed=1.0 \
  --collect-episodes 50 \
  --warmup-episodes 200 \
  --collect-max-steps 3600 \
  --sim-buffer-after-last-depart 1000 \
  --marl-episodes 100 \
  --marl-dreamer-actor-horizon 8 \
  --marl-grpo-route-cost-weight 0.05 \
  --marl-grpo-rollout-weight 1.0 \
  --marl-grpo-rollout-group-size 4 \
  --marl-grpo-policy-kl-coef 0.0 \
  --marl-grpo-combined-advantage-clip 5.0 \
  --marl-vehicle-obs-include-route-costs \
  --marl-vehicle-obs-include-fleet-pressure \
  --checkpoint-episodes 100 \
  --eval-test-eps 1 \
  --marl-seed 48 \
  --device cuda:0 \
  2>&1 | tee logs/gretz_armainvilliers/full_cycle_20260526_seed48.log
