#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-/home/wangwenrui/opt/miniconda3/envs/URB/bin/python}"
output_root="${HDV_RESPONSE_OUTPUT_ROOT:-experiment_result/hdv_od_route_predictor_train_fit_v4_20260723}"
requested_cities=("$@")
if (( ${#requested_cities[@]} == 0 )); then
  requested_cities=(gretz ingolstadt fontenay)
fi

common_args=(
  --route-selection-split train
  --route-response-model residual
  --route-epochs 300
  --patience 150
  --route-lr 0.01
  --weight-decay 0
  --gate-penalty 0
  --route-hidden-dim 128
  --route-depth 2
  --residual-scales 0,0.25,0.5,1,2,4
  --bootstrap-samples 2000
  --bootstrap-block-size 10
  --seed 42
  --quiet
)

run_gretz() {
  "${python_bin}" scripts/train_hdv_response_predictor_offline.py \
    --run-dir results/gretz_armainvilliers_online_wm_mainline_reuse8k_cachemerge_b4_450ep_resume150_20260718/online_wm_mainline \
    --edge-registry networks/gretz_armainvilliers/edge_registry.json \
    --initial-hdv-state results/gretz_armainvilliers_online_wm_mainline_reuse8k_cachemerge_b4_150ep_20260717/online_wm_mainline/hdv_learning_state_after_online_wm_train.json \
    --legacy-checkpoint results/gretz_armainvilliers_online_wm_mainline_reuse8k_cachemerge_b4_450ep_resume150_20260718/online_wm_mainline/checkpoints/checkpoint_ep00300.pt \
    --output-dir "${output_root}/gretz" \
    --train-end-episode 250 \
    --validation-end-episode 300 \
    --test-start-episode 301 \
    --progress-denominator 450 \
    "${common_args[@]}"
}

run_ingolstadt() {
  "${python_bin}" scripts/train_hdv_response_predictor_offline.py \
    --run-dir results/ingolstadt_custom_online_wm_mainline_reuse8k_cachemerge_b4_450ep_resume150_20260718/online_wm_mainline \
    --edge-registry networks/ingolstadt_custom/edge_registry.json \
    --initial-hdv-state results/ingolstadt_custom_online_wm_mainline_reuse8k_cachemerge_b4_150ep_20260717/online_wm_mainline/hdv_learning_state_after_online_wm_train.json \
    --legacy-checkpoint results/ingolstadt_custom_online_wm_mainline_reuse8k_cachemerge_b4_450ep_resume150_20260718/online_wm_mainline/checkpoints/checkpoint_ep00300.pt \
    --output-dir "${output_root}/ingolstadt" \
    --train-end-episode 250 \
    --validation-end-episode 300 \
    --test-start-episode 301 \
    --progress-denominator 450 \
    "${common_args[@]}"
}

run_fontenay() {
  "${python_bin}" scripts/train_hdv_response_predictor_offline.py \
    --run-dir worldmodel_AAAI_2027/artifacts/fontenay_mainline_results_ckpt100_150_20260721/results/fontenay_en_parisis_online_wm_fontenay_reusewarmup_b4_150ep_server/online_wm_mainline \
    --edge-registry networks/fontenay_en_parisis/edge_registry.json \
    --initial-hdv-state networks/fontenay_en_parisis/post_mutation_hdv_learning_state.json \
    --legacy-checkpoint worldmodel_AAAI_2027/artifacts/fontenay_mainline_results_ckpt100_150_20260721/results/fontenay_en_parisis_online_wm_fontenay_reusewarmup_b4_150ep_server/online_wm_mainline/checkpoints/checkpoint_ep00100.pt \
    --output-dir "${output_root}/fontenay" \
    --train-end-episode 50 \
    --validation-end-episode 100 \
    --test-start-episode 101 \
    --progress-denominator 150 \
    "${common_args[@]}"
}

for city in "${requested_cities[@]}"; do
  case "${city}" in
    gretz) run_gretz ;;
    ingolstadt) run_ingolstadt ;;
    fontenay) run_fontenay ;;
    *)
      echo "unknown city '${city}'; expected gretz, ingolstadt, or fontenay" >&2
      exit 2
      ;;
  esac
done
