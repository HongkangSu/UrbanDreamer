#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="/home/wangwenrui/opt/miniconda3/envs/URB/bin/python"
ROOT="${ROOT:-ingolstadt_custom_current_to_dest_incident_20260528}"
NETWORK_DIR="${NETWORK_DIR:-networks/ingolstadt_custom}"
NETWORK_NAME="${NETWORK_NAME:-ingolstadt_custom}"
FLOW_TASK_NAME="${FLOW_TASK_NAME:-${NETWORK_NAME}_urb_fair_eval_postmutation}"
RATIO_MACHINES="${RATIO_MACHINES:-0.4}"
DATA_DIR="${DATA_DIR:-data/${ROOT}/collect_urb_fair_eval_postmutation}"
RECORDS_DIR="${RECORDS_DIR:-results/${ROOT}/collect_urb_fair_eval_postmutation_records}"
WM_DIR="${WM_DIR:-logs/${ROOT}/dynamics_urb_fair_mainline_8k}"
SETTINGS_DIR="${SETTINGS_DIR:-.urbandreamer/generated/${ROOT}}"
SEEDS="${SEEDS:-43 48 52}"
MARL_EPISODES="${MARL_EPISODES:-200}"
MODE="${MODE:-ac}"

case "${MODE}" in
  ac)
    GRPO_ROUTE_WEIGHT="${GRPO_ROUTE_WEIGHT:-0.0}"
    GRPO_ROLLOUT_WEIGHT="${GRPO_ROLLOUT_WEIGHT:-0.0}"
    GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-1}"
    ROLLOUT_REWARD_WEIGHT="${ROLLOUT_REWARD_WEIGHT:-0.0}"
    ;;
  grpo)
    GRPO_ROUTE_WEIGHT="${GRPO_ROUTE_WEIGHT:-0.05}"
    GRPO_ROLLOUT_WEIGHT="${GRPO_ROLLOUT_WEIGHT:-1.0}"
    GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-4}"
    ROLLOUT_REWARD_WEIGHT="${ROLLOUT_REWARD_WEIGHT:-0.0}"
    ;;
  *)
    echo "Unsupported MODE=${MODE}; expected ac or grpo" >&2
    exit 2
    ;;
esac

for seed in ${SEEDS}; do
  RUN_DIR="logs/${ROOT}/marl_${MODE}_current_to_dest_seed${seed}_ep${MARL_EPISODES}"
  EVAL_DIR="results/${ROOT}/eval_${MODE}_current_to_dest_seed${seed}_ep${MARL_EPISODES}"
  mkdir -p "${RUN_DIR}" "${EVAL_DIR}"

  "${PY}" -u dreamer4/urban_routing/train_marl.py \
    --settings "${SETTINGS_DIR}/urb_fair_marl.env" \
    --data_dir "${DATA_DIR}" \
    --meta_file "${FLOW_TASK_NAME}.pt" \
    --projector_ckpt "${WM_DIR}/flow_projector_latest.pt" \
    --dynamics_ckpt "${WM_DIR}/latest.pt" \
    --edge_registry "${NETWORK_DIR}/edge_registry.json" \
    --network_folder "${NETWORK_DIR}" \
    --network_name "${NETWORK_NAME}" \
    --agents_csv "${NETWORK_DIR}/agents.csv" \
    --ratio_machines "${RATIO_MACHINES}" \
    --checkpoint_dir "${RUN_DIR}" \
    --training_family dreamer_ctde \
    --wm_cfm_init_mode zero \
    --wm_policy_obs_source decoder_volume_tt \
    --wm_edge_cost_source decoded_speed \
    --wm_candidate_edge_cost_source decoded_speed \
    --vehicle_candidate_source current_to_dest \
    --vehicle_path_cache_mode lazy \
    --vehicle_actor_architecture route_shared \
    --vehicle_routes_csv "${RECORDS_DIR}/routes.csv" \
    --vehicle_reward_scale 1.0 \
    --vehicle_obs_include_route_costs \
    --vehicle_obs_include_fleet_pressure \
    --dreamer_actor_objective discrete_actor_critic \
    --gamma 1.0 \
    --gae_lambda 1.0 \
    --dreamer_all_reward_weight 0.0 \
    --dreamer_remaining_time_progress_weight 0.0 \
    --dreamer_route_cost_reward_weight 0.0 \
    --dreamer_rollout_cost_reward_weight "${ROLLOUT_REWARD_WEIGHT}" \
    --dreamer_ac_step_advantage_weight "${AC_STEP_WEIGHT:-0.1}" \
    --dreamer_ac_route_advantage_weight "${AC_ROUTE_WEIGHT:-1.0}" \
    --dreamer_ac_route_credit_mode "${AC_ROUTE_CREDIT_MODE:-route_cost}" \
    --dreamer_ac_route_credit_base_cost_mode "${AC_BASE_COST_MODE:-progress_fraction}" \
    --dreamer_ac_route_advantage_temperature 60.0 \
    --dreamer_grpo_route_cost_weight "${GRPO_ROUTE_WEIGHT}" \
    --dreamer_grpo_rollout_weight "${GRPO_ROLLOUT_WEIGHT}" \
    --dreamer_grpo_rollout_group_size "${GRPO_GROUP_SIZE}" \
    --dreamer_grpo_policy_clip_ratio "${GRPO_CLIP_RATIO:-0.2}" \
    --dreamer_grpo_policy_kl_coef "${GRPO_KL_COEF:-0.0}" \
    --dreamer_grpo_combined_advantage_clip "${GRPO_ADV_CLIP:-5.0}" \
    --dreamer_grpo_rollout_baseline_policy "${GRPO_BASELINE_POLICY:-none}" \
    --dreamer_hdv_background_mode episode_hdv_head_response \
    --dreamer_hdv_background_state "${RECORDS_DIR}/hdv_learning_state_final.json" \
    --dreamer_hdv_background_weight 1.0 \
    --dreamer_hdv_background_combine additive \
    --dreamer_hdv_response_head_fallback error \
    --dreamer_hdv_response_pressure_weight 60.0 \
    --dreamer_hdv_response_temperature 60.0 \
    --dreamer_hdv_response_prior_mix 0.0 \
    --terminal_tall_reward_weight 0.0 \
    --dreamer_entropy_coef "${ENTROPY_COEF:-0.001}" \
    --dreamer_action_mode "${ACTION_MODE:-expected}" \
    --dreamer_action_temperature "${ACTION_TEMPERATURE:-1.0}" \
    --dreamer_critic_aggregation global_attention \
    --dreamer_critic_attention_heads 4 \
    --n_total_episodes "${MARL_EPISODES}" \
    --dreamer_batch_episodes "${BATCH_EPISODES:-8}" \
    --dreamer_actor_horizon "${ACTOR_HORIZON:-8}" \
    --dreamer_departure_cap_mode none \
    --seed "${seed}" \
    --log_every 5 \
    --checkpoint_every "${CHECKPOINT_EVERY:-50}" \
    --device cuda:0 \
    2>&1 | tee "${RUN_DIR}/train.log"

  "${PY}" -u scripts/eval_urb_fair.py \
    --settings "${SETTINGS_DIR}/urb_fair_eval.env" \
    --policy_ckpt "${RUN_DIR}/checkpoint_ep$(printf '%05d' "${MARL_EPISODES}").pt" \
    --projector_ckpt "${WM_DIR}/flow_projector_latest.pt" \
    --dynamics_ckpt "${WM_DIR}/latest.pt" \
    --post_mutation_state_in "${RECORDS_DIR}/post_mutation_state.json" \
    --results_dir "${EVAL_DIR}/checkpoint_ep$(printf '%05d' "${MARL_EPISODES}")" \
    --test_eps "${EVAL_TEST_EPS:-1}" \
    --device cuda:0 \
    --deterministic \
    --wm_edge_cost_source decoded_speed \
    --wm_candidate_edge_cost_source decoded_speed \
    --av_candidate_source current_shortest_wm \
    2>&1 | tee "${EVAL_DIR}/eval.log"
done
