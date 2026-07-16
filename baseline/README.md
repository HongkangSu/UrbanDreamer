# Online Actor-Critic Baseline

This folder contains the real-SUMO online actor-critic baseline used for
downstream training without the world model's CFM transition or imagined
rollouts. Its legacy no-argument Gretz scenario matches
`experiment_result/gretz_stronglivebad_completion100_full_cycle_20260602.md`:
incident edge `-986733117#1`, window `[500,2500)`, speed `0.1m/s`, with the
reused `20260530_odcompact` post-mutation state and HDV state.

The real-network baseline keeps the frozen world-model encoder and policy
observation decoder so the actor sees the same route-local observation family
as the Gretz run. It does not use the world-model transition, model rollout,
MPC, or reranking for actor-critic updates. The toy adapter uses live edge
features instead and is shared by the IPPO, MAPPO, and IQL runners.

The current cross-network baseline recipe uses:

- `dreamer_route_cost_reward_scope=vehicle_critic`
- `dreamer_route_cost_reward_state=current`
- `dreamer_route_cost_reward_mode=capped_step_cost`
- `dreamer_route_cost_reward_weight=1.0`
- `vehicle_reward_scale=1.0`
- replay buffer of 8 real episodes with 2 update epochs
- full remaining real-trajectory returns (`return_horizon_steps=0`)
- stochastic policy sampling during training and deterministic masked argmax
  during evaluation
- online HDV learning from the restored post-mutation state

For a valid selected route the capped live SUMO reward is:

```text
-min(step_interval, remaining_route_tt) / reward_temperature
```

Invalid routes receive zero reward.  Final SUMO travel times (`t_all`, `t_CAV`,
`t_HDV`) are evaluation metrics, not training rewards.

Dry-run a real-network configuration:

```bash
python baseline/run_online_actor_critic.py \
  --settings .urbandreamer/generated/<tag>/urb_fair_eval.env \
  --post-mutation-state-in results/<tag>/collect_urb_fair_eval_postmutation_records/post_mutation_state.json \
  --hdv-state-in results/<tag>/collect_urb_fair_eval_postmutation_records/post_mutation_hdv_learning_state.json \
  --dry-run
```

Train without running the post-training evaluation phase:

```bash
python baseline/run_online_actor_critic.py <scenario arguments> \
  --train-episodes 500 \
  --checkpoint-episodes 100,200,300,400,500 \
  --reward-mode capped_step_cost \
  --replay-buffer-episodes 8 \
  --replay-update-epochs 2 \
  --return-horizon-steps 0 \
  --real-policy-route-selection sample \
  --keep-human-learning-after-mutation \
  --skip-eval
```

Train the replay-buffer variant currently used for stability probing:

```bash
bash tmp/run_gretz_no_cfm_replay_20260609.sh
```

New cross-network outputs are written under:

```text
results/<tag>/baselines_online_ac_500ep_hdvlearn_<stamp>/online_ac/
```

The checkpoint format is compatible with `scripts/eval_sumo_policy.py` and uses
`model_family="dreamer_ctde"`.
