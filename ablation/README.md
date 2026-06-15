# Gretz Online No-CFM Actor-Critic Ablation

This folder contains the Gretz-only ablation requested for downstream
actor-critic training without using the world model's CFM transition during
training.  Its default Gretz scenario matches
`experiment_result/gretz_stronglivebad_completion100_full_cycle_20260602.md`:
incident edge `-986733117#1`, window `[500,2500)`, speed `0.1m/s`, with the
reused `20260530_odcompact` post-mutation state and HDV state.

The ablation keeps the existing world-model encoder and policy observation
decoder so the actor sees the same route-local observation family as the offline
Gretz run.  It does not use the world model transition, model rollout, MPC, or
reranking for the actor-critic update.  By default each update uses one real
SUMO episode.  Replay-buffer variants can reuse a bounded window of recent real
SUMO episodes with `--replay-buffer-episodes`, still without imagined rollout.

The online training reward matches the offline downstream AC semantics:

- `dreamer_route_cost_reward_scope=vehicle_critic`
- `dreamer_route_cost_reward_state=current`
- `dreamer_route_cost_reward_mode=progress_fraction`
- `dreamer_route_cost_reward_weight=1.0`
- `vehicle_reward_scale=1.0`

For a valid selected route the live SUMO reward is:

```text
min(step_interval, remaining_route_tt) / (elapsed_time + remaining_route_tt)
```

Invalid routes receive zero reward.  Final SUMO travel times (`t_all`, `t_CAV`,
`t_HDV`) are evaluation metrics, not training rewards.

Dry run:

```bash
python ablation/run_gretz_online_no_cfm_ablation.py --dry-run
```

Train 100 online SUMO episodes and evaluate checkpoints 25/50/100 for three
episodes each:

```bash
python ablation/run_gretz_online_no_cfm_ablation.py
```

Train the replay-buffer variant currently used for stability probing:

```bash
bash tmp/run_gretz_no_cfm_replay_20260609.sh
```

Outputs are written under:

```text
results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/ablation_online_no_cfm_ac/
```

The checkpoint format is compatible with `scripts/eval_sumo_policy.py` and uses
`model_family="dreamer_ctde"`.
