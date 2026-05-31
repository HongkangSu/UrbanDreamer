# Gretz Current-to-Destination Full-Cycle Result

Date: 2026-05-31

Network/tag:
`gretz_armainvilliers_current_to_dest_incident_20260530_odcompact`

## Mainline Setting

This run used the current per-vehicle critic mainline:

- Train candidate source: `current_to_dest`
- Eval candidate source: `current_shortest_freeflow`
- Candidate path generation edge cost: `freeflow`
- Route-cost reward scope: `vehicle_critic`
- Actor objective: `discrete_actor_critic`
- Vehicle critic: enabled
- Vehicle route-Q critic: disabled
- Legacy actor-only route rewards/losses: disabled
- `gamma=1.0`, `gae_lambda=1.0`

The reward-path invariant was satisfied in the MARL diagnosis:

- `legacy_actor_only_enabled=false`
- no legacy actor-only lines were detected
- mainline log marker reports `vehicle_critic=True` and
  `vehicle_route_q=False`

## Full-Cycle Status

The full-cycle status file reports:

```text
STAGE='complete'
STATE='succeeded'
```

Collection, WM training, MARL training, and SUMO eval all completed.

## Collection Quality

The collection readiness check reports warmup and collection complete:

- warmup: `200/200`
- saved collection episodes: `50/50`
- collection manifest: present
- collected `.pt` artifacts: present

Collection signal diagnostics reported no warnings:

| Metric | Value |
| --- | ---: |
| unique episode count | 50 |
| reward mean | -4.2555365562 |
| reward std | 0.3566047847 |
| AV active edge fraction | 0.4251950979 |
| AV row-sum mean | 5.0044937134 |
| HDV active edge fraction | 0.2309921980 |
| OD groups | 629 |
| route count per OD group | 4 |
| route0 is free-flow best fraction | 1.0 |
| nonmonotonic free-flow group fraction | 0.5771065183 |

Interpretation: free-flow Dijkstra candidates are sorted with route0 as the
free-flow best route. Nonmonotonic free-flow groups still exist because the
raw route alternatives are not monotonically ordered by all downstream cost
notions, but route0 is consistently the free-flow best candidate under the
current candidate-generation semantics.

## World Model

The WM recovered from the earlier OOM by using smaller sequence/batch settings
and completed to the expected 8000-step checkpoint.

WM diagnosis status:

- status: `healthy_so_far`
- warnings: none
- completed step: `8000`
- final checkpoint saved: yes
- `latest.pt`: present
- `flow_projector_latest.pt`: present

Selected final/recent WM metrics:

| Metric | Last | Best |
| --- | ---: | ---: |
| loss | 1.699115 | 1.681939 |
| flow_mse | 1.035574 | 1.035574 |
| hdv_ce | 8.313459 | 8.313459 |
| dec_mh | 0.066397 | 0.058912 |
| pred_next_dec | 0.141950 | 0.108828 |
| pred_next_speed_mse | 0.108122 | 0.061456 |
| t_all_mse | 0.183474 | 0.183474 |
| t_all_raw_mse | 3.195169 | 3.195169 |

Interpretation: WM training produced usable checkpoints with no nonfinite or
save-failure warnings. Some prediction metrics are not at their best at the
last logged row, but the overall diagnosis is healthy and the final checkpoint
exists.

## Offline MARL Convergence

MARL produced checkpoints:

- `checkpoint_ep00100.pt`
- `checkpoint_ep00200.pt`
- `checkpoint_ep00300.pt`
- `checkpoint_ep00400.pt`
- `latest.pt`

The convergence diagnosis status is `healthy_so_far` with no warnings.

Key convergence signals:

| Signal | Value |
| --- | ---: |
| first50 mean entropy | 1.38614 |
| last logged entropy | 1.3722 |
| recent20 mean entropy | 1.37586 |
| last route0 argmax fraction | 0.73 |
| last nonzero-route argmax fraction | 0.27 |
| recent20 route0 argmax fraction | 0.8015 |
| last vehicle reward mean | 0.13 |
| last vehicle target std | 0.28 |
| last vehicle return std | 0.27 |

Interpretation: there was no fast entropy collapse. The policy did not simply
lock to route0, and the per-vehicle critic target/reward signal remained
nonzero.

## SUMO Eval

Eval directory:
`results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/eval_urb_fair_mainline_sweep`

Each checkpoint was evaluated with `test_eps=1`.

| Checkpoint | t_all | t_CAV | t_HDV | Rank |
| --- | ---: | ---: | ---: | ---: |
| ep200 | 2.7366456362 | 2.5756561680 | 2.8442543860 | 1 |
| ep100 | 2.7379600421 | 2.5793963255 | 2.8439473684 | 2 |
| ep400 | 2.9672712934 | 3.1448818898 | 2.8485526316 | 3 |

Best checkpoint:

```text
checkpoint_ep00200.pt
t_all = 2.73664563617245
```

The numeric `t_all <= 4.35` gate is met for Gretz, but this is not SOTA on
Gretz after adding the two requested greedy baselines below.

## Gretz Baseline Comparison

Baseline script:
`baseline/eval_gretz_current_to_dest_freeflow_candidates.sh`

Both baselines use the same free-flow generated candidate paths as the learned
policy:

- `--av_candidate_source current_shortest_freeflow`
- `--wm_candidate_edge_cost_source freeflow`
- `--wm_edge_cost_source decoded_speed`
- reused Gretz post-mutation and HDV learning state
- `test_eps=1`

| Run | t_all | t_CAV | t_HDV | Route counts | Route0 share |
| --- | ---: | ---: | ---: | --- | ---: |
| `baseline_live_current_greedy` | 2.7317034700 | 2.5639107612 | 2.8438596491 | r0=1029, r1=11, r2=1, r3=0 | 98.85% |
| `baseline_freeflow_greedy` | 2.7324921136 | 2.5666666667 | 2.8433333333 | r0=1032, r1=0, r2=0, r3=0 | 100.00% |
| `learned_ep200` | 2.7366456362 | 2.5756561680 | 2.8442543860 | r0=1017, r1=10, r2=0, r3=1 | 98.93% |
| `learned_ep100` | 2.7379600421 | 2.5793963255 | 2.8439473684 | r0=1003, r1=19, r2=0, r3=3 | 97.85% |
| `learned_ep400` | 2.9672712934 | 3.1448818898 | 2.8485526316 | r0=704, r1=231, r2=160, r3=130 | 57.47% |

Differences versus the best learned checkpoint:

- `baseline_live_current_greedy` is better than `learned_ep200` by
  `0.0049421661` t_all.
- `baseline_freeflow_greedy` is better than `learned_ep200` by
  `0.0041535226` t_all.

Interpretation: on Gretz, the current learned policy is valid and close to the
greedy baselines, but it does not beat either requested baseline in the
single-episode eval. Because the best baseline is mostly route0, Gretz appears
to be a weak learning benchmark under this candidate/incident setting: there is
little room for the actor to improve over greedy route0-like behavior, and late
training drift at ep400 is harmful.

## Current Conclusion

The Gretz full cycle proves that the current per-vehicle critic mainline can
complete:

```text
new collection -> new WM -> per-vehicle critic MARL -> SUMO eval
```

and can produce a valid learned policy without direct actor-only reward paths.

The best policy appears around ep100/ep200. ep400 is worse, mostly through CAV
degradation (`t_CAV` rises from about 2.58 to 3.14), so checkpoint selection is
important. This is not the old fast-collapse failure mode; it looks more like
late training drift/overtraining under the current trust/anchor/reward schedule.

After the Gretz baseline comparison, the stronger conclusion is narrower:
Gretz verifies pipeline health and near-greedy policy quality, but it does not
show learned-policy superiority over the two requested greedy baselines.

## Remaining Risks

- Gretz should not be used alone to claim SOTA; the requested baselines are
  slightly better than the learned policy in the current single-episode eval.
- The eval uses `test_eps=1`, so the result is a valid smoke/full-cycle signal
  but not a strong statistical estimate.
- Les is the next full-cycle network to run. Plaisir was stopped per user
  instruction and should not be included in the current validation set.

## Source Evidence

- Status: `logs/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/urb_fair_full_cycle.status.env`
- Collection data:
  `data/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/collect_urb_fair_eval_postmutation`
- Collection records:
  `results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/collect_urb_fair_eval_postmutation_records`
- WM checkpoints:
  `logs/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/dynamics_urb_fair_mainline_8k`
- MARL checkpoints:
  `logs/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/marl_urb_fair_mainline_500ep`
- Eval summaries:
  `results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/eval_urb_fair_mainline_sweep/checkpoint_ep00100/collector_eval_summary.json`
  `results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/eval_urb_fair_mainline_sweep/checkpoint_ep00200/collector_eval_summary.json`
  `results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/eval_urb_fair_mainline_sweep/checkpoint_ep00400/collector_eval_summary.json`
- Baseline eval summaries:
  `results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/baseline_gretz_ep200_ref_policy_freeflow_candidates_freeflow_greedy/collector_eval_summary.json`
  `results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/baseline_gretz_ep200_ref_policy_freeflow_candidates_live_current_greedy/collector_eval_summary.json`
