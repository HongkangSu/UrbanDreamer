# Gretz Route0-Break Incident Final Report

Date: 2026-05-31

## Run status

Active tag:

```text
gretz_armainvilliers_current_to_dest_incident_route0break_reusepost_20260531
```

The full cycle completed successfully:

```text
STAGE='complete'
STATE='succeeded'
```

The run reused the previous Gretz 200-episode post-mutation bundle and skipped
new HDV warmup under the route0-break incident:

- reused post-mutation state:
  `results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/collect_urb_fair_eval_postmutation_records/post_mutation_state.json`
- reused HDV state:
  `results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/collect_urb_fair_eval_postmutation_records/post_mutation_hdv_learning_state.json`
- effective new warmup: `0`

Incident:

```text
edge=-221452293#6
window=[500,2500)
speed=1.0m/s
```

Candidate semantics:

```text
collection / MARL vehicle_candidate_source = current_to_dest
wm_candidate_edge_cost_source = freeflow
eval_candidate_source = current_shortest_freeflow
```

## Baseline gate

The post-state baseline probe passed:

| Policy | t_all | t_CAV | t_HDV | completion | route0 | route1 | route2 | route3 | route0_share | fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `freeflow_greedy` | 2.882545 | 2.758268 | 2.965614 | 0.996855 | 1110 | 0 | 0 | 0 | 1.000000 | 0 |
| `live_current_tt` | 2.837855 | 2.660958 | 2.956096 | 0.996855 | 980 | 81 | 10 | 4 | 0.911628 | 0 |

This established non-route0 dynamic routing space before training.

## Collection

Collection completed:

```json
{
  "warmup_episodes": 0,
  "num_episodes": 50,
  "episodes_saved": 50,
  "partial": false,
  "should_humans_adapt": false
}
```

Flow artifacts:

- `data/gretz_armainvilliers_current_to_dest_incident_route0break_reusepost_20260531/collect_urb_fair_eval_postmutation/gretz_armainvilliers_urb_fair_eval_postmutation.pt`
- `data/gretz_armainvilliers_current_to_dest_incident_route0break_reusepost_20260531/collect_urb_fair_eval_postmutation/flows/gretz_armainvilliers_urb_fair_eval_postmutation/shard_0000.pt`
- `data/gretz_armainvilliers_current_to_dest_incident_route0break_reusepost_20260531/collect_urb_fair_eval_postmutation/flows/gretz_armainvilliers_urb_fair_eval_postmutation/shard_0001.pt`
- AV decision trace:
  `data/gretz_armainvilliers_current_to_dest_incident_route0break_reusepost_20260531/collect_urb_fair_eval_postmutation/gretz_armainvilliers_urb_fair_eval_postmutation_av_decision_trace.csv`

## World model

WM artifacts exist:

- `logs/gretz_armainvilliers_current_to_dest_incident_route0break_reusepost_20260531/dynamics_urb_fair_mainline_8k/flow_projector_latest.pt`
- `logs/gretz_armainvilliers_current_to_dest_incident_route0break_reusepost_20260531/dynamics_urb_fair_mainline_8k/latest.pt`

No traceback/error was found in `full_cycle.log`.

## MARL training

MARL produced checkpoints through episode 400:

```text
checkpoint_ep00050.pt
checkpoint_ep00100.pt
checkpoint_ep00150.pt
checkpoint_ep00200.pt
checkpoint_ep00250.pt
checkpoint_ep00300.pt
checkpoint_ep00350.pt
checkpoint_ep00400.pt
latest.pt
```

Late-training logs show entropy did not collapse; examples near the end:

```text
ep00350 ent=1.2863
ep00360 ent=1.3028
ep00370 ent=1.3065
ep00380 ent=1.3111
ep00390 ent=1.2913
```

There were repeated `actor_anchor_reject` entries, so actor updates were
strongly constrained by the anchor/trust-region logic late in training.

## Checkpoint SUMO eval

All requested checkpoint evals completed:

| Checkpoint | t_all | t_CAV | t_HDV | completion | route0 | route1 | route2 | route3 | fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ep00050 | 2.859175 | 2.707021 | 2.960877 | 0.996855 | 768 | 156 | 103 | 78 | 0 |
| ep00100 | 2.880731 | 2.757677 | 2.962982 | 0.996855 | 922 | 162 | 21 | 10 | 0 |
| ep00200 | 2.856770 | 2.735792 | 2.936766 | 0.963836 | 1058 | 149 | 36 | 15 | 0 |
| ep00400 | 2.885410 | 2.770604 | 2.962149 | 0.996855 | 926 | 136 | 31 | 8 | 0 |

Best checkpoint by `t_all`:

```text
logs/gretz_armainvilliers_current_to_dest_incident_route0break_reusepost_20260531/marl_urb_fair_mainline_500ep/checkpoint_ep00200.pt
```

Best eval summary:

```json
{
  "t_all": 2.8567699836867866,
  "t_CAV": 2.735792349726776,
  "t_HDV": 2.9367660343270097,
  "completion_rate": 0.9638364779874213,
  "dynamic_selected_route_counts": {
    "route0": 1058,
    "route1": 149,
    "route2": 36,
    "route3": 15
  },
  "candidate_dynamic_total": 1258,
  "candidate_fallback_total": 0,
  "candidate_unique_mean": 3.5133615255355837
}
```

## Independent trained-policy test

After checkpoint selection, the best trained MARL policy was tested again with
5 SUMO test episodes:

```text
policy=logs/gretz_armainvilliers_current_to_dest_incident_route0break_reusepost_20260531/marl_urb_fair_mainline_500ep/checkpoint_ep00200.pt
results=results/gretz_armainvilliers_current_to_dest_incident_route0break_reusepost_20260531/eval_best_ep200_5eps_20260601
test_eps=5
deterministic=true
post_mutation_state_in=results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/collect_urb_fair_eval_postmutation_records/post_mutation_state.json
incident=edge:-221452293#6 window:[500,2500) speed:1.0m/s
```

Summary:

```json
{
  "t_all": 2.8569874932028276,
  "t_CAV": 2.735655737704918,
  "t_HDV": 2.9372177055103887,
  "completion_rate": 0.9638364779874213,
  "dynamic_selected_route_counts": {
    "route0": 5290,
    "route1": 745,
    "route2": 180,
    "route3": 75
  },
  "live_route_best_counts": {
    "route0": 5805,
    "route1": 410,
    "route2": 65,
    "route3": 10
  },
  "candidate_dynamic_total": 6290,
  "candidate_fallback_total": 0,
  "candidate_unique_mean": 3.5133615255355837
}
```

Per-episode metrics were stable across the 5 deterministic eval episodes:

| Episode | t_all | t_CAV | t_HDV | completion | unfinished |
| --- | ---: | ---: | ---: | ---: | ---: |
| 201 | 2.856770 | 2.735792 | 2.936766 | 0.963836 | 23 |
| 202 | 2.856770 | 2.735792 | 2.936766 | 0.963836 | 23 |
| 203 | 2.856987 | 2.735656 | 2.937218 | 0.963836 | 23 |
| 204 | 2.856987 | 2.735656 | 2.937218 | 0.963836 | 23 |
| 205 | 2.856987 | 2.735656 | 2.937218 | 0.963836 | 23 |

Comparison to the baseline gate:

| Policy | Test eps | t_all | t_CAV | t_HDV | completion | fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `freeflow_greedy` | 1 | 2.882545 | 2.758268 | 2.965614 | 0.996855 | 0 |
| `live_current_tt` | 1 | 2.837855 | 2.660958 | 2.956096 | 0.996855 | 0 |
| `marl_ep200` | 5 | 2.856987 | 2.735656 | 2.937218 | 0.963836 | 0 |

This confirms the trained ep200 policy reproducibly beats the route0/freeflow
greedy baseline on travel time, but still trails the live-current greedy
baseline and carries the same completion-rate risk seen in the original
single-episode sweep.

Best-selection artifact:

```text
results/gretz_armainvilliers_current_to_dest_incident_route0break_reusepost_20260531/eval_urb_fair_mainline_sweep/best_checkpoint_selection.json
```

## Interpretation

- The learned ep200 policy improves over the route0/freeflow-greedy baseline
  on `t_all`: `2.856987` in the 5-episode retest vs `2.882545`.
- It does not beat the live-current greedy upper baseline: `2.856987` vs
  `2.837855`.
- Completion is weaker for the selected ep200 checkpoint (`0.963836`) than for
  the greedy baselines and the other learned checkpoints (`0.996855`). This is
  the main risk in calling ep200 a clean win.
- ep50 is very close in `t_all` (`2.859175`) and has better completion
  (`0.996855`) plus much stronger non-route0 use.
- Because this run reused the old Gretz post-mutation HDV state under a changed
  incident, HDV behavior is not fully re-equilibrated to the route0-break
  incident. Treat results as a fast retrain/probe, not the strictest final
  experimental condition.

## Follow-up

Recommended next comparison:

1. Evaluate ep50 and ep200 over more than one SUMO seed/test episode.
2. Rank with a completion-aware metric, not only `t_all`, because ep200 has
   lower completion.
3. If ep50 remains stable, use it as the practical candidate; if ep200's
   incomplete vehicles vanish with more eval seeds, keep ep200.
