# Gretz Strong Incident Full-Cycle Result

Date: 2026-06-02

## Selected incident

Network/tag:

```text
gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1
```

Incident:

```text
edge=-986733117#1
window=[500,2500)
speed=0.1m/s
```

The full cycle reused the previous Gretz post-mutation state and did not rerun
HDV warmup:

```text
HDV_ADAPT_EP=0
post_mutation_state_in=results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/collect_urb_fair_eval_postmutation_records/post_mutation_state.json
hdv_state_in=results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/collect_urb_fair_eval_postmutation_records/post_mutation_hdv_learning_state.json
```

Status:

```text
STAGE=complete
STATE=succeeded
```

## Baseline sanity

Both baseline probes completed with 100% arrival and no candidate fallback:

| Policy | t_all | t_CAV | t_HDV | completion | route0 | route1 | route2 | route3 | fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| freeflow_greedy | 4.817820 | 4.133924 | 5.272557 | 1.000000 | 1352 | 0 | 0 | 0 | 0 |
| live_current_tt | 4.689492 | 3.970735 | 5.167408 | 1.000000 | 1271 | 17 | 3 | 0 | 0 |
| live_current_tt + planned pressure p120 | 4.613810 | 3.788517 | 5.162565 | 1.000000 | 1122 | 109 | 36 | 13 | 0 |

This incident is strong enough to raise the route0/freeflow baseline from the
old weak-Gretz level to about 4.8 minutes, while simple live-current greedy is
still route0-dominated.

## Full-cycle training

The full cycle completed and produced WM and MARL artifacts:

```text
logs/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/dynamics_urb_fair_mainline_8k/latest.pt
logs/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/dynamics_urb_fair_mainline_8k/flow_projector_latest.pt
logs/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/marl_urb_fair_mainline_500ep/checkpoint_ep00050.pt
logs/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/marl_urb_fair_mainline_500ep/checkpoint_ep00400.pt
```

Checkpoint evals:

| Checkpoint | t_all | t_CAV | t_HDV | completion | route0 | route1 | route2 | route3 | fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ep00050 | 4.072233 | 5.738833 | 3.014074 | 0.809748 | 1577 | 242 | 491 | 1007 | 0 |
| ep00100 | 6.523716 | 8.673885 | 5.094023 | 1.000000 | 1081 | 255 | 762 | 1161 | 0 |
| ep00200 | 4.304971 | 5.758790 | 3.300437 | 0.911950 | 1015 | 374 | 268 | 871 | 0 |
| ep00400 | 4.105765 | 3.229003 | 4.688743 | 1.000000 | 698 | 207 | 143 | 120 | 0 |

The automatic best-by-`t_all` selector picked ep00050, but that checkpoint is
not acceptable under the strict 100% completion constraint. Applying the 100%
completion gate first, ep00400 is the useful checkpoint:

```text
checkpoint=logs/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/marl_urb_fair_mainline_500ep/checkpoint_ep00400.pt
t_all=4.105765
completion_rate=1.000000
```

## Conclusion

This incident satisfies the requested benchmark shape:

- congestion is strong: freeflow route0 baseline reaches `t_all=4.817820`;
- both requested baseline probes reach `completion_rate=1.0`;
- live greedy is not optimal: the valid learned checkpoint ep00400 reaches
  `t_all=4.105765`, better than both plain live-current greedy (`4.689492`) and
  the pressure sanity baseline (`4.613810`);
- the learned policy is not route0-dominant at ep00400 (`route0=698/1168`,
  about 59.8%).

## Source evidence

- settings:
  `.urbandreamer/generated/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/urb_fair_marl.env`
- status:
  `logs/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/urb_fair_full_cycle.status.env`
- baseline summaries:
  `results/gretz_armainvilliers_current_to_dest_incident_completion100_probe_20260601/neg_986733117_1_S0b_s500_e2500_v0p1/freeflow_greedy_1eps/collector_eval_summary.json`
  `results/gretz_armainvilliers_current_to_dest_incident_completion100_probe_20260601/neg_986733117_1_S0b_s500_e2500_v0p1/live_current_tt_1eps/collector_eval_summary.json`
  `results/gretz_armainvilliers_current_to_dest_incident_completion100_confirm_noteleport_20260601/live_current_tt_planned_pressure_1eps_p120/collector_eval_summary.json`
- MARL eval:
  `results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_urb_fair_mainline_sweep/checkpoint_ep00400/collector_eval_summary.json`
