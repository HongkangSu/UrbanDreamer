# Gretz Route0-Break Incident Interim Report

Date: 2026-05-31

## Incident setting

- Network: `gretz_armainvilliers`
- Probe tag: `gretz_armainvilliers_current_to_dest_incident_route0break_20260531`
- Full-cycle tag: `gretz_armainvilliers_current_to_dest_incident_route0break_full_20260531`
- Incident edge: `-221452293#6`
- Incident window: `[500, 2500)`
- Incident speed: `1.0 m/s`
- Candidate semantics:
  - Collection / MARL vehicle candidate source: `current_to_dest`
  - WM candidate edge cost source: `freeflow`
  - SUMO eval candidate source: `current_shortest_freeflow`

## Baseline probe

The probe reused an existing post-mutation HDV state and did not rerun HDV
adaptation. `collector_eval_summary.json` reports
`effective_human_learning_episodes = 0` and `summary_episode = 201`.

| Policy | t_all | t_CAV | t_HDV | completion_rate | route0 | route1 | route2 | route3 | route0_share | fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `freeflow_greedy` | 2.882545 | 2.758268 | 2.965614 | 0.996855 | 1110 | 0 | 0 | 0 | 1.000000 | 0 |
| `live_current_tt` | 2.837855 | 2.660958 | 2.956096 | 0.996855 | 980 | 81 | 10 | 4 | 0.911628 | 0 |

Acceptance readout:

- `live_current_tt` route0 share is `980 / 1075 = 91.16%`, below the required 95% threshold.
- Non-route0 choices are present: route1 81, route2 10, route3 4.
- `live_current_tt` improves `t_all` by `0.044690 min` over `freeflow_greedy`.
- `candidate_fallback_total = 0` for both baselines.
- SUMO stats report `teleports total = 0` for both baselines.

Conclusion: the incident passes the baseline gate and provides measurable
dynamic rerouting space without changing candidate semantics.

## Full-cycle run

The initial clean full-cycle run
`gretz_armainvilliers_current_to_dest_incident_route0break_full_20260531`
was started with a fresh 200-episode HDV warmup, but that would take roughly
the same wall-clock time as previous Gretz collections. It was stopped during
warmup and its status file was marked `STATE='cancelled'`.

The active full training pipeline now reuses the previous Gretz 200-episode
post-mutation bundle, then runs the new route0-break incident collection and
training. This is a deliberate runtime tradeoff: the incident differs from the
old Gretz incident, so the HDV state is not a perfect re-adaptation for the new
edge, but it avoids rerunning the expensive 200-episode Gretz warmup.

- Active tag: `gretz_armainvilliers_current_to_dest_incident_route0break_reusepost_20260531`
- tmux session: `gretz_route0break_reusepost_20260531`
- launch script: `tmp/run_gretz_route0break_reusepost_20260531.sh`
- log file: `logs/gretz_armainvilliers_current_to_dest_incident_route0break_reusepost_20260531/full_cycle.log`
- status file: `logs/gretz_armainvilliers_current_to_dest_incident_route0break_reusepost_20260531/urb_fair_full_cycle.status.env`
- reused post-mutation state:
  `results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/collect_urb_fair_eval_postmutation_records/post_mutation_state.json`
- reused HDV state:
  `results/gretz_armainvilliers_current_to_dest_incident_20260530_odcompact/collect_urb_fair_eval_postmutation_records/post_mutation_hdv_learning_state.json`

Planned full-cycle settings:

- warmup episodes: loaded from previous Gretz post-mutation bundle; effective new warmup is 0
- collection episodes: 50
- WM max steps: 8000
- MARL episodes: 400
- checkpoint eval episodes: 50, 100, 200, 400
- SUMO eval test episodes per checkpoint: 1
- process-visible device: `cuda:0`
- launcher maps physical GPU 1 through `CUDA_VISIBLE_DEVICES=1`

At startup, the active status file showed:

```text
STAGE='collect_urb_fair'
STATE='running'
```

The collect stage confirmed:

- Incident: `edge=-221452293#6 window=[500.0, 2500.0) speed=1.00m/s`
- `av_candidate_source=current_to_dest`
- `random_av_fraction=0.6`
- `epsilon=0.4`
- `Warmup episodes: 0`
- `[HDV state] loaded 636 agent entries`
- `=== Phase 1: Reusing post-mutation bundle ===`
- `Human learning after mutation: disabled`

## Code change

`scripts/run_current_to_dest_candidate_full_cycle.py` now passes incident edges
as a single argparse token:

```text
--incident-edge=-221452293#6
```

This is required for dash-prefixed SUMO edge IDs. Without it, the child
argparse parser can misread the edge ID as an option.

Verification:

```text
/home/wangwenrui/opt/miniconda3/envs/URB/bin/python -m pytest tmp/test_candidate_full_cycle_runner.py
8 passed
```

## Remaining work

- Wait for background full-cycle completion.
- Verify collection produced enough flow shards and clean post-mutation state.
- Inspect WM train/eval loss from `dynamics_urb_fair_mainline_8k`.
- Inspect MARL entropy/reward/critic traces through episode 50 and 400.
- Read checkpoint SUMO eval summaries for 50, 100, 200, 400.
- Update this report with checkpoint comparison and best policy selection.
