# Current-to-Destination Free-Flow Candidate Experiment Results

Date: 2026-05-30

Network: `ingolstadt_custom_current_to_dest_incident_20260528`

## Mainline Setting

The current mainline uses free-flow only for candidate path generation:

- Train candidate source: `current_to_dest`
- Eval candidate source: `current_shortest_freeflow`
- Candidate path generation edge cost: `--wm_candidate_edge_cost_source freeflow`
- Route cost / observation / reward cost source: `--wm_edge_cost_source decoded_speed`
- Eval route-cost override for learned policy: `--av_route_cost_policy none`

So "free-flow candidate" means: generate current-to-destination candidate paths with free-flow Dijkstra, then still evaluate those candidate routes using decoded-speed costs for observations and critic reward. It does not mean using free-flow cost as the actor-critic reward.

## Bottom Line

The controlled `seed98` run shows that the current free-flow candidate mainline can converge to a strong policy:

| Run | Role | t_all | t_CAV | t_HDV | Selected route counts |
| --- | --- | ---: | ---: | ---: | --- |
| `seed98_freeflow_candidate_ac_ep100` | learned AC, free-flow candidates | 4.3437 | 4.2244 | 4.4233 | r0=2494, r1=219, r2=63, r3=21 |
| `baseline_freeflow_greedy` | baseline | 4.4082 | 4.3910 | 4.4196 | r0=2970, r1=0, r2=0, r3=4 |
| `baseline_live_current_greedy` | baseline | 4.3767 | 4.2955 | 4.4308 | r0=2533, r1=260, r2=66, r3=23 |
| `policy_ep400_reference` | earlier good policy reference | 4.3450 | 4.2263 | 4.4242 | r0=2484, r1=249, r2=46, r3=21 |
| `seed96_freeflow_candidate_ac_ep400` | seed-sensitive weak run | 4.4005 | 4.3800 | 4.4143 | r0=1327, r1=630, r2=500, r3=426 |

`seed98_freeflow_candidate_ac_ep100` improves over:

- Free-flow greedy by `0.0644` t_all, about `1.46%`.
- Live-current greedy by `0.0330` t_all, about `0.75%`.

This is enough evidence that free-flow generated candidates are not inherently preventing convergence. The remaining problem is seed/optimization stability: `seed96` with the same free-flow candidate semantics converged to a much more route-diverse deterministic policy and only reached `t_all=4.4005`.

## Baseline Definitions

Both baselines were evaluated on the same free-flow generated candidate paths:

- `baseline_freeflow_greedy`: chooses the candidate with lowest free-flow route cost.
- `baseline_live_current_greedy`: chooses the candidate with lowest live/current travel-time route cost.

Both use:

- `--av_candidate_source current_shortest_freeflow`
- `--wm_candidate_edge_cost_source freeflow`
- `--wm_edge_cost_source decoded_speed`
- `--dynamic_routing`

The baseline script is [baseline/eval_current_to_dest_freeflow_candidates.sh](/home/wangwenrui/shk/UrbanDreamer/baseline/eval_current_to_dest_freeflow_candidates.sh).

## Candidate Source Ablation

A controlled seed98 pair compared train-time candidate generation sources while evaluating both policies on free-flow candidates:

| Train candidate edge cost | Eval candidate source | t_all | Selected route counts |
| --- | --- | ---: | --- |
| free-flow | current_shortest_freeflow | 4.3437 | r0=2494, r1=219, r2=63, r3=21 |
| decoded-speed | current_shortest_freeflow | 4.3508 | r0=2521, r1=219, r2=59, r3=21 |

The free-flow candidate run is slightly better in this controlled comparison. This does not support switching the mainline to decoded-speed candidate generation.

There is also an action-set gap diagnostic in `tmp/train_candidate_action_set_gap_summary.json`:

- Compared active vehicles: `5872`
- Mean candidate set Jaccard: `0.8647`
- Median candidate set Jaccard: `1.0`
- Fraction where decoded-speed candidates had a strictly better best route than free-flow candidates: `0.0191`
- Mean best-route gap, free-flow set minus decoded-speed set: `0.1689s`
- p95 best-route gap: `0.0s`

Interpretation: free-flow candidate sets are usually not missing a materially better decoded-speed route. The weaker `seed96` result is better explained by actor-critic optimization / seed sensitivity than by candidate-set mismatch.

## Route Choice Interpretation

The learned good policies are not equivalent to pure route0 greedy:

- `seed98_freeflow_candidate_ac_ep100`: route0 share `89.17%`
- `policy_ep400_reference`: route0 share `88.71%`
- `baseline_freeflow_greedy`: route0 share `99.87%`
- `baseline_live_current_greedy`: route0 share `87.89%`
- `seed96_freeflow_candidate_ac_ep400`: route0 share `46.03%`

The strong learned policy resembles the live-current greedy route mix more than pure free-flow greedy, but gets better `t_all` than both baselines. The weak seed96 run is the clear outlier: it sends too many decisions to route1/2/3 and loses most of the benefit.

The route/path visualizations are in [visualization/current_to_dest_choices](/home/wangwenrui/shk/UrbanDreamer/visualization/current_to_dest_choices).

## Source Files

Primary raw summaries:

- `results/ingolstadt_custom_current_to_dest_incident_20260528/eval_candpair_freeflow_seed98_ep100_current_shortest_freeflow/collector_eval_summary.json`
- `results/ingolstadt_custom_current_to_dest_incident_20260528/baseline_ep400_ref_freeflow_candidates_freeflow_greedy/collector_eval_summary.json`
- `results/ingolstadt_custom_current_to_dest_incident_20260528/baseline_ep400_ref_freeflow_candidates_live_current_greedy/collector_eval_summary.json`
- `results/ingolstadt_custom_current_to_dest_incident_20260528/eval_ep400_current_shortest_freeflow_exact/collector_eval_summary.json`
- `results/ingolstadt_custom_current_to_dest_incident_20260528/eval_seed96_freeflowcand_fromscratch_anchortrust_ep400/collector_eval_summary.json`
- `results/ingolstadt_custom_current_to_dest_incident_20260528/eval_candpair_decoded_speed_seed98_ep100_current_shortest_freeflow/collector_eval_summary.json`

Generated summary files in this directory:

- [comparison.csv](/home/wangwenrui/shk/UrbanDreamer/experiment_result/comparison.csv)
- [summary.json](/home/wangwenrui/shk/UrbanDreamer/experiment_result/summary.json)

## Current Limitations

The claim here is deliberately narrow:

- Confirmed: free-flow candidate generation can produce a strong actor-critic policy on this Ingolstadt current-to-destination setup.
- Confirmed: the learned policy beats the two requested greedy baselines in SUMO eval.
- Not yet confirmed: multi-seed stability.
- Not yet confirmed: final `seed98` 400ep from-scratch result, because that long run was still training when this report was written.
- Still likely issue: actor optimization is seed-sensitive; bad seeds can keep logits too diffuse or produce route1/2/3 over-selection.
