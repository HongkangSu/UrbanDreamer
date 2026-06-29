# Toy Cross-Bottleneck Case Study

## Main Conclusions

- Dynamic game evidence: dynamic rerouting improves t_all by 0.1444 min over first-decision-only on the pure online AC checkpoint.
- Best setting-matched online+WM dynamic policy does not beat the pure online AC baseline: online_wm_hdvlearn_ep100_manual_5ep_dynamic t_all=7.5087, online baseline t_all=7.4553, delta=0.0534 min.
- Best legacy/reference WM-family dynamic row: wm_marl_mainline_ep400_dynamic t_all=7.4453, delta_vs_online=-0.0101 min.
- Route-probability separation is reported only as diagnostics; it is not a success criterion for this toy case.

## Dynamic Evaluation Comparison

| rank | policy | family | human learn | test_eps | t_all | t_CAV | t_HDV | completion | lnK-H | unique paths | note |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | online_wm_ep100_hdvfreeze_diagnostic_dynamic | online_wm_diagnostic | true | 5 | 7.4350 | 8.1975 | 6.9266 | 1.000 | 0.0981 | 1.763 | diagnostic only: online+WM ep100 dynamic eval with HDV learning frozen after mutation |
| 2 | wm_marl_mainline_ep400_dynamic | wm_marl | unknown | 3 | 7.4453 | 8.4733 | 6.7599 | 1.000 | 0.0107 | 1.724 | WM/MARL mainline checkpoint; not the pure online baseline |
| 3 | pure_online_ac_ep400_dynamic_hdvlearn | pure_online | true | 5 | 7.4553 | 7.1079 | 7.6869 | 1.000 | 0.2364 | nan | pure SUMO online AC; dynamic eval only, fixed roles, HDV learning enabled |
| 4 | offline_wm_noqhat_ep2000_dynamic | offline_wm | unknown | 3 | 7.4825 | 7.3477 | 7.5723 | 1.000 | 0.0359 | 1.884 | offline WM/Dreamer no-Qhat long train |
| 5 | offline_wm_capped_step_ep400_dynamic | offline_wm | unknown | 3 | 7.4903 | 8.3152 | 6.9403 | 1.000 | 0.0003 | 1.869 | older capped-step WM-family run |
| 6 | online_wm_hdvlearn_ep100_manual_5ep_dynamic | online_wm | true | 5 | 7.5087 | 7.7425 | 7.3528 | 1.000 | 0.0926 | 1.820 | online+WM corrected mainline ep100 manual 5ep eval; dynamic, fixed roles, HDV learning enabled |
| 7 | offline_wm_noqhat_ep1000_dynamic | offline_wm | unknown | 3 | 7.5808 | 7.4062 | 7.6972 | 1.000 | 0.0309 | 1.841 | offline WM/Dreamer no-Qhat |
| 8 | offline_wm_progress_ep400_dynamic | offline_wm | unknown | 3 | 7.7921 | 7.6237 | 7.9043 | 1.000 | 0.0001 | 1.717 | offline WM/Dreamer progress reward |
| 9 | offline_wm_noqhat_ep400_dynamic | offline_wm | unknown | 3 | 7.8034 | 8.0245 | 7.6559 | 1.000 | 0.0001 | 1.749 | offline WM/Dreamer no-Qhat |

## Dynamic Game Ablation

| mode | episodes | t_all | t_CAV | t_HDV | completion |
|---|---:|---:|---:|---:|---:|
| dynamic | 5 | 7.3646 | 7.4486 | 7.3086 | 1.000 |
| first_decision_only | 5 | 7.5090 | 7.6896 | 7.3886 | 1.000 |

Delta is dynamic minus first-decision-only; negative is better.

```json
{
  "t_all_dynamic_minus_first_only": -0.14435416666666612,
  "t_CAV_dynamic_minus_first_only": -0.240963541666666,
  "t_HDV_dynamic_minus_first_only": -0.07994791666666679
}
```

## Route Diagnostics

The toy network intentionally has limited effective route diversity. Route0 and route3 are often duplicate candidate paths, so route entropy is a diagnostic rather than a pass/fail criterion.

| policy | selected/live match | selected/WM match | live regret | route0=route3 | selected counts | mean probs |
|---|---:|---:|---:|---:|---|---|
| online_wm_ep100_hdvfreeze_diagnostic_dynamic | 0.279 | 0.379 | 8.823 | 1.000 | `{'route0': 3370, 'route1': 3075, 'route2': 3350, 'route3': 3185}` | `{'route0': 0.2652, 'route1': 0.1935, 'route2': 0.2751, 'route3': 0.2662}` |
| wm_marl_mainline_ep400_dynamic | 0.824 | 0.958 | 11.433 | 1.000 | `{'route0': 6861, 'route1': 927, 'route2': 225, 'route3': 0}` | `{'route0': 0.2569, 'route1': 0.2271, 'route2': 0.2591, 'route3': 0.2569}` |
| pure_online_ac_ep400_dynamic_hdvlearn | nan | nan | 10.642 | nan | `{'route0': 3204, 'route1': 2673, 'route2': 2171, 'route3': 2257}` | `{'route0': 0.3085, 'route1': 0.2592, 'route2': 0.2107, 'route3': 0.2216}` |
| offline_wm_noqhat_ep2000_dynamic | 0.824 | 0.880 | 13.847 | 1.000 | `{'route0': 6212, 'route1': 665, 'route2': 324, 'route3': 0}` | `{'route0': 0.2675, 'route1': 0.204, 'route2': 0.2611, 'route3': 0.2675}` |
| offline_wm_capped_step_ep400_dynamic | 0.839 | 0.863 | 9.605 | 1.000 | `{'route0': 5874, 'route1': 813, 'route2': 876, 'route3': 0}` | `{'route0': 0.2505, 'route1': 0.2466, 'route2': 0.2525, 'route3': 0.2505}` |
| online_wm_hdvlearn_ep100_manual_5ep_dynamic | 0.270 | 0.370 | 10.787 | 1.000 | `{'route0': 3238, 'route1': 2799, 'route2': 3181, 'route3': 3057}` | `{'route0': 0.2731, 'route1': 0.1922, 'route2': 0.2606, 'route3': 0.2741}` |
| offline_wm_noqhat_ep1000_dynamic | 0.825 | 0.907 | 14.798 | 1.000 | `{'route0': 6615, 'route1': 670, 'route2': 179, 'route3': 0}` | `{'route0': 0.2672, 'route1': 0.2056, 'route2': 0.2599, 'route3': 0.2672}` |
| offline_wm_progress_ep400_dynamic | 0.866 | 0.935 | 7.110 | 1.000 | `{'route0': 6645, 'route1': 722, 'route2': 274, 'route3': 0}` | `{'route0': 0.2508, 'route1': 0.2477, 'route2': 0.2507, 'route3': 0.2508}` |
| offline_wm_noqhat_ep400_dynamic | 0.853 | 0.901 | 12.207 | 1.000 | `{'route0': 7091, 'route1': 531, 'route2': 535, 'route3': 0}` | `{'route0': 0.251, 'route1': 0.2472, 'route2': 0.2508, 'route3': 0.251}` |

## Source Files

- JSON: `experiment_result/toy_cross_bottleneck_case_study_20260629.json`
- This report: `experiment_result/toy_cross_bottleneck_case_study_20260629.md`
