# Active Vehicle WM Cost Diagnostic

- Input: `results/toy_cross_bottleneck/online_wm_case_study_hdvlearn_dynamic_400ep_20260629/manual_eval_ep00100_5ep/dynamic_av_agent_decisions.csv`
- Rows with valid route costs: 12275 / 12275
- Index top-1 match: 0.8033
- Path-equivalent top-1 match: 0.8033
- Top-2 path overlap mean: 1.0310
- WM/live MAE mean: 104.1082
- WM best live regret mean: 10.8070
- Selected live regret mean: 10.7865
- Spearman mean: 0.6090
- Kendall mean: 0.5923

## By Current Edge

| edge | rows | path top-1 | MAE | wm-best live regret | selected live regret |
|---|---:|---:|---:|---:|---:|
| R_to_M | 2004 | 1.0000 | 86.2185 | 0.0000 | 0.0000 |
| A_pre_Y | 1699 | 0.9929 | 104.5059 | 0.0188 | 0.0188 |
| A_hold | 1192 | 0.2374 | 188.1096 | 43.7209 | 43.9837 |
| Y | 1117 | 0.9937 | 39.4487 | 0.0151 | 0.0151 |
| X | 892 | 0.7702 | 193.5632 | 42.9948 | 42.9116 |
| A_in | 864 | 0.2535 | 209.2945 | 29.9263 | 29.3599 |
| X_to_M | 767 | 1.0000 | 111.7705 | 0.0000 | 0.0000 |
| M | 603 | 1.0000 | 3.6989 | 0.0000 | 0.0000 |
| R_tail | 602 | 1.0000 | 132.8231 | 0.0000 | 0.0000 |
| B_hold | 520 | 0.3135 | 78.4128 | 12.9296 | 12.9399 |
| B_pre_Y | 379 | 0.9921 | 44.0943 | 0.0211 | 0.0211 |
| B_in | 345 | 0.2812 | 82.9841 | 13.1066 | 13.0864 |
| R_shared | 314 | 1.0000 | 121.9255 | 0.0000 | 0.0000 |
| Y_to_M | 266 | 1.0000 | 4.6230 | 0.0000 | 0.0000 |
| M_to_B | 260 | 1.0000 | 1.4504 | 0.0000 | 0.0000 |
| B_out | 174 | 1.0000 | 0.2312 | 0.0000 | 0.0000 |
| A_pre_X | 89 | 0.7528 | 88.3683 | 46.2020 | 46.2020 |
| A_to_R | 54 | 1.0000 | 38.4100 | 0.0000 | 0.0000 |
| B_pre_X | 45 | 0.8444 | 30.2510 | 20.4316 | 20.4316 |
| B_to_R | 35 | 1.0000 | 6.4856 | 0.0000 | 0.0000 |
| A_out | 26 | 1.0000 | 0.1948 | 0.0000 | 0.0000 |
| M_to_A | 24 | 1.0000 | 0.3030 | 0.0000 | 0.0000 |
| X_to_R | 4 | 1.0000 | 76.3802 | 0.0000 | 0.0000 |
