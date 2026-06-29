# Active Vehicle WM Cost Diagnostic

- Input: `results/toy_cross_bottleneck/online_wm_case_study_hdvlearn_dynamic_400ep_20260629/manual_eval_ep00100_5ep_hdvfreeze/dynamic_av_agent_decisions.csv`
- Rows with valid route costs: 12980 / 12980
- Index top-1 match: 0.8147
- Path-equivalent top-1 match: 0.8147
- Top-2 path overlap mean: 1.1144
- WM/live MAE mean: 104.1532
- WM best live regret mean: 9.2515
- Selected live regret mean: 8.8230
- Spearman mean: 0.7739
- Kendall mean: 0.7339

## By Current Edge

| edge | rows | path top-1 | MAE | wm-best live regret | selected live regret |
|---|---:|---:|---:|---:|---:|
| R_to_M | 2785 | 1.0000 | 83.6545 | 0.0000 | 0.0000 |
| A_pre_Y | 2425 | 1.0000 | 106.2818 | 0.0000 | 0.0000 |
| Y | 1230 | 1.0000 | 40.4297 | 0.0000 | 0.0000 |
| R_tail | 1180 | 1.0000 | 128.6130 | 0.0000 | 0.0000 |
| A_hold | 1045 | 0.1340 | 209.4843 | 28.8226 | 26.8505 |
| A_in | 780 | 0.1410 | 241.8235 | 22.7090 | 18.2159 |
| M | 545 | 1.0000 | 3.7497 | 0.0000 | 0.0000 |
| B_hold | 520 | 0.2308 | 86.5841 | 15.8691 | 15.8795 |
| B_pre_Y | 440 | 1.0000 | 45.2998 | 0.0000 | 0.0000 |
| X | 405 | 0.5802 | 276.3463 | 129.4985 | 129.4985 |
| R_shared | 350 | 1.0000 | 85.1495 | 0.0000 | 0.0000 |
| B_in | 345 | 0.2754 | 92.8104 | 15.2465 | 15.2410 |
| Y_to_M | 295 | 1.0000 | 4.5696 | 0.0000 | 0.0000 |
| M_to_B | 245 | 1.0000 | 1.5726 | 0.0000 | 0.0000 |
| B_out | 160 | 1.0000 | 0.2651 | 0.0000 | 0.0000 |
| A_to_R | 85 | 1.0000 | 33.7432 | 0.0000 | 0.0000 |
| B_to_R | 55 | 1.0000 | 4.6209 | 0.0000 | 0.0000 |
| X_to_M | 40 | 1.0000 | 38.0875 | 0.0000 | 0.0000 |
| A_pre_X | 25 | 0.6000 | 200.2468 | 251.7018 | 251.7018 |
| A_out | 15 | 1.0000 | 0.1748 | 0.0000 | 0.0000 |
| M_to_A | 10 | 1.0000 | 0.0635 | 0.0000 | 0.0000 |
