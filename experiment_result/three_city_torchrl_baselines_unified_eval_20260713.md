# 三城市 TorchRL baseline 统一评测最终表

生成时间：2026-07-13T17:54:34.716126+00:00。

统一协议：每个方法 5 个 episode；TorchRL 使用 strict valid-mask argmax，主线使用安全候选集上的 deterministic argmax；HDV post-mutation 状态冻结；候选路由为 `current_shortest_freeflow`；`sim_buffer_after_last_depart=7200`；`max_steps_per_episode=9000`；`sumo_time_to_teleport=-1`。

**重要口径说明：本报告保留未启用 `ignore-junction-blocker` 的原统一协议结果。Fontenay 后续在继续保持 `sumo_time_to_teleport=-1` 的前提下，仅增加 `ignore-junction-blocker=60`，已实现五种方法 25/25 个 episode 全部 100% completion、teleports=0。该补充结果没有覆盖进本表，以免把只改变 Fontenay SUMO 协议的结果混入三城市统一比较；详见 [Fontenay 无-teleport 100% 补充评测](fontenay_ignore_junction_blocker60_eval_20260713.md)。**

表内数值为 episode 等权宏平均 ± 总体标准差（ddof=0），单位为分钟。Δt_all = 方法 t_all − 当前主线 t_all，因此负 delta 更好。

## Gretz-Armainvilliers

| 方法 | t_all | t_CAV | t_HDV | Completion | Δt_all vs 主线 | 口径 |
|---|---:|---:|---:|---:|---:|---|
| UrbanDreamer（当前主线） | 4.1136 ± 0.0064 | 3.2349 ± 0.0049 | 4.6979 ± 0.0075 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 3180/3180 到达) | +0.0000 | 全体车辆 |
| IPPO | 5.4346 ± 0.0013 | 5.8716 ± 0.0034 | 5.1440 ± 0.0044 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 3180/3180 到达) | +1.3210 | 全体车辆 |
| MAPPO | 6.2977 ± 0.0193† | 7.9954 ± 0.0243† | 5.2695 ± 0.0163† | 89.62% ± 0.00% (min 89.62%; 0/5 完整; 2850/3180 到达) | — | 仅到达车辆† |
| IQL | 7.1367 ± 0.0024 | 10.3764 ± 0.0049 | 4.9825 ± 0.0008 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 3180/3180 到达) | +3.0231 | 全体车辆 |
| Greedy（freeflow） | 4.8212 ± 0.0027 | 4.1366 ± 0.0022 | 5.2764 ± 0.0031 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 3180/3180 到达) | +0.7075 | 全体车辆 |
| Greedy（live current TT） | 4.6897 ± 0.0002 | 3.9693 ± 0.0012 | 5.1687 ± 0.0010 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 3180/3180 到达) | +0.5761 | 全体车辆 |

## Ingolstadt

| 方法 | t_all | t_CAV | t_HDV | Completion | Δt_all vs 主线 | 口径 |
|---|---:|---:|---:|---:|---:|---|
| UrbanDreamer（当前主线） | 4.4684 ± 0.0000 | 4.4991 ± 0.0000 | 4.4479 ± 0.0000 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +0.0000 | 全体车辆 |
| IPPO | 5.6533 ± 0.0000 | 7.4683 ± 0.0000 | 4.4434 ± 0.0000 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +1.1850 | 全体车辆 |
| MAPPO | 7.2070 ± 0.0000 | 11.4150 ± 0.0000 | 4.4017 ± 0.0000 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +2.7386 | 全体车辆 |
| IQL | 7.6572 ± 0.0000 | 12.5587 ± 0.0000 | 4.3896 ± 0.0000 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +3.1889 | 全体车辆 |
| Greedy（freeflow） | 4.5419 ± 0.0000 | 4.5907 ± 0.0000 | 4.5093 ± 0.0000 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +0.0735 | 全体车辆 |
| Greedy（live current TT） | 4.4100 ± 0.0000 | 4.3751 ± 0.0000 | 4.4333 ± 0.0000 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | -0.0584 | 全体车辆 |

## Fontenay-en-Parisis

| 方法 | t_all | t_CAV | t_HDV | Completion | Δt_all vs 主线 | 口径 |
|---|---:|---:|---:|---:|---:|---|
| UrbanDreamer（当前主线） | 4.5876 ± 0.1347† | 4.6295 ± 0.0123† | 4.5565 ± 0.2365† | 97.38% ± 5.24% (min 86.90%; 4/5 完整; 10069/10340 到达) | — | 仅到达车辆† |
| IPPO | 4.5182 ± 0.2792† | 5.2094 ± 0.3945† | 4.0575 ± 0.2091† | 86.05% ± 5.52% (min 82.11%; 0/5 完整; 8898/10340 到达) | — | 仅到达车辆† |
| MAPPO | 5.8091 ± 0.5256† | 8.5565 ± 0.6535† | 4.0305 ± 0.4363† | 83.38% ± 12.20% (min 68.38%; 1/5 完整; 8621/10340 到达) | — | 仅到达车辆† |
| IQL | — | — | — | — | — | 占位 |
| Greedy（freeflow） | 4.1930 ± 0.3927† | 4.0952 ± 0.4714† | 4.2588 ± 0.3401† | 90.47% ± 7.78% (min 83.85%; 2/5 完整; 9355/10340 到达) | — | 仅到达车辆† |
| Greedy（live current TT） | 4.0537 ± 0.3121† | 3.9370 ± 0.3660† | 4.1331 ± 0.2755† | 87.67% ± 6.17% (min 84.24%; 1/5 完整; 9065/10340 到达) | — | 仅到达车辆† |

## 审计与解释

- `†` 表示至少一个 episode 未达到 100% completion。对应 t 只统计已到达车辆，存在幸存者偏差（arrived_only），因此不计算 delta，也不与完整结果直接排序。
- Fontenay IQL 是明确占位项：它不属于本轮已完成的 8 个 TorchRL checkpoint，未被当作 missing 输入。
- TorchRL 每个有效 attempt 均检查 config、eval summary、严格 episode ID/整数完成性，并逐行验证 `dynamic_agent_decisions.csv` 的 argmax 与 valid mask。
- 主线/Greedy 每个有效 attempt 均检查 collector summary、URB-fair manifest、冻结 HDV、7200 秒 buffer、9000 秒总时域、路由策略以及 post/reuse 路径和route artifact 哈希。
- 所有方法均从 5 个逐车原始 episode CSV 独立复算 `travel_time`：校验唯一车辆 ID、post-state CAV/HDV 身份与有限到达数，并按 `sum(seconds) / arrivals / 60` 交叉验证 t_all/t_CAV/t_HDV。

## 选用的 attempt

- Gretz-Armainvilliers / UrbanDreamer（当前主线）：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_urbfair_unified_detargmax_long9000_5ep_20260713/mainline/attempt_001`
- Gretz-Armainvilliers / IPPO：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_torchrl_ep500_detargmax_long9000_5ep_20260713/ippo/attempt_001`（逐行审计 9708 条决策）
- Gretz-Armainvilliers / MAPPO：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_torchrl_ep500_detargmax_long9000_5ep_20260713/mappo/attempt_001`（逐行审计 15218 条决策）
- Gretz-Armainvilliers / IQL：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_torchrl_ep500_detargmax_long9000_5ep_20260713/iql/attempt_001`（逐行审计 16891 条决策）
- Gretz-Armainvilliers / Greedy（freeflow）：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_urbfair_unified_detargmax_long9000_5ep_20260713/freeflow_greedy/attempt_001`
- Gretz-Armainvilliers / Greedy（live current TT）：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_urbfair_unified_detargmax_long9000_5ep_20260713/live_current_tt/attempt_001`
- Ingolstadt / UrbanDreamer（当前主线）：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_urbfair_unified_detargmax_long9000_5ep_20260713/mainline/attempt_001`
- Ingolstadt / IPPO：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_torchrl_ep500_detargmax_long9000_5ep_20260713/ippo/attempt_001`（逐行审计 21870 条决策）
- Ingolstadt / MAPPO：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_torchrl_ep500_detargmax_long9000_5ep_20260713/mappo/attempt_001`（逐行审计 30585 条决策）
- Ingolstadt / IQL：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_torchrl_ep500_detargmax_long9000_5ep_20260713/iql/attempt_001`（逐行审计 33755 条决策）
- Ingolstadt / Greedy（freeflow）：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_urbfair_unified_detargmax_long9000_5ep_20260713/freeflow_greedy/attempt_001`
- Ingolstadt / Greedy（live current TT）：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_urbfair_unified_detargmax_long9000_5ep_20260713/live_current_tt/attempt_001`
- Fontenay-en-Parisis / UrbanDreamer（当前主线）：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_urbfair_unified_detargmax_long9000_5ep_20260713/mainline/attempt_001`
- Fontenay-en-Parisis / IPPO：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_torchrl_ep500_detargmax_long9000_5ep_20260713/ippo/attempt_001`（逐行审计 35341 条决策）
- Fontenay-en-Parisis / MAPPO：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_torchrl_ep500_detargmax_long9000_5ep_20260713/mappo/attempt_001`（逐行审计 55991 条决策）
- Fontenay-en-Parisis / IQL：占位：Fontenay IQL 尚未包含在已完成的 8 个 TorchRL checkpoint 中。
- Fontenay-en-Parisis / Greedy（freeflow）：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_urbfair_unified_detargmax_long9000_5ep_20260713/freeflow_greedy/attempt_001`
- Fontenay-en-Parisis / Greedy（live current TT）：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_urbfair_unified_detargmax_long9000_5ep_20260713/live_current_tt/attempt_001`
