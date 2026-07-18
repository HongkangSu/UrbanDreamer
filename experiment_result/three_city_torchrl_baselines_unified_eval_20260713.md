# 三城市 TorchRL baseline 统一评测表（训练时执行器阶段更新）

最近更新：2026-07-17 15:32:51 UTC。

统一仿真协议：每个方法 5 个 episode；HDV post-mutation 状态冻结；候选路由为 `current_shortest_freeflow`；`sim_buffer_after_last_depart=7200`；`max_steps_per_episode=9000`；`sumo_time_to_teleport=-1`。

策略执行口径：Gretz 和 Ingolstadt 的 IPPO、MAPPO、IQL 已在本次更新中改为各自训练时的执行器。IPPO/MAPPO 使用 strict valid-mask categorical sampling；IQL 使用训练终局的 valid-mask epsilon-greedy（`epsilon=0.05`）。主线、Greedy 以及尚在重跑的 Fontenay baseline 行暂时保留 2026-07-13 的 deterministic 结果，因此本文件当前是阶段表，而不是三城市全部完成后的最终同口径表。

同名 `.csv`/`.json` 已同步替换 Gretz 和 Ingolstadt 的六组 TorchRL 结果，并保留 Fontenay 尚未完成的旧记录/占位项。

**重要口径说明：本表仍以未启用 `ignore-junction-blocker` 的原统一协议为主。Fontenay blocker60 结果见 [Fontenay 无-teleport补充评测](fontenay_ignore_junction_blocker60_eval_20260713.md)，其中 IPPO 已替换为训练时执行器结果；Fontenay 原协议三组以及 blocker60 MAPPO/IQL 仍在运行或等待。**

表内数值为 episode 等权宏平均 ± 总体标准差（ddof=0），单位为分钟。Δt_all = 方法 t_all − 当前主线 t_all，因此负 delta 更好。

## 已完成六组的阶段结论

- Gretz 对随机执行较敏感。IPPO、MAPPO、IQL 分别只有 3/5、2/5、2/5 个完整 episode，平均 completion 为 96.48%、95.50%、91.16%。其 t 值只来自已到达车辆，存在明显幸存者偏差，不能据此认为其旅行时间优于旧 argmax 结果或当前主线。
- Ingolstadt 三组仍是 5/5 完整。IPPO、MAPPO、IQL 的 t_all 分别为 5.6794、7.1569、7.6120 分钟；相对当前表中主线分别为 +1.2110、+2.6885、+3.1437 分钟，排序仍为 IPPO 优于 MAPPO、IQL，但均弱于主线。
- 六组共审计 125,983 条决策，所有 selected route 均满足 valid mask。实际非 argmax 决策为 2,693 条；IQL 在两城的非 argmax 比例为 3.69% 和 3.75%，与 `epsilon=0.05` 经 valid-mask/greedy 分配后的预期量级一致。六组 SUMO teleports 均为 0。

## Gretz-Armainvilliers

| 方法 | 策略执行 | t_all | t_CAV | t_HDV | Completion | Δt_all vs 主线 | 口径 |
|---|---|---:|---:|---:|---:|---:|---|
| UrbanDreamer（当前主线） | deterministic argmax（旧） | 4.1136 ± 0.0064 | 3.2349 ± 0.0049 | 4.6979 ± 0.0075 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 3180/3180 到达) | +0.0000 | 全体车辆 |
| IPPO | categorical sample | 4.6518 ± 0.7000† | 5.1370 ± 0.5144† | 4.3230 ± 0.8461† | 96.48% ± 4.31% (min 91.19%; 3/5 完整; 3068/3180 到达) | — | 仅到达车辆† |
| MAPPO | categorical sample | 6.0667 ± 0.6801† | 8.0737 ± 0.7083† | 4.7736 ± 0.6454† | 95.50% ± 4.75% (min 89.15%; 2/5 完整; 3037/3180 到达) | — | 仅到达车辆† |
| IQL | epsilon-greedy (`epsilon=0.05`) | 5.9639 ± 0.9391† | 8.7659 ± 1.1859† | 4.1878 ± 0.7936† | 91.16% ± 7.77% (min 80.03%; 2/5 完整; 2899/3180 到达) | — | 仅到达车辆† |
| Greedy（freeflow） | deterministic rule | 4.8212 ± 0.0027 | 4.1366 ± 0.0022 | 5.2764 ± 0.0031 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 3180/3180 到达) | +0.7075 | 全体车辆 |
| Greedy（live current TT） | deterministic rule | 4.6897 ± 0.0002 | 3.9693 ± 0.0012 | 5.1687 ± 0.0010 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 3180/3180 到达) | +0.5761 | 全体车辆 |

## Ingolstadt

| 方法 | 策略执行 | t_all | t_CAV | t_HDV | Completion | Δt_all vs 主线 | 口径 |
|---|---|---:|---:|---:|---:|---:|---|
| UrbanDreamer（当前主线） | deterministic argmax（旧） | 4.4684 ± 0.0000 | 4.4991 ± 0.0000 | 4.4479 ± 0.0000 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +0.0000 | 全体车辆 |
| IPPO | categorical sample | 5.6794 ± 0.0321 | 7.5324 ± 0.0759 | 4.4440 ± 0.0166 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +1.2110 | 全体车辆 |
| MAPPO | categorical sample | 7.1569 ± 0.0469 | 11.2906 ± 0.1146 | 4.4011 ± 0.0063 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +2.6885 | 全体车辆 |
| IQL | epsilon-greedy (`epsilon=0.05`) | 7.6120 ± 0.1023 | 12.3958 ± 0.2520 | 4.4229 ± 0.0100 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +3.1437 | 全体车辆 |
| Greedy（freeflow） | deterministic rule | 4.5419 ± 0.0000 | 4.5907 ± 0.0000 | 4.5093 ± 0.0000 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +0.0735 | 全体车辆 |
| Greedy（live current TT） | deterministic rule | 4.4100 ± 0.0000 | 4.3751 ± 0.0000 | 4.4333 ± 0.0000 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | -0.0584 | 全体车辆 |

## Fontenay-en-Parisis

| 方法 | 策略执行 | t_all | t_CAV | t_HDV | Completion | Δt_all vs 主线 | 口径 |
|---|---|---:|---:|---:|---:|---:|---|
| UrbanDreamer（当前主线） | deterministic argmax（旧） | 4.5876 ± 0.1347† | 4.6295 ± 0.0123† | 4.5565 ± 0.2365† | 97.38% ± 5.24% (min 86.90%; 4/5 完整; 10069/10340 到达) | — | 仅到达车辆† |
| IPPO | argmax（旧，重跑中） | 4.5182 ± 0.2792† | 5.2094 ± 0.3945† | 4.0575 ± 0.2091† | 86.05% ± 5.52% (min 82.11%; 0/5 完整; 8898/10340 到达) | — | 仅到达车辆† |
| MAPPO | argmax（旧，重跑中） | 5.8091 ± 0.5256† | 8.5565 ± 0.6535† | 4.0305 ± 0.4363† | 83.38% ± 12.20% (min 68.38%; 1/5 完整; 8621/10340 到达) | — | 仅到达车辆† |
| IQL | 等待 500ep checkpoint | — | — | — | — | — | 占位 |
| Greedy（freeflow） | deterministic rule | 4.1930 ± 0.3927† | 4.0952 ± 0.4714† | 4.2588 ± 0.3401† | 90.47% ± 7.78% (min 83.85%; 2/5 完整; 9355/10340 到达) | — | 仅到达车辆† |
| Greedy（live current TT） | deterministic rule | 4.0537 ± 0.3121† | 3.9370 ± 0.3660† | 4.1331 ± 0.2755† | 87.67% ± 6.17% (min 84.24%; 1/5 完整; 9065/10340 到达) | — | 仅到达车辆† |

## 审计与解释

- `†` 表示至少一个 episode 未达到 100% completion。对应 t 只统计已到达车辆，存在幸存者偏差（arrived_only），因此不计算 delta，也不与完整结果直接排序。
- Fontenay IQL 仍为占位项：500ep checkpoint 尚未完成，原协议与 blocker60 的训练时执行器评估已排队等待。
- 本次更新的六个 TorchRL attempt 均检查 config、eval summary、严格 episode ID/整数完成性、冻结 HDV、9000/7200 时域与无 teleport；逐行验证 `dynamic_agent_decisions.csv` 的 selected route 满足 valid mask，并确认存在真实非 argmax 采样。IPPO/MAPPO 使用 categorical sampling；IQL 额外校验 `epsilon=0.05`。
- 主线/Greedy 每个有效 attempt 均检查 collector summary、URB-fair manifest、冻结 HDV、7200 秒 buffer、9000 秒总时域、路由策略以及 post/reuse 路径和route artifact 哈希。
- 所有方法均从 5 个逐车原始 episode CSV 独立复算 `travel_time`：校验唯一车辆 ID、post-state CAV/HDV 身份与有限到达数，并按 `sum(seconds) / arrivals / 60` 交叉验证 t_all/t_CAV/t_HDV。

## 选用的 attempt

- Gretz-Armainvilliers / UrbanDreamer（当前主线）：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_urbfair_unified_detargmax_long9000_5ep_20260713/mainline/attempt_001`
- Gretz-Armainvilliers / IPPO：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_torchrl_ep500_native_long9000_5ep_20260716/ippo/attempt_002`（审计 9378 条决策；227 条非 argmax）
- Gretz-Armainvilliers / MAPPO：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_torchrl_ep500_native_long9000_5ep_20260716/mappo/attempt_002`（审计 14067 条决策；94 条非 argmax）
- Gretz-Armainvilliers / IQL：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_torchrl_ep500_native_long9000_5ep_20260716/iql/attempt_002`（审计 16819 条决策；621 条非 argmax）
- Gretz-Armainvilliers / Greedy（freeflow）：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_urbfair_unified_detargmax_long9000_5ep_20260713/freeflow_greedy/attempt_001`
- Gretz-Armainvilliers / Greedy（live current TT）：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_urbfair_unified_detargmax_long9000_5ep_20260713/live_current_tt/attempt_001`
- Ingolstadt / UrbanDreamer（当前主线）：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_urbfair_unified_detargmax_long9000_5ep_20260713/mainline/attempt_001`
- Ingolstadt / IPPO：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_torchrl_ep500_native_long9000_5ep_20260716/ippo/attempt_002`（审计 21810 条决策；205 条非 argmax）
- Ingolstadt / MAPPO：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_torchrl_ep500_native_long9000_5ep_20260716/mappo/attempt_002`（审计 30324 条决策；288 条非 argmax）
- Ingolstadt / IQL：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_torchrl_ep500_native_long9000_5ep_20260716/iql/attempt_002`（审计 33585 条决策；1258 条非 argmax）
- Ingolstadt / Greedy（freeflow）：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_urbfair_unified_detargmax_long9000_5ep_20260713/freeflow_greedy/attempt_001`
- Ingolstadt / Greedy（live current TT）：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_urbfair_unified_detargmax_long9000_5ep_20260713/live_current_tt/attempt_001`
- Fontenay-en-Parisis / UrbanDreamer（当前主线）：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_urbfair_unified_detargmax_long9000_5ep_20260713/mainline/attempt_001`
- Fontenay-en-Parisis / IPPO：旧 argmax attempt：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_torchrl_ep500_detargmax_long9000_5ep_20260713/ippo/attempt_001`；训练时执行器重跑中。
- Fontenay-en-Parisis / MAPPO：旧 argmax attempt：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_torchrl_ep500_detargmax_long9000_5ep_20260713/mappo/attempt_001`；训练时执行器重跑中。
- Fontenay-en-Parisis / IQL：占位：等待 500ep checkpoint 后自动运行训练时执行器评估。
- Fontenay-en-Parisis / Greedy（freeflow）：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_urbfair_unified_detargmax_long9000_5ep_20260713/freeflow_greedy/attempt_001`
- Fontenay-en-Parisis / Greedy（live current TT）：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_urbfair_unified_detargmax_long9000_5ep_20260713/live_current_tt/attempt_001`
