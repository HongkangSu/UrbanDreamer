# 三城市 TorchRL baseline 统一评测主表

最近更新：2026-07-22T17:33:30.328227+00:00。

统一协议：每个方法 5 个 checkpoint-eval episode；HDV post-mutation 状态冻结；`sim_buffer_after_last_depart=7200`；`max_steps_per_episode=9000`；`sumo_time_to_teleport=-1`。

MAPPO 使用训练时 valid-mask categorical sampling；IQL 使用终局 ε=0.05 的 valid-mask epsilon-greedy，并保持 `first_decision_per_vehicle` 探索范围。

Fontenay 训练使用 blocker60 以避免 junction blocker，但本主表评测仍使用与另外两城相同的原统一协议，即只设置 `collision.action=warn`，不启用 `ignore-junction-blocker`。Fontenay MAPPO/IQL 为 200ep checkpoint；Gretz 与 Ingolstadt 为 500ep checkpoint。

数值单位均为分钟。t 指标为 episode 等权宏平均 ± 总体标准差（ddof=0）。P95 为 5 个评测 episode 的逐车 pooled P95，TT max 为全局逐车最大旅行时间。

训练代价采用完成率校正代理：`c_g = mean_ep(mean_arrived_tt_g / completion_g - matched_t_pre,g)`，其中每一组使用自己的完成率，`t_pre,g` 来自同需求、同 incident 的 matched pure-HDV reference。

## Gretz-Armainvilliers

| 方法 | 策略执行 | t_all | t_CAV | t_HDV | c_all (adj.) | c_CAV (adj.) | c_HDV (adj.) | P95 | TT max | Completion | Δt_all vs 主线 | 口径 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| UrbanDreamer（当前主线） | deterministic argmax | 4.1136 ± 0.0064 | 3.2349 ± 0.0049 | 4.6979 ± 0.0075 | 2.7465 | 4.4054 | 1.6417 | 15.9333 | 38.4833 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 3180/3180 到达) | +0.0000 | 全体车辆 |
| IPPO | categorical sample | 4.6518 ± 0.7000† | 5.1370 ± 0.5144† | 4.3230 ± 0.8461† | 1.0770 | 2.8922 | -0.1155 | 21.6600† | 44.5500† | 96.48% ± 4.31% (min 91.19%; 3/5 完整; 3068/3180 到达) | — | 仅到达车辆† |
| MAPPO | categorical sample | 4.1571 ± 0.6096† | 3.4227 ± 0.4386† | 4.6477 ± 0.7210† | 0.2100 | 0.7233 | -0.1398 | 18.3800† | 38.5000† | 98.27% ± 3.46% (min 91.35%; 4/5 完整; 3125/3180 到达) | — | 仅到达车辆† |
| IQL | epsilon-greedy (ε=0.05), first_decision_per_vehicle | 4.3207 ± 0.1445 | 3.6129 ± 0.1365 | 4.7913 ± 0.1543 | 0.6407 | 1.7738 | -0.1044 | 20.2875 | 38.5833 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 3180/3180 到达) | +0.2071 | 全体车辆 |
| Greedy（freeflow） | deterministic freeflow | 4.8212 ± 0.0027 | 4.1366 ± 0.0022 | 5.2764 ± 0.0031 | 1.7789 | 1.3136 | 2.0788 | 24.9333 | 38.5500 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 3180/3180 到达) | +0.7075 | 全体车辆 |
| Greedy（live current TT） | deterministic live-TT | 4.6897 ± 0.0002 | 3.9693 ± 0.0012 | 5.1687 ± 0.0010 | 1.6474 | 1.1463 | 1.9711 | 23.0667 | 38.5333 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 3180/3180 到达) | +0.5761 | 全体车辆 |

## Ingolstadt

| 方法 | 策略执行 | t_all | t_CAV | t_HDV | c_all (adj.) | c_CAV (adj.) | c_HDV (adj.) | P95 | TT max | Completion | Δt_all vs 主线 | 口径 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| UrbanDreamer（当前主线） | deterministic argmax | 4.4684 ± 0.0000 | 4.4991 ± 0.0000 | 4.4479 ± 0.0000 | 4.1992 | 9.4574 | 0.6938 | 7.0500 | 13.9333 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +0.0000 | 全体车辆 |
| IPPO | categorical sample | 5.6794 ± 0.0321 | 7.5324 ± 0.0759 | 4.4440 ± 0.0166 | 1.9944 | 4.9405 | 0.0620 | 14.6933 | 38.9667 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +1.2110 | 全体车辆 |
| MAPPO | categorical sample | 4.6163 ± 0.0150 | 4.8738 ± 0.0310 | 4.4447 ± 0.0117 | 0.4431 | 1.0329 | 0.0555 | 7.5333 | 22.5000 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +0.1479 | 全体车辆 |
| IQL | epsilon-greedy (ε=0.05), first_decision_per_vehicle | 4.5851 ± 0.0143 | 4.8325 ± 0.0359 | 4.4202 ± 0.0148 | 0.7501 | 1.9042 | -0.0030 | 7.4167 | 27.8000 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +0.1167 | 全体车辆 |
| Greedy（freeflow） | deterministic freeflow | 4.5419 ± 0.0000 | 4.5907 ± 0.0000 | 4.5093 ± 0.0000 | 0.1548 | 0.3195 | 0.0449 | 7.3333 | 15.4500 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | +0.0735 | 全体车辆 |
| Greedy（live current TT） | deterministic live-TT | 4.4100 ± 0.0000 | 4.3751 ± 0.0000 | 4.4333 ± 0.0000 | 0.0229 | 0.1039 | -0.0311 | 6.6167 | 10.7167 | 100.00% ± 0.00% (min 100.00%; 5/5 完整; 5175/5175 到达) | -0.0584 | 全体车辆 |

## Fontenay-en-Parisis

| 方法 | 策略执行 | t_all | t_CAV | t_HDV | c_all (adj.) | c_CAV (adj.) | c_HDV (adj.) | P95 | TT max | Completion | Δt_all vs 主线 | 口径 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| UrbanDreamer（当前主线） | deterministic argmax | 4.5876 ± 0.1347† | 4.6295 ± 0.0123† | 4.5565 ± 0.2365† | 0.7771 | 2.4418 | -0.3329 | 22.3800† | 40.4667† | 97.38% ± 5.24% (min 86.90%; 4/5 完整; 10069/10340 到达) | — | 仅到达车辆† |
| IPPO | deterministic argmax | 4.5182 ± 0.2792† | 5.2094 ± 0.3945† | 4.0575 ± 0.2091† | 0.7039 | 2.6535 | -0.5745 | 20.3983† | 44.0500† | 86.05% ± 5.52% (min 82.11%; 0/5 完整; 8898/10340 到达) | — | 仅到达车辆† |
| MAPPO | categorical sample | 4.5859 ± 0.4334† | 4.8561 ± 0.6334† | 4.4059 ± 0.3003† | 0.0128 | 0.7927 | -0.5075 | 21.6767† | 57.4167† | 93.86% ± 7.52% (min 84.43%; 3/5 完整; 9705/10340 到达) | — | 仅到达车辆† |
| IQL | epsilon-greedy (ε=0.05), first_decision_per_vehicle | 4.3609 ± 0.5383† | 4.5938 ± 0.8068† | 4.2056 ± 0.3614† | 0.3312 | 1.5939 | -0.5083 | 20.2533† | 48.4333† | 89.91% ± 8.24% (min 82.69%; 2/5 完整; 9297/10340 到达) | — | 仅到达车辆† |
| Greedy（freeflow） | deterministic freeflow | 4.1930 ± 0.3927† | 4.0952 ± 0.4714† | 4.2588 ± 0.3401† | -0.3828 | -0.4356 | -0.3464 | 20.1217† | 38.5667† | 90.47% ± 7.78% (min 83.85%; 2/5 完整; 9355/10340 到达) | — | 仅到达车辆† |
| Greedy（live current TT） | deterministic live-TT | 4.0537 ± 0.3121† | 3.9370 ± 0.3660† | 4.1331 ± 0.2755† | -0.3924 | -0.4955 | -0.3197 | 18.5067† | 39.8833† | 87.67% ± 6.17% (min 84.24%; 1/5 完整; 9065/10340 到达) | — | 仅到达车辆† |

## 审计与解释

- `c_all/c_CAV/c_HDV` 是 completion-adjusted proxy，不等同于把未到达车辆旅行时间设为 0；CAV 与 HDV 分别使用各自的完成率。
- UrbanDreamer 的 c 使用真实环境 collection；IPPO/MAPPO/IQL 使用所评 checkpoint 的在线训练轨迹；两个 Greedy 无学习轨迹，使用统一 5-episode fixed-policy eval proxy。
- Gretz 的 legacy UrbanDreamer collection 实际按 episode 改变 `vType` 角色 bundle；该行按每个 episode 的实际 CAV/HDV ID 分组，并用 matched pure-HDV reference 中同一组车辆作基准。其他行使用 checkpoint 对应的固定 post-mutation 角色。
- `†` 表示至少一个评测 episode 未达到 100% completion。t、P95 与 TT max 只基于已到达车辆，属于 right-censored/arrived-only 指标；缺失旅行时间没有当作 0。
- Δt_all = 方法 t_all − 当前主线 t_all；负值更好。不完整评测存在幸存者偏差，因此不报告 delta。
- P95 与 TT max 不做完成率除法；该操作对分位数与极值没有可解释的统计含义。
- JSON 中保留每行训练 source phase、checkpoint/path、episode IDs、coverage、group completion 和 reference 审计。

## 选用的 attempt

- Gretz-Armainvilliers / UrbanDreamer（当前主线）：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_urbfair_unified_detargmax_long9000_5ep_20260713/mainline/attempt_001`
- Gretz-Armainvilliers / IPPO：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_torchrl_ep500_native_long9000_5ep_20260716/ippo/attempt_002`
- Gretz-Armainvilliers / MAPPO：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_torchrl_latest_mappo_iql_native_long9000_5ep_20260720_124428_retry2_20260722_0737/mappo/attempt_001`
- Gretz-Armainvilliers / IQL：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_torchrl_latest_mappo_iql_native_long9000_5ep_20260720_124428_retry2_20260722_0737/iql/attempt_001`
- Gretz-Armainvilliers / Greedy（freeflow）：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_urbfair_unified_detargmax_long9000_5ep_20260713/freeflow_greedy/attempt_001`
- Gretz-Armainvilliers / Greedy（live current TT）：`results/gretz_armainvilliers_current_to_dest_incident_completion100_stronglivebad_collisionwarn_drain7200_20260601_neg_986733117_1_S0b_s500_e2500_v0p1/eval_urbfair_unified_detargmax_long9000_5ep_20260713/live_current_tt/attempt_001`
- Ingolstadt / UrbanDreamer（当前主线）：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_urbfair_unified_detargmax_long9000_5ep_20260713/mainline/attempt_001`
- Ingolstadt / IPPO：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_torchrl_ep500_native_long9000_5ep_20260716/ippo/attempt_002`
- Ingolstadt / MAPPO：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_torchrl_latest_mappo_iql_native_long9000_5ep_20260720_124428_retry2_20260722_0737/mappo/attempt_001`
- Ingolstadt / IQL：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_torchrl_latest_mappo_iql_native_long9000_5ep_20260720_124428_retry2_20260722_0737/iql/attempt_001`
- Ingolstadt / Greedy（freeflow）：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_urbfair_unified_detargmax_long9000_5ep_20260713/freeflow_greedy/attempt_001`
- Ingolstadt / Greedy（live current TT）：`results/ingolstadt_custom_actionfix_incident_20260516_185810/eval_urbfair_unified_detargmax_long9000_5ep_20260713/live_current_tt/attempt_001`
- Fontenay-en-Parisis / UrbanDreamer（当前主线）：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_urbfair_unified_detargmax_long9000_5ep_20260713/mainline/attempt_001`
- Fontenay-en-Parisis / IPPO：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_torchrl_ep500_detargmax_long9000_5ep_20260713/ippo/attempt_001`
- Fontenay-en-Parisis / MAPPO：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_torchrl_latest_mappo_iql_native_long9000_5ep_20260720_124428_retry2_20260722_0737/mappo/attempt_001`
- Fontenay-en-Parisis / IQL：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_torchrl_latest_mappo_iql_native_long9000_5ep_20260720_124428_retry2_20260722_0737/iql/attempt_001`
- Fontenay-en-Parisis / Greedy（freeflow）：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_urbfair_unified_detargmax_long9000_5ep_20260713/freeflow_greedy/attempt_001`
- Fontenay-en-Parisis / Greedy（live current TT）：`results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_urbfair_unified_detargmax_long9000_5ep_20260713/live_current_tt/attempt_001`
