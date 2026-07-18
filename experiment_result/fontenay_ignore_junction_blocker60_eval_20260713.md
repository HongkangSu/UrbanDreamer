# Fontenay-en-Parisis 无 teleport 100% completion 补充评测（训练时执行器阶段更新）

最近更新：2026-07-17 15:35:43 UTC。

## 结论

保持 `sumo_time_to_teleport=-1`，仅在原 Fontenay 评测协议中增加
`ignore-junction-blocker=60` 后，UrbanDreamer 当前主线、freeflow greedy、
live-current-TT greedy、IPPO 和 MAPPO 均在 episode 51–55 达到 100% completion。
本次更新已将 IPPO 替换为训练时使用的 strict valid-mask categorical
sampling 结果；其余行暂时保留 2026-07-13 的 deterministic 结果，MAPPO
训练时执行器重跑仍在进行中。

这是一组独立的 Fontenay rescue-protocol 结果，不覆盖
[三城市原统一协议正式表](three_city_torchrl_baselines_unified_eval_20260713.md)。
如果将该设置采纳为正式 benchmark 协议，三座城市应统一使用该设置重跑。

## SUMO 协议

```text
--time-to-teleport -1
--collision.action warn
--ignore-junction-blocker 60
```

其余关键条件不变：每种方法 5 个 episode；episode IDs 为 51–55；每轮
2068 辆车，其中 827 辆 CAV、1241 辆 HDV；冻结 post-mutation/HDV 状态；
事故、路由候选、seed、7200 秒 drain buffer 和 9000 秒上限均与原正式评测一致。

## 最终结果

数值为 episode 等权宏平均 ± population standard deviation（ddof=0），单位为分钟。

| 方法 | 策略执行 | Completion | 完整 episode | t_all | t_CAV | t_HDV | Collisions | Teleports |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| UrbanDreamer 当前主线 | deterministic argmax（旧） | 100.00% ± 0.00% | 5/5 | 4.6601 ± 0.0113 | 4.6310 ± 0.0146 | 4.6794 ± 0.0097 | 0 | 0 |
| Greedy freeflow | deterministic rule（旧） | 100.00% ± 0.00% | 5/5 | 4.6824 ± 0.0098 | 4.6762 ± 0.0065 | 4.6865 ± 0.0123 | 0 | 0 |
| Greedy live-current-TT | deterministic rule（旧） | 100.00% ± 0.00% | 5/5 | 4.6909 ± 0.0233 | 4.6925 ± 0.0384 | 4.6899 ± 0.0150 | 0 | 0 |
| IPPO | categorical sample | 100.00% ± 0.00% | 5/5 | 5.4382 ± 0.0217 | 6.6107 ± 0.0584 | 4.6569 ± 0.0051 | 1 | 0 |
| MAPPO | deterministic argmax（旧） | 100.00% ± 0.00% | 5/5 | 6.7111 ± 0.0351 | 9.7062 ± 0.0471 | 4.7152 ± 0.0339 | 3 | 0 |

## 单变量 A/B 证据

在原协议中稳定失败的 freeflow episode 51 上，仅增加
`ignore-junction-blocker=60`：

| 配置 | 到达 | Completion | 终局 running/waiting | Teleports |
|---|---:|---:|---:|---:|
| 原统一协议 | 1743/2068 | 84.2843% | 321/4 | 0 |
| 增加 `ignore-junction-blocker=60` | 2068/2068 | 100.0000% | 0/0 | 0 |

## 完整性与无 teleport 审计

- 5 种方法 × 5 个 episode，共 25/25 个 episode 有效。
- 25 个逐车 CSV 共 51,700/51,700 行；每轮 2068/2068 辆到达。
- 25/25 份稳定 ordinal SUMO statistics 均记录
  `time-to-teleport=-1` 和 `ignore-junction-blocker=60`。
- 每轮均为 `loaded=2068`、`inserted=2068`、`running=0`、`waiting=0`。
- `teleports.total/jam/yield/wrongLane` 在全部 25 轮中均为 0。
- `emergencyStops=0`；collisions 总计 4 次：IPPO 1 次、MAPPO 3 次。
  `collision.action=warn` 没有删除车辆，所有车辆均有正常到达记录。
- 新 IPPO 共审计 32,813 条决策，其中 343 条不是 masked argmax；0 条选择
  违反 valid mask。逐车原始记录、执行器配置和 5 份 SUMO statistics 的复核
  均通过，teleports 为 0。
- 旧结果独立审计器普通输出与 `--json` 输出均以退出码 0 完成；新训练时
  执行器审计器同样以退出码 0 完成。

审计脚本：
[`tmp/audit_fontenay_jblock60_5ep_20260713.py`](../tmp/audit_fontenay_jblock60_5ep_20260713.py)。
新执行器审计脚本：
[`tmp/audit_torchrl_native_eval_20260716.py`](../tmp/audit_torchrl_native_eval_20260716.py)。

结果目录：
[`eval_sumo_ignore_junction_blocker60_5ep_20260713`](../results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_sumo_ignore_junction_blocker60_5ep_20260713)。
新 IPPO 结果：
[`attempt_004`](../results/fontenay_en_parisis_current_to_dest_both494_incident_warmup50_fastcap_20260609/eval_torchrl_ep500_native_blocker60_long9000_5ep_20260716/ippo/attempt_004)。
