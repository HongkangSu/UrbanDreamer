> 历史验证记录：下文对应分阶段 codec 改造前的 schema 7。当前主线为
> `CODEC_PRETRAIN → PRETRAIN → ONLINE_TRAINING → FINAL_EVALUATION`、checkpoint 8 / replay 6；
> 当前验收见 [分阶段编解码与初始车辆集合预训练](staged_codec_cohort_pretrain_20261002.md)。

# 离线预训练主线验证

验证日期：2026-10-02。当前生命周期为 `PRETRAIN → ONLINE_TRAINING →
FINAL_EVALUATION → COMPLETED`，checkpoint schema 为 7。之前 warmup 生命周期
的验证报告保留为历史记录；本报告描述直接读取既有离线 NPZ 的主线。

## 行为与恢复

- `world_model_initialization_epochs` 控制离线预训练 epoch；主线不包含 warmup
  采集阶段、warmup replay 或 warmup 数据源。
- 预训练只更新世界模型及 active-fraction readout，actor、宏观 V、事件 Q 参数
  保持原值。离线轨迹不进入在线 replay，也不消耗真实 episode 编号。
- 没有配置来源或没有合法序列时明确失败，不启动 SUMO。已存在但为空的来源目录
  不生成空 manifest 或初始 checkpoint，补齐数据后可以在原输出目录正常启动。
- 完整 epoch 的序列起点访问一次，保留尾部不足 batch；epoch 边界 checkpoint
  恢复后访问次序及模型参数一致。在线首个需求 episode 编号为 0。
- schema 6 的完整 checkpoint 和包含 warmup replay 的旧 trainer 状态被拒绝；
  NPZ schema 1/2 和 replay schema 5 的数据格式保持兼容。

## 已有 NPZ 的短程运行

输入来源为已存在的
`outputs/event_mainline_verify_20261002_completed/episodes/warmup/episode_000000.npz`。
它来自先前验证，当前运行只复制原始存档字节供独立冻结，没有生成新的预训练数据。
验证 profile 为 reference-8/T30、CPU、180 秒真实期限；预训练 sequence 1、
batch 2、1 个 epoch，在线世界模型更新 1 次、imagination pass 1 次、
2 条 rollout、最多 3 个窗口。

TraCI 和 libsumo 均完成离线预训练、在线微调、确定性评估以及 checkpoint 保存／
恢复，恢复后的需求人口、事件数、观察费用及完成率与保存前完全一致。两种后端
的这些结果也一致。

| 指标 | 结果 |
|---|---:|
| 离线导入 episode / 原始事件 / 窗口 | 1 / 10 / 5 |
| 预训练 SUMO session | 0 |
| 预训练合法序列 / optimizer 更新 | 5 / 3 |
| 预训练世界模型 loss | 1.0943108916 |
| 首个在线 demand index | 0 |
| 在线人口 / 原始事件 / 窗口 | 8 / 13 / 6 |
| 在线世界模型 loss | 0.3720310330 |
| 想象事件 / Q loss | 8 / 1.2107874155 |
| 反事实基线 mean / std | 0.0368601605 / 0.0170778595 |
| Actor 优势 mean / std | -1.0732581615 / 0.2343576550 |
| 固定评估人口 / 事件 | 12 / 24 |
| 固定评估完成率 | 11/12 |

固定评估的 180 秒短期限留下 1 辆未完成车辆；报告中的 `evaluation_t_all` 是
实际观察 exposure 的人口均值，不能当作完成 episode 的平均旅行时间。
此次验收要求功能、数值有限和可恢复，不作为交通性能提升证据。

完整机器可读结果：

- `outputs/offline_pretrain_verify_20261002_traci/verification.json`
- `outputs/offline_pretrain_verify_20261002_libsumo/verification.json`

另构造与原 NPZ 数值相同、但配置位于外部 `protocol.json` 的兼容来源，完成
同样的离线预训练／在线／恢复验证。验证工具先解析原来源的冻结配置，再把它与
原始存档字节一起复制到独立验证目录，避免丢失旧存档的祖先目录来源信息。
该报告为 `outputs/offline_pretrain_verify_20261002_external_config/verification.json`。

## 配置与回归

三个城市的主配置和 `_offline` 配置指向两个既有采集根目录，使用独立的
`*_pretrain_eventq` 输出。六份冻结城市采集 TOML 仍可解析，历史 warmup 数值
在来源 snapshot 中保留；主线不执行这些弃用字段，也不将它们纳入语义指纹。

生产 Toy 要求 reference-320/T420，当前发现的 reference-50/T420 和
reference-8/T30 NPZ 不满足其需求配置。生产 Toy 的来源列表保持为空，并在
启动时明确要求补齐兼容来源；本次没有启动采集来补数据。

主线 pipeline、training、checkpoint、CLI 和 FM 集成专项 **72 passed**。
配置、离线来源及既有采集工具兼容专项 **75 passed**。Python 编译和
`git diff --check` 通过；未新增依赖。收尾相关回归 **133 passed**；
最终完整回归 **525 passed**，耗时 **350.11 秒**。
