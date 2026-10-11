# 历史记录：含 warmup 阶段的 Event 主线实施验证

本报告验证的是先前 **warmup → 世界模型初始化 → online → evaluation** 生命周期，
以及 checkpoint schema 6。当前主线已改为 **PRETRAIN → ONLINE → EVAL → COMPLETED**，
使用 checkpoint schema 7，仅从已有兼容 archive 预训练，不再现场采集 warmup。
下列原始指标和报告路径保留用于历史对照，不能作为新生命周期的验收记录。
当前验证见 [离线预训练验证记录](offline_pretrain_validation.md)。三城市旧来源的
事件编码／FM 梯度证据仍描述对应已验证的数据与模型操作。

验证日期：2026-10-02。实施前基线提交为 `0041fae`。当前验证使用已有 URB
环境，没有新增依赖。验证对象包括原始事件、观测时钟、FM 条件、事件 Q、actor、
离线来源和完整 checkpoint。短程结果只证明功能、数值有限和恢复能力。

## Toy 真实执行与恢复

TraCI 和 libsumo 均完成真实 warmup、世界模型初始化、真实在线采集、世界模型
在线更新、事件 Q/actor 更新及确定性评估。完整 checkpoint schema 6 保存后，
新 trainer 加载并重复固定种子评估，事件数、需求人口、旅行时间、共享回报和完成率
完全一致。两个后端的这些统计也一致。

短程预算为 CPU、8 辆参考需求、30 秒 Poisson 到达期、180 秒真实期限；初始化
sequence length 1、batch 2，在线世界模型更新 1 次、imagination pass 1 次、
2 条 rollout、最多 3 个宏观窗口。该缩减预算只属于验证工具。

| 指标 | 值 |
|---|---:|
| Warmup 人口 / 原始动作事件 / 窗口 | 8 / 10 / 5 |
| 初始化序列 / optimizer 更新 | 5 / 3 |
| 初始化世界模型 loss | 1.0900634587 |
| 在线人口 / 原始动作事件 / 窗口 | 7 / 12 / 6 |
| 在线世界模型 loss | 0.3632777631 |
| 想象动作事件 | 8 |
| 事件 Q loss | 1.2153385878 |
| 宏观 V loss | 1.1200278997 |
| 反事实基线 mean / std | 0.0225850418 / 0.0168616436 |
| Global advantage mean / std | -1.0662631989 / 0.2731405795 |
| Actor advantage mean / std | -1.0662631989 / 0.2731405795 |
| Actor loss | -0.9423780441 |
| 固定评估人口 / 事件 | 12 / 24 |
| 固定评估 mean travel minutes | 2.3583333333 |
| 固定评估完成率 | 1.0 |

本次 Toy 样本没有个体超时，因此 actor 与 global advantage 统计一致。独立回归
测试覆盖有超时的优势缩放和 Q 目标排除个体惩罚。验证的 warmup 去重统计为
1 个发现文件、1 个导入轨迹、0 个重复；未配置外部离线来源。

原始报告：

- TraCI：`outputs/event_mainline_verify_20261002_traci_restored/verification.json`。
- 最终 libsumo：`outputs/event_mainline_verify_20261002_completed/verification.json`。
- 完整事件 NPZ 和 checkpoint 位于对应输出目录。

## 三城市既有 archive 与模型梯度

每城市抽取 random、online_ac、gnac 的第一个训练 archive，共 9 个 episode、
1,143 个窗口、522,714 个原始动作事件。9 个文件均为独立轨迹，本次真实抽样
去重数为 0；跨目录、跨 provenance、跨 offline/warmup 的重复由回归构造验证。
这 9 个 episode 均截断，保留 exposure 监督，并保持非终止标记和无完成旅行时间标签。

| 城市 | 导入 episode | 导入事件 | 全路网边数 | 模型样本窗口事件 | Recursive FM loss |
|---|---:|---:|---:|---:|---:|
| Gretz | 3 | 104,939 | 4,485 | 479 | 0.06463585794 |
| Fontenay | 3 | 313,982 | 7,585 | 2,111 | 0.07603225112 |
| Ingolstadt | 3 | 103,793 | 853 | 611 | 0.07252916694 |

各城市在完整路网上进行 batch 1、深度 1 的 FM 前向与反向；预测、损失和事件
embedding / MLP 梯度均有限且非零。为检查学习开始后的梯度路径，验证脚本对
默认零初始化输出头设置确定性的 `std=0.01` 非零权重。Fontenay 验证进程峰值
内存约 3.74 GiB，CPU 单线程耗时约 144 秒。详细来源、manifest 身份及梯度
统计见 [城市验证 JSON](event_city_validation.json)。既有 archive 和采集任务保持原状态。

## 回归与检查

第一轮完整回归为 **509 passed**，耗时 375.81 秒；最终完整回归为
**513 passed**，耗时 354.84 秒。随后增加的离线哈希回归包含在最新专项
**20/20 passed** 中；离线、archive、配置和压缩专项共 **69 passed**。
主线集成的七个测试文件共 **88 passed**，包含实际 TraCI 生命周期和 checkpoint 恢复。
专项测试同时检查被排除重复文件在同次运行内变化时立即拒绝，且每个 epoch 不对
选中的 archive 重复执行完整哈希扫描。

关键行为覆盖事件边界、短分叉跳过、同车多事件、内部连接和跨窗口承诺；联合编码
可加性、置换不变性、替换与重聚合一致、FM 梯度及条件缓存；Q 的因果前缀、合法
mask、冻结基线、梯度隔离与个体惩罚分离；NPZ 兼容、空人口、截断、缓存、完整
遍历、manifest 去重、重压缩恢复、追加隔离和内容变更拒绝。

Python 编译和 `git diff --check` 通过。当前环境没有安装 ruff、flake8、mypy 或
pyright，未为检查安装新工具。初始化恢复在 pipeline 的完整 epoch 边界执行。
全量城市训练和交通性能改善验证不属于本次短程验收。
