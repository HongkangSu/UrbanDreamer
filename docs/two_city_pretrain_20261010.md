# 两城已有数据的 UrbanDreamer 预训练

2026-10-10 启动 Coulommiers 和 Montereau 的独立离线预训练。运行目录为
`outputs/two_city_urbandreamer_pretrain_20261010`，两城各有独立的输入副本、
训练/验证划分、codec 和世界模型检查点。执行源码及 SHA256 清单也已冻结。

| 网络 | 冻结的已提交轨迹文件 | 去重后轨迹 | 世界模型训练集 | 世界模型验证集 |
| --- | ---: | ---: | ---: | ---: |
| Coulommiers | 3912 | 3908 | 3294 | 614 |
| Montereau (`montereau_fault_yonne`) | 2952 | 2949 | 2505 | 444 |

数据只来自 `city_baselines_v5_1500_noprior_penalty_20261009` 的 train 子目录。
先读取各方法已持久化的训练游标，再冻结明确文件名单；不包含运行中的尾部、
评估轨迹或其他网络。已核对路网、车辆需求、道路顺序、事件协议、Poisson
需求过程、meso 物理设置及无 incident 设置。旧批次的路网或需求版本不同，
没有混入本轮。对 CRC 相同的轨迹候选使用完整轨迹 SHA256 核实去重。

按实际需求实例分组，确定性地保留 15% 的需求组作验证，同一个实例下所有
方法的轨迹属于同一分区。因为各组包含的方法数量不同，验证轨迹占比不必
恰好为 15%。两城 codec 各使用按方法均衡抽取的 256 条训练轨迹和 48 条
验证轨迹；世界模型使用表中完整划分，并在训练前复查完整验证分区。

每城先并行进行 codec 训练与 compact-cache 构建。codec 使用 batch 4、
最多 20000 次更新、每 500 次更新验证，使用配置中的 occupied/slow speed
损失权重和速度质量门槛。原始帧、cohort 帧均须通过车辆计数和速度质量门。
`decoder_allow_unready=false`。全部缓存就绪、codec 验收通过及显存达到启动
条件后，监督器自动调用 `run_complete_pretrain.py` 完成世界模型预训练。
最初启动使用五个 epoch、batch1/累积32的低显存配置；下文记录用户要求
改用 Ingolstadt 配置后的实际运行设置。配置使用统一的文稿奖励默认值；
现有数据用于物理动态监督。

Coulommiers 使用 GPU 0，Montereau 使用 GPU 1。codec 的 allocator 上限
分别为各 GPU 的 3%。现有 baseline、Ingolstadt 在线训练与数据压缩进程继续运行。
监督器只包含离线预训练阶段。新增采集轨迹保留在原数据目录，本次冻结后
不自动改变训练集或验证集。

每城的进度和证据文件：

- `preparation_summary.json`：冻结规模、方法组成、已提交游标。
- `supervisor_status.json`：阶段、心跳、进程 PID、实际执行命令及异常。
- `codec.log`、`codec/progress.json`、`codec/metrics.jsonl`：codec 更新和验证。
- `cache/cache_progress.json`、`cache/cache_manifest.json`：缓存进度及最终清单。
- `world_model.log`、`world_model_ingo_profile.status.json`：当前完整验收和 FM 更新。
- `world_model_ingo_profile/pretrain_summary.json`：当前预算完成后的预训练总结。

后台 tmux session：`pretrain_coulommiers_20261010` 和
`pretrain_montereau_fault_yonne_20261010` 是原始配置的历史启动记录；当前为
`pretrain_coulommiers_ingo64_20261010` 和 `pretrain_montereau_fault_yonne_ingo64_20261010`。
`ingo_profile_launch_manifest.json` 保存当前启动命令；
如需恢复，应先检查当前监督器状态和日志，避免重复启动同一城市。

相关准备、阶段命令和 codec 验收测试共 115 项通过。启动阶段仍需以真实
GPU 更新、缓存进度和最终质量指标为准；预训练启动不代表模型已经通过验收。

启动复核已确认两城各完成至少 200 次实际 codec 梯度更新，缓存也已产生
已验证的 compact episode；两个监督器、现有 baseline controller、Ingolstadt
在线训练和原始数据压缩进程均保持运行。`startup_verification.json` 保存冻结
源码 184 个文件的 hash 复核、需求分区及首次实际更新证据。

## 用户要求的 Ingolstadt 配置

2026-10-10 用户要求两城采用与 Ingolstadt 相同的训练配置，明确使用 batch64。
当前预算和核心运行参数为：一个 epoch、batch64、累积1、single-step FM、
sequence length1、BF16、activation checkpointing、编译完整批次、四个线程、
space/time/MLP chunk64/65536/65536。数据、输入版本和奖励设置保持原有身份。

Montereau 从已提交的420次优化更新、13440个训练窗口继续。迁移证明保留
全部模型、Adam状态、RNG和采样顺序/游标，只有checkpoint外层配置身份改变。
原有检查点保留在 `world_model`，新目录为 `world_model_ingo_profile`；
`runtime_migration.json` 保存来源、目标和保留证明。Coulommiers 使用codec4000
检查点，Montereau使用codec4500检查点，两城均已通过原始及cohort pilot质量门。

共享GPU的其他训练占用约17–19GiB，直接运行大路网batch64出现过显存不足。
最终经过真实CUDA验证的执行方式只调整运行时激活存储和不足64的尾批：
保存到CPU的张量保留dtype和stride，广播邻接只保存唯一数据；63窗口尾批
在transformer内分成32+31执行，再拼接回63并使用原来的整体loss和一次
optimizer更新。没有重复、跳过或重新加权训练窗口。完整批次仍为64并编译。

实际Montereau大路网probe的完整64和尾批63均完成前向、反向、readout及decoder
更新，encoder保持冻结，梯度均有限。峰值allocated5422.59MiB、reserved5736MiB；
两种批次的实测时长约5.01/5.15秒。记录位于
`montereau_fault_yonne/benchmark_ingo64_tail_split/batch_64.json`。
这个私有吞吐测试使用完整缓存中两个真实的高事件数轨迹，生产训练仍使用
全部3294/2505条训练轨迹及614/444条完整验证轨迹。

用户随后明确要求保持全部baseline运行。Coulommiers已从codec4000补训到4500，
完整cohort总量相对误差降为0.0568067，全部质量门通过，PRETRAIN检查点已保存。
其第一次FM更新遇到同卡baseline显存增长而OOM，没有已提交的FM更新丢失。
当前两城都以batch64在资源等待队列中，GPU0/1空闲约5272/5074MiB。

当前监督器使用 `source_baseline_priority_20261010` 冻结源码，增加2GiB启动
余量，要求至少8192MiB空闲显存后才启动；CUDA OOM会释放自己的进程、
等待资源，再从同一配置和已提交检查点恢复。只对当前attempt的CUDA OOM
重试，codec质量错误等其他异常仍明确报告。baseline和Ingolstadt在线训练
没有被暂停。选择记录在 `baseline_priority_manifest.json`。
相关代码测试127项通过，另有实际CUDA张量布局及梯度一致性检查通过。
