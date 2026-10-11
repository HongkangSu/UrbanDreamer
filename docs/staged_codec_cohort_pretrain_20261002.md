> 已被用户澄清的逐窗口标签取代。本记录的固定序列C0转移训练已停止，旧checkpoint保留；不能作为新主线严格续训点。仅独立codec可提取后重新验收。当前见 [逐窗口标签与吞吐修正](per_transition_pretrain_20261002.md)。

# 分阶段编解码与初始车辆集合预训练

2026-10-02，Ingolstadt。用户要求先将编解码训练到可用，再启动完整离线预训练，并去除车辆数预测标签中未来新增车辆。当前运行使用独立输出目录，不改既有存档或采集进程，不新增依赖。

## 训练契约

生命周期为 `CODEC_PRETRAIN → PRETRAIN → ONLINE_TRAINING → FINAL_EVALUATION`。本次完整城市任务使用 `--pretrain-only` 的等价 pipeline 控制，仅离线训练，不自动启动在线微调。完整 checkpoint 为 schema 8，有限在线 replay 为 schema 6；旧 NPZ schema 1/2 仍只读导入。

Codec 单独优化原始 FlowProjector。车辆密度先乘道路容量成为物理车辆数，监督使用 log1p 计数误差、全路／有车路加权及归一化质量误差；速度使用比例误差，移除先前近零速度倒数带来的梯度失衡。实际解码速度为正值。FM 不再同时更新 codec，也不以编解码重建项训练 dynamics。

所有验收使用独立需求身份的验证轨迹，验证数据不用于增强或梯度更新。原始状态及 cohort 混合状态都须通过：有车路车辆数 MAE ≤0.5、全路 MAE ≤0.1、空路 MAE ≤0.05、总车辆数相对误差 ≤0.1、实际速度 MAE ≤1.5 m/s。缺失指标、非有限指标、过期的更新版本或不同验证清单不能授权 FM。

转移监督固定整条序列开始时已出发且未到达的车辆集合 C0。后续出发车辆不进入车辆数标签、动作事件或 active exposure。处于内部连接的车辆保留集合成员资格，但不计入外部道路车辆数；到达按实际观察到达时间积分。动作与 exposure 仍以原 episode 人口归一化。同车多次动作保留。编解码重建标签始终是输入帧自身车辆数。

速度、占有率、停车量保留真实完整背景：旧存档没有逐车速度，不能精确构建这些通道的 cohort 专属数值。因此这是混合状态预测，不是删除未来交通需求后的物理反事实仿真。

## 冻结数据与资源

输入冻结于 `2026-10-02 11:50:02 UTC`，两个既有采集根目录共 3,597 个已提交 Ingolstadt 训练 NPZ。78 个未提交文件未纳入。按解压内容验证出 157 个重复，去重后 3,440 个 episode；共 419,916 个窗口、76,435,347 个动作事件。后续新增文件不扩充本次训练集。

按 realized demand identity 分组：2,920 个训练 episode（356,341 窗口、64,874,952 事件、312,541 个 L16 起点），520 个验证 episode（63,575 窗口、11,560,395 事件）。完整 FM 预算 5 epoch，共 1,562,705 条序列。验证需求保留集合沿用 pilot 的全部 176 个保留身份，不将同一需求不同策略轨迹拆到两边。Codec 首轮使用冻结 pilot 的 80 个训练／20 个验证 episode，每训练 epoch 10,050 个有效帧，每次原始状态验证覆盖 2,540 帧。C0 验证每 episode 4 个均匀序列起点、每序列 4 个均匀有效当前帧及真实下一边界速度；终止边界不伪造速度标签。新增强／验收使用实际 16 步预测跨度，保留旧 4 步失败诊断。

缓存通过完整解压成员 SHA256 绑定源内容，使用注册车辆表、边界 CSR 和事件索引存储，保留原始流量和事件。最多缓存两个 episode。无损重新压缩允许恢复，实质内容变化拒绝。每 FM epoch 打乱 episode 和合法序列起点，完整访问一次并保留尾 batch；checkpoint 保存游标、随机状态、清单身份和有限在线 replay，不嵌入数据集。完整城市 runner 在第一次 FM optimizer 更新后保存，并在后续最多 300 秒或 500 次更新的完整更新边界保存，支持恢复未完成 epoch 的剩余起点；标准 CLI 仍默认按 epoch 保存。

现有 GPU 剩余显存不足，codec 使用 CPU 单线程及与稠密图乘法输出／参数梯度等价的稀疏乘法。完整任务监测空闲资源后选择 GPU1 或 CPU，不修改其他任务。

## 当前实测

首轮第 4,500 次 codec 更新仅原始状态通过，C0 状态失败；随后仅使用训练80 episode构造序列长度16的C0增强，保留原始监督。第6,000次更新，两组20-episode独立验证同时通过：

| 指标 | 原始状态 | C0 混合状态 | 门槛 |
|---|---:|---:|---:|
| 有车路车辆数 MAE | 0.361826 | 0.476098 | ≤0.5 |
| 全路车辆数 MAE | 0.048614 | 0.040635 | ≤0.1 |
| 空路车辆数 MAE | 0.001237 | 0.001056 | ≤0.05 |
| 总车辆数相对误差 | 0.026179 | 0.098563 | ≤0.1 |
| 速度 MAE，m/s | 0.177723 | 0.159456 | ≤1.5 |

恢复6,000 checkpoint后重验，所有指标逐bit相同，没有追加优化更新，正常退出。完整520 episode复验：原始状态有车路0.345374、全路0.043317、空路0.001092、总量相对0.030774、速度0.176221，均通过；C0相应0.453881、0.034151、0.000902、0.104104、0.157933，总量相对误差略超门槛。因此流程继续在完整训练分区校准codec，未将原始／20episode验收冒充完整分区通过。复验时GPU1有足够空闲显存，完整任务采用cuda:1及6%进程显存上限。

C0验证是每episode4起点×4帧的代表性抽样，不是全部合法序列与帧组合；原始状态验证遍历所有有效当前帧。

完整520 episode复验首次略微失败后，pipeline仅在训练分区继续独立codec更新至6,500，全部门槛同时通过：

| 指标 | 完整留出原始状态（63,575帧） | 完整留出C0状态（8,320帧） | 门槛 |
|---|---:|---:|---:|
| 有车路车辆数 MAE | 0.287882 | 0.378856 | ≤0.5 |
| 全路车辆数 MAE | 0.036038 | 0.028437 | ≤0.1 |
| 空路车辆数 MAE | 0.000833 | 0.000678 | ≤0.05 |
| 总车辆数相对误差 | 0.027944 | 0.076805 | ≤0.1 |
| 速度 MAE，m/s | 0.164271 | 0.146005 | ≤1.5 |

完整FM已经在2026-10-02 16:03:22 UTC开始，首次更新world loss0.058710、exposure读出loss0.378316，codec_frozen=true，labels=fixed_initial_cohort。第一个完整optimizer group为4个batch×2条序列，8个起点；其后立即保存schema8 PRETRAIN/epoch0 checkpoint。16:05:26 UTC运行心跳为116次更新，world loss0.008449、readout loss0.002073，数值有限。这是训练批次损失，不是held-out交通预测性能证据。

在独立CPU进程严格载入这个真实GPU训练checkpoint，配置指纹及输入哈希匹配，codec6500就绪proof有效、codec冻结、world_model_updates=1、sampler游标8。恢复参数与保存状态逐bit相同，离线游标相同，在线replay为空、环境episode cursor=0。真实checkpoint复验保存于 `outputs/ingolstadt_complete_cohort_pretrain_20261002/resume_verification.json`；单测还验证实际FM半个epoch中断后的剩余起点、最终模型／optimizer／全部RNG与无中断运行逐bit一致。

实时任务为tmux `ud_full_pretrain_20261002`，当前PID1865631。5个完整FM epoch尚未完成，任务持续运行；不会自动进入在线训练。日志：`outputs/ingolstadt_complete_cohort_pretrain_20261002_supervisor/run.log`；状态：`outputs/ingolstadt_complete_cohort_pretrain_20261002.status.json`；实际模型及checkpoint：`outputs/ingolstadt_complete_cohort_pretrain_20261002/checkpoints/latest.pt`。注意初始80episode生产者的`codec_latest.pt`仍为6,000步；完整分区进一步校准后的6,500步codec位于实际完整任务checkpoint中。

最终完整回归 **648 passed，363.20秒**，Python编译及`git diff --check`通过。环境未安装ruff/mypy/pyright，没有为检查新增依赖。上述结果证明编解码重建、标签接口、完整预训练启动与恢复的正确性，不证明交通控制性能提升。

完整任务目录另导出 `run_config.toml`，所有来源／输入／输出路径为绝对路径。已重新严格解析并与真实checkpoint语义指纹逐字相等。五epoch任务完成并退出后，可用 `python -m urban_dreamer --config outputs/ingolstadt_complete_cohort_pretrain_20261002/run_config.toml` 接续在线微调；本次未执行该命令。
