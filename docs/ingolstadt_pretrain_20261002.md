## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-02
- Verification Status: VERIFIED (checkpoint reload reproduces fixed validation)
- Version Label: ingolstadt_offline_pretrain_pilot_v1

此文记录旧 FM pilot 的历史诊断结果。它没有当前独立 codec 原始／C0 双 gate，
不表示 codec 已达到当前物理重建阈值。`tools/pretrain_offline.py` 的旧执行入口现已
禁用；split 和采样 helper 保留。当前训练入口与验收条件见
[分阶段 codec/C0 预训练记录](staged_codec_cohort_pretrain_20261002.md)。

## 数据快照

2026-10-02 11:50:02 UTC 冻结两个采集根目录的持久提交文件清单。原始存档和
运行中的采集任务保持原状态，后续新增文件不进入本次试验。

| 来源 | Online AC | GNAC | Dreamer | MBPO | Random | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| 原始 collection | 1175 | 750 | 25 | 300 | 612 | 2862 |
| SP prior collection | 290 | 200 | 20 | 70 | 155 | 735 |
| 合计 | 1465 | 950 | 45 | 370 | 767 | 3597 |

当时有 3675 个稳定 episode 文件名，其中 78 个尚未持久提交，未纳入快照。
纳入的 3597 个文件为 3.303 GB 压缩内容，解压成员约 230.14 GB，包含
439855 个 30 秒窗口、81644059 个成功动作事件。

按 trajectory 内容去重，确认 157 个重复文件，得到 **3440 条独立轨迹、419916
个窗口、76435347 个动作事件**；其中 542 条完整、2898 条截断。先用 ZIP 成员
名称／长度／CRC 找候选，再对 314 个候选文件核验解压后 trajectory SHA256。
完整盘点验证元数据、输入哈希、边顺序、物理配置、需求与计数；逐事件转向和完整
车辆状态的严格检查在实际导入的训练／验证子集上执行，100 个文件全部通过。

完整清单及每文件统计位于 `outputs/ingolstadt_pretrain_census_20261002_1148/`。
冻结集合覆盖 1175 个实际需求身份；不同方法共享需求，不能按文件独立随机切分。

## 首轮预训练预算

为先观察模型预测质量，本轮按两个来源 × 五种方法分层，每层取 8 条训练与
2 条验证轨迹。训练 80 条、验证 20 条；对应 70 个训练需求身份和 15 个验证
需求身份，严格不交叉，跨来源重复轨迹也排除。

训练子集包含 10050 个窗口、1850345 个事件，4 条完整、76 条截断；验证子集
包含 2540 个窗口、469712 个事件，20 条均截断。截断数据不监督完成总旅行时间。

- 训练器：现有 event 联合编码世界模型与 active-fraction readout；actor、V、Q
  参数保持初始化值，不运行 SUMO、不写入在线 replay。
- 设备：RTX3090 GPU1，FP32、CPU 单线程；PyTorch GPU allocator 上限为设备的
  11%。不停止或重配置已有任务。
- 优化器：主线 AdamW `lr=1e-4`、`weight_decay=1e-2`；readout Adam `lr=1e-3`；
  clip 10，训练 seed42，batch2、sequence16、gradient accumulation1。
- 抽取每训练 episode 8 个覆盖起止与中间阶段的合法序列起点，共 640 个序列、
  320 次优化更新。使用原有每 100 更新递增监督深度，最终深度为 3。
- 验证每 episode 4 个固定起点，共 80 个序列，预测 30／60／90 秒；给定真实
  记录动作，保留两条独立潜状态噪声轨迹，再平均解码结果。噪声 seed31415 固定。
- 预训练和验证共耗时 966.6 秒（包含来源严格导入与 checkpoint）。进程 exit0。

这是固定预算试跑，尚未遍历全部合法序列起点，canonical epoch 数为 0。
它不等于全数据 5 epoch 预训练，也不提供策略或交通性能提升证据。

## 固定验证结果

使用未来真实占用路段 mask，避免大量空路段稀释误差。速度以道路 speed limit
换算 m/s；车辆数以容量换算真实车辆单位。速度标签与主线 forward endpoint mean
一致。持续性基线保持起始观测速度、车辆数及活跃比例。

| 验证指标 | 随机初始化 | 320 更新后 | 持续性基线 |
|---|---:|---:|---:|
| 30 秒占用路段速度 MAE (m/s) | 7.8793 | 1.4919 | 0.7761 |
| 60 秒占用路段速度 MAE (m/s) | 8.2267 | 1.7834 | 0.8839 |
| 90 秒占用路段速度 MAE (m/s) | 8.2155 | 2.0007 | 0.9159 |
| 30 秒占用路段车辆数 MAE (辆／路段) | 5.0602 | 4.0986 | 0.7255 |
| 60 秒占用路段车辆数 MAE (辆／路段) | 4.7760 | 3.9308 | 0.8166 |
| 90 秒占用路段车辆数 MAE (辆／路段) | 4.8464 | 3.9398 | 0.8108 |
| 30 秒外部路网总车辆数 MAE (辆) | 1157.5747 | 281.9791 | 8.0625 |
| 30 秒窗口活跃比例 MAE | 0.3333 | 0.2612 | 0.0033 |

速度改善约 81.1%（30 秒）和 75.6%（90 秒），但三种 horizon 的速度、车辆数
和活跃比例误差仍弱于持续性基线。速度在 update240 更好（30 秒 1.1850 m/s），
update320 有反弹；车辆数在 update80 后基本平台化。单批训练 loss 下降不能
作为预测质量或交通性能的充分证据。

## 编码／解码诊断

用真实端点 flow 编码再解码，和正常 rollout 使用完全相同的端点目标及占用 mask：

- 30 秒车辆数 MAE：真实端点重建 **4.0685**，正常预测 **4.0986**。
- 90 秒车辆数 MAE：真实端点重建 **3.8855**，正常预测 **3.9398**。
- 30 秒外部路网总车辆数均值：真实 **354.6**，预测 **202.4**，平均低估
  **152.2 辆**（约 42.9%）。这是所选验证时段的均值，不是 episode 总人口。
- 原始解码速度为负的比例：初始化约 76.4%，训练后 0%。

这支持当前主要车辆数瓶颈已出现在编码／解码环节，而不是仅由递归转移增加。
下一轮宜检查容量归一化 volume 监督的尺度与权重，以及如何保留观测车辆数；
再评估增加训练预算。验证仅一次 seed42、15 个 held-out 需求身份、四个时段；
这些点估计不宣称统计显著性或其他城市上的泛化。

## 产物与复核

运行目录：`outputs/ingolstadt_pretrain_pilot_20261002/`。

- `result.json`、`metrics.jsonl`：原运行前后指标、训练／验证曲线数据。
- `restored_validation.json`：初始化重现、checkpoint 恢复与配对端点诊断。
- `learning_curves.png`：训练 loss 与 held-out 预测误差。
- `checkpoints/latest.pt`：schema7 完整训练状态，停留 PRETRAIN。
- `checkpoints/world_model_pretrained.pt`：便于加载的 CPU 世界模型权重和来源身份。
- `run_config.toml`、`resolved_config.json`、`command.json`：实际配置和命令。
- `offline_manifest.json`、`validation_manifest.json`、`split.json`、`schedule.json`：
  内容身份、需求隔离与固定抽样计划。

恢复后与原记录比较：初始化指标最大差异 0，训练后最大差异
1.9073486328125e-6。新增 split、sampler、预测／单位／状态恢复测试共 13 项通过；
Python 编译、`git diff --check` 通过。训练主线、损失和架构没有为这轮试验修改。
