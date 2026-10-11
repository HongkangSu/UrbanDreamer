# Event 主线：离线预训练与在线微调

主线真实采集、想象训练和评估使用 branch-only event 协议：每 1 秒观察一次，
世界模型按 30 秒转移。TraCI 是默认后端；配置 `backend = "libsumo"` 可使用
已有的进程内 SUMO 后端。其他基线调用共享 imagination 接口时，默认保留原协议。

主线生命周期固定为 **PRETRAIN → ONLINE → EVAL → COMPLETED**。PRETRAIN 仅使用
已有、兼容的真实 SUMO 训练 NPZ，完成世界模型预训练后才开启在线真实采集、
世界模型微调与想象策略更新。预训练期间不创建 SUMO session，也不现场采集来源。

## 动作事件与世界模型

每次成功决策保存不可变 `ActionEvent`，包含车辆 ID、episode 内事件索引、决策时间、
入口边、下一跳和目的地。同车在同一窗口内的多次决策分别保存。窗口使用
`[start, start + 30)`，边界观察产生的动作属于下一个窗口。

Replay 在 batch 中按事件稀疏打包，包含 batch、时间步和连续窗口相位；车辆 ID
只用于关联和精确事件替换。主线必须提供事件，空窗口提供显式空事件集合。
缺失事件不会回退到旧边计数条件。

世界模型将事件按合法转向与三个 10 秒时间桶聚合。入口、下一跳和目的地共享
16 维路段 embedding，事件 MLP 为 49→64→32。原始聚合保存整数计数和可加
特征和；读出使用 `log1p(count)`、均值和入口/出口静态特征，空组置零，再将
入口/出口汇总的 64 维动作特征接入现有 action GNN。训练时重算 learned 特征。
FM 求解及 disagreement 多噪声采样在同一条件内复用一次编码。

想象运动在观察间隔内连续沿承诺路线推进，仅对观察到的多候选外部边调用 actor。
未观察到的短分叉沿现有确定性路线通过。承诺和路口剩余距离跨窗口保存。
活动、非待决策、非目的地快照必须提供下一跳承诺。到达和费用结算使用观察时刻；
新出发车辆从实际出发时刻计费。

## 事件 Q 与 actor

独立 `EventActionCritic` 用共享两层 128 维 MLP 批量评分合法候选。它使用窗口
起始潜状态、候选观测、目的地、相位、当前策略 context、此前事件前缀和其他
同时动作。前缀与同时动作分别保存；只删除当前事件的原始贡献，保留同车历史。
候选替换在可加计数/特征和上进行，之后使用小读出，不重跑世界模型或完整 GNN。

一次 imagination pass 完成全部采样后才更新参数。采样时保存 detached 基线：

\[
b_{i,k}=\sum_{a\in\mathcal A_i}\pi(a\mid o_{i,k})Q(C_k,a_{-i,k},a),
\qquad A^{global}_{i,k}=G_{i,k}-b_{i,k}.
\]

事件 Q 只用当前 actor 想象轨迹的已执行动作及 detached 共享回报做 MSE 监督，
使用独立 Adam，学习率 `3e-4`、梯度裁剪 10。共享目标包含交通费用与端点风险。
个体超时回报继续在 actor 优势缩放阶段添加。宏观 V 继续训练宏观价值，并用于
想象截断 bootstrap。Q 拟合同批回报后不重算 actor 基线。

## 离线来源、manifest 与恢复

主线必须配置含有兼容样本的 `[data].archive_roots`。缺少来源、集合为空或配置不兼容
时，在任何在线采集前明确失败。预训练遍历冻结离线轨迹；在线采集产生的数据用于
后续微调和策略更新，不进入本次冻结的预训练集合。

```toml
[data]
archive_roots = ["/absolute/path/to/collection_root"]
cache_episodes = 2
```

三个城市的主配置和 `configs/*_offline.toml` 均包含两个已有数据根目录，分别使用
独立的 `*_pretrain_eventq` 输出目录。Toy 主配置要求参考人口 320、Poisson 到达期
420 秒，并匹配输入哈希和物理配置。目前未发现满足条件的既有 Toy NPZ，因此
Toy 配置显式保留空来源并说明必须补充。reference-50/T420 的吞吐验证来源与
reference-8/T30 的短程验证来源都不兼容生产 Toy 配置。
导入检查城市、真实 SUMO 来源、训练分区、输入哈希、边顺序、协议、观察步长、
宏观窗口、物理与需求配置、车辆关联、合法转向及事件重建计数。旧 NPZ 两种字段
名称可读；目的地通过完整车辆表补齐。配置缺失时使用采集任务的冻结配置快照；
无法确认的来源明确拒绝并给出原因。

首次使用时生成 `offline_manifest.json`。
完整性按解压内容哈希检查，轨迹去重独立于 provenance/metrics；无损重新压缩和
ZIP 成员重排不改变身份。集合冻结后新增文件不加入；冻结文件发生内容变化时拒绝。
重复文件仍在完整性清单中；包括被排除的重复来源，内容变化也会在遍历前检查。
最多缓存两个解码 episode。每 epoch 打乱
episode 和序列起点，每个合法起点访问一次，保留不足 batch 的尾部。
预训练 epoch 完成后释放来源缓存。

截断轨迹保留实际人口、exposure 和非终止状态，不作为完成的总旅行时间标签。
导入 context 按当前主线完成 30 秒窗口后的实际期限计算，原期限保留为来源信息。
checkpoint 保存 manifest 身份、采样进度、随机状态和有限在线 replay，不嵌入离线集合。
主线 pipeline 在完成预训练 epoch 后保存 checkpoint；中断的 epoch 从前一边界重跑。
数据集支持恢复序列游标。完整 `run_complete_pretrain.py` 在首个成功 optimizer group 后保存，
此后每 300 秒或 500 次 optimizer 更新保存（先满足者触发），同时保存参数、optimizer、
来源游标、epoch 累计统计和 RNG；因此支持从已完成 group 的 epoch 内边界恢复。

预训练预算继续使用 `[run].world_model_initialization_epochs`，batch、sequence length
和梯度累积继续使用对应 `world_model_initialization_*` resource 字段。为读取冻结
采集任务配置，仍接受弃用的 `run.warmup_episodes`（非负整数，默认 0），及原有
simulation `warmup_*` 字段；这些字段不调度主线采集，也不进入主线语义指纹。
三个 simulation 字段同样可省略，默认 horizon/drain/patience 分别为
1800/3000/0 秒；冻结配置中的原始值仍保留。新主线 TOML 不包含这些字段。

新 checkpoint schema 为 **10**，replay schema 为 **7**，新 archive schema 为 **2**
（可读取 archive schema 1）。旧完整 checkpoint 被拒绝为新架构严格续训点。
语义指纹包含 `独立完整前态输入／仅减当窗新车标签／冻结codec FM语义`。先前含 warmup 阶段的完整
checkpoint 不能作为当前生命周期的严格续训点；新运行使用独立输出目录。

## 验证与当前训练入口

`tools/verify_event_mainline.py` 的单 archive Toy 验证、以及
`tools/pretrain_offline.py` 的旧 pilot 执行入口属于历史工具，现已在读档、
创建输出目录、构造模型或启动 SUMO 之前明确报错。一个 Toy archive 无法建立独立
需求身份的训练／验证分区；旧工具也没有通过实际 codec 的原始与 C0 双重质量门槛。
历史验证结果可以用于回顾功能实现，不是当前 codec 已可用于 FM 的证明。
`pretrain_offline.py` 的 split 和 sampled-schedule helper 继续保留供诊断使用。

以下示例使用已有冻结 Ingolstadt 来源和新的输出目录，展示 codec、缓存及完整 FM
的分阶段执行方式；它不会自动生成新的 Toy 采集数据。

```bash
python tools/train_codec_ready.py \
  --pilot-dir outputs/ingolstadt_pretrain_pilot_20261002 \
  --output-dir outputs/ingolstadt_codec_new
python tools/build_pretrain_cache.py \
  --config configs/ingolstadt_custom.toml \
  --census-dir outputs/ingolstadt_pretrain_census_20261002_1148 \
  --pilot-dir outputs/ingolstadt_pretrain_pilot_20261002 \
  --output-dir outputs/ingolstadt_compact_cache_new
python tools/run_complete_pretrain.py \
  --config configs/ingolstadt_custom.toml --codec-dir outputs/ingolstadt_codec_new \
  --cache-manifest outputs/ingolstadt_compact_cache_new/cache_manifest.json \
  --codec-checkpoint outputs/ingolstadt_complete_cohort_pretrain_20261002/checkpoints/latest.pt \
  --device cuda:1 \
  --output-dir outputs/ingolstadt_per_transition_pretrain_new
```

Codec producer 只在原始状态与 16 步 C0 状态的独立 held-out 误差均通过实际阈值后
发布 ready。完整 runner 验证 gate 数值、checkpoint 与来源身份、2,920／520 分区、
完整数据缓存，并在 520 条验证轨迹上重新校验双 gate 后推进 FM；FM 更新冻结 codec。
它只进行预训练，保留在线起点 checkpoint，不自动运行在线训练或评估。
失败的 hybrid gate 会明确阻止 FM；有新鲜 heartbeat 的 augmentation producer
则被视为正在恢复，等待实际验收结果。

当前数值、配置与运行记录见 [逐窗口标签与吞吐修正记录](per_transition_pretrain_20261002.md)。
模块行为与 checkpoint 恢复由相关 pytest 回归覆盖；短程功能与有限数值不能证明交通性能提升。

### 在线想象训练的批处理实现（2026-10-04）

在线预算保持每个真实 episode 4 次世界模型更新，以及 16 个 actor–critic pass；
每个 pass 采样 8 条轨迹、每条最多推进 8 个 30 秒窗口。优化不改变奖励、
超时惩罚、优势缩放、合法候选或世界模型噪声设置。

决策时保存实际观察、context、策略概率、所选动作和事件编号。窗口结束后，
Q 按原始事件时刻批量构造各行的历史和同时动作条件；严格屏蔽未来事件，
只删除当前事件自身贡献，同车更早事件仍保留。整个 pass 内 actor、世界模型和
critics 参数固定，反事实基线在任何 critic 更新前固定，不能在拟合回报后重算。
候选 Q 在无梯度模式下计算基线，只有已执行候选重新建立 Q 训练图。
Actor 的 log-probability 和 entropy 同样在参数更新前按窗口重算训练图。
该重算依赖当前 actor/Q 使用无 dropout、无 batch normalization 的确定性 MLP；
以后修改 scorer 架构时需要重新验证采样条件与训练前向的一致性。

CUDA 训练期间，每个 pass 建立一次冻结的 CPU actor 副本，负责小批量 logits、
softmax 和 `multinomial` 采样；车辆推进不再为每次决策等待 GPU 往返。
概率、所选动作和观察在窗口结束后统一传到 CUDA。原 actor 的参数与 Adam
状态、世界模型及批量反向更新继续留在 CUDA。动作仍服从相同的合法 categorical
分布，但采样改用 CPU RNG，旧 seed 的后续具体轨迹会发生分叉；此外还存在
CPU/GPU FP32 舍入差异。checkpoint 原本就同时保存 CPU/CUDA RNG，恢复测试
确认新采样流可以准确恢复，无需修改模型参数结构或 checkpoint schema。
固定输入的概率与梯度对照、各阶段采样实验见基准产物。

物理推进对本秒不会跨道路／连接边界的车辆批量执行原本的单秒减法，
在每次策略回调前同步字典状态；跨边界车辆保留标量推进。
确定性路线缓存绑定静态 `NextHopCandidates` 实例，上限 8,192 项，
首跳与后缀规则分别缓存。窗口交通观察复用已有不可变 `PreparedActorWindow`。
想象专用 replay 采样只复制起始车辆快照，轨迹拥有该副本，不复制未使用的未来状态。

性能证据位于 `outputs/online_profile_20261004/`。使用
`tools/profile_online_imagination.py --snapshot-dir ... --output-dir ... --no-profile --passes 16`
测量包含实际 actor/V/Q 优化器更新的完整 16-pass 阶段；模型加载、数据身份校验
不计入该阶段，CUDA 在计时边界同步。`source*` 目录保留各版代码快照，
`comparison.json` 区分组件 profiling、完整阶段实测与 SUMO 采集基准。

当时采用 TraCI／SUMO 1.25 的正常在线验收为 episode 33（第 34 轮）：同轮 SUMO 采集 **156.507 秒**，
完整想象与 actor–critic 阶段 **140.917 秒**，处理 **469,968 个事件**，
actor／V／Q 各执行 **16 次更新**；另有 4 次世界模型更新合计 **1.637 秒**。
学习阶段占采集时间的 **90.04%**，满足本轮“学习短于 SUMO 采集”的目标。
从真实 episode 开始到 checkpoint 保存约 **329 秒**，已继续 2,000 轮任务。
证据为 `outputs/online_profile_20261004/live_acceptance_v7.json`；
最终完整回归 **786 passed, 2 skipped**，编译与 diff 检查通过。
此前未达标的冷启动和 V6 同轮记录均保留；时长受交通状态及共享硬件负载影响，
本次验收不保证每一轮的固定耗时，也不证明交通策略性能改善。

### 与 baseline 共用 SUMO 采样模块

`learning/sumo_sampling.py` 的 `SumoEpisodeSampler` 现在同时服务
`tools/train_toy_actor_critic.py`、`learning/baselines/common.py` 和主线
`MainlineTrainer`。模块统一 session 生命周期、窗口推进、截止时间与超时、
窗口交通观察缓存和策略回调；需求生成、各算法的学习行、replay 和指标仍由调用方负责。
baseline 显式选择 `context_mode="window"`，主线选择 `"observation"`，
保留各自原有 context 时间口径。零人口与终止窗口也保留各调用方的原行为。

共享模块在相同 libsumo 1.27.1、冻结 actor 和需求下，与抽取前实现的 11 个物理数组、
状态/事件 JSON、18 个非耗时指标及旅行时间全部精确一致；完整 3,810 秒采集
实测 **46.791 秒**。见 `outputs/online_profile_round2_20261004/shared_sampler_libsumo/parity.json`。

用户明确选择与 baseline 相同的 libsumo 1.27.1 后，主线在已完成 **50 轮**的边界
迁移到 `outputs/ingolstadt_online_libsumo_20261004`。这是显式引擎迁移，
不宣称与旧 SUMO 1.25 轨迹等价。旧目录与原 checkpoint 保留；新配置重算语义指纹，
迁移逐项验证模型、优化器、RNG、replay 和计数不变，仍累计训练至 2,000 轮。
旧引擎评估归档，新引擎从迁移时 actor 重建三组基线，禁止跨引擎混比。

继续该运行使用：

```bash
python tools/run_online_pilot.py \
  --source-dir outputs/ingolstadt_bf16_pretrain_20261002 \
  --output-dir outputs/ingolstadt_online_libsumo_20261004 \
  --run-config outputs/ingolstadt_online_libsumo_20261004/run_config.toml \
  --device cuda:1 --threads 4 --gpu-memory-fraction .06 \
  --online-episodes 2000 --evaluation-every 20 --extend-run
```

首轮新引擎在线采集（episode 50）实测 **33.242 秒**，1,012 辆车，完整 3,810 秒，
14,247 个动作事件；16 个完整 actor–critic pass 合计 **114.993 秒**，
249,149 个想象事件，actor／V／Q 各更新 16 次。4 次世界模型更新共 **1.943 秒**，
从采集开始到保存 checkpoint 的整轮时间约 **166 秒（2 分 46 秒）**，累计完成 51 轮后继续运行。
证据为 `outputs/online_profile_round2_20261004/shared_sampler_live_validation.json`。
采集提速后，学习阶段约为采集的 **3.46 倍**，旧的相对时间目标在新后端下未满足；
应分别报告两阶段绝对时间及整个训练周期，不能沿用旧后端下的耗时比例。
本次完整回归 **819 passed, 2 skipped**，编译与 diff 检查通过。

同时保留了两项不改格式或数学定义的小优化：checkpoint 标量规范化与事件字典序列化
在冻结数据上合计准备时间 **10.393 → 7.114 秒**；大窗口 Q 身份定位使用二分查找，
避免额外的大型匹配矩阵，重复 ID 仍按原严格匹配处理。这些组件结果不代表整轮同比例加速。

### 并行想象执行

`learning/parallel_imagination.py` 提供持久 CPU 进程池和集中 GPU 学习。
启用方式为 `URB_IMAGINATION_WORKERS=8`；0 保留串行对照路径。所选 worker 数写入
checkpoint metrics，恢复时未显式设置环境变量则沿用已保存数量。训练器退出或
worker 失败时关闭进程池，失败的 pass 不允许用缺失 rollout 的结果更新模型。

worker 每次推进一个 30 秒窗口，使用冻结 CPU actor、事件编码器和 Q 计算
决策条件及更新前的反事实基线；不持有 CUDA 或反向图。父进程批量执行 8 条
rollout 的世界模型预测，在全部轨迹采样结束后统一重建 actor/V/已执行 Q 的
梯度图并更新。16 pass、8 rollout、最多 8 窗口及奖励规则不变；不同长度 rollout
的 V loss 仍按原规则等权平均，部分尾窗口不虚构 30 秒转移或端点风险。

每条 rollout 的策略和世界模型噪声子流由已保存的父 RNG 明确派生，不依赖 worker
调度顺序。与历史串行随机轨迹不保证逐位一致；CPU oracle 验证了 1/2 worker
一致性、串行目标与梯度、截断/超时/费用及失败清理。真实 CUDA 批计算允许正常
浮点舍入误差，不宣称所有 seed 下物理轨迹逐位相同。

第 153 轮冻结 checkpoint 的实测基准：串行完整 16 pass **90.381 秒**；
8 worker 首次含启动 **49.950 秒**，紧接的完整热运行 **32.904 秒**。
热运行有 128 条 rollout、994 个实际窗口、215,792 个事件，actor/V/Q 各更新 16 次。
工作者启动及静态缓存是一次性开销，不能把热运行数字用于首次冷启动。
证据见 `outputs/imagination_40s_20261004/optimized_v1_summary.json`。

实际在线任务从完成 160 轮的 checkpoint 接入 8-worker 执行：首次完整阶段
**49.061 秒**（含冷启动），之后两轮分别 **37.133 / 42.057 秒**，
平均 **39.595 秒**。三轮 actor/V/Q 均各更新 16 次，数值有限且参数发生变化。
两轮热运行从开始采集到 checkpoint 完成分别约 **76 / 97 秒**；不要将 runner
保存前的 `seconds` 字段误报为含序列化的总时长。checkpoint 已确认保存 worker 数 8，
任务继续累计至 2,000 轮。新执行路径的完整回归为 **875 passed, 2 skipped**。
实际证据见 `outputs/imagination_40s_20261004/live_parallel_validation.json`。

实施前的工作区已提交为 `0041fae`，可作为本次 event 集成的对照基线。
当前生命周期验收见 [离线预训练验证记录](offline_pretrain_validation.md)；先前 warmup
生命周期的数字保留在 [历史验证记录](event_mainline_validation.md)。城市梯度和来源
身份见 [城市验证 JSON](event_city_validation.json)。
