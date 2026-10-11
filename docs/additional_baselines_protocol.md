# 中观新增 baseline 的实验协议

IPPO、MAPPO、IQL、MAMBA 使用 `urban_dreamer.learning.baselines.common.EpisodeRunner`。2026-10-07 新批次按用户要求统一启用 `global_with_individual_timeout`；纯全局模式仍可显式选择，检查点禁止跨奖励模式直接恢复。完整训练预算为每个任务 5000 回合，实际进度以批次队列状态为准。

## 共同采集和奖励设置

- 城市继续使用既有中观配置；配置、源码和许可证随新批次冻结。
- 独立 OD 泊松需求：OD 生成率跨回合固定，每回合采样数量和出发时间；种子为 42，训练回合使用 `42 + episode`，不重采样零人口回合。
- 使用真实本回合 N；分母为 `max(N, 1)`。中观、libsumo、1 秒观察、30 秒宏窗口；只在真实下一跳分支事件调用策略，同车窗口内多次动作全部保存。
- 不屏蔽历史访问路段。所有真实执行仍使用合法、目的地可达的 next-hop mask。
- **本批次带个体超时惩罚**：各方法的全局窗口奖励仍为 `-active_vehicle_seconds / (N * 60)`。真实截止时，每辆未到达车在自身决策链上承担一次 `-5000`；策略梯度方法先缩放全局 advantage，再加入个体 timeout return 并裁剪。IQL 在该车辆最后一条 Bellman transition 加入一次 terminal cost。MAMBA 仅在想象真正到达实际 deadline 时按车辆身份加入惩罚，短 horizon 截断不处罚。全局 critic／世界模型标签保持团队在途费用。评估 `selection_score` 加入未到达惩罚。
- 所有批次命令明确传入 `--no-global-reward-only`。显式使用 `--global-reward-only` 时，超时罚项不进入学习或排名，保留纯全局模式行为。
- 训练按 Categorical 采样，评估 argmax；IQL 使用 Boltzmann Q logits，放弃旧脚本的 epsilon-greedy，以对齐采样方式。
- **2026-10-08 起关闭 shortest-path prior**：所有随附主线配置、学习型 baseline CLI 和新队列批次均使用权重 0；采集、更新和评估都不增加最短路 logits 偏好。显式非零 prior 的可选功能仍保留。历史 20261007 冻结批次曾使用权重 1、温度 30 秒、保持 200 回合、余弦衰减至第 1000 回合归零；该批次现已按用户要求停止，历史记录保持原语义，不能按关闭 prior 的新协议直接恢复。
- 每 100 回合在固定需求种子 10042/10043/10044 上作确定性评估，best 按带个体超时惩罚的 `selection_score` 选取。只有显式开启非零 prior 时才分别评估 scheduled prior 和 zero prior，并按 zero prior 结果选取 best。最终 holdout 种子为 20042/20043/20044。
- 每回合保存恢复检查点及完整真实 NPZ；episode 不含想象转移。恢复会拒绝 reward mode、prior、revisit 或物理身份改变。

已停止的 20261006_160058 批次使用历史 `global_with_individual_timeout`：共享 critic 用全局奖励，actor 和 checkpoint selection 有个体未到达罚项。20261006_174632 批次使用纯全局奖励。两批旧结果不能宣称同目标比较；不得改标签。现在所有方法均支持显式奖励选项，新增方法的惩罚模式具有独立恢复协议。奖励协议改变会拒绝直接恢复旧任务。

## 算法实现及明确适配

| 方法 | 学习实现 | 信用和模型边界 |
|---|---|---|
| IPPO | 共享 next-hop actor、局部候选集 critic；最多4轮 clipped PPO，clip0.2，batch512 | 个体决策链使用全局奖励；最终动作保留团队尾部；centered advantage、整批KL早停、critic最佳拟合保护 |
| MAPPO | 共享 next-hop actor、集中全网动态 flow critic；最多4轮 clipped PPO | frozen behavior log-prob和完整合法分布；宏窗口 lambda return；critic默认最多8轮，IPPO最多4轮 |
| IQL | 共享 candidate Q、200000条replay；默认每新增transition预算一次优化抽样 | 小数预算跨回合累计；target tau=.01每512条抽样，尾batch按比例同步；raw-Q target、prior仅行为、完整团队尾部 |
| MAMBA | 离散车辆RSSM/agent attention + 逐路段全局RSSM + 物理想象 | local按真实决策事件推进，global按宏窗口推进；动作显式条件来自固定起点active车辆群；physical arrival，不注入未来demand |

所有车辆网络共享参数，以适应随机实际人口及每回合变化的 ID；旧版固定角色的独立网络没有直接续用。

IPPO/IQL 根据精确物理窗口边界为异步事件积分全局奖励。档案只提供 30 秒窗口的总 exposure，因此窗口内均匀 reward rate 是明确的时序近似，而非每秒精确 reward trace。

MAMBA 指 [Scalable Multi-Agent Model-Based Reinforcement Learning](https://arxiv.org/abs/2205.15023) 的算法，不是 selective-SSM 网络。原模块和MIT许可证保存在 `mamba_core/`；本实现是动态路由适配，不是原benchmark逐配置复现。修正后global输入保留逐路段flow、固定起点群的起终点占用、时间及N；global动作保留该群逐路段来源/所选下一跳计数。每条训练序列与想象都固定同一群身份；其他车辆/未来入流的未知作用通过条件预测统计吸收。通信仍按块，块间无通信。

车辆RSSM在物理想象的真实决策事件观察候选集；宏窗口内允许零次或多次决策。候选mask和到达由现有图/物理推进检查。种子为宏边界快照，模型更新后按一致起点重建posterior并burn-in，不再复用旧latent。局部车辆到达停止其决策；全局预测保留团队奖励尾部到想象horizon/deadline。默认至少100次含真实车辆决策的局部模型更新且100个非空cohort宏窗口后才训练想象actor；零人口回合保留但不能凑足路由warm-up。Toy验证可显式降低门槛。短模型训练、块通信和统计背景仍限制预测质量。

PPO的整批KL阈值默认0.01，每完整actor epoch后检查；它停止后续epochs，不保证已执行epoch的KL始终小于阈值。critic最佳MSE保护包含对应Adam状态，只保证当前固定target批次拟合不变差，不保证泛化。原始优势、KL、ratio分位数、clip fraction、critic bias/explained variance和实际更新数量进入日志。

IQL sample-to-insert默认1.0表示优化抽样次数预算，并不保证每条新增数据都被抽到；target时间单位是抽样transition。终端/持续transition的TD指标分开记录，零人口回合不消耗额外更新。

新PPO/IQL/MAMBA检查点保存版本和完整学习设置；旧学习协议检查点不允许静默续训。旧档案和检查点保留作历史分析/世界模型数据，不混进新同协议训练曲线。

Dreamer的真实采集仍使用共享 `NextHopActor`，默认CPU采集镜像，GPU保留learner。想象actor同样可用CPU镜像，按pass同步最新权重/prior；跨设备保持策略分布，改变随机数流，不声称逐轨迹一致。想象rollout数、horizon及世界模型更新预算不减少。新检查点记录想象推理设备；旧未标记检查点如要把想象采样移出learner设备，需要显式 `--dreamer-allow-legacy-imagination-migration`，原设备恢复可使用 `--dreamer-imagination-device learner`。本轮保留Dreamer原有起点群耗尽即结束想象的全局尾部近似。

MAPPO/IPPO 的集中/局部 critic 区别遵循 [PPO 多智能体论文](https://arxiv.org/abs/2103.01955)；本项目采用适应动态车辆人口的路由观测和纯全局团队目标，而非声称原 benchmark 的逐配置复现。

## 验证证据

新增单元测试覆盖真正的 PPO clipping、冻结行为概率、raw-Q/prior 分离、非法动作屏蔽、异步同窗口 reward、个体到达后的团队尾部、动态人口和零人口、个体 reward/timeout 投毒不影响学习、模型/actor/critic 参数更新及恢复。

四种方法在同一 Toy 中观真实需求上完成 CUDA 采集、学习、存档和检查点恢复；需求哈希和实际 N 完全一致，且 `selection_score == global_reward_sum`。记录为 `outputs/additional_baselines_alignment_20261006/meso_smoke.json`。此小规模检查验证功能与协议，不证明城市策略收敛。
