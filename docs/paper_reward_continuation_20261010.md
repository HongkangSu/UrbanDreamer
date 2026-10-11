# 累计平均时间奖励统一与 baseline 继跑

2026-10-10，按用户指令将 UrbanDreamer 和各 baseline 的默认共享物理奖励统一为

\[
r^g[a,b]=-{1\over60}\left[{C(b)\over B(b)\vee1}-{C(a)\over B(a)\vee1}\right].
\]

`C` 是累计实际 active 车辆·秒；`B` 是累计实际驶入路网的车辆数，实际出发时间必须
严格早于物理截止时刻。未驶入和恰在截止时刻开始的车辆不计入目标分母。已开始但
未完成的行程只计截止前累计时间；没有估计剩余时间或默认附加终端惩罚。空需求为零。
现有 runners 的有效截止时间仍按 30 秒窗口取整，例如 3800 秒配置对应 3810 秒。
物理观测采用已有 SUMO 步长；此次运行使用兼容该时间网格的截止时刻。

## 实现与默认设置

- `urban_dreamer/learning/global_reward.py` 是共用奖励与累计统计实现，契约为
  `causal_cumulative_mean_global_v1`。真实 SUMO sessions 在每个物理步更新累计量，
  sampler 使用同一记账对象，避免重复累计。分母增长的修正可合法产生正区间奖励。
- Actor 候选输入仍为十维，将人口特征改为当前 active/B。Critic context 由二维扩展
  为四维：`t/T, active/B, log1p(C/60), log1p(B)`，没有最终需求人数这一输入。
- 默认 `--reward-objective observed_mean --global-reward-only`；个体 timeout shaping 为零，
  `selection_score` 等于共享回报。旧 exposure 语义只作为显式 `legacy_exposure` 保留。
- 默认模型分歧风险权重为零。显式启用时，它作为额外算法风险项单独记录，不冒充
  物理共享奖励。gamma=1 和原有 entropy 退火进度保持。
- 主线 replay schema 8 保存起止累计 C/B 及准确奖励；事件回报使用事件时刻与末端的
  累计平均时间差。真实训练和想象的奖励记账共享口径。
- Dreamer、MBPO、MAMBA 的学习型奖励模型改用准确的带符号 paper reward 标签；其
  想象回报仍是模型近似。已有世界模型的交通量/控制归一化单位保留，以兼容正在运行
  的预训练和已学到的动力学，未在本次切换中重建整套离线交通标签。
- 新配置语义指纹明确标记 paper reward。旧 source 可用历史指纹检查，但旧 online
  checkpoint 不能无标记地作为新奖励状态直接继续。

## baseline 迁移和运行

新 controller 位于 `outputs/city_baselines_paper_reward_20261010`，使用各城市原来的
冻结源码为基础，保留原路网和需求文件，叠加审核后的奖励实现。原训练输出目录保留。

所有 22 个未完成任务的命令均已改为 paper reward。18 个已有 checkpoint 完成显式
迁移；4 个尚无 checkpoint 的任务将在资源可用时以新默认设置启动。原来运行的九个
任务优先恢复，另外十三个继续等待 GPU。五个已完成任务/参考结果保持历史状态。

迁移保留原有 Actor、世界模型和其他不需扩展的权重；必要的 Critic/奖励模型输入
增加累计统计量，新维度置零，Adam moments 同位置扩展。优化器原内容、随机数状态、
已完成回合数和原 5000 集预算保留。旧奖励的 replay/synthetic cache 清理后重新积累；
旧 best 文件归档，新目标 best 比较阈值重置，不能跨奖励版本直接比较旧分数。

原 checkpoint、best、协议与日志备份保存在新 controller 的 `handoff_original`，
具体迁移和保留状态 hash 在 `handoff_protocol.json`；每个原输出目录写入
`reward_objective_changes.jsonl` 记录切换回合。

独立核验于 09:11 UTC 完成的运行快照：

|任务|迁移边界|已提交新奖励回合|
|---|---:|---:|
|Ingolstadt MAPPO|2561|2580|
|Ingolstadt IQL|2653|2694|
|Ingolstadt MAMBA|718|723|
|Coulommiers AC|551|577|
|Coulommiers IPPO|937|954|
|Coulommiers MAPPO|477|486|
|Coulommiers IQL|315|323|
|Montereau AC|600|618|
|Montereau IPPO|883|902|

九个任务均为拥有正确 PID 身份的新源码进程，回合和 seed 连续。新提交回合满足
`global_reward_sum = -C(T)/(60*max(B(T),1))`、个体惩罚零、selection score 等于共享
回报，最大恒等式误差为 `7.11e-15`。共享奖励正确不代表切换后性能立即提高；新观测
统计和目标需要适应，清理的 replay 也需要重新积累。

## UrbanDreamer 预训练与后续 online

预训练 PID `3839431`、原启动身份、源码、配置与输入保持。新的 paper online waiter
PID `4025475` 正常等待该预训练完成，计划为
`outputs/ingolstadt_full_retrain_20261009/paper_reward_future_plan.json`，执行源码为同目录
`source_paper_reward_20261010`。

完成预训练后，queue 将执行明确的 pretrain-to-online 奖励迁移：保留 Actor/世界模型、
原训练计数、RNG、离线数据与 codec 验证；latent critic 输入 66→68，event critic 输入
413→415，后者在第 147 列插入新 context 列，而不是简单在末端追加。原预训练文件
不被改写，online 仍按原计划执行 2000 集，并采用风险权重零的 paper reward 配置。

## 验证证据与边界

- 主线相关 141 项、baseline 有界回归 98 项、指纹/归档边界 40 项、队列和迁移相关
  64 项、root 最终配置/sampler/队列相关 170 项检查分别通过；其中有重叠，不相加
  作为独立测试数。真实 SUMO 相关 30 项和 baseline integration 三项分别通过。
- 额外真实 30 秒 cutoff 核验：362 辆预生成需求，仅 17 辆实际开始，C=207 车辆·秒，
  observed mean=12.17647 秒，共享回报=-0.202941，额外惩罚零。
- 独立核验重读全部十八个原始备份 SHA，重新计算保留学习状态证明；十七个仍存在
  的初始迁移文件进一步核对了 28 个扩展参数和 56 个 Adam moments 的原值及新增零值。
  Montereau IPPO 的初始迁移 latest/best 已在新目标第 900 集评估后正常更新；它的
  原始备份和保留状态证明仍已核验。
- 源码包第一次模板检查发现缺少一个已有迁移辅助脚本，在任何迁移 checkpoint
  发布前停止。补齐依赖后从同一持久化步骤恢复；修复记录保存在 `bundle_repairs`。
- Python 编译检查、`git diff --check` 通过。无新增第三方依赖。

主要证据：
`outputs/city_baselines_paper_reward_20261010/independent_verification.json`、
`continuation_verification.json`、`handoff_protocol.json`，以及主线
`paper_reward_handoff.json` 与实时 `status.json`。
