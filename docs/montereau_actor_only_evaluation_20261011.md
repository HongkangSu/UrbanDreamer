# Montereau AC：确定性评估移除最短路先验

用户明确要求确定性执行不带 prior。此前 `deterministic` 仅将采样改为
argmax，仍对 `actor_logits - dynamic_cost_excess / 30` 取最大值。
因此此前约97.72%的完成率属于组合策略，不能解释为 actor 独立执行的能力。

## 正确口径与代码

`tools/train_dynamic_prior_ac.py` 的默认 `deterministic` 改为对 masked
神经网络输出直接取 argmax，静态/动态加性 prior 均不参与；该分支不查询
动态最短路距离。静态距离观察特征、合法动作 mask、1秒事件决策和30秒
路况快照不变。`deterministic_with_prior` 显式保留旧行为供历史诊断。

训练仍使用 stochastic + dynamic prior，保持用户原先的500轮训练预算。
初始化的纯先验一致性验证单独保留；initial best、周期验证、最终/最佳
holdout 都改为 actor-only。日志/存档增加 `evaluation_policy`、
`action_policy` 和 `routing_prior_applied`；checkpoint 协议明确记录新口径。

## 已完成的同种子实测

冻结启动评估时最新的第179轮checkpoint，严格加载原协议后，在独立进程中
显式将评估 dynamic prior weight置零（static prior本来为零），三个种子
并行运行。测试证明该实现与新代码直接调用原始神经网络输出逐元素完全一致。
不改动训练进程的权重、随机流或优化器。

| 策略 | 三种子平均完成率 | 完成车辆/总车辆 | 三种子平均ATT_T | 从未出发 |
|---|---:|---:|---:|---:|
| 纯动态最短路，30秒刷新 | 97.7231% | 4378/4482 | 3.19114分钟 | 0 |
| AC179，无prior argmax | 16.6804% | 747/4482 | 40.61674分钟 | 273 |

种子10042/10043/10044的无prior完成率分别为
17.9522%、15.8927%、16.1962%。ATT_T仍按既有实际出发人口及截止时刻截断
费用口径计算，不是只对完成车辆取均值；未出发车辆另列，不隐藏。

当前训练目标是“固定prior + 学习修正”的组合策略。移除prior后直接使用
修正网络，不保证获得独立路由策略。这次测量证明当前checkpoint对prior
依赖很强，不能再用带prior的确定性成绩声称actor已独立学会最短时间路由。
没有据此擅自改变训练目标或启动新的蒸馏/退火实验。

## 历史与迁移

新冻结源码为实验目录下 `source_actor_only_eval_20261011`，相对前一加速版本
仅driver发生变化。严格校验协议差异只包括新评估定义及driver源码哈希。
在下一已提交checkpoint边界恢复训练。旧best依据带prior成绩选出，不能
继承为新口径下的best；从已评估的179轮checkpoint重新开始actor-only选择。
旧best、latest、initial及旧评估汇总备份到 `before_actor_only_evaluation/`。
历史验证序列保存在checkpoint `legacy_with_prior_selection` 中。

后续验证存于 `run/evaluation/actor_only/episode_*`；旧
`run/evaluation/episode_*` 为带prior历史记录。旧汇总保留在
`validation_assessment_with_prior.json`，新的 `validation_assessment.json`
只汇总 actor-only。原NPZ不改写；独立评估的原始NPZ通过
`metadata.dynamic_prior_protocol.evaluation_override` 明确记录置零操作。

原始证据：`outputs/montereau_actor_only_eval_20261011/`，含checkpoint、
三个SUMO存档、results.json、测试和迁移记录。训练状态以deployment.json
和实验launch.json为准。历史checkpoint未逐个补做无prior评估，所以
重新建立的best不是对全部过去训练轮次的回溯最优。

## 验证与运行状态

本轮主进程复核的动态AC、prior、因果路由、精确距离及观察时钟测试
**71项通过（3.48秒）**；实现子任务包含额外相关路由测试共79项通过。
编译与 `git diff --check` 通过。无prior输出与旧实现prior权重置零的
logits逐元素一致；训练随机采样动作/行为logprob保持一致。

第183轮提交后已迁移，正在第184/500轮。迁移前后latest的模型、优化器、
RNG和episode编号递归精确比较通过；best的actor确认为已实测的179轮。
只更改评估协议和相应选择历史，不更改训练参数。
