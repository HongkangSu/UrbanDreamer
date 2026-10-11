# Montereau AC：衰减先验与无先验的500轮对照

用户要求比较先验对训练效率的影响，并明确采样先验应随episode降到零。
旧固定先验实验已在第199轮checkpoint完整提交后停止，保留全部记录。
它不能重标为“衰减先验”数据。新实验两组均从同一个第0轮状态开始。

## 对照设置

| 项目 | 衰减先验组 | 无先验组 |
|---|---|---|
| 训练轮数 | 500 | 500 |
| 初始先验系数 | 1 | 0 |
| 系数调度 | 第1轮为1，线性降至第250轮为0 | 始终为0 |
| 第250–500轮 | 无先验采样 | 无先验采样 |
| 确定性评估 | 无先验actor argmax | 无先验actor argmax |

公式为 `initial_weight * max(0, 1 - max(0, episode - 1) / 249)`。
系数在一轮的采集及actor更新期间固定；每条训练行记录实际采样系数，
更新前校验不变。checkpoint协议保存调度与端点，恢复时重建对应系数。
不使用的动态最短路计算和道路费用观测跳过，避免向无先验组加入无用开销。

两组共享冻结源码、初始化actor/critic、Adam状态、Python/NumPy/Torch RNG。
原第0轮checkpoint逐张量与新初始化核对一致。协议除初始prior系数和
初始化参考策略标记外全部一致。不是从已经学习过的第199轮继续做对照。

城市Montereau，OD-stagger，训练需求种子43..542；每50轮使用
10042/10043/10044做actor-only验证。完成训练后使用20042..20046做独立
holdout，训练中不用于选择。meso/libsumo、1秒物理观察、30秒路况快照、
3810秒截止、奖励/网络/优化器/熵预算保持相同。每组CPU单线程，并行运行。

共享初始actor-only评估三种子的完成率为2.0478%、2.8777%、2.6882%。
这是共同零初始化输出头的未训练argmax表现，不能用于评价学习效果。
初始参考使用已核验的相同需求纯动态最短路记录，并检查输入哈希和需求一致。

## 如何比较效率

- 样本效率：相同训练轮数时，三个固定种子的无prior完成率、ATT_T和未出发数量。
- 计算效率：同一组验证表现对累计采集＋学习墙钟耗时；不把验证及checkpoint I/O
  混入该横轴。完整实验端到端耗时可另由启动/完成记录检查。
- 训练诊断：随机采样完成率、prior系数、事件数、每轮耗时、actor/critic loss、
  行为logprob误差。随机采样完成率不是无prior部署策略的最终效果。
- 两组是一个配对初始化，不能单凭这次实验声称跨初始化统计显著性。

## 文件和监督

根目录 `outputs/montereau_ac_prior_ablation_500_20261011/`：

- `decaying_prior/run/` 和 `no_prior/run/`：独立checkpoint、metrics、训练存档和评估。
- `initial_evaluation/`：共同初始状态及三个真实SUMO验证存档。
- `plan.json`、`paired_protocol_verification.json`、各组`initialization_verification.json`：对照协议证据。
- `comparison.json`、`comparison.csv`、`comparison.png`：自动更新的对照记录和图。
- `supervise.py`：仅管理这两组进程，失败时最多三次尝试从checkpoint恢复；
  `supervisor_pid.json`及各组`launch.json`记录PID、启动时间和命令。
- `fixed_prior_stop.json`：旧固定先验实验的停止轮次和原因。

## 修改与验证

`tools/train_dynamic_prior_ac.py` 增加linear/constant采样调度及归零端点，
无prior跳过费用查询，checkpoint恢复调度，行为系数一致性检查；
`tests/test_dynamic_prior_ac.py` 增加对应回归测试。
相关动态AC、prior、费用观察、动态最短路测试95项通过。
没有新增依赖、改动其他baseline进程或改动物理设置。

## 启动后的真实训练验证

两组均完成第1轮真实采集、actor/critic更新及checkpoint保存，继续第2轮。
需求种子43、人口1540、需求内容哈希完全一致。

| 第1轮诊断 | 衰减先验（此轮系数1） | 无先验（系数0） |
|---|---:|---:|
| 随机采样完成率 | 57.3377% | 16.7532% |
| 采集耗时 | 70.47秒 | 97.40秒 |
| 学习耗时 | 3.78秒 | 4.19秒 |
| 动作事件数 | 55,235 | 67,593 |
| 行为logprob最大误差 | 0 | 0 |

双方actor梯度非零，loss有限。第一轮衰减组与旧固定prior第一轮（均系数1）
全部状态、动作、flow、费用、终止/截断及旅行时间精确相同，证明保留原行为。
首轮差异反映先验对采样路径的直接引导，不能据此认定样本效率或最终收敛更好。
后者必须看后续无先验argmax固定种子评估。
