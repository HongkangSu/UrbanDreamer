# 当前候选网络补丁与单轮计时

按用户要求，正式Coulommiers和Montereau已完成如下修改：

- 外部车道长度不超过10米时，采用固定种子42、按路段ID确定的10～15米随机长度，同一条道路的合格车道共用长度。
- 严格删除乘用车可用的相互掉头道路对的两个方向连接：Coulommiers369对，Montereau332对；不保留必要掉头例外。netconvert重建内部连接、路口请求和信号连接索引，保留外部道路/车道属性，再应用长度修正。
- 所有其余外部车道连接与原网络一致，保留连接的交通灯指示、相位时长和偏移经逐连接验证一致。
- 删除由此不可达的全部OD车辆记录，不重新分配需求。Coulommiers删除53个OD/189行，保留489个OD/1811行；Montereau删除49个OD/194行，保留712个OD/2806行。剩余车辆ID、OD索引及出发时间原样保留，全部需求OD可达。
- Montereau的8对铁路bidi绑定短路段原先分别取随机值，触发不等长warning。现按bidi配对的规范ID共用随机值，并修复当前8条对应车道。全部65对bidi绑定长度一致；Coulommiers没有bidi绑定。
- 发布时间1800秒、训练截止3810秒、无事故、prior0、带个体超时惩罚继续保留。Ingo未修改。完整训练仍停止。

正式文件在networks/coulommiers和networks/montereau_fault_yonne内，格式仍为net.xml、edge_registry.json、agents.csv、od_*.txt。旧文件完整备份，并将原来链接到当前输入的历史源码快照固定到补丁前输入。

## AC单轮计时

最新冻结源码和路网、CPU单线程、初始化策略、需求种子43，各跑一轮真实采样和actor/critic更新：

| 网络 | 本轮实际车辆 | 采样含存档 | 参数更新 | 训练部分合计 | 首次初始化 |
|---|---:|---:|---:|---:|---:|
| Coulommiers | 1811 | 109.46秒 | 4.13秒 | 113.59秒 | 10.34秒 |
| Montereau | 2827 | 180.12秒 | 4.22秒 | 184.34秒 | 9.60秒 |

训练部分合计不含首次初始化、定期评估和检查点序列化。泊松需求使实际车辆数围绕1811/2806的期望值变化。单轮初始策略完成率为9.72%/3.82%，这只是耗时测量，不代表已训练策略收敛。运行未再出现bidi不等长warning，仍有中观忽略微观internal links的正常提示。

验证：剩余OD全部可达，短路段下限分别10.001/10.004米，相互乘用车掉头对为0；每网3辆原生SUMO加载/通行检查全部到达、无teleport；补丁/默认配置/队列相关68项测试通过。

产物：outputs/candidate_random_lengths_no_mutual_uturn_20261008/manifest.json、signal_verification.json、sumo_smoke.json；outputs/candidate_bidi_length_fix_20261008/manifest.json、episode_timing_summary.json及ac_one_episode/。完整初始源码、网络与OD备份均保留。
