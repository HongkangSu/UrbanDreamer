# 城市采样速度与需求发布调整（2026-10-08）

**后续指令更新：用户要求发布时间与 Ingo 一致，当前 Coulommiers/Montereau 已恢复 1800 秒发布、60 个 nominal 窗口、3810 秒训练截止，CSV 恢复原时间戳。下文 3600 秒部分仅为已完成的历史对照，不是当前配置。最新积压诊断见 `outputs/inflight_reduction_20261008/`。**

用户要求先停止所有正在运行的 baseline、关闭最短路 prior，再解决候选城市明显慢于 Ingo 的问题；随后要求拉长 Coulommiers 和 Montereau 的需求发布时间。

## 当前实验状态

原 20261007 批次的 22 个训练进程及全部调度/归档控制器已停止，30 个训练任务标记 stopped，3 个最短路参考已完成。检查点、原始 prior 协议和冻结源码保留。当前代码拒绝自动恢复用户已停止的批次。新队列默认 prior 权重为 0，继续使用带个体超时惩罚的配置；本次没有重新开启完整训练。

## 原先慢在哪里

停止前最近 20 轮 AC 的均值如下；不同城市策略处于不同训练阶段，这不是算法公平排名。

| 网络 | 注册路段 | 期望车辆 | 每秒平均在途车辆 | 采样（含存档） | 参数更新 |
|---|---:|---:|---:|---:|---:|
| Ingo | 853 | 1035 | 126.4 | 31.6 秒 | 1.16 秒 |
| Coulommiers | 4483 | 2000 | 1247.9 | 412.2 秒 | 1.99 秒 |
| Montereau | 4302 | 3000 | 1703.0 | 520.7 秒 | 2.12 秒 |

平均在途车辆按 active_vehicle_seconds / microsteps 计算。候选网络约为 Ingo 的 5 倍路段数；低完成率使车辆持续累积，将逐秒观察工作量放大约 10 和 13.5 倍。原先十几倍的耗时差距主要在采样，不在梯度更新。

## 保持轨迹不变的优化

- meso + libsumo 每秒直接读取 road ID、插值位置和 route index，避免订阅包装和整批结果字典；微观及 socket TraCI 保留订阅实现。
- 每次出发及 setRoute 后保存 SUMO 实际接受的完整路线，后续观察只读实时 route index；保留原有路段/下一跳/目的地校验。
- 跨轮复用同一网络的 destination tables；缓存有上限的不可变自由流路线后缀，保留原 tie breaking，禁止重访路径继续使用原专用计算。
- 批量转换路段 flow，保留 float64 中间计算与 float32 输出。
- 减少丢弃的全车状态复制，避免归档原生标量的昂贵类型检查，移除 JSON 独立复制之前的重复 deepcopy。

每个城市以相同初始 actor、种子 43 的相同需求、prior 0、CPU 单线程跑两轮（第二轮包含第一轮的真实参数更新），完整存档；使用冻结的原 1800 秒发布期和 3810 秒截止时间。未开启 cProfile 的计时结果：

| 网络 | 原采样均值 | 优化后均值 | 耗时减少 |
|---|---:|---:|---:|
| Ingo | 82.1 秒 | 40.2 秒 | 51.0% |
| Coulommiers | 365.6 秒 | 251.9 秒 | 31.1% |
| Montereau | 455.4 秒 | 325.2 秒 | 28.6% |

六组逐轮动作/特征摘要、全部状态/决策/flow/action/departure/exposure/termination 档案数组及学习标量精确一致。summary_json 包含性能计时和配置出处，未要求其字节相同。两轮是有上限的点测量，不能当作所有方法的稳定吞吐量或收敛结论。

剩余大头是拥堵情况下的 SUMO 内部推进。需注意 [SUMO 1.27.1 Simulation::step](https://github.com/eclipse-sumo/sumo/blob/v1_27_1/src/libsumo/Simulation.cpp#L159) 同时包含物理步进与订阅刷新，原 cProfile 的 simulation_step 桶不能全部归于物理仿真。没有降低每秒观察频率或关闭信号灯/中观物理控制。[MSEdge::getMesoPositions](https://github.com/eclipse-sumo/sumo/blob/v1_27_1/src/microsim/MSEdge.cpp#L1653) 使用上一时刻的位置保证不后退，因此跳过中间插值读取可能改变后续轨迹；本次仍每秒读取每辆在途车的位置。

原始证据：`outputs/city_speed_diagnosis_20261008/`，含 before_profile、before_timing、production_revision、production_timing、comparison.json 和图。

## 新需求发布设置

Coulommiers 仍为期望 2000 辆，Montereau 仍为期望 3000 辆。两者 online/offline TOML 的发布期从 1800 延长到 3600 秒，OD 分配不变，CSV 时间戳同步乘 2。实际训练需求由配置中的 PoissonDemand 控制；同一随机种子下，数量、车辆 ID 和 OD 不变，连续出发时间精确乘 2。Ingo 保持原配置。

nominal 宏窗口改为 120，训练完成缓冲仍为 2000 秒，训练 deadline 向上取整为 5610 秒；配置的 evaluation horizon 为 5600 秒。平均发布率为 Coulommiers 2000 辆/小时、Montereau 3000 辆/小时，均为原来的一半。无事故、prior 0 保持。

固定种子 10042 的最短路短测试（单种子，不是多需求收敛评估）：

| 网络 | 相同实际车辆数 | 原 1800 秒发布完成率 | 新 3600 秒发布完成率 |
|---|---:|---:|---:|
| Coulommiers | 2032 | 72.05% | 87.06% |
| Montereau | 3030 | 40.56% | 45.02% |

Coulommiers 有明显改善；Montereau 仍存在较严重的未完成问题，不能说拉长发布期已消除其持续拥堵。新旧窗口及 deadline 不同，不能把时长或仅已到达车辆的均值当作同物理时段的等价比较。

修改前配置及 CSV 保存在 `outputs/city_demand_release_3600s_20261008/original_inputs/`；manifest 和 shortest_smoke 保存新发布设置及验证结果。准备器默认生成 3600 秒发布期并同步 horizon，使用新的时间戳 manifest，防止覆盖原准备记录。

## 验证

最终选定的采样、真实 SUMO、归档、baseline、prior、队列和需求配置测试共 300 项通过；修改的 Python 文件编译检查通过。真实小网逐秒对照覆盖普通及禁止重访路由，直接读取和订阅读取的全部逐秒状态与宏窗口结果相同。原批次 137 个冻结源码/配置文件哈希不变。完整训练、调度及诊断进程均已结束或停止。
