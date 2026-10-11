# Montereau 动态先验 AC 的采集加速（2026-10-11）

当前 500 轮实验已经采用 meso、libsumo、1 秒 event 观察、30 秒路况快照。
本次保持这些设置、需求、随机流、策略和优化器不变，优化 Python 采集路径。

## 定位

冻结 episode 31 checkpoint，使用随机状态恢复后 seed 43 的 OD-stagger 需求回放。
带 cProfile 的整轮采集为 176.7 秒，不能用它和普通训练计时直接比较。
关键累计时间：动态最短路 48.16 秒，逐车观测 34.96 秒，底层
`libsumo.simulation_step` 24.26 秒，自由流完整路线构建 14.24 秒。
这些函数存在调用层级；不得把所有累计时间直接相加。

## 改动

- `tools/causal_routing_policies.py`：Dijkstra 的工作距离和费用使用 Python
  float 列表，避开 Python 循环中的 NumPy 标量和 NumPy 标量堆比较；最终
  转为原有 float64 数组缓存。算法、堆顺序、代价相加顺序、缓存失效均不变。
- `urban_dreamer/traffic/event_session.py`：每次观察时绑定 meso/revisit 标志、
  vehicle API 和 edge index 字典，减少逐车循环内重复属性查找。
  仍每秒读取 road ID、插值位置、实时 route index，并执行所有原有校验。
- `tests/test_routing_distance_performance.py`：冻结旧实现作为独立参考，验证
  随机图、零代价环、并列、不可达路段、乘用车权限、缓存重置和 GA 子类。
  距离数组逐字节相同，堆弹出顺序也完全相同。

## 端到端证据

未开启 profiler、相同 checkpoint/RNG/需求的完整回放（包括存档）：

| 项目 | 原版 | 优化版 |
|---|---:|---:|
| 整轮采集 | 94.044 秒 | 74.996 秒 |
| 动态最短路计算 | 26.473 秒 | 12.707 秒 |
| 最短路实际计算次数 | 7,221 | 7,221 |
| 动作事件数 | 66,509 | 66,509 |

整轮耗时下降 20.25%，吞吐约为原来的 1.254 倍。
动作训练行摘要、全部非 summary NPZ 数组（包含状态、动作、流量、出发、费用、
完成/截断）、各车到达旅行时间和排除墙钟计时的 metrics 精确一致。
这是单个完整轨迹的配对测量；机器同时运行其他实验，不能推广为所有种子的固定加速比。
较早一次无 profiler 回放保存的采集阶段（不含最终存档）为 86.88 秒，说明墙钟存在波动。

原始证据在 `outputs/montereau_collection_speed_20261011/`：
`before_profile.txt`、`reference.json`、`optimized.json`、`comparison.json`、
`replay.py`、完整 NPZ、测试日志和部署记录。早期 `baseline.py` 已完成仿真及
归档，但打印空 profiler 统计时失败；最终计时使用修正后的 `replay.py` 两组结果。

## 运行迁移

旧实验冻结源码保留。新源码为
`outputs/montereau_dynamic_prior_ac_500_20261011/source_collection_fast_20261011`，
与原冻结副本仅上述两个实现文件不同。
迁移必须在每轮 latest checkpoint 与 metrics 提交后进行，原恢复点备份至
`before_collection_speed_migration/`。严格比对协议：只有这两个源码哈希变化，
重新计算 fingerprint；actor、critic、优化器、RNG、episode 计数和评估记录不变。
initial/best/latest 一起迁移以保持最后 holdout 的 best checkpoint 可加载。
进程只针对本次 Montereau 500 轮任务，其他训练不受影响。
实际提交轮次和 PID 以 `deployment.json` 与实验 `launch.json` 为准。

## 限制

未修改 SUMO 引擎、没有降低物理/观察频率，也没有减少训练样本。
SUMO 内部推进仍需约二十多秒；车辆状态读取、动作执行、存档等仍有开销。
本轮证据支持约 20% 的采集加速，不支持已经稳定达到每轮一分钟以内，
也不是交通性能改善的证据。未重跑整个仓库的长时间训练测试集。

## 验证完成情况

相关路由、动态 prior/AC、事件 session、micro/meso、订阅/直接读取、
no-revisit、实时费用观察测试合计 **142 passed in 268.48s**。
Python 编译检查与工作区 `git diff --check` 通过。
迁移发生在第 36 轮完整提交后；initial/best/latest 的全部非协议状态
（包括参数、优化器、随机状态和进度）递归精确比较通过。

优化版已经完成第 37 轮采集、actor/critic 更新和新 checkpoint 保存，继续第 38 轮。
该轮真实采集 92.21 秒、学习 4.66 秒。它与计时回放的需求/轨迹不同，不能
直接与 75 秒或前一训练轮相除计算加速率；正式吞吐仍需跨轮统计。
