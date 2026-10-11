# 每轮想象训练的一分钟预算

用户允许减少想象推理批数。新在线运行从原完整checkpoint的第83轮接续，保留所有模型、Adam、replay、RNG、历史曲线和固定需求评估；总在线目标仍2000轮。原输出保留，新输出为 `outputs/ingolstadt_noincident_pretrain_20261008/online_actor_critic_2000_1min`，监督日志在 `online_1min_supervisor`。

## 耗时证据与方案选择

原始16批每轮，真实稳态每批约42.97秒；世界模型占38.64秒（89.9%），CPU车辆推进/actor/Q/worker等待3.20秒，目标构建0.79秒，backward0.023秒，Adam0.0034秒。单窗GPU kernels中SGEMM约49.6%，FP32 attention约39.4%，主要是计算量。完整预算矩阵工作量约5.14PFLOP，本机3090最高时钟理论峰值约44.08TFLOPS时也需116.55秒，未计其他工作与共享竞争。

完整Euler100 CUDA Graph无可靠完整批量收益且driver额外消耗约2GiB，未默认接入。降低Euler至4/8/16/32步虽然显著提速，但四个条件30/90/240秒的latent/variance/局部慢路变化均未过数值门槛，未部署。Euler100只是数值对照，不是交通真值；这些试验也不能证明100步已经预测准确。

采用预算调整，每轮最多两次完整imagination pass、60秒准入目标。至少完成一次，使用最新实测耗时与持久化EMA估算是否有时间执行第二次。若当前约42秒一批，则完成一次后不会启动第二次；单批低于30秒时可能两次。每批仍保持8条轨迹、3份噪声、最多8个30秒宏观窗口、100步Euler、严格FP32。每个完整pass均做一次actor、V、事件Q更新；每轮4次真实数据世界模型更新不变。损失平均按真实完成批数计算。

60秒是完整批次准入目标，不是硬中断。首次编译、CPU worker启动、共享GPU突然变慢可能超过目标，代码会完成当前完整批并标记budget_exceeded。日志记录requested/completed pass数、想象秒数、预算、EMA与停止原因。一次batch减少并非推理kernel本身加速，会降低每轮policy/critic更新次数；后续收敛需评估。

历史配置缺字段仍16/0且语义指纹不变，新17个预设明确2/60。新预算进入指纹，已有在线状态通过显式 `imagination_budget_migration.json` 迁移，不默默把旧checkpoint当相同实验。未提交的源episode档仅冻结在source_history，工作目录排除它们，防止新预算provenance冲突；原源档未改。

核心代码：`training.py::resolved_imagination_budget` 和完整pass预算循环，`config.py`资源字段，`tools/migrate_imagination_budget.py`与在线runner预算来源。相关223项回归和76项专项检查通过；默认兼容、Adam/RNG精确迁移、warm/cold/意外慢批、完整step与未提交档处理均有测试。源码冻结在source_online_1min，模型不使用正在校准的decoder候选。

Decoder独立校准已完成一轮20682次更新：慢路MAE降至约0.341m/s、局部条件平均约0.338m/s，但最差局部路段仍超过1m/s门槛，因此候选未部署。速度优化不放宽该验收。

## 接续运行实测

2026-10-09 14:13 UTC 核对：原83轮完整状态接续后，已提交4轮，累计87轮，2000轮任务继续运行。零基episode_index=83为冷启动，想象与AC共78.21秒，正确标记budget_exceeded；后续84、85、86分别为41.21、41.99、38.15秒，平均40.45秒。每轮均完成一批，第二批因剩余时间不足而未启动。

三轮稳态的episode计时分别57.98、59.73、57.59秒，包括真实采样、四次世界模型更新及想象/AC，**不包括checkpoint写盘和定期评估**。真实SUMO执行为13.98–16.45秒。与旧每轮16批约650秒想象耗时相比，减少预算后约16倍提速；这主要来自工作量减少，单批推理速度没有获得相同倍数提升，也尚未达到与SUMO采集相同的耗时。

检查了新四轮的有限损失、Q/优势/反事实基线统计、实际完成批数、actor/V/Q的optimizer步数与参数变化，三者每轮各更新一次。稳态三轮Q损失0.00749–0.01990，优势均值0.00134–0.10686、标准差0.08581–0.09436，反事实基线均值−0.76085至−0.59434；这些指标仅验证更新执行及数值有限，不能证明交通性能改善。证据保存于输出父目录的one_minute_budget_verification.json及新运行pilot_metrics.jsonl。最后重跑预算、迁移、在线runner与推理实验相关76项测试均通过，Python编译和git diff --check通过。
