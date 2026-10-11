# Ingolstadt BF16与每轮八次AC更新

用户明确批准直接使用BF16，并保持每批8条rollout、每轮8次actor更新。2026-10-09从双噪声FP32主线已完成233轮的完整checkpoint迁移至`online_actor_critic_2000_bf16_8updates`，总目标仍2000轮。

实际配置：velocity-only BF16、Euler100、双噪声（8×2=模型batch16）、每批最多8个30秒窗口、每轮8批，actor/V/Q各更新8次。每轮4次真实世界模型更新保留。60秒准入截断关闭，以保证8次更新；不能同时承诺完整8批低于一分钟。

`imagination_precision_continuation`仅允许FP32→BF16以及此次明确的2批/60秒→8批/不限时组合，记录新旧配置、预算与原始checkpoint身份。模型、优化器、replay、RNG、历史metrics、全局游标精确保留。源运行与存档不覆盖，新的源码快照在`source_online_bf16_8updates`，supervisor位于`online_bf16_8updates_supervisor`。Ingolstadt两个预设同步BF16与8/0，其他城市预设保持原精度和预算。

已验证接续第234轮（零基233）完整提交。八批分别40.724、7.264、9.783、8.328、8.732、8.089、8.470、10.482秒；第一批含编译/worker启动，其余七批平均约8.74秒。想象+AC合计101.88秒，训练计时118.40秒（不含checkpoint写盘），actor/V/Q日志均确认为8次optimizer更新且参数有变化。actor loss −0.002816、Q loss 0.005142、V loss 0.002571，指标均有限。不能把单轮功能验证视为策略收敛证明。

迁移/runner相关79项测试和配置/推理/预算相关142项测试通过；Python编译及git diff --check通过。原始证据为输出父目录`bf16_8updates_verification.json`和新运行`pilot_metrics.jsonl`。

BF16不修复原decoder的局部低速偏差。当前独立decoder校准候选仍未验收上线；AC慢路的真实状态直接编解码诊断见`precision_mae_noise2_20261009.md`与`ac_slow_oracle.json`。
