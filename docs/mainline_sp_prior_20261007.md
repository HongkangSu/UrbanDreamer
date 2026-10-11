# 主线最短路先验续训（2026-10-07）

用户要求与正在运行的 Online AC baseline 使用相同的最短路先验。复用 `models.policy` 的先验函数，在合法候选 logits 上加入 `-w * (cost - min_cost) / 30`；categorical 温度仍为 1。初始权重 1，前 200 轮保持不变，随后余弦退火，第 1000 轮起归零。

每轮在真实采集前按全局一基轮次设置权重，此权重保持到该轮全部 16 次想象更新结束。真实采样、CPU 想象、并行想象、GPU 重算 log-prob 和反事实基线期望使用同一行为概率。确定性评估使用零先验 argmax，与此前评估保持相同定义。

## 续训

从 `outputs/ingolstadt_online_meso_20261006` 第 482 轮完整 checkpoint 迁移到 `outputs/ingolstadt_online_meso_sp_prior_20261007`。新任务从第 483 轮开始，累计预算仍为 2000 轮。原有模型、优化器、RNG、有限 replay、训练计数、历史评估逐字段保留；没有再次离线预训练，也没有重置先验退火时钟。baseline 进程未改动。

迁移工具 `tools/migrate_mainline_prior.py` 只允许修改四个先验字段、输出目录及设备。未启用先验的配置保留原语义指纹；启用后生成新指纹，必须显式迁移，不能作为原架构的静默续训。不可变的源 checkpoint、初始续训 checkpoint、源配置及协议均保存 SHA256，并在每次启动时验证。

`policy_prior_migration.json` 记录来源和全局轮次；`continuation_verification.json` 记录实际逐字段状态校验；`source_changes.json` 记录新冻结代码中替换的文件。原始 run 保留全部旧存档，新 run 的历史曲线继承 checkpoint 中的记录，新增事件存档从当前轮次继续。

## 使用

新配置的 `[method]` 设置 `sp_prior_weight=1.0`、`sp_prior_temperature=30.0`、`sp_prior_hold_episodes=200`、`sp_prior_off_episode=1000`。当前默认已按用户要求改回关闭：所有随附主线 TOML 的 sp_prior_weight=0；设为 1 可启用上述调度。旧配置省略字段时仍按权重 0 解析，保持已有 checkpoint 语义兼容；需要关闭的新任务应显式设 sp_prior_weight=0。学习型 baseline 的 CLI 同样默认关闭，--sp-prior-weight 1 可显式开启。运行中的冻结任务不随默认值变化。

训练存档和指标新增实际先验权重、温度及全局调度轮次。先验变化是行为策略干预，刚切换后的采样奖励改善不能直接归因于模型学习；应同时观察零先验确定性评估和先验归零后的表现。先验不修复世界模型的局部预测偏差。

## 实际验证

完整 pytest：1115 passed（496.34 秒）；Python 编译与 git diff --check 通过。第 483 轮已完成真实采样、4 次世界模型更新和 16 次想象 actor–critic 更新，并保存 checkpoint，随后进入第 484 轮。首轮先验权重 0.7217311573；1026 辆车全部到达，平均旅行时间 5.775406 分钟/车；平均事件 Q loss 0.192392，actor advantage 均值 -0.141131、标准差 0.195387，均为有限值。actor、宏观 V 与事件 Q 参数均有更新。明细保存在新输出目录的 first_prior_round_verification.json。单轮结果仅验证功能运行，不能宣称收敛或交通性能提升。
