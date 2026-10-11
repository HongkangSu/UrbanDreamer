# 主线 entropy 单位修正（2026-10-08）

用户要求先修正 entropy，保留当前世界模型与 critic，世界模型重训方案另行决定。共享回报按完整 episode 人口 N 除过，旧固定 entropy 系数0.002未除人口。现在每个事件使用0.002 / (N * max(std(global_advantage),1))，其中统计量 detach。个体超时惩罚与优势裁剪不变。默认 categorical 温度仍为1，prior为0。

随附主线配置显式 actor_entropy_mode="population"；历史配置省略时按legacy解析，保持旧语义指纹。population/off是不同新指纹。训练配置决定实际模式，恢复checkpoint的旧metrics['actor_entropy_mode']不会将其覆盖。生产runner禁止与配置冲突的环境变量。

当前运行从第525轮完整checkpoint迁移到 outputs/ingolstadt_online_meso_noincident_entropyfix_20261008，下一轮526，总预算2000。源checkpoint、源配置、源protocol和初始continuation checkpoint均不可变并保存哈希；模型、优化器、replay、RNG、轮次与历史评估逐字段保持相同。actor_entropy_mode是唯一学习设置变化。GPU上限仍为25%（6GiB），batch4，4次WM更新与16次想象pass，8个rollout和8个30秒窗口。

新代码和运行协议在source/、source_changes.json、entropy_migration.json及pilot_protocol.json可复核。旧run输出全部保留，曲线可继承checkpoint的历史记录并在525轮标记切换。

本修正解决entropy与回报尺度不一致，不承诺消除世界模型的局部队列偏差或critic尾部价值偏差。此前冻结493轮的两组需求对照中，人口缩放基本保持起点真实费用；关闭entropy仍退化，故不能将其认定为全部问题。

## 实际运行验证

第526轮完成4次世界模型更新和16次想象actor–critic更新，已保存checkpoint，随后进入527轮。平均有效entropy系数1.882859763e-6，优势缩放系数1，prior0；所有记录损失和统计量有限。第一轮真实采集1049辆，完成率99.619%，共享奖励-15.0342；该采集发生在本轮actor更新之前，因此不能将其归因于entropy修正。部署验证以有效系数与恢复成功为准，不宣称收敛。主集成/恢复回归121项通过，另一个checkpoint/CLI/pipeline/FM组76项通过，Python编译及git diff --check通过。
