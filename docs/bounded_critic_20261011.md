# 有限时域、固定车辆集合的有界 critic

本次实现保持物理奖励、gamma=1、lambda=.95、actor entropy 和路由先验不变。
配置显式启用 `method.critic_value_mode = "bounded_cohort_v1"`；省略时保留旧行为和旧配置指纹。

对状态或决策事件，用实际物理计数计算
`B = (deadline - now) * active_count / (60 * seed_started_count)`。
V 和每个合法候选 Q 的网络输出是成本比例，接口返回的是 `-B * sigmoid(raw)`，即具体的分钟/起点已出发车辆。
V/Q loss、lambda return、bootstrap、反事实 baseline 和 actor advantage 都使用这些物理单位，不使用比例替代回报。
零活跃车辆、零剩余时间输出精确零；空系统避免除零。分母不是最终需求数，也不随到达而减少。

新模式只允许零模型分歧惩罚；其他额外负奖励不在该物理界内。预测和训练 target 均检查范围；无效 target 报错，不裁剪掩盖错误。
FP32 预算和输出在 BF16 运行入口保持原单位。极小预算在比例诊断除法前排除。

## 自举与终止样本

完整 target V 从新 V 复制，关闭梯度，每个 pass 固定。
全部中间 next-value 和事件末尾尾值使用 target V；每次在线 V optimizer 更新后进行一次 EMA，默认 tau=.01，下一 pass 才使用新目标。

原来的 actor rollout 流保持不变。默认额外采样 `ceil(.25 * rollout_count)` 条能触达截止时间的轨迹，**仅用于 V**。
当前 16 条原轨迹对应四条辅助轨迹；按每条轨迹的平均 loss 等权平均，辅助部分占 V loss 的 20%。
辅助轨迹不进入 actor/Q loss 或 actor advantage 统计；NumPy、Torch CPU/CUDA 随机状态独立保存，并在异常路径恢复主随机状态。
终止候选池为空时不增加轨迹。候选池只用已观测起点时间和实际 horizon/deadline。

## Checkpoint 与迁移

新 critic 带持久化的输出契约标记。旧权重不能直接按比例语义加载，即使调用者使用 `strict=False`。
新状态包含 `target_latent_critic`、辅助随机状态和 mode/tau/ratio；恢复时检查这些约定。
CUDA 恢复要求保存的辅助 CUDA 状态数量与当前可见设备一致，避免悄悄借用主随机流。CPU 检查可以读取 CUDA 产生的状态。

`tools/migrate_bounded_critics.py` 要求新的输出目录，只读输入 checkpoint，重新初始化 V/Q 和它们的 optimizer state，建立 target V。
保留 actor、世界模型、其他 optimizer、replay、原随机状态、episode 和 entropy 时钟。它不会启动或停止训练。

本次第 345 轮隔离候选在 `outputs/ingo_bounded_critic_candidate_20261011`。
`critic_migration.json` 记录来源、哈希和迁移边界；`preservation_verification.json` 独立比较实际模型及学习状态。
这是未做真实交通校准的候选初始化，不能把零越界率理解为训练已经恢复。

## 验证与边界

测试覆盖模型范围、物理单位、零预算梯度、非法输入、旧权重拒绝、目标 EMA、串并行一致性、target 范围、辅助样本不影响 actor/Q、随机状态和 checkpoint 恢复。
精确回报的小环境测试采用事先固定的训练预算与门槛，比较相同共享参数初始化的有界/无界 critic，检查物理分钟误差和可分动作排序。
验收协议在 `.omx/plans/critic-learning-validation-protocol.json`。

这些测试不证明 SUMO 到达率改善，也不证明 TD 长期稳定或交通状态表示充分。当前主线仍使用旧冻结源码，未被本次更改重启。
真实同策略校准、成功/失败状态覆盖及 actor 解冻后的闭环实验仍是独立发布门槛。
当前实现刻意不改变 critic 输入特征，也不通过重新缩放旧输入来掩盖饱和问题；保留物理 MSE，避免悄悄引入 1/B² 的样本加权。
