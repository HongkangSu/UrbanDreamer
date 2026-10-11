# 中观主线与 baseline 的共享执行设置

本次迁移将仓库提供的 Toy、Ingolstadt、Gretz、Fontenay 主线、离线配置和 baseline 启动配置切换为相同的中观 profile。旧采集任务的冻结 source/config、NPZ 和 checkpoint 保持原样。迁移配置不会自动启动全量采集、预训练或 2000 轮训练。

## 配置与执行

```toml
[simulation]
traffic_model = "meso"
session_protocol = "event"
observation_step_seconds = 1.0
backend = "libsumo"
meso_junction_control = true
meso_lane_queue = false
meso_multi_queue = false
meso_edge_length_m = 98.0
meso_interpolate_positions = true
```

以上布尔设置是本轮验证的固定 profile，配置解析会拒绝冲突值。段长度允许正数，改变它会改变语义指纹。中观只接受 1 秒 event 观察；车辆按外部路段转移，动作记录仍为 branch-only，世界模型窗口仍为 30 秒。红灯、路口优先权、下游容量和排队保留；单队列意味着前车等待红灯时也会阻挡后车，即使后车计划走另一个绿灯方向。

共享 `traffic/simulation_settings.py` 生成 SUMO 参数和物理身份。主线、Online AC、GNAC、Dreamer、MBPO、DTA、最短路及随机采集均使用同一配置。省略 CLI 的 session/backend/step 参数时从配置读取；中观时显式传入冲突参数会报错。不能通过 `sumo_extra_args` 注入 `--mesosim`、`--meso-*` 或 `--no-internal-links` 绕过身份记录。

已在当前 URB 环境的 **libsumo 1.27.1** 验证位置插值。现有旧外部 SUMO 不提供 `--meso-interpolate-pos`，因此新配置使用 libsumo；没有新增或安装依赖。缺少 `traffic_model` 的历史配置仍默认为 micro，旧微观语义指纹保持兼容。

事故限速使用中观有效的 edge API，并恢复原有车道速度。状态快照不记录虚构内部车道进度；`remaining_distance` 来自外部边的插值位置，已核查四个网络的 registry 长度与第一车道长度一致到厘米级。出发、到达、观察边界费用和 `[t_n,t_{n+1})` 动作归属保留。

## 策略与想象

`NextHopCandidates` 使用独立的 micro/meso 视图。中观路线成本、剩余距离、actor 候选特征和路线后缀不再包含微观内部连接长度。原始网络连接没有被删除；候选缓存不会跨物理视图混用。

主线 serial/parallel 想象和 model-based baseline 显式传入相同模型。中观想象仅采用 event 协议，外部边末端直接进入后继，拒绝旧微观内部车道种子；1 秒观察、短边跳过、跨窗口承诺和边界动作规则均有测试。

中观还取消了旧微观想象的 0.05 归一化速度下限。主线 serial/parallel、Dreamer 和 MBPO 共用速度裁剪规则：预测零速允许保持停驶，完整计算在途费用；速度上限及车辆最大速度约束保留。旧微观调用保持原下限。

**仍存在的模型限制：**想象车辆目前按解码路段速度独立推进，并未成为完整的 SUMO 中观队列仿真器。真实单队列的相互阻塞、信号和容量作用需要由学习模型及后续奖励/动力学改进处理。此次验证证明迁移和有限训练链路可运行、可恢复，不证明策略已收敛或交通性能提升。

## 数据与 checkpoint

新存档保存完整 `physics_identity`，离线 manifest、压缩 cache、codec 源和 ready 证明都检查物理身份。没有模型字段的历史数据按 micro 处理；不能通过修改一个标签或复制新配置，将旧微观轨迹声明成中观。即使 codec 只读取 flow，cache 的物理身份仍会核验。

城市配置的离线根目录改为 `outputs/city_baseline_dataset_meso`，输出目录追加 `_meso`，Ingolstadt 在线预算为 2000 轮。这个根目录用于新采集的数据；不能直接填入原微观集合。Toy 仍需要指定兼容的离线训练源，短程验证的需求设置与生产 profile 不同，不自动注册为生产数据。

新物理模型/设置进入配置与 checkpoint 指纹。旧完整微观 checkpoint 不是中观严格续训点；旧网络权重只可作为明确迁移的初始化，并重新验证 codec、拟合中观转移和价值目标。当前迁移没有把旧 actor/Q/V 优化器或旧 replay 冒充新的续训状态。

例如，采集新的 Ingolstadt 中观随机轨迹（需自行设定合适的 episode 数，下面仅为短程检查）：

```bash
python tools/collect_random_episodes.py \
  --config configs/ingolstadt_custom.toml \
  --output-dir outputs/city_baseline_dataset_meso/ingolstadt_custom/random \
  --episodes 2 --threads 1
```

baseline 同样从配置获取物理设置，例如：

```bash
python tools/train_toy_actor_critic.py \
  --config configs/ingolstadt_custom.toml \
  --output-dir outputs/ingolstadt_online_ac_meso \
  --episodes 2 --save-episodes --skip-reference-baselines
```

常规主线继续是独立 codec 预训练、世界模型离线预训练、在线 actor–critic 训练。兼容数据准备好后：

```bash
python -m urban_dreamer --config configs/ingolstadt_custom.toml
```

## 验证记录

真实 SUMO 测试覆盖位置插值、合法路由承诺、自动唯一后继、短分叉跳过、完整 no-revisit 历史、1 秒动作/30 秒窗口、到达与 exposure 守恒、事故限速/恢复、红灯阻塞/绿灯释放、单队列前车阻塞。

数据测试覆盖旧微观指纹、模型/段长度混用、存档错误重标记、cache/flow-only 身份检查、微观 codec ready 证明被中观配置拒绝，以及真实旧 schema9 微观迁移兼容性。

隔离工具 `tools/verify_meso_migration.py` 将真实 Toy 中观轨迹送入留出 codec 验证、FM、actor/V/Q、确定性评估和严格 checkpoint 恢复。测试预算明确缩小，用于功能验证，不代替生产 2000 轮训练。原高学习率失败试验与按生产学习率成功试验均保留，未绕过 codec gate。

详细结果在 `outputs/meso_migration_validation_20261006/`。最终完整测试 **1019 passed in 323.69s**，日志为 `outputs/meso_feasibility_20261006/full_pytest_final.log`。Python 全目录编译检查和 `git diff --check` 通过。现有环境没有 ruff/mypy/flake8，不为检查安装依赖。

Toy 四条真实轨迹共 75 辆车、134 次动作；按生产 codec 学习率 `1e-4`，2672 次更新后原始及 hybrid 留出门均通过：有车道路 MAE 0.3464/0.2928、全路段 MAE 0.0893/0.0785、总量相对误差 0.0994/0.0785、速度 MAE 0.1223/0.1397 m/s。之后完成 12 次离线 FM 更新、真实在线采集、actor/V/Q 更新和严格 checkpoint 恢复。取消零速下限后，另从这个 checkpoint 执行一次 **8 rollout、8 窗口 horizon** 的想象 pass，16 次事件，世界模型、actor、V、Q 均有实际参数更新；Q loss 0.4804，保存于 `production_lr/zero_speed_update_full_horizon.json`。额外的 2 rollout/2 窗口测试没有事件，V 正常更新且 actor/Q 正确跳过；保留日志，不把短 horizon 下无事件当作训练已收敛。

Ingolstadt 八条实现（主线、最短路、随机、Online AC、GNAC、Dreamer、MBPO、DTA）在相同测试需求上实际执行，物理身份与需求哈希一致。这一组参考人口限制为 16、实际 22 辆，使用 fresh actors，不能证明城市策略质量。

正式 profile 另完成 1026 辆车的完整确定性路段成本采集，10682 次动作、2089 个物理步、全部到达，耗时 4.92 秒。此时间不含神经 actor、模型更新和存档，不能当作整轮训练耗时或相对微观环境的策略改进。记录为 `outputs/meso_feasibility_20261006/migrated_full_city.json`。

已在新生产数据根目录采集两条标准 Ingolstadt 随机策略存档：1007/1041 辆车、96416/65984 次动作、各 127 个窗口。总计 **2048 辆车、162400 次动作、254 个窗口**，严格导入、内容完整性与第一批 `[1,4,853,8]` 张量有限性检查通过。路径为 `outputs/city_baseline_dataset_meso/ingolstadt_custom/random/episodes/train/`，导入报告为 `outputs/meso_migration_validation_20261006/standard_ingolstadt_import/report.json`。随机策略完成率 30.19%/18.92%，两个 episode 均按截断保留；这是可用的兼容性种子数据，不是足够训练城市 world model 的数据规模，也不是性能改善证据。

车辆数标签约定保持不变：上一状态输入保留完整观测；本窗口下一状态车辆数监督只排除本窗口新增出发车辆；后续 teacher-forced 输入使用完整真实下一状态。
