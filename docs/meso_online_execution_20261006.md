# 现有预训练 checkpoint 的中观在线运行

按用户要求，复用现有 checkpoint 启动 Ingolstadt 2000 轮中观在线训练，并停止、替换原 baseline。用户随后明确要求不再守 baseline 进程，工作重点为主线；新 baseline 的后台任务保留运行。

## 主线初始化

源为 `outputs/ingolstadt_bf16_pretrain_20261002/checkpoints/latest.pt`，已完成 5 个微观世界模型预训练 epoch，codec 6500 次更新，尚无在线 actor–critic 训练。源文件 SHA256 为 `56aff81065c521593eff0ab59ddf37505b75442f11dd9634cfd991c472ed99c1`，迁移未修改它。

新运行 `outputs/ingolstadt_online_meso_20261006`。原始 codec 在两个独立中观需求上的 raw/hybrid 车辆数验证未过门，故只校准 decoder；encoder、dynamics、辅助读出、actor、V、Q 均保持与源一致。900 次 decoder 更新、229.6 秒后通过所有原阈值。raw 全道路车辆 MAE 0.09812、有车 MAE 0.38736、总数相对误差 0.04695、速度 MAE 0.33218 m/s；hybrid 对应 0.09073、0.46276、0.06944、0.29218 m/s。独立 CPU 重新打开并重新计算指标通过。

不重新执行完整 FM 预训练，不复用旧微观 replay、optimizer、随机状态和评估。初始化物理语义显式记录为迁移；目标中观预训练 epoch 为 0，源预训练证明单独保存。新权重在 `initialization_checkpoint.pt` 中冻结，SHA256 `fddba6008a251ae01312e580de4b593e5723116603b377aba91d73f8787c2396`。新运行从在线 episode 0 开始。

## 持续运行

后台监督脚本 `outputs/ingolstadt_online_meso_20261006_supervisor/run.py`，运行状态 `status.json`，日志 `run.log`。主线源码冻结于新输出的 `source/`，输入 network 文件保持只读引用。监督脚本最多 3 次故障重试，恢复已经完成的 checkpoint；用户停止信号不会触发自动重启。

配置为 meso/libsumo1.27.1/event，1 秒观察、30 秒窗口；GPU1、4 个父进程线程、8 个 CPU imagination workers，CUDA memory fraction 0.08。在线预算保持 2000：每 episode 4 次世界模型更新，16 个 imagination pass，每 pass 8 rollout、最多 8 窗口；每 20 episode 固定 3 个需求评估，每完整 episode 保存 checkpoint。

运行已完成初始 3 个固定评估并进入真实在线训练。核验时严格 checkpoint 已完整保存第 4 轮，实时 JSON 进度随后推进到第 5 轮、第 6 轮正在更新。前 5 轮每轮均有 16 次 actor、event Q、macro V optimizer step；耗时 179.7/170.0/168.3/157.7/161.7 秒。进度不断变化，应读实时 status/progress。

初始和前几轮策略质量较低，部分 actor 优势接近裁剪下限，不能由训练完成或有限 loss 断言收敛。源是世界模型完成预训练时的 checkpoint，其策略尚未在线训练；新的世界模型转移也来自微观，需要在线适配中观动力学。

## Baseline 替换

旧 `outputs/city_baseline_dataset_no_revisit_5000ep_20261004_201004` 的 15 个 worker 和 owned controllers 已停，无遗留 owned 进程；保存 9682 个 NPZ、36 个 checkpoint ZIP CRC/长度清单、源快照与指标。新 `outputs/city_baseline_dataset_meso_no_revisit_5000ep_20261006_155101` 的 15 个独立任务均已启动并有 protocol/初始 checkpoint。

保留原 baseline 设置：3 城市 × Online AC/GNAC/Dreamer/MBPO/random，每个 5000 episode，seed42、禁止 revisit、零最短路先验、checkpoint1/eval100/3seeds。baseline 从新模型/optimizer/replay开始。mainline 继续其原来的目的地可达动作集合，不新增 baseline 的 no-revisit 实验约束；本轮统一的是中观物理配置，不能假定全部方法的策略约束都相同。

GPU0：Ingolstadt Dreamer/GNAC、Fontenay GNAC。GPU1：Gretz GNAC/Dreamer/MBPO、Ingolstadt MBPO。Fontenay Dreamer/MBPO及所有 Online AC/random 在 CPU。baseline 的 supervisor/watch/compactor 保持后台运行，不再额外等待全部首轮完成。

## 验证

27 个 checkpoint reuse、online pilot、backend migration 相关测试通过；新增初始化工具及监督脚本编译、`git diff --check` 通过。实际主线已进行物理采集、4 次/轮 FM 更新、完整 16 pass actor–critic 更新和严格 checkpoint 保存/加载。2000 轮尚在运行，本记录不宣称全程完成。
