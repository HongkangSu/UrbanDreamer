# Ingolstadt 无事故、无先验主线重跑（2026-10-07）

用户要求按当前设置重新进行2000轮在线训练。运行目录：`outputs/ingolstadt_online_meso_noincident_noprior_20261007`；监护进程目录：`outputs/ingolstadt_online_meso_noincident_noprior_20261007_supervisor`。原带先验主线在612轮停止并保留全部旧数据。

## 初始化来源

- 微观离线预训练：`outputs/ingolstadt_bf16_pretrain_20261002/checkpoints/latest.pt`（原冻结预训练源）。
- 校准后的中观零轮初始化：`outputs/ingolstadt_online_meso_20261006/initialization_checkpoint.pt`。新任务**只**读取这个零轮初始化，不继续旧线上 actor。
- 7组新采集的无事故随机合法动作轨迹位于 `outputs/ingolstadt_noincident_codec_parallel_20261007`；6组用于数据清单，1组独立需求验证编解码。它们仅用来重新验证已有模型的解码器，不进入线上策略 replay，不构成新一轮世界模型预训练。
- 所有模型权重的 SHA256 与旧零轮初始化逐项相同；无需 decoder 校准。优化器、在线 replay、随机状态、在线轮次和评估计数从零开始。
- 新初始化 checkpoint SHA256：`93a2427fc91f7f831338e37a441faa1f7028b350b70ada9ab1663dc3d30b77b4`；目标配置指纹：`9f428c9fc1eec6bcf9c308d34ef384fa624fb761c0594982136648fda6f85f09`。

## 物理与训练设置

中观 libsumo 1.27.1；event 决策与1秒观察；世界模型30秒更新。事故配置为1,000,000–1,001,000秒，超过本任务所有时间上限，因此不会触发。最短路 prior 权重0。完整预算2000轮，每轮4次世界模型更新、16次想象 actor–critic 更新，每次8个rollout、最多8个30秒窗口。GPU1；8个想象采样进程；每20轮固定需求确定性评估；每轮 checkpoint。

独立无事故数据上的 codec 验证：原始占用路段计数MAE 0.302954，总路段计数MAE 0.082626，速度MAE 0.345915 m/s；hybrid占用计数MAE 0.359960。全部满足未修改的质量门槛，`codec_ready=true`。

初始化文件 `calibrated_noincident_initialization.json` 保留旧checkpoint、旧meso manifest、新checkpoint、配置、数据manifest的哈希以及解码验证证明。训练器启动与恢复都检查这些身份。新运行冻结代码在 `source/`，差异哈希在 `source_changes.json`。

该验证证明复用世界模型的物理编解码达到门槛，并不证明想象中的长期交通动态与真实SUMO完全一致。训练奖励和之前有事故／带prior运行不直接拼接比较。

## 启动验收

三组固定需求的初始评估完成。第一轮在线训练完成：1050辆车，真实完成率26.667%，共享交通奖励-39.1830，事件数80161；4次世界模型更新，16次完整想象actor–critic pass。FM损失均值0.0003757555，事件Q损失均值10.6271，actor优势均值-2.0616、标准差0.3843，均为有限值。模型、Q和actor各自参数均发生更新。首轮耗时177秒，冷启动拥堵与完整时限使SUMO采集约80秒；后续速度须以多轮样本判断，不能据此推断长期平均。

严格加载第1轮保存的 checkpoint 成功，config/input身份一致，next_unit=1、训练计数和durable progress一致，actor及world optimizer含状态。证明记录见 `first_round_verification.json`。监护进程继续运行，保留2000轮预算和每20轮评估。

## 显存上限调整

第14轮完整checkpoint后，按用户要求将GPU1的PyTorch显存上限从0.08（1.92 GiB）提高到0.25（6 GiB）。checkpoint和run_config均未改动，batch仍为4，所有采样与更新预算保持原值。原checkpoint、旧运行协议和旧监护脚本备份于runtime_resource_changes/gpu_fraction_20261007。新进程已从第15轮继续；实际显存按需分配，放宽上限不意味着自动使用6 GiB或自动加速。
