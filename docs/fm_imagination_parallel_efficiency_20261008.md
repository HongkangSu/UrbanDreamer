# Euler 想象推理的并行效率优化（2026-10-08）

新主线配置开启严格 FP32 的静态整批速度场编译，保留 Euler100、每条轨迹三份噪声和八条想象轨迹。优化范围是世界模型的无梯度速度场前向，不改变 actor、V、事件Q 的计算精度或更新次数。

## 实现与配置

`urban_dreamer/learning/fm_inference.py::configure_imagination_inference` 在 `_train_imagination_pass` 开始时应用。默认预设为：

```toml
[resources]
world_model_imagination_precision = "fp32"
world_model_imagination_compile = true
world_model_imagination_compile_mode = "default"
world_model_imagination_space_chunk = 16
world_model_imagination_time_chunk = 32768
world_model_imagination_mlp_chunk = 16384
```

静态编译只用于 eval、CUDA、no_grad、已有窗口条件、完整24样本、单个物理时间token的速度场调用。尾批、CPU和有梯度/训练模式保持eager；CPU effective profile记录FP32且不编译。没有padding或丢弃轨迹。噪声、条件编码、prior、Euler累加与返回latent始终保持FP32。参数对象、optimizer引用和state keys不变；checkpoint只记录可安全加载的普通profile字典。复制模型时重新绑定编译函数，避免引用原模型。

新17份预设显式开启，低层缺字段时compile仍为False，默认新资源字段从语义指纹剔除，旧配置身份与行为保持兼容。非默认设置进入指纹。正在运行的预训练和旧在线任务使用各自冻结代码/配置，本次未重启它们。

## 同条件 GPU 证据

固定同一个城市checkpoint、起点、窗口动作事件和噪声。每条轨迹仍三份噪声，每次仍100个Euler步。最终确认使用实际helper，模型eval/no_grad但dynamics参数requires_grad=True，与训练主线相同。每方法三次warmup后做九轮交替计时。

| 八轨迹×三噪声 | 普通执行 | FP32静态编译 |
|---|---:|---:|
| 单次中位耗时 | 7.52049秒 | 6.74208秒 |
| 九次总耗时 | 66.0534秒 | 60.2965秒 |

九轮中八轮编译更快，总耗时减少8.72%，配对加速比中位数1.13964。保守报告约9%的完整批量收益，不宣称数倍加速或整轮在线训练同幅提速。首次编译执行18.20秒包含编译和一次求解，不计入稳态。B1两种设置均走eager，其耗时差异体现共享负载波动，不能作为尾批优化收益。

此前三轮确认的连续八个物理窗口，在固定记录动作与相同噪声下：latent相对差0.01576%，全局样本方差变化+0.0062%，解码速度平均差约2.2e-5m/s、车辆数差约9e-6辆/路段。无非有限结果或CUDA Graph输出覆盖。该验证是数值一致性检查，不是交通性能或策略收敛证明。

## 没有采用的设置

更大MLP块或完全关闭分块没有稳定提速。保持严格FP32但关闭分块的编译也更慢，最终选择原有在线分块。reduce-overhead/CUDA Graph模式只得到约4%的初测收益，未选为默认。

只在速度场使用BF16初测快约2.57倍，但B8连续八步latent偏差7.42%、全局方差+17.04%；局部TF32快约1.60倍，但latent偏差3.98%。这些可能改变actor/Q看到的latent，未默认启用。两者仍作为显式可选项用于后续诊断。TF32使用PyTorch2.9+的局部matmul精度API，退出或异常时恢复原状态，不改变外部模型阶段。

## 证据位置与限制

`outputs/ingolstadt_noincident_pretrain_20261008/euler_inference_benchmark/` 包含 `REPORT.md`、`summary.json`、`fp32_confirm9.json` 和各候选实验原始结果。新工具 `tools/benchmark_fm_inference.py` 复用固定快照，记录冷编译、重复计时、精度与八步递归差异，不写生产checkpoint。

GPU与其他任务共享，基线耗时跨实验明显波动；只使用同轮配对结果比较。八轨迹性能测试复制同一拥堵起点并使用独立噪声，验证实际batch形状，未覆盖八组不同需求的完整策略执行。世界模型时间不包含车辆推进、actor决策和actor–critic更新。保持100步意味着仍需100次顺序速度场推进，此优化不能将其计算量消除。

相关回归269项通过，覆盖配置、GPU/CPU与尾批分派、训练梯度原路、复制与weights_only恢复、串并行想象、Euler100、事件Q/actor更新和固定反事实基线。Python编译及 `git diff --check` 通过；未新增依赖。GPU数值与性能证据来自上述实际城市快照；单元测试不替代城市性能评估。
