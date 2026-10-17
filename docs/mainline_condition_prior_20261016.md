# 主线：软滤波背景仅作为 FM 条件

新主线配置固定为软滤波、状态独立于动作的背景预测、仅输出端过滤，默认 `tau=0.1`。模型使用

`source = Z + graph_noise`，`reference = reference_proj(Z)`，

`velocity = flow_head(H) + g([H, Z, b(Z), X_t-Z, t])`。

`g` 为 449→128→64 的 SiLU 网络，末层零初始化。背景增量不再移动起点或参考状态，FM 条件中的背景增量停止梯度，prior 继续通过独立的辅助预测损失训练。输入 Z 保留全部图频率；输出滤波算子为 `(I + tau Lsym)^-1`。图噪声仍由原 rank-16 基底采样，不随 tau 改变。

```toml
[method]
prior_filter = "soft"
prior_filter_strength = 0.1
prior_action_conditioned = false
prior_filter_input = false
prior_usage = "condition_only"
```

全部已发布配置以及 MainlineTrainer、配置驱动的 benchmark/evaluation 构造器已接线。独立/递归 CFM、Euler/midpoint/Picard、分歧采样及 CUDA Graph replay 使用一致的条件；加速缓存每次更新当前状态和背景。非 FM regression/Gaussian 消融移除 FM 专用条件残差，保留各自原有预测定义，并在变体元信息中记录该适配。

历史配置缺少 `prior_usage` 时仍按 `source_reference` 解析，缺省的旧滤波强度仍为 10；低层模型构造默认也保留旧行为和参数键。新用法进入语义 fingerprint，不能静默以新结构恢复旧优化器检查点。已保存的完整预训练配置两种历史 fingerprint 均核对不变。正在运行的冻结实验快照没有被替换或重启。

此次固定的是结构和未来新运行的配置，不是自动迁移实验 B 的训练权重。新主线仍训练 GNN prior，条件 RMS 缓冲区初始为 1；实验 B 使用已拟合的固定 prior 输出头和训练集 RMS。需要迁移 B 时必须一并映射输出头、条件残差与三个归一化标量，不能直接宣称新初始化主线具有 B 的测试成绩。原生条件分支已用相同权重和 RMS 与实验 B 的完整 transition 及分歧样本核对。

低成本 tau 搜索使用固定的已适配 B 权重、固定 prior 输出头，在 16 个验证需求组、32 个状态上执行 K=8、Euler100 的完整 FM 推理。搜索 `0, .01, .1, 1, 10, 100`，未使用测试集选参。修正后的 Energy Score 为：

| tau | Fair Energy Score |
| --- | ---: |
| 0 | 8.303703 |
| 0.01 | 8.303745 |
| 0.1 | 8.304029 |
| 1 | 8.304653 |
| 10 | 8.304382 |
| 100 | 8.304290 |

六组最大差距约 0.0114%；数值最优 tau=0 相对 .1 改善约 0.0039%，而车辆数与速度 CRPS 偏向 tau=100。没有发现实际明显更好的统一参数，因此保留 .1。这是固定权重推理敏感性，不是每个 tau 独立重训后的最优性结论。

结果位于 `outputs/tau_condition_20261016/v1`；独立复算 1344 项指标通过。主线验证覆盖配置与历史身份、真实 Dynamics 梯度和求解器、训练器/优化器与检查点恢复、实验 B 等价性、非 FM 消融，以及真实小图 CUDA Graph capture/replay。
