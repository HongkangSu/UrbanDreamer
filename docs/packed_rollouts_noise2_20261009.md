# 合批投影与双噪声想象推理

用户要求轨迹和噪声样本共同沿 batch 维拼接，线性投影与 MLP 使用整批路段，attention 保持不同轨迹独立，并将噪声样本由三个改成两个。

## 实现

`SpaceSelfAttention.packed_projections` 提供显式可选路径：QKV 对 batch×路段展平后一次投影，SDPA 仅沿独立 batch 分块，输出再次整批投影。参数与 checkpoint state keys 不变，默认保留旧路径。MLP 使用已有 token_chunk_size=0 即可整批执行，无需新建网络层。

八轨迹×两个噪声=16 模型样本，853条路段加5个特殊token共13,728个token，已经落在原空间attention块16和MLP块16,384之内。因此双噪声八轨迹下，旧路径本身已执行全批QKV、MLP及输出投影；attention仍在16个独立路网内分别计算。

17份主线预设明确设置 disagreement_samples=2。低层MethodConfig缺字段默认3保留，避免重新解释旧配置和旧checkpoint。实际编译full_batch根据8×disagreement_samples计算，回归测试确认双噪声使用16样本。

## 城市实测

固定同一Ingolstadt在线checkpoint（第70轮快照，SHA256 b0de75527763dfa169b374450895fd6ccc12f0309f00ec163b9e826528f650ed），从验证集选择不同窗口作为轨迹起点、固定噪声，严格FP32/Euler100/双噪声。每个batch尺寸两种路径都使用适配该尺寸的整模型静态编译，排除冷编译后交替测三对。GPU与其他任务共享，绝对耗时和小幅差异有竞争噪声。这里计时为单个30秒世界模型窗口，不是完整8窗口rollout，也不含CPU车辆推进和AC更新。

| Rollout数 | 模型batch | 原路径中位秒 | 全批投影中位秒 | 原路径轨迹窗口/秒 | 全批投影轨迹窗口/秒 |
|---|---|---|---|---|---|
| 1 | 2 | 0.952 | 0.963 | 1.051 | 1.038 |
| 8 | 16 | 3.024 | 3.019 | 2.645 | 2.650 |
| 16 | 32 | 5.421 | 4.871 | 2.951 | 3.285 |

八轨迹两条路径输出逐位一致、显存峰值均303.72MiB。16轨迹latent相对L2差2.52e-7，方差相对L2差8.41e-6，峰值分配显存从415.74MiB升到545.49MiB；三对总时间比为1.063，即约6%的收益，中位时间比约1.113。8轨迹的三对总时间比0.957，不支持其有额外加速收益。显存峰值为PyTorch分配口径，非整个CUDA进程显存。

扩大batch可以提高吞吐，但此次并非倍数级收益。16轨迹整批投影尚未做8窗口连续策略闭环和完整AC验证，保持实验选项；生产双噪声8轨迹本身已不分块，无需强行切换候选路径。

完整记录：`outputs/ingolstadt_noincident_pretrain_20261008/packed_rollouts_20261009/samples2.json`，复现工具 `tools/benchmark_packed_rollouts.py`。

## 续训与验证

旧运行在149轮完整checkpoint接续至`online_actor_critic_2000_noise2`。显式disagreement_samples_continuation仅允许3→2及输出/device变化，记录源与目标身份，逐项验证模型、优化器、replay、RNG、metrics和全局游标一致。旧历史存档保留，未提交episode不会写入新运行的有效存档。每轮最大批数2、60秒准入预算、8轨迹、8窗口、Euler100、4次真实WM更新和2000轮目标由源运行保留。

噪声数降低会减少计算，也会提高disagreement方差估计的不稳定性；不宣称因batch为2的幂就必然加速，也不宣称交通性能改善。

相关配置、迁移、编译形状、独立attention与梯度测试188项通过，世界模型/并行想象/批量AC/多样性相关50项通过，Python编译与git diff --check通过。独立attention测试证明修改其他轨迹不改变当前轨迹输出，投影hook确认QKV和输出各只调用一次。

2026-10-09 15:32 UTC 接续运行已提交第150轮（零基149）：首次包含编译/worker启动的想象与AC耗时57.22秒，完成一批，actor/V/Q各更新一次，loss分别0.003897/0.003123/0.007274且均有限。第151轮已启动。该57.22秒为冷启动，不作为双噪声稳态耗时估计；采样方差和回报曲线还需长期评估。
