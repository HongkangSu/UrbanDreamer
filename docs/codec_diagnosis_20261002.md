# Ingolstadt codec 误差与分阶段训练诊断

## 实测结论

当前证据指向 codec 内的监督尺度与梯度失衡。车辆数标签在检查样本中正确，
梯度没有断开；FM 与 codec 的直接梯度冲突没有表现为主要因素。分阶段训练
可稳定潜空间并降低 codec 训练成本，但单独训练原 loss 不会自动修正速度／
车辆数通道之间的失衡。

本轮保持原始标签、模型结构和生产训练代码，使用 pilot320 权重进行独立诊断。
没有新增数据、修改存档或启动 SUMO。

## 标签与梯度

对真实 Dreamer episode6 的事故与末期样本检查：990、1020、3750、3780 秒，
853 个路段的 `volume * capacity` 与车辆表独立重建的外部路段计数完全一致，
每个边界 mismatch 数与最大误差均为0。总车辆数分别为490、507、961、962。
codec 的车辆数重建目标与编码输入的同一时刻 channel2 完全相同。

| 指标 | 初始化：事故／末期 | pilot320：事故／末期 |
|---|---:|---:|
| Encoder speed/volume 梯度范数比 | 261580 / 251614 | 2077 / 1472 |
| Shared decoder speed/volume 梯度范数比 | 318923 / 317451 | 2508 / 2058 |
| 全局梯度范数 | 31817 / 27869 | 3.150 / 5.988 |
| 裁剪阈值10对应的缩放因子 | 0.000314 / 0.000359 | 1 / 1 |
| Speed/volume encoder 梯度余弦 | -0.371 / -0.346 | -0.201 / -0.187 |
| FM/volume encoder 梯度余弦 | 初始化FM encoder梯度为0 | +0.052 / +0.092 |

速度项用倒数变换，预测速度接近0时导数很大；车辆数项用容量归一化 density
再 `log1p`，信号较小。输出 Softplus(beta10) 在320更新后的 occupied-road
平均导数约0.125／0.097，只有0.21%／0.94%的导数低于1e-3，属于衰减，不能
称为整个 volume head 梯度消失。BPR 四次幂项在这些样本的车辆数梯度很小。
此 source 截断，total-travel-time loss 按规则禁用。

## 固定数据 codec 拟合

三组使用同样2条真实 episode、6个固定完整路网帧、同样 pilot320 初始codec
权重；500步、AdamW1e-4、weight decay0.01、clip10，不运行FM。

| 条件 | 起始 occupied vehicle MAE | 100步 | 500步 |
|---|---:|---:|---:|
| 原始 density loss，codec-only | 6.3078 | 6.2810 | 3.1913 |
| 仅改为真实车辆数 log1p，全道路平均 | 6.3078 | 5.5808 | 2.2515 |
| 真实车辆数 log1p，空／有车道路等权 | 6.3078 | 3.2394 | 1.4345 |

不增加数据也能显著减小误差，因此不能仅以数据量不足解释问题。单位控制的
起始 volume-head 梯度约为原始242倍。等权控制进一步减小occupied误差，但
500步时 empty-road MAE 从原始0.021升到0.177，总车辆数bias转为+89.4辆，
不能只看 occupied 指标决定新目标。

500步并未逼近零误差，这不是表达能力无限或泛化改善的证明。以上全部是
固定训练样本拟合，不是独立验证集成绩。

## 联合／分开训练的匹配对照

另用同一个 episode 的事故与末期连续序列（start33和125，sequence2），
从同样pilot320权重开始，使用同一批数据、原codec loss、BPR、LR、裁剪和
300步优化。唯一差别是是否加入正常的两步FM／prior训练。

| 条件 | 300步 occupied vehicle MAE | Speed MAE (m/s) | 耗时 |
|---|---:|---:|---:|
| Codec-only，原loss | 3.0399 | 0.6987 | 2.97秒 |
| 联合codec+FM，原loss | 3.0486 | 0.7005 | 163.67秒 |

本对照分开／联合的车辆数拟合差异约0.29%，没有显示FM是主要干扰源；codec-only
明显更便宜。该单batch控制不能保证所有多步训练阶段均无冲突，但与梯度探针一致。

## 建议的训练顺序

1. 首先修codec监督尺度：车辆数loss用 `log1p(pred_density * capacity)` 与
   `log1p(observed_count)`，同时监控空路假阳性、占用路误差与全网总量偏差。
   输出仍保持原normalized-density接口；speed/volume有效梯度需重新检查。
2. 单独预训练codec，依据 held-out 重建指标验收，不能只依据总loss。
3. 编码器和解码器固定后训练事件条件、图prior和FM主体，使latent目标稳定。
4. 最后再评估小学习率联合微调，并持续检查codec重建是否退化。

这是一项依据诊断的建议，尚未改动主线的默认训练阶段或生产loss。未来新增车辆
的cohort标签讨论与这里的同帧codec监督问题分别验证，不在本轮同时改变。

## 证据文件

目录：`outputs/ingolstadt_codec_diagnosis_20261002/`。

- `gradients.json`：逐项loss、参数block梯度、余弦及独立标签审计。
- `codec_result.json`：三组固定数据控制、相同初始hash与非codec参数不变断言。
- `joint_vs_codec.json`：同数据／同loss／同预算的分开与联合对照。
- `codec_checkpoint_*.pt`：仅用于诊断的权重，不替代生产 checkpoint。

新增诊断工具和相关测试通过，编译与 `git diff --check` 通过。结果为单次seed的
局部诊断，不宣称统计显著性、全数据收敛或交通性能改善。
