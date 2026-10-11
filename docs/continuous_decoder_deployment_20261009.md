# 校准decoder上线与持续监督训练

用户明确要求直接部署校准decoder，并在离线世界模型预训练、在线微调中都持续训练decoder。新的线上输出为`outputs/ingolstadt_noincident_pretrain_20261008/online_actor_critic_2000_decoder_continuous`，由`online_decoder_continuous_supervisor`继续2000轮目标。

## 模型与训练状态

从已完成244轮的BF16/8批主线完整checkpoint迁移。仅替换独立校准产物的`flow_projector.decoder_*`八个参数张量；encoder、全局读出头、转移模型、actor/V/Q、既有优化器、replay、随机状态和全局进度保留。逐张量校验候选的非decoder参数与源模型一致。源运行和校准checkpoint均保留，冻结源码在`source_online_decoder_continuous`。

部署的候选来自20682次独立校准更新，仍保留candidate_unready原始质量记录。用户明确允许部署，因此新配置`decoder_allow_unready=true`表示运行授权，`codec_ready`仍为0，不伪造质量验收。旧验证证明失效，原值保存在冻结的源checkpoint和校准产物中。

decoder有两套独立Adam：离线预训练`decoder_pretrain`学习率1e-4，在线微调`decoder_online`学习率1e-5；本次两个新优化器采用空状态初始化，原actor/V/Q等优化器状态完整保留，不把校准AdamW状态强行当成Adam。参数保持可训练，加载checkpoint和完成更新后不会再次冻结。Encoder保持冻结，真实监督的latent先detach；不改变已训练的latent坐标定义。

每轮四次真实数据世界模型更新时同步优化decoder，使用原车辆数/总量/速度/其他通道监督，加上有车、慢路组内速度损失各权重1。沿用真实相邻边界速度标签，不伪造终点之后的下一帧。纯想象阶段不使用未来真实标签，也不更新decoder。

所有17份主线预设显式开启预训练与在线decoder训练；低层旧配置默认保持关闭，避免静默改变旧checkpoint语义。预设默认不放宽质量门槛：通过初始验证的模型可在持续训练中使用初始准入证明，质量结果随参数更新失效并独立记录；本次未过最差局部门槛的候选通过显式授权部署。

想象配置保留BF16、Euler100、两个噪声、每批8条rollout、最多8个30秒窗口、每轮8次actor/V/Q更新，时间截断关闭。

## 运行证据

首个接续episode为第245轮（零基244）。四次真实WM+decoder更新耗时1.275、0.205、0.204、0.178秒，第一项含启动开销；decoder梯度范数0.887、0.758、0.466、0.134，均有限非零。

首轮八批想象+AC分别29.585、9.355、9.477、10.489、11.779、9.914、11.056、8.815秒。首批包含worker启动，其余七批平均10.126秒；整个想象阶段100.47秒，episode训练117.59秒，不含checkpoint写盘和定期评估。对比接入前最近40批平均9.992秒，目前没有明显单批性能退化证据；GPU共享，不能将约0.13秒差视为严格新增开销。

第245轮checkpoint实测decoder累计4次更新，八个decoder张量均已变化；所有非decoder的codec张量逐位不变，Adam每个已用参数step=4。实际推理profile确认BF16、full_batch=16；actor/V/Q各8次更新。证据存于父目录`continuous_decoder_verification.json`和当前supervisor/run.log。迁移复制的pilot_metrics.jsonl保留旧源未提交的部分轨迹日志；本次计时只读取新supervisor日志和完整提交行，避免把旧未提交批次重复计入。

第二个接续episode第246轮也已完整提交：八批均值10.067秒，想象+AC共80.543秒，SUMO内部计时11.164秒，episode训练94.659秒（不含checkpoint写盘/定期评估），actor/V/Q各8次，Q损失0.004786。checkpoint的decoder累计更新8次、Adam step=8，encoder仍逐位未变。与接入前每轮75–84秒、每批约9.99秒处于相近范围，持续decoder训练未显示明显推理减速。固定八批的想象总时长尚未达到一分钟。

## 验证

核心训练/配置/推理/迁移/runner相关208项测试通过，checkpoint/pipeline/codec/旧迁移等相邻测试134项通过（部分重复，不相加）。修复了单次decoder更新指标与累计计数重名的问题，真实两轮训练回归证明累计计数4→8，checkpoint恢复后优化器与RNG一致。迁移测试覆盖encoder漂移、非有限权重、冻结文件篡改、未经允许的配置变更和初始actor状态篡改。Python编译与git diff --check通过，无新依赖。

上述验证说明部署生效且decoder持续更新，不证明交通性能收敛或残余慢路误差已全部消除。
