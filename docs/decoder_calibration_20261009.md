# 冻结 encoder 的完整 decoder 校准

用户要求先校准。正式任务已在GPU0独立启动，源模型为已完成五epoch的无事故Ingolstadt `world_model_single_step` checkpoint。当前2000轮在线任务保留原冻结源码和模型，不做现场热替换，不开始新的完整世界模型预训练。

训练使用全部10341条训练轨迹、1052731个合法窗口，batch64，每epoch保留每episode尾批，共20682更新。预算上限25000，完成一个完整epoch后停止。训练和1834条留出按需求组、路径和trajectory内容身份严格隔离；之前挑选的错误案例都属于留出，不进入校准训练。

冻结input_proj、所有GNN和t_all_head，参数SHA在整个过程严格检查。仅decoder_shared/speed/volume/other可更新，新AdamW学习率1e-4。原车辆数、总量、整体速度和其他通道损失保留；追加有车和均值慢路组内速度损失，各权重1。速度优化仍使用归一化速度，报告转换为m/s。Encoder前向不建立梯度图，旧latent定义保持不变。

当前raw状态配真实下一边界；出生车剔除后的hybrid endpoint配真实再下一边界。末端没有真实following时不构造hybrid速度标签。慢路定义为当前目标车辆数>=5−1e-5、两边界均值速度<=2m/s；persistent另外只作诊断。

完整留出raw和hybrid在训练前、每5000次更新、epoch结束验证。全路段速度<=1.5m/s、有车速度<=1.5m/s、慢路速度<=0.5m/s；每条至少20个慢观测的路段均值最大值<=1m/s。同时单独检验一个帧只有1–5条慢路的局部条件，局部慢路均值<=0.5m/s、合格路段最大均值<=1m/s；无足够覆盖不会误报为零误差通过。车辆数与总量仍使用原门槛。

正式基线验证表明：完整raw留出慢路MAE1.269m/s，局部拥堵MAE2.976m/s，局部最大路段均值5.692m/s；hybrid结果相近。不能再用原all-road平均速度门槛代替局部验收。

执行源码140个Python文件冻结在 `outputs/ingolstadt_noincident_pretrain_20261008/source_decoder_calibration/`；supervisor和日志在 `decoder_calibration_supervisor/`，校准状态/独立checkpoint在 `decoder_calibration/`。checkpoint保存decoder、独立Adam、FrameSampler和RNG以及源码来源/数据身份，原世界模型checkpoint不变。恢复需要相同完整契约。候选仅标记candidate_ready/unready，无自动部署逻辑。

启动已验证真实训练、checkpoint写入和冻结参数哈希。相关163项回归通过、Python编译与git diff --check通过。监督脚本最多重试3次，每次恢复已保存校准进度。独立泛化结果需等完整校准后报告，私有12帧过拟合探针不作为验收。

后续代码支持世界模型离线预训练时保持encoder冻结、decoder用独立监督/Adam持续更新；FM损失在latent空间，单纯解冻decoder并不能训练它。该改动本轮没有启动新的完整WM训练。当前先完成校准，再评价生成状态与真实车辆推进的影响。
