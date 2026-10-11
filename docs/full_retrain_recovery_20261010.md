# 全量重训恢复与新增数据

2026-10-09 19:20 UTC，原数据构建因冻结清单中的prior/IQL原档缺失而退出，总控制器随后停止，预训练未开始。2026-10-10核对发现历史prior源目录12,181档、原预训练系列962档和复用的旧compact目录已不在原路径，共13,143个原冻结源无法恢复。原失败状态和日志保存在failures/20261009_source_loss及原build.log中，不把这一期间报告为训练进度。

## 恢复和追加

保留下来的23,303份有效compact receipt及1,842份现存未构建原档全部复核/补建，形成25,145个源别名、25,140个去重轨迹。用户随后明确要求加入新增数据，于2026-10-10 03:37:24.544606 UTC冻结8,278条新增候选；其中4,062条兼容Ingolstadt无事故meso/event/1秒观察/30秒窗口，另4,216条均属于其他城市，未导入。

最终数据为29,207个源别名、29,202条去重轨迹，训练24,818条/2,605,729窗口，验证4,384条/460,640窗口；五个重复源保留审计。旧分组不改变，新需求组按同一确定性15%留出规则赋值。无法恢复的13,143个源明确列入recovery_inventory/recovery_audit，不合成原始档案，也不将之后追加文件混入本次已冻结清单。

全部compact文件重新核对解压内容SHA256，原档核对成员签名/stamp；新增、重建、stamp变化原档重算完整解压SHA。29,207个可用原档均hardlink封存在data/source_archives，使用原路径身份+内容hash命名，重复源别名保持独立。删除原目录不会删除仍被封存引用的inode；没有额外复制数据块。最终manifest身份5254f01c7863d3e8524d4cf118983fae1bfbe755832ece48380acbdd11eab443。

旧完整WM源checkpoint不可读取，但新运行inputs/calibration_latest.pt已事先冻结，SHA4fbc4e1cb0b569a6dfa0eabb602149ddf69643c074219ea8053e29c0a363b84c。归档绑定严格核对该权重SHA、frozenencoder参数hash、当前网络和物理配置及训练/验证分组。明确记录source_checkpoint_revalidated=False：仅使用已批准的codec，不声称重新校验不可读取的旧WM，更不继承旧dynamics/actor/critic。

## 启动与显存

总控制器和data monitor已恢复。第一次自动估算选128，但共享GPU空闲仅约3GiB，128实测reserved3.6GiB不适合。尚无参数更新时停止并归档该初始化attempt，增加实时显存约束：计划比较reserved峰值+768MiB余量，并在严格恢复时使用已经冻结的原显存决策，避免计划随GPU波动改变。

当前使用batch64、BF16、single_step seq1、decoder持续训练、Euler100想象、双噪声、16rollout/4次更新。一个完整epoch共49,636个microbatch；预计含数据加载/验证/保存约4.56小时，两个完整epoch超出8小时，因此按用户时间要求冻结为一epoch。在线目标2000轮，仍由控制器在预训练完成后自动启动。

有效控制器源码冻结在source_memory_safe_20261010；source_recovery_20261010及原source保留。初始化/恢复工具和日志在launch_command_memory_safe.json、controller.log、pretrain.log、world_model.status.json。

## 真实训练证据

实际FM更新从2026-10-10 05:22:59 UTC开始。核对05:32:54的checkpoint：世界模型和decoder均已完成2,001次更新、104,672个窗口；decoder八个参数张量全已变化，Adam step=2,001，非decoder codec参数逐位不变。Actor/V/Q/world_model_online/decoder_online优化器仍为空，符合新预训练阶段。全部世界模型张量有限，checkpoint可加载；累计更新、数据游标与decoder计数一致。

首条训练记录FM loss0.038632，后续单组约0.0007–0.003，样本不同存在波动；这是训练loss，不是独立预测准确性证明。按前后实际记录窗口数/墙钟时间约176.4窗口/秒，剩余训练约3.9小时，之后执行结束验证。GPU1实测利用率90%，reserved约2.18GiB；并行其他实验会使吞吐波动。

训练前完整留出codec验证：有车车辆MAE0.1791辆/路段、整体速度0.2581m/s、有车速度0.6786m/s、慢路速度0.3595m/s、hybrid慢路0.3462m/s、总量相对误差1.99%。本次总体门槛通过；未重新证明最差局部路段全部达标。训练更新后旧验证proof按正常逻辑失效，epoch结束再验证。

原始证据restart_training_verification.json、data/recovery_dataset_smoke.json、data/recovery_sealed.json与data/calibrated_codec_prerequisites.json。恢复/扩展8项、归档校准及runner66项、控制器12项和整合111项检查通过（有重叠，不相加）；Python编译与git diff --check通过。
