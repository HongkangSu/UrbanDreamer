# 无事故 Ingolstadt 新数据预训练（2026-10-08）

用户要求使用昨晚的无事故 Ingolstadt 数据预训练一版世界模型并检查加速。本版沿用现有结构和速度标签，作为数据重训对照；单独报告持续慢队列误差。输出根目录 outputs/ingolstadt_noincident_pretrain_20261008。

## 冻结数据与隔离

数据源 outputs/city_baselines_noincident_prior_penalty_5000ep_20261007/ingolstadt_custom。09:58:37UTC按已提交checkpoint/progress冻结12180个训练NPZ；排除IQL一个未提交文件。全部元数据符合 meso/libsumo、event/1秒观察、30秒窗口、无事故、1800秒Poisson需求和相同输入哈希。原始ZIP20.80GB，成员解压约604GB；按解压trajectory内容SHA去重5文件，12175独立轨迹。4198个需求身份，按需求身份分割避免不同方法间泄漏。

训练10341条轨迹、1052731窗口、281093334事件；留出1834条、187223窗口、50247263事件。codec预训练分方法均衡选200训练+40留出，属于上述固定划分。初始版本每epoch1032049合法L3起点。现行单窗版本每epoch1052731个窗口，5epoch共5263655窗口，保留每episode不足batch的尾部。

## 已完成编解码训练

相同架构新初始化，GPU0，batch64，FP32，sparse图运算。4500更新、约615秒通过原始与hybrid双gate。40留出中的持续慢队列（两边界均至少5车、速度<=2m/s）MAE旧1.43485→新1.15523m/s，有车整体速度MAE旧.85806→新.88140；hybrid总量相对误差旧.32077→新.08809。原5个特定极端队列仍有5–7m/s的oracle高估；不能声称decoder问题已解决，详见codec_diagnostics/REPORT.md。

对完整1834留出重新验证：原始计数MAE全路段.034540、有车.292371、总量相对误差.040353、全部速度MAE.224004m/s；hybrid分别.023456、.314541、.057489、.184901m/s。原有阈值全部满足，随后冻结codec开始FM。

## 加速与实际配置

- compact缓存：7.276GB，12进程严格构建12180源文件，约1584秒；避免每epoch解码巨大的车辆JSON。
- FlowCache命中时改为stat+ZIP中央目录签名检查，首次或变化时验证解压内容SHA；保留2episode上限与无损重压缩兼容性。
- 通过显式数量参数适配新语料，GPU0/1设备编号与allocator对应，不放松源数据或codec证明检查。
- GPU0、BF16 dynamics/readout、FP32codec/参数/Adam，batch32、accum1、L3、space64/time65536/MLP65536，保留梯度重算；allocator上限18%。
- B16/32/48/64、重算开关和两张GPU均做私有无更新测试。B64在18%和22%显存上限下OOM，关闭重算也OOM；B48没有稳定提速。GPU1同profile有约2倍共享负载波动，最终选GPU0 B32。没有把旧attention优化当成新收益。
- GPU0 B32缓存短测102.78序列/秒，完整5epoch理论下限13.95小时；实际早期约80–95序列/秒且会波动，暂估16–20小时。未减少数据或epoch以伪称加速。

## 现行 FM 训练与 Euler 生成设置

用户要求默认保留 Euler 并增加积分步数。17份新配置显式使用 `fm_solver="euler"`、`fm_steps=100`，在流时间[0,1]上步长0.01，每次30秒物理转移生成调用100次速度场；条件编码复用，噪声只在起点采样。论文 [Simulation-Free Schrödinger Bridges via Score and Flow Matching](https://proceedings.mlr.press/v238/tong24a/tong24a.pdf) 附录F.1采用100步Euler作为ODE采样设置（部分高斯实验20步）。步数不是全领域统一标准。低层缺字段的历史2步Euler默认保留以兼容旧checkpoint；中点法保留为显式可选。

新训练模式 `single_step`、初始化序列长度1。输入包含上一边界全部已出现车辆，下一边界车辆数标签仅剔除本窗口新出发车辆。速度场训练仍采用随机流时间的标准FM回归，默认observed-endpoint辅助读出不执行ODE。生成100步并不意味着每个训练batch执行100次前向。可选 `recursive_3step` 连续滚动物理三步，默认关闭；该模式需要seq>=3/horizon>=3及完整车辆/到达元数据，跟踪起点车辆集合以避免已剔除出生车重新进入后续标签，并保留早期生成终点梯度。

CUDA0、B32、BF16、L1的实际完整Adam更新，静态整批编译中位数0.15926秒，eager为0.22629秒，约1.421倍；峰值allocated1089MiB。loss差0.00190%，梯度cosine0.9999503，全部有限。B23/B31尾批确实回退eager、不新增编译图，重复更新后参数与Adam状态均有限。保留非有限梯度拦截。详见 `core_speed/compile/single_step_static.json`。L3独立窗口改L1减少重复监督预算，不能把这部分算作编译收益。

## 运行与验收

supervisor/run.py使用冻结source/代码；world_model.status.json记录每30秒训练心跳，每500更新或5分钟保存可恢复checkpoint。目标仅离线预训练，完成5epoch后停止于ONLINE_TRAINING的0轮边界，随后自动做分方法留出预测检查，不自动替换正在运行的主线。

旧预训练12:03:36UTC在第14001个完整优化器更新checkpoint暂停（390343个L3起点，epoch0）。新输出 `world_model_single_step/` 明确迁移全部模型、Adam状态、RNG和累计更新数，L1覆盖计数/采样器从零开始；旧进度、配置和checkpoint冻结在 `single_step_migration.json` 及其引用文件，旧输出保留。12:32UTC启动 `supervisor_single_step/run.py`，使用 `source_single_step/` 冻结源码，恢复后对1834条留出重新验证codec，不复用错误的L3数据身份。新5epoch任务沿用全部10341条训练轨迹。当前在线主线旧任务使用其冻结的2步Euler配置，新配置会用于新模型后续生成和新运行。

10:46UTC开始FM；保存的第501次更新checkpoint严格加载成功：13856序列，codec_updates4500、codec_ready1、epoch_cursor0；actor/V/Q优化器为空、online replay为空。pretrain_start_verification.json保存证明。相关92项测试、Python编译、git diff --check通过。早期loss下降只表明优化在运行，不证明真实交通性能提升。

新版本启动验证 `single_step_start_verification.json`：第15002累计更新checkpoint严格恢复成功，新L1覆盖28571窗口，codec ready/frozen，actor/V/Q优化器和online replay仍为空。在实际853路段留出窗口上执行Euler100，恰好100次速度场调用，latent和decoder结果全部有限。CPU诊断耗时约19.56秒，不能作为GPU想象吞吐量。

12:36:23–12:39:54UTC训练心跳覆盖约211秒，访问窗口50364，累计更新15778；按区间样本差计算约238.54窗口/秒。若该速率保持，剩余5epoch覆盖约6.07小时；这是早期共享GPU负载下的估计，不是完成承诺，也不包含训练后Euler100留出预测评估。旧任务的16–20小时估计不适用于新L1预算。

验证：本次相关284项通过，随后真实Euler100测试9项、恢复/模式补充62项通过。完整测试集1443通过、5失败；失败全部位于 `test_baseline_repaired_protocol.py`，旧测试要求IPPO/MAPPO/IQL/Mamba拒绝 `--no-global-reward-only`，而当前baseline工具允许该显式模式；其mock绕过配置加载，未进入FM路径。未为本次FM调整覆盖这部分既有baseline行为。全Python编译与 `git diff --check` 通过，环境没有ruff/mypy/pyflakes，未安装新的检查依赖。

## 预训练完成后自动在线训练

用户已授权自动接入2000轮在线actor–critic。`online_queue/`中的独立watcher已启动，PID2776167；状态记录为 `online_queue/status.json`，启动参数为 `online_queue/launch.json`。执行源码冻结在 `source_online/`，135个Python文件的SHA位于 `source_online/source_online_manifest.json`。等待期绑定源码manifest和预训练run_config哈希，不修改正在运行的pretrainer。

watcher每30秒检查：五个初始化epoch全部完成、`pretrain_summary.json`为pretrained、最终checkpoint为ONLINE_TRAINING/0、原始及hybrid codec证明仍有效、尚无在线数据/在线优化器更新、原pretrainer已退出。满足条件后将最终状态原样冻结到 `online_seed/`，只启用经过验证的FP32 imagination编译并更新配置指纹；模型、Adam、RNG、replay和计数严格一致。当前预训练的独立留出预测评估继续由原supervisor执行，不作为未经用户要求的额外性能审批门槛。

在线输出为 `online_actor_critic_2000/`。先对三个固定需求做确定性初始评估，然后进行2000轮SUMO采集、世界模型在线更新与想象actor/V/事件Q更新，每20轮确定性评估。预算保持每轮4次WM更新、16个imagination pass、每pass8条轨迹、最多8个宏观窗口，Euler100和3份噪声不变。维持无事故meso/libsumo、event1秒观察、WM30秒、prior0及population entropy；个体超时惩罚保留。GPU0/1自动选择有至少3GiB空闲显存的设备，首次选定后恢复沿用同设备，allocator上限25%。

文件锁和PID启动时间检查防止重复启动；中断从在线逐episode checkpoint恢复，最多自动启动3次，耗尽重试或完整性错误记录明确状态。队列只管理新输出，不停止既有基线或旧在线实验。当前等待阶段不会提前创建seed或在线输出。新的queue和GPU设备选择等相关142项测试通过，Python编译和diff检查通过。

完整计划位于 `online_handoff_plan.json`。截至14:22UTC，预训练已完成1个epoch，累计访问1366130窗口，处于第2个epoch；队列已实测为waiting_for_pretrain。
