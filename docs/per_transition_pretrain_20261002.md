# 逐窗口车辆数标签与完整预训练吞吐修正

2026-10-02，Ingolstadt。用户明确：输入为完整上一观测状态，包括上一窗口新出发车辆；下一观测的车辆数标签仅排除本次窗口新出发车辆。此前固定整段序列起始C0的理解错误，旧任务PID1865631已停止，保留所有旧checkpoint与源数据。

## 正确的数据定义

对每个窗口重新选择在其起点已经观察到出发、尚未到达的车辆集合C_t。模型输入X_t是完整真实观测；目标Y_{t+1}保留真实下一观测全部字段，仅以C_t成员在终点外部道路上的车辆数替换volume。内部连接中的成员仍属于C_t，在驶出后可进入道路标签；到达车辆不进入终点计数。入口departure_counts不能直接从终点道路减去，因为新出发车辆会在窗口中移动。

A在0秒已存在，B在30秒首次观察到，C在60秒首次观察到：第一步输入计A，终点标签计A；第二步输入计A+B，终点标签计A+B。B只在出生的那次标签中被排除，下次重新纳入。所有原始actions、action events、departure、context和exposure保持不变。只改用户指定的车辆数标签。

主线使用`previous_state_excluding_current_births`。批次保留原始`flow`，另提供`transition_target_flow[B,T,E,8]`及逐窗口`transition_cohort_count[B,T]`。compact cache原始文件无须重建；派生数据契约已更新，拒绝旧标签的采样续训身份。新完整checkpoint为schema9、replay为schema7；旧固定C0 dynamics不能严格续训。

训练每个窗口使用完整真实起点，展平B×H作独立FM监督。图先验与CFM速度损失形式保留，目标detach；仅供exposure辅助readout使用的Euler/Picard终点求解不需要梯度。readout前态也是该窗口完整真实编码。H=3现在表示三个各自30秒的窗口目标，不是从单一初态连续递归90秒。历史recursive helper保留供其他实验。

Codec仍监督其输入状态自身的车辆数。完整验证原始状态覆盖63,575有效帧；修正终点分布验证覆盖7,800个代表性帧（520episode×15），速度均值使用真实下一边界，无后续边界的最后终点不伪造速度标签。初始化只从旧任务抽取6,500步codec和其optimizer；dynamics、actor、V、Q与FMoptimizer重新初始化，并在新目标分布上重新验证codec。

## 速度和显存的实测根因

旧任务确实在cuda:1运行，不是CPU训练。世界模型8,218,422参数，FP32权重31.35MiB；dynamics 8,173,328参数，权重31.18MiB；Adam状态约62.35MiB；codec只有36,229参数（0.138MiB），FM冻结。其他actor/V/Q总权重约0.85MiB。静态buffers约9.86MiB，CUDA上下文及其他分配约316MiB。显存并不由参数量单独决定：attention/MLP激活、batch、计算分块和activation checkpointing才是本任务主要内存变量。

旧初始配置空间attention分块1、时间attention分块256，产生大量细碎算子。相同B4/H3测试中，旧分块attention forward/backward分别1272/318次，大分块48/12次；addmm从2723降至179。两者attention GPU实际算时接近，但调度及Python循环使旧端到端前后向0.617秒、大分块0.353秒，纯分块优化1.75倍。

| 修正后的独立窗口目标，H3 | 前后向中位秒 | 序列/秒 | CUDA分配峰值MiB |
|---|---:|---:|---:|
| B2，小分块，activation checkpoint开 | 0.350 | 5.72 | 346 |
| B4，小分块，checkpoint开 | 0.617 | 6.48 | 592 |
| B4，大分块，checkpoint开 | 0.353 | 11.32 | 1060 |
| B2，大分块，checkpoint关 | 0.178 | 11.25 | 1482 |

大分块为space16/time32768/mlp16384。B4 checkpoint关闭、B8 checkpoint开启分别在单独进程的2.35GiB allocator预算内OOM，因此没有直接选择更大batch或关闭重算。推荐B4、累积2保持有效batch8，保留checkpoint。缓存命中数据整理约18毫秒、已打包标签校验约1毫秒，模型约353毫秒，数据准备不是主要瓶颈。

计时是共享RTX3090上的1次warmup和3次测量，同一缓存episode，无optimizerstep；它不能替代完整任务吞吐，也不建立codec可用性证明。新标签的独立训练与无梯度sample solve本身比旧递归目标更快，这部分不能归功于batch调优。完整benchmark、参数计数及profiler见`outputs/per_transition_throughput_20261002/`和`tools/benchmark_pretrain_throughput.py`。

## 当前运行与验证

新任务输出为`outputs/ingolstadt_per_transition_pretrain_20261002`，只读复用冻结3,440个去重episode的cache：2,920训练、520验证，训练356,341窗口、64,874,952事件、每epoch312,541合法起点，共5epoch。新执行配置B4/累积2/大分块/重算开启，allocator上限8%，显式cuda:1，避免悄悄降级CPU。保留全部源存档与其他采集任务。

完整新目标验证已通过：6500步codec在520episode的原始状态有车路MAE0.287882、全路0.036038、空路0.000833、总量相对0.027944、速度0.164271；新标签终点分布相应0.292228、0.031893、0.000777、0.035053、0.155629。所有阈值保持不变。

真实任务于17:42:39 UTC产生第一个FM更新、schema9 PRETRAIN checkpoint和8条序列进度。PID1920366、tmux ud_per_transition_pretrain_20261002，codec冻结，dynamics从新随机参数训练。真实GPUcheckpoint在独立CPU进程严格加载成功，标签、数据游标及optimizer均正确；原始前态含上一窗口新生车，终点只排除本窗新生。

稳定H3生产心跳从2026-10-02 17:44:10 UTC到2026-10-02 17:53:48 UTC共578秒，处理5829条序列，平均**10.08序列/秒**，比旧约2.19序列/秒快**4.61倍**。该总提升同时包含必要标签／独立窗口训练改变，不能全部归因于batch；同目标执行分块/batch短测独立提升约1.98倍。数据准备中位0.040秒/组，模型更新中位0.751秒/组。GPU更新后活跃tensor约158MiB，reserved约1702MiB，NVIDIA进程约2018MiB；中间激活释放后进入缓存，显存快照不能代表前后向峰值。

最终完整回归**704 passed，361.83秒**；Python编译和git diff --check通过。首次完整回归仅旧__all__断言未包含新增FM helper，更新该API契约后专项128passed、完整重跑704全过。纯分块／重算的loss、预测及所有参数梯度等价测试通过，未新增依赖。

实时状态`outputs/ingolstadt_per_transition_pretrain_20261002.status.json`；checkpoint、严格匹配配置run_config.toml及实测throughput_verification.json位于新任务目录。完整5epoch仍运行，不自动开始online。


## 4–8小时预算下的第二轮加速

用户明确要求4–8小时，并允许增加batch。正确标签的schema9任务在17:57:40保存checkpoint，18:00左右暂停；该checkpoint的1,225次WM更新、6,500次codec更新、所有模型／optimizer／RNG已通过CPU迁移验证保留。下一运行为schema10、BF16训练、FP32参数／梯度／Adam，codec、留出评估和想象推断保持FP32。

主要优化与验证：

- 时间attention只有一个时刻时，softmax恒1，精确退化为value投影加output投影。BF16通用FlashAttention backward对极短长度申请大padding workspace，B4约1.26GiB。等价shortcut消除该工作区；T>1仍保留因果attention。FP32输出／输入和参数梯度回归通过。
- 移除空间attention的全True冗余mask，允许更优SDPA分派；单独影响很小（约40MiB内存），不能把主要收益归于mask。
- BF16只用于dynamics及辅助readout。事件feature_sum保持FP32，MLP BF16输出先cast回FP32后index_add，整数count不变。已验证固定RNG的FMloss相对差约0.39%、FMvelocity相对L2约0.68%、dynamics梯度cosine0.997469／relativeL2约7.20%、范数差小于1%，全部有限；不是逐bit等价。
- exposure readout在已经编码的真实终点latent上学习，不再为了这个辅助头额外运行两次Euler Transformer。相同noise和tau下，FM／prior loss及dynamics梯度完全不变；生成终点的readout误差仍应单独留出验证。
- FM只整理实际三个监督窗口，不构造无用车辆snapshot字典，B4数据准备25.96ms降到2.86ms。所有canonical起点、RNG与原始字段保持；codec流不受影响。

B16、BF16、observed endpoint、prefix3、chunk64/65536/65536、checkpointON：私有模型前后向中位0.196秒，prefix数据整理约0.042秒，综合约67.3sequence/s（旧312,541start预算的5epoch约6.45小时）。生产使用新sequence_length3，合法起点350,501／epoch，增加12.15%以覆盖原先完全漏掉的37,960尾部窗口；5个完整epoch1,752,505起点，按上述短测约7.23小时。真实FM-only元数据优化进一步降低CPU成本，但硬件共享负载、Adam和checkpoint开销应通过实际心跳重新估计。B8备选显存更稳，理论接近8小时。

新运行输出`outputs/ingolstadt_bf16_pretrain_20261002`，迁移schema9正确标签模型而非重训。L3改变来源契约，因此重置epoch采样及统计为0，旧计数保存在迁移记录中；WM更新及Adam/RNG继续，codec readiness必须在真实L3留出分布上重验。保持5epoch，不删数据。

当前GPU0剩1MiB、GPU1剩212MiB，无法启动新GPU训练。未停止任何其他任务。持久等待器tmux`ud_bf16_pretrain_20261002`，PID1951614，每30秒检查GPU1：空闲>=2304MiB选B16，>=1536MiB选B8；不CPUfallback、不改源存档，不提前污染新训练输出目录。资源状态及精确命令位于`outputs/ingolstadt_bf16_pretrain_20261002_supervisor/`。这里的4–8小时不包含未知排队时间，不能表述为已经开始BF16实际训练。

完整测试首次为730passed、2项CUDA资源检测因GPU满导致初始化OOM；改进测试的资源检查后专项66passed/2skipped，完整回归最终为**730passed、2skipped，359.15秒**（CUDA显存被外部任务占满，GPU用例按资源不足跳过）。核心模型／标签／cache／training196passed，迁移／workflow168passed，shipped L3配置82passed，编译和diffcheck通过。未新增依赖。


## 2026-10-03完成与loss曲线

资源等待器在2026-10-02 19:43 UTC发现GPU1空闲4686MiB并自动启动B16；无需停止其他任务。实际FM从19:44:24到2026-10-03 02:20:07，5个完整L3 epoch耗时6小时35分43秒；处理1,752,505序列、112,880次新optimizer更新（总WM更新114,105，含迁移前1,225），平均73.81sequence/s。checkpoint schema10最终phaseONLINE_TRAINING/cursor0，状态pretrained，尚未在线训练。

世界模型最后批次loss3.38809e-5（FM2.60864e-5+0.1×prior7.79450e-5），首批1.25496e-3。按epoch采样批次中位数：4.68894e-4、2.69623e-4、1.69977e-4、1.20923e-4、8.24689e-5；首末epoch中位下降82.41%。Exposure readout MSE中位：7.12769e-4、3.33111e-4、3.29782e-4、2.49832e-4、2.51160e-4，第二epoch后主要趋于平台。所有心跳loss有限。

789个约30秒采样点与末次update已绘图，原始细线、最近31点中位粗线、log纵轴与epoch竖线；CSV/PNG/PDF/统计在run的plots目录，脚本tools/plot_pretrain_loss.py。不是逐update完整loss序列，不能从loss下降直接推出交通性能提升。

完成后只读加载最终checkpoint，以FP32在20个代表性held-out episode、40条序列上作3个独立30秒窗口的预测评价（每窗3个随机样本，seed31415，来自与train分离的需求身份）。首窗已有车辆数MAE0.6333辆，保持原状态基线0.8790；有车路速度MAE0.7694m/s，基线0.9442。第三窗从完整60秒真实状态重新预测30秒，车辆数0.6136vs0.8278、速度0.8033vs0.9370，不能称90秒递归forecast。Exposure readout预测MAE约0.028而persistence约0.0024，辅助读出仍未胜过该基线，应单独诊断；它未参与当前actor实际车辆计费。完整机器结果heldout_prediction_evaluation.json。这是代表性抽样与事实动作条件下预测评估，不是全520episode遍历或实际交通策略评估。

attention模块编译CPUproof完成：输出／输入及所有参数梯度近似一致，参数对象和state_dict键保持，兼容activation checkpoint；小CPU例约1.4倍，未自动纳入已完成的GPU训练，不能当作GPU速度增益。
