# UrbanDreamer Methods v2: Flow-Matching World Model, HDV Response, and Two-level Actor-Critic

> 本文档根据当前仓库代码重写 UrbanDreamer 的 Methods 叙述。术语上，本文使用 **CAV** 表示可控联网自动驾驶车辆，使用 **HDV** 表示人类驾驶车辆。代码和旧文档中的 `av_*` 字段在本文中均对应 CAV 侧信息，例如 `av_planned_edge_flow` 表示 CAV planned edge pressure。

UrbanDreamer 面向 mixed CAV--HDV traffic routing。系统的核心思想是把可执行的 vehicle-level route decision 与可快速 rollout 的 edge-level world model 解耦：actor 为每辆 active CAV 从候选路径中选择 route，aggregation layer 将这些离散 route choices 投影为 edge-aligned planned pressure，flow-matching world model 再根据 traffic latent、CAV pressure、departure demand 和 HDV response context 预测下一步宏观交通状态。训练阶段采用 centralized training with decentralized execution：actor 执行时仍是单车路由策略，critic 则可以使用 global latent 和 fleet-level 信息做 credit assignment。

当前主线由四个部分组成：

1. **World model**：包含 edge-state encoder、decoder/readouts 和 flow-matching transition；CAV pressure、departure 与 HDV response 作为 graph context 输入，上一 latent state \(Z_t\) 作为 flow source \(x_0\)。
2. **HDV response module**：预测 HDV 对 CAV episode-level pressure pattern 的背景响应。
3. **Actor-critic**：vehicle-level CAV actor，加上 global-level critic 和 agent-level critic 的双层设计。
4. **Training pipeline**：SUMO/RouteRL 数据采集、collection warmup、world-model optimization warmup、offline world model training、imagined actor-critic training 和 simulator-grounded evaluation。

## 1. World Model

UrbanDreamer 的 world model 是一个 flow-matching edge-token dynamics model。当前代码接口仍沿用 `transition_mode="cfm"`、`use_cfm=True`、`cfm_loss` 等历史命名；新版 method 叙述中，这些名字对应的是 state-to-state flow matching，而不是以上一状态为条件、再从 zero/noise 生成下一状态的 formulation。因此新版主线不再把 deterministic dynamics、PCFM 或 Stackelberg equilibrium 作为必需 transition，而是将它们视为保留扩展、约束项或消融接口。

World model 接收的是固定路网 edge graph 上的聚合交通状态。令 \(\mathcal{E}=\{1,\ldots,E\}\) 为 SUMO network 中所有非 internal road edge 的固定集合。每个 decision step 的原始宏观状态为
\[
X_t \in \mathbb{R}^{E \times 8}.
\]
其中前四个通道是动态交通量：
\[
[\mathrm{occupancy},\ \mathrm{speed}/\mathrm{speedLimit},\ \mathrm{vehicle}/\mathrm{capacity},\ \mathrm{halt}/\mathrm{capacity}],
\]
后四个通道是静态路网属性：
\[
[\mathrm{length},\ \mathrm{speedLimit},\ \mathrm{lanes},\ \mathrm{roadType}].
\]

`EdgeRegistry` 负责构造稳定的 edge ordering、静态特征、capacity/free-flow travel time，以及 edge-line graph adjacency。这里的 graph node 是 road edge；如果 SUMO 中 edge \(e_i\) 可以接入 \(e_j\)，则 line graph 中存在 \(i\rightarrow j\) 的邻接。邻接矩阵按行归一化后供 encoder 和 graph-context encoder 使用。

### 1.1 Encoder: Edge-state Tokenization

World-model encoder 对应 `FlowProjector`。它把 \(X_t\) 映射为 edge latent tokens：
\[
Z_t = f_{\mathrm{enc}}(X_t) \in \mathbb{R}^{E \times d_z},
\]
当前主线中 \(d_z=64\)。实现上，`FlowProjector` 先对每条 edge 的 8 维特征做 linear projection，再通过若干 `EdgeGNNLayer` 在 line graph 上传播局部拓扑信息：
\[
h_e^0 = W_x X_t(e),
\]
\[
m_i = \sum_j A_{ij} W_{\mathrm{msg}}h_j,
\quad
\tilde h_i = \mathrm{GELU}(W_{\mathrm{self}}h_i + m_i),
\]
\[
h_i' = \mathrm{LayerNorm}(h_i + \tilde h_i).
\]
最终的 \(h^L\) 即 \(Z_t\)。该 encoder 提供两个关键 inductive biases：第一，所有状态都 edge-aligned，便于与 route pressure 对齐；第二，line-graph message passing 能表达局部 queue propagation、bottleneck spillback 和 route overlap 对邻接道路的影响。

### 1.2 Graph-context Encoder: CAV Pressure, Departure, and HDV Context

World model 的 action interface 不是 per-vehicle route id，而是所有 active CAV route choices 在物理路网上诱导出的 planned pressure field。对当前 step 的 active CAV 集合 \(\mathcal{I}_t^{\mathrm{CAV}}\)，每辆车 \(i\) 有 \(K_i(t)\le K\) 条 current-edge-to-destination candidate routes \(\{P_{i,t}^k\}\)。actor 输出 route distribution \(\pi_\theta(k\mid o_{i,t})\) 后，aggregation layer 将其投影到 edge field：
\[
A_t^{\mathrm{CAV}}(e)
=
\frac{1}{\max(|\mathcal{I}_t^{\mathrm{CAV}}|,1)}
\sum_{i\in\mathcal{I}_t^{\mathrm{CAV}}}\sum_{k=1}^{K_i(t)}
\pi_\theta(k\mid o_{i,t})\,
\mathbf{1}\{e\in P_{i,t}^{k,H}\}.
\]
其中 \(P_{i,t}^{k,H}\) 表示在一个短期 decision interval 内预计进入的 route suffix edge 集合。在 actor 驱动的 online SUMO collection 与 imagined training 中，从 masked policy 随机采样路线，\(\pi_\theta\) 被本次采样得到的 one-hot route 替代；同一个 sampled one-hot 同时用于车辆执行、replay action 与 WM condition。evaluation 单独使用 masked argmax。这里不包括 warmup 的探索性 collection。代码中该 field 保存为 `av_planned_edge_flow`。

Graph-context encoder 对应 `ActionDepartGNNEncoder`，当前 world-model training 只允许 `--action_encoder_type graph_cond`。它接收四类外部转移上下文：
\[
A_t^{\mathrm{CAV}}\in\mathbb{R}^E,\quad
D_t\in\mathbb{R}^E,\quad
H_t^{\mathrm{HDV}}\in\mathbb{R}^E,\quad
C_t\in\mathbb{R}^{d_c}.
\]
\(D_t\) 是 departure schedule 聚合出的 edge-aligned departure field，\(H_t^{\mathrm{HDV}}\) 是 HDV response context，\(C_t\) 是低维上下文，例如 time/load/episode context。`ActionDepartGNNEncoder` 为 CAV pressure 和 departure 分别使用独立 edge GNN branch；当 `transition_hdv_context_mode` 启用时，HDV context 使用第三个独立 branch。每个 branch 都将 edge scalar 与 static edge features 拼接，再沿 line graph message passing。

该 encoder 输出两类 context token：

- **Global ACTION token**：对 CAV pressure branch 和 departure branch 做 mean/max pooling，并与 context MLP 输出拼接；启用 HDV context 时，还会把 HDV branch 的 pooled summary 加入 global token。
- **Edge context tokens**：当 `graph_cond_edge_conditioning` 启用时，将 per-edge CAV/departure/HDV hidden states 拼接后映射到 \(d_{\mathrm{model}}\)，并注入对应 spatial token。这样 transition 不仅看到全局 demand summary，也能看到每条 edge 上的局部 pressure shape。

这一区分很重要：global ACTION token 表示整体负载和时间上下文，edge context tokens 保留“哪些 edge 被 CAV planned pressure、departure demand 和 HDV response 覆盖”的空间结构。它们是 \(v_\phi\) 的外部上下文，不改变 \(Z_t=x_0\) 的 flow source 角色。

### 1.3 Decoder and Readouts

`FlowProjector` 同时包含 decoder/readout heads。它们不是独立 simulator，而是让 latent \(Z_t\) 可解释、可监督，并为 actor-critic 提供 reward/readout signal。

主要 readouts 包括：

- **Dynamic decoder**：`decode(z)` 将 \(Z_t\) 解码为 edge-level dynamic channels。volume 通道通过 softplus 映射为非负值，避免硬截断导致负 logits 无梯度。
- **Policy observation projection**：`project_obs(z)` 或 decoder-derived compact features 可作为 MARL edge observation。当前 full-cycle runner 默认 `--wm_policy_obs_source decoder_volume_tt`，即使用 decoded volume 和 BPR/decoded travel time 相关特征。
- **Route-cost head**：`predict_edge_route_cost(z, free_flow)` 输出 planner-calibrated edge travel time。它用于 candidate route cost、route ranking/distillation 或 reward 信号。
- **Episode \(t_{\mathrm{all}}\) head**：`predict_t_all_from_z(z)` 在启用 `enable_t_all_head` 或 `t_all_supervision_weight>0` 时预测 episode-level mean travel time，作为辅助监督和诊断信号。

World-model training 会根据配置组合 decoder reconstruction、BPR supervision、transition decoded supervision、route ranking/distillation 和 \(t_{\mathrm{all}}\) supervision。核心 transition 由 flow-matching loss 驱动；decoder/readouts 的作用是约束 latent representation，使其对交通速度、流量、路径成本和全局 travel-time 指标可读。

### 1.4 Flow-Matching Transition

Flow-matching transition 对应代码中的 `Dynamics`。新版 method 中，上一状态 \(Z_t\) 不再作为 condition 输入；它是 flow path 的 source \(x_0\)。外部上下文只包含 actor/demand/background 等转移因素，以及可选的更早历史摘要：
\[
\mathcal C_t =
\left(A_t^{\mathrm{CAV}},D_t,C_t,H_t^{\mathrm{HDV}},Z_{<t}^{\mathrm{hist}}\right).
\]
其中 \(Z_{<t}^{\mathrm{hist}}\) 是可选的 temporal context；如果使用它，也只表示 \(t\) 之前的更早状态摘要。\(Z_t\) 本身不放入 \(\mathcal C_t\)，而是严格作为 \(x_0\)。

模型学习 velocity field：
\[
v_\phi(x_\tau,\tau,\mathcal C_t)
\in\mathbb{R}^{E\times d_z}.
\]

每个时间步的 token layout 由以下部分组成：

- spatial tokens：当前 flow interpolation state \(x_\tau\)；
- ACTION token：graph global context；
- FLOW_TIME token：\(\tau\in[0,1]\) 的 MLP embedding；
- REGISTER tokens：learnable memory slots；
- 可选 edge context tokens：加到对应 spatial token 上。

所有 tokens 加入二维正弦位置编码：一维编码 history/rollout time index，另一维编码 token position。随后进入 block-causal transformer。每层包含 space/modality attention、causal time attention 和 MLP residual block。`space_mode="wm_agent"` 允许跨 modality full mixing；`wm_agent_isolated` 则保持 agent token 隔离。当前 world-model 主线关注 world tokens 与 edge context tokens 的时空混合。

训练使用以上一状态为 source 的 state-to-state flow matching。给定一个真实 transition pair \((Z_t,Z_{t+1})\)，定义
\[
x_0=Z_t,\qquad x_1=Z_{t+1},\qquad \tau\sim\mathcal{U}(0,1).
\]
线性 probability path 为
\[
x_\tau
=(1-\tau)x_0+\tau x_1
=(1-\tau)Z_t+\tau Z_{t+1}.
\]
该 path 的真实 velocity 是常量：
\[
u_\tau
=
\frac{d x_\tau}{d\tau}
=
x_1-x_0
=
Z_{t+1}-Z_t.
\]
模型以 \(x_\tau\)、flow time \(\tau\) 和外部上下文 \(\mathcal C_t\) 为输入，预测
\[
\hat u_\phi
=
v_\phi(x_\tau,\tau,\mathcal C_t).
\]
无权重版本的 flow-matching loss 为
\[
\mathcal{L}_{\mathrm{FM}}
=
\mathbb{E}_{(t,\tau)}
\left[
\left\|
v_\phi(x_\tau,\tau,\mathcal C_t)
-
(Z_{t+1}-Z_t)
\right\|_2^2
\right],
\]
这里的范数对所有 edge token 和 latent channel 求平均。实现中还可加入 active/busy edge weighting，使高负载或非空 edge 获得更高监督权重。设 \(w_{t,e}\ge 0\) 是第 \(e\) 条 edge 的权重，则加权版本为
\[
\mathcal{L}_{\mathrm{FM}}^{\mathrm{weighted}}
=
\mathbb{E}_{(t,\tau)}
\left[
\frac{
\sum_{e=1}^{E}
w_{t,e}
\left\|
v_{\phi,e}(x_\tau,\tau,\mathcal C_t)
-
(Z_{t+1,e}-Z_{t,e})
\right\|_2^2
}{
\sum_{e=1}^{E} w_{t,e}+\epsilon
}
\right].
\]
其中 \(w_{t,e}\) 由 `cfm_active_weight`、`cfm_busy_weight`、`cfm_busy_threshold` 等配置控制；这些参数名仍沿用代码里的 `cfm_*` 前缀，但 method 语义是 flow matching。

推理时，单步 transition 从 \(x_{\tau=0}=Z_t\) 初始化，沿 \(\tau:0\rightarrow1\) 解 ODE：
\[
\frac{dx_\tau}{d\tau}
=
v_\phi(x_\tau,\tau,\mathcal C_t),
\qquad
x_{\tau=0}=Z_t.
\]
积分终点作为下一步预测：
\[
\hat Z_{t+1}
=
Z_t+
\int_0^1
v_\phi(x_\tau,\tau,\mathcal C_t)\,d\tau.
\]
实际 sampling 可用 Euler / ODE solver 近似。若使用 \(K\) 个 Euler steps，\(\Delta\tau=1/K\)，则
\[
x^{(0)}=Z_t,\qquad
x^{(k+1)}
=
x^{(k)}
+
\Delta\tau\,
v_\phi(x^{(k)},\tau_k,\mathcal C_t),
\quad
\hat Z_{t+1}=x^{(K)}.
\]
需要强调的是，\(Z_t\) 在新版 method 中只承担 flow source \(x_0\) 的角色，不再作为 condition。actor-critic 中的多步 imagination 是反复调用 one-step flow-matching transition，将 \(\hat Z_{t+1}\) 作为下一步的 source 继续预测，而不是一次性生成整个 future sequence。

代码同步时需要重点修改两类位置：world-model training 中的 `cfm_loss` 应从 batch latent sequence 取 \(Z_t\) 作为 source，而不是采样 noise；inference/sampling 中的 `_init_cfm_state`、`cfm_init_mode` 或等价 one-step sampler 应以当前 latent \(Z_t\) 初始化 flow state，避免再使用 `zero` 作为主线初值。

## 2. HDV Response Module

HDV 在真实 SUMO/RouteRL 中不是静态背景。它们会根据历史 travel time、拥堵经验和 routing adaptation 机制改变路线。如果 imagined rollout 中把 HDV 完全固定，CAV actor 容易低估自身 route pressure 对 HDV 背景交通的外部性；但如果每个 policy-gradient step 都显式求解 HDV equilibrium，训练又会过慢且不稳定。因此 UrbanDreamer 使用 episode-level neural HDV response module 作为主线近似。

### 2.1 Response Variable and Target Construction

HDV response module 的监督信号来自采集数据中的 `hdv_edge_action`。该字段是每个 decision step 的 edge-aligned HDV background distribution，形状为 \((N,E)\)，其中 \(E\) 是路网 edge 数。它不是某个单车 route label，而是“当前 HDV 背景交通在各 edge 上的占用/压力分布”。记原始记录为
\[
h_t^{\mathrm{raw}}\in\mathbb{R}_{\ge0}^{E}.
\]

训练时先根据 `hdv_policy_target_mode` 把 raw HDV edge field 转成监督目标 \(H_t^{\mathrm{tar}}\)：

- `episode_mean` / `episode_response` / `response`：对同一 episode 内的 HDV edge field 求均值，并把同一个 episode-level target 重复到该 episode 的每个时间步。`FlowDataset._episode_mean_aux_target` 会在离线 shard 加载时完成这个处理；online-WM replay 路径则用当前 episode/window 的 `hdv_target_current` 构造等价 target。

进入 loss 前会做非负化和 edge 维归一化：
\[
\tilde H_{t,e}
=
\begin{cases}
\dfrac{\max(H_{t,e}^{\mathrm{tar}},0)}
{\sum_{e'}\max(H_{t,e'}^{\mathrm{tar}},0)},
&
\sum_{e'}\max(H_{t,e'}^{\mathrm{tar}},0)>0,\\[1.1em]
\dfrac{1}{E},
&
\text{otherwise}.
\end{cases}
\]
这样 HDV head 学到的是 edge distribution，而不是未归一化车流计数。

### 2.2 Prediction Head and Context

`Dynamics` 在 flow-matching forward 中得到每条 edge 的 transformer 输出 \(s_{t,e}\in\mathbb{R}^{d_{\mathrm{model}}}\)，然后用 edge-level `hdv_policy_head` 独立作用在每条 edge token 上：
\[
r_{t,e}
=
\mathrm{MLP}_{\psi}(s_{t,e}),
\quad
r_t\in\mathbb{R}^{E}.
\]
当前代码要求 `hdv_policy_mode="edge"`，并且 `hdv_policy_dim == n_edges`。`hdv_policy_head` 的实现是
\[
\mathrm{Linear}(d_{\mathrm{model}},d_{\mathrm{model}})
\rightarrow
\mathrm{GELU}
\rightarrow
\mathrm{Linear}(d_{\mathrm{model}},1),
\]
对所有 edge token 共享参数。预测分布由 edge 维 softmax 得到：
\[
\hat H_t^{\mathrm{HDV}}
=
\mathrm{softmax}(r_t),
\quad
\hat H_t^{\mathrm{HDV}}\in\Delta^{E-1}.
\]

HDV head 使用的 CAV-pressure context 可以和 transition flow matching 使用的外部转移上下文不同。Transition 主损失仍使用 causal/per-step CAV pressure \(A_t^{\mathrm{CAV}}\)、departure、context 和可选 HDV context 来预测 velocity field；HDV response head 则通过 `hdv_policy_action_mode` 单独构造 `hdv_actions`：

- `step` / `legacy`：不额外构造 `hdv_actions`，HDV logits 和 transition 共用当前 model actions；
- `window_mean`：用当前 sampled window 内的平均 CAV pressure 作为 HDV head 的 context；若 departure/context 可用，也使用 window mean；
- `episode_mean` / `episode_response` / `response`：用 episode-level CAV pressure context。离线训练时可直接使用 `FlowDataset._episode_mean_action_condition` 产生的 episode mean CAV pressure；为了匹配 MARL 推理接口，departure/context 在该模式下置空。

因此 episode-response 语义下，HDV head 学习的映射可以写成
\[
\hat H^{\mathrm{HDV}}
=
g_\psi(\bar A^{\mathrm{CAV}}, Z_{\mathrm{ctx}}),
\quad
\bar A^{\mathrm{CAV}}
=
\frac{1}{T}\sum_{t=1}^{T} A_t^{\mathrm{CAV}},
\]
其中 \(Z_{\mathrm{ctx}}\) 是 episode 起点或查询时的 latent context。代码接口 `predict_episode_hdv_response` 允许显式传入 `z_context`；没有可用 latent context 时，它使用 zero latent query，并用 `episode_av_pressure` 构造一个单步 action context，最后返回 softmax 后的 edge distribution。

当 `hdv_actions` 不为空时，`cfm_loss` 会把 transition velocity 和 HDV response 拆成两次 `Dynamics` 调用：一次使用 transition actions 预测 flow-matching velocity，另一次使用 HDV-specific actions 预测 `hdv_logits`。这样可以避免 HDV head 被 causally shifted per-step transition action 语义污染；它监督的是 CAV policy pressure 对 HDV background 的慢时间尺度响应。这里的 `cfm_loss` 是代码历史函数名，method 语义仍是上一状态作为 \(x_0\) 的 flow matching。

### 2.3 HDV Response Loss

HDV response 使用交叉熵拟合。给定 logits \(r_t\) 和归一化 target \(\tilde H_t\)，损失为
\[
\mathcal{L}_{\mathrm{HDV}}
=
-
\frac{1}{BT}
\sum_{b=1}^{B}
\sum_{t=1}^{T}
\sum_{e=1}^{E}
\tilde H_{b,t,e}
\log
\mathrm{softmax}(r_{b,t})_e.
\]

总 world-model loss 中，HDV response 是辅助项：
\[
\mathcal{L}_{\mathrm{WM}}
=
\mathcal{L}_{\mathrm{FM}}
+
\lambda_{\mathrm{HDV}}\mathcal{L}_{\mathrm{HDV}}
+
\cdots,
\]


### 2.4 Transition-time Injection

HDV response 不直接混入 CAV action，而是作为独立 edge context 注入 transition 的外部上下文。

把 HDV auxiliary background 作为 separate graph context branch 输入 `ActionDepartGNNEncoder`。这是新版主线更推荐的语义，因为它保持 controllable CAV pressure 与 adaptive HDV response 的可解释分离。

因此 transition 不是写成“以上一状态为条件生成下一状态”的条件分布，而是写成带外部上下文的 velocity field：
\[
\mathcal C_t=
\left(A_t^{\mathrm{CAV}},D_t,C_t,H_t^{\mathrm{HDV}}\right),
\qquad
v_\phi(x_\tau,\tau,\mathcal C_t).
\]
这里的初值仍然是 \(x_0=Z_t\)，目标是沿
\[
x_\tau=(1-\tau)Z_t+\tau Z_{t+1}
\]
学习 velocity \(Z_{t+1}-Z_t\)。\(A_t^{\mathrm{CAV}}\) 表示当前 actor 可控制的短期 planned edge pressure；\(H_t^{\mathrm{HDV}}\) 表示 HDV background 对 episode-level CAV policy pressure 的慢时间尺度响应。二者通过不同 graph branch 进入 world model，作为 velocity predictor 的外部上下文，而不是把 \(Z_t\) 放入 condition。

训练 transition 时，会把来自 `hdv_edge_action` 的 episode/window HDV target 归一化后作为 teacher-forced context 输入；imagined rollout 时，则先用 `predict_episode_hdv_response` 从 actor 的 aggregate CAV pressure 预测 \(\hat H^{\mathrm{HDV}}\)，再把该预测分布作为后续 flow-matching transition 的 HDV context。这样 HDV response head 负责估计背景响应，flow-matching transition 负责在给定 CAV action 和 HDV context 后推进 traffic latent。



## 3. Actor-Critic

Actor-critic 部分使用 `training_family=dreamer_ctde`。策略是 vehicle-level route actor，训练是 centralized training with decentralized execution。执行时，每辆 CAV 只需要自己的 route observation；训练时，critic 可以读取 latent traffic state、action edge field、context，以及同一 decision step 内其他 active CAV 的观测。

### 3.1 Vehicle-level CAV Actor

对每辆 active CAV \(i\)，系统根据其 current edge \(e_i^t\) 和 destination \(g_i\) 动态生成 candidate routes：
\[
\mathcal{P}_{i,t}=\{P_{i,t}^1,\ldots,P_{i,t}^{K_i(t)}\}.
\]
当前主线使用 `current_to_dest` 语义：候选路径每个 decision point 都从车辆当前位置重新到目的地生成。因此 route slot 只在该车辆当前 candidate set 内有意义，`route0` 不表示全局固定物理路线。full-cycle runner 在 MARL 训练中传入 `--vehicle_candidate_source current_to_dest`，evaluation 侧可使用 `current_shortest_freeflow` 等稳定候选策略。

Vehicle observation \(o_{i,t}\) 由以下信息拼接：

- 每条 candidate route 上的 edge latent/decoded feature summary；
- optional global edge summary；
- route-cost features，例如 relative cost gap、log route cost 和 valid mask；
- optional fleet pressure features，例如同 tick 已规划 route share 或 pressure overlap；
- 低维 context，例如 episode progress、active ratio 和 time/load features。

`VehicleRouteActor` 支持 MLP 和 `route_shared` 两种结构。代码默认 `vehicle_actor_architecture="mlp"`，即将整条 vehicle observation 拼接后输入 MLP policy head；`route_shared` 是同一接口下的候选路径共享打分器，适合在论文叙述中强调 route-slot invariant 的策略结构。使用 `route_shared` 时，每条 candidate route 由共享参数函数打分：
\[
\ell_{i,t,k}
=
f_\theta(
\phi(P_{i,t}^k),
\phi_{\mathrm{global}}(Z_t),
\mathrm{cost}_{i,t,k},
\mathrm{fleet}_{i,t,k},
C_t
),
\]
然后通过 masked softmax 得到
\[
\pi_\theta(k\mid o_{i,t})
=
\frac{\exp(\ell_{i,t,k})}
{\sum_{k':\mathrm{valid}}\exp(\ell_{i,t,k'})}.
\]
invalid routes 被 mask 为极低 logit；如果全部 invalid，则代码回退到零 logits，避免 NaN。route-shared 结构降低 slot bias，因为每个 candidate slot 共享同一套 route preference function；MLP 结构则保留当前默认的 concatenated-observation actor 行为。

Actor 的输出不会直接作为 world-model action id。它会被 aggregation layer 转换成 \(A_t^{\mathrm{CAV}}\)，再输入 flow-matching world model。真实 SUMO evaluation 时，route distribution 会通过 argmax、sampling 或受控混合策略落成可执行 rerouting 指令。

### 3.2 Global-level Critic

Global critic 对应 `CentralizedLatentValueCritic`。它读取 latent edge state \(Z_t\)，通过 mean pooling 或 global-attention pooling 得到 traffic summary：
\[
s_t^{\mathrm{global}}=\mathrm{Pool}(Z_t).
\]
当 `dreamer_critic_variant=action_conditioned` 或 context critic 启用时，它还可拼接 action edge field \(A_t^{\mathrm{CAV}}\) 和 critic context \(C_t^{\mathrm{critic}}\)，预测 step-level global value：
\[
V_\omega^{\mathrm{global}}
=
V_\omega(Z_t,A_t^{\mathrm{CAV}},C_t^{\mathrm{critic}}).
\]
full-cycle runner 默认打开 `--dreamer_critic_context`，并可通过 `--dreamer_critic_aggregation` 选择 mean 或 attention aggregation。global critic 适合估计 aggregate rollout value，例如全网 travel-time 或 action-conditioned social cost。

需要注意的是，当前 full-cycle 默认 `dreamer_all_reward_weight=0.0`，并将 route-cost reward scope 设为 `vehicle_critic`。因此 global critic 仍是架构组成部分，但默认主优化信号更多通过 agent-level critic 承担。

### 3.3 Agent-level Critic

Agent-level critic 对每个 sampled CAV decision 预测 per-vehicle value：
\[
V_\eta^{\mathrm{veh}}(o_{i,t},C_t^{\mathrm{critic}}).
\]
该 critic 支持 `mlp` 和 `self_attention` 架构。full-cycle runner 默认启用 `--dreamer_vehicle_critic`、`--dreamer_vehicle_critic_architecture self_attention`、`--dreamer_vehicle_critic_context` 和 return normalizer。`self_attention` 模式会把同一 imagined step 中的 active vehicles 按 group index 分组，在 value head 前做 multi-head self-attention：
\[
\tilde h_{i,t}
=
\mathrm{SelfAttention}(\{h_{j,t}:j\in\mathcal{I}_t^{\mathrm{CAV}}\}).
\]
这样 critic 输出仍是一车一个 value，但 value estimate 可以利用同 tick fleet information，表达 CAV 之间的 route overlap、congestion externality 和 shared bottleneck competition。

### 3.4 Reward Function and Actor-Critic Objective

UrbanDreamer 的 reward 由 world-model decoded readouts 和 route-level bookkeeping 计算。`_dreamer_step_reward` 先从 \(Z_{t+1}\) 解码 edge dynamics，得到 edge volume 与 edge travel time。edge travel time 可以来自 decoded speed、BPR volume 或 route-cost head：

- `decoded_speed`：根据 decoded speed ratio 与 free-flow time 得到 edge cost；
- `bpr_volume`：根据 decoded volume 和 BPR layer 得到 congestion travel time；
- `route_head`：使用 `FlowProjector.predict_edge_route_cost` 的 planner-calibrated cost。

全局平均 travel time 可写为
\[
\bar T_t^{\mathrm{all}}
=
\frac{\sum_e \hat v_t(e)\hat T_t(e)}
{\max(\sum_e \hat v_t(e),1)}.
\]
但当前 full-cycle 主线默认 `dreamer_all_reward_weight=0.0`，不把该全局项作为主要 reward。

默认主奖励是 vehicle-critic route progress reward。full-cycle runner 默认：

- `dreamer_route_cost_reward_weight=1.0`；
- `dreamer_route_cost_reward_scope=vehicle_critic`；
- `dreamer_route_cost_reward_mode=progress_fraction`；
- `dreamer_route_cost_reward_state=current`；
- `dreamer_grpo_route_cost_weight=0.0`；
- `dreamer_route_cost_aux_weight=0.0`。

`progress_fraction` 奖励不是 free-flow shortest-path potential，而是当前候选 route 的 route-progress fraction。对车辆 \(i\) 的候选 route \(k\)，令
\[
\hat T_{i,t,k}^{\mathrm{rem}}
\]
表示根据 world-model edge travel-time readout 估计的该候选 route 剩余 travel time，\(\tau_{i,t}^{\mathrm{elapsed}}\) 表示车辆已行驶时间，\(\Delta t\) 表示一个 decision interval。一个 step 内最多推进 \(\Delta t\)，但不能超过该 route 的剩余时间，因此有效推进量为
\[
\Delta \hat T_{i,t,k}
=
\min(\Delta t,\hat T_{i,t,k}^{\mathrm{rem}}).
\]
该 route 的 progress reward 为
\[
r_{i,t,k}^{\mathrm{prog}}
=
\min\left(
\frac{
\Delta \hat T_{i,t,k}
}{
\max(\tau_{i,t}^{\mathrm{elapsed}}+\hat T_{i,t,k}^{\mathrm{rem}},\epsilon)
},
c_{\mathrm{prog}}
\right),
\]
其中 \(c_{\mathrm{prog}}\) 对应 `dreamer_ac_route_credit_progress_cap`，默认上限为 1。若使用 selected route，则取
\[
r_{i,t}^{\mathrm{veh}}
=
w_{\mathrm{route}}\,
r_{i,t,a_{i,t}}^{\mathrm{prog}},
\]
其中 \(w_{\mathrm{route}}\) 对应 `dreamer_route_cost_reward_weight`。若计算 per-route credit，则保留整行 \(r_{i,t,k}^{\mathrm{prog}}\) 供 route advantage 使用。

该设计把每个 decision step 的奖励解释为“车辆本次决策使自己完成了整段 trip 的多少比例”，避免直接用 remaining route seconds 让长途车辆主导 critic target。`freeflow_shortest_progress` 仍保留在 `train_marl.py` 中作为兼容/消融模式，但不是当前 full-cycle 训练主线。


Critic target 使用 lambda return。对 global critic：
\[
G_t =
r_t+\gamma\left((1-\lambda)V(Z_{t+1})+\lambda G_{t+1}\right),
\]
并最小化
\[
\mathcal{L}_{V}^{\mathrm{global}}
=
\left\|
V_\omega^{\mathrm{global}}(Z_t,A_t,C_t)
-
\mathrm{stopgrad}(G_t)
\right\|_2^2.
\]
Agent-level critic 对每个 vehicle decision 构造对应的 per-vehicle lambda target，并最小化
\[
\mathcal{L}_{V}^{\mathrm{veh}}
=
\mathbb{E}_{i,t}
\left[
\left(
V_\eta^{\mathrm{veh}}(o_{i,t},C_t)
-
\mathrm{stopgrad}(G_{i,t}^{\mathrm{veh}})
\right)^2
\right].
\]

Actor 默认使用 discrete actor-critic objective。对 sampled route action \(a_{i,t}\)，advantage 来自 vehicle critic、global critic 或 route-credit signal 的组合：
\[
\mathcal{L}_{\pi}
=
-
\mathbb{E}_{i,t}
\left[
\log\pi_\theta(a_{i,t}\mid o_{i,t})
\cdot
\mathrm{clip}(\hat A_{i,t},[-c,c])
\right]
-
\beta\mathcal{H}(\pi_\theta).
\]
实现还包含 PPO-style ratio clipping、advantage normalization、entropy floor、actor trust-region gate、anchor policy KL 等稳定器。新版主线应把这些视为 imagined policy learning 的 safeguards，而不是额外的问题定义。

## 4. Training Pipeline

UrbanDreamer 的训练流程是 simulator-grounded collection 与 offline model/policy learning 的闭环。真实 SUMO/RouteRL 提供高保真但昂贵的 trajectories；flow-matching world model 提供低成本 imagined rollouts；actor-critic 在 imagined rollouts 中更新策略；最终 checkpoint 仍回到 SUMO/RouteRL 中选择和刷新数据。

### 4.1 Data Collection in SUMO/RouteRL

数据采集由 `scripts/collect_urb_fair.py` 和 full-cycle runner 调度。每个 episode 中，车辆按照 `ratio_machines` 分为 CAV 与 HDV。CAV 根据采集阶段使用 random、epsilon-greedy、fixed/current policy 或混合策略产生 route choices；HDV 由 SUMO/RouteRL 背景 route adaptation 机制控制。

每个 decision step 保存 world-model 和 actor-critic 所需的监督字段：

- raw edge traffic field \(X_t\)，即 `(E,8)` flow matrix；
- `av_planned_edge_flow`，即 CAV selected/planned route 在 edge 上的 pressure；
- `hdv_edge_action` 或其他 HDV auxiliary target/context；
- `depart`，即 departure field \(D_t\)；
- `context`，即低维时间/负载上下文 \(C_t\)；
- reward、episode id、vehicle/route metadata、candidate route information。

这些 records 同时承担两个角色：为 flow-matching world model 提供 transition supervision，为 imagined actor-critic 初始化 latent state、active vehicle set 和 route candidates。

### 4.2 Collection Warmup

本文把 collection warmup 与 world-model optimization warmup 分开。collection warmup 对应 full-cycle runner 中的 `--warmup-episodes` 和 settings 中的 `HDV_ADAPT_EP`。它发生在 SUMO/RouteRL 数据采集或评估前，目的是让 HDV route adaptation 先根据网络和 incident/demand 条件形成较稳定的背景状态。这样主采集阶段保存的数据更接近“CAV policy 与已适应 HDV background 交互”的分布，而不是 HDV 初始随机状态。

该 warmup 不训练 neural world model；它改变的是真实仿真器中的 HDV background state。evaluation 阶段也可以通过保存/加载 `post_mutation_state` 或 HDV final state 来复现实验条件。

### 4.3 World-model Optimization Warmup

world-model optimization warmup 指训练配置层面的预热，而不是一个代码强制的固定两阶段 pipeline。当前 `train_dynamics.py` 会联合构造 `FlowProjector` 和 `Dynamics`，并在同一个 optimizer 中训练所有未冻结参数；如果传入 `--freeze_flow_projector`、`--flow_projector_ckpt`、`--resume` 或 `--train_route_cost_head_only`，则可以形成不同的 staged/freeze training 行为。

主线可采用以下 warmup 思路：

- 先提高 decoder reconstruction、decoded supervision、BPR/route-cost supervision 或 \(t_{\mathrm{all}}\) supervision 的权重，使 \(Z_t\) 对交通 readouts 可解释；
- 使用 one-step flow-matching transition 作为基础监督，即 `rollout_horizon=1`；
- 启用 scheduled sampling 时，通过 `sched_sample_warmup` 将 sampling probability 线性 ramp 到 `sched_sample_end`，降低 autoregressive distribution shift；
- 再逐步使用 transition decoded supervision、multi-step rollout horizon 或 ADM-style direct k-step losses。

需要明确的是：当前代码没有硬编码“先只训练 encoder/decoder，再训练 flow-matching transition”的必经阶段。warmup 是通过 loss weights、resume/freeze flags、scheduled sampling 和 rollout horizon 等配置实现的训练策略。

### 4.4 Offline World-model Training

World model 使用 `dreamer4/train_dynamics.py` 离线训练。full-cycle runner 构造的主线命令包含：

- `--action_key av_planned_edge_flow`；
- `--aux_action_key hdv_edge_action`；
- `--hdv_policy_weight` 控制 \(\mathcal{L}_{\mathrm{HDV}}\) 的权重；
- `--hdv_policy_target_mode` 选择 `step`、`window_mean` 或 `episode_mean/episode_response` target 语义；
- `--hdv_policy_action_mode` 选择 HDV head 使用 per-step、window-level 或 episode-level CAV pressure context；
- `--action_encoder_type graph_cond`；
- `--edge_action_injection` 和 `--edge_action_value_norm active_count`；
- `--graph_cond_edge_conditioning` 可选启用；
- `--transition_hdv_context_mode` 可选启用；
- `--decoder_recon_weight`、`--transition_decoded_supervision_weight`、`--bpr_supervision_weight`、`--t_all_supervision_weight` 等辅助监督权重。

每个 training batch 从 collected flow shards 中采样长度为 `seq_len` 的序列。训练步骤为：

1. `FlowProjector` 编码 flow sequence，得到 \(Z_{1:T}\)；
2. `_build_graph_cond_model_actions` 将 CAV action、departure、context 和 HDV context 打包为 graph context tuple；
3. `cfm_loss` 取上一 latent \(Z_t\) 作为 \(x_0\)，对 \(\tau\) 采样并插值到目标 latent \(Z_{t+1}\)，训练 `Dynamics` 输出 flow velocity \(Z_{t+1}-Z_t\)；该函数名沿用代码历史命名，但该步骤的新版 method 语义是 flow matching；
4. 根据配置加入 HDV response CE、decoder reconstruction、transition decoded supervision、BPR/route-cost、route ranking/distillation、\(t_{\mathrm{all}}\) 等辅助损失；
5. 保存 `flow_projector` 与 `dynamics` checkpoint，供 MARL imagination 加载。

核心优化目标可写成
\[
\mathcal{L}_{\mathrm{WM}}
=
\mathcal{L}_{\mathrm{FM}}
+
\lambda_{\mathrm{dec}}\mathcal{L}_{\mathrm{dec}}
+
\lambda_{\mathrm{trans}}\mathcal{L}_{\mathrm{trans-dec}}
+
\lambda_{\mathrm{BPR}}\mathcal{L}_{\mathrm{BPR}}
+
\lambda_{\mathrm{HDV}}\mathcal{L}_{\mathrm{HDV}}
+
\lambda_{t_{\mathrm{all}}}\mathcal{L}_{t_{\mathrm{all}}}
\cdots .
\]
其中只有 \(\mathcal{L}_{\mathrm{FM}}\) 是 transition 的必要主损失，其余项由配置启用。

### 4.5 Imagined Actor-Critic Training

Actor-critic 使用 `dreamer4/urban_routing/train_marl.py`。full-cycle runner 默认设置：

- `--training_family dreamer_ctde`；
- `--wm_cfm_init_mode` 需同步改为以上一 latent \(Z_t\) 初始化的 state-to-state 模式；该参数名仍沿用代码历史命名，当前代码中的 `zero` 语义属于待替换的旧实现；
- `--wm_policy_obs_source decoder_volume_tt`；
- `--wm_edge_cost_source decoded_speed`；
- `--vehicle_candidate_source current_to_dest`；
- `--dreamer_actor_objective discrete_actor_critic`；
- `--dreamer_actor_horizon 8`；
- `--dreamer_vehicle_critic`；
- `--dreamer_vehicle_critic_architecture self_attention`；
- `--no-dreamer_vehicle_route_q_critic`；
- `--dreamer_all_reward_weight 0.0`；
- `--dreamer_route_cost_reward_scope vehicle_critic`；
- `--dreamer_route_cost_reward_mode progress_fraction`；
- `--dreamer_route_cost_reward_state current`。

一次 imagined rollout 的流程如下：

1. 从 offline dataset 采样真实 latent state 和 route/vehicle metadata，重建 active CAV set；
2. 为每辆 active CAV 生成 current-to-destination candidate routes；
3. 构建 vehicle route observation，并用 `VehicleRouteActor` 输出 route probabilities；
4. 将 route probabilities 聚合为 \(A_t^{\mathrm{CAV}}\)；
5. 查询或复用 episode-level HDV response background \(H^{\mathrm{HDV}}\)；
6. 用 flow-matching world model 预测 \(Z_{t+1}\)；
7. 用 decoder/readouts 和 route lifecycle wrapper 计算 reward、route credit、arrival/progress bookkeeping；
8. 用 global critic 和 vehicle critic 构造 lambda returns；
9. 更新 actor、critics 和 slow/target critics。

这里需要区分两条状态更新路径。World model 只推进宏观 edge latent：
\[
Z_t \xrightarrow{\mathrm{FM}(A_t^{\mathrm{CAV}},D_t,C_t,H_t^{\mathrm{HDV}})}
Z_{t+1}.
\]
车辆级位置和到达状态不是 neural world model 的输出，而是由 imagined rollout 外层维护的 route lifecycle wrapper 推进。对每辆 active CAV，wrapper 维护一个轻量状态：
\[
\chi_{i,t}
=
\left(
e_{i,t}^{\mathrm{cur}},
\rho_{i,t}^{\mathrm{edge}},
g_i,
d_i^{\mathrm{dep}},
P_{i,t}^{\mathrm{cur}},
\mathrm{arrived}_{i,t}
\right),
\]
其中 \(e_{i,t}^{\mathrm{cur}}\) 是当前 edge，\(\rho_{i,t}^{\mathrm{edge}}\) 是当前 edge 上剩余 travel time，\(g_i\) 是 destination，\(d_i^{\mathrm{dep}}\) 是 departure step，\(P_{i,t}^{\mathrm{cur}}\) 是当前选中的 route suffix。

每个 imagined decision step 中，先根据 `departure_schedule` 注入新出发的 CAV/HDV，再移除已到达或超过 maximum lifetime cutoff 的车辆；对于从 episode 中途重建出来的车辆，`_ensure_active_vehicle_current_edges` 会用 departure step、参考 route 和平均 edge time 估计其 current edge，避免把所有历史车辆错误地放回 origin。随后对每辆未到达 CAV 以 current edge 和 destination 生成 candidate routes。宏观 world-model action \(A_t^{\mathrm{CAV}}\) 可以由 route probabilities 聚合得到；车辆级推进则使用 rollout policy 对应的 selected route \(P_{i,t}^{a_i}\)，并把该 route 记为新的 `current_route`。

车辆推进使用 world-model decoded/readout 得到的 edge travel time \(\hat T_t(e)\) 和一个 decision interval \(\Delta t\)。令 selected route suffix 从当前 edge 开始，若当前 edge 不在该 suffix 中，则从候选 route 的首条 edge 开始。wrapper 将 \(\Delta t\) 当作 time budget 沿 route suffix 消耗：
\[
b\leftarrow \Delta t,\qquad
\rho \leftarrow
\begin{cases}
\rho_{i,t}^{\mathrm{edge}}, & \text{如果缓存的剩余时间仍对应当前 edge},\\
\hat T_t(e_{i,t}^{\mathrm{cur}}), & \text{否则}.
\end{cases}
\]
若 \(b<\rho\)，车辆仍停留在当前 edge，并更新
\[
e_{i,t+1}^{\mathrm{cur}}=e_{i,t}^{\mathrm{cur}},
\qquad
\rho_{i,t+1}^{\mathrm{edge}}=\rho-b.
\]
若 \(b\ge \rho\)，则先消耗当前 edge，再把剩余 budget 用于 suffix 中后续 edge；每跨过一条 edge，就把 current edge 更新为下一条 edge，并把该 edge 的剩余时间初始化为 \(\hat T_t(e)\)。如果 budget 消耗完 selected suffix 的最后一条 edge，则
\[
\mathrm{arrived}_{i,t+1}=\mathrm{true},
\]
该车辆在后续 imagined steps 中不再生成 actor sample，也不再贡献 per-vehicle route-progress reward。否则，未到达车辆携带更新后的 current edge、edge remaining time 和 current route 进入下一步，并重新从 current edge 到 destination 生成 candidate routes。

因此 actor-critic 的 imagination 是“宏观 latent 由 flow-matching world model 推进，车辆微观位置由 deterministic route-progress wrapper 推进”的混合过程。这样可以让 reward、route credit、arrival/progress bookkeeping 与 selected route 对齐，同时避免让 flow-matching transition 同时学习 per-vehicle terminal dynamics。

### 4.6 SUMO Evaluation and Data Refresh

训练出的 actor checkpoint 会放回 SUMO/RouteRL 中评估。evaluation 记录真实 simulator 指标，包括：

- \(t_{\mathrm{all}}\)：所有车辆平均 travel time；
- \(t_{\mathrm{CAV}}\)：CAV 平均 travel time；
- \(t_{\mathrm{HDV}}\)：HDV 平均 travel time；
- completion/arrival fraction；
- route usage、fallback statistics、rerouting diagnostics；
- incident/post-mutation state 相关诊断。

Checkpoint selection 以真实 SUMO diagnostics 为准，而不是 imagined reward 单独决定。若启用 refresh cycle，则将当前 actor 生成的新 trajectories 加入 world-model data pool，下一轮重新训练或 fine-tune world model，使 surrogate distribution 更贴近当前 policy 诱导出的 state-action distribution。

## 5. Implementation Notes

- 当前主线使用 CAV planned edge pressure 作为 action interface。代码字段名仍是 `av_planned_edge_flow`，但论文叙述中可写作 \(A_t^{\mathrm{CAV}}\)。
- `route0` 只在单车当前 candidate set 内有局部意义，不能作为 world model 的全局动作。因此 world model 不预测 route id dynamics，而预测 edge-token traffic transition。
- `ActionDepartGNNEncoder` 将 CAV pressure、departure 和 HDV context 分支建模；当 `graph_cond_edge_conditioning` 启用时，局部 edge context 直接注入 spatial tokens。
- `hdv_policy_head` 是 neural HDV response 主线；Stackelberg 和 Gawron/RouteRL 主要作为真实环境背景、fallback 或 ablation。
- GRPO rollout objective、route-cost auxiliary direct actor loss、vehicle route-Q critic、PCFM sampler 等机制仍保留在代码中，但当前 full-cycle 默认不把它们作为主优化路径。
- 新版 method 的默认叙述应围绕 `graph_cond` world model、flow-matching transition、episode-level HDV response、`dreamer_ctde` actor-critic、vehicle critic route-progress reward 和 SUMO-grounded evaluation 展开。
