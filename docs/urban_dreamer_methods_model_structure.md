# UrbanDreamer Methods: CFM 世界模型与 Vehicle-level Actor-Critic 主线

> 中文 Methods 草稿。本文档按机器学习顶会论文的 Methods 写法组织当前代码主线，重点覆盖模型结构、状态/动作语义、训练目标和闭环训练流程。实现细节以当前仓库为准；其中 rollout GRPO、Stackelberg/PCFM 等保留为可选扩展或消融，不作为主线叙事。

## 1. Problem Formulation

我们研究一个混合人机交通网络中的路由博弈问题。路网被表示为有向 road-edge graph \(\mathcal G=(\mathcal V,\mathcal E)\)，其非内部 SUMO edge 构成固定集合 \(\mathcal E=\{1,\ldots,E\}\)。交通参与者包括可控自动驾驶车辆（AV）和人类驾驶车辆（HDV）。由于 HDV 的路线选择会随交通状态、历史经验和 AV 诱导策略而变化，AV 之间也通过共享道路容量产生拥堵外部性，该问题不能被建模为独立单智能体最短路问题，而应首先被视为一个动态多智能体路由博弈。

### 1.1 Routing as a Stochastic Game

令 \(\mathcal N_t=\mathcal N_t^{\mathrm{AV}}\cup\mathcal N_t^{\mathrm{HDV}}\) 表示决策步 \(t\) 网络中的 active vehicles。真实交通系统可形式化为一个随机（Stochastic）博弈：
\[
\mathcal G_{\mathrm{route}}
=
\langle
\mathcal N,\mathcal S,
\{\mathcal A_i\}_{i\in\mathcal N},
P,
\{r_i\}_{i\in\mathcal N},
\{\mathcal O_i\}_{i\in\mathcal N},
\Omega,
\gamma
\rangle .
\]
全局状态 \(s_t\in\mathcal S\) 包含完整但不可被单车观测的交通微观状态和慢变量，例如车辆位置、速度、排队长度、lane/junction 状态、每辆车的剩余路径、OD/departure schedule、incident 状态、HDV 路由学习状态以及当前网络拥堵场。

Departure schedule 是该博弈的重要外生过程。每辆车 \(i\) 具有出发时间 \(\tau_i^{\mathrm{dep}}\)、origin edge \(o_i\)、destination edge \(g_i\) 和类型 \(m_i\in\{\mathrm{AV},\mathrm{HDV}\}\)。在决策步 \(t\)，满足 \(\tau_i^{\mathrm{dep}}=t\) 的车辆被注入网络，形成新的 active agents。该过程决定当前网络中有多少 AV/HDV 参与博弈，也决定世界模型中的 departure field \(D_t\)。因此，转移不仅由当前 route choices 决定，也由外生发车需求驱动。

每个智能体 \(i\) 的动作 \(a_{i,t}\in\mathcal A_i(s_t)\) 是在给定候选路由集合中的 route choice 或 route commitment。对任意车辆，其候选动作空间可写为
\[
\mathcal A_i(t)=\{1,\ldots,K_i(t)\},
\]
其中第 \(k\) 个动作表示选择候选路径 \(P_{i,t}^{k}\)。不过 AV 和 HDV 的决策时机不同：HDV 在 departure 时刻只做一次路线选择，并在 episode 内保持该路径；AV 则在行驶过程中每隔一个 rerouting interval 重新生成 current-edge-to-destination candidate routes，并重新选择路线。对 AV 来说这是系统要学习的控制动作；对 HDV 来说则由人类响应模型或 RouteRL/Gawron 风格策略产生。

联合动作 \(a_t=(a_{i,t})_{i\in\mathcal N_t}\) 与 departure process 共同通过 SUMO 微观交通动力学和 HDV 适应机制诱导转移：
\[
s_{t+1}\sim P(\cdot\mid s_t,a_t,D_t).
\]
转移是随机的，原因包括：HDV 策略对 AV pressure 的响应、车辆出发和到达过程、微观 car-following/lane-changing/junction 交互、以及从单车视角不可观测的网络内部状态。奖励 \(r_i(s_t,a_t)\) 通常与个体旅行时间、剩余行程进展、拥堵外部性或系统级 travel time 相关。本文只学习 AV 策略；HDV 被建模为环境中的自适应智能体，其行为影响 AV 的状态转移和回报。

### 1.2 Agent-level POMDP

对任意单个车辆智能体 \(i\)，由于它无法观测全局微观状态 \(s_t\)、其他车辆未来动作以及 HDV 内部学习状态，其决策过程是一个部分可观测马尔可夫决策过程（POMDP）：
\[
\mathcal M_i
=
\langle
\mathcal S,\mathcal A_i,P_i,r_i,\mathcal O_i,O_i,\gamma
\rangle .
\]
其中 \(O_i(o_{i,t}\mid s_t)\) 是观测函数。车辆只有在其决策时刻才需要输出动作：HDV 的决策时刻是 departure step，AV 的决策时刻是每个 rerouting interval。对 AV actor 来说，局部观测 \(o_{i,t}\) 由可获得的路由和交通摘要构成，包括：

- 当前 edge、destination edge、已行驶时间和 episode/time context；
- 当前和未来外生发车需求的摘要，例如当前 decision step 的 departure field；
- 从当前 edge 到 destination 的 \(K\) 条 candidate routes；
- 每条候选路径上的 edge latent/dynamic features、route travel-time estimate、relative cost gap 和 valid mask；
- 可选的 global traffic summary；
- 同一决策步内已规划 AV fleet pressure 或候选路径 overlap features。

每个 agent 的离散动作空间都是给定候选路径集合上的选择：
\[
\mathcal A_i(o_{i,t})
=
\{1,\ldots,K_i(t)\},
\]
其中 \(K_i(t)\le K\)，主线默认 \(K=4\)。区别在于动作是否会被后续更新：HDV 在 \(\tau_i^{\mathrm{dep}}\) 选择一次
\[
a_i^{\mathrm{HDV}} \sim \pi^{\mathrm{HDV}}(\cdot\mid o_{i,\tau_i^{\mathrm{dep}}}),
\]
并在 episode 内保持该路径；AV 在每个 rerouting decision step \(t\in\mathcal T_i^{\mathrm{AV}}\) 重新选择
\[
\pi_\theta(a_{i,t}=k\mid o_{i,t}),
\]
即选择第 \(k\) 条 current-edge-to-destination candidate route。HDV 策略不由本文优化；在真实采集/评估中由 HDV 学习或响应机制决定，在 imagined rollout 中由 episode-level HDV response module 近似。

### 1.3 Macroscopic World-model State and Action Spaces

直接在完整 stochastic game 的微观状态空间 \(\mathcal S\) 上做规划不可行。因此 UrbanDreamer 使用一个宏观 edge-level world model 近似该博弈的聚合转移。宏观状态不再保留每辆车的全部微观状态，而是使用固定路网上的 edge traffic field：
\[
X_t \in \mathbb R^{E\times 8}.
\]
前四个通道是动态交通量：
\[
[\mathrm{occ},\ \mathrm{speed}/\mathrm{speedLimit},\ \mathrm{veh}/\mathrm{capacity},\ \mathrm{halt}/\mathrm{capacity}],
\]
后四个通道是归一化静态路网属性：
\[
[\mathrm{length},\ \mathrm{speedLimit},\ \mathrm{lanes},\ \mathrm{roadType}].
\]
世界模型将 \(X_t\) 编码为 latent edge tokens \(Z_t\in\mathbb R^{E\times d_z}\)，其中当前主线默认 \(d_z=64\)。宏观状态还包含历史 latent \(Z_{\le t}\)、departure field \(D_t\in\mathbb R^E\)、低维上下文 \(C_t\in\mathbb R^{d_c}\)，以及可选 HDV response context \(H_t^{\mathrm{HDV}}\in\mathbb R^E\)。其中 \(D_t(e)\) 表示当前 decision step 从 edge \(e\) 注入网络的外生发车需求，是由 departure schedule 聚合得到的 edge-aligned demand field。

宏观动作不是单车 route id，而是所有处于 AV rerouting decision step 的车辆选择聚合后的 edge-aligned planned pressure。令 \(\mathcal I_t=\mathcal N_t^{\mathrm{AV}}\) 表示当前 active AV 集合。每辆 AV \(i\in\mathcal I_t\) 具有当前位置 \(e_i^t\)、目的 edge \(g_i\)，并在当前状态下生成 \(K\) 条 current-edge-to-destination candidate routes：
\[
\mathcal P_{i,t}=\{P_{i,t}^{1},\ldots,P_{i,t}^{K}\}.
\]
给定一个短期进入窗口 \(H_{\mathrm{edge}}\)（实现中对应一个 decision interval），定义 \(P_{i,t}^{k,H}\) 为车辆选择第 \(k\) 条路线后在该窗口内预计会进入的后续 edge 集合；当前已占据 edge 不计入 future-entry pressure。soft actor rollout 下的宏观 AV action 为
\[
A_t^{\mathrm{AV}}(e)
= \frac{1}{\max(|\mathcal I_t|,1)}
\sum_{i\in\mathcal I_t}\sum_{k=1}^{K}
\pi_\theta(k\mid o_{i,t})\,
\mathbf 1\{e\in P_{i,t}^{k,H}\}.
\]
在真实 SUMO 采集和评估中，\(\pi_\theta\) 被实际选择的 one-hot route 替代。因此，`av_planned_edge_flow` 表示“当前受控 AV 计划在下一决策间隔内进入某条 edge 的比例”，不是真实观测 flow、occupancy 或 veh/s。

在宏观层面，世界模型学习 action-conditioned aggregate transition：
\[
p_\phi(Z_{t+1}\mid Z_{\le t},A_t^{\mathrm{AV}},D_t,C_t,H_t^{\mathrm{HDV}}).
\]
该转移可以被看作对底层 stochastic routing game 的聚合近似：AV 的 vehicle-level route choices 通过 \(A_t^{\mathrm{AV}}\) 影响道路压力，HDV 的自适应响应通过 \(H_t^{\mathrm{HDV}}\) 进入条件，CFM dynamics 则预测下一步 edge-level traffic latent。

从系统层面看，UrbanDreamer 将原始交通博弈分解为三个可学习接口：

1. **Agent-level control:** \(o_{i,t}\mapsto \pi_\theta(\cdot\mid o_{i,t})\)，在每辆 AV 的 POMDP 观测上进行 route choice。
2. **Macro action aggregation:** \(\{\pi_\theta(\cdot\mid o_{i,t})\}_{i\in\mathcal I_t}\mapsto A_t^{\mathrm{AV}}\)，把多车离散动作转成 edge-aligned planned pressure。
3. **Action-conditioned imagination:** \((Z_{\le t},A_t^{\mathrm{AV}},D_t,C_t,H_t^{\mathrm{HDV}})\mapsto Z_{t+1}\)，用 CFM dynamics 近似真实博弈的宏观转移。

这三个接口构成一个闭环：真实仿真器提供离线监督数据，世界模型提供可快速 rollout 的 latent simulator，actor-critic 在该 simulator 中学习策略，再回到真实仿真器评估和刷新数据。

## 2. World Model

UrbanDreamer 的世界模型由三个核心模块组成：

1. **Tokenizer.** 将宏观交通状态和宏观动作分别 token 化，包括 state tokenizer 与 action tokenizer。
2. **CFM dynamics.** 只负责在已 token 化的状态与条件下建模 transition，即学习 \(Z_t\rightarrow Z_{t+1}\) 的 latent dynamics。
3. **HDV response module.** 从 episode-level AV pressure 预测 HDV edge background，使 imagined rollout 能表达 HDV 对 AV 策略的自适应响应。

其中 tokenizer 是世界模型的输入接口：state tokenizer 负责把 SUMO edge state 转为 latent edge tokens，action tokenizer 负责把 AV pressure、departure、context 和 HDV response 转为 dynamics 可使用的条件 tokens。Dynamics 不再承担动作语义解析或辅助读出，它只消费 tokens 并预测下一步 latent state。世界模型的目标也不是从零学习微观交通规则，而是在固定路网、聚合 edge state 和 planned route pressure 条件下，学习一个快速、可 rollout 的 aggregate traffic transition surrogate。

### 2.1 Tokenizer

Tokenizer 包含两个并列部分：state tokenizer 和 action tokenizer。前者处理交通状态，后者处理控制动作与外生条件。二者都利用固定 road-edge graph 的拓扑结构，使 dynamics 接收到的输入已经是 edge-aligned token representation。

#### State Tokenizer

State tokenizer 对应实现中的 EdgeRegistry 和 FlowProjector。它把每个 SUMO decision step 的 edge-level traffic field
\[
X_t\in\mathbb R^{E\times 8}
\]
编码为 latent edge tokens
\[
Z_t=f_{\mathrm{state}}(X_t)\in\mathbb R^{E\times d_z}.
\]
其中 \(E\) 是固定非内部 SUMO edge 数量，当前主线默认 \(d_z=64\)。

系统首先由 SUMO 路网构建固定 EdgeRegistry。每条 edge 有稳定 index、静态属性、capacity/free-flow travel time，以及 edge-line graph adjacency。这里的 graph node 是 road edge；若一条 road edge 可以通向另一条 road edge，则在 line graph 中形成邻接关系。邻接矩阵 \(A\in\mathbb R^{E\times E}\) 经过 row normalization 后供 state/action tokenizers 使用。

FlowProjector 先对 \(X_t(e)\in\mathbb R^8\) 做 linear projection，再用若干轻量 GCN-style EdgeGNNLayer 注入局部拓扑信息：
\[
h_e^0=W_x X_t(e),
\]
\[
m_i=\sum_{j=1}^{E} A_{ij} W_{\mathrm{msg}} h_j,
\]
\[
\tilde h_i=\mathrm{GELU}(W_{\mathrm{self}}h_i+m_i),
\]
\[
h_i'=\mathrm{LayerNorm}(h_i+\tilde h_i).
\]
最终 \(Z_t=h^L\) 作为 dynamics 的 spatial state tokens。FlowProjector 还包含 decoder/readout，用于把 latent token 解码为 speed、volume、route cost 或 \(t_{\mathrm{all}}\) 等下游可读量；这些 readouts 属于 state tokenizer 的可解释读出接口，不改变 dynamics 的 transition 定义。

#### Action Tokenizer

Action tokenizer 对应实现中的 `ActionDepartGNNEncoder`。它不是离散 route-id tokenizer，而是 continuous graph-conditioned tokenizer。其输入包括：

\[
A_t^{\mathrm{AV}}\in\mathbb R^E,\quad
D_t\in\mathbb R^E,\quad
H_t^{\mathrm{HDV}}\in\mathbb R^E,\quad
C_t\in\mathbb R^{d_c}.
\]

其中 \(A_t^{\mathrm{AV}}\) 是短期 AV planned edge pressure，\(D_t\) 是 departure field，\(H_t^{\mathrm{HDV}}\) 是 episode-level HDV response background，\(C_t\) 是低维时间/负载上下文。每个 edge field 与 static edge features 拼接后进入独立 GNN 分支：
\[
u^{A}=\mathrm{GNN}_A([A_t^{\mathrm{AV}},S]),
\quad
u^{D}=\mathrm{GNN}_D([D_t,S]),
\quad
u^{H}=\mathrm{GNN}_H([H_t^{\mathrm{HDV}},S]).
\]

Action tokenizer 输出两类 tokens：

1. **Global ACTION token.** 对 edge hidden states 做 mean/max pooling，并与 context MLP 输出拼接：
   \[
   q_t^{\mathrm{act}}=
   W_o[
   \mathrm{mean}_e u_e^A,\max_e u_e^A,
   \mathrm{mean}_e u_e^D,\max_e u_e^D,
   \mathrm{MLP}(C_t)
   ].
   \]
   如果启用 HDV context，则额外加入 pooled HDV branch。

2. **Edge-conditioned ACTION tokens.** 为避免全局 pooling 抹掉 action 的空间形状，主线启用 graph-conditioned edge conditioning：
   \[
   q_{t,e}^{\mathrm{edge}}=
   W_{\mathrm{edge}}[u_e^A,u_e^D,u_e^H].
   \]
   这些 token 被直接加到对应 edge 的 spatial token 上，使 dynamics 在预测 \(Z_{t+1,e}\) 时可以看到同一 edge 上的 action/depart/HDV pressure 语义。

因此，action tokenizer 同时提供全局动作摘要和局部 edge-aligned 条件。全局 ACTION token 描述 episode/tick 级负载概况，edge-conditioned tokens 保留具体哪几条道路被 planned pressure 覆盖。Dynamics 只消费这些 tokens，不再负责从原始 action field 中自行恢复动作语义。

### 2.2 Conditional CFM Dynamics

Conditional CFM dynamics 是世界模型的 transition 模块。给定 state tokenizer 输出的历史 latent tokens \(Z_{\le t}\)，以及 action tokenizer 输出的 global/edge condition tokens，dynamics 预测下一步 latent state \(Z_{t+1}\)。该模块只负责 transition：
\[
(Z_{\le t},Q_t^{\mathrm{act}},Q_t^{\mathrm{edge}})
\longmapsto
Z_{t+1}.
\]

具体地，CFM dynamics 预测从噪声到 \(Z_{t+1}\) 的 continuous-time vector field。每个时间步的 token layout 是
\[
[\mathrm{ACTION},\ \mathrm{FLOW\_TIME},\ \mathrm{SPATIAL}_1,\ldots,\mathrm{SPATIAL}_E,\ \mathrm{REGISTER}_1,\ldots,\mathrm{REGISTER}_R].
\]
其中 SPATIAL tokens 来自 state tokenizer，ACTION 与 edge-conditioned tokens 来自 action tokenizer，FLOW_TIME token 是 CFM interpolation time \(\tau\in[0,1]\) 的 MLP embedding，REGISTER tokens 是可学习 memory slots。

所有 token 加上二维正弦位置编码：一部分编码 rollout/history time index，另一部分编码 token position。随后进入 block-causal transformer。该 transformer 交替执行：

1. **Space/modality attention.** 在同一时间步内混合 action、flow-time、spatial 和 register token。主线使用的 space mode 允许 world-model tokens 在同一 step 内充分交互，同时保留 modality-aware mask 机制。
2. **Causal time attention.** 沿历史维度做因果 attention，使 \(t+1\) transition 只能使用当前及过去 latent history。
3. **MLP residual block.** 使用 gated MLP 和 residual normalization。

Transformer 的 spatial output 经过 linear head 得到 CFM velocity：
\[
v_\phi(X_\tau,\tau,Z_{\le t},A_t,D_t,C_t,H_t^{\mathrm{HDV}})
\in\mathbb R^{E\times d_z}.
\]

世界模型 transition 采用 conditional flow matching。给定目标 latent \(Z_{t+1}\)，采样 Gaussian noise \(X_0\sim\mathcal N(0,I)\) 和 interpolation time \(\tau\sim\mathcal U(0,1)\)，构造线性 probability path：
\[
X_\tau=(1-\tau)X_0+\tau Z_{t+1}.
\]
目标 velocity 是
\[
v^\star=Z_{t+1}-X_0.
\]
核心损失为
\[
\mathcal L_{\mathrm{CFM}}
=
\mathbb E\left[
w_e\,
\left\|
v_\phi(X_\tau,\tau,Z_{\le t},A_t,D_t,C_t,H_t^{\mathrm{HDV}})
-v^\star
\right\|_2^2
\right],
\]
其中 \(w_e\) 是可选 active/busy edge weighting，用于提高对非空或高负载道路的监督密度。

推理时，单个 transition 从 \(X_0\) 初始化并解 ODE：
\[
\frac{dX_\tau}{d\tau}=v_\phi(X_\tau,\tau,\cdot),\quad \tau:0\rightarrow 1.
\]
当前主线常用轻量 ODE sampling；下游 actor-critic 的 8-step rollout 是反复调用该 single-transition predictor，而不是一次 CFM 直接生成 8 个未来状态。

### 2.3 HDV Response Module

HDV 在真实 SUMO/RouteRL 中会根据交通状态和历史经验形成自适应路线选择。若在 imagined rollout 中把 HDV 视作固定背景，会低估 AV route choice 对人类驾驶车辆的外部性；若每一步都显式求解 HDV equilibrium，又会使 actor training 过慢且难以稳定。因此当前主线使用 episode-level neural HDV response。

给定 actor 在一个 imagined episode 中产生的 AV edge pressure 序列 \(\{A_t^{\mathrm{AV}}\}_{t=1}^{T}\)，先聚合为 episode pressure：
\[
\bar A^{\mathrm{AV}}
=
\mathrm{Normalize}\left(\sum_{t=1}^{T} A_t^{\mathrm{AV}}\right).
\]
HDV response head 复用 world model dynamics 中的 edge-level `hdv_policy_head`。在 episode-response 语义下，训练时该 head 不使用 causally shifted per-step action，而是使用 unshifted episode-level AV pressure condition；departure/context 被置空以匹配 MARL 推理接口。该 head 学习映射
\[
\hat H^{\mathrm{HDV}}
=
g_\psi(\bar A^{\mathrm{AV}},Z_{\mathrm{ctx}}),
\quad
\hat H^{\mathrm{HDV}}\in\Delta^{E-1},
\]
其中 \(Z_{\mathrm{ctx}}\) 是 episode 起点或查询 latent context；没有可用 latent context 时实现可用零 latent query。输出是一个 episode-level HDV edge background distribution。该背景随后作为 \(H_t^{\mathrm{HDV}}\) 注入 world model 的 condition encoder，在整个 imagined rollout 中作为相对静态的 HDV response context 使用。

该设计把 HDV response 与 controllable AV action 分离：

- \(A_t^{\mathrm{AV}}\) 表示 actor 当前可控的 planned edge pressure；
- \(H^{\mathrm{HDV}}\) 表示 HDV 对整个 episode AV policy 的背景响应；
- dynamics 通过 separate graph-conditioned branch 同时看到二者，而不是把 HDV 直接混入 AV action 语义。

解析 Wardrop/Gawron response 在系统中仍可作为 fallback、prior 或 ablation。论文主线中，它的角色是提供交通分配先验与对照，而不是替代 neural HDV response head。

### 2.4 Vehicle Lifecycle Boundary

世界模型本身只预测宏观 edge-level latent transition，并不显式预测每辆车是否仍然 active，也没有单独的 arrival/done head。换言之，\(Z_{t+1}\) 描述的是聚合交通场的下一步状态，而不是一组 per-vehicle terminal labels。

从 actor-critic rollout 的角度，active vehicle set 是世界模型外层维护的离散 bookkeeping state。对 AV 可写为
\[
\mathcal I_{t+1}^{\mathrm{AV}}
=
\Big(\mathcal I_t^{\mathrm{AV}}\cup \mathcal D_t^{\mathrm{AV}}\Big)
\setminus
\left\{
i:
b_{i,t+1}^{\mathrm{arr}}=1
\ \mathrm{or}\
t+1-\tau_i^{\mathrm{dep}}>L_{\max}
\right\},
\]
其中 \(\mathcal D_t^{\mathrm{AV}}\) 是当前 step 由 departure schedule 注入的新 AV，\(L_{\max}\) 是由平均路径长度给出的生命周期上界，\(b_{i,t+1}^{\mathrm{arr}}\) 不是神经网络预测，而是由外层车辆推进器根据当前选中 route suffix 和 edge travel-time estimates 计算得到。

车辆生命周期由 world-model rollout 外层的 bookkeeping 维护：

- 在真实 SUMO 采集和评估中，车辆到达由仿真器给出，例如通过 arrived vehicle events 记录真实 travel time。
- 在 imagined rollout 中，新车辆由 departure schedule 注入 active set。
- AV 的当前位置由轻量路径推进器维护：actor 选择候选 route 后，系统用当前 WM 解码/估计的 edge travel times \(\hat T_t(e)\) 消耗一个 decision interval 的时间预算 \(\Delta t\)，沿选中 route suffix 前进；只有当该时间预算走完所选 suffix 的最后一条 edge 时，车辆才被标记为 arrived。
- 为避免车辆无限停留，active set 还使用基于平均路径长度的最大生命周期窗口；超过该窗口的车辆会被移出 active set。
- HDV 在 imagined rollout 中主要通过 episode-level HDV response/background 影响宏观 dynamics；其 active count 同样由 departure schedule 与生命周期窗口近似维护，而不是由神经 head 逐车预测。

因此，arrival modeling 是当前实现中的非神经外层机制：世界模型提供 edge travel-time/traffic latent，vehicle lifecycle wrapper 用这些读数推进 per-vehicle proxy state。这个设计把宏观交通转移学习和 per-vehicle bookkeeping 解耦，降低了 world model 的输出复杂度。

## 3. Vehicle-level Actor-Critic

下游策略学习使用 vehicle-level actor-critic，而不是 rollout GRPO。策略每个决策步为所有 active AV 选择 candidate route；世界模型将这些 vehicle-level choices 聚合成 edge pressure 并推进 latent traffic state。

### 3.1 Candidate Route Semantics

对每辆 active AV \(i\)，系统根据当前 edge \(e_i^t\) 和 destination \(g_i\) 生成 \(K\) 条候选路径。主线使用 `current_to_dest` semantics：候选集在每个决策点重新从当前位置搜索到目的地，因此不同车辆的 `route0` 不是全局同一条物理路线，而是该车辆当前候选集中的第 0 条路径。

候选路径由 edge travel-time estimates 定价。候选生成可使用 free-flow、decoded speed/BPR 或 route-head cost 等边成本来源；当前 full-cycle 默认候选生成侧偏向 free-flow/current-shortest-freeflow，以避免把候选集合本身过度绑定到未成熟 world model。

### 3.2 Vehicle Actor

VehicleRouteActor 输入每辆车的 route observation \(o_{i,t}\)。该 observation 包括：

- 每条候选 route 聚合的 edge latent/dynamic features；
- 当前 edge-to-destination route cost、relative cost gap 和 valid mask；
- 可选 global edge summary；
- 已规划 fleet pressure 或同 tick 内其他车辆选择带来的 overlap features；
- 时间进度、active vehicle ratio 等低维 context。

主线 actor architecture 是 route-shared scorer：对每个候选 route 使用共享参数网络打分
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
再通过 softmax 得到 route distribution：
\[
\pi_\theta(k\mid o_{i,t})=
\frac{\exp(\ell_{i,t,k})}
\sum_{k'} \exp(\ell_{i,t,k'})}.
\]
invalid candidate routes 被 mask 掉。共享 route scorer 的好处是每个 route slot 使用同一套偏好函数，降低 route0/route1 slot bias，并支持不同车辆的 candidate set 语义。

actor 输出的 per-vehicle route probabilities 会通过第 1 节中的 aggregation 变成 edge-aligned \(A_t^{\mathrm{AV}}\)。在真实环境评估时，route distribution 通过 argmax、sampling 或受控混合策略落成实际 rerouting 指令。

### 3.3 Centralized Critics

训练采用 centralized training with decentralized execution。actor 执行时只依赖每辆车的 route observation；critic 在训练中可以使用 richer global information。

系统包含两个主要 critic：

1. **Global latent critic.** 对 latent edge state \(Z_t\) 做 mean 或 attention pooling，并可拼接 action edge \(A_t^{\mathrm{AV}}\) 与 context，预测 step-level value：
   \[
   V_\omega^{\mathrm{global}}(Z_t,A_t,C_t).
   \]
   attention pooling 使用 learnable query token 读取 edge latent summary。

2. **Vehicle critic.** 对每辆车的 route observation 预测 per-vehicle value：
   \[
   V_\eta^{\mathrm{veh}}(o_{i,t},C_t).
   \]
   主线可使用 self-attention 让同一步的 active vehicles 互相通信，从而在保持 per-vehicle output 的同时提供 centralized credit assignment。

Global critic 更适合整体 rollout value bootstrap；vehicle critic 更适合为每个 AV 的 route choice 提供局部但 centralized 的 advantage signal。当前主线强调 vehicle critic 和 route-cost reward scope，而非旧的 actor-only global reward。

### 3.4 Actor-Critic Objective

在世界模型中生成长度 \(H\) 的 imagined rollout，主线默认 \(H=8\)。每一步：

1. actor 为 active AV 输出 route distribution；
2. route distribution 聚合为 \(A_t^{\mathrm{AV}}\)；
3. world model 预测 \(Z_{t+1}\)；
4. decoder/route-cost readout 计算 reward 和 route-credit signal；
5. critic bootstrap 构造 actor-critic targets。

Global critic 使用 lambda return：
\[
G_t =
r_t+\gamma\left((1-\lambda)V(Z_{t+1})+\lambda G_{t+1}\right),
\]
并最小化
\[
\mathcal L_V
=
\left\|
V_\omega(Z_t,A_t,C_t)-\mathrm{stopgrad}(G_t)
\right\|_2^2.
\]

Actor 的 score-function 主线可概括为
\[
\mathcal L_\pi
=
-
\mathbb E_{i,t}
\left[
\log \pi_\theta(a_{i,t}\mid o_{i,t})
\cdot
\mathrm{clip}\left(\hat A_{i,t},[-c,c]\right)
\right]
-\beta\mathcal H(\pi_\theta),
\]
其中 \(\hat A_{i,t}\) 来自 global/vehicle critic 的 return baseline、route-cost credit 或二者组合。实践中还使用 PPO-style ratio clipping、advantage normalization、entropy floor 和 actor trust-region gate 控制策略更新幅度。

Route-credit reward 将同一车辆不同候选路径的 cost、planned pressure overlap 或 counterfactual externality 转换为 per-route advantage。主线默认将 route-cost reward 主要送入 vehicle critic，而不是直接把所有全局 reward 梯度穿过 world model 回传给 actor。这样能减少 actor 利用世界模型误差的风险，并提高多车 credit assignment 稳定性。

## 4. Online Collection, Offline Training, and Evaluation Loop

UrbanDreamer 的训练流程是一个 online sampling/evaluation 与 offline model/policy training 的闭环：

1. **SUMO/RouteRL collection.** 在真实仿真器中运行若干 episode。AV 使用随机、epsilon-greedy、固定 route 或当前 policy 混合产生多样路线选择；HDV 使用 Gawron/RouteRL 风格自适应背景。每个决策步保存 raw flow state \(X_t\)、`av_planned_edge_flow`、`hdv_edge_action`、`depart`、`context`、reward 和 episode id。

2. **World model training.** 从 flow shards 中采样长度为 \(T_{\mathrm{seq}}\) 的序列，训练 FlowProjector、CFM dynamics、decoder heads 和 neural HDV response head。主线 default full-cycle 使用 edge-conditioned graph condition、episode-response HDV context、decoder reconstruction、transition decoded supervision、BPR/route-cost supervision 和 \(t_{\mathrm{all}}\) supervision。

3. **Imagined actor-critic training.** 冻结或半冻结 world model，用 vehicle actor 在 latent environment 中执行 \(H=8\) step imagined rollout。actor 输出 vehicle-level route distribution，聚合为 edge pressure，world model 预测下一 latent state，critics 计算 returns 和 route-credit advantages。

4. **SUMO evaluation.** 将 actor checkpoint 放回真实 SUMO/RouteRL 中评估，记录 \(t_{\mathrm{all}}\)、\(t_{\mathrm{CAV}}\)、\(t_{\mathrm{HDV}}\)、completion、route usage 和 fallback statistics。

5. **Checkpoint/data refresh.** 根据真实评估选择 checkpoint；必要时把新策略产生的数据加入 world-model dataset，进入下一轮 collection-training-evaluation cycle。

该闭环把真实 SUMO 用作高保真但昂贵的 oracle，把 CFM world model 用作低成本、多步、可条件化的 imagined simulator。世界模型负责学习 aggregate transition 和 action externality，actor-critic 负责在该 surrogate 中优化 vehicle-level routing policy。

一个简化的主线算法如下。

```text
Algorithm: UrbanDreamer main training loop

Input: SUMO network, OD/departure schedule, route candidate generator

1. Collect real trajectories in SUMO:
   for each real episode:
       split vehicles into AV and HDV
       adapt HDV routes with Gawron/RouteRL state
       for each decision step t:
           choose AV candidate routes
           aggregate selected routes into av_planned_edge_flow
           save X_t, A_t^AV, D_t, C_t, HDV edge target
           advance SUMO

2. Train CFM world model offline:
   sample flow sequences from collected shards
   encode X_t with FlowProjector
   train CFM dynamics to match Z_{t+1}
   train decoder, route-cost, t_all, and HDV response heads

3. Train actor-critic in imagined rollouts:
   initialize latent state from dataset or sampled real state
   for h = 1..H:
       build per-vehicle current_to_dest candidate routes
       actor scores each candidate route
       aggregate route probabilities to edge pressure
       query episode-level HDV response background when enabled
       advance latent state with CFM world model
       compute route-cost and travel-time rewards
   update centralized critics and vehicle route actor

4. Evaluate actor checkpoints in SUMO and retain the best policy.
```

## 5. Discussion of Design Choices

### 为什么使用 edge-token world model

交通状态天然定义在 road-edge graph 上。与全局 MLP 或 OD-level assignment 相比，edge-token 表示能直接表达 bottleneck、spillback、route overlap 和局部拥堵传播。FlowProjector 的 GCN-style line-graph message passing 提供局部拓扑归纳偏置；Transformer dynamics 则允许非局部路径竞争和跨时间依赖。

### 为什么 action 是 pressure 而不是 per-vehicle route id

不同车辆的 route slot 只在其局部 candidate set 内有意义；`route0` 对不同车辆可能对应不同物理路径。因此，world model 不使用 route id 作为动作，而使用所有 vehicle route choices 聚合后的 edge pressure。这样 transition model 看到的是物理路网上哪里将被短期 planned demand 覆盖，从而与 traffic dynamics 的 edge state 对齐。

### 为什么 CFM 能在较少 episode 上工作

当前 CFM world model 不是开放域图像扩散模型。它学习的是固定路网、强条件、低维 edge latent 的 one-transition vector field。每个 episode 提供数十个 decision steps，每个 step 又提供 \(E\) 个 edge 的 dense supervision；因此 50 个 episode 约 3000 step 实际对应大量 edge-level transition samples。CFM 在这里主要提供稳定的 continuous transition parameterization 和可采样 latent dynamics，而不是承担高熵开放生成任务。

### 为什么需要 neural HDV response

如果只把 HDV 当作静态背景，actor 可能学到只优化 CAV 的路径重分配，而忽略 HDV 对 AV pressure 的响应。episode-level neural HDV response 把“AV 策略改变 HDV 背景”的慢时间尺度效应压缩成一个可学习 edge distribution，使 imagined rollout 同时包含 controllable AV action 和 adaptive human background。

