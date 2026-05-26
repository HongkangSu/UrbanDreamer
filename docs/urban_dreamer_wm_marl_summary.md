## 问题定义

- 参与者 CAV（自动驾驶汽车）为可控智能体，HDV（人类驾驶汽车）为自适应背景交通。
- 状态：路网边级状态，包括 volume、travel time、拥堵结构、车辆当前位置、OD 需求和历史 latent。
- 动作：每个决策步为每辆 active AV 从候选路径集合中选择一条 current edge -> destination 路径；聚合后形成 edge-level action。
- 转移：真实环境中由 SUMO/RouteRL 动态网络加载产生；训练中由 CFM world model 近似。
- 目标：降低 CAV 与全体车辆 travel time，主要指标为 `t_CAV`、`t_HDV`、`t_all`。

核心不是静态 OD assignment，而是车辆在行驶过程中反复根据当前交通状态和其他车辆行为更新路径选择。本任务中 CAV 策略不能只优化自身局部路径，还会通过拥堵外部性影响 HDV 和全局流量。

## 整体模型结构

当前 UrbanDreamer 路线可以分为三层：世界模型、vehicle-level MARL actor，以及 reward / advantage 训练目标。

### 1. 世界模型

世界模型输入是路网 edge-level 状态和 action：

- `S_t`：每条 edge 的交通状态，包括速度、流量、占用等动态特征。
- `A_t`：AV 的 planned edge flow，即当前决策时刻所有 CAV 选择路线后聚合到 edge 上的计划流量。
- `depart/context`：出发车辆、时间上下文等外生信息。

世界模型主体包括：

- `FlowProjector`：把原始 edge flow 编码成 edge-token latent。
- `Graph-conditioned CFM dynamics`：根据历史 latent、AV planned action、depart/context 预测下一步 latent。
- `Decoder / heads`：从 latent 解码速度、流量，并预测 `t_all`、`t_CAV proxy` 等全局指标。


### 2. MARL Actor

当前训练的是 vehicle-level actor：

- 每辆 CAV 从自己的候选 route 中输出 route distribution。
- actor observation 包含同一 tick 已经选择的 AV route信息，以及候选 route 的负载。


### 3. 训练目标

当前最好版本主要使用：

- wm-level GRPO advantage
- vehicle-level route-cost reward


## 实验结果

| method | t_all | t_CAV | t_HDV |
|---|---:|---:|---:|
| only route0 | 4.4120 | 4.3890 | 4.4274 |
| freeflow_greedy | 4.5531 | 4.5571 | 4.5472 |
| pure HDV | 4.3913 | N/A | 4.3913 |
| live_greedy | 4.3888 | 4.3398 | 4.4214 |
| **WM+MARL** | **4.3402** | **4.2813** | **4.3794** |



