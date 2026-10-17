# IQL diagnosis and independent rerun

IQL in this repository means parameter-shared independent Q-learning for dynamic
vehicle agents. It is not offline implicit Q-learning. The three-city rerun uses
the explicit implementation profile `double_dueling_v2`; historical results use
`legacy_v1` and remain preserved.

## Observed problem

The old 2,000-episode runs used fixed-temperature softmax collection over raw Q
values, followed by greedy evaluation. Sampling 2,048 replay states from each
final checkpoint gives the following mean probability of taking the greedy action
at a branching decision:

| City | Mean greedy-action probability | Median legal Q range |
|---|---:|---:|
| Ingolstadt | 0.2561 | 0.02036 |
| Coulommiers | 0.2638 | 0.02606 |
| Montereau | 0.2757 | 0.04798 |

With four legal routes, collection therefore remains close to uniform despite
2,000 training episodes. Softmax is a valid exploration scheme, but this fixed
temperature does not move collection towards the greedy evaluation policy when
the learned action gaps are small.

The final 100 training episodes versus final fixed-seed greedy evaluation have
completion rates of 93.05% versus 4.42% (Ingolstadt), 80.97% versus 40.62%
(Coulommiers), and 99.54% versus 13.53% (Montereau). These use different demand
samples and different action-selection rules, so the gap alone does not identify
a coding bug or prove that exploration is the only cause. It does rule out using
the good sampled training return as evidence of a good greedy routing policy.

None of the sampled old replay rows had a forced action. Forced-event compression
is tested for the general collector, but is not presented as the explanation of
these three completed runs.

## Implementation changes

- Double DQN selects the legal next action with online raw Q and evaluates it
  with target raw Q. The shortest-path behavior prior never enters this target.
- A masked dueling scorer separates the pooled local state value from centered
  candidate advantages. Centering happens before projection; behavior logits
  omit the common value so value magnitude cannot erase small action differences.
  Only the same local candidate observations are used; no future demand is added.
- Training uses legal epsilon-greedy actions. Epsilon decays linearly from 1 to
  0.05 over 1,000 completed episodes and then stays at 0.05. Its actor buffer is
  synchronized to the CPU collection mirror and restored with checkpoints.
  Deterministic evaluation remains greedy; explicitly requested sampled
  evaluation retains its temperature semantics.
- Forced events do not consume branch-replay update draws. Their integrated
  rewards remain between consecutive real choices; the last choice still
  includes the whole remaining team return, including time after vehicle arrival.
  A terminal individual penalty, if explicitly enabled, is counted once.
- The optimizer, replay draw ratio, gamma=1, target synchronization in sampled
  transition units, reward definition and legal route space retain their previous
  settings. The profile, exploration settings and replay schedule are checked
  before loading weights. Legacy and revised runs cannot be silently resumed
  into one another.

These are disclosed algorithm changes, not proof of improved SUMO performance.
The new method is independent Q-learning implemented with Double DQN and dueling
heads; comparisons should label the implementation accordingly.

Primary algorithm references: [Double DQN](https://arxiv.org/abs/1509.06461) and
[Dueling networks](https://proceedings.mlr.press/v48/wangf16.pdf).

## Rerun protocol

`outputs/city_iql_v2_2000_20261017/` contains independent fresh runs for Ingolstadt,
Coulommiers and Montereau, 2,000 real SUMO episodes each, training seed 42. They
retain the old simulation configuration, network input hashes, current-to-
destination route space, global observed-mean objective, zero shortest-path
prior, learning rate 0.0002, replay capacity 200,000, batch size 512 and draw ratio
1. Checkpoints are evaluated every 100 episodes on demand seeds 10042–10044.
All three runs are scheduled concurrently and preserve episode checkpoints,
metrics and trajectories. A compactor compresses committed trajectories.

The completed-baseline curves and the diagnosis are reproducible using
`tools/plot_completed_baselines.py`; the original run provenance is retained in
its generated summary. The new launch and diagnostic records are under
`outputs/iql_repair_20261017/`.

Regression checks cover legal Double DQN selection/evaluation, dueling masking
and candidate permutations, behavior precision under a large common value,
epsilon distribution and annealing, CPU mirror synchronization, exact update
continuation, legacy rejection, team-tail accounting through forced events, and
a terminal two-action learning task with an explicit greedy ranking assertion.
