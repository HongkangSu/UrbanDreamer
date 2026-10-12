# Disagreement penalty × multi-step supervision gate

Status: bounded design specification, 2026-10-12. No source implementation or
training is authorized by this artifact alone. The current assignment authorizes
this planning file only; campaign execution authority remains with the leader.

## Scope and evidence

The requested MS gate weights **multi-step world-model supervision loss**. It
does not truncate policy imagination, stop SUMO, change legal actions, filter
policy returns, or change the physical reward. Earlier rollout-truncation advice
is superseded for this experiment; truncation is a separate optional hypothesis.

Inspected local evidence:

- `tools/train_component_world_model.py::supervised_loss` implements
  `recursive_fm`, `sample_ms`, and `mean_ms`. The last two add endpoint MSE to
  recursive FM; no uncertainty gate exists. `sample_ms` averages squared sample
  errors; `mean_ms` scores the sample mean. These are different learning problems.
- `urban_dreamer/models/world_model.py::recursive_fm_loss` preserves gradients
  through generated recursive endpoints, with detached reference targets. Its
  `depth_weights` are normalized by their sum and apply to FM/prior loss, so they
  are NOT a suitable shortcut for the proposed MS-only, fixed-denominator gate.
- Searched current docs/plans, sibling `../worldmodel_AAAI_2027/sections/04_method.tex`,
  `urban_dreamer_method_section.tex`, and `../../worldmodel_kdd/sections/{method,
  training pipeline,mathematical details,experiments}.tex`. No predictive-
  disagreement MS-loss gate definition was found in those inspected sources.
  They discuss one-step training and repeated multi-step imagination; do not
  claim this proposed formula reproduces an established manuscript component.
- `../ablation/online_wm_dreamer_actor_critic.py::_actor_gate_passes` gates actor
  update counts from diagnostics. It is a different historical mechanism.
- Genuine RL already exists in `tools/run_component_online_policy.py`, via
  `ComponentPilotTrainer` and `OnlinePilot`: real SUMO collection, online WM
  updates, actor/V/Q parameter-change checks, paired evaluation, strict resume.
  Admission currently requires `mode=single_step,horizon=1`; its online loss is
  also single-step. MS checkpoint admission requires an explicit new protocol.

## Primary 2×2 experiment

Use the same stochastic dynamics family, frozen codec, initialization/data seeds,
replay schedule, cohort labels, optimizer, solver, sample count, and update budget.

| Arm | Actor risk penalty | MS endpoint-loss gate |
|---|---|---|
| 00 | off | off: every endpoint weight is 1 |
| 10 | on | off: every endpoint weight is 1 |
| 01 | off | on: detached disagreement weights |
| 11 | on | on: detached disagreement weights |

All arms use the SAME MS objective; single-step versus MS is a separate factor.
Primary recommendation: `mean_ms`, avoiding sample-MSE's explicit spread penalty.
Choose this before results; a `sample_ms` replication is separately labeled.

Minimal staging: train matched ungated/gated offline MS checkpoints for each WM
seed, then run penalty-off/on genuine policy learning from each checkpoint.
Use the same single-step online WM adaptation in all four cells and label the
factor **offline MS supervision gate**. This is the smallest integration with
the present runner. If the intended factor must persist online, add and test the
same cohort/MS objective in ALL four online arms before execution; do not enable
MS online only in gate-on arms. Report stage explicitly, not just `gate=true`.

Separate component seed from policy seed: the current runner conflates both in
`--seed`. Pair policy initialization/evaluation demands across cells; report at
least three independent initializations for final claims. Existing screens and
small smokes are not converged policy evidence.

## MS gate definition and gradients

For batch size B, horizon H, latent width d, E roads, K endpoint draws, and fixed
cohort targets y[b,h], recursively generate z[k,b,h] under recorded cohort actions
and causal sparse action events. Each draw retains its own recursive trajectory.

Primary endpoint error:

    e[b,h] = mean_(edge,channel)((mean_k z[k,b,h] - y[b,h])**2)

For the separate sample-MS variant, average squared error over k instead. Do not
switch between these variants inside a gate comparison.

Compute an independent no-gradient recursive ensemble from the SAME initial
cohort and recorded actions, using a private checkpointed RNG stream. Its
per-depth endpoint dispersion is:

    u[b,h] = mean_(edge,channel)(Var_k(z_score[k,b,h], unbiased=True))

This is accumulated recursive predictive dispersion; it is not one-step
conditional disagreement at a common predecessor and not parameter uncertainty.
Do not mix definitions in calibration, reports, or reward-penalty normalization.

Freeze depth-specific thresholds tau[h] from an identified disjoint development
calibration corpus and a named reference checkpoint before gated training. Record
sample count, solver/precision, codec/graph IDs, percentile rule, realized numeric
thresholds, calibration demand identities, and checkpoint SHA. Never select a
threshold using the target residual of the window being weighted or final test
outcomes. No threshold values are invented in this design; a filled numeric
calibration artifact is a launch prerequisite. Positive finite values and a
declared zero-dispersion fallback are required.

Minimal hard gate, version `ms_endpoint_disagreement_gate_v1`:

    w[b,1] = 1                         # retain one-step endpoint anchor
    w[b,h>1] = stop_gradient(1[u[b,h] <= tau[h]])
    L = L_recursive_FM + alpha * sum_(b,h)(w[b,h] * e[b,h]) / (B * H)

Gate-off sets every w to 1. Keep the recursive FM and prior terms, codec,
auxiliary readout, targets, action conditioning, recursive trajectory generation,
and alpha identical. Do NOT divide by sum(w), surviving samples, or gate mass.
For padded variable horizons, replace B*H with the pre-gate valid-target count;
do not change which targets are valid based on disagreement.

Stop gradients through the scorer inputs, score ensemble, thresholds, and weights;
retain gradients through the complete supervised recursive path. Rejecting depth
h does not detach its state: accepted later depths may backpropagate through it.
Do not stop recursion or resample until a sample passes. Both arms consume the
same scheduled windows and score-ensemble compute/RNG streams.

Detached current-model scores still permit indirect gate collapse as training
changes uncertainty. Log it; never silently unmask or retune thresholds. A frozen
scorer is a separately declared robustness variant, not an unnoticed replacement.

## Exposure and statistical interpretation

Log per depth: raw windows, positive-weight windows, sum(w), sum(w**2), acceptance,
u quantiles, tau, ungated error, weighted MS contribution, full FM/prior losses,
gradient norm, samples/NFE and wall time. Report effective exposure sum(w) and
ESS=(sum(w)**2)/sum(w**2), with zero convention; retain cumulative counts and
congestion/demand strata. Fixed denominator intentionally changes auxiliary-loss
magnitude. A matched constant-weight control is optional and separately budgeted
to distinguish uncertainty selection from merely reducing alpha.

Single-model sample dispersion is not identified epistemic uncertainty. For
sample MSE, mean_k ||z_k-y||² = ||mean(z)-y||² + (K-1)/K * sample_variance: apparent
disagreement/sample-error association partly follows by construction. Independent
score draws avoid shared Monte Carlo coupling but not all state-difficulty or
finite-mean-estimation effects. Preserve existing independent horizon-1 risk
diagnostics in `evaluate_component_world_models.py::evaluate_condition` and
`summarize_component_predictions.py`; longer-horizon same-ensemble diagnostics
cannot be relabeled independent. Add independent recursive score draws when
validating this MS gate, with congestion controls and demand-cluster statistics.

## Reward penalty remains a separate bounded mechanism

`critic_budget.py::remaining_cost_budget` gives B(t)=(D-t)*active/(60*started).
Physical rewards/returns remain explicit vehicle-minutes and physical V/Q stay
inside [-B,0]. Preserve config/trainer checks forbidding nonzero legacy
`disagreement_weight` with `bounded_cohort_v1`. Clipping variance alone cannot
make physical cost plus risk fit B when physical cost already saturates B.

Use separate risk V/Q and target risk V with independent parameters/optimizers
and contract, then combine ONLY actor advantages: A=A_physical+beta*A_risk.
Risk-on optimizes a regularized training objective; real evaluation remains
physical cost, completion and censoring. It does not preserve the exact same
optimal policy by mathematical identity.

Example bounded endpoint risk: q=min(u_one_step/s,1), with separately calibrated
fixed s, private score RNG, and no charge on partial terminal windows. Risk V/Q
may use -K*sigmoid(raw), where K bounds remaining chargeable full-window endpoints.
At macro start K=floor((D-t)/30); a late event includes the entire current pending
endpoint charge using its window-start phase. True terminal future risk is zero.
Validate physical and risk targets separately; never clip invalid physical
targets or reinterpret risk weights under the physical checkpoint marker.

Reuse causal detached candidate baselines from `_complete_window_event_q`, retain
late-event physical cost removal, and retain the whole current endpoint risk in
the risk event return. Preserve target-V bootstrap/terminal semantics. Declare
combined advantage normalization/clipping and entropy scaling; log their scales.
Normalized mixed rewards could fit [-B,0] after rescaling, but then V/Q cease to
predict physical minutes and reward/entropy scale changes confound the factor;
this is rejected for the primary experiment.

## Implementation and acceptance specification

1. Add an experimental gate helper and calibration artifact reader; extend
   `supervised_loss` at its endpoint-MSE reduction only. Keep production FM
   default behavior unchanged. Add gate identity, threshold SHA and RNG to
   protocol/checkpoints and `CONTINUATION_FIELDS`; incompatible resumes fail.
2. Extend component policy admission explicitly for selected MS mode/horizon,
   source gate protocol and hashes. Do not relabel MS checkpoints single-step.
   Keep the chosen online supervision stage common to all four cells.
3. Add separate experimental risk models, optimizer/target updates, actor
   combination and strict state schema in the experimental trainer; begin serial
   only. Reject parallel execution until matching target/RNG tests exist.
4. Algebra/gradient tests: all-one weights exactly recover existing loss and
   gradients under fixed draws; partial masks equal hand calculations with fixed
   denominator; all later depths rejected retain FM and depth-one anchor; no
   scorer/weight/codec gradients; later accepted depths still train earlier
   recursive transitions. Exposures/NFE and schedules stay unchanged.
5. Causality/data tests: identical cohort targets and newborn exclusions across
   cells; gates never consume target residuals; legal event prefixes and draws
   are unchanged; no true future SUMO state enters policy or score conditions.
6. RNG/resume tests: independent score draws do not advance main trajectory RNG;
   exact checkpoint replay restores masks, losses, exposure counters, private RNG,
   risk target/optimizer state; threshold/checkpoint/source mismatch rejects.
7. Bounds tests: saturated physical budget plus positive risk never alters the
   physical target; late events, empty/terminal states, partial tails and risk
   count budgets; beta=0 exactly preserves physical actor path under fixed data.
8. Bounded real SUMO smoke per cell: actual actor/V/Q steps and parameter deltas,
   finite losses, zero physical/risk bound violations, immutable codec/sources,
   paired demand hashes and complete strict reload. Run no city-scale comparison
   until these pass and gate coverage/risk calibration are reviewed.

Deliverables are distinct: calibrated gate behavior, valid risk estimator,
genuine four-cell learning evidence, and physical-policy performance with paired
uncertainty. Range compliance, reduced training loss, or a fixed-policy rollout
does not establish any of the latter claims.
