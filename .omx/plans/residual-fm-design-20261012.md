# Frozen regression center + normalized residual FM

Implementation-ready experimental specification, 2026-10-12. This file is the
only write in this task. No source changes, scale fitting, training or experiments
were performed. This design does not alter the concurrently launched
`isolated_prior_graph_fm` experiment or any frozen production source.

## 1. First variant and immutable source

Variant identifier: `residual_fm_regression_center_v1`. Stage-one scope is
single-step, fixed-codec component prediction; recursive EVALUATION is supported,
recursive TRAINING, MS gates, risk penalties and parallel policy imagination are
out of scope. This is a lightweight staged distribution model, not a pure
loss-only comparison with the six-layer production transformer.

Use the completed, previously evaluated deterministic `regression` checkpoint:

    outputs/component_experiments_20261011/training/regression_seed42_single_step_v1/latest.pt
    SHA256 b468124d6f84d85155360637bf7f49732a93ab68671fefbd551f0ec0222fdece

Read-only SHA verification matched on 2026-10-12. Its recorded budget is 2,048
Adam updates, batch 16, 32,768 transition exposures, seed 42, data seed 20261011,
learning rate 1e-4, BF16 training; recorded training time is 939.439179 seconds.
Do not substitute the moving `convergence_v1/.../latest.pt` checkpoint.

Require exact parent campaign/config/manifest identities and codec state equality
with the source. The source training manifest identity is
`f1f1b010555b1fe3adfdf0cd792baac27cdda879f5d4fc89eb9ddeb928843d04`.
The parent codec checkpoint SHA is
`2f63f0b2a5fe18f3a3bd9be27ffd1e63a2755464e7f2944e1227334f0ffe6af6`.
Import codec and regression dynamics only. Do not import actor, critic, old
reward targets or regression Adam moments into the residual optimizer.

## 2. Distribution and training target

For encoded current traffic z, recorded action a and sparse action events E:

    m = frozen_regression.transition(z, a, action_events=E)
    r_target = (z_next - m) / s
    eps ~ Normal(0, I_(roads*64)); tau ~ Uniform[0,1]
    r_tau = (1-tau)*eps + tau*r_target
    v_target = r_target - eps
    L_residual_FM = mean((v_theta(r_tau,tau; z,m,C(E)) - v_target)**2)
    r_sample = ODESolve(v_theta, initial=eps, tau:0->1)
    predict_next = m + s*r_sample

All z, z_next, m, C(E), and s are detached. Only the new residual network learns.
The loss mean is over the original batch, all roads and all 64 latent channels.
No endpoint MSE, sample-mean MSE, prior loss, adversarial term, learned s, target
clipping or zero-mean residual constraint is added in v1. Numerical nonfinite
values fail explicitly. The center is an approximate deterministic prediction;
the residual distribution may learn bias correction, so the final predictive
mean need not equal m. Do not call m the true conditional mean.

Source support is full latent space, not a fixed low-frequency graph subspace.
eps is unit coordinate Gaussian in normalized residual coordinates; physical
source covariance is diag(s**2), repeated across roads. Report this covariance
and expected source energy E*sum(s**2). Do not claim source-energy matching with
the existing spectral FM: normalized residual modeling is a different parameterization.

## 3. Training-only scale artifact

Fit once, without optimization or validation access, on 4,096 training windows
selected by a private NumPy generator seed 20261012, using the existing legal-
window sampler (episode probability proportional to legal-window count, starts
uniform within episode). Record every selected episode/start and duplicates,
their digest, manifest identity, count and demand groups. Horizon is 1 and targets
come from `prepare_transition_targets`, exactly as in the source regression.

Use FP32 codec/center inference and FP64 streaming squared-error accumulation:

    raw_s[j] = sqrt(mean_(windows,roads)((z_next[j]-m[j])**2))
    global_rms = sqrt(mean_j(raw_s[j]**2))
    floor = max(1e-4, 0.05*global_rms)
    s[j] = max(raw_s[j], floor),  j=0..63

Use RMS about zero, not centered standard deviation: center bias remains a
learnable residual. Use one per-channel scale pooled across roads, not per-road
parameters. All scales must be finite and positive; store raw values, floor,
floored-channel count and FP32 s[64] in JSON plus a persistent model buffer.
Hash the artifact and validate JSON/buffer equality. Never refit on resume or
change s online. Dataset/center/codec changes require a new artifact/run.
Record these 4,096 calibration-only forward exposures and their wall time.

## 4. Minimal residual architecture and resource cost

Reuse the existing regression dynamics object as a frozen child `mean_dynamics`;
do not deepcopy a second full transformer. The world model retains ONE frozen
codec outside the adapter. The frozen mean's condition encoder computes C(E)
once per transition; use the SAME condition for its deterministic center call
and residual context, preserving the regression forward computation exactly.
Implement `mean_and_condition` by the no-prior regression formula in
`ExperimentalDynamics.distribution`: C=mean.encode_condition(a,E), reference=
mean.reference_proj(z), delta=mean.backbone(a,packed_latents=z,flow_time=0,
condition=ConditionEncoding(C.edge_tokens,reference)).latent_velocity, m=z+delta.
Assert parity with the source transition; do not call an extra center forward
just to recover conditions. Reject graph_regression/Gaussian mean sources in v1.

New `ResidualVelocity` module, all trainable, hidden width 128:

    c = Linear64to128(z) + Linear64to128(m) + Linear256to128(C(E))
    h = c + Linear64to128(r_tau) + Linear33to128(time_features(tau))
    h = EdgeGNNLayer128(h, adjacency)
    h = EdgeGNNLayer128(h, adjacency)
    v = Linear128to64(h)

Use the existing directed normalized adjacency and existing `EdgeGNNLayer`
implementation, including its residual/LayerNorm behavior. Time features are
`[tau, sin(pi*j*tau), cos(pi*j*tau) for j=1..16]` (33 dimensions). Use standard
PyTorch Linear initialization for context/input projections, zero output-head
weight/bias, and record initialization seed. Zero initial velocity produces the
declared m+s*eps prior; it does NOT initially reproduce the deterministic center.
Cache c within a solve; never cache its trainable graph across optimizer updates.

This adds about 0.14M parameters (implementation reports exact count), compared
with 8,140,368 parameters recorded for the frozen regression dynamics. Frozen
FP32 mean weights occupy about 31.1 MiB, excluding codec/buffers/activations; they
have no gradient or optimizer tensors. Added FP32 parameters+gradients+two Adam
moments are roughly 2.2 MiB, excluding activations/workspaces. No double-backbone
memory advantage is claimed without runtime measurement. Frozen mean still costs
one full transformer forward per transition, plus residual NFE cheap graph calls.
Any later full-transformer residual network is a separately budgeted variant.

The two-hop residual GNN plus globally informed m/C is a deliberate capacity
restriction. It may fail to model nonlocal residual dependence; failure is not
proof that residual flow matching as a class fails. Missing signal/history inputs
remain missing; this design does not silently add new traffic observations.

## 5. Adapter, solver and evaluator contracts

Prefer an isolated `urban_dreamer/models/residual_dynamics.py` module. Adapter
owns frozen `mean_dynamics`, trainable `residual_velocity`, persistent s and
versioned extra state. Expose `n_edges`, `event_encoder` and `encode_condition`
through the frozen mean for existing event-Q feature construction. Override
`train(mode)` to keep mean_dynamics.eval() even when the residual head trains.
Every center/condition call uses no_grad; neither module mode nor codec gradients
may drift. The active-fraction readout remains outside this adapter.
Implement the existing chunk-profile setters by configuring the frozen mean's
forward chunks and leaving its activation checkpointing disabled; the small
residual graph head needs no transformer checkpointing/compiled-block setup.

Preserve public methods:

- `transition(z, action, noise=None, steps=2, solver='euler',
  picard_iterations=None, picard_batch_chunk_size=64, action_events=None)` accepts
  (B,E,64) or (B,T,E,64), returning the same latent shape. Explicit noise has full
  latent shape, never spectral coefficients. Validate shape/finite inputs.
- `latent_disagreement(...,n_samples,noise=None,...)` returns `(samples, scalar,
  edge_variance)` matching production layouts and unbiased variance convention.
  Compute variance AFTER m+s*r conversion, in original latent units. Explicit
  noise shape is (S,B,E,64) or (S,B,T,E,64); sample ordering is stable.
- Compute mean/context once per input state and repeat these over noise samples;
  preserve original joint draw ordering when chunking. Recompute m and C at each
  recursive physical transition from its generated current state.
- v1 supports Euler only: FP32 r/s/m and integrator; time=node/steps and step=1/steps.
  Reject midpoint/Picard requests explicitly rather than silently replacing them.
  Later solver support requires independent numerical tests.
- Use ordinary torch RNG semantics for noise=None; explicit noise consumes no
  random numbers. Builders and scale fitting use forked/private streams and
  restore caller RNG. Save torch CPU/CUDA, data RNG and initialization provenance.

Add explicit residual dispatch BEFORE generic `.endswith('fm')`/regression loss
branches in `experimental_loss` and `variant_metadata`. Add the variant to the
factory with construction metadata; do not run `independent_transition_fm_loss`
or `prior_start` on normalized residual states. Do not fabricate a graph_prior
with rank=E merely to satisfy legacy metadata/parallel noise assumptions.

The factory must support strict self-contained restore: construct a regression
mean shell and small residual head from architecture metadata, then strict-load
the embedded mean weights, residual weights and s. Fresh initialization requires
the verified mean source; resume/evaluation must not require a mutable external
mean file. Keep the archived source SHA as ancestry, and preserve the original
regression extra-state marker inside the frozen child.

Update evaluator factory call in `evaluate_component_world_models.py::main` to
pass architecture metadata. Preserve `sample_trajectories` and physical scoring
APIs. Add a `center_only` prediction row as an explicit ablation; do not replace
the sample mean with m when computing predictor mean metrics.

Initial policy use is serial eager FP32 only. `configure_imagination_inference`
needs an explicit residual-adapter branch reporting its true eager profile;
production velocity wrapping must not pretend to cover this different forward
path. Reject worker/parallel modes because they assume graph_prior.rank.
Filtering trainable parameters is required in component and online optimizers:
`[p for p in dynamics.parameters() if p.requires_grad]`. Online updates, if later
authorized after offline admission, update residual head only with s/mean/codec
frozen; do not leave frozen mean parameters in an ambiguous optimizer contract.

## 6. Checkpoint and exact training plan

Extend component protocol with `residual_contract_version=1`, mean source
variant/seed/SHA/update/exposures/training time, codec identity, scale artifact
SHA/sampler/values, residual initialization seed, architecture, target/noise units,
solver, precision and trainable/frozen counts. Record mean-pretrain and residual
update/exposure counters separately and their totals; do not relabel parent
updates as residual optimization. Embed all inference weights/scales. Resume
compares these fields exactly; continuation preserves them in CONTINUATION_FIELDS.

Execution order once implementation is approved by the leader:

1. Run unit/small-model acceptance tests below (no live producer dependency).
2. Fit and freeze the 4,096-window training-only scale artifact; verify no held-out
   identities, zero parameter mutation and unchanged caller RNG.
3. CPU/small-model gradient smoke; then admitted real-graph GPU smoke: 8 updates,
   B=2, seed42, FP32, Adam1e-4, clip1, isolated output; no claims from smoke.
4. New fresh development run (do not extend the B=2 smoke): 2,048 residual updates,
   B=16, horizon1, Adam1e-4, no weight decay, clip1, BF16 residual-head autocast,
   FP32 codec/center/normalization/interpolants/loss, two threads, data seed
   20261011 and residual seed42. Reuse the legal-window sampling contract, record
   all sample identities, checkpoint every128 and milestones512/1024/1536/2048.
   Gate admission on available GPU memory; do not terminate production jobs.
   Require BF16 hardware support without silent fallback. Comparators retain
   their recorded BF16 training protocols; all inference/scale comparisons use
   FP32. Record the frozen-center precision exception rather than claiming every
   forward operation has identical precision or cost across architectures.
5. Evaluate fixed development conditions/demand hashes using K=16 samples,
   chunk4, horizons1/4/8, Euler steps20 (primary), and steps100 numerical audit.
   Use existing count/speed physical MAE, CRPS, energy score, interval coverage/
   width, cohort excess and endpoint-codec/persistence baselines. Check inference
   solver sensitivity before interpreting probability scores. Keep independent
   horizon1 disagreement ensemble and dataset grouping conventions.
6. Report this as seed42 development evidence. Further seeds or policy learning
   require the leader's subsequent execution plan, not automatic expansion here.

## 7. Budget fairness and decision criteria

The seed42 initial center has U_m=2,048 updates/X_m=32,768 exposures. The proposed
residual stage has U_r=2,048/X_r=32,768: total 4,096 optimization updates and
65,536 transition exposures, plus 4,096 calibration-only forwards and common
codec/data-generation cost. Report total and incremental wall time, total/frozen
parameters, peak memory and center+residual inference NFE separately.

Required comparisons before any advantage claim:

- frozen center at U_m (incremental effect; explicitly additional training);
- ordinary regression continued to U_m+U_r with its original optimizer/RNG and
  completed-budget checkpoint (equal total update/exposure budget);
- from-scratch FM at U_m+U_r with common codec/data and full protocol provenance;
- mean-only center+s*training diagonal Gaussian reference, requiring no residual
  optimization (does learned residual dependence outperform simple noise?).

Equal updates are not equal FLOPs or wall time, and residual head capacity is
different; show both counts and measured time. Existing convergence checkpoints
can be used only at exact declared milestone budgets; never choose post hoc best
update using the reporting set. Existing graph/no-prior/isolated-prior screens
remain separate source-identical comparisons. Primary evidence for distribution
benefit is proper physical-unit scoring plus calibration; lower latent FM loss
alone is insufficient. Do not assert improvement if extra pretraining explains
the comparison or mean error/censoring masks differ. Negative results are retained.

## 8. Required tests and failure handling

1. Affine units: fixed eps/tau, known m/s/target yield exact r_target, interpolant,
   v_target and loss. A constant analytic residual velocity integrates to known
   m+s*r; zero velocity returns m+s*eps. No accidental second addition of z/m.
2. Full support: explicit noise outside the old low-frequency graph span survives
   with zero velocity; residual source shape is E*64, independent of prior rank.
3. Gradients: frozen codec/mean/event encoder have no gradients or weight changes;
   residual head has finite nonzero representative gradients/optimizer updates.
   Context uses both z and m plus C; tau/r perturbations affect the velocity.
4. Frozen behavior: mean.train is false after parent train(); center matches the
   source regression at fixed FP32 inputs/events; no duplicated codec/backbone
   registration; optimizer membership equals trainable residual parameters only.
5. Scale integrity: training-only identities, deterministic subset/digest,
   FP64 accumulation, zero-residual floor, nonfinite failure, exact checkpoint
   buffer/artifact match and rejection of source/codec/scale mismatches.
6. API: squeezed/unsqueezed inputs, empty/legal sparse events, sample/chunk ordering,
   explicit-noise determinism, original-unit disagreement (including s**2), and
   unsupported solver/worker rejection; independent conditioning per recursive step.
7. Resume: exact outputs/loss/update after full optimizer+RNG restore; self-contained
   evaluation after external mean source is unavailable; no old graph-FM state
   accepted as residual state. Initialization/statistics do not shift policy RNG.
8. Physical diagnostics: output distributions scored on identical target masks;
   no clipping of negative/over-cohort decoded predictions before error reporting;
   parameter/memory/time counts include mean and residual stages. Euler20/100
   discrepancy is reported, not mistaken for model stochastic uncertainty.

Failure modes include underestimated in-sample residual scale, shifted recursive
input states, extra mean-forward cost, inadequate two-hop residual capacity,
decoder-amplified latent noise, small-sample mean bias, and persistent missing
history/TLS/cohort-commitment inputs. Revisions change explicit protocol/version;
none justify silent validation-scale fitting, endpoint-MSE addition or center
fine-tuning inside v1.
