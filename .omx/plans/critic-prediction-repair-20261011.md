# Critic prediction repair — approved isolated implementation plan

## Principles
- Preserve physical reward and undiscounted finite-horizon objective.
- Encode exact physical constraints instead of requiring the critic to rediscover them.
- Improve target grounding as well as output range; bounds alone do not demonstrate policy improvement.
- Keep serial/parallel definitions identical and changes explicit in checkpoint provenance.

## Drivers
1. Six of eight observed-state predictions violate feasible remaining-cost bounds.
2. Most 240s rollouts use the current online critic as their own bootstrap target.
3. A valid value must still distinguish actions and support gradients; clipping an old -56 output is insufficient.

## Options
A. Clip existing V/Q to feasible ranges: smallest change but creates zero gradients for severely invalid values and can hide failed learning. Reject as standalone fix; allow diagnostic logging only.
B. Predict bounded remaining-cost fractions in V and Q, with explicit remaining-time budgets; add a frozen/EMA target V and terminal-stratified starts. Preferred staged repair. Changes critic output meaning and initialization; requires explicit migration and experiments.
C. Predict a survival curve for current vehicles and integrate expected active time: interpretable temporal structure, but substantially larger architecture and labeling changes. Defer until bounded critic fails to provide useful action ranking.
D. Keep physical-minute outputs, improve critic-specific feature conditioning and provide terminal-reaching/same-policy real-return grounding. Strong alternative: directly addresses learning/data failure and avoids sigmoid endpoint saturation. It cannot structurally exclude impossible values. Include as a diagnostic control rather than treating clipping as the only alternative.

## Proposed output contract
For each state/event let B=(deadline-now)*active_count/(60*seed_started_count), computed by trainer from actual physical counters, not inferred from decoder volume or final demand. B is a detached explicit argument.
V=-B*sigmoid(f_V(z, normalized_context)); Q_a=-B_event*sigmoid(f_Q(event_features_a)). Outputs and actor advantages remain mean minutes; raw heads predict unitless cost fractions. B=0 forces exact zero at deadline and with no active vehicles. No sigmoid transform of the old learned raw values without retraining. No additional risk term may silently use these pure-physical bounds.

Use physical V/Q for returns and actor baseline. Initially retain physical-minute MSE for both V and Q to isolate the output-contract change. Log fraction error where B is safely positive, but do not silently switch to fraction MSE: it reweights the original loss by 1/B^2, particularly near the deadline. A later V-only fraction-loss ablation can be tested explicitly. Candidate masking remains unchanged; gradient flows to critic parameters only. Both serial and parallel must derive the same per-event budget before motion changes it.

Priority is V, whose bootstrap changes actor return targets. Q is a detached counterfactual policy-expectation baseline, not an objective the actor directly maximizes. Baseline error can increase variance; actor advantage normalization/clipping complicates ideal baseline invariance. Do not claim poor candidate Q ranking alone proves policy-gradient bias. Keep explicit baseline-independence tests.

Compute B in FP32 from detached physical counters. N_started=0 with N_active=0 is an exact-zero empty state, without evaluating 0/0; active>0 with started=0 is invalid. At B=0 return exact zero; no division by B occurs in training loss. Fraction diagnostics mask rows with B<=1e-6 minutes BEFORE division; log their count. Only endpoint roundoff within 1e-7 seconds may map negative remaining time to zero; invalid times/counts fail. Initialize fresh heads near their nonsaturated interior and monitor both raw-head and hidden-layer saturation. Fraction-one targets are legitimate in nonarrival-heavy data and can saturate sigmoid; finite boundedness alone is not acceptance. Boundary/endpoint tests use explicitly declared FP32/BF16 tolerances.

Critic-specific features may omit historical C because it cancels from fixed-cohort future costs, while retaining C in real accounting/replay. Any conditioning change must be tested separately: the measured first Tanh layer is not saturated; second-layer saturation is 48%-97%; zeroing only logC in the old network worsens saturation. Do not apply an inference-only feature rescaling to old weights or assert raw context magnitude is the proven cause. No global change to actor inputs or shared observation schema.

## Target and sampling contract
- Bootstrap every intermediate lambda next-value and final event tail from a full detached target V copied at initialization; propose EMA tau=.01 once after each actual online V optimizer step. Current implementation performs one V step per pass: collect all original and auxiliary rollouts with a frozen target snapshot, fit V, update EMA, expose new target only to the next pass. If multiple V steps are later introduced, rollout targets still use their fixed snapshot. Tau is experimental, not assumed optimal. Event Q uses the same physical tail V; a separate target Q is not required by current target equations.
- Retain gamma=1 and lambda=.95. Terminal flags still remove bootstrap; architectural zero at deadline is additional protection.
- Keep the original uniform actor rollout stream unchanged. Add ceil(.25 * original_rollout_count) TERMINAL-REACHING V-ONLY rollouts per pass: for the current 16 original rollouts add four, so auxiliaries are 20% of the 20 V-training rollouts. At current replay coverage, about (4+16*.0635)/20=25.08% of V rollouts reach the deadline, absent early completion. Derive terminal pool from actual horizon and episode deadline. If pool empty, omit auxiliary rollouts and log the omission. Auxiliary tracks contribute only to V loss, never Q loss, actor loss, actor advantage normalization or actor sampling RNG; use a separately checkpointed auxiliary RNG stream. This intentionally reweights V fitting, not actor state occupancy. Ratio is an experimental setting, not a performance claim.
- Auxiliary randomness is isolated for NumPy replay selection AND Torch CPU/CUDA action/model draws; snapshot/restore global states if an API cannot accept a generator. V loss is the mean of per-rollout mean physical-minute MSE across original plus auxiliary tracks, matching existing reduction: four auxiliary tracks among twenty have 20% loss weight regardless of their lengths. Do not pool all macro transitions as an implicit alternative weighting.
- Do not directly fit current critics to greedy audit returns as if they were on-policy stochastic targets. Real-return auxiliary training requires a separate same-policy cohort label specification.

## Implementation sequence after design completion
1. Add a small shared budget calculation and bounded V/Q parameterization, normalized feature handling and config contract; preserve unchanged actor and WM interfaces where possible.
2. Wire all V, next-V, candidate-Q, executed-Q and baseline paths, including workers and late-in-window event times. Use target V for every bootstrap; update/save/restore it explicitly.
3. Add opt-in stratified seed sampling and diagnostics: output/target bound violations, normalized targets, terminal/truncated counts, time sensitivity, actual seed-time distribution.
4. Explicitly initialize new critic parameters/optimizers/target model for changed output contract in an isolated continuation artifact; preserve actor/world weights and RNG provenance. Existing critic weights cannot be silently reinterpreted. Do not restart live training as part of planning.
5. Run a bounded recovery experiment with actor frozen before any coupled actor update; compare unbounded baseline, bounded-only, then target/sampling additions. Do not claim success merely because values are bounded.

## Acceptance and tests
- All finite valid states/actions obey -B<=V,Q<=0; exact zero for B=0, including float32/BF16 use sites. Reject invalid counts/deadlines.
- Physical-minute losses finite at zero/tiny B; fraction diagnostics never divide unmasked zero/tiny budgets. Gradients nonzero for representative unsaturated interior targets; test saturation near endpoints without asserting impossible nonzero gradients at B=0.
- Verify cost(t,t')+B(t')<=B(t) for fixed cohorts. Combined with bounded target V this guarantees bounded one-step and lambda returns at gamma=1. Test arrivals inside windows, late/simultaneous events, partial windows and deadline. Inconsistent targets are diagnosed, not clipped away.
- A deterministic fixed-cohort environment with known remaining travel times validates Bellman/lambda target units and state/action ordering, not only interval bounds.
- Construct equal mean-pooled traffic/time/count features with different destination/cohort configurations. If V inputs alias their different true costs, document this representational limitation; a compact causal remaining-route summary is a separately tested extension, not a hidden reward prior.
- Perturb historical C holding physical state, clock, denominator and policy fixed. Future fixed-cohort costs are invariant; critic-only feature conditioning must not change actor/shared replay semantics.
- Terminal transitions use no bootstrap. Old reward risk modes cannot silently reuse physical-only bound semantics.
- Serial/parallel event budgets, returns, baselines, losses and gradients agree with shared frozen samples.
- With critic outputs held fixed, enabling auxiliary V-only samples leaves original actor sample identities, advantage-normalization inputs, loss and gradients unchanged; test independent RNG advancement and restore.
- Save/resume reproduces target-critic state, optimizer, new contract and seed sampling; old checkpoints rejected unless explicit migration is used.
- Audit held-out seed-time bins including t=deadline; report physical-return calibration, action ranking, and saturation fractions independently of bound compliance.
- Current eight-case audit provides provenance and regression evidence only; it does not establish policy improvement. Real closed-loop evaluation required after critic-only calibration and controlled actor unfreezing.

## Risks and mitigations
Sigmoid saturation: initialize near interior and inspect head gradients; do not reuse old -56 raw head. Population averaging makes action effects small: retain physical advantage units and measure action ranking separately. EMA can stabilize wrong predictions: terminal supervision and diagnostics remain required. Stratification alters training distribution: expose ratio and report distributions. Migration breaks critic semantics: fresh critics with recorded boundary and preserved actor/world; no silent resume.

## Decision record
Draft decision: option B, staged output/target/boundary changes with unchanged reward; include option D as a learning/grounding control. Reject clipping-only and discount changes as substitutes for finite-horizon correctness. Consequence: explicit new critic contract and isolated validation required before deployment. Preserve episode/entropy clocks when continuing; resetting them is a separate intervention. Follow-up: consider richer cohort representation or survival outputs only if calibration/credit assignment remains poor.

Experiment order: frozen actor/world current critic baseline; MATCHED fresh unbounded versus fresh bounded critics with identical shared-backbone initialization, features, replay, optimizer settings, update count and uniform sampling (document head initialization); then change feature conditioning alone if warranted; then add target V; then auxiliary terminal V-only rollouts. If budget requires combining target and boundary additions, label their combined effect. Eight low-arrival audit cases are regression data only; require successful and failing held-out same-policy cases and an exact-return toy problem before claiming learned value quality.

Learned-quality gate before actor unfreezing: freeze an evaluation-protocol artifact BEFORE fitting, with finite update/time budget, same-policy data split, toy absolute-error tolerance, physical-minute calibration noninferiority tolerance relative to both matched fresh-unbounded and all-remain baselines in each time bin, and required ranking accuracy on controlled pairs whose true return difference exceeds a declared label-noise threshold. Choose numerical tolerances from the frozen label precision/variance and downstream sensitivity, not after results. Implementation cannot advance to training without that filled protocol, and actor cannot unfreeze unless every declared gate passes. Exact-range compliance alone is insufficient. If meaningful action contrasts or same-policy calibration data are absent, result is inconclusive and requires additional diagnostic cases. Candidate-Q contrasts are reported separately from actual actor-gradient tests.

Architect review applied: distinguish V target bias from Q baseline variance; preserve loss units; specify budgets and Bellman invariant; acknowledge state aliasing and measured saturation; define EMA and overlapping mixture semantics; make migration explicit.
Critical review revisions: auxiliary terminal samples restricted to V only; matched fresh-unbounded initialization/control required; predeclared learned-quality gates block actor unfreezing; empty states and target snapshot timing clarified.

## Execution handoff
Available native roles: default/executor, test-engineer, architect, critic, verifier. Standard role presets previously failed model availability in this session; inherited native default agents can perform equivalent bounded roles.
Solo/ralph-style ownership: one implementation owner (high reasoning), one independent verifier. Native parallel/team-style ownership: models+config lane, training+replay lane, independent tests/review lane; coordinate shared schemas before parallel edits. In this App session prefer native agents; only if OMX runtime is explicitly available use `omx team 3:executor` or `$team` with this plan, or `$ralph` with PRD and test spec artifacts. Team verification must prove unit/integration parity, explicit migration/resume and isolated real rollout before shutdown; root verifies artifacts and live-job status.

Sequential reviews completed: Architect APPROVE, independent Critic APPROVE after one revision. Approved for isolated implementation and validation only; no live deployment/restart has been performed or implied. Numerical learned-quality thresholds must be frozen in the evaluation protocol before fitting.
