# Baseline learning repair and Dreamer inference optimization

User authorized implementation after the algorithm audit. Production city workers remain stopped. Preserve all old archives, source snapshots and checkpoints. No new dependencies or long training launches.

## Ownership and implementation sequence

- Root: reward defaults/launcher protocol, CLI integration, documentation, integrated verification.
- PPO lane: ppo.py and PPO/IPPO tests; centered robust advantages, full legal-policy KL, explicit critic fit budget and best-fit protection, versioned checkpoint settings.
- IQL lane: iql.py and IQL tests; transition-scaled draw credit, transition-unit target tau, TD diagnostics and strict schedule checkpoints.
- MAMBA lane: mamba.py/core and MAMBA tests; physical imagination event loop couples local decision RSSM with macro global RSSM. Global training and imagination condition on same identity-fixed starting active cohort with per-road actions/observations. Fresh matched-time posteriors with burn-in; physical arrival; world-model warm-up; new checkpoint schema.
- Dreamer lane: dreamer.py and dedicated tests/benchmark; profile before editing, remove repeated small GPU actor inference/synchronization using frozen inference mirror where measured beneficial, preserve rollout/update counts and record device/RNG semantics.

## Cleanup and regression plan

Before each optimization/refactor, lock team-tail reward, actual N, zero-demand, legal actions and rollout scope with regression cases. Add semantic counterexamples for MAMBA clock/continuation/action-support mismatches and update-budget tests for IQL. Remove obsolete cached latent/continuation/pooled action paths only once replacement is protected. Reuse existing ActorMirror and physical imagination; do not introduce duplicate collector or motion loops.

All new trainer/launcher defaults use pure global rewards; explicit legacy timeout option remains to inspect old experiments. Protocol changes must reject silent continuation of old learning configurations. Do not change fixed evaluation demand seeds, Poisson rates or prior schedule.

## Verification

Run lane-specific tests first; integrate CLI and protocol tests; run complete repository pytest and compilation/diff checks. Then bounded real Toy end-to-end archive/update/resume for all affected methods including nonzero/zero population and actual MAMBA imagination after explicit small validation warm-up. Measure Dreamer before/after with identical frozen support/budget and report variability and cross-device RNG limitations. No city batch restart is part of this request.

Completion report: changed files/behavior, verified results, measured performance and remaining modeling limits. Test success does not establish convergence.
