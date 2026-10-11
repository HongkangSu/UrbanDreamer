# Active goal handoff: execute the full UrbanDreamer experiment campaign

The user authorized experiments, evidence-driven structural fixes, and requested a
workspace commit before structure changes. Goal remains ACTIVE; do not mark it
complete after development screening. First preservation commit: `2a084d7` (483
pre-existing files). Current code includes many subsequent Lore commits. Existing
main experiments still run from their original frozen sources; none was stopped.

## Authoritative paths and live queues

Campaign root: `outputs/component_experiments_20261011`.
Use `queue_status.json` plus host `ps` to verify liveness, not stale status alone.

- `screen_v1`: eight variants, seed42, 2048 updates, batch16, 32768 window
  exposures each, GPU1 cap .08. At last audit graph_fm, no_prior_fm, regression,
  graph_regression completed; random_prior_fm running; Gaussian/r64/r128 follow.
  Controller PID 904468, tmux `component_screen_20261011`.
- `prediction_screen_v1`: automatically waits for each committed training result;
  8 demand groups, 3 traffic phases + separate empty stress, K16, Euler20,
  h1/4/8, GPU1 cap .04. Controller PID913157,
  tmux `component_predictions_20261011`.
- `numerics_v1`: COMPLETED matched old pretrained snapshot diagnostics at Euler20
  and100, four groups, K8. `comparison.json` summarizes differences. Do not pool
  NFE20/100. Both contain complete identical conditions.
- `noise_support_screen_v1`: two extra full-coordinate-noise variants, source
  energy matched to rank16 (10.24 per state); GPU0 cap .08 train/.04 evaluate.
  Both 2048-update training jobs completed. Graph-noise candidate evaluation
  completed; no-prior-noise evaluation running at last audit. PID927873, tmux
  `component_noise_support_20261011`.
- `supervision_screen_v1`: waits until original eight trainings finish, then
  independent_cohort, recursive_fm, sample_ms, mean_ms, same fixed cohort labels,
  seed42, batch4,H3,2048 updates (24576 window exposures), training Euler4,K4 MS.
  GPU1 cap .12. PID922923, tmux `component_supervision_20261011`.
  Real CPU 2-update integration for all four already verified under
  `smoke_supervision/verified_results.json` and `label_audit.json`.

Queues have immutable plans/sources and source manifests. Never edit an active
frozen plan to change treatment. Failed jobs remain failed; inspect and create a
declared continuation/revision. Source snapshots are in each queue directory.

## Data and protocol

Snapshot from `outputs/ingolstadt_full_retrain_20261009/world_model`, sha
`2f63f0b2a5fe18f3a3bd9be27ffd1e63a2755464e7f2944e1227334f0ffe6af6`.
Frozen local snapshot + protocol at campaign root. This historical checkpoint
has legacy reward fingerprint; loader explicitly verifies the supported legacy
fingerprint and imports only traffic model/codec, never relabels old policy reward.

Development manifest: 128 train demand groups,819 episodes,87685 windows;
32 validation groups,187 episodes,19576 windows. Full parent corpus has
24818 train episodes/2605729 windows,4384 validation/460640 windows. Development
validation has been used before; it is NOT untouched final test. Fresh final
demand seeds710xxx have not been used. Development SUMO seeds610001..610003.

All model screens use same frozen codec, paired initializations/data schedules,
fresh dynamics. Optimizer is Adam1e-4, unlike historical production AdamW;
comparisons within screen are controlled, not claimed to reproduce main training.
2048-update screen is not convergence. Need learning-curve/budget checks and >=3
independent initialization seeds before final claims. `load_frozen(...fresh... )`
loads codec only; active-fraction readout is fresh/frozen in offline screen and
must be handled explicitly for later policy integration. Non-FM online training
must use `experimental_loss`, not production FM objective. Compiled FM inference
also requires compatibility checks for adapters before policy use.

## Measured evidence (development, not final conclusions)

- Spectrum: 853 roads, rank16 raw count energy .410%, standardized count+speed
  1.481%, latent1.472%; uniform-rank reference1.876%.32 demand groups,384 windows.
  Basis overlap with exact eigenspace .99999999944. Connectivity audit: one
  850-road component +3 isolated roads,4 zero modes. See `spectral/`.
- Numerical snapshot: NFE100 single-step occupied count MAE .5644 veh,
  speed .8148m/s, count CRPS .49235, raw count 90%coverage .3394 (K8). NFE20
  count MAE .6070,coverage .2561. Larger NFE does not remove calibration issue;
  total mass errors can worsen. K8 interval Monte Carlo uncertainty matters.
- Fresh graph_fm screen vs no_prior_fm (same8 groups,K16,NFE20): h1 count MAE
  1.07397 vs .99270; speed1.33655 vs1.17471; CRPS .97458 vs .88557;
  total count error20.0385 vs10.6779. Do not call final significance.
- Direct regression h1 count MAE .65828,speed .8723,CRPS .65828; h8 count .8379,
  speed1.4290,total mass error7.7331. Deterministic coverage0 is expected; no
  assertion that it is a calibrated stochastic model. FM necessity is unproven.
- Isotropic-noise graph FM h1 coverage .369 vs .301 original, but h8 total
  mass error137.725 and Energy Score20.743 vs19.756/10.288 original. More dispersion
  is NOT automatically an improvement. Preserve this failure; do not promote
  the candidate merely because coverage rose. Other isotropic result pending.
- Generalization:72 real heuristic evaluations +36 fixed Urban ep22 evaluations.
  Normal demand means: static239.410s,dynamic200.282s,Urban339.130s.
  1.3x burst:455.689/345.363/786.475s. Early policy is worse; not final convergence.
  Urban one seed has4/1267 unfinished (99.6843%),0 never-started. C/B differs from
  completed-only ATT. See `generalization/urban_checkpoint_development_v2`.

## Executor defect and repair — important unfinished branch

True SUMO1.27.1 meso toy single queue: independent average-speed execution sends
red-light head vehicle across stop line even with true per-second road speed.
Strict controls/routes69/69 match;135 vehicle steps,46 wrong-road-timing cases.
No WM or 30s speed approximation involved. Evidence `executor_audit/toy_queue_oracle_v2`.

Implemented optional `passage_constraint` in imagination.py plus
`meso_constraints.py`. Real paired replay reduces46 errors to4/5 and42/41 red
violations to0, no new road errors, same89 common-road position errors unchanged.
59 author tests,93 root related tests, independent queue/offset review. Default
None preserves legacy; production frozen runs unchanged. Commit `b448020`.
Evidence `executor_audit/toy_queue_constrained_v2`.

This is NOT city-ready/full meso physics: strict parser rejects permissive g,
actuated/external overrides, unknown tied FIFO and simultaneous upstream merges;
no segment supply/headway/priority arbitration. Read-only real-city preflight
shows all3 networks static but contain g, so all fail this strict prototype.
See `executor_audit/city_schedule_preflight.json`. Next repair needs explicit
city-supported semantics or richer causal queue/permission inputs, then free
rollout/SUMO validation. Do not claim existing main experiments were repaired or
silently enable unsupported prototype. This remains meaningful work, not a blocker.

## Evaluation integrity repairs already made

- Decode each stochastic trajectory before means; no hidden teacher forcing.
- Match fixed initial cohort counts/events; speed label is endpoint+following
  boundary average, not instantaneous speed.
- Independent noise ensemble for h1 disagreement; h>1 same-ensemble diagnostic.
- Default phase selection uses real traffic only; empty stress separate.
- CLI complete-horizon intersection; old frozen queues also have sufficient windows.
- Report tool permits old/new horizon policies only after proof every condition has
  all horizons and full count/speed labels; all other pairing checks stay strict.
- Imagination fixed-horizon summaries preserve full condition/sample sets, avoid
  survivor selection, and separate post-deadline stress. Validate external condition
  identities. Old raw outputs untouched; revised summary stored in new directory.
- Training journal archives uncommitted tail on resume; exact CPU restore tested.

## Tools / next actions

`tools/summarize_component_predictions.py` produces strict paired CSV/PNG/PDF
and bootstrap demand-group intervals. Reports currently only for old snapshot
steps20; once models finish, run on matched `prediction_screen/*` directories.
Keep training seeds separate; current CIs do not measure training variability.

1. Verify live jobs, collect finished matrix, inspect negative outcomes, test
   convergence/production NFE before selecting variants. Expand to multiple seeds
   and final held-out demands; don't reduce goal to screens.
2. Finish real-city executor repair/validation; then selected RL ablations with
   consistent execution/critic/actor/codec budgets and fresh paired seeds.
3. Complete MS experiments and probability/recursion analyses, including full
   source-noise candidates without hiding mass-drift failures.
4. Disagreement diagnosis then compatible risk penalty/gate experiments. Current
   bounded critic forbids beta>0; MS gate is not presently a production component.
5. Final learned-policy demand shifts, three-city compatible main table, sample
   efficiency including offline acquisition/warmup, censored/completed-trip metrics.
6. Full requirement audit before update_goal complete. Nothing is blocked merely
   because jobs take hours. Preserve all negative results and protocol identities.
