# Completed-pretrain reference and sufficiently trained offline comparators

Prepared 2026-10-13 by a bounded read-only architecture audit. This is a runner implementation specification, not evidence of trained comparators or a launched campaign. No GPU, SUMO, training, checkpoint modification, or production-source change was performed.

## Fixed reference and data evidence

- FM reference is ONLY `outputs/component_experiments_20261011/snapshot/checkpoints/latest.pt`, SHA256 `2f63f0b2a5fe18f3a3bd9be27ffd1e63a2755464e7f2944e1227334f0ffe6af6`. Its prescribed one-epoch initialization completed 49,636 optimizer updates and 2,605,729 observed-window exposures. Completion is not independently established convergence.
- Source cache manifest: `/home/wangwenrui/shk/UrbanDreamer/urban_dreamer/outputs/ingolstadt_full_retrain_20261009/data/cache_manifest.json`; current file SHA256 `34169f27525094b6103b5a0556f64c14c3e176d7985f19fd0d5543bc68c4c30e`; content identity `5254f01c7863d3e8524d4cf118983fae1bfbe755832ece48380acbdd11eab443`.
- Full training split: 4,291 realized demand groups, 24,818 trajectories, 2,605,729 single-step windows. Full development validation: 759 disjoint demand groups, 4,384 trajectories, 460,640 windows. Previously inspected validation remains development data, never final test.
- Original B=64 batches do not concatenate episode tails. Thus one epoch has `sum(ceil(windows_i/64)) = 49,636` updates, not `ceil(2605729/64)`. B=32/16/8 would produce 90,502/170,556/332,526 updates per complete epoch respectively. This is a material optimizer-schedule difference.
- Original launch is retained in `outputs/ingolstadt_full_retrain_20261009/launch_command_memory_safe.json`; training code is `source_memory_safe_20261010/urban_dreamer/learning/training.py`. Original optimizer is AdamW(lr=1e-4, weight_decay=1e-2), not the component screen's Adam. Frozen config uses single_step, sequence length 1, rank16, source coefficient std .1, prior loss .1, BF16 dynamics, Euler100 inference. One-step FM training fits velocity and does not perform 100 inference steps per update.

## Exact reuse boundaries

1. Reuse `tools/component_experiment_data.py:78` `load_frozen(..., fresh_dynamics_seed=seed)`. It verifies exact checkpoint bytes, supported legacy semantic fingerprint and prepared input hash, constructs the unchanged production backbone, imports the completed flow_projector codec only, and freezes it. It imports no old actor/critic/reward optimizer contract. `fresh_dynamics_seed=None` is reserved for the completed reference, not comparator initialization.
2. Reuse `urban_dreamer/models/experimental_dynamics.py:119` `apply_variant` and `tools/train_component_world_model.py:68` `supervised_loss(mode='single_step')`. Input remains **full observed prior traffic state** and **all recorded action events**. Target count is the previous-window active cohort Ct excluding current-window births; occupancy/halting/speed retain full-background semantics. The contract is explicit in `urban_dreamer/learning/cohort_targets.py:63`. Do not silently replace these with fixed initial C0 targets/controls used by imagination diagnostics.
3. Instantiate `CompactPretrainDataset(config, prepared, topology, 1, cache_manifest_path=FULL_MANIFEST, split='train')` directly. Its `iter_epoch` at `urban_dreamer/learning/pretrain_cache.py:448` visits each legal start exactly once, shuffling episode order and then within-episode starts with the supplied independent NumPy Generator. Source/cache identity verification remains active.
4. Reuse `atomic_json`, `atomic_checkpoint` and `reconcile_training_journal`; add dataset cursor state to the new runner's durable checkpoint. Do not reuse the old screen protocol identity, output directory, status files, or old 8,192-update weights.

The smallest new surface is a separate `tools/train_full_corpus_world_model.py` importing those existing utilities, with its own protocol role `full_corpus_comparator`, plus bounded runner tests. Leave the existing screen runner and campaign artifacts unchanged.

### Full selection through existing data preparation is possible, but not sufficient

`select_manifest(source, train_groups=4291, validation_groups=759, seed=20261013)` preserves all original trajectories. It still creates a different derived manifest identity and labels it development-only; that is acceptable provenance. Preparing with the **frozen component snapshot** is preferable to copying mutable production latest. The CLI `prepare` currently copies the latest checkpoint from run-dir, so a new wrapper must explicitly copy the immutable 2f63 snapshot or assert its bytes before and after preparation.

However, existing `train_component_world_model.py:235` samples legal windows **with replacement**; it gives no unique-coverage guarantee and uses a different optimizer. Selecting all groups and running enough replacements is a valid separately declared training protocol, but cannot claim to reproduce original finite unique traversal. Direct full-manifest traversal is simpler and more defensible. Equal exposures alone are not equal unique data coverage or equal update budget.

## Minimal comparator matrix and parameter accounting

Initial production-quality preparation: regression, Gaussian, no_prior_fm, all independent new dynamics initialization seed42 and independent **paired data seed20261013**. Stage regression and Gaussian first; no_prior_fm is the graph-prior diagnostic. Add graph_regression only if the prior question remains unresolved after those three. Do not launch a factorial matrix.

| Predictor | Dynamics parameters | Prediction/training contract |
| --- | ---: | --- |
| Completed graph_fm reference | 8,173,328 | Already completed prescribed full pretraining; keep exact checkpoint |
| regression | 8,140,368 | Same graph-conditioned Transformer, deterministic latent residual, latent MSE |
| Gaussian | 8,156,816 | Same backbone; output 64 latent means + 64 diagonal stds, Gaussian NLL |
| no_prior_fm | 8,140,368 | Same FM backbone and rank16 source noise; deterministic coarse prior removed |
| graph_regression, optional | 8,173,328 | Same deterministic latent regressor with trained rank16 prior |

Gaussian adds exactly `64*(256+1)=16,448` head parameters to regression; removing the coarse prior removes 32,960 parameters (<.5%). Obtain `variant_metadata` at runtime and verify these counts before training. Gaussian std is softplus(raw)+1e-4 (`experimental_dynamics.py:174`); its random draws are full road/latent coordinates, whereas FM source noise is rank16 spectral coefficients. Do not call these equal stochastic families.

For each comparator, load the same codec SHA, same graph and static features, same feature/history/action inputs, and seed before fresh model construction. Variant Gaussian replaces only the output head; shared backbone initialization should match seed42 regression, verified by state-key tensor hashes. Independent data RNG ensures extra Gaussian/FM draws never alter the replay sequence. Initialization/data pairing applies among **new comparator runs**; the final reference snapshot does not establish that its original data order equaled data seed20261013.

## Concrete training and resume specification

- Freeze source/code manifest and full data/protocol hashes before starting. Assert completed reference counters and exact SHA. Do not require old reward semantics on imported traffic-only weights.
- Model codec eval/frozen; dynamics train. Optimizer AdamW(lr=1e-4, weight_decay=.01), gradient clip1, FP32 codec input/target encode under no_grad, BF16 dynamics on supported CUDA only. **Keep encode outside autocast**, matching original `source_memory_safe_20261010/.../training.py:673`; current component caller wraps its entire supervised_loss in autocast and therefore should not be copied without moving that boundary.
- Requested effective B64, one update per original within-episode B64 batch including its actual tail. If GPU admission cannot fit, preserve update semantics by materializing that batch and accumulating sample-weighted microbatches; do not concatenate unrelated episode tails or claim that B16 updates are the original schedule. This requires carefully remapping SparseActionEvents batch indices. An explicit B16 full-epoch protocol is acceptable only with its changed 170,556-update schedule disclosed; it must not be equated with 49,636 updates.
- Initial minimum coverage: one whole unique epoch for all comparators, 2,605,729 windows each. This is a **coverage floor**, not the stopping criterion. Preserve checkpoints at regular update boundaries and epoch boundaries. Use a declared continuation segment if further epochs are necessary.
- At each successfully committed optimizer step save, as one atomic state: model, optimizer, independent data_rng, CPU/CUDA model RNG, `source.state_dict()`, update, exposures, epoch counters, source/protocol hashes, elapsed training time. Codec hash must remain unchanged. Never save a sampler cursor ahead of a committed update.
- `iter_epoch` advances its start cursor before yielding and increments `epochs_completed` only after it is exhausted; save the cursor at successful-update boundaries, and save again after exhaustion. A checkpoint after the last yielded batch but before exhaustion is legal: resume consumes no duplicate window and exhausts the saved iterator to commit the epoch counter. Do not begin a new epoch merely because the final batch count reached its expected value.
- Restore cursor with `source.load_state_dict` (inherited `offline_data.py:372` validates manifest, complete permutations and cursors), optimizer/RNG, then recreate iterator. Existing `tests/test_pretrain_cache.py:142` proves remaining sequence reproducibility; add actual interrupted-run vs uninterrupted model/optimizer/curve parity for this new runner.
- Retain failed-attempt logs; reconcile only rows beyond saved update using existing journal helper. No optimizer reset or reseeding on resume. New epoch budget continuation preserves the same learning/data contract and records its reason/parent checkpoint SHA.

## Validation and sufficient-training acceptance

Prespecify development selection before any new comparator sees scores. A bounded first implementation can use 64 disjoint existing-validation demand groups with two deterministic legal anchors each (128 matched single-step states), plus explicitly retained empty states. Persist condition IDs and feature masks. Select by hash/traffic occupancy, never by which predictor performs well. Validation has its own NumPy/Torch seed and must save/restore training RNG; never mutate the training dataset cursor.

At every 4,096 updates and each epoch end, record common physical count/speed MAE/RMSE using decoded sample means (Gaussian/FM K8; regression one deterministic prediction), Energy/CRPS/PICP/MPIW, and predictor-specific native validation loss on the exact same states. Original FM anchor is evaluated from its completed checkpoint on these states, with K32/NFE100 when generating the final paired report. Do not compare raw FM velocity loss, Gaussian NLL and regression MSE as if they were one scale. Cheap validation can use a prespecified smaller NFE profile for trend, but final profile must match Euler100 and both profiles must be labeled.

Before running, declare a continuation rule: after full coverage, extend another unique epoch if primary predictor-appropriate development scores still improve materially across successive fixed checkpoints; no stop on one noisy point or flat training loss alone. A reasonable prespecified review threshold is <1% relative change in relevant physical MAE and proper score over three consecutive checks spanning at least 8,192 updates, with no widening train/validation divergence. That threshold is a diagnostic to review, not proof of global convergence. If it fails, report **still training** and retain the run. Gaussian undercoverage or exploding std warrants optimization review, not early best-case selection.

Final sufficiency evidence must include completed unique coverage, uninterrupted/resumed data trace proof, learning curves, development trend, chosen checkpoint/selection rule, actual exposures/updates/wall time, and remaining convergence uncertainty. Paper-level learned-baseline claims additionally need independent training seeds (at least3) and fresh held-out demand groups; seed42 full coverage is the first rigorous comparator, not the final multi-seed conclusion.

## Limits of the proposed comparison

- The original pretrained FM encoder was frozen throughout dynamics training, and the decoder was supervised for 49,636 updates (`source_memory_safe_20261010/.../training.py:288,715,755`). Sharing its completed codec does **not** shift latent coordinates, but does give all comparators an FM-campaign-trained observation decoder. Disclose this common inherited asset. Do not re-train decoder differently per comparator.
- Original FM also trained its active_fraction readout on observed endpoints. The fresh-comparator branch of `load_frozen` does not import a trained readout. Offline physical count/speed comparisons do not use it, so it can remain frozen/unused there. **A downstream policy comparison must explicitly load the same completed readout or train an identically budgeted readout for every variant**; a frozen random fresh readout is unacceptable.
- Original dynamics optimizer/data history differs from this new shared-codec campaign even when unique corpus, AdamW and B64 match. These runs answer whether sufficiently trained alternatives using the available completed representation beat the deployed checkpoint. They alone do not isolate every architecture effect. A later completed matched graph_fm training run is needed for a strict same-initialization/training-history architecture study; keep 2f63 as the deployment anchor regardless.
- Evaluation on full state/full controls/Ct must precede fixed-C0 autoregression. The latter introduces state/control-scope shifts already audited, and must have a separate target-family label. Do not use the shifted task as the only learned-baseline metric.
- Current objective optimizes expected traffic cost; better proper scores do not automatically prove lower routing regret. Action-conditioned short probes and actual replicated policy training remain separate evidence requirements.

## Runner acceptance checks before GPU admission

1. Full-manifest identity/hash/count verification and train/validation group disjointness; deterministic finite traversal visits every `(episode,start)` once and episode-tail update accounting matches49,636 at B64.
2. Shared codec hashes and paired backbone initialization; only specified variant keys differ; expected parameter counts.
3. Same full inputs/actions/hybridCt labels as archived original; no C0 substitution or future-demand leakage.
4. Mid-episode, mid-last-batch and epoch-boundary restore parity; model/optimizer/data/RNG/curves match uninterrupted execution.
5. Validation does not perturb training RNG/cursor/model mode/weights; train-specific optimizer excludes codec/readout.
6. Small synthetic/small frozen-data CPU smoke and resource-limited GPU admission check; this smoke is mechanical proof, not a quality baseline. No production training process may be stopped to make room.

Once implemented and these checks pass, stage serial regression then Gaussian, preserve their full-coverage curves, and launch no_prior_fm only after available resources and completed-checkpoint evaluation priorities permit. No training was launched by this audit.
