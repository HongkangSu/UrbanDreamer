# Main experiment audit — 2026-10-12

Read-only audit of main runs, excluding component-experiment queues. Host PID snapshot: 2026-10-12 03:24:51 UTC; artifact reads through approximately 03:30 UTC. Only this audit file was written. No source, checkpoint, running process or scheduler was changed. Counts below are a moving snapshot, not final results. Host inspection was limited to user wangwenrui and relevant project processes.

## Authoritative current status

Aliases used below (all relative to repository root):

- **U** = `outputs/ingolstadt_full_retrain_20261009/online_bounded_noprior_fresh_t2_2000_20261011`
- **F** = `outputs/city_baselines_two_city_fresh2500_20261011`
- **I** = `outputs/city_baselines_noincident_noprior_penalty_5000ep_20261008/ingolstadt_custom`
- **P** = `outputs/two_city_urbandreamer_pretrain_supplement_20261011`

Each live entry below has both a matching host command/PID and a recent persisted record. All active RL commands use one independent training seed, **42**. Per-episode demand seeds and evaluation seeds are not independent training replicas.

| City / method | Status and PID | Last observed completed episode / updates | Budget / ancestry |
|---|---|---:|---|
| Ingolstadt UrbanDreamer | online, 840021 | 87 episodes; last zero-based index 86 | 2000; fresh actor/critics/replay, pretrained WM retained |
| Ingolstadt IQL | online, 839299 | 4803 | 5000; resumed, reward migration at 2653 |
| Ingolstadt MAMBA | online, 839306 | 1704 | 5000; resumed, reward migration at 718 |
| Ingolstadt Dreamer | online, 839573 | 2843 | 5000; resumed, reward migration at 1503 |
| Ingolstadt MAPPO | failed, no live child | 2754 | 5000; reward migration at 2561 |
| Ingolstadt GNAC | queue completed, no live child | 5000, holdout_best present | reward migration at 3447 |
| Ingolstadt IPPO | queue completed, no live child | tail contains holdout_best for selected episode 300 | historical protocol; not proof of final training length |
| Ingolstadt online AC | queue completed, no live child | tail contains holdout_best for selected episode 600 | historical protocol; not proof of final training length |
| Ingolstadt MBPO | waiting_gpu, no child | not established | queue waiting |
| Coulommiers online AC | online, 849809 | 385 | fresh 2500 |
| Coulommiers IPPO | online, 849810 | 416 | fresh 2500 |
| Coulommiers MAPPO | online, 849811 | 369 | fresh 2500 |
| Coulommiers IQL | online, 849814 | 273 | fresh 2500 |
| Coulommiers MAMBA | online, 849816 | 178 | fresh 2500 |
| Coulommiers GNAC | online, 849935 | 336 | fresh 2500 |
| Coulommiers Dreamer / MBPO | waiting_gpu, no child | not started in this queue | fresh 2500 |
| Montereau online AC | online, 849815 | 398 | fresh 2500 |
| Montereau IPPO | online, 849934 | 359 | fresh 2500 |
| Montereau MAPPO | online, 849945 | 344 | fresh 2500 |
| Montereau IQL | online, 849953 | 234 | fresh 2500 |
| Montereau MAMBA | online, 849966 | 162 | fresh 2500 |
| Montereau MBPO | online, 862781 | 287 | fresh 2500 |
| Montereau GNAC / Dreamer | waiting_gpu, no child | not started in this queue | fresh 2500 |
| Coulommiers UrbanDreamer | offline pretraining complete; no live pretrainer | WM updates 19864, supplemental samples 843026, epoch 1 | no online episodes; codec_ready true |
| Montereau UrbanDreamer | decoder recovery failed; no live recovery child | latest resume failed before recovering update counter | no supplemental pretrain completion summary |

Episode evidence: `F/<city>/<method>/metrics.jsonl`, `I/<method>/metrics.jsonl`, `U/pilot_progress.json`. Queue evidence: `F/queue_state.json`, current controller PID **849799**. Coulommiers evidence: `P/coulommiers/supervisor_status.json` (`complete`) and `world_model/pretrain_summary.json` (`pretrained`, online_episodes_completed=0). Montereau authoritative evidence: `P/montereau_fault_yonne/supervisor_status.json` and `decoder_recovery_chain_status.json` (`failed`), not the older `world_model.status.json` still reporting revalidation_running.

Numerous controller processes remain alive: old noincident/v5/entropy/paper_reward/sampling_t2/sampling_t0p2/argmax controllers. Their existence is **not** evidence of independent active training replicas. In particular `outputs/city_baselines_argmax_20261011/queue_state.json` still names old two-city child PIDs 839366 etc., absent from the host snapshot; F correctly names new 849xxx children. Ingolstadt children retain the argmax frozen source and I output directories while F's current queue includes their PIDs. Do not add duplicated controller views as extra experiments. This audit does not claim that every old controller is inert internally; it establishes which actual training children were alive.

## Reward, actions, selection and data-budget contracts

| Group | Real reward / reporting | Evaluation action and demand bank | Checkpoint selection |
|---|---|---|---|
| Current U | cumulative observed mean: -delta(C/max(B,1))/60; beta=0; bounded_cohort_v1 | argmax, zero SP prior; seed family 314159, evaluation indices reused before/milestones/after; latest milestone 80 episodes | pilot saves latest and milestone evaluation records; inspected runner has no baseline-style best-selection + independent holdout procedure |
| Fresh F baselines | causal_cumulative_mean_global_v1; global-only; no individual timeout penalty; reward_objective observed_mean | deterministic/argmax, SP prior0; development seeds 10042–10044 | best.pt minimizes negative selection_score on periodic evaluation every100; planned holdout_final and holdout_best seeds20042–20044 at run end |
| Resumed I baselines | current ongoing rows use same causal contract; earlier segments use historical objective | latest live commands argmax; historical sampled/temperature evaluation regimes exist | reward handoff reset best threshold; action-selector changes archive old best; current best is not a global comparison across all historical scoring regimes |
| Completed historical I methods | inspect each protocol; some tail rows lack causal reward_contract | historical deterministic holdout rows exist | cannot infer current target compatibility from queue completed status |

U config evidence: `run_config.toml:62` low_frequency_rank16, `:64` prior_loss_weight0.1, `:70–72` K2/beta0/bounded critic, `:75–76` SP routing prior0, `:83` previous_state_excluding_current_births, `:84` observed_endpoint, `:90` single_step FM. The directory term noprior means no **routing** prior; low-frequency graph prior is present. Current U is not an MS+gate+penalty Full variant. `supervision_horizon=3` does not override single_step mode.

Source selection evidence: `F/source/tools/train_toy_baseline.py:264` development seeds, `:430–455` best selection and holdout. `U` executes `outputs/ingolstadt_full_retrain_20261009/source_bounded_noprior_argmax_20261011/tools/run_bounded_online_continuation.py`; its `tools/run_online_pilot.py:269–276` saves latest, `:363–384` evaluates milestones, `:413–420` repeats after-evaluation and records total consumed episodes. No independent final test bank was found in this pilot flow.

U `fresh_start_verification.json` records actor, bounded V/Q, target V, optimizers, online replay, online clock and RNG reset; WM, offline corpus, codec validation and pretrain counters retained. `latest_evaluation_change.json` records a **selector-only switch at episode4**, sampled temperature2 -> argmax, preserving models/optimizer/replay/RNG/online clock. Its new before-evaluation reference is episode4, **not random initialization**. Do not plot that reference at x=0. Current commands and latest_evaluation_change supersede original fresh_start selector metadata.

Physical horizon: U config demand1800s + nominal evaluation3800s, implemented macro rounding3810s. Recent F training rows have microsteps3810, window_count127. Completion can finish earlier; interaction accounting should sum actual microsteps rather than multiplying every episode by3810. For U, the 87 stored online records summed to331470 one-second SUMO transitions at this snapshot. This excludes pretraining collection, codec ancestry, development evaluation, and any discarded warmup/attempt costs.

Mainline offline training interaction floor, from current complete cache manifests (training split only):

| City | Unique training trajectories | 30-second training windows | Validation trajectories / windows |
|---|---:|---:|---:|
| Ingolstadt | 24818 | 2605729 | 4384 / 460640 |
| Coulommiers | 6638 | 843026 | 1216 / 154432 |
| Montereau | 6306 | 800862 | 1092 / 138684 |

Ingolstadt manifest `outputs/ingolstadt_full_retrain_20261009/data/cache_manifest.json`; other manifests `P/<city>/cache/cache_manifest.json`. These are sequence-window counts, **not** one-second transitions. Windows also may include a final partially elapsed window, so recover exact simulation steps from source episode metadata where available. Count a deduplicated physical trajectory once for collection, not once per replay epoch/gradient use. Data originate from prior baseline trajectories; WM's offline access cannot be treated as free when claiming total environment sample efficiency. Separate (a) equal-access offline-data comparison, (b) total collection-budget comparison, and optionally (c) online fine-tuning cost with explicit common pretraining offset. Include codec training ancestry when it used additional physical trajectories.

## Heldout status and direct comparability blockers

1. **Not yet a complete three-city table.** Two-city U has no completed online main runs; Montereau recovery failed. Several baselines remain queued, and Ingolstadt MAPPO failed. Its queue last_error is `ValueError('PPO collected behavior probabilities do not match the frozen actor')`, log `outputs/city_baselines_paper_reward_20261010/logs/ingolstadt_custom_mappo.attempt3.log`. This audit did not restart or diagnose that method further.
2. **Only seed42 is established.** Repeated demand seeds, checkpoint resumes, multiple source snapshots, and repeated controller PIDs do not provide training-seed uncertainty. Need independent training seeds for the selected comparison set; current results are development/single-seed evidence.
3. **Ingolstadt is a continuation study, not uniform fresh training.** Reward migrations preserve actor and optimizer history. Actual boundaries: IQL2653, MAPPO2561, MAMBA718, Dreamer1503, GNAC3447 (`I/*/reward_objective_changes.jsonl`). Do not relabel post-migration episode0 or discard prior interactions for a from-scratch sample-efficiency claim. Fresh F is cleaner, but has2500 budget versus I5000/U2000 and different optimizer/entropy progress histories.
4. **Evaluation currently is not paired across U and baselines.** U uses314159-indexed requests, baselines10042–10044 development seeds; even equal integer seeds would require matching actual demand hashes, input/network hashes and episode-generation conventions. Freeze a common realized demand bank and reuse identical files/hash identities for every final method.
5. **Final heldout has already been consumed in historical runs.** I/GNAC metrics tail includes holdout_best seed20043/20044 with causal contract; I/IPPO, online_ac and DTA also have holdout_best on20043/20044 under older rows. Therefore20042–20044 is not a never-observed bank for the whole study. Fresh F runs have not reached planned end-of-run holdout in this snapshot; absence of final rows there does not restore study-wide blindness. U's repeated before/milestone/after bank is development, not independent final test.
6. **Checkpoint selection is asymmetric.** Baselines retain development-best; U inspected pilot retains latest and evaluation milestones without the same best protocol. Predeclare final-at-equal-interaction budget or development-best under the same evaluation budget. If historical U milestone weights were not retained, evaluation records alone cannot reconstruct them. Audit checkpoint retention before committing to best-checkpoint comparisons.
7. **Risk of outcome-dependent reporting.** Do not pick between best and final on final-test outcomes. Use already-consumed holdouts as development; freeze new demand bank before final evaluation and keep its outcomes out of model/selector/hyperparameter selection. Codec/WM validation used for acceptance is also not an untouched final prediction-test set.
8. **Censored objective requires auxiliary reporting.** Main cost is C(T)/(60B(T)), not complete-trip mean. Preserve started/never-departed/completed/unfinished counts and completed-trip tail times, plus an agreed drain rule. Queue reference jobs marked completed need physical contract/hash revalidation before final comparison; status alone does not ensure this.

Recommended final-main protocol: preserve all current work as development; freeze exact method/source/checkpoint ancestry and reward/simulator/action protocol; choose a minimal strong comparison set; run required independent seeds under the chosen budget; separate offline-data-aware sample-efficiency claims; evaluate a common newly frozen heldout demand bank once after checkpoint selection. This is planning guidance, not authorization exercised by this audit.

## Montereau recovery handoff: exact failure/provenance

Frozen source: `P/source_montereau_fullgpu_decoder_recovery_20261011`. Wrapper: `P/montereau_fault_yonne/recovery_fullgpu_chain.py`, COMMANDS at lines19–31. The failed child command is:

```text
/home/wangwenrui/opt/miniconda3/envs/URB/bin/python -u P/source_montereau_fullgpu_decoder_recovery_20261011/tools/recover_pretrain_decoder.py --run-dir P/montereau_fault_yonne/world_model --output-dir P/montereau_fault_yonne/decoder_recovery_fullgpu_20261011_163817 --initial-gate-report P/montereau_fault_yonne/diagnostics/codec_gate_20261011_160339/gate_result.json --device cuda:1 --gpu-memory-fraction 1 --threads 4 --batch-size 64 --codec-validation-batch-size 32 --max-updates 3000 --validation-every 500 --resume
```

Here P expands to the absolute repository path plus the alias above. Failed supervisor887656/child887663 were **not live** in host snapshot. Source checkpoint: `P/montereau_fault_yonne/world_model/checkpoints/latest.pt`. Resume checkpoint: `P/montereau_fault_yonne/decoder_recovery_fullgpu_20261011_163817/recovery_state.pt`.

Log: `P/montereau_fault_yonne/decoder_recovery_chain_decoder_recovery.log`. It contains **two distinct failures**, so fixing deserialization alone does not prove acceptance:

```text
recover_pretrain_decoder.py:205
RuntimeError: decoder calibration budget exhausted without passing the unchanged heldout gate
...
recover_pretrain_decoder.py:136
saved = torch.load(recovery_state, map_location='cpu', weights_only=True)
_pickle.UnpicklingError: Weights only load failed.
Unsupported global: GLOBAL numpy._core.multiarray._reconstruct
```

Recovery `status.json` has UTC2026-10-12 02:46:05, phase failed. Its `calibration_updates=0` is the failed resume's local counter, **not evidence that the earlier2000-update attempt never trained**. The3000 maximum is the later resume's intended total ceiling, not the completed budget. This code initializes updates=0 before loading the retained state. Publication and normal FM continuation are later chain stages and were not reached after this failure.

Hashes computed directly read-only in this audit:

- Frozen `tools/recover_pretrain_decoder.py`: 13738bytes, SHA256 `f86f09c12bb9e30a36f9b7a5902418aefe0731d7133db4138e34f38aa87a879e`.
- Retained `recovery_state.pt`: 100615207bytes, SHA256 `c11f08dc1dab1a8d1b5eb2134717e22a4b2b666d63405faf43c1e293f006973f`.

Protected-source hashes recorded by recovery status (not independently recomputed here): original world_model/checkpoints/latest.pt `4ca1cae03967b81ea29f83ed2b3688786eba45d6db896c97c6b5b98ff1f0e04d`; run_config.toml `ff07933be4af483cfad0f848ff54128d73a6852ecaea4429b6f58a19c9430419`; cache manifest file `3ea42f42ef3811bd44fac8a9af423bd52b82fe6344323025a407102fda1ef438`. Keep unchanged quality gates and source identity checks when root assesses a bounded recovery. No recovery or unsafe checkpoint load was performed by this audit.

## Addendum: Montereau quality-gate diagnosis (2026-10-12, read-only)

This addendum corrects the initial informal assumption that the exhausted attempt had run3000 updates. The retained directory has validation records at500/1000/1500/**2000**, latest checkpoint mtime2026-10-11 17:26. There is no2500/3000 validation record. The subsequent chain requests `--max-updates3000 --resume`, but its load failed before entering the update loop. Thus2000 is the completed budget supported by durable validation evidence; reading saved update counters safely is delegated to the loader-fix lane. No checkpoint unpickling occurred in this diagnosis.

The initial full-gate report (`P/montereau_fault_yonne/diagnostics/codec_gate_20261011_160339/gate_result.json`) checked1092 validation trajectories/138684 raw windows, with encoder codec_updates4500 and decoder_revision420. Its **only** failed key was `hybrid.codec_total_relative_error`; all raw/hybrid road-count and speed thresholds passed. All four subsequent `decoder_recovery_fullgpu_20261011_163817/validation_*.json` retain the same single failing measure.

| Additional decoder updates | Raw occupied count MAE | Raw total relative error | Hybrid occupied count MAE | Hybrid total relative error | Hybrid occupied / slow speed MAE (m/s) |
|---:|---:|---:|---:|---:|---:|
|0 (initial report)|0.278235|0.024066|0.277756|0.175792|0.297016 /0.178746|
|500|0.259669|0.022830|0.259086|0.123355|0.274777 /0.126624|
|1000|0.251394|0.030258|0.250931|0.118990|0.267436 /0.124317|
|1500|0.218486|0.020258|0.217811|0.119490|0.273015 /0.131399|
|2000|0.203466|0.017048|0.202653|**0.106516**|0.263798 /0.134212|

The total-relative threshold remains0.1 for both spaces; occupied count threshold0.5, all-road count0.1, empty count0.05, overall/occupied speed1.5, slow speed0.5. Raw/hybrid slow-road counts stay4823153/84180, satisfying nonempty-subset coverage. The best observed checkpoint under the blocking metric is the latest2000 state. Hybrid relative error improved39.4% from initial,13.65% from500, and10.86% from1500; it needs another0.006516 absolute reduction (about6.12% of current error) to pass. The1000–1500 fluctuation does not establish a plateau, given improvement at2000. Speed is comfortably within thresholds and not the gating problem.

Frozen-source inspection supports the following diagnosis, separating facts from hypotheses:

1. **Objective and scope are internally aligned.** `tools/calibrate_frozen_decoder.py:53–62` constructs raw `(flow[t],flow[t+1])` and hybrid `(corrected cohort endpoint[t+1],raw[t+2])`; endpoints without real following frames are skipped. `codec_training.py:104–160` reconstructs input-frame count, so it does not accidentally supervise hybrid count against newly born population. Forward-mean speed uses the following raw frame by established codec contract. Decoder-only mode detaches encoder latents. No specific target/mask bug was found.
2. **Sampler visits training trajectories and frames, not heldout states.** `tools/train_codec_ready.py:154–194` uses shuffled without-replacement episode/frame orders and persists cursor/RNG. Recovery passes half requested batch size to this sampler, then concatenates raw and hybrid frames, approximately32+32 for batch64. Entire train episodes contain127 windows, so2000 updates cover approximately500 trajectories/63500 distinct raw starts (tail batches31), not a full6306-trajectory epoch. This is useful supervised progress but modest corpus coverage.
3. **There is a documented training/gate weighting mismatch for empty initial cohorts.** `pretrain_cache.py:545–586` samples four uniformly spaced sequence starts for validation; with seq1/windows127, valid hybrid starts are0,42,84 and126 is skipped. Therefore start0 accounts for one third of hybrid gate frames, versus1/126 of uniform hybrid training frames. The existing read-only `decoder_recovery_sampling_evidence.json` independently records the42-fold difference and checks12 source initial populations, all empty. Those checks are recorded prior evidence, not an exhaustive re-scan performed here.
4. **Small-population total error is a plausible driver, not yet a proven numerical decomposition.** `codec_training.py:82–99` computes mean per-frame `abs(sum(pred_count)-sum(target_count))/max(total_target,1)`. At empty cohort frames even submillivehicle per-edge positive leakage may accumulate over a large graph and contribute directly to this score. Training mass loss instead uses `asinh((pred_total-target_total)/(target_total+1))²` at weight0.1; road count uses80% all-road log-count MSE plus20% occupied-road MSE. Consequently low per-road MAE and good high-population reconstruction can coexist with a failed sparse-frame mass gate. The saved aggregate validation reports do not contain signed mass bias or error by anchor/population, so they cannot establish that empty-start frames cause all of the observed0.106516.
5. **No evidence supports changing FM, encoder, speed masks or representation at this point.** The scope of the failure is decoder hybrid mass calibration; count quality is improving and all speed gates pass. Config declares decoder pretrain Adam learning rate1e-4 (`world_model/run_config.toml:111`); recovery restores existing decoder optimizer/moments without changing LR. Actual saved optimizer LR should be verified by the safe-loader lane rather than inferred solely from config.

Recommended bounded next action for root: after a verified safe load confirms saved updates2000, continue the already intended total ceiling3000 (**1000 additional updates**, full unchanged validation at2500/3000), preserving sampler cursor, optimizer, frozen encoder/dynamics, source hashes and all thresholds. At the original update/validation timestamps,500-step intervals took roughly10–12minutes, but present shared-GPU availability may change this. First passing original gate can publish through the existing checked candidate workflow; no passing result is claimed now.

If3000 still fails, do **not** silently extend forever or loosen the0.1 gate. First add a read-only decomposition of the same validation predictions by anchors0/42/84 and target population0,1–5,>5, with signed/absolute total-count error and total relative error. Confirm on analogous **training** strata. If zero/small-cohort undercoverage is supported, use a separate explicitly recorded decoder-only calibration variant with modest train-only stratified zero/small-pop hybrid examples while retaining uniform raw/nonempty data, identical original loss, the entire existing validation partition and thresholds. Start conservatively (for example10–20% of hybrid slots from the underrepresented training strata), and predeclare an additional bounded1000-step budget; this changes sampling weights and must be documented. It is a conditional follow-up proposal, not executed here. A count/mass architectural change is not justified by the current evidence.
