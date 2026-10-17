# Main experiment evidence audit — 2026-10-13 08:41–08:47 UTC

Bounded read-only audit by main_experiment_evidence. Only this new note was written; no model/source/runtime/checkpoint/process was changed. Counts are moving observations, not completed final results. Prior audit remains retained at main-experiment-audit-20261012.md.

## Authoritative running/completed evidence

All listed live PIDs had an existing /proc cmdline matching the exact UrbanDreamer outputs run path. Queue states alone were not accepted.

| Method/run | Last durable work | Runtime evidence |
|---|---|---|
| Ingolstadt UrbanDreamer bounded zero-routing-prior argmax | 375 online episodes; 1,299,040 actual online microsteps | PID 840021 live; 2000 target |
| Coulommiers UrbanDreamer mainline | 87 online episodes; 331,470 actual online microsteps | PID 2341554 live under source_online_fp16_continuation_20261013; 2000 target |
| Montereau UrbanDreamer pretrain | decoder candidate ready at 2500 calibration updates; FM epoch0, 1664 local optimizer updates and 78,994 local sampled sequences at08:43:34 | PID2471575 live pretrainer, supervisor2471391; no completed pretrain_summary; online watcher2471388 live waiting |
| Ingo baselines | IQL/GNAC/IPPO/onlineAC queue-complete; Dreamer4837/MAMBA2242 still training | Dreamer839573/MAMBA839306 live; MAPPO failed2754; MBPO waiting GPU |
| Coulommiers baselines | onlineAC1326/IPPO1276/GNAC1186/IQL968/MAMBA641 | five matching live processes; MAPPO failed532; Dreamer/MBPO waitingGPU |
| Montereau baselines | onlineAC1398/IPPO1158/IQL868/MAMBA636 | four matching live processes; MAPPO failed390; GNAC/Dreamer waitingGPU; MBPO latest639 but queue names dead PID862781 |

Episode evidence: `outputs/city_baselines_two_city_fresh2500_20261011/<city>/<method>/metrics.jsonl`; Ingo baseline root `outputs/city_baselines_noincident_noprior_penalty_5000ep_20261008/ingolstadt_custom`; Urban `pilot_progress.json` in Ingo online_bounded_noprior_fresh_t2_2000_20261011 and two_city_urbandreamer_pretrain_supplement_20261012/coulommiers/online_mainline_2000. Current baseline status.json had stale committed episode numbers, so metrics tail supersedes it. Three MAPPO failures share `PPO collected behavior probabilities do not match the frozen actor`; this audit did not restart them.

## Pretrain and fair-comparison contract

Ingo completed pretrain: 49,636 WM/decoder updates, 2,605,729 sampled windows, epoch1, zero online episodes, codec_ready. Immutable comparison snapshot SHA2f63f0b2a5fe18f3a3bd9be27ffd1e63a2755464e7f2944e1227334f0ffe6af6 remains valid completed-pretrain source. Completion of an epoch is not independent proof of statistical convergence.

Expanded Coulommiers completed pretrain: 38,056 cumulative WM/decoder updates, latest-epoch1,155,192 samples, epoch1, codec_ready; no online episodes in pretrain summary. Prior-corpus continuation counters are explicitly retained. Current train/validation corpus9096/1694 trajectories,1,155,192/215,138 windows. Montereau expanded corpus8824/1550 trajectories,1,120,648/196,850 windows; pretrain incomplete. Ingo original corpus24818/4384 trajectories,2,605,729/460,640 windows, identity5254f01c7863d3e8524d4cf118983fae1bfbe755832ece48380acbdd11eab443.

All three current configs declare `fm_training_mode="single_step"`; Ingo `_world_loss` sets depth1 and calls independent_transition_fm_loss. `supervision_horizon=3` does not imply recursive MS. No MS gate in this pretrain path. Current main online configs set disagreement_weight0.0. Therefore main run is GraphPrior+FM with no recursive multi-step/gate/reward penalty, not proposed Full=MS+Gate+Penalty. Removing nonexistent MS/gate cannot be presented as a valid ablation of this checkpoint.

No sufficiently trained standalone deterministic/Gaussian world-model baseline was found in inspected pretrain-summary inventories. Regression/Gaussian component models are labeled development_screen and their data loader hardcodes development_manifest: ~87,685 unique training windows versus Ingo2,605,729. Existing sampler uses replacement at episode/window level. 2k/8k screens must remain excluded from formal comparison per user instruction. Simply running49,636 updates on87k replacement windows would not match full-pretrain data coverage.

Recommended independent baseline protocol: new immutable campaign bound to full original Ingo train/validation manifest and completed shared codec; same observed-state/control/previous-cohort count and background-speed semantics; matched backbone/history/action event inputs; sufficiently train deterministic MSE and heteroscedastic Gaussian NLL at least one genuine without-replacement full-corpus epoch, preserve optimizer/sampler/RNG cursors, then bounded prespecified extra epochs based on development convergence. Use matching train exposure rather than just nominal update count (batch/tails vary); report parameter/runtime costs. Use new unseen final demand bank after decisions, not inspected validation. Freeze codec for all new variants; disclose difference from original FM whose decoder was jointly calibrated, or retrain a matched FM control on shared frozen codec when drawing architectural causal conclusions. No such training was executed by this audit.

## Reporting defect and immediate repair handoff

Frozen Coulommiers evaluator `source_online_fp16_continuation_20261013/tools/run_online_pilot.py:143–149` defines complete from last transition terminal and censor flag from not-terminal. Deadline termination is not completion. Exact milestone80 example in online_mainline_2000/pilot_progress.json evaluations[2]: hash53c81df7a5b0881d919f2f5c8f48cf7cf20a55f61b36c2a5bafa66e538dc34c6, population1814,started1775,arrived960,nonarrivals854,3810microsteps, yet completeTrue/censoredFalse/travel_minutes26.546667. All three milestone80 rows have same defect. Root notified and owns surgical current-source fix with tests; do not rewrite archived records or change running frozen source.

Real reward is observed cumulative C(T)/(60*B(T)), denominator started vehicles, not final completed-trip average. Report deadline cost, started/completed/never-departed/unfinished and completed-only tail separately; final paired evaluator needs declared drain rule. Censor flag repair alone does not produce completed trip times for unfinished vehicles.

## Coverage of six requested experiment sections

1. Overall routing: incomplete3city final table, one independent training seed42 only; current milestones/development seeds are not independent seeds. Strong heuristic controls exist. No final equal-budget trained3city MARL comparison. Mainmethod seed bank314159 differs baselines10042–44; source/demand hashes must be paired before ranking. Baseline20042–44 holdout has already been consumed historically. Need newly frozen final bank and same checkpoint-selection rule.
2. World prediction: completed-pretrained FM probability/multistep/dependence/action probes exist and root owns their reports. Fully trained Reg/Gaussian and NoPrior model comparison absent in inspected summaries. Do not import old screen rankings.
3. Graph spectrum: completed development raw/hybrid/latent spectral analysis exists at component_experiments_20261011/spectral/summary.json. Hybrid rank16 energy1.48584%, modefraction1.87573%; does not support broad lowfrequency domination. Training-matched no-prior/random-rank/lowfrequency performance controls still need sufficient training.
4. Stochastic/MS: current checkpoint has independent single-step training. Old sample_ms/mean_ms/recursive_fm diagnostics are development-screen artifacts; cannot claim current pretrained FM+MS versus noMS. Need a separately trained, explicit recursive or sample/meanMS objective under full sufficient-budget protocol if this methodological claim remains.
5. Disagreement: new pretrained_evaluation_20261012/disagreement_v1 currently has plan/source/prepared verification only, no scored completed report found; root owns this active lane. Four-way reward-penalty/gate policy ablation absent; beta0 main does not verify penalty usefulness.
6. Reliability/generalization: conditional-futures/action/dependence outputs exist (root auditing). Demand development grid36conditions at scales.7/1/1.3,4profiles,3demandseeds compares heuristics and early Urban actor episode22, checkpointc89e7ff4...3d712, explicitly development. This is not final mature-policy OOD or independent-training evidence. Current improved actor can be evaluated on same DEVELOPMENT bank without finaltest claim; never tune on fresh final bank.

Sample efficiency needs two declared views: total unique real collection including offline/codec ancestry, and online fine-tuning with explicit common offline-data offset/equal offline access. 30s windows are not 1s interactions; repeated replay epochs do not add physical collection, but source trajectories need deduplication and actual executed microsteps. Resumed Ingo baselines retain historical reward/optimizer trajectories and must include all prior real interactions. Running/queued jobs do not satisfy remaining sections.

Next low-impact full-goal actions compatible production: fix evaluator censor reporting (root already acting), audit/train full-corpus matched Reg/Gaussian in independent campaign subject to resource admission, and add common paired DEVELOPEMENT evaluation for matured Ingo checkpoint versus strongest selected baselines/heuristics while retaining final-bank blindness. Parallel infrastructure diagnosis of MAPPO behavior-probability mismatch is needed before baseline final table, with a new explicit resume/frozen-source provenance; never silently alter active baseline children.

## Baseline implementation and exact failure handoff

Verified latest traceback paths: Ingo `outputs/city_baselines_paper_reward_20261010/logs/ingolstadt_custom_mappo.attempt3.log` (train_toy_baseline.py374→ppo.py317); Coulommiers/Montereau `outputs/city_baselines_two_city_fresh2500_20261011/logs/{coulommiers,montereau_fault_yonne}_mappo.attempt1.log` (train420→ppo318). All fail on frozen-actor collected-behavior probability equality invariant. No cause or safe bypass is proven by this read-only audit.

Existing native component counterparts are available in experimental_dynamics.py (`regression`, `graph_regression`, `gaussian`, `no_prior_fm` etc.), sharing Dynamics condition/action encoding. run_component_online_policy.py separates component_seed from policy_seed, initializes fresh actor/latentV/eventQ/targetV/readout and preserves frozen codec. This infrastructure is currently explicitly development_screen-only, bound to development_manifest and single-step beta0 runtime; completed-pretrain admission/full-corpus protocol must be added before using it as a formal counterpart. Existing interface is a usable starting point, not evidence that sufficiently trained counterparts or3independent downstream seeds have been run.
