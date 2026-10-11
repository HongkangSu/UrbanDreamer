# Short-cycle network repair: verified candidates, not deployed

Update 2026-10-09 17:52 UTC: **Coulommiers v5 is now deployed** following
explicit user acceptance for fresh baseline training. Montereau uses the
separately validated occupancy1500 demand on its original geometry. New queue:
`outputs/city_baselines_v5_1500_noprior_penalty_20261009`. DTA and Random are
excluded. Historical diagnostic conclusion below describes the state before
this user-authorized deployment.

Historical status: **not deployed at the end of the original diagnostic**. The requested network-level intervention was implemented and tested, but the candidates did not resolve cold-start sampling gridlock. Production networks, demand, running baseline jobs, and their queue remain unchanged. No new training batch was launched.

## Implemented preparation

`tools/patch_short_road_cycles.py` prepares isolated network bundles. It removes passenger connection pairs using cumulative active-OD reachability checks, accepts evidence-bearing additional restrictions, rebuilds junction internals using netconvert, and regenerates the existing registry format. It preserves source inputs and supports `--input-root` for reproducible preparation from backups. Signal validation compares retained connection state-versus-time functions exactly, including safe static-phase compaction.

The strongest moderate candidate is `outputs/short_cycle_patch_20261009/v5`:

| Property | Coulommiers | Montereau |
|---|---:|---:|
| Removed road connection pairs | 304 | 346 |
| Removed lane connections | 308 | 353 |
| Active OD retained | 489/489 | 712/712 |
| Expected demand retained | 1811 | 2806 |
| Cycles entirely composed of roads at most30m | 0 | 0 |
| Observed blocking-region road count, now acyclic | 75 | 281 |

All original road/lane numeric attributes, OD files and agents were preserved in this candidate. Remaining signal indications and timing, bidi lengths, and registry/XML correspondence passed independent checks. This does not make the entire city acyclic: longer roads can close new loops.

## Cold-start checks versus trained-policy checks

All following cold-start checks use the same fresh AC initialization seed42, no shortest-path prior, 1800s demand release and 3810s deadline. Frozen means continuously active on the same external road with at most1m change in remaining distance across21 samples in the final600s. It is sampled persistent-stall evidence, not a claim about mathematical permanent deadlock.

| Network / candidate | Completion | Final600s frozen | Final600s arrivals |
|---|---:|---:|---:|
| Coulommiers original, fresh AC | 7.451% | 1561 | 1 |
| Coulommiers v5, fresh AC | 7.937% | 1528 | 1 |
| Montereau original, fresh AC | 5.187% | 2111 | 0 |
| Montereau v4, fresh AC | 4.611% | 2192 | 0 |
| Montereau core length floor30m | 6.304% | 2332 | 0 |
| Montereau core length floor90m | 6.736% | 2485 | 0 |
| Montereau junction joining15m | 5.403% | 2204 | 0 |
| Montereau whole-graph OD-safe feedback pruning | 8.069% | 2261 | 2 |
| Montereau additional119 cuts,10OD/39 rows removed | 5.757% | 2300 | 2 |
| Montereau core minimum3 passenger lanes, safe new yield signals | 7.313% | 2420 | 1 |

The OD-removal trial changes the population and demand hash: it has2779 realized vehicles versus2776 in the original trial. The other Montereau fresh-policy comparisons retain the exact original demand hash. Enlarging lane lengths or lane count changes physics; these are explicitly separate candidates.

The previously trained Coulommiers AC at episode699 performs much better on v5:98.293% pooled completion over two demands and no final600s frozen vehicles. DTA after five updates completes all three evaluation demands. **This does not establish that restarting training from random weights becomes stable.** The fresh-actor check above is why the candidate has not been published as a cold-start repair.

## Constraint and mechanism findings

Ingo is not immune: current IPPO episode3590 had280 unfinished vehicles, all with unchanged sampled road/position in the final600s and no new arrivals. The initial comparison therefore mixed network structure with policy maturity.

Montereau's active-OD requirement graph contains a121-endpoint strongly connected component. Preserving those OD relations necessarily preserves some road cycles. A specific13-road ring is already OD-essential before the whole-graph pruning experiment. Opening that ring alone requires losing at least2OD/8 demand rows. A combined <=100m and observed-AC-region cut loses10OD/39 rows, but actual simulation still freezes on longer loops.

Pruning every currently removable cyclic arc preserves all712OD but adds539 restrictions relative to v5. It raises the demand-weighted free-flow proxy by416.5s and still produces a seven-road waiting core in simulation. It is not a suitable production benchmark replacement.

Storage enlargement also failed. In the90m trial, several ring roads each filled with12 vehicles and a157.51m road held20. Increasing storage admitted more vehicles before freezing. A final throughput experiment widened217 core roads by423 passenger lanes. Its67 new controlled lane connections were assigned separate signal indices and yield-green during their template green windows; the original85 signal waveforms remained exact, and libsumo startup had no unsafe-green warning. It still froze elsewhere.

These checks support persistent cyclic storage/FIFO blocking under exploratory routing. They do not support claiming that moderate static road pruning alone solves cold-start training stability. A further intervention needs a deliberate change of scope, such as a different network/demand design or explicit simulation-side prevention/recovery of cyclic waits. Shortest-path prior was not enabled in any trial.

## Artifacts

- Corrected episode reward plots: `outputs/episode_reward_curves_20261009/episode_rewards.png`; raw shared episode reward, individual timeout penalties plotted separately.
- Initial extended-time experiments: `outputs/unfinished_diagnosis_20261009/summary.json`, `completion_extension.png`.
- Candidate inputs and original backups: `outputs/short_cycle_patch_20261009/v5/`.
- Fixed-policy/DTA validation: `outputs/short_cycle_patch_20261009/validation_v5/`.
- Fresh AC old/new checks: `outputs/short_cycle_patch_20261009/fresh_initial_ac/`.
- Independent static verification: `outputs/short_cycle_patch_20261009/verification/verification_final/`.
- Capacity and topology counterexamples: `outputs/short_cycle_patch_20261009/capacity_variants/`.
- Conditional handoff helper: `outputs/short_cycle_patch_20261009/handoff_training.py`; tested but **not executed**. Its integrity gate is not a dynamic quality gate. Do not run it merely because static validation passed.

The failed signal-cloning lane candidate was stopped and rejected; its successor with safe new-lane yield signals completed normally. All diagnostic processes were finished or explicitly stopped. Existing production training was not interrupted.
