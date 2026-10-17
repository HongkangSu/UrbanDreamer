# Completed-pretraining evaluation protocol

User scope: all new formal model evaluations use the completed Ingolstadt Prior+FM snapshot, SHA256 2f63f0b2a5fe18f3a3bd9be27ffd1e63a2755464e7f2944e1227334f0ffe6af6. Its durable metrics record 49,636 dynamics updates, 2,605,729 training-window exposures, one completed epoch, and zero online episodes. Completion of the prescribed pretraining is not a separate convergence proof.

The 2k/8k fresh-dynamics screens remain development diagnostics. Their controllers stopped scheduling new work; already-running children may finish and all artifacts remain preserved. The regression-2048-centered residual candidate is dormant. No screen checkpoint may become a formal baseline without adequate training and matching validation.

## First requested evaluations

1. Shared K32 physical trajectory bank, nested K1/8/16/32 sample-versus-mean diagnostics, decoded-draw mean, fixed condition seeds, Euler100 and horizons1/2/4/8. Pilot2 demand groups, three observed traffic phases plus empty stress pergroup; extend8 groups only after integrity/finite-output audit. Conditions are fixed initial cohort and recorded actions; speed labels retain factual background traffic.
2. CRPS and Energy Score (declared empirical V and supplementary U estimators),90% interval coverage/width, thresholded congestion Brier, physical count/bounds diagnostics. K1 and pointreferences do not supply estimated prediction intervals. Stochastic-bank ensemble mean is a pointreference, not separately trained regression. Zero-source-noise FM retains the same pretrained parameters and is not necessarily the conditional mean.
3. Bounded finite-action decision fidelity using the same fully pretrained FM. Verify complete initial state/Python/controller/RNG consistency and no-intervention traces. Native meso loadState has failed; startupload loses interpolatedpositions. Preserve these failures and use explicitly labeled pairedprefixreplay if complete boundary-state proof succeeds. Firsttrial uses fixed preregistered future demand across actions, not repeated-condition Poisson outcomes. Model trajectories generate their own causal future actionevents; no truthfuture events are inputs. Report realized finite-action regret separately from estimated model expectation.

## Scientific boundaries

These are development evaluations, not untouched finaltests or downstream trained-policy performance. Two demand groups provide a bounded pilot, not publication-grade replication. Repeated bank subsets are Monte Carlo sensitivity diagnostics, not independent traffic outcomes. Future independent-demand branch experiments must preserve active initial traffic and explicitly reconstruct external demand generation; neither prefixreplay nor independent episodes alone proves conditional future-demand calibration.

A fully trained deterministic/probabilistic baseline is still required before a cross-model superiority claim. Existing undertrained screens are excluded. Equal-real-interaction downstream policy comparisons, paired demand tests, action-ranking expected regret, and finalholdout evaluations remain laterrequirements.

## Completed-stage evidence and current continuation

Completed formal-development panels use the same completed pretrained snapshot only.
- Eight-group K32 prediction panel: K1→K32 H1 occupied count MAE0.6360→0.5994, speed0.9430→0.8899; H8 K32MAE0.7715/1.4452. Actual count coverage43.73/45.77/46.63/39.36% versus sampler finite-bank reference≈85%; no learned-regression comparison.
- Spatial/temporal dependence: adjacent speed Variogram improves against marginal-preserving spatialshuffle, but count and stackedtrajectory Energy do not. Original/temporalshuffle trajectory Energy_U14.8111/14.7969. Small mixed development effects, not global superiority.
- Uniform prospective finite-action panel:8 states,19 real actions,K8,NFE100. Sampled/zero both7of8 best; each1aggregate cohort-vehicle-second regret; paired normalized-regret interval crosses0.4states entirely tied. No stochastic-policy advantage established.
- Fourstate repeatedfuture panel:2initialseeds×2times,R8each,K32+zero,NFE100. All32real,128sampled,4zero retained; complete SUMO/Python/RNG pairing and independent audits pass. H240 mean-minute costMAE0.0216551 versuszero0.0240578; CRPS0.0201622; empirical90% costcoverage1/32. Bias flips byinitialseed. Root independently rehashedbanks and recomputed allH costMAE/CRPS/coverage.

49combined targetedtests passed; evaluator sources preserved incfa49bf. Production/model code unchanged. Existing unrelated supplement/recovery edits are not part of this commit.

Next bounded diagnostic: completed-checkpoint H1 fullrecorded vsC0recorded action events, sameinitialstate/target/noise on24 nonempty developmentconditions. Factualfull events are privileged offline inputs, not deployable forecasts. Verify C0draw reproduction before scoring full arm, useK8,NFE100, freezeinputs/source/plan, preserve reports. No newtraining or undertrainedbaselines.

Independent architecture audit: firstwindow executorcost uses initialdecode BEFORE first stochasticFMtransition. H1 real andmodel costvariances areboth0, so H1coveragefailure reflects deterministic bias. AtH8 acondition hasmodelSD26.78≥realSD24.51butcoverage0, exposing important meanbias; noiseinflation notjustified. Model futurebackgroundbirths absent; full observed single-step pretraining and C0autoregressive execution differ. Lowranknoise restriction is a hypothesis, not a proven cause.
