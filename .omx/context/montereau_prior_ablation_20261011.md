# Active Montereau experiment after user prior correction

Active task: compare training efficiency with decaying dynamic-shortest prior vs no prior.
User requires deterministic evaluation WITHOUT prior; sampling prior must decay to zero.

Current primary run root: outputs/montereau_ac_prior_ablation_500_20261011/
Two fresh arms: decaying_prior/run and no_prior/run; both500episodes, same original
zero-episode actor/critic/Adam/RNG (verified), same seed43..542, MontereauOD-stagger.
Linear prior weight1 at episode1 to0 at250, then0 through500; otherarmalways0.
Evalevery50seeds10042..44, actor-onlyargmax; finalholdout20042..46.
Main artifacts comparison.json/csv/png auto-updated; launch PID identity perarm;
supervisor_pid.json ownsboth and resumes max3attempts.
95 tests passed plus firstdecayingarmepisode1 fullNPZ/travel times exactly match
legacyfixedprior episode1 (bothweight1). RuntimeGPUunchanged CPU1perarm.

Old outputs/montereau_dynamic_prior_ac_500_20261011 run stopped at episode199;
its supervisedprocesses stopped safely and all checkpoints/archives preserved.
Do not accidentally resume it: fixedprior phase cannot be relabeleddecaydata.
Oldwithprior argmaxvalidation97.72% wascombinedpolicy, notstandaloneactor.
Corrected actoronly ep179 completed16.6804%,ATT40.6167min on3seeds. PureSP30s
97.7231%,ATT3.19114min. Those are historical, notnewarmperformance.

Implementation: tools/train_dynamic_prior_ac.py; schedule protocolfields,
set_episode_prior, saved rowweight assertion, restorefromcheckpointepisode,
zero prior skipsunusedrouterandcostobservations, deterministicrawactoronly.
Detailed methodology docs/montereau_prior_ablation_20261011.md.

USER STOPPED BOTH ARMS. Supervisor637317 and trainers637319/637320 exited; stopped_by_user.json prevents automatic restart. Read stop_verification.json for saved rounds. Do not resume without explicit request.
