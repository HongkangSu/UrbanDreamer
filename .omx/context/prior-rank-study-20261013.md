# Explicit prior-rank request (broad goal paused)

Completed only the user-requested rank study using completed checkpoint2f63 and frozen distribution source. Outputs outputs/pretrained_prior_rank_20261013/v1. New tool/tests evaluate_pretrained_prior_ranks/test_pretrained_prior_ranks. CPU2threads, no SUMO/GPU/FMgeneration or fullWMtraining.

Same64training/32validation demandgroups,768/384windows. All prior hidden/condition/codec weights frozen; each rank8/16/32/64/128/256/512/852 has originalhead transfer and fullysolved 4160-coefficient ridgeoutput fit. Spectralprojected-target oracle is a capacitybound, notdeployableprediction. Rank853 analyticalunprojectedreference only because actualprior clamps E−1.

Heldout rank16 targetlatent increment capture1.44954%, projection residual98.55046%; original fullSSE improvementvszeroprior.85131%, refit.94986%. Within projected error originalreduction58.7295%, refit65.5285%. Rank128 capture17.4007%, originalgain6.15353%, refitgain9.23199%; rank512capture67.2583%,refitgain32.5753%; rank852capture99.8915%,refitgain42.9062%. Frozenbody trainedr16 remainsanimportantlimit.

Physicalcoarse decodedmetrics are mixed: zerolatentprior countall.244450/occupied1.249096/speedoccupied1.343828; refit852 countall.194185/occupied1.357189/speedoccupied1.490288. Do not hide occupiedcount/speed worsening behind allroadcount improvement. Root paired128−16 refit: fullrelativeSSE−.082821CI[−.089602,−.077483], occupiedcountMAE+.0468435CI[+.035289,+.054930], occupiedspeedMAE+.112078CI[+.085600,+.131412].

9tests passed. Root independentlyverified all8normal equations/heldout SSE skills fromsufficientstatistics andpairedrankCI. All218 originalruntimefiles/checkpoint/data unchanged. Geometry-only correction normalizesnativeF32basis forsubspacecomparison; predictionsremainactualstoredbasis. Originalcompletefirst_runpreserved; everymetricsCSV/fit/selectionJSONbyteidentical andcoefficients/stats arrays exact. Correctnative/exactbasis overlap.99999999944424; outputreldiff3.29e−5.

No productionrank/config changes. Potential followup fullytrainedr64/128/256prior-body/FM comparisons, controlled source noise energy andpriorlossweight. Currentrank controlsBOTHprojection andnoise support: expectedGaussianenergy sigma²*r*64, so sigma_r=.1*sqrt(16/r) preservesenergy ifnoise rankchanged; projection-only test keepsnoisefixed. Fullgraph-normalized priorMSE also scales withactive r/M and shouldbe accounted for whencomparinglosses.

Broadgoal remainsuserpaused. ExistingbackgroundpreviouslyauthorizedReg/queuedGauss andCoulommierslongjob continue; oldagents instructed noadditionaloldgoallaunches, Ingo/Montereau longrecoveriesheld. Ranktask mustnotmarkbroadergoalcomplete.
