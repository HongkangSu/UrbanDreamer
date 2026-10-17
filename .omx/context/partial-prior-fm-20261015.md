# Authorized coarse-source validation — live handoff

User authorized the two-stage low-cost validation and then said continue.
Stage 1 finished and independently verified: frozen selected direct heads,
GPU Euler4 and Euler100 paired on old32states16groups, no retraining/reselection.
Commit357dd55; docs/direct_context_steps_20261015.json. At100: zeroES8.259512,
hintES8.189493 (difference CI crosses0), shiftedsource+refES9.634003 worse.
Both occupiedCRPS improve forhint. Noise/control/weights fixed. This is solver
robustness on reused test states, not an independent fresh test.

Stage 2 RUNNING, owned PID996234 on GPU0, output
outputs/partial_context_fm_20261015/v2_gpu0. Complete checkpoint49,636updates,
SHA2f63f0b2a5fe18f3a3bd9be27ffd1e63a2755464e7f2944e1227334f0ffe6af6.
Fixed weak state-only tau.1 HYBRID matched ridge prior, oldrank16Gaussian noise.
Three equal-budget partialadapted arms, all hintb:
B(sourceZ,refZ), C1(sourceZ+b,refZ), C2(sourceZ+b,refZ+b).
Train ONLYlastTransformerblock/reference_proj/originalflow_head+newDirectResidual,
1,414,528params/arm. Gradient through frozenprefix retained; fullvelocity CFM
loss target=targetZ-source. Freshidenticalnewheads and same originalbackbone.
64traingroups768windows×4pairedtime/noise=3072graphviews; B2 wholegraphs.
Max4epochs/min3,1536updates/epoch/arm, max6144; valK4Euler100 everyepoch,
8fixedstates4groups, bestES independentperarm epoch0eligible, .2%meaningful
patience3, capmustflagbudgetlimited. Newtest16groups32states K8Euler100,
only opened AFTERvalidation_selection. Newlyreservedtestgroups excludeallprior
studies including directv2. Do not tune using test.

Trainonlycostpilot outputpilot_v1: 2warmups3measuredFB NOoptimizersteps,
hashes/RNGunchanged, correctgradients. Tripleepoch~348sec estimate, peak351MiB.
Original trainingv1 PID977393 stopped ONLY ownedjob beforefirstepochfinished:
GPU1 two other compute jobs sharedSM. v1/stop_reason.json preserved; no selection
or test. Restartv2 source/dataset/split bytesidentical, protocolONLYdevicechange.
Split continues referencing v1 INPUTcache_manifest, intentional andunchanged.
Generated arraymanifest is frozen_input_manifest.json (collisionfixed beforelaunch).
Do not stop otherjobs.

Owned uncommitted files:
tools/train_partial_context_fm.py (pilot/helpers), tools/run_partial_context_fm.py
(fulltrainer), tests/test_partial_context_fm.py, tests/test_run_partial_context_fm.py.
28focusedtests pass, compile/diffcheckpass. No dependencies/corechanges.
Nativeagentfm_background_scope_audit implementedhelpers/tests, reviewedrunner,
prepared tmp/verify_partial_context_fm_20261015.py. Run atcompletion:
URBpython tmp/verify_partial_context_fm_20261015.py --study outputs/partial_context_fm_20261015/v2_gpu0
Independent bankmetrics/CI/selection/equalbudget/cache/noise/rms/checkpointscope.
No newpartialtestread beforeselection.

OptionalH4 gate prescribedBEFOREtest: h4_followup_protocol.json copiedv2.
C1 source_effect orC2 joint_effect qualifies only ES+bothoccupiedCRPS95CIupper<0
andtotalcountabserrorpointdelta<=0. Thenfirst8fixedteststates initialC0cohort
H4Euler100K8 baselineB+eligible, no tuning. Rootrunnerprepared(notrun):
tmp/evaluate_partial_h4_20261015.py [study-path], defaultv2_gpu0 device0.
Run aftermainsummaryevenifgatefails tosaveh4_gate; ifpositive, independently
verify H4banks using NumPyhelper tmp/verify_direct_context_fm_20261014.py.

Remaining: monitor tocompletion; independentmainverification; gatedH4;
docs/partial_context_fm_20261015.json actualresults andlimitations;
commit ONLYowned4files+docs followingLoreprotocol. Chinese simplefinal,
sourceeffect BETWEEN separatelyequaladaptedmodels not within-fixedweights.
If cap, no convergenceclaim; no policy/generalarchitecture necessityclaim.
Allotherdirtyfiles belong otherwork; leave them untouched. No broadpausedgoal
resume or newgoal. Userprefers finishedpretrain, lowSUMObudget.

Latest: GPU0 epoch3 finished elapsed1160sec; BbestES8.92693@1,
C1best12.25734@3, C2best10.28614@3. Epoch4running. Actualepochtimes
~369s then335s then~298s; postprocessingwatcher PID1028003 waits summary.
It sequentially runs independentmainverifier, H4gate/followup, independentH4
verification, then docs generator. Logs: v2_gpu0/postprocess.log; completionfile
postprocess_complete.json. Root must inspect errors and finishcommit/final.
Scripts tmp/watch_partial_postprocess_20261015.py, verify_partial_h4_20261015.py,
report_partial_context_fm_20261015.py prepared. H4runner acceptsstudyarg,
defaultv2_gpu0/device0. IfpositiveH4 runs ~fewminutes; ifgatefails noH4dir.
Source andheadscopedtests28pass. Independentcheck loadedrawCPweights_onlyTrue:
18allowedmodeltensors1,348,672elements +4residualtensors65,856=1,414,528.

COMPLETE: v2GPU0 finished1683.8sec~28min; independentpostprocess allpass,
noH4eligible. Commit3bb9e1d owns4code/testfiles+docs/partial_context_fm_20261015.json.
Freshtest B ES8.838768 countCRPS.511316 speedCRPS.890532 massabs9.338744;
C1 ES14.042972 count.864280 speed1.097508 mass43.006282;
C2 ES9.618947 count.597519 speed.909456 mass14.787629.
C2−B ES+.780179CI[.551035,1.013429]; count+.086202CI[.068183,.106136];
speed+.018924CI[-.025956,.063071]NS. C1−B ES+5.204203CI[2.73514,9.368984].
6144updates/arm, bestepochs B1/C1=3/C2=4, budgetlimitedtrue (C2stillimproving).
Mainverifier12312bankmetricchecks702CI,435fileshashed, all3072draws/RMSreconstructed.
28focusedtestspass; compile/diffcheckpass; all4liveownedfilesmatchfrozenbytes.
Postprocesswatcherfinished; no ownedtrainingstillrunning. Globalpausedgoal untouched.
Report currentbudgetno coarse-sourcebenefit, notuniversalno-use/fullconvergence;
allarms hintb so no proofhintnecessity; no regression/policycomparison here.
