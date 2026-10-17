# Current task and accepted experiment

User requests all baseline training on Ingolstadt, Coulommiers, Montereau, each
1000 real episodes, using restored K4 current_to_dest route actions. Latest user
explicitly confirms all policies including SP/DSP/DTA should reroute every30s
for a fair setting. Priority is Online AC versus dynamic shortest.

Run9 methods percity: online_ac,gnac,dreamer,mbpo,ippo,mappo,iql,mamba,dta.
Prepare new frozen source/network/config copies; preserve currentmeso,noincident,
Poisson1800s release,3810s deadline,globalobservedmeanreward,noprior,argmax.
OnlineAC evalevery20; othermethods every100. Eval3fixedseeds10042..10044 and
holdout20042..20044, both matched SP/DSP references. Seed42. Everyepisodecheckpoint
andrealarchive; memory-aware existingqueue + losslesscompactor. No oldjob restarted
or overwritten. GPUs currently0free,1about3GBused. Diskabout142GBfree.

Prior task completed: 5pairedseed314159 SP/DSP comparisons in
outputs/ingo_current_to_dest_shortest_20261013_173936; dynamic pooled198.347s vs
static226.647s (12.486%lower),P95316vs529 (40.265%lower),all5223carscompleted.
All69114routechoices+149475conditioningevents independentlyaudited. DTA50trial
wasnotlaunched; expanded1000episodeDTA supersedes that trial.

Implementation ownership: route_session agent baselinecommon+RealActorCritic+
routecollectiontests; route_imagination agent Dreamer/MBPO/Mamba+modelbasedtests;
root queuepreparation+SP/DSP reference CLI+OnlineACcomparisonmonitor+launchverification.
DTAadapter doneand19tests passed; next_hop behavior kept. Existingroot route feature
previously478tests passed, newSP/DSPdriver22tests passed. Sourceuncommitted.

Before launch: finishagents,testintegration,snapshot+27train/6refmanifest audit,
smallrealCLIwarmup/smokesincludingOnlineACandimaginedmodels, thenstartqueueandmonitor.
Verify runningjobs commitrealcheckpoints/allprotocolsroute; report actualrunningvsqueued
andcomparisonartifact. Training completionmustnotbeclaimed atlaunch.
