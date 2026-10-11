# Paper reward implementation and continuation

User authorized changing default UrbanDreamer/all baseline rewards and resuming
currently running baseline jobs after migration. No further permission needed.

Default shared physical reward: -delta(C / max(B, 1)) / 60, C is actual started
vehicle-seconds, B counts actual starts strictly before physical cutoff. All
default individual timeout penalties are zero. Actor remains ten features, with
causal active/B fraction. Critic context expands 2 -> 4: time fraction, active/B,
log1p(C/60), log1p(B). Default model-disagreement risk weight becomes zero;
explicit algorithm regularization stays separately recorded.

Ownership: root global_reward.py, SUMO sessions, sampler, workflow checkpoint and
config, root tests/docs. paper_reward_baselines owns common/all baseline methods,
AC/baseline CLI/tests. paper_reward_mainline owns trainer, parallel imagination,
replay/models policy/event critics/mainline tests. paper_reward_continuation owns
new city continuation utility and tests. paper_reward_future_mainline owns new
future mainline handoff utility and pretrained queue initialization migration.
All agents preserve unrelated workspace edits.

Runtime inventory: entropy controller 3937519 owns nine active city baselines:
Ingolstadt MAPPO/IQL/MAMBA; Coulommiers AC/IPPO/MAPPO/IQL; Montereau AC/IPPO.
Twelve other jobs are GPU-wait queued. Existing completed jobs and archive
compactors remain preserved. Stage fresh execution sources from each currently
frozen city source (preserve each city's network patches), overlay reviewed code,
backup durable checkpoints, expand context weights/Adam moments with zero columns,
preserve all other learned tensors/cursors/RNG, clear obsolete reward replay,
invalidate old best comparisons, and resume --global-reward-only
--reward-objective observed_mean under new controller. Require new-episode
evidence for all active learners and ownership-guarded recovery on failure.

Mainline pretrainer 3839431 must remain running unchanged. Old original mainline
controller is already SIGSTOP; entropy future waiter 3896491 owns handoff. Stage
new source/new plan, explicitly migrate completed-pretrain checkpoint at online
initialization, then replace only future waiter. Verify same pretrainer PID and
immutable original plan/config. Mainline event critic context expansion inserts
two columns in the middle, not at matrix end.

Completed so far: shared reward utility and continuous session accounting,
sampler accounting/live causal event context, mainline and baseline consumers,
new replay8 fields, migration utilities underway. Before migration run targeted
and broad appropriate tests plus real SUMO smoke. Agents have passed substantial
isolated tests; root owns final integration, migration application and liveness.
