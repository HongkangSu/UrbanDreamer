# Fixed entropy annealing and checkpoint continuation, 2026-10-10

The user requested a shared fixed actor-entropy schedule and continuation from
current training episodes. The accepted coefficient starts at0.002, holds for
10% of the original episode budget, follows a cosine decay to0.0001 at80%, then
stays at0.0001. It is not divided by episode population or advantage scale.
Evaluation and imagination-pass counts do not advance the schedule. All actor
updates within one real episode use the same coefficient.

`urban_dreamer/learning/entropy_schedule.py` is the common scheduler. AC, GNAC,
Dreamer, MBPO, IPPO, MAPPO and MAMBA use it. UrbanDreamer imagination training uses
the same scheduler and the global completed real-episode counter. IQL has no
entropy loss; its original epsilon-greedy mechanism is retained and the protocol
marks entropy as not applicable. DTA-SO and Random remain excluded.

For5000 episodes, beta remains0.002 through500 completed episodes, is0.00105
at2250 completed episodes, and is0.0001 from4000 onward. For a2000-episode mainline,
the corresponding boundaries are200 and1600. Restoring a checkpoint never resets
this clock. Completed experiments retain their original results and budgets.

The baseline handoff uses `tools/continue_city_entropy_annealing.py` and the new
controller directory `outputs/city_baselines_entropy_annealed_20261010`. Existing
per-method output directories, history and archive compactors are retained.
Original checkpoints are backed up before explicit config-fingerprint migration;
all remaining learning state is compared by a semantic digest, including model
weights, optimizers, replay, RNG and counters. In-flight uncommitted episodes may
be repeated from the durable checkpoint boundary. Source revisions overlay only
entropy-specific changes, preserving the frozen model and simulation code.

The completed UrbanDreamer entropyfix2000 run is not restarted. The active
world-model pretrainer retains PID3839431 and its original source/config/checkpoint.
A verified replacement bridge waits for that pretrainer to finish, then launches
the originally planned2000-episode online run with an annealed seed. Its plan is
`outputs/ingolstadt_full_retrain_20261009/entropy_annealing_future_plan.json`.

Old controllers remain stopped as wait-status keepers. Never resume them after
handoff, since the new controller owns future launches. Immutable backups and
explicit migration records distinguish old fixed/population runs from the new
continuation. This is a mid-training intervention, not a from-scratch matched
annealing benchmark.
