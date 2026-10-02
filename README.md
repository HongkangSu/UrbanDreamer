# UrbanDreamer

UrbanDreamer trains a routing policy for a fully controlled CAV fleet in SUMO.
Every vehicle chooses its next road edge when it first departs and whenever it
enters a new external edge. A learned world model predicts the network state
at fixed 30-second intervals. Vehicle decisions and world-model updates run on
separate clocks.

```bash
git lfs install
git lfs pull
conda env create -f environment.yaml
conda activate urban-dreamer
python -m urban_dreamer --config configs/toy_cross_bottleneck.toml --device cpu \
  --output-dir outputs/toy_next_hop
```

The reference environment is Python 3.12 with SUMO 1.25. Alternatively, create a
Python 3.12 virtual environment and install `requirements.txt`. Ensure `sumo` is
on `PATH`. The runtime controls its own TraCI connection directly.

Bundled profiles cover Toy, Ingolstadt, Gretz, and Fontenay under `configs/`.
The Toy profile still specifies 150 online episodes; reduce the episode counts
in a copied configuration for a smoke run. City profiles have substantial
runtime costs, especially networks containing very short edges.

Each input directory contains exactly the following required files:

```text
networks/<network_name>/
├── <network_name>.net.xml
├── agents.csv
├── od_<network_name>.txt
└── edge_registry.json
```

`agents.csv` has columns `id,origin,destination,start_time`. Every row is a
controlled vehicle. Origin and destination are zero-based indices into the
`origins` and `destinations` lists in the OD file, whose entries are edge IDs.
Departure times are finite, nonnegative seconds and may be fractional. Vehicle
IDs must be unique. The registry fixes the ordering of the network edges;
large city registries use Git LFS. There are no vehicle role assignments or
precomputed candidate routes.

Copy `configs/mainline.example.toml` to configure another network. The strict
TOML schema has four sections: `[inputs]` contains only `network_name` and
`network_dir`; `[run]` sets output directory, device, seed, and phase lengths;
`[simulation]` controls horizons, incidents, and the explicit
`vehicle_max_speed_mps` cap; `[resources]` controls batch, sequence, and gradient
accumulation sizes. Relative paths resolve from the TOML file. The macro budget
is `world_model_intervals_per_episode`, with 30 seconds per interval. A replay
sequence length of `T` means `T` transitions and `T+1` physical states.

The CLI accepts `--config` and optional `--device` and `--output-dir` overrides.
It runs warmup collection, world-model initialization, online collection and
imagined policy updates, then deterministic evaluation. Run the same command
to resume a matching `checkpoints/latest.pt`. Checkpoint schema 5 stores model,
optimizer, replay, random state, metrics, and environment episode cursor, plus
the pipeline phase and next unit. Inputs and semantic configuration must match;
relocating data or changing device/output path does not alter the semantic
fingerprint. Old checkpoint schemas and old configuration keys are rejected.
Existing output directories are preserved; use a fresh output directory for
this architecture. Completed runs return their saved summary.

World-model initialization uses `[run].world_model_initialization_epochs` and
`[resources].world_model_initialization_batch_size`. One epoch shuffles and visits
each valid replay sequence start once, retains the final smaller batch, and does
not cycle or duplicate samples to fill it. Gradient accumulation weights batches
by their actual sample counts. A checkpoint is saved after each complete epoch;
an interrupted epoch restarts from the preceding boundary with its RNG restored.
The old `world_model_initialization_steps` key is rejected. Environment warmup
episode counts remain separate from optimization epochs. Metrics report completed
epochs, batches, sampled sequences and optimizer updates separately.

Real baseline episodes can be archived independently of the bounded training
replay with `--save-episodes` in either baseline trainer. Each
`episodes/train/episode_XXXXXX.npz` stores boundary traffic, raw and normalized
action/departure counts, physical exposure, vehicle snapshots, successful action
events, realized demand and population, completion/truncation and provenance.
Evaluation and holdout episodes are excluded. NPZ files use concrete numeric
arrays and UTF-8 JSON bytes (`allow_pickle=False`), compressed atomic writes,
and stable episode filenames for restart replay. Zero and incomplete episodes
are retained. Load with `read_episode_archive`; `to_replay_episode` requires an
explicit matching session protocol before conversion to mainline replay.

`tools/run_city_baseline_dataset.py launch --output-root <directory>` launches
Online AC, GNAC, Dreamer, MBPO and uniform random collection in each of Ingolstadt,
Gretz and Fontenay: 15 independent jobs, 5000 training episodes per job by default.
It freezes a code/config snapshot, uses event/libsumo at 1 second, checkpoints
learned baselines every 25 episodes and evaluates every 100 episodes on fixed
seeds. Each task has its own tmux supervisor, log and recovery status; a monitor
writes `status.json`. The branch-only event action protocol is recorded explicitly
and differs from the all-next-hop microstep protocol. Downstream world-model
training and rollout conditions must use compatible action semantics. The launcher
does not start world-model pretraining automatically.
An independent compactor losslessly repacks checkpoint-committed NPZ files with
ZIP LZMA, verifying the decompressed bytes before atomic replacement. It retains
every array and field, and leaves newer episodes untouched so checkpoint replay
cannot race compaction. Collection supervisors preserve a 10 GiB disk reserve.
MBPO keeps its existing approximate recovery behavior: its real model buffer is
not checkpointed and only a tail of synthetic rows is restored after a failure;
the separately archived real episodes remain complete.

Learned baselines can use an annealed shortest-path policy prior via
`--sp-prior-weight 1 --sp-prior-temperature 30 --sp-prior-hold-episodes 200
--sp-prior-off-episode 1000`. The actor logits receive
`-weight * (full_free_flow_cost - minimum_legal_cost) / temperature_seconds`,
where the full cost includes the junction, candidate road and downstream route.
Weight is fixed during a real training episode and its updates/imagination;
it is held through episode 200, cosine-annealed to exactly zero at episode 1000,
and remains zero thereafter. Rewards, masks and network parameter counts retain
their existing definitions. Default weight zero preserves the unassisted policy.

Enabled runs evaluate both scheduled-prior and bare (`deterministic_no_prior`)
actors on the same demand seeds, and select the best checkpoint by bare-actor
performance. Prior settings and realized weights are stored in protocols, metrics
and training archives. MBPO drops cached synthetic rows whenever the weight
changes, avoiding mixing different prior schedules; its existing same-weight
historical-policy reuse remains approximate. The plotting tool uses bare-actor
evaluation as the default for a prior-enabled manifest and exports the scheduled
evaluation separately. Uniform random collection remains an unassisted control.

At a vehicle event, candidates are passenger-permitted successor edges that
can still reach the destination. Stable edge order and padding masks support
variable outdegree. A shared scorer uses candidate traffic, static cost, and
shortest remaining distance. Warmup samples a softmin over static remaining
cost; online collection samples the actor, and evaluation chooses its highest
score. A sole candidate still counts as an action. Entering the destination
edge ends decision-making, while travel time continues until actual arrival.

SUMO receives a short execution route consisting of current edge, selected
next edge, and a deterministic placeholder successor when the next edge is
not the destination. The next real event replaces that placeholder. Only the
selected next edge enters the action aggregate. Vehicles are inserted with
zero departure position and speed. Internal junction edges and waiting on the
same road edge do not produce new decisions. Failed route execution, missed
committed edges, and premature disappearance invalidate the episode.

The microstep is an integer number of milliseconds that divides one second,
no greater than one second, bounded using the shortest passenger edge and the
configured vehicle speed cap. Networks requiring less than one millisecond
are rejected. SUMO arguments that replace the network, routes, configuration,
simulation clock, speed settings, or managed routing are reserved. Vehicle
subscriptions track microstep events; full-network observations are extracted
at macro boundaries.

For a physical state `S_t`, departures, arrivals, and edge changes observed at
`t` are already reflected, but policy actions at `t` are pending. The action
aggregate is

```text
A_t[e] = number of successful choices of next edge e during [t, t+30) / N
D_t[e] = number of first-observed departures on e during (t, t+30] / N
```

`N` is the total episode demand. Repeated decisions by the same vehicle count
separately, so the action vector may sum to more than one. Empty windows remain
valid transitions in storage. Replay retains observed `departure_t` as training
telemetry. The world model takes latent traffic and edge-aligned `action_t`,
with graph/static features and FM integration time. Actions enter each road token
through the action GNN and edge-aligned injection; there is no global action token
or episode-context input in the dynamics or graph prior. Replay retains the
two-feature `context_t` for policy/value contexts and diagnostics. The world model
has no departure encoder or timetable input.
SUMO alone uses the demand departure schedule to inject vehicles. Future
scheduled departures keep an otherwise empty real episode alive.

The mainline regenerates demand in every warmup, online and evaluation episode.
Each supported OD has a fixed rate `lambda_od = reference_OD_count / T`, where
`reference_OD_count` comes from `agents.csv` and `T` is
`[simulation].demand_duration_seconds`. The original vehicle IDs and scheduled
times are not reused. For each OD, sample `N_od ~ Poisson(lambda_od * T)` and,
conditional on that count, independent uniform departure times in `[0, T)`.
This is a homogeneous continuous-time Poisson process with independent OD
streams. A generated vehicle keeps its OD for its entire trip.

The supplied Toy configs use `T=420` seconds; city configs use `T=1800` seconds.
Expected episode populations remain 320, 1035, 636 and 2068 respectively for
Toy, Ingolstadt, Gretz and Fontenay. Rates remain fixed across episodes; counts,
OD composition and departure times vary as realizations of the same demand
distribution. `T` only determines the arrival period, not the simulation cutoff.
Demand uses a separate random stream keyed by the run seed and persisted
environment episode counter, so checkpoint recovery reproduces the next demand
independently of policy or replay sampling.

`N` is the realized episode population. Replay schema 4 stores it with each
episode and returns it with sampled seeds, so imagination keeps the original
episode denominator. A zero-vehicle realization is retained without resampling:
it produces one empty terminal window, zero reward and no active training seed.
Future vehicle schedules are excluded from model conditions and imagination.
Actual `N` still normalizes model actions and rewards, while episode progress and
active-vehicle fraction remain available to actor observations and the critic.
The world model receives no explicit episode-progress or active-fraction context;
without a physical-time input it must statistically average phase differences
between demand generation and clearance when latent traffic and actions are
otherwise similar. Standalone baseline/AC experiments
also use this Poisson law, keyed by the base seed and SUMO episode seed, so
different methods can evaluate exactly matching demand realizations.

World-model training samples the full collected physical timeline, including
empty traffic, demand generation and the period after departures have ended.
Sequences remain contiguous and never cross episode boundaries.
Imagined rollouts require an active vehicle snapshot at their starting boundary.
Episode progress uses the full simulation budget, including
the completion buffer. After the final actual arrival, the last window is padded with empty
traffic while travel time and active-vehicle time retain their actual values.

The decoder speed channel is supervised with the forward window's endpoint
mean: `decode(z_t).speed -> (speed_t + speed_{t+30}) / 2`. Labels use the same
per-edge speed normalization as the observations. Replay and the latent encoder
retain physical boundary observations; other decoder channels retain their
boundary-state targets. The final sampled frame has no following observation,
so only its speed loss is masked. Predicted next latents use the same alignment:
`decode(predicted_z_{t+30})` describes the speed of `[t+30, t+60]` when both
endpoint labels are available. This applies to warmup and online training.
The label convention is included in the semantic fingerprint, so checkpoints
trained with instantaneous speed labels cannot silently resume this training.

Imagined rollouts freeze decoded traffic within each 30-second window and
advance the vehicles already active at the sampled start through external-edge
and junction events. Future demand is not inserted during imagination.
Each rollout lasts at most **8 windows (240 seconds)**, ending earlier if its
vehicles arrive or the real episode deadline is reached. Extending the eligible
sampling range does not extend the eight-window rollout length. Boundary physics precedes the next prediction; decisions
exactly on a boundary belong to the next window. Snapshots preserve departure
status/time, remaining distance, committed next edge, pending decisions, and
junction progress. New predicted speeds change subsequent travel rates without
resetting progress or resampling a committed choice. Junction time uses a
deterministic connection-length/speed approximation.

Training shares the global traffic reward. In a real 30-second window it is
`-active_vehicle_seconds / (N * 60)`, where `N` is the episode population.
Imagination uses exactly the explicit seed cohort's integrated active vehicle
seconds divided by its originating episode's actual `N * 60`. The active-fraction
readout remains auxiliary supervised output. Each completed 30-second imagined
transition subtracts `disagreement_weight` times the unbiased sample variance of
same-condition latent predictions, once at the right endpoint. A partial terminal
tail incurs only its actual physical cost and no fictitious full-window transition.
At the absolute episode deadline, **only an unfinished vehicle receives -5000**.
This is an individual terminal cost, excluded from shared window rewards and
all global critic targets. Arrivals exactly at the deadline are processed first.
The last own decision records the penalty once; with gamma=1 its Monte Carlo
return reaches that vehicle's earlier decisions only. Arrived vehicles receive
zero timeout return. Vehicles without policy decisions create no synthetic
training rows. Imagined rollouts use their active cohort; future departures
remain excluded.

The actor samples its masked categorical distribution during online collection
and imagined training. Evaluation and final actor holdouts use **argmax**.
Warmup retains static-cost softmin sampling. Macro critic returns use `gamma=1`
and `lambda=0.95`. Each decision receives global Monte Carlo reward-to-go from
its event timestamp, excluding earlier travel cost and retaining all later
endpoint risks. Simultaneous decisions share a target and an action-independent
baseline evaluated before the joint action. Event advantages are
scaled by the standard deviation of **global advantages only**, floored at 1,
without mean-centering. The actor then adds each row's own timeout return and
clips the result to ±2.5. Thus a failed vehicle does not rescale successful
vehicles' advantages through batch normalization. Entropy regularization
remains 0.002. There is no vehicle critic, local elapsed-time reward or
configurable reward mixture.

An ordinary eight-window imagination cutoff bootstraps the global critic and
adds no timeout penalty. Windows with no decisions still train the world model
and global critic. The online deadline is the world-model interval budget plus
`training_completion_buffer_seconds`, rounded up to a 30-second boundary.
Warmup and evaluation use their respective configured limits. Real collection
reports completion, unfinished counts, `vehicle_timeout_penalty_sum` and the
unpenalized shared `global_reward_sum`. `selection_score` adds the diagnostic
penalty sum for checkpoint ranking only; it is never a shared training reward.
physical travel-time statistics remain unchanged. Failed action execution
invalidates the episode. Censored online episodes supervise interval dynamics
and exposure but not completed-episode travel time. The individual timeout scope,
policy execution and advantage normalization are included in the semantic
fingerprint; old shared-timeout checkpoints are rejected.

The transition starts from `z + P_low prior(z, action)` plus
`sigma * U_low * GaussianNoise`. The graph spectrum uses a symmetric normalized
Laplacian; directed upstream/downstream messages still feed the prior. FM regresses
velocity toward detached real endpoint encodings. Multi-step supervision recursively
feeds predicted endpoints into subsequent recorded conditions, retaining gradients
through earlier solves. It does not force each stochastic endpoint onto the same
observed future with endpoint MSE. Reconstruction and BPR losses retain their
existing decoder semantics. Euler and Picard share the same transition interface;
Picard batches independent integration nodes, and each solve caches only its
state/time-independent condition branches. Caches are local to a call and rebuilt
after parameter or condition changes.

The attention sequence is `[FM time, edge states, register tokens]`. FM time is
the numerical integration coordinate, distinct from physical episode progress.
World-model APIs accept edge-aligned actions and no `context` argument. Removing
the global action/context branch changes the semantic fingerprint, so checkpoints
from the earlier conditioning architecture require a fresh run. Physical clocks,
deadlines, actual population normalization, and actor/critic contexts retain their
existing semantics.

An optional `[method]` table configures graph rank, noise, solver, supervision depth
and risk; see [the method configuration](configs/mainline.example.toml). The default
supervision curriculum increases from one to at most three transitions every 100
optimizer updates, bounded by the sampled sequence length. Online replay supports
that horizon; warmup length remains the resource setting. Latent variance uses raw
coordinates (`S_Z=I`) with parameters frozen during imagination. It describes
conditional predictive dispersion, which can include random demand, and is not a
measurement of pure epistemic uncertainty or an error bound.

Evaluation demand uses `[method].evaluation_demand_seed` and the evaluation index,
independently of the number of training episodes. SUMO uses this independent seed
too. Each realization is saved with a SHA-256 identity in `evaluation_demands/`;
use these identities to verify paired comparisons. Summaries include both
`evaluation_t_all_mean` (episode mean, including empty episodes) and
`evaluation_t_all_vehicle_weighted` (total vehicle minutes / total realized vehicles).
Incomplete evaluations remain censored exposure statistics; consult completion
and nonarrival counts before interpreting them as completed travel times.
Historical fixed-demand baseline outputs have a different demand protocol and
must be labeled separately. The small real-SUMO paired-demand verification is
reproducible with `python tools/verify_fm_prior.py --output-dir outputs/fm_prior_check`.

Measured four-city stepping costs and real short-edge checks are recorded in
[the verification report](docs/next_hop_verification.md).

The aggregate loses decision order, timing within the window, and source edge.
Frozen traffic, approximate junction times, and placeholder routes affecting
SUMO lane anticipation are accepted modeling approximations. Per-edge events
also have a real computational cost: Gretz's 0.1-metre edges can require
1-millisecond stepping, or 30,000 TraCI microsteps per window. Record chosen
step size, microstep count, decision count, and wall time for each city and
vehicle count. Toy throughput does not establish full-city feasibility.

For development verification, use an environment with pytest already installed:

```bash
OMP_NUM_THREADS=1 python -m pytest -q
python -m compileall -q urban_dreamer
```

Tests exercise next-hop reachability and masks, boundary timing, action and
departure normalization, replay alignment, imagined progress and rewards,
checkpoint integrity, and live SUMO behavior when SUMO is available.

A separate real-SUMO actor-critic diagnostic uses the same fixed global-reward
setting without a world model. Its global critic consumes full-network dynamic
traffic plus episode progress and active-vehicle fraction. Training always
samples categorical actions, and evaluation always uses argmax. Random and
static-shortest reference baselines are reported separately.

```bash
OMP_NUM_THREADS=1 python tools/train_toy_actor_critic.py \
  --output-dir outputs/toy_ac_global_new --episodes 150 --eval-every 10 --eval-seeds 3
python tools/plot_toy_actor_critic.py --input-dir outputs/toy_ac_global_new
```

The diagnostic uses all 320 Toy vehicles by default. For fixed-step collection,
add `--session event --backend libsumo --step-seconds 1`; libsumo reads network
traffic only at macro boundaries. Its checkpoint format is
`toy_global_individual_timeout_poisson_ac_v1`, containing actor/global-critic models, optimizers
and random states. `--resume` restores a matching new-format `latest.pt`.
Reward weights, vehicle-critic controls and deterministic-training overrides
are no longer accepted. Old artifacts remain available for historical analysis
but cannot resume under the new training contract. New logs distinguish
`global_reward_sum`, `vehicle_timeout_penalty_sum` and `selection_score`;
plotting can also read archived vehicle-reward records.

The GNAC, Dreamer and MBPO baseline runner uses the same shared traffic reward,
individual timeout cost and sampling/evaluation convention, with checkpoint
format `toy_global_individual_timeout_poisson_baseline_v1`.
DTA allocates each generated vehicle according to learned OD path fractions
using a reproducible draw keyed by the episode seed, OD and vehicle ID. The draw
does not depend on future OD counts or departure times. Under Poisson demand,
zero assignment changes no longer stop DTA; inspect fixed-seed evaluation,
completion, marginal-cost gap and path-fraction changes over the chosen budget.

```bash
python tools/train_toy_baseline.py --method gnac \
  --output-dir outputs/toy_gnac_global_new --episodes 150 \
  --session event --backend libsumo --step-seconds 1
```

The plotting tool requires matplotlib in the analysis environment. Judge
convergence using evaluated travel times, completion and total training cost.
Toy is acyclic, so it cannot demonstrate learned avoidance of loops.

Launch the three-city online AC / approximate DTA-SO comparison with matching
Poisson seeds and optimized event/libsumo collection:

```bash
python tools/run_city_poisson_comparison.py \
  --output-root outputs/city_poisson_comparison_new --episodes 500 --eval-every 25
python tools/monitor_city_poisson_comparison.py \
  --root outputs/city_poisson_comparison_new --plot
```

The launcher creates six independent tmux sessions and records their commands,
PID and exit files in `manifest.json`. Monitoring exports `status.json` and
`learning_curves.png`, checks paired demand hashes, and requires host process
visibility to verify running PIDs. Missing sandbox `/proc` entries are reported
as unknown rather than stopped. `--skip-reference-baselines` omits extra random
and shortest evaluations; AC evaluation itself remains argmax.
