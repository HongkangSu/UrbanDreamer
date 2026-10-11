# UrbanDreamer

## Default city inputs (2026-10-09)

Coulommiers uses **v5 short-cycle repair, 1811 expected vehicles / 489 OD pairs**.
Montereau uses **occupancy1500, 1500 expected vehicles / 283 OD pairs**, with the
geometry validated for that demand. The online and `_offline` profiles in
`configs/` both reference these canonical bundles under `networks/`. Release
window: 1800 seconds; training cutoff: 3810 seconds; no incidents; prior zero.
Input versions and hashes are pinned in [default_city_inputs.json](configs/default_city_inputs.json).
New baseline queues default to eight learned methods; DTA-SO and Random are
excluded. Actor entropy now follows a shared fixed coefficient schedule:
0.002 for the first 10% of training, cosine decay to 0.0001 at 80%, then a
constant floor. The schedule uses restored progress and the original total
budget, without population scaling. IQL has no entropy loss and retains its
epsilon-greedy exploration. See [the continuation protocol](docs/entropy_annealing_20261010.md). The historical proportional-demand importer refuses to overwrite
installed bundles. Existing frozen experiment snapshots remain unchanged.

UrbanDreamer trains a routing policy for a fully controlled CAV fleet in SUMO.
The mainline observes vehicles once per second and chooses a next road edge
when an observed external entry has multiple destination-reachable candidates.
A learned world model predicts the network state
at fixed 30-second intervals. Vehicle decisions and world-model updates run on
separate clocks.

Bundled mainline and baseline profiles now use **SUMO mesoscopic traffic** through
libsumo: one-second event observations, 30-second world-model windows, active
traffic-light/priority control, combined lanes with a single queue per segment,
98-metre maximum segment length, and interpolated external-edge positions. The
profile is explicit in `[simulation]`; baselines and collectors resolve their
settings from the same configuration. Microscopic configurations remain readable
when `traffic_model` is omitted or set to `"micro"`.
See [the migration contract and validation](docs/mesoscopic_migration.md).

The mainline first pretrains its world model on existing compatible real SUMO
training archives, then collects online episodes to fine-tune the world model
and routing policy, and finally evaluates the policy. It requires a nonempty
offline source and starts no SUMO session during pretraining.

```bash
git lfs install
git lfs pull
conda env create -f environment.yaml
conda activate urban-dreamer
# Set [data].archive_roots in a copied Toy config to an existing compatible
# Toy training dataset before invoking the mainline.
python -m urban_dreamer --config configs/toy_cross_bottleneck.toml --device cpu \
  --output-dir outputs/toy_pretrain_eventq_new
```

The historical microscopic reference environment is Python 3.12 with SUMO 1.25,
as recorded in `requirements.txt`. The new mesoscopic profile has been verified
in the existing URB environment with **libsumo 1.27.1**; it requires the
`--meso-interpolate-pos` capability. The older external SUMO executable in that
environment does not provide it. No dependencies are installed by the migration.
Use the available libsumo runtime for these profiles; an unmodified historical
reference installation is insufficient for the new mesoscopic settings.

Bundled profiles cover Toy, Ingolstadt, Gretz, and Fontenay under `configs/`.
The Toy profile still specifies 150 online episodes; reduce the episode counts
in a copied configuration for a smoke run. Its offline roots are explicitly
empty because no compatible reference-320 / 420-second Toy archive dataset was
found; configure such a source before running it. The existing throughput
archives use reference-50 demand, and earlier short validation archives use
reference-8 demand over 30 seconds, so neither matches the production Toy profile.
City profiles point to `outputs/city_baseline_dataset_meso`, a separate location
for newly collected compatible mesoscopic training archives. The existing
microscopic collections remain available under their original frozen profiles,
but are rejected as pretraining sources for a mesoscopic run. Collect compatible
data before starting the new mainline; changing the profile does not launch a
collection or a full training job.

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
TOML schema has four required sections: `[inputs]` contains only `network_name` and
`network_dir`; `[run]` sets output directory, device, seed, and phase lengths;
`[simulation]` controls horizons, incidents, and the explicit
`vehicle_max_speed_mps` cap; `[resources]` controls batch, sequence, and gradient
accumulation sizes. Optional `[method]` controls FM, risk and per-window vehicle-count labels;
`[codec]` sets separate reconstruction training and held-out acceptance limits; `[data]` selects
frozen offline archives with at most two cached episodes. The parser allows an
omitted `[data]` table for standalone collector and baseline compatibility, but
mainline execution requires compatible offline samples. Relative paths resolve from the TOML file. The macro budget
is `world_model_intervals_per_episode`, with 30 seconds per interval. A replay
sequence length of `T` means `T` transitions and `T+1` physical states.

The CLI accepts `--config` and optional `--device` and `--output-dir` overrides.
It runs **CODEC_PRETRAIN → PRETRAIN → ONLINE → EVAL → COMPLETED**.
CODEC_PRETRAIN fits physical reconstruction on training trajectories and requires
independent held-out raw and birth-adjusted endpoint errors to pass all five configured
limits. PRETRAIN freezes the accepted codec and traverses only existing
offline archives; ONLINE collects real episodes and performs world-model
fine-tuning and imagined policy updates; EVAL is deterministic. Missing, empty
or incompatible sources fail before online collection. Run the same command
to resume a matching `checkpoints/latest.pt`. Use `--pretrain-only` to stop before any online collection. Checkpoint schema 10 stores model,
optimizer, bounded online replay (schema 7), train/validation manifest identity/progress,
codec acceptance evidence, random state,
metrics, and environment episode cursor, plus
the pipeline phase and next unit. Inputs and semantic configuration must match;
changing device/output path does not alter the semantic fingerprint. The frozen
manifest also binds archive roots and file contents. Old checkpoint schemas and
superseded configuration keys are rejected. Deprecated `run.warmup_episodes`
is still accepted for frozen collector configurations, defaults to zero, and
does not schedule mainline collection; legacy simulation `warmup_*` fields also
remain optional and collector-only. Their defaults are 1800 seconds for the
horizon, 3000 seconds for the completion drain, and zero for completion patience.
Shipped mainline profiles omit all warmup settings. These unused fields do not
affect the mainline fingerprint.
Existing output directories are preserved; use a fresh output directory for
this architecture. Completed runs return their saved summary.

Offline world-model pretraining keeps the budget key `[run].world_model_initialization_epochs` and
`[resources].world_model_initialization_batch_size`. One epoch shuffles and visits
each valid archived sequence start once, retains the final smaller batch, and does
not cycle or duplicate samples to fill it. Gradient accumulation weights batches
by their actual sample counts. A checkpoint is saved after each complete epoch;
the standard CLI restores the preceding boundary after an interrupted epoch.
The complete-city supervisor additionally saves a consistent model/optimizer/
sampler/RNG checkpoint after its first update and at bounded intervals within
an epoch, allowing that runner to continue the remaining starts.
The old `world_model_initialization_steps` key is rejected. Pretraining does not
start a session or collect new source trajectories. Metrics report completed
epochs, batches, sampled sequences and optimizer updates separately.

Every transition uses its complete observed previous state as input. Its endpoint
vehicle-count target excludes only vehicles not yet departed in that previous
observation. A vehicle born in one window appears in the next full observed
input and becomes eligible for the next target. Actions, sparse events,
departures, context and exposure remain the original physical records. Speed,
occupancy and halt labels retain the observed endpoint background. Endpoint
counts are reconstructed by vehicle membership at their actual endpoint roads,
not by subtracting origin departure counts. Codec reconstruction always uses
its input frame's own counts.

For the corrected labels, measured performance and current Ingolstadt run, see
[the per-window pretraining record](docs/per_transition_pretrain_20261002.md).
The [previous fixed-cohort run](docs/staged_codec_cohort_pretrain_20261002.md)
was stopped after the user clarified the label definition; its dynamics weights
are not resumed. Its independently trained codec may be imported and revalidated.

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

Shipped mainline configurations and learned-baseline CLI entry points default to
no shortest-path policy prior (weight 0). To enable the optional annealed prior, use
`--sp-prior-weight 1 --sp-prior-temperature 30 --sp-prior-hold-episodes 200
--sp-prior-off-episode 1000`. The actor logits receive
`-weight * (full_free_flow_cost - minimum_legal_cost) / temperature_seconds`,
where the full cost includes the junction, candidate road and downstream route.
Weight is fixed during a real training episode and its updates/imagination;
it is held through episode 200, cosine-annealed to exactly zero at episode 1000,
and remains zero thereafter. Rewards, masks and network parameter counts retain
their existing definitions. To keep it disabled explicitly, use `--sp-prior-weight 0`
for baselines or `[method].sp_prior_weight = 0.0` for the mainline. Shipped
mainline TOML files explicitly record all four settings. Legacy configurations
and checkpoints with no prior fields retain their historical zero-prior semantics;
changing a resumed run requires an explicit protocol migration. Existing frozen
runs are not changed by these new-run defaults.

Enabled baseline runs evaluate both scheduled-prior and bare (`deterministic_no_prior`)
actors on the same demand seeds, and select the best checkpoint by bare-actor
performance. Prior settings and realized weights are stored in protocols, metrics
and training archives. MBPO drops cached synthetic rows whenever the weight
changes, avoiding mixing different prior schedules; its existing same-weight
historical-policy reuse remains approximate. The plotting tool uses bare-actor
evaluation as the default for a prior-enabled manifest and exports the scheduled
evaluation separately. Mainline primary evaluation uses zero-prior argmax.
Uniform random, shortest-path references and DTA remain unassisted controls.

Mainline actor entropy now uses `[method].actor_entropy_mode = "population"`
in shipped presets: each event receives coefficient
`0.002 / (full_episode_vehicle_count * max(global_advantage_std, 1))`.
Population and scaling statistics are detached. This keeps the entropy term in
the same units as the population-normalized shared return; individual timeout
penalties retain their separate addition and clipping. Missing fields in legacy
snapshots resolve to `legacy` and retain the old fingerprint. Changing a live
run requires `tools/migrate_mainline_entropy.py`; the online runner rejects an
environment override that conflicts with its fingerprinted configuration.
Baseline entropy settings are independent of this mainline correction.

Fixed evaluation references can be collected with `tools/evaluate_city_references.py`
using `--root <dataset> --city <city> --method shortest|dta_so --output-dir <new-directory>`.
The shortest-path policy is evaluated directly; DTA-SO performs exactly one
assignment update on training seed `seed + 1`, then evaluates its fixed costs on
the dataset's independent evaluation seeds. The existing DTA uses BPR marginal
costs and MSA; one update is not a converged system optimum. Combined results
can be passed to `tools/plot_city_baseline_dataset.py --reference-results <json>`
or selected by the dataset manifest's `reference_results` path. Evaluation plots
show their horizontal means and demand-seed standard deviations, with Random
removed from evaluation legends. Training plots still show the random collector.

Baseline collection supports `--forbid-revisit` with an event session and zero
shortest-path policy-prior weight. Nodes here mean directed external road edges.
Every vehicle carries its complete `visited_edges` history; candidate masks
exclude visited/current edges and choices that cannot reach the destination in
the remaining graph. SUMO execution routes use that same graph, and route-index
tracking includes singleton roads and roads crossed between observation ticks.
Imagination, archived snapshots and restored seeds preserve this history.
An inconsistent history or an empty feasible action set raises an error rather
than relaxing the constraint. Static distance/cost observation features and the
route suffix needed by SUMO remain available; no shortest-path bias is added to
actor logits. The mode is recorded in archives, protocols and checkpoint args,
and cannot be changed when resuming. `tools/audit_no_revisit_archives.py` checks
complete physical histories and decision records. New constrained experiments
use separate output directories and fresh model initialization.

Baseline training metrics include `collection_seconds` (demand setup, real
simulation, recording and session close), `archive_seconds` (the final archive
write, included in collection), `learning_seconds`, and `episode_total_seconds`
(collection plus learning, excluding evaluation and checkpoints). Dreamer/MBPO
also report `world_model_seconds`, `imagination_seconds`, `actor_update_seconds`
and `critic_update_seconds`; these are components of learning, not additional
costs. Timing uses host wall clocks without forcing per-event CUDA barriers.
The completed timing fields are in `metrics.jsonl`; an archive saved during
collection cannot contain its own final write duration.

Actor observations reuse bounded, immutable topology templates and prepare the
frozen traffic features once per window. Mutable traffic, remaining distance,
episode progress and active population remain fresh. Dreamer/MBPO imagined
decision records retain local actor features rather than copying the entire
network flow at every decision. Update counts, rollout horizons and RNG draws
are unchanged. `tools/benchmark_baseline_throughput.py` compares a bounded real
Toy run with a source snapshot using identical seeds and training budgets.

At a vehicle event, candidates are passenger-permitted successor edges that
can still reach the destination. Stable edge order and padding masks support
variable outdegree. A shared scorer uses candidate traffic, static cost, and
shortest remaining distance. Warmup samples a softmin over static remaining
cost; online collection samples the actor, and evaluation chooses its highest
score. A sole candidate follows the existing route automatically. Entering the destination
edge ends decision-making, while travel time continues until actual arrival.

SUMO carries a complete route rebuilt from the current edge, selected next edge,
and a deterministic destination suffix. Forks crossed between observations
follow that route. Each successful branch decision retains its source,
destination, time, and identity; edge counts remain consistency telemetry. Vehicles are inserted with
zero departure position and speed. Internal junction edges and waiting on the
same road edge do not produce new decisions. Failed route execution, missed
committed edges, and premature disappearance invalidate the episode.

Mainline observation steps are fixed at one second; world-model windows are
30 seconds. The legacy all-next-hop session retains its adaptive microsteps for
existing callers. SUMO arguments that replace the network, routes, configuration,
simulation clock, speed settings, or managed routing are reserved. Vehicle
subscriptions track observed events; full-network observations are extracted
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
telemetry. The mainline world model conditions on raw events grouped by legal
turn and three ten-second buckets, with shared source/next/destination embeddings
and continuous phase. Nonlinear turn readouts produce edge-aligned features
with graph/static features and FM integration time. Actions enter each road token
through the action GNN and edge-aligned injection; there is no global action token
or episode-context input in the dynamics or graph prior. Replay retains the
two-feature `context_t` for policy/value contexts and diagnostics. The world model
has no departure encoder or timetable input.
SUMO alone uses the demand departure schedule to inject vehicles. Future
scheduled departures keep an otherwise empty real episode alive.

The mainline regenerates demand in every online and evaluation episode.
Offline pretraining retains the realized demand of each compatible source archive.
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

`N` is the realized episode population. Replay schema 7 stores it with each
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
endpoint labels are available. This applies to offline pretraining and online fine-tuning.
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
The default shared reward for UrbanDreamer and all routing baselines is
`-delta(C / max(B, 1)) / 60`, where `C` is cumulative active vehicle-seconds
and `B` is cumulative actual starts strictly before the physical cutoff.
Both real and imagined rewards carry their cumulative accounting history.
The four critic context fields are time progress, active/B, log1p(C/60),
and log1p(B); Actor observations never use the final future vehicle count.
Empty demand contributes zero. Unfinished trips contribute observed time only;
default individual timeout penalties and the default model-risk weight are zero.
The new reward contract is `causal_cumulative_mean_global_v1` and replay schema 8.
Legacy checkpoints need an explicit migration before paper-reward continuation.

Warmup retains static-cost softmin sampling. Macro critic returns use `gamma=1`
and `lambda=0.95`. Each decision receives global Monte Carlo reward-to-go from
its event timestamp, excluding earlier travel cost and retaining all later
endpoint risks. Simultaneous decisions share a return target. Each event uses a
detached legal-policy expectation of its candidate event Q values, conditioned
on history and the other simultaneous actions. Baselines are saved at sampling
time and reused after Q fitting. Q uses current-policy imagined returns only;
macro V remains responsible for macro value learning and cutoff bootstrap. Event advantages are
scaled by the standard deviation of **global advantages only**, floored at 1,
without mean-centering, and clips the result to ±2.5 without an individual timeout
return. Entropy regularization follows the configured schedule. There is no
vehicle critic, local elapsed-time reward or
configurable reward mixture.

An ordinary eight-window imagination cutoff bootstraps the global critic and
adds no timeout penalty. Windows with no decisions still train the world model
and global critic. The online deadline is the world-model interval budget plus
`training_completion_buffer_seconds`, rounded up to a 30-second boundary.
Warmup and evaluation use their respective configured limits. Real collection
reports completion, unfinished counts, `vehicle_timeout_penalty_sum` and the
shared `global_reward_sum`. Under the default paper objective, `selection_score`
equals `global_reward_sum` and `vehicle_timeout_penalty_sum` is zero.
physical travel-time statistics remain unchanged. Failed action execution
invalidates the episode. Censored online episodes supervise interval dynamics
and exposure but not completed-episode travel time. Reward semantics, policy
execution and advantage normalization are included in the semantic fingerprint;
legacy sources can be inspected but require recorded migration for learning.

The transition starts from `z + P_low prior(z, action)` plus
`sigma * U_low * GaussianNoise`. The graph spectrum uses a symmetric normalized
Laplacian; directed upstream/downstream messages still feed the prior. FM regresses
velocity toward detached birth-adjusted endpoint encodings. Mainline supervision
uses independent windows with full observed starts; a target cleaned of current
births is never fed as the next window's input. The default three supervised
windows are three 30-second predictions, not a recursive 90-second training
objective. The legacy recursive helper remains available to other experiments.
Sampled solves run without gradients and train only the detached exposure
readout; the FM velocity and graph prior retain gradients. Codec reconstruction
has its own optimizer and is frozen during dynamics training. Euler and Picard share the same transition interface;
Picard batches independent integration nodes, and each solve caches only its
state/time-independent condition branches. Caches are local to a call and rebuilt
after parameter or condition changes.

The attention sequence is `[FM time, edge states, register tokens]`. FM time is
the numerical integration coordinate, distinct from physical episode progress.
World-model APIs retain edge-aligned count telemetry, require sparse raw events
in mainline event mode, and accept no `context` argument. Removing
the global action/context branch changes the semantic fingerprint, so checkpoints
from the earlier conditioning architecture require a fresh run. Physical clocks,
deadlines, actual population normalization, and actor/critic contexts retain their
existing semantics.

Pretraining execution chunks and activation checkpointing are explicit resource
settings. Larger chunks reduce repeated small CUDA launches; memory permits
microbatch 4 with accumulation 2 in the measured Ingolstadt run. Gradient
accumulation preserves the effective batch but does not make microbatches run
in parallel. The transformer keeps activation checkpointing in that run because
turning it off exceeded the tested shared-GPU budget. Optional CUDA BF16
pretraining keeps codec, parameters, gradients, optimizer state and evaluation
in FP32. The singleton temporal attention path avoids a padded FlashAttention
workspace. Auxiliary exposure reads are learned from detached observed endpoint
latents, removing unnecessary integration passes from FM training. The shipped
pretraining sequence length is three, matching the independent supervision
horizon and covering episode tail windows.

An optional `[method]` table configures graph rank, noise, solver, supervision depth
and risk; see [the method configuration](configs/mainline.example.toml). The default
supervision curriculum increases from one to at most three transitions every 100
optimizer updates, bounded by the sampled sequence length. Online replay supports
that horizon; offline sequence length remains the resource setting. Latent variance uses raw
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

Raw events preserve source, destination, identity and continuous phase. The
joint encoder pools each turn/time bucket, so it does not preserve every ordering
detail within a bucket. Frozen traffic and approximate junction times remain
modeling approximations. One-second observations can skip short forks; this is
shared by real event execution and event imagination. Legacy all-next-hop runs
on Gretz can still require millisecond microsteps. Record chosen
step size, microstep count, decision count, and wall time for each city and
vehicle count. Toy throughput does not establish full-city feasibility.

See [the event mainline implementation and verification guide](docs/event_mainline.md)
for the event Q formula, offline manifests, version compatibility and bounded
Toy/city verification commands, and [the offline-pretraining verification record](docs/offline_pretrain_validation.md)
for the current lifecycle. The earlier warmup validation report is retained as historical evidence.

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

The GNAC, Dreamer, MBPO, PPO, IQL and MAMBA baseline runners use the same causal
observed-mean global reward and sampling/evaluation convention, with checkpoint
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
