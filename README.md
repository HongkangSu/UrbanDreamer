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
to resume a matching `checkpoints/latest.pt`. Checkpoint schema 3 stores model,
optimizer, replay, random state, metrics, and environment episode cursor, plus
the pipeline phase and next unit. Inputs and semantic configuration must match;
relocating data or changing device/output path does not alter the semantic
fingerprint. Old checkpoint schemas and old configuration keys are rejected.
Existing output directories are preserved; use a fresh output directory for
this architecture. Completed runs return their saved summary.

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
telemetry, but the world model takes only latent traffic, `action_t`, and the
two-feature `context_t`; it has no departure encoder or timetable input.
SUMO alone uses the demand departure schedule to inject vehicles. Future
scheduled departures keep an otherwise empty real episode alive.

Training samples active windows over the entire collected episode, including
the period after departures have ended. A window with integrated vehicle
activity is valid for world-model training even if its starting state was empty.
Imagined rollouts require an active vehicle snapshot at their starting boundary.
Purely empty windows are excluded from training samples; sequences remain
contiguous and never join across an empty gap. Replay still stores the full
physical timeline. Episode progress uses the full simulation budget, including
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

Local rewards are negative elapsed seconds between vehicle decisions divided
by 30, including the destination edge after the final decision. Returns use
`gamma=1` and `lambda=0.95`. An ordinary eight-window rollout cutoff settles
elapsed time and bootstraps the in-transit vehicle critic. At the real episode
deadline, each unfinished vehicle instead receives an additive **-5000** on its
final action reward, with no bootstrap. This penalty is not divided by 30 and is
not charged at intermediate macro boundaries. A vehicle arriving exactly at the
deadline receives no penalty. Critic-only intervals use the same deadline target
without inserting additional actor events or counting the penalty twice in an
event trajectory.

The online episode deadline is the configured world-model interval budget plus
`training_completion_buffer_seconds`, rounded up to a 30-second boundary just
like real collection. Warmup and evaluation use their respective configured
limits. Valid real runs reaching those limits report unfinished vehicle counts,
penalty sums, and penalized local reward sums; physical travel-time statistics
remain unmodified. Failed action execution is still an invalid episode. The world model learns the interval's active-vehicle fraction from
`active_vehicle_seconds / (N * 30)`; macro reward is its predicted fraction
multiplied by `-30/60`. Macro advantages are associated with each window's
vehicle decisions. Windows without decisions still train dynamics and the
macro critic. Separate critic-only intervals also train committed-edge and junction
progress states used for horizon bootstrap; they do not create policy decisions or
change action aggregates. Censored online episodes supervise interval dynamics
and exposure, but are excluded from completed-episode travel-time supervision.

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

A separate real-SUMO diagnostic isolates the actor and vehicle critic from the
world model. It reuses the same observations and event rewards, trains directly
on real trajectories with on-policy lambda returns, and compares random,
static-cost, initial-policy and trained-policy performance:

```bash
OMP_NUM_THREADS=1 python tools/train_toy_actor_critic.py \
  --output-dir outputs/toy_ac_real_new --episodes 150 --eval-every 10 --eval-seeds 3
python tools/plot_toy_actor_critic.py --input-dir outputs/toy_ac_real_new
```

This diagnostic uses all 320 Toy vehicles by default, writes its own checkpoints
and JSONL metrics, and does not instantiate a world model or latent macro critic.
`--resume` continues its own `latest.pt` in the same output directory. Its
checkpoints are separate from the mainline schema. The plotting tool requires
matplotlib in the analysis environment; no new runtime dependency was added.
Judge convergence using evaluated trip times and completion, not loss alone.
Toy is acyclic, so this experiment cannot demonstrate learned avoidance of loops.

For the paired global-reward ablation, use a new experiment directory and
`--global-reward-weight 1`. The additional critic consumes the real full-network
dynamic traffic fields plus progress/active fraction; its reward is measured
`-active_vehicle_seconds / (N * 60)`. Per-window lambda advantages are added to
local event advantages before the existing normalization. No world model or
learned reward predictor is involved. Weight zero retains the local-only mode.

```bash
OMP_NUM_THREADS=1 python tools/train_toy_actor_critic.py \
  --output-dir outputs/toy_ac_global_new --global-reward-weight 1 \
  --episodes 150 --eval-every 10 --eval-seeds 3
python tools/plot_toy_actor_critic.py --input-dir outputs/toy_ac_global_new \
  --compare-dir outputs/toy_ac_real_new
```

The diagnostic checkpoint format is now `toy_real_actor_critic_v2` and also
stores global-critic optimizer state. Input/config fingerprints and global
reward settings are checked on resume. Baseline v1 artifacts remain readable
for analysis, but are not resumed by the updated experiment runner.

To test **global reward only**, use `--local-reward-weight 0
--global-reward-weight 1` in a new output directory. No vehicle critic is created
or updated; local trip-reward metrics are diagnostic only. The global critic
supplies the baseline. Pure-global advantages use scale-only normalization
(std floor 1) and clipping, without subtracting the current batch mean: a shared
constant team advantage must not disappear when centered. This normalization
choice is recorded and checked on resume. At the real deadline, each unfinished
vehicle contributes an unscaled `-5000` to the terminal team reward.
