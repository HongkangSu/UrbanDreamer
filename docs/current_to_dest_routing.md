# Periodic current-to-destination routing

The default `[simulation].routing_mode = "current_to_dest"` restores the route
selection setting from commit `80ad9d0e3cb9293ddb5773a5979bb0b5b40caa45`, the
parent of next-hop/HDV-removal commit `c31c650854fffb25d1a140638a9aa9eeee779ed3`.
Fresh configuration files and `SimulationConfig` instances default to
`current_to_dest`; shipped profiles state this setting explicitly. Select
`routing_mode = "next_hop"` to retain next-hop control.

Historical frozen exports named `run_config.toml` or `source_config.toml` infer
`next_hop` when the setting is omitted, preserving checkpoint fingerprints and
archive snapshots for existing resume and pretraining-transfer tools. An explicit
setting always wins, including in those files. These filenames are reserved for
frozen run snapshots: when copying an old snapshot to another filename, add
`routing_mode = "next_hop"` to preserve its semantics. New run exports always
include the setting for either mode and therefore round-trip under any filename.
Resolved snapshots and fingerprints still omit `next_hop` to preserve historical
identities. Starting or resuming an old pilot keeps its source mode; converting
to full-route control requires the explicit transfer described below.

## Control behavior

At each global 30-second boundary, the policy selects one of four complete
remaining routes for each departed, unarrived vehicle on an external edge other
than its destination. Vehicles that depart between boundaries initially follow
their existing SUMO route and become eligible at the next boundary. Entering a
fork does not trigger another policy sample in this mode. Vehicles on the
destination edge need no further routing action.

Candidates start at the vehicle's **current edge**, include that edge, and end at
its fixed destination edge. The historical penalty-Dijkstra algorithm uses
positive static free-flow edge travel times, multiplies every edge of each
selected path by three, retries a repeated path once, and pads missing slots
with the first route. Unreachable routes fail explicitly. The bounded LRU stores
200000 current/destination pairs. `forbid_revisit` is not supported in this mode.

Set `URB_ROUTE_CACHE_DIR` to a common directory to share a persistent 200000-pair
candidate cache between the real collector, SUMO route validation and spawned
imagination workers. Network identities include ordered adjacency, edge IDs,
passenger permissions, free-flow costs and the candidate algorithm version.
SQLite WAL permits concurrent readers; per-pair locks prevent duplicate cold
searches, while different pairs can be computed concurrently. The shared store
evicts in insertion order so cache hits do not contend on a write lock. It
survives session/worker/trainer restarts and does not accumulate a separate route
LRU in each process. Cache access errors visibly fall back to computing candidates.
Dynamic congestion features, policy parameters and RNG state are not cached.
This directory is an execution setting, separate from checkpoint/configuration
fingerprints. Initial SUMO routes and destination distance tables still use the
existing next-hop topology cache; this sharing concerns the four full-route
policy candidates.

The current optimized event session, one-second observation clock, libsumo/TraCI
backend, meso junction/queue settings, demand law, incident handling, and observed
cumulative vehicle-time reward remain in use. HDV modules are not restored.
SUMO receives the complete chosen suffix through `setRoute`; observed route
indices maintain `committed_route` in vehicle snapshots. Imagination follows the
same suffix across edges and macro windows, including after replay restoration.

## World-model compatibility

Policy actions and world-model conditions have separate records. The actor still
uses a shared ten-feature candidate scorer, with four route slots. Full-route
traffic, cost and distance summaries distinguish paths that share a first hop.
`edge_indices` and `next_edge` continue to identify physical next edges. Model
parameter shapes and the flow `(E,8)`, latent `(E,64)`, action `(E,)`, and sparse
`ActionEvent` interfaces are unchanged.

World-model conditions continue to report observed fork commitments at the actual
one-second observation timestamp: current edge, next edge, destination, and phase.
These include route-followed forks without another policy callback. Unobserved
forks skipped during a SUMO step do not produce synthesized events, and future
turns are not packed at route-selection time. A turn observed exactly at a macro
window's end stays pending until the following half-open window. Archives retain
the branch-only conditioning stream in `decisions`/`action_counts`; route policy
actions are independently audited in `policy_decisions`. Policy/Q learning uses
actual route-choice samples; world-model updates use the conditioning stream.

## Reusing completed pretraining

Use the existing online runner with a separate output directory:

```bash
python tools/run_online_pilot.py \
  --source-dir /path/to/completed_pretraining \
  --output-dir /path/to/new_route_experiment \
  --routing-mode current_to_dest \
  --device cuda:1
```

The source must be a completed-pretraining checkpoint with no prior online
activity. The routing transfer allows only the control setting and output/device
changes; inputs, seed, physics/backend, method, codec and resources must match.
The existing legacy reward conversion still applies where required. Alternatively
provide a `--run-config` copied from the source with only `routing_mode` changed.
This is an isolated policy experiment with a different configuration fingerprint;
strict resume of its output uses that route fingerprint. Re-run with the same
source, output and routing flag to resume through the existing per-episode runner.

The runner restores the existing trainer state and verifies every world-model
tensor, including the codec and readouts, against the original checkpoint. The
protocol records source checkpoint and world-model hashes and both configuration
fingerprints. No dynamics pretraining or decoder recalibration is required by a
routing-only transfer. This verifies interface and weight reuse; forecast quality
under the new routing policy still requires empirical evaluation.

## Extending baseline training

Both baseline trainers support a stop budget separate from the entropy schedule.
To continue an existing 1000-episode experiment to 2000 episodes, keep its config
and output directory and add:

```bash
--episodes 2000 --entropy-schedule-episodes 1000 --resume
```

This restores model weights, optimizers, replay where applicable, and RNG state.
The entropy coefficient continues on its original clock and stays at its final
floor after the original decay finishes. Annealed checkpoints reject a changed
entropy horizon. Fresh experiments without an override use their episode budget
as the horizon. Existing frozen configs and their fingerprints remain valid; the
CLI stop budget and batch manifest record the extension. The ongoing three-city
batch keeps its historical directory name containing `1000` while its manifest
records a 2000-episode target for each of the 27 training jobs.
