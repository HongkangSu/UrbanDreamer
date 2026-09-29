# Event-driven SUMO verification

Measured on 2026-09-29 with SUMO 1.25.0 and the existing `URB` Python environment. Original city network files were not modified. These measurements exercise direct TraCI control with a 40 m/s vehicle speed ceiling, frozen macro observations, vehicle subscriptions, and successful next-hop decisions.

Each city probe starts four vehicles at scheduled time zero, choosing distinct OD pairs where available and filling from the demand table otherwise. The policy minimizes static successor cost plus remaining destination cost. Incidents are disabled for this probe. One 30-second window is collected; all four vehicles departed and none had arrived by its end.

| City | External edges | Step (seconds) | Microsteps | Successful decisions | Measured wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Toy cross bottleneck | 24 | 0.5 | 60 | 12 | 0.039 |
| Ingolstadt custom | 853 | 0.001 | 30,000 | 25 | 13.140 |
| Gretz-Armainvilliers | 4,485 | 0.001 | 30,000 | 35 | 14.648 |
| Fontenay-en-Parisis | 7,585 | 0.01 | 3,000 | 26 | 4.223 |

Wall seconds use `SumoSession.metrics['wall_seconds']`: timing begins after SUMO starts and includes demand insertion, initial network observation, and interval execution. It excludes process startup and the final network extraction performed after the metric is recorded. These bounded four-vehicle probes do not estimate full demand training cost. SUMO emitted existing network warnings concerning Ingolstadt traffic-light links and Gretz rail signals.

The safe step uses the shortest passenger-permitted **lane in the original XML**, the explicit vehicle speed ceiling, and an integer millisecond step dividing one second. Registry edge lengths do not override a shorter lane. Actual vehicle snapshots use the subscribed lane's remaining length; internal snapshots preserve the full remaining connection suffix from the current internal lane.

The automated live tests additionally cover:

- Three vehicles on Toy through full completion, including a future scheduled departure, exact travel-time/active-time equality, and terminal empty-state padding.
- A temporary copy of Toy with both `A_in` and `A_hold` reduced to **0.1 meters**, three vehicles, and a **1 millisecond** step. All three receive decisions on both short edges, override the original placeholder at the following branch, and arrive normally. The initial isolated test took 38.88 seconds.
- Invalid next-hop selection and a simulated TraCI lane-connectivity execution failure invalidate the episode and count no successful action.

Unit tests cover a departure and arrival observed together at 30 seconds, deferred boundary decisions, a long edge spanning multiple windows without repeated decisions, missed committed edges, premature arrivals, exact lane progress, internal connection suffixes, destination reachability, single successors, and placeholder route construction.

Reproduce the automated checks from the repository root:

```sh
/home/wangwenrui/opt/miniconda3/envs/URB/bin/python -m pytest \
  tests/test_urban_dreamer_environment.py \
  tests/test_urban_dreamer_topology.py \
  tests/test_urban_dreamer_live_environment.py -q
```

The short-edge test mutates a pytest temporary XML copy only. The live failure test deliberately injects a TraCI execution exception; it verifies strict failure handling, while real multi-lane routing coverage comes from the original city probes. This is not an exhaustive multi-lane maneuver test.


## Training and recovery integration

The automated Toy lifecycle test uses two vehicles and a reduced update budget:
one warmup episode, one world-model initialization microbatch, one online episode,
one world-model online update, one two-window imagined rollout, and one deterministic
evaluation episode. It deliberately interrupts immediately before online collection,
loads the phase-boundary checkpoint into a fresh trainer, completes the remaining
phases, and checks that a repeated completed run does not instantiate a trainer.
This validates the full lifecycle; it is not a convergence or full-demand benchmark.

Additional training regressions check learned active-fraction readout changes reach
macro rewards and event advantages; decision-free windows update the macro critic;
and critic-only in-transit samples train commitment and junction features used by
horizon bootstrap. A destination-edge critic sample ends at actual arrival and does
not bootstrap. Incomplete online episode totals are excluded from final episode
travel-time supervision, while their interval data remain usable.

A separate integration diagnostic exercised B=2, T=3, E=24 on the real Toy trainer.
After a model update and an on-disk checkpoint roundtrip, the subsequent losses and
dynamics output-layer weights reproduced exactly (zero relative/absolute tolerance),
covering optimizer, random-state and replay-sampler restoration together.

Verification environment: Python 3.13.1, PyTorch 2.10.0, NumPy 2.3.5,
TraCI 1.21.0, sumolib 1.25.0, SUMO 1.25.0, pytest 9.0.2. The documented
installation reference remains Python 3.12; that clean installation was not repeated.
No lint or static type-checker packages are installed in this environment, and none
were added. Python AST parsing, compilation, and Git whitespace checks pass.

Final full-suite result after the in-transit critic update: **127 tests passed in
67.94 seconds**. Command:

```sh
OMP_NUM_THREADS=1 /home/wangwenrui/opt/miniconda3/envs/URB/bin/python -m pytest -q
```

`python -m compileall -q urban_dreamer tests tools` and `git diff --check` also
passed after the final code changes. Production code, configuration, dependency
files and README were scanned for the removed predictor, role, four-route,
whole-path advancement and RouteRL identifiers; no implementation remnants were
found. Old identifiers occur only in regression assertions that reject them.

## Decoder window-mean speed labels

The speed target for `decode(z_t)` is now `(speed_t + speed_{t+30}) / 2`,
computed in the existing per-edge normalized units. Encoder inputs, latent
transition targets, replay data, and all non-speed decoder targets retain their
physical boundary meaning. Both initialization and online updates use this
label convention; imagined vehicle advancement continues consuming the decoder
output directly, without a second endpoint prediction or correction pass.

The final sampled latent lacks a following boundary observation, so its speed
loss is masked while other channels still train. Predicted-next decoder outputs
use the following window's label where available, with the same tail mask.
Regressions check one-transition and multi-transition batches, unchanged encoder
inputs/replay states, exact endpoint arithmetic, and zero masked-speed gradient
with nonzero gradients for the remaining channels. The label convention is
included in the semantic fingerprint to reject silent resumption from models
trained with instantaneous-speed labels.

After this label change, the complete suite passed: **130 tests in 67.50 seconds**.
Python compilation and `git diff --check` also passed.

## Full-episode sampling, eight-window rollouts, and nonarrival penalties

The imagination horizon remains **eight 30-second windows**, capped by the real
online episode deadline. Seed selection now spans every stored active boundary
throughout the full simulation, including after the departure period; there is
no departure-duration cutoff on eligible seeds. Both replay buffers train on
active intervals, retain the original full physical timeline, and never stitch
sequences across an empty gap. Exposure-only intervals remain valid world-model
examples but cannot become imagination seeds without active starting snapshots.

Departure timing remains in SUMO demand and observed training telemetry. It is
not a world-model condition: the departure GNN/embedding and `depart` argument
were removed. Imagination advances only the active seed cohort, without future
scheduled vehicle insertions. All cohort vehicles arriving ends the rollout;
otherwise the usual cutoff at eight windows retains value bootstrap.

At the actual episode deadline, an unfinished vehicle's last event receives
`-elapsed_seconds / 30 - 5000` and no bootstrap. The penalty is not scaled by 30,
is not applied at a normal rollout cutoff, and is not applied to arrival exactly
at the deadline. In-transit critic-only samples use the corresponding terminal
value target without adding actor events. Actual collection reports unfinished
counts, penalty sums and penalized local reward sums separately from physical
travel time. Legitimate warmup/evaluation timeouts are retained as incomplete
trajectories, while action failures still invalidate collection.

New regressions cover a seed with eleven windows remaining (only eight are
imagined, no timeout penalty), a late seed with one window remaining (one
unscaled timeout penalty), deadline arrival, future-demand exclusion, activity
sampling after departures, empty gaps, filtered sampler restoration, and all
three real-collection phase timeout metrics. Checkpoint and replay schema 3
reject older model/sampling contracts; semantic fingerprints also encode the
reward and sampling rules.

Final verification for this update: **144 tests passed in 66.00 seconds**,
including the real Toy lifecycle/recovery and multi-vehicle short-edge tests.
Python compilation and `git diff --check` passed. Rollout length remains eight
windows; only its eligible start-time range spans the full simulation timeline.
