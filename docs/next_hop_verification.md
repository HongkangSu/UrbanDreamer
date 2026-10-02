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

## Event-session libsumo boundary reads

The fixed-step `SumoEventSession` now reads edge occupancy, mean speed, vehicle
count and halting count directly at window boundaries when using `libsumo`.
It retains the per-time flow cache and vehicle subscriptions for event detection.
Socket TraCI retains batched edge subscriptions. In-process edge subscriptions
previously refreshed every edge on every simulation step even though the runner
only consumed network traffic at 30-second boundaries.

On 2026-10-01, the saved Fontenay DTA checkpoint at iteration 334 was tested with
all 2,068 vehicles, seed 10044, the configured incident and a one-second step.
The optimized session ran first, followed by a reference session implementing
the former persistent edge subscriptions. Neither episode performed training.

| Measurement | Boundary reads | Persistent subscriptions |
| --- | ---: | ---: |
| Episode collection wall seconds | 66.784 | 144.638 |
| Time inside simulationStep, seconds | 7.505 | 99.563 |
| Completion rate | 100% | 100% |
| Mean vehicle travel seconds | 267.831238 | 267.831238 |
| Decision events | 40,817 | 40,817 |
| Simulation steps | 3,176 | 3,176 |
| Macro windows | 106 | 106 |

This pair measured a 2.166x speedup. Exact equality checks passed for physical
metrics, every vehicle's travel time, the complete ordered decision trace, and
all macro transitions (flow, next flow, actions, departures, context, vehicle
snapshots, exposure and reward). Results and the run log are saved under
`outputs/event_session_boundary_reads_fontenay_20261001/`.

Collection timing includes session startup, demand insertion and runtime cache
construction; it excludes runner/method construction, checkpoint loading,
post-episode algorithm updates, checkpoint saving and extra evaluations. This
is one paired seed, and timing varies with host load and cache warmth. These
measurements apply to the fixed-step event session rather than the strict
per-edge microstep session.

Reproduce with the saved DTA experiment and a new report path:

```sh
python tools/benchmark_event_session.py \
  --experiment outputs/dta_fontenay_en_parisis_seed42_20260930 \
  --report outputs/fontenay_boundary_read_benchmark/report.json --seed 10044
```

The benchmark refuses to overwrite a report, loads the checkpoint with normal
input/config validation, and asserts episode equivalence. Its reference session
is confined to the diagnostic tool; training uses the optimized implementation.

Verification: **46 tests passed in 227.44 seconds** across event sessions, DTA,
and the baseline/model-based runners. New regressions first failed under the
old libsumo behavior, then passed after the change. They cover all four traffic
fields, sub-one-percent occupancy, cache reuse and refresh, terminal empty flow,
and real Toy execution with edge subscriptions forbidden. Python AST parsing,
compilation and `git diff --check` also passed. Ruff and mypy are not installed;
no additional checker dependencies were added.

## Fixed global-reward setting and Poisson mainline demand

On 2026-10-01, the retained training contract was fixed to global reward only:
`-active_vehicle_seconds / (N * 60)` per real window, with an additive `-5000`
per unfinished vehicle in the actual deadline window. Imagined macro rewards
still use the active-fraction readout; the real deadline adds the corresponding
unfinished active-cohort penalty. Ordinary rollout cutoffs bootstrap without
a deadline penalty. Arrivals at the deadline are processed before the count.

Online and imagined actors sample their masked categorical distributions;
standard evaluation and final actor holdouts use argmax. Actor advantages use
the global critic only, scale by standard deviation floored at one, clip to
±2.5, and are not batch-centered. Gamma remains one, lambda 0.95 and entropy
coefficient 0.002. Vehicle critics, their optimizers, per-vehicle event rewards,
local return functions, reward weights and deterministic-training CLI switches
were deleted. New logs report `global_reward_sum`, not a vehicle-reward alias.
Historical plotting remains able to read archived experiment records.

Mainline demand is now a homogeneous independent Poisson process for each
reference OD. Its fixed rate is the OD's count in `agents.csv` divided by
`demand_duration_seconds`. Per-OD counts are Poisson with mean `lambda_od * T`;
conditional times are uniform on `[0,T)`. This samples the continuous-time
process without discretizing arrivals to whole seconds or forcing a nonempty
episode. Generated IDs, OD counts and departure times vary by episode; each
vehicle's OD is fixed after generation. Toy uses T=420 seconds and each city
uses T=1800 seconds. Expected populations match the reference tables.

Demand sampling has a dedicated stream keyed by run seed and the checkpoint's
environment episode counter, independent of actor and replay RNG consumption.
Replay schema 4 stores each episode's realized population, and imagination uses
the sampled seed's population. Actual N continues to normalize actions and
context, making aggregate population information available; future individual
vehicle schedules are not given to the model or inserted during imagination.
Zero-population episodes keep a valid empty terminal window and are excluded
from active training samples. Mainline checkpoint schema 4 and semantic
fingerprints reject the former mixed/local and fixed-timetable contracts.

An actual all-vehicle Toy diagnostic completed two warmup episodes, one model
initialization update, a deliberate phase interruption and resume, two online
episodes and one argmax evaluation. It used the strict default TraCI session
with a 0.5-second step. Model update budgets were reduced to one online update,
one imagination pass and two rollouts per episode; the eight-window rollout
horizon remained unchanged. This validates execution and restoration, not
policy convergence.

| Stage | Environment episode | Realized population | Completion |
| --- | ---: | ---: | ---: |
| Warmup | 0 | 305 | 100% |
| Warmup | 1 | 310 | 100% |
| Online | 2 | 335 | 100% |
| Online | 3 | 334 | 100% |
| Argmax evaluation | 4 | 341 | 100% |

The Toy OD rates were each 160/420 = 0.380952381 vehicles per second, and the
expected total population was 320. Artifacts, sampled episode metrics, demand
parameters, checkpoint, configuration, diagnostic source and logs are under
`outputs/mainline_global_poisson_smoke_20261001/`.

The completed full suite passed **252 tests in 296.67 seconds**. Tests include
Poisson count mean/variance, independent OD counts, arrival-window increments,
population-aware replay restore, zero-demand real sessions, sampling versus
argmax, no-decision critic learning, readout-to-advantage propagation, terminal
penalties, fixed-policy CLI enforcement and random-state restoration. The
suite ran on the host because sandbox port binding gave TraCI `remote-port=None`.
Python compilation, whole-repository AST parsing and `git diff --check` passed;
Ruff and mypy remain unavailable and no dependencies were added.

## Three-city online AC and approximate DTA-SO Poisson comparison

The standalone AC and baseline runners now also use independent fixed-rate OD
Poisson demand. Both generate `sample(seed=base_seed, episode_index=SUMO_seed)`
and log the complete demand realization hash, allowing paired training and
evaluation comparisons. Realized population controls normalization and zero
demand remains a valid terminal episode. Model-based baseline replay and
imagination retain each seed's population as well.

DTA allocates newly generated vehicles from its learned OD path fractions.
Each vehicle's categorical draw is reproducible from the episode seed, its own
OD and its own ID; it does not use future departure times, future OD counts or
other vehicle ordering. Evaluation preparation does not update learned costs,
fractions or learning RNG. The Poisson version disables the former unchanged-ID
early stopping rule and logs reference-demand-weighted marginal-cost gap and
path-fraction changes; small MSA changes alone are not an optimum certificate.

Six 500-episode runs were launched on 2026-10-01 at 16:34 UTC, one AC and one
DTA-SO for each of Ingolstadt, Gretz and Fontenay. They use base seed 42,
event/libsumo collection, one-second steps and one CPU thread per process.
AC trains with categorical sampling and evaluates by argmax. Fixed evaluation
seeds are 10042--10044 every 25 episodes; holdout seeds are 20042--20044. The
extra random/shortest reference suite is omitted for this initial comparison.
All runs retain the configured incidents and 3810-second rounded deadlines.

The experiment manifest, commands, source snapshot/hashes, PID/exit files,
logs, status and curves are under
`outputs/city_online_ac_dta_so_poisson_seed42_20261001/`. The monitor verifies
that common episode/seed records have identical offered-demand hashes. Read
host process information to verify job status; sandbox-hidden PIDs are unknown,
not evidence that training has stopped.

The changed runner/DTA/model-based/plot suite passed **73 tests in 83.02
seconds**, including actual SUMO train/resume/evaluation. DTA's allocation tests
also verify that adding future vehicles or changing their schedules leaves an
existing vehicle's choice unchanged. Compilation and whitespace checks passed.
These are ongoing training experiments; convergence is assessed from paired
evaluation, completion and reward curves, not just loss or arrived-only time.

## Individual nonarrival penalties

The nonarrival contract was corrected on 2026-10-01: the shared traffic reward
and global value targets exclude timeout penalties. Only a vehicle unfinished
at the actual deadline receives a -5000 terminal cost. Its last own decision
records this cost once (`timeout_penalty`); its own earlier decisions receive
the corresponding undiscounted Monte Carlo `timeout_return`. Arrived vehicles
receive zero timeout return, including arrivals exactly at the deadline.
Normal imagination truncation is not failure. No policy rows are fabricated
for unfinished vehicles that never made a decision.

Actor updates normalize global advantages using global-only batch statistics,
then add each action's own timeout return and apply the existing ±2.5 clipping.
This keeps an arrived action's coefficient unchanged when another vehicle is
marked failed. The shared actor parameters still learn from all trajectories.
No vehicle critic or local elapsed-time reward was reintroduced. FM prior,
disagreement risk and causal event-global returns retain their existing logic.

`global_reward_sum` reports shared traffic cost only.
`vehicle_timeout_penalty_sum` is the sum of individual penalties for diagnostics.
`selection_score` combines these two for checkpoint ranking, never as a shared
critic target or a reward broadcast to other vehicles. DTA has no actor update;
its assignment algorithm is unchanged and this score is only evaluation data.

Checkpoint schema 5 and the individual-timeout reward marker reject former
shared-timeout checkpoints. The standalone formats are
`toy_global_individual_timeout_poisson_ac_v1` and
`toy_global_individual_timeout_poisson_baseline_v1`.

The complete suite passed **317 tests in 305.94 seconds**. Regressions include
mixed successful/failed cohorts, once-only terminal accounting, per-vehicle
return propagation, no-action failures, cross-rollout isolation, exact-deadline
arrivals, unpenalized global critic targets, normalization isolation, FM causal
returns and physical train/checkpoint/evaluation lifecycles. Compilation and
whitespace checks passed.

The six former city jobs were intentionally stopped and preserved under
`outputs/city_online_ac_dta_so_poisson_seed42_20261001/`. Six fresh 500-episode
runs started at 17:52 UTC under
`outputs/city_online_ac_dta_so_individual_timeout_seed42_20261001/`, using the
same demand seeds and evaluation schedule. The latter directory stores the
new reward contract, source snapshot, verification log and automatically
refreshed curves/status. Old-policy training is not continued under new labels.
