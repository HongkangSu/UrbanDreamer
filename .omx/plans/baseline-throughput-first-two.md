# Baseline throughput: first two improvements

Scope: preserve demand, rewards, prior, RNG draw order, vehicle physics and update counts. Running dataset snapshots remain unchanged.

1. Lock existing observations, prior, event boundaries and model-based updates with targeted regression tests before edits.
2. Delete per-decision full-network flow copies from Dreamer/MBPO imagined rows; actor consumers only need local candidate features. Verify identical decisions, returns and gradient updates against the pre-change source.
3. Add wall-clock timing for real collection, archive writing, world model updates, imagination generation, actor/critic updates and complete episode updates. Timings must not force CUDA synchronization per event.
4. Cache immutable candidate features by (current edge, destination) on each candidate provider; bound cache memory. Keep mutable traffic/progress/population/distance freshly filled. Reuse frozen window edge features/global means and batch action/entropy host transfers.
5. Validate exact cached/uncached observations, cache isolation and stale dynamic values; run relevant regression tests and real Toy smokes. Benchmark repeated observations on real city topologies and equivalent seeded model-based update workloads.

No dependency additions, changed simulation step, shortened rollouts, reduced update count, AMP, actor device changes, or new vehicle advancement algorithm.

Completed: implementation, seeded parity, CPU/GPU real SUMO smokes, whole repository regression (507 passed; 2 CUDA cases verified separately), compilation and whitespace checks. Evidence: outputs/baseline_throughput_first_two_20261002/verification.json. Running batch remains on its original frozen source.
