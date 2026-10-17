# City executor performance diagnosis — 2026-10-12

Scope: diagnosis and proposal only. No production code edits, optimizer calls,
GPU work, or longer policy runs are authorized in this slice.

Evidence motivating diagnosis: the city integration smoke recorded a 113.6 s
imagination pass; an earlier legacy smoke recorded 3.2 s. Those runs used different
GPUs and shared-machine conditions, so this is a motivation, not a causal benchmark.

## Bounded measurement

1. Use the city smoke's immutable source bundle and completed checkpoint read-only.
2. Reconstruct the online replay sampler from policy seed 42, four WM batches, and
   two active-seed draws; verify its final NumPy RNG state against the checkpoint.
3. Use the first sampled seed, saved post-WM weights, fresh actor/critics, CPU FP32,
   two Torch threads, and one 30 s window. This preserves the physical seed and
   pre-actor-update policy, while declaring the CPU/narrow-horizon diagnostic.
4. Run an unprofiled reference and a cProfile replica with identical RNG/model
   states. Save complete physical/arbitration traces, endpoint states, event actions,
   costs, and RNG hashes. Require exact trace/output equality before interpreting
   performance evidence.
5. Attribute executor, priority/calendar, sorting, motion-loop, actor, and model
   work. Record profiler overhead rather than treating its wall time as throughput.

## Review gate

Append measured findings, a minimal optimization proposal, exact-semantics risks,
and a verification matrix. Submit to the root agent before changing executor code.
Output root: `outputs/component_experiments_20261011/executor_performance_v1/`.
