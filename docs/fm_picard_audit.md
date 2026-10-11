# Picard imagination solver audit

Read-only audit of `urban_dreamer/models/world_model.py::Dynamics.solve` and serial/parallel imagination call sites. The reference is the configured 100-step explicit Euler flow solve for one 30-second physical transition. This audit does not change the running solver.

## What the current Picard implementation computes

Let `N=100`, `h=1/N`, initial latent `x0`, and fixed window condition `c`. Sequential Euler is

`x[j+1] = x[j] + h * v(j*h, x[j], c)`.

The implementation initializes every trajectory node to `x0`, then performs simultaneous updates

`X_new[j] = x0 + h * sum(v(i*h, X_old[i], c), i=0..j-1)`.

The velocity calls at all nodes are independent inside each Picard iteration and can be batched. The fixed point is exactly the same discretized Euler trajectory, not a different higher-order integrator. After K iterations, the first K Euler nodes are exact in exact arithmetic. After N iterations, the entire N-node trajectory matches Euler, without requiring a global contraction assumption, because this discrete system is strictly triangular. Floating-point batching and cumulative sums can introduce roundoff differences.

**Two Picard iterations at 100 nodes do not guarantee an accurate final sample.** Existing four-node/four-iteration equality tests verify the finite-propagation property, not the accuracy of two iterations on 100 nodes. Example for `x'=a*x`, `x(0)=1`:

| a | Euler100 endpoint | Picard2 | Picard4 | Picard8 |
|---:|---:|---:|---:|---:|
| 1 | 2.704814 | 2.495000 | 2.695912 | 2.704812 |
| 5 | 131.501258 | 18.375000 | 63.095156 | 125.023290 |
| 10 | 13780.612340 | 60.500000 | 614.322500 | 6020.885123 |

These examples are mathematical checks, not claims about the learned field's Lipschitz constant. The model currently enforces no global bound establishing that K=2/4/8 is sufficient. Increasing the node count does not by itself make the full cumulative Picard map contractive; its sup-norm Lipschitz bound scales with the whole integration interval, not just one node's step size.

## Speed and memory

Picard trades more total network work for reduced sequential depth. Euler100 processes 100 velocity evaluations per sample in series. Picard K over 100 nodes processes K*100 sample-equivalent velocity evaluations, arranged into K larger network calls (or more calls when chunked). Wall-clock improvement depends on GPU spare throughput and batching efficiency; a speedup must be measured at the real imagination batch, not inferred as `100/K`.

Production parallel imagination can have eight active rollouts and three disagreement samples. The effective Picard batch is then `100*8*3=2400`. At 853 roads, 256 condition channels and FP32, each fully repeated condition tensor occupies approximately **1.95 GiB**. Both edge tokens and reference tokens therefore require **3.90 GiB**, before trajectories, velocities, transformer token activations and attention workspace. One 100-node latent trajectory is approximately 0.49 GiB; simultaneously retaining old/new trajectories and velocity adds more.

The existing `space_chunk` controls attention sub-batches, but the entire `spatial_proj`, condition arrays and token arrays can still be materialized first. Thus attention chunking alone cannot solve this memory growth.

A bounded implementation should chunk **complete velocity calls** across flow nodes/effective batch and construct expanded condition/action/time tensors only for the current chunk. Do not materialize the full repeated condition and slice it afterward. Retain the old trajectory while collecting velocities; never update nodes in-place and use already-updated nodes within the same iteration, since that changes the Jacobi update. For inference under `no_grad`, preallocated buffers can reduce peak allocation. A differentiable recursive-training path needs graph-preserving concatenation or otherwise validated autograd behavior.

## Noise, disagreement, and causality

The current code correctly samples noise once in `prior_start` (or once for the stack of disagreement source samples) and reuses the resulting start throughout the solve. It does not resample noise inside Picard iterations. Conditions are encoded once, then replicated without rerunning the event encoder.

Flow nodes are flattened into the batch dimension, separate from the physical-window dimension. This preserves the same causal structure as the corresponding Euler call; flow node guesses must never be inserted as physical time tokens. Multiple disagreement samples remain independent batch members.

Chunking must preserve ordering: node, then sample/rollout batch, then physical interval. A mismatch can silently mix conditions or noise between vehicles/rollouts. Test reordered input batches, multiple noise draws, and per-sample serial equality.

Disagreement is particularly sensitive to approximate integration: Picard error shared across samples can bias their means, and different integration errors across samples can alter variance. An acceptable endpoint-average speed error alone does not establish that the uncertainty penalty remains comparable.

## Acceptance proposal

Keep Euler100 as the reference until paired tests pass. For each actual checkpoint condition, reuse exactly the same noise and compare Picard2/4/6/8 (and larger counts if needed) against Euler100. Include congested/slow roads and varied traffic loads rather than only empty or free-flow states.

Measure:

- latent RMS, relative RMS and road-wise tail errors to Euler100;
- decoded speed/count error, occupied and slow-road error, and error against real labels separately;
- active-fraction/reward readout changes;
- disagreement mean/variance changes using the production three noise samples;
- eight-window physical rollout drift with the same seed states and controlled action policy;
- full production-sized batch wall time and peak GPU memory.

Comparing successive iterations is useful for early stopping, but endpoint-only cancellation can hide interior errors. Monitor the entire trajectory using an absolute-plus-relative scale, with road-wise tails in addition to global RMS. The difference `X[k]-X[k-1]` is the fixed-point residual of the previous iterate; checking the final iterate's residual requires another field sweep. No residual threshold is a rigorous endpoint-error guarantee without a stability/Lipschitz bound. Calibrate candidate thresholds against paired Euler100 errors and observed downstream reward/uncertainty differences.

A practical deployment interface may expose a bounded node/effective-batch chunk size, maximum iterations, optional absolute/relative trajectory tolerances, and a convergence/fallback metric. Solver and iteration/tolerance choices affect generated trajectories and belong in the semantic fingerprint. Chunk size is a resource parameter, provided chunked and unchunked results agree to numerical tolerance. Fixed-iteration runs are easiest to benchmark reproducibly; adaptive stopping should report the actual iterations and failure rate.

If a bounded iteration budget fails its acceptance checks, preserve Euler100 for that use case. Do not silently select Picard2 merely because `picard_iterations=2` was an old configuration default.

## Primary reference

Shih et al., *Parallel Sampling of Diffusion Models*, NeurIPS 2023, introduces ParaDiGMS: parallel guesses of future solver nodes refined by Picard iteration. Its central tradeoff is extra parallel computation for less sequential latency; practical convergence checks and adequate compute matter. This supports investigating Picard, not assuming that an arbitrary two iterations match a sequential trajectory.

- Paper: https://arxiv.org/abs/2305.16317
- Authors' implementation: https://github.com/AndyShih12/paradigms

This document's finite-step triangular-system argument and memory arithmetic follow directly from the UrbanDreamer implementation. Actual urban-model benchmark results are produced by the separate paired GPU experiment.

## Bounded implementation and measured city results (2026-10-08)

`Dynamics.solve`, `transition`, and `latent_disagreement` now accept `picard_batch_chunk_size`, default 64 effective batch members. Zero explicitly selects the old unbounded path for diagnostics. Conditions, actions, and flow times are indexed for each chunk before the complete network call; full 256-channel repeated conditions are never built in advance. All chunks read the same old trajectory, and the cumulative update happens after every chunk finishes. Physical intervals and disagreement-sample ordering remain unchanged. Full 64-channel trajectories/velocities remain resident; this reduces inference memory but does not make total memory independent of node/sample count. Differentiable callers still retain their autograd graphs.

The chunked and unbounded implementation agree in FP64 analytic output/gradient tests and in real Dynamics fixed-noise disagreement tests. A separate analytic test demonstrates that Picard2 on 100 nodes is not equivalent to Euler100, while Picard100 reaches the same discrete endpoint. Default Euler configuration and the running pretrain/online experiments were not switched.

Paired GPU1 FP32 tests freeze a checkpoint in `outputs/ingolstadt_noincident_pretrain_20261008/picard_benchmark/source_checkpoint.pt`. These are partial-pretrain diagnostics, not evidence about every eventual model. The congested held-out case has 577 observed vehicles and 517 vehicles on slow queues; three noise samples match the imagination disagreement calculation. At effective batch 1, chunk64, 100 nodes, medians from three alternating repeats were:

| Solver | Seconds | Latent relative L2 to Euler | Mean variance / Euler | Occupied-speed MAE to real label (m/s) |
|---|---:|---:|---:|---:|
| Euler100 | 0.901 | 0 | 1 | 0.9584 |
| Picard2 | 1.626 | 7.35% | 14.05% | 1.1873 |
| Picard6 | 3.977 | 1.19% | 89.83% | 0.9590 |

Peak allocated memory is 112.6MiB for Euler and 842.1MiB for chunked Picard in this case. The old unbounded three-noise path on a lighter case used 2195.9MiB for Picard. These are different traffic cases, so that latter pair is not a strict same-condition memory ablation. Single-sample Picard2 on the light case was only modestly faster (0.474 vs Euler0.563 seconds), while three-sample Picard2 was slower (1.607 vs0.915 seconds). Shared-GPU timings fluctuate; they do not establish an absolute latency guarantee.

The current acceptance checks reject a default switch to Picard: low iteration counts introduce substantial latent/disagreement differences, and more accurate counts cost more time. Euler is only the discrete integration reference in these comparisons, not a true traffic label. Speed/count metrics against observed labels are recorded separately in each JSON result.

The final production-shaped test uses eight copies of the congested start with independent noise, giving 8 rollouts ×3 samples. GPU1 FP32, space64/time65536/MLP8192, two warmups and three interleaved repetitions:

| Solver | Seconds per macro transition | Peak allocated MiB | Effective velocity batch members |
|---|---:|---:|---:|
| Euler100 | 5.412 | 381.6 | 2400 |
| Picard2 / chunk64 | 10.236 | 2681.9 | 4800 |
| Picard6 / chunk64 | 32.165 | 2681.9 | 14400 |

This measures the world-model/disagreement prediction call with the imagination batch shape; it excludes actor updates and physical vehicle advancement, and does not simulate eight distinct demands. The observed slowdowns are 1.89× and 5.94×. Four distinct held-out windows spanning 1/124/244/577 active vehicles give Picard2 latent relative errors4.34%–13.27% and mean-variance ratios10.8%–20.9%; Picard6 gives0.865%–1.908% and76.8%–89.8%. Three-sample variance is noisy; paired comparisons use identical noise.

Complete experiment summary: `outputs/ingolstadt_noincident_pretrain_20261008/picard_benchmark/REPORT.md` and `summary.json`. All tests share one frozen checkpoint SHA256. The broad initial sweep is explicitly `bounded_partial`; the final specified chunked cases are complete. All private benchmark processes are finished. Related solver/rollout/trainer tests:149 passed; additional configuration/diversity/chunk tests:141 passed. Python compilation and `git diff --check` passed. No new dependencies.
