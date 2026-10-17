# Recursive training efficiency plan (2026-10-17)

The user requested stopping the three cached single-step pretrainers before
optimizing the current three-transition recursive training path. Preserve the
three supervised physical transitions, 100-step Euler definition, noise law,
recorded cohort controls, loss targets, and gradients through earlier predictions.

1. Establish loss, parameter-gradient, and RNG-state equivalence against the
   existing recursive loss with fixed inputs, flow times, and noise.
2. Skip the final endpoint solve when its generated output is not consumed.
   The final flow-matching and prior losses still run; earlier endpoint solves
   retain their complete backward paths. Keep the existing endpoint-returning
   API as the compatibility default for external callers.
3. Profile forward/backward time and saved activation memory. Apply a bounded
   optimization to repeated integration work only when equivalence is verified.
   Do not reduce solver steps, truncate gradients, or change supervision depth.
4. Run focused regression tests and bounded hardware benchmarks; record actual
   device, graph size, batch, precision, solver, timing, and memory. Do not restart
   production training during this optimization task.

Initial stop: SIGTERM to queue controller 3329948, followed by SIGINT to
pretrain PIDs 3291922, 3330434, 3330537. All seven related controller/child PIDs
were absent in the subsequent process check. Their logs end in KeyboardInterrupt;
existing atomic checkpoints remain in the original output directories.

## Implemented execution changes

- `recursive_fm_loss(predict_endpoints=False)` omits the unconsumed final solve;
  its compatibility default still returns all endpoints.
- The mainline uses 10-node integration checkpoints, preserving the complete
  Euler/midpoint backward path through both prefix transitions.
- Recursive observed frames are encoded once (four frames instead of six).
- Deterministic positional tensors use a bounded cache, and deterministic
  transformer checkpointing omits RNG preservation.
- Remainder-batch execution slices all current-state/background conditions.
  Its existing regression fixture exposed a stale two-field condition path;
  the updated fixture exercises nonzero conditional outputs and gradients.

## Validation

The focused regression command covering recursive losses, positional constants,
world-model training, codec isolation, solvers, compiled-block compatibility,
activation offload, tail execution, imagination solvers, and pipeline returned
**271 passed, 1 skipped**. Source compilation and `git diff --check` passed.
Fixed-input CPU checks cover loss, endpoint, input/action and parameter gradients,
and RNG identity. AOT-compiled block backward compatibility also passed.

Bounded CUDA BF16 probes use the real 853-segment Ingolstadt graph with synthetic
latent targets, three supervised transitions, 100 Euler nodes per transition,
and the full six-block dynamics network. They measure FM forward/backward only;
they do not include archive collation, codec, readout or Adam updates, and do not
compile blocks. Actual training throughput on the corpus still requires a run.
Raw records and stopped-process evidence are in
`tmp/recursive_training_efficiency_20261017/`.

After one warmup, medians of two measurements on RTX 3090 / CUDA 0:

| Path | Batch | Forward + backward (s) | Peak allocated MiB | Sequences/s |
| --- | ---: | ---: | ---: | ---: |
| Original recursive implementation | 1 | 10.733 | 2144.2 | 0.0932 |
| Skip unused final solve + deterministic execution reuse | 1 | 9.017 | 1481.5 | 0.1109 |
| Above + integration checkpoints every 10 nodes | 1 | 10.875 | 210.1 | 0.0920 |
| Above + integration checkpoints every 10 nodes | 8 | 15.438 | 1019.3 | 0.5182 |

At batch 1, the speed-only path improved throughput by 19.0%. Integration
checkpointing reduced peak allocated tensor memory by 90.2%, with approximately
unchanged elapsed time relative to the original implementation. Batch 8 with
checkpointing produced 5.56 times the sequence throughput of original batch 1,
while using less peak tensor memory. These are short, shared-GPU computational
probes, not measured full-corpus training speedups. Allocated tensor memory is
not total process memory reported by nvidia-smi. Losses for the batch-1 variants
were identical in the probes; the batch-8 case contains a different set of
synthetic examples and is not a loss-equivalence comparison.

For subsequent production trials, increase the microbatch only after checking
actual codec/event/optimizer overhead and preserve the desired effective batch
with gradient accumulation (for example 8 x 8 for an effective batch of 64).
No long-running production job was restarted. The 4,302/4,483-segment cities
have not yet received an independent hardware benchmark under these changes.
