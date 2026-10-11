# Requested Ingolstadt profile for two-city world-model pretraining

User requested the Ingolstadt execution configuration, specifically batch64.
The final profile uses batch64, accumulation1, BF16, compiled full blocks,
activation checkpointing, chunks64/65536/65536, four threads and one epoch.

Montereau's committed checkpoint preserves420 updates,13440 samples, all models,
optimizer states, RNG states and sampler permutations/cursors. Coulommiers reuses
its codec4000 checkpoint and the completed full-corpus compact cache.

Actual probes show the original full profile exceeds available shared GPU memory.
The verified runtime retains the same joint loss and sample/optimizer counts,
offloads saved tensors to CPU without expanding broadcast adjacency, and executes
large remainder transformer forwards in subgroups before concatenating outputs.
Full batches still execute64 together. Full64 and remainder63 actual backward,
decoder updates and frozen-encoder invariants pass.

Finish plan: remove the unsuccessful experimental compiled-tail option and its
test while retaining original static-full dispatch tests; keep the proven
tail-execution and offload tests. Validate the source bundle after final edits,
preserving already running process source by publishing a subsequent revision
if needed. Verify actual Coulommiers64 updates and Montereau's queue state.

Resource preference is pending: user previously requested continuing baselines,
so no baseline job is paused without explicit priority selection. GPU1 currently
has5074MiBfree versus6272MiBadmission requirement. Continue independent GPU0 work.
