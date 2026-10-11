# Offline pretrain -> online fine-tuning

User change: replace real warmup collection with existing offline archive pretraining.

1. Remove warmup stage and redundant warmup replay/source from the mainline. Preserve
   legacy configuration parsing used by baseline collectors and frozen archive snapshots.
2. Fresh runs begin at pretrain. Use the existing world_model_initialization_epochs
   budget and stream only the frozen offline manifest. Missing/empty sources fail
   before any SUMO session or initial checkpoint; no collection fallback.
3. Continue with real event online collection, world-model fine-tuning and existing
   Q/actor imagination, then deterministic evaluation. Offline samples never enter
   online policy replay or advance the real environment episode counter.
4. Update checkpoint lifecycle/version, profiles, documentation and verification tool.
5. Regression proof: pretrain cannot invoke session factory; all offline sequences
   visited once; epoch-boundary restore agrees; first online episode is zero;
   pretrain->online->evaluation smoke, full tests, compile and diff checks.

No new dependencies, no full city training, no mutation of source archives/jobs.
