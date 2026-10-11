# Coulommiers and Montereau partial-corpus pretraining

Authorized scope: start independent UrbanDreamer offline pretraining on currently
available, checkpoint-committed data for both cities. Continue current collectors,
baselines, compactors, and Ingolstadt online training.

1. Freeze exact committed train archives from the current v5 collection family.
   Check input hashes, segment ordering, demand law, and traffic physics. Copy
   prepared network inputs and execution source into the new run.
2. Split by realized demand identity, with 15% held out across all methods.
   Verify full SHA256 for CRC duplicate candidates. Use a representative codec
   pilot; retain every unique trajectory in the full world-model corpus.
3. Run fresh city-specific codec training and compact-cache construction in
   parallel, bounded to available GPU memory and two CPU cache workers per city.
4. Require raw/cohort count and speed gates, including configured occupied/slow
   speed limits. Revalidate on the full held-out partition before dynamics
   training. Never admit an unready codec implicitly.
5. Automatically run five offline world-model epochs after prerequisites pass.
   Use causal paper-reward defaults. No automatic online phase is included.

Evidence: preparation_summary.json, immutable source_manifest.json, codec
metrics/progress/checkpoints, cache receipts/manifest, supervisor_status.json,
world_model.status.json, and pretrain_summary.json. Verify actual codec update
progress for both cities and leave durable background supervisors running.
