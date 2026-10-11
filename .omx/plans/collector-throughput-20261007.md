# Collector throughput implementation

User authorized implementing the preceding collector/device optimization proposal.

1. Lock collector parity (actions, demand, reward, population, boundaries, empty episodes, archives and resume), then route Online AC through the shared EpisodeRunner without changing its public/checkpoint interface.
2. Add an explicit collection device distinct from learner device. Use episode-frozen CPU policy mirrors for small MLP actors, synchronizing weights and prior before every training/evaluation episode. Keep learner optimizers/replay on their existing devices. Record migration provenance; cross-device random draws need not be bit-identical.
3. Optimize IPPO return construction without changing rewards, gamma/lambda, clipping, epochs or minibatch counts. Optimize MAMBA online output transfers and reservoir allocations while preserving recurrent updates and reservoir RNG semantics.
4. Test real SUMO same-demand parity, CPU/GPU policy probabilities and prior restoration, old checkpoint compatibility, and bounded performance comparisons. Preserve all existing unrelated work.
5. Freeze verified source, preserve archives/metrics/checkpoints, restart affected workers from durable episode boundaries using existing targeted deployment tools. Preserve reward mode, prior schedule, actual N, revisit setting, evaluation seeds and5000-episode budgets; do not restart from scratch or alter the other online experiment.

Acceptance: shared collector regression tests pass; no stale CPU policy weights/priors; exact same-device action/reward parity where expected; improved measured inference/return construction; all restarted workers have verified live PIDs and durable resumed progress. In-flight episodes are preserved separately before replay. No new dependencies.

## Final status

Implementation and verification complete. User subsequently requested stopping the experiments. Both batches (27 workers and31 controllers) are stopped;13777 completed archives and durable checkpoints retained. Full suite1115passed; final targeted28passed; compilation/diff checks passed. No automatic restart is pending. Evidence: outputs/collector_optimization_20261007/stop_and_validation.json.
