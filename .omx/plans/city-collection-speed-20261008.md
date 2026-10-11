# City collection speed correction

User request: stop current baselines, disable all shortest-path priors, and fix the excessive collection time on Coulommiers/Montereau before any new full training.

1. Stop batch workers and all admission/compaction owners; retain checkpoints and historical inputs. Set future launch weights to zero, preserving previous protocols as historical evidence. Do not resume training during diagnosis.
2. Freeze original source and record recent phase timings and map/vehicle statistics. Profile complete no-prior, full-population AC episodes for each city using fixed actor initialization, demand seed, one CPU thread, unchanged SUMO physics and archive output.
3. Protect flow representation and route/observation semantics with regression tests before optimizing the identified scalar and repeated route/table work. Reuse existing topology/session caches; do not simplify the physical network, alter sampling intervals or remove episode data.
4. Compare original and revised full episodes using identical seeds and digest all policy features/actions and traffic results. Measure production timing separately from profiler overhead, include warm episodes, and distinguish unavoidable active-population/network scaling from removable Python overhead.
5. Run focused session/topology/baseline/prior tests and broader checks as warranted. Save figures and a report containing timing ratios, behavior-equivalence evidence and remaining limitations. Leave full training stopped.
