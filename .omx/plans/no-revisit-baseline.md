# Hard no-revisit baseline, no shortest-path policy prior

User request: preserve/stop current baseline batch; fresh three cities × Online AC/GNAC/Dreamer/MBPO/Random, 5000 episodes each. Prior weight must be zero throughout. Scope excludes unrelated mainline/offline work.

Node definition: existing decision graph nodes are directed external road edges (same as earlier revisit metric). Clarification requested; use this context-consistent definition absent contrary steering.

Contract:
- Opt-in `--forbid-revisit`, default false, requires event session and zero policy-prior weight.
- Per-vehicle immutable `visited_edges` tuple in snapshots, archives, replay and checkpoint seeds. Current edge is always blocked for the next action.
- Candidate mask removes visited nodes and successors unable to reach the destination in the graph with visited/current nodes removed. Never relax mask to permit a repeat.
- SUMO full execution route must use the same constrained graph, including automatic singleton/skipped roads. Track actual route-index traversals, not only observed fork decisions; arrival consumes remaining route suffix.
- No valid candidate is an explicit contract failure (should be unreachable under certified path invariant), never fabricated arrival or unmasked fallback.
- Imagination updates history on physical edge entry, uses history-aware masks and validates selected commitments in legacy and event stepping.
- Fresh initialization/output; old archives remain separate. Checkpoints/config/protocol/archive metadata carry mode and reject incompatible resume.
- Preserve actual Poisson episode N, zero-population semantics, event costs, demand seeds, budgets and old default behavior.

Verification: history-sensitive/cache-isolated masks, trapped successor pruning, skipped physical roads and arrivals, real/imagined branch commitment consistency, archive/replay history, resume guards, zero prior, Toy real SUMO no-repeat audits, small city real audits, then launch15 processes and verify source/config/PIDs/first archives.

Implementation and verification complete. Oldbatch preserved29241archives and stopped. New15jobs running at /home/wangwenrui/shk/UrbanDreamer/urban_dreamer/outputs/city_baseline_dataset_no_revisit_5000ep_20261004_201004. Final evidence: verification.json, first_episode_audit.json. Three full realcity audits have zero physical/decision revisits; CPU/GPU Toy and resume checks passed.
