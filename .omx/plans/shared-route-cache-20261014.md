# Shared route candidate cache

User scope: the three running UrbanDreamer current_to_dest online jobs share
static route candidates between real collection, route validation and all eight
imagination workers. Cache capacity is 200000 pairs per city/network. Keep the
1000-episode target, FP16, eight workers/eight rollouts and two imagination passes.

1. Protect existing candidate ordering, padding, tie handling, route legality,
   feature values and process-scheduling-independent training with existing tests.
2. Add one bounded cross-process cache for static candidate tuples, isolated by
   topology, passenger permissions, free-flow costs and algorithm version. Do not
   cache traffic-dependent observations, policies, RNG, or optimizer state.
3. Reuse the same cache in real collection/session validation and spawned workers;
   avoid introducing dependencies or changing configuration/checkpoint fingerprints.
4. Test cross-instance and spawned-process reuse, concurrent cold misses, bounds,
   topology isolation and serial/parallel actions/losses/gradients/RNG equivalence.
   Measure lookup overhead before selecting the backend for deployment.
5. Create a new immutable source derived from the currently running frozen source,
   changing only cache-related files. Preserve/checksum configuration and committed
   model/optimizer/replay/RNG checkpoints; stop old jobs at an episode boundary and
   resume the same outputs/cursors under that new source. Keep unrelated baselines.
6. Verify live parent/worker cache identity and capacity, actual cache reuse and
   preserved 1000-episode/FP16/8-worker/8-rollout/2-pass protocol; retain handoff and
   verification records. Long initialization remains a startup phase, not training.
