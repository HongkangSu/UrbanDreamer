# Accepted execution: codec quality gate, then complete FM pretraining

User authorized training the codec until usable, then starting the full offline
pretraining workflow, and removing future newborn vehicle contributions from
the transition volume labels.

- Preserve full observed input/codec same-frame count targets. Fix count loss
  units/foreground weighting/mass constraint and actual positive speed decode.
- Validate codec on genuinely disjoint demand identities. Gate: occupied count
  MAE<=0.5, all-road<=0.1, empty<=0.05, total relative<=0.1, speed<=1.5m/s.
  Failure stops before FM; budget limits never imply readiness.
- Each sequence fixes its initial active cohort C0. Future volume, actions and
  exposure track that cohort, while speed/occupancy/halts retain full observed
  traffic background. This is not a physical no-newborn counterfactual.
- Stage FM freezes codec, consumes cohort targets, and traverses all eligible
  starts of the frozen training partition. Held-out demand groups never train.
- Full Ingolstadt corpus preprocessing uses committed frozen source list,
  verified dedup and content hashes, compact membership cache; source files and
  running collectors remain unchanged. Cache building overlaps codec training.
- Complete implementation/test/real codec acceptance, then launch durable
  pretrain-only full job and verify that it enters FM with codec frozen.
- Checkpoint8/replay6, stage+validation/cache identity and RNG restore.
