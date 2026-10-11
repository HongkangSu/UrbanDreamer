# UrbanDreamer component experiment campaign

User-authorized goal: execute the six experimental sections, report measured
results, diagnose failures and revise models when warranted. Baseline worktree
preserved in commit `2a084d7` before experimental implementation.

## Completion requirements

1. Overall routing: audit current runs; compare compatible fixed checkpoints
   with common demand realizations, cutoff accounting, completion and never-started
   counts; report true interaction costs including offline collection and warmup.
2. Prediction: train graph FM, no-coarse-prior FM, direct regression, graph
   regression, Gaussian and random-coarse-prior FM on identical replay/codec
   controls. Measure single and genuinely recursive predictions and distributions.
3. Spectral: raw/hybrid/fixed-encoder increments, ranks including current r16,
   random projection references, congestion strata and episode-level intervals.
4. Supervision: independent vs recursive FM on identical cohort targets;
   sample/mean endpoint MSE experiments clearly identified as new mechanisms.
5. Disagreement: held-out mean prediction error association and conditional
   congestion analysis; implement/test a compatible policy-risk mechanism before
   penalty policy comparisons (bounded physical critics currently forbid beta>0).
   Any gate experiment is an addition, not deletion of an existing component.
6. Reliability/generalization: recorded-action recursion, explicit vehicle/count
   consistency, demand intensity/timing shifts and unseen demand groups with frozen
   policies; distinguish model internal consistency from external SUMO accuracy.

## Staging and integrity

- Existing production runs keep their frozen sources and output directories.
- First development corpus: 128 train demand groups (819 trajectories, 87685
  windows), 32 existing validation groups (187 trajectories, 19576 windows).
  The complete parent corpus remains available for final converged runs.
- All validation is development data. Never relabel it as untouched final test.
  Fresh evaluation demand seeds start at 710001; development uses 610001.
- First screen: same fixed codec, paired initialization seeds and window schedule;
  budget calibrated by actual gradient throughput and validation learning curves.
  Screening alone does not fulfill the final comparison requirement.
- Preserve negative results. Log every revision, reason, checkpoint/source hash,
  training exposure, solver NFE and wall time. Revisions use validation only.
- Representative final comparisons need >=3 independent training initializations;
  paired demand evaluation with episode/demand-group uncertainty intervals.
- Exact target scope, speed interval semantics, masks and future-demand visibility
  are checked before interpreting recursion or generalization.
- Tests verify transitions, loss gradients, projection/noise separation,
  checkpoint replay, probability scores, cohort membership and paired demand.
- Resource admission checks free GPU memory; no automatic production termination.

## Work lanes

Leader: frozen corpus/checkpoint, training/evaluation orchestration, integration,
failure diagnosis and final verification.
Spectrum: CPU spectral analysis and artifacts.
Variants: isolated dynamics adapters and tests.
Metrics: physical/probabilistic evaluation and grouped statistics.
Generalization: SUMO demand manifests, paired evaluation and checkpoint adapters.

## Status

2026-10-11: baseline committed; source data/checkpoint revalidated and snapshotted;
implementation underway. No experimental performance claim established yet.
