# MAPPO behavior check failure: numerical diagnostic (2026-10-13)

## Scope and confirmed runtime evidence

Read-only diagnosis; no production source changes, training, SUMO execution, process restarts, or checkpoint writes.

Coulommiers frozen campaign `outputs/city_baselines_two_city_fresh2500_20261011` ends in `RoutingPpoMethod.on_episode` before critic/actor updates for episode 533. `coulommiers/mappo/latest.pt` contains episode 532; checkpoint SHA-256 `ef8eca24e77c2be5ca42e48f7fe632bbdfd783df9da4c522aad1f4782866de14`. Collection is CPU, learner CUDA:0; shortest-path weight zero and training temperature one. CollectionPolicy mirrors weights before every episode. Actor scorer has no dropout/batchnorm. Actor and optimizer are unchanged between episode collection and the guard. The guard compares sampled **log probabilities**, not raw probabilities, with `atol=1e-4, rtol=1e-5`.

The checkpoint policy head has common scalar bias −6.34956 and mean absolute weights 1.00080. Montereau last saved episode 390 has bias −6.09206 and mean absolute head weight 1.48155, consistent with a shared numerical mechanism but not independently reproduced here. Ingo attempt3 log has the same guard error after episode 2754; its checkpoint was not loaded because this campaign stores handoff paths outside the log tree.

Archived/current ppo.py and collection_policy.py are byte-identical. Current policy.py has additional dynamic-prior/bounded-critic features; NextHopActor's score computation remains equivalent for zero prior and temperature one used here. Source archives remain untouched.

## Archive limitation

The failed episode NPZ contains 50,608 decision records, frozen 30-second flow, boundary states, counts and summary. Decision records retain vehicle ID, current/destination/selected edge and time, **not** collected `old_log_prob`, features/mask, event batch membership, live active fraction or decision remaining distance. Exact failed-row reproduction is therefore unavailable without a deterministic simulator replay, which was outside this diagnostic scope.

The probe uses recorded current/destination/action edges and actual window flow, but sets remaining distance to zero and uses window-start context instead of the missing live context. It demonstrates a failure mechanism with real checkpoint/traffic-derived inputs, not the exact in-memory failing row. No data leakage or source changes are needed for this demonstration.

## Observed mechanism

On all 50,608 derived events, many legal logits share a large common offset near −258 while meaningful candidate differences are around 0–3. CPU single-event inference versus CUDA batches of 512 produces only one sampled-logprob mismatch above the existing guard tolerance:

- Derived event ordinal 14457: CPU legal logits `[−256.5604858, −258.0604248]`; GPU `[−256.5604248, −258.0604858]`.
- Selected logprob CPU −1.70135498 vs GPU −1.70147705; absolute discrepancy 0.0001220703125, greater than the row threshold approximately 0.000117.
- Selected action probability changes only 0.00001818 (≈0.18243462 to ≈0.18241644).
- Each raw logit moved at most two float32 ULPs. Opposite movements and categorical log-normalization amplify this common-offset cancellation.
- CPU1→CPU512 maximum sampled-logprob difference 0.000091552734375, zero guard violations; GPU512 maximum 0.0001220703125, one violation.

The policy is effectively unchanged; a strict behavior guard mistakes device/batch float32 roundoff for changed behavior. This is stronger than merely guessing a tolerance issue. It is not evidence of policy mutation or causal collector leakage.

## Numerically stable candidate-score gauge

Temporary monkeypatch only, evaluated with the same checkpoint and all 50,608 derived inputs:

```python
hidden = actor.candidate_scorer(features)
first = mask.to(torch.int64).argmax(dim=-1)
reference = hidden[torch.arange(len(hidden), device=hidden.device), first]
logits = F.linear(hidden - reference[:, None, :], actor.policy_head.weight, bias=None).squeeze(-1)
logits = logits.masked_fill(~mask, -torch.inf)
# Existing shortest-path prior still applied after this score.
```

This computes `w·(h_k−h_ref)` before the large absolute score exists. Removing shared scalar head bias changes no categorical policy in exact arithmetic. Subtracting raw logits only after head projection cannot recover precision already lost. Mask-selected reference avoids using invalid padding; no parameter shapes or checkpoint tensors need change.

| Comparison to CPU1 | Existing max | Centered max | Existing guard violations | Centered violations |
|---|---:|---:|---:|---:|
| CPU512 sampled logprob | 9.1553e−5 | 3.0994e−6 | 0 | 0 |
| GPU512 sampled logprob | 1.2207e−4 | 3.5763e−6 | 1 | 0 |
| CPU64 sampled logprob | 5.4266e−5 | 4.9454e−6 | 0 | 0 |

GPU CPU probability discrepancy falls from 3.0577e−5 to 1.1325e−6. All CUDA inference used physical GPU1 (`CUDA_VISIBLE_DEVICES=1`), .025 memory cap, one CPU thread; physical GPU0 was not allocated.

These measurements support candidate-score gauge stabilization as the precise first fix. They do not prove all future saturated policies/device combinations will satisfy a fixed tolerance.

## Recommended minimal implementation and tests

1. Add explicit opt-in/stamped centered score gauge for MAPPO/IPPO, or a reviewed actor-wide implementation with a protocol version. Keep old frozen checkpoints/evaluation snapshots unchanged. Use the existing policy head weight and retain bias tensor for checkpoint compatibility; omit its contribution under centered gauge.
2. Preserve the current behavior integrity guard. Do not replace `old_log_prob`, bypass checks, or merely increase atol. Include max absolute error, failing count/index and actor/collector identity in future failure diagnostics; persist only a small mismatch record and collection batch metadata, not huge corpora.
3. Unit tests: scalar head bias shift has no effect; padded illegal reference cannot affect scores; candidate permutation leaves probabilities invariant after undoing permutation; float64 equivalence with old formula including nonzero prior and temperature; genuine stale/mutated actor still rejected.
4. Migration tests: old actor tensors/Adam state restore unchanged; explicit protocol migration is required for centered gauge; rollout/update share the same gauge. Resume from last successfully updated episode with a NEW immutable source snapshot and provenance pointing to old snapshot/checkpoint.
5. Bounded post-fix verification: this archived-derived CPU1/CPU512/GPU512 benchmark plus a short genuine collected episode on the new source before long continuation. A failed collection already archived should be regenerated from episode 532 state without mixing the old and new score contract; preserve failed archive and original logs.

## Evidence artifacts

- `tmp/mappo_behavior_diagnosis_20261013/probe_all.py` and `probe_all.json`: full original benchmark.
- `tmp/mappo_behavior_diagnosis_20261013/probe_centered.py` and `probe_centered.json`: temporary centered benchmark.
- `probe.py` and `probe.json`: bounded first 12,000-event precursor.

No new repository dependencies or production code changes were made. This is a diagnosis/recommendation artifact, not a completed runtime migration or formal routing performance result.

## Core fix implemented after root preservation commits

Root authorized implementation after `cfa49bf` and `1ed5625`. Changed only:

- `urban_dreamer/models/policy.py`: opt-in `actor.score_gauge=pre_head_legal_reference_v1` in NextHopActor.forward_logits; missing attribute uses original legacy computation. Bias tensor retained; hidden first LEGAL reference subtracted before bias-free head projection. Existing prior and temperature paths retained.
- `urban_dreamer/learning/baselines/ppo.py`: constructor validates args.score_gauge; enabled learning_protocol version 4 includes score_gauge. Legacy version 2/3 and all old fields unchanged; strict load guard enforces explicit migration both directions. Original frozen-behavior logprob guard unchanged.
- `tools/train_toy_baseline.py`: --score-gauge choices legacy/pre_head_legal_reference_v1; nonlegacy rejects methods outside MAPPO/IPPO before any setup.
- NEW `tests/test_policy_score_gauge.py`: 21 test cases.

Final core regression command (physical GPU1):

```sh
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home/wangwenrui/opt/miniconda3/envs/URB/bin/python -m pytest -q tests/test_policy_score_gauge.py tests/test_routing_ppo_baselines.py tests/test_actor_sampling_temperature.py tests/test_collection_policy_device.py tests/test_urban_dreamer_policy.py
```

Result: **79 passed in 9.52 seconds**. py_compile of all four files and git diff --check passed. No new dependencies.

A broader invocation including test_additional_penalty_ppo_iql.py had 90 passes and 2 failures. Both failures are pre-existing MAPPO fixture observation dimension mismatch (10 supplied vs 12 expected because args lacks explicit legacy_exposure). Both were independently reproduced against untouched `outputs/city_baselines_two_city_fresh2500_20261011/source` using /tmp cwd and PYTHONPATH explicitly pointing at that frozen source; stack traces confirmed archived ppo.py and common.py. Unrelated fixtures were not changed.

No new three-city jobs, recovery checkpoint migration, runtime configuration changes or restarts were executed. Root owns migration/launch. Current fix remains uncommitted for root review and integration.
