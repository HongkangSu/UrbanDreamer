# Keep / Drop Matrix

| Area | Decision | Reason |
| --- | --- | --- |
| `scripts/run_urb_fair_full_cycle.py` | keep | Canonical four-stage URB-fair runner with stable network flags. |
| `scripts/collect_urb_fair.py` | keep | Required to regenerate the post-mutation bundle and collection manifest. |
| `dreamer4/train_dynamics.py` | keep | Required world-model training stage. |
| `dreamer4/urban_routing/train_marl.py` | keep | Required imagined Dreamer CTDE training stage. |
| `scripts/eval_urb_fair.py` | keep | Required SUMO eval wrapper and manifest writer. |
| `scripts/eval_sumo.py` | keep | Required evaluator implementation used by the wrapper. |
| `scripts/select_best_eval_summary.py` | keep | Required best-checkpoint selector for eval probes. |
| Target-local `dreamer4/` package layout | keep for now | Preserves imports and behavior during migration. |
| Network folders | drop from target | Networks are selected externally through `--network-dir`. |
| Fixed-route eval wrapper | drop | Not part of the retained training mainline. |
| Sweep and ablation shell scripts | drop | Replaced by the canonical full-cycle runner. |
| Public legacy training-family choices | drop | UrbanDreamer exposes `dreamer_ctde` only. |
| External metric package integration | drop | Eval truth stays in target-local summaries and manifests. |
