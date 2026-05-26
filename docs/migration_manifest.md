# Migration Manifest

UrbanDreamer was populated from the current URB-fair mainline in the adjacent
`dreamer4` workspace and then reduced to the active training path.

## Compatibility Package

The target keeps a `dreamer4/` package name in this migration pass. This is a
deliberate compatibility boundary: the training and evaluation code share many
intra-package imports, and retaining the package name preserves behavior while
the public project surface is slimmed down.

## Retained Entrypoints

- `scripts/run_urb_fair_full_cycle.py`
- `scripts/collect_urb_fair.py`
- `dreamer4/train_dynamics.py`
- `dreamer4/urban_routing/train_marl.py`
- `scripts/eval_urb_fair.py`
- `scripts/eval_sumo.py`
- `scripts/select_best_eval_summary.py`

## Retained Supporting Modules

The supporting modules under `dreamer4/urban_routing/` are the runtime closure
needed by collection, world-model training, imagined policy training, and SUMO
evaluation. They remain internal implementation detail; the stable public
network contract is documented in `docs/contract_freeze.md`.

## Verification Scope

The migrated test set covers:

- URB-fair collection protocol contracts;
- evaluation wrapper argument construction and manifest behavior;
- full-cycle command translation for `--network-dir` and `--network-name`;
- Dreamer CTDE MARL mode behavior;
- absence of out-of-scope public surfaces in source code.
