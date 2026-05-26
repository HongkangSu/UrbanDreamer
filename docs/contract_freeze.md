# URB-fair Contract Freeze

This document freezes the public network-switching surface for the migrated
UrbanDreamer mainline.

## Required Public Flags

```bash
--network-dir /path/to/network
--network-name <name>
```

These names are stable. New code should translate them inward rather than
replacing them with stage-specific names.

## Translation Matrix

| Public input | Collect stage | Dynamics stage | MARL stage | Eval stage |
| --- | --- | --- | --- | --- |
| `--network-dir` | `NETWORK_DIR`, `network_folder` | not used directly except `EDGE_REGISTRY` default | `--network_folder` | `--network_folder` |
| `--network-name` | `NETWORK_NAME`, `network_name` | flow task naming defaults | `--network_name` | `--network` |
| `<network-dir>/edge_registry.json` | `EDGE_REGISTRY` | `--edge_registry` | `--edge_registry` | `--edge_registry` |
| `<network-dir>/agents.csv` | `AGENTS_CSV` | no direct use | `--agents_csv`, derived `--n_av`, `--n_hdv` | route artifact manifest input |
| `<network-dir>/od_<name>.txt` | `OD_FILE` | no direct use | no direct use | evaluation defaults through settings |
| `<network-dir>/route_registry.json` | `ROUTE_REGISTRY` | no direct use | no direct use | evaluation defaults through settings |

`--route-artifacts-dir` is an auxiliary artifact-location override, not a
replacement for the network selector. It defaults to `--network-dir` and is
used only for the post-mutation route artifact contract:

- `agents.csv`
- `routes.csv`
- `route.rou.xml`

## Retained Evaluation Outputs

The retained evaluation outputs are:

- `collector_eval_summary.json`
- `collector_eval_episodes.csv` when produced by the evaluator
- `urb_fair_run_manifest.json`
- `best_checkpoint_selection.json` from `scripts/select_best_eval_summary.py`

External metric-generation scripts from neighboring repositories are not part
of this target contract.
