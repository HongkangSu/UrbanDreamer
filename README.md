# UrbanDreamer

UrbanDreamer is the slim URB-fair training mainline extracted from the
`dreamer4` research workspace. It keeps the executable four-stage path and the
minimal compatibility package needed by that path:

1. collect the URB-fair post-mutation bundle;
2. train the world model on collected flow data;
3. train the imagined Dreamer CTDE policy;
4. run SUMO evaluation probes and select the best checkpoint summary.

## Installation

The reproducible environment targets Linux x86_64, CPython 3.12, and an
NVIDIA driver capable of running the CUDA runtime installed with PyTorch.
Create it in one command from the repository root:

```bash
conda env create -f environment.yaml
```

Then activate it and expose the pip-installed SUMO runtime:

```bash
conda activate URB
export SUMO_HOME="$CONDA_PREFIX"
```

For an existing Python 3.12 virtual environment, install the same locked stack
directly:

```bash
python -m pip install -r requirements.txt
export SUMO_HOME="$(python -c 'import sys; print(sys.prefix)')"
```

Verify the GPU, SUMO binary, and Python imports before launching a long run:

```bash
python -c "import torch, routerl, traci; print(torch.__version__, torch.cuda.is_available())"
"$SUMO_HOME/bin/sumo" --version
python -m pytest tests/test_three_city_online_wm_full_pipeline.py -q
```

The public network-switching contract is:

```bash
python scripts/run_urb_fair_full_cycle.py \
  --network-dir /path/to/network \
  --network-name <name>
```

`--network-dir` points at the selected SUMO network assets. `--network-name`
is forwarded to RouteRL/SUMO as the logical network name. If the post-mutation
route artifacts live outside the network directory, pass:

```bash
  --route-artifacts-dir /path/to/artifacts
```

That artifact directory must contain `agents.csv`, `routes.csv`, and
`route.rou.xml`. When omitted, `--route-artifacts-dir` defaults to
`--network-dir`.

## What Remains

The retained code is intentionally centered on URB-fair:

- `scripts/run_urb_fair_full_cycle.py` is the canonical full-cycle entrypoint.
- `scripts/collect_urb_fair.py` regenerates the post-mutation collection.
- `dreamer4/train_dynamics.py` trains the world model.
- `dreamer4/urban_routing/train_marl.py` trains the Dreamer CTDE policy.
- `scripts/eval_urb_fair.py` wraps SUMO evaluation and writes
  `urb_fair_run_manifest.json` alongside the collector evaluation summaries.
- `scripts/select_best_eval_summary.py` chooses the best evaluated checkpoint.

The target package keeps the `dreamer4/` import layout for compatibility during
this migration. That preserves the current training behavior without a risky
package rename inside the same change.

## What Was Dropped

UrbanDreamer does not expose the old fixed-route evaluation wrapper, sweep
shells, ablation scripts, or legacy public training-family switches. Evaluation
truth is the target-local eval manifest and collector summaries, not an
external metrics package from a neighboring repository.

## Dry Run

Use `--dry-run` to inspect generated settings and exact commands without
starting training:

```bash
python scripts/run_urb_fair_full_cycle.py \
  --network-dir /path/to/network \
  --network-name <name> \
  --dry-run
```

The runner writes generated `.env` files under `.urbandreamer/generated/<name>/`
and prints a JSON command plan.

## Test

```bash
python -m pytest tests -q
```

Some end-to-end training commands require the local SUMO/RouteRL environment,
network artifacts, and checkpoints. The included tests focus on import
contracts, command construction, and retained URB-fair protocol behavior.
