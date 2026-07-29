# UrbanDreamer


## Quick start

```bash
# 1. Download the bundled network data.
git lfs install
git lfs pull

# 2. Create the reference environment.
conda env create -f environment.yaml
conda activate urban-dreamer

# 3. Run one bundled city profile.
python -m urban_dreamer --config configs/toy_cross_bottleneck.toml
```

The other ready-to-run profiles are:

- `configs/ingolstadt_custom.toml`
- `configs/gretz_armainvilliers.toml`
- `configs/fontenay_en_parisis.toml`

Edit the selected TOML to change its device or training budget. Running the
same command again resumes from `<output_dir>/checkpoints/latest.pt`.

## Install

The reference environment uses Python 3.12 and SUMO 1.25.

```bash
conda env create -f environment.yaml
conda activate urban-dreamer
```

Alternatively:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Make sure `sumo` is available on `PATH`. When SUMO is installed by the
`eclipse-sumo` wheel, set:

```bash
export SUMO_HOME="$(
  python -c 'from pathlib import Path; import sumo; print(Path(sumo.__file__).resolve().parent)'
)"
sumo --version
```

## Bundled networks

Each city config reads its matching prepared bundle from
`networks/<network_name>/`:

```text
networks/<network_name>/
├── <network_name>.net.xml
├── agents.csv
├── od_<network_name>.txt
├── edge_registry.json
├── routes.csv
├── route.rou.xml
├── post_mutation_state.json
├── initial_hdv_state.json
└── hdv_response_predictor_v4.pt
```

The repository includes bundles for Toy, Ingolstadt, Gretz, and Fontenay. The
runtime validates their network, vehicles, OD pairs, four routes per OD,
AV/HDV split, hashes, and HDV predictor before training.

Large city `edge_registry.json` files are tracked with Git LFS.

## Configure

Bundled city configs can run directly. To add another network, copy the
portable example:

```bash
cp configs/mainline.example.toml configs/my_network.toml
```

Edit its four sections:

- `[inputs]`: network name and prepared-data paths;
- `[run]`: output directory, device, seed, and episode counts;
- `[simulation]`: episode horizons, incident settings, and SUMO arguments;
- `[resources]`: world-model batch, sequence, and accumulation sizes.

Relative paths are resolved from the TOML file location. The example therefore
uses `../networks/...` and `../outputs/...`.

## Run

Start the complete pipeline:

```bash
python -m urban_dreamer --config configs/my_network.toml
```

Optional command-line overrides:

```bash
python -m urban_dreamer \
  --config configs/my_network.toml \
  --output-dir outputs/my_network \
  --device cuda:0
```
