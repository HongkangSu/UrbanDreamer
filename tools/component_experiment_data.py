"""Frozen inputs and shared loading for the component experiment campaign.

Existing validation is development data. Final tests must use new demand groups;
this module never relabels a previously inspected validation set as final test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from urban_dreamer.learning.offline_data import _identity
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.models.world_model import UrbanDreamerWorldModel
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.checkpoint import load_checkpoint
from urban_dreamer.workflow.config import load_config, legacy_semantic_fingerprint
from urban_dreamer.workflow.pipeline import combined_input_hash


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def select_manifest(source, *, train_groups, validation_groups, seed):
    """Select entire demand groups without crossing original partitions."""
    if source.get('status') != 'complete' or source.get('identity') != _identity(
            {key: value for key, value in source.items() if key != 'identity'}):
        raise ValueError('source manifest must be complete with a valid identity')
    groups = {split: {entry['demand_group'] for entry in source['entries']
                      if entry['split'] == split} for split in ('train', 'validation')}
    if groups['train'] & groups['validation']:
        raise ValueError('source demand groups overlap')
    selected = {}
    for split, count in [('train', train_groups), ('validation', validation_groups)]:
        if count < 1 or count > len(groups[split]):
            raise ValueError(f'invalid {split} group budget {count}/{len(groups[split])}')
        ranked = sorted(groups[split], key=lambda group: hashlib.sha256(
            f'{seed}:{split}:{group}'.encode()).hexdigest())
        selected[split] = set(ranked[:count])
    entries = [entry for entry in source['entries']
               if entry['demand_group'] in selected[entry['split']]]
    paths = {entry['path'] for entry in entries}
    manifest = dict(source, entries=entries,
                    sources=[row for row in source['sources'] if row['path'] in paths],
                    duplicate_files=0)
    manifest.pop('identity')
    manifest['component_selection'] = dict(parent_identity=source['identity'], seed=seed,
        role='development_only', groups={k: sorted(v) for k, v in selected.items()},
        preserves_all_methods_per_selected_demand=True)
    manifest['identity'] = _identity(manifest)
    return manifest


def load_frozen(experiment_dir, *, device='cpu', fresh_dynamics_seed=None):
    directory = Path(experiment_dir)
    protocol = json.loads((directory / 'protocol.json').read_text())
    snapshot = directory / 'snapshot'
    checkpoint = snapshot / 'checkpoints/latest.pt'
    if sha256(checkpoint) != protocol['checkpoint_sha256']:
        raise ValueError('frozen checkpoint changed')
    config = load_config(snapshot / 'run_config.toml')
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    raw = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # The frozen pretrain predates the paper-reward migration. Verify its exact
    # supported historical contract; import only traffic predictions, no actor,
    # critic, reward targets or optimizer state under a new reward definition.
    compatible = {config.semantic_fingerprint, legacy_semantic_fingerprint(config)}
    if raw['config_fingerprint'] not in compatible:
        raise ValueError('checkpoint matches neither current nor supported historical configuration')
    loaded = load_checkpoint(output_dir=snapshot,
        expected_config_fingerprint=raw['config_fingerprint'],
        expected_input_hash=combined_input_hash(prepared))
    if fresh_dynamics_seed is not None:
        torch.manual_seed(fresh_dynamics_seed)
    model = UrbanDreamerWorldModel(
        adjacency=torch.as_tensor(topology.network.adjacency, dtype=torch.float32),
        static_features=torch.as_tensor(topology.network.static_features, dtype=torch.float32),
        low_frequency_rank=config.method.low_frequency_rank,
        prior_filter=getattr(config.method, 'prior_filter', 'hard'),
        prior_filter_strength=getattr(config.method, 'prior_filter_strength', 10.0),
        prior_action_conditioned=getattr(config.method, 'prior_action_conditioned', True),
        prior_filter_input=getattr(config.method, 'prior_filter_input', False), prior_usage=getattr(config.method, 'prior_usage', 'source_reference'),
        prior_noise_scale=config.method.prior_noise_scale, event_protocol=True,
        codec_speed_positive=config.codec.speed_positive)
    state = loaded.training_state['models']['world_model']
    if fresh_dynamics_seed is None:
        model.load_state_dict(state, strict=True)
    else:
        model.flow_projector.load_state_dict({key.removeprefix('flow_projector.'): value
            for key, value in state.items() if key.startswith('flow_projector.')}, strict=True)
    model.flow_projector.requires_grad_(False)
    model.active_vehicle_fraction_readout.requires_grad_(False)
    model.to(device)
    return config, prepared, topology, model, protocol


def dataset(experiment_dir, config, prepared, topology, *, split, horizon):
    return CompactPretrainDataset(config, prepared, topology, horizon,
        cache_manifest_path=Path(experiment_dir) / 'development_manifest.json', split=split)


def prepare(args):
    output = args.output.resolve()
    if (output / 'protocol.json').exists():
        raise ValueError('campaign already prepared; never silently replace its inputs')
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / 'snapshot'
    (snapshot / 'checkpoints').mkdir(parents=True, exist_ok=True)
    source = json.loads(args.manifest.read_text())
    selected = select_manifest(source, train_groups=args.train_groups,
        validation_groups=args.validation_groups, seed=args.seed)
    shutil.copyfile(args.run_dir / 'run_config.toml', snapshot / 'run_config.toml')
    # Opening one inode also handles a producer's atomic latest replacement.
    with (args.run_dir / 'checkpoints/latest.pt').open('rb') as reader, \
            (snapshot / 'checkpoints/latest.pt').open('wb') as writer:
        shutil.copyfileobj(reader, writer)
    atomic_json(output / 'development_manifest.json', selected)
    protocol = dict(schema_version=1, baseline_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        checkpoint_source=str(args.run_dir.resolve()),
        checkpoint_sha256=sha256(snapshot / 'checkpoints/latest.pt'),
        config_sha256=sha256(snapshot / 'run_config.toml'),
        source_manifest=str(args.manifest.resolve()), source_manifest_identity=source['identity'],
        development_manifest_identity=selected['identity'],
        selection_seed=args.seed, role='development_not_final_test',
        counts={split: dict(groups=len(selected['component_selection']['groups'][split]),
            episodes=sum(e['split'] == split for e in selected['entries']),
            windows=sum(e['windows'] for e in selected['entries'] if e['split'] == split))
            for split in ('train', 'validation')})
    atomic_json(output / 'protocol.json', protocol)
    print(json.dumps(protocol, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-groups', type=int, default=128)
    parser.add_argument('--validation-groups', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261011)
    prepare(parser.parse_args())
