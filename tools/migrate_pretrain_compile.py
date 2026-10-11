"""Lossless, explicit continuation of a partial pretrain with compiled blocks."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.migrate_mainline_prior import _assert_equal, _read_frozen
from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.workflow.checkpoint import CheckpointPhase, latest_checkpoint_path, load_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config

FIELD = 'world_model_pretrain_compile_blocks'


def validate_compile_transfer_config(source, target):
    before, after = source.resolved_snapshot(), target.resolved_snapshot()
    for snapshot in (before, after):
        snapshot['resources'].pop(FIELD)
        for field in ('output_dir', 'device'):
            snapshot['run'].pop(field)
    if before != after:
        raise ValueError('compile continuation may change only compile flag, output directory and device')
    if getattr(source.resources, FIELD) or not getattr(target.resources, FIELD):
        raise ValueError('compile continuation requires disabled to enabled compilation')


def _validate_pretrain(checkpoint, config):
    state = checkpoint.training_state
    metrics = state['metrics']
    if checkpoint.phase is not CheckpointPhase.PRETRAIN:
        raise ValueError('compile continuation requires PRETRAIN phase')
    if (checkpoint.next_unit != metrics.get('world_model_initialization_epochs_completed')
            or not 0 <= checkpoint.next_unit < config.run.world_model_initialization_epochs):
        raise ValueError('compile continuation requires a consistent unfinished epoch cursor')
    if not metrics.get('codec_ready'):
        raise ValueError('compile continuation requires a ready codec')
    if (state['next_environment_episode'] != 0 or metrics.get('online_episodes_completed', 0) != 0
            or metrics.get('evaluation_episodes_completed', 0) != 0
            or state['replay']['online']['episodes']):
        raise ValueError('compile continuation requires no online training or evaluation')
    for name in ('actor', 'latent_critic', 'event_critic', 'world_model_online'):
        if state['optimizers'].get(name, {}).get('state'):
            raise ValueError('compile continuation refuses trained online optimizers')
    offline = state['replay']['offline']
    if offline['epochs_completed'] != checkpoint.next_unit:
        raise ValueError('offline epoch cursor differs from checkpoint')
    if state['replay'].get('codec_validation') is None:
        raise ValueError('compile continuation requires independent codec validation state')


def validate_compile_continuation(output_dir, config, input_hash):
    output = Path(output_dir).resolve()
    manifest = json.loads((output / 'compile_migration.json').read_text())
    if (manifest['target_output'] != str(output)
            or manifest['target_config_fingerprint'] != config.semantic_fingerprint
            or manifest['input_hash'] != input_hash):
        raise ValueError('compile continuation target identity differs')
    for filename, expected in manifest['frozen_artifacts'].items():
        if sha256_file(output / filename) != expected:
            raise ValueError(f'compile continuation {filename} hash differs')
    source_config = load_config(output / 'source_config.toml')
    validate_compile_transfer_config(source_config, config)
    if source_config.semantic_fingerprint != manifest['source_config_fingerprint']:
        raise ValueError('compile continuation source fingerprint differs')
    source = _read_frozen(output / 'source_checkpoint.pt', source_config, input_hash)
    initial = _read_frozen(output / 'continuation_checkpoint.pt', config, input_hash)
    _validate_pretrain(source, source_config)
    _validate_pretrain(initial, config)
    if source.next_unit != initial.next_unit:
        raise ValueError('compile continuation changed epoch cursor')
    _assert_equal(source.training_state, initial.training_state)
    latest = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                             expected_input_hash=input_hash)
    if latest.training_state['metrics']['world_model_initialization_samples_seen'] < manifest['samples_seen']:
        raise ValueError('compile continuation latest samples moved backwards')
    return manifest


def migrate(source_dir, output_dir, *, device=None):
    source_dir, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise ValueError('compile continuation requires a new absent output directory')
    source_config = load_config(source_dir / 'run_config.toml')
    config = replace(source_config,
        run=replace(source_config.run, output_dir=output, device=device or source_config.run.device),
        resources=replace(source_config.resources, **{FIELD: True}))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    validate_compile_transfer_config(source_config, config)
    output.mkdir(parents=True)
    shutil.copy2(source_dir / 'run_config.toml', output / 'source_config.toml')
    # Checkpoints are atomically replaced: this copy freezes one complete update.
    shutil.copy2(latest_checkpoint_path(source_dir), output / 'source_checkpoint.pt')
    raw = torch.load(output / 'source_checkpoint.pt', map_location='cpu', weights_only=True)
    source = _read_frozen(output / 'source_checkpoint.pt', source_config, raw['input_hash'])
    _validate_pretrain(source, source_config)
    export_run_config(config, output / 'run_config.toml')
    raw['config_fingerprint'] = config.semantic_fingerprint
    torch.save(raw, output / 'continuation_checkpoint.pt')
    latest_checkpoint_path(output).parent.mkdir(parents=True)
    shutil.copy2(output / 'continuation_checkpoint.pt', latest_checkpoint_path(output))
    filenames = ['source_config.toml', 'source_checkpoint.pt', 'continuation_checkpoint.pt', 'run_config.toml']
    for filename in ('offline_manifest.json', 'codec_validation_manifest.json'):
        if (source_dir / filename).is_file():
            shutil.copy2(source_dir / filename, output / filename)
            filenames.append(filename)
    state = source.training_state
    manifest = dict(schema_version=1, kind='compile_blocks_pretrain_continuation',
        source_output=str(source_dir), target_output=str(output),
        source_config_fingerprint=source_config.semantic_fingerprint,
        target_config_fingerprint=config.semantic_fingerprint, input_hash=source.input_hash,
        frozen_artifacts={filename: sha256_file(output / filename) for filename in filenames},
        completed_epochs=source.next_unit,
        samples_seen=state['metrics']['world_model_initialization_samples_seen'],
        epoch_samples=state['metrics']['world_model_epoch_samples'],
        optimizer_updates=state['metrics']['world_model_initialization_updates'],
        offline_cursor={key: state['replay']['offline'][key] for key in
                        ('epochs_completed', 'episode_cursor', 'start_cursor')},
        hardware=dict(host=platform.node(), torch_version=str(torch.__version__),
                      source_device=source_config.run.device, target_device=config.run.device),
        preserved='all model/optimizer tensors, RNG, sampler orders/cursors, validation state and counters')
    atomic_json(output / 'compile_migration.json', manifest)
    validate_compile_continuation(output, config, source.input_hash)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device')
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(migrate(args.source_dir, args.output_dir, device=args.device), indent=2))


if __name__ == '__main__':
    main()
