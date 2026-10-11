"""Preserve partial pretraining while adopting the approved Ingolstadt execution profile."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.migrate_mainline_prior import _assert_equal, _read_frozen
from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.workflow.checkpoint import CheckpointPhase, latest_checkpoint_path, load_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


PROFILE = dict(world_model_initialization_batch_size=64,
               world_model_gradient_accumulation_steps=1,
               world_model_pretrain_space_chunk=64,
               world_model_pretrain_time_chunk=65536,
               world_model_pretrain_mlp_chunk=65536,
               world_model_pretrain_precision='bf16',
               world_model_pretrain_compile_blocks=True,
               world_model_pretrain_activation_checkpointing=True)


def target_config(source, output_dir):
    target = replace(source,
        run=replace(source.run, output_dir=Path(output_dir).resolve(), world_model_initialization_epochs=1),
        resources=replace(source.resources, **PROFILE))
    return replace(target, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(target, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))


def validate_runtime_transfer_config(source, target):
    before, after = source.resolved_snapshot(), target.resolved_snapshot()
    for snapshot in (before, after):
        for field in PROFILE:
            snapshot['resources'].pop(field)
        for field in ('output_dir', 'world_model_initialization_epochs'):
            snapshot['run'].pop(field)
    if before != after:
        raise ValueError('runtime continuation may change only the approved execution fields, epoch budget and output directory')
    if any(getattr(target.resources, key) != value for key, value in PROFILE.items()):
        raise ValueError('target must use the approved batch64 Ingolstadt execution profile')
    if (source.run.world_model_initialization_epochs != 5
            or target.run.world_model_initialization_epochs != 1):
        raise ValueError('runtime continuation requires the approved epoch budget change from five to one')


def validate_pretrain(checkpoint, config):
    state, metrics = checkpoint.training_state, checkpoint.training_state['metrics']
    if state['reward_contract'] != 'causal_cumulative_mean_global_v1':
        raise ValueError('runtime continuation requires the current causal global reward contract')
    if checkpoint.phase is not CheckpointPhase.PRETRAIN:
        raise ValueError('runtime continuation requires PRETRAIN phase')
    if (checkpoint.next_unit != metrics.get('world_model_initialization_epochs_completed')
            or not 0 <= checkpoint.next_unit < config.run.world_model_initialization_epochs):
        raise ValueError('epoch budget must exceed the consistent completed-epoch cursor')
    if (state['next_environment_episode'] != 0 or metrics.get('online_episodes_completed', 0) != 0
            or metrics.get('evaluation_episodes_completed', 0) != 0
            or state['replay']['online']['episodes']):
        raise ValueError('runtime continuation requires no online training or evaluation')
    for name in ('actor', 'latent_critic', 'event_critic', 'world_model_online'):
        if state['optimizers'].get(name, {}).get('state'):
            raise ValueError('runtime continuation refuses trained online optimizers')
    offline, validation = state['replay']['offline'], state['replay'].get('codec_validation')
    if offline['epochs_completed'] != checkpoint.next_unit:
        raise ValueError('offline epoch cursor differs from checkpoint')
    if validation is None or not validation.get('manifest_identity'):
        raise ValueError('runtime continuation requires independent codec validation state')
    if 'batch_size' in offline:
        raise ValueError('unexpected batch-size-dependent sampler schema; explicit migration required')
    if config.codec.decoder_allow_unready:
        raise ValueError('runtime continuation preserves strict codec acceptance; unready override forbidden')
    # Continuous decoder updates invalidate the current quality stamp. The
    # initial held-out admission remains valid until the encoder or data change.
    trainer = object.__new__(MainlineTrainer)
    trainer.config = config
    trainer.metrics = dict(metrics)
    trainer.codec_validation_data = SimpleNamespace(
        manifest={'identity': validation['manifest_identity']}, has_samples=True)
    trainer._validate_codec_pretraining_ready()


def validate_runtime_continuation(output_dir, config, input_hash):
    output = Path(output_dir).resolve()
    proof = json.loads((output / 'runtime_migration.json').read_text())
    if (proof['target_output'] != str(output)
            or proof['target_config_fingerprint'] != config.semantic_fingerprint
            or proof['input_hash'] != input_hash):
        raise ValueError('runtime continuation target identity differs')
    for filename, expected in proof['frozen_artifacts'].items():
        if sha256_file(output / filename) != expected:
            raise ValueError(f'runtime continuation {filename} hash differs')
    source_config = load_config(output / 'source_config.toml')
    validate_runtime_transfer_config(source_config, config)
    if source_config.semantic_fingerprint != proof['source_config_fingerprint']:
        raise ValueError('runtime continuation source fingerprint differs')
    source = _read_frozen(output / 'source_checkpoint.pt', source_config, input_hash)
    initial = _read_frozen(output / 'continuation_checkpoint.pt', config, input_hash)
    validate_pretrain(source, source_config)
    validate_pretrain(initial, config)
    if source.next_unit != initial.next_unit:
        raise ValueError('runtime continuation changed epoch cursor')
    _assert_equal(source.training_state, initial.training_state)
    latest = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                             expected_input_hash=input_hash)
    if latest.training_state['metrics']['world_model_initialization_samples_seen'] < proof['samples_seen']:
        raise ValueError('runtime continuation latest samples moved backwards')
    return proof


def migrate(source_dir, output_dir):
    source_dir, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise ValueError('runtime continuation requires a new absent output directory')
    source_config = load_config(source_dir / 'run_config.toml')
    config = target_config(source_config, output)
    validate_runtime_transfer_config(source_config, config)
    output.mkdir(parents=True)
    shutil.copy2(source_dir / 'run_config.toml', output / 'source_config.toml')
    shutil.copy2(latest_checkpoint_path(source_dir), output / 'source_checkpoint.pt')
    raw = torch.load(output / 'source_checkpoint.pt', map_location='cpu', weights_only=True)
    source = _read_frozen(output / 'source_checkpoint.pt', source_config, raw['input_hash'])
    validate_pretrain(source, source_config)
    validate_pretrain(source, config)
    export_run_config(config, output / 'run_config.toml')
    # Only the checkpoint envelope's configuration identity changes. Samplers
    # store consumed start cursors independently of their next batch size.
    raw['config_fingerprint'] = config.semantic_fingerprint
    torch.save(raw, output / 'continuation_checkpoint.pt')
    latest_checkpoint_path(output).parent.mkdir(parents=True)
    shutil.copy2(output / 'continuation_checkpoint.pt', latest_checkpoint_path(output))
    filenames = ['source_config.toml', 'source_checkpoint.pt', 'continuation_checkpoint.pt', 'run_config.toml']
    for filename in ('offline_manifest.json', 'codec_validation_manifest.json'):
        if (source_dir / filename).is_file():
            shutil.copy2(source_dir / filename, output / filename)
            filenames.append(filename)
    for field in ('pretrain_cache_manifest', 'validation_cache_manifest'):
        path = getattr(source_config.data, field)
        if path is not None:
            filename = f'source_{field}.json'
            shutil.copy2(path, output / filename)
            filenames.append(filename)
    metrics, offline = source.training_state['metrics'], source.training_state['replay']['offline']
    proof = dict(schema_version=1, kind='approved_ingo_runtime_pretrain_continuation',
        source_output=str(source_dir), target_output=str(output),
        source_config_fingerprint=source_config.semantic_fingerprint,
        target_config_fingerprint=config.semantic_fingerprint, input_hash=source.input_hash,
        frozen_artifacts={filename: sha256_file(output / filename) for filename in filenames},
        completed_epochs=source.next_unit, target_epochs=1,
        samples_seen=metrics['world_model_initialization_samples_seen'],
        epoch_samples=metrics['world_model_epoch_samples'],
        optimizer_updates=metrics['world_model_initialization_updates'],
        offline_cursor={key: offline[key] for key in ('epochs_completed', 'episode_cursor', 'start_cursor')},
        runtime_profile=PROFILE, source_device=source_config.run.device, target_device=config.run.device,
        preserved='all training_state values unchanged: models, optimizers, RNG, sampler permutations/cursors, counters and decoder admission',
        consequence='Remaining starts are grouped into batches64; historic batch/update counters remain intact. Completed-epoch budget reduced from5 to1.')
    atomic_json(output / 'runtime_migration.json', proof)
    validate_runtime_continuation(output, config, source.input_hash)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    print(json.dumps(migrate(args.source_dir, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
