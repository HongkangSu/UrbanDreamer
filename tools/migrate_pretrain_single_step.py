"""Explicit partial-pretrain migration to single-step FM and Euler100 inference."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.migrate_mainline_prior import _assert_equal, _read_frozen
from tools.migrate_pretrain_compile import _validate_pretrain
from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.learning.offline_data import _identity
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.checkpoint import CheckpointPhase, latest_checkpoint_path, load_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config
from urban_dreamer.workflow.pipeline import combined_input_hash


def validate_single_step_transfer_config(source, target):
    before, after = source.resolved_snapshot(), target.resolved_snapshot()
    for snapshot in (before, after):
        for field in ('fm_training_mode', 'fm_solver', 'fm_steps'):
            snapshot['method'].pop(field)
        snapshot['resources'].pop('world_model_initialization_sequence_length')
        snapshot['resources'].pop('world_model_pretrain_compile_blocks')
        for field in ('output_dir', 'device'):
            snapshot['run'].pop(field)
    if before != after:
        raise ValueError('single-step migration may change only FM mode/solver/steps, sequence length, compile flag and location')
    if (source.method.fm_training_mode != 'legacy_independent'
            or source.resources.world_model_initialization_sequence_length != 3
            or source.method.fm_solver != 'euler' or source.method.fm_steps != 2):
        raise ValueError('single-step migration requires legacy L3 Euler2 source')
    if not target.resources.world_model_pretrain_compile_blocks:
        raise ValueError('single-step migration requires compiled blocks')
    if (target.method.fm_training_mode != 'single_step'
            or target.resources.world_model_initialization_sequence_length != 1
            or target.method.fm_solver != 'euler' or target.method.fm_steps != 100):
        raise ValueError('single-step migration requires single_step L1 Euler100 target')


def migrate_training_state(state, source_config, training, validation):
    """Rebind checked datasets and reset coverage, preserving learned state exactly."""
    if validation is None:
        raise ValueError('migration requires independent compact validation data')
    for key, dataset in (('offline', training), ('codec_validation', validation)):
        manifest = copy.deepcopy(dataset.manifest)
        manifest['contract']['sequence_length'] = source_config.resources.world_model_initialization_sequence_length
        expected = _identity({name: value for name, value in manifest.items() if name != 'identity'})
        if state['replay'][key]['manifest_identity'] != expected:
            raise ValueError(f'{key} manifest changed beyond sequence length')
    migrated = copy.deepcopy(state)
    migrated['replay']['offline'] = training.state_dict()
    migrated['replay']['codec_validation'] = validation.state_dict()
    metrics = migrated['metrics']
    old = {key: value for key, value in metrics.items()
           if key.startswith(('world_model_initialization_', 'world_model_epoch_', 'pretrain_progress_'))}
    for key in list(metrics):
        if key.startswith(('world_model_epoch_', 'pretrain_progress_')) or key in (
                'world_model_initialization_epochs_completed', 'world_model_initialization_samples_seen',
                'world_model_initialization_batches_completed'):
            metrics[key] = 0.
        if key.startswith(('codec_validation_', 'codec_hybrid_validation_')):
            del metrics[key]
    metrics['codec_ready'] = 0.
    metrics['single_step_migration_source_counters'] = old
    metrics['single_step_migration_codec_revalidation_required'] = True
    return migrated


def validate_optimizer_state(trainer, state):
    """Reject nonfinite or shape-corrupt Adam moments before accepting continuation."""
    if set(state['optimizers']) != set(trainer.optimizers):
        raise ValueError('optimizer names differ from current trainer')
    if (state['metrics'].get('world_model_initialization_updates', 0) > 0
            and not state['optimizers']['world_model_initial']['state']):
        raise ValueError('world_model_initial optimizer lost warm state')
    for name, optimizer in trainer.optimizers.items():
        saved = state['optimizers'][name]
        groups = saved['param_groups']
        if len(groups) != len(optimizer.param_groups):
            raise ValueError(f'{name} optimizer group count differs')
        known = set()
        for group, current in zip(groups, optimizer.param_groups):
            if len(group['params']) != len(current['params']):
                raise ValueError(f'{name} optimizer parameter count differs')
            for identifier, parameter in zip(group['params'], current['params']):
                if identifier in known:
                    raise ValueError(f'{name} optimizer repeats parameter identity')
                known.add(identifier)
                moments = saved['state'].get(identifier, {})
                if moments and not {'step', 'exp_avg', 'exp_avg_sq'} <= set(moments):
                    raise ValueError(f'{name} optimizer has incomplete Adam state')
                for key, value in moments.items():
                    if isinstance(value, torch.Tensor):
                        if not torch.isfinite(value).all():
                            raise ValueError(f'{name} optimizer contains nonfinite {key}')
                        if key == 'step':
                            if value.numel() != 1 or value.item() < 0:
                                raise ValueError(f'{name} optimizer has invalid step')
                        elif value.shape != parameter.shape:
                            raise ValueError(f'{name} optimizer moment shape differs')
                        if key in ('exp_avg_sq', 'max_exp_avg_sq') and (value < 0).any():
                            raise ValueError(f'{name} optimizer variance is negative')
        if set(saved['state']) - known:
            raise ValueError(f'{name} optimizer has unknown parameter state')
    for model in state['models'].values():
        for value in model.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError('checkpoint model contains nonfinite values')


def _make_trainer(config):
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    cpu_config = replace(config, run=replace(config.run, device='cpu'))
    trainer = MainlineTrainer(config=cpu_config, prepared_inputs=prepared, topology=topology)
    return trainer, combined_input_hash(prepared)


def validate_single_step_continuation(output_dir, config, input_hash):
    output = Path(output_dir).resolve()
    manifest = json.loads((output / 'single_step_migration.json').read_text())
    if (manifest['target_output'] != str(output) or manifest['input_hash'] != input_hash
            or manifest['target_config_fingerprint'] != config.semantic_fingerprint):
        raise ValueError('single-step continuation target identity differs')
    for filename, expected in manifest['frozen_artifacts'].items():
        if sha256_file(output / filename) != expected:
            raise ValueError(f'single-step continuation {filename} hash differs')
    old_config = load_config(output / 'source_config.toml')
    validate_single_step_transfer_config(old_config, config)
    source = _read_frozen(output / 'source_checkpoint.pt', old_config, input_hash)
    _validate_pretrain(source, old_config)
    initial = _read_frozen(output / 'continuation_checkpoint.pt', config, input_hash)
    if initial.phase is not CheckpointPhase.PRETRAIN or initial.next_unit != 0:
        raise ValueError('single-step continuation must restart PRETRAIN coverage at epoch zero')
    for key in ('models', 'optimizers', 'rng', 'reward_contract', 'next_environment_episode'):
        _assert_equal(source.training_state[key], initial.training_state[key])
    _assert_equal(source.training_state['replay']['online'], initial.training_state['replay']['online'])
    metrics = initial.training_state['metrics']
    if (metrics['codec_ready'] or not metrics['single_step_migration_codec_revalidation_required']
            or metrics['world_model_initialization_samples_seen'] != 0
            or metrics['world_model_initialization_epochs_completed'] != 0):
        raise ValueError('single-step continuation has invalid coverage or codec proof')
    for key in ('world_model_updates', 'world_model_initialization_updates'):
        if metrics[key] != source.training_state['metrics'][key]:
            raise ValueError('single-step continuation changed lifetime updates')
    latest = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                             expected_input_hash=input_hash)
    if latest.training_state['metrics']['world_model_updates'] < metrics['world_model_updates']:
        raise ValueError('single-step continuation latest updates moved backwards')
    return manifest


def migrate(source_dir, output_dir, *, device=None):
    source_dir, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise ValueError('single-step migration requires a new absent output directory')
    old_config = load_config(source_dir / 'run_config.toml')
    config = replace(old_config,
        run=replace(old_config.run, output_dir=output, device=device or old_config.run.device),
        resources=replace(old_config.resources, world_model_initialization_sequence_length=1,
                          world_model_pretrain_compile_blocks=True),
        method=replace(old_config.method, fm_training_mode='single_step', fm_solver='euler', fm_steps=100))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    validate_single_step_transfer_config(old_config, config)
    if config.data.pretrain_cache_manifest is None:
        raise ValueError('single-step migration requires the verified compact cache')
    output.mkdir(parents=True)
    shutil.copy2(source_dir / 'run_config.toml', output / 'source_config.toml')
    shutil.copy2(latest_checkpoint_path(source_dir), output / 'source_checkpoint.pt')
    raw = torch.load(output / 'source_checkpoint.pt', map_location='cpu', weights_only=True)
    source = _read_frozen(output / 'source_checkpoint.pt', old_config, raw['input_hash'])
    _validate_pretrain(source, old_config)
    trainer, input_hash = _make_trainer(config)
    if input_hash != source.input_hash:
        raise ValueError('single-step migration prepared input hash differs')
    state = migrate_training_state(source.training_state, old_config,
                                   trainer.offline_data, trainer.codec_validation_data)
    validate_optimizer_state(trainer, state)
    trainer.load_state_dict(state)
    export_run_config(config, output / 'run_config.toml')
    raw.update(config_fingerprint=config.semantic_fingerprint, phase=CheckpointPhase.PRETRAIN.value,
               next_unit=0, training_state=state)
    torch.save(raw, output / 'continuation_checkpoint.pt')
    latest_checkpoint_path(output).parent.mkdir(parents=True)
    shutil.copy2(output / 'continuation_checkpoint.pt', latest_checkpoint_path(output))
    filenames = ['source_config.toml', 'source_checkpoint.pt', 'run_config.toml', 'continuation_checkpoint.pt']
    manifest = dict(schema_version=1, kind='single_step_L1_Euler100_pretrain_migration',
        source_output=str(source_dir), target_output=str(output), input_hash=input_hash,
        source_config_fingerprint=old_config.semantic_fingerprint,
        target_config_fingerprint=config.semantic_fingerprint,
        frozen_artifacts={name: sha256_file(output / name) for name in filenames},
        source_completed_epochs=source.next_unit,
        source_progress=state['metrics']['single_step_migration_source_counters'],
        source_sampler=source.training_state['replay']['offline'],
        target_sampler=state['replay']['offline'],
        target_validation_sampler=state['replay']['codec_validation'],
        target_coverage_epochs=config.run.world_model_initialization_epochs,
        codec_revalidation_required=True,
        preserved='model/optimizer tensors, lifetime updates, RNG, empty online replay',
        reset='L1 coverage epochs/samples/batches and sampler orders/cursors; codec validation proof')
    atomic_json(output / 'single_step_migration.json', manifest)
    validate_single_step_continuation(output, config, input_hash)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device')
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(migrate(args.source_dir, args.output_dir, device=args.device), indent=2))


if __name__ == '__main__':
    main()
