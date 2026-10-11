"""Start fresh no-incident online training from the calibrated meso initialization.

The source is the *initialization* checkpoint, never a trained online checkpoint.
Only incident timing and the output, device and offline archive locations may
change. The new corpus supplies independent codec validation before publication.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.migrate_pretrained_meso import (
    calibrate_decoder, install_sparse_codec, model_hashes, nondecoder_models,
    transfer_models, validate_initialization_manifest, validate_target_initialization,
)
from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.checkpoint import (
    CheckpointPhase, _validate_loaded_payload, latest_checkpoint_path,
    load_checkpoint, save_checkpoint,
)
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config
from urban_dreamer.workflow.pipeline import combined_input_hash

MANIFEST_NAME = 'calibrated_noincident_initialization.json'


def validate_noincident_config(source, target):
    """Accept only a fresh runtime location, offline source and incident timing."""
    # Actor/critics are untrained at this zero-round source; entropy is a
    # fingerprinted target objective, independent of decoder validation.
    if source.inputs != target.inputs or source.method != replace(target.method, actor_entropy_mode=source.method.actor_entropy_mode) or source.codec != target.codec:
        raise ValueError('no-incident transfer must preserve inputs, method and codec settings')
    if source.resources != target.resources:
        raise ValueError('no-incident transfer must preserve all resource settings')
    if source.run != replace(target.run, output_dir=source.run.output_dir, device=source.run.device):
        raise ValueError('no-incident transfer may change only run output and device')
    if source.simulation != replace(target.simulation,
            incident_start_seconds=source.simulation.incident_start_seconds,
            incident_end_seconds=source.simulation.incident_end_seconds):
        raise ValueError('no-incident transfer may change only incident start/end times')
    if source.simulation.traffic_model != 'meso' or target.simulation.traffic_model != 'meso':
        raise ValueError('no-incident transfer requires unchanged mesoscopic physics')
    horizon = max(target.simulation.evaluation_horizon_seconds,
                  target.simulation.demand_duration_seconds + target.simulation.training_completion_buffer_seconds)
    if target.simulation.incident_start_seconds <= horizon:
        raise ValueError('incident must start strictly after every configured episode horizon')
    if (target.data.cache_episodes != source.data.cache_episodes
            or target.data.pretrain_cache_manifest is not None
            or target.data.validation_cache_manifest is not None
            or not target.data.archive_roots):
        raise ValueError('no-incident transfer requires fresh archive roots and unchanged cache size')


def _source_initialization(source_dir, input_hash):
    source_dir = Path(source_dir).resolve()
    source_config = load_config(source_dir / 'run_config.toml')
    source_manifest_path = source_dir / 'meso_initialization.json'
    source_manifest = json.loads(source_manifest_path.read_text())
    frozen = source_dir / 'initialization_checkpoint.pt'
    payload = torch.load(frozen, map_location='cpu', weights_only=True)
    source = _validate_loaded_payload(payload=payload, path=frozen,
        expected_config_fingerprint=source_config.semantic_fingerprint,
        expected_input_hash=input_hash)
    pretrain_dir = Path(source_manifest['source_output']).resolve()
    pretrain_config = load_config(pretrain_dir / 'run_config.toml')
    pretrain = load_checkpoint(output_dir=pretrain_dir,
        expected_config_fingerprint=pretrain_config.semantic_fingerprint,
        expected_input_hash=input_hash)
    validate_initialization_manifest(source_manifest, pretrain, pretrain_config,
                                     source_config, input_hash, source_dir)
    validate_target_initialization(source.training_state, pretrain, source_manifest)
    if source.phase is not CheckpointPhase.ONLINE_TRAINING or source.next_unit != 0:
        raise ValueError('source must be the calibrated meso online-start checkpoint')
    if not source.training_state['metrics'].get('codec_ready'):
        raise ValueError('source calibrated codec is not ready')
    return source, source_config, source_manifest_path


def validate_calibrated_noincident_manifest(manifest, target_config, output, input_hash):
    """Validate the complete immutable source/target chain before runner startup."""
    output = Path(output).resolve()
    if manifest.get('schema_version') != 1 or manifest.get('kind') != 'calibrated_meso_to_fresh_noincident_online':
        raise ValueError('invalid no-incident initialization manifest schema')
    source_dir = Path(manifest['source_output']).resolve()
    source, source_config, source_manifest_path = _source_initialization(source_dir, input_hash)
    validate_noincident_config(source_config, target_config)
    expected = {
        'source_output': str(source_dir),
        'target_output': str(output),
        'source_config_fingerprint': source_config.semantic_fingerprint,
        'target_config_fingerprint': target_config.semantic_fingerprint,
        'input_hash': input_hash,
        'source_initialization_sha256': sha256_file(source.path),
        'source_meso_manifest_sha256': sha256_file(source_manifest_path),
        'source_model_hashes': model_hashes(source.training_state['models']),
        'target_initialization_sha256': sha256_file(output / 'initialization_checkpoint.pt'),
        'training_manifest_sha256': sha256_file(output / 'offline_manifest.json'),
        'validation_manifest_sha256': sha256_file(output / 'codec_validation_manifest.json'),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'no-incident initialization {key} identity differs')
    for key, path in (('training_manifest_identity', output / 'offline_manifest.json'),
                      ('validation_manifest_identity', output / 'codec_validation_manifest.json')):
        if manifest.get(key) != json.loads(path.read_text())['identity']:
            raise ValueError(f'no-incident initialization {key} differs')
    import libsumo
    if manifest.get('target_engine_version') != libsumo.__version__:
        raise ValueError('no-incident initialization engine version differs')
    payload = torch.load(output / 'initialization_checkpoint.pt', map_location='cpu', weights_only=True)
    target = _validate_loaded_payload(payload=payload, path=output / 'initialization_checkpoint.pt',
        expected_config_fingerprint=target_config.semantic_fingerprint,
        expected_input_hash=input_hash)
    validate_target_initialization(target.training_state, source, manifest)
    metrics = target.training_state['metrics']
    if (not manifest.get('codec_validation', {}).get('codec_ready')
            or not metrics.get('codec_ready')
            or metrics.get('codec_validation_manifest_identity') != manifest['validation_manifest_identity']
            or metrics.get('codec_validation_updates') != metrics.get('codec_updates')):
        raise ValueError('no-incident held-out codec proof is invalid')
    return target


def migrate(source_dir, output_dir, *, config_path=ROOT / 'configs/ingolstadt_custom.toml',
            archive_root, validation_archive_root=None, device='cpu', threads=4,
            sampler_workers=8, codec_calibration_updates=0,
            gpu_memory_fraction=.08, continue_failed_validation=False):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir:
        raise ValueError('no-incident initialization requires a separate output directory')
    if output_dir.exists() and not continue_failed_validation:
        raise ValueError('no-incident initialization requires a new absent output directory')
    if continue_failed_validation and latest_checkpoint_path(output_dir).exists():
        raise ValueError('cannot reinitialize an output that already has a checkpoint')
    if not 0 <= codec_calibration_updates <= 1000 or threads < 1 or not 0 <= sampler_workers <= 8:
        raise ValueError('invalid bounded calibration, thread or sampler worker count')
    base = load_config(config_path)
    archive_root = Path(archive_root).resolve()
    validation_roots = ((Path(validation_archive_root).resolve(),) if validation_archive_root else ())
    config = replace(base, run=replace(base.run, output_dir=output_dir, device=device),
        data=replace(base.data, archive_roots=(archive_root,), validation_archive_roots=validation_roots,
                     pretrain_cache_manifest=None, validation_cache_manifest=None))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    prepared = validate_prepared_inputs(config)
    input_hash = combined_input_hash(prepared)
    source, source_config, source_manifest_path = _source_initialization(source_dir, input_hash)
    validate_noincident_config(source_config, config)
    source_hash = sha256_file(source.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, torch.device(device))
    topology = load_traffic_topology(prepared)
    trainer = MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology)
    try:
        install_sparse_codec(trainer)
        transfer_models(trainer, source)
        trainer.metrics['imagination_workers'] = sampler_workers
        trainer.metrics['source_pretraining_provenance'] = source.training_state['metrics'].get(
            'source_pretraining_provenance', {})
        trainer.validate_pretraining_data()
        codec = trainer._validate_codec()
        report = dict(schema_version=1, kind='calibrated_meso_to_fresh_noincident_online',
            source_output=str(source_dir), target_output=str(output_dir),
            source_config_fingerprint=source_config.semantic_fingerprint,
            target_config_fingerprint=config.semantic_fingerprint, input_hash=input_hash,
            source_initialization_sha256=source_hash,
            source_meso_manifest_sha256=sha256_file(source_manifest_path),
            source_model_hashes=model_hashes(source.training_state['models']),
            training_manifest_identity=trainer.offline_data.manifest['identity'],
            validation_manifest_identity=trainer.codec_validation_data.manifest['identity'],
            training_manifest_sha256=sha256_file(output_dir / 'offline_manifest.json'),
            validation_manifest_sha256=sha256_file(output_dir / 'codec_validation_manifest.json'),
            codec_initial_validation=codec, codec_validation=codec,
            target_world_model_initialization_epochs=0,
            online_episode_budget=config.run.online_training_episodes,
            sampler_workers=sampler_workers,
            preserved='all calibrated source neural weights unless bounded decoder-only calibration is required',
            reset='optimizers, online replay, RNG, online counters and evaluations')
        if not trainer.codec_ready() and codec_calibration_updates:
            report['decoder_calibration'] = calibrate_decoder(trainer,
                max_updates=codec_calibration_updates, output=output_dir,
                source_models=source.training_state['models'])
            report['codec_validation'] = trainer._validate_codec()
        trainer.validate_codec_ready()
        trainer.world_model.flow_projector.requires_grad_(False)
        state = trainer.state_dict()
        report['model_hashes'] = model_hashes(state['models'])
        report['frozen_nondecoder_hashes'] = model_hashes(nondecoder_models(source.training_state['models']))
        validate_target_initialization(state, source, report)
        import libsumo
        report['target_engine_version'] = libsumo.__version__
        checkpoint = save_checkpoint(output_dir=output_dir,
            config_fingerprint=config.semantic_fingerprint, input_hash=input_hash,
            phase=CheckpointPhase.ONLINE_TRAINING, next_unit=0, training_state=state)
        shutil.copy2(checkpoint, output_dir / 'initialization_checkpoint.pt')
        report['target_initialization_sha256'] = sha256_file(checkpoint)
        export_run_config(config, output_dir / 'run_config.toml')
        atomic_json(output_dir / MANIFEST_NAME, report)
        validate_calibrated_noincident_manifest(report, config, output_dir, input_hash)
        if sha256_file(source.path) != source_hash:
            raise ValueError('calibrated source initialization changed during migration')
        return report
    finally:
        trainer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ingolstadt_custom.toml')
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--validation-archive-root', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--sampler-workers', type=int, default=8)
    parser.add_argument('--codec-calibration-updates', type=int, default=0)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.08)
    parser.add_argument('--continue-failed-validation', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(migrate(args.source_dir, args.output_dir, config_path=args.config,
        archive_root=args.archive_root, validation_archive_root=args.validation_archive_root,
        device=args.device, threads=args.threads, sampler_workers=args.sampler_workers,
        codec_calibration_updates=args.codec_calibration_updates,
        gpu_memory_fraction=args.gpu_memory_fraction,
        continue_failed_validation=args.continue_failed_validation), indent=2), flush=True)


if __name__ == '__main__':
    main()
