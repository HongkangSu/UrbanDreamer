"""Explicit neural-weight transfer from completed micro pretraining to fresh meso online training.

This is an initialization, not a strict cross-physics checkpoint resume. Codec
readiness is remeasured on independent meso demands; no dynamics pretraining runs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import copy

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, sha256_file, validate_pretrained_source
from tools.train_codec_ready import SparseGraphCodec
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.learning.codec_training import codec_train_step, make_codec_optimizer
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.checkpoint import CheckpointPhase, latest_checkpoint_path, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config
from urban_dreamer.workflow.pipeline import combined_input_hash

MODELS = ('world_model', 'actor', 'latent_critic', 'event_critic')
DECODER_PREFIX = 'flow_projector.decoder_'


def model_hashes(models):
    result = {}
    for name, state in models.items():
        digest = hashlib.sha256()
        for key, value in sorted(state.items()):
            value = value.detach().cpu().contiguous()
            digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def nondecoder_models(models):
    return {name: {key: value for key, value in state.items()
                   if name != 'world_model' or not key.startswith(DECODER_PREFIX)}
            for name, state in models.items()}


def calibrate_decoder(trainer, *, max_updates, output, source_models):
    """Calibrate physical decoding without changing learned latent representations."""
    if not 0 < max_updates <= 1000:
        raise ValueError('decoder calibration must be explicitly bounded to at most 1000 updates')
    before_hashes = model_hashes(nondecoder_models(source_models))
    original_rng = copy.deepcopy(trainer.state_dict()['rng'])
    flags = [(parameter, parameter.requires_grad) for name in MODELS for parameter in getattr(trainer, name).parameters()]
    for parameter, _ in flags:
        parameter.requires_grad_(False)
    for name, parameter in trainer.world_model.flow_projector.named_parameters():
        if name.startswith('decoder_'):
            parameter.requires_grad_(True)
    optimizer = make_codec_optimizer(trainer.world_model, lr=trainer.config.codec.learning_rate)
    updates, history, started = 0, [], time.monotonic()
    try:
        while updates < max_updates and not trainer.codec_ready():
            hybrid = iter(trainer._transition_codec_batches(trainer.offline_data, shuffle=True))
            for ordinal, raw in enumerate(trainer._codec_batches(trainer.offline_data, shuffle=True)):
                batches = [raw]
                extra = next(hybrid, None) if ordinal % 3 == 0 else None
                if extra is not None:
                    batches.append(extra)
                for flow, following in batches:
                    loss = codec_train_step(trainer.world_model, flow, following, trainer.network,
                                            trainer.config.method, optimizer)
                    updates += 1
                    trainer.metrics['codec_calibration_updates'] = updates
                    if updates % 100 == 0 or updates == max_updates:
                        codec = trainer._validate_codec()
                        if model_hashes(nondecoder_models(trainer.state_dict()['models'])) != before_hashes:
                            raise ValueError('decoder calibration changed a frozen neural tensor')
                        row = dict(updates=updates, seconds=time.monotonic() - started,
                                   codec_loss=float(loss['codec_loss']), codec_validation=codec,
                                   frozen_nondecoder_hashes=before_hashes)
                        history.append(row)
                        atomic_json(Path(output) / 'decoder_calibration_progress.json', dict(
                            max_updates=max_updates, updates=updates, history=history))
                        torch.save(dict(models=trainer.state_dict()['models'], updates=updates,
                            optimizer=optimizer.state_dict()), Path(output) / 'decoder_calibration_latest.pt')
                        print(json.dumps(dict(phase='decoder_calibration_gate', **row)), flush=True)
                    if trainer.codec_ready() or updates == max_updates:
                        break
                if trainer.codec_ready() or updates == max_updates:
                    break
        return dict(updates=updates, max_updates=max_updates, history=history,
                    frozen_nondecoder_hashes=before_hashes, seconds=time.monotonic() - started)
    finally:
        for parameter, enabled in flags:
            parameter.requires_grad_(enabled)
        # Calibration is a separate initialization operation. Online RNG and
        # every trainer optimizer start from the fresh target seed/state.
        state = trainer.state_dict()
        state['rng'] = original_rng
        trainer.load_state_dict(state)


def validate_transfer_config(source, target):
    """Allow physics/backend/data/resource transfer, retain input and learning objectives."""
    # Fresh actor initialization may choose a new entropy objective without
    # changing the pretrained world-model objective or any source model tensor.
    if source.inputs != target.inputs or source.method != replace(target.method, actor_entropy_mode=source.method.actor_entropy_mode) or source.codec != target.codec:
        raise ValueError('meso initialization must preserve inputs, method and codec objective/thresholds')
    if source.simulation.traffic_model != 'micro' or target.simulation.traffic_model != 'meso':
        raise ValueError('initialization requires explicit micro to meso physics transfer')
    before, after = source.resolved_snapshot()['simulation'], target.resolved_snapshot()['simulation']
    for values in (before, after):
        for key in tuple(values):
            if key in ('backend', 'traffic_model') or key.startswith('meso_'):
                values.pop(key)
    if before != after or target.simulation.backend != 'libsumo':
        raise ValueError('only traffic physics and traci/libsumo backend may change simulation settings')
    if source.run.seed != target.run.seed:
        raise ValueError('meso initialization must retain the configured seed')
    if target.data.pretrain_cache_manifest or target.data.validation_cache_manifest:
        raise ValueError('meso initialization cannot inherit micro pretraining caches')
    if source.resources.online_world_model_batch_size != target.resources.online_world_model_batch_size:
        raise ValueError('meso initialization must preserve the online world-model batch size')
    if source.resources.world_model_initialization_sequence_length != target.resources.world_model_initialization_sequence_length:
        raise ValueError('meso initialization must preserve sequence input shape')


def install_sparse_codec(trainer):
    trainer.world_model.flow_projector.__class__ = SparseGraphCodec
    trainer.world_model.flow_projector.register_buffer('sparse_adjacency',
        trainer._tensor(trainer.topology.network.adjacency).to_sparse().coalesce(), persistent=False)


def transfer_models(trainer, source):
    if set(source.training_state['models']) != set(MODELS):
        raise ValueError('source must contain world model, actor, macro V and event Q')
    for name in MODELS:
        getattr(trainer, name).load_state_dict(source.training_state['models'][name], strict=True)
    if model_hashes(trainer.state_dict()['models']) != model_hashes(source.training_state['models']):
        raise ValueError('transferred model tensors differ from the pretrained source')


def validate_source_provenance(source, config, prepared, topology):
    """Validate the source's bound cache proofs without importing its old data state."""
    validate_pretrained_source(source, config)
    if not config.data.pretrain_cache_manifest:
        raise ValueError('completed source requires its original frozen compact cache proof')
    identities = {}
    for partition, replay_key, path in (
        ('train', 'offline', config.data.pretrain_cache_manifest),
        ('validation', 'codec_validation', config.data.validation_cache_manifest or config.data.pretrain_cache_manifest),
    ):
        dataset = CompactPretrainDataset(config, prepared, topology,
            config.resources.world_model_initialization_sequence_length,
            cache_manifest_path=path, split=partition)
        dataset.load_state_dict(source.training_state['replay'][replay_key])
        identities[partition] = dataset.manifest['identity']
    metrics = source.training_state['metrics']
    if (not metrics.get('codec_ready') or metrics.get('codec_validation_manifest_identity') != identities['validation']
            or metrics.get('codec_validation_updates') != metrics.get('codec_updates')):
        raise ValueError('source codec proof does not match its frozen validation cache')
    limits = dict(codec_count_mae_occupied=config.codec.occupied_vehicle_mae_max,
        codec_count_mae_all=config.codec.vehicle_mae_max, codec_count_mae_empty=config.codec.empty_vehicle_mae_max,
        codec_total_relative_error=config.codec.total_relative_mae_max, codec_speed_mae_mps=config.codec.speed_mae_mps_max)
    for prefix in ('codec_validation_', 'codec_hybrid_validation_'):
        if any(not 0 <= metrics.get(prefix + key, float('inf')) <= maximum for key, maximum in limits.items()):
            raise ValueError('source codec metrics do not meet its unchanged quality gate')
    return identities


def validate_initialization_manifest(manifest, source, source_config, target_config, input_hash, output):
    validate_transfer_config(source_config, target_config)
    expected = dict(source_checkpoint_sha256=sha256_file(source.path),
        source_config_fingerprint=source_config.semantic_fingerprint,
        target_config_fingerprint=target_config.semantic_fingerprint,
        input_hash=input_hash, target_output=str(Path(output).resolve()))
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('meso initialization source/config/input identity differs')
    if manifest.get('source_model_hashes', manifest.get('model_hashes')) != model_hashes(source.training_state['models']):
        raise ValueError('meso initialization source model hashes differ')
    frozen = Path(output) / 'initialization_checkpoint.pt'
    if not frozen.is_file() or sha256_file(frozen) != manifest.get('target_initialization_sha256'):
        raise ValueError('meso initialization checkpoint hash differs')
    if not manifest.get('codec_validation', {}).get('codec_ready'):
        raise ValueError('meso initialization codec quality gate has not passed')
    import libsumo
    if libsumo.__version__ != manifest.get('target_engine_version'):
        raise ValueError('meso initialization libsumo engine version differs')


def validate_target_initialization(state, source, manifest):
    if model_hashes(state['models']) != manifest['model_hashes']:
        raise ValueError('meso initialization model hashes differ from the calibrated target')
    if model_hashes(nondecoder_models(state['models'])) != model_hashes(nondecoder_models(source.training_state['models'])):
        raise ValueError('meso initialization changed frozen nondecoder source tensors')
    metrics = state['metrics']
    if (state['next_environment_episode'] != 0 or state['replay']['online']['episodes']
            or any(metrics.get(key, 0) != 0 for key in ('world_model_initialization_epochs_completed',
                'online_episodes_completed', 'evaluation_episodes_completed', 'codec_epochs_completed', 'codec_updates'))):
        raise ValueError('meso initialization must use fresh target training/collection counters')
    if any(optimizer['state'] for optimizer in state['optimizers'].values()):
        raise ValueError('meso initialization optimizers must be fresh')
    calibration = manifest.get('decoder_calibration', {})
    if not 0 <= calibration.get('updates', 0) <= calibration.get('max_updates', 0) <= 1000:
        raise ValueError('meso decoder calibration exceeds its explicit bound')


def migrate(source_dir, output_dir, *, config_path=ROOT / 'configs/ingolstadt_custom.toml',
            device='cpu', threads=4, sampler_workers=8, codec_calibration_updates=0,
            continue_failed_validation=False, gpu_memory_fraction=.08):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir or (output_dir.exists() and not continue_failed_validation):
        raise ValueError('meso initialization requires a new absent output directory')
    if continue_failed_validation and (not (output_dir / 'meso_codec_transfer_validation.json').is_file()
                                      or latest_checkpoint_path(output_dir).exists()):
        raise ValueError('continuation requires a failed validation report without a published checkpoint')
    source_config = load_config(source_dir / 'run_config.toml')
    base = load_config(config_path)
    config = replace(base, run=replace(base.run, output_dir=output_dir, device=device),
        data=replace(base.data, archive_roots=(ROOT / 'outputs/city_baseline_dataset_meso/ingolstadt_custom/random',),
                     validation_archive_roots=(), pretrain_cache_manifest=None, validation_cache_manifest=None))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    validate_transfer_config(source_config, config)
    prepared = validate_prepared_inputs(config)
    source_prepared = validate_prepared_inputs(source_config)
    input_hash = combined_input_hash(prepared)
    if input_hash != combined_input_hash(source_prepared):
        raise ValueError('meso initialization prepared input hashes differ')
    source_hash = sha256_file(latest_checkpoint_path(source_dir))
    source = load_checkpoint(output_dir=source_dir, expected_config_fingerprint=source_config.semantic_fingerprint,
                             expected_input_hash=input_hash)
    validate_pretrained_source(source, source_config)
    topology = load_traffic_topology(prepared)
    source_cache_identities = validate_source_provenance(source, source_config, source_prepared, topology)
    torch.set_num_threads(threads)
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, torch.device(device))
    trainer = MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology)
    try:
        install_sparse_codec(trainer)
        transfer_models(trainer, source)
        trainer.metrics['imagination_workers'] = sampler_workers
        trainer.validate_pretraining_data()
        codec = trainer._validate_codec()
        report = dict(schema_version=1, kind='micro_pretrained_neural_weights_to_fresh_meso_online',
            source_output=str(source_dir), target_output=str(output_dir),
            source_checkpoint_sha256=source_hash, source_config_fingerprint=source_config.semantic_fingerprint,
            target_config_fingerprint=config.semantic_fingerprint, input_hash=input_hash,
            model_hashes=model_hashes(source.training_state['models']),
            source_pretraining=dict(traffic_model='micro',
                world_model_epochs=source.training_state['metrics']['world_model_initialization_epochs_completed'],
                codec_updates=source.training_state['metrics']['codec_updates']),
            source_cache_manifest_identities=source_cache_identities,
            target_meso_pretraining_epochs=0, codec_validation=codec,
            training_manifest_identity=trainer.offline_data.manifest['identity'],
            validation_manifest_identity=trainer.codec_validation_data.manifest['identity'],
            preserved='all neural model tensors exactly', reset='all optimizers, replay, RNG, counters and evaluations',
            online_episode_budget=config.run.online_training_episodes, evaluation_every=20,
            sampler_workers=sampler_workers)
        import libsumo
        report['target_engine_version'] = libsumo.__version__
        initial_report = output_dir / 'meso_codec_transfer_validation.json'
        if initial_report.exists():
            previous = json.loads(initial_report.read_text())
            for key in ('source_checkpoint_sha256', 'source_config_fingerprint', 'target_config_fingerprint',
                        'input_hash', 'model_hashes', 'training_manifest_identity', 'validation_manifest_identity'):
                if previous.get(key) != report[key]:
                    raise ValueError('failed validation source/config/manifests changed before calibration')
        else:
            atomic_json(initial_report, report)
        report['source_model_hashes'] = report['model_hashes']
        report['codec_initial_validation'] = codec
        if not trainer.codec_ready() and codec_calibration_updates:
            report['decoder_calibration'] = calibrate_decoder(trainer, max_updates=codec_calibration_updates,
                output=output_dir, source_models=source.training_state['models'])
            report['codec_validation'] = trainer._validate_codec()
            report['preserved'] = 'encoder, dynamics, auxiliary readout, actor, macro V and event Q tensors exactly'
            report['changed'] = 'decoder tensors only through explicitly bounded production-loss calibration'
        trainer.validate_codec_ready()
        trainer.world_model.flow_projector.requires_grad_(False)
        trainer.metrics['source_pretraining_provenance'] = report['source_pretraining']
        if model_hashes(nondecoder_models(trainer.state_dict()['models'])) != model_hashes(nondecoder_models(source.training_state['models'])):
            raise ValueError('initialization changed a nondecoder source model tensor')
        report['model_hashes'] = model_hashes(trainer.state_dict()['models'])
        report['frozen_nondecoder_hashes'] = model_hashes(nondecoder_models(source.training_state['models']))
        path = save_checkpoint(output_dir=output_dir, config_fingerprint=config.semantic_fingerprint,
            input_hash=input_hash, phase=CheckpointPhase.ONLINE_TRAINING, next_unit=0,
            training_state=trainer.state_dict())
        shutil.copy2(path, output_dir / 'initialization_checkpoint.pt')
        report['target_initialization_sha256'] = sha256_file(path)
        export_run_config(config, output_dir / 'run_config.toml')
        atomic_json(output_dir / 'meso_initialization.json', report)
        loaded = load_checkpoint(output_dir=output_dir, expected_config_fingerprint=config.semantic_fingerprint,
                                 expected_input_hash=input_hash)
        validate_target_initialization(loaded.training_state, source, report)
        trainer.load_state_dict(loaded.training_state)
        trainer.validate_codec_ready()
        if any(optimizer.state for optimizer in trainer.optimizers.values()):
            raise ValueError('target optimizers must start fresh')
        if sha256_file(source.path) != source_hash:
            raise ValueError('source checkpoint changed during initialization')
        return report
    finally:
        trainer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ingolstadt_custom.toml')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--sampler-workers', type=int, default=8)
    parser.add_argument('--codec-calibration-updates', type=int, default=0)
    parser.add_argument('--continue-failed-validation', action='store_true')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.08)
    args = parser.parse_args(argv)
    if args.threads < 1 or not 0 <= args.sampler_workers <= 8 or not 0 <= args.codec_calibration_updates <= 1000:
        parser.error('positive threads and sampler workers from zero to eight required')
    print(json.dumps(migrate(args.source_dir, args.output_dir, config_path=args.config,
        device=args.device, threads=args.threads, sampler_workers=args.sampler_workers,
        codec_calibration_updates=args.codec_calibration_updates,
        continue_failed_validation=args.continue_failed_validation,
        gpu_memory_fraction=args.gpu_memory_fraction), indent=2), flush=True)


if __name__ == '__main__':
    main()
