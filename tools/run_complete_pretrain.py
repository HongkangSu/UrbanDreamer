"""Pretrain independent full-prior windows with current-interval births removed from endpoint volume."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from urban_dreamer.models.world_model import BlockCausalLayer, WORLD_MODEL_DIM
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.learning.episode_archive import archive_physics_identity
from urban_dreamer.learning.offline_data import _identity
from urban_dreamer.learning.pretrain_cache import CACHE_SCHEMA_VERSION
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.simulation_settings import physics_identity
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.checkpoint import CheckpointPhase, latest_checkpoint_path, save_checkpoint
from urban_dreamer.workflow.config import DataConfig, load_config, _semantic_fingerprint, validate_fm_training_profile
from urban_dreamer.workflow.pipeline import MainlinePipeline, combined_input_hash
from tools.train_codec_ready import FrameSampler, GATES, SparseGraphCodec


FULL_TRAIN_EPISODES = 2920
FULL_VALIDATION_EPISODES = 520
VOLUME_SCOPE = 'previous_state_excluding_current_births'


def export_run_config(config, path):
    """Write the exact flat mainline sections without adding a TOML dependency."""
    lines = []
    snapshot = config.resolved_snapshot()
    # Snapshot identities omit legacy next_hop, but new exports must retain
    # their mode when loaded under any filename (including the temporary file).
    snapshot['simulation']['routing_mode'] = config.simulation.routing_mode
    for section, values in snapshot.items():
        lines.append(f'[{section}]')
        for name, value in values.items():
            if value is not None:
                lines.append(f'{name} = {json.dumps(value, ensure_ascii=False, allow_nan=False)}')
        lines.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    if load_config(temporary).semantic_fingerprint != config.semantic_fingerprint:
        temporary.unlink()
        raise ValueError('exported run configuration differs from actual training semantics')
    temporary.replace(path)


def runtime_profile(config, *, gpu_memory_fraction, threads):
    resources = config.resources
    return dict(device=config.run.device, threads=threads,
        batch_size=resources.world_model_initialization_batch_size,
        accumulation_steps=resources.world_model_gradient_accumulation_steps,
        effective_batch_size=resources.world_model_initialization_batch_size
                             * resources.world_model_gradient_accumulation_steps,
        space_chunk=resources.world_model_pretrain_space_chunk,
        time_chunk=resources.world_model_pretrain_time_chunk,
        mlp_chunk=resources.world_model_pretrain_mlp_chunk,
        activation_checkpointing=resources.world_model_pretrain_activation_checkpointing,
        precision=resources.world_model_pretrain_precision,
        compile_blocks=resources.world_model_pretrain_compile_blocks,
        compile_policy=('static_full_batch_bf16_training_eager_tails_v1'
                        if resources.world_model_pretrain_compile_blocks else 'eager'),
        readout_supervision=config.method.exposure_readout_supervision,
        fm_training_mode=config.method.fm_training_mode,
        sequence_length=resources.world_model_initialization_sequence_length,
        gpu_memory_fraction=gpu_memory_fraction)



def _dispatch_pretrain_block(eager_forward, compiled_forward, full_shape):
    def forward(x):
        if (tuple(x.shape) == full_shape and x.device.type == 'cuda'
                and x.dtype in (torch.float32, torch.bfloat16) and torch.is_grad_enabled()
                and torch.is_autocast_enabled('cuda')
                and torch.get_autocast_dtype('cuda') == torch.bfloat16):
            return compiled_forward(x)
        return eager_forward(x)
    return forward


def compile_pretrain_blocks(trainer):
    """Compile only the validated full batch; smaller tails use eager execution."""
    if not trainer.config.resources.world_model_pretrain_compile_blocks:
        return 0
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '2')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',
        str((trainer.config.run.output_dir.parent / 'inductor_cache').resolve()))
    model = trainer.world_model
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    state_keys = tuple(model.state_dict())
    blocks = [module for module in model.modules() if isinstance(module, BlockCausalLayer)]
    if not blocks:
        raise ValueError('compile-blocks requested but no BlockCausalLayer was found')
    resources = trainer.config.resources
    full_batch = resources.world_model_initialization_batch_size
    if trainer.config.method.fm_training_mode == 'legacy_independent':
        full_batch *= min(resources.world_model_initialization_sequence_length,
                          trainer.config.method.supervision_horizon)
    for block in blocks:
        full_shape = (full_batch, 1, block.space.attn_mask.shape[-1], WORLD_MODEL_DIM)
        eager_forward = block.forward
        compiled_forward = torch.compile(eager_forward, backend='inductor', mode='default',
                                         fullgraph=True, dynamic=False)
        block.forward = _dispatch_pretrain_block(eager_forward, compiled_forward, full_shape)
    if parameter_ids != tuple(id(parameter) for parameter in model.parameters()):
        raise RuntimeError('block compilation changed optimizer parameter identity')
    if state_keys != tuple(model.state_dict()):
        raise RuntimeError('block compilation changed checkpoint state keys')
    return len(blocks)


def resolve_compile_blocks(config, override):
    return (config.resources.world_model_pretrain_compile_blocks
            if override is None else override == 'on')


def migrate_correct_scope_checkpoint(trainer, path):
    """Preserve learned correct-label models, while resetting changed data coverage."""
    path = Path(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('schema_version') != 9:
        raise ValueError('model migration requires a trusted correct-label schema9 checkpoint')
    if payload.get('input_hash') != combined_input_hash(trainer.prepared_inputs):
        raise ValueError('migration checkpoint has different prepared inputs')
    source_path = path.parent.parent / 'resolved_config.json'
    source = json.loads(source_path.read_text())
    target = trainer.config.resolved_snapshot()
    source_method = dict(source.get('method', {}))
    if source_method.get('transition_volume_scope') != VOLUME_SCOPE:
        raise ValueError('migration requires previous_state_excluding_current_births source labels')
    source_method.setdefault('exposure_readout_supervision', 'sampled_endpoint')
    if physics_identity(source.get('simulation', {})) != physics_identity(trainer.config.simulation):
        raise ValueError('migration simulation traffic physics contract differs from the trusted source')
    for section in ('inputs', 'simulation', 'codec'):
        source_values, target_values = source.get(section), target[section]
        if not isinstance(source_values, dict):
            raise ValueError(f'migration {section} contract differs from the trusted source')
        if section == 'simulation':
            # Canonical physics was checked above. Historical micro snapshots
            # predate these fields; compare all remaining settings unchanged.
            source_values = {key: value for key, value in source_values.items()
                if key != 'traffic_model' and not key.startswith('meso_')}
            target_values = {key: value for key, value in target_values.items()
                if key != 'traffic_model' and not key.startswith('meso_')}
        if source_values != target_values:
            raise ValueError(f'migration {section} contract differs from the trusted source')
    for key, value in target['method'].items():
        if key != 'exposure_readout_supervision' and source_method.get(key) != value:
            raise ValueError(f'migration changes method.{key}; only readout supervision may differ')
    for key in ('seed', 'online_training_episodes', 'evaluation_episodes'):
        if source.get('run', {}).get(key) != target['run'][key]:
            raise ValueError(f'migration changes run.{key}')
    allowed = {'world_model_initialization_batch_size', 'world_model_initialization_sequence_length',
        'world_model_gradient_accumulation_steps', 'world_model_pretrain_space_chunk',
        'world_model_pretrain_time_chunk', 'world_model_pretrain_mlp_chunk',
        'world_model_pretrain_activation_checkpointing', 'world_model_pretrain_precision'}
    for key, value in target['resources'].items():
        if key not in allowed and source.get('resources', {}).get(key) != value:
            raise ValueError(f'migration changes resources.{key}')
    state = payload.get('training_state', {})
    metrics = state.get('metrics', {})
    if (state.get('next_environment_episode') != 0 or metrics.get('online_episodes_completed', 0) != 0
            or metrics.get('evaluation_episodes_completed', 0) != 0
            or state.get('replay', {}).get('online', {}).get('episodes')):
        raise ValueError('migration refuses checkpoints that have collected/trained online or evaluated')
    phase = payload.get('phase')
    if phase != 'pretrain':
        raise ValueError('migration requires a pretraining-only source checkpoint')
    old_length = source['resources']['world_model_initialization_sequence_length']
    new_length = target['resources']['world_model_initialization_sequence_length']
    replay = state.get('replay', {})
    for name, dataset in (('offline', trainer.offline_data),
                          ('codec_validation', trainer.codec_validation_data)):
        if dataset is None or replay.get(name) is None:
            raise ValueError('migration requires independent bound training and validation sources')
        manifest = copy.deepcopy(dataset.manifest)
        manifest['contract']['sequence_length'] = old_length
        identity = _identity({key: value for key, value in manifest.items() if key != 'identity'})
        if replay[name].get('manifest_identity') != identity:
            raise ValueError(f'migration {name} source manifest differs beyond sequence coverage')
    migrated = copy.deepcopy(state)
    # This is an explicit new full-coverage experiment, not a strict same-run
    # resume. A normal schema10 resume retains its cursor through the pipeline.
    reset = True
    copied_counters = {key: value for key, value in metrics.items()
                       if key.startswith('world_model_initialization_') or key.startswith('world_model_epoch_')}
    if reset:
        migrated['replay']['offline'] = trainer.offline_data.state_dict()
        migrated['replay']['codec_validation'] = trainer.codec_validation_data.state_dict()
        for key in copied_counters:
            migrated['metrics'][key] = 0.
    migrated['metrics']['codec_ready'] = 0.
    migrated['metrics']['codec_epochs_completed'] = 0.
    for key in list(migrated['metrics']):
        if key.startswith('codec_validation_') or key.startswith('codec_hybrid_validation_'):
            del migrated['metrics'][key]
    migrated['metrics']['migration_source_checkpoint'] = str(path.resolve())
    migrated['metrics']['migration_source_schema'] = 9
    migrated['metrics']['migration_source_initialization_counters'] = copied_counters
    migrated['metrics']['migration_source_sequence_length'] = old_length
    migrated['metrics']['migration_source_world_model_updates'] = metrics['world_model_updates']
    migrated['metrics']['migration_sampler_reset'] = reset
    trainer.load_state_dict(migrated)
    return dict(source_checkpoint=str(path.resolve()), source_schema=9,
        preserved_world_model_updates=metrics['world_model_updates'],
        old_sequence_length=old_length, new_sequence_length=new_length, sampler_reset=reset,
        imported_components=['all_models', 'all_optimizers', 'rng'], codec_revalidation_required=True)


def import_codec_only(trainer, checkpoint_path):
    """Warm physical reconstruction only; schema8 dynamics never enter the new run."""
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('schema_version') != 8:
        raise ValueError('codec-only import requires the historical schema8 mainline checkpoint')
    if payload.get('input_hash') != combined_input_hash(trainer.prepared_inputs):
        raise ValueError('codec-only import checkpoint has different prepared network inputs')
    state = payload.get('training_state', {})
    world = state.get('models', {}).get('world_model', {})
    prefix = 'flow_projector.'
    codec = {name[len(prefix):]: value for name, value in world.items() if name.startswith(prefix)}
    if (not codec or any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
                         for value in codec.values())):
        raise ValueError('codec-only import requires finite flow_projector parameters')
    updates = state.get('metrics', {}).get('codec_updates')
    if (isinstance(updates, bool) or not isinstance(updates, (int, float))
            or not math.isfinite(updates) or updates < 0 or updates != int(updates)):
        raise ValueError('codec-only import checkpoint has invalid codec update accounting')
    optimizer = state.get('optimizers', {}).get('codec')
    if not isinstance(optimizer, dict) or not optimizer.get('param_groups'):
        raise ValueError('codec-only import checkpoint lacks codec optimizer state')
    trainer.world_model.flow_projector.load_state_dict(codec, strict=True)
    trainer.optimizers['codec'].load_state_dict(optimizer)
    trainer.metrics['codec_updates'] = int(updates)
    # Readiness, FM counters, actor/critic states and all other optimizers belong
    # to the new semantics. Only actual validation can set the new ready proof.
    trainer.metrics['codec_ready'] = 0.
    trainer.metrics['codec_warm_start_checkpoint'] = str(Path(checkpoint_path).resolve())
    trainer.metrics['codec_warm_start_schema'] = 8
    trainer.metrics['codec_warm_start_updates'] = int(updates)
    return dict(checkpoint=str(Path(checkpoint_path).resolve()), source_schema=8,
                imported_codec_updates=int(updates), imported_components=['flow_projector', 'codec_optimizer'])


def checked_gate(metrics, report, *, label):
    """A producer's ready flag must agree with every finite physical error."""
    if not isinstance(metrics, dict) or not isinstance(report, dict):
        raise ValueError(f'{label} validation metrics and gate proof are missing')
    if report.get('thresholds') != GATES:
        raise ValueError(f'{label} codec thresholds differ from the accepted physical contract')
    checks = {}
    for name, limit in GATES.items():
        value = metrics.get(name)
        checks[name] = (not isinstance(value, bool) and isinstance(value, (int, float))
                        and math.isfinite(value) and 0 <= value <= limit)
    passed = all(checks.values())
    if report.get('checks') != checks or report.get('passed') is not passed:
        raise ValueError(f'{label} reported codec gate does not match its physical metrics')
    return passed


def codec_checkpoint_proof(codec_dir, result):
    """Freeze accepted parameter/optimizer objects, bound to the frozen pilot."""
    manifest_path = codec_dir / 'codec_manifest.json'
    checkpoint_path = codec_dir / 'codec_latest.pt'
    manifest = json.loads(manifest_path.read_text())
    manifest_identity = hashlib.sha256(json.dumps(
        {key: value for key, value in manifest.items() if key != 'identity'},
        sort_keys=True).encode()).hexdigest()
    if manifest.get('identity') != manifest_identity or result.get('manifest_identity') != manifest_identity:
        raise ValueError('codec result is not bound to the frozen codec manifest')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    required = {'schema_version', 'codec', 'optimizer', 'sampler', 'manifest_identity',
                'update', 'batch_size', 'codec_speed_positive'}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys() or checkpoint['schema_version'] != 1:
        raise ValueError('accepted codec checkpoint has an invalid schema')
    if (checkpoint['manifest_identity'] != manifest_identity
            or checkpoint['update'] != result.get('update')
            or isinstance(checkpoint['update'], bool) or not isinstance(checkpoint['update'], int)
            or checkpoint['update'] < 0):
        raise ValueError('codec checkpoint does not match the evaluated result/source/update')
    if checkpoint['codec_speed_positive'] is not True:
        raise ValueError('accepted codec must expose positive physical decoder speed')
    if (isinstance(checkpoint['batch_size'], bool) or not isinstance(checkpoint['batch_size'], int)
            or checkpoint['batch_size'] < 1):
        raise ValueError('accepted codec checkpoint has an invalid sampler batch size')
    entries = manifest.get('entries', {})
    if not entries.get('train') or not entries.get('validation'):
        raise ValueError('codec acceptance requires nonempty training and independent held-out sources')
    groups = {split: {row['demand_group'] for row in entries[split]} for split in ('train', 'validation')}
    if groups['train'] & groups['validation']:
        raise ValueError('codec acceptance sources share held-out demand identities')
    sampler = FrameSampler(entries['train'], checkpoint['batch_size'], 0)
    try:
        sampler.load_state_dict(checkpoint['sampler'])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError('accepted codec checkpoint sampler is invalid') from exc
    state = checkpoint['codec']
    if (not isinstance(state, dict) or not state
            or any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
                   for value in state.values())):
        raise ValueError('accepted codec weights must be nonempty finite tensors')
    if not isinstance(checkpoint['optimizer'], dict) or not checkpoint['optimizer'].get('param_groups'):
        raise ValueError('accepted codec optimizer state is missing')
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode() + str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    state_hash = digest.hexdigest()
    if result.get('codec_state_sha256', state_hash) != state_hash:
        raise ValueError('evaluated codec parameter hash differs from checkpoint weights')
    return checkpoint, manifest, dict(codec_manifest_identity=manifest_identity,
        codec_update=checkpoint['update'], codec_state_sha256=state_hash)


def checked_cache_manifest(cache, *, expected_train=FULL_TRAIN_EPISODES,
                           expected_validation=FULL_VALIDATION_EPISODES):
    if (cache.get('status') != 'complete' or cache.get('schema_version') != CACHE_SCHEMA_VERSION
            or cache.get('identity') != _identity({key: value for key, value in cache.items() if key != 'identity'})):
        raise ValueError('full cache lacks a complete verified manifest')
    entries = cache.get('entries', [])
    sources = cache.get('sources', [])
    if not entries or not sources:
        raise ValueError('full pretraining cache cannot have empty episodes or sources')
    source_paths = {source['path'] for source in sources}
    if (len(source_paths) != len(sources) or len(sources) < len(entries)
            or cache.get('duplicate_files') != len(sources) - len(entries)):
        raise ValueError('full cache source membership or duplicate count is inconsistent')
    groups = {split: set() for split in ('train', 'validation')}
    counts = dict(train=0, validation=0)
    for entry in entries:
        split = entry.get('split')
        if (split not in groups or not isinstance(entry.get('demand_group'), str)
                or not entry['demand_group'] or entry.get('windows', 0) < 1
                or entry.get('path') not in source_paths
                or not entry.get('cache_content_hash') or not Path(entry['cache_path']).is_file()):
            raise ValueError('full cache episode lacks split, demand identity, windows or checked artifact')
        groups[split].add(entry['demand_group'])
        counts[split] += 1
    if groups['train'] & groups['validation']:
        raise ValueError('full cache training and validation share demand identities')
    for source in sources:
        if source.get('split') not in groups or source.get('demand_group') not in groups[source['split']]:
            raise ValueError('full cache source split differs from its frozen demand membership')
    if (counts != dict(train=expected_train, validation=expected_validation)
            or cache.get('train_episodes') != counts['train']
            or cache.get('validation_episodes') != counts['validation']):
        raise ValueError(f'full cache membership must be {expected_train} train/{expected_validation} held-out episodes')
    return groups


def prerequisite_state(codec_dir, cache_path, *, expected_train=FULL_TRAIN_EPISODES,
                       expected_validation=FULL_VALIDATION_EPISODES, expected_simulation=None):
    codec_path = codec_dir / 'result.json'
    result = json.loads(codec_path.read_text()) if codec_path.is_file() else None
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else None
    # A recovery producer explicitly publishes this marker and refreshes it.
    # A terminal failed hybrid result cannot turn into an indefinite wait for
    # a producer that has already exited.
    markers = []
    if result and result.get('status') == 'hybrid_training_active':
        markers.append(codec_path)
    progress_path = codec_dir / 'progress.json'
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get('status') == 'hybrid_training_active':
            markers.append(progress_path)
    terminal_success = bool(result and result.get('status') != 'hybrid_training_active'
                            and (result.get('gate') or {}).get('passed')
                            and (result.get('hybrid_gate') or {}).get('passed'))
    if markers and not terminal_success:
        marker = max(markers, key=lambda path: path.stat().st_mtime)
        if time.time() - marker.stat().st_mtime > 300:
            raise RuntimeError('hybrid codec recovery heartbeat is stale; FM remains blocked')
        return dict(ready=False, codec_ready=False, cache_ready=bool(cache and cache.get('status') == 'complete'),
                    recovery='hybrid_training_active')
    if result is None:
        return dict(ready=False, codec_ready=False, cache_ready=bool(cache and cache.get('status') == 'complete'))
    if result.get('status') in {'budget_exhausted', 'hybrid_gate_failed'}:
        raise RuntimeError('codec producer ended without raw/hybrid quality acceptance; restart recovery explicitly; FM remains blocked')
    raw_ready = checked_gate(result.get('validation'), result.get('gate'), label='raw')
    hybrid_ready = checked_gate(result.get('hybrid_validation'), result.get('hybrid_gate'), label='fixed-cohort hybrid')
    if not raw_ready or not hybrid_ready:
        raise RuntimeError('codec producer ended with failed raw/hybrid quality gate; recovery must run before FM')
    if cache is None or cache.get('status') != 'complete':
        if expected_simulation is not None:
            manifest = json.loads((codec_dir / 'codec_manifest.json').read_text())
            codec_physics = archive_physics_identity({'session_protocol': 'event', **manifest})
            if codec_physics != physics_identity(expected_simulation):
                raise ValueError('accepted codec traffic physics differ from the requested run')
        return dict(ready=False, codec_ready=True, cache_ready=False)
    groups = checked_cache_manifest(cache, expected_train=expected_train, expected_validation=expected_validation)
    checkpoint, manifest, proof = codec_checkpoint_proof(codec_dir, result)
    if (manifest.get('input_hashes') != cache.get('contract', {}).get('input_hashes')
            or manifest.get('physical') != cache.get('contract', {}).get('physical')):
        raise ValueError('accepted codec and full cache have different network/physical contracts')
    codec_physics = archive_physics_identity({'session_protocol': 'event', **manifest})
    cache_physics = archive_physics_identity({'session_protocol': 'event', **cache['contract']})
    if codec_physics != cache_physics:
        raise ValueError('accepted codec and full cache have different traffic physics contracts')
    if expected_simulation is not None and codec_physics != physics_identity(expected_simulation):
        raise ValueError('accepted codec and full cache traffic physics differ from the requested run')
    pilot_train = {row['demand_group'] for row in manifest['entries']['train']}
    pilot_validation = {row['demand_group'] for row in manifest['entries']['validation']}
    if pilot_train & groups['validation'] or not pilot_validation <= groups['validation']:
        raise ValueError('full held-out cache leaked into codec training or lacks codec held-out membership')
    source_paths = {entry['path'] for entry in cache['sources']}
    if any(row['path'] not in source_paths for rows in manifest['entries'].values() for row in rows):
        raise ValueError('accepted codec sources are outside the frozen complete-corpus cache')
    return dict(ready=True, codec_ready=True, cache_ready=True, checkpoint=checkpoint,
                proof={**proof, 'cache_manifest_identity': cache['identity']})


def _archived_calibration_binding(binding_path, path, contract, cache, groups, prepared, config):
    """Check preserved evidence, explicitly without revalidating the lost original."""
    from tools.calibrate_frozen_decoder import file_hash
    binding_path = Path(binding_path).resolve()
    binding = json.loads(binding_path.read_text())
    if (binding.get('schema_version') != 1 or binding.get('source_original_unavailable') is not True
            or binding.get('identity') != _identity({k: v for k, v in binding.items() if k != 'identity'})):
        raise ValueError('archived calibration binding schema, declaration or identity is invalid')
    if binding.get('calibration_checkpoint_sha256') != file_hash(path):
        raise ValueError('archived calibration checkpoint hash differs from binding')
    for field, contract_field in (('recorded_source_checkpoint_sha256', 'source_checkpoint_sha256'),
                                   ('frozen_parameters_sha256', 'frozen_parameters_sha256')):
        if not binding.get(field) or binding[field] != contract.get(contract_field):
            raise ValueError(f'archived calibration binding differs in {field}')
    cache_contract = cache['contract']
    if (binding.get('input_hashes') != prepared.input_hashes
            or binding.get('input_hashes') != cache_contract.get('input_hashes')):
        raise ValueError('archived calibration binding has different network inputs')
    if binding.get('physical') != cache_contract.get('physical'):
        raise ValueError('archived calibration binding has different physical contract')
    target_physics = physics_identity(config.simulation)
    if (binding.get('physics_identity') != target_physics
            or archive_physics_identity({'session_protocol': 'event', **cache_contract}) != target_physics):
        raise ValueError('archived calibration binding/cache use different traffic physics')
    assignments = binding.get('original_demand_groups')
    if (not isinstance(assignments, dict) or not assignments
            or any(not isinstance(group, str) or not group or split not in ('train', 'validation')
                   for group, split in assignments.items())):
        raise ValueError('archived calibration binding lacks original demand assignments')
    for split, group_set in groups.items():
        if any(assignments.get(group) != split for group in group_set):
            raise ValueError('archived calibration training/held-out demand partition changed')
    return dict(archived_calibration_binding=True, source_checkpoint_revalidated=False,
                calibration_binding=str(binding_path), calibration_binding_sha256=file_hash(binding_path),
                calibration_binding_identity=binding['identity'])


def calibrated_codec_prerequisites(path, cache_path, config, *, binding_path=None, **expected_counts):
    """Admit the explicitly approved candidate without claiming its gates passed."""
    from tools.calibrate_frozen_decoder import file_hash
    from tools.probe_slow_codec import freeze_encoder_for_decoder_probe, frozen_state_hash
    path = Path(path).resolve()
    if not (config.codec.decoder_allow_unready and config.codec.decoder_pretrain_enabled
            and config.codec.decoder_online_enabled):
        raise ValueError('calibrated codec requires explicit unready admission and continuous decoder training')
    if not cache_path.is_file():
        return dict(ready=False, codec_ready=False, cache_ready=False)
    cache = json.loads(cache_path.read_text())
    if cache.get('status') != 'complete':
        return dict(ready=False, codec_ready=False, cache_ready=False)
    groups = checked_cache_manifest(cache, **expected_counts)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    contract = checkpoint.get('contract', {})
    if checkpoint.get('schema_version') != 1 or checkpoint.get('status') not in {'candidate_ready', 'candidate_unready', 'ready'}:
        raise ValueError('calibrated codec requires a completed calibration checkpoint')
    if not isinstance(checkpoint.get('updates'), int) or isinstance(checkpoint['updates'], bool) or checkpoint['updates'] <= 0:
        raise ValueError('calibrated codec requires positive training updates')
    prepared = validate_prepared_inputs(config)
    source = None
    if binding_path is not None:
        binding_proof = _archived_calibration_binding(
            binding_path, path, contract, cache, groups, prepared, config)
    else:
        source_path = Path(contract['source_checkpoint'])
        if file_hash(source_path) != contract.get('source_checkpoint_sha256'):
            raise ValueError('calibrated codec source checkpoint hash changed')
        source_config = load_config(source_path.parent.parent / 'run_config.toml')
        source = torch.load(source_path, map_location='cpu', weights_only=True)
        if source.get('input_hash') != combined_input_hash(prepared):
            raise ValueError('calibrated codec uses different network inputs')
        if (physics_identity(source_config.simulation) != physics_identity(config.simulation)
                or archive_physics_identity({'session_protocol': 'event', **cache['contract']}) != physics_identity(config.simulation)):
            raise ValueError('calibrated codec/cache use different traffic physics')
        old_cache = json.loads(Path(source_config.data.pretrain_cache_manifest).read_text())
        for key in ('input_hashes', 'physical'):
            if old_cache['contract'].get(key) != cache['contract'].get(key):
                raise ValueError(f'calibrated codec/cache differ in {key}')
        old_train = {entry['demand_group'] for entry in old_cache['entries'] if entry['split'] == 'train'}
        old_validation = {entry['demand_group'] for entry in old_cache['entries'] if entry['split'] == 'validation'}
        if old_train & groups['validation'] or old_validation & groups['train']:
            raise ValueError('calibrated codec training/held-out demand partition changed')
        binding_proof = dict(archived_calibration_binding=False, source_checkpoint_revalidated=True)
    topology = load_traffic_topology(prepared)
    codec = SparseGraphCodec(adjacency=torch.tensor(topology.network.adjacency),
                              positive_speed=config.codec.speed_positive)
    state = checkpoint.get('codec', {})
    if not state or any(not isinstance(v, torch.Tensor) or not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('calibrated codec weights must be finite')
    codec.load_state_dict(state, strict=True)
    freeze_encoder_for_decoder_probe(codec)
    if frozen_state_hash(codec) != contract.get('frozen_parameters_sha256'):
        raise ValueError('calibrated codec changed its frozen encoder')
    # Adjacency is a nonpersistent buffer, rebuilt from the checked target inputs.
    target_adjacency = torch.tensor(topology.network.adjacency, dtype=torch.float32)
    if (not torch.equal(codec.adjacency, target_adjacency)
            or not torch.equal(codec.sparse_adjacency.to_dense(), target_adjacency)):
        raise ValueError('calibrated codec adjacency differs from target network')
    if source is not None:
        original = {k.removeprefix('flow_projector.'): v for k, v in source['training_state']['models']['world_model'].items()
                    if k.startswith('flow_projector.')}
        for name, parameter in codec.named_parameters():
            if not parameter.requires_grad and not torch.equal(parameter.detach(), original[name]):
                raise ValueError('calibrated codec encoder differs from pretrained source')
    return dict(ready=True, codec_ready=False, cache_ready=True, checkpoint=checkpoint,
        proof=dict(calibration_checkpoint=str(path), calibration_checkpoint_sha256=file_hash(path),
            calibration_status=checkpoint['status'], calibration_updates=checkpoint['updates'],
            explicit_unready_admission=True, cache_manifest_identity=cache['identity'],
            source_checkpoint_sha256=contract['source_checkpoint_sha256'],
            imported_components=['flow_projector'], all_optimizers_fresh=True, **binding_proof))


def import_calibrated_codec(trainer, checkpoint):
    """Import only physical reconstruction; all dynamics, policies and Adam are fresh."""
    trainer.world_model.flow_projector.load_state_dict(checkpoint['codec'], strict=True)
    trainer.metrics.update(codec_ready=0., codec_updates=0.,
                           calibrated_codec_only_warm_start=True,
                           calibrated_codec_source_updates=int(checkpoint['updates']))
    trainer._set_dynamics_codec_gradients()
    return dict(source='explicitly_approved_calibration', imported_components=['flow_projector'],
                calibration_updates=checkpoint['updates'], all_optimizers_fresh=True)


def status(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), **value)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)
    print(json.dumps(value, sort_keys=True), flush=True)


class MonitoredTrainer(MainlineTrainer):
    def __init__(self, *, monitor_path, **kwargs):
        super().__init__(**kwargs)
        self.monitor_path = monitor_path
        self._last_heartbeat = 0.
        self._last_checkpoint_time = None
        self._last_checkpoint_update = None
        self._pretrain_progress_hook = self._save_pretrain_progress
        self._runtime_profile = {}
        self._parameter_count = sum(parameter.numel() for model in
            (self.world_model, self.actor, self.latent_critic, self.event_critic)
            for parameter in model.parameters())

    def _validate_codec(self):
        original = self.config
        batch_size = getattr(self, '_codec_validation_batch_size', original.codec.batch_size)
        self.config = replace(original, codec=replace(original.codec, batch_size=batch_size))
        try:
            return super()._validate_codec()
        finally:
            self.config = original

    def initialize_codec_epoch(self, epoch_index):
        if self.metrics.get('calibrated_codec_only_warm_start'):
            self.validate_codec_pretraining_ready()
            self._set_dynamics_codec_gradients()
            self.metrics['codec_epochs_completed'] = epoch_index + 1
            return dict(epoch=epoch_index + 1, skipped=True,
                        reason='approved_calibrated_codec_continues_training_in_pretrain',
                        codec_ready=self.codec_ready())
        return super().initialize_codec_epoch(epoch_index)

    def load_state_dict(self, state):
        super().load_state_dict(state)
        corpus_migration = self.config.run.output_dir / 'corpus_migration.json'
        if corpus_migration.is_file() and self.metrics.get('corpus_migration_codec_revalidation_required'):
            from tools.migrate_pretrain_corpus import revalidate_corpus_codec
            status(self.monitor_path, dict(phase='supplemental_codec_revalidation_running',
                validation=self.codec_validation_data.summary,
                runtime_profile=self._runtime_profile))
            proof = revalidate_corpus_codec(self)
            save_checkpoint(output_dir=self.config.run.output_dir,
                config_fingerprint=self.config.semantic_fingerprint,
                input_hash=combined_input_hash(self.prepared_inputs),
                phase=CheckpointPhase.PRETRAIN,
                next_unit=int(self.metrics['world_model_initialization_epochs_completed']),
                training_state=self.state_dict())
            status(self.monitor_path, dict(phase='supplemental_codec_revalidation', **proof,
                runtime_profile=self._runtime_profile))
        migration = self.config.run.output_dir / 'single_step_migration.json'
        if migration.is_file() and not self.codec_ready():
            status(self.monitor_path, dict(phase='single_step_codec_revalidation_running',
                validation=self.codec_validation_data.summary,
                runtime_profile=self._runtime_profile))
            proof = self._validate_codec()
            self.validate_codec_ready()
            self.world_model.flow_projector.requires_grad_(False)
            self.metrics['single_step_migration_codec_revalidation_required'] = False
            save_checkpoint(output_dir=self.config.run.output_dir,
                config_fingerprint=self.config.semantic_fingerprint,
                input_hash=combined_input_hash(self.prepared_inputs),
                phase=CheckpointPhase.PRETRAIN,
                next_unit=int(self.metrics['world_model_initialization_epochs_completed']),
                training_state=self.state_dict())
            status(self.monitor_path, dict(phase='single_step_codec_revalidation', **proof,
                runtime_profile=self._runtime_profile))

    def _save_pretrain_progress(self, epoch_index):
        """Persist only completed optimizer groups, including the first one."""
        now = time.monotonic()
        updates = int(self.metrics['world_model_updates'])
        if (self._last_checkpoint_time is not None
                and now - self._last_checkpoint_time < 300
                and updates - self._last_checkpoint_update < 500):
            return
        checkpoint_utc = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        self.metrics['pretrain_progress_checkpoint_utc'] = checkpoint_utc
        self.metrics['pretrain_progress_checkpoint_updates'] = updates
        save_checkpoint(output_dir=self.config.run.output_dir,
            config_fingerprint=self.config.semantic_fingerprint,
            input_hash=combined_input_hash(self.prepared_inputs),
            phase=CheckpointPhase.PRETRAIN, next_unit=epoch_index,
            training_state=self.state_dict())
        self._last_checkpoint_time = now
        self._last_checkpoint_update = updates
        status(self.monitor_path, dict(phase='fm_pretrain_checkpoint', epoch=epoch_index,
            optimizer_updates=updates, checkpoint_utc=checkpoint_utc,
            volume_labels=VOLUME_SCOPE,
            sampled_sequences=self.metrics['world_model_initialization_samples_seen'],
            runtime_profile=getattr(self, '_runtime_profile', {})))

    def _update_world(self, *args, **kwargs):
        result = super()._update_world(*args, **kwargs)
        if time.monotonic() - self._last_heartbeat >= 30:
            self._last_heartbeat = time.monotonic()
            group_samples = int(getattr(self, '_last_pretrain_group_samples', 0))
            update_seconds = float(getattr(self, '_last_world_update_seconds', 0.))
            data_seconds = float(getattr(self, '_last_pretrain_data_seconds', 0.))
            elapsed = update_seconds + data_seconds
            cuda = self.device.type == 'cuda'
            status(self.monitor_path, dict(phase='fm_pretrain',
                epoch=int(self.metrics['world_model_initialization_epochs_completed']),
                optimizer_updates=int(self.metrics['world_model_updates']),
                sampled_sequences=int(self.metrics['world_model_initialization_samples_seen']) + group_samples,
                supervision_depth=self.metrics.get('world_model_supervision_depth_last'),
                group_samples=group_samples, update_seconds=update_seconds, data_seconds=data_seconds,
                sequences_per_second_group=group_samples / elapsed if elapsed > 0 else None,
                gpu_memory_allocated_bytes=torch.cuda.memory_allocated(self.device) if cuda else 0,
                gpu_memory_reserved_bytes=torch.cuda.memory_reserved(self.device) if cuda else 0,
                params_total=self._parameter_count, runtime_profile=self._runtime_profile,
                codec_ready=self.codec_ready(), codec_frozen=all(
                    not parameter.requires_grad for parameter in self.world_model.flow_projector.parameters()),
                volume_labels=VOLUME_SCOPE, train=self.offline_data.summary,
                **result))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--codec-dir', type=Path)
    parser.add_argument('--calibrated-codec', type=Path,
        help='explicitly approved decoder calibration; import codec only with fresh models and optimizers')
    parser.add_argument('--calibration-binding', type=Path,
        help='preserved calibration evidence when original source checkpoint/cache are unavailable')
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--codec-checkpoint', type=Path,
        help='optional historical schema8 mainline checkpoint; import only codec weights/optimizer, never old dynamics')
    parser.add_argument('--migrate-checkpoint', type=Path,
        help='trusted schema9 correct-label pretraining checkpoint; preserve models/optimizer/RNG and revalidate codec')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda:0', 'cuda:1'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation-steps', type=int, default=2)
    parser.add_argument('--space-chunk', type=int, default=16)
    parser.add_argument('--time-chunk', type=int, default=32768)
    parser.add_argument('--mlp-chunk', type=int, default=16384)
    parser.add_argument('--activation-checkpointing', choices=('on', 'off'), default='on')
    parser.add_argument('--compile-blocks', choices=('on', 'off'), default=None,
                        help='compile transformer blocks in place; omitted inherits configuration')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.08)
    parser.add_argument('--activation-offload', action='store_true',
                        help='save backward activations on CPU while retaining the selected batch and compiled profile')
    parser.add_argument('--tail-execution-batch', type=int,
                        help='execute large remainders in smaller transformer chunks, retaining one joint loss/update')
    parser.add_argument('--codec-validation-batch-size', type=int,
                        help='batch independent codec validation frames without changing the training checkpoint contract')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--readout-supervision', choices=('observed_endpoint', 'sampled_endpoint'),
                        default='observed_endpoint')
    parser.add_argument('--sequence-length', type=int)
    parser.add_argument('--fm-training-mode', choices=('legacy_independent', 'single_step', 'recursive_3step'))
    parser.add_argument('--expected-train-episodes', type=int)
    parser.add_argument('--expected-validation-episodes', type=int)
    args = parser.parse_args(argv)
    if any(getattr(args, name) < 1 for name in ('threads', 'batch_size', 'accumulation_steps',
                                               'space_chunk', 'time_chunk', 'mlp_chunk')):
        parser.error('threads, batch size, accumulation steps and chunk sizes must be positive')
    if not math.isfinite(args.gpu_memory_fraction) or not 0 < args.gpu_memory_fraction <= 1:
        parser.error('gpu memory fraction must be finite and in (0, 1]')
    if args.sequence_length is not None and args.sequence_length < 1:
        parser.error('sequence length must be positive')
    if args.tail_execution_batch is not None and not 1 <= args.tail_execution_batch < args.batch_size:
        parser.error('tail execution batch must be positive and smaller than the full batch')
    if args.codec_validation_batch_size is not None and args.codec_validation_batch_size < 1:
        parser.error('codec validation batch size must be positive')
    if args.codec_checkpoint is not None and args.migrate_checkpoint is not None:
        parser.error('choose either codec-only warm start or full correct-label migration')
    if (args.codec_dir is None) == (args.calibrated_codec is None):
        parser.error('provide exactly one of --codec-dir or --calibrated-codec')
    if args.calibrated_codec and (args.codec_checkpoint or args.migrate_checkpoint):
        parser.error('calibrated codec import cannot be combined with other migrations')
    if args.calibration_binding is not None and args.calibrated_codec is None:
        parser.error('--calibration-binding requires --calibrated-codec')
    expected_counts = {}
    for argument, key in (('expected_train_episodes', 'expected_train'),
                          ('expected_validation_episodes', 'expected_validation')):
        value = getattr(args, argument)
        if value is not None:
            if value < 1:
                parser.error('expected episode counts must be positive')
            expected_counts[key] = value
    output = args.output_dir.resolve()
    config = load_config(args.config)
    # Monitoring lives outside the training directory so prerequisite waiting
    # never turns a fresh directory into an invalid partial training output.
    monitor = output.with_suffix('.status.json')
    while True:
        if args.calibrated_codec:
            prerequisites = calibrated_codec_prerequisites(args.calibrated_codec, args.cache_manifest,
                                                          config, binding_path=args.calibration_binding, **expected_counts)
        else:
            prerequisites = prerequisite_state(args.codec_dir, args.cache_manifest,
                expected_simulation=config.simulation, **expected_counts)
        if prerequisites['ready']:
            break
        status(monitor, dict(phase='waiting_for_quality_and_cache', **prerequisites))
        time.sleep(30)
    torch.set_num_threads(args.threads)
    chosen = args.device
    if chosen == 'auto':
        chosen = 'cpu'
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            free, _ = torch.cuda.mem_get_info(1)
            if free >= 2.25 * 1024 ** 3:
                chosen = 'cuda:1'
    if chosen.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, torch.device(chosen))
    config = replace(config,
        run=replace(config.run, output_dir=output, device=chosen),
        resources=replace(config.resources, world_model_initialization_batch_size=args.batch_size,
            world_model_gradient_accumulation_steps=args.accumulation_steps,
            world_model_pretrain_space_chunk=args.space_chunk,
            world_model_pretrain_time_chunk=args.time_chunk,
            world_model_pretrain_mlp_chunk=args.mlp_chunk,
            world_model_pretrain_activation_checkpointing=args.activation_checkpointing == 'on',
            world_model_pretrain_precision=args.precision,
            world_model_pretrain_compile_blocks=resolve_compile_blocks(config, args.compile_blocks),
            world_model_initialization_sequence_length=args.sequence_length
                or config.resources.world_model_initialization_sequence_length),
        method=replace(config.method, exposure_readout_supervision=args.readout_supervision,
                       fm_training_mode=args.fm_training_mode or config.method.fm_training_mode),
        data=DataConfig(archive_roots=(), cache_episodes=2,
                       pretrain_cache_manifest=args.cache_manifest.resolve(),
                       validation_cache_manifest=args.cache_manifest.resolve()))
    validate_fm_training_profile(config.method, config.resources)
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(inputs=config.inputs,
        run=config.run, simulation=config.simulation, resources=config.resources,
        method=config.method, data=config.data, codec=config.codec))
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    fresh = not latest_checkpoint_path(output).is_file()
    if (output / 'corpus_migration.json').is_file():
        from tools.migrate_pretrain_corpus import validate_corpus_continuation
        validate_corpus_continuation(output, config, combined_input_hash(prepared))
    if (output / 'single_step_migration.json').is_file():
        from tools.migrate_pretrain_single_step import validate_single_step_continuation
        validate_single_step_continuation(output, config, combined_input_hash(prepared))
    if (output / 'runtime_migration.json').is_file():
        from tools.migrate_pretrain_runtime import validate_runtime_continuation
        validate_runtime_continuation(output, config, combined_input_hash(prepared))
    def factory():
        trainer = MonitoredTrainer(config=config, prepared_inputs=prepared, topology=topology,
                                   monitor_path=monitor)
        trainer._runtime_profile = runtime_profile(config,
            gpu_memory_fraction=args.gpu_memory_fraction, threads=args.threads)
        trainer._runtime_profile['activation_offload'] = args.activation_offload
        trainer._runtime_profile['tail_execution_batch'] = args.tail_execution_batch
        trainer._codec_validation_batch_size = args.codec_validation_batch_size or config.codec.batch_size
        trainer._runtime_profile['codec_validation_batch_size'] = trainer._codec_validation_batch_size
        if args.tail_execution_batch:
            from tools.pretrain_tail_execution import enable_pretrain_tail_execution
            enable_pretrain_tail_execution(trainer, max_batch=args.tail_execution_batch)
        if args.activation_offload:
            from tools.pretrain_activation_offload import enable_pretrain_activation_offload
            enable_pretrain_activation_offload(trainer)
        sparse = SparseGraphCodec(adjacency=trainer._tensor(topology.network.adjacency),
                                  positive_speed=config.codec.speed_positive).to(trainer.device)
        # Parameter objects remain those registered with all optimizer groups.
        trainer.world_model.flow_projector.__class__ = SparseGraphCodec
        trainer.world_model.flow_projector.register_buffer('sparse_adjacency', sparse.sparse_adjacency,
                                                          persistent=False)
        compile_pretrain_blocks(trainer)
        if fresh:
            if args.calibrated_codec is not None:
                warm_start = import_calibrated_codec(trainer, prerequisites['checkpoint'])
                dynamics = 'fresh_random_initialization'
            elif args.migrate_checkpoint is not None:
                warm_start = migrate_correct_scope_checkpoint(trainer, args.migrate_checkpoint)
                dynamics = 'migrated_correct_label_model'
            elif args.codec_checkpoint is not None:
                warm_start = import_codec_only(trainer, args.codec_checkpoint)
                dynamics = 'fresh_random_initialization'
            else:
                checkpoint = prerequisites['checkpoint']
                if not checkpoint['codec_speed_positive']:
                    raise ValueError('codec decoder semantics do not match positive-speed mainline')
                trainer.world_model.flow_projector.load_state_dict(checkpoint['codec'], strict=True)
                trainer.optimizers['codec'].load_state_dict(checkpoint['optimizer'])
                trainer.metrics['codec_updates'] = int(checkpoint['update'])
                warm_start = dict(imported_codec_updates=int(checkpoint['update']),
                                  source='accepted_pilot_codec', imported_components=['flow_projector', 'codec_optimizer'])
                dynamics = 'fresh_random_initialization'
            trainer.metrics.update({'codec_acceptance_' + key: value
                                    for key, value in prerequisites['proof'].items()})
            # Revalidate on the complete held-out partition, rather than trusting
            # the initial twenty-episode acceptance alone.
            status(monitor, dict(phase='full_partition_codec_validation_running',
                device=chosen, train=trainer.offline_data.summary,
                validation=trainer.codec_validation_data.summary,
                acceptance_artifacts=prerequisites['proof'], volume_labels=VOLUME_SCOPE,
                warm_start=warm_start, dynamics=dynamics,
                runtime_profile=trainer._runtime_profile, params_total=trainer._parameter_count))
            proof = trainer._validate_codec()
            status(monitor, dict(phase='full_partition_codec_validation', **proof,
                device=chosen, train=trainer.offline_data.summary,
                validation=trainer.codec_validation_data.summary,
                acceptance_artifacts=prerequisites['proof'], volume_labels=VOLUME_SCOPE,
                warm_start=warm_start, dynamics=dynamics,
                runtime_profile=trainer._runtime_profile))
            export_run_config(config, output / 'run_config.toml')
        return trainer
    result = MainlinePipeline(config=config, input_hash=combined_input_hash(prepared),
                              trainer_factory=factory, pretrain_only=True).run()
    status(monitor, dict(phase='pretraining_completed', summary=result))


if __name__ == '__main__':
    main()
