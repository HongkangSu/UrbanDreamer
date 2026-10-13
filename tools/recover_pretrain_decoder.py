"""Calibrate a partial pretrained decoder; publish a candidate only after its codec gate passes."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.calibrate_frozen_decoder import calibration_pairs, file_hash, verify_partition_separation
from tools.migrate_mainline_prior import _assert_equal
from tools.migrate_pretrained_meso import install_sparse_codec
from tools.probe_slow_codec import atomic_json, freeze_encoder_for_decoder_probe
from tools.run_city_pretrain import gpu_admission, write_status
from tools.train_codec_ready import FrameSampler
from urban_dreamer.learning.codec_training import codec_objective, decoder_parameters
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.checkpoint import _normalize_plain_value
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.pipeline import combined_input_hash

DECODER_PREFIXES = tuple('flow_projector.' + name + '.' for name in
                        ('decoder_shared', 'decoder_speed', 'decoder_volume', 'decoder_other'))


def load_recovery_state(path, *, protected_hashes, initial_decoder_updates):
    # Historical recovery saves contain only one NumPy array: MT19937's
    # uint32 RNG keys. Keep weights-only loading and scope its exact allowlist.
    with torch.serialization.safe_globals([
            np._core.multiarray._reconstruct, np.ndarray, np.dtype,
            type(np.dtype(np.uint32))]):
        saved = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or set(saved) != {
            'training_state', 'sampler', 'updates', 'protected_hashes'}:
        raise ValueError('decoder recovery checkpoint has an invalid schema')
    if saved['protected_hashes'] != protected_hashes:
        raise ValueError('decoder recovery source changed before resume')
    updates = saved['updates']
    if (type(updates) is not int or updates < 0
            or saved['training_state']['metrics']['decoder_pretrain_updates']
            != initial_decoder_updates + updates):
        raise ValueError('decoder recovery update count differs')
    return saved


def save_recovery_state(path, *, training_state, sampler, updates, protected_hashes):
    # Reuse the main checkpoint serializer so future saves need no allowlist;
    # this also preserves integer optimizer keys and all replay/RNG fields.
    payload = _normalize_plain_value(dict(training_state=training_state, sampler=sampler,
        updates=updates, protected_hashes=protected_hashes), path='decoder_recovery')
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def verify_initial_report(report, protected_hashes, decoder_revision):
    if (report.get('protected_sha256_before') != protected_hashes
            or report.get('protected_sha256_after') != protected_hashes
            or report.get('optimizer_steps') != 0
            or report.get('models_optimizers_unchanged') is not True
            or report.get('decoder_revision') != decoder_revision):
        raise ValueError('initial codec diagnostic does not match the frozen recovery source')


def validate_recovery_codec(trainer, batch_size):
    original = trainer.config
    trainer.config = replace(original, codec=replace(original.codec, batch_size=batch_size))
    try:
        return trainer._validate_codec()
    finally:
        trainer.config = original


def candidate_training_state(source, calibrated, *, updates, accepted):
    """Copy only decoder tensors/moments and verified codec admission into the source state."""
    if not accepted or not calibrated['metrics'].get('codec_ready'):
        raise ValueError('decoder recovery requires the unchanged full heldout quality gate')
    if updates < 1:
        raise ValueError('decoder recovery requires supervised calibration updates')
    expected = source['metrics']['decoder_pretrain_updates'] + updates
    if calibrated['metrics']['decoder_pretrain_updates'] != expected:
        raise ValueError('decoder recovery update count differs')
    if set(source['models']) != set(calibrated['models']) or set(source['optimizers']) != set(calibrated['optimizers']):
        raise ValueError('decoder recovery changed model or optimizer schema')
    for key in ('codec_updates', 'world_model_updates', 'world_model_initialization_updates',
                'decoder_online_updates'):
        if source['metrics'].get(key) != calibrated['metrics'].get(key):
            raise ValueError('decoder recovery changed non-calibration update counters')
    for model, values in source['models'].items():
        if set(values) != set(calibrated['models'][model]):
            raise ValueError('decoder recovery changed model tensor schema')
        for key, value in values.items():
            if model != 'world_model' or not key.startswith(DECODER_PREFIXES):
                _assert_equal(value, calibrated['models'][model][key])
    for key, value in source['optimizers'].items():
        if key != 'decoder_pretrain':
            _assert_equal(value, calibrated['optimizers'][key])
    result = copy.deepcopy(source)
    for key, value in calibrated['models']['world_model'].items():
        if key.startswith(DECODER_PREFIXES):
            result['models']['world_model'][key] = value.detach().clone()
    result['optimizers']['decoder_pretrain'] = copy.deepcopy(calibrated['optimizers']['decoder_pretrain'])
    for key, value in calibrated['metrics'].items():
        if key.startswith(('codec_validation_', 'codec_hybrid_validation_', 'decoder_pretrain_authorized_')):
            result['metrics'][key] = value
    result['metrics'].update(codec_ready=1., decoder_pretrain_updates=expected,
                             corpus_migration_codec_revalidation_required=False,
                             decoder_recovery_updates=updates)
    return result


def run(args):
    source, output = args.run_dir.resolve(), args.output_dir.resolve()
    if source == output or source in output.parents:
        raise ValueError('decoder recovery output must be separate from the training run')
    if output.exists() and not args.resume:
        raise ValueError('recovery output already exists; use --resume')
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = source / 'checkpoints/latest.pt'
    config = load_config(source / 'run_config.toml')
    protected = [checkpoint, source / 'run_config.toml', config.data.pretrain_cache_manifest]
    hashes = {str(path): file_hash(path) for path in protected}
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    initial = payload['training_state']
    if (payload['phase'] != 'pretrain' or payload['next_unit'] != 0
            or initial['metrics'].get('online_episodes_completed', 0) != 0
            or not config.codec.decoder_pretrain_enabled or config.codec.decoder_allow_unready):
        raise ValueError('recovery requires strict decoder-enabled offline pretraining')
    revision = initial['metrics']['decoder_pretrain_updates'] + initial['metrics'].get('decoder_online_updates', 0)
    if args.initial_gate_report:
        verify_initial_report(json.loads(args.initial_gate_report.read_text()), hashes, revision)
    if args.device.startswith('cuda'):
        while True:
            # Admission uses an 8 GiB working-space estimate, independently of
            # the allocator cap; a full allocator allowance cannot require
            # all physical VRAM plus a margin that no GPU can satisfy.
            admission = gpu_admission(args.device, .25, margin_mib=2048)
            write_status(output / 'status.json', phase='waiting_for_decoder_gpu', admission=admission)
            if admission['ready']:
                break
            time.sleep(15)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, torch.device(args.device))
    torch.set_num_threads(args.threads)
    prepared = validate_prepared_inputs(config)
    if payload['config_fingerprint'] != config.semantic_fingerprint or payload['input_hash'] != combined_input_hash(prepared):
        raise ValueError('recovery checkpoint identity differs from its frozen configuration')
    trainer = MainlineTrainer(config=replace(config, run=replace(config.run, device=args.device, output_dir=output)),
        prepared_inputs=prepared, topology=load_traffic_topology(prepared))
    updates = 0
    try:
        install_sparse_codec(trainer)
        MainlineTrainer.load_state_dict(trainer, initial)
        verify_partition_separation(trainer.offline_data.manifest['entries'], trainer.codec_validation_data.manifest['entries'])
        entries = [dict(entry, frames=entry['windows']) for entry in trainer.offline_data.manifest['entries']]
        sampler = FrameSampler(entries, args.batch_size // 2, config.run.seed)
        recovery_state = output / 'recovery_state.pt'
        if args.resume:
            saved = load_recovery_state(recovery_state, protected_hashes=hashes,
                initial_decoder_updates=initial['metrics']['decoder_pretrain_updates'])
            MainlineTrainer.load_state_dict(trainer, saved['training_state'])
            sampler.load_state_dict(saved['sampler'])
            updates = saved['updates']
        trainer.world_model.requires_grad_(False)
        freeze_encoder_for_decoder_probe(trainer.world_model.flow_projector)
        optimizer = trainer.optimizers['decoder_pretrain']
        for name, other in trainer.optimizers.items():
            if name != 'decoder_pretrain':
                def forbidden_step(*a, _name=name, **kw):
                    raise RuntimeError(f'decoder recovery forbids {_name} optimizer updates')
                other.step = forbidden_step

        def save_progress(phase, **values):
            current_hashes = {str(path): file_hash(path) for path in protected}
            if current_hashes != hashes:
                raise ValueError('frozen source changed during decoder recovery')
            save_recovery_state(recovery_state, training_state=trainer.state_dict(),
                sampler=sampler.state_dict(), updates=updates, protected_hashes=hashes)
            write_status(output / 'status.json', phase=phase, calibration_updates=updates,
                         frozen_source_sha256=hashes, **values)

        while updates < args.max_updates:
            index, starts = sampler.next()
            arrays = trainer.offline_data._episode(index)
            pairs = [pair for pair in calibration_pairs(arrays, starts, trainer.network.capacities) if pair is not None]
            flow = torch.cat([pair[0] for pair in pairs]).to(trainer.device)
            following = torch.cat([pair[1] for pair in pairs]).to(trainer.device)
            trainer.world_model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = codec_objective(trainer.world_model, flow, following, trainer.network,
                                     config.method, codec=config.codec, decoder_only=True)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite recovery loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(decoder_parameters(trainer.world_model), 10., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            trainer.metrics['decoder_pretrain_updates'] = initial['metrics']['decoder_pretrain_updates'] + updates
            trainer.metrics['codec_ready'] = 0.
            if updates % 100 == 0:
                save_progress('decoder_calibration', loss=float(loss), gradient_norm=float(norm))
            if updates % args.validation_every != 0 and updates != args.max_updates:
                continue
            save_progress('full_heldout_codec_validation')
            proof = validate_recovery_codec(trainer, args.codec_validation_batch_size)
            atomic_json(output / f'validation_{updates:06d}.json', dict(updates=updates, proof=proof,
                        limits=trainer._codec_limits(), passed=trainer.codec_ready()))
            if not trainer.codec_ready():
                continue
            trainer.validate_codec_pretraining_ready()
            candidate = candidate_training_state(initial, trainer.state_dict(), updates=updates, accepted=True)
            recovered = copy.deepcopy(payload)
            recovered['training_state'] = candidate
            candidate_path = output / 'candidate_checkpoint.pt'
            torch.save(recovered, candidate_path)
            atomic_json(output / 'recovery_proof.json', dict(status='candidate_ready', updates=updates,
                source_checkpoint=str(checkpoint), protected_sha256_before=hashes,
                protected_sha256_after={str(path): file_hash(path) for path in protected},
                candidate_sha256=file_hash(candidate_path), proof=proof,
                preserved='encoder, latent coordinates, dynamics/readout, actor/critics, all non-decoder optimizers, FM counters/coverage, source replay and RNG',
                changed='decoder weights and existing decoder_pretrain Adam moments, decoder revision and new heldout admission'))
            save_progress('candidate_ready', candidate=str(candidate_path), codec_ready=True)
            return
        save_progress('budget_exhausted', codec_ready=False)
        raise RuntimeError('decoder calibration budget exhausted without passing the unchanged heldout gate')
    except BaseException as error:
        write_status(output / 'status.json', phase='failed', error=repr(error),
                     calibration_updates=updates, frozen_source_sha256=hashes,
                     retained_recovery_checkpoint=str(output / 'recovery_state.pt'))
        raise
    finally:
        trainer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--initial-gate-report', type=Path)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--gpu-memory-fraction', type=float, default=1.)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--codec-validation-batch-size', type=int, default=32)
    parser.add_argument('--max-updates', type=int, default=2000)
    parser.add_argument('--validation-every', type=int, default=500)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (args.batch_size < 2 or args.batch_size % 2
            or min(args.threads, args.max_updates, args.validation_every, args.codec_validation_batch_size) < 1
            or not 0 < args.gpu_memory_fraction <= 1):
        parser.error('positive budgets and an even batch size >= 2 are required')
    run(args)


if __name__ == '__main__':
    main()
