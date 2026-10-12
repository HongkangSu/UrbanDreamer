"""Isolated, resumable residual CFM with frozen regression center and codec."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.component_experiment_data import atomic_json, dataset, load_frozen, sha256
from tools.residual_experiment_data import (VARIANT, SCHEMA, calibrate_scale, install_residual,
    json_sha256, latent_batch, load_center, preserved_torch_rng, restore_residual_model,
    state_digest, training_lengths, validate_campaign, validate_scale)
from tools.train_component_world_model import atomic_checkpoint, reconcile_training_journal


def stage_protocol(model, source, parent, provenance, artifact, *, seed, data_seed,
                   updates, batch_size, precision, learning_rate, calibration_seconds=0., resources=None):
    validate_scale(artifact, source=source, provenance=provenance)
    if provenance['mean_training_protocol']['training_manifest_identity'] != source.manifest['identity']:
        raise ValueError('mean and residual training manifests differ')
    mean_protocol = provenance['mean_training_protocol']
    if (data_seed != mean_protocol['data_seed'] or batch_size != mean_protocol['batch_size']
            or mean_protocol['sampler'] != 'uniform_legal_windows_with_replacement_episode_batches'):
        raise ValueError('residual stage must continue mean data seed, batch size and sampler')
    return dict(schema_version=1, kind='frozen_regression_residual_cfm_v1', parent=parent,
        variant=model.dynamics.metadata(), center=provenance,
        scale_artifact_sha256=json_sha256(artifact), initialization_seed=seed, training_seed=seed, data_seed=data_seed,
        sampler_continuation=dict(kind='completed_mean_data_rng_v1',
            inherited_state=provenance['mean_data_rng'], state_identity=provenance['mean_data_rng_identity'],
            completed_updates=provenance['mean_updates'], completed_exposures=provenance['mean_exposures']),
        planned_updates=updates, batch_size=batch_size, learning_rate=learning_rate,
        precision=precision, frozen_mean_precision='fp32', codec_precision='fp32',
        optimizer='Adam', gradient_clip=1., mode='single_step', horizon=1,
        frozen_codec=True, frozen_center=True, training_manifest_identity=source.manifest['identity'],
        frozen_mean_sha256=state_digest(model.dynamics.mean_dynamics.state_dict()),
        sampler='uniform_legal_windows_with_replacement_episode_batches',
        calibration_forward_exposures=artifact['count'],
        calibration_seconds=calibration_seconds,
        resources=resources or dict(device=str(next(model.parameters()).device),
            threads=torch.get_num_threads(), mean_space_chunk=16, mean_time_chunk=32768,
            mean_mlp_chunk=16384, mean_activation_checkpointing=False),
        total_planned_optimization_updates=provenance['mean_updates'] + updates,
        total_planned_training_exposures=provenance['mean_exposures'] + updates * batch_size,
        budget_claim='center pretraining plus residual stage; calibration forwards reported separately; not equal FLOPs',
        role='development_screen_not_converged_evidence',
        source_sha256={name: sha256(ROOT/name) for name in (
            'tools/train_residual_world_model.py', 'tools/residual_experiment_data.py',
            'tools/component_experiment_data.py', 'urban_dreamer/models/residual_dynamics.py',
            'urban_dreamer/models/world_model.py', 'urban_dreamer/models/experimental_dynamics.py',
            'urban_dreamer/learning/cohort_targets.py', 'urban_dreamer/learning/pretrain_cache.py')})


def assert_frozen(model, protocol):
    if (any(p.requires_grad for p in model.flow_projector.parameters())
            or any(p.requires_grad for p in model.dynamics.mean_dynamics.parameters())
            or state_digest(model.flow_projector.state_dict()) != protocol['center']['codec_sha256']
            or state_digest(model.dynamics.mean_dynamics.state_dict()) != protocol['frozen_mean_sha256']):
        raise RuntimeError('frozen codec/mean identity or gradient contract changed')


def read_stage_checkpoint(output):
    output = Path(output)
    seal = json.loads((output/'checkpoint_identity.json').read_text())
    checkpoint = output/seal.get('checkpoint_file', 'latest.pt')
    if sha256(checkpoint) != seal['checkpoint_sha256']:
        raise ValueError('residual checkpoint SHA256 differs from committed identity')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload['update'] != seal['update'] or state_digest(payload['model']) != payload['model_sha256']:
        raise ValueError('residual checkpoint cursor or model identity differs')
    return payload


def run_stage(model, source, capacities, output, protocol, artifact, *, resume=None,
              checkpoint_every=128, stop_after=None):
    """Train one declared stage; stop_after is a bounded test/smoke checkpoint boundary."""
    from urban_dreamer.models.residual_dynamics import residual_fm_loss
    output = Path(output)
    budget, size = protocol['planned_updates'], protocol['batch_size']
    if min(budget, size, checkpoint_every) < 1 or protocol['learning_rate'] <= 0:
        raise ValueError('stage budgets and learning rate must be positive')
    if protocol['precision'] not in ('fp32', 'bf16'):
        raise ValueError('unsupported residual precision')
    device = next(model.parameters()).device
    if protocol['precision'] == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 requires supported CUDA hardware')
    lengths = training_lengths(source)
    if source.manifest['identity'] != protocol['training_manifest_identity']:
        raise ValueError('stage training manifest differs from protocol')
    validate_scale(artifact, source=source, provenance=protocol['center'])
    if json_sha256(artifact) != protocol['scale_artifact_sha256']:
        raise ValueError('scale artifact differs from stage protocol')
    assert_frozen(model, protocol)
    parameters = [p for p in model.dynamics.parameters() if p.requires_grad]
    if not parameters or {id(p) for p in parameters} != {
            id(p) for p in model.dynamics.residual_velocity.parameters()}:
        raise ValueError('residual optimizer must own exactly residual head parameters')
    optimizer = torch.optim.Adam(parameters, lr=protocol['learning_rate'])
    rng = np.random.default_rng(protocol['data_seed'])
    from urban_dreamer.learning.offline_data import _identity
    inherited = protocol['sampler_continuation']
    if (_identity(inherited['inherited_state']) != inherited['state_identity']
            or inherited['inherited_state'] != protocol['center']['mean_data_rng']):
        raise ValueError('inherited mean sampler state identity differs')
    rng.bit_generator.state = copy.deepcopy(inherited['inherited_state'])
    start, exposures, elapsed_before = 0, 0, 0.
    if resume is not None:
        if resume['protocol'] != protocol or resume['calibration'] != artifact:
            raise ValueError('resume protocol/calibration differs; create an explicit new experiment')
        if (resume['schema_version'] != SCHEMA or resume['variant'] != VARIANT
                or resume['seed'] != protocol['initialization_seed']
                or not 0 <= resume['update'] <= budget
                or resume['exposures'] != resume['update'] * size
                or state_digest(resume['model']) != resume['model_sha256']):
            raise ValueError('invalid residual resume schema/model/counters')
        if len(resume['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('residual resume requires the same CUDA RNG device count')
        model.load_state_dict(resume['model'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        # Adam moments follow parameters; noncapturable Adam keeps its scalar
        # step on CPU. Make this explicit rather than assuming GPU load behavior.
        for parameter, state in optimizer.state.items():
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in state:
                    if state[key].shape != parameter.shape:
                        raise ValueError('optimizer moment shape differs from residual parameter')
                    state[key] = state[key].to(device=parameter.device, dtype=parameter.dtype)
            if torch.is_tensor(state.get('step')):
                state['step'] = state['step'].cpu()
        rng.bit_generator.state = copy.deepcopy(resume['data_rng'])
        torch.set_rng_state(resume['torch_rng'].cpu())
        if resume['cuda_rng']:
            torch.cuda.set_rng_state_all([state.cpu() for state in resume['cuda_rng']])
        start, exposures, elapsed_before = resume['update'], resume['exposures'], resume['training_seconds']
        assert_frozen(model, protocol)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('fresh residual stage requires a new empty output directory')
        torch.manual_seed(protocol['initialization_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(protocol['initialization_seed'])
    output.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        reconcile_training_journal(output/'metrics.jsonl', start)
    atomic_json(output/'protocol.json', protocol)
    atomic_json(output/'scale.json', artifact)
    if sha256(output/'scale.json') != protocol['scale_artifact_sha256']:
        raise RuntimeError('serialized scale artifact identity differs')
    target = budget if stop_after is None else min(budget, int(stop_after))
    if target < start:
        raise ValueError('stop boundary precedes the restored cursor')
    model.train()
    model.dynamics.set_online_chunk_profile()
    rows = []
    started = time.perf_counter()
    status = dict(phase='training', pid=os.getpid(), update=start, variant=VARIANT)
    atomic_json(output/'status.json', status)
    amp = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if protocol['precision'] == 'bf16' else nullcontext
    try:
        for update in range(start + 1, target + 1):
            episode = int(rng.choice(len(lengths), p=lengths / lengths.sum()))
            starts = rng.integers(lengths[episode], size=size).tolist()
            batch = source.batch_from_starts(episode, starts, fm_only=True)
            encoded = latent_batch(model, batch, capacities, device)
            optimizer.zero_grad(set_to_none=True)
            with amp():
                loss, metrics, _ = residual_fm_loss(model.dynamics, **encoded, predict_endpoints=False)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite residual loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, protocol['gradient_clip'], error_if_nonfinite=True)
            if any(p.grad is not None for p in model.dynamics.mean_dynamics.parameters()):
                raise RuntimeError('residual loss created frozen mean gradients')
            optimizer.step()
            exposures += size
            row = dict(update=update, exposures=exposures, loss=float(loss.detach()),
                gradient_norm=float(norm), episode=episode, starts=starts,
                demand_group=source.manifest['entries'][episode]['demand_group'],
                training_seconds=elapsed_before + time.perf_counter()-started,
                total_training_and_calibration_seconds=(protocol['center']['mean_training_seconds']
                    + protocol['calibration_seconds'] + elapsed_before + time.perf_counter()-started),
                total_optimization_updates=protocol['center']['mean_updates'] + update,
                total_training_exposures=protocol['center']['mean_exposures'] + exposures,
                **{key: float(value) for key, value in metrics.items()})
            rows.append(row)
            with (output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
            if update == 1 or update % 25 == 0 or update == target:
                status.update(row)
                atomic_json(output/'status.json', status)
                print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
            if update % checkpoint_every == 0 or update == target:
                assert_frozen(model, protocol)
                state = model.state_dict()
                payload = dict(schema_version=SCHEMA, variant=VARIANT,
                    seed=protocol['initialization_seed'], update=update, exposures=exposures,
                    training_seconds=row['training_seconds'], protocol=protocol, calibration=artifact,
                    model=state, model_sha256=state_digest(state), optimizer=optimizer.state_dict(),
                    data_rng=copy.deepcopy(rng.bit_generator.state), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
                generation = output/'checkpoints'/f'update_{update:06d}_{time.time_ns()}.pt'
                generation.parent.mkdir(exist_ok=True)
                if generation.exists():
                    raise ValueError('immutable checkpoint generation already exists')
                atomic_checkpoint(generation, payload)
                seal = dict(update=update, checkpoint_file=str(generation.relative_to(output)),
                            checkpoint_sha256=sha256(generation))
                atomic_json(output/'checkpoint_identity.json', seal)
                alias = output/'latest.alias.tmp'
                if alias.is_symlink():
                    alias.unlink()
                alias.symlink_to(generation.relative_to(output))
                alias.replace(output/'latest.pt')
                status.update(seal)
                if update % 512 == 0:
                    atomic_checkpoint(output/f'model_update_{update:06d}.pt', payload)
        status['phase'] = 'completed' if target == budget else 'checkpointed'
        atomic_json(output/'status.json', status)
        return rows
    except BaseException as exc:
        status.update(phase='failed', error=repr(exc))
        atomic_json(output/'status.json', status)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--experiment-dir', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--mean-checkpoint', type=Path)
    result.add_argument('--mean-checkpoint-sha256')
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--data-seed', type=int, default=20261011)
    result.add_argument('--calibration-seed', type=int, default=20261012)
    result.add_argument('--calibration-windows', type=int, default=4096)
    result.add_argument('--calibration-batch-size', type=int, default=16)
    result.add_argument('--updates', type=int, default=2048)
    result.add_argument('--batch-size', type=int, default=16)
    result.add_argument('--learning-rate', type=float, default=1e-4)
    result.add_argument('--checkpoint-every', type=int, default=128)
    result.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    result.add_argument('--device', default='cuda:1')
    result.add_argument('--threads', type=int, default=2)
    result.add_argument('--gpu-memory-fraction', type=float, default=.06)
    result.add_argument('--resume', action='store_true')
    return result


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (min(args.updates, args.batch_size, args.calibration_windows, args.calibration_batch_size,
            args.threads, args.checkpoint_every) < 1 or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0 or not 0 < args.gpu_memory_fraction <= .06):
        p.error('invalid training/resource budget')
    if not args.resume and (args.mean_checkpoint is None or not args.mean_checkpoint_sha256):
        p.error('fresh training requires the immutable mean checkpoint and SHA256')
    if args.resume and (args.mean_checkpoint is not None or args.mean_checkpoint_sha256 is not None):
        p.error('resume uses the embedded center; do not supply an external mean source')
    if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('fresh residual stage requires a new output directory')
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        if free < max(2*1024**3, args.gpu_memory_fraction*total + 512*1024**2):
            raise RuntimeError('GPU admission lacks required free headroom; existing runs remain untouched')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    if args.precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        p.error('BF16 requires supported CUDA hardware')
    if args.resume:
        parent, manifest_sha = validate_campaign(args.experiment_dir)
        resumed = read_stage_checkpoint(args.output_dir)
        with preserved_torch_rng():
            config, prepared, topology, model, _ = load_frozen(args.experiment_dir,
                device='cpu', fresh_dynamics_seed=resumed['protocol']['center']['mean_seed'])
        restore_residual_model(model, resumed, parent)
        model.to(device)
        provenance, artifact = resumed['protocol']['center'], resumed['calibration']
        calibration_seconds = resumed['protocol']['calibration_seconds']
        if (provenance['manifest_sha256'] != manifest_sha
                or artifact['seed'] != args.calibration_seed or artifact['count'] != args.calibration_windows
                or artifact['batch_size'] != args.calibration_batch_size):
            raise ValueError('resume campaign/calibration configuration differs')
    else:
        resumed = None
        config, prepared, topology, model, parent, provenance = load_center(
            args.experiment_dir, args.mean_checkpoint, args.mean_checkpoint_sha256, device=device)
    source = dataset(args.experiment_dir, config, prepared, topology, split='train', horizon=1)
    if source.manifest['identity'] != provenance['mean_training_protocol']['training_manifest_identity']:
        raise ValueError('mean and residual training manifests differ')
    if not args.resume:
        calibration_started = time.perf_counter()
        artifact = calibrate_scale(model, source, topology.network.capacities, provenance,
            count=args.calibration_windows, seed=args.calibration_seed, batch_size=args.calibration_batch_size)
        calibration_seconds = time.perf_counter() - calibration_started
        install_residual(model, artifact, initialization_seed=args.seed)
    protocol = stage_protocol(model, source, parent, provenance, artifact,
        seed=args.seed, data_seed=args.data_seed, updates=args.updates, batch_size=args.batch_size,
        precision=args.precision, learning_rate=args.learning_rate, calibration_seconds=calibration_seconds,
        resources=dict(device=str(device), threads=args.threads,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            gpu_memory_fraction=args.gpu_memory_fraction, checkpoint_every=args.checkpoint_every,
            mean_space_chunk=16, mean_time_chunk=32768, mean_mlp_chunk=16384,
            mean_activation_checkpointing=False, torch_version=str(torch.__version__)))
    return run_stage(model, source, topology.network.capacities, args.output_dir, protocol, artifact,
                     resume=resumed, checkpoint_every=args.checkpoint_every)


if __name__ == '__main__':
    main()
