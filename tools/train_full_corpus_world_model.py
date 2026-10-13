"""Finite full-corpus comparator training; completed epochs are not convergence.

Prepare from the immutable distribution runtime, then invoke THIS tool inside
the prepared source tree. No production checkpoint, schema or runner is edited.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from tools.component_experiment_data import atomic_json, load_frozen, sha256
from tools.component_prediction_metrics import evaluate_prediction_samples, summarize_state_records
from tools.evaluate_component_world_models import endpoint_target, estimate_training_scales, physical, select_conditions
from tools.train_component_world_model import atomic_checkpoint, reconcile_training_journal
from urban_dreamer.learning.action_events import SparseActionEvents
from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.learning.offline_data import _identity
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.models.experimental_dynamics import apply_variant, experimental_loss, variant_metadata

CHECKPOINT_SHA = '2f63f0b2a5fe18f3a3bd9be27ffd1e63a2755464e7f2944e1227334f0ffe6af6'
MANIFEST_SHA = '34169f27525094b6103b5a0556f64c14c3e176d7985f19fd0d5543bc68c4c30e'
MANIFEST_ID = '5254f01c7863d3e8524d4cf118983fae1bfbe755832ece48380acbdd11eab443'


def tensor_identity(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def manifest_counts(source, batch_size=64):
    if (source.get('status') != 'complete' or source.get('identity') != _identity(
            {k: v for k, v in source.items() if k != 'identity'})):
        raise ValueError('incomplete or invalid full manifest')
    result, groups = {}, {}
    for split in ('train', 'validation'):
        entries = [row for row in source['entries'] if row['split'] == split]
        if not entries or any(row['windows'] < 1 for row in entries):
            raise ValueError('empty split or invalid single-step coverage')
        groups[split] = {row['demand_group'] for row in entries}
        result[split] = dict(groups=len(groups[split]), episodes=len(entries),
            windows=sum(row['windows'] for row in entries),
            updates_per_epoch=sum((row['windows']+batch_size-1)//batch_size for row in entries))
    if groups['train'] & groups['validation']:
        raise ValueError('train and development demand groups overlap')
    return result


def prepare(campaign, frozen_campaign, frozen_source):
    """Copy exact completed snapshot/data identity and reviewed runtime; no model load."""
    campaign = Path(campaign).resolve()
    if campaign.exists():
        raise ValueError('new campaign required; never overwrite prepared evidence')
    frozen_campaign, frozen_source = map(Path, (frozen_campaign, frozen_source))
    parent = json.loads((frozen_campaign/'protocol.json').read_text())
    checkpoint = frozen_campaign/'snapshot/checkpoints/latest.pt'
    manifest = Path(parent['source_manifest'])
    if sha256(checkpoint) != CHECKPOINT_SHA or sha256(manifest) != MANIFEST_SHA:
        raise ValueError('completed reference or full data manifest identity changed')
    source_manifest = json.loads((frozen_source.parent/'source_manifest.json').read_text())
    for name, expected in source_manifest.items():
        if sha256(frozen_source/name) != expected:
            raise ValueError(f'frozen runtime changed: {name}')
    source = json.loads(manifest.read_text())
    counts = manifest_counts(source)
    if source['identity'] != MANIFEST_ID or counts['train']['windows'] != 2605729:
        raise ValueError('unexpected full corpus')
    (campaign/'snapshot/checkpoints').mkdir(parents=True)
    shutil.copyfile(checkpoint, campaign/'snapshot/checkpoints/latest.pt')
    shutil.copyfile(frozen_campaign/'snapshot/run_config.toml', campaign/'snapshot/run_config.toml')
    shutil.copyfile(manifest, campaign/'full_manifest.json')
    shutil.copytree(frozen_source, campaign/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(__file__, campaign/'source/tools/train_full_corpus_world_model.py')
    hashes = {str(p.relative_to(campaign/'source')): sha256(p)
              for p in sorted((campaign/'source').rglob('*.py'))}
    atomic_json(campaign/'source_manifest.json', hashes)
    atomic_json(campaign/'protocol.json', dict(schema_version=1, role='full_corpus_development_comparators',
        checkpoint_sha256=CHECKPOINT_SHA, source_manifest_sha256=MANIFEST_SHA,
        source_manifest_identity=MANIFEST_ID, config_sha256=sha256(campaign/'snapshot/run_config.toml'),
        full_manifest_identity=MANIFEST_ID, counts=counts,
        parent_component_protocol=parent, runtime_manifest_sha256=sha256(campaign/'source_manifest.json'),
        intended_variants=['regression', 'gaussian'], comparison_reference='completed immutable pretrained FM',
        convergence='not established by prescribed epoch completion',
        readout='completed shared readout imported explicitly; frozen with codec'))
    return campaign


def slice_batch(batch, begin, end):
    size = len(batch.flow)
    if not 0 <= begin < end <= size:
        raise ValueError('invalid microbatch bounds')
    changes = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, SparseActionEvents):
            selected = value.select((value.batch_index >= begin) & (value.batch_index < end))
            value = replace(selected, batch_index=selected.batch_index-begin, batch_size=end-begin)
        elif torch.is_tensor(value):
            if value.ndim < 1 or value.shape[0] != size:
                raise ValueError(f'non-aligned batch tensor {field.name}')
            value = value[begin:end]
        elif isinstance(value, tuple):
            if len(value) != size:
                raise ValueError(f'non-aligned batch tuple {field.name}')
            value = value[begin:end]
        changes[field.name] = value
    return replace(batch, **changes)


def encoded_inputs(model, batch, capacities, device):
    scoped = prepare_transition_targets(batch, capacities=capacities)
    size, _, edges, _ = batch.flow.shape
    # Explicitly disable caller autocast: codec coordinates match original FP32.
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        initial = model.encode(batch.flow[:, 0].to(device).float()).reshape(size, 1, edges, -1)
        target = model.encode(scoped.transition_target_flow[:, 0].to(device).float()).reshape_as(initial)
    return initial, target, scoped


def objective(model, batch, capacities, *, device, precision):
    initial, target, _ = encoded_inputs(model, batch, capacities, device)
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
    with amp:
        return experimental_loss(model, z_inputs=initial, z_targets=target,
            action=batch.action[:, :1].to(device), action_events=batch.action_events.for_step(0).to(device),
            prior_weight=.1, predict_endpoints=False)[:2]


def update(model, optimizer, batch, capacities, *, device, precision, microbatch_size):
    optimizer.zero_grad(set_to_none=True)
    values = {}
    size = len(batch.flow)
    for begin in range(0, size, microbatch_size):
        part = slice_batch(batch, begin, min(size, begin+microbatch_size))
        loss, metrics = objective(model, part, capacities, device=device, precision=precision)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite comparator objective')
        weight = len(part.flow)/size
        (loss*weight).backward()
        for key, value in dict(loss=loss.detach(), **metrics).items():
            values[key] = values.get(key, 0.)+float(value)*weight
    values['gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(model.dynamics.parameters(),
        1., error_if_nonfinite=True))
    optimizer.step()
    return values


@contextmanager
def validation_guard(model, source):
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == 'cuda' else []
    cursor = source.state_dict()
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda:
            torch.cuda.set_rng_state_all(cuda)
        for module, mode in modes.items():
            module.training = mode
        if source.state_dict() != cursor:
            raise RuntimeError('validation mutated training cursor')


@torch.no_grad()
def validate(model, training, validation, conditions, capacities, limits, scales, *, device, seed, samples):
    rows, native = [], []
    with validation_guard(model, training):
        torch.manual_seed(seed)
        for item in conditions:
            batch = validation.batch_from_starts(item['episode_index'], [item['start']], fm_only=True)
            initial, target, scoped = encoded_inputs(model, batch, capacities, device)
            loss, metrics, _ = experimental_loss(model, z_inputs=initial, z_targets=target,
                action=batch.action.to(device), action_events=batch.action_events.to(device), predict_endpoints=False)
            native.append(dict(loss=float(loss), **{key: float(value) for key, value in metrics.items()}))
            draws = [physical(model.decode(model.transition(initial[:, 0], batch.action[:, 0].to(device),
                action_events=batch.action_events.for_step(0).to(device))), capacities, limits).cpu()
                for _ in range(samples if variant_metadata(model)['variant'] == 'gaussian' else 1)]
            arrays = validation._episode(item['episode_index'])
            following = torch.as_tensor(arrays['flow'][item['start']+2]).unsqueeze(0)
            truth = endpoint_target(scoped.transition_target_flow[:, 0], following, capacities.cpu(), limits.cpu())
            rows.extend(evaluate_prediction_samples(torch.stack(draws), truth, scales=scales,
                state_metadata=[item])['states'])
    return dict(role='fixed_development_only', states=len(rows), samples=samples,
        native={key: float(np.mean([row[key] for row in native])) for key in native[0]},
        physical=summarize_state_records(rows), records=rows)


def checkpoint_payload(model, optimizer, source, data_rng, *, update_count, exposures, protocol):
    return dict(schema_version=1, protocol=protocol, model=model.state_dict(), optimizer=optimizer.state_dict(),
        dataset=source.state_dict(), data_rng=data_rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == 'cuda' else [],
        update=update_count, exposures=exposures, epochs_completed=source.epochs_completed)


def fit(model, source, optimizer, args, protocol, *, capacities, validation_callback=None):
    """Only successful optimizer steps advance the durable sampler cursor."""
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path, journal = output/'latest.pt', output/'metrics.jsonl'
    rng = np.random.default_rng(args.data_seed)
    count, exposures = 0, 0
    def save():
        atomic_checkpoint(path, checkpoint_payload(model, optimizer, source, rng,
            update_count=count, exposures=exposures, protocol=protocol))
    if args.resume:
        old = torch.load(path, map_location='cpu', weights_only=True)
        if old['protocol'] != protocol:
            raise ValueError('resume protocol changed; create a declared continuation')
        model.load_state_dict(old['model'], strict=True)
        optimizer.load_state_dict(old['optimizer'])
        source.load_state_dict(old['dataset'])
        rng.bit_generator.state = old['data_rng']
        torch.set_rng_state(old['torch_rng'])
        if old['cuda_rng']:
            torch.cuda.set_rng_state_all(old['cuda_rng'])
        count, exposures = old['update'], old['exposures']
        reconcile_training_journal(journal, count)
        rows = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        if [row['update'] for row in rows] != list(range(1, count+1)):
            raise ValueError('committed training journal has missing updates')
    else:
        if path.exists() or (output/'protocol.json').exists():
            raise ValueError('existing run requires explicit resume')
        atomic_json(output/'protocol.json', protocol)
        save()
    started = time.perf_counter()
    try:
        while source.epochs_completed < args.epochs:
            for batch in source.iter_epoch(args.batch_size, rng):
                metrics = update(model, optimizer, batch, capacities, device=args.device,
                    precision=args.precision, microbatch_size=args.microbatch_size)
                count += 1
                exposures += len(batch.flow)
                row = dict(update=count, exposures=exposures, epoch=source.epochs_completed,
                    episode=int(batch.episode[0]), starts=batch.start_index.tolist(), **metrics)
                if validation_callback and count % args.validation_every == 0:
                    row['validation'] = validation_callback()
                with journal.open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                if count % args.checkpoint_every == 0:
                    save()
                if count == 1 or count % 25 == 0:
                    atomic_json(output/'status.json', dict(phase='training', update=count,
                        exposures=exposures, epochs_completed=source.epochs_completed,
                        segment_seconds=time.perf_counter()-started, last_loss=metrics['loss']))
                if getattr(args, 'stop_after_updates', None) is not None and count >= args.stop_after_updates:
                    save()
                    atomic_json(output/'status.json', dict(phase='interrupted_for_mechanical_smoke', update=count))
                    return
            # Iterator exhaustion, rather than last-batch arithmetic, commits the epoch.
            save()
            if validation_callback:
                atomic_json(output/f'validation_epoch_{source.epochs_completed:03d}.json', validation_callback())
        expected_exposures = args.epochs*sum(row['windows'] for row in source.manifest['entries'])
        expected_updates = args.epochs*sum((row['windows']+args.batch_size-1)//args.batch_size
            for row in source.manifest['entries'])
        if exposures != expected_exposures or count != expected_updates:
            raise RuntimeError('completed traversal does not match unique coverage/update accounting')
        atomic_json(output/'status.json', dict(phase='prescribed_epochs_completed', update=count,
            exposures=exposures, epochs_completed=source.epochs_completed,
            convergence_established=False, checkpoint_sha256=sha256(path)))
    except Exception as exc:
        atomic_json(output/'status.json', dict(phase='failed', error=f'{type(exc).__name__}: {exc}',
            update=count, exposures=exposures, durable_checkpoint_sha256=sha256(path)))
        raise


def run(args):
    campaign = args.campaign.resolve()
    campaign_protocol = json.loads((campaign/'protocol.json').read_text())
    if args.batch_size != 64:
        raise ValueError('formal full-corpus campaign requires original logical B64; reduce microbatch instead')
    if ROOT != (campaign/'source').resolve():
        raise ValueError('run the prepared frozen source tool, not the mutable workspace tool')
    for name, digest in json.loads((campaign/'source_manifest.json').read_text()).items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'runtime source changed: {name}')
    if sha256(campaign/'full_manifest.json') != MANIFEST_SHA:
        raise ValueError('full training manifest changed')
    torch.set_num_threads(args.threads)
    if args.precision == 'bf16' and torch.device(args.device).type != 'cuda':
        raise ValueError('BF16 requires CUDA')
    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_device(args.device)
        free, total = torch.cuda.mem_get_info()
        if free < args.gpu_memory_fraction*total+512*1024**2:
            raise RuntimeError('GPU admission headroom unavailable; existing jobs remain untouched')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    config, prepared, topology, model, _ = load_frozen(campaign, device=args.device, fresh_dynamics_seed=args.seed)
    raw = torch.load(campaign/'snapshot/checkpoints/latest.pt', map_location='cpu', weights_only=True)
    metrics = raw['training_state']['metrics']
    if (raw['phase'] != 'online_training' or raw['next_unit'] != 0
            or metrics['world_model_initialization_updates'] != 49636
            or metrics['world_model_initialization_samples_seen'] != 2605729
            or metrics['world_model_initialization_epochs_completed'] != 1
            or metrics['online_episodes_completed'] != 0):
        raise ValueError('shared asset source is not the completed original pretraining')
    shared = {key.removeprefix('active_vehicle_fraction_readout.'): value
        for key, value in raw['training_state']['models']['world_model'].items()
        if key.startswith('active_vehicle_fraction_readout.')}
    model.active_vehicle_fraction_readout.load_state_dict(shared, strict=True)
    del raw
    codec_sha = tensor_identity(model.flow_projector.state_dict())
    readout_sha = tensor_identity(model.active_vehicle_fraction_readout.state_dict())
    apply_variant(model, args.variant, seed=args.seed)
    expected_parameters = {'regression': 8140368, 'gaussian': 8156816}
    if variant_metadata(model)['parameters'] != expected_parameters[args.variant]:
        raise ValueError('comparator parameter budget differs from frozen runtime')
    model.train()
    model.flow_projector.eval().requires_grad_(False)
    model.active_vehicle_fraction_readout.eval().requires_grad_(False)
    if args.inspect_assets_only:
        atomic_json(campaign/f'asset_identity_{args.variant}_seed{args.seed}.json', dict(
            role='CPU_asset_loading_identity_only_not_trained_baseline', checkpoint_sha256=CHECKPOINT_SHA,
            codec_sha256=codec_sha, readout_sha256=readout_sha, variant=variant_metadata(model),
            device=args.device, frozen_codec=True, frozen_readout=True))
        return
    model.dynamics.set_pretraining_execution_profile(space_chunk=args.microbatch_size,
        time_chunk=32768, mlp_chunk=16384, activation_checkpointing=True)
    source = CompactPretrainDataset(config, prepared, topology, 1,
        cache_manifest_path=campaign/'full_manifest.json', split='train')
    validation = CompactPretrainDataset(config, prepared, topology, 1,
        cache_manifest_path=campaign/'full_manifest.json', split='validation')
    conditions = select_conditions(validation.manifest['entries'], per_group=2,
        max_groups=args.validation_groups, horizon=2, seed=args.validation_seed)
    capacities = torch.as_tensor(topology.network.capacities, dtype=torch.float32)
    limits = torch.as_tensor(topology.network.speed_limits_mps, dtype=torch.float32)
    scales, scales_protocol = estimate_training_scales(source, capacities, limits,
        seed=args.validation_seed, max_groups=32)
    optimizer = torch.optim.AdamW(model.dynamics.parameters(), lr=1e-4, weight_decay=.01)
    protocol = dict(schema_version=1, campaign=campaign_protocol, variant=variant_metadata(model),
        seed=args.seed, data_seed=args.data_seed, planned_epochs=args.epochs,
        batch_size=args.batch_size, microbatch_size=args.microbatch_size, precision=args.precision,
        sampler='finite_unique_shuffled_episode_windows', optimizer='AdamW', learning_rate=1e-4,
        weight_decay=.01, gradient_clip=1., target='full_inputs_full_events_hybrid_Ct',
        codec_sha256=codec_sha, readout_sha256=readout_sha,
        training_manifest_identity=source.manifest['identity'], conditions=conditions, scales=scales_protocol,
        validation_seed=args.validation_seed, validation_every=args.validation_every,
        validation_samples=args.validation_samples, checkpoint_every=args.checkpoint_every,
        source_manifest_sha256=sha256(campaign/'source_manifest.json'),
        convergence='not implied by planned epoch completion')
    def evaluate():
        return validate(model, source, validation, conditions, capacities, limits, scales,
            device=args.device, seed=args.validation_seed, samples=args.validation_samples)
    fit(model, source, optimizer, args, protocol, capacities=capacities, validation_callback=evaluate)
    if tensor_identity(model.flow_projector.state_dict()) != codec_sha or tensor_identity(
            model.active_vehicle_fraction_readout.state_dict()) != readout_sha:
        raise RuntimeError('frozen shared codec/readout changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--inspect-assets-only', action='store_true')
    parser.add_argument('--frozen-campaign', type=Path, default=ROOT/'outputs/component_experiments_20261011')
    parser.add_argument('--frozen-source', type=Path, default=ROOT/'outputs/pretrained_evaluation_20261012/distribution_v1/source')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--variant', choices=('regression', 'gaussian'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-seed', type=int, default=20261013)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--microbatch-size', type=int, default=8)
    parser.add_argument('--validation-groups', type=int, default=64)
    parser.add_argument('--validation-seed', type=int, default=2026101301)
    parser.add_argument('--validation-samples', type=int, default=8)
    parser.add_argument('--validation-every', type=int, default=4096)
    parser.add_argument('--checkpoint-every', type=int, default=128)
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.06)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-updates', type=int, help='mechanical interruption only; not a quality run')
    args = parser.parse_args()
    if args.prepare_only:
        print(prepare(args.campaign, args.frozen_campaign, args.frozen_source))
        return
    if not args.variant or (not args.output_dir and not args.inspect_assets_only):
        parser.error('training requires --output-dir and --variant')
    if min(args.epochs, args.batch_size, args.microbatch_size, args.validation_groups,
            args.validation_samples, args.validation_every, args.checkpoint_every, args.threads) < 1:
        parser.error('counts must be positive')
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error('invalid GPU memory fraction')
    run(args)


if __name__ == '__main__':
    main()
