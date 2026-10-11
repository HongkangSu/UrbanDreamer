"""Controlled offline dynamics comparisons with a frozen shared codec.

This runner never mutates production checkpoints or uses final test results.
Training schedules are independent of model RNG; every variant sees the same
real replay windows. A development screen is not a converged final experiment.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.component_experiment_data import atomic_json, dataset, load_frozen, sha256
from urban_dreamer.learning.cohort_targets import prepare_cohort_batch, prepare_transition_targets
from urban_dreamer.models.experimental_dynamics import VARIANTS, apply_variant, experimental_loss, variant_metadata
from urban_dreamer.models.world_model import recursive_fm_loss


SUPERVISION = ('single_step', 'independent_cohort', 'recursive_fm', 'sample_ms', 'mean_ms')


def supervised_loss(model, batch, capacities, *, mode, device, steps=4, ms_samples=4,
                    ms_weight=1., prior_weight=.1):
    if mode == 'single_step':
        scoped = prepare_transition_targets(batch, capacities=capacities)
        inputs, targets = batch.flow[:, :1], scoped.transition_target_flow[:, :1]
        actions, events = batch.action[:, :1], batch.action_events.for_step(0)
    else:
        scoped = prepare_cohort_batch(batch, capacities=capacities)
        inputs, targets = scoped.transition_flow[:, :-1], scoped.transition_flow[:, 1:]
        actions, events = scoped.transition_action, scoped.transition_action_events
    size, horizon, edges, _ = inputs.shape
    with torch.no_grad():
        z_inputs = model.encode(inputs.reshape(-1, edges, 8).to(device)).reshape(size, horizon, edges, -1)
        z_targets = model.encode(targets.reshape(-1, edges, 8).to(device)).reshape_as(z_inputs)
    actions, events = actions.to(device), events.to(device)
    if mode in ('single_step', 'independent_cohort'):
        loss, metrics, _ = experimental_loss(model, z_inputs=z_inputs, z_targets=z_targets,
            action=actions, action_events=events, prior_weight=prior_weight, predict_endpoints=False)
    else:
        if variant_metadata(model)['variant'] not in ('graph_fm', 'no_prior_fm', 'random_prior_fm'):
            raise ValueError('recursive supervision comparison currently requires FM variants')
        sequence = torch.cat([z_inputs[:, :1], z_targets], dim=1)
        # The same cohort labels/actions are used in all supervision variants.
        loss, metrics, endpoints = recursive_fm_loss(model.dynamics, z_sequence=sequence,
            action=actions, action_events=events, prior_weight=prior_weight,
            steps=steps, solver='euler')
        if mode in ('sample_ms', 'mean_ms'):
            predictions = [endpoints]
            for _ in range(ms_samples - 1):
                current, trajectory = z_inputs[:, 0], []
                for h in range(horizon):
                    current = model.transition(current, actions[:, h],
                        action_events=events.for_step(h), steps=steps, solver='euler')
                    trajectory.append(current)
                predictions.append(torch.stack(trajectory, dim=1))
            draws = torch.stack(predictions)
            estimate = draws if mode == 'sample_ms' else draws.mean(0)
            ms_loss = (estimate.float() - z_targets.float()).square().mean()
            loss = loss + ms_weight * ms_loss
            metrics['endpoint_ms_loss'] = ms_loss.detach()
    return loss, metrics, size * horizon


def atomic_checkpoint(path, payload):
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def reconcile_training_journal(path, committed_update):
    """Keep failed-attempt evidence separately from the committed learning curve."""
    if not path.exists():
        return
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError('training journal has corruption before its last line') from None
            path.with_name(f'metrics_partial_{time.time_ns()}.txt').write_text(line)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            temporary.replace(path)
    committed = [row for row in rows if row['update'] <= committed_update]
    updates = [row['update'] for row in committed]
    if updates != sorted(set(updates)):
        raise ValueError('committed journal updates are duplicated or out of order')
    if len(committed) != len(rows):
        archive = path.with_name(f'metrics_uncommitted_{time.time_ns()}.jsonl')
        archive.write_text(''.join(json.dumps(row) + '\n' for row in rows if row['update'] > committed_update))
        temporary = path.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(row)+'\n' for row in committed))
        temporary.replace(path)


def train(args):
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        if free < args.gpu_memory_fraction * total + 512 * 1024 ** 2:
            raise RuntimeError('insufficient GPU admission headroom; existing runs remain untouched')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    if args.precision == 'bf16' and device.type != 'cuda':
        raise ValueError('BF16 training requires CUDA')
    if args.mode not in ('single_step', 'independent_cohort') and args.variant not in (
            'graph_fm', 'no_prior_fm', 'random_prior_fm'):
        raise ValueError('recursive comparison requires an FM variant')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / 'latest.pt'
    if (output / 'protocol.json').exists() and not args.resume:
        raise ValueError('existing run requires explicit resume; never overwrite a previous result')
    config, prepared, topology, model, parent = load_frozen(args.experiment_dir,
        device=device, fresh_dynamics_seed=args.seed)
    apply_variant(model, args.variant, seed=args.seed)
    model.dynamics.set_pretraining_execution_profile(space_chunk=args.batch_size,
        time_chunk=32768, mlp_chunk=16384, activation_checkpointing=True)
    horizon = 1 if args.mode == 'single_step' else args.horizon
    source = dataset(args.experiment_dir, config, prepared, topology, split='train', horizon=horizon)
    model.train()
    model.flow_projector.eval()
    optimizer = torch.optim.Adam(model.dynamics.parameters(), lr=args.learning_rate)
    train_rng = np.random.default_rng(args.data_seed)
    lengths = np.asarray([max(0, row['windows'] - horizon + 1) for row in source.manifest['entries']])
    probabilities = lengths / lengths.sum()
    protocol = dict(schema_version=1, parent=parent, variant=variant_metadata(model),
        training_seed=args.seed, data_seed=args.data_seed, mode=args.mode, horizon=horizon,
        batch_size=args.batch_size, planned_updates=args.updates, learning_rate=args.learning_rate,
        precision=args.precision, frozen_codec=True, optimizer='Adam', gradient_clip=1.,
        prior_weight=config.method.prior_loss_weight, recursive_solver_steps=args.steps,
        ms_samples=args.ms_samples, ms_weight=args.ms_weight,
        role='development_screen', sampler='uniform_legal_windows_with_replacement_episode_batches',
        script_sha256=sha256(__file__), dynamics_sha256=sha256(ROOT/'urban_dreamer/models/experimental_dynamics.py'),
        training_manifest_identity=source.manifest['identity'])
    start_update = 0
    exposures = 0
    elapsed_before = 0.
    if args.resume:
        old = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if old['protocol'] != protocol:
            raise ValueError('resume protocol differs; preserve old run and create a declared revision')
        model.load_state_dict(old['model'], strict=True)
        optimizer.load_state_dict(old['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        train_rng.bit_generator.state = old['data_rng']
        torch.set_rng_state(old['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(old['cuda_rng'], device)
        start_update, exposures, elapsed_before = old['update'], old['exposures'], old['training_seconds']
        reconcile_training_journal(output/'metrics.jsonl', start_update)
    else:
        atomic_json(output/'protocol.json', protocol)
    start = time.perf_counter()
    status = dict(phase='training', pid=os.getpid(), update=start_update, output=str(output))
    atomic_json(output/'status.json', status)
    amp = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if args.precision == 'bf16' else nullcontext
    try:
        for update in range(start_update + 1, args.updates + 1):
            episode = int(train_rng.choice(len(lengths), p=probabilities))
            starts = train_rng.integers(0, lengths[episode], size=args.batch_size).tolist()
            batch = source.batch_from_starts(episode, starts, fm_only=args.mode == 'single_step')
            optimizer.zero_grad(set_to_none=True)
            with amp():
                loss, metrics, windows = supervised_loss(model, batch, topology.network.capacities,
                    mode=args.mode, device=device, steps=args.steps, ms_samples=args.ms_samples,
                    ms_weight=args.ms_weight, prior_weight=config.method.prior_loss_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'nonfinite loss at update {update}')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.dynamics.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            exposures += windows
            row = dict(update=update, exposures=exposures, loss=float(loss.detach()),
                gradient_norm=float(gradient), training_seconds=elapsed_before + time.perf_counter()-start,
                episode=int(batch.episode[0]), demand_group=source.manifest['entries'][episode]['demand_group'],
                **{key: float(value) for key, value in metrics.items()})
            with (output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            if update == 1 or update % 25 == 0 or update == args.updates:
                if device.type == 'cuda':
                    row['gpu_peak_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    row['gpu_peak_reserved_mib'] = torch.cuda.max_memory_reserved(device) / 1024 ** 2
                status.update(row)
                atomic_json(output/'status.json', status)
                print(json.dumps(row), flush=True)
            if update % args.checkpoint_every == 0 or update == args.updates:
                atomic_checkpoint(checkpoint_path, dict(schema_version=1, variant=args.variant, seed=args.seed,
                    update=update, exposures=exposures, training_seconds=row['training_seconds'], protocol=protocol,
                    model=model.state_dict(), optimizer=optimizer.state_dict(), data_rng=train_rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None))
                if update % args.milestone_every == 0:
                    atomic_checkpoint(output/f'model_update_{update:06d}.pt', dict(
                        schema_version=1, variant=args.variant, seed=args.seed, update=update,
                        exposures=exposures, protocol=protocol, model=model.state_dict()))
        status['phase'] = 'completed'
        status['checkpoint_sha256'] = sha256(checkpoint_path)
        atomic_json(output/'status.json', status)
    except Exception as exc:
        status.update(phase='failed', error=f'{type(exc).__name__}: {exc}')
        atomic_json(output/'status.json', status)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-seed', type=int, default=20261011)
    p.add_argument('--updates', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--mode', choices=SUPERVISION, default='single_step')
    p.add_argument('--horizon', type=int, default=3)
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--ms-samples', type=int, default=4)
    p.add_argument('--ms-weight', type=float, default=1.)
    p.add_argument('--precision', choices=('fp32','bf16'), default='bf16')
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--gpu-memory-fraction', type=float, default=.08)
    p.add_argument('--checkpoint-every', type=int, default=128)
    p.add_argument('--milestone-every', type=int, default=512)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.updates,args.batch_size,args.horizon,args.steps,args.ms_samples,args.checkpoint_every,args.milestone_every,args.threads)<1:
        p.error('counts and budgets must be positive')
    if not 0 < args.gpu_memory_fraction <= 1 or args.learning_rate <= 0 or args.ms_weight < 0:
        p.error('invalid memory fraction, learning rate or MS weight')
    if args.milestone_every % args.checkpoint_every:
        p.error('milestone-every must be a multiple of checkpoint-every')
    train(args)


if __name__ == '__main__':
    main()
