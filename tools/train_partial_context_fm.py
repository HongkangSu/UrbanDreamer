"""Pilot-first partial FM adaptation with independently controlled source/reference.

The CLI deliberately runs TRAIN-ONLY timing/gradient pilots, never optimization
or validation/test forecasting. Its helpers support the later budgeted trainer.
Training reference_proj requires autograd through the frozen five-block prefix:
final-feature caches cannot be reused for this phase.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = Path(os.environ.get('PARTIAL_FM_FROZEN_ROOT', ROOT)).resolve()
sys.path.insert(0, str(FROZEN_ROOT))
import numpy as np
import torch
from torch import nn

from tools.component_experiment_data import atomic_json, load_frozen, sha256
from tools.evaluate_pretrained_distribution import CHECKPOINT_SHA256
from tools.evaluate_soft_prior_fm import MeanOverridePrior
from tools.train_direct_context_fm import DirectResidual, DirectVelocityHead
from tools.train_nonlinear_fm_residual import stratified_draws
from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.learning.offline_data import archive_content_hash
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.models.world_model import ConditionEncoding

ARMS = {'B': (False, False), 'C1': (True, False), 'C2': (True, True)}
PREFIXES = ('dynamics.transformer.layers.5.', 'dynamics.reference_proj.', 'dynamics.flow_head.')
EXPECTED_PARAMETERS = 1_414_528


def allowed_name(name):
    return name.startswith(PREFIXES)


def tensor_digest(items):
    result = hashlib.sha256()
    for name, value in sorted(items):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def parameter_digest(model, residual, *, frozen_only=False):
    values = [(name, p) for name, p in model.named_parameters()
              if not frozen_only or not allowed_name(name)]
    if not frozen_only:
        values += [('residual.' + name, p) for name, p in residual.named_parameters()]
    return tensor_digest(values)


def configure_trainable(model, residual):
    """Freeze weights, not the gradient path from the last block to reference."""
    model.eval().requires_grad_(False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(allowed_name(name))
    residual.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad] + list(residual.parameters())
    count = sum(p.numel() for p in params)
    if count != EXPECTED_PARAMETERS:
        raise ValueError(f'expected {EXPECTED_PARAMETERS} trainable parameters, got {count}')
    return params


@contextmanager
def partial_context(dynamics, residual, arm, scales, *, background=None):
    """All arms receive hint b; source and reference can differ (especially C1).

    ``background`` is an optional detached cached b for observed training states.
    Generated rollouts recompute b from their current state. No future target is
    available to this context manager or its forward prehook.
    """
    if arm not in ARMS:
        raise ValueError('arm must be B, C1 or C2')
    source_on, reference_on = ARMS[arm]
    context = {}
    old_head = dynamics.flow_head
    had_start = 'prior_start' in dynamics.__dict__
    instance_start = dynamics.__dict__.get('prior_start')

    def start(this, z, condition, noise=None, *, detach_prior_increment=False):
        if not isinstance(detach_prior_increment, bool):
            raise ValueError('detach flag must be boolean')
        with torch.no_grad():
            b = this.graph_prior(z, condition) if background is None else background
        if b.shape != z.shape:
            raise ValueError('cached background shape differs from current states')
        b = b.detach()
        mean = z + b if source_on else z
        reference = z + b if reference_on else z
        # This projection MUST stay outside no_grad: the frozen prefix also
        # MUST retain the graph carrying its gradient back to this parameter.
        cached = ConditionEncoding(condition.edge_tokens, this.reference_proj(reference))
        context.update(z=z, hint=b, mean=mean)
        increment = b if source_on else torch.zeros_like(b)
        return mean + this.graph_prior.sample_noise(z, noise), increment, cached

    def capture(module, args, kwargs):
        context['xt'] = kwargs['packed_latents']
        context['time'] = kwargs['flow_time']

    dynamics.prior_start = MethodType(start, dynamics)
    hook = dynamics.register_forward_pre_hook(capture, with_kwargs=True)
    dynamics.flow_head = DirectVelocityHead(old_head, residual, context, scales)
    try:
        yield context
    finally:
        hook.remove()
        dynamics.flow_head = old_head
        if had_start:
            dynamics.prior_start = instance_start
        else:
            del dynamics.__dict__['prior_start']


def cfm_loss(model, residual, arm, batch, scales):
    """Independent H1 CFM, full velocity target, whole graph retained."""
    z, target = batch['z'].detach(), batch['target'].detach()
    if z.shape != target.shape or z.ndim != 4 or z.shape[1] != 1:
        raise ValueError('observed z/target must align as B,1,E,64')
    condition = ConditionEncoding(batch['condition'].detach())
    with partial_context(model.dynamics, residual, arm, scales, background=batch['b'].detach()):
        source, _, cached = model.dynamics.prior_start(z, condition, batch['noise'])
        t = batch['time'][:, :, None, None]
        xt = (1 - t) * source + t * target
        velocity = model.dynamics(batch['action'], packed_latents=xt,
                                  flow_time=batch['time'], condition=cached).latent_velocity
        target_velocity = (target - source).detach()
        loss = (velocity.float() - target_velocity.float()).square().mean()
    return loss


def gradient_proof(model, residual):
    named = dict(model.named_parameters())
    frozen_grads = [name for name, p in named.items() if not allowed_name(name) and p.grad is not None]
    if frozen_grads:
        raise ValueError('frozen parameters received gradients: ' + ', '.join(frozen_grads[:4]))
    required = ('dynamics.reference_proj.weight', 'dynamics.flow_head.weight',
                'dynamics.transformer.layers.5.norm_space.scale')
    norms = {}
    for name in required:
        gradient = named[name].grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise ValueError('missing/nonfinite trainable gradient: ' + name)
        norms[name] = float(torch.linalg.vector_norm(gradient))
        if norms[name] == 0:
            raise ValueError('zero trainable gradient: ' + name)
    if residual.output.weight.grad is None:
        raise ValueError('new residual output has no gradient')
    norms['residual.output.weight'] = float(torch.linalg.vector_norm(residual.output.weight.grad))
    return dict(frozen_parameter_gradients_absent=True, nonzero_gradient_norms=norms,
                time_attention_qk_note='T=1 bypasses temporal Q/K; their rows are structurally inactive')


def frozen_window_batch(model, batch, capacities):
    """Only codec, prior and action conditioning may be cached; no Transformer H."""
    device = next(model.parameters()).device
    scoped = prepare_transition_targets(batch, capacities=capacities)
    with torch.no_grad():
        z = model.encode(batch.flow[:, 0].to(device))[:, None]
        target = model.encode(scoped.transition_target_flow[:, 0].to(device))[:, None]
        action = batch.action[:, :1].to(device)
        events = batch.action_events.for_step(0).to(device)
        condition = model.dynamics.encode_condition(action, action_events=events)
        b = model.dynamics.graph_prior(z, condition)
    return dict(z=z.detach(), target=target.detach(), b=b.detach(),
                action=action.detach(), condition=condition.edge_tokens.detach())


def selected_state(model, residual):
    """Serialize only the explicitly adapted modules plus the new residual."""
    return dict(model={name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                       if allowed_name(name)},
                residual={name: value.detach().cpu().clone() for name, value in residual.state_dict().items()})


def run_pilot(models, residuals, params, batch, scales, *, repeats=3, warmups=2):
    if repeats < 1 or warmups < 0:
        raise ValueError('positive repeats and nonnegative warmups required')
    device = batch['z'].device
    before = {arm: parameter_digest(models[arm], residuals[arm]) for arm in ARMS}
    rng_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    records = []
    for repeat in range(warmups + repeats):
        for arm in ARMS:
            for p in params[arm]:
                p.grad = None
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            loss = cfm_loss(models[arm], residuals[arm], arm, batch, scales)
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite pilot loss')
            loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            proof = gradient_proof(models[arm], residuals[arm])
            if repeat >= warmups:
                records.append(dict(arm=arm, repeat=repeat - warmups, seconds=elapsed,
                                    peak_gpu_mib=torch.cuda.max_memory_allocated(device) / 1024**2
                                    if device.type == 'cuda' else 0., gradient_proof=proof))
    for arm in ARMS:
        for p in params[arm]:
            p.grad = None
        if parameter_digest(models[arm], residuals[arm]) != before[arm]:
            raise ValueError('timing pilot changed parameters')
    if not torch.equal(torch.random.get_rng_state(), rng_before):
        raise ValueError('timing pilot consumed CPU RNG')
    if device.type == 'cuda' and not torch.equal(torch.cuda.get_rng_state(device), cuda_before):
        raise ValueError('timing pilot consumed CUDA RNG')
    return dict(records=records, optimizer_steps=0, warmups=warmups, measured_repeats=repeats,
                parameters_unchanged=True, gradients_cleared=True, rng_unchanged=True,
                batch_graphs=int(batch['z'].shape[0]), roads=int(batch['z'].shape[-2]))


def run(args):
    if not args.pilot_only:
        raise ValueError('This entry point is train-only timing/gradient pilot; pass --pilot-only.')
    if FROZEN_ROOT != (args.output / 'source').resolve():
        raise ValueError('freeze this study source under output/source first')
    if (args.output / 'pilot_summary.json').exists():
        raise ValueError('preserve completed pilot; choose a fresh output')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_manifest = json.loads((args.output / 'source_manifest.json').read_text())
    for name, expected in source_manifest.items():
        if sha256(FROZEN_ROOT / name) != expected:
            raise ValueError('frozen source changed')
    checkpoint = args.experiment / 'snapshot/checkpoints/latest.pt'
    references = dict(checkpoint=checkpoint, split=args.split, basis=args.basis, weak_head=args.weak_head)
    hashes = {key: sha256(path) for key, path in references.items()}
    if hashes['checkpoint'] != CHECKPOINT_SHA256:
        raise ValueError('completed pretrained checkpoint required')
    split = json.loads(args.split.read_text())
    entry = split['train'][0]
    starts = entry['starts'][:args.batch_graphs]
    if len(starts) != args.batch_graphs or not starts:
        raise ValueError('pilot batch exceeds first fixed training entry')
    protocol = dict(status='prespecified_before_pilot', role='train-only timing/gradient pilot; NO optimizer or scoring',
                    arms={k: dict(source_on=v[0], reference_on=v[1], hint='same frozen b') for k, v in ARMS.items()},
                    reference_hashes=hashes, seed=args.seed, quarter=0, starts=starts,
                    cache_scope='only Z/target/b/edge-conditioning/action; frozen prefix must retain autograd',
                    trainable_parameters_per_arm=EXPECTED_PARAMETERS, allowed_prefixes=PREFIXES,
                    trainable_breakdown=dict(last_block=1315840, reference_proj=16384, flow_head=16448, direct_residual=65856),
                    full_epoch_graph_examples=3072, batch_graphs=args.batch_graphs,
                    warmups=args.warmups, repeats=args.repeats, device=args.device, threads=args.threads,
                    validation_test_predictions=False, normalizers='positive dummy scales1 for timing only; train-only RMS needed before adaptation')
    atomic_json(args.output / 'protocol.json', protocol)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        if free < args.gpu_memory_fraction * total + 256 * 1024**2:
            raise RuntimeError('insufficient GPU admission headroom')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    config, prepared, topology, base, _ = load_frozen(args.experiment, device=device)
    base.eval().requires_grad_(False)
    base.set_online_chunk_profile()
    with np.load(args.basis) as a:
        basis, values = a['basis'], a['eigenvalues']
    with np.load(args.weak_head) as a:
        coefficients = a['coefficients']
    original_prior = base.dynamics.graph_prior
    base.dynamics.graph_prior = MeanOverridePrior(original_prior,
        operator=(basis * (1 / (1 + .1 * values))) @ basis.T, coefficients=coefficients, state_only=True)
    if base.dynamics.graph_prior.rank != 16:
        raise ValueError('original rank16 noise required')
    data = CompactPretrainDataset(config, prepared, topology, 1,
                                  cache_manifest_path=Path(split['cache_manifest_path']), split='train')
    index = {e['cache_path']: i for i, e in enumerate(data.manifest['entries'])}[entry['cache_path']]
    batch = data.batch_from_starts(index, starts, materialization_horizon=1, fm_only=True)
    capacities = torch.tensor(topology.network.capacities, dtype=torch.float32).clamp_min(1)
    cached = frozen_window_batch(base, batch, capacities)
    times, noise, schedule = stratified_draws(args.seed, 'train', entry, starts, 0, original_prior.rank)
    cached.update(time=times.to(device), noise=noise.to(cached['z']))
    models, residuals, params = {}, {}, {}
    for arm in ARMS:
        models[arm] = copy.deepcopy(base)
        torch.manual_seed(args.seed)
        residuals[arm] = DirectResidual().to(device)
        params[arm] = configure_trainable(models[arm], residuals[arm])
    for arm in ('C1', 'C2'):
        for name, value in residuals['B'].state_dict().items():
            torch.testing.assert_close(value, residuals[arm].state_dict()[name], rtol=0, atol=0)
    result = run_pilot(models, residuals, params, cached, dict(z=1., b=1., relative=1.),
                       repeats=args.repeats, warmups=args.warmups)
    per_arm = {arm: float(np.median([r['seconds'] for r in result['records'] if r['arm'] == arm])) for arm in ARMS}
    examples = sum(len(e['starts']) for e in split['train']) * 4
    steps = (examples + args.batch_graphs - 1) // args.batch_graphs
    result.update(status='pilot_complete', reference_hashes=hashes, median_seconds_per_arm=per_arm,
                  graph_examples_per_epoch=examples, optimizer_steps_per_epoch_per_arm=steps,
                  estimated_triple_epoch_seconds=steps * sum(per_arm.values()),
                  budget_note='rough extrapolation excludes encoding, I/O, optimizer step and validation; choose budget before any scores',
                  selected_train_entry=dict(cache_path=entry['cache_path'], starts=starts), schedule=schedule)
    if archive_content_hash(Path(entry['cache_path'])) != entry['cache_content_hash']:
        raise ValueError('pilot replay changed')
    for key, path in references.items():
        if sha256(path) != hashes[key]:
            raise ValueError('reference changed')
    for name, expected in source_manifest.items():
        if sha256(FROZEN_ROOT / name) != expected:
            raise ValueError('frozen source changed')
    imports = {name: str(Path(mod.__file__).resolve()) for name, mod in sys.modules.items()
               if name.startswith(('tools.', 'urban_dreamer.')) and getattr(mod, '__file__', None)}
    if any(not Path(path).is_relative_to(FROZEN_ROOT) for path in imports.values()):
        raise ValueError('live code contaminated frozen pilot')
    atomic_json(args.output / 'import_audit.json', imports)
    atomic_json(args.output / 'pilot_summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment', 'split', 'basis', 'weak-head', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pilot-only', action='store_true', default=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.08)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-graphs', type=int, default=2)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261021)
    run(parser.parse_args())
