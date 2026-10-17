"""Bounded, private recursive FM forward/backward benchmark; no training resume."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from urban_dreamer.learning.action_events import ActionEvent, pack_action_events
from urban_dreamer.models import world_model as current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('baseline', 'optimized', 'checkpointed'), required=True)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--edges', type=int, default=128)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--memory-fraction', type=float, default=.25)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.edges, args.batch, args.steps, args.repeats, args.checkpoint_interval) < 1 or args.warmups < 0:
        parser.error('positive dimensions, repeats and checkpoint interval required')
    if not 0 < args.memory_fraction <= 1:
        parser.error('memory fraction must be in (0,1]')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        if free < args.memory_fraction * total + 512 * 1024**2:
            raise RuntimeError('insufficient free memory for isolated bounded benchmark')
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    if args.variant == 'baseline':
        spec = importlib.util.spec_from_file_location('_recursive_fm_baseline', args.baseline_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = current
    if args.registry:
        registry = json.loads(args.registry.read_text())
        adjacency = torch.as_tensor(registry['adjacency'], dtype=torch.float32)
        static = torch.as_tensor(registry['static_features'], dtype=torch.float32)
        args.edges = len(static)
    else:
        adjacency = torch.roll(torch.eye(args.edges), 1, dims=1)
        static = torch.zeros(args.edges, 4)
    torch.manual_seed(117)
    model = module.Dynamics(adjacency=adjacency, static_features=static,
        low_frequency_rank=min(16, args.edges - 1), prior_filter='soft', prior_filter_strength=.1,
        prior_action_conditioned=False, prior_usage='condition_only', event_protocol=True).to(device)
    model.set_pretraining_execution_profile(space_chunk=16, time_chunk=32768, mlp_chunk=16384,
                                           activation_checkpointing=True)
    with torch.no_grad():
        model.flow_head.weight.normal_(std=.01)
        model.condition_residual.output.weight.normal_(std=.01)
    generator = torch.Generator().manual_seed(439)
    sequence = torch.randn(args.batch, 4, args.edges, 64, generator=generator).to(device)
    action = torch.zeros(args.batch, 3, args.edges, device=device)
    noise = torch.randn(args.batch, 3, model.graph_prior.rank, 64, generator=generator).to(device)
    flow_time = torch.rand(args.batch, 3, generator=generator).to(device)
    turns = torch.nonzero(adjacency > 0, as_tuple=False)
    edge, successor = turns[0].tolist()
    events = pack_action_events([[tuple([ActionEvent('probe', h, h * 30. + 1., edge,
                     successor, successor)]) for h in range(3)] for _ in range(args.batch)]).to(device)
    kwargs = {} if args.variant == 'baseline' else dict(predict_endpoints=False,
        checkpoint_interval=args.checkpoint_interval if args.variant == 'checkpointed' else 0)
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else lambda: None
    amp = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if args.precision == 'bf16' else nullcontext
    result = dict(variant=args.variant, status='running', device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        precision=args.precision, batch=args.batch, edges=args.edges, physical_transitions=3,
        solver='euler', integration_steps=args.steps, checkpoint_interval=kwargs.get('checkpoint_interval', 0),
        compile_blocks=False, activation_checkpointing=True,
        torch_version=torch.__version__, data='synthetic latent targets, real topology if registry supplied',
        training_modified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    save()
    try:
        measurements = []
        for index in range(args.warmups + args.repeats):
            model.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            synchronize()
            begin = time.perf_counter()
            with amp():
                loss, metrics, predictions = module.recursive_fm_loss(model, z_sequence=sequence, action=action,
                    action_events=events, noise=noise, flow_time=flow_time, steps=args.steps, **kwargs)
            # Match observed-readout training: endpoints have no external loss.
            # Do not keep a previous iteration's unused final-solve graph alive.
            del metrics, predictions
            synchronize()
            forward_end = time.perf_counter()
            loss.backward()
            synchronize()
            end = time.perf_counter()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                               for p in model.parameters()):
                raise FloatingPointError('nonfinite loss or parameter gradients')
            if index >= args.warmups:
                row = dict(forward_seconds=forward_end - begin, backward_seconds=end - forward_end,
                    total_seconds=end - begin, loss=float(loss.detach()),
                    peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20 if device.type == 'cuda' else None)
                measurements.append(row)
                print(json.dumps(row), flush=True)
            result['measurements'] = measurements
            save()
            del loss
        result.update(status='complete', total_seconds_median=statistics.median(r['total_seconds'] for r in measurements),
            forward_seconds_median=statistics.median(r['forward_seconds'] for r in measurements),
            backward_seconds_median=statistics.median(r['backward_seconds'] for r in measurements))
    except torch.OutOfMemoryError as exc:
        result.update(status='oom', error=str(exc))
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        save()
    print(json.dumps({key: value for key, value in result.items() if key != 'measurements'}), flush=True)


if __name__ == '__main__':
    main()
