"""Profile real isolated imagination updates, preserving the full sampling budget.

Uses a frozen source tree/checkpoint and never writes to the live run. Phase GPU
spans include scheduling/contending work and are not exclusive kernel timings.
A single steady-state solve may additionally be traced with torch.profiler.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


class PhaseRecorder:
    def __init__(self, *, device=None):
        self.device = device
        self.rows = []

    @contextmanager
    def phase(self, name, **details):
        import torch
        events = None
        if self.device is not None and self.device.type == 'cuda':
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        started = time.perf_counter()
        try:
            yield
        finally:
            if events:
                events[1].record()
            self.rows.append(dict(phase=name, wall_seconds=time.perf_counter() - started,
                                  details=details, _events=events))

    def finish(self):
        import torch
        if self.device is not None and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        rows, self.rows = self.rows, []
        for row in rows:
            events = row.pop('_events')
            row['gpu_span_seconds'] = events[0].elapsed_time(events[1]) / 1000 if events else None
        return rows


def velocity_matrix_flops(*, edges, tokens, width, hidden, depth, latent_width):
    """Count multiply/add pairs, excluding nonlinearities, norms and graph work.

    The physical interval dimension is one: temporal attention reduces to V/out.
    Gated MLP uses width -> 2*hidden -> width. Spatial attention is dense.
    """
    if min(edges, tokens, width, hidden, depth, latent_width) < 1 or tokens < edges:
        raise ValueError('positive model dimensions and tokens >= edges required')
    projections = 12 * tokens * width * width  # Spatial QKV/out + temporal V/out.
    attention = 4 * tokens * tokens * width
    mlp = 6 * tokens * width * hidden
    edge_projection = 4 * edges * latent_width * width
    return depth * (projections + attention + mlp) + edge_projection


def summarize_phases(rows):
    summary = {}
    for row in rows:
        item = summary.setdefault(row['phase'], dict(calls=0, wall_seconds=0., gpu_span_seconds=0.))
        item['calls'] += 1
        item['wall_seconds'] += row['wall_seconds']
        if row['gpu_span_seconds'] is not None:
            item['gpu_span_seconds'] += row['gpu_span_seconds']
    return summary


def summarize_trace(events):
    """Kernel work and host CUDA launch calls, excluding Python phase spans."""
    summary = {}
    for event in events:
        category = event.get('cat')
        if event.get('ph') != 'X' or category not in {'kernel', 'cuda_runtime'}:
            continue
        key = (category, event.get('name', 'unknown'))
        row = summary.setdefault(key, dict(category=category, name=key[1], calls=0, seconds=0.))
        row['calls'] += 1
        row['seconds'] += event.get('dur', 0.) / 1e6
    rows = sorted(summary.values(), key=lambda row: row['seconds'], reverse=True)
    return dict(kernel_seconds=sum(row['seconds'] for row in rows if row['category'] == 'kernel'),
                cuda_runtime_host_seconds=sum(row['seconds'] for row in rows if row['category'] == 'cuda_runtime'),
                rows=rows,
                note='Kernel durations can include device contention; overlapping kernels may be counted twice.')


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--online-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-fraction', type=float, default=.10)
    parser.add_argument('--passes', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--trace-pass', type=int, default=1)
    parser.add_argument('--assumed-fp32-tflops', type=float, default=35.58)
    args = parser.parse_args()
    if min(args.passes, args.threads, args.workers) < 1 or not 0 < args.memory_fraction <= 1:
        parser.error('positive passes/threads/workers and memory fraction in (0,1] required')
    if args.assumed_fp32_tflops <= 0:
        parser.error('assumed peak throughput must be positive')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.json').exists():
        raise ValueError('refusing to overwrite existing profiling results')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(args.source_dir.resolve()))
    import torch
    from urban_dreamer.learning import training, parallel_imagination
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.topology import load_traffic_topology
    from urban_dreamer.workflow.checkpoint import load_checkpoint
    from urban_dreamer.workflow.config import load_config
    from urban_dreamer.workflow.pipeline import combined_input_hash
    from tools.migrate_pretrained_meso import install_sparse_codec

    snapshot = output / 'snapshot'
    (snapshot / 'checkpoints').mkdir(parents=True, exist_ok=True)
    checkpoint_path = snapshot / 'checkpoints/latest.pt'
    if not checkpoint_path.exists():
        # Opening an atomically replaced checkpoint pins that completed inode.
        shutil.copyfile(args.online_dir / 'checkpoints/latest.pt', checkpoint_path)
        shutil.copyfile(args.online_dir / 'run_config.toml', snapshot / 'run_config.toml')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    os.environ['URB_IMAGINATION_WORKERS'] = str(args.workers)
    config = load_config(snapshot / 'run_config.toml')
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    checkpoint = load_checkpoint(output_dir=snapshot,
        expected_config_fingerprint=config.semantic_fingerprint,
        expected_input_hash=combined_input_hash(prepared))
    trainer = training.MainlineTrainer(config=replace(config, run=replace(config.run,
        device=args.device, output_dir=output)), prepared_inputs=prepared, topology=topology)
    install_sparse_codec(trainer)
    trainer.load_state_dict(checkpoint.training_state)
    dynamics = trainer.world_model.dynamics
    layer = dynamics.transformer.layers[0]
    dimensions = dict(edges=dynamics.n_edges, tokens=int(layer.space.attn_mask.shape[-1]),
        width=layer.space.attention.width, hidden=layer.mlp.output.in_features,
        depth=len(dynamics.transformer.layers), latent_width=dynamics.spatial_proj.in_features)
    flops = velocity_matrix_flops(**dimensions)
    protocol = dict(source_file=training.__file__, source_dir=str(args.source_dir.resolve()),
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        completed_episodes=checkpoint.next_unit, device=args.device, dimensions=dimensions,
        matrix_flops_per_sample_velocity=flops, fp32_tflops_assumed=args.assumed_fp32_tflops,
        passes_per_episode=training.IMAGINED_PASSES_PER_EPISODE,
        rollouts_per_pass=training.IMAGINED_ROLLOUTS_PER_PASS, horizon=training.IMAGINED_HORIZON,
        fm_steps=config.method.fm_steps, solver=config.method.fm_solver,
        samples=config.method.disagreement_samples, threads=args.threads, workers=args.workers,
        note='Private optimizer updates only. GPU spans include queueing/contention; phase totals may overlap.')
    atomic_json(output / 'protocol.json', protocol)
    recorder = PhaseRecorder(device=device)
    patches = []
    pass_index = -1
    solve_index = 0
    traces = []

    def patch(obj, attribute, wrapped):
        original = getattr(obj, attribute)
        patches.append((obj, attribute, original))
        setattr(obj, attribute, wrapped(original))

    def timed(name, details=lambda *_a, **_k: {}):
        def wrap(original):
            def call(*a, **k):
                with recorder.phase(name, **details(*a, **k)):
                    return original(*a, **k)
            return call
        return wrap

    patch(trainer.world_model, 'encode', timed('codec_encode'))
    patch(trainer.world_model, 'decode', timed('codec_decode'))
    patch(torch.Tensor, 'backward', timed('backward'))
    patch(parallel_imagination, '_finish_pass', timed('build_differentiable_objectives'))
    for name in ('actor', 'event_critic', 'latent_critic'):
        patch(trainer.optimizers[name], 'step', timed('adam_' + name))

    def wrapped_disagreement(original):
        def call(z, action, **kwargs):
            nonlocal solve_index
            index = solve_index
            solve_index += 1
            details = dict(rollouts=int(z.shape[0]), samples=kwargs['n_samples'],
                           steps=kwargs['steps'], window_index=index)
            trace = pass_index == args.trace_pass and index == 0
            with recorder.phase('world_prediction', **details):
                if not trace:
                    return original(z, action, **kwargs)
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device.type == 'cuda':
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
                    result = original(z, action, **kwargs)
                prof.export_chrome_trace(str(output / 'velocity_trace.json'))
                (output / 'velocity_cpu_table.txt').write_text(prof.key_averages().table(
                    sort_by='self_cpu_time_total', row_limit=35))
                (output / 'velocity_cuda_table.txt').write_text(prof.key_averages().table(
                    sort_by='self_device_time_total', row_limit=35))
                trace_data = json.loads((output / 'velocity_trace.json').read_text())
                trace_summary = summarize_trace(trace_data['traceEvents'])
                atomic_json(output / 'velocity_trace_summary.json', trace_summary)
                traces.append(dict(pass_index=pass_index, window_index=index, **trace_summary))
                return result
        return call
    patch(trainer.world_model, 'latent_disagreement', wrapped_disagreement)

    class ExecutorProxy:
        def __init__(self, executor):
            self.executor = executor
        def map(self, *a, **k):
            with recorder.phase('cpu_physics_actor_q_worker_wait'):
                return list(self.executor.map(*a, **k))
    patch(parallel_imagination, '_executor', lambda original:
          lambda *a, **k: ExecutorProxy(original(*a, **k)))
    results = dict(protocol=protocol, passes=[], traces=traces)
    try:
        for pass_index in range(args.passes):
            solve_index = 0
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            metrics = trainer._train_imagination_pass()
            phases = recorder.finish()
            wall = time.perf_counter() - started
            sample_evaluations = sum(row['details']['rollouts'] * row['details']['samples'] * row['details']['steps']
                for row in phases if row['phase'] == 'world_prediction')
            row = dict(index=pass_index, wall_seconds=wall, metrics=metrics, phases=phases,
                summary=summarize_phases(phases), sample_velocity_evaluations=sample_evaluations,
                matrix_flops=sample_evaluations * flops,
                ideal_matrix_compute_seconds=sample_evaluations * flops / (args.assumed_fp32_tflops * 1e12),
                physical_work=getattr(trainer, '_last_parallel_work', None),
                first_pass_includes_compile_and_worker_startup=pass_index == 0,
                profiler_enabled=pass_index == args.trace_pass)
            if device.type == 'cuda':
                row['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                row['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
            results['passes'].append(row)
            atomic_json(output / 'results.json', results)
            print(json.dumps(row), flush=True)
    finally:
        for obj, attribute, original in reversed(patches):
            setattr(obj, attribute, original)
        trainer.close()


if __name__ == '__main__':
    main()
