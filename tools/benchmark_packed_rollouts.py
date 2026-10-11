"""Paired full-Euler benchmarks for packed projections and independent road attention."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '2')

import torch
from tools.evaluate_world_accuracy_snapshot import atomic_json
from tools.benchmark_euler_resolution import relative_l2
from urban_dreamer.learning.fm_inference import ImaginationInferenceProfile, _VelocityInference
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.models.world_model import UrbanDreamerWorldModel
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 8, 16])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--memory-fraction', type=float, default=.15)
    args = parser.parse_args()
    if args.samples < 2 or min(args.batches) < 1 or args.repeats < 1 or args.horizon < 1:
        parser.error('positive batches, repeats and horizon; at least two samples required')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(args.output.parent.resolve() / 'inductor_cache'))
    config = load_config(args.source_dir / 'run_config.toml')
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    model = UrbanDreamerWorldModel(
        adjacency=torch.as_tensor(topology.network.adjacency).float(),
        static_features=torch.as_tensor(topology.network.static_features).float(),
        event_protocol=True, low_frequency_rank=config.method.low_frequency_rank,
        prior_noise_scale=config.method.prior_noise_scale,
        codec_speed_positive=config.codec.speed_positive).to(device).eval()
    checkpoint = args.source_dir / 'checkpoints/latest.pt'
    payload = torch.load(checkpoint, weights_only=True, map_location='cpu')
    model.load_state_dict(payload['training_state']['models']['world_model'])
    del payload
    dataset = CompactPretrainDataset(config, prepared, topology, args.horizon,
        cache_manifest_path=config.data.validation_cache_manifest or config.data.pretrain_cache_manifest,
        split='validation')
    case = json.loads(args.cases.read_text())['recursive_cases'][-1]
    index = next(i for i, entry in enumerate(dataset.manifest['entries']) if entry['path'] == case['path'])
    result = dict(status='running', production_modified=False, solver='euler', steps=100,
        samples=args.samples, precision='fp32', horizon=args.horizon,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__, timings=[])
    original = model.dynamics.forward
    def save():
        atomic_json(args.output, result)
    with torch.no_grad():
        for batch_size in args.batches:
            # Distinct observed windows in one episode, with per-trajectory noise.
            maximum = dataset.manifest['entries'][index]['windows'] - args.horizon
            starts = [min(case['actual_start'] + i, maximum) for i in range(batch_size)]
            batch = dataset.batch_from_starts(index, starts, fm_only=True)
            initial = model.encode(batch.flow[:, 0].to(device))
            controls = [(batch.action[:, t].to(device), batch.action_events.for_step(t).to(device))
                        for t in range(args.horizon)]
            noise = torch.randn(args.horizon, args.samples, batch_size, model.dynamics.graph_prior.rank, 64,
                generator=torch.Generator().manual_seed(1947)).to(device)
            wrappers = {}
            for name in ('original', 'packed'):
                model.dynamics.forward = original
                wrappers[name] = _VelocityInference(model.dynamics, ImaginationInferenceProfile(
                    16, 32768, 16384 if name == 'original' else 0, 'fp32', True,
                    full_batch=args.samples * batch_size))
            def select(name):
                packed = name == 'packed'
                model.dynamics._set_chunk_profile(space_chunk=16, time_chunk=32768,
                                                   mlp_chunk=0 if packed else 16384)
                for layer in model.dynamics.transformer.layers:
                    layer.space.packed_projections = packed
                model.dynamics.forward = wrappers[name]
            def execute():
                current = initial
                for t, (action, events) in enumerate(controls):
                    samples, _, _ = model.latent_disagreement(current, action, n_samples=args.samples,
                        noise=noise[t], action_events=events, steps=100, solver='euler')
                    current = samples[0]
                return samples
            row = dict(rollouts=batch_size, velocity_batch=batch_size * args.samples,
                seconds={'original': [], 'packed': []}, cold_seconds={}, peak_allocated_mib={})
            result['timings'].append(row)
            reference = None
            for name in ('original', 'packed'):
                select(name)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                values = execute()
                torch.cuda.synchronize(device)
                row['cold_seconds'][name] = time.perf_counter() - start
                if reference is None:
                    reference = values.clone()
                else:
                    row['latent_relative_l2'] = relative_l2(values, reference)
                    row['variance_relative_l2'] = relative_l2(values.var(0), reference.var(0))
                save()
            for repeat in range(args.repeats):
                for name in (('original', 'packed') if repeat % 2 == 0 else ('packed', 'original')):
                    select(name)
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    values = execute()
                    torch.cuda.synchronize(device)
                    row['seconds'][name].append(time.perf_counter() - start)
                    row['peak_allocated_mib'][name] = max(row['peak_allocated_mib'].get(name, 0),
                        torch.cuda.max_memory_allocated(device) / 1024**2)
                save()
                print(json.dumps(dict(rollouts=batch_size, repeat=repeat+1,
                    seconds={k: v[-1] for k, v in row['seconds'].items()})), flush=True)
            row['median_seconds'] = {k: statistics.median(v) for k, v in row['seconds'].items()}
            row['paired_total_speedup'] = sum(row['seconds']['original']) / sum(row['seconds']['packed'])
            row['trajectory_windows_per_second'] = {k: batch_size * args.horizon / v
                                                    for k, v in row['median_seconds'].items()}
            save()
            model.dynamics.forward = original
            del wrappers, reference, values, batch, initial, controls, noise
            torch.cuda.empty_cache()
    result['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
