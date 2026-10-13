"""Private FP32 full Euler100 CUDA graph timing and eight-window correctness proof."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from urban_dreamer.learning.fm_graph_solver import EulerGraphSolver
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.models.world_model import UrbanDreamerWorldModel
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config
from tools.benchmark_fm_inference import quality
from tools.benchmark_fm_picard import tensor_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--windows-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-fraction', type=float, default=.08)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--cases', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rollout-steps', type=int, default=8)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.source_dir / 'checkpoints/latest.pt'
    result = dict(status='running', production_modified=False, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  precision='fp32', solver='euler', steps=100, noise_samples=3, timings=[], quality=[], mutations=[],
                  torch_version=torch.__version__,
                  solver_source_sha256=hashlib.sha256((ROOT/'urban_dreamer/learning/fm_graph_solver.py').read_bytes()).hexdigest())
    def save():
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2)+'\n'); temp.replace(args.output)
    save()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    if free < total * args.memory_fraction + 512 * 1024**2:
        raise RuntimeError('insufficient free GPU memory for isolated benchmark')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    config = load_config(args.source_dir / 'run_config.toml')
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    dataset = CompactPretrainDataset(config, prepared, topology, args.rollout_steps,
                                    cache_manifest_path=args.cache_manifest, split='validation')
    windows = json.loads(args.windows_json.read_text())['windows']
    model = UrbanDreamerWorldModel(adjacency=torch.as_tensor(topology.network.adjacency, dtype=torch.float32),
        static_features=torch.as_tensor(topology.network.static_features, dtype=torch.float32), event_protocol=True,
        low_frequency_rank=config.method.low_frequency_rank,
        prior_filter=getattr(config.method, 'prior_filter', 'hard'),
        prior_filter_strength=getattr(config.method, 'prior_filter_strength', 10.0),
        prior_action_conditioned=getattr(config.method, 'prior_action_conditioned', True),
        prior_filter_input=getattr(config.method, 'prior_filter_input', False), prior_noise_scale=config.method.prior_noise_scale,
        codec_speed_positive=config.codec.speed_positive).to(device).eval()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(payload['training_state']['models']['world_model']); del payload
    model.dynamics._set_chunk_profile(space_chunk=16, time_chunk=32768, mlp_chunk=16384)
    eager = model.dynamics.solve
    graphs = EulerGraphSolver(model.dynamics, max_cached_shapes=1)
    capacity = torch.as_tensor(topology.network.capacities)
    limits = torch.as_tensor(topology.network.speed_limits_mps)
    def data(case, batch_size):
        row = windows[case]
        episode = row['episode_index']
        start = min(row['window'], dataset.manifest['entries'][episode]['windows'] - args.rollout_steps)
        batch = dataset.batch_from_starts(episode, [max(start, 0)] * batch_size, fm_only=True)
        z = model.encode(batch.flow[:, 0].to(device))
        controls = [(batch.action[:, i].to(device), batch.action_events.for_step(i).to(device))
                    for i in range(args.rollout_steps)]
        noise = torch.randn(args.rollout_steps, 3, batch_size, model.dynamics.graph_prior.rank, 64,
                            generator=torch.Generator().manual_seed(113000+case)).to(device)
        return z, controls, noise
    def execute(work, rollout=False):
        z, controls, noise = work
        records = []
        for step in range(args.rollout_steps if rollout else 1):
            action, events = controls[step]
            samples, _, variance = model.latent_disagreement(z, action, n_samples=3, noise=noise[step],
                                                            steps=100, solver='euler', action_events=events)
            if rollout:
                decoded = model.decode(samples)
                readout = model.active_vehicle_fraction_readout(samples, z_prev=z.unsqueeze(0).expand_as(samples),
                    raw_decoder_fraction=decoded[...,2].mean(-1), transition_seconds=30.)
                records.append(dict(latent=samples.cpu(), variance=variance.cpu(), decoded=decoded.mean(0).cpu(), readout=readout.cpu()))
            z = samples[0]
        return records if rollout else samples
    with torch.no_grad():
        for batch_size in args.batches:
            work = data(3, batch_size)
            model.dynamics.solve = eager
            reference = execute(work).cpu()
            model.dynamics.solve = graphs
            free_before_capture, _ = torch.cuda.mem_get_info(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device); begin = time.perf_counter()
            candidate = execute(work).cpu()
            torch.cuda.synchronize(device)
            row = dict(trajectories=batch_size, velocity_batch=3*batch_size, cold_seconds=time.perf_counter()-begin,
                       comparison=tensor_comparison(candidate, reference), eager_seconds=[], graph_seconds=[],
                       peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
                       reserved_mib=torch.cuda.memory_reserved(device)/1024**2,
                       device_memory_delta_mib=(free_before_capture-torch.cuda.mem_get_info(device)[0])/1024**2,
                       graph_stats=dict(graphs.stats))
            result['timings'].append(row); save()
            if not graphs.cache:
                result['status'] = 'capture_failed'; save(); return
            for repeat in range(args.repeats):
                methods = [('eager', eager), ('graph', graphs)]
                if repeat % 2: methods.reverse()
                for name, solve in methods:
                    model.dynamics.solve = solve
                    torch.cuda.synchronize(device); begin = time.perf_counter()
                    execute(work)
                    torch.cuda.synchronize(device)
                    row[name+'_seconds'].append(time.perf_counter()-begin)
            row['eager_median_seconds'] = statistics.median(row['eager_seconds'])
            row['graph_median_seconds'] = statistics.median(row['graph_seconds'])
            row['speedup'] = row['eager_median_seconds']/row['graph_median_seconds']; save()
            print('timings', batch_size, row, flush=True)
            # The graph reads live parameter storage, including after Adam-style updates.
            parameter = model.dynamics.flow_head.bias
            parameter.add_(.0001)
            model.dynamics.solve = eager; updated_reference = execute(work).cpu()
            before_captures = graphs.stats['captures']
            model.dynamics.solve = graphs; updated_candidate = execute(work).cpu()
            result['mutations'].append(dict(batch=batch_size, comparison=tensor_comparison(updated_candidate, updated_reference),
                recaptured=graphs.stats['captures']!=before_captures, change_from_old=tensor_comparison(updated_reference, reference)))
            parameter.sub_(.0001); save()
            # Four stress conditions at batch 1 cover consecutive propagation cheaply.
            if batch_size == 1:
                for case in args.cases:
                    work = data(case, batch_size)
                    model.dynamics.solve = eager; reference_roll = execute(work, True)
                    model.dynamics.solve = graphs; candidate_roll = execute(work, True)
                    result['quality'].append(dict(case=case, batch=batch_size, steps=quality(candidate_roll,reference_roll,capacity,limits)))
                    save(); print('quality', case, flush=True)
        result['graph_stats'] = graphs.stats
        result['status'] = 'complete'; save()


if __name__ == '__main__':
    main()
