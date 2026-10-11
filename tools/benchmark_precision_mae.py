"""Compare velocity-only BF16/TF32 with FP32 against held-out traffic labels."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '2')

import torch
from tools.benchmark_euler_resolution import compare_samples
from tools.evaluate_world_accuracy_snapshot import atomic_json, road_scores
from tools.migrate_pretrained_meso import install_sparse_codec
from urban_dreamer.learning.cohort_targets import prepare_cohort_batch
from urban_dreamer.learning.fm_inference import configure_imagination_inference
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.models.world_model import UrbanDreamerWorldModel
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config


def aggregate_scores(rows):
    """Weight masked errors by their actual road observations, not case count."""
    output = {}
    for metric, denominator in (
        ('speed_mae_mps', 'road_observations'),
        ('occupied_speed_mae_mps', 'occupied_roads'),
        ('slow_speed_mae_mps', 'slow_roads'),
        ('occupied_count_mae_vehicles', 'occupied_roads'),
        ('total_count_mae_per_trajectory', 'trajectories')):
        available = [row for row in rows if row.get(metric) is not None and row[denominator] > 0]
        count = sum(row[denominator] for row in available)
        output[metric] = sum(row[metric] * row[denominator] for row in available) / count if count else None
    output.update({key: sum(row[key] for row in rows)
                   for key in ('road_observations', 'occupied_roads', 'slow_roads', 'trajectories')})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--pairs', type=int, default=3)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error('pairs must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.10, device)
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(args.output_dir.resolve() / 'inductor_cache'))
    config = load_config(args.source_dir / 'run_config.toml')
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    model = UrbanDreamerWorldModel(adjacency=torch.as_tensor(topology.network.adjacency).float(),
        static_features=torch.as_tensor(topology.network.static_features).float(), event_protocol=True,
        low_frequency_rank=config.method.low_frequency_rank, prior_noise_scale=config.method.prior_noise_scale,
        codec_speed_positive=config.codec.speed_positive).to(device).eval()
    trainer = SimpleNamespace(world_model=model, topology=topology, network=topology.network,
        device=device, config=config, _tensor=lambda x: torch.as_tensor(x, device=device).float())
    install_sparse_codec(trainer)
    checkpoint = args.source_dir / 'checkpoints/latest.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    completed = payload['training_state']['metrics']['online_episodes_completed']
    model.load_state_dict(payload['training_state']['models']['world_model'])
    del payload
    dataset = CompactPretrainDataset(config, prepared, topology, 9,
        cache_manifest_path=config.data.validation_cache_manifest or config.data.pretrain_cache_manifest,
        split='validation')
    capacities = torch.as_tensor(topology.network.capacities, device=device).float().clamp_min(1)
    limits = torch.as_tensor(topology.network.speed_limits_mps, device=device).float()
    profiles = ('fp32', 'tf32', 'bf16')
    result = dict(status='running', production_modified=False, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        online_completed=completed, samples=2, rollouts=8, velocity_batch=16, solver='euler', steps=100,
        horizon=8, seed=31415, precision_scope='velocity only; prior, noise, Euler state and decoder remain FP32',
        quality=[], timings=[], profiles={}, cold_seconds=[])
    def save():
        atomic_json(args.output_dir / 'results.json', result)
    def select(precision):
        trainer.config = replace(config, method=replace(config.method, disagreement_samples=2),
            resources=replace(config.resources, world_model_imagination_precision=precision,
                              world_model_imagination_compile=True))
        result['profiles'][precision] = configure_imagination_inference(trainer)
    def scores(prediction, truth, speed):
        row = road_scores(prediction, truth, speed, capacities, limits)
        row.update(road_observations=truth.shape[0] * truth.shape[1], trajectories=truth.shape[0],
            total_count_mae_per_trajectory=float(((prediction[..., 2] - truth[..., 2]) * capacities).sum(-1).abs().mean()))
        return row
    cases = json.loads(args.cases.read_text())['recursive_cases']
    timing_work = None
    save()
    with torch.no_grad():
        for case_index, case in enumerate(cases):
            entry = next(i for i, item in enumerate(dataset.manifest['entries']) if item['path'] == case['path'])
            max_start = dataset.manifest['entries'][entry]['windows'] - 9
            first = min(case['actual_start'], max_start - 7)
            if first < 0:
                raise ValueError('case needs eight distinct starts and nine following windows')
            starts = list(range(first, first + 8))
            batch = dataset.batch_from_starts(entry, starts, materialization_horizon=9, fm_only=False)
            scoped = prepare_cohort_batch(batch, capacities=capacities)
            truth = scoped.transition_flow.to(device)
            initial = model.encode(truth[:, 0])
            controls = [(scoped.transition_action[:, t].to(device),
                scoped.transition_action_events.for_step(t).to(device)) for t in range(8)]
            generator = torch.Generator(device=device).manual_seed(31415 + case_index)
            noises = [torch.randn(2, 8, model.dynamics.graph_prior.rank, 64, device=device, generator=generator)
                      for _ in range(8)]
            if timing_work is None:
                timing_work = (initial.clone(), controls[0], noises[0])
            reference = {}
            for precision in profiles:
                select(precision)
                current = initial.clone()
                torch.cuda.synchronize(device)
                begin = time.perf_counter()
                for step, (action, events) in enumerate(controls):
                    samples, global_variance, _ = model.latent_disagreement(current, action,
                        n_samples=2, noise=noises[step], action_events=events, steps=100, solver='euler')
                    current = samples[0]
                    if step + 1 not in (1, 3, 8):
                        continue
                    decoded = model.decode(samples.reshape(-1, samples.shape[-2], 64)).reshape(2, 8, samples.shape[-2], -1)
                    speed = .5 * (truth[:, step + 1, :, 1] + truth[:, step + 2, :, 1])
                    row = dict(case=case['method'], starts=starts, precision=precision, horizon_seconds=(step+1)*30,
                        sample0=scores(decoded[0], truth[:, step+1], speed),
                        sample_mean=scores(decoded.mean(0), truth[:, step+1], speed),
                        variance_mean=float(global_variance.mean()))
                    if precision == 'fp32':
                        reference[step] = (samples.clone(), decoded.clone(), global_variance.clone())
                    else:
                        ref_samples, ref_decoded, ref_var = reference[step]
                        row['comparison'] = compare_samples(samples, ref_samples, decoded, ref_decoded,
                            speed, truth[:, step+1, :, 2] * capacities, limits, capacities)
                        row['risk_mean_absolute_difference'] = float((global_variance - ref_var).abs().mean()) * config.method.disagreement_weight
                    result['quality'].append(row)
                    save()
                torch.cuda.synchronize(device)
                result['cold_seconds'].append(dict(case=case['method'], precision=precision,
                    seconds=time.perf_counter()-begin, includes_possible_compilation=True))
                save()
                print(json.dumps(dict(case=case['method'], precision=precision,
                    seconds=time.monotonic()-started)), flush=True)
        z, (action, events), noise = timing_work
        def timed():
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            model.latent_disagreement(z, action, n_samples=2, noise=noise, action_events=events, steps=100, solver='euler')
            torch.cuda.synchronize(device)
            return time.perf_counter()-begin
        for precision in ('tf32', 'bf16'):
            pairs = []
            for name in ('fp32', precision):
                select(name)
                timed()
            for index in range(args.pairs):
                row = {}
                for name in (('fp32', precision) if index % 2 == 0 else (precision, 'fp32')):
                    select(name)
                    row[name] = timed()
                pairs.append(row)
            result['timings'].append(dict(precision=precision, pairs=pairs,
                fp32_median_seconds=statistics.median(r['fp32'] for r in pairs),
                candidate_median_seconds=statistics.median(r[precision] for r in pairs),
                paired_total_speedup=sum(r['fp32'] for r in pairs)/sum(r[precision] for r in pairs)))
            save()
        result['aggregate'] = {str(horizon): {precision: aggregate_scores(
            [row['sample0'] for row in result['quality'] if row['precision'] == precision
             and row['horizon_seconds'] == horizon]) for precision in profiles} for horizon in (30, 90, 240)}
    if torch.backends.cuda.matmul.fp32_precision != 'ieee':
        raise RuntimeError('velocity precision context leaked outside its scope')
    result.update(status='complete', elapsed_seconds=time.monotonic()-started,
        notes=['32 starting windows from four held-out episodes; neighboring starts overlap.',
               'Future labels never feed recursive latent; controls are recorded fixed-cohort controls.',
               'Single frozen checkpoint and noise seed, not policy closed-loop or population generalization.',
               'Timing alternates paired inputs on a shared GPU; production was not changed.'])
    save()


if __name__ == '__main__':
    main()
