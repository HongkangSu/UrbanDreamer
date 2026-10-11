"""Private Euler-resolution tradeoff benchmark; never changes production settings."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_world_accuracy_snapshot import atomic_json, road_scores
from tools.migrate_pretrained_meso import install_sparse_codec
from urban_dreamer.learning.cohort_targets import prepare_cohort_batch
from urban_dreamer.learning.fm_inference import configure_imagination_inference
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.models.world_model import UrbanDreamerWorldModel, FLOW_LATENT_DIM
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.checkpoint import load_checkpoint
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.pipeline import combined_input_hash


def relative_l2(value, reference):
    value, reference = value.double(), reference.double()
    if not bool(torch.isfinite(value).all() and torch.isfinite(reference).all()):
        raise ValueError('nonfinite Euler comparison')
    denominator = float(torch.linalg.vector_norm(reference))
    numerator = float(torch.linalg.vector_norm(value - reference))
    return numerator / denominator if denominator > 0 else (0. if numerator == 0 else None)


def compare_samples(samples, reference, decoded, decoded_reference, speed_target, target_count, limits, capacities):
    """Teacher differences are numerical diagnostics, never ground-truth accuracy."""
    variance = samples.float().var(0, unbiased=True).mean(-1)
    reference_variance = reference.float().var(0, unbiased=True).mean(-1)
    ref_mean = float(reference_variance.mean())
    ratio = float(variance.mean()) / ref_mean if ref_mean else None
    slow = (target_count >= 5 - 1e-5) & (speed_target * limits <= 2)
    speed_difference = (decoded[0, ..., 1] - decoded_reference[0, ..., 1]) * limits
    count_difference = (decoded[0, ..., 2] - decoded_reference[0, ..., 2]) * capacities
    slow_max = float(speed_difference[slow].abs().max()) if bool(slow.any()) else None
    result = dict(sample0_latent_relative_l2=relative_l2(samples[0], reference[0]),
        sample_mean_latent_relative_l2=relative_l2(samples.mean(0), reference.mean(0)),
        all_samples_latent_relative_l2=relative_l2(samples, reference),
        global_variance_ratio=ratio, road_variance_relative_l2=relative_l2(variance, reference_variance),
        sample0_all_road_speed_mae_vs_reference_mps=float(speed_difference.abs().mean()),
        sample0_all_road_count_mae_vs_reference_vehicles=float(count_difference.abs().mean()),
        sample0_slow_speed_max_abs_vs_reference_mps=slow_max,
        slow_roads=int(slow.sum()))
    result['numerical_gate'] = bool(result['sample0_latent_relative_l2'] is not None
        and result['sample0_latent_relative_l2'] <= .01 and ratio is not None
        and abs(ratio - 1) <= .05 and (slow_max is None or slow_max <= .1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, nargs='+', default=[4, 8, 16, 32])
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--samples', type=int, default=3)
    args = parser.parse_args()
    if args.samples < 2 or args.pairs < 1:
        raise ValueError('need at least two noise samples and one timing pair')
    if any(step < 1 or step >= 100 for step in args.steps):
        raise ValueError('candidate Euler resolution must be within 1..99')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config = load_config(args.source_dir / 'run_config.toml')
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    checkpoint = load_checkpoint(output_dir=args.source_dir,
        expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash=combined_input_hash(prepared))
    torch.set_num_threads(2)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.06, device)
    model = UrbanDreamerWorldModel(adjacency=torch.as_tensor(topology.network.adjacency).float(),
        static_features=torch.as_tensor(topology.network.static_features).float(), event_protocol=True,
        low_frequency_rank=config.method.low_frequency_rank, prior_noise_scale=config.method.prior_noise_scale,
        codec_speed_positive=config.codec.speed_positive).to(device).eval().requires_grad_(False)
    trainer = SimpleNamespace(world_model=model, network=topology.network, topology=topology,
        config=replace(config, run=replace(config.run, device=args.device),
            method=replace(config.method, disagreement_samples=args.samples)), device=device,
        _tensor=lambda value: torch.as_tensor(value, device=device).float())
    install_sparse_codec(trainer)
    model.load_state_dict(checkpoint.training_state['models']['world_model'], strict=True)
    del checkpoint.training_state['models']
    dataset = CompactPretrainDataset(config, prepared, topology, 9,
        cache_manifest_path=config.data.validation_cache_manifest or config.data.pretrain_cache_manifest,
        split='validation')
    profile = configure_imagination_inference(trainer)
    result = dict(status='running', production_modified=False, benchmark_only=True,
        source_checkpoint=str(checkpoint.path), checkpoint_sha256=hashlib.sha256(checkpoint.path.read_bytes()).hexdigest(),
        online_completed=checkpoint.next_unit, solver='euler', reference_steps=100,
        candidates=args.steps, stochastic_samples=args.samples, seed=31415, inference_profile=profile,
        quality=[], timings=[])
    atomic_json(output / 'results.json', result)
    source_cases = json.loads(args.cases.read_text())['recursive_cases']
    capacities = torch.as_tensor(topology.network.capacities, device=device).float().clamp_min(1.)
    limits = torch.as_tensor(topology.network.speed_limits_mps, device=device).float()
    counter = dict(calls=0)
    def count_calls(module, inputs, kwargs):
        counter['calls'] += 1
    hook = model.dynamics.register_forward_pre_hook(count_calls, with_kwargs=True)
    first_batch = None
    with torch.no_grad():
        for case_index, case in enumerate(source_cases):
            index = next(i for i, entry in enumerate(dataset.manifest['entries']) if entry['path'] == case['path'])
            batch = dataset.batch_from_starts(index, [case['actual_start']], materialization_horizon=9, fm_only=False)
            if first_batch is None:
                first_batch = dataset.batch_from_starts(index, [case['actual_start']] * 8, materialization_horizon=1, fm_only=True)
            scoped = prepare_cohort_batch(batch, capacities=capacities)
            truth = scoped.transition_flow.to(device)
            initial = model.encode(truth[:, 0])
            generator = torch.Generator(device=device).manual_seed(31415 + case_index)
            noises = [torch.randn(args.samples, 1, model.dynamics.graph_prior.rank, FLOW_LATENT_DIM,
                device=device, generator=generator) for _ in range(8)]
            reference = []
            for resolution in [100] + args.steps:
                current = initial.clone()
                for step in range(8):
                    counter['calls'] = 0
                    samples, global_variance, edge_variance = model.dynamics.latent_disagreement(current,
                        scoped.transition_action[:, step].to(device), n_samples=args.samples, noise=noises[step],
                        action_events=scoped.transition_action_events.for_step(step).to(device),
                        steps=resolution, solver='euler')
                    if counter['calls'] != resolution:
                        raise RuntimeError('Euler velocity evaluation count differs from resolution')
                    current = samples[0]
                    if resolution == 100:
                        reference.append(samples.clone())
                    elif step + 1 in (1, 3, 8):
                        decoded = model.decode(samples.reshape(-1, samples.shape[-2], samples.shape[-1])).reshape(args.samples, 1, samples.shape[-2], -1)
                        ref = reference[step]
                        decoded_ref = model.decode(ref.reshape(-1, ref.shape[-2], ref.shape[-1])).reshape(args.samples, 1, ref.shape[-2], -1)
                        speed_target = .5 * (truth[:, step + 1, :, 1] + truth[:, step + 2, :, 1])
                        scores = compare_samples(samples, ref, decoded, decoded_ref, speed_target,
                            truth[:, step + 1, :, 2] * capacities, limits, capacities)
                        result['quality'].append(dict(case=case['method'], start=case['actual_start'],
                            horizon_seconds=30 * (step + 1), steps=resolution, velocity_calls=resolution,
                            comparison=scores, ground_truth_sample0=road_scores(decoded[0], truth[:, step + 1], speed_target, capacities, limits),
                            ground_truth_reference_sample0=road_scores(decoded_ref[0], truth[:, step + 1], speed_target, capacities, limits)))
                atomic_json(output / 'results.json', result)
            print(json.dumps(dict(stage='case_complete', case=case_index, seconds=time.monotonic()-started)), flush=True)
        z = model.encode(first_batch.flow[:, 0].to(device))
        action = first_batch.action[:, 0].to(device)
        events = first_batch.action_events.for_step(0).to(device)
        noise = torch.randn(args.samples, 8, model.dynamics.graph_prior.rank, FLOW_LATENT_DIM, device=device,
            generator=torch.Generator(device=device).manual_seed(31415))
        def timed(resolution):
            counter['calls'] = 0
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            model.dynamics.latent_disagreement(z, action, n_samples=args.samples, noise=noise, action_events=events,
                steps=resolution, solver='euler')
            torch.cuda.synchronize(device)
            return time.perf_counter() - begin, counter['calls']
        timed(100)
        for resolution in [value for value in args.steps if value >= 8]:
            pairs = []
            timed(resolution)
            for pair in range(args.pairs):
                measurements = {}
                for value in ([100, resolution] if pair % 2 == 0 else [resolution, 100]):
                    elapsed, calls = timed(value)
                    if calls != value:
                        raise RuntimeError('timing velocity count differs')
                    measurements[str(value)] = elapsed
                pairs.append(measurements)
            ref = [pair['100'] for pair in pairs]
            candidate = [pair[str(resolution)] for pair in pairs]
            result['timings'].append(dict(steps=resolution, pairs=pairs,
                reference_median_seconds=statistics.median(ref), candidate_median_seconds=statistics.median(candidate),
                paired_total_speedup=sum(ref)/sum(candidate), trajectories=8, stochastic_samples=args.samples,
                full_velocity_batch=8 * args.samples))
            atomic_json(output / 'results.json', result)
            print(json.dumps(dict(stage='timing_complete', steps=resolution, seconds=time.monotonic()-started)), flush=True)
    hook.remove()
    result.update(status='complete', elapsed_seconds=time.monotonic()-started,
        gates=dict(latent_relative_l2_max=.01, global_variance_relative_change_max=.05, slow_speed_max_abs_mps=.1),
        notes=['Euler100 is a numerical reference, not ground truth.',
               'Every resolution uses identical source parameters, initial state and per-window noise; sample zero recurses.',
               'Fixed C0 volume and recorded C0 controls; future observed states are scoring labels only.',
               'Full-batch timing uses identical inputs and paired alternating measurements on a shared GPU.',
               'No mainline settings or imagination budget were changed.'])
    atomic_json(output / 'results.json', result)


if __name__ == '__main__':
    main()
