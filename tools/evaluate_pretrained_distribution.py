"""Frozen pretrained Prior+FM distribution diagnostics, never screen-model results.

Each condition owns one genuine 32-trajectory bank. K prefixes, empirical sample
means and finite-pool Monte Carlo diagnostics reuse that bank without reselection.
Zero source noise is a separate inference setting of the SAME pretrained FM.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.component_experiment_data import atomic_json, dataset, load_frozen, sha256
from tools.component_prediction_metrics import empirical_crps, evaluate_prediction_samples, summarize_state_records
from tools.evaluate_component_world_models import (
    endpoint_target, estimate_training_scales, physical, select_traffic_conditions, stable_seed,
)
from tools.summarize_component_predictions import demand_bootstrap
from tools.world_model_evaluation import _repeat_events
from urban_dreamer.learning.cohort_targets import prepare_cohort_batch

CHECKPOINT_SHA256 = '2f63f0b2a5fe18f3a3bd9be27ffd1e63a2755464e7f2944e1227334f0ffe6af6'
KS = (1, 8, 16, 32)
HORIZONS = (1, 2, 4, 8)


@torch.no_grad()
def trajectory_bank(model, initial, actions, events, *, capacities, limits,
                    seed, horizons=HORIZONS, samples=32, chunk=4, steps=100, zero_source=False):
    """Per-draw/depth coefficient RNG makes bank identities chunk-independent."""
    if initial.shape[0] != 1 or samples < 1 or chunk < 1 or steps < 1:
        raise ValueError('one condition per bank and positive sampling budgets required')
    rank = model.dynamics.graph_prior.rank
    bank = {h: [] for h in horizons}
    for begin in range(0, samples, chunk):
        size = min(chunk, samples-begin)
        current = initial.repeat(size, 1, 1)
        for depth in range(max(horizons)):
            noise = []
            for ordinal in range(begin, begin+size):
                shape = (1, rank, initial.shape[-1])
                value = (torch.zeros(shape) if zero_source else torch.randn(shape,
                    generator=torch.Generator(device='cpu').manual_seed(
                        stable_seed(seed, 'source_coefficients_v1', ordinal, depth))))
                noise.append(value)
            current = model.transition(current, actions[:, depth].repeat(size, 1),
                action_events=_repeat_events(events.for_step(depth), size),
                noise=torch.cat(noise).to(current), steps=steps, solver='euler')
            if depth+1 in bank:
                decoded = physical(model.decode(current), capacities, limits)
                bank[depth+1].append(decoded[:, None].float().cpu().numpy())
    return {h: np.concatenate(values, axis=0) for h, values in bank.items()}


def supplementary_scores(samples, target, scales):
    """Fair finite-K U-statistic CRPS/energy plus prespecified congestion Brier.

    K=1 has no pairwise U-statistic. Brier is the ordinary empirical probability
    score, not a claim that repeated finite-bank subsets are independent data.
    """
    samples, target, scales = map(lambda x: np.asarray(x, dtype=np.float64), (samples, target, scales))
    if (samples.ndim != 4 or samples.shape[1:] != target.shape or target.shape[0] != 1
            or target.shape[-1] != 2 or scales.shape != (2,) or np.any(scales <= 0)
            or not np.isfinite(samples).all() or np.isinf(target).any() or not np.isfinite(scales).all()):
        raise ValueError('finite (K,1,E,2) samples, matching targets and positive scales required')
    valid = np.isfinite(target[0])
    occupied = valid[:, 0] & (target[0, :, 0] > 0)
    k = len(samples)
    scores = {}
    for feature, name in enumerate(('count', 'speed')):
        for scope, selected in (('all', valid[:, feature]), ('occupied', valid[:, feature] & occupied)):
            score = None
            if k > 1 and selected.any():
                draws = samples[:, 0, selected, feature]
                labels = target[0, selected, feature]
                first = np.abs(draws-labels).mean(0)
                empirical = empirical_crps(draws, labels)
                score = float((first-(first-empirical)*k/(k-1)).mean())
            scores[f'{name}_crps_u_{scope}'] = score
    scores['energy_score_u'] = None
    if k > 1 and valid.any():
        draws, labels = (samples[:, 0]/scales)[:, valid], (target[0]/scales)[valid]
        first = np.linalg.norm(draws-labels, axis=1).mean()
        pairs = sum(np.linalg.norm(draws-draw, axis=1).sum() for draw in draws)
        scores['energy_score_u'] = float(first-pairs/(2*k*(k-1)))
    event_valid = valid.all(-1)
    for scope, selected in (('all', event_valid), ('occupied', event_valid & occupied)):
        scores[f'congestion_brier_{scope}'] = None
        scores[f'congestion_prevalence_{scope}'] = None
        scores[f'congestion_probability_{scope}'] = None
        scores[f'congestion_valid_roads_{scope}'] = int(selected.sum())
        if selected.any():
            probability = ((samples[:, 0, selected, 0] >= 5.) &
                           (samples[:, 0, selected, 1] <= 2.)).mean(0)
            observed = (target[0, selected, 0] >= 5.) & (target[0, selected, 1] <= 2.)
            scores[f'congestion_brier_{scope}'] = float(np.square(probability-observed).mean())
            scores[f'congestion_prevalence_{scope}'] = float(observed.mean())
            scores[f'congestion_probability_{scope}'] = float(probability.mean())
    return scores


def score_bank(samples, target, zero_draw, *, scales, cohort_count, limits, metadata):
    """Per-K distribution vs decoded-ensemble-mean point references."""
    if len(samples) != 32:
        raise ValueError('the predeclared shared bank must contain exactly 32 trajectories')
    rows = []
    definitions = []
    for k in KS:
        definitions.extend((('empirical_ensemble', k, samples[:k]),
                            ('ensemble_mean_point', k, samples[:k].mean(0, keepdims=True))))
    definitions.append(('zero_source_noise_same_fm', 1, zero_draw))
    for kind, k, draws in definitions:
        details = dict(metadata, predictor=kind, k=k,
            interval_available=(kind == 'empirical_ensemble' and k > 1),
            bank_subset='first K fixed draw ordinals' if kind != 'zero_source_noise_same_fm' else 'separate zero coefficients')
        upper = np.stack((np.full_like(limits, float(cohort_count)), limits), axis=-1)
        row = evaluate_prediction_samples(draws, target, scales=scales,
            cohort_count=cohort_count, state_metadata=[details], upper_bounds=upper)['states'][0]
        if not details['interval_available']:
            for field in row['metric_keys']:
                if '_coverage' in field or '_width' in field or '_sample_variance_' in field:
                    row[field] = None
        extra = supplementary_scores(draws, target, scales)
        row.update(extra)
        row['metric_keys'].extend(extra)
        rows.append(row)
    return rows


def nested_mean_diagnostics(samples, *, seed, repetitions=32):
    """Fixed random permutations, nested prefixes; finite-pool sensitivity only.

    The 32-draw mean is a finite reference, not the exact conditional expectation.
    Repeated subsets overlap and are not independent uncertainty replicates.
    Disjoint K-vs-K means are included for K <= 16 to expose sampling sensitivity.
    """
    values = np.asarray(samples, dtype=np.float64)
    if len(values) != 32 or repetitions < 1:
        raise ValueError('32-draw bank and positive repetition count required')
    orders = np.stack([np.random.default_rng(stable_seed(seed, 'nested_subset_v1', repeat)).permutation(32)
                       for repeat in range(repetitions)])
    full = values.mean(0)
    rows = []
    for k in KS:
        deviations, splits = [], []
        for order in orders:
            point = values[order[:k]].mean(0)
            deviations.append(np.abs(point-full).mean(axis=(0, 1)))
            if 2*k <= 32:
                other = values[order[k:2*k]].mean(0)
                splits.append(np.abs(point-other).mean(axis=(0, 1)))
        for feature, name in enumerate(('count', 'speed')):
            observations = np.asarray(deviations)[:, feature]
            rows.append(dict(k=k, feature=name, repetitions=repetitions,
                mean_absolute_deviation_from_full_bank=float(observations.mean()),
                p95_absolute_deviation_from_full_bank=float(np.quantile(observations, .95)),
                disjoint_equal_k_mean_absolute_difference=(float(np.asarray(splits)[:, feature].mean()) if splits else None)))
    return rows, orders


def run(args):
    if args.output_dir.exists():
        raise ValueError('preserve previous results; choose a new output directory')
    if args.max_groups not in (2, 8) or args.sample_chunk < 1 or args.threads < 1:
        raise ValueError('declared pilot/full budgets are 2 or 8 groups, with positive chunk and threads')
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError('invalid GPU memory fraction')
    checkpoint = args.experiment_dir/'snapshot/checkpoints/latest.pt'
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('only the declared fully pretrained Ingolstadt checkpoint is allowed')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        if free < args.gpu_memory_fraction*total + 512*1024**2:
            raise RuntimeError('insufficient GPU admission headroom')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    config, prepared, topology, model, parent = load_frozen(args.experiment_dir, device=device)
    model.eval()
    validation = dataset(args.experiment_dir, config, prepared, topology, split='validation', horizon=9)
    training = dataset(args.experiment_dir, config, prepared, topology, split='train', horizon=1)
    capacities = torch.tensor(topology.network.capacities, dtype=torch.float32).clamp_min(1)
    limits = torch.tensor(topology.network.speed_limits_mps, dtype=torch.float32)
    conditions = select_traffic_conditions(validation, capacities, per_group=3, max_groups=args.max_groups,
        horizon=9, seed=parent['selection_seed'], include_empty=True)
    scales, scale_source = estimate_training_scales(training, capacities, limits, seed=parent['selection_seed'])
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'banks').mkdir()
    protocol = dict(schema_version=1, role='fully_pretrained_development_distribution', checkpoint_sha256=CHECKPOINT_SHA256,
        campaign=parent, code_sha256=sha256(__file__), horizons=list(HORIZONS), ks=list(KS), bank_draws=32,
        solver='euler', solver_steps=100, sample_chunk=args.sample_chunk, conditions=len(conditions), scales=scale_source,
        condition_protocol='fixed initial cohort; recorded actions; background-traffic speed labels; not a no-birth SUMO counterfactual',
        speed_target='endpoint/following-boundary mean in m/s; paired with cohort endpoint count',
        congestion_event='decoded/target count >= 5 vehicles AND speed <= 2 m/s; both target coordinates valid',
        scores='empirical V-statistic plus off-diagonal U-statistic CRPS/energy when K>1; standard Brier',
        point_reference='mean of decoded draws for each K, not decoding mean latent; intervals unavailable',
        zero_noise='same pretrained prior+FM parameters and Euler100; zero source coefficients at every depth; not regression',
        k1='one predictive draw; no variance estimate or credible/prediction interval is reported',
        mc='32 predeclared permutations; nested prefixes and disjoint K/K means; overlapping subsets are not independent replicates',
        noise='CPU Gaussian coefficient seed=stable_seed(condition_seed,source_coefficients_v1,draw_ordinal,depth)',
        final_test_used=False, training_performed=False)
    atomic_json(args.output_dir/'protocol.json', protocol)
    atomic_json(args.output_dir/'conditions.json', conditions)
    all_rows, mc_rows, bank_manifest = [], [], []
    with (args.output_dir/'states.jsonl').open('x') as stream:
        for index, item in enumerate(conditions):
            batch = validation.batch_from_starts(item['episode_index'], [item['start']], fm_only=False)
            if batch.flow.shape[1] < 10:
                raise ValueError('condition lacks complete H8 plus following-boundary speed label')
            cohort = prepare_cohort_batch(batch, capacities=capacities)
            with torch.no_grad():
                initial = model.encode(cohort.transition_flow[:, 0].to(device))
            actions, events = cohort.transition_action.to(device), cohort.transition_action_events.to(device)
            options = dict(capacities=capacities.to(device), limits=limits.to(device), seed=item['condition_seed'],
                           chunk=args.sample_chunk)
            bank = trajectory_bank(model, initial, actions, events, **options)
            zero = trajectory_bank(model, initial, actions, events, samples=1, zero_source=True, **options)
            targets = {h: endpoint_target(cohort.transition_flow[:, h], batch.flow[:, h+1], capacities, limits).numpy()
                       for h in HORIZONS}
            counts = int(cohort.cohort_count[0])
            stored = dict(horizons=np.asarray(HORIZONS), draws=np.stack([bank[h] for h in HORIZONS]),
                targets=np.stack([targets[h] for h in HORIZONS]), zero_source_draws=np.stack([zero[h] for h in HORIZONS]),
                scales=scales, capacities=capacities.numpy(), speed_limits=limits.numpy(), initial_cohort_count=np.asarray(counts),
                recorded_actions=cohort.transition_action.numpy())
            for h in HORIZONS:
                details = dict(item, horizon=h, forecast_seconds=h*30,
                    protocol='recorded_action_recursive_initial_cohort', model='fully_pretrained_prior_fm')
                rows = score_bank(bank[h], targets[h], zero[h], scales=scales, cohort_count=counts,
                                  limits=limits.numpy(), metadata=details)
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                all_rows.extend(rows)
                diagnostics, orders = nested_mean_diagnostics(bank[h], seed=item['condition_seed'])
                mc_rows.extend(dict(details, **row) for row in diagnostics)
                stored['nested_subset_ordinals'] = orders
            destination = args.output_dir/'banks'/f'condition_{index:03d}.npz'
            np.savez_compressed(destination, **stored)
            bank_manifest.append(dict(condition=item, file=str(destination.resolve()), sha256=sha256(destination)))
            atomic_json(args.output_dir/'bank_manifest.json', bank_manifest)
            stream.flush()
            print(json.dumps(dict(conditions_done=index+1, conditions_total=len(conditions))), flush=True)
    groups = defaultdict(list)
    for row in all_rows:
        stratum = 'empty_stress' if row['traffic_phase'] == 'empty_stress' else 'traffic_phases'
        groups[(row['predictor'], row['k'], row['horizon'], stratum)].append(row)
    summaries = []
    fields = ('count_mae_all', 'count_mae_occupied', 'speed_mae_occupied', 'count_crps_all',
        'count_crps_occupied', 'speed_crps_occupied', 'count_crps_u_occupied', 'speed_crps_u_occupied',
        'count_coverage90_occupied', 'speed_coverage90_occupied', 'count_width90_occupied', 'speed_width90_occupied',
        'energy_score', 'energy_score_u', 'congestion_brier_all', 'congestion_brier_occupied', 'total_count_absolute_error')
    for (predictor, k, h, stratum), rows in groups.items():
        summaries.append(dict(predictor=predictor, k=k, horizon=h, stratum=stratum,
            descriptive=summarize_state_records(rows), clustered={field:demand_bootstrap(rows, field,
                repetitions=1000, seed=parent['selection_seed']) for field in fields}))
    atomic_json(args.output_dir/'mc_diagnostics.json', dict(role='finite_bank_sensitivity_not_independent_replicates', rows=mc_rows))
    atomic_json(args.output_dir/'summary.json', dict(status='complete', conditions=len(conditions), evaluations=summaries,
        checkpoint_sha256=CHECKPOINT_SHA256, final_test_used=False))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--experiment-dir', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--max-groups', type=int, default=2)
    result.add_argument('--sample-chunk', type=int, default=4)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--threads', type=int, default=2)
    result.add_argument('--gpu-memory-fraction', type=float, default=.06)
    return result


if __name__ == '__main__':
    run(parser().parse_args())
