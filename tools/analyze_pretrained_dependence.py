"""Post-hoc dependence diagnostics on immutable fully pretrained K32 banks.

Spatial shuffling preserves each road's complete marginal trajectory. Horizon
shuffling preserves each horizon's complete joint state distribution. Neither
reference is a causal rollout or a newly trained model. The existing bank is
reused without sampling a new world-model prediction or fitting any scale.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.component_experiment_data import atomic_json, sha256
from tools.summarize_component_predictions import demand_bootstrap

CHECKPOINT_SHA = '2f63f0b2a5fe18f3a3bd9be27ffd1e63a2755464e7f2944e1227334f0ffe6af6'
REFERENCES = ('original', 'spatial_shuffle', 'horizon_shuffle')
CONDITION_KEYS = ('demand_group', 'episode', 'start', 'traffic_phase', 'condition_seed')


def condition_key(condition):
    return tuple(condition[key] for key in CONDITION_KEYS)


def _array_digest(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str(value.shape).encode() + str(value.dtype).encode() + value.tobytes()).hexdigest()


def _private_seed(base, condition):
    encoded = json.dumps([int(base), list(condition_key(condition))], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'little')


def validate_pairs(pairs, roads):
    array = np.asarray(pairs)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if (array.ndim != 2 or array.shape[1] != 2 or not np.issubdtype(array.dtype, np.integer)
            or (array < 0).any() or (array >= roads).any() or (array[:, 0] >= array[:, 1]).any()):
        raise ValueError('pairs require unique canonical integer road indices 0 <= i < j < E')
    if len(set(map(tuple, array.tolist()))) != len(array):
        raise ValueError('duplicate road pairs')
    return array.astype(np.int64, copy=False)


def select_pairs(adjacency, *, seed, nonneighbors=1024):
    graph = np.asarray(adjacency)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1] or not np.isfinite(graph).all() or (graph < 0).any():
        raise ValueError('finite nonnegative square adjacency required')
    if not isinstance(nonneighbors, int) or isinstance(nonneighbors, bool) or nonneighbors < 1:
        raise ValueError('positive nonneighbor pair count required')
    roads = len(graph)
    i, j = np.triu_indices(roads, 1)
    connected = (graph[i, j] > 0) | (graph[j, i] > 0)
    adjacent = np.stack((i[connected], j[connected]), axis=-1)
    complement = np.stack((i[~connected], j[~connected]), axis=-1)
    if len(complement) < nonneighbors:
        raise ValueError('network has fewer nonneighbor pairs than prescribed')
    indices = np.random.default_rng(seed).choice(len(complement), size=nonneighbors, replace=False)
    distant = complement[np.sort(indices)]
    return adjacent, distant


def _validate_draws(draws):
    array = np.asarray(draws)
    if array.ndim != 4 or array.shape[1] < 2 or min(array.shape) < 1 or array.shape[-1] != 2 or not np.isfinite(array).all():
        raise ValueError('finite (H,K,E,2) draws with K>=2 required')
    return array


def spatial_shuffle(draws, *, seed):
    array = _validate_draws(draws)
    _, count, roads, _ = array.shape
    rng = np.random.default_rng(seed)
    permutations = np.stack([rng.permutation(count) for _ in range(roads)], axis=1)
    result = np.take_along_axis(array, permutations[None, :, :, None], axis=1)
    return result, permutations


def horizon_shuffle(draws, *, seed):
    array = _validate_draws(draws)
    rng = np.random.default_rng(seed)
    permutations = np.stack([rng.permutation(array.shape[1]) for _ in range(array.shape[0])])
    return np.take_along_axis(array, permutations[:, :, None, None], axis=1), permutations


def _validate_scores(samples, target, scales, target_mask=None):
    samples, target, scales = np.asarray(samples, dtype=np.float64), np.asarray(target, dtype=np.float64), np.asarray(scales, dtype=np.float64)
    if (samples.ndim < 3 or samples.shape[0] < 2 or samples.shape[1:] != target.shape or target.shape[-1] != 2
            or scales.shape != (2,) or not np.isfinite(scales).all() or (scales <= 0).any()
            or not np.isfinite(samples).all() or np.isinf(target).any()):
        raise ValueError('finite K>=2 samples, matching targets and two positive finite training scales required')
    mask = np.isfinite(target)
    if target_mask is not None:
        supplied = np.asarray(target_mask)
        if supplied.dtype != bool or supplied.shape != target.shape or (supplied & ~mask).any():
            raise ValueError('target_mask must be matching boolean and cannot select nonfinite labels')
        mask &= supplied
    return samples, target, scales, mask


def _canonical_vectors(vectors):
    # Byte ordering is arbitrary but deterministic, and invariant to a row
    # permutation. It makes finite-precision score equality directly testable.
    vectors = np.ascontiguousarray(vectors)
    order = sorted(range(len(vectors)), key=lambda index: vectors[index].tobytes())
    return vectors[order]


def energy_score_u(samples, target, scales, target_mask=None):
    samples, target, scales, mask = _validate_scores(samples, target, scales, target_mask)
    dimensions = int(mask.sum())
    if not dimensions:
        return dict(score=None, dimensions=0)
    values = _canonical_vectors((samples / scales)[:, mask])
    truth = (target / scales)[mask]
    first = np.linalg.norm(values - truth, axis=1).mean()
    # Sum each unordered pair exactly once; equivalent to the off-diagonal
    # ordered sum divided by 2*K*(K-1), without a K*K*D allocation.
    pair_sum = sum(np.linalg.norm(values[index + 1:] - values[index], axis=1).sum()
                   for index in range(len(values) - 1))
    score = first - pair_sum / (len(values) * (len(values) - 1))
    return dict(score=float(score), dimensions=dimensions)


def variogram_scores(samples, target, scales, pairs, *, target_mask=None, power=.5):
    samples, target, scales, valid = _validate_scores(samples, target, scales, target_mask)
    if target.ndim != 2:
        raise ValueError('spatial variogram target must have (E,2) shape')
    if power != .5:
        raise ValueError('this analysis prespecifies variogram power 0.5')
    pairs = validate_pairs(pairs, len(target))
    values, truth = samples / scales, target / scales
    result, all_plugin, all_u = {}, [], []
    for feature, name in enumerate(('count', 'speed')):
        selected = valid[pairs[:, 0], feature] & valid[pairs[:, 1], feature]
        a, b = pairs[selected].T
        count = len(a)
        plugin = corrected = None
        if count:
            observed = np.abs(truth[a, feature] - truth[b, feature]) ** power
            # Sorting transformed draws preserves the pair's sample statistics
            # under a whole-state horizon permutation, exactly in float64.
            powered = np.sort(np.abs(values[:, a, feature] - values[:, b, feature]) ** power, axis=0)
            errors = np.square(observed - powered.mean(0))
            # Unbiased estimate of squared population discrepancy for IID
            # original draws: plugin - sample_var(powered)/K. Shuffled draws
            # are finite-pool references, so this label makes no IID claim.
            corrected_errors = errors - powered.var(0, ddof=1) / len(samples)
            plugin, corrected = float(errors.mean()), float(corrected_errors.mean())
            all_plugin.extend(errors.tolist())
            all_u.extend(corrected_errors.tolist())
        result[name] = dict(plugin=plugin, u_corrected=corrected, valid_pairs=count,
                            omitted_pairs=len(pairs) - count)
    result['joint'] = dict(plugin=float(np.mean(all_plugin)) if all_plugin else None,
        u_corrected=float(np.mean(all_u)) if all_u else None, valid_pair_features=len(all_plugin))
    return result


def marginal_statistics(draws, targets):
    array = _validate_draws(draws).astype(np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (array.shape[0], array.shape[2], 2) or np.isinf(targets).any():
        raise ValueError('marginal targets must match (H,E,2), with NaN only for missing labels')
    ordered = np.sort(array, axis=1)
    count = array.shape[1]
    coefficients = (2 * np.arange(1, count + 1) - count - 1)[None, :, None, None]
    first = np.abs(ordered - targets[:, None]).mean(1)
    pair_term = (ordered * coefficients).sum(1)
    result = dict(sorted=ordered, mean=ordered.mean(1), crps_v=first - pair_term / count ** 2,
                  crps_u=first - pair_term / (count * (count - 1)))
    for level in (.5, .8, .9, .95):
        lower, upper = np.quantile(ordered, [(1 - level) / 2, (1 + level) / 2], axis=1)
        result[f'picp_{level}'] = np.where(np.isfinite(targets),
            ((targets >= lower) & (targets <= upper)).astype(float), np.nan)
    return result


def verify_invariance(original, spatial, temporal, targets, scales):
    before = marginal_statistics(original, targets)
    for name, changed in (('spatial', spatial), ('temporal', temporal)):
        after = marginal_statistics(changed, targets)
        for metric in before:
            if not np.array_equal(before[metric], after[metric], equal_nan=True):
                raise ValueError(f'{name} shuffle changed exact marginal {metric}')
    for road in range(original.shape[2]):
        left = _canonical_vectors(original[:, :, road, :].transpose(1, 0, 2).reshape(original.shape[1], -1))
        right = _canonical_vectors(spatial[:, :, road, :].transpose(1, 0, 2).reshape(original.shape[1], -1))
        if not np.array_equal(left, right):
            raise ValueError('spatial shuffle changed local count-speed/temporal trajectories')
    for horizon in range(original.shape[0]):
        if not np.array_equal(_canonical_vectors(original[horizon].reshape(original.shape[1], -1)),
                              _canonical_vectors(temporal[horizon].reshape(original.shape[1], -1))):
            raise ValueError('horizon shuffle changed a full joint state distribution')
        if energy_score_u(original[horizon], targets[horizon], scales) != energy_score_u(temporal[horizon], targets[horizon], scales):
            raise ValueError('horizon shuffle changed exact per-horizon energy score')
    return dict(exact_marginals=True, exact_mean_crps_picp=True,
                spatial_local_trajectory_multisets=True, temporal_joint_state_multisets=True,
                temporal_per_horizon_energy_exact=True,
                arithmetic='canonical sorted float64 sums eliminate permutation-dependent rounding')


def mass_diagnosis(predicted_mean, target_count):
    predicted, target = np.asarray(predicted_mean, dtype=np.float64), np.asarray(target_count, dtype=np.float64)
    if predicted.ndim != 1 or target.shape != predicted.shape or not np.isfinite(predicted).all() or np.isinf(target).any():
        raise ValueError('mass diagnosis requires matching road vectors and finite predictions')
    if np.any(target[np.isfinite(target)] < 0):
        raise ValueError('target count must be nonnegative')
    if not np.isfinite(target).all():
        return dict(status='missing_count_labels', missing_count_labels=int(np.isnan(target).sum()),
                    normalized_spatial_tv=None)
    occupied = target > 0
    bias = predicted - target
    occupied_bias = float(bias[occupied].sum())
    ghost = float(predicted[~occupied].sum())
    missing = float(np.maximum(-bias[occupied], 0).sum())
    surplus = float(np.maximum(bias[occupied], 0).sum())
    total_true, total_predicted = float(target.sum()), float(predicted.sum())
    global_bias = total_predicted - total_true
    residual = global_bias - occupied_bias - ghost
    if not np.isclose(residual, 0., atol=1e-10 * max(1., abs(global_bias), total_true), rtol=0):
        raise ValueError('global signed mass decomposition failed')
    tv, status = None, 'ok'
    if total_true <= 0:
        status = 'no_positive_target_mass'
    elif (predicted < 0).any():
        status = 'negative_predicted_mass'
    elif total_predicted <= 0:
        status = 'no_positive_predicted_mass'
    else:
        tv = float(.5 * np.abs(predicted / total_predicted - target / total_true).sum())
    return dict(status=status, occupied_roads=int(occupied.sum()), true_total_count=total_true,
        predicted_total_count=total_predicted, occupied_signed_bias=occupied_bias,
        ghost_mass_true_empty=ghost, positive_ghost_mass=float(np.maximum(predicted[~occupied], 0).sum()),
        negative_empty_mass=float(np.maximum(-predicted[~occupied], 0).sum()),
        missing_occupied_mass=missing, surplus_occupied_mass=surplus,
        global_signed_mass=global_bias, decomposition_residual=residual,
        raw_spatial_l1=float(np.abs(bias).sum()), normalized_spatial_tv=tv)


def prepare_pairs(experiment_dir, prespecification, destination):
    from urban_dreamer.workflow.config import load_config
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.topology import load_traffic_topology
    destination = Path(destination)
    if destination.exists():
        return json.loads(destination.read_text())
    config = load_config(Path(experiment_dir) / 'snapshot/run_config.toml')
    prepared = validate_prepared_inputs(config)
    network = load_traffic_topology(prepared).network
    adjacent, distant = select_pairs(network.adjacency,
        seed=prespecification['nonneighbor_pair_seed'], nonneighbors=prespecification['nonneighbor_pairs'])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        network_xml=str(prepared.network_xml), network_sha256=sha256(prepared.network_xml),
        edge_registry=str(prepared.edge_registry), registry_sha256=sha256(prepared.edge_registry),
        edge_ids=list(network.edge_ids), adjacency_sha256=_array_digest(network.adjacency),
        adjacent=adjacent.tolist(), nonneighbor=distant.tolist(),
        seed=prespecification['nonneighbor_pair_seed'], selection='prespecified without prediction/target access')
    atomic_json(destination, result)
    return result


def _source_files(source, experiment_dir, pairs):
    files = {str(source / name): sha256(source / name)
             for name in ('protocol.json', 'conditions.json', 'bank_manifest.json', 'summary.json')}
    checkpoint = Path(experiment_dir) / 'snapshot/checkpoints/latest.pt'
    if sha256(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('only the declared fully pretrained checkpoint may be analyzed')
    files[str(checkpoint)] = CHECKPOINT_SHA
    manifest_path = source.parent / 'source_manifest.json'
    files[str(manifest_path)] = sha256(manifest_path)
    frozen = json.loads(manifest_path.read_text())
    for relative, expected in frozen.items():
        path = source.parent / 'source' / relative
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f'frozen evaluation source changed: {relative}')
        files[str(path)] = observed
    for key, expected in (('network_xml', 'network_sha256'), ('edge_registry', 'registry_sha256')):
        if sha256(pairs[key]) != pairs[expected]:
            raise ValueError('prespecified network/registry changed')
        files[pairs[key]] = pairs[expected]
    return files


def _aggregate(rows, metrics, *, prespecification, pilot):
    groups = defaultdict(list)
    for row in rows:
        groups[row['scope'], row.get('horizon'), row['stratum']].append(row)
    results = []
    for (scope, horizon, stratum), selected in sorted(groups.items(), key=lambda item: str(item[0])):
        index = {(r['condition_id'], r['reference']): r for r in selected}
        for metric in metrics:
            for reference in REFERENCES:
                records = [r for r in selected if r['reference'] == reference]
                if not any(metric in r for r in records):
                    continue
                estimate = demand_bootstrap(records, metric, seed=prespecification['bootstrap_seed'],
                    repetitions=prespecification['bootstrap_repetitions'])
                if pilot:
                    estimate.update(ci_low=None, ci_high=None)
                results.append(dict(scope=scope, horizon=horizon, stratum=stratum, metric=metric,
                    contrast=reference, states=len(records), available_states=sum(r.get(metric) is not None for r in records),
                    inference='descriptive_two_group_pilot' if pilot else 'paired_demand_group_bootstrap', **estimate))
                if reference != 'original':
                    differences = []
                    for row in records:
                        original = index[row['condition_id'], 'original']
                        a, b = original.get(metric), row.get(metric)
                        if (a is None) != (b is None):
                            raise ValueError('paired score target availability differs')
                        differences.append(dict(row, delta=None if a is None else a - b))
                    paired = demand_bootstrap(differences, 'delta', seed=prespecification['bootstrap_seed'],
                        repetitions=prespecification['bootstrap_repetitions'])
                    if pilot:
                        paired.update(ci_low=None, ci_high=None)
                    results.append(dict(scope=scope, horizon=horizon, stratum=stratum, metric=metric,
                        contrast='original_minus_' + reference, states=len(records),
                        available_states=sum(r['delta'] is not None for r in differences),
                        inference='descriptive_two_group_pilot' if pilot else 'paired_demand_group_bootstrap', **paired))
    return results


def analyze(source, output, *, experiment_dir, prespecification_path, pairs_path):
    import torch
    torch.set_num_threads(2)
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('new output directory required; preserve prior results')
    prespecification = json.loads(Path(prespecification_path).read_text())
    pairs = prepare_pairs(experiment_dir, prespecification, pairs_path)
    protocol = json.loads((source / 'protocol.json').read_text())
    summary = json.loads((source / 'summary.json').read_text())
    if (summary.get('status') != 'complete' or protocol.get('role') != 'fully_pretrained_development_distribution'
            or protocol.get('checkpoint_sha256') != CHECKPOINT_SHA or protocol.get('training_performed') is not False):
        raise ValueError('complete fully pretrained, no-training bank evaluation required')
    manifest = json.loads((source / 'bank_manifest.json').read_text())
    conditions = json.loads((source / 'conditions.json').read_text())
    keys = [condition_key(c) for c in conditions]
    if len(keys) != len(set(keys)) or len(keys) != protocol['conditions'] or len(keys) != summary['conditions']:
        raise ValueError('duplicate/missing declared condition identities')
    if len(manifest) != len(keys) or {condition_key(row['condition']) for row in manifest} != set(keys):
        raise ValueError('banks must cover the full declared condition intersection')
    hashes = _source_files(source, experiment_dir, pairs)
    for artifact in (Path(prespecification_path), Path(pairs_path)):
        hashes[str(artifact.resolve())] = sha256(artifact)
    banks = []
    for row in manifest:
        path = Path(row['file'])
        if path.resolve().parent != (source / 'banks').resolve() or sha256(path) != row['sha256']:
            raise ValueError('bank location or immutable hash mismatch')
        hashes[str(path)] = row['sha256']
        banks.append((row['condition'], path))
    groups = {row['demand_group'] for row in conditions}
    pilot = len(groups) <= 2
    if not pilot and len(groups) != 8:
        raise ValueError('formal dependence analysis requires the prescribed eight demand groups')
    adjacent = validate_pairs(pairs['adjacent'], len(pairs['edge_ids']))
    distant = validate_pairs(pairs['nonneighbor'], len(pairs['edge_ids']))
    if len(distant) != prespecification['nonneighbor_pairs'] or set(map(tuple, adjacent)) & set(map(tuple, distant)):
        raise ValueError('prespecified adjacent/nonneighbor pair sets inconsistent')
    rows, checks, permutations = [], [], {}
    for index, (condition, path) in enumerate(banks):
        with np.load(path, allow_pickle=False) as saved:
            horizons, raw, targets = saved['horizons'], saved['draws'], saved['targets']
            scales = saved['scales']
        if (raw.shape != (4, 32, 1, len(pairs['edge_ids']), 2)
                or targets.shape != (4, 1, len(pairs['edge_ids']), 2)
                or not np.array_equal(horizons, [1, 2, 4, 8]) or not np.array_equal(scales, protocol['scales']['scales'])):
            raise ValueError('bank shape/horizon/training-scale identity mismatch')
        draws, targets = raw[:, :, 0], targets[:, 0]
        _validate_draws(draws)
        _validate_scores(draws.transpose(1, 0, 2, 3), targets, scales)
        spatial, road_orders = spatial_shuffle(draws, seed=_private_seed(prespecification['spatial_shuffle_seed'], condition))
        temporal, horizon_orders = horizon_shuffle(draws, seed=_private_seed(prespecification['horizon_shuffle_seed'], condition))
        checks.append(dict(condition_id=index, **verify_invariance(draws, spatial, temporal, targets, scales)))
        permutations[f'condition_{index:03d}_road_orders'] = road_orders
        permutations[f'condition_{index:03d}_horizon_orders'] = horizon_orders
        mean = marginal_statistics(draws, targets)['mean']
        base = dict(condition, condition_id=index,
                    stratum='empty_stress' if condition['traffic_phase'] == 'empty_stress' else 'traffic_phases')
        for reference, bank in zip(REFERENCES, (draws, spatial, temporal)):
            trajectory = energy_score_u(bank.transpose(1, 0, 2, 3), targets, scales)
            rows.append(dict(base, scope='trajectory', horizon=None, reference=reference,
                trajectory_energy_u=trajectory['score'], target_dimensions=trajectory['dimensions']))
            for h, horizon in enumerate(horizons.tolist()):
                energy = energy_score_u(bank[h], targets[h], scales)
                row = dict(base, scope='horizon', horizon=horizon, reference=reference,
                    energy_u=energy['score'], target_dimensions=energy['dimensions'])
                for label, selected_pairs in (('adjacent', adjacent), ('nonneighbor', distant)):
                    scores = variogram_scores(bank[h], targets[h], scales, selected_pairs)
                    for feature in ('count', 'speed', 'joint'):
                        row[f'{label}_{feature}_vs'] = scores[feature]['plugin']
                        row[f'{label}_{feature}_vs_u_corrected'] = scores[feature]['u_corrected']
                        row[f'{label}_{feature}_valid_pairs'] = scores[feature].get('valid_pairs', scores[feature].get('valid_pair_features'))
                mass = mass_diagnosis(mean[h, :, 0], targets[h, :, 0])
                row.update({'mass_' + key: value for key, value in mass.items()})
                rows.append(row)
    metrics = ['energy_u', 'trajectory_energy_u'] + [f'{pairs}_{feature}_vs{suffix}'
        for pairs in ('adjacent', 'nonneighbor') for feature in ('count', 'speed', 'joint')
        for suffix in ('', '_u_corrected')] + ['mass_' + name for name in (
            'occupied_signed_bias', 'ghost_mass_true_empty', 'missing_occupied_mass', 'surplus_occupied_mass',
            'global_signed_mass', 'normalized_spatial_tv', 'true_total_count', 'predicted_total_count', 'raw_spatial_l1')]
    aggregates = _aggregate(rows, metrics, prespecification=prespecification, pilot=pilot)
    after = {path: sha256(path) for path in hashes}
    if hashes != after:
        raise RuntimeError('bank/source/checkpoint/plan inputs changed during dependence analysis')
    output.mkdir(parents=True)
    np.savez_compressed(output / 'permutations.npz', **permutations)
    (output / 'states.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
    atomic_json(output / 'aggregates.json', aggregates)
    atomic_json(output / 'invariance.json', checks)
    atomic_json(output / 'provenance.json', dict(before=hashes, after=after, unchanged=True,
        analysis_code_sha256=sha256(__file__), source=str(source.resolve()),
        prespecification_sha256=sha256(prespecification_path), pairs_sha256=sha256(pairs_path),
        score_reference='Scheuerer and Hamill (2015), equations 3 and 4, DOI 10.1175/MWR-D-14-00269.1',
        score_reference_url='https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf',
        variogram_warning='VS is proper but not strictly proper; shared additive errors cancel. U correction is supplementary and may be negative; finite-pool synthetic draws are not IID new forecasts.',
        reference_warning='Synthetic dependence diagnostics only; not causal trajectories or independently trained/replicated models. No training or held-out scale fitting.',
        marginal_statistics='Sorted float64 canonical sums; exact per-road marginal, mean, CRPS and PICP equality verified.',
        energy_dimensions='finite target coordinates only; saved training scales; no dimension normalization',
        mass_interpretation='Raw mean counts, no clipping; occupied signed bias + empty signed ghost mass = global signed mass. TV uses normalized nonnegative mass and preserves invalid/no-positive-target cases explicitly.',
        cpu_threads=2))
    report = dict(status='complete', conditions=len(conditions), demand_groups=len(groups),
        pilot_descriptive_only=pilot, adjacent_pairs=len(adjacent), nonneighbor_pairs=len(distant),
        scored_rows=len(rows), invariance_conditions_passed=len(checks),
        target_dimensions_complete=all(row['target_dimensions'] == len(pairs['edge_ids']) * 2 * (4 if row['scope'] == 'trajectory' else 1) for row in rows),
        no_positive_target_mass_rows=sum(row.get('mass_status') == 'no_positive_target_mass' for row in rows),
        unavailable_mass_rows=sum(row.get('scope') == 'horizon' and row.get('mass_status') != 'ok' for row in rows),
        immutable_inputs_verified=len(hashes), aggregates=aggregates)
    atomic_json(output / 'summary.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--experiment-dir', type=Path, default=ROOT / 'outputs/component_experiments_20261011')
    parser.add_argument('--prespecification', type=Path, default=ROOT / 'outputs/pretrained_evaluation_20261012/dependence_v1/prespecification.json')
    parser.add_argument('--pairs', type=Path, default=ROOT / 'outputs/pretrained_evaluation_20261012/dependence_v1/pairs.json')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--wait-for-complete', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        result = prepare_pairs(args.experiment_dir, json.loads(args.prespecification.read_text()), args.pairs)
        print(json.dumps(dict(adjacent_pairs=len(result['adjacent']), nonneighbor_pairs=len(result['nonneighbor']))))
    else:
        if args.source is None or args.output_dir is None:
            parser.error('--source and --output-dir required for analysis')
        if args.wait_for_complete:
            while not (args.source / 'summary.json').exists():
                time.sleep(30)
        result = analyze(args.source, args.output_dir, experiment_dir=args.experiment_dir,
                         prespecification_path=args.prespecification, pairs_path=args.pairs)
        print(json.dumps({key: value for key, value in result.items() if key != 'aggregates'}, indent=2))
