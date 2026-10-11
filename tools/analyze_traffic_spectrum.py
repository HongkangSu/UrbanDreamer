"""Reproducible CPU graph-spectrum diagnostic on grouped compact replay.

Validation is a development split, never claimed to be an untouched test set.
The encoder is frozen; raw-feature spectra independently check representation bias.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.linalg import eigh

from urban_dreamer.learning.offline_data import _identity, archive_content_hash
from urban_dreamer.learning.pretrain_cache import compact_transition_targets
from urban_dreamer.models.world_model import FlowProjector
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.pipeline import combined_input_hash


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_laplacian(adjacency):
    adjacency = np.asarray(adjacency, dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError('adjacency must be square')
    if not np.isfinite(adjacency).all() or (adjacency < 0).any():
        raise ValueError('adjacency must be finite and nonnegative')
    symmetric = (adjacency + adjacency.T) / 2
    degree = symmetric.sum(axis=1)
    inverse = np.maximum(degree, 1e-12) ** -.5
    return np.diag((degree > 0).astype(float)) - inverse[:, None] * symmetric * inverse[None, :]


def select_grouped(entries, split, count, seed):
    groups = {}
    for entry in entries:
        if entry['split'] == split:
            groups.setdefault(entry['demand_group'], []).append(entry)
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    selected = rng.choice(len(keys), min(count, len(keys)), replace=False)
    result = []
    for index in sorted(selected):
        candidates = sorted(groups[keys[index]], key=lambda x: x['cache_path'])
        result.append(candidates[int(rng.integers(len(candidates)))])
    return result


def select_starts(flow, count):
    if len(flow) < 2:
        raise ValueError('episode must contain transitions')
    return np.unique(np.linspace(0, len(flow) - 2, min(count, len(flow) - 1)).round().astype(int))


def read_episode(entry):
    path = Path(entry['cache_path'])
    if archive_content_hash(path) != entry['cache_content_hash']:
        raise ValueError(f"cache content mismatch: {entry['cache_path']}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['metadata'] = json.loads(arrays.pop('metadata_json').tobytes().decode())
    return arrays


def physical_features(flow, capacities, limits):
    return np.stack((flow[..., 2] * capacities, flow[..., 1] * limits), axis=-1).astype(np.float64)


def congestion(flow):
    occupied = flow[..., 2] > 0
    ratio = np.clip(1 - flow[..., 1], 0, 1)
    return (ratio * occupied).sum(axis=-1) / np.maximum(occupied.sum(axis=-1), 1)


def projection_energy(delta, basis, ranks):
    """Return per-window cumulative projected energy and total energy."""
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 3 or not np.isfinite(delta).all():
        raise ValueError('increments must be finite [windows, roads, channels]')
    if basis.shape[0] != delta.shape[1] or max(ranks) > basis.shape[1]:
        raise ValueError('projection dimensions do not match')
    coefficients = np.einsum('nr,tnc->trc', basis, delta, optimize=True)
    cumulative = np.cumsum(np.square(coefficients).sum(axis=-1), axis=-1)
    return cumulative[:, np.asarray(ranks) - 1], np.square(delta).sum(axis=(1, 2))


def bootstrap_ratio(numerators, denominators, *, repetitions=500, seed=0):
    numerators, denominators = np.asarray(numerators), np.asarray(denominators)
    if len(numerators) != len(denominators) or not len(numerators):
        raise ValueError('bootstrap requires matched nonempty episode aggregates')
    if denominators.sum() <= 0:
        return None, None, None
    estimate = float(numerators.sum() / denominators.sum())
    rng = np.random.default_rng(seed)
    index = rng.integers(len(numerators), size=(repetitions, len(numerators)))
    totals = denominators[index].sum(axis=1)
    values = numerators[index].sum(axis=1)[totals > 0] / totals[totals > 0]
    lower, upper = np.quantile(values, [.025, .975])
    return estimate, float(lower), float(upper)


def summarize(records, ranks, roads, repetitions, seed):
    result = []
    keys = sorted({(r['space'], r['stratum'], r['projection'], r['rank']) for r in records})
    for space, stratum, projection, rank in keys:
        selected = [r for r in records if (r['space'], r['stratum'], r['projection'], r['rank']) == (space, stratum, projection, rank)]
        value, lower, upper = bootstrap_ratio([r['projected_energy'] for r in selected],
            [r['total_energy'] for r in selected], repetitions=repetitions, seed=seed)
        result.append(dict(space=space, stratum=stratum, projection=projection, rank=rank,
            rank_fraction=rank / roads, energy_fraction=value, ci95_lower=lower, ci95_upper=upper,
            episodes=len(selected), windows=sum(r['windows'] for r in selected)))
    return result


def write_csv(path, rows):
    with Path(path).open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spaces = ['raw_count', 'raw_speed', 'raw_joint', 'hybrid_joint', 'latent_raw', 'latent_hybrid']
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for axis, space in zip(axes.flat, spaces):
        selected = [r for r in rows if r['space'] == space and r['stratum'] == 'all']
        spectral = [r for r in selected if r['projection'] == 'graph_low_frequency']
        x = [r['rank_fraction'] for r in spectral]
        axis.plot(x, [r['energy_fraction'] for r in spectral], 'o-', label='Graph low frequency')
        axis.fill_between(x, [r['ci95_lower'] for r in spectral], [r['ci95_upper'] for r in spectral], alpha=.2)
        random = [[r['energy_fraction'] for r in selected if r['projection'] == key]
                  for key in sorted({r['projection'] for r in selected if r['projection'].startswith('random_')})]
        if random:
            axis.plot(x, np.mean(random, axis=0), '--', label='Random subspaces mean')
            axis.fill_between(x, np.min(random, axis=0), np.max(random, axis=0), alpha=.15)
        axis.plot(x, x, ':', color='gray', label='r / M')
        rank16 = [r for r in spectral if r['rank'] == 16]
        if rank16:
            axis.axvline(rank16[0]['rank_fraction'], color='gray', alpha=.35)
        axis.set(title=space.replace('_', ' '), xlabel='Fraction of graph modes', ylabel='Increment energy fraction')
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.savefig(output / 'spectral_energy.png', dpi=180)
    figure.savefig(output / 'spectral_energy.pdf')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-groups', type=int, default=128)
    parser.add_argument('--validation-groups', type=int, default=96)
    parser.add_argument('--windows-per-episode', type=int, default=12)
    parser.add_argument('--ranks', type=int, nargs='+', default=[1, 4, 8, 16, 32, 64, 128])
    parser.add_argument('--random-seeds', type=int, nargs='+', default=[11, 23, 37, 53, 71])
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261011)
    args = parser.parse_args()
    if min(args.train_groups, args.validation_groups, args.windows_per_episode, args.bootstrap, args.threads) < 1:
        parser.error('sampling, bootstrap and thread counts must be positive')
    started = time.time()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('status') != 'complete' or manifest['identity'] != _identity({k: v for k, v in manifest.items() if k != 'identity'}):
        raise ValueError('invalid or incomplete manifest')
    if manifest['contract']['input_hashes'] != prepared.input_hashes:
        raise ValueError('manifest and configuration input mismatch')
    train_groups = {e['demand_group'] for e in manifest['entries'] if e['split'] == 'train'}
    validation_groups = {e['demand_group'] for e in manifest['entries'] if e['split'] == 'validation'}
    if train_groups & validation_groups:
        raise ValueError('demand groups overlap between training and validation')
    train = select_grouped(manifest['entries'], 'train', args.train_groups, args.seed)
    validation = select_grouped(manifest['entries'], 'validation', args.validation_groups, args.seed + 1)
    if not train or not validation:
        raise ValueError('both train and validation episodes required')
    (args.output / 'selection.json').write_text(json.dumps(dict(train=train, validation=validation), indent=2) + '\n')
    capacities = np.maximum(np.asarray(topology.network.capacities), 1.)
    limits = np.asarray(topology.network.speed_limits_mps)
    square_sum, elements, congestion_values = np.zeros(2), 0, []
    for number, entry in enumerate(train):
        arrays = read_episode(entry)
        starts = select_starts(arrays['flow'], args.windows_per_episode)
        features = physical_features(arrays['flow'], capacities, limits)
        delta = features[starts + 1] - features[starts]
        square_sum += np.square(delta).sum(axis=(0, 1))
        elements += delta.shape[0] * delta.shape[1]
        congestion_values.extend(congestion(arrays['flow'][starts]).tolist())
        if number % 16 == 0:
            print(json.dumps(dict(stage='training_scale', episode=number, total=len(train))), flush=True)
    scale = np.maximum(np.sqrt(square_sum / elements), 1e-12)
    thresholds = np.quantile(congestion_values, [1 / 3, 2 / 3])
    adjacency = np.asarray(topology.network.adjacency)
    laplacian = normalized_laplacian(adjacency)
    roads = len(adjacency)
    ranks = sorted(set(r for r in args.ranks if 1 <= r <= roads))
    if not ranks:
        raise ValueError('no valid ranks')
    eigenvalues, basis = eigh(laplacian, subset_by_index=[0, max(ranks) - 1], check_finite=True)
    residual = float(np.linalg.norm(laplacian @ basis - basis * eigenvalues, axis=0).max())
    projections = {'graph_low_frequency': basis}
    for seed in args.random_seeds:
        projections[f'random_{seed}'] = np.linalg.qr(np.random.default_rng(seed).normal(size=(roads, max(ranks))), mode='reduced')[0]
    checkpoint_hash = sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if payload['input_hash'] != combined_input_hash(prepared):
        raise ValueError('checkpoint input hash does not match config')
    state = payload['training_state']['models']['world_model']
    checkpoint_basis = state['dynamics.graph_prior.basis'].numpy().astype(np.float64)
    checkpoint_rank = checkpoint_basis.shape[1]
    checkpoint_basis, _ = np.linalg.qr(checkpoint_basis, mode='reduced')
    projections['checkpoint_prior_basis'] = checkpoint_basis
    checkpoint_basis_overlap = float(np.square(basis[:, :checkpoint_rank].T @ checkpoint_basis).sum() / checkpoint_rank)
    encoder = FlowProjector(adjacency=torch.as_tensor(adjacency, dtype=torch.float32), positive_speed=config.codec.speed_positive)
    encoder.load_state_dict({k.removeprefix('flow_projector.'): v for k, v in state.items() if k.startswith('flow_projector.')}, strict=True)
    encoder.eval().requires_grad_(False)
    records = []
    for number, entry in enumerate(validation):
        arrays = read_episode(entry)
        starts = select_starts(arrays['flow'], args.windows_per_episode)
        before, after = arrays['flow'][starts], arrays['flow'][starts + 1]
        target, _ = compact_transition_targets(arrays, starts.tolist(), 1, capacities)
        target = target[:, 0]
        raw = (physical_features(after, capacities, limits) - physical_features(before, capacities, limits)) / scale
        hybrid = (physical_features(target, capacities, limits) - physical_features(before, capacities, limits)) / scale
        with torch.inference_mode():
            z_before = encoder(torch.as_tensor(before, dtype=torch.float32))
            z_after = encoder(torch.as_tensor(after, dtype=torch.float32))
            z_target = encoder(torch.as_tensor(target, dtype=torch.float32))
        spaces = dict(raw_count=raw[..., :1], raw_speed=raw[..., 1:], raw_joint=raw,
                      hybrid_joint=hybrid, latent_raw=(z_after - z_before).numpy(), latent_hybrid=(z_target - z_before).numpy())
        congestion_levels = np.digitize(congestion(before), thresholds)
        strata = dict(all=np.ones(len(starts), dtype=bool), low=congestion_levels == 0,
                      medium=congestion_levels == 1, high=congestion_levels == 2)
        for space, delta in spaces.items():
            for projection, directions in projections.items():
                projection_ranks = [checkpoint_rank] if projection == 'checkpoint_prior_basis' else ranks
                numerator, denominator = projection_energy(delta, directions, projection_ranks)
                for stratum, mask in strata.items():
                    if not mask.any():
                        continue
                    for rank_index, rank in enumerate(projection_ranks):
                        records.append(dict(demand_group=entry['demand_group'], episode=entry['episode'], method=entry['method'],
                            space=space, stratum=stratum, projection=projection, rank=rank,
                            projected_energy=float(numerator[mask, rank_index].sum()), total_energy=float(denominator[mask].sum()), windows=int(mask.sum())))
        print(json.dumps(dict(stage='validation_spectrum', episode=number + 1, total=len(validation), elapsed_seconds=time.time() - started)), flush=True)
    rows = summarize(records, ranks, roads, args.bootstrap, args.seed)
    write_csv(args.output / 'episode_energy.csv', records)
    write_csv(args.output / 'spectral_summary.csv', rows)
    plot_results(rows, args.output)
    if sha256(args.checkpoint) != checkpoint_hash:
        raise ValueError('checkpoint changed during analysis')
    result = dict(status='complete', analysis_split='development_validation', seed=args.seed, roads=roads,
        train_groups=len(train), validation_groups=len(validation), windows_per_episode=args.windows_per_episode,
        increment_scale_rms=dict(count=float(scale[0]), speed_mps=float(scale[1])),
        scaling='Training sampled raw increment RMS per channel; increments are not mean-centered.',
        congestion_definition='Mean clipped (1-speed_ratio) on occupied roads; thresholds from training windows.',
        congestion_thresholds=thresholds.tolist(), eigenvalues=eigenvalues.tolist(), max_eigen_residual=residual,
        checkpoint_basis_rank=checkpoint_rank, checkpoint_to_exact_low_subspace_overlap=checkpoint_basis_overlap,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
        config=str(args.config.resolve()), config_sha256=sha256(args.config), manifest_identity=manifest['identity'],
        script_sha256=sha256(__file__), checkpoint_phase=payload['phase'], checkpoint_next_unit=payload['next_unit'],
        bootstrap='Resample episode aggregates (one episode per unique demand group), ratio of summed energies; 95% percentile interval.',
        bootstrap_repetitions=args.bootstrap, random_seeds=args.random_seeds,
        caveats=['Validation was used in model development; this is not an untouched test result.',
                 'Frozen learned encoder can itself favor low frequencies; raw feature results are the independent control.',
                 'Hybrid target removes current-window births from count only; speed retains raw interval semantics.',
                 'Partial spectrum only: full cumulative curve would require additional ranks.',
                 'Random-subspace band measures projection-seed range, not a statistical confidence interval.'],
        elapsed_seconds=time.time() - started, summary=rows)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(stage='complete', output=str(args.output), elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
