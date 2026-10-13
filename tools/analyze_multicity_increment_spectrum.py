"""CPU-only prespecified spectra of observed 30-second traffic increments.

Uses real-city-network simulation replay, not field sensor measurements. No
encoder, cohort target, temporal padding, centering, or model training is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
import torch

from tools.analyze_traffic_spectrum import (
    bootstrap_ratio, congestion, normalized_laplacian, physical_features,
    read_episode, select_grouped, select_starts, sha256, write_csv,
)
from urban_dreamer.learning.offline_data import _identity
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config

SOURCES = {
    'ingolstadt_custom': (
        'outputs/ingolstadt_full_retrain_20261009/world_model/run_config.toml',
        'outputs/ingolstadt_full_retrain_20261009/data/cache_manifest.json'),
    'coulommiers': (
        'outputs/two_city_urbandreamer_pretrain_supplement_20261012/coulommiers/base_config.toml',
        'outputs/two_city_urbandreamer_pretrain_supplement_20261012/coulommiers/cache/cache_manifest.json'),
    'montereau_fault_yonne': (
        'outputs/two_city_urbandreamer_pretrain_supplement_20261012/montereau_fault_yonne/base_config.toml',
        'outputs/two_city_urbandreamer_pretrain_supplement_20261012/montereau_fault_yonne/cache/cache_manifest.json'),
}
FRACTIONS = [.05, .10, .20, .40, .60, .80, 1.]
CUTOFFS = [.1, .25, .5]
SPACES = ['raw_count', 'raw_speed', 'raw_joint']
STRATA = ['all', 'low', 'medium', 'high']


def raw_pairs(arrays, starts, capacities, limits, *, interval=30.):
    """Only genuine adjacent observation boundaries; retain births and mean."""
    flow, times = arrays['flow'], arrays['time']
    starts = np.asarray(starts, dtype=int)
    if len(times) != len(flow) or np.any(starts < 0) or np.any(starts + 1 >= len(flow)):
        raise ValueError('missing actual endpoint; terminal padding is prohibited')
    if not np.allclose(times[starts + 1] - times[starts], interval, rtol=0, atol=1e-8):
        raise ValueError('observations do not have the prespecified 30-second interval')
    before = physical_features(flow[starts], capacities, limits)
    after = physical_features(flow[starts + 1], capacities, limits)
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('nonfinite observed physical features')
    return before, after, after - before


def training_scale(deltas):
    """RMS without mean subtraction; callers supply the training partition."""
    square_sum, elements = np.zeros(2), 0
    for delta in deltas:
        square_sum += np.square(delta).sum(axis=(0, 1))
        elements += delta.shape[0] * delta.shape[1]
    if not elements:
        raise ValueError('training increments required for scaling')
    rms = np.sqrt(square_sum / elements)
    return np.maximum(rms, 1e-12), rms, square_sum, elements


def mode_energy(delta, basis):
    if delta.ndim != 3 or not np.isfinite(delta).all() or basis.shape != (delta.shape[1], delta.shape[1]):
        raise ValueError('finite increments and a complete square graph basis required')
    coefficients = np.einsum('nr,tnc->trc', basis, delta, optimize=True)
    channel_energy = np.square(coefficients)
    energy = np.stack((channel_energy[..., 0], channel_energy[..., 1], channel_energy.sum(axis=-1)), axis=1)
    total = np.stack((np.square(delta[..., 0]).sum(axis=-1), np.square(delta[..., 1]).sum(axis=-1),
                      np.square(delta).sum(axis=(1, 2))), axis=1)
    np.testing.assert_allclose(energy.sum(axis=-1), total, rtol=1e-9, atol=1e-9)
    return energy, total


def group_aggregates(energy, total, labels):
    """Aggregate windows inside a demand group before resampling groups."""
    numerators = np.zeros((len(STRATA), len(SPACES), energy.shape[-1]))
    denominators = np.zeros((len(STRATA), len(SPACES)))
    windows = np.zeros(len(STRATA), dtype=int)
    for index, name in enumerate(STRATA):
        selected = np.ones(len(labels), dtype=bool) if name == 'all' else labels == index - 1
        numerators[index] = energy[selected].sum(axis=0)
        denominators[index] = total[selected].sum(axis=0)
        windows[index] = selected.sum()
    return numerators, denominators, windows


def prespecified_cuts(eigenvalues):
    roads = len(eigenvalues)
    cuts = [('rank', str(rank), rank) for rank in [16, 32, 64, 128] if rank <= roads]
    cuts += [('mode_fraction', str(fraction), int(np.ceil(fraction * roads))) for fraction in FRACTIONS]
    cuts += [('lambda_cutoff', str(cutoff), int(np.count_nonzero(eigenvalues <= cutoff))) for cutoff in CUTOFFS]
    return cuts


def complete_eigenspace_cut(eigenvalues, nominal_modes, tolerance=1e-8):
    """Include a boundary eigenvalue block, making capture rotation invariant."""
    if not nominal_modes:
        return 0
    return int(np.searchsorted(eigenvalues, eigenvalues[nominal_modes - 1] + tolerance, side='right'))


def summarize(city, eigenvalues, numerators, denominators, windows, bootstrap, seed):
    cumulative = np.cumsum(numerators, axis=-1)
    rows = []
    for s, stratum in enumerate(STRATA):
        for f, space in enumerate(SPACES):
            keep = windows[:, s] > 0
            denominator = denominators[keep, s, f]
            for kind, requested, nominal_modes in prespecified_cuts(eigenvalues):
                modes = (complete_eigenspace_cut(eigenvalues, nominal_modes)
                         if kind != 'lambda_cutoff' else nominal_modes)
                numerator = cumulative[keep, s, f, modes - 1] if modes else np.zeros(keep.sum())
                value, lower, upper = (bootstrap_ratio(numerator, denominator, repetitions=bootstrap, seed=seed)
                                       if keep.any() else (None, None, None))
                rows.append(dict(city=city, space=space, stratum=stratum, cut_kind=kind, requested=requested,
                    nominal_modes=nominal_modes, modes=modes, expanded_boundary_tie=modes != nominal_modes,
                    roads=len(eigenvalues), mode_fraction=modes / len(eigenvalues),
                    lambda_max_included=float(eigenvalues[modes - 1]) if modes else None,
                    energy_fraction=value, ci95_lower=lower, ci95_upper=upper,
                    uniform_energy_reference=modes / len(eigenvalues), groups=int(keep.sum()),
                    zero_energy_groups=int(np.count_nonzero(denominator == 0)), windows=int(windows[keep, s].sum())))
            centroid_num = (numerators[keep, s, f] * eigenvalues).sum(axis=-1)
            value, lower, upper = (bootstrap_ratio(centroid_num, denominator, repetitions=bootstrap, seed=seed)
                                   if keep.any() else (None, None, None))
            rows.append(dict(city=city, space=space, stratum=stratum, cut_kind='spectral_centroid', requested='lambda / 2',
                nominal_modes=len(eigenvalues), modes=len(eigenvalues), expanded_boundary_tie=False,
                roads=len(eigenvalues), mode_fraction=1., lambda_max_included=float(eigenvalues[-1]),
                energy_fraction=None if value is None else value / 2,
                ci95_lower=None if lower is None else lower / 2, ci95_upper=None if upper is None else upper / 2,
                uniform_energy_reference=float(eigenvalues.mean() / 2), groups=int(keep.sum()),
                zero_energy_groups=int(np.count_nonzero(denominator == 0)), windows=int(windows[keep, s].sum())))
    return rows


def plot_results(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, space in zip(axes, SPACES):
        for city in SOURCES:
            selected = [r for r in rows if r['city'] == city and r['space'] == space
                        and r['stratum'] == 'all' and r['cut_kind'] == 'mode_fraction']
            x = [0.] + [r['mode_fraction'] for r in selected]
            axis.plot(x, [0.] + [r['energy_fraction'] for r in selected], 'o-', label=city.replace('_', ' '))
            axis.fill_between(x, [0.] + [r['ci95_lower'] for r in selected],
                              [0.] + [r['ci95_upper'] for r in selected], alpha=.15)
        axis.plot([0, 1], [0, 1], ':', color='gray', label='Uniform spectral energy')
        axis.axvline(.1, color='gray', alpha=.35)
        axis.axhline(.5, color='gray', alpha=.35)
        axis.set(title=space.replace('_', ' '), xlabel='Fraction of graph modes (lowest first)',
                 ylabel='Fraction of observed increment energy', xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.savefig(output / 'spectral_energy.png', dpi=180)
    figure.savefig(output / 'spectral_energy.pdf')
    plt.close(figure)


def model_coordinate_supplement(output):
    """Separate, fixed supplementary normalization; reuse primary selections."""
    primary_spec = json.loads((output / 'prespec.json').read_text())
    primary_result = json.loads((output / 'result.json').read_text())
    if primary_result['status'] != 'complete' or primary_result['prespec_sha256'] != sha256(output / 'prespec.json'):
        raise ValueError('completed, unchanged primary source freeze required')
    supplemental_spec = dict(frozen_at=datetime.now(timezone.utc).isoformat(),
        primary_prespec_sha256=sha256(output / 'prespec.json'),
        purpose='supplement only: model input count/capacity and speed/limit coordinates',
        normalization='same raw observed increment divided roadwise by capacity/speed limit; separate train-only channel RMS',
        sampling='exactly primary frozen training/validation episodes and windows',
        cuts='exactly primary fixed cuts and boundary-eigenspace expansion',
        primary_unchanged=True, source_tool_sha256=sha256(__file__),
        artifacts={city: {name: sha256(output / city / name) for name in ['graph.npz', 'raw_observed_increments.npz']}
                   for city in SOURCES})
    supplement = output / 'model_coordinates_supplement'
    supplement.mkdir()
    (supplement / 'prespec.json').write_text(json.dumps(supplemental_spec, indent=2) + '\n')
    (supplement / 'analysis_source.py').write_text(Path(__file__).read_text())
    rows, scales = [], {}
    for city, source in primary_spec['sources'].items():
        print(json.dumps(dict(city=city, stage='model_coordinate_supplement')), flush=True)
        with np.load(output / city / 'graph.npz', allow_pickle=False) as graph:
            capacities, limits = graph['capacities'], graph['speed_limits_mps']
            basis, eigenvalues = graph['basis'], graph['eigenvalues']
        normalizers = np.stack((capacities, limits), axis=-1)
        training = []
        for entry in source['train']:
            arrays = read_episode(entry)
            _, _, delta = raw_pairs(arrays, entry['selected_starts'], capacities, limits)
            training.append(delta / normalizers)
        scale, raw_rms, square_sum, elements = training_scale(training)
        del training
        with np.load(output / city / 'raw_observed_increments.npz', allow_pickle=False) as archive:
            delta = archive['delta'] / normalizers / scale
            groups = archive['demand_groups']
        numerators, denominators, windows = [], [], []
        for group_delta in delta:
            energy, total = mode_energy(group_delta, basis)
            num, denom, count = group_aggregates(energy, total, np.zeros(len(energy), dtype=int))
            numerators.append(num); denominators.append(denom); windows.append(count)
        num, den, win = np.asarray(numerators), np.asarray(denominators), np.asarray(windows)
        city_rows = [r for r in summarize(city, eigenvalues, num, den, win,
                      primary_spec['bootstrap'], primary_spec['seed'] + 2) if r['stratum'] == 'all']
        names = dict(zip(SPACES, ['model_count_capacity', 'model_speed_ratio', 'model_joint']))
        for row in city_rows:
            row['space'] = names[row['space']]
        rows.extend(city_rows)
        np.savez_compressed(supplement / f'{city}_group_energy.npz', numerators=num[:, 0],
            denominators=den[:, 0], windows=win[:, 0], eigenvalues=eigenvalues,
            demand_groups=groups, training_rms=scale, train_square_sum=square_sum, train_elements=elements)
        scales[city] = raw_rms.tolist()
    write_csv(supplement / 'energy_summary.csv', rows)
    selected = [r for r in rows if r['cut_kind'] == 'mode_fraction' and r['requested'] == '0.1']
    lines = ['# Model-coordinate normalization supplement', '',
        'This supplement applies each road’s capacity and speed limit to the same observed raw increment, '
        'then fixes channel RMS from the original training selection. It uses the same complete graph basis, '
        'validation windows, and demand-group bootstrap as the primary. It does not use an encoder or '
        'cohort-adjusted target, and does not change the primary physical-unit claim.', '',
        '| City | Feature | Energy at nominal lowest 10% modes | 95% CI | Actual modes |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in selected:
        lines.append(f"| {row['city']} | {row['space']} | {row['energy_fraction']:.2%} | "
                     f"[{row['ci95_lower']:.2%}, {row['ci95_upper']:.2%}] | {row['modes']} / {row['roads']} |")
    lines += ['', f'Training channel RMS by city: `{json.dumps(scales)}`.', '',
        'Roadwise normalization changes the spatial distribution of energy. Differences from the primary '
        'therefore quantify coordinate dependence; neither normalization estimates a learned encoder’s latent spectrum.', '']
    (supplement / 'report.md').write_text('\n'.join(lines))
    (supplement / 'result.json').write_text(json.dumps(dict(status='complete', scales=scales,
        primary_prespec_sha256=supplemental_spec['primary_prespec_sha256'], primary_results=selected), indent=2) + '\n')


def active_road_supplement(output):
    """Induced graph using only a frozen training-support mask, as a caveat check."""
    spec = json.loads((output / 'prespec.json').read_text())
    if json.loads((output / 'result.json').read_text())['status'] != 'complete':
        raise ValueError('completed primary analysis required')
    supplement = output / 'training_active_graph_supplement'
    if not supplement.exists():
        supplement.mkdir()
        supplemental_spec = dict(frozen_at=datetime.now(timezone.utc).isoformat(),
            primary_prespec_sha256=sha256(output / 'prespec.json'),
            mask='positive count in any before/after boundary of the frozen sampled training pairs only',
            operator='original model adjacency induced on training mask, then normalized symmetric Laplacian',
            increments='unchanged observed physical count/speed delta including births',
            scale='unchanged primary training channel RMS',
            validation='unchanged primary groups and windows',
            cut_rules='unchanged primary cuts and complete boundary eigenvalue blocks',
            coverage='fraction of full validation count/speed/joint energy omitted by mask',
            primary_unchanged=True)
        (supplement / 'prespec.json').write_text(json.dumps(supplemental_spec, indent=2) + '\n')
    supplemental_spec = json.loads((supplement / 'prespec.json').read_text())
    if supplemental_spec['primary_prespec_sha256'] != sha256(output / 'prespec.json'):
        raise ValueError('supplement source freeze changed')
    (supplement / 'analysis_source.py').write_text(Path(__file__).read_text())
    rows, audits = [], {}
    for city in SOURCES:
        print(json.dumps(dict(city=city, stage='training_active_graph_supplement')), flush=True)
        with np.load(output / city / 'graph.npz', allow_pickle=False) as graph:
            mask = graph['train_positive_roads']
            adjacency = graph['adjacency'][np.ix_(mask, mask)]
            edge_ids = graph['edge_ids'][mask]
        laplacian = normalized_laplacian(adjacency)
        eigenvalues, basis = eigh(laplacian, check_finite=True, driver='evd')
        components, labels = connected_components(adjacency + adjacency.T, directed=False)
        with np.load(output / city / 'raw_observed_increments.npz', allow_pickle=False) as archive:
            raw_delta, scale, groups = archive['delta'], archive['rms_scale'], archive['demand_groups']
        standardized = raw_delta / scale
        full_channel_energy = np.square(standardized).sum(axis=(0, 1, 2))
        omitted_channel_energy = np.square(standardized[:, :, ~mask]).sum(axis=(0, 1, 2))
        omitted_fractions = np.append(omitted_channel_energy / np.maximum(full_channel_energy, 1e-30),
                                     omitted_channel_energy.sum() / max(full_channel_energy.sum(), 1e-30))
        numerators, denominators, windows = [], [], []
        for group_delta in standardized[:, :, mask]:
            energy, total = mode_energy(group_delta, basis)
            num, den, count = group_aggregates(energy, total, np.zeros(len(energy), dtype=int))
            numerators.append(num); denominators.append(den); windows.append(count)
        num, den, win = np.asarray(numerators), np.asarray(denominators), np.asarray(windows)
        rows.extend(r for r in summarize(city, eigenvalues, num, den, win, spec['bootstrap'], spec['seed'] + 2)
                    if r['stratum'] == 'all')
        np.savez_compressed(supplement / f'{city}_group_energy.npz', numerators=num[:, 0], denominators=den[:, 0],
            windows=win[:, 0], eigenvalues=eigenvalues, demand_groups=groups, rms_scale=scale,
            train_positive_mask=mask, basis=basis, adjacency=adjacency, edge_ids=edge_ids)
        audits[city] = dict(induced_roads=int(mask.sum()), full_roads=len(mask), components=int(components),
            largest_component=int(np.bincount(labels).max()),
            isolated_roads=int(np.count_nonzero((adjacency + adjacency.T).sum(axis=1) == 0)),
            zero_modes=int(np.count_nonzero(np.abs(eigenvalues) < 1e-8)),
            omitted_validation_energy_fraction=dict(zip(SPACES, omitted_fractions.tolist())),
            omitted_validation_nonzero_count_roads=int(np.count_nonzero(np.any(raw_delta[:, :, ~mask, 0] != 0, axis=(0, 1)))))
    write_csv(supplement / 'energy_summary.csv', rows)
    plot_results(rows, supplement)
    selected = [r for r in rows if r['cut_kind'] == 'mode_fraction' and r['requested'] == '0.1']
    lines = ['# Training-active induced-graph supplement', '',
        'This sensitivity check removes roads with no positive count in any of the frozen sampled training '
        'boundaries. The mask is fixed entirely from training. The graph is induced from the original '
        'model adjacency on this mask and then renormalized, so it is a different graph operator and is '
        'reported separately. Physical observed increments, training channel RMS, demand groups, windows, '
        'and all spectral cut rules remain fixed. Validation coverage below quantifies omitted traffic energy.', '',
        '| City | Feature | Energy in nominal lowest 10% modes | 95% CI | Actual modes | Omitted full-graph validation energy |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in selected:
        omitted = audits[row['city']]['omitted_validation_energy_fraction'][row['space']]
        lines.append(f"| {row['city']} | {row['space']} | {row['energy_fraction']:.2%} | "
            f"[{row['ci95_lower']:.2%}, {row['ci95_upper']:.2%}] | {row['modes']} / {row['roads']} | {omitted:.2%} |")
    lines += ['', 'The full-graph nominal 10% cuts in cities with more than 10% nullspace modes '
        'cannot uniquely specify a low-frequency subspace. This induced-graph check helps separate that '
        'representation issue from the spectrum of traffic-bearing roads. Its conclusions are conditional '
        'on the training-derived road mask and observed replay distribution.', '']
    (supplement / 'report.md').write_text('\n'.join(lines))
    (supplement / 'result.json').write_text(json.dumps(dict(status='complete', graph_audits=audits,
        primary_results=selected, primary_prespec_sha256=supplemental_spec['primary_prespec_sha256']), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', type=int, default=64)
    parser.add_argument('--windows', type=int, default=12)
    parser.add_argument('--seed', type=int, default=2026101303)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--model-coordinate-supplement', action='store_true')
    parser.add_argument('--active-road-supplement', action='store_true')
    args = parser.parse_args()
    if min(args.groups, args.windows, args.bootstrap) < 1:
        parser.error('sample and bootstrap counts must be positive')
    torch.set_num_threads(2)
    if args.model_coordinate_supplement:
        model_coordinate_supplement(args.output)
        return
    if args.active_road_supplement:
        active_road_supplement(args.output)
        return
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'prespec.json').exists():
        raise ValueError('output already prespecified; use a new version directory')
    print(json.dumps(dict(pid=os.getpid(), stage='freeze_sources')), flush=True)
    frozen, loaded = {}, {}
    for city, (config_relative, manifest_relative) in SOURCES.items():
        config_path, manifest_path = ROOT / config_relative, ROOT / manifest_relative
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') != 'complete' or manifest['identity'] != _identity({k: v for k, v in manifest.items() if k != 'identity'}):
            raise ValueError(f'{city}: invalid complete manifest identity')
        entries = [entry for entry in manifest['entries'] if entry.get('canonical', False)]
        train_groups = {e['demand_group'] for e in entries if e['split'] == 'train'}
        validation_groups = {e['demand_group'] for e in entries if e['split'] == 'validation'}
        if train_groups & validation_groups or min(len(train_groups), len(validation_groups)) < args.groups:
            raise ValueError(f'{city}: disjoint sufficiently large partitions required')
        train = select_grouped(entries, 'train', args.groups, args.seed)
        validation = select_grouped(entries, 'validation', args.groups, args.seed + 1)
        source = dict(config_path=str(config_path), config_sha256=sha256(config_path),
            manifest_path=str(manifest_path), manifest_sha256=sha256(manifest_path),
            manifest_identity=manifest['identity'], contract=manifest['contract'],
            train_available_groups=len(train_groups), validation_available_groups=len(validation_groups),
            demand_group_overlap=0, train=train, validation=validation)
        for entry in train + validation:
            entry['selected_starts'] = np.unique(np.linspace(0, entry['windows'] - 1,
                min(args.windows, entry['windows'])).round().astype(int)).tolist()
        frozen[city], loaded[city] = source, manifest
    # Written before loading selected traffic observations or computing energies.
    prespec = dict(frozen_at=datetime.now(timezone.utc).isoformat(), seed=args.seed,
        train_seed=args.seed, validation_seed=args.seed + 1, groups_per_partition=args.groups,
        windows_per_group=args.windows, uniform_windows_including_last_actual_pair=True,
        bootstrap=args.bootstrap, bootstrap_unit='demand_group', mean_centering=False,
        primary='raw observed X(t+30)-X(t), physical vehicle count and mean speed m/s, including births',
        joint_scale='training-only per-channel increment RMS; no mean subtraction',
        graph='full normalized symmetric Laplacian of model directed road adjacency, float64 complete eigh',
        ranks=[16, 32, 64, 128], mode_fractions=FRACTIONS, lambda_cutoffs=CUTOFFS,
        primary_criterion='first ceil(10%*M) modes capture >50% energy with demand-group bootstrap 95% CI lower >50%, across all cities and all count/speed/joint features',
        uniform_reference='included_modes / full_graph_roads',
        boundary_ties='expand rank/fraction cuts to complete eigenvalue blocks with absolute tolerance 1e-8; primary 10% criterion is ambiguous if expansion changes its nominal rank',
        congestion='occupied-road mean clip(1-speed_ratio,0,1), training-window tertiles',
        evidence_domain='SUMO mesoscopic simulation replay on real city networks; not field sensor observations',
        cpu_threads=2, source_tool_sha256=sha256(__file__), sources=frozen)
    (args.output / 'prespec.json').write_text(json.dumps(prespec, indent=2) + '\n')
    (args.output / 'prespec.sha256').write_text(sha256(args.output / 'prespec.json') + '\n')
    (args.output / 'analysis_source.py').write_text(Path(__file__).read_text())
    all_rows, audits, feature_rows = [], {}, []
    for city, source in frozen.items():
        print(json.dumps(dict(city=city, stage='load_topology')), flush=True)
        config = load_config(source['config_path'])
        prepared = validate_prepared_inputs(config)
        if prepared.input_hashes != source['contract']['input_hashes']:
            raise ValueError(f'{city}: replay and graph input hashes mismatch')
        network = load_traffic_topology(prepared).network
        capacities, limits = np.maximum(np.asarray(network.capacities), 1.), np.asarray(network.speed_limits_mps)
        city_output = args.output / city
        city_output.mkdir()
        deltas, train_congestion = [], []
        train_positive_roads = np.zeros(len(capacities), dtype=bool)
        selected_raw_hashes, checked = [], []
        def episode(entry):
            arrays = read_episode(entry)
            if arrays['metadata']['city'] != city or arrays['metadata']['physics_identity'] != source['contract']['physics_identity']:
                raise ValueError(f'{city}: replay metadata mismatch')
            if arrays['flow'].shape[1] != len(capacities) or not np.allclose(arrays['flow'][:, :, 4:], network.static_features[None], rtol=1e-6, atol=1e-6):
                raise ValueError(f'{city}: cached road order/static topology mismatch')
            starts = select_starts(arrays['flow'], args.windows)
            if starts.tolist() != entry['selected_starts']:
                raise ValueError('observed episode differs from frozen window count')
            checked.append(dict(cache_path=entry['cache_path'], cache_content_hash=entry['cache_content_hash'],
                raw_cache_file_sha256=sha256(entry['cache_path']), demand_group=entry['demand_group'],
                split=entry['split'], frames=len(arrays['flow']), last_time=float(arrays['time'][-1]),
                last_selected_pair=starts[-1].item(), last_actual_endpoint=float(arrays['time'][starts[-1] + 1])))
            selected_raw_hashes.append(entry['cache_content_hash'])
            return arrays, starts
        for i, entry in enumerate(source['train']):
            arrays, starts = episode(entry)
            before, after, delta = raw_pairs(arrays, starts, capacities, limits)
            deltas.append(delta)
            train_positive_roads |= np.any((before[..., 0] > 0) | (after[..., 0] > 0), axis=0)
            train_congestion.extend(congestion(arrays['flow'][starts]).tolist())
            if i % 16 == 0:
                print(json.dumps(dict(city=city, stage='training_rms', groups=i)), flush=True)
        scale, raw_rms, square_sum, elements = training_scale(deltas)
        del deltas
        thresholds = np.quantile(train_congestion, [1 / 3, 2 / 3])
        adjacency = np.asarray(network.adjacency, dtype=np.float64)
        laplacian = normalized_laplacian(adjacency)
        components, labels = connected_components(adjacency + adjacency.T, directed=False)
        degree = (adjacency + adjacency.T).sum(axis=1)
        pre_energy_audit = dict(roads=len(adjacency), components=int(components),
            component_sizes=np.bincount(labels).tolist(), isolated_roads=int(np.count_nonzero(degree == 0)),
            passenger_allowed_roads=int(np.count_nonzero(network.passenger_allowed)),
            train_positive_count_roads=int(np.count_nonzero(train_positive_roads)),
            training_positive_isolated_roads=int(np.count_nonzero(train_positive_roads & (degree == 0))),
            training_rms=raw_rms.tolist(), congestion_thresholds=thresholds.tolist())
        (city_output / 'pre_validation_graph_audit.json').write_text(json.dumps(pre_energy_audit, indent=2) + '\n')
        print(json.dumps(dict(city=city, stage='complete_float64_eigh', roads=len(adjacency))), flush=True)
        eigenvalues, basis = eigh(laplacian, check_finite=True, driver='evd')
        residual = float(np.max(np.linalg.norm(laplacian @ basis - basis * eigenvalues, axis=0)))
        orthogonality = float(np.max(np.abs(basis.T @ basis - np.eye(len(adjacency)))))
        np.savez_compressed(city_output / 'graph.npz', adjacency=adjacency, eigenvalues=eigenvalues,
                            basis=basis, edge_ids=np.asarray(network.edge_ids), capacities=capacities,
                            speed_limits_mps=limits, component_labels=labels,
                            train_positive_roads=train_positive_roads, passenger_allowed=network.passenger_allowed)
        numerators, denominators, group_windows, raw_deltas = [], [], [], []
        support, departures, statistics = [], [], []
        for i, entry in enumerate(source['validation']):
            arrays, starts = episode(entry)
            before, after, delta = raw_pairs(arrays, starts, capacities, limits)
            raw_deltas.append(delta)
            energy, total = mode_energy(delta / scale, basis)
            strata = np.searchsorted(thresholds, congestion(arrays['flow'][starts]), side='right')
            num, denom, counts = group_aggregates(energy, total, strata)
            numerators.append(num); denominators.append(denom); group_windows.append(counts)
            support.append(dict(occupied_road_fraction=float(np.mean((before[..., 0] > 0) | (after[..., 0] > 0))),
                count_increment_support=float(np.mean(delta[..., 0] != 0)), speed_increment_support=float(np.mean(delta[..., 1] != 0))))
            departures.append(int(arrays['departure_counts'][starts].sum()))
            statistics.append(dict(demand_group=entry['demand_group'], windows=len(starts),
                count_before_mean=float(before[..., 0].mean()), count_before_max=float(before[..., 0].max()),
                speed_before_mean_mps=float(before[..., 1].mean()),
                count_delta_rms=float(np.sqrt(np.square(delta[..., 0]).mean())),
                speed_delta_rms_mps=float(np.sqrt(np.square(delta[..., 1]).mean())),
                count_delta_mean=float(delta[..., 0].mean()), speed_delta_mean_mps=float(delta[..., 1].mean()),
                births_in_selected_intervals=departures[-1],
                count_energy_outside_train_positive_mask=float(np.square(delta[:, ~train_positive_roads, 0]).sum()),
                speed_energy_outside_train_positive_mask=float(np.square(delta[:, ~train_positive_roads, 1]).sum()),
                isolated_count_increment_energy=float(np.square(delta[:, degree == 0, 0]).sum()),
                isolated_speed_increment_energy=float(np.square(delta[:, degree == 0, 1]).sum()), **support[-1]))
            if i % 16 == 0:
                print(json.dumps(dict(city=city, stage='validation_raw_spectrum', groups=i)), flush=True)
        numerators, denominators, group_windows = np.asarray(numerators), np.asarray(denominators), np.asarray(group_windows)
        np.savez_compressed(city_output / 'group_energy.npz', numerators=numerators, denominators=denominators,
            windows=group_windows, strata=np.asarray(STRATA), spaces=np.asarray(SPACES), eigenvalues=eigenvalues,
            demand_groups=np.asarray([e['demand_group'] for e in source['validation']]), rms_scale=scale)
        np.savez_compressed(city_output / 'raw_observed_increments.npz', delta=np.asarray(raw_deltas),
            rms_scale=scale, train_square_sum=square_sum, train_elements=elements,
            starts=np.asarray([e['selected_starts'] for e in source['validation']]),
            demand_groups=np.asarray([e['demand_group'] for e in source['validation']]))
        for row in statistics:
            feature_rows.append(dict(city=city, **row))
        all_rows.extend(summarize(city, eigenvalues, numerators, denominators, group_windows, args.bootstrap, args.seed + 2))
        audits[city] = dict(**pre_energy_audit,
            zero_modes=int(np.count_nonzero(np.abs(eigenvalues) < 1e-8)), min_lambda=float(eigenvalues[0]),
            max_lambda=float(eigenvalues[-1]), eigen_residual=residual, basis_orthogonality=orthogonality,
            raw_training_rms=raw_rms.tolist(), applied_rms_scale=scale.tolist(),
            train_rms_elements_per_channel=elements,
            unique_selected_cache_content_hashes=len(set(selected_raw_hashes)),
            selected_train_groups=len(source['train']), selected_validation_groups=len(source['validation']),
            selected_validation_windows=int(group_windows[:, 0].sum()),
            births_in_validation_selected_intervals=sum(departures), verified_cache_files=checked,
            static_road_order_matches=True, input_hashes_match=True, actual_endpoint_pairs_only=True)
        (city_output / 'audit.json').write_text(json.dumps(audits[city], indent=2) + '\n')
        print(json.dumps(dict(city=city, stage='complete')), flush=True)
    write_csv(args.output / 'energy_summary.csv', all_rows)
    write_csv(args.output / 'raw_feature_statistics.csv', feature_rows)
    plot_results(all_rows, args.output)
    primary = [r for r in all_rows if r['stratum'] == 'all' and r['cut_kind'] == 'mode_fraction' and r['requested'] == '0.1']
    ambiguous = any(r['expanded_boundary_tie'] for r in primary)
    strong = not ambiguous and all(r['ci95_lower'] is not None and r['ci95_lower'] > .5 for r in primary)
    lines = ['# Observed traffic increment graph spectra', '',
        f'Prespecified strong low-frequency-dominance criterion met across all cities/features: **{strong}**.', '',
        f'Primary nominal 10% cuts ambiguous because they split a tied eigenspace: **{ambiguous}**. '
        'Energy below is computed with complete boundary eigenspaces; CSV records nominal and actual ranks.', '',
        'Evidence is recorded SUMO mesoscopic simulation on three real city networks, not field observations. '
        'Validation is a development split of the existing replay corpus, not an untouched external test set.', '',
        '| City | Feature | Energy at nominal 10% cut (complete ties) | Demand-group bootstrap 95% CI | Actual included modes | Uniform reference |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in primary:
        lines.append(f"| {r['city']} | {r['space']} | {r['energy_fraction']:.2%} | [{r['ci95_lower']:.2%}, {r['ci95_upper']:.2%}] | {r['modes']} / {r['roads']} | {r['mode_fraction']:.2%} |")
    lines += ['', 'The primary increments are actual `X(t+30)-X(t)` road counts and mean speeds, including births. '
        'Physical count is reconstructed from the cached float32 count/capacity feature; speed is reconstructed in m/s. '
        'No encoder, cohort-adjusted/hybrid target, terminal padding, or spatial/temporal mean subtraction is used. '
        f'Each city uses one frozen episode from each of {args.groups} train demand groups and {args.groups} disjoint validation groups, '
        f'with {args.windows} uniformly spaced actual adjacent pairs per episode, including the final observed pair. '
        'RMS channel scaling and congestion tertiles are fixed from training only. '
        'Separate count and speed energy fractions are invariant to their scalar RMS scaling; joint energy gives '
        'channels equal training RMS scale.', '',
        'The graph operator is the model symmetric normalized Laplacian: `A_s=(A+A.T)/2`, '
        '`L=diag(d>0)-D^(-1/2) A_s D^(-1/2)`. Its quadratic form penalizes differences in '
        '`x_i/sqrt(d_i)` across adjacent roads, not differences in unweighted counts; isolated roads have zero operator. '
        'All graph roads and zero modes remain in the basis. Thus disconnected components and isolated-road signals '
        'can influence low-mode capture, and inactive roads contribute zero increment energy rather than a large static baseline. '
        'The uniform reference is the expected energy of an isotropic vector in an equally sized subspace.', '',
        'Reported energy is a pooled ratio of projected energy to total energy, with groups resampled as whole units '
        f'({args.bootstrap} bootstrap repetitions). It is not an average of window ratios. Zero-energy groups remain '
        'in the aggregates and their individual ratios are undefined. Empty strata are retained in NPZ arrays and excluded '
        'only from that stratum’s resampling. The CI is descriptive for this replay distribution, not simultaneous across '
        'all reported cuts. Rank/fraction cuts expand to include whole tied boundary eigenvalue spaces '
        '(absolute tolerance 1e-8); eigenvalue cutoffs include complete spaces. An expanded nominal 10% cut '
        'cannot establish the primary 10% criterion.', '',
        'See `energy_summary.csv` for all prespecified ranks, mode fractions, frequency cutoffs, congestion strata, '
        'and relative spectral centroid (energy-weighted lambda divided by 2). '
        '`prespec.json` freezes identities, source configuration/content hashes, partitions, and sampled window IDs '
        'before loading sampled observations. Each city’s `audit.json` records verified file hashes and graph checks. '
        '`graph.npz`, `raw_observed_increments.npz`, and `group_energy.npz` allow independent reprojection and '
        'bootstrap verification. `raw_feature_statistics.csv` reports traffic activity and increment support.', '',
        '| City | Roads | Components | Isolated roads | Zero modes | Training count/speed increment RMS |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for city, audit in audits.items():
        lines.append(f"| {city} | {audit['roads']} | {audit['components']} | {audit['isolated_roads']} | {audit['zero_modes']} | {audit['raw_training_rms']} |")
    lines += ['', f'Elapsed CPU-only analysis wall time: {time.time() - started:.1f} seconds.', '',
        'Reproduce in a new version directory:', '',
        '```bash', 'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 '
        '/home/wangwenrui/opt/miniconda3/envs/URB/bin/python tools/analyze_multicity_increment_spectrum.py '
        '--output outputs/traffic_increment_spectrum_20261013/v2', '```', '']
    (args.output / 'report.md').write_text('\n'.join(lines))
    result = dict(status='complete', strong_low_frequency_dominance=strong, primary_tie_ambiguity=ambiguous,
        elapsed_seconds=time.time() - started, primary_results=primary,
        prespec_sha256=sha256(args.output / 'prespec.json'), cross_city_selected_cache_hashes_disjoint=
        len({e['cache_content_hash'] for source in frozen.values() for e in source['train'] + source['validation']}) == args.groups * 2 * len(SOURCES))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
