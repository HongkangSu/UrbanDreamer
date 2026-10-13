"""CPU diagnostic of projection capacity and prediction by a frozen prior body.

This is a one-step prior-head experiment, not FM inference or retraining of the
world model. Exact nested graph modes are used at every rank. Only an output
linear head is independently refit at each rank, with a prespecified ridge.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = Path(os.environ.get('PRIOR_RANK_FROZEN_ROOT', ROOT)).resolve()
sys.path.insert(0, str(FROZEN_ROOT))

import numpy as np
import torch
from scipy.linalg import eigh
from torch.nn import functional as F

from tools.analyze_traffic_spectrum import normalized_laplacian, select_grouped, select_starts, bootstrap_ratio
from tools.component_experiment_data import atomic_json, dataset, load_frozen, sha256
from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.learning.offline_data import _identity, archive_content_hash

CHECKPOINT_SHA = '2f63f0b2a5fe18f3a3bd9be27ffd1e63a2755464e7f2944e1227334f0ffe6af6'
RANKS = (8, 16, 32, 64, 128, 256, 512, 852)
RIDGE_RELATIVE = 1e-6


def frozen_hidden(prior, z, condition):
    """Reproduce the native prior's complete frozen hidden front."""
    graph = prior.adjacency.to(z)
    downstream = graph / graph.sum(-1, keepdim=True).clamp_min(1e-12)
    upstream = graph.T / graph.T.sum(-1, keepdim=True).clamp_min(1e-12)
    return F.silu(prior.state_proj(z)
        + downstream @ prior.downstream_proj(z)
        + upstream @ prior.upstream_proj(z)
        + prior.condition_proj(condition.edge_tokens))


def subspace_geometry(exact_basis, checkpoint_basis):
    """Normalize only for geometric comparison, preserving actual predictions."""
    exact_basis = np.asarray(exact_basis, dtype=np.float64)
    checkpoint_basis = np.asarray(checkpoint_basis, dtype=np.float64)
    if exact_basis.shape != checkpoint_basis.shape or exact_basis.ndim != 2:
        raise ValueError('matching matrix bases required')
    orthogonal, _ = np.linalg.qr(checkpoint_basis, mode='reduced')
    rank = exact_basis.shape[1]
    shared = min(float(rank), max(0., float(np.square(exact_basis.T @ orthogonal).sum())))
    return dict(native_subspace_overlap=shared/rank,
        native_subspace_chordal_distance_squared=float(rank-shared),
        checkpoint_basis_original_orthogonality_max_error=float(np.max(np.abs(checkpoint_basis.T@checkpoint_basis-np.eye(rank)))),
        geometry='QR-normalized checkpoint basis; native predictions retain original stored FP32 basis')


def spectral_design(hidden, delta, basis):
    """U.T [H, 1] and U.T delta; the bias is projected with the features."""
    hidden, delta, basis = [np.asarray(value, dtype=np.float64) for value in (hidden, delta, basis)]
    if (hidden.ndim != 3 or delta.shape != hidden.shape or basis.ndim != 2
            or basis.shape[0] != hidden.shape[1] or not all(np.isfinite(v).all() for v in (hidden, delta, basis))):
        raise ValueError('finite matched [windows, roads, channels] and matching basis required')
    augmented = np.concatenate((hidden, np.ones((*hidden.shape[:-1], 1))), axis=-1)
    return basis.T @ augmented, basis.T @ delta


def solve_ridge(gram, cross, target_energy, *, relative=RIDGE_RELATIVE):
    """Fully solve equal-capacity output heads; penalize bias as prespecified."""
    gram, cross = np.asarray(gram, dtype=np.float64), np.asarray(cross, dtype=np.float64)
    if (gram.ndim != 2 or gram.shape[0] != gram.shape[1] or cross.shape[0] != len(gram)
            or not np.isfinite(gram).all() or not np.isfinite(cross).all() or relative <= 0):
        raise ValueError('finite square Gram, matching cross moment and positive ridge required')
    ridge = relative * max(float(np.trace(gram) / len(gram)), 1e-12)
    regularized = gram + np.eye(len(gram)) * ridge
    coefficients = np.linalg.solve(regularized, cross)
    normal_residual = regularized @ coefficients - cross
    residual = float(target_energy - 2 * np.sum(coefficients * cross)
        + np.sum(coefficients * (gram @ coefficients)))
    proof = dict(ridge=ridge, relative_ridge=relative, bias_penalized=True,
        gram_trace=float(np.trace(gram)), gram_condition=(float(np.linalg.cond(gram)) if np.linalg.matrix_rank(gram)==len(gram) else None),
        regularized_condition=float(np.linalg.cond(regularized)), training_projected_sse=max(0., residual),
        normal_equation_relative_residual=float(np.linalg.norm(normal_residual) / max(np.linalg.norm(cross), 1e-30)),
        coefficient_shape=list(coefficients.shape), output_parameters=int(coefficients.size))
    return coefficients, proof


def metric_sums(delta, prediction, target_coefficients):
    delta, prediction, target_coefficients = [np.asarray(v, dtype=np.float64) for v in (delta, prediction, target_coefficients)]
    total = float(np.square(delta).sum())
    projected = float(np.square(target_coefficients).sum())
    dot = float(np.sum(delta * prediction))
    predicted = float(np.square(prediction).sum())
    full = float(np.square(delta - prediction).sum())
    return dict(total_delta_energy=total, captured_energy=projected, full_sse=full,
        projected_sse=max(0., full - (total - projected)), prediction_energy=predicted, dot_product=dot)


def aggregate_groups(records):
    """Keep demand groups as the resampling unit, including multiple episodes."""
    totals = {}
    identity = ('split', 'arm', 'rank', 'demand_group')
    for row in records:
        key = tuple(row[k] for k in identity)
        if key not in totals:
            totals[key] = {k: row[k] for k in identity}
            totals[key].update({k: 0. for k in row if k not in (*identity, 'episode')})
        for field in totals[key]:
            if field not in identity:
                totals[key][field] += row[field]
    return list(totals.values())


def summarize(records, repetitions=2000, seed=20261013):
    groups = aggregate_groups(records)
    rows = []
    for split, arm, rank in sorted({(r['split'], r['arm'], r['rank']) for r in groups}):
        selected = [r for r in groups if (r['split'], r['arm'], r['rank']) == (split, arm, rank)]
        row = dict(split=split, arm=arm, rank=rank, groups=len(selected), windows=int(sum(r['windows'] for r in selected)))
        def ratio(field, numerator, denominator):
            value, lower, upper = bootstrap_ratio(numerator, denominator, repetitions=repetitions, seed=seed)
            row.update({field: value, field+'_ci95_low': lower, field+'_ci95_high': upper})
        energy = [r['total_delta_energy'] for r in selected]
        capture = [r['captured_energy'] for r in selected]
        ratio('captured_energy_fraction', capture, energy)
        ratio('irreducible_projection_fraction', [a-b for a, b in zip(energy, capture)], energy)
        ratio('full_normalized_sse', [r['full_sse'] for r in selected], energy)
        ratio('within_projected_normalized_sse', [r['projected_sse'] for r in selected], capture)
        ratio('improvement_vs_zero_prior', [r['total_delta_energy']-r['full_sse'] for r in selected], energy)
        ratio('prediction_norm_ratio_squared', [r['prediction_energy'] for r in selected], energy)
        squared = row['prediction_norm_ratio_squared']
        row['prediction_norm_ratio'] = None if squared is None else float(np.sqrt(max(squared, 0.)))
        product = sum(r['prediction_energy'] for r in selected) * sum(energy)
        row['alignment'] = None if product <= 0 else float(sum(r['dot_product'] for r in selected) / np.sqrt(product))
        for metric in ('count', 'speed'):
            ratio(metric+'_mae', [r[metric+'_absolute_error'] for r in selected], [r[metric+'_elements'] for r in selected])
            ratio(metric+'_mae_occupied', [r[metric+'_occupied_absolute_error'] for r in selected],
                [r[metric+'_occupied_elements'] for r in selected])
        rows.append(row)
    return groups, rows


def physical_sums(model, endpoint, targets, following, following_valid, capacities, limits):
    decoded = model.decode(endpoint.float()).cpu().numpy().astype(np.float64)
    truth = targets.cpu().numpy().astype(np.float64)
    count_target = truth[..., 2] * capacities
    count_error = np.abs(decoded[..., 2] * capacities - count_target)
    # Decoder instantaneous speed is supervised by the adjacent interval mean.
    speed_target = .5 * (truth[..., 1] + following[..., 1]) * limits
    speed_error = np.abs(decoded[..., 1] * limits - speed_target)
    occupied = count_target > 0
    valid = np.broadcast_to(following_valid[:, None], occupied.shape)
    return dict(count_absolute_error=float(count_error.sum()), count_elements=int(count_error.size),
        count_occupied_absolute_error=float(count_error[occupied].sum()), count_occupied_elements=int(occupied.sum()),
        speed_absolute_error=float(speed_error[valid].sum()), speed_elements=int(valid.sum()),
        speed_occupied_absolute_error=float(speed_error[valid & occupied].sum()),
        speed_occupied_elements=int((valid & occupied).sum()))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    validation = [r for r in rows if r['split'] == 'validation']
    for arm, label in [('oracle', 'Oracle projected target'), ('original_transfer', 'Original trained output head'),
                       ('refit_output', 'Independently solved output head')]:
        selected = [r for r in validation if r['arm'] == arm]
        for axis, metric in zip(axes.flat, ['full_normalized_sse', 'within_projected_normalized_sse', 'count_mae', 'speed_mae_occupied']):
            x = [r['rank'] for r in selected]
            axis.plot(x, [r[metric] for r in selected], 'o-', label=label)
            axis.fill_between(x, [r[metric+'_ci95_low'] for r in selected], [r[metric+'_ci95_high'] for r in selected], alpha=.12)
    persistence = next(r for r in validation if r['arm'] == 'zero_prior')
    titles = ['Full latent SSE / zero-prior SSE', 'Projected SSE / projected zero-prior SSE',
              'Coarse count MAE (vehicles)', 'Coarse occupied-road speed MAE (m/s)']
    for axis, metric, title in zip(axes.flat, ['full_normalized_sse', 'within_projected_normalized_sse', 'count_mae', 'speed_mae_occupied'], titles):
        axis.axhline(persistence[metric], linestyle=':', color='gray', label='z0 persistence')
        axis.axvline(16, color='gray', alpha=.35)
        axis.set(xscale='log', xlabel='Exact nested projection rank', title=title)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for extension in ('png', 'pdf'):
        figure.savefig(output / ('prior_rank_prediction.'+extension), dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    selected = [r for r in validation if r['arm'] == 'oracle']
    axis.plot([r['rank'] for r in selected], [r['captured_energy_fraction'] for r in selected], 'o-')
    axis.fill_between([r['rank'] for r in selected], [r['captured_energy_fraction_ci95_low'] for r in selected],
        [r['captured_energy_fraction_ci95_high'] for r in selected], alpha=.2)
    axis.axvline(16, color='gray', alpha=.35)
    axis.set(xscale='log', xlabel='Exact nested projection rank', ylabel='Captured target-increment energy fraction',
        title='Frozen-encoder target capacity; demand-group bootstrap 95% intervals')
    axis.grid(alpha=.2)
    for extension in ('png', 'pdf'):
        figure.savefig(output / ('prior_rank_capacity.'+extension), dpi=180)
    plt.close(figure)


def verify_runtime(source, manifest):
    entries = json.loads(manifest.read_text())
    entries = entries.get('files', entries)
    for relative, expected in entries.items():
        if sha256(source / relative) != expected:
            raise ValueError('immutable frozen runtime changed: '+relative)
    return dict(files=len(entries), manifest_sha256=sha256(manifest))


def run(args):
    started = time.time()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(args.seed)
    if args.output.exists() and (args.output/'summary.json').exists():
        raise ValueError('preserve prior completed results')
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.experiment/'snapshot/checkpoints/latest.pt'
    if sha256(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('completed fixed checkpoint required')
    runtime_proof = verify_runtime(args.frozen_source, args.source_manifest)
    manifest_path = args.experiment/'development_manifest.json'
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'complete' or manifest['identity'] != _identity({k: v for k, v in manifest.items() if k != 'identity'}):
        raise ValueError('complete development manifest required')
    selection = {split: select_grouped(manifest['entries'], split, count, args.seed+(split=='validation'))
                 for split, count in [('train', args.train_groups), ('validation', args.validation_groups)]}
    if set(e['demand_group'] for e in selection['train']) & set(e['demand_group'] for e in selection['validation']):
        raise ValueError('demand groups overlap')
    atomic_json(args.output/'selection.json', selection)
    protocol = dict(status='prespecified_before_target_results', checkpoint_sha256=CHECKPOINT_SHA,
        frozen_source=str(args.frozen_source.resolve()), runtime_proof=runtime_proof,
        development_manifest_sha256=manifest_hash, development_manifest_identity=manifest['identity'],
        code_sha256=sha256(__file__), seed=args.seed, ranks=list(RANKS), windows_per_group=args.windows,
        train_groups=len(selection['train']), validation_groups=len(selection['validation']),
        ridge_relative=RIDGE_RELATIVE, ridge='lambda=1e-6 trace(X.T X)/65; all 65 coefficients including bias penalized; no validation tuning',
        output_shape=[65, 64], model_body_frozen=True, encoder_frozen=True, full_world_model_training=False,
        arms=['oracle', 'original_transfer', 'refit_output', 'zero_prior'],
        target='single_step prepare_transition_targets; full observed inputs/events, hybrid C_t endpoint',
        projection='exact full normalized symmetric graph Laplacian eigendecomposition, nested low modes',
        native_rank_limit='min(requested_rank,E-1); r853 is analytical unprojected reference only',
        physical_speed_target='0.5*(raw next interval speed + raw next-next interval speed); omit unavailable final interval',
        bootstrap='2000 demand-group resamples; ratios of summed sufficient statistics; same groups/windows every rank',
        device='cpu', threads=2, final_test_used=False, prior_noise_changed=False,
        caveats=['Development validation was previously inspected.', 'Encoder and prior hidden/condition body were trained at rank16.',
                 'Refitting only the output head does not establish performance of a fully retrained higher-rank body.',
                 'Full factual recorded events are an offline dynamics diagnostic, not causal actor-available controls.',
                 'Coarse decoded point metrics contain no FM correction and are not probabilistic forecasts.'])
    atomic_json(args.output/'protocol.json', protocol)
    print(json.dumps(dict(stage='initializing', pid=os.getpid(), elapsed_seconds=time.time()-started)), flush=True)
    config, prepared, topology, model, _ = load_frozen(args.experiment, device='cpu')
    model.eval().requires_grad_(False)
    if config.method.fm_training_mode != 'single_step':
        raise ValueError('fixed checkpoint must use single-step pretraining')
    roads = len(topology.network.adjacency)
    if roads != 853:
        raise ValueError('this prespecified study requires 853 roads')
    capacities = np.maximum(np.asarray(topology.network.capacities, dtype=np.float64), 1.)
    limits = np.asarray(topology.network.speed_limits_mps, dtype=np.float64)
    laplacian = normalized_laplacian(topology.network.adjacency)
    eigenvalues, basis = eigh(laplacian, check_finite=True)
    np.savez_compressed(args.output/'exact_basis.npz', basis=basis, eigenvalues=eigenvalues)
    prior = model.dynamics.graph_prior
    native_basis = prior.basis.cpu().numpy().astype(np.float64)
    proof = dict(native_rank=prior.rank, **subspace_geometry(basis[:,:16], native_basis),
        exact_max_eigen_residual=float(np.linalg.norm(laplacian@basis-basis*eigenvalues, axis=0).max()),
        exact_orthogonality_max_error=float(np.max(np.abs(basis.T@basis-np.eye(roads)))))
    gram = {r: np.zeros((65, 65), dtype=np.float64) for r in RANKS}
    cross = {r: np.zeros((65, 64), dtype=np.float64) for r in RANKS}
    energies = {r: 0. for r in RANKS}
    original = np.concatenate((prior.output.weight.cpu().numpy().T, prior.output.bias.cpu().numpy()[None]), axis=0).astype(np.float64)
    coefficient_heads = {}
    data_proofs = []
    records = []
    native_records = []
    validation_sufficient = {r: dict(gram=np.zeros((65,65)), cross=np.zeros((65,64)), energy=0.) for r in RANKS}
    def materialize(source, entry, index):
        arrays = source._episode(index)
        starts = select_starts(arrays['flow'], args.windows)
        batch = source.batch_from_starts(index, starts.tolist(), materialization_horizon=1, fm_only=True)
        scoped = prepare_transition_targets(batch, capacities=capacities)
        z = model.encode(batch.flow[:, 0])
        target = model.encode(scoped.transition_target_flow[:, 0])
        condition = model.dynamics.encode_condition(batch.action, action_events=batch.action_events)
        hidden = frozen_hidden(prior, z[:,None], condition)[:,0]
        delta = (target-z).cpu().numpy().astype(np.float64)
        x, y = spectral_design(hidden.cpu().numpy(), delta, basis)
        valid = starts+2 < len(arrays['flow'])
        following = np.zeros_like(arrays['flow'][starts+1])
        following[valid] = arrays['flow'][starts[valid]+2]
        if not proof.get('native_forward_algebra_verified'):
            native = prior(z[:,None], condition)[:,0]
            manual = prior.project(prior.output(hidden))
            torch.testing.assert_close(native, manual, atol=2e-6, rtol=2e-6)
            direct = native_basis @ (native_basis.T @ (np.concatenate((hidden.cpu().numpy(),np.ones((*hidden.shape[:-1],1))),-1) @ original))
            np.testing.assert_allclose(native.cpu().numpy(), direct, atol=2e-5, rtol=2e-5)
            exact_native = basis[:,:16]@(x[:,:16]@original)
            proof.update(native_forward_algebra_verified=True, native_to_exact_r16_relative_output_difference=float(
                np.linalg.norm(exact_native-native.cpu().numpy())/max(np.linalg.norm(native.cpu().numpy()),1e-30)))
        data_proofs.append(dict(split=entry['split'], demand_group=entry['demand_group'], episode=entry['episode'],
            cache_path=entry['cache_path'], expected_cache_content_hash=entry['cache_content_hash'],
            starts=starts.tolist(), valid_following_speed=valid.tolist()))
        native_delta = prior(z[:,None], condition)[:,0].cpu().numpy().astype(np.float64)
        return z, scoped.transition_target_flow[:,0], delta, x, y, following, valid, native_delta
    with torch.inference_mode():
        for split in ('train', 'validation'):
            source = dataset(args.experiment, config, prepared, topology, split=split, horizon=1)
            indices = {e['cache_path']: i for i,e in enumerate(source.manifest['entries'])}
            for ordinal, entry in enumerate(selection[split]):
                z, targets, delta, x, y, following, valid, native_delta = materialize(source, entry, indices[entry['cache_path']])
                for rank in RANKS:
                    xr, yr = x[:,:rank].reshape(-1,65), y[:,:rank].reshape(-1,64)
                    if split == 'train':
                        gram[rank] += xr.T@xr
                        cross[rank] += xr.T@yr
                        energies[rank] += float(np.square(yr).sum())
                    else:
                        validation_sufficient[rank]['gram'] += xr.T@xr
                        validation_sufficient[rank]['cross'] += xr.T@yr
                        validation_sufficient[rank]['energy'] += float(np.square(yr).sum())
                        for arm, head in [('oracle', None), ('original_transfer', original), ('refit_output', coefficient_heads[rank])]:
                            prediction = basis[:,:rank] @ (y[:,:rank] if arm == 'oracle' else x[:,:rank] @ head)
                            row = dict(split=split, arm=arm, rank=rank, demand_group=entry['demand_group'], episode=entry['episode'], windows=len(z))
                            row.update(metric_sums(delta, prediction, y[:,:rank]))
                            row.update(physical_sums(model, z+torch.as_tensor(prediction,dtype=z.dtype), targets, following, valid, capacities, limits))
                            records.append(row)
                if split == 'validation':
                    row = dict(split=split, arm='zero_prior', rank=0, demand_group=entry['demand_group'], episode=entry['episode'], windows=len(z))
                    row.update(metric_sums(delta, np.zeros_like(delta), y))
                    row.update(physical_sums(model,z,targets,following,valid,capacities,limits))
                    records.append(row)
                    # Preserve actual checkpoint-basis rank16 separately from exact nested rank16.
                    native_row = dict(split=split,arm='native_checkpoint_r16',rank=16,demand_group=entry['demand_group'],episode=entry['episode'],windows=len(z))
                    native_row.update(metric_sums(delta,native_delta,native_basis.T@delta))
                    native_row.update(physical_sums(model,z+torch.as_tensor(native_delta,dtype=z.dtype),targets,following,valid,capacities,limits))
                    native_records.append(native_row)
                print(json.dumps(dict(stage=split, group=ordinal+1, total=len(selection[split]), pid=os.getpid(), elapsed_seconds=time.time()-started)), flush=True)
            if split == 'train':
                fits = {}
                for rank in RANKS:
                    coefficient_heads[rank], fits[str(rank)] = solve_ridge(gram[rank],cross[rank],energies[rank])
                np.savez_compressed(args.output/'output_head_coefficients.npz', original=original,
                    **{f'rank_{r}': h for r,h in coefficient_heads.items()})
                np.savez_compressed(args.output/'training_sufficient_statistics.npz',
                    **{f'{kind}_{r}': data[r] for kind,data in [('gram',gram),('cross',cross),('target_energy',energies)] for r in RANKS})
                atomic_json(args.output/'fit_diagnostics.json', fits)
    records.extend(native_records)
    groups, rows = summarize(records, seed=args.seed)
    write_csv(args.output/'episode_metrics.csv', records)
    write_csv(args.output/'demand_group_metrics.csv', groups)
    write_csv(args.output/'rank_summary.csv', rows)
    atomic_json(args.output/'data_audit.json', data_proofs)
    np.savez_compressed(args.output/'validation_sufficient_statistics.npz',
        **{f'{kind}_{r}': value for r,fields in validation_sufficient.items() for kind,value in fields.items()})
    plot(rows,args.output)
    # Revalidate immutable artifacts after computation; no producer inputs change.
    if sha256(checkpoint) != CHECKPOINT_SHA or sha256(manifest_path) != manifest_hash:
        raise ValueError('immutable model/manifest changed')
    verify_runtime(args.frozen_source,args.source_manifest)
    for entry in data_proofs:
        if archive_content_hash(Path(entry['cache_path'])) != entry['expected_cache_content_hash']:
            raise ValueError('selected compact data changed')
    imported = {name: str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
        if name.startswith(('urban_dreamer.','tools.')) and getattr(module,'__file__',None)}
    if any(not Path(path).is_relative_to(FROZEN_ROOT) for path in imported.values()):
        raise ValueError('live source contaminated frozen execution')
    atomic_json(args.output/'import_audit.json',imported)
    atomic_json(args.output/'summary.json',dict(status='complete',protocol=protocol, algebra_and_basis=proof,
        fit_diagnostics=fits,summary=rows,elapsed_seconds=time.time()-started,
        checkpoint_unchanged=True,source_unchanged=True,data_unchanged=True,
        analytical_unprojected_reference=dict(rank=roads,captured_energy_fraction=1.,irreducible_projection_fraction=0.,
            native_implementation_supports_this_rank=False), imported_modules=len(imported)))
    print(json.dumps(dict(stage='complete',output=str(args.output), elapsed_seconds=time.time()-started)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=Path('outputs/component_experiments_20261011'))
    parser.add_argument('--frozen-source',type=Path,default=Path('outputs/pretrained_evaluation_20261012/distribution_v1/source'))
    parser.add_argument('--source-manifest',type=Path,default=Path('outputs/pretrained_evaluation_20261012/distribution_v1/source_manifest.json'))
    parser.add_argument('--output',type=Path,default=Path('outputs/pretrained_prior_rank_20261013/v1'))
    parser.add_argument('--train-groups',type=int,default=64)
    parser.add_argument('--validation-groups',type=int,default=32)
    parser.add_argument('--windows',type=int,default=12)
    parser.add_argument('--seed',type=int,default=20261013)
    run(parser.parse_args())
