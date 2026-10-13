"""CPU frozen-body probe: graph-frequency energy is distinct from predictability.

Fits a physical increment head on train only; reuses the saved identity latent
head from the completed checkpoint study. No action features, FM, or SUMO runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from tools.analyze_traffic_spectrum import bootstrap_ratio, normalized_laplacian, physical_features, select_starts
from tools.component_experiment_data import atomic_json, dataset, load_frozen, sha256
from tools.evaluate_pretrained_prior_ranks import CHECKPOINT_SHA, solve_ridge, write_csv
from tools.evaluate_prior_input_filter import state_hidden
from urban_dreamer.learning.offline_data import archive_content_hash

BAND_EDGES = np.array([0., .1, .25, .5, 1., 1.5, 2.0000001])


def frequency_bands(eigenvalues, edges=BAND_EDGES, tolerance=1e-8):
    """Assign near-equal eigenvalues together, including cutoff eigenspaces."""
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('finite nonempty spectrum required')
    if np.any(np.diff(values) < 0) or values.min() < -tolerance or values.max() > edges[-1]:
        raise ValueError('sorted normalized Laplacian spectrum required')
    bins = np.empty(len(values), dtype=int)
    start = 0
    for end in range(1, len(values)+1):
        if end == len(values) or values[end]-values[end-1] > tolerance:
            representative = max(0., float(values[start:end].mean()))
            index = int(np.searchsorted(edges, representative, side='right')-1)
            bins[start:end] = min(index, len(edges)-2)
            start = end
    return bins


def augmented(hidden):
    hidden = np.asarray(hidden, dtype=np.float64)
    return np.concatenate((hidden, np.ones((*hidden.shape[:-1], 1))), axis=-1)


def band_statistics(target, prediction, mean, basis, bins):
    arrays = [np.asarray(a, dtype=np.float64) for a in (target, prediction, mean)]
    target, prediction, mean = arrays
    if target.shape != prediction.shape or mean.shape not in (target.shape, target.shape[1:]):
        raise ValueError('matching targets, predictions, and road-wise means required')
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('finite values required')
    error, mean_error = target-prediction, target-mean
    coefficients = [np.einsum('nr,tnc->trc', basis, a, optimize=True) for a in (target, error, mean_error)]
    energies = []
    for band in range(len(BAND_EDGES)-1):
        energies.append(np.stack([np.square(a[:, bins == band]).sum(axis=(0, 1)) for a in coefficients]))
    result = np.stack(energies)
    direct = np.stack([np.square(a).sum(axis=(0, 1)) for a in (target, error, mean_error)])
    np.testing.assert_allclose(result.sum(axis=0), direct, rtol=2e-10, atol=1e-8)
    return result


def summarize(groups, seed):
    rows = []
    for space in sorted({r['space'] for r in groups}):
        selected = [r for r in groups if r['space'] == space]
        for band in range(len(BAND_EDGES)-1):
            band_rows = [r for r in selected if r['band'] == band]
            row = dict(space=space, band=band, lower=float(BAND_EDGES[band]), upper=float(BAND_EDGES[band+1]),
                       modes=band_rows[0]['modes'], groups=len(band_rows), windows=int(sum(r['windows'] for r in band_rows)))
            for metric, numerator, denominator in [('energy_fraction', 'zero_sse', 'total_energy'),
                    ('skill_vs_zero', 'prediction_sse', 'zero_sse'), ('skill_vs_train_mean', 'prediction_sse', 'mean_sse')]:
                value, low, high = bootstrap_ratio([r[numerator] for r in band_rows], [r[denominator] for r in band_rows],
                                                  repetitions=2000, seed=seed)
                if metric.startswith('skill') and value is not None:
                    value, low, high = 1-value, 1-high, 1-low
                row.update({metric:value, metric+'_ci95_low':low, metric+'_ci95_high':high})
            rows.append(row)
    return rows


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for column, space in enumerate(('latent', 'count', 'speed', 'joint')):
        chosen = [r for r in rows if r['space'] == space]
        x = np.arange(len(chosen))
        axes[0,column].bar(x, [r['energy_fraction'] for r in chosen], color='#4c78a8')
        axes[0,column].set_title(space+' increment energy')
        for metric, label, color in [('skill_vs_zero','vs persistence','#4c78a8'),
                                     ('skill_vs_train_mean','vs train mean','#e45756')]:
            values = np.array([r[metric] for r in chosen])
            lower = np.array([r[metric+'_ci95_low'] for r in chosen])
            upper = np.array([r[metric+'_ci95_high'] for r in chosen])
            axes[1,column].errorbar(x,values,yerr=[values-lower,upper-values],label=label,color=color,marker='o',capsize=2)
        axes[1,column].axhline(0,color='gray',lw=.8)
        axes[1,column].set_title(space+' held-out predictive skill')
        for axis in axes[:,column]:
            axis.set_xticks(x,[f'{r["lower"]:g}–{min(r["upper"],2):g}' for r in chosen],rotation=35)
            axis.set_xlabel('normalized-Laplacian frequency band'); axis.grid(axis='y',alpha=.2)
    axes[1,0].legend(fontsize=8)
    fig.suptitle('Frozen state-only front; group bootstrap 95% CI; no action inputs')
    for suffix in ('png','pdf'):
        fig.savefig(output/('frequency_predictability.'+suffix),dpi=180)
    plt.close(fig)


def run(args):
    started=time.time(); torch.set_num_threads(2);torch.set_num_interop_threads(2)
    if args.output.exists():
        raise ValueError('preserve existing experiment; choose a new output')
    args.output.mkdir(parents=True)
    selection=json.loads(args.selection.read_text())
    if {r['demand_group'] for r in selection['train']} & {r['demand_group'] for r in selection['validation']}:
        raise ValueError('disjoint demand groups required')
    checkpoint=args.experiment/'snapshot/checkpoints/latest.pt'
    references={'checkpoint':checkpoint,'selection':args.selection,'basis':args.basis,
                'identity_coefficients':args.identity/'output_head_coefficients.npz'}
    reference_hashes={name:sha256(path) for name,path in references.items()}
    if reference_hashes['checkpoint'] != CHECKPOINT_SHA:
        raise ValueError('completed pretrained checkpoint required')
    protocol=dict(status='prespecified_before_scores',device='cpu',threads=2,seed=args.seed,
        frequency_edges=BAND_EDGES.tolist(),eigspace_tolerance=1e-8,reference_hashes=reference_hashes,
        train_groups=len(selection['train']),validation_groups=len(selection['validation']),windows_per_group=12,
        latent_head='saved raw_next__identity_tau0__output_only; 65x64; no refit',
        physical_head='train-only F64 ridge, 65x2, relative lambda 1e-6; frozen same state-only front',
        targets='raw actual adjacent 30-second endpoint increments; births included; no speed interval averaging',
        state_inputs='current flow encoding, native state/downstream/upstream branches; condition branch omitted',
        baselines=['zero increment persistence','per-road/channel training mean increment'],
        joint_scaling='each physical channel divided by its train-only global increment RMS; no centering',
        metrics=['held-out energy fraction','1-SSE_pred/SSE_zero','1-SSE_pred/SSE_train_mean'],
        bootstrap='2000 whole demand-group resamples; pooled sums; same resample seed',
        limits=['development validation previously inspected','frozen old action-conditioned body with condition removed',
                'one-step linear-head probe; not fully retrained soft/FМ model',
                'state-only prediction is not a causal zero-action background','Ingolstadt SUMO replay, not sensor traffic'],
        noise_used=False,actions_used=False,validation_fitting=False)
    atomic_json(args.output/'protocol.json',protocol); atomic_json(args.output/'selection.json',selection)
    source_manifest={}
    for path in [*ROOT.joinpath('urban_dreamer').rglob('*.py'),*ROOT.joinpath('tools').glob('*.py')]:
        relative=path.relative_to(ROOT); destination=args.output/'source'/relative
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        source_manifest[str(relative)]=sha256(path)
    atomic_json(args.output/'source_manifest.json',source_manifest)
    with np.load(args.basis) as asset:
        basis=asset['basis'];eigenvalues=asset['eigenvalues']
    bins=frequency_bands(eigenvalues)
    config,prepared,topology,model,_=load_frozen(args.experiment,device='cpu')
    model.eval().requires_grad_(False);prior=model.dynamics.graph_prior
    assert prior.prior_filter=='hard' and prior.prior_action_conditioned and not prior.filter_input
    np.testing.assert_allclose(basis.T@basis,np.eye(len(basis)),atol=2e-10)
    lap=normalized_laplacian(prior.adjacency.numpy())
    np.testing.assert_allclose(lap@basis,basis*eigenvalues,atol=2e-10)
    capacities=np.maximum(np.asarray(topology.network.capacities,dtype=np.float64),1.)
    limits=np.asarray(topology.network.speed_limits_mps,dtype=np.float64)
    with np.load(references['identity_coefficients']) as asset:
        latent_head=asset['raw_next__identity_tau0__output_only'].copy()
    gram=np.zeros((65,65));cross=np.zeros((65,2));physical_energy=np.zeros(2)
    train_sums={'latent':np.zeros((len(basis),64)),'physical':np.zeros((len(basis),2))};train_windows=0
    groups=[];audit=[];validation_identity_sse=0.;validation_full_stats={'gram':np.zeros((65,65)),
                    'physical_cross':np.zeros((65,2)),'physical_energy':0.,'latent_cross':np.zeros((65,64)),'latent_energy':0.}
    with torch.inference_mode():
        for split in ('train','validation'):
            source=dataset(args.experiment,config,prepared,topology,split=split,horizon=1)
            indices={e['cache_path']:i for i,e in enumerate(source.manifest['entries'])}
            for ordinal,entry in enumerate(selection[split]):
                index=indices[entry['cache_path']];arrays=source._episode(index);starts=select_starts(arrays['flow'],12)
                batch=source.batch_from_starts(index,starts.tolist(),materialization_horizon=1,fm_only=True)
                current=batch.flow[:,0];future=torch.as_tensor(arrays['flow'][starts+1],dtype=current.dtype)
                z=model.encode(current);delta_latent=(model.encode(future)-z).double().numpy()
                hidden=state_hidden(prior,z[:,None])[:,0].double().numpy();x=augmented(hidden);flat=x.reshape(-1,65)
                physical=physical_features(arrays['flow'][starts+1],capacities,limits)-physical_features(arrays['flow'][starts],capacities,limits)
                if split=='train':
                    gram+=flat.T@flat;cross+=flat.T@physical.reshape(-1,2)
                    physical_energy+=np.square(physical).sum(axis=(0,1))
                    train_sums['latent']+=delta_latent.sum(axis=0);train_sums['physical']+=physical.sum(axis=0)
                    train_windows+=len(starts)
                else:
                    prediction_latent=x@latent_head;prediction_physical=x@physical_head
                    validation_identity_sse+=float(np.square(delta_latent-prediction_latent).sum())
                    validation_full_stats['gram']+=flat.T@flat
                    for name,target in [('physical',physical),('latent',delta_latent)]:
                        validation_full_stats[name+'_cross']+=flat.T@target.reshape(-1,target.shape[-1])
                        validation_full_stats[name+'_energy']+=float(np.square(target).sum())
                    latent_stats=band_statistics(delta_latent,prediction_latent,train_means['latent'],basis,bins)
                    physical_stats=band_statistics(physical,prediction_physical,train_means['physical'],basis,bins)
                    all_stats={'latent':latent_stats.sum(axis=-1),'count':physical_stats[...,0],
                        'speed':physical_stats[...,1],'joint':(physical_stats/rms[None,None,:]**2).sum(axis=-1)}
                    for space,stats in all_stats.items():
                        for band,values in enumerate(stats):
                            groups.append(dict(demand_group=entry['demand_group'],space=space,band=band,
                                modes=int((bins==band).sum()),windows=len(starts),zero_sse=float(values[0]),
                                prediction_sse=float(values[1]),mean_sse=float(values[2]),total_energy=float(stats[:,0].sum())))
                audit.append(dict(split=split,cache_path=entry['cache_path'],content_hash=entry['cache_content_hash'],
                    demand_group=entry['demand_group'],starts=starts.tolist()))
                print(json.dumps(dict(stage=split,group=ordinal+1,elapsed_seconds=time.time()-started)),flush=True)
            if split=='train':
                physical_head,fit=solve_ridge(gram,cross,float(physical_energy.sum()))
                train_means={name:values/train_windows for name,values in train_sums.items()}
                rms=np.sqrt(physical_energy/(train_windows*len(basis)))
                if (rms<=0).any():raise ValueError('nonzero physical training RMS required')
                atomic_json(args.output/'fit_diagnostics.json',dict(physical=fit,train_windows=train_windows,physical_increment_rms=rms.tolist()))
                np.savez_compressed(args.output/'fitted_arrays.npz',physical_head=physical_head,
                    latent_head=latent_head,train_mean_latent=train_means['latent'],train_mean_physical=train_means['physical'],rms=rms)
    for entry in audit:
        if archive_content_hash(Path(entry['cache_path'])) != entry['content_hash']:
            raise ValueError('selected data changed')
    independent={}
    for space,head in [('physical',physical_head),('latent',latent_head)]:
        s=validation_full_stats;sse=s[space+'_energy']-2*np.sum(head*s[space+'_cross'])+np.sum(head*(s['gram']@head))
        direct=sum(r['prediction_sse'] for r in groups if r['space']==('joint' if space=='physical' else 'latent'))
        if space=='physical':
            direct=sum(r['prediction_sse'] for r in groups if r['space'] in ('count','speed'))
        np.testing.assert_allclose(sse,direct,rtol=1e-10,atol=1e-7)
        independent[space]=dict(sufficient_statistics_sse=float(sse),band_summed_sse=direct)
    reference=json.loads((args.identity/'summary.json').read_text())
    expected=next(r for r in reference['metrics'] if r['target']=='raw_next' and r['arm']=='output_only')['full_normalized_sse']
    actual=validation_identity_sse/sum(r['zero_sse'] for r in groups if r['space']=='latent')
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    rows=summarize(groups,args.seed);write_csv(args.output/'demand_group_metrics.csv',groups);write_csv(args.output/'metric_summary.csv',rows)
    atomic_json(args.output/'data_audit.json',audit);plot(rows,args.output)
    imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
             if name.startswith(('tools.','urban_dreamer.')) and getattr(module,'__file__',None)}
    for name,path in imports.items():
        relative=str(Path(path).relative_to(ROOT))
        if relative not in source_manifest or sha256(Path(path)) != source_manifest[relative]:
            raise ValueError('source mismatch: '+name)
    for name,path in references.items():
        if sha256(path)!=reference_hashes[name]:raise ValueError('reference changed: '+name)
    atomic_json(args.output/'import_audit.json',imports)
    np.savez_compressed(args.output/'validation_sufficient_statistics.npz',**validation_full_stats)
    atomic_json(args.output/'summary.json',dict(status='complete',protocol=protocol,metrics=rows,
        physical_increment_rms=rms.tolist(),identity_reference_normalized_sse=actual,
        bands=[dict(band=i,modes=int((bins==i).sum()),lambda_min=float(eigenvalues[bins==i].min()) if (bins==i).any() else None,
                    lambda_max=float(eigenvalues[bins==i].max()) if (bins==i).any() else None) for i in range(len(BAND_EDGES)-1)],
        independent_verification=independent,parseval_verified=True,source_imports_verified=True,
        references_unchanged=True,data_unchanged=True,elapsed_seconds=time.time()-started))
    print(json.dumps(dict(stage='complete',elapsed_seconds=time.time()-started)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=Path('outputs/component_experiments_20261011'))
    parser.add_argument('--selection',type=Path,default=Path('outputs/pretrained_prior_rank_20261013/v1/selection.json'))
    parser.add_argument('--basis',type=Path,default=Path('outputs/pretrained_prior_rank_20261013/v1/exact_basis.npz'))
    parser.add_argument('--identity',type=Path,default=Path('outputs/traffic_increment_spectrum_20261013/input_output_v1/identity_filter_supplement'))
    parser.add_argument('--output',type=Path,default=Path('outputs/traffic_increment_spectrum_20261013/predictability_v1'))
    parser.add_argument('--seed',type=int,default=20261013)
    run(parser.parse_args())
