"""Fully solved, frozen-front soft-background heads on immutable CPU replay.

The study distinguishes an output filter from a filtered supervision target.
It does not retrain the encoder, prior front, decoder, or Flow Matching model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = Path(os.environ.get('SOFT_BACKGROUND_FROZEN_ROOT', ROOT)).resolve()
sys.path.insert(0, str(FROZEN_ROOT))

import numpy as np
import torch

from tools.analyze_traffic_spectrum import bootstrap_ratio, normalized_laplacian, physical_features, select_starts, select_grouped
from tools.component_experiment_data import atomic_json, dataset, load_frozen, sha256
from tools.evaluate_pretrained_prior_ranks import CHECKPOINT_SHA, physical_sums, solve_ridge, write_csv
from tools.evaluate_prior_input_filter import state_hidden
from tools.evaluate_frequency_predictability import augmented, frequency_bands, BAND_EDGES
from urban_dreamer.learning.offline_data import archive_content_hash
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.models.world_model import LowFrequencyGraphPrior

TAUS = (0., .01, .03, .1, .3, 1., 3., 10., 30., 100.)


def spectral_moments(x, y):
    """Per-mode training moments; no fitting or cross-road mixing here."""
    x, y = np.asarray(x,dtype=np.float64), np.asarray(y,dtype=np.float64)
    if x.ndim != 3 or y.ndim != 3 or x.shape[:2] != y.shape[:2]:
        raise ValueError('matching [windows,modes,features] arrays required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite moments required')
    return dict(gram=np.einsum('tnd,tne->nde',x,x,optimize=True),
                cross=np.einsum('tnd,tnc->ndc',x,y,optimize=True),
                energy=np.square(y).sum(axis=(0,2)))


def weighted_moments(moments, gains, supervision):
    """F(XW) versus F(y) uses g² cross; F(XW) versus y uses g cross."""
    gains=np.asarray(gains,dtype=np.float64)
    if gains.shape != moments['energy'].shape or not np.isfinite(gains).all():
        raise ValueError('matching finite frequency gains required')
    if supervision not in ('filtered','full'):
        raise ValueError('supervision must be filtered or full')
    weights=gains**2
    return dict(gram=np.einsum('n,nde->de',weights,moments['gram'],optimize=True),
        cross=np.einsum('n,ndc->dc',weights if supervision=='filtered' else gains,moments['cross'],optimize=True),
        energy=float(np.sum((weights if supervision=='filtered' else 1)*moments['energy'])))


def solve_weighted(moments,gains,supervision):
    aggregate=weighted_moments(moments,gains,supervision)
    head,proof=solve_ridge(aggregate['gram'],aggregate['cross'],aggregate['energy'])
    return head,proof,aggregate


def sufficient_sse(head,aggregate):
    return float(aggregate['energy']-2*np.sum(head*aggregate['cross'])+
                 np.sum(head*(aggregate['gram']@head)))


def add_moments(total,addition):
    for key,value in addition.items():
        total[key]+=value


def project(basis,array):
    return np.einsum('nr,tnc->trc',basis,array,optimize=True)


def spectral_sums(target,prediction,gains,bins):
    target,prediction=np.asarray(target),np.asarray(prediction)
    if target.shape!=prediction.shape or gains.shape!=(target.shape[1],):
        raise ValueError('matching spectra and gains required')
    error=target-prediction;filtered=gains[None,:,None]*target
    bands=np.array([float(np.square(error[:,bins==i]).sum()) for i in range(len(BAND_EDGES)-1)])
    full=float(np.square(error).sum())
    np.testing.assert_allclose(bands.sum(),full,rtol=1e-12,atol=1e-8)
    return dict(full_sse=full,total_delta_energy=float(np.square(target).sum()),
        filtered_target_sse=float(np.square(filtered-prediction).sum()),
        filtered_delta_energy=float(np.square(filtered).sum()),
        filtered_error_sse=float(np.square(gains[None,:,None]*error).sum()),
        prediction_energy=float(np.square(prediction).sum())),bands


METRICS=[('full_normalized_sse','full_sse','total_delta_energy'),
         ('filtered_target_normalized_sse','filtered_target_sse','filtered_delta_energy'),
         ('filtered_error_normalized_sse','filtered_error_sse','filtered_delta_energy'),
         ('direct_count_normalized_sse','direct_count_sse','direct_count_energy'),
         ('direct_speed_normalized_sse','direct_speed_sse','direct_speed_energy'),
         ('direct_joint_normalized_sse','direct_joint_sse','direct_joint_energy'),
         *[(metric+'_mae'+suffix,metric+suffix+'_absolute_error',metric+suffix+'_elements')
           for metric in ('count','speed') for suffix in ('','_occupied')]]


def summarize(groups,seed):
    rows=[]
    for candidate in sorted({r['candidate'] for r in groups}):
        selected=[r for r in groups if r['candidate']==candidate]
        row={key:selected[0][key] for key in ('candidate','filter','tau','supervision')}
        row.update(groups=len(selected),windows=int(sum(r['windows'] for r in selected)))
        for metric,numerator,denominator in METRICS:
            value,lo,hi=bootstrap_ratio([r[numerator] for r in selected],[r[denominator] for r in selected],
                repetitions=2000,seed=seed)
            row.update({metric:value,metric+'_ci95_low':lo,metric+'_ci95_high':hi})
        row['full_latent_skill_vs_zero']=1-row['full_normalized_sse']
        rows.append(row)
    return rows


def contrasts(groups,seed):
    indexes={candidate:{r['demand_group']:r for r in groups if r['candidate']==candidate}
        for candidate in sorted({r['candidate'] for r in groups})}
    rows=[]
    reference_names=('zero','soft_tau0__filtered','hard_r16__filtered')
    for candidate,data in indexes.items():
        for reference in reference_names:
            if candidate==reference:continue
            other=indexes[reference];keys=sorted(data)
            if keys!=sorted(other):raise ValueError('paired groups required')
            for metric,numerator,denominator in [r for r in METRICS if not r[0].startswith('filtered')]:
                denominators=[data[k][denominator] for k in keys]
                np.testing.assert_allclose(denominators,[other[k][denominator] for k in keys],rtol=1e-12)
                value,lo,hi=bootstrap_ratio([data[k][numerator]-other[k][numerator] for k in keys],denominators,
                    repetitions=2000,seed=seed)
                rows.append(dict(candidate=candidate,reference=reference,metric=metric,
                    candidate_minus_reference=value,ci95_low=lo,ci95_high=hi))
    for tau in TAUS:
        a,b=indexes[f'soft_tau{tau:g}__full'],indexes[f'soft_tau{tau:g}__filtered'];keys=sorted(a)
        for metric,numerator,denominator in METRICS:
            value,lo,hi=bootstrap_ratio([a[k][numerator]-b[k][numerator] for k in keys],[a[k][denominator] for k in keys],
                repetitions=2000,seed=seed)
            rows.append(dict(candidate=f'soft_tau{tau:g}__full',reference=f'soft_tau{tau:g}__filtered',
                metric=metric,candidate_minus_reference=value,ci95_low=lo,ci95_high=hi))
    return rows


def band_summary(records,seed):
    rows=[]
    for candidate in sorted({r['candidate'] for r in records}):
        for band in range(len(BAND_EDGES)-1):
            selected=[r for r in records if r['candidate']==candidate and r['band']==band]
            value,lo,hi=bootstrap_ratio([r['prediction_sse'] for r in selected],[r['target_energy'] for r in selected],
                repetitions=2000,seed=seed)
            rows.append(dict(candidate=candidate,band=band,lower=float(BAND_EDGES[band]),upper=float(BAND_EDGES[band+1]),
                modes=selected[0]['modes'],skill_vs_zero=1-value,ci95_low=1-hi,ci95_high=1-lo))
    return rows


def pareto(rows,metrics):
    """No scalarization: only point-estimate nondominance on named metrics."""
    return [r['candidate'] for r in rows if not any(
        all(other[k]<=r[k] for k in metrics) and any(other[k]<r[k] for k in metrics)
        for other in rows)]


def plot(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    metrics=('full_normalized_sse','direct_joint_normalized_sse','count_mae',
             'count_mae_occupied','speed_mae_occupied','speed_mae')
    for axis,metric in zip(axes.flat,metrics):
        for supervision,color in [('filtered','#4c78a8'),('full','#e45756')]:
            selected=sorted([r for r in rows if r['filter']=='soft' and r['supervision']==supervision],key=lambda r:r['tau'])
            x=[r['tau'] for r in selected];y=np.array([r[metric] for r in selected])
            lo=np.array([r[metric+'_ci95_low'] for r in selected]);hi=np.array([r[metric+'_ci95_high'] for r in selected])
            axis.errorbar(x,y,yerr=[y-lo,hi-y],label=supervision+' target',color=color,marker='o',capsize=2)
        for reference,color in [('zero','gray'),('hard_r16__filtered','#59a14f')]:
            row=next(r for r in rows if r['candidate']==reference)
            axis.axhline(row[metric],label=reference,color=color,linestyle='--',alpha=.8)
        axis.set_xscale('symlog',linthresh=.1);axis.set_xlabel('soft-filter strength tau');axis.set_title(metric)
        axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Frozen state-only front, solved heads: target shrinkage versus output filtering')
    for suffix in ('png','pdf'):fig.savefig(output/('soft_background_tau.'+suffix),dpi=180)
    plt.close(fig)


def confirmation_selection(entries,previous,count,seed):
    excluded={r['demand_group'] for split in ('train','validation') for r in previous[split]}
    eligible=[r for r in entries if r['split']=='validation' and r['demand_group'] not in excluded]
    selected=select_grouped(eligible,'validation',count,seed)
    if len(selected)!=count:raise ValueError('insufficient unused validation demand groups')
    if {r['demand_group'] for r in selected}&excluded:raise ValueError('confirmation group overlap')
    return selected


def confirm(args):
    """Evaluate fixed already-solved heads on unused groups; never fit here."""
    started=time.time();torch.set_num_threads(2);torch.set_num_interop_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():raise ValueError('preserve completed confirmation')
    if FROZEN_ROOT!=(args.output/'source').resolve():raise ValueError('confirmation source must be frozen independently')
    manifest=json.loads((args.output/'source_manifest.json').read_text())
    def verify_source():
        for relative,digest in manifest.items():
            if sha256(FROZEN_ROOT/relative)!=digest:raise ValueError('source changed')
    verify_source()
    previous=json.loads(args.selection.read_text());full_manifest=json.loads(args.validation_manifest.read_text())
    selected=confirmation_selection(full_manifest['entries'],previous,args.confirmation_groups,args.seed)
    original=json.loads((args.confirm_from/'summary.json').read_text())
    checkpoint=args.experiment/'snapshot/checkpoints/latest.pt'
    refs={'checkpoint':checkpoint,'selection':args.selection,'basis':args.basis,
        'heads':args.confirm_from/'heads.npz','screen_summary':args.confirm_from/'summary.json',
        'full_manifest':args.validation_manifest}
    hashes={name:sha256(path) for name,path in refs.items()}
    if hashes['checkpoint']!=CHECKPOINT_SHA:raise ValueError('completed checkpoint required')
    candidates=list(dict.fromkeys([*args.candidates,'zero']))
    for name in candidates:
        if name not in {r['candidate'] for r in original['metrics']}:raise ValueError('unknown fixed candidate')
    protocol=dict(status='prespecified_before_confirmation_scores',study_role='unused-development-group confirmation; no refit',
        device='cpu',threads=2,seed=args.seed,validation_groups=len(selected),windows_per_group=12,
        selected_candidates=candidates,selection='deterministic random full-manifest validation demand groups and episodes; excludes prior64train+32validation groups',
        candidate_selection_rationale='Fixed after first screen, before new-group scores: persistence and native hard controls; tau0 identity; tau.01 screened latent minimum; tau.1 practical weak-bias candidate preserving latent skill and improving decoded all-road count; tau1 intermediate; old tau10 filtered; tau10 full-target supervision diagnostic. No candidate or tau tuning on confirmation.',
        source_manifest_sha256=sha256(args.output/'source_manifest.json'),reference_hashes=hashes,
        limitations=['Candidates chosen after development screening.',
            'These groups were unused in this hyperparameter study but belong to the existing pretraining validation corpus; not a pristine external test.',
            'Frozen encoder/body/decoder and copied solved heads; not joint training or full FM dynamics.',
            'RAW next-state latent/physical targets differ from production cohort-adjusted hybrid FM target.',
            'State-only forecast is not causal zero-action traffic evolution.'],
        decoder_speed='mean raw next/next-next interval speed; unavailable following omitted',
        direct_physical='actual raw next-current count/speed increments; train RMS from original screen retained',
        fit_performed=False,actions_used=False,noise_used=False,bootstrap='2000 whole-demand-group resamples; pooled paired sufficient-statistic ratios')
    atomic_json(args.output/'protocol.json',protocol);atomic_json(args.output/'selection.json',dict(validation=selected))
    config,prepared,topology,model,_=load_frozen(args.experiment,device='cpu');model.eval().requires_grad_(False)
    prior=model.dynamics.graph_prior
    assert prior.prior_filter=='hard' and prior.prior_action_conditioned and not prior.filter_input
    source=CompactPretrainDataset(config,prepared,topology,1,cache_manifest_path=args.validation_manifest,split='validation')
    indices={r['cache_path']:i for i,r in enumerate(source.manifest['entries'])}
    with np.load(args.basis) as asset:basis=asset['basis'];eigenvalues=asset['eigenvalues']
    bins=frequency_bands(eigenvalues);roads=len(eigenvalues)
    np.testing.assert_allclose(basis.T@basis,np.eye(roads),atol=2e-10)
    np.testing.assert_allclose(normalized_laplacian(prior.adjacency.numpy())@basis,basis*eigenvalues,atol=2e-10)
    native_hard=(prior.basis@prior.basis.T).double().numpy()
    with np.load(args.confirm_from/'heads.npz') as asset:
        heads={name:{space:asset[name+'__'+space].copy() for space in ('latent','physical')} for name in candidates if name!='zero'}
    rms=np.asarray(original['proof']['physical_increment_train_rms'])
    capacities=np.maximum(np.asarray(topology.network.capacities,dtype=np.float64),1.)
    limits=np.asarray(topology.network.speed_limits_mps,dtype=np.float64)
    records=[];bands=[];audit=[]
    with torch.inference_mode():
        for ordinal,entry in enumerate(selected):
            index=indices[entry['cache_path']];arrays=source._episode(index);starts=select_starts(arrays['flow'],12)
            batch=source.batch_from_starts(index,starts.tolist(),materialization_horizon=1,fm_only=True)
            current=batch.flow[:,0];future=torch.as_tensor(arrays['flow'][starts+1],dtype=current.dtype)
            np.testing.assert_array_equal(current.numpy(),arrays['flow'][starts])
            z=model.encode(current);delta=(model.encode(future)-z).double().numpy();x=augmented(state_hidden(prior,z[:,None])[:,0].numpy())
            actual=physical_features(arrays['flow'][starts+1],capacities,limits)-physical_features(arrays['flow'][starts],capacities,limits)
            xs=project(basis,x);ys=project(basis,delta);actual_spectrum=project(basis,actual)
            np.testing.assert_allclose(np.square(ys).sum(),np.square(delta).sum(),rtol=2e-10)
            np.testing.assert_allclose(np.square(actual_spectrum).sum(),np.square(actual).sum(),rtol=2e-10)
            valid=starts+2<len(arrays['flow']);following=np.zeros_like(arrays['flow'][starts+1]);following[valid]=arrays['flow'][starts[valid]+2]
            for name in candidates:
                original_row=next(r for r in original['metrics'] if r['candidate']==name)
                gains=np.ones(roads) if original_row['tau'] is None else 1/(1+original_row['tau']*eigenvalues)
                if name=='zero':
                    prediction=np.zeros_like(delta);ps=np.zeros_like(ys);phy_ps=np.zeros_like(actual_spectrum)
                    metric,band_sse=spectral_sums(ys,ps,gains,bins)
                elif original_row['filter']=='hard':
                    prediction=(native_hard@x)@heads[name]['latent'];ps=project(basis,prediction)
                    phy_ps=project(basis,(native_hard@x)@heads[name]['physical']);filtered=native_hard@delta
                    full=float(np.square(delta-prediction).sum())
                    metric=dict(full_sse=full,total_delta_energy=float(np.square(delta).sum()),
                        filtered_target_sse=float(np.square(filtered-prediction).sum()),filtered_delta_energy=float(np.square(filtered).sum()),
                        filtered_error_sse=float(np.square(native_hard@(delta-prediction)).sum()),prediction_energy=float(np.square(prediction).sum()))
                    band_sse=np.array([np.square((ys-ps)[:,bins==i]).sum() for i in range(len(BAND_EDGES)-1)])
                    np.testing.assert_allclose(band_sse.sum(),full,rtol=2e-10)
                else:
                    ps=gains[None,:,None]*(xs@heads[name]['latent']);phy_ps=gains[None,:,None]*(xs@heads[name]['physical'])
                    prediction=x@heads[name]['latent'] if original_row['tau']==0 else np.einsum('nr,trc->tnc',basis,ps,optimize=True)
                    metric,band_sse=spectral_sums(ys,ps,gains,bins)
                physical_sse=np.square(actual_spectrum-phy_ps).sum(axis=(0,1));physical_energy=np.square(actual_spectrum).sum(axis=(0,1))
                row={k:original_row[k] for k in ('filter','tau','supervision')}
                row.update(candidate=name,demand_group=entry['demand_group'],windows=len(starts),**metric,
                    direct_count_sse=float(physical_sse[0]),direct_count_energy=float(physical_energy[0]),
                    direct_speed_sse=float(physical_sse[1]),direct_speed_energy=float(physical_energy[1]),
                    direct_joint_sse=float(np.sum(physical_sse/rms**2)),direct_joint_energy=float(np.sum(physical_energy/rms**2)))
                row.update(physical_sums(model,z+torch.as_tensor(prediction,dtype=z.dtype),future,following,valid,capacities,limits));records.append(row)
                for band,value in enumerate(band_sse):
                    bands.append(dict(candidate=name,demand_group=entry['demand_group'],band=band,modes=int((bins==band).sum()),
                        prediction_sse=float(value),target_energy=float(np.square(ys[:,bins==band]).sum())))
            audit.append(dict(cache_path=entry['cache_path'],content_hash=entry['cache_content_hash'],demand_group=entry['demand_group'],starts=starts.tolist()))
            print(json.dumps(dict(stage='confirmation',group=ordinal+1,total=len(selected),elapsed_seconds=time.time()-started)),flush=True)
    for name,path in refs.items():
        if sha256(path)!=hashes[name]:raise ValueError('reference changed')
    for entry in audit:
        if archive_content_hash(Path(entry['cache_path']))!=entry['content_hash']:raise ValueError('data changed')
    imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
        if name.startswith(('tools.','urban_dreamer.')) and getattr(module,'__file__',None)}
    if any(not Path(path).is_relative_to(FROZEN_ROOT) for path in imports.values()):raise ValueError('live import contaminated frozen execution')
    verify_source();atomic_json(args.output/'import_audit.json',imports)
    rows=summarize(records,args.seed);band_rows=band_summary(bands,args.seed)
    # Only representative candidates are present: compute applicable paired references.
    indexes={name:{r['demand_group']:r for r in records if r['candidate']==name} for name in candidates};paired=[]
    for name,selected_rows in indexes.items():
        for reference in ('zero','soft_tau0__filtered','soft_tau10__filtered','hard_r16__filtered'):
            if reference not in indexes or reference==name:continue
            keys=sorted(selected_rows);other=indexes[reference]
            if keys!=sorted(other):raise ValueError('paired groups required')
            for metric,numerator,denominator in [r for r in METRICS if not r[0].startswith('filtered')]:
                value,lo,hi=bootstrap_ratio([selected_rows[k][numerator]-other[k][numerator] for k in keys],
                    [selected_rows[k][denominator] for k in keys],repetitions=2000,seed=args.seed)
                paired.append(dict(candidate=name,reference=reference,metric=metric,candidate_minus_reference=value,ci95_low=lo,ci95_high=hi))
    write_csv(args.output/'demand_group_metrics.csv',records);write_csv(args.output/'frequency_group_metrics.csv',bands)
    write_csv(args.output/'metric_summary.csv',rows);write_csv(args.output/'frequency_summary.csv',band_rows)
    atomic_json(args.output/'paired_contrasts.json',paired);atomic_json(args.output/'data_audit.json',audit)
    plot(rows,args.output)
    atomic_json(args.output/'summary.json',dict(status='complete',protocol=protocol,metrics=rows,paired_contrasts=paired,
        frequency_metrics=band_rows,heads_unchanged=True,checkpoint_unchanged=True,data_unchanged=True,
        source_unchanged=True,parseval_all_bands_verified=True,train_and_screen_group_overlap=False,
        elapsed_seconds=time.time()-started))
    print(json.dumps(dict(stage='complete_confirmation',elapsed_seconds=time.time()-started)),flush=True)


def run(args):
    started=time.time();torch.set_num_threads(2);torch.set_num_interop_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():raise ValueError('preserve completed results')
    if FROZEN_ROOT != (args.output/'source').resolve():
        raise ValueError('execute copied script using this new study immutable source')
    manifest=json.loads((args.output/'source_manifest.json').read_text())
    def verify_source():
        for relative,digest in manifest.items():
            if sha256(FROZEN_ROOT/relative)!=digest:raise ValueError('source changed: '+relative)
    verify_source()
    checkpoint=args.experiment/'snapshot/checkpoints/latest.pt'
    references={'checkpoint':checkpoint,'selection':args.selection,'graph_basis':args.basis,
        'identity_coefficients':args.identity/'output_head_coefficients.npz',
        'tau10_coefficients':args.reference/'output_head_coefficients.npz',
        'identity_summary':args.identity/'summary.json','tau10_summary':args.reference/'summary.json'}
    hashes={name:sha256(path) for name,path in references.items()}
    if hashes['checkpoint']!=CHECKPOINT_SHA:raise ValueError('completed checkpoint required')
    selection=json.loads(args.selection.read_text())
    if {r['demand_group'] for r in selection['train']} & {r['demand_group'] for r in selection['validation']}:
        raise ValueError('train/validation groups overlap')
    protocol=dict(status='prespecified_before_scores',device='cpu',threads=2,seed=args.seed,taus=list(TAUS),
        reference_hashes=hashes,source_manifest_sha256=sha256(args.output/'source_manifest.json'),
        train_groups=len(selection['train']),validation_groups=len(selection['validation']),windows_per_group=12,
        latent_heads='65x64, 4160 parameters each; full input, state-only frozen front',
        direct_physical_heads='65x2, 130 parameters each; same front, actual raw count/speed increments',
        filter='F_tau=(I+tau Lsym)^-1; exact fixed full spectrum; all modes retained',
        supervision={'filtered':'||F_tau(delta)-F_tau(XW)||²','full':'||delta-F_tau(XW)||²'},
        spectral_fit={'gram':'sum_k g_k² G_k','filtered_cross':'sum_k g_k² C_k','full_cross':'sum_k g_k C_k'},
        ridge='F64 closed form; 1e-6 * trace(filtered Gram)/65; bias penalized; no validation tuning',
        tau0_equivalence='two targets coincide; solve once, share identical coefficients and forecasts',
        hard_control='checkpoint native U16 U16.T; filtered supervision; no replacement eigenbasis',
        zero_control='decode current latent; zero raw physical increment',
        latent_target='RAW actual next encode minus current encode, births included; no hybrid labels',
        decoder_count='decoded endpoint count versus raw next count; all and raw-next occupied roads',
        decoder_speed='decoded endpoint speed versus mean raw next/next-next interval speed; unavailable following omitted',
        direct_physical='RAW next-current count/speed increment; no interval averaging, no decoder',
        primary='FULL unfiltered latent SSE, same raw target and denominator for every candidate',
        secondary='decoded physical MAE and direct actual physical SSE reported separately; Pareto not scalar weighting',
        frequency_edges=BAND_EDGES.tolist(),bootstrap='2000 paired whole-demand-group resamples; pooled sufficient-statistic ratios',
        ranking_gate='reproduce prior native tau0/tau10 and hard control before reporting sweep rankings',
        reference_tolerances={'native_normalized_sse_absolute':1e-6,'native_decoder_mae_absolute':1e-5},
        noise_used=False,actions_used=False,SUMO_used=False,fully_retrained=False,final_test_used=False,
        caveats=['Previously inspected development validation; hyperparameter screening, not final test evidence.',
            'Only output heads solved; body/encoder/decoder trained with old hard/action-conditioned structure.',
            'State-only background is not causal zero-action prediction.',
            'Full versus filtered target changes supervised shrinkage, not only output architecture.',
            'No FM residual training, recursive distribution assessment, or downstream policy comparison.'])
    atomic_json(args.output/'protocol.json',protocol);atomic_json(args.output/'selection.json',selection)
    config,prepared,topology,model,_=load_frozen(args.experiment,device='cpu');model.eval().requires_grad_(False)
    prior=model.dynamics.graph_prior
    assert prior.prior_filter=='hard' and prior.prior_action_conditioned and not prior.filter_input
    original_basis=prior.basis.clone();original_noise=prior.noise_scale
    with np.load(args.basis) as asset:eigenvalues=asset['eigenvalues'];basis=asset['basis']
    bins=frequency_bands(eigenvalues);roads=len(eigenvalues)
    np.testing.assert_allclose(basis.T@basis,np.eye(roads),atol=2e-10)
    np.testing.assert_allclose(normalized_laplacian(prior.adjacency.numpy())@basis,basis*eigenvalues,atol=2e-10)
    native_hard=(prior.basis@prior.basis.T).double().numpy()
    native_soft=LowFrequencyGraphPrior(prior.adjacency,rank=16,noise_scale=original_noise,
        prior_filter='soft',prior_filter_strength=10.,prior_action_conditioned=False,prior_filter_input=False)
    native_soft.load_state_dict(dict(prior.state_dict(),soft_filter_operator=native_soft.soft_filter_operator),strict=True)
    native_tau10=native_soft.soft_filter_operator.double().numpy()
    gain10=1/(1+10*eigenvalues)
    proof={'soft_native_vs_exact_max_gain_error':float(np.max(np.abs(native_tau10@basis-basis*gain10))),
           'native_hard_rank':int(prior.basis.shape[1]),'archived_model_hard_action_conditioned':True,
           'front_condition_branch_omitted':True,'source_noise_changed':False}
    capacities=np.maximum(np.asarray(topology.network.capacities,dtype=np.float64),1.)
    limits=np.asarray(topology.network.speed_limits_mps,dtype=np.float64)
    empty=lambda channels:dict(gram=np.zeros((roads,65,65)),cross=np.zeros((roads,65,channels)),energy=np.zeros(roads))
    train={'latent':empty(64),'physical':empty(2)}
    val={'latent':empty(64),'physical':empty(2)}
    native_empty=lambda channels:dict(gram=np.zeros((65,65)),cross=np.zeros((65,channels)),energy=0.)
    native_stats={name:{space:native_empty(channels) for space,channels in [('latent',64),('physical',2)]}
                  for name in ('hard','tau10')}
    native_validation={name:{space:native_empty(channels) for space,channels in [('latent',64),('physical',2)]}
                  for name in ('hard','tau10')}
    candidates={};fits={};saved_stats={};records=[];bands=[];audit=[];native_reference_records=[]
    train_energy=np.zeros(2);train_elements=0
    with torch.inference_mode():
        for split in ('train','validation'):
            source=dataset(args.experiment,config,prepared,topology,split=split,horizon=1)
            indices={e['cache_path']:i for i,e in enumerate(source.manifest['entries'])}
            for ordinal,entry in enumerate(selection[split]):
                index=indices[entry['cache_path']];arrays=source._episode(index);starts=select_starts(arrays['flow'],12)
                batch=source.batch_from_starts(index,starts.tolist(),materialization_horizon=1,fm_only=True)
                current=batch.flow[:,0];future=torch.as_tensor(arrays['flow'][starts+1],dtype=current.dtype)
                np.testing.assert_array_equal(current.numpy(),arrays['flow'][starts])
                z=model.encode(current);delta=(model.encode(future)-z).double().numpy()
                x=augmented(state_hidden(prior,z[:,None])[:,0].numpy())
                physical=physical_features(arrays['flow'][starts+1],capacities,limits)-physical_features(arrays['flow'][starts],capacities,limits)
                xs=project(basis,x);ys={'latent':project(basis,delta),'physical':project(basis,physical)}
                for space in ('latent','physical'):
                    np.testing.assert_allclose(np.square(ys[space]).sum(),np.square(delta if space=='latent' else physical).sum(),rtol=2e-10)
                    add_moments((train if split=='train' else val)[space],spectral_moments(xs,ys[space]))
                for native_name,operator in [('hard',native_hard),('tau10',native_tau10)]:
                    xf=operator@x;flat=xf.reshape(-1,65)
                    for space,target in [('latent',delta),('physical',physical)]:
                        yf=operator@target;bucket=(native_stats if split=='train' else native_validation)[native_name][space]
                        bucket['gram']+=flat.T@flat;bucket['cross']+=flat.T@yf.reshape(-1,yf.shape[-1]);bucket['energy']+=float(np.square(yf).sum())
                if split=='train':
                    train_energy+=np.square(physical).sum(axis=(0,1));train_elements+=len(starts)*roads
                else:
                    valid=starts+2<len(arrays['flow']);following=np.zeros_like(arrays['flow'][starts+1])
                    following[valid]=arrays['flow'][starts[valid]+2]
                    reference_predictions={'tau0':x@candidates['soft_tau0__filtered']['latent'],
                        'tau10':(native_tau10@x)@native_heads['tau10']['latent'],
                        'hard':(native_hard@x)@native_heads['hard']['latent']}
                    for native_name,prediction in reference_predictions.items():
                        row=dict(candidate=native_name,demand_group=entry['demand_group'],
                            full_sse=float(np.square(delta-prediction).sum()),total_delta_energy=float(np.square(delta).sum()))
                        row.update(physical_sums(model,z+torch.as_tensor(prediction,dtype=z.dtype),future,following,valid,capacities,limits))
                        native_reference_records.append(row)
                    tau0_predictions={}
                    for name,candidate in candidates.items():
                        gains=candidate['gains']
                        if candidate['filter']=='hard':
                            prediction=(native_hard@x)@candidate['latent'];ps=project(basis,prediction)
                            phy_prediction=(native_hard@x)@candidate['physical'];phy_ps=project(basis,phy_prediction)
                            filtered_delta=native_hard@delta
                            full=float(np.square(delta-prediction).sum())
                            metric=dict(full_sse=full,total_delta_energy=float(np.square(delta).sum()),
                                filtered_target_sse=float(np.square(filtered_delta-prediction).sum()),
                                filtered_delta_energy=float(np.square(filtered_delta).sum()),
                                filtered_error_sse=float(np.square(native_hard@(delta-prediction)).sum()),
                                prediction_energy=float(np.square(prediction).sum()))
                            band_sse=np.array([np.square((ys['latent']-ps)[:,bins==i]).sum() for i in range(len(BAND_EDGES)-1)])
                            np.testing.assert_allclose(band_sse.sum(),full,rtol=2e-10)
                        else:
                            ps=np.zeros_like(ys['latent']) if candidate['filter']=='zero' else gains[None,:,None]*(xs@candidate['latent'])
                            phy_ps=np.zeros_like(ys['physical']) if candidate['filter']=='zero' else gains[None,:,None]*(xs@candidate['physical'])
                            metric,band_sse=spectral_sums(ys['latent'],ps,gains,bins)
                            if candidate['filter']=='zero':prediction=np.zeros_like(delta)
                            elif candidate['tau']==0:prediction=x@candidate['latent']
                            else:prediction=np.einsum('nr,trc->tnc',basis,ps,optimize=True)
                            if candidate['tau']==0:
                                if tau0_predictions:
                                    np.testing.assert_array_equal(prediction,tau0_predictions['latent'])
                                    np.testing.assert_array_equal(phy_ps,tau0_predictions['physical_spectrum'])
                                tau0_predictions=dict(latent=prediction,physical_spectrum=phy_ps)
                        phy_error=ys['physical']-phy_ps
                        physical_sse=np.square(phy_error).sum(axis=(0,1));physical_delta_energy=np.square(ys['physical']).sum(axis=(0,1))
                        row={key:candidate[key] for key in ('filter','tau','supervision')}
                        row.update(candidate=name,demand_group=entry['demand_group'],windows=len(starts),**metric,
                            direct_count_sse=float(physical_sse[0]),direct_count_energy=float(physical_delta_energy[0]),
                            direct_speed_sse=float(physical_sse[1]),direct_speed_energy=float(physical_delta_energy[1]),
                            direct_joint_sse=float(np.sum(physical_sse/rms**2)),direct_joint_energy=float(np.sum(physical_delta_energy/rms**2)))
                        row.update(physical_sums(model,z+torch.as_tensor(prediction,dtype=z.dtype),future,following,valid,capacities,limits))
                        records.append(row)
                        for band,value in enumerate(band_sse):
                            bands.append(dict(candidate=name,demand_group=entry['demand_group'],band=band,
                                modes=int((bins==band).sum()),prediction_sse=float(value),target_energy=float(np.square(ys['latent'][:,bins==band]).sum())))
                        if name=='soft_tau10__filtered':
                            difference=np.linalg.norm(prediction-reference_predictions['tau10'])/max(np.linalg.norm(reference_predictions['tau10']),1e-30)
                            proof['spectral_vs_native_tau10_prediction_relative_l2_max']=max(proof.get('spectral_vs_native_tau10_prediction_relative_l2_max',0.),float(difference))
                audit.append(dict(split=split,cache_path=entry['cache_path'],content_hash=entry['cache_content_hash'],
                    demand_group=entry['demand_group'],starts=starts.tolist(),current_raw_flow_exact_match=True))
                print(json.dumps(dict(stage=split,group=ordinal+1,total=len(selection[split]),elapsed_seconds=time.time()-started)),flush=True)
            if split=='train':
                rms=np.sqrt(train_energy/train_elements);native_heads={}
                for native_name,spaces in native_stats.items():
                    native_heads[native_name]={}
                    for space,aggregate in spaces.items():
                        head,fit=solve_ridge(aggregate['gram'],aggregate['cross'],aggregate['energy'])
                        native_heads[native_name][space]=head;fits[f'native_{native_name}__{space}']=fit
                for tau in TAUS:
                    gains=1/(1+tau*eigenvalues)
                    shared={}
                    for supervision in ('filtered','full'):
                        candidate=dict(filter='soft',tau=tau,supervision=supervision,gains=gains)
                        for space in ('latent','physical'):
                            if tau==0 and supervision=='full':
                                head,fit,aggregate=shared[space]
                            else:head,fit,aggregate=solve_weighted(train[space],gains,supervision)
                            shared[space]=(head,fit,aggregate);candidate[space]=head
                            name=f'soft_tau{tau:g}__{supervision}'
                            fits[name+'__'+space]=fit
                            for key,value in aggregate.items():saved_stats['train__'+name+'__'+space+'__'+key]=value
                        candidates[name]=candidate
                candidates['hard_r16__filtered']=dict(filter='hard',tau=None,supervision='filtered',gains=None,**native_heads['hard'])
                candidates['zero']=dict(filter='zero',tau=None,supervision='none',gains=np.ones(roads),latent=None,physical=None)
                for space in ('latent','physical'):
                    np.testing.assert_array_equal(candidates['soft_tau0__filtered'][space],candidates['soft_tau0__full'][space])
                with np.load(references['identity_coefficients']) as prior_asset:
                    old_identity=prior_asset['raw_next__identity_tau0__output_only']
                    np.testing.assert_allclose(candidates['soft_tau0__filtered']['latent'],old_identity,rtol=1e-9,atol=1e-10)
                    proof['tau0_prior_coefficients_max_absolute_difference']=float(np.max(np.abs(candidates['soft_tau0__filtered']['latent']-old_identity)))
                with np.load(references['tau10_coefficients']) as prior_asset:
                    old_tau10=prior_asset['raw_next__soft_tau10__output_only'];old_hard=prior_asset['raw_next__hard_r16__output_only']
                    np.testing.assert_allclose(native_heads['tau10']['latent'],old_tau10,rtol=1e-10,atol=1e-10)
                    np.testing.assert_allclose(native_heads['hard']['latent'],old_hard,rtol=1e-10,atol=1e-10)
                    proof['native_tau10_previous_coefficients_max_absolute_difference']=float(np.max(np.abs(native_heads['tau10']['latent']-old_tau10)))
                atomic_json(args.output/'fit_diagnostics.json',fits)
                np.savez_compressed(args.output/'heads.npz',**{name+'__'+space:candidate[space] for name,candidate in candidates.items()
                    if candidate['filter']!='zero' for space in ('latent','physical')})
    # Gate all summaries on reproducible historical native predictions first.
    reference_proof=[]
    for native_name,reference_dir,filter_name in [('tau0',args.identity,'identity_tau0'),('tau10',args.reference,'soft_tau10'),('hard',args.reference,'hard_r16')]:
        old_summary=json.loads((reference_dir/'summary.json').read_text())
        expected=next(r for r in old_summary['metrics'] if r['target']=='raw_next' and r['filter']==filter_name and r['arm']=='output_only')
        selected=[r for r in native_reference_records if r['candidate']==native_name]
        for metric,numerator,denominator in [m for m in METRICS if m[0]=='full_normalized_sse' or '_mae' in m[0]]:
            value=sum(r[numerator] for r in selected)/sum(r[denominator] for r in selected)
            tolerance=1e-6 if metric=='full_normalized_sse' else 1e-5
            np.testing.assert_allclose(value,expected[metric],rtol=0,atol=tolerance)
            reference_proof.append(dict(reference=native_name,metric=metric,current=value,previous=expected[metric],absolute_difference=abs(value-expected[metric]),tolerance=tolerance))
    independent=[]
    for name,candidate in candidates.items():
        if candidate['filter']=='zero':continue
        for space in ('latent','physical'):
            if candidate['filter']=='hard':
                # Hard is not assumed to equal an arbitrary exact rank-16 eigenspace.
                prediction_rows=[r for r in records if r['candidate']==name]
                head=candidate[space];agg=native_validation['hard'][space]
                direct=sum(r['filtered_target_sse'] if space=='latent' else 0. for r in prediction_rows)
                sse=sufficient_sse(head,agg)
                if space=='latent':np.testing.assert_allclose(sse,direct,rtol=1e-10,atol=1e-7)
                independent.append(dict(candidate=name,space=space,objective='filtered_target',sse=sse,direct_sse=direct if space=='latent' else None))
                for key,value in agg.items():saved_stats['validation__'+name+'__'+space+'__'+key]=value
                continue
            agg=weighted_moments(val[space],candidate['gains'],'full')
            sse=sufficient_sse(candidate[space],agg)
            direct=sum(r['full_sse'] if space=='latent' else r['direct_count_sse']+r['direct_speed_sse'] for r in records if r['candidate']==name)
            np.testing.assert_allclose(sse,direct,rtol=1e-10,atol=1e-7)
            independent.append(dict(candidate=name,space=space,objective='full_target',sse=sse,direct_sse=direct))
            for key,value in agg.items():saved_stats['validation__'+name+'__'+space+'__'+key]=value
    proof.update(native_reference_gate_passed=True,tau0_full_filtered_bit_identical=True,parseval_all_bands_verified=True,
        physical_increment_train_rms=rms.tolist(),independent_sse_verified=True)
    torch.testing.assert_close(prior.basis,original_basis,rtol=0,atol=0)
    assert prior.noise_scale==original_noise
    for name,path in references.items():
        if sha256(path)!=hashes[name]:raise ValueError('reference changed: '+name)
    for entry in audit:
        if archive_content_hash(Path(entry['cache_path']))!=entry['content_hash']:raise ValueError('data changed')
    imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
        if name.startswith(('tools.','urban_dreamer.')) and getattr(module,'__file__',None)}
    if any(not Path(path).is_relative_to(FROZEN_ROOT) for path in imports.values()):raise ValueError('live import contaminated frozen execution')
    verify_source();atomic_json(args.output/'import_audit.json',imports)
    rows=summarize(records,args.seed);paired=contrasts(records,args.seed);band_rows=band_summary(bands,args.seed)
    write_csv(args.output/'demand_group_metrics.csv',records);write_csv(args.output/'frequency_group_metrics.csv',bands)
    write_csv(args.output/'metric_summary.csv',rows);write_csv(args.output/'frequency_summary.csv',band_rows)
    atomic_json(args.output/'paired_contrasts.json',paired);atomic_json(args.output/'native_reference_gate.json',reference_proof)
    atomic_json(args.output/'independent_sse_verification.json',independent);atomic_json(args.output/'data_audit.json',audit)
    np.savez_compressed(args.output/'aggregate_sufficient_statistics.npz',**saved_stats)
    pareto_metrics=['full_normalized_sse','count_mae','count_mae_occupied','speed_mae_occupied']
    plot(rows,args.output)
    atomic_json(args.output/'summary.json',dict(status='complete',protocol=protocol,proof=proof,metrics=rows,
        paired_contrasts=paired,frequency_metrics=band_rows,
        point_estimate_pareto=dict(metrics=pareto_metrics,candidates=pareto(rows,pareto_metrics),inferential_claim=False),
        elapsed_seconds=time.time()-started,source_unchanged=True,checkpoint_unchanged=True,data_unchanged=True))
    print(json.dumps(dict(stage='complete',elapsed_seconds=time.time()-started)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--basis',type=Path,required=True)
    parser.add_argument('--identity',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261014)
    parser.add_argument('--confirm-from',type=Path)
    parser.add_argument('--validation-manifest',type=Path)
    parser.add_argument('--confirmation-groups',type=int,default=64)
    parser.add_argument('--candidates',nargs='+')
    args=parser.parse_args()
    if args.confirm_from:
        if args.validation_manifest is None or args.candidates is None:
            parser.error('confirmation requires --validation-manifest and --candidates')
        confirm(args)
    else:run(args)
