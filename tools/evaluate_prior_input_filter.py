"""Frozen-body, state-only probe of output versus input-and-output filtering.

CPU only. Fits each 65x64 head by F64 ridge using training windows only.
This probes an old action-conditioned/rank16 body with its condition term removed;
it does not establish performance of a newly trained background dynamics prior.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = Path(os.environ.get('PRIOR_INPUT_FROZEN_ROOT', ROOT)).resolve()
sys.path.insert(0, str(FROZEN_ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from tools.analyze_traffic_spectrum import bootstrap_ratio, normalized_laplacian, select_starts
from tools.component_experiment_data import atomic_json, dataset, load_frozen, sha256
from tools.evaluate_pretrained_prior_ranks import CHECKPOINT_SHA, physical_sums, solve_ridge, write_csv
from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.learning.offline_data import archive_content_hash
from urban_dreamer.learning.pretrain_cache import compact_transition_targets
from urban_dreamer.models.world_model import ConditionEncoding, LowFrequencyGraphPrior


def state_hidden(prior, z):
    """Native hidden front with the action/condition branch omitted."""
    graph = prior.adjacency.to(z)
    downstream = graph / graph.sum(-1, keepdim=True).clamp_min(1e-12)
    upstream = graph.T / graph.T.sum(-1, keepdim=True).clamp_min(1e-12)
    values = prior.project(z) if prior.filter_input else z
    return F.silu(prior.state_proj(values) + downstream @ prior.downstream_proj(values)
                  + upstream @ prior.upstream_proj(values))


def filtered_design(hidden, delta, operator):
    hidden, delta, operator = [np.asarray(v, dtype=np.float64) for v in (hidden, delta, operator)]
    if hidden.ndim != 3 or delta.shape != hidden.shape or operator.shape != (hidden.shape[1],)*2:
        raise ValueError('matching windows/roads/channels and square filter required')
    if not all(np.isfinite(v).all() for v in (hidden, delta, operator)):
        raise ValueError('finite designs required')
    augmented = np.concatenate((hidden, np.ones((*hidden.shape[:-1], 1))), axis=-1)
    return operator @ augmented, operator @ delta


def sums(delta, prediction, filtered_target, operator):
    error = delta - prediction
    return dict(total_delta_energy=float(np.square(delta).sum()),
        filtered_delta_energy=float(np.square(filtered_target).sum()),
        full_sse=float(np.square(error).sum()),
        filtered_target_sse=float(np.square(filtered_target-prediction).sum()),
        filtered_error_sse=float(np.square(operator@error).sum()),
        prediction_energy=float(np.square(prediction).sum()), dot_product=float(np.sum(delta*prediction)))


def summarize(records, seed=20261013, repetitions=2000):
    identity = ('target', 'filter', 'arm', 'demand_group')
    groups = {}
    for record in records:
        key = tuple(record[k] for k in identity)
        row = groups.setdefault(key, {k: record[k] for k in identity})
        for field, value in record.items():
            if field not in (*identity, 'episode'):
                row[field] = row.get(field, 0.) + value
    groups = list(groups.values())
    rows = []
    for target, family, arm in sorted({tuple(r[k] for k in identity[:3]) for r in groups}):
        selected = [r for r in groups if (r['target'],r['filter'],r['arm']) == (target,family,arm)]
        row = dict(target=target, filter=family, arm=arm, groups=len(selected), windows=int(sum(r['windows'] for r in selected)))
        for name, numerator, denominator in [
            ('full_normalized_sse','full_sse','total_delta_energy'),
            ('filtered_normalized_sse','filtered_target_sse','filtered_delta_energy'),
            ('filtered_error_normalized_sse','filtered_error_sse','filtered_delta_energy'),
            ('prediction_norm_ratio_squared','prediction_energy','total_delta_energy'),
            *[(metric+'_mae'+suffix, metric+suffix+'_absolute_error', metric+suffix+'_elements')
              for metric in ('count','speed') for suffix in ('','_occupied')]]:
            value, lower, upper = bootstrap_ratio([r[numerator] for r in selected], [r[denominator] for r in selected], repetitions=repetitions, seed=seed)
            row.update({name:value,name+'_ci95_low':lower,name+'_ci95_high':upper})
        product = sum(r['prediction_energy'] for r in selected)*sum(r['total_delta_energy'] for r in selected)
        row['alignment'] = sum(r['dot_product'] for r in selected)/np.sqrt(product) if product>0 else None
        row['prediction_norm_ratio'] = np.sqrt(row['prediction_norm_ratio_squared']) if row['prediction_norm_ratio_squared'] is not None else None
        row['improvement_vs_zero_prior'] = 1-row['full_normalized_sse'] if row['full_normalized_sse'] is not None else None
        rows.append(row)
    return groups, rows


def paired_contrasts(groups, seed=20261013, repetitions=2000, *, include_filtered=True):
    results = []
    for target, family in sorted({(r['target'],r['filter']) for r in groups}):
        selected = {arm:{r['demand_group']:r for r in groups if (r['target'],r['filter'],r['arm']) == (target,family,arm)}
                    for arm in ('output_only','input_output')}
        keys = sorted(selected['output_only'])
        if keys != sorted(selected['input_output']):
            raise ValueError('paired groups must match')
        metrics=[
            ('full_normalized_sse','full_sse','total_delta_energy'),
            *([('filtered_normalized_sse','filtered_target_sse','filtered_delta_energy')] if include_filtered else []),
            *[(m+'_mae'+s,m+s+'_absolute_error',m+s+'_elements') for m in ('count','speed') for s in ('','_occupied')]]
        for metric,numerator,denominator in metrics:
            differences, denominators = [], []
            for key in keys:
                a,b = selected['output_only'][key],selected['input_output'][key]
                if not np.isclose(a[denominator],b[denominator],rtol=1e-12,atol=1e-12):
                    raise ValueError('paired targets/denominators differ')
                differences.append(b[numerator]-a[numerator]); denominators.append(a[denominator])
            value,low,high = bootstrap_ratio(differences,denominators,repetitions=repetitions,seed=seed)
            results.append(dict(target=target,filter=family,metric=metric,input_output_minus_output_only=value,
                                ci95_low=low,ci95_high=high,groups=len(keys),lower_is_better=True))
    return results


def identity_equivalence(output_prior, input_prior, z):
    """Prove that filtering either end at tau zero leaves the front identical."""
    torch.testing.assert_close(output_prior.project(z),z,rtol=0,atol=0)
    a,b=state_hidden(output_prior,z),state_hidden(input_prior,z)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    return a


def identity_reference_contrasts(groups, reference_groups, seed=20261013):
    """Compare common full-target losses; filtered denominators differ by tau."""
    pairs=[]
    for reference in ('soft_tau10_output_only','zero_prior'):
        for row in groups:
            if row['arm']=='output_only':
                pairs.append(dict(row,filter=reference,arm='input_output'))
        for row in reference_groups:
            if row['filter']=='soft_tau10' and row['arm']==('output_only' if reference!='zero_prior' else 'zero_prior'):
                pairs.append(dict(row,filter=reference,arm='output_only'))
    result=paired_contrasts(pairs,seed,include_filtered=False)
    for row in result:
        row['reference']=row.pop('filter')
        row['identity_minus_reference']=row.pop('input_output_minus_output_only')
    return result


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    metrics=('full_normalized_sse','filtered_normalized_sse','count_mae','speed_mae_occupied')
    chosen=[r for r in rows if r['target']=='raw_next']
    for axis,metric in zip(axes.flat,metrics):
        labels=[]
        for i,row in enumerate(chosen):
            value=row[metric]
            axis.bar(i,value,color={'output_only':'#386cb0','input_output':'#f0027f','zero_prior':'#888888'}[row['arm']],alpha=.8)
            axis.errorbar(i,value,yerr=[[value-row[metric+'_ci95_low']],[row[metric+'_ci95_high']-value]],color='black',capsize=3)
            labels.append(row['filter']+'\n'+row['arm'])
        axis.set_xticks(range(len(labels)),labels,fontsize=8);axis.set_title(metric);axis.grid(axis='y',alpha=.2)
    fig.suptitle('State-only frozen-body head probe: raw traffic endpoint; group bootstrap 95% CI')
    for extension in ('png','pdf'):
        fig.savefig(output/('prior_input_output.'+extension),dpi=180)
    plt.close(fig)


def run(args):
    started=time.time()
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(args.seed)
    if (args.output/'summary.json').exists():
        raise ValueError('preserve completed results')
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=args.experiment/'snapshot/checkpoints/latest.pt'
    if sha256(checkpoint)!=CHECKPOINT_SHA:
        raise ValueError('fixed completed checkpoint required')
    selection=json.loads(args.selection.read_text())
    if {r['demand_group'] for r in selection['train']} & {r['demand_group'] for r in selection['validation']}:
        raise ValueError('train/validation groups overlap')
    source_manifest=json.loads((FROZEN_ROOT.parent/'source_manifest.json').read_text())
    def verify_source():
        for relative,digest in source_manifest.items():
            if sha256(FROZEN_ROOT/relative)!=digest:
                raise ValueError('frozen source changed: '+relative)
    verify_source()
    identity_control=args.identity_control
    strength=0. if identity_control else args.prior_filter_strength
    soft_family='identity_tau0' if identity_control else f'soft_tau{strength:g}'
    families=[(soft_family,'soft')] + ([] if identity_control else [('hard_r16','hard')])
    protocol=dict(status='prespecified_before_results',seed=args.seed,device='cpu',threads=2,
        checkpoint_sha256=CHECKPOINT_SHA,selection_sha256=sha256(args.selection),windows_per_episode=12,
        train_groups=len(selection['train']),validation_groups=len(selection['validation']),
        filters={soft_family:f'(I+{strength:g}*Lsym)^-1; full rank, no truncation','hard_r16':'checkpoint own stored U16 U16.T'},
        arms=['output_only','input_output','zero_prior'],output_parameters=4160,output_shape=[65,64],
        target_families=['hybrid_Ct','raw_next'],fit_target='F(delta); same target within filter families',
        design='F[H(z),1] versus F[H(Fz),1]; both native state-only fronts',
        ridge='F64 closed form, lambda=1e-6*trace(train Gram)/65; bias penalized; no validation tuning',
        physical_target='RAW next count; speed average RAW next and following; unavailable following omitted',
        full_sse='||delta-prediction||^2',filtered_sse='||F(delta)-prediction||^2',
        filtered_error_sse='||F(delta-prediction)||^2; separately reported',
        bootstrap='2000 paired demand-group resamples; ratios of aggregate sufficient statistics',
        prior_noise_changed=False,noise_used=False,full_world_model_training=False,final_test_used=False,
        caveats=['Development validation was previously inspected.',
          'The encoder and body were trained with an action-conditioned hard rank16 prior. Condition term is removed in this probe.',
          'Only the output head is fitted; this is not a fully trained state-only background prior.',
          'One-step decoded point predictions have no FM correction.'])
    if identity_control:
        protocol.update(status='prespecified_extension_before_identity_scores',
            study_role='exploratory_identity_filter_control',
            motivation='Posthoc extension motivated by absence of raw low-frequency dominance; preserve original prespecified ablation.',
            filters={'identity_tau0':'soft tau=0 => F=I'},arms=['output_only','zero_prior'],
            input_output_equivalence='Bit-identical fronts and designs checked on every selected window; one head per target suffices.',
            comparison_primary='Common FULL delta SSE vs original tau10 output-only and persistence; filtered denominators not compared across tau.',
            reference_groups_sha256=sha256(args.reference/'demand_group_metrics.csv'))
    protocol['prior_filter_strength']=strength
    atomic_json(args.output/'protocol.json',protocol);atomic_json(args.output/'selection.json',selection)
    config,prepared,topology,model,_=load_frozen(args.experiment,device='cpu')
    model.eval().requires_grad_(False)
    original=model.dynamics.graph_prior
    capacities=np.maximum(np.asarray(topology.network.capacities,dtype=np.float64),1.)
    limits=np.asarray(topology.network.speed_limits_mps,dtype=np.float64)
    adjacency=original.adjacency
    priors={};operators={};proof={}
    identity_input_prior=None
    for family,kind in families:
        for input_filter in (False,True):
            prior=LowFrequencyGraphPrior(adjacency,rank=16,noise_scale=original.noise_scale,
                prior_filter=kind,prior_filter_strength=strength,prior_action_conditioned=False,prior_filter_input=input_filter)
            state=prior.state_dict()
            for key,value in original.state_dict().items():
                if key in state and state[key].shape==value.shape:
                    state[key]=value.clone()
            prior.load_state_dict(state,strict=True);prior.eval().requires_grad_(False)
            arm='input_output' if input_filter else 'output_only'
            if identity_control and input_filter:
                identity_input_prior=prior
            else:
                priors[(family,arm)]=prior
            operators[family]=(prior.soft_filter_operator if kind=='soft' else prior.basis@prior.basis.T).double().numpy()
            torch.testing.assert_close(prior.basis,original.basis,rtol=0,atol=0)
            for name,param in prior.named_parameters():
                torch.testing.assert_close(param,dict(original.named_parameters())[name],rtol=0,atol=0)
    with np.load(args.basis) as basis_asset:
        eigenvalues=basis_asset['eigenvalues'];basis=basis_asset['basis']
    soft=operators[soft_family]; gains=1/(1+strength*eigenvalues)
    proof.update(soft_full_rank=bool(np.linalg.matrix_rank(soft)==len(soft)),soft_min_gain=float(gains.min()),
        soft_max_gain_error=float(np.max(np.abs(soft@basis-basis*gains))),
        hard_basis_matches_checkpoint=True,body_parameters_match_checkpoint=True,condition_branch_omitted=True,
        soft_noise_basis_identical_to_hard=True,noise_scale=original.noise_scale,
        projector_sha256=sha256(FROZEN_ROOT/'urban_dreamer/models/world_model.py'),graph_sha256=sha256(args.basis),
        hard_basis_orthogonality_error=float(np.max(np.abs(original.basis.double().numpy().T@original.basis.double().numpy()-np.eye(16)))))
    heads={};stats={};validation_stats={};records=[];data_audit=[]
    for target in ('hybrid_Ct','raw_next'):
        for key in priors:
            stats[(target,*key)]=dict(gram=np.zeros((65,65)),cross=np.zeros((65,64)),energy=0.)
            validation_stats[(target,*key)]=dict(gram=np.zeros((65,65)),cross=np.zeros((65,64)),energy=0.)
    with torch.inference_mode():
        for split in ('train','validation'):
            source=dataset(args.experiment,config,prepared,topology,split=split,horizon=1)
            indices={e['cache_path']:i for i,e in enumerate(source.manifest['entries'])}
            for ordinal,entry in enumerate(selection[split]):
                index=indices[entry['cache_path']];arrays=source._episode(index)
                starts=select_starts(arrays['flow'],12)
                batch=source.batch_from_starts(index,starts.tolist(),materialization_horizon=1,fm_only=True)
                scoped=prepare_transition_targets(batch,capacities=capacities)
                cached_target,_=compact_transition_targets(arrays,starts,1,capacities)
                np.testing.assert_allclose(scoped.transition_target_flow[:,0].numpy(),cached_target[:,0],rtol=1e-6,atol=1e-6)
                raw=torch.as_tensor(arrays['flow'][starts+1],dtype=batch.flow.dtype)
                z=model.encode(batch.flow[:,0]);deltas={name:(model.encode(flow)-z).double().numpy()
                    for name,flow in [('hybrid_Ct',scoped.transition_target_flow[:,0]),('raw_next',raw)]}
                valid=starts+2<len(arrays['flow']);following=np.zeros_like(arrays['flow'][starts+1])
                following[valid]=arrays['flow'][starts[valid]+2]
                hidden={key:state_hidden(prior,z[:,None])[:,0] for key,prior in priors.items()}
                if identity_control:
                    identical=identity_equivalence(priors[(soft_family,'output_only')],identity_input_prior,z[:,None])[:,0]
                    torch.testing.assert_close(identical,hidden[(soft_family,'output_only')],rtol=0,atol=0)
                    proof['tau0_input_output_bit_identical']=True
                    proof['tau0_equivalence_windows']=proof.get('tau0_equivalence_windows',0)+len(z)
                if not proof.get('native_forward_verified'):
                    condition=ConditionEncoding(torch.randn(len(z),1,len(adjacency),256))
                    alternate=ConditionEncoding(condition.edge_tokens*99)
                    for key,prior in priors.items():
                        native=prior(z[:,None],condition)[:,0]
                        torch.testing.assert_close(native,prior.project(prior.output(hidden[key])),rtol=2e-6,atol=2e-6)
                        torch.testing.assert_close(native,prior(z[:,None],alternate)[:,0],rtol=0,atol=0)
                        coefficients=np.concatenate((prior.output.weight.numpy().T,prior.output.bias.numpy()[None]),axis=0)
                        x,_=filtered_design(hidden[key].numpy(),deltas['raw_next'],operators[key[0]])
                        np.testing.assert_allclose(native.numpy(),x@coefficients,rtol=3e-5,atol=3e-6)
                    proof['native_forward_verified']=True
                for target,delta in deltas.items():
                    for (family,arm),h in hidden.items():
                        key=(target,family,arm);operator=operators[family]
                        x,y=filtered_design(h.numpy(),delta,operator);flat=x.reshape(-1,65);fy=y.reshape(-1,64)
                        bucket=stats[key] if split=='train' else validation_stats[key]
                        bucket['gram']+=flat.T@flat;bucket['cross']+=flat.T@fy;bucket['energy']+=float(np.square(y).sum())
                        if split=='validation':
                            prediction=x@heads[key]
                            row=dict(target=target,filter=family,arm=arm,demand_group=entry['demand_group'],episode=entry['episode'],windows=len(z))
                            row.update(sums(delta,prediction,y,operator))
                            row.update(physical_sums(model,z+torch.as_tensor(prediction,dtype=z.dtype),raw,following,valid,capacities,limits));records.append(row)
                            if arm=='output_only':
                                zero=dict(row,arm='zero_prior');zero.update(sums(delta,np.zeros_like(delta),y,operator))
                                zero.update(physical_sums(model,z,raw,following,valid,capacities,limits));records.append(zero)
                data_audit.append(dict(split=split,cache_path=entry['cache_path'],expected_cache_content_hash=entry['cache_content_hash'],
                    demand_group=entry['demand_group'],starts=starts.tolist(),valid_following_speed=valid.tolist(),hybrid_cache_label_match=True))
                print(json.dumps(dict(stage=split,group=ordinal+1,total=len(selection[split]),elapsed_seconds=time.time()-started)),flush=True)
            if split=='train':
                fits={}
                for key,bucket in stats.items():
                    heads[key],diagnostic=solve_ridge(bucket['gram'],bucket['cross'],bucket['energy']);fits['__'.join(key)]=diagnostic
                atomic_json(args.output/'fit_diagnostics.json',fits)
                np.savez_compressed(args.output/'output_head_coefficients.npz',**{'__'.join(k):v for k,v in heads.items()})
    independent={}
    for key,bucket in validation_stats.items():
        head=heads[key]
        sse=float(bucket['energy']-2*np.sum(head*bucket['cross'])+np.sum(head*(bucket['gram']@head)))
        direct=sum(r['filtered_target_sse'] for r in records if (r['target'],r['filter'],r['arm'])==key)
        np.testing.assert_allclose(sse,direct,rtol=1e-10,atol=1e-8)
        independent['__'.join(key)]=dict(sufficient_statistic_filtered_sse=sse,direct_filtered_sse=direct)
    groups,rows=summarize(records,args.seed)
    if identity_control:
        import csv
        reference_groups=[]
        with (args.reference/'demand_group_metrics.csv').open() as stream:
            for row in csv.DictReader(stream):
                reference_groups.append({key:value if key in ('target','filter','arm','demand_group') else float(value) for key,value in row.items()})
        contrasts=identity_reference_contrasts(groups,reference_groups,args.seed)
    else:
        contrasts=paired_contrasts(groups,args.seed)
    write_csv(args.output/'episode_metrics.csv',records);write_csv(args.output/'demand_group_metrics.csv',groups)
    write_csv(args.output/'metric_summary.csv',rows);atomic_json(args.output/'paired_contrasts.json',contrasts)
    atomic_json(args.output/'data_audit.json',data_audit);plot(rows,args.output)
    for label,buckets in [('training',stats),('validation',validation_stats)]:
        np.savez_compressed(args.output/(label+'_sufficient_statistics.npz'),**{'__'.join((*key,kind)):value for key,bucket in buckets.items() for kind,value in bucket.items()})
    verify_source()
    if sha256(checkpoint)!=CHECKPOINT_SHA or sha256(args.selection)!=protocol['selection_sha256']:
        raise ValueError('source artifacts changed')
    if identity_control and sha256(args.reference/'demand_group_metrics.csv')!=protocol['reference_groups_sha256']:
        raise ValueError('original ablation reference changed')
    for entry in data_audit:
        if archive_content_hash(Path(entry['cache_path']))!=entry['expected_cache_content_hash']:
            raise ValueError('selected data changed')
    imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
        if name.startswith(('urban_dreamer.','tools.')) and getattr(module,'__file__',None)}
    if any(not Path(path).is_relative_to(FROZEN_ROOT) for path in imports.values()):
        raise ValueError('live source contaminated frozen execution')
    atomic_json(args.output/'import_audit.json',imports)
    atomic_json(args.output/'summary.json',dict(status='complete',protocol=protocol,proof=proof,
        metrics=rows,paired_contrasts=contrasts,independent_sse_verification=independent,
        elapsed_seconds=time.time()-started,source_unchanged=True,checkpoint_unchanged=True,data_unchanged=True))
    print(json.dumps(dict(stage='complete',elapsed_seconds=time.time()-started)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=Path('outputs/component_experiments_20261011'))
    parser.add_argument('--selection',type=Path,default=Path('outputs/pretrained_prior_rank_20261013/v1/selection.json'))
    parser.add_argument('--basis',type=Path,default=Path('outputs/pretrained_prior_rank_20261013/v1/exact_basis.npz'))
    parser.add_argument('--output',type=Path,default=Path('outputs/traffic_increment_spectrum_20261013/input_output_v1'))
    parser.add_argument('--seed',type=int,default=20261013)
    parser.add_argument('--prior-filter-strength',type=float,default=10.)
    parser.add_argument('--identity-control',action='store_true')
    parser.add_argument('--reference',type=Path,default=Path('outputs/traffic_increment_spectrum_20261013/input_output_v1'))
    run(parser.parse_args())
