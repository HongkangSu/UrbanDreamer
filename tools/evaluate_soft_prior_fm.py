"""Complete frozen FM forecasts under explicit prior-mean interventions.

These are inference interventions on a completed hard/action-prior FM, not
fully trained soft/state-only world models. Noise coordinates never change.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

ROOT=Path(__file__).resolve().parents[1]
FROZEN_ROOT=Path(os.environ.get('SOFT_PRIOR_FM_FROZEN_ROOT',ROOT)).resolve()
sys.path.insert(0,str(FROZEN_ROOT))
import numpy as np
import torch
from torch import nn

from tools.component_experiment_data import atomic_json,load_frozen,sha256
from tools.evaluate_component_world_models import endpoint_target,stable_seed
from tools.evaluate_pretrained_distribution import trajectory_bank,CHECKPOINT_SHA256
from tools.evaluate_pretrained_condition_scope import score
from tools.summarize_component_predictions import demand_bootstrap
from tools.component_prediction_metrics import summarize_state_records
from urban_dreamer.learning.cohort_targets import prepare_cohort_batch,prepare_transition_targets
from urban_dreamer.learning.offline_data import archive_content_hash,_identity
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.models.world_model import LowFrequencyGraphPrior,ConditionEncoding
from tools.world_model_evaluation import _repeat_events

ARMS=('native','zero_mean','action_soft0.1','action_soft10','state_soft0','state_soft0.1','state_soft10')
METRICS=tuple(f'{feature}_{metric}_{scope}' for feature in ('count','speed')
    for metric in ('mae','mse','rmse','crps','sample_variance','coverage90','width90') for scope in ('all','occupied'))+(
    'energy_score','energy_score_u','total_count_bias','total_count_absolute_error',
    'total_count_sample_variance','total_count_over_cohort_fraction')


class MeanOverridePrior(LowFrequencyGraphPrior):
    """Copy native buffers/modules; only mean operator/conditions/head differ."""
    def __init__(self,original,*,operator=None,coefficients=None,state_only=False,zero=False):
        nn.Module.__init__(self)
        for name in ('rank','noise_scale','prior_filter','prior_filter_strength','prior_action_conditioned','filter_input'):
            setattr(self,name,getattr(original,name))
        self.filter_input=False
        self.prior_action_conditioned=not state_only
        self.zero_mean=bool(zero)
        for name,module in original._modules.items():self.add_module(name,copy.deepcopy(module))
        for name,value in original._buffers.items():
            self.register_buffer(name,None if value is None else value.detach().clone(),
                persistent=name not in original._non_persistent_buffers_set)
        value=None if operator is None else torch.as_tensor(operator,device=original.basis.device,dtype=original.basis.dtype).clone()
        if value is not None and (value.shape!=(len(original.basis),)*2 or not bool(torch.isfinite(value).all())):
            raise ValueError('finite matching square mean operator required')
        self.register_buffer('mean_operator',value,persistent=False)
        if coefficients is not None:
            c=torch.as_tensor(coefficients,device=self.output.weight.device,dtype=self.output.weight.dtype)
            if c.shape!=(self.output.in_features+1,self.output.out_features) or not bool(torch.isfinite(c).all()):
                raise ValueError('finite augmented output coefficients required')
            with torch.no_grad():self.output.weight.copy_(c[:-1].T);self.output.bias.copy_(c[-1])
        self.eval().requires_grad_(False)

    def project(self,values):
        return super().project(values) if self.mean_operator is None else self.mean_operator.to(values)@values

    def forward(self,z,condition):
        return torch.zeros_like(z) if self.zero_mean else super().forward(z,condition)

    @property
    def projection_rank(self):
        return 0 if self.zero_mean else self.rank if self.mean_operator is None else len(self.basis)


def digest(model,exclude_prior=False):
    result=hashlib.sha256()
    for name,value in model.state_dict().items():
        if exclude_prior and name.startswith('dynamics.graph_prior.'):continue
        result.update(name.encode());result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def summaries(rows,seed):
    result=[];paired=[]
    for steps,horizon,arm in sorted({(r['steps'],r['horizon'],r['arm']) for r in rows}):
        selected=[r for r in rows if (r['steps'],r['horizon'],r['arm'])==(steps,horizon,arm)]
        descriptive=summarize_state_records(selected)
        row=dict(steps=steps,horizon=horizon,arm=arm,states=len(selected),
            metrics={metric:demand_bootstrap(selected,metric,repetitions=2000,seed=seed) for metric in METRICS},
            pooled_mean_prediction_rmse={key:descriptive[key] for key in descriptive if '_rmse_' in key})
        for metric in METRICS:
            if '_rmse_' in metric:
                mse=demand_bootstrap(selected,metric.replace('_rmse_','_mse_'),repetitions=2000,seed=seed)
                row['metrics'][metric]={key:(float(np.sqrt(value)) if value is not None else None)
                    if key in ('estimate','ci_low','ci_high') else value for key,value in mse.items()}
        result.append(row)
        if arm=='native':continue
        native={r['condition_index']:r for r in rows if (r['steps'],r['horizon'],r['arm'])==(steps,horizon,'native')}
        differences=[]
        for item in selected:
            other=native[item['condition_index']]
            delta={key:item[key] for key in ('demand_group','episode')}
            for metric in METRICS:
                if (item.get(metric) is None)!=(other.get(metric) is None):raise ValueError('paired masks differ')
                delta[metric]=None if item.get(metric) is None else item[metric]-other[metric]
            differences.append(delta)
        pair_metrics={metric:demand_bootstrap(differences,metric,repetitions=2000,seed=seed) for metric in METRICS}
        for metric in METRICS:
            if '_rmse_' not in metric:continue
            mse=metric.replace('_rmse_','_mse_');groups=sorted({r['demand_group'] for r in selected if r.get(mse) is not None})
            if not groups:continue
            a=np.array([np.mean([r[mse] for r in selected if r['demand_group']==g and r.get(mse) is not None]) for g in groups])
            b=np.array([np.mean([native[r['condition_index']][mse] for r in selected
                if r['demand_group']==g and r.get(mse) is not None]) for g in groups])
            estimate=float(np.sqrt(a.mean())-np.sqrt(b.mean()))
            draws=np.random.default_rng(seed).integers(len(a),size=(2000,len(a)))
            values=np.sqrt(a[draws].mean(axis=1))-np.sqrt(b[draws].mean(axis=1))
            low,high=np.quantile(values,[.025,.975])
            pair_metrics[metric]=dict(estimate=estimate,ci_low=float(low) if len(a)>1 else None,
                ci_high=float(high) if len(a)>1 else None,demand_groups=len(a))
        paired.append(dict(steps=steps,horizon=horizon,arm=arm,reference='native',metrics=pair_metrics))
    return dict(arms=result,intervention_minus_native=paired)


def plot(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for row_index,horizon in enumerate((1,4)):
        max_states=max(r['states'] for r in summary['arms'] if r['horizon']==horizon and r['steps'] in (2,20))
        plot_arms=[arm for arm in ARMS if any(r['arm']==arm and r['horizon']==horizon and r['states']==max_states
            and r['steps'] in (2,20) for r in summary['arms'])]
        for column,metric in enumerate(('count_mae_occupied','speed_mae_occupied','energy_score')):
            axis=axes[row_index,column]
            for steps,offset,color in [(2,-.17,'#4c78a8'),(20,.17,'#e45756')]:
                chosen=[next((r for r in summary['arms'] if (r['steps'],r['horizon'],r['arm'])==(steps,horizon,arm)
                    and r['states']==max_states),None) for arm in plot_arms]
                if any(r is None for r in chosen):continue
                values=[r['metrics'][metric]['estimate'] for r in chosen]
                low=[r['metrics'][metric]['ci_low'] for r in chosen];high=[r['metrics'][metric]['ci_high'] for r in chosen]
                if any(v is None for v in values):continue
                axis.bar(np.arange(len(plot_arms))+offset,values,width=.32,color=color,label=f'Euler {steps}')
                axis.errorbar(np.arange(len(plot_arms))+offset,values,yerr=[np.array(values)-low,np.array(high)-values],fmt='none',ecolor='black',capsize=2)
            axis.set_xticks(np.arange(len(plot_arms)),plot_arms,rotation=40,fontsize=8);axis.set_title(f'H{horizon}, N={max_states}: {metric}');axis.grid(axis='y',alpha=.2)
    axes[0,0].legend();fig.suptitle('Complete frozen FM predictions: prior-mean inference interventions')
    for suffix in ('png','pdf'):fig.savefig(output/('integrated_fm.'+suffix),dpi=180)
    plt.close(fig)


def run_primary100(args):
    """Training-target/solver-aligned H1 supplement; unchanged frozen FM."""
    started=time.time();torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    if FROZEN_ROOT!=(args.output/'source').resolve():raise ValueError('frozen supplement source required')
    if (args.output/'summary.json').exists():raise ValueError('preserve completed supplement')
    manifest=json.loads((args.output/'source_manifest.json').read_text())
    def verify_source():
        for relative,expected in manifest.items():
            if sha256(FROZEN_ROOT/relative)!=expected:raise ValueError('source changed')
    verify_source();(args.output/'banks').mkdir(exist_ok=True)
    refs={'checkpoint':args.experiment/'snapshot/checkpoints/latest.pt','basis':args.basis,
        'hybrid_tau0.1_head':args.hybrid_head,'hybrid_tau10_heads':args.hybrid_reference,
        'screen_conditions':args.screen_reference/'conditions.json','screen_cache':args.screen_reference/'cache_manifest.json',
        'scales':args.scale_reference}
    hashes={name:sha256(path) for name,path in refs.items()}
    if hashes['checkpoint']!=CHECKPOINT_SHA256:raise ValueError('completed checkpoint required')
    conditions=json.loads((args.screen_reference/'conditions.json').read_text())
    arms=('native','zero_mean','hybrid_state_soft0.1','hybrid_state_soft10')
    scale_meta=json.loads(args.scale_reference.read_text())['scales'];scales=np.asarray(scale_meta['scales'])
    protocol=dict(status='prespecified_before_scores',role='production-aligned complete FM H1 Euler100 confirmation',
        rationale='Numerical20to100 drift and known RAW-head/hybridCt-target confound; fix these compatibility issues, no method-score selection.',
        arms=list(arms),horizons=[1],steps=[100],conditions=len(conditions),groups=len({r['demand_group'] for r in conditions}),
        samples=8,sample_chunk=args.chunk,seed=args.seed,device=args.device,gpu_memory_fraction=args.gpu_memory_fraction,
        execution_profile='online chunk profile; FP32 deterministic, TF32off, noAMP/compile',
        inputs='FULL factual initialstate/batchactions/events, including withininterval birth actions',
        target='prepare_transition_targets hybridCt count + RAW next/following interval mean speed, same original production single_step',
        background_heads='state-only same frozen full-input body; filtered supervision trained on same hybridCt latent target; tau.1 newly solved training64groups768windows, tau10 archived fullysolved head',
        reference_hashes=hashes,noise='same stored native rank16 basis/scale, CPUdraw/depth seeds; no-zero-noise arm',
        gate='ALL first2 native100 banks reproduced withinmeanabs count/speed1e-5,max1e-3 vs savedv2initial-profile100 BEFORE intervention scoring. Same-profile native clone/noise remain bit-exact.',
        profile_parity_scope='H1 only; H4 online maxdrift failedpredeclared gate andisnot used',
        point_metrics='mean of decoded8draws, not decode(meanlatent); CRPS/energy/90%coverage all8draws',
        scales=scale_meta,bootstrap='2000 paired demandgroup resamples; count/speed RMSE pooled fromMSE',
        training_performed=False,caveats=['Frozen FM trained oldhard/actionconditioned prior; replacement is inferenceintervention, not jointlytrainedarchitecture.',
            'Development randomdemand groups; no externaltest, policy or generalizationclaim.',
            'State-only prior is conditional background predictor, not causal zero-action simulator.'])
    atomic_json(args.output/'protocol.json',protocol);atomic_json(args.output/'conditions.json',conditions)
    device=torch.device(args.device);torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
    if free<args.gpu_memory_fraction*total+256*1024**2:raise RuntimeError('insufficient supplementGPUheadroom')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)
    config,prepared,topology,model,_=load_frozen(args.experiment,device=device);model.eval().requires_grad_(False)
    model.set_online_chunk_profile();native=model.dynamics.graph_prior;original=digest(model);body=digest(model,True)
    capacities=torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1)
    limits=torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
    source=CompactPretrainDataset(config,prepared,topology,5,cache_manifest_path=args.screen_reference/'cache_manifest.json',split='validation')
    with np.load(args.basis) as asset:basis=asset['basis'];values=asset['eigenvalues']
    with np.load(args.hybrid_head) as asset:weak=asset['coefficients'].copy()
    with np.load(args.hybrid_reference) as asset:strong=asset['hybrid_Ct__soft_tau10__output_only'].copy()
    variants={'native':native,'zero_mean':MeanOverridePrior(native,zero=True)}
    for tau,coefficients in [(.1,weak),(10.,strong)]:
        operator=(basis*(1/(1+tau*values)))@basis.T
        variants[f'hybrid_state_soft{tau:g}']=MeanOverridePrior(native,operator=operator,coefficients=coefficients,state_only=True)
    def materialize(item):
        batch=source.batch_from_starts(item['episode_index'],[item['start']],fm_only=False)
        target=prepare_transition_targets(batch,capacities=capacities)
        z=model.encode(batch.flow[:,0].to(device))
        truth=endpoint_target(target.transition_target_flow[:,0],batch.flow[:,2],capacities,limits).numpy()
        count=int(target.transition_cohort_count[0,0])
        options=dict(capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(1,),samples=8,chunk=args.chunk,steps=100)
        return batch,z,truth,count,options
    native_bank={};gates=[]
    with torch.inference_mode():
        # Global alignment gate before any replaced-prior scoring.
        for item in conditions[:2]:
            batch,z,truth,count,options=materialize(item);model.dynamics.graph_prior=native
            draws=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),**options)[1]
            index=item['condition_index']
            with np.load(args.screen_reference/'banks'/f'condition_{index:03d}.npz') as old:
                expected=old['native__steps100__h1'];np.testing.assert_array_equal(truth,old['targets'][0])
            difference=np.abs(draws-expected);mean=np.abs(draws.mean(0)-expected.mean(0))
            check=dict(condition_index=index,mean_count_abs_drift=float(mean[...,0].mean()),mean_speed_abs_drift=float(mean[...,1].mean()),
                max_count_drift=float(difference[...,0].max()),max_speed_drift=float(difference[...,1].max()))
            atomic_json(args.output/'profile100_gate.json',gates+[check])
            if max(check['mean_count_abs_drift'],check['mean_speed_abs_drift'])>=1e-5 or max(check['max_count_drift'],check['max_speed_drift'])>=1e-3:
                raise ValueError('H1online100 profile parity failed')
            gates.append(check);native_bank[index]=draws
        rows=[];proofs=[];bank_manifest=[]
        for item in conditions:
            batch,z,truth,count,options=materialize(item);index=item['condition_index']
            model.dynamics.graph_prior=native
            reference=native_bank.pop(index) if index in native_bank else trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),**options)[1]
            model.dynamics.graph_prior=MeanOverridePrior(native)
            cloned=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),**options)[1]
            np.testing.assert_array_equal(reference,cloned)
            coefficients=torch.stack([torch.randn(1,native.rank,z.shape[-1],generator=torch.Generator(device='cpu').manual_seed(
                stable_seed(item['condition_seed'],'source_coefficients_v1',ordinal,0))) for ordinal in range(8)])
            for arm,prior in variants.items():
                for begin in (0,4):
                    rep=z.repeat(4,1,1)[:,None];noise=coefficients[begin:begin+4].to(z)
                    torch.testing.assert_close(prior.sample_noise(rep,noise),native.sample_noise(rep,noise),rtol=0,atol=0)
                if arm.startswith('hybrid_'):
                    a=ConditionEncoding(torch.zeros(1,1,len(basis),256,device=device));b=ConditionEncoding(torch.ones_like(a.edge_tokens))
                    torch.testing.assert_close(prior(z[:,None],a),prior(z[:,None],b),rtol=0,atol=0)
            stored=dict(target=truth,scales=scales,source_coefficients=coefficients.numpy(),initial_latent=z.cpu().numpy(),
                native_noise_basis=native.basis.cpu().numpy(),native_noise_scale=np.array(native.noise_scale))
            for arm in arms:
                model.dynamics.graph_prior=variants[arm]
                draws=reference if arm=='native' else trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),**options)[1]
                stored[arm]=draws;rows.append(score(draws,truth,scales,count,limits.numpy(),dict(item,arm=arm,steps=100,horizon=1,k=8,
                    forecast_seconds=30,control_scope='full_factual_hybrid_Ct',prior_target='HYBRID_Ct' if arm.startswith('hybrid_') else 'native')))
            model.dynamics.graph_prior=native
            path=args.output/'banks'/f'condition_{index:03d}.npz';np.savez_compressed(path,**stored)
            bank_manifest.append(dict(condition_index=index,path=str(path.resolve()),sha256=sha256(path)))
            proofs.append(dict(condition_index=index,native_clone_bit_equal=True,all_arms_noise_bit_equal=True,state_prior_action_invariant=True))
            with (args.output/'states.jsonl').open('w') as stream:
                for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
            atomic_json(args.output/'bank_manifest.json',bank_manifest);atomic_json(args.output/'native_noise_gates.json',proofs)
            progress=dict(stage='primary100',conditions_complete=len(proofs),total_conditions=len(conditions),elapsed_seconds=time.time()-started,
                max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2)
            atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            if len(proofs)==4:atomic_json(args.output/'pilot_summary.json',summaries(rows,args.seed))
    model.dynamics.graph_prior=native
    if digest(model)!=original or digest(model,True)!=body:raise ValueError('frozen model changed')
    for name,path in refs.items():
        if sha256(path)!=hashes[name]:raise ValueError('reference changed')
    for item in conditions:
        if archive_content_hash(Path(item['cache_path']))!=item['cache_content_hash']:raise ValueError('cache changed')
    imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
        if name.startswith(('tools.','urban_dreamer.')) and getattr(module,'__file__',None)}
    if any(not Path(path).is_relative_to(FROZEN_ROOT) for path in imports.values()):raise ValueError('live import')
    verify_source();atomic_json(args.output/'import_audit.json',imports)
    result=summaries(rows,args.seed)
    atomic_json(args.output/'summary.json',dict(status='complete',protocol=protocol,**result,body_digest=body,
        native_model_digest=original,checkpoint_unchanged=True,body_unchanged=True,noise_unchanged=True,
        source_unchanged=True,data_unchanged=True,elapsed_seconds=time.time()-started))
    print(json.dumps(dict(stage='complete_primary100',elapsed_seconds=time.time()-started)),flush=True)


def run(args):
    started=time.time();torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    if FROZEN_ROOT!=(args.output/'source').resolve():raise ValueError('execute new study frozen source')
    if (args.output/'summary.json').exists():raise ValueError('preserve completed results')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'banks').mkdir(exist_ok=True)
    manifest=json.loads((args.output/'source_manifest.json').read_text())
    def verify_source():
        for name,expected in manifest.items():
            if sha256(FROZEN_ROOT/name)!=expected:raise ValueError('frozen source changed')
    verify_source()
    refs={'checkpoint':args.experiment/'snapshot/checkpoints/latest.pt','basis':args.basis,
        'heads':args.heads,'selection':args.selection,'full_manifest':args.cache_manifest,'scale_reference':args.scale_reference}
    hashes={k:sha256(v) for k,v in refs.items()}
    if hashes['checkpoint']!=CHECKPOINT_SHA256:raise ValueError('completed checkpoint required')
    selected=json.loads(args.selection.read_text())['validation'][:args.groups]
    if len(selected)!=args.groups:raise ValueError('not enough prespecified groups')
    full=json.loads(args.cache_manifest.read_text());paths={r['path'] for r in selected}
    subset=dict(full,entries=selected,sources=[r for r in full['sources'] if r['path'] in paths],
        train_episodes=0,validation_episodes=len(selected),duplicate_files=0,
        parent_manifest_sha256=hashes['full_manifest'],study_selection='fixed first groups from prior independent selection')
    subset.pop('identity');subset['identity']=_identity(subset)
    atomic_json(args.output/'cache_manifest.json',subset)
    scale_meta=json.loads(args.scale_reference.read_text())['scales'];scales=np.asarray(scale_meta['scales'],dtype=np.float64)
    protocol=dict(status='prespecified_before_scores',checkpoint_sha256=CHECKPOINT_SHA256,
        reference_hashes=hashes,device=args.device,gpu_memory_fraction=args.gpu_memory_fraction,threads=2,
        groups=args.groups,conditions_per_group=2,samples=8,sample_chunk=args.chunk,horizons=[1,4],steps=[2,20],seed=args.seed,
        arms=list(ARMS),condition_sampling='fixed selected fresh-group episodes; eligible starts near one-third/two-thirds; no prediction selection',
        target={'H1_primary':'production single-step hybrid Ct count target from prepare_transition_targets; factual following-boundary mean speed',
                'H4_diagnostic':'fixed INITIAL cohort count and factual background following-boundary mean speed'},
        controls={'H1_primary':'FULL factual recorded batch actions/events including within-window birth actions; initial full state',
                  'H4_diagnostic':'initial-cohort factual recorded actions/events, fixed initial cohort state; no new-policy decisions'},
        recursive='initial real latent only; every subsequent latent generated; recorded controls teacher-forced, states never teacher-forced',
        state_background_fit='RAW next-state head trained previously; mismatch with production hybridCt H1 and fixedcohort H4 targets explicitly labeled',
        source_noise='original stored rank16 basis and scale; matched CPU draw/depth coefficient seeds at every arm',
        point_metrics='MEAN OF DECODED eight samples; empirical CRPS/energy, 90% coverage andwidth; same actual-target occupied mask',
        numerics={'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'TF32':False,'deterministic_algorithms':True},
        scales=scale_meta,gate='same-process/batch native versus cloned-native bit identity; no historical CUDA-bank bitwise gate',
        pilot='first four conditions full7arms×H1/H4×Euler2/20; runtime-only estimated fullmatrix<=900s keeps matrix. Otherwise remaining conditions all7armsH1 Euler2/20, H4Euler20 native/zero/state0/state.1 only. No score-based trimming.',
        solver100_check='first two fixed conditions only; H1 native andstate_soft.1, K8; compare with same-seed Euler20',
        uncertainty='2000 paired whole-demand-group bootstrap; equal demand then episode/state weighting; RMSE pooling separately reported',
        training_performed=False,FM_body_changed=False,SUMO_used=False,
        caveats=['Frozen FM was trained with hard rank16 action-conditioned prior; replacements are inference interventions, not joint-trained models.',
            'New background heads were fitted on raw next-state increments; primary complete FM uses production hybridCt H1, H4 is separate fixedcohort diagnostic.',
            'This development replay does not constitute external test or policy evaluation.',
            'Latent prior state-only branch is not a causal zero-action baseline.'])
    atomic_json(args.output/'protocol.json',protocol)
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
        if free<args.gpu_memory_fraction*total+512*1024**2:raise RuntimeError('insufficient GPU headroom')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)
    config,prepared,topology,model,_=load_frozen(args.experiment,device=device);model.eval().requires_grad_(False)
    native=model.dynamics.graph_prior;original=digest(model);body=digest(model,True)
    capacities=torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1)
    limits=torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
    source=CompactPretrainDataset(config,prepared,topology,5,cache_manifest_path=args.output/'cache_manifest.json',split='validation')
    indices={r['cache_path']:i for i,r in enumerate(source.manifest['entries'])}
    with np.load(args.basis) as asset:eigenvalues=asset['eigenvalues'];basis=asset['basis']
    operators={tau:(basis*(1/(1+tau*eigenvalues)))@basis.T for tau in (0.,.1,10.)}
    operators[0.]=np.eye(len(basis))
    variants={'native':native,'zero_mean':MeanOverridePrior(native,zero=True)}
    with np.load(args.heads) as coefficients:
        for tau in (.1,10.):variants[f'action_soft{tau:g}']=MeanOverridePrior(native,operator=operators[tau])
        for tau in (0.,.1,10.):variants[f'state_soft{tau:g}']=MeanOverridePrior(native,operator=operators[tau],
            coefficients=coefficients[f'soft_tau{tau:g}__filtered__latent'],state_only=True)
    conditions=[]
    for entry in selected:
        index=indices[entry['cache_path']];arrays=source._episode(index);legal=len(arrays['flow'])-6
        if legal<2:raise ValueError('insufficient complete H4 plus speed-following labels')
        starts=[int(round(legal/3)),int(round(2*legal/3))]
        if len(set(starts))!=2:raise ValueError('state starts must differ')
        for start in starts:
            conditions.append(dict(condition_index=len(conditions),episode_index=index,start=start,
                demand_group=entry['demand_group'],episode=entry['path']+'::episode='+str(entry['episode']),
                cache_path=entry['cache_path'],cache_content_hash=entry['cache_content_hash'],
                condition_seed=stable_seed(args.seed,entry['demand_group'],start)))
    atomic_json(args.output/'conditions.json',conditions)
    rows=[];proofs=[];bank_manifest=[];reduced_h4=False;scoring_started=time.time()
    with torch.inference_mode():
        for ordinal,item in enumerate(conditions):
            batch=source.batch_from_starts(item['episode_index'],[item['start']],fm_only=False)
            cohort=prepare_cohort_batch(batch,capacities=capacities)
            single_step=prepare_transition_targets(batch,capacities=capacities)
            if cohort.transition_flow.shape[1]<6:raise ValueError('missing complete labels')
            initial=model.encode(batch.flow[:,0].to(device))
            cohort_initial=model.encode(cohort.transition_flow[:,0].to(device))
            targets={1:endpoint_target(single_step.transition_target_flow[:,0],batch.flow[:,2],capacities,limits).numpy(),
                     4:endpoint_target(cohort.transition_flow[:,4],batch.flow[:,5],capacities,limits).numpy()}
            actions=cohort.transition_action.to(device);events=cohort.transition_action_events.to(device)
            source_coefficients=torch.stack([torch.stack([torch.randn(1,native.rank,initial.shape[-1],
                generator=torch.Generator(device='cpu').manual_seed(stable_seed(item['condition_seed'],'source_coefficients_v1',draw,depth)))
                for draw in range(8)]) for depth in range(4)])
            for arm,prior in variants.items():
                for depth in range(4):
                    for begin in (0,4):
                        coefficient=source_coefficients[depth,begin:begin+4].to(initial)
                        repeated=initial.repeat(4,1,1)[:,None]
                        torch.testing.assert_close(prior.sample_noise(repeated,coefficient),native.sample_noise(repeated,coefficient),rtol=0,atol=0)
                if arm.startswith('state_'):
                    condition=ConditionEncoding(torch.zeros(1,1,len(basis),256,device=device))
                    altered=ConditionEncoding(torch.ones_like(condition.edge_tokens)*37)
                    torch.testing.assert_close(prior(initial[:,None],condition),prior(initial[:,None],altered),rtol=0,atol=0)
            stored=dict(targets=np.stack([targets[h] for h in (1,4)]),initial_latent=initial.cpu().numpy(),cohort_initial_latent=cohort_initial.cpu().numpy(),
                horizons=np.array([1,4]),scales=scales,source_coefficients=source_coefficients.numpy(),
                native_noise_basis=native.basis.cpu().numpy(),native_noise_scale=np.array(native.noise_scale))
            options=dict(capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],samples=8,chunk=args.chunk)
            def generate(steps,horizons):
                result={}
                if 1 in horizons:
                    result[1]=trajectory_bank(model,initial,batch.action.to(device),batch.action_events.to(device),
                        steps=steps,horizons=(1,),**options)[1]
                if 4 in horizons:
                    result[4]=trajectory_bank(model,cohort_initial,actions,events,steps=steps,horizons=(4,),**options)[4]
                return result
            for steps in ((2,20,100) if ordinal<2 else (2,20)):
                active_arms=('native','state_soft0.1') if steps==100 else ARMS
                selected_horizons={arm:((1,) if steps==100 or
                    (reduced_h4 and (steps==2 or arm not in ('native','zero_mean','state_soft0','state_soft0.1')))
                    else (1,4)) for arm in active_arms}
                reference_horizons=tuple(sorted({h for value in selected_horizons.values() for h in value}))
                model.dynamics.graph_prior=native
                reference=generate(steps,reference_horizons)
                if ordinal==0 and steps==2:
                    repeated_reference=generate(steps,reference_horizons)
                    noop=MeanOverridePrior(native)
                    diagnostic={}
                    act=batch.action[:,0].to(device).repeat(4,1)[:,None]
                    evt=_repeat_events(batch.action_events.for_step(0).to(device),4)
                    first=model.dynamics.encode_condition(act,action_events=evt)
                    second=model.dynamics.encode_condition(act,action_events=evt)
                    a=native(repeated,first);b=noop(repeated,first)
                    start_a,_,cached_a=model.dynamics.prior_start(repeated,first,coefficient)
                    model.dynamics.graph_prior=noop
                    start_b,_,cached_b=model.dynamics.prior_start(repeated,first,coefficient)
                    diagnostic.update(condition_repeat_bit_equal=torch.equal(first.edge_tokens,second.edge_tokens),
                        condition_repeat_maxabs=float((first.edge_tokens-second.edge_tokens).abs().max()),
                        prior_mean_clone_bit_equal=torch.equal(a,b),prior_mean_clone_maxabs=float((a-b).abs().max()),
                        source_start_clone_bit_equal=torch.equal(start_a,start_b),source_start_clone_maxabs=float((start_a-start_b).abs().max()),
                        reference_context_clone_bit_equal=torch.equal(cached_a.reference_tokens,cached_b.reference_tokens),
                        reference_context_clone_maxabs=float((cached_a.reference_tokens-cached_b.reference_tokens).abs().max()),
                        native_repeat={str(h):dict(bit_equal=np.array_equal(reference[h],repeated_reference[h]),
                            maxabs=float(np.max(np.abs(reference[h]-repeated_reference[h])))) for h in reference_horizons},
                        prior_layout={name:dict(native_stride=list(value.stride()),clone_stride=list(noop.state_dict()[name].stride()))
                            for name,value in native.state_dict().items()})
                    atomic_json(args.output/'numerical_diagnosis.json',diagnostic)
                    for h in reference_horizons:np.testing.assert_array_equal(reference[h],repeated_reference[h])
                model.dynamics.graph_prior=MeanOverridePrior(native)
                cloned=generate(steps,reference_horizons)
                for h in reference_horizons:np.testing.assert_array_equal(reference[h],cloned[h])
                proofs.append(dict(condition_index=ordinal,steps=steps,native_clone_bit_identical=True,all_arm_noise_bit_identical=True,state_only_action_invariant=True))
                for arm in active_arms:
                    model.dynamics.graph_prior=variants[arm]
                    draws=reference if arm=='native' else generate(steps,selected_horizons[arm])
                    for h in selected_horizons[arm]:
                        stored[f'{arm}__steps{steps}__h{h}']=draws[h]
                        row=score(draws[h],targets[h],scales,int(cohort.cohort_count[0]),limits.numpy(),
                            dict(item,arm=arm,steps=steps,horizon=h,k=8,forecast_seconds=30*h,
                                control_scope='full_factual_hybrid_Ct' if h==1 else 'fixed_initial_cohort'))
                        rows.append(row)
                model.dynamics.graph_prior=native
            path=args.output/'banks'/f'condition_{ordinal:03d}.npz';np.savez_compressed(path,**stored)
            bank_manifest.append(dict(condition_index=ordinal,path=str(path.resolve()),sha256=sha256(path)))
            atomic_json(args.output/'native_noise_gates.json',proofs);atomic_json(args.output/'bank_manifest.json',bank_manifest)
            with (args.output/'states.jsonl').open('w') as stream:
                for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
            progress=dict(stage='integrated_fm',conditions_complete=ordinal+1,total_conditions=len(conditions),
                elapsed_seconds=time.time()-started,max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2)
            atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            if ordinal==3:
                if digest(model,True)!=body or digest(model)!=original:raise ValueError('model changed in pilot')
                atomic_json(args.output/'pilot_summary.json',summaries(rows,args.seed));plot(summaries(rows,args.seed),args.output)
                estimate=(time.time()-scoring_started)/4*len(conditions)
                reduced_h4=estimate>900
                atomic_json(args.output/'runtime_gate.json',dict(score_based=False,pilot_conditions=4,
                    estimated_full_seconds=estimate,threshold_seconds=900,reduced_remaining_h4=reduced_h4,
                    remaining_matrix='allarmsH1Euler2/20; H4Euler20 native/zero/state0/state.1' if reduced_h4 else 'full7armsH1/H4Euler2/20'))
    model.dynamics.graph_prior=native
    if digest(model,True)!=body or digest(model)!=original:raise ValueError('completed model changed')
    for key,path in refs.items():
        if sha256(path)!=hashes[key]:raise ValueError('reference changed')
    for entry in selected:
        if archive_content_hash(Path(entry['cache_path']))!=entry['cache_content_hash']:raise ValueError('cache changed')
    imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
        if name.startswith(('tools.','urban_dreamer.')) and getattr(module,'__file__',None)}
    if any(not Path(path).is_relative_to(FROZEN_ROOT) for path in imports.values()):raise ValueError('live source contaminated frozen eval')
    verify_source();atomic_json(args.output/'import_audit.json',imports)
    summary=summaries(rows,args.seed);plot(summary,args.output)
    atomic_json(args.output/'summary.json',dict(status='complete',protocol=protocol,**summary,
        body_digest=body,native_model_digest=original,checkpoint_unchanged=True,body_unchanged=True,
        noise_unchanged=True,source_unchanged=True,data_unchanged=True,elapsed_seconds=time.time()-started))
    print(json.dumps(dict(stage='complete',elapsed_seconds=time.time()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','basis','heads','selection','cache-manifest','scale-reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--groups',type=int,default=16);p.add_argument('--seed',type=int,default=20261016)
    p.add_argument('--device',default='cuda:1');p.add_argument('--gpu-memory-fraction',type=float,default=.06)
    p.add_argument('--chunk',type=int,default=4)
    p.add_argument('--primary100',action='store_true');p.add_argument('--hybrid-head',type=Path)
    p.add_argument('--hybrid-reference',type=Path);p.add_argument('--screen-reference',type=Path)
    args=p.parse_args()
    if args.primary100:
        if not all((args.hybrid_head,args.hybrid_reference,args.screen_reference)):p.error('primary100 requires three reference arguments')
        run_primary100(args)
    else:run(args)
