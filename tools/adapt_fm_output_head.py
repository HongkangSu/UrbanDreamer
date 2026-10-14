"""Low-budget coupled FM calibration: anchored closed-form final-head fit.

All codec, prior, action, reference, and transformer parameters stay frozen.
The feature phase uses teacher interpolation, not rollout or backpropagation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
FROZEN_ROOT=Path(os.environ.get('FM_HEAD_ADAPT_FROZEN_ROOT',ROOT)).resolve()
sys.path.insert(0,str(FROZEN_ROOT))
import numpy as np
import torch

from tools.evaluate_soft_prior_fm import MeanOverridePrior,digest,summaries,score,METRICS
from tools.evaluate_hard_rank_fm import HardMeanPrior
from tools.component_experiment_data import atomic_json,load_frozen,dataset,sha256
from tools.evaluate_pretrained_distribution import CHECKPOINT_SHA256,trajectory_bank
from tools.evaluate_component_world_models import endpoint_target,stable_seed
from tools.analyze_traffic_spectrum import select_starts
from tools.summarize_component_predictions import demand_bootstrap
from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.learning.offline_data import archive_content_hash


def anchored_ridge(gram,cross,energy,anchor,relative=1e-4):
    gram,cross,anchor=[np.asarray(v,dtype=np.float64) for v in (gram,cross,anchor)]
    if gram.ndim!=2 or gram.shape[0]!=gram.shape[1] or cross.shape!=anchor.shape or cross.shape[0]!=len(gram):
        raise ValueError('matching square Gram, cross moment and anchor required')
    if not all(np.isfinite(v).all() for v in (gram,cross,anchor)) or not np.isfinite(energy) or relative<=0:
        raise ValueError('finite statistics and positive anchor strength required')
    ridge=relative*max(float(np.trace(gram)/len(gram)),1e-12)
    delta=np.linalg.solve(gram+np.eye(len(gram))*ridge,cross-gram@anchor);head=anchor+delta
    def sse(w):return float(energy-2*np.sum(w*cross)+np.sum(w*(gram@w)))
    residual=(gram+np.eye(len(gram))*ridge)@head-(cross+ridge*anchor)
    proof=dict(relative_ridge=relative,ridge=ridge,coefficients_shape=list(head.shape),parameters=int(head.size),
        bias_anchored=True,original_train_sse=sse(anchor),adapted_train_sse=sse(head),
        anchor_penalty=float(ridge*np.square(delta).sum()),parameter_relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(anchor),1e-30)),
        normal_equation_relative_residual=float(np.linalg.norm(residual)/max(np.linalg.norm(cross+ridge*anchor),1e-30)))
    return head,proof


def capture_final_features(head,forward):
    """Install/remove a temporary hook even if forward raises; no parameter edit."""
    captures=[]
    hook=head.register_forward_pre_hook(lambda module,inputs:captures.append(inputs[0].detach()))
    try:
        with torch.no_grad():result=forward()
    finally:hook.remove()
    if len(captures)!=1:raise ValueError('exactly one final affine-head call expected')
    return captures[0],result


def head_coefficients(head):
    return np.concatenate((head.weight.detach().cpu().numpy().T,head.bias.detach().cpu().numpy()[None]),axis=0).astype(np.float64)


def install_head(head,coefficients):
    values=torch.as_tensor(coefficients,device=head.weight.device,dtype=head.weight.dtype)
    if values.shape!=(head.in_features+1,head.out_features) or not bool(torch.isfinite(values).all()):raise ValueError('bad affine coefficients')
    with torch.no_grad():head.weight.copy_(values[:-1].T);head.bias.copy_(values[-1])


def frozen_digest(model):
    result=hashlib.sha256()
    for name,value in model.state_dict().items():
        if name.startswith(('dynamics.graph_prior.','dynamics.flow_head.')):continue
        result.update(name.encode());result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def same_prior_contrasts(rows,seed):
    results=[]
    for a,b in [(f'adapt_{name}',f'old_{name}') for name in ('hard_r16','hard_r64','soft_tau0.1','soft_tau10')]+[
            ('adapt_hard_r64','adapt_hard_r16'),('adapt_soft_tau0.1','adapt_soft_tau10')]:
        left={r['condition_index']:r for r in rows if r['arm']==a};right={r['condition_index']:r for r in rows if r['arm']==b}
        if left.keys()!=right.keys():raise ValueError('same states required')
        differences=[]
        for index,row in left.items():
            reference=right[index];delta={key:row[key] for key in ('demand_group','episode')}
            for metric in METRICS:
                if '_rmse_' in metric:continue
                if (row[metric] is None)!=(reference[metric] is None):raise ValueError('paired masks differ')
                delta[metric]=None if row[metric] is None else row[metric]-reference[metric]
            differences.append(delta)
        results.append(dict(arm=a,reference=b,states=len(left),metrics={m:demand_bootstrap(differences,m,repetitions=2000,seed=seed)
            for m in METRICS if '_rmse_' not in m}))
    return results


def run(args):
    started=time.time();torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    if FROZEN_ROOT!=(args.output/'source').resolve():raise ValueError('new frozen source required')
    if (args.output/'summary.json').exists():raise ValueError('preserve completed result')
    manifest=json.loads((args.output/'source_manifest.json').read_text())
    def verify_source():
        for relative,expected in manifest.items():
            if sha256(FROZEN_ROOT/relative)!=expected:raise ValueError('source changed')
    verify_source();(args.output/'banks').mkdir(exist_ok=True)
    refs={'checkpoint':args.experiment/'snapshot/checkpoints/latest.pt','basis':args.basis,'prior_heads':args.prior_heads,
        'train_selection':args.train_selection,'conditions':args.reference/'conditions.json','cached_banks':args.reference/'bank_manifest.json',
        'cache_manifest':args.cache_manifest,'scale_reference':args.scale_reference,
        'soft_weak_head':args.soft_weak_head,'soft_strong_head':args.soft_strong_heads,
        'soft_old_banks':args.soft_reference/'bank_manifest.json','soft_old_conditions':args.soft_reference/'conditions.json'}
    hashes={k:sha256(v) for k,v in refs.items()}
    if hashes['checkpoint']!=CHECKPOINT_SHA256:raise ValueError('completed checkpoint required')
    selection=json.loads(args.train_selection.read_text())['train'];conditions=json.loads((args.reference/'conditions.json').read_text())[:16]
    if {r['demand_group'] for r in selection}&{r['demand_group'] for r in conditions}:raise ValueError('train/eval groups overlap')
    protocol=dict(status='prespecified_before_features_or_fit',checkpoint_sha256=CHECKPOINT_SHA256,reference_hashes=hashes,
        bootstrap_seed=args.seed,train_groups=64,train_windows=768,main_eval_conditions=16,main_eval_demand_groups=8,
        prior_families=['hard_r16','hard_r64','soft_tau0.1','soft_tau10'],times=[.25,.75],source_coefficients='paired CPUstable_seed(seed,traincache,start,time) across ALL4matchedHYBRIDpriors',
        feature='temporary flow_head forward_pre_hook underno_grad; final256D transformer features, originalflowhead retainedduring extraction',
        source='z+b_state_prior+originalnoise16; HYBRIDCt encodedtarget',interpolation='x_t=(1-t)start+t target',velocity_target='target-start',
        only_fitted='flow_head weight256x64+bias64=16448parameters; codec/prior/actionencoder/transformer/referenceproj frozen',
        objective='F64anchoredridge ||HW-Y||²+lambda||W-W0||²; lambda=1e-4traceGram/257, bias anchored, notunedusingvalidation',
        solver='W=W0+(G+lambdaI)^-1(C-GW0), exactconvex optimum of frozen featureproblem; notfulljointtraining',
        budget_choice='Twofixedt upfront for ALL4priors afteruserrequestedsoftarms; all768realwindowsretained,1536augmentations/prior; noquality-based time/ridge tuning',
        eval='H1Euler100K8FP32online, FULL observedinitialstate/FULL factualactions includingbirthactions, HYBRIDCtcount/RAWfollowingmean speed',
        comparisons=['unchangednative',*[f'old_{name}' for name in ('hard_r16','hard_r64','soft_tau0.1','soft_tau10')],
            *[f'adapt_{name}' for name in ('hard_r16','hard_r64','soft_tau0.1','soft_tau10')]],
        primary_pairs=['eachadapted-sameprioroldhead','adaptedhard64-adaptedhard16','adaptedsoft.1-adaptedsoft10'],
        gate='first2nativeoldhead+samepriors reproducibleagainstcachedsame-profilebanks, strictnoise/body equalitybeforeadaptedscores',
        device=args.device,gpu_memory_fraction=args.gpu_memory_fraction,threads=2,sample_chunk=args.chunk,
        state_training_updated=False,FM_backbone_training_updated=False,no_SGD=True,
        caveats=['Lowbudgetcoupled lastlayer calibration of oldfrozenFM, not fullyretrainedarchitecture.',
            'Finitefeature augmentation/64groups; closedformconvergence is onlyforchosenlinearhead, notfullFM optimization.',
            'One checkpoint, eightvalidationdemandgroups, developmentreplay, nopolicyorexternaltestclaim.'])
    atomic_json(args.output/'protocol.json',protocol);atomic_json(args.output/'conditions.json',conditions)
    device=torch.device(args.device);torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
    if free<args.gpu_memory_fraction*total+256*1024**2:raise RuntimeError('insufficient GPUadmission')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)
    config,prepared,topology,model,_=load_frozen(args.experiment,device=device);model.eval().requires_grad_(False);model.set_online_chunk_profile()
    native=model.dynamics.graph_prior;original=digest(model);frozen=frozen_digest(model);anchor=head_coefficients(model.dynamics.flow_head)
    if anchor.shape!=(257,64):raise ValueError('expected256to64flowhead')
    with np.load(args.basis) as a:basis=a['basis'];eigenvalues=a['eigenvalues']
    with np.load(args.prior_heads) as a:priors={f'hard_r{r}':HardMeanPrior(native,basis[:,:r],coefficients=a[f'rank_{r}'],state_only=True) for r in (16,64)}
    with np.load(args.soft_weak_head) as a:weak=a['coefficients'].copy()
    with np.load(args.soft_strong_heads) as a:strong=a['hybrid_Ct__soft_tau10__output_only'].copy()
    for tau,coefficients in [(.1,weak),(10.,strong)]:
        operator=(basis*(1/(1+tau*eigenvalues)))@basis.T
        priors[f'soft_tau{tau:g}']=MeanOverridePrior(native,operator=operator,coefficients=coefficients,state_only=True)
    capacities=torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1);limits=torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
    training=dataset(args.experiment,config,prepared,topology,split='train',horizon=1);indices={e['cache_path']:i for i,e in enumerate(training.manifest['entries'])}
    training_audit=[]
    def cache_features(times,gate=False):
        phase_start=time.time();stats={r:dict(gram=np.zeros((257,257)),cross=np.zeros((257,64)),energy=0.) for r in priors}
        windows=0
        with torch.inference_mode():
            for ordinal,entry in enumerate(selection):
                index=indices[entry['cache_path']];arrays=training._episode(index);starts=select_starts(arrays['flow'],12)
                batch=training.batch_from_starts(index,starts.tolist(),materialization_horizon=1,fm_only=True)
                targets=prepare_transition_targets(batch,capacities=capacities)
                z=model.encode(batch.flow[:,0].to(device))[:,None];target=model.encode(targets.transition_target_flow[:,0].to(device))[:,None]
                actions=batch.action.to(device);events=batch.action_events.to(device);condition=model.dynamics.encode_condition(actions,action_events=events)
                for time_value in times:
                    coefficient=torch.stack([torch.randn(1,native.rank,64,generator=torch.Generator(device='cpu').manual_seed(
                        stable_seed(args.seed,'head_calibration_noise_v1',entry['cache_path'],int(start),float(time_value)))) for start in starts]).to(z)
                    for rank,prior in priors.items():
                        torch.testing.assert_close(prior.sample_noise(z,coefficient),native.sample_noise(z,coefficient),rtol=0,atol=0)
                        model.dynamics.graph_prior=prior;start,_,cached=model.dynamics.prior_start(z,condition,coefficient)
                        x=(1-time_value)*start+time_value*target;velocity=target-start;t=torch.full(z.shape[:2],time_value,device=device,dtype=z.dtype)
                        features,output=capture_final_features(model.dynamics.flow_head,lambda:model.dynamics(actions,packed_latents=x,flow_time=t,condition=cached))
                        if ordinal==0:
                            torch.testing.assert_close(model.dynamics.flow_head(features),output.latent_velocity,rtol=0,atol=0)
                        hidden=features.cpu().numpy().astype(np.float64).reshape(-1,256);y=velocity.cpu().numpy().astype(np.float64).reshape(-1,64)
                        design=np.concatenate((hidden,np.ones((len(hidden),1))),axis=1)
                        stats[rank]['gram']+=design.T@design;stats[rank]['cross']+=design.T@y;stats[rank]['energy']+=float(np.square(y).sum())
                windows+=len(starts)*len(times)
                if len(training_audit)<64:training_audit.append(dict(cache_path=entry['cache_path'],content_hash=entry['cache_content_hash'],demand_group=entry['demand_group'],starts=starts.tolist()))
                print(json.dumps(dict(stage='features',group=ordinal+1,total=64,times=list(times),elapsed_seconds=time.time()-phase_start)),flush=True)
                if ordinal==3 and gate:
                    estimate=(time.time()-phase_start)*16
                    atomic_json(args.output/'timing_gate.json',dict(score_based=False,estimated_feature_seconds=estimate,threshold_seconds=480,reduce_times=estimate>480))
                    if estimate>480:return None,None
        return stats,windows
    used_times=[.25,.75];stats,augmentations=cache_features(used_times,gate=False)
    model.dynamics.graph_prior=native
    if digest(model)!=original or frozen_digest(model)!=frozen:raise ValueError('featurephase changed model')
    heads={};fits={}
    for rank,bucket in stats.items():heads[rank],fits[str(rank)]=anchored_ridge(bucket['gram'],bucket['cross'],bucket['energy'],anchor)
    atomic_json(args.output/'fit_diagnostics.json',dict(fits=fits,used_times=used_times,augmented_windows_per_prior=augmentations,training_windows=768,
        backbone_unchanged=True,original_flowhead_unchanged_during_features=True))
    atomic_json(args.output/'training_data_audit.json',training_audit)
    np.savez_compressed(args.output/'heads.npz',anchor=anchor,**{f'head_{r}':h for r,h in heads.items()})
    np.savez_compressed(args.output/'training_statistics.npz',**{f'rank_{r}__{key}':value for r,b in stats.items() for key,value in b.items()})
    evaluation=CompactPretrainDataset(config,prepared,topology,5,cache_manifest_path=args.cache_manifest,split='validation')
    reference_banks={b['condition_index']:b for b in json.loads((args.reference/'bank_manifest.json').read_text())}
    for bank in reference_banks.values():
        if sha256(Path(bank['path']))!=bank['sha256']:raise ValueError('referencebankchanged')
    soft_banks={b['condition_index']:b for b in json.loads((args.soft_reference/'bank_manifest.json').read_text())}
    for bank in soft_banks.values():
        if sha256(Path(bank['path']))!=bank['sha256']:raise ValueError('softreferencebankchanged')
    soft_conditions=json.loads((args.soft_reference/'conditions.json').read_text())
    if any(item!=soft_conditions[i] for i,item in enumerate(conditions)):raise ValueError('soft/hard condition pairing differs')
    scale=np.asarray(json.loads(args.scale_reference.read_text())['scales']['scales'])
    def materialize(item):
        batch=evaluation.batch_from_starts(item['episode_index'],[item['start']],fm_only=False);target=prepare_transition_targets(batch,capacities=capacities)
        z=model.encode(batch.flow[:,0].to(device));truth=endpoint_target(target.transition_target_flow[:,0],batch.flow[:,2],capacities,limits).numpy()
        count=int(target.transition_cohort_count[0,0]);options=dict(capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(1,),samples=8,chunk=args.chunk,steps=100)
        return batch,z,truth,count,options
    gates=[];rows=[];bank_manifest=[]
    with torch.inference_mode():
        for item in conditions[:2]:
            batch,z,truth,count,options=materialize(item);install_head(model.dynamics.flow_head,anchor)
            for family,prior in [('native',native),*priors.items()]:
                model.dynamics.graph_prior=prior;actual=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),**options)[1]
                bank=reference_banks[item['condition_index']] if family=='native' or family.startswith('hard') else soft_banks[item['condition_index']]
                key={'native':'native','hard_r16':'state_r16','hard_r64':'state_r64','soft_tau0.1':'hybrid_state_soft0.1','soft_tau10':'hybrid_state_soft10'}[family]
                with np.load(bank['path']) as old:
                    np.testing.assert_array_equal(truth,old['target']);np.testing.assert_array_equal(actual,old[key])
                gates.append(dict(condition_index=item['condition_index'],prior=family,old_head_bank_bit_identical=True))
            atomic_json(args.output/'old_head_native_gates.json',gates)
        for item in conditions:
            batch,z,truth,count,options=materialize(item);index=item['condition_index']
            with np.load(reference_banks[index]['path']) as old:
                np.testing.assert_array_equal(truth,old['target'])
                cached={arm:old[key].copy() for arm,key in [('native','native'),('old_hard_r16','state_r16'),('old_hard_r64','state_r64')]}
                coefficients=old['source_coefficients'].copy()
            with np.load(soft_banks[index]['path']) as old:
                np.testing.assert_array_equal(truth,old['target']);np.testing.assert_array_equal(coefficients,old['source_coefficients'])
                cached.update(old_soft_tau0_1=old['hybrid_state_soft0.1'].copy(),old_soft_tau10=old['hybrid_state_soft10'].copy())
                cached['old_soft_tau0.1']=cached.pop('old_soft_tau0_1')
            stored=dict(target=truth,scales=scale,source_coefficients=coefficients,initial_latent=z.cpu().numpy(),**cached)
            for arm,draws in cached.items():rows.append(score(draws,truth,scale,count,limits.numpy(),dict(item,arm=arm,steps=100,horizon=1,k=8)))
            for rank,prior in priors.items():
                for begin in (0,4):
                    noise=torch.as_tensor(coefficients[begin:begin+4],device=device,dtype=z.dtype);rep=z.repeat(4,1,1)[:,None]
                    torch.testing.assert_close(prior.sample_noise(rep,noise),native.sample_noise(rep,noise),rtol=0,atol=0)
                model.dynamics.graph_prior=prior;install_head(model.dynamics.flow_head,heads[rank])
                draws=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),**options)[1]
                stored[f'adapt_{rank}']=draws;rows.append(score(draws,truth,scale,count,limits.numpy(),dict(item,arm=f'adapt_{rank}',steps=100,horizon=1,k=8)))
            model.dynamics.graph_prior=native;install_head(model.dynamics.flow_head,anchor)
            if frozen_digest(model)!=frozen or digest(model)!=original:raise ValueError('backbone changed or originalheadnotrestored')
            path=args.output/'banks'/f'condition_{index:03d}.npz';np.savez_compressed(path,**stored);bank_manifest.append(dict(condition_index=index,path=str(path.resolve()),sha256=sha256(path)))
            with (args.output/'states.jsonl').open('w') as stream:
                for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
            atomic_json(args.output/'bank_manifest.json',bank_manifest)
            progress=dict(stage='adapted_evaluation',conditions_complete=index+1,total_conditions=16,elapsed_seconds=time.time()-started,max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2)
            atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            if index==3:atomic_json(args.output/'pilot_summary.json',summaries(rows,args.seed))
    for name,path in refs.items():
        if sha256(path)!=hashes[name]:raise ValueError('referencechanged')
    for entry in training_audit:
        if archive_content_hash(Path(entry['cache_path']))!=entry['content_hash']:raise ValueError('traincachechanged')
    for item in conditions:
        if archive_content_hash(Path(item['cache_path']))!=item['cache_content_hash']:raise ValueError('evalcachechanged')
    verify_source();imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if name.startswith(('tools.','urban_dreamer.')) and getattr(module,'__file__',None)}
    if any(not Path(path).is_relative_to(FROZEN_ROOT) for path in imports.values()):raise ValueError('liveimport')
    atomic_json(args.output/'import_audit.json',imports)
    result=summaries(rows,args.seed);pairs=same_prior_contrasts(rows,args.seed)
    atomic_json(args.output/'summary.json',dict(status='complete',protocol=protocol,**result,same_prior_and_rank_pairs=pairs,
        frozen_backbone_digest=frozen,native_model_digest=original,backbone_unchanged=True,head_only_adapted=True,original_head_restored=True,
        noise_unchanged=True,checkpoint_unchanged=True,source_unchanged=True,data_unchanged=True,elapsed_seconds=time.time()-started))
    print(json.dumps(dict(stage='complete',elapsed_seconds=time.time()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','basis','prior-heads','train-selection','reference','cache-manifest','scale-reference','output',
                 'soft-weak-head','soft-strong-heads','soft-reference'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--gpu-memory-fraction',type=float,default=.03)
    p.add_argument('--chunk',type=int,default=4);p.add_argument('--seed',type=int,default=20261018)
    run(p.parse_args())
