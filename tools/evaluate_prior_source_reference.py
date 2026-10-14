"""Split native prior influence on source mean versus reference conditioning.

Only the two cross arms are new forecasts. Completed native/zero-mean banks
are reused after strict within-process diagonal checks. No parameter training.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
FROZEN_ROOT=Path(os.environ.get('PRIOR_SPLIT_FROZEN_ROOT',ROOT)).resolve()
sys.path.insert(0,str(FROZEN_ROOT))
import numpy as np
import torch

from tools.component_experiment_data import atomic_json,load_frozen,sha256
from tools.evaluate_pretrained_distribution import CHECKPOINT_SHA256,trajectory_bank
from tools.evaluate_component_world_models import endpoint_target
from tools.evaluate_soft_prior_fm import digest,summaries,score,METRICS
from tools.summarize_component_predictions import demand_bootstrap
from tools.world_model_evaluation import _repeat_events
from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.learning.offline_data import archive_content_hash
from urban_dreamer.models.world_model import ConditionEncoding

ARMS={'native':(True,True),'zero_mean':(False,False),
      'source_only':(True,False),'reference_only':(False,True)}


@contextmanager
def split_prior_start(dynamics,*,source_on,reference_on):
    """Temporary instance method, restoring descriptor/instance presence exactly."""
    if not isinstance(source_on,bool) or not isinstance(reference_on,bool):raise ValueError('boolean switches required')
    present='prior_start' in dynamics.__dict__;old=dynamics.__dict__.get('prior_start')
    def prior_start(self,z,condition,noise=None,*,detach_prior_increment=False):
        if not isinstance(detach_prior_increment,bool):raise ValueError('boolean detach flag required')
        increment=self.graph_prior(z,condition)
        detach=detach_prior_increment or not getattr(self.graph_prior,'prior_action_conditioned',True)
        b=increment.detach() if detach else increment
        source_mean=z+b if source_on else z
        reference=z+b if reference_on else z
        cached=ConditionEncoding(condition.edge_tokens,self.reference_proj(reference))
        return source_mean+self.graph_prior.sample_noise(z,noise),increment,cached
    dynamics.prior_start=MethodType(prior_start,dynamics)
    try:yield
    finally:
        if present:dynamics.prior_start=old
        else:delattr(dynamics,'prior_start')


def contrasts(rows,seed):
    indexes={arm:{r['condition_index']:r for r in rows if r['arm']==arm} for arm in ARMS}
    pairs=[('source_only','zero_mean','source_effect_reference_off'),
        ('native','reference_only','source_effect_reference_on'),
        ('reference_only','zero_mean','reference_effect_source_off'),
        ('native','source_only','reference_effect_source_on'),
        ('native','zero_mean','joint_effect'),
        ('source_only','zero_mean','source_only_vs_zero'),
        ('reference_only','zero_mean','reference_only_vs_zero')]
    result=[];metrics=[m for m in METRICS if '_rmse_' not in m]
    for a,b,label in pairs:
        differences=[]
        if indexes[a].keys()!=indexes[b].keys():raise ValueError('identical paired conditions required')
        for index,row in indexes[a].items():
            other=indexes[b][index];item={k:row[k] for k in ('demand_group','episode')}
            for metric in metrics:
                if (row[metric] is None)!=(other[metric] is None):raise ValueError('paired metric masks differ')
                item[metric]=None if row[metric] is None else row[metric]-other[metric]
            differences.append(item)
        result.append(dict(label=label,arm=a,reference=b,metrics={m:demand_bootstrap(differences,m,repetitions=2000,seed=seed) for m in metrics}))
    interactions=[]
    for index,row in indexes['native'].items():
        item={k:row[k] for k in ('demand_group','episode')}
        for metric in metrics:
            values=[indexes[arm][index][metric] for arm in ('native','source_only','reference_only','zero_mean')]
            if all(v is None for v in values):item[metric]=None
            elif any(v is None for v in values):raise ValueError('interaction mask mismatch')
            else:item[metric]=values[0]-values[1]-values[2]+values[3]
        interactions.append(item)
    result.append(dict(label='interaction_E11_minus_E10_minus_E01_plus_E00',
        metrics={m:demand_bootstrap(interactions,m,repetitions=2000,seed=seed) for m in metrics}))
    return result


def run(args):
    started=time.time();torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    if FROZEN_ROOT!=(args.output/'source').resolve():raise ValueError('frozen new study source required')
    if (args.output/'summary.json').exists():raise ValueError('preserve completed result')
    manifest=json.loads((args.output/'source_manifest.json').read_text())
    def verify_source():
        for relative,expected in manifest.items():
            if sha256(FROZEN_ROOT/relative)!=expected:raise ValueError('frozen source changed')
    verify_source();(args.output/'banks').mkdir(exist_ok=True)
    refs={'checkpoint':args.experiment/'snapshot/checkpoints/latest.pt','conditions':args.reference/'conditions.json',
        'cached_banks':args.reference/'bank_manifest.json','cached_summary':args.reference/'summary.json',
        'cache_manifest':args.cache_manifest,'scales':args.scale_reference}
    hashes={k:sha256(v) for k,v in refs.items()}
    if hashes['checkpoint']!=CHECKPOINT_SHA256:raise ValueError('completed checkpoint required')
    conditions=json.loads((args.reference/'conditions.json').read_text())
    if len(conditions)!=32 or len({r['demand_group'] for r in conditions})!=16:raise ValueError('same32conditions16groups required')
    protocol=dict(status='prespecified_before_scores',checkpoint_sha256=CHECKPOINT_SHA256,reference_hashes=hashes,
        bootstrap_seed=args.seed,arms={k:dict(source_on=v[0],reference_on=v[1]) for k,v in ARMS.items()},
        increment='ORIGINAL action-conditioned native b(z,A), allweights unchanged; not state-only background',
        source='z+(b ifsource_on else0)+sameoriginalU16Gaussiannoise',
        reference='reference_proj(z+b ifreference_on elsez); referenceoff iscurrentstate, notNone/zeros',
        cached_diagonals=['native','zero_mean'],new_forecasts=['source_only','reference_only'],
        preservation='Native prior modules unchanged; temporary prior_start restored original instanceattribute presence evenonexception',
        target='Productionaligned HYBRIDCt nextcount andRAWnext/followingmean speed; FULLfactualstate/actions/events',
        conditions=32,demand_groups=16,horizons=[1],steps=100,samples=8,sample_chunk=args.chunk,
        point_metrics='mean of decoded8draws; probabilitymetrics useALL8',
        execution='FP32onlinechunks deterministicCUBLAS/TF32off; CPU/BLAS2; noAMP/compile',device=args.device,gpu_memory_fraction=args.gpu_memory_fraction,
        gate='First2bothdiagonals exactcachedbank checks beforeanycross forecasts; originalsource noise bitexact',
        metrics='count/speed MAE/pooledRMSE,CRPS/EnergyScore,90%coverage/width,totalcount,actualtarget occupiedmask',
        contrasts=['source_effect_reference_off','source_effect_reference_on','reference_effect_source_off','reference_effect_source_on',
                   'joint_effect','source_only_vs_zero','reference_only_vs_zero','interaction_E11-E10-E01+E00'],
        bootstrap='2000pairedwhole demandgroups; fixedsame32states16groups; one frozencheckpoint uncertaintyonly',
        training_performed=False,new_diagonal_predictions_only_for_reproducibility_gate=True,
        caveats=['Mechanism inferenceintervention on oldcompletedFM, notretraining orpolicynecessity proof.',
            'Source/reference effects mayinteract; reportallmetrics, noassumptionnativealwaysworse.',
            'Development SUMO replay onrealcitynetwork; notexternaltest orfield sensor traffic.'])
    atomic_json(args.output/'protocol.json',protocol);atomic_json(args.output/'conditions.json',conditions)
    device=torch.device(args.device);torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
    if free<args.gpu_memory_fraction*total+256*1024**2:raise RuntimeError('insufficient GPUadmission')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)
    config,prepared,topology,model,_=load_frozen(args.experiment,device=device);model.eval().requires_grad_(False);model.set_online_chunk_profile()
    if model.dynamics.graph_prior.rank!=16 or not model.dynamics.graph_prior.prior_action_conditioned:raise ValueError('originalrank16actionprior required')
    original=digest(model);method_was_local='prior_start' in model.dynamics.__dict__
    capacities=torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1);limits=torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
    data=CompactPretrainDataset(config,prepared,topology,5,cache_manifest_path=args.cache_manifest,split='validation')
    scales=np.asarray(json.loads(args.scale_reference.read_text())['scales']['scales'])
    bank_index={b['condition_index']:b for b in json.loads((args.reference/'bank_manifest.json').read_text())}
    for bank in bank_index.values():
        if sha256(Path(bank['path']))!=bank['sha256']:raise ValueError('cached bank changed')
    def materialize(item):
        batch=data.batch_from_starts(item['episode_index'],[item['start']],fm_only=False)
        scoped=prepare_transition_targets(batch,capacities=capacities);z=model.encode(batch.flow[:,0].to(device))
        target=endpoint_target(scoped.transition_target_flow[:,0],batch.flow[:,2],capacities,limits).numpy()
        count=int(scoped.transition_cohort_count[0,0]);options=dict(capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(1,),samples=8,chunk=args.chunk,steps=100)
        return batch,z,target,count,options
    gates=[];rows=[];bank_manifest=[];diagnostics=[]
    with torch.inference_mode():
        for item in conditions[:2]:
            batch,z,target,count,options=materialize(item)
            for arm in ('native','zero_mean'):
                with split_prior_start(model.dynamics,source_on=ARMS[arm][0],reference_on=ARMS[arm][1]):
                    actual=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),**options)[1]
                with np.load(bank_index[item['condition_index']]['path']) as saved:
                    np.testing.assert_array_equal(target,saved['target']);np.testing.assert_array_equal(actual,saved[arm])
                gates.append(dict(condition_index=item['condition_index'],arm=arm,cached_diagonal_bit_equal=True))
            atomic_json(args.output/'diagonal_gate.json',gates)
        for item in conditions:
            batch,z,target,count,options=materialize(item);index=item['condition_index']
            with np.load(bank_index[index]['path']) as saved:
                np.testing.assert_array_equal(target,saved['target']);draws={arm:saved[arm].copy() for arm in ('native','zero_mean')}
                coefficients=saved['source_coefficients'].copy()
            action=batch.action[:,0:1].to(device);events=batch.action_events.for_step(0).to(device)
            condition=model.dynamics.encode_condition(action,action_events=events)
            b=model.dynamics.graph_prior(z[:,None],condition);old_ref=model.dynamics.reference_proj(z[:,None]+b);current_ref=model.dynamics.reference_proj(z[:,None])
            diagnostic=dict(condition_index=index,increment_norm=float(torch.linalg.vector_norm(b)),
                increment_rms=float(b.square().mean().sqrt()),increment_relative_to_current=float(torch.linalg.vector_norm(b)/torch.linalg.vector_norm(z).clamp_min(1e-30)),
                reference_token_change_norm=float(torch.linalg.vector_norm(old_ref-current_ref)),
                reference_token_relative_change=float(torch.linalg.vector_norm(old_ref-current_ref)/torch.linalg.vector_norm(current_ref).clamp_min(1e-30)))
            for begin in (0,4):
                rep=z.repeat(4,1,1)[:,None];coeff=torch.as_tensor(coefficients[begin:begin+4],device=device,dtype=z.dtype)
                original_noise=model.dynamics.graph_prior.sample_noise(rep,coeff)
                common=model.dynamics.encode_condition(batch.action[:,0:1].to(device).repeat(4,1,1),action_events=_repeat_events(events,4))
                base_increment=model.dynamics.graph_prior(rep,common)
                for arm,(source_on,reference_on) in ARMS.items():
                    with split_prior_start(model.dynamics,source_on=source_on,reference_on=reference_on):
                        start,_,cached=model.dynamics.prior_start(rep,common,coeff)
                    source=rep+base_increment if source_on else rep
                    ref=rep+base_increment if reference_on else rep
                    torch.testing.assert_close(start,source+original_noise,rtol=0,atol=0)
                    torch.testing.assert_close(cached.reference_tokens,model.dynamics.reference_proj(ref),rtol=0,atol=0)
                    if cached.edge_tokens is not common.edge_tokens:raise ValueError('edgecondition changed')
            diagnostic['same_original_noise_start_and_reference_exact']=True;diagnostics.append(diagnostic)
            for arm in ('source_only','reference_only'):
                with split_prior_start(model.dynamics,source_on=ARMS[arm][0],reference_on=ARMS[arm][1]):
                    draws[arm]=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),**options)[1]
            if digest(model)!=original or ('prior_start' in model.dynamics.__dict__)!=method_was_local:raise ValueError('originalmodel/methodnotrestored')
            for arm in ARMS:
                rows.append(score(draws[arm],target,scales,count,limits.numpy(),dict(item,arm=arm,steps=100,horizon=1,k=8,
                    source_on=ARMS[arm][0],reference_on=ARMS[arm][1])))
            path=args.output/'banks'/f'condition_{index:03d}.npz'
            np.savez_compressed(path,target=target,scales=scales,source_coefficients=coefficients,initial_latent=z.cpu().numpy(),
                native_noise_basis=model.dynamics.graph_prior.basis.cpu().numpy(),native_noise_scale=np.array(model.dynamics.graph_prior.noise_scale),**draws)
            bank_manifest.append(dict(condition_index=index,path=str(path.resolve()),sha256=sha256(path)))
            with (args.output/'states.jsonl').open('w') as stream:
                for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
            atomic_json(args.output/'bank_manifest.json',bank_manifest);atomic_json(args.output/'increment_diagnostics.json',diagnostics)
            progress=dict(stage='cross_mechanisms',conditions_complete=index+1,total_conditions=32,elapsed_seconds=time.time()-started,max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2)
            atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            if index==3:atomic_json(args.output/'pilot_summary.json',dict(**summaries(rows,args.seed),mechanism_contrasts=contrasts(rows,args.seed)))
    for key,path in refs.items():
        if sha256(path)!=hashes[key]:raise ValueError('reference changed')
    for item in conditions:
        if archive_content_hash(Path(item['cache_path']))!=item['cache_content_hash']:raise ValueError('cache changed')
    verify_source();imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if name.startswith(('tools.','urban_dreamer.')) and getattr(module,'__file__',None)}
    if any(not Path(path).is_relative_to(FROZEN_ROOT) for path in imports.values()):raise ValueError('live import')
    atomic_json(args.output/'import_audit.json',imports)
    atomic_json(args.output/'summary.json',dict(status='complete',protocol=protocol,**summaries(rows,args.seed),
        mechanism_contrasts=contrasts(rows,args.seed),model_digest=original,parameters_unchanged=True,method_restored=True,
        source_unchanged=True,checkpoint_unchanged=True,data_unchanged=True,noise_unchanged=True,elapsed_seconds=time.time()-started))
    print(json.dumps(dict(stage='complete',elapsed_seconds=time.time()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','reference','cache-manifest','scale-reference','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--gpu-memory-fraction',type=float,default=.03)
    p.add_argument('--chunk',type=int,default=4);p.add_argument('--seed',type=int,default=20261019)
    run(p.parse_args())
