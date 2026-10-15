"""Small FM residuals with directly observed state/background context.

Reuses immutable H/Y CFM caches; original checkpoint never receives gradients.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import MethodType
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
FROZEN_ROOT=Path(os.environ.get('DIRECT_FM_FROZEN_ROOT',ROOT)).resolve()
sys.path.insert(0,str(FROZEN_ROOT))
import numpy as np
import torch
from torch import nn
from tools.component_experiment_data import atomic_json,load_frozen,sha256
from tools.evaluate_soft_prior_fm import MeanOverridePrior,digest,score
from tools.evaluate_pretrained_distribution import CHECKPOINT_SHA256,trajectory_bank
from tools.evaluate_component_world_models import endpoint_target,stable_seed
from tools.adapt_fm_output_head import capture_final_features
from tools.train_nonlinear_fm_residual import NonlinearResidual,residual_head,study_summary,paired_metrics,stratified_draws
from urban_dreamer.learning.cohort_targets import prepare_transition_targets,prepare_cohort_batch
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.learning.offline_data import archive_content_hash
ARMS=('zero_direct','soft_direct','condition_only')
FAMILY={'zero_direct':'zero_mean','soft_direct':'weak_soft','condition_only':'zero_mean'}


class DirectResidual(nn.Module):
    def __init__(self):
        super().__init__();self.hidden=nn.Linear(449,128);self.output=nn.Linear(128,64)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):return self.output(torch.nn.functional.silu(self.hidden(x)))


def direct_inputs(h,z,hint,relative,t,scales):
    if tuple(h.shape[:-1])!=tuple(z.shape[:-1]) or h.shape[-1]!=256 or z.shape[-1]!=64:raise ValueError('matching H256/Z64 rows required')
    if hint.shape!=z.shape or relative.shape!=z.shape:raise ValueError('context blocks must align')
    tt=t
    while tt.ndim<h.ndim:tt=tt.unsqueeze(-1)
    tt=tt.expand(*h.shape[:-1],1)
    return torch.cat((h,z/scales['z'],hint/scales['b'],relative/scales['relative'],tt),-1)


class DirectVelocityHead(nn.Module):
    def __init__(self,old,new,context,scales):
        super().__init__();self.old=old;self.new=new;self.context=context;self.scales=scales
    def forward(self,h):
        c=self.context
        return self.old(h)+self.new(direct_inputs(h,c['z'],c['hint'],c['xt']-c['mean'],c['time'],self.scales))


@contextmanager
def direct_context(dynamics,residual,weak_prior,arm,scales):
    """Capture macro state and GENERATED x_t, never a true future endpoint."""
    if arm not in ARMS:raise ValueError('unknown arm')
    context={};old_head=dynamics.flow_head;old_start=dynamics.prior_start
    had_start='prior_start' in dynamics.__dict__;instance_start=dynamics.__dict__.get('prior_start')
    def start(this,z,condition,noise=None,**kwargs):
        result=old_start(z,condition,noise,**kwargs);increment=result[1]
        hint=torch.zeros_like(z) if arm=='zero_direct' else weak_prior(z,condition) if arm=='condition_only' else increment
        context.update(z=z,hint=hint,mean=z+increment)
        return result
    def capture(module,args,kwargs):
        context['xt']=kwargs['packed_latents'];context['time']=kwargs['flow_time']
    dynamics.prior_start=MethodType(start,dynamics)
    hook=dynamics.register_forward_pre_hook(capture,with_kwargs=True)
    dynamics.flow_head=DirectVelocityHead(old_head,residual,context,scales)
    try:yield context
    finally:
        hook.remove();dynamics.flow_head=old_head
        if had_start:dynamics.prior_start=instance_start
        else:del dynamics.__dict__['prior_start']


def selection_update(previous,es,epoch):
    result=dict(previous);improved=es<previous['ES']-1e-5
    if improved:result.update(ES=es,epoch=epoch)
    meaningful=es<=previous['reset_ES']*(1-.002)
    if meaningful:result.update(reset_ES=result['ES'],reset_epoch=epoch,patience=0)
    else:result['patience']+=1
    return result,improved,meaningful


def read_parent(parent):
    manifest=json.loads((parent/'feature_manifest.json').read_text());caches={}
    for stage,arms in manifest.items():
        caches[stage]={}
        for arm,entry in arms.items():
            for field in ('x','y'):
                if sha256(Path(entry[field]))!=entry[field+'_sha256']:raise ValueError('parent features changed')
            caches[stage][arm]={key:np.load(entry[key],mmap_mode='r') for key in ('x','y')}
    return manifest,caches


def context_rows(aux,indices,arm,device):
    windows=np.array(aux['window'][indices],copy=True);roads=indices%aux['roads']
    z=torch.from_numpy(np.array(aux['z'][windows,roads],copy=True)).to(device)
    b=torch.zeros_like(z) if arm=='zero_direct' else torch.from_numpy(np.array(aux['b'][windows,roads],copy=True)).to(device)
    relative=torch.from_numpy(np.array(aux['relative'][FAMILY[arm]][indices],copy=True)).to(device)
    t=torch.from_numpy(np.array(aux['time'][indices],copy=True)).to(device)
    return z,b,relative,t


def run(args):
    started=time.time();torch.set_num_threads(args.threads);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    if FROZEN_ROOT!=(args.output/'source').resolve():raise ValueError('new immutable source required')
    if (args.output/'summary.json').exists():raise ValueError('preserve existing study')
    source_manifest=json.loads((args.output/'source_manifest.json').read_text())
    def source_gate():
        for name,h in source_manifest.items():
            if sha256(FROZEN_ROOT/name)!=h:raise ValueError('source changed')
    source_gate();split=json.loads(args.split.read_text());parent_split=json.loads((args.parent/'split.json').read_text())
    for key in ('train','validation'):
        if split[key]!=parent_split[key]:raise ValueError('parent feature selection/order must match')
    strip_index=lambda items:[{k:v for k,v in item.items() if k!='episode_index'} for item in items]
    if strip_index(split['validation_conditions'])!=strip_index(parent_split['validation_conditions']):raise ValueError('validation identity/seed/order changed')
    refs={'checkpoint':args.experiment/'snapshot/checkpoints/latest.pt','split':args.split,'cache_manifest':Path(split['cache_manifest_path']),
        'parent_features':args.parent/'feature_manifest.json','parent_draws':args.parent/'feature_draw_schedule.json','parent_source':args.parent/'source_manifest.json',
        'weak_head':args.weak_head,'basis':args.basis,'known_banks':args.known_reference/'bank_manifest.json','known_conditions':args.known_reference/'conditions.json',
        'known_cache':args.known_cache,'scales':args.scale_reference,'old_zero_head':args.parent/'checkpoints/best_zero_mean.pt','old_soft_head':args.parent/'checkpoints/best_weak_soft.pt'}
    hashes={k:sha256(v) for k,v in refs.items()}
    if hashes['checkpoint']!=CHECKPOINT_SHA256:raise ValueError('completed checkpoint required')
    groups=[{e['demand_group'] for e in split[k]} for k in ('train','validation','test')]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise ValueError('split overlap')
    protocol=dict(status='prespecified_before_fit',reference_hashes=hashes,seed=args.seed,arms=list(ARMS),
        network='449->128SiLU->64,65856params,lastlayerzeros; H256/Z64/hint64/(GENERATEDXt-sourceMean)64/time1',
        priors='zero_direct zero source/ref and zero hint; soft_direct weak matchedHYBRIDtau.1 source/ref/hint; condition_only zero source/ref with sameweak hint',
        controls='FULLcurrenttraffic/FULLfactualactionevents/HYBRIDCt endpoint; source noiseoriginalrank16 exactpaired draws',
        normalizers='scalar training-only uncenteredRMS: Z, weakb, pooledzero+softrelative; H unchanged; zero hint remainszero',
        caches='reuse all oldH/Y immutable; newcompactperwindow Z/b and parentrow-relative/time/windowindex; NOnewTransformerforwardcache',
        training=dict(minimum_epochs=20,maximum_epochs=120,batch_rows=32768,optimizer='AdamW',lr=.001,weight_decay=.0001,clip_norm=1,
            best_epsilon=1e-5,patience_checks=3,meaningful_reset_relative=.002,stop='ALL3checkswithoutmeaningfulESimprovement; sameepoch/updatebudget'),
        validation=f'epoch0 candidate and8samefixedstates4groupsK4Euler{args.steps} at20,25,...120; independentbestvalES',
        test='32states16NEWblindgroups afterselection;3newarms+2oldH-onlybestheads descriptive references',
        optional_h4=f'eachpriorarm eligible vszero ifES+bothoccupiedCRPS CIupper<0 andtotalcountpoint<=0; ifany qualifies fixedfirst8teststatesC0 H4Euler{args.steps}K8',
        freeze='alloriginalCP/body/codec/prior/action/ref/Linearflowhead frozen; onlynewg receivesgradients',
        gpu_memory_fraction=args.gpu_memory_fraction,device=args.device,precision=f'FP32noAMP/compile CPU/BLAS{args.threads}',integration_steps=args.steps,screening=args.steps!=100,
        initial_gate='same-backend original vs zero residual exact; archived bit identity additionally required onlyCUDA100',
        cache_compatibility='firsttraininggroup/firstquarter bothfamilies H/Y allrows: rtol1e-4 atol1e-4 beforeSGD; failclosed',
        limitations=['Onefrozenbackbone/headseed; notfulljointtraining orpolicybenefit.','maxcapbudgetlimited; no fullmodelconvergenceclaim.'])
    atomic_json(args.output/'protocol.json',protocol)
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
        if free<args.gpu_memory_fraction*total+256*1024**2:raise RuntimeError('insufficient GPU headroom')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)
    config,prepared,topology,model,_=load_frozen(args.experiment,device=device);model.eval().requires_grad_(False);model.set_online_chunk_profile()
    native=model.dynamics.graph_prior;original=digest(model)
    capacities=torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1);limits=torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
    with np.load(args.basis) as a:basis=a['basis'];values=a['eigenvalues']
    with np.load(args.weak_head) as a:coefficients=a['coefficients']
    weak=MeanOverridePrior(native,operator=(basis*(1/(1+.1*values)))@basis.T,coefficients=coefficients,state_only=True)
    priors={'zero_mean':MeanOverridePrior(native,zero=True),'weak_soft':weak};heads={}
    for arm in ARMS:torch.manual_seed(args.seed);heads[arm]=DirectResidual().to(device)
    if sum(p.numel() for p in heads[ARMS[0]].parameters())!=65856:raise ValueError('wrongheadsize')
    for arm in ARMS[1:]:
        for k,v in heads[ARMS[0]].state_dict().items():torch.testing.assert_close(v,heads[arm].state_dict()[k],rtol=0,atol=0)
    scales=np.asarray(json.loads(args.scale_reference.read_text())['scales']['scales'])
    datasets={k:CompactPretrainDataset(config,prepared,topology,5 if k=='validation' else 1,cache_manifest_path=refs['cache_manifest'],split=k) for k in ('train','validation')}
    # Exact no-op initial forecast gate before any SGD.
    known=CompactPretrainDataset(config,prepared,topology,5,cache_manifest_path=args.known_cache,split='validation')
    known_items=json.loads((args.known_reference/'conditions.json').read_text())[:2]
    known_banks={b['condition_index']:b for b in json.loads((args.known_reference/'bank_manifest.json').read_text())};gates=[]
    with torch.inference_mode():
        for item in known_items:
            batch=known.batch_from_starts(item['episode_index'],[item['start']],fm_only=False);z=model.encode(batch.flow[:,0].to(device))
            bank=known_banks[item['condition_index']]
            if sha256(Path(bank['path']))!=bank['sha256']:raise ValueError('oldbank changed')
            with np.load(bank['path']) as saved:
                for arm in ARMS:
                    model.dynamics.graph_prior=priors[FAMILY[arm]]
                    baseline=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(1,),samples=8,chunk=4,steps=args.steps)[1]
                    with direct_context(model.dynamics,heads[arm],weak,arm,dict(z=1.,b=1.,relative=1.)):
                        draws=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(1,),samples=8,chunk=4,steps=args.steps)[1]
                    np.testing.assert_array_equal(draws,baseline)
                    archive_exact=device.type=='cuda' and args.steps==100
                    if archive_exact:np.testing.assert_array_equal(draws,saved['hybrid_state_soft0.1' if FAMILY[arm]=='weak_soft' else 'zero_mean'])
                    gates.append(dict(condition=item['condition_index'],arm=arm,zero_head_exact_old=True,archived_bit_identity_checked=archive_exact,device=str(device),steps=args.steps))
    model.dynamics.graph_prior=native
    if digest(model)!=original or 'prior_start' in model.dynamics.__dict__ or model.dynamics._forward_pre_hooks:raise ValueError('gate restoration failed')
    atomic_json(args.output/'initial_gates.json',gates)
    parent_manifest,caches=read_parent(args.parent);aux={};cache_dir=args.output/'context';cache_dir.mkdir(exist_ok=True);roads=len(basis);schedule=[];rms_sums=dict(z=0.,b=0.,relative=0.);rms_counts=dict(z=0,b=0,relative=0);ancillary_started=time.time()
    windows=sum(len(e['starts']) for stage in ('train','validation') for e in split[stage])
    if shutil.disk_usage(args.output).free<windows*roads*(128*4+4*(128*4+8))+4*1024**3:raise RuntimeError('insufficient ancillary disk reserve')
    with torch.inference_mode():
        for stage in ('train','validation'):
            entries=split[stage];data=datasets[stage];indices={e['cache_path']:i for i,e in enumerate(data.manifest['entries'])}
            nwindows=sum(len(e['starts']) for e in entries);rows=nwindows*4*roads
            if any(len(caches[stage][family]['x'])!=rows for family in priors):raise ValueError('parent row count mismatch')
            def mapped(name,shape,dtype=np.float32):return np.lib.format.open_memmap(cache_dir/f'{stage}_{name}.npy',mode='w+',dtype=dtype,shape=shape)
            a=aux[stage]=dict(z=mapped('z',(nwindows,roads,64)),b=mapped('b',(nwindows,roads,64)),window=mapped('window',(rows,),np.int32),time=mapped('time',(rows,)),relative={family:mapped('relative_'+family,(rows,64)) for family in priors},roads=roads)
            offset=0;window_offset=0
            for ordinal,e in enumerate(entries):
                starts=e['starts'];batch=data.batch_from_starts(indices[e['cache_path']],starts,materialization_horizon=1,fm_only=True);target_batch=prepare_transition_targets(batch,capacities=capacities)
                z=model.encode(batch.flow[:,0].to(device))[:,None];target=model.encode(target_batch.transition_target_flow[:,0].to(device))[:,None]
                condition=model.dynamics.encode_condition(batch.action.to(device),action_events=batch.action_events.to(device));b=weak(z,condition)
                a['z'][window_offset:window_offset+len(starts)]=z[:,0].cpu().numpy();a['b'][window_offset:window_offset+len(starts)]=b[:,0].cpu().numpy()
                if stage=='train' and ordinal==0:
                    times,noise,_=stratified_draws(args.seed,stage,e,starts,0,native.rank);times=times.to(device);noise=noise.to(z);compatibility=[]
                    for family,prior in priors.items():
                        model.dynamics.graph_prior=prior;source,_,cached=model.dynamics.prior_start(z,condition,noise)
                        interpolated=(1-times[:,:,None,None])*source+times[:,:,None,None]*target
                        features,out=capture_final_features(model.dynamics.flow_head,lambda:model.dynamics(batch.action.to(device),packed_latents=interpolated,flow_time=times,condition=cached))
                        for key,value in [('x',features),('y',target-source-out.latent_velocity)]:
                            actual=value.cpu().numpy().reshape(-1,256 if key=='x' else 64);expected=caches[stage][family][key][:len(actual)]
                            np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-4)
                            compatibility.append(dict(family=family,field=key,rows=len(actual),max_absolute_error=float(np.max(np.abs(actual-expected))),rtol=1e-4,atol=1e-4))
                    model.dynamics.graph_prior=native;atomic_json(args.output/'parent_cache_compatibility.json',compatibility)
                if stage=='train':
                    for key,value in [('z',z),('b',b)]:rms_sums[key]+=float(value.double().square().sum());rms_counts[key]+=value.numel()
                for quarter in range(4):
                    times,noise,records=stratified_draws(args.seed,stage,e,starts,quarter,native.rank);schedule.append(dict(stage=stage,cache_path=e['cache_path'],quarter=quarter,draws=records))
                    times=times.to(device);noise=noise.to(z);n=len(starts)*roads
                    a['window'][offset:offset+n]=np.repeat(np.arange(window_offset,window_offset+len(starts)),roads);a['time'][offset:offset+n]=np.repeat(times.cpu().numpy().reshape(-1),roads)
                    for family,prior in priors.items():
                        mean=z+(b if family=='weak_soft' else 0);source=mean+prior.sample_noise(z,noise);t=times[:,:,None,None]
                        relative=(1-t)*source+t*target-mean;a['relative'][family][offset:offset+n]=relative.cpu().numpy().reshape(-1,64)
                        if stage=='train':rms_sums['relative']+=float(relative.double().square().sum());rms_counts['relative']+=relative.numel()
                    offset+=n
                window_offset+=len(starts)
                print(json.dumps(dict(stage='ancillary_cache',split=stage,groups=ordinal+1,total=len(entries),elapsed_seconds=time.time()-ancillary_started)),flush=True)
            if offset!=rows or window_offset!=nwindows:raise ValueError('context index mismatch')
            for key in ('z','b','window','time'):a[key].flush()
            for value in a['relative'].values():value.flush()
    if schedule!=json.loads((args.parent/'feature_draw_schedule.json').read_text()):raise ValueError('parent time/noise schedule differs')
    normalizers={key:max(np.sqrt(rms_sums[key]/rms_counts[key]),1e-8) for key in rms_sums}
    atomic_json(args.output/'normalizers.json',dict(scales=normalizers,sums=rms_sums,counts=rms_counts,fit_split='trainonly',centering=False))
    atomic_json(args.output/'context_manifest.json',{str(p.name):sha256(p) for p in cache_dir.glob('*.npy')})
    old_heads={}
    for arm,key in [('old_zero','old_zero_head'),('old_soft','old_soft_head')]:
        h=NonlinearResidual().to(device);h.load_state_dict(torch.load(refs[key],map_location=device,weights_only=True)['state']);old_heads[arm]=h.eval().requires_grad_(False)
    outcheck=args.output/'checkpoints';outcheck.mkdir(exist_ok=True);valdir=args.output/'validation';valdir.mkdir(exist_ok=True)
    def forecast(items,variants,k,prefix):
        rows=[];banks=[]
        with torch.inference_mode():
            for ordinal,item in enumerate(items):
                data=datasets['validation'];idx=item['episode_index']
                if data.manifest['entries'][idx]['cache_path']!=item['cache_path']:raise ValueError('condition index changed')
                batch=data.batch_from_starts(idx,[item['start']],fm_only=False);target=prepare_transition_targets(batch,capacities=capacities);z=model.encode(batch.flow[:,0].to(device))
                truth=endpoint_target(target.transition_target_flow[:,0],batch.flow[:,2],capacities,limits).numpy();stored=dict(target=truth,scales=scales,initial_latent=z.cpu().numpy())
                for label,(family,head,kind) in variants.items():
                    model.dynamics.graph_prior=priors[family]
                    ctx=direct_context(model.dynamics,head,weak,label,normalizers) if kind=='direct' else residual_head(model.dynamics,head)
                    with ctx:draws=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(1,),samples=k,chunk=4,steps=args.steps)[1]
                    stored[label]=draws;rows.append(score(draws,truth,scales,int(target.transition_cohort_count[0,0]),limits.numpy(),dict(item,arm=label,episode=str(item['source_path'])+'::'+str(item['episode']),steps=args.steps,horizon=1,k=k)))
                model.dynamics.graph_prior=native
                if digest(model)!=original:raise ValueError('forecast changed original model')
                path=Path(str(prefix)+f'_{ordinal:03d}.npz');np.savez_compressed(path,**stored);banks.append(dict(condition_index=item['condition_index'],path=str(path.resolve()),sha256=sha256(path)))
        return rows,banks
    def variants():return {a:(FAMILY[a],heads[a],'direct') for a in ARMS}
    zero_rows,zero_banks=forecast(split['validation_conditions'],variants(),4,valdir/'epoch000');best={}
    for arm in ARMS:
        es=float(np.mean([r['energy_score'] for r in zero_rows if r['arm']==arm]));best[arm]=dict(ES=es,epoch=0,reset_ES=es,reset_epoch=0,patience=0)
        torch.save(dict(epoch=0,validation_ES=es,state={k:v.detach().cpu().clone() for k,v in heads[arm].state_dict().items()}),outcheck/f'best_{arm}.pt')
    atomic_json(valdir/'epoch000_summary.json',dict(**study_summary(zero_rows,args.seed),rows=zero_rows,banks=zero_banks))
    optimizers={a:torch.optim.AdamW(heads[a].parameters(),lr=.001,weight_decay=.0001) for a in ARMS};history=[];stop_reason='maximum_epoch_cap'
    def batch_inputs(stage,arm,index):
        family=FAMILY[arm];h=torch.from_numpy(np.array(caches[stage][family]['x'][index],copy=True)).to(device)
        z,b,r,t=context_rows(aux[stage],index,arm,device)
        return direct_inputs(h,z,b,r,t,normalizers),torch.from_numpy(np.array(caches[stage][family]['y'][index],copy=True)).to(device)
    def feature_mse(arm):
        total=0.;n=len(caches['validation'][FAMILY[arm]]['x']);heads[arm].eval()
        with torch.no_grad():
            for start in range(0,n,32768):
                x,y=batch_inputs('validation',arm,np.arange(start,min(start+32768,n)));total+=float((heads[arm](x)-y).square().sum())
        return total/(n*64)
    for epoch in range(1,121):
        epoch_started=time.time();n=len(caches['train']['zero_mean']['x']);order=np.random.default_rng(stable_seed(args.seed,'training_order',epoch)).permutation(n);losses={a:0. for a in ARMS};updates=0
        for start in range(0,n,32768):
            index=order[start:start+32768]
            for arm in ARMS:
                heads[arm].train();x,y=batch_inputs('train',arm,index);optim=optimizers[arm];optim.zero_grad(set_to_none=True);loss=(heads[arm](x)-y).square().mean()
                if not bool(torch.isfinite(loss)):raise ValueError('nonfinite trainloss')
                loss.backward();torch.nn.utils.clip_grad_norm_(heads[arm].parameters(),1.);optim.step();losses[arm]+=float(loss.detach())*len(index)
            updates+=1
        row=dict(epoch=epoch,updates_per_arm=updates,train_mse={a:losses[a]/n for a in ARMS},validation_feature_mse={a:feature_mse(a) for a in ARMS},epoch_seconds=time.time()-epoch_started)
        if epoch==1:atomic_json(args.output/'epoch_timing_pilot.json',dict(full_epoch=True,continued_exact_state=True,extra_updates=0,**row))
        if epoch>=20 and epoch%5==0:
            vr,vb=forecast(split['validation_conditions'],variants(),4,valdir/f'epoch{epoch:03d}');row['integrated_validation']={}
            for arm in ARMS:
                es=float(np.mean([r['energy_score'] for r in vr if r['arm']==arm]));best[arm],improved,meaningful=selection_update(best[arm],es,epoch)
                row['integrated_validation'][arm]=dict(ES=es,best_update=improved,meaningful_reset=meaningful,selection=best[arm])
                if improved:torch.save(dict(epoch=epoch,validation_ES=es,state={k:v.detach().cpu().clone() for k,v in heads[arm].state_dict().items()}),outcheck/f'best_{arm}.pt')
                torch.save(dict(epoch=epoch,state=heads[arm].state_dict(),optimizer=optimizers[arm].state_dict()),outcheck/f'last_{arm}.pt')
            atomic_json(valdir/f'epoch{epoch:03d}_summary.json',dict(**study_summary(vr,args.seed),rows=vr,banks=vb))
        history.append(row);atomic_json(args.output/'training_history.json',history)
        progress=dict(stage='training',epoch=epoch,best=best,elapsed_seconds=time.time()-started,max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0.)
        atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if epoch>=20 and all(best[a]['patience']>=3 for a in ARMS):stop_reason='all_meaningful_ES_patience3';break
    if digest(model)!=original or any(p.grad is not None for p in model.parameters()):raise ValueError('old model gradient/update')
    for a in ARMS:heads[a].load_state_dict(torch.load(outcheck/f'best_{a}.pt',map_location=device,weights_only=True)['state']);heads[a].eval()
    atomic_json(args.output/'validation_selection.json',dict(status='frozen_before_new_blindtest',best=best,stop_reason=stop_reason,actual_equal_epochs=epoch,updates_per_arm=sum(r['updates_per_arm'] for r in history),budget_limited=stop_reason=='maximum_epoch_cap'))
    testdir=args.output/'test';testdir.mkdir(exist_ok=True);allvariants=variants();allvariants.update(old_zero=('zero_mean',old_heads['old_zero'],'old'),old_soft=('weak_soft',old_heads['old_soft'],'old'))
    testrows,testbanks=forecast(split['test_conditions'],allvariants,8,testdir/'condition')
    pairs={name:paired_metrics(testrows,a,b,args.seed) for name,a,b in [('soft_vs_zero','soft_direct','zero_direct'),('hint_vs_zero','condition_only','zero_direct'),('shift_vs_hint','soft_direct','condition_only'),('direct_zero_vs_old','zero_direct','old_zero'),('direct_soft_vs_old','soft_direct','old_soft')]}
    atomic_json(testdir/'states.json',testrows);atomic_json(testdir/'bank_manifest.json',testbanks)
    eligible=[]
    for arm,key in [('soft_direct','soft_vs_zero'),('condition_only','hint_vs_zero')]:
        p=pairs[key]
        if all(p[m]['ci_high'] is not None and p[m]['ci_high']<0 for m in ('energy_score','count_crps_occupied','speed_crps_occupied')) and p['total_count_absolute_error']['estimate']<=0:eligible.append(arm)
    atomic_json(args.output/'h4_gate.json',dict(prespecified=True,eligible=eligible,pairs={k:pairs[k] for k in ('soft_vs_zero','hint_vs_zero')},tuning_after_test=False))
    if eligible:
        h4rows=[];h4banks=[]
        with torch.inference_mode():
            for ordinal,item in enumerate(split['test_conditions'][:8]):
                batch=datasets['validation'].batch_from_starts(item['episode_index'],[item['start']],fm_only=False);cohort=prepare_cohort_batch(batch,capacities=capacities)
                z=model.encode(cohort.transition_flow[:,0].to(device));truth=endpoint_target(cohort.transition_flow[:,4],batch.flow[:,5],capacities,limits).numpy();stored=dict(target=truth,scales=scales)
                for arm in ['zero_direct']+eligible:
                    model.dynamics.graph_prior=priors[FAMILY[arm]]
                    with direct_context(model.dynamics,heads[arm],weak,arm,normalizers):draws=trajectory_bank(model,z,cohort.transition_action.to(device),cohort.transition_action_events.to(device),capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(4,),samples=8,chunk=4,steps=args.steps)[4]
                    stored[arm]=draws;h4rows.append(score(draws,truth,scales,int(cohort.cohort_count[0]),limits.numpy(),dict(item,arm=arm,episode=str(item['source_path'])+'::'+str(item['episode']),steps=args.steps,horizon=4,k=8)))
                model.dynamics.graph_prior=native;path=args.output/f'h4_{ordinal:03d}.npz';np.savez_compressed(path,**stored);h4banks.append(dict(path=str(path.resolve()),sha256=sha256(path)))
        atomic_json(args.output/'h4_summary.json',dict(**study_summary(h4rows,args.seed),rows=h4rows,banks=h4banks,diagnostic='C0recordedcontrolautoregressive'))
    for name,path in refs.items():
        if sha256(path)!=hashes[name]:raise ValueError('reference input changed')
    for stage,arms in parent_manifest.items():
        for family,entry in arms.items():
            for key in ('x','y'):
                if sha256(Path(entry[key]))!=entry[key+'_sha256']:raise ValueError('parent feature altered')
    for stage in ('train','validation','test'):
        for e in split[stage]:
            if archive_content_hash(Path(e['cache_path']))!=e['cache_content_hash']:raise ValueError('data cache altered')
    source_gate()
    if digest(model)!=original or 'prior_start' in model.dynamics.__dict__ or model.dynamics._forward_pre_hooks:raise ValueError('old model/context not restored')
    imports={name:str(Path(mod.__file__).resolve()) for name,mod in sys.modules.items() if name.startswith(('tools.','urban_dreamer.')) and getattr(mod,'__file__',None)}
    if any(not Path(p).is_relative_to(FROZEN_ROOT) for p in imports.values()):raise ValueError('live runtime import')
    atomic_json(args.output/'import_audit.json',imports)
    atomic_json(args.output/'summary.json',dict(status='complete',test=study_summary(testrows,args.seed),test_pairs=pairs,best=best,normalizers=normalizers,
        original_model_digest=original,original_model_unchanged=True,parent_features_unchanged=True,source_unchanged=True,data_unchanged=True,only_new_parameters_per_arm=65856,
        actual_equal_epochs=epoch,updates_per_arm=sum(r['updates_per_arm'] for r in history),stop_reason=stop_reason,budget_limited=stop_reason=='maximum_epoch_cap',H4_eligible=eligible,elapsed_seconds=time.time()-started))
    print(json.dumps(dict(stage='complete',elapsed_seconds=time.time()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','split','parent','weak-head','basis','known-reference','known-cache','scale-reference','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--steps',type=int,default=100);
    p.add_argument('--device',default='cuda:0');p.add_argument('--gpu-memory-fraction',type=float,default=.04);p.add_argument('--seed',type=int,default=20261021)
    run(p.parse_args())
