"""Continue C2 with complete AdamW state; test only after validation plateaus."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
FROZEN_ROOT=Path(os.environ.get('PARTIAL_FM_FROZEN_ROOT',ROOT)).resolve()
sys.path.insert(0,str(FROZEN_ROOT))
import numpy as np
import torch
from tools.run_partial_context_fm import (read,load_selected,frozen_digest,configure_trainable,
    selected_state,DirectResidual,MeanOverridePrior,cfm_loss,gradient_proof,atomic_json,
    load_frozen,sha256,CHECKPOINT_SHA256,CompactPretrainDataset,trajectory_bank,
    endpoint_target,score,study_summary,paired_metrics,stable_seed,prepare_transition_targets,partial_context,archive_content_hash)


def plateau_step(state, value, epoch):
    if not math.isfinite(value):raise ValueError('finite validation ES required')
    state=copy.deepcopy(state)
    improved=value<state['best_ES']-1e-5
    if improved:state.update(best_ES=value,best_epoch=epoch)
    meaningful=value<=state['anchor_ES']*.998
    state['patience']=0 if meaningful else state['patience']+1
    if meaningful:state.update(anchor_ES=state['best_ES'],anchor_epoch=epoch)
    event='meaningful_improvement' if meaningful else 'waiting'
    if state['patience']>=5:
        if state['reductions']<2:
            state['reductions']+=1;state['lrs']=[lr*.3 for lr in state['lrs']]
            state.update(patience=0,anchor_ES=state['best_ES'],anchor_epoch=epoch)
            event='reduce_lr'
        else:state['converged']=True;event='validation_plateau_at_min_lr'
    return state,improved,event


def restore_optimizer(model,head,payload):
    parameters=configure_trainable(model,head)
    opt=torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.requires_grad],lr=.0001),
        dict(params=list(head.parameters()),lr=.001)],weight_decay=.0001)
    opt.load_state_dict(payload['optimizer'])
    if [len(g['params']) for g in opt.param_groups]!=[18,4]:raise ValueError('optimizer parameter order differs')
    for state in opt.state.values():
        if state['step'].device.type!='cpu':raise ValueError('AdamW noncapturable steps must remain CPU scalars')
    return parameters,opt


def assert_nested_equal(actual, expected):
    if isinstance(expected,torch.Tensor):
        torch.testing.assert_close(actual.detach().cpu(),expected.detach().cpu(),rtol=0,atol=0)
    elif isinstance(expected,dict):
        if actual.keys()!=expected.keys():raise AssertionError('state keys differ')
        for key in expected:assert_nested_equal(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        if len(actual)!=len(expected):raise AssertionError('state lengths differ')
        for a,b in zip(actual,expected):assert_nested_equal(a,b)
    elif actual!=expected:raise AssertionError('scalar state differs')


def save_checkpoint(path,payload):
    temporary=path.with_suffix('.tmp');torch.save(payload,temporary);os.replace(temporary,path)


def run(args):
    started=time.time();torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if FROZEN_ROOT!=(args.output/'source').resolve():raise ValueError('freeze new source first')
    if (args.output/'training_history.json').exists() or (args.output/'summary.json').exists():raise ValueError('preserve existing run')
    source_manifest=read(args.output/'source_manifest.json')
    def source_gate():
        for name,h in source_manifest.items():
            if sha256(FROZEN_ROOT/name)!=h:raise ValueError('frozen source changed')
    source_gate()
    launch=read(args.parent/'launch.json')['argv'];options={launch[i]:launch[i+1] for i in range(len(launch)-1) if launch[i].startswith('--')}
    experiment=Path(options['--experiment']);split=read(args.split);parent_split=read(args.parent/'split.json')
    for key in ('train','validation'):
        if split[key]!=parent_split[key]:raise ValueError('training and validation selections must stay fixed')
    stripped=lambda items:[{k:v for k,v in i.items() if k!='episode_index'} for i in items]
    if stripped(split['validation_conditions'])!=stripped(parent_split['validation_conditions']):raise ValueError('validation states changed')
    groups=[{e['demand_group'] for e in split[s]} for s in ('train','validation','test')]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise ValueError('split overlap')
    if groups[2]&{e['demand_group'] for e in parent_split['test']}:raise ValueError('previous test reused')
    resume=args.resume or args.parent/'checkpoints/last_C2.pt'
    refs=dict(checkpoint=experiment/'snapshot/checkpoints/latest.pt',resume=resume,split=args.split,
        data_manifest=Path(split['cache_manifest_path']),parent_protocol=args.parent/'protocol.json',
        parent_cache=args.parent/'frozen_input_manifest.json',normalizers=args.parent/'normalizers.json',
        parent_selection=args.parent/'validation_selection.json',basis=Path(options['--basis']),weak_head=Path(options['--weak-head']),
        scales=Path(options['--scale-reference']),reference_B=args.parent/'checkpoints/best_B.pt',reference_C2=args.parent/'checkpoints/last_C2.pt')
    hashes={k:sha256(p) for k,p in refs.items()}
    if hashes['checkpoint']!=CHECKPOINT_SHA256:raise ValueError('completed pretrain required')
    # CPU loading preserves noncapturable AdamW scalar-step device placement.
    payload=torch.load(resume,map_location='cpu',weights_only=True)
    first_epoch=int(payload['epoch'])
    if args.maximum_epoch<=first_epoch:raise ValueError('maximum epoch must exceed resume epoch')
    initial_steps={int(float(v['step'])) for v in payload['optimizer']['state'].values()}
    if initial_steps!={first_epoch*1536}:raise ValueError('optimizer steps do not match global epoch')
    protocol=dict(status='prespecified_before_continuation',reference_hashes=hashes,seed=20261021,
        resume_epoch=first_epoch,resume_updates=first_epoch*1536,variant='C2 source/refZ+b, same frozen backgroundhint',
        trainable_parameters=1414528,training='same3072cachedwholegraphviews,B2,originalglobalepochshuffle;fullvelocityCFM',
        controller=dict(patience=5,meaningful_relative_improvement=.002,best_epsilon=1e-5,lr_factor=.3,lr_reductions=2,maximum_global_epoch=args.maximum_epoch),
        stopping='fivecheckswithoutmeaningfulimprovement AFTERtworeductions; cap isbudgetlimited and leavesnewtestclosed',
        validation='same8states4groups,K4Euler100 everyepoch; parentepoch4candidateeligible',
        test='new32states16groups onlyafterplateau;K8Euler100;B_parent,C2_epoch4,C2_converged',
        fairness='C2extended has greater optimization budget; references unchanged, NOTequal-budget superiorityproof',
        precision='FP32noAMP/noTF32/noCompile;evalmode,gradientthroughfrozenprefix',device=args.device,gpu_memory_fraction=args.gpu_memory_fraction,
        resume_forward_gate=dict(conditions=2,rtol=1e-5,atol=1e-5),
        caveats=['Validationplateau ofthispartialparameterfamily, notglobaloptimum orfulljointconvergence.','Onecheckpoint/adaptationseed; no policy benefit claim.'])
    atomic_json(args.output/'protocol.json',protocol)
    cache_manifest=read(refs['parent_cache']);cache_dir=args.parent/'frozen_inputs'
    for name,h in cache_manifest.items():
        if sha256(cache_dir/name)!=h:raise ValueError('parent cache changed')
    arrays={p.stem:np.load(p,mmap_mode='r') for p in (cache_dir/name for name in cache_manifest)}
    cache={k:arrays[k] for k in ('z','target','b','condition','action')};mapping=arrays['example_window']
    if len(mapping)!=3072:raise ValueError('expected3072pairedviews')
    scales_context=read(refs['normalizers'])['scales'];scales=np.asarray(read(refs['scales'])['scales']['scales'])
    device=torch.device(args.device);torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
    if free<args.gpu_memory_fraction*total+256*1024**2:raise RuntimeError('insufficient GPU headroom')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)
    config,prepared,topology,model,_=load_frozen(experiment,device=device);model.eval().requires_grad_(False);model.set_online_chunk_profile()
    native=model.dynamics.graph_prior
    with np.load(refs['basis']) as a:basis,eigenvalues=a['basis'],a['eigenvalues']
    with np.load(refs['weak_head']) as a:coefficients=a['coefficients']
    model.dynamics.graph_prior=MeanOverridePrior(native,operator=(basis*(1/(1+.1*eigenvalues)))@basis.T,coefficients=coefficients,state_only=True)
    head=DirectResidual().to(device);load_selected(model,head,payload['state']);parameters,opt=restore_optimizer(model,head,payload)
    for kind in ('model','residual'):
        actual=selected_state(model,head)[kind]
        for name,v in payload['state'][kind].items():torch.testing.assert_close(actual[name],v,rtol=0,atol=0)
    assert_nested_equal(opt.state_dict(),payload['optimizer'])
    frozen=frozen_digest(model);normal=read(refs['parent_selection'])['best']['C2']
    controller=copy.deepcopy(payload.get('controller',dict(best_ES=normal['ES'],best_epoch=normal['epoch'],anchor_ES=normal['reset_ES'],anchor_epoch=normal['reset_epoch'],patience=normal['patience'],reductions=0,lrs=[g['lr'] for g in opt.param_groups],converged=False)))
    if controller['converged']:raise ValueError('checkpoint already reached validation plateau')
    if [g['lr'] for g in opt.param_groups]!=controller['lrs']:raise ValueError('scheduler and optimizer disagree')
    if 'best_state' in payload:best_state=copy.deepcopy(payload['best_state'])
    else:best_state=copy.deepcopy(torch.load(args.parent/'checkpoints/best_C2.pt',map_location='cpu',weights_only=True)['state'])
    history=copy.deepcopy(payload.get('history',[]));capacities=torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1)
    limits=torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
    data=CompactPretrainDataset(config,prepared,topology,5,cache_manifest_path=refs['data_manifest'],split='validation')
    valdir=args.output/'validation';valdir.mkdir(exist_ok=True);check=args.output/'checkpoints';check.mkdir(exist_ok=True)
    def unchanged():
        if frozen_digest(model)!=frozen or 'prior_start' in model.dynamics.__dict__ or model.dynamics._forward_pre_hooks:raise ValueError('frozen state or context changed')
    def forecast(items,k,directory,variants=None):
        directory.mkdir(exist_ok=True);rows=[];banks=[]
        with torch.no_grad():
            for ordinal,item in enumerate(items):
                if data.manifest['entries'][item['episode_index']]['cache_path']!=item['cache_path']:raise ValueError('condition identity changed')
                batch=data.batch_from_starts(item['episode_index'],[item['start']],fm_only=False);scoped=prepare_transition_targets(batch,capacities=capacities)
                truth=endpoint_target(scoped.transition_target_flow[:,0],batch.flow[:,2],capacities,limits).numpy();z=model.encode(batch.flow[:,0].to(device));stored=dict(target=truth,scales=scales)
                for label,(arm,state) in (variants or {'C2':('C2',None)}).items():
                    if state is not None:load_selected(model,head,state)
                    with partial_context(model.dynamics,head,arm,scales_context):
                        draws=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(1,),samples=k,chunk=4,steps=100)[1]
                    stored[label]=draws;rows.append(score(draws,truth,scales,int(scoped.transition_cohort_count[0,0]),limits.numpy(),dict(item,arm=label,episode=str(item['source_path'])+'::'+str(item['episode']),steps=100,horizon=1,k=k)))
                path=directory/f'condition_{ordinal:03d}.npz';np.savez_compressed(path,**stored);banks.append(dict(condition_index=item['condition_index'],path=str(path.resolve()),sha256=sha256(path)))
        unchanged();atomic_json(directory/'states.json',rows);atomic_json(directory/'bank_manifest.json',banks)
        return rows,banks
    old_banks=payload.get('validation_banks') or read(args.parent/f'validation/epoch{first_epoch:03d}/bank_manifest.json')
    gate_rows,gate_banks=forecast(split['validation_conditions'][:2],4,valdir/'resume_gate')
    old_index={b['condition_index']:b for b in old_banks};gates=[]
    for new in gate_banks:
        old=old_index[new['condition_index']]
        if sha256(Path(old['path']))!=old['sha256']:raise ValueError('parent bank changed')
        with np.load(new['path']) as actual,np.load(old['path']) as expected:
            np.testing.assert_array_equal(actual['target'],expected['target']);np.testing.assert_allclose(actual['C2'],expected['C2'],rtol=1e-5,atol=1e-5)
            gates.append(dict(condition_index=new['condition_index'],max_abs=float(np.abs(actual['C2']-expected['C2']).max()),bit_exact=bool(np.array_equal(actual['C2'],expected['C2']))))
    atomic_json(args.output/'resume_gates.json',dict(epoch=first_epoch,optimizer_steps=first_epoch*1536,optimizer_tensors=len(opt.state),checkpoint_weights_bitexact=True,optimizer_state_bitexact=True,step_tensors_cpu=True,forward=gates))
    atomic_json(args.output/'initial_controller.json',controller);reason='maximum_epoch_cap';gradient=None
    for epoch in range(first_epoch+1,args.maximum_epoch+1):
        begin=time.time();order=np.random.default_rng(stable_seed(20261021,'partial_training_order',epoch)).permutation(len(mapping));loss_sum=0.;updates=0;lrs_used=[g['lr'] for g in opt.param_groups]
        for start in range(0,len(order),2):
            ix=order[start:start+2];wi=mapping[ix];batch={k:torch.from_numpy(np.array(v[wi],copy=True)).to(device) for k,v in cache.items()}
            batch.update(time=torch.from_numpy(np.array(arrays['time'][ix],copy=True)).to(device),noise=torch.from_numpy(np.array(arrays['noise'][ix],copy=True)).to(device))
            opt.zero_grad(set_to_none=True);loss=cfm_loss(model,head,'C2',batch,scales_context)
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite trainloss')
            loss.backward()
            if gradient is None:gradient=gradient_proof(model,head);atomic_json(args.output/'gradient_proof.json',gradient)
            torch.nn.utils.clip_grad_norm_(parameters,1.);opt.step();loss_sum+=float(loss.detach())*len(ix);updates+=1
        vr,vb=forecast(split['validation_conditions'],4,valdir/f'epoch{epoch:03d}');es=float(np.mean([r['energy_score'] for r in vr]));before=copy.deepcopy(controller)
        controller,improved,event=plateau_step(controller,es,epoch)
        if improved:best_state=selected_state(model,head)
        for group,lr in zip(opt.param_groups,controller['lrs']):group['lr']=lr
        row=dict(epoch=epoch,validation_summary_path=str((valdir/f'epoch{epoch:03d}_summary.json').resolve()),updates=updates,total_updates=epoch*1536,train_CFM_MSE=loss_sum/len(mapping),validation_ES=es,lrs_used=lrs_used,controller_before=before,controller=copy.deepcopy(controller),event=event,best_update=improved,epoch_seconds=time.time()-begin)
        history.append(row);atomic_json(valdir/f'epoch{epoch:03d}_summary.json',dict(**study_summary(vr,20261021),rows=vr,banks=vb))
        atomic_json(args.output/'training_history.json',history)
        save_checkpoint(check/'last_C2.pt',dict(epoch=epoch,state=selected_state(model,head),optimizer=opt.state_dict(),best_state=best_state,controller=controller,history=history,validation_banks=vb))
        save_checkpoint(check/'best_C2.pt',dict(epoch=controller['best_epoch'],validation_ES=controller['best_ES'],state=best_state))
        progress=dict(stage='training',**row,elapsed_seconds=time.time()-started,max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2)
        atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if controller['converged']:reason='validation_plateau_at_min_lr';break
    unchanged();selection=dict(status='frozen_before_new_blindtest' if controller['converged'] else 'budget_limited_test_closed',controller=controller,last_epoch=epoch,total_updates=epoch*1536,continued_updates=(epoch-first_epoch)*1536,reason=reason,budget_limited=not controller['converged'])
    atomic_json(args.output/'validation_selection.json',selection)
    if not controller['converged']:
        source_gate()
        for k,p in refs.items():
            if sha256(p)!=hashes[k]:raise ValueError('reference changed:'+k)
        atomic_json(args.output/'summary.json',dict(status='budget_limited',selection=selection,new_test_opened=False));return
    opt.zero_grad(set_to_none=True)
    variants=dict(B_parent=('B',torch.load(refs['reference_B'],map_location='cpu',weights_only=True)['state']),C2_epoch4=('C2',torch.load(refs['reference_C2'],map_location='cpu',weights_only=True)['state']),C2_converged=('C2',best_state))
    tr,tb=forecast(split['test_conditions'],8,args.output/'test',variants)
    pairs={name:paired_metrics(tr,a,b,20261021) for name,a,b in [('extended_vs_B','C2_converged','B_parent'),('extended_vs_epoch4','C2_converged','C2_epoch4'),('epoch4_vs_B','C2_epoch4','B_parent')]}
    for k,p in refs.items():
        if sha256(p)!=hashes[k]:raise ValueError('reference changed:'+k)
    for name,h in cache_manifest.items():
        if sha256(cache_dir/name)!=h:raise ValueError('parent frozeninput modified')
    for stage in ('train','validation','test'):
        for item in split[stage]:
            if archive_content_hash(Path(item['cache_path']))!=item['cache_content_hash']:raise ValueError('data changed')
    unchanged();source_gate();imports={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items() if n.startswith(('tools.','urban_dreamer.')) and getattr(m,'__file__',None)}
    if any(not Path(p).is_relative_to(FROZEN_ROOT) for p in imports.values()):raise ValueError('live import')
    atomic_json(args.output/'import_audit.json',imports)
    atomic_json(args.output/'summary.json',dict(status='complete',test=study_summary(tr,20261021),pairs=pairs,selection=selection,continued_only_C2=True,training_budgets_equal=False,frozen_state_unchanged=True,parent_cache_unchanged=True,elapsed_seconds=time.time()-started,max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','split','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--maximum-epoch',type=int,default=80)
    p.add_argument('--device',default='cuda:0');p.add_argument('--gpu-memory-fraction',type=float,default=.08)
    run(p.parse_args())
