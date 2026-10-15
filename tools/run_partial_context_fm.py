"""Equal-budget partial FM adaptation after a separate training-only cost pilot."""
from __future__ import annotations
import argparse
import copy
import json
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
from tools.train_partial_context_fm import (ARMS, EXPECTED_PARAMETERS, allowed_name, tensor_digest,
    configure_trainable, partial_context, cfm_loss, gradient_proof, frozen_window_batch,
    selected_state, DirectResidual, MeanOverridePrior, ConditionEncoding, stratified_draws,
    atomic_json, load_frozen, sha256, CHECKPOINT_SHA256, archive_content_hash, CompactPretrainDataset)
from tools.train_direct_context_fm import selection_update
from tools.train_nonlinear_fm_residual import study_summary,paired_metrics
from tools.evaluate_pretrained_distribution import trajectory_bank
from tools.evaluate_component_world_models import endpoint_target,stable_seed
from tools.evaluate_soft_prior_fm import score
from urban_dreamer.learning.cohort_targets import prepare_transition_targets


def read(path):return json.loads(Path(path).read_text())


def frozen_digest(model):
    return tensor_digest([(name,value) for name,value in model.state_dict().items() if not allowed_name(name)])


def load_selected(model,residual,state):
    existing=model.state_dict()
    expected={name for name in existing if allowed_name(name)}
    if set(state['model'])!=expected:raise ValueError('selected state must contain exactly adapted modules')
    with torch.no_grad():
        for name,value in state['model'].items():existing[name].copy_(value)
    residual.load_state_dict(state['residual'])


def run(args):
    started=time.time()
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if FROZEN_ROOT!=(args.output/'source').resolve():raise ValueError('immutable source required')
    if (args.output/'training_history.json').exists() or (args.output/'summary.json').exists():raise ValueError('preserve prior training')
    if not 1<=args.minimum_epochs<=args.maximum_epochs:raise ValueError('positive bounded epoch budget required')
    manifest=read(args.output/'source_manifest.json')
    def source_gate():
        for name,h in manifest.items():
            if sha256(FROZEN_ROOT/name)!=h:raise ValueError('source changed')
    source_gate()
    split=read(args.split);pilot=read(args.pilot/'pilot_summary.json')
    if pilot['status']!='pilot_complete' or pilot['optimizer_steps']!=0 or not pilot['parameters_unchanged']:raise ValueError('valid no-update pilot required')
    refs=dict(checkpoint=args.experiment/'snapshot/checkpoints/latest.pt',split=args.split,
        cache_manifest=Path(split['cache_manifest_path']),pilot=args.pilot/'pilot_summary.json',basis=args.basis,
        weak_head=args.weak_head,scale_reference=args.scale_reference)
    hashes={k:sha256(p) for k,p in refs.items()}
    if hashes['checkpoint']!=CHECKPOINT_SHA256:raise ValueError('completed pretrain required')
    if pilot['reference_hashes']['split']!=hashes['split']:raise ValueError('pilot/train split mismatch')
    groups=[{e['demand_group'] for e in split[s]} for s in ('train','validation','test')]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise ValueError('split overlap')
    protocol=dict(status='prespecified_before_partial_fit_or_scores',reference_hashes=hashes,seed=args.seed,
        arms={a:dict(source_on=v[0],reference_on=v[1],hint='same frozen state-only b') for a,v in ARMS.items()},
        training=dict(minimum_epochs=args.minimum_epochs,maximum_epochs=args.maximum_epochs,batch_whole_graphs=args.batch_graphs,
            validation_every_epochs=args.validation_every,optimizer='AdamW',original_module_lr=args.lr,new_residual_lr=args.residual_lr,
            weight_decay=.0001,clip_norm=1.,patience_checks=3,meaningful_ES_reset_relative=.002,best_epsilon=1e-5,
            equal_actual_consumed_budget=True,initial_epoch0_candidate=True),
        trainable_parameters_per_arm=EXPECTED_PARAMETERS,freeze='codec/prior/actionconditioning/first5Transformerblocks allfrozen; gradients retained THROUGH frozenprefix to reference_proj',
        loss='fullvelocity CFM MSE target=encodedHYBRIDfuture-source, paired4stratifiedtime/noise draws; no oldresidualY',
        cache='only training Z/targetZ/background/action/edgecondition/time/noisecoefficients; no H cache',
        normalizers='scalaruncenteredRMS fromsame64traininggroups; relative pooledzero+coarsesource; no validation/test fit',
        validation='samefixed8states4groups K4Euler100 everycheck; independentlybestES, all3patience3 stops',
        test='fresh32states16groups only afterselection; K8Euler100, pairedwhole-demandgroupbootstrap2000',
        comparisons='C1vsB sourceeffect atfixedrefZ; C2vsC1 referenceeffect atfixedcoarsesource; C2vsBjoint',
        execution='FP32 noAMP noTF32 noCompile eval-mode dropoutdisabled activationcheckpointing',device=args.device,
        gpu_memory_fraction=args.gpu_memory_fraction,
        limitations=['Onepretrainedbackbone and adaptationseed; validationplateau does notprove globaloptimum.',
            'Cap without patience isbudgetlimited; only partialadaptation, not fulljointtraining.',
            'ALLarms havebackgroundhint: tests coarse source/reference incremental value, not hint necessity or policy benefit.'])
    atomic_json(args.output/'protocol.json',protocol)
    device=torch.device(args.device);torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
    if free<args.gpu_memory_fraction*total+256*1024**2:raise RuntimeError('insufficient GPU headroom')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)
    config,prepared,topology,base,_=load_frozen(args.experiment,device=device);base.eval().requires_grad_(False);base.set_online_chunk_profile()
    native=base.dynamics.graph_prior
    with np.load(args.basis) as a:basis,values=a['basis'],a['eigenvalues']
    with np.load(args.weak_head) as a:coefficients=a['coefficients']
    base.dynamics.graph_prior=MeanOverridePrior(native,operator=(basis*(1/(1+.1*values)))@basis.T,coefficients=coefficients,state_only=True)
    capacities=torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1)
    limits=torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
    scales=np.asarray(read(args.scale_reference)['scales']['scales'])
    train=CompactPretrainDataset(config,prepared,topology,1,cache_manifest_path=refs['cache_manifest'],split='train')
    validation=CompactPretrainDataset(config,prepared,topology,5,cache_manifest_path=refs['cache_manifest'],split='validation')
    indices={e['cache_path']:i for i,e in enumerate(train.manifest['entries'])}
    cache_dir=args.output/'frozen_inputs';cache_dir.mkdir(exist_ok=True)
    nw=sum(len(e['starts']) for e in split['train']);roads=len(basis)
    shapes=dict(z=(nw,1,roads,64),target=(nw,1,roads,64),b=(nw,1,roads,64),condition=(nw,1,roads,256),action=(nw,1,roads))
    cache={key:np.lib.format.open_memmap(cache_dir/f'{key}.npy',mode='w+',dtype=np.float32,shape=shape) for key,shape in shapes.items()}
    example_windows=[];times=[];noises=[];schedule=[];offset=0
    sums=dict(z=0.,b=0.,relative=0.);counts=dict(z=0,b=0,relative=0)
    for ordinal,e in enumerate(split['train']):
        starts=e['starts'];batch=train.batch_from_starts(indices[e['cache_path']],starts,materialization_horizon=1,fm_only=True)
        values=frozen_window_batch(base,batch,capacities);size=len(starts)
        for key in shapes:cache[key][offset:offset+size]=values[key].cpu().numpy()
        with torch.no_grad():
            for key in ('z','b'):sums[key]+=float(values[key].double().square().sum());counts[key]+=values[key].numel()
            for quarter in range(4):
                tt,noise,records=stratified_draws(args.seed,'train',e,starts,quarter,native.rank)
                example_windows.extend(range(offset,offset+size));times.extend(tt.numpy());noises.extend(noise.numpy())
                schedule.append(dict(stage='train',cache_path=e['cache_path'],quarter=quarter,draws=records))
                t=tt.to(device)[:,:,None,None];epsilon=native.sample_noise(values['z'],noise.to(values['z']))
                for source_on in (False,True):
                    mean=values['z']+values['b'] if source_on else values['z']
                    relative=(1-t)*(mean+epsilon)+t*values['target']-mean
                    sums['relative']+=float(relative.double().square().sum());counts['relative']+=relative.numel()
        offset+=size
        print(json.dumps(dict(stage='frozen_input_cache',groups=ordinal+1,total=len(split['train']),elapsed_seconds=time.time()-started)),flush=True)
    if offset!=nw:raise ValueError('cache length mismatch')
    for value in cache.values():value.flush()
    mapping=np.asarray(example_windows,dtype=np.int64);time_array=np.asarray(times,dtype=np.float32);noise_array=np.asarray(noises,dtype=np.float32)
    for name,array in [('example_window',mapping),('time',time_array),('noise',noise_array)]:np.save(cache_dir/f'{name}.npy',array)
    normalizers={key:max(np.sqrt(sums[key]/counts[key]),1e-8) for key in sums}
    atomic_json(args.output/'normalizers.json',dict(scales=normalizers,sums=sums,counts=counts,fit_split='trainonly',centering=False))
    atomic_json(args.output/'draw_schedule.json',schedule)
    atomic_json(args.output/'frozen_input_manifest.json',{p.name:sha256(p) for p in cache_dir.glob('*.npy')})
    models={};heads={};parameters={};frozen={};optimizers={}
    for arm in ARMS:
        models[arm]=copy.deepcopy(base);torch.manual_seed(args.seed);heads[arm]=DirectResidual().to(device)
        parameters[arm]=configure_trainable(models[arm],heads[arm]);frozen[arm]=frozen_digest(models[arm])
        optimizers[arm]=torch.optim.AdamW([dict(params=[p for p in models[arm].parameters() if p.requires_grad],lr=args.lr),dict(params=list(heads[arm].parameters()),lr=args.residual_lr)],weight_decay=.0001)
    initial={a:selected_state(models[a],heads[a]) for a in ARMS}
    for arm in ('C1','C2'):
        for kind in ('model','residual'):
            for k,v in initial['B'][kind].items():torch.testing.assert_close(v,initial[arm][kind][k],rtol=0,atol=0)
    atomic_json(args.output/'initial_gates.json',dict(same_initial_trainable_weights=True,all_arms_parameters=EXPECTED_PARAMETERS,new_heads_zero_output=True))
    check=args.output/'checkpoints';check.mkdir(exist_ok=True);valdir=args.output/'validation';valdir.mkdir(exist_ok=True)
    def unchanged():
        for arm in ARMS:
            if frozen_digest(models[arm])!=frozen[arm]:raise ValueError('frozen model state changed')
            if 'prior_start' in models[arm].dynamics.__dict__ or models[arm].dynamics._forward_pre_hooks:raise ValueError('context not restored')
    def forecast(items,k,directory):
        directory.mkdir(exist_ok=True);rows=[];banks=[]
        with torch.no_grad():
            for ordinal,item in enumerate(items):
                if validation.manifest['entries'][item['episode_index']]['cache_path']!=item['cache_path']:raise ValueError('condition mismatch')
                batch=validation.batch_from_starts(item['episode_index'],[item['start']],fm_only=False)
                scoped=prepare_transition_targets(batch,capacities=capacities)
                truth=endpoint_target(scoped.transition_target_flow[:,0],batch.flow[:,2],capacities,limits).numpy()
                z=base.encode(batch.flow[:,0].to(device));stored=dict(target=truth,scales=scales)
                for arm in ARMS:
                    with partial_context(models[arm].dynamics,heads[arm],arm,normalizers):
                        draws=trajectory_bank(models[arm],z,batch.action.to(device),batch.action_events.to(device),capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(1,),samples=k,chunk=4,steps=100)[1]
                    stored[arm]=draws
                    rows.append(score(draws,truth,scales,int(scoped.transition_cohort_count[0,0]),limits.numpy(),dict(item,arm=arm,episode=str(item['source_path'])+'::'+str(item['episode']),steps=100,horizon=1,k=k)))
                path=directory/f'condition_{ordinal:03d}.npz';np.savez_compressed(path,**stored);banks.append(dict(condition_index=item['condition_index'],path=str(path.resolve()),sha256=sha256(path)))
        unchanged();atomic_json(directory/'states.json',rows);atomic_json(directory/'bank_manifest.json',banks)
        return rows,banks
    def save_best(arm,epoch,es):torch.save(dict(epoch=epoch,validation_ES=es,state=selected_state(models[arm],heads[arm])),check/f'best_{arm}.pt')
    rows,banks=forecast(split['validation_conditions'],4,valdir/'epoch000');best={}
    for arm in ARMS:
        es=float(np.mean([r['energy_score'] for r in rows if r['arm']==arm]));best[arm]=dict(ES=es,epoch=0,reset_ES=es,reset_epoch=0,patience=0);save_best(arm,0,es)
    atomic_json(valdir/'epoch000_summary.json',dict(**study_summary(rows,args.seed),rows=rows,banks=banks))
    history=[];reason='maximum_epoch_cap';gradient_records={}
    for epoch in range(1,args.maximum_epochs+1):
        begin=time.time();order=np.random.default_rng(stable_seed(args.seed,'partial_training_order',epoch)).permutation(len(mapping))
        losses={a:0. for a in ARMS};updates=0
        for start in range(0,len(order),args.batch_graphs):
            ix=order[start:start+args.batch_graphs];wi=mapping[ix]
            batch={key:torch.from_numpy(np.array(value[wi],copy=True)).to(device) for key,value in cache.items()}
            batch.update(time=torch.from_numpy(time_array[ix].copy()).to(device),noise=torch.from_numpy(noise_array[ix].copy()).to(device))
            for arm in ARMS:
                opt=optimizers[arm];opt.zero_grad(set_to_none=True);loss=cfm_loss(models[arm],heads[arm],arm,batch,normalizers)
                if not bool(torch.isfinite(loss)):raise ValueError('nonfinite train loss')
                loss.backward()
                if epoch==1 and updates==0:gradient_records[arm]=gradient_proof(models[arm],heads[arm])
                torch.nn.utils.clip_grad_norm_(parameters[arm],1.);opt.step();losses[arm]+=float(loss.detach())*len(ix)
            updates+=1
        row=dict(epoch=epoch,updates_per_arm=updates,train_CFM_MSE={a:losses[a]/len(mapping) for a in ARMS},epoch_seconds=time.time()-begin)
        if epoch%args.validation_every==0 or epoch==args.maximum_epochs:
            vr,vb=forecast(split['validation_conditions'],4,valdir/f'epoch{epoch:03d}');row['integrated_validation']={}
            for arm in ARMS:
                es=float(np.mean([r['energy_score'] for r in vr if r['arm']==arm]));best[arm],improved,meaningful=selection_update(best[arm],es,epoch)
                row['integrated_validation'][arm]=dict(ES=es,best_update=improved,meaningful_reset=meaningful,selection=copy.deepcopy(best[arm]))
                if improved:save_best(arm,epoch,es)
            atomic_json(valdir/f'epoch{epoch:03d}_summary.json',dict(**study_summary(vr,args.seed),rows=vr,banks=vb))
        history.append(row);atomic_json(args.output/'training_history.json',history)
        progress=dict(stage='training',epoch=epoch,best=best,elapsed_seconds=time.time()-started,max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2)
        atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if epoch>=args.minimum_epochs and all(best[a]['patience']>=3 for a in ARMS):reason='all_meaningful_ES_patience3';break
    for arm in ARMS:
        torch.save(dict(epoch=epoch,state=selected_state(models[arm],heads[arm]),optimizer=optimizers[arm].state_dict()),check/f'last_{arm}.pt')
        cp=torch.load(check/f'best_{arm}.pt',map_location=device,weights_only=True);load_selected(models[arm],heads[arm],cp['state'])
        models[arm].zero_grad(set_to_none=True);heads[arm].zero_grad(set_to_none=True)
    unchanged();atomic_json(args.output/'gradient_proof.json',gradient_records)
    selection=dict(status='frozen_before_new_blindtest',best=best,actual_equal_epochs=epoch,updates_per_arm=sum(r['updates_per_arm'] for r in history),stop_reason=reason,budget_limited=reason=='maximum_epoch_cap')
    atomic_json(args.output/'validation_selection.json',selection)
    tr,tb=forecast(split['test_conditions'],8,args.output/'test')
    pairs={name:paired_metrics(tr,a,b,args.seed) for name,a,b in [('source_effect','C1','B'),('reference_effect','C2','C1'),('joint_effect','C2','B')]}
    for key,path in refs.items():
        if sha256(path)!=hashes[key]:raise ValueError('reference changed: '+key)
    for stage in ('train','validation','test'):
        for e in split[stage]:
            if archive_content_hash(Path(e['cache_path']))!=e['cache_content_hash']:raise ValueError('data changed')
    for name,h in read(args.output/'frozen_input_manifest.json').items():
        if sha256(cache_dir/name)!=h:raise ValueError('frozen input cache changed')
    unchanged();source_gate()
    imports={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items() if n.startswith(('tools.','urban_dreamer.')) and getattr(m,'__file__',None)}
    if any(not Path(p).is_relative_to(FROZEN_ROOT) for p in imports.values()):raise ValueError('live import')
    atomic_json(args.output/'import_audit.json',imports)
    atomic_json(args.output/'summary.json',dict(status='complete',test=study_summary(tr,args.seed),pairs=pairs,selection=selection,
        trainable_parameters_per_arm=EXPECTED_PARAMETERS,frozen_state_unchanged=True,source_unchanged=True,data_unchanged=True,
        elapsed_seconds=time.time()-started,max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','split','pilot','basis','weak-head','scale-reference','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--minimum-epochs',type=int,required=True);p.add_argument('--maximum-epochs',type=int,required=True)
    p.add_argument('--validation-every',type=int,default=1);p.add_argument('--batch-graphs',type=int,default=2)
    p.add_argument('--lr',type=float,default=.0001);p.add_argument('--residual-lr',type=float,default=.001)
    p.add_argument('--device',default='cuda:1');p.add_argument('--gpu-memory-fraction',type=float,default=.08)
    p.add_argument('--seed',type=int,default=20261021)
    run(p.parse_args())
