"""Paired small nonlinear FM residuals on frozen CFM features and fresh splits.

Only 41,152 new parameters per arm receive gradients. No original checkpoint,
codec, prior, action, reference or transformer parameter is retrained.
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

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
FROZEN_ROOT=Path(os.environ.get('NONLINEAR_FM_FROZEN_ROOT',ROOT)).resolve()
sys.path.insert(0,str(FROZEN_ROOT))
import numpy as np
import torch
from torch import nn

from tools.component_experiment_data import atomic_json,load_frozen,sha256
from tools.evaluate_soft_prior_fm import MeanOverridePrior,digest,score,METRICS
from tools.adapt_fm_output_head import capture_final_features
from tools.evaluate_pretrained_distribution import CHECKPOINT_SHA256,trajectory_bank
from tools.evaluate_component_world_models import endpoint_target,stable_seed
from tools.summarize_component_predictions import demand_bootstrap
from urban_dreamer.learning.cohort_targets import prepare_transition_targets,prepare_cohort_batch
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.learning.offline_data import archive_content_hash

ARMS=('zero_mean','weak_soft')


class NonlinearResidual(nn.Module):
    def __init__(self):
        super().__init__();self.hidden=nn.Linear(256,128);self.output=nn.Linear(128,64)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):return self.output(torch.nn.functional.silu(self.hidden(x)))


class ResidualVelocityHead(nn.Module):
    def __init__(self,original,residual):
        super().__init__();self.original=original;self.residual=residual
    def forward(self,x):return self.original(x)+self.residual(x)


@contextmanager
def residual_head(dynamics,residual):
    original=dynamics.flow_head;dynamics.flow_head=ResidualVelocityHead(original,residual)
    try:yield
    finally:dynamics.flow_head=original


def paired_metrics(rows,arm,reference,seed):
    a={r['condition_index']:r for r in rows if r['arm']==arm};b={r['condition_index']:r for r in rows if r['arm']==reference}
    if a.keys()!=b.keys():raise ValueError('paired conditions required')
    metrics=[m for m in METRICS if '_rmse_' not in m];differences=[]
    for index,row in a.items():
        other=b[index];item={key:row[key] for key in ('demand_group','episode')}
        for metric in metrics:
            if (row[metric] is None)!=(other[metric] is None):raise ValueError('paired masks differ')
            item[metric]=None if row[metric] is None else row[metric]-other[metric]
        differences.append(item)
    return {m:demand_bootstrap(differences,m,repetitions=2000,seed=seed) for m in metrics}


def study_summary(rows,seed):
    """Actual study arms only; no implicit native-baseline lookup/forecasts."""
    result=[]
    for arm in sorted({r['arm'] for r in rows}):
        selected=[r for r in rows if r['arm']==arm]
        metrics={m:demand_bootstrap(selected,m,repetitions=2000,seed=seed) for m in METRICS}
        for metric in METRICS:
            if '_rmse_' in metric:
                mse=demand_bootstrap(selected,metric.replace('_rmse_','_mse_'),repetitions=2000,seed=seed)
                metrics[metric]={key:float(np.sqrt(value)) if key in ('estimate','ci_low','ci_high') and value is not None else value for key,value in mse.items()}
        result.append(dict(arm=arm,states=len(selected),metrics=metrics))
    return dict(arms=result)


def feature_rows(entries,roads):return sum(len(e['starts']) for e in entries)*4*roads


def selection_improves(best,es):
    return best is None or es<best['ES']-1e-5


def stratified_draws(seed,stage,entry,starts,quarter,rank):
    """One independent uniform draw per quarter/window, same for both arms."""
    times=[];coefficients=[];records=[]
    for start in starts:
        time_seed=stable_seed(seed,stage,'uniform_time',entry['cache_path'],int(start),quarter)
        noise_seed=stable_seed(seed,stage,'source_noise',entry['cache_path'],int(start),quarter)
        uniform=float(torch.rand((),generator=torch.Generator().manual_seed(time_seed)))
        value=np.float32((quarter+uniform)/4)
        times.append(value);coefficients.append(torch.randn(1,rank,64,generator=torch.Generator().manual_seed(noise_seed)))
        records.append(dict(start=int(start),quarter=quarter,time=float(value),time_seed=time_seed,noise_seed=noise_seed))
    return torch.tensor(np.asarray(times))[:,None],torch.stack(coefficients),records


def run(args):
    started=time.time();torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    if FROZEN_ROOT!=(args.output/'source').resolve():raise ValueError('new frozen source required')
    if (args.output/'summary.json').exists():raise ValueError('preserve completed study')
    manifest=json.loads((args.output/'source_manifest.json').read_text())
    def verify_source():
        for relative,expected in manifest.items():
            if sha256(FROZEN_ROOT/relative)!=expected:raise ValueError('frozen source changed')
    verify_source();split=json.loads(args.split.read_text())
    refs={'checkpoint':args.experiment/'snapshot/checkpoints/latest.pt','split':args.split,
        'cache_manifest':Path(split['cache_manifest_path']),'split_audit':args.output/'data_split_audit.json',
        'weak_head':args.weak_head,'basis':args.basis,'known_banks':args.known_reference/'bank_manifest.json',
        'known_conditions':args.known_reference/'conditions.json','known_cache':args.known_cache,'scales':args.scale_reference}
    hashes={k:sha256(v) for k,v in refs.items()}
    if hashes['checkpoint']!=CHECKPOINT_SHA256:raise ValueError('completed checkpoint required')
    group_sets=[{e['demand_group'] for e in split[name]} for name in ('train','validation','test')]
    if any(group_sets[i]&group_sets[j] for i in range(3) for j in range(i)):raise ValueError('split leakage')
    protocol=dict(status='prespecified_before_feature_cache_or_training',reference_hashes=hashes,checkpoint_sha256=CHECKPOINT_SHA256,
        seed=args.seed,arms=list(ARMS),prior='weakstate-onlymatchedHYBRIDtau.1 versuszero_mean; sameoriginalnoise16',
        references='weaksource/ref=z+b; zerosource/ref=z; neitherreferenceisNone',
        original_frozen='codec/prior/actionencoder/referenceprojection/fullTransformer/originalLinearflowhead',
        residual='Linear256128+SiLU+Linear12864,41152params; lastlayerzero; identicalinitseed',
        cached_target='(encodedHYBRIDnext-start)-old_linear_velocity at x_t=(1-t)start+t target; oldbody no_grad pre-flow_head hook',
        times='4paireduniformstrata eachwindow [j/4,(j+1)/4), stagekey train/validation, sameindependentnoiseperarm',
        split_counts=split['counts'],cache='FP32 memmapsH256/Y64, minibatchcopies only, CPU/BLAS2; NOTESTfeatures',
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,batch_rows=32768,clip_norm=1),
        epochs=dict(minimum=20,maximum=80,integrated_validation_every=5,patience_checks=3,
            stop='BOTHarms3checkswithoutESimprovement, sameactualepochs/updates; independentlybestvalEScheckpoint',improvement_epsilon=1e-5),
        validation='epoch0 zeroresidual eligible; featureMSEall16freshgroups monitored; integratedES8fixedstates4groups K4Euler100 epoch0 and eachcheckafter20; minimum20 consumed regardlessselection',
        test='oneblindH1 all32freshstates16groups K8Euler100 AFTERselection; originalzero/weakplusbestzero/weak4variants',
        optional_h4_gate='bestweak-bestzero ES AND bothoccupiedCRPS95CIupper<0, totalcountabsoluteerror pointdelta<=0; ifpasses first8FIXEDteststates H4C0autoregressive K8Euler100, no retuning',
        gate='first2knownzero/weak+zeroresidual EXACTcachedK8beforeSGD; originalbody/noise/methodchecks',
        runtime='cachefirst4groups timing+FULLfirstepoch continuingexactstate; no partialpilotSGD hiddenexposures',
        device=args.device,gpu_memory_fraction=args.gpu_memory_fraction,precision='FP32noamp/compile',
        caveats=['Onlysmallnewresidualtrained; notfulljointmodelconvergence.',
            'Max80cap beforepatience=>budgetlimited, no convergenceclaim.',
            'Testnotopenedforscoring/featuresuntilvalidationselection; oneCP/newheadseed, notpolicy/MARLbenefit.'])
    atomic_json(args.output/'protocol.json',protocol)
    device=torch.device(args.device);torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
    if free<args.gpu_memory_fraction*total+256*1024**2:raise RuntimeError('insufficient GPUheadroom')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)
    config,prepared,topology,model,_=load_frozen(args.experiment,device=device);model.eval().requires_grad_(False);model.set_online_chunk_profile()
    native=model.dynamics.graph_prior;old_linear=model.dynamics.flow_head;original=digest(model)
    capacities=torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1);limits=torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
    with np.load(args.basis) as a:basis=a['basis'];values=a['eigenvalues']
    with np.load(args.weak_head) as a:coefficients=a['coefficients']
    operator=(basis*(1/(1+.1*values)))@basis.T
    priors={'zero_mean':MeanOverridePrior(native,zero=True),'weak_soft':MeanOverridePrior(native,operator=operator,coefficients=coefficients,state_only=True)}
    heads={}
    for arm in ARMS:
        torch.manual_seed(args.seed);heads[arm]=NonlinearResidual().to(device)
    if sum(p.numel() for p in heads['zero_mean'].parameters())!=41152:raise ValueError('wrongresidualsize')
    for key,value in heads['zero_mean'].state_dict().items():torch.testing.assert_close(value,heads['weak_soft'].state_dict()[key],rtol=0,atol=0)
    scales=np.asarray(json.loads(args.scale_reference.read_text())['scales']['scales'])
    datasets={name:CompactPretrainDataset(config,prepared,topology,5 if name=='validation' else 1,
        cache_manifest_path=refs['cache_manifest'],split=name) for name in ('train','validation')}
    # Verify initial zero residual against old known predictions before learning.
    known=CompactPretrainDataset(config,prepared,topology,5,cache_manifest_path=args.known_cache,split='validation')
    known_items=json.loads((args.known_reference/'conditions.json').read_text())[:2]
    known_banks={b['condition_index']:b for b in json.loads((args.known_reference/'bank_manifest.json').read_text())}
    for b in known_banks.values():
        if sha256(Path(b['path']))!=b['sha256']:raise ValueError('knownbankchanged')
    gates=[]
    with torch.inference_mode():
        for item in known_items:
            batch=known.batch_from_starts(item['episode_index'],[item['start']],fm_only=False);target=prepare_transition_targets(batch,capacities=capacities)
            z=model.encode(batch.flow[:,0].to(device));truth=endpoint_target(target.transition_target_flow[:,0],batch.flow[:,2],capacities,limits).numpy()
            with np.load(known_banks[item['condition_index']]['path']) as saved:
                np.testing.assert_array_equal(truth,saved['target'])
                for arm,oldkey in [('zero_mean','zero_mean'),('weak_soft','hybrid_state_soft0.1')]:
                    model.dynamics.graph_prior=priors[arm]
                    with residual_head(model.dynamics,heads[arm]):
                        draws=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(1,),samples=8,chunk=4,steps=100)[1]
                    np.testing.assert_array_equal(draws,saved[oldkey]);gates.append(dict(condition=item['condition_index'],arm=arm,zero_residual_exact_oldforecast=True))
    model.dynamics.graph_prior=native
    if digest(model)!=original:raise ValueError('initial gates modifiedoldmodel')
    atomic_json(args.output/'initial_gates.json',gates)
    cache_dir=args.output/'features';cache_dir.mkdir(exist_ok=True);roads=len(basis)
    required=sum(feature_rows(split[s],roads) for s in ('train','validation'))*320*4*2
    if shutil.disk_usage(args.output).free<required+4*1024**3:raise RuntimeError('insufficientcache diskreserve')
    caches={};cache_audit=[];alignment=[];schedules=[];feature_start=time.time()
    with torch.inference_mode():
        for stage in ('train','validation'):
            entries=split[stage];data=datasets[stage];indices={e['cache_path']:i for i,e in enumerate(data.manifest['entries'])}
            rows=feature_rows(entries,roads);caches[stage]={}
            for arm in ARMS:
                xp=cache_dir/f'{stage}_{arm}_x.npy';yp=cache_dir/f'{stage}_{arm}_y.npy'
                caches[stage][arm]=dict(x=np.lib.format.open_memmap(xp,mode='w+',dtype=np.float32,shape=(rows,256)),
                    y=np.lib.format.open_memmap(yp,mode='w+',dtype=np.float32,shape=(rows,64)),x_path=xp,y_path=yp)
            offset=0
            for ordinal,entry in enumerate(entries):
                index=indices[entry['cache_path']];starts=entry['starts'];batch=data.batch_from_starts(index,starts,materialization_horizon=1,fm_only=True)
                target_batch=prepare_transition_targets(batch,capacities=capacities)
                z=model.encode(batch.flow[:,0].to(device))[:,None];target=model.encode(target_batch.transition_target_flow[:,0].to(device))[:,None]
                action=batch.action.to(device);condition=model.dynamics.encode_condition(action,action_events=batch.action_events.to(device))
                b=priors['weak_soft'](z,condition)
                align=dict(stage=stage,demand_group=entry['demand_group'],windows=len(starts),zero_delta_energy=float((target-z).square().sum()),weak_no_noise_sse=float((target-z-b).square().sum()),zero_noisy_sse=0.,weak_noisy_sse=0.)
                for quarter in range(4):
                    times,noise,records=stratified_draws(args.seed,stage,entry,starts,quarter,native.rank)
                    schedules.append(dict(stage=stage,cache_path=entry['cache_path'],quarter=quarter,draws=records))
                    times=times.to(device);noise=noise.to(z);size=len(starts)*roads
                    for arm in ARMS:
                        prior=priors[arm];torch.testing.assert_close(prior.sample_noise(z,noise),native.sample_noise(z,noise),rtol=0,atol=0)
                        model.dynamics.graph_prior=prior;start,_,cached=model.dynamics.prior_start(z,condition,noise)
                        t=times[:,:,None,None];interpolated=(1-t)*start+t*target
                        features,out=capture_final_features(old_linear,lambda:model.dynamics(action,packed_latents=interpolated,flow_time=times,condition=cached))
                        y=target-start-out.latent_velocity
                        caches[stage][arm]['x'][offset:offset+size]=features.cpu().numpy().reshape(-1,256)
                        caches[stage][arm]['y'][offset:offset+size]=y.cpu().numpy().reshape(-1,64)
                        align['zero_noisy_sse' if arm=='zero_mean' else 'weak_noisy_sse']+=float((target-start).square().sum())
                    offset+=size
                alignment.append(align);cache_audit.append(dict(stage=stage,cache_path=entry['cache_path'],hash=entry['cache_content_hash'],starts=starts))
                print(json.dumps(dict(stage='feature_cache',split=stage,groups_complete=ordinal+1,total_groups=len(entries),elapsed_seconds=time.time()-feature_start)),flush=True)
                if stage=='train' and ordinal==3:
                    atomic_json(args.output/'cache_timing_pilot.json',dict(groups=4,windows=48,uniform_quarters=4,head_training_performed=False,
                        measured_seconds=time.time()-feature_start,estimated_train_seconds=(time.time()-feature_start)*16))
            if offset!=rows:raise ValueError('cache rowsnotfullywritten')
            for arm in ARMS:caches[stage][arm]['x'].flush();caches[stage][arm]['y'].flush()
    model.dynamics.graph_prior=native
    if digest(model)!=original or old_linear._forward_pre_hooks:raise ValueError('cache changedbody/lefthook')
    atomic_json(args.output/'alignment_stats.json',alignment);atomic_json(args.output/'feature_draw_schedule.json',schedules);atomic_json(args.output/'cache_data_audit.json',cache_audit)
    atomic_json(args.output/'feature_manifest.json',{stage:{arm:dict(rows=len(data['x']),dtype='float32',x=str(data['x_path']),y=str(data['y_path']),
        x_sha256=sha256(data['x_path']),y_sha256=sha256(data['y_path'])) for arm,data in arms.items()} for stage,arms in caches.items()})
    # Epoch1 is the FULL timing pilot and continues unchanged; no extra exposures.
    optimizers={arm:torch.optim.AdamW(heads[arm].parameters(),lr=.001,weight_decay=.0001) for arm in ARMS}
    checkpoints=args.output/'checkpoints';checkpoints.mkdir(exist_ok=True);validation_dir=args.output/'validation';validation_dir.mkdir(exist_ok=True)
    def feature_mse(arm):
        x,y=caches['validation'][arm]['x'],caches['validation'][arm]['y'];total=0.
        heads[arm].eval()
        with torch.no_grad():
            for begin in range(0,len(x),32768):
                xx=torch.from_numpy(np.array(x[begin:begin+32768],copy=True)).to(device);yy=torch.from_numpy(np.array(y[begin:begin+32768],copy=True)).to(device)
                total+=float((heads[arm](xx)-yy).square().sum())
        return total/(len(x)*64)
    def evaluate(items,variants,samples,path_prefix):
        rows=[];banks=[]
        with torch.inference_mode():
            for ordinal,item in enumerate(items):
                data=datasets['validation'];index=item['episode_index']
                if data.manifest['entries'][index]['cache_path']!=item['cache_path']:raise ValueError('rootconditionindicesreordered')
                batch=data.batch_from_starts(index,[item['start']],fm_only=False);target=prepare_transition_targets(batch,capacities=capacities)
                z=model.encode(batch.flow[:,0].to(device));truth=endpoint_target(target.transition_target_flow[:,0],batch.flow[:,2],capacities,limits).numpy();count=int(target.transition_cohort_count[0,0])
                stored=dict(target=truth,scales=scales,initial_latent=z.cpu().numpy())
                for label,(arm,residual) in variants.items():
                    model.dynamics.graph_prior=priors[arm]
                    if residual is None:
                        draws=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(1,),samples=samples,chunk=4,steps=100)[1]
                    else:
                        residual.eval()
                        with residual_head(model.dynamics,residual):draws=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(1,),samples=samples,chunk=4,steps=100)[1]
                    stored[label]=draws;rows.append(score(draws,truth,scales,count,limits.numpy(),dict(item,arm=label,episode=str(item['source_path'])+'::'+str(item['episode']),steps=100,horizon=1,k=samples)))
                model.dynamics.graph_prior=native
                if digest(model)!=original:raise ValueError('evaluationmodifiedoriginalbody/head')
                path=Path(str(path_prefix)+f'_{ordinal:03d}.npz');np.savez_compressed(path,**stored);banks.append(dict(condition_index=item['condition_index'],path=str(path.resolve()),sha256=sha256(path)))
        return rows,banks
    initial_rows,initial_banks=evaluate(split['validation_conditions'],{a:(a,heads[a]) for a in ARMS},4,validation_dir/'epoch000')
    best={}
    for arm in ARMS:
        es=float(np.mean([r['energy_score'] for r in initial_rows if r['arm']==arm]));best[arm]=dict(ES=es,epoch=0,patience=0)
        torch.save(dict(epoch=0,validation_ES=es,state={k:v.detach().cpu().clone() for k,v in heads[arm].state_dict().items()}),checkpoints/f'best_{arm}.pt')
    atomic_json(validation_dir/'epoch000_summary.json',dict(**study_summary(initial_rows,args.seed),bank_manifest=initial_banks,rows=initial_rows))
    history=[];stop_reason='maximum_epoch_cap'
    for epoch in range(1,81):
        epoch_start=time.time();order=np.random.default_rng(stable_seed(args.seed,'training_order',epoch)).permutation(len(caches['train']['zero_mean']['x']))
        losses={arm:0. for arm in ARMS};updates=0
        for begin in range(0,len(order),32768):
            index=order[begin:begin+32768]
            for arm in ARMS:
                heads[arm].train();data=caches['train'][arm]
                x=torch.from_numpy(np.array(data['x'][index],copy=True)).to(device);y=torch.from_numpy(np.array(data['y'][index],copy=True)).to(device)
                optimizer=optimizers[arm];optimizer.zero_grad(set_to_none=True);loss=(heads[arm](x)-y).square().mean()
                if not bool(torch.isfinite(loss)):raise ValueError('nonfinitetrainingloss')
                loss.backward();torch.nn.utils.clip_grad_norm_(heads[arm].parameters(),1.);optimizer.step();losses[arm]+=float(loss)*len(index)
            updates+=1
        row=dict(epoch=epoch,updates_per_arm=updates,train_mse={a:losses[a]/len(order) for a in ARMS},
            validation_feature_mse={a:feature_mse(a) for a in ARMS},epoch_seconds=time.time()-epoch_start)
        if epoch==1:atomic_json(args.output/'epoch_timing_pilot.json',dict(full_epoch=True,continued_exact_state=True,extra_pilot_updates=0,**row))
        if epoch>=20 and epoch%5==0:
            validation_rows,banks=evaluate(split['validation_conditions'],{a:(a,heads[a]) for a in ARMS},4,validation_dir/f'epoch{epoch:03d}')
            validation_summary=study_summary(validation_rows,args.seed);row['integrated_validation']={}
            for arm in ARMS:
                selected=[r for r in validation_rows if r['arm']==arm];es=float(np.mean([r['energy_score'] for r in selected]));row['integrated_validation'][arm]=es
                if selection_improves(best[arm],es):
                    best[arm]=dict(ES=es,epoch=epoch,patience=0)
                    torch.save(dict(epoch=epoch,validation_ES=es,state={k:v.detach().cpu().clone() for k,v in heads[arm].state_dict().items()}),checkpoints/f'best_{arm}.pt')
                else:best[arm]['patience']+=1
                torch.save(dict(epoch=epoch,state=heads[arm].state_dict(),optimizer=optimizers[arm].state_dict()),checkpoints/f'last_{arm}.pt')
            atomic_json(validation_dir/f'epoch{epoch:03d}_summary.json',dict(**validation_summary,bank_manifest=banks,rows=validation_rows))
        history.append(row);atomic_json(args.output/'training_history.json',history)
        progress=dict(stage='residual_training',epoch=epoch,best=best,elapsed_seconds=time.time()-started,max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2)
        atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if epoch>=20 and all(best[a]['patience']>=3 for a in ARMS):stop_reason='both_validation_ES_patience3';break
    if digest(model)!=original or any(p.grad is not None for p in model.parameters()):raise ValueError('originalbackboneupdated/receivedgradient')
    for arm in ARMS:
        saved=torch.load(checkpoints/f'best_{arm}.pt',map_location=device,weights_only=True);heads[arm].load_state_dict(saved['state']);heads[arm].eval()
    atomic_json(args.output/'validation_selection.json',dict(status='frozen_before_blind_test',best=best,stop_reason=stop_reason,
        actual_equal_epochs=epoch,budget_limited=stop_reason=='maximum_epoch_cap',actual_updates_per_arm=sum(r['updates_per_arm'] for r in history),hyperparameter_grid=False))
    test_dir=args.output/'test';test_dir.mkdir(exist_ok=True)
    variants={'old_zero':('zero_mean',None),'old_soft':('weak_soft',None),'adapter_zero':('zero_mean',heads['zero_mean']),'adapter_soft':('weak_soft',heads['weak_soft'])}
    test_rows,test_banks=evaluate(split['test_conditions'],variants,8,test_dir/'condition')
    test_summary=study_summary(test_rows,args.seed);pairs={label:paired_metrics(test_rows,a,b,args.seed) for label,a,b in [
        ('prior_effect_after_equal_adapter','adapter_soft','adapter_zero'),('zero_adapter_effect','adapter_zero','old_zero'),('soft_adapter_effect','adapter_soft','old_soft'),('prior_effect_before_adapter','old_soft','old_zero')]}
    atomic_json(test_dir/'states.json',test_rows);atomic_json(test_dir/'bank_manifest.json',test_banks)
    p=pairs['prior_effect_after_equal_adapter'];qualifies=all(p[m]['ci_high'] is not None and p[m]['ci_high']<0 for m in ('energy_score','count_crps_occupied','speed_crps_occupied')) and p['total_count_absolute_error']['estimate']<=0
    atomic_json(args.output/'h4_gate.json',dict(prespecified=True,pass_gate=qualifies,metrics={m:p[m] for m in ('energy_score','count_crps_occupied','speed_crps_occupied','total_count_absolute_error')},tuning_after_test=False))
    if qualifies:
        h4rows=[]
        with torch.inference_mode():
            for item in split['test_conditions'][:8]:
                batch=datasets['validation'].batch_from_starts(item['episode_index'],[item['start']],fm_only=False);cohort=prepare_cohort_batch(batch,capacities=capacities)
                z=model.encode(cohort.transition_flow[:,0].to(device));truth=endpoint_target(cohort.transition_flow[:,4],batch.flow[:,5],capacities,limits).numpy()
                for arm in ARMS:
                    model.dynamics.graph_prior=priors[arm]
                    with residual_head(model.dynamics,heads[arm]):draws=trajectory_bank(model,z,cohort.transition_action.to(device),cohort.transition_action_events.to(device),capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(4,),samples=8,chunk=4,steps=100)[4]
                    h4rows.append(score(draws,truth,scales,int(cohort.cohort_count[0]),limits.numpy(),dict(item,arm=arm,episode=str(item['source_path'])+'::'+str(item['episode']),steps=100,horizon=4,k=8,control_scope='fixed_initial_cohort')))
                model.dynamics.graph_prior=native
        atomic_json(args.output/'h4_summary.json',dict(**study_summary(h4rows,args.seed),rows=h4rows,diagnostic_not_counterfactual=True))
    if digest(model)!=original:raise ValueError('originalmodelnotrestored')
    for name,path in refs.items():
        if sha256(path)!=hashes[name]:raise ValueError('inputchanged')
    for entry in cache_audit:
        if archive_content_hash(Path(entry['cache_path']))!=entry['hash']:raise ValueError('cachechanged')
    for entry in split['test']:
        if archive_content_hash(Path(entry['cache_path']))!=entry['cache_content_hash']:raise ValueError('blindtestcachechanged')
    verify_source();imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if name.startswith(('tools.','urban_dreamer.')) and getattr(module,'__file__',None)}
    if any(not Path(path).is_relative_to(FROZEN_ROOT) for path in imports.values()):raise ValueError('liveimport')
    atomic_json(args.output/'import_audit.json',imports)
    atomic_json(args.output/'summary.json',dict(status='complete',protocol=protocol,test=test_summary,test_pairs=pairs,best=best,
        stop_reason=stop_reason,actual_equal_epochs=epoch,updates_per_arm=sum(r['updates_per_arm'] for r in history),budget_limited=stop_reason=='maximum_epoch_cap',
        original_model_digest=original,original_body_unchanged=True,original_head_unchanged=True,only_new_residual_parameters_trained=41152,
        original_noise_unchanged=True,blind_test_once=True,H4_gate_passed=qualifies,source_unchanged=True,data_unchanged=True,elapsed_seconds=time.time()-started))
    print(json.dumps(dict(stage='complete',elapsed_seconds=time.time()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','split','weak-head','basis','known-reference','known-cache','scale-reference','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--gpu-memory-fraction',type=float,default=.04);p.add_argument('--seed',type=int,default=20261021)
    run(p.parse_args())
