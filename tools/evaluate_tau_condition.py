"""Validation-only tau sensitivity of the selected, condition-only B model.

This is a frozen-head/frozen-FM inference sweep, not retraining each tau.
All imports use the immutable completed source-control experiment snapshot.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'outputs/source_condition_controls_20261015/v1/source'
os.environ['PARTIAL_FM_FROZEN_ROOT']=str(FROZEN)
os.environ['DIRECT_FM_FROZEN_ROOT']=str(FROZEN)
sys.path.insert(0,str(FROZEN))
import numpy as np
import torch
from tools.source_condition_controls import source_context
from tools.run_partial_context_fm import (read,load_selected,DirectResidual,MeanOverridePrior,
    atomic_json,load_frozen,sha256,CHECKPOINT_SHA256,CompactPretrainDataset,
    trajectory_bank,endpoint_target,score,prepare_transition_targets,stable_seed)
from tools.summarize_component_predictions import demand_bootstrap

TAUS=(0.,.01,.1,1.,10.,100.)
FIELDS=('energy_score_u','count_crps_u_occupied','speed_crps_u_occupied',
        'count_coverage90_occupied','speed_coverage90_occupied',
        'count_width90_occupied','speed_width90_occupied','energy_score',
        'count_mae_occupied','speed_mae_occupied','total_count_absolute_error')

def soft_operator(basis,eigenvalues,tau):
    if not np.isfinite(tau) or tau<0:raise ValueError('tau must be finite and nonnegative')
    return (basis*(1/(1+tau*eigenvalues)))@basis.T

def validation_conditions(split):
    """Use all existing validation groups; never inspect any test entry."""
    manifest=read(Path(split['cache_manifest_path']))
    entries=[x for x in manifest['entries'] if x['split']=='validation']
    indices={entry['cache_path']:i for i,entry in enumerate(entries)}
    old={(x['cache_path'],x['start']):x for x in split['validation_conditions']}
    conditions=[]
    for entry in split['validation']:
        for start in entry['evaluation_starts']:
            item=dict(old.get((entry['cache_path'],start),{}))
            item.update(condition_index=len(conditions),episode_index=indices[entry['cache_path']],
                cache_path=entry['cache_path'],source_path=entry['original_source_path'],episode=entry['episode'],
                demand_group=entry['demand_group'],start=start,role='validation')
            item.setdefault('condition_seed',stable_seed(20261016,'tau_condition',entry['demand_group'],start))
            conditions.append(item)
    if len({x['demand_group'] for x in conditions})!=16:raise ValueError('16 validation groups required')
    return conditions

def summarize(rows):
    result={}
    reference={r['condition_index']:r for r in rows if r['tau']==.1}
    for tau in TAUS:
        selected=[r for r in rows if r['tau']==tau]
        metrics={field:demand_bootstrap(selected,field,repetitions=2000,seed=20261016) for field in FIELDS}
        deltas=[]
        for row in selected:
            ref=reference[row['condition_index']]
            delta={key:row[key] for key in ('demand_group','episode')}
            delta.update({field:None if row[field] is None else row[field]-ref[field] for field in FIELDS})
            deltas.append(delta)
        result[str(tau)]=dict(metrics=metrics,minus_tau_0_1={field:demand_bootstrap(deltas,field,repetitions=2000,seed=20261016) for field in FIELDS})
    best=min(TAUS,key=lambda tau:result[str(tau)]['metrics']['energy_score_u']['estimate'])
    return dict(status='complete',validation_selected_tau=best,selection_metric='energy_score_u',
        interpretation='Inference sensitivity at fixed B weights and fixed prior regression head; not independently trained tau optima',
        arms=result,conditions=len(reference),demand_groups=16,test_evaluated=False)

def run(args):
    study=ROOT/'outputs/source_condition_controls_20261015/v1'
    parent=ROOT/'outputs/partial_context_fm_20261015/v2_gpu0'
    argv=read(parent/'launch.json')['argv'];opts={argv[i]:argv[i+1] for i in range(len(argv)-1) if argv[i].startswith('--')}
    refs=dict(checkpoint=Path(opts['--experiment'])/'snapshot/checkpoints/latest.pt',
        B_checkpoint=study/'B/checkpoints/best_B.pt',basis=Path(opts['--basis']),prior_head=Path(opts['--weak-head']),
        normalizers=study/'B/normalizers.json',physical_scales=Path(opts['--scale-reference']),
        split=study/'split.json',script=Path(__file__).resolve())
    hashes={key:sha256(path) for key,path in refs.items()}
    if hashes['checkpoint']!=CHECKPOINT_SHA256:raise ValueError('complete pretraining required')
    manifest=read(study/'source_manifest.json')
    for name,expected in manifest.items():
        if sha256(FROZEN/name)!=expected:raise ValueError('immutable source changed: '+name)
    split=read(refs['split']);conditions=validation_conditions(split)
    if args.output.exists():raise ValueError('preserve existing output')
    args.output.mkdir(parents=True);(args.output/'banks').mkdir()
    atomic_json(args.output/'protocol.json',dict(taus=TAUS,source='Z+native rank16 graph noise',
        reference='Z',hint='soft-filtered action-independent background',head_fixed=True,FM_fixed=True,
        prior_filter='output only',conditions=conditions,selection_split='validation only',
        samples=8,solver='euler',steps=100,device=args.device,reference_hashes=hashes,
        source_snapshot=str(FROZEN),test_read=False,train_updates=0))
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.06,device)
    config,prepared,topology,model,_=load_frozen(Path(opts['--experiment']),device=device)
    model.eval().requires_grad_(False);model.set_online_chunk_profile()
    head=DirectResidual().to(device).eval().requires_grad_(False)
    checkpoint=torch.load(refs['B_checkpoint'],map_location='cpu',weights_only=True)
    load_selected(model,head,checkpoint['state'])
    norms=read(refs['normalizers'])['scales'];scales=np.asarray(read(refs['physical_scales'])['scales']['scales'])
    with np.load(refs['basis']) as data:basis,eigenvalues=data['basis'],data['eigenvalues']
    with np.load(refs['prior_head']) as data:coefficients=data['coefficients']
    native=model.dynamics.graph_prior
    priors={tau:MeanOverridePrior(native,operator=soft_operator(basis,eigenvalues,tau),coefficients=coefficients,state_only=True) for tau in TAUS}
    capacities=torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1)
    limits=torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
    data=CompactPretrainDataset(config,prepared,topology,5,cache_manifest_path=Path(split['cache_manifest_path']),split='validation')
    rows=[];banks=[];anchor_checks=[]
    anchor_index={(x['cache_path'],x['start']):i for i,x in enumerate(split['validation_conditions'])}
    with torch.no_grad():
        for ordinal,item in enumerate(conditions):
            if data.manifest['entries'][item['episode_index']]['cache_path']!=item['cache_path']:raise ValueError('validation identity mismatch')
            batch=data.batch_from_starts(item['episode_index'],[item['start']],fm_only=False)
            scoped=prepare_transition_targets(batch,capacities=capacities)
            truth=endpoint_target(scoped.transition_target_flow[:,0],batch.flow[:,2],capacities,limits).numpy()
            z=model.encode(batch.flow[:,0].to(device));stored=dict(target=truth,scales=scales)
            for tau in TAUS:
                model.dynamics.graph_prior=priors[tau]
                with source_context(model.dynamics,head,'B',norms):
                    draws=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),
                        capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],
                        horizons=(1,),samples=8,chunk=4,steps=100)[1]
                row=score(draws,truth,scales,int(scoped.transition_cohort_count[0,0]),limits.numpy(),
                    dict(item,tau=tau,arm=f'tau_{tau:g}',episode=str(item['source_path'])+'::'+str(item['episode']),steps=100,horizon=1,k=8))
                rows.append(row);stored[f'tau_{tau:g}']=draws
                if tau==.1 and (item['cache_path'],item['start']) in anchor_index:
                    old_ordinal=anchor_index[(item['cache_path'],item['start'])]
                    anchor_path=parent/'validation'/f'epoch{checkpoint["epoch"]:03d}'/f'condition_{old_ordinal:03d}.npz'
                    with np.load(anchor_path) as archive:expected=archive['B']
                    np.testing.assert_allclose(draws[:4],expected,atol=2e-4,rtol=0)
                    anchor_checks.append(dict(condition=ordinal,anchor=str(anchor_path),max_absolute_error=float(np.max(np.abs(draws[:4]-expected)))))
            path=args.output/'banks'/f'condition_{ordinal:03d}.npz';np.savez_compressed(path,**stored)
            banks.append(dict(condition_index=ordinal,path=str(path.resolve()),sha256=sha256(path)))
            atomic_json(args.output/'states.json',rows);atomic_json(args.output/'bank_manifest.json',banks)
            atomic_json(args.output/'progress.json',dict(complete=ordinal+1,total=len(conditions)))
            print(f'{ordinal+1}/{len(conditions)}',flush=True)
    for key,path in refs.items():
        if sha256(path)!=hashes[key]:raise ValueError('input changed: '+key)
    atomic_json(args.output/'anchor_checks.json',anchor_checks)
    atomic_json(args.output/'summary.json',summarize(rows))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/tau_condition_20261016/v1')
    run(parser.parse_args())
