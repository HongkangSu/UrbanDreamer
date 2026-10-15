"""One shared blind test after A/G/B validation-only selection is complete.

Run an immutable copy from study/postprocess with PARTIAL_FM_FROZEN_ROOT set
to study/source. All model/helper imports must come from that training snapshot;
the evaluator's own path/hash are recorded separately, never added to its manifest.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
FROZEN_ROOT=Path(os.environ.get('PARTIAL_FM_FROZEN_ROOT',ROOT)).resolve()
sys.path.insert(0,str(FROZEN_ROOT))
import numpy as np
import torch

from tools.train_source_condition_control import install_prior
from tools.source_condition_controls import source_context
from tools.run_partial_context_fm import (
    read,load_selected,frozen_digest,DirectResidual,atomic_json,load_frozen,sha256,
    CHECKPOINT_SHA256,CompactPretrainDataset,trajectory_bank,endpoint_target,score,
    study_summary,paired_metrics,prepare_transition_targets,
)
from urban_dreamer.learning.offline_data import archive_content_hash

ARMS=('G','A','B','C2')


def require_ready(study):
    """No split/test datum is read here; failure leaves blind-test outputs absent."""
    selections={}
    for arm in ('A','G','B'):
        path=study/arm/'validation_selection.json'
        if not path.exists():raise RuntimeError(f'{arm} has not frozen validation selection')
        selection=read(path);controller=selection.get('controller',{})
        if selection.get('status')!='ready_for_shared_test' or not controller.get('converged',False):
            raise RuntimeError(f'{arm} has not reached the preregistered validation plateau')
        if selection.get('budget_limited',False):raise RuntimeError(f'{arm} is budget limited')
        if controller.get('reductions')!=2 or controller.get('patience',0)<5:
            raise RuntimeError(f'{arm} plateau controller is incomplete')
        selections[arm]=selection
    return selections


def load_selected_checkpoint(path,selection,label):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    controller=selection['controller']
    if checkpoint['epoch']!=controller['best_epoch']:
        raise ValueError(f'{label} checkpoint is not the validation-selected epoch')
    if not np.isclose(checkpoint['validation_ES'],controller['best_ES'],rtol=0,atol=1e-10):
        raise ValueError(f'{label} checkpoint validation score differs')
    if checkpoint['epoch']>selection['last_epoch']:raise ValueError('selected epoch exceeds consumed budget')
    return checkpoint


def load_fixed_reference(reference):
    path=Path(reference['path'])
    if sha256(path)!=reference['sha256']:raise ValueError('fixed C2 checkpoint hash differs from design')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['epoch']!=reference.get('epoch',10):
        raise ValueError('fixed C2 must remain the predeclared epoch10 reference')
    return checkpoint


def context_scales(payload):
    if payload.get('fit_split')!='trainonly' or payload.get('centering') is not False:
        raise ValueError('context normalizers must be uncentered and fitted only on training')
    result={key:float(payload['scales'][key]) for key in ('z','b','relative')}
    if any(not np.isfinite(value) or value<=0 for value in result.values()):
        raise ValueError('positive finite context scales required')
    return result


def command_options(argv):
    return {argv[i]:argv[i+1] for i in range(len(argv)-1) if argv[i].startswith('--')}


def build_priors(model,basis_path,head_path):
    """Each wrapper starts from the original native prior, never another wrapper."""
    native=model.dynamics.graph_prior;priors={}
    try:
        for arm in ARMS:
            model.dynamics.graph_prior=native
            install_prior(model,arm,basis_path,head_path)
            priors[arm]=model.dynamics.graph_prior.eval().requires_grad_(False)
    finally:model.dynamics.graph_prior=native
    return priors


def run(args):
    # This guard MUST precede test split access, dataset materialization and outputs.
    selections=require_ready(args.study)
    if FROZEN_ROOT!=(args.study/'source').resolve():raise ValueError('import from the immutable study/source snapshot')
    output=args.output or args.study/'shared_test'
    if output.exists():raise ValueError('preserve existing test run; choose a fresh --output')
    design=read(args.study/'design.json')
    if design.get('test_opened') is not False:raise ValueError('design must be the unchanged prespecified input')
    source_manifest=read(args.study/'source_manifest.json')
    def source_gate():
        for name,expected in source_manifest.items():
            if sha256(FROZEN_ROOT/name)!=expected:raise ValueError('frozen source changed: '+name)
    source_gate()
    launches=read(args.study/'launch.json')['arms']
    parents={command_options(launches[arm]['argv'])['--parent'] for arm in ('A','G','B')}
    if len(parents)!=1:raise ValueError('all controls must share the same completed-pretrain parent')
    parent=Path(parents.pop());parent_opts=command_options(read(parent/'launch.json')['argv'])
    experiment=Path(parent_opts['--experiment'])
    refs=dict(design=args.study/'design.json',source_manifest=args.study/'source_manifest.json',
        split=args.study/'split.json',checkpoint=experiment/'snapshot/checkpoints/latest.pt',
        basis=Path(parent_opts['--basis']),weak_head=Path(parent_opts['--weak-head']),
        physical_scales=Path(parent_opts['--scale-reference']),parent_cache=parent/'frozen_input_manifest.json',
        parent_context_normalizers=parent/'normalizers.json',evaluator=Path(__file__).resolve())
    checkpoints={};norms={};budgets={}
    for arm in ('A','G','B'):
        refs[arm+'_selection']=args.study/arm/'validation_selection.json'
        refs[arm+'_checkpoint']=args.study/arm/'checkpoints'/f'best_{arm}.pt'
        refs[arm+'_normalizers']=args.study/arm/'normalizers.json'
        refs[arm+'_protocol']=args.study/arm/'protocol.json'
        if read(refs[arm+'_protocol'])['reference_hashes']['checkpoint']!=CHECKPOINT_SHA256:
            raise ValueError(f'{arm} was not fitted from the completed pretrained checkpoint')
        checkpoints[arm]=load_selected_checkpoint(refs[arm+'_checkpoint'],selections[arm],arm)
        norms[arm]=context_scales(read(refs[arm+'_normalizers']))
        budgets[arm]=dict(selected_epoch=checkpoints[arm]['epoch'],consumed_epoch=selections[arm]['last_epoch'],
                          total_updates=selections[arm]['total_updates'])
    checkpoints['C2']=load_fixed_reference(design['reference_C2'])
    c2_study=Path(design['reference_C2']['path']).parents[1]
    refs.update(C2_checkpoint=Path(design['reference_C2']['path']),C2_selection=c2_study/'validation_selection.json',
                C2_protocol=c2_study/'protocol.json')
    c2_protocol=read(refs['C2_protocol']);c2_selection=read(refs['C2_selection'])
    if not c2_selection['controller']['converged']:raise ValueError('fixed C2 was not validation selected at plateau')
    if c2_protocol['reference_hashes']['normalizers']!=sha256(refs['parent_context_normalizers']):
        raise ValueError('C2 context normalizers differ from its original training scales')
    norms['C2']=context_scales(read(refs['parent_context_normalizers']))
    budgets['C2']=dict(selected_epoch=checkpoints['C2']['epoch'],consumed_epoch=c2_selection['last_epoch'],
                       total_updates=c2_selection['total_updates'],fixed_before_this_test_split=True)
    split=read(refs['split']);refs['data_manifest']=Path(split['cache_manifest_path'])
    conditions=split['test_conditions']
    if len(conditions)!=32 or len({item['demand_group'] for item in conditions})!=16:
        raise ValueError('prespecified 32 states / 16 demand groups required')
    hashes={key:sha256(path) for key,path in refs.items()}
    if hashes['checkpoint']!=CHECKPOINT_SHA256:raise ValueError('completed pretrained checkpoint required')
    if hashes['C2_checkpoint']!=design['reference_C2']['sha256']:raise ValueError('C2 reference changed')
    parent_cache=read(refs['parent_cache'])
    for name,expected in parent_cache.items():
        if sha256(parent/'frozen_inputs'/name)!=expected:raise ValueError('parent training cache changed')
    data_entries={item['cache_path']:item['cache_content_hash'] for item in conditions}
    for path,expected in data_entries.items():
        if archive_content_hash(Path(path))!=expected:raise ValueError('test replay content differs from fixed selection')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
        if free<args.gpu_memory_fraction*total+256*1024**2:raise RuntimeError('insufficient GPU headroom')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)
    output.mkdir(parents=True);(output/'banks').mkdir()
    protocol=dict(status='prespecified_before_shared_test',arms=list(ARMS),samples=8,sample_chunk=4,
        horizons=[1],solver='euler',steps=100,conditions=32,demand_groups=16,bootstrap_repetitions=2000,bootstrap_seed=20261021,
        device=args.device,gpu_memory_fraction=args.gpu_memory_fraction,reference_hashes=hashes,
        selections_frozen_before_test=True,training_budgets_equal=False,consumed_budgets=budgets,
        context_normalizers=norms,inputs='FULL observed current state/actions/events; HYBRIDCt target and factual following-speed mean',
        point_metrics='mean of decoded eight draws; never decoding a mean latent',
        noise='matched condition/draw seeds; G dense iid unit noise, A/B/C2 original rank16 graph noise',
        comparisons=['B_vs_A background hint','A_vs_G source mean AND covariance','C2_vs_B coarse source/reference beyond hint'],
        design_is_prespecified_input_unchanged=True,evaluator_path=str(Path(__file__).resolve()),
        limitations=['Partial adaptation of a pretrained frozen prefix, not a fully optimized Gaussian FM ceiling.',
                     'Independent validation stopping gives differing optimization budgets; no equal-budget or policy claim.'])
    atomic_json(output/'test_protocol.json',protocol)
    atomic_json(output/'selection_hashes.json',{key:dict(path=str(path.resolve()),sha256=hashes[key]) for key,path in refs.items()})
    config,prepared,topology,model,_=load_frozen(experiment,device=device)
    model.eval().requires_grad_(False);model.set_online_chunk_profile()
    native=model.dynamics.graph_prior;base_frozen=frozen_digest(model)
    priors=build_priors(model,refs['basis'],refs['weak_head']);frozen={}
    for arm in ARMS:
        model.dynamics.graph_prior=priors[arm]
        frozen[arm]=frozen_digest(model)
    model.dynamics.graph_prior=native
    head=DirectResidual().to(device).eval().requires_grad_(False)
    capacities=torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1)
    limits=torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
    scales=np.asarray(read(refs['physical_scales'])['scales']['scales'])
    data=CompactPretrainDataset(config,prepared,topology,5,cache_manifest_path=refs['data_manifest'],split='validation')
    rows=[];banks=[]
    with torch.no_grad():
        for ordinal,item in enumerate(conditions):
            index=item['episode_index']
            if data.manifest['entries'][index]['cache_path']!=item['cache_path']:raise ValueError('test condition index changed')
            batch=data.batch_from_starts(index,[item['start']],fm_only=False)
            scoped=prepare_transition_targets(batch,capacities=capacities)
            truth=endpoint_target(scoped.transition_target_flow[:,0],batch.flow[:,2],capacities,limits).numpy()
            z=model.encode(batch.flow[:,0].to(device));stored=dict(target=truth,scales=scales,initial_latent=z.cpu().numpy())
            for arm in ARMS:
                model.dynamics.graph_prior=priors[arm];load_selected(model,head,checkpoints[arm]['state'])
                with source_context(model.dynamics,head,arm,norms[arm]):
                    draws=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),
                        capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],
                        horizons=(1,),samples=8,chunk=4,steps=100)[1]
                stored[arm]=draws
                rows.append(score(draws,truth,scales,int(scoped.transition_cohort_count[0,0]),limits.numpy(),
                    dict(item,arm=arm,episode=str(item['source_path'])+'::'+str(item['episode']),steps=100,horizon=1,k=8)))
                if frozen_digest(model)!=frozen[arm] or 'prior_start' in model.dynamics.__dict__ or model.dynamics._forward_pre_hooks:
                    raise ValueError('evaluation mutated frozen weights or context')
            path=output/'banks'/f'condition_{ordinal:03d}.npz';np.savez_compressed(path,**stored)
            banks.append(dict(condition_index=item['condition_index'],path=str(path.resolve()),sha256=sha256(path)))
            atomic_json(output/'states.json',rows);atomic_json(output/'bank_manifest.json',banks)
            atomic_json(output/'progress.json',dict(conditions_complete=ordinal+1,total_conditions=32))
            print(json.dumps(dict(conditions_complete=ordinal+1,total_conditions=32)),flush=True)
    model.dynamics.graph_prior=native
    if frozen_digest(model)!=base_frozen:raise ValueError('shared frozen model changed')
    for key,path in refs.items():
        if sha256(path)!=hashes[key]:raise ValueError('input changed during test: '+key)
    for name,expected in parent_cache.items():
        if sha256(parent/'frozen_inputs'/name)!=expected:raise ValueError('parent training cache changed')
    for path,expected in data_entries.items():
        if archive_content_hash(Path(path))!=expected:raise ValueError('test replay changed')
    source_gate()
    imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
             if name.startswith(('tools.','urban_dreamer.')) and getattr(module,'__file__',None)}
    if any(not Path(path).is_relative_to(FROZEN_ROOT) for path in imports.values()):raise ValueError('live imports contaminated evaluation')
    atomic_json(output/'import_audit.json',imports)
    pairs={name:paired_metrics(rows,a,b,20261021) for name,a,b in
           [('B_vs_A','B','A'),('A_vs_G','A','G'),('C2_vs_B','C2','B')]}
    atomic_json(output/'summary.json',dict(status='complete',**study_summary(rows,20261021),pairs=pairs,
        consumed_budgets=budgets,training_budgets_equal=False,test_opened=True,source_unchanged=True,
        design_unchanged=True,parent_cache_unchanged=True,data_unchanged=True,selected_checkpoints_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--gpu-memory-fraction',type=float,default=.08)
    run(parser.parse_args())
