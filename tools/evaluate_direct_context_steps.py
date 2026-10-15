"""Fixed-head Euler4/Euler100 confirmation; no training or checkpoint selection."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = Path(os.environ.get('DIRECT_FM_FROZEN_ROOT', ROOT)).resolve()
sys.path.insert(0, str(FROZEN_ROOT))
import numpy as np
import torch
from tools.train_direct_context_fm import (ARMS, FAMILY, DirectResidual, direct_context,
    MeanOverridePrior, digest, score, trajectory_bank, endpoint_target, prepare_transition_targets,
    CompactPretrainDataset, atomic_json, load_frozen, sha256, CHECKPOINT_SHA256,
    study_summary, paired_metrics, archive_content_hash)


def read(path):
    return json.loads(Path(path).read_text())


def run(args):
    started = time.time()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    if FROZEN_ROOT != (args.output/'source').resolve():
        raise ValueError('immutable evaluation source required')
    if (args.output/'summary.json').exists():
        raise ValueError('preserve completed evaluation')
    manifest = read(args.output/'source_manifest.json')
    def source_gate():
        for name, expected in manifest.items():
            if sha256(FROZEN_ROOT/name) != expected:
                raise ValueError('frozen source changed: '+name)
    source_gate()
    parent_protocol = read(args.study/'protocol.json')
    selection = read(args.study/'validation_selection.json')
    if selection['status'] != 'frozen_before_new_blindtest':
        raise ValueError('selection must already be frozen')
    launch = read(args.study/'launch.json')['argv']
    options = {launch[i]: launch[i+1] for i in range(len(launch)-1) if launch[i].startswith('--')}
    experiment = Path(options['--experiment'])
    refs = dict(checkpoint=experiment/'snapshot/checkpoints/latest.pt',
        split=args.study/'split.json', selection=args.study/'validation_selection.json',
        normalizers=args.study/'normalizers.json', basis=Path(options['--basis']),
        weak_head=Path(options['--weak-head']), scale_reference=Path(options['--scale-reference']),
        parent_protocol=args.study/'protocol.json', parent_source=args.study/'source_manifest.json',
        cpu4_banks=args.study/'test/bank_manifest.json')
    refs.update({a: args.study/'checkpoints'/f'best_{a}.pt' for a in ARMS})
    hashes = {k:sha256(v) for k,v in refs.items()}
    if hashes['checkpoint'] != CHECKPOINT_SHA256:
        raise ValueError('completed pretrain checkpoint required')
    split = read(refs['split'])
    conditions = split['test_conditions']
    if len(conditions) != 32 or len({i['demand_group'] for i in conditions}) != 16:
        raise ValueError('same fixed 32 test conditions required')
    seed = parent_protocol['seed']
    protocol = dict(status='prespecified_before_new_forecasts', steps=[4,100], samples=8, chunk=4,
        device=args.device, gpu_memory_fraction=args.gpu_memory_fraction, precision='FP32 noAMP noTF32 noCompile',
        arms=list(ARMS), reference_hashes=hashes, training=False, selection=False,
        selected_epochs={a:selection['best'][a]['epoch'] for a in ARMS},
        weights='existing CPU4-validation-selected heads used unchanged for BOTH integration budgets',
        test='same32previouslyscoredstates/16groups; solver robustness diagnostic, NOT a fresh independent confirmation set',
        comparisons='soft_vs_zero; hint_vs_zero; shift_vs_hint at each NFE; samearm100minus4',
        cpu4='rerun all GPU4 as backend control; CPU4 differences descriptive only',
        target='FULLfactualcontrol/currentstate HYBRIDCt count, raw next/following mean speed',
        noise='same old rank16 graph Gaussian paired per-condition/per-draw seeds',
        bootstrap='2000paired demandgroups, same seed as parent; not training-seed uncertainty',
        limitations=['No retraining or NFE100 checkpoint reselection.',
            'Soft source arm changes BOTH source and reference; not a source-only causal effect.',
            'One frozen pretrained backbone and head training seed; no policy benefit claim.'])
    atomic_json(args.output/'protocol.json',protocol)
    atomic_json(args.output/'conditions.json',conditions)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    free,total = torch.cuda.mem_get_info(device)
    if free < args.gpu_memory_fraction*total + 256*1024**2:
        raise RuntimeError('insufficient GPU headroom')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)
    config,prepared,topology,model,_ = load_frozen(experiment,device=device)
    model.eval().requires_grad_(False)
    model.set_online_chunk_profile()
    native = model.dynamics.graph_prior
    original = digest(model)
    with np.load(refs['basis']) as a:
        basis,values = a['basis'],a['eigenvalues']
    with np.load(refs['weak_head']) as a:
        coefficients = a['coefficients']
    weak = MeanOverridePrior(native,operator=(basis*(1/(1+.1*values)))@basis.T,
        coefficients=coefficients,state_only=True)
    priors = dict(zero_mean=MeanOverridePrior(native,zero=True),weak_soft=weak)
    heads = {}
    for arm in ARMS:
        head = DirectResidual().to(device)
        cp = torch.load(refs[arm],map_location=device,weights_only=True)
        if cp['epoch'] != selection['best'][arm]['epoch']:
            raise ValueError('selected checkpoint changed')
        head.load_state_dict(cp['state'])
        heads[arm] = head.eval().requires_grad_(False)
    normalizers = read(refs['normalizers'])['scales']
    scales = np.asarray(read(refs['scale_reference'])['scales']['scales'])
    capacities = torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1)
    limits = torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
    data = CompactPretrainDataset(config,prepared,topology,5,
        cache_manifest_path=Path(split['cache_manifest_path']),split='validation')
    cpu_banks = {b['condition_index']:b for b in read(refs['cpu4_banks'])}
    rows=[];banks=[];backend=[];gates=[]
    (args.output/'banks').mkdir(exist_ok=True)
    with torch.inference_mode():
        for ordinal,item in enumerate(conditions):
            if data.manifest['entries'][item['episode_index']]['cache_path'] != item['cache_path']:
                raise ValueError('condition identity changed')
            batch = data.batch_from_starts(item['episode_index'],[item['start']],fm_only=False)
            target = prepare_transition_targets(batch,capacities=capacities)
            truth = endpoint_target(target.transition_target_flow[:,0],batch.flow[:,2],capacities,limits).numpy()
            z = model.encode(batch.flow[:,0].to(device))
            action,events = batch.action.to(device),batch.action_events.to(device)
            kwargs = dict(capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],
                horizons=(1,),samples=8,chunk=4)
            if ordinal == 0:
                zero = DirectResidual().to(device).eval().requires_grad_(False)
                for arm in ARMS:
                    model.dynamics.graph_prior = priors[FAMILY[arm]]
                    base = trajectory_bank(model,z,action,events,steps=4,**kwargs)[1]
                    with direct_context(model.dynamics,zero,weak,arm,normalizers):
                        noop = trajectory_bank(model,z,action,events,steps=4,**kwargs)[1]
                    np.testing.assert_array_equal(base,noop)
                    gates.append(dict(arm=arm,zero_residual_exact=True,device=str(device),steps=4))
                atomic_json(args.output/'initial_gates.json',gates)
            entry = cpu_banks[item['condition_index']]
            if sha256(Path(entry['path'])) != entry['sha256']:
                raise ValueError('CPU4 bank changed')
            with np.load(entry['path']) as old:
                np.testing.assert_array_equal(truth,old['target'])
                stored = dict(target=truth,scales=scales,initial_latent=z.cpu().numpy())
                for steps in (4,100):
                    for arm in ARMS:
                        model.dynamics.graph_prior = priors[FAMILY[arm]]
                        with direct_context(model.dynamics,heads[arm],weak,arm,normalizers):
                            draws = trajectory_bank(model,z,action,events,steps=steps,**kwargs)[1]
                        label = f'{arm}_s{steps}'
                        stored[label] = draws
                        rows.append(score(draws,truth,scales,int(target.transition_cohort_count[0,0]),limits.numpy(),
                            dict(item,arm=label,episode=str(item['source_path'])+'::'+str(item['episode']),steps=steps,horizon=1,k=8)))
                        if steps == 4:
                            delta = draws.astype(np.float64)-old[arm]
                            backend.append(dict(condition_index=item['condition_index'],arm=arm,
                                max_absolute_difference=float(np.abs(delta).max()),rms_difference=float(np.sqrt(np.square(delta).mean()))))
            model.dynamics.graph_prior = native
            if digest(model) != original or 'prior_start' in model.dynamics.__dict__ or model.dynamics._forward_pre_hooks:
                raise ValueError('original model or methods changed')
            path = args.output/'banks'/f'condition_{ordinal:03d}.npz'
            np.savez_compressed(path,**stored)
            banks.append(dict(condition_index=item['condition_index'],path=str(path.resolve()),sha256=sha256(path)))
            atomic_json(args.output/'states.json',rows)
            atomic_json(args.output/'bank_manifest.json',banks)
            progress=dict(conditions_complete=ordinal+1,total=32,elapsed_seconds=time.time()-started,
                max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2)
            atomic_json(args.output/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    pairs={}
    for steps in (4,100):
        for label,a,b in [('soft_vs_zero','soft_direct','zero_direct'),('hint_vs_zero','condition_only','zero_direct'),('shift_vs_hint','soft_direct','condition_only')]:
            pairs[f'{label}_s{steps}']=paired_metrics(rows,f'{a}_s{steps}',f'{b}_s{steps}',seed)
    for arm in ARMS:
        pairs[f'{arm}_100minus4']=paired_metrics(rows,f'{arm}_s100',f'{arm}_s4',seed)
    for k,p in refs.items():
        if sha256(p) != hashes[k]:raise ValueError('reference input changed: '+k)
    for item in conditions:
        if archive_content_hash(Path(item['cache_path'])) != item['cache_content_hash']:
            raise ValueError('replay changed')
    source_gate()
    imports = {name:str(Path(m.__file__).resolve()) for name,m in sys.modules.items()
        if name.startswith(('tools.','urban_dreamer.')) and getattr(m,'__file__',None)}
    if any(not Path(p).is_relative_to(FROZEN_ROOT) for p in imports.values()):raise ValueError('live import')
    atomic_json(args.output/'import_audit.json',imports)
    atomic_json(args.output/'cpu4_backend_differences.json',backend)
    atomic_json(args.output/'summary.json',dict(status='complete',test=study_summary(rows,seed),pairs=pairs,
        model_unchanged=True,heads_unchanged=True,inputs_unchanged=True,training=False,selection=False,
        elapsed_seconds=time.time()-started,max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--gpu-memory-fraction',type=float,default=.04)
    run(p.parse_args())
