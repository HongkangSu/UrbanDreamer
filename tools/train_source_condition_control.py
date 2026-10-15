"""Adapt source/background controls; never open the shared test split here."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = Path(os.environ.get('PARTIAL_FM_FROZEN_ROOT', ROOT)).resolve()
sys.path.insert(0, str(FROZEN_ROOT))
import numpy as np
import torch
from tools.continue_partial_context_fm import plateau_step, restore_optimizer, assert_nested_equal, save_checkpoint
from tools.run_partial_context_fm import (read, load_selected, frozen_digest, configure_trainable,
    selected_state, DirectResidual, MeanOverridePrior, gradient_proof, atomic_json,
    load_frozen, sha256, CHECKPOINT_SHA256, CompactPretrainDataset, trajectory_bank,
    endpoint_target, score, study_summary, stable_seed, prepare_transition_targets)
from tools.source_condition_controls import StandardGaussianPrior, source_context, source_cfm_loss


def install_prior(model, arm, basis_path, head_path):
    native = model.dynamics.graph_prior
    if arm == 'G':
        model.dynamics.graph_prior = StandardGaussianPrior(native, native.basis.shape[0])
    elif arm == 'A':
        model.dynamics.graph_prior = MeanOverridePrior(native, zero=True)
    else:
        with np.load(basis_path) as a:
            basis, values = a['basis'], a['eigenvalues']
        with np.load(head_path) as a:
            coefficients = a['coefficients']
        model.dynamics.graph_prior = MeanOverridePrior(native,
            operator=(basis * (1 / (1 + .1 * values))) @ basis.T,
            coefficients=coefficients, state_only=True)


def gaussian_training_noise(parent, output, arrays):
    """Keep paired time/seed draws, replacing graph coefficients by dense iid noise."""
    shape = (len(arrays['example_window']), *arrays['z'].shape[1:])
    path = output / 'gaussian_noise.npy'
    if path.exists():
        raise ValueError('preserve existing Gaussian cache')
    noise = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
    offset = 0
    for group in read(parent / 'draw_schedule.json'):
        for draw in group['draws']:
            generator = torch.Generator(device='cpu').manual_seed(draw['noise_seed'])
            noise[offset] = torch.randn(shape[1:], generator=generator).numpy()
            offset += 1
    if offset != shape[0]:
        raise ValueError('Gaussian schedule length mismatch')
    noise.flush()
    squared, count = 0., 0
    for start in range(0, shape[0], 8):
        ix = slice(start, min(start + 8, shape[0]))
        target = arrays['target'][arrays['example_window'][ix]]
        t = arrays['time'][ix, :, None, None]
        relative = (1 - t) * noise[ix] + t * target
        squared += float(np.square(relative.astype(np.float64)).sum())
        count += relative.size
    return noise, dict(relative=max(float(np.sqrt(squared / count)), 1e-8),
                       squared_sum=squared, count=count, sha256=sha256(path))


def run(args):
    started = time.time()
    output = args.study / args.arm
    output.mkdir(exist_ok=True)
    if (output / 'protocol.json').exists():
        raise ValueError('preserve existing training run')
    if FROZEN_ROOT != (args.study / 'source').resolve():
        raise ValueError('immutable source required')
    source_manifest = read(args.study / 'source_manifest.json')
    def source_gate():
        for name, expected in source_manifest.items():
            if sha256(FROZEN_ROOT / name) != expected:
                raise ValueError('frozen source changed: ' + name)
    source_gate()
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    launch = read(args.parent / 'launch.json')['argv']
    options = {launch[i]: launch[i+1] for i in range(len(launch)-1) if launch[i].startswith('--')}
    experiment = Path(options['--experiment'])
    refs = dict(checkpoint=experiment/'snapshot/checkpoints/latest.pt',
        basis=Path(options['--basis']), weak_head=Path(options['--weak-head']),
        scale_reference=Path(options['--scale-reference']), split=args.study/'split.json',
        parent_cache=args.parent/'frozen_input_manifest.json',
        parent_normalizers=args.parent/'normalizers.json', parent_selection=args.parent/'validation_selection.json')
    if args.arm == 'B':
        refs['resume'] = args.parent/'checkpoints/last_B.pt'
        refs['parent_best'] = args.parent/'checkpoints/best_B.pt'
    hashes = {k: sha256(p) for k,p in refs.items()}
    if hashes['checkpoint'] != CHECKPOINT_SHA256:
        raise ValueError('completed pretrain required')
    split = read(refs['split']); parent_split = read(args.parent/'split.json')
    for key in ('train', 'validation'):
        if split[key] != parent_split[key]:
            raise ValueError('train/validation groups changed')
    strip = lambda items: [{k:v for k,v in item.items() if k != 'episode_index'} for item in items]
    if strip(split['validation_conditions']) != strip(parent_split['validation_conditions']):
        raise ValueError('validation states changed')
    groups = [{e['demand_group'] for e in split[s]} for s in ('train','validation','test')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('split overlap')
    cache_manifest = read(refs['parent_cache'])
    for name, expected in cache_manifest.items():
        if sha256(args.parent/'frozen_inputs'/name) != expected:
            raise ValueError('parent cache changed')
    arrays = {Path(name).stem: np.load(args.parent/'frozen_inputs'/name, mmap_mode='r') for name in cache_manifest}
    mapping = arrays['example_window']
    if len(mapping) != 3072:
        raise ValueError('3072 graph views required')
    normalizers = copy.deepcopy(read(refs['parent_normalizers'])['scales'])
    noise = arrays['noise']; noise_audit = None
    if args.arm == 'G':
        noise, noise_audit = gaussian_training_noise(args.parent, output, arrays)
        normalizers['relative'] = noise_audit['relative']
    atomic_json(output/'normalizers.json', dict(scales=normalizers, fit_split='trainonly',
        centering=False, gaussian_source_audit=noise_audit,
        note='G refits only relative-Xt RMS for its dense unit source; Z/b RMS shared, A/G b identically zero'))
    protocol = dict(arm=args.arm, status='prespecified_before_updates_or_validation_scores',
        reference_hashes=hashes, seed=20261021, trainable_parameters=1414528,
        source={'G':'N(0,I) iid full-road full-latent', 'A':'Z+native rank16 graph noise', 'B':'Z+native rank16 graph noise'}[args.arm],
        background_condition=args.arm=='B', reference='Z', legitimate_conditions='current Z and recorded action/events retained',
        training='3072 paired whole-graph views, B2, full velocity CFM; same global epoch shuffle',
        controller=dict(patience=5, meaningful_relative_improvement=.002, best_epsilon=1e-5,
                        lr_factor=.3, lr_reductions=2, maximum_global_epoch=args.maximum_epoch),
        stopping='validation plateau after two LR reductions; cap leaves shared test closed',
        validation='same 8 states/4 groups, K4 Euler100 each epoch; select minimum ES',
        test='NEVER read by this trainer; common fresh 32 states/16 groups after all arms select',
        freeze='completed pretrained codec/prior/action encoder/first5Transformerblocks frozen; gradients through prefix retained',
        precision='FP32 no AMP/TF32/compile; dropout disabled', device=args.device,
        caveats=['Partial adaptation, not full joint optimization or a Gaussian FM performance ceiling.',
                 'G/A change source mean AND covariance. Only A/B isolates the background hint.',
                 'Same controller, differing updates after independent stopping; not equal training budgets.'])
    atomic_json(output/'protocol.json', protocol)
    device = torch.device(args.device); torch.cuda.set_device(device)
    free,total = torch.cuda.mem_get_info(device)
    if free < args.gpu_memory_fraction*total + 256*1024**2:
        raise RuntimeError('insufficient GPU headroom')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    config,prepared,topology,model,_ = load_frozen(experiment, device=device)
    model.eval().requires_grad_(False); model.set_online_chunk_profile()
    install_prior(model, args.arm, refs['basis'], refs['weak_head'])
    torch.manual_seed(20261021); head = DirectResidual().to(device)
    if args.arm == 'B':
        payload = torch.load(refs['resume'], map_location='cpu', weights_only=True)
        load_selected(model, head, payload['state'])
        parameters,opt = restore_optimizer(model, head, payload)
        assert_nested_equal(opt.state_dict(), payload['optimizer'])
        first_epoch = int(payload['epoch'])
        if {int(float(v['step'])) for v in payload['optimizer']['state'].values()} != {first_epoch*1536}:
            raise ValueError('resume AdamW step mismatch')
        old = read(refs['parent_selection'])['best']['B']
        controller = dict(best_ES=old['ES'], best_epoch=old['epoch'], anchor_ES=old['reset_ES'],
            anchor_epoch=old['reset_epoch'], patience=old['patience'], reductions=0,
            lrs=[g['lr'] for g in opt.param_groups], converged=False)
        best_state = copy.deepcopy(torch.load(refs['parent_best'], map_location='cpu', weights_only=True)['state'])
    else:
        first_epoch = 0
        parameters = configure_trainable(model, head)
        opt = torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.requires_grad],lr=.0001),
                                dict(params=list(head.parameters()),lr=.001)],weight_decay=.0001)
        controller = None
        best_state = selected_state(model, head)
    frozen = frozen_digest(model)
    capacities = torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1)
    limits = torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
    scales = np.asarray(read(refs['scale_reference'])['scales']['scales'])
    data = CompactPretrainDataset(config,prepared,topology,5,
        cache_manifest_path=Path(split['cache_manifest_path']),split='validation')
    valdir = output/'validation'; valdir.mkdir(exist_ok=True)
    check = output/'checkpoints'; check.mkdir(exist_ok=True)
    def unchanged():
        if frozen_digest(model)!=frozen or 'prior_start' in model.dynamics.__dict__ or model.dynamics._forward_pre_hooks:
            raise ValueError('frozen weights or source context changed')
    def forecast(epoch):
        directory=valdir/f'epoch{epoch:03d}'; directory.mkdir(exist_ok=True); rows=[]; banks=[]
        with torch.no_grad():
            for ordinal,item in enumerate(split['validation_conditions']):
                if data.manifest['entries'][item['episode_index']]['cache_path']!=item['cache_path']:
                    raise ValueError('validation identity changed')
                batch=data.batch_from_starts(item['episode_index'],[item['start']],fm_only=False)
                scoped=prepare_transition_targets(batch,capacities=capacities)
                truth=endpoint_target(scoped.transition_target_flow[:,0],batch.flow[:,2],capacities,limits).numpy()
                z=model.encode(batch.flow[:,0].to(device))
                with source_context(model.dynamics,head,args.arm,normalizers):
                    draws=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),
                        capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],
                        horizons=(1,),samples=4,chunk=4,steps=100)[1]
                row=score(draws,truth,scales,int(scoped.transition_cohort_count[0,0]),limits.numpy(),
                    dict(item,arm=args.arm,episode=str(item['source_path'])+'::'+str(item['episode']),steps=100,horizon=1,k=4))
                rows.append(row); path=directory/f'condition_{ordinal:03d}.npz'
                np.savez_compressed(path,target=truth,scales=scales,**{args.arm:draws})
                banks.append(dict(condition_index=item['condition_index'],path=str(path.resolve()),sha256=sha256(path)))
        unchanged();atomic_json(directory/'states.json',rows);atomic_json(directory/'bank_manifest.json',banks)
        es=float(np.mean([r['energy_score'] for r in rows]))
        atomic_json(valdir/f'epoch{epoch:03d}_summary.json',dict(**study_summary(rows,20261021),rows=rows,banks=banks))
        return es,banks
    initial_es,initial_banks=forecast(first_epoch)
    if args.arm=='B':
        old_banks=read(args.parent/f'validation/epoch{first_epoch:03d}/bank_manifest.json')
        gates=[]
        for new,old in zip(initial_banks,old_banks):
            if new['condition_index']!=old['condition_index'] or sha256(Path(old['path']))!=old['sha256']:
                raise ValueError('B resume bank mismatch')
            with np.load(new['path']) as actual,np.load(old['path']) as expected:
                np.testing.assert_array_equal(actual['target'],expected['target'])
                np.testing.assert_allclose(actual['B'],expected['B'],rtol=1e-5,atol=1e-5)
                gates.append(dict(condition_index=new['condition_index'],max_abs=float(np.abs(actual['B']-expected['B']).max())))
        atomic_json(output/'resume_gates.json',dict(optimizer_state_bitexact=True,initial_epoch=first_epoch,forward=gates))
    else:
        controller=dict(best_ES=initial_es,best_epoch=0,anchor_ES=initial_es,anchor_epoch=0,
                        patience=0,reductions=0,lrs=[g['lr'] for g in opt.param_groups],converged=False)
    atomic_json(output/'initial_controller.json',controller)
    history=[]; gradient=None; reason='maximum_epoch_cap'
    for epoch in range(first_epoch+1,args.maximum_epoch+1):
        begin=time.time(); order=np.random.default_rng(stable_seed(20261021,'partial_training_order',epoch)).permutation(len(mapping))
        loss_sum=0.; updates=0; lrs_used=[g['lr'] for g in opt.param_groups]
        for start in range(0,len(order),2):
            ix=order[start:start+2];wi=mapping[ix]
            batch={k:torch.from_numpy(np.array(arrays[k][wi],copy=True)).to(device) for k in ('z','target','b','condition','action')}
            batch.update(time=torch.from_numpy(np.array(arrays['time'][ix],copy=True)).to(device),
                         noise=torch.from_numpy(np.array(noise[ix],copy=True)).to(device))
            opt.zero_grad(set_to_none=True)
            loss=source_cfm_loss(model,head,args.arm,batch,normalizers)
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite CFM loss')
            loss.backward()
            if gradient is None:
                gradient=gradient_proof(model,head);atomic_json(output/'gradient_proof.json',gradient)
            torch.nn.utils.clip_grad_norm_(parameters,1.);opt.step()
            loss_sum+=float(loss.detach())*len(ix);updates+=1
        es,banks=forecast(epoch)
        controller,improved,event=plateau_step(controller,es,epoch)
        if improved:best_state=selected_state(model,head)
        for group,lr in zip(opt.param_groups,controller['lrs']):group['lr']=lr
        row=dict(epoch=epoch,updates=updates,total_updates=epoch*1536,train_CFM_MSE=loss_sum/len(mapping),
            validation_ES=es,lrs_used=lrs_used,controller=copy.deepcopy(controller),event=event,
            best_update=improved,epoch_seconds=time.time()-begin)
        history.append(row);atomic_json(output/'training_history.json',history)
        save_checkpoint(check/f'last_{args.arm}.pt',dict(epoch=epoch,state=selected_state(model,head),
            optimizer=opt.state_dict(),best_state=best_state,controller=controller,history=history,validation_banks=banks))
        save_checkpoint(check/f'best_{args.arm}.pt',dict(epoch=controller['best_epoch'],validation_ES=controller['best_ES'],state=best_state))
        progress=dict(stage='training',arm=args.arm,**row,elapsed_seconds=time.time()-started,
                      max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2)
        atomic_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if controller['converged']:reason='validation_plateau_at_min_lr';break
    unchanged();source_gate()
    for key,path in refs.items():
        if sha256(path)!=hashes[key]:raise ValueError('reference changed: '+key)
    for name,expected in cache_manifest.items():
        if sha256(args.parent/'frozen_inputs'/name)!=expected:raise ValueError('parent cache changed')
    imports={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items()
             if n.startswith(('tools.','urban_dreamer.')) and getattr(m,'__file__',None)}
    if any(not Path(p).is_relative_to(FROZEN_ROOT) for p in imports.values()):raise ValueError('live import')
    atomic_json(output/'import_audit.json',imports)
    selection=dict(status='ready_for_shared_test' if controller['converged'] else 'budget_limited_test_closed',
        controller=controller,last_epoch=epoch,total_updates=epoch*1536,continued_updates=(epoch-first_epoch)*1536,
        reason=reason,budget_limited=not controller['converged'])
    atomic_json(output/'validation_selection.json',selection)
    atomic_json(output/'summary.json',dict(status=selection['status'],arm=args.arm,selection=selection,
        test_opened=False,frozen_state_unchanged=True,source_unchanged=True,parent_cache_unchanged=True,
        elapsed_seconds=time.time()-started,max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('A','G','B'),required=True)
    for name in ('parent','study'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--maximum-epoch',type=int,default=80)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--gpu-memory-fraction',type=float,default=.08)
    run(p.parse_args())
