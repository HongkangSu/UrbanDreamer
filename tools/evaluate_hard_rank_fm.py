"""Hard PRIOR-MEAN rank interventions in complete pretrained Euler100 FM.

Prediction rank is independent of the immutable checkpoint rank16 noise.
The main family uses state-only, HYBRID-Ct-supervised solved output heads.
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
FROZEN_ROOT=Path(os.environ.get('HARD_RANK_FM_FROZEN_ROOT',ROOT)).resolve()
sys.path.insert(0,str(FROZEN_ROOT))
import numpy as np
import torch

from tools.evaluate_soft_prior_fm import MeanOverridePrior,digest,summaries,score
from tools.component_experiment_data import atomic_json,load_frozen,dataset,sha256
from tools.evaluate_pretrained_distribution import CHECKPOINT_SHA256,trajectory_bank
from tools.evaluate_component_world_models import endpoint_target
from tools.evaluate_prior_input_filter import state_hidden
from tools.evaluate_frequency_predictability import augmented
from tools.evaluate_pretrained_prior_ranks import solve_ridge
from tools.analyze_traffic_spectrum import select_starts
from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.learning.offline_data import archive_content_hash
from urban_dreamer.models.world_model import ConditionEncoding

RANKS=(16,32,64,128,256)
MAIN=('native',*(f'state_r{r}' for r in RANKS))
AUX=('conditional_refit_r16','original_r64','original_r128','conditional_refit_r64','conditional_refit_r128')


def complete_rank(values,rank,tolerance=1e-8):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=1 or not np.isfinite(values).all() or np.any(np.diff(values)<0) or not 1<=rank<=len(values):
        raise ValueError('sorted finite spectrum and valid positive rank required')
    end=int(rank)
    while end<len(values) and values[end]-values[end-1]<=tolerance:end+=1
    return end


class HardMeanPrior(MeanOverridePrior):
    """Thin hard projection for means; inherited native basis remains noise-only."""
    def __init__(self,native,prediction_basis,*,coefficients=None,state_only=False):
        super().__init__(native,coefficients=coefficients,state_only=state_only)
        value=torch.as_tensor(prediction_basis,device=native.basis.device,dtype=native.basis.dtype).clone()
        if value.ndim!=2 or value.shape[0]!=len(native.basis) or not 1<=value.shape[1]<=len(native.basis) or not bool(torch.isfinite(value).all()):
            raise ValueError('finite matching thin prediction basis required')
        self.register_buffer('prediction_basis',value,persistent=False)

    def project(self,values):
        basis=self.prediction_basis.to(values)
        return basis@(basis.T@values)

    @property
    def projection_rank(self):return self.prediction_basis.shape[1]


def fit_state_heads(args,basis,actual_ranks):
    """One TRAIN-only CPU pass, common frozen state front and equal head sizes."""
    started=time.time();config,prepared,topology,model,_=load_frozen(args.experiment,device='cpu')
    model.eval().requires_grad_(False)
    capacities=np.maximum(np.asarray(topology.network.capacities,dtype=np.float64),1.)
    selection=json.loads(args.train_selection.read_text())['train']
    source=dataset(args.experiment,config,prepared,topology,split='train',horizon=1)
    indices={r['cache_path']:i for i,r in enumerate(source.manifest['entries'])}
    stats={r:dict(gram=np.zeros((65,65)),cross=np.zeros((65,64)),energy=0.) for r in RANKS}
    audit=[];top=basis[:,:max(actual_ranks.values())]
    with torch.inference_mode():
        for ordinal,entry in enumerate(selection):
            index=indices[entry['cache_path']];arrays=source._episode(index);starts=select_starts(arrays['flow'],12)
            batch=source.batch_from_starts(index,starts.tolist(),materialization_horizon=1,fm_only=True)
            target=prepare_transition_targets(batch,capacities=capacities);z=model.encode(batch.flow[:,0])
            delta=(model.encode(target.transition_target_flow[:,0])-z).double().numpy()
            x=augmented(state_hidden(model.dynamics.graph_prior,z[:,None])[:,0].numpy())
            xs=np.einsum('nr,tnc->trc',top,x,optimize=True);ys=np.einsum('nr,tnc->trc',top,delta,optimize=True)
            for rank,actual in actual_ranks.items():
                xf=xs[:,:actual].reshape(-1,65);yf=ys[:,:actual].reshape(-1,64)
                stats[rank]['gram']+=xf.T@xf;stats[rank]['cross']+=xf.T@yf;stats[rank]['energy']+=float(np.square(yf).sum())
            audit.append(dict(cache_path=entry['cache_path'],content_hash=entry['cache_content_hash'],demand_group=entry['demand_group'],starts=starts.tolist()))
            print(json.dumps(dict(stage='fit_state_heads',group=ordinal+1,total=len(selection),elapsed_seconds=time.time()-started)),flush=True)
    heads={};fits={}
    for rank,bucket in stats.items():
        heads[rank],fits[str(rank)]=solve_ridge(bucket['gram'],bucket['cross'],bucket['energy'])
    for entry in audit:
        if archive_content_hash(Path(entry['cache_path']))!=entry['content_hash']:raise ValueError('training cache changed')
    atomic_json(args.output/'head_fit_diagnostics.json',dict(status='complete',training_groups=len(selection),windows=sum(len(r['starts']) for r in audit),
        validation_used=False,output_shape=[65,64],parameters_per_head=4160,actual_ranks=actual_ranks,fits=fits,elapsed_seconds=time.time()-started))
    atomic_json(args.output/'training_data_audit.json',audit)
    np.savez_compressed(args.output/'state_heads.npz',**{f'rank_{r}':h for r,h in heads.items()})
    np.savez_compressed(args.output/'head_training_sufficient_statistics.npz',**{f'rank_{r}__{k}':v for r,b in stats.items() for k,v in b.items()})
    return heads


def run(args):
    started=time.time();torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    if FROZEN_ROOT!=(args.output/'source').resolve():raise ValueError('new immutable source required')
    if (args.output/'summary.json').exists():raise ValueError('preserve completed study')
    manifest=json.loads((args.output/'source_manifest.json').read_text())
    def verify_source():
        for path,expected in manifest.items():
            if sha256(FROZEN_ROOT/path)!=expected:raise ValueError('frozen source changed')
    verify_source();(args.output/'banks').mkdir(exist_ok=True)
    refs={'checkpoint':args.experiment/'snapshot/checkpoints/latest.pt','basis':args.basis,'train_selection':args.train_selection,
        'conditional_heads':args.conditional_heads,'conditional_protocol':args.conditional_heads.parent/'protocol.json',
        'conditions':args.native_reference/'conditions.json','native_banks':args.native_reference/'bank_manifest.json',
        'native_summary':args.native_reference/'summary.json','cache_manifest':args.cache_manifest,'scales':args.scale_reference}
    hashes={k:sha256(v) for k,v in refs.items()}
    if hashes['checkpoint']!=CHECKPOINT_SHA256:raise ValueError('completed checkpoint required')
    with np.load(args.basis) as asset:basis=asset['basis'];values=asset['eigenvalues']
    actual={r:complete_rank(values,r) for r in RANKS}
    conditions=json.loads((args.native_reference/'conditions.json').read_text())
    if len(conditions)!=32 or len({r['demand_group'] for r in conditions})!=16:raise ValueError('fixed32conditions16groups required')
    training_groups={r['demand_group'] for r in json.loads(args.train_selection.read_text())['train']}
    if training_groups&{r['demand_group'] for r in conditions}:raise ValueError('training/evaluation groups overlap')
    conditional_protocol=json.loads(refs['conditional_protocol'].read_text())
    if conditional_protocol['checkpoint_sha256']!=CHECKPOINT_SHA256 or 'hybrid' not in conditional_protocol['target'].lower():
        raise ValueError('conditional refits must use completed checkpoint and hybridCt labels')
    protocol=dict(status='prespecified_before_scores',checkpoint_sha256=CHECKPOINT_SHA256,reference_hashes=hashes,bootstrap_seed=args.seed,
        main_arms=list(MAIN),main_conditions=32,main_demand_groups=16,
        auxiliary_arms=list(AUX),auxiliary_conditions='first16fixedconditions,8groups; all compared to native/state on exactly same subset',
        rationale='state-only background is main hypothesis; limited conditional refit/original controls isolate action conditioning and head refit from rank',
        requested_ranks=list(RANKS),actual_complete_eigenspace_ranks=actual,tie_tolerance=1e-8,
        source_noise='ALWAYS original checkpoint storedrank16 basis/scale andmatched8CPUdrawseeds; changingglobal low_frequency_rank wouldalsochangenoise andisNOTthisexperiment',
        mean_projection='exact full symnormalized Laplacian lowest modes; include entire repeated boundary eigenspaces',
        main_heads='state-only same frozen body, fullinput, HYBRIDCt filtered-target F64ridge,64train768windows,4160params/head',
        aux_heads='original actionconditioned body/head projection64/128; archived conditionalrefit16/64/128 fullysolvedsameHYBRID/factualcontrols',
        metrics='mean of decoded8samples MAE/pooledRMSE; all8CRPS/EnergyScore/90%coverage/totalcount; actualtarget occupiedmask',
        inputs='FULL observed initialstate,FULLfactualactions/events inclwithininterval births',target='HYBRIDCt count plus RAWnext/followingmean speed',
        steps=100,horizons=[1],sample_chunk=args.chunk,samples=8,device=args.device,gpu_memory_fraction=args.gpu_memory_fraction,
        execution_profile='FP32online chunk profile; deterministicCUDA/CUBLAS; TF32off; noAMP/compile',
        native_gate='first2native/clonebitidentity andpriorreferenceparity mean1e-5/max1e-3 beforeinterventions; verifyallcachednative hashesbefore reuse',
        runtime='Compact matrix fixed before scores: 5stateheads×32 +5aux×16; noH4, no score-based trimming. Remainingoriginal/conditional ranksnotpartofmatrix.',
        bootstrap='2000pairedwhole demandgroups; samepopulation within each family; 8group AUX separated from16group MAIN',
        full_world_model_training=False,FM_body_changed=False,config_changed=False,
        caveats=['Frozen FM was trained nativehard/actionmeanrank16; this tests explicit inferenceinterventions, not fullyjointtrainedlarger-rankworldmodels.',
            'Body andencoder learned withrank16; outputheadrefit only isfullysolved, not newbodytraining.',
            'State-only predicts behaviour-averaged background, not zero-action causal dynamics.',
            'One pretrained checkpoint; CIs aredemand variability, not trainingseed uncertainty; development notexternaltest.'])
    atomic_json(args.output/'protocol.json',protocol);atomic_json(args.output/'conditions.json',conditions)
    state_heads=fit_state_heads(args,basis,actual)
    device=torch.device(args.device);torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
    if free<args.gpu_memory_fraction*total+256*1024**2:raise RuntimeError('insufficient GPU admission headroom')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)
    config,prepared,topology,model,_=load_frozen(args.experiment,device=device);model.eval().requires_grad_(False);model.set_online_chunk_profile()
    native=model.dynamics.graph_prior;original=digest(model);body=digest(model,True)
    capacities=torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1);limits=torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
    data=CompactPretrainDataset(config,prepared,topology,5,cache_manifest_path=args.cache_manifest,split='validation')
    with np.load(args.conditional_heads) as heads:
        original_coeff=np.concatenate((native.output.weight.cpu().numpy().T,native.output.bias.cpu().numpy()[None]),axis=0)
        np.testing.assert_array_equal(original_coeff.astype(np.float64),heads['original'])
        variants={f'state_r{r}':HardMeanPrior(native,basis[:,:actual[r]],coefficients=state_heads[r],state_only=True) for r in RANKS}
        for r in (64,128):variants[f'original_r{r}']=HardMeanPrior(native,basis[:,:actual[r]])
        for r in (16,64,128):variants[f'conditional_refit_r{r}']=HardMeanPrior(native,basis[:,:actual[r]],coefficients=heads[f'rank_{r}'])
    scale_meta=json.loads(args.scale_reference.read_text())['scales'];scales=np.asarray(scale_meta['scales'])
    banks=json.loads((args.native_reference/'bank_manifest.json').read_text())
    indexed={b['condition_index']:b for b in banks}
    for bank in banks:
        if sha256(Path(bank['path']))!=bank['sha256']:raise ValueError('cachednativebank changed')
    def materialize(item):
        batch=data.batch_from_starts(item['episode_index'],[item['start']],fm_only=False)
        target=prepare_transition_targets(batch,capacities=capacities);z=model.encode(batch.flow[:,0].to(device))
        truth=endpoint_target(target.transition_target_flow[:,0],batch.flow[:,2],capacities,limits).numpy()
        count=int(target.transition_cohort_count[0,0]);options=dict(capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(1,),samples=8,chunk=args.chunk,steps=100)
        return batch,z,truth,count,options
    gates=[];rows=[];noise_proof=[];bank_manifest=[]
    with torch.inference_mode():
        for item in conditions[:2]:
            batch,z,truth,count,options=materialize(item);model.dynamics.graph_prior=native
            predicted=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),**options)[1]
            model.dynamics.graph_prior=MeanOverridePrior(native)
            cloned=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),**options)[1]
            np.testing.assert_array_equal(predicted,cloned)
            with np.load(indexed[item['condition_index']]['path']) as saved:
                expected=saved['native'];np.testing.assert_array_equal(truth,saved['target'])
            difference=np.abs(predicted-expected);mean=np.abs(predicted.mean(0)-expected.mean(0))
            gate=dict(condition_index=item['condition_index'],native_clone_bit_equal=True,mean_count_drift=float(mean[...,0].mean()),mean_speed_drift=float(mean[...,1].mean()),max_count_drift=float(difference[...,0].max()),max_speed_drift=float(difference[...,1].max()))
            atomic_json(args.output/'native_gate.json',gates+[gate])
            if max(gate['mean_count_drift'],gate['mean_speed_drift'])>=1e-5 or max(gate['max_count_drift'],gate['max_speed_drift'])>=1e-3:raise ValueError('native reference parity failed')
            gates.append(gate)
        for item in conditions:
            batch,z,truth,count,options=materialize(item);index=item['condition_index']
            with np.load(indexed[index]['path']) as previous:
                reference=previous['native'].copy();np.testing.assert_array_equal(truth,previous['target'])
                coefficients=previous['source_coefficients'].copy()
            active=(*MAIN[1:],*(AUX if index<16 else ()))
            stored=dict(target=truth,scales=scales,initial_latent=z.cpu().numpy(),source_coefficients=coefficients,
                native_noise_basis=native.basis.cpu().numpy(),native_noise_scale=np.array(native.noise_scale),native=reference)
            rows.append(score(reference,truth,scales,count,limits.numpy(),dict(item,arm='native',steps=100,horizon=1,k=8,family='main',control_scope='full_factual_hybrid_Ct',noise_rank=16)))
            for arm in active:
                prior=variants[arm]
                for begin in (0,4):
                    rep=z.repeat(4,1,1)[:,None];noise=torch.as_tensor(coefficients[begin:begin+4],device=device,dtype=z.dtype)
                    torch.testing.assert_close(prior.sample_noise(rep,noise),native.sample_noise(rep,noise),rtol=0,atol=0)
                if arm.startswith('state'):
                    a=ConditionEncoding(torch.zeros(1,1,len(basis),256,device=device));b=ConditionEncoding(torch.ones_like(a.edge_tokens)*51)
                    torch.testing.assert_close(prior(z[:,None],a),prior(z[:,None],b),rtol=0,atol=0)
                model.dynamics.graph_prior=prior
                draws=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),**options)[1];stored[arm]=draws
                nominal=int(arm.split('_r')[-1])
                rows.append(score(draws,truth,scales,count,limits.numpy(),dict(item,arm=arm,steps=100,horizon=1,k=8,
                    family='main' if arm.startswith('state') else 'auxiliary',requested_rank=nominal,prediction_rank=actual[nominal],noise_rank=16,control_scope='full_factual_hybrid_Ct')))
            model.dynamics.graph_prior=native
            path=args.output/'banks'/f'condition_{index:03d}.npz';np.savez_compressed(path,**stored)
            bank_manifest.append(dict(condition_index=index,path=str(path.resolve()),sha256=sha256(path)))
            noise_proof.append(dict(condition_index=index,arms=list(active),all_noise_bit_equal=True,noise_rank=16,state_action_invariant=True))
            with (args.output/'states.jsonl').open('w') as stream:
                for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
            atomic_json(args.output/'bank_manifest.json',bank_manifest);atomic_json(args.output/'noise_proof.json',noise_proof)
            progress=dict(stage='complete_fm',conditions_complete=index+1,total_conditions=32,elapsed_seconds=time.time()-started,
                max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2)
            atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            if index==3:
                atomic_json(args.output/'pilot_summary.json',summaries(rows,args.seed))
                atomic_json(args.output/'runtime_note.json',dict(score_selection=False,pilot_completed=4,scope_matrix_unchanged=True,
                    explanation='Compactmain32/aux16 matrix frozen upfront; optional fullconditional/original grid excludedbefore scores.'))
    model.dynamics.graph_prior=native
    if digest(model)!=original or digest(model,True)!=body:raise ValueError('native model/body changed')
    for name,path in refs.items():
        if sha256(path)!=hashes[name]:raise ValueError('reference changed')
    for item in conditions:
        if archive_content_hash(Path(item['cache_path']))!=item['cache_content_hash']:raise ValueError('cache changed')
    verify_source();imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
        if name.startswith(('tools.','urban_dreamer.')) and getattr(module,'__file__',None)}
    if any(not Path(p).is_relative_to(FROZEN_ROOT) for p in imports.values()):raise ValueError('live import contaminatedfrozen run')
    atomic_json(args.output/'import_audit.json',imports)
    main=summaries([r for r in rows if r['arm'] in MAIN],args.seed)
    auxiliary=summaries([r for r in rows if r['condition_index']<16],args.seed)
    atomic_json(args.output/'summary.json',dict(status='complete',protocol=protocol,main=main,auxiliary=auxiliary,
        native_model_digest=original,body_digest=body,body_unchanged=True,source_unchanged=True,noise_unchanged=True,
        checkpoint_unchanged=True,data_unchanged=True,elapsed_seconds=time.time()-started))
    print(json.dumps(dict(stage='complete',elapsed_seconds=time.time()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','basis','train-selection','conditional-heads','native-reference','cache-manifest','scale-reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--gpu-memory-fraction',type=float,default=.03)
    p.add_argument('--chunk',type=int,default=4);p.add_argument('--seed',type=int,default=20261017)
    run(p.parse_args())
