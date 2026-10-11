"""Small paired fresh-dynamics ablation; never resumes or updates the full mainline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.models.world_model import Dynamics, FlowProjector, ConditionEncoding, independent_transition_fm_loss
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.checkpoint import load_checkpoint, CheckpointPhase
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.pipeline import combined_input_hash
from tools.analyze_world_model_diversity import summarize_samples


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def module_hash(module):
    digest=hashlib.sha256()
    for name,value in module.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def paired_rng(seed,step,batch,horizon,rank):
    generator=torch.Generator(device='cpu').manual_seed(seed+step)
    tau=torch.rand(batch,horizon,generator=generator,dtype=torch.float32)
    noise=torch.randn(batch,horizon,rank,64,generator=generator,dtype=torch.float32)
    return tau,noise


def make_schedule(entries,updates,batch,horizon,seed):
    eligible=[index for index,entry in enumerate(entries) if entry['windows']>=horizon+batch-1]
    if not eligible:
        raise ValueError('no sufficiently long training episodes for the paired schedule')
    rng=np.random.default_rng(seed);order=rng.permutation(eligible);schedule=[]
    for step in range(updates):
        index=int(order[step%len(order)]);entry=entries[index]
        legal=entry['windows']-horizon+1
        starts=rng.choice(legal,size=batch,replace=False).astype(int).tolist()
        schedule.append(dict(step=step,episode_index=index,starts=starts,
            source_path=entry['path'],source_content_hash=entry['content_hash'],
            cache_content_hash=entry['cache_content_hash'],demand_group=entry['demand_group'],method=entry['method']))
    return schedule


def open_sources(run_dir):
    config=load_config(run_dir/'run_config.toml')
    prepared=validate_prepared_inputs(config);topology=load_traffic_topology(prepared)
    loaded=load_checkpoint(output_dir=run_dir,expected_config_fingerprint=config.semantic_fingerprint,
                           expected_input_hash=combined_input_hash(prepared))
    if (loaded.schema_version!=10 or loaded.phase!=CheckpointPhase.ONLINE_TRAINING or loaded.next_unit!=0
            or loaded.training_state['metrics']['world_model_initialization_epochs_completed']!=5
            or loaded.training_state['metrics']['codec_updates']!=6500
            or loaded.training_state['metrics']['online_episodes_completed']!=0):
        raise ValueError('requires the completed schema10 pretrain checkpoint and shared6500-update codec')
    sources={split:CompactPretrainDataset(config,prepared,topology,3,
        cache_manifest_path=config.data.pretrain_cache_manifest,split=split) for split in ('train','validation')}
    groups={split:{row['demand_group'] for row in source.manifest['entries']} for split,source in sources.items()}
    if groups['train']&groups['validation']:
        raise ValueError('paired training and held-out demands overlap')
    return config,topology,loaded,sources


def evaluate(codec,dynamics,source,conditions,config,network,device,seed):
    results=[];capacities=np.maximum(network.capacities,1.);limits=network.speed_limits_mps
    codec.eval();dynamics.eval()
    with torch.no_grad():
        for index,condition in enumerate(conditions):
            batch=source.batch_from_starts(condition['episode_index'],[condition['start']],fm_only=True)
            scoped=prepare_transition_targets(batch,capacities=network.capacities)
            target=scoped.transition_target_flow[0,0].numpy()
            speed=(batch.flow[0,1,:,1]+batch.flow[0,2,:,1]).numpy()/2.
            z=codec(batch.flow[:,0].to(device)).unsqueeze(1)
            action=batch.action[:,:1].to(device)
            background=dynamics.encode_condition(action,action_events=batch.action_events.for_step(0).to(device))
            coarse=z+dynamics.graph_prior(z,background)
            context=ConditionEncoding(background.edge_tokens,dynamics.reference_proj(coarse))
            generator=torch.Generator(device='cpu').manual_seed(seed+index)
            noise=torch.randn(32,1,dynamics.graph_prior.rank,64,generator=generator)
            values=[]
            for sigma in (0.,.1):
                dynamics.graph_prior.noise_scale=sigma;starts=[];ends=[];decoded=[]
                for offset in range(0,32,4):
                    previous=z.repeat(4,1,1,1)
                    start=coarse.repeat(4,1,1,1)+dynamics.graph_prior.sample_noise(previous,noise[offset:offset+4].to(device))
                    cached=ConditionEncoding(context.edge_tokens.repeat(4,1,1,1),context.reference_tokens.repeat(4,1,1,1))
                    endpoint=dynamics.solve(start,action.repeat(4,1,1),cached,steps=config.method.fm_steps,
                        solver=config.method.fm_solver,picard_iterations=config.method.picard_iterations)
                    starts.append(start[:,0].cpu().numpy());ends.append(endpoint[:,0].cpu().numpy())
                    decoded.append(codec.decode(endpoint[:,0]).cpu().numpy())
                values.append(dict(sigma=sigma,**summarize_samples(np.concatenate(starts),np.concatenate(ends),
                    np.concatenate(decoded),target,speed,capacities,limits,int(scoped.transition_cohort_count[0,0]))))
            results.append(dict(condition=condition,sigmas=values))
    # aggregate_rows uses four cells; pair direct means to avoid fabricating cells.
    aggregate=[]
    for index,sigma in enumerate((0.,.1)):
        rows=[row['sigmas'][index] for row in results];summary=dict(test_sigma=sigma,conditions=len(rows))
        for name,value in rows[0].items():
            if isinstance(value,(int,float)) and not isinstance(value,bool) and name!='sigma':
                candidates=[row[name] for row in rows if row[name] is not None]
                summary[name]=float(np.mean(candidates)) if candidates else None
        occupied=sum(row['occupied_roads'] for row in rows)
        summary['occupied_road_weighted_coverage90']=sum(row['interval90_coverage_occupied']*row['occupied_roads']
            for row in rows if row['occupied_roads'])/occupied
        aggregate.append(summary)
    return dict(conditions=results,aggregate=aggregate)


def run_arm(args,plan):
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:1');torch.cuda.set_device(device)
    free,total=torch.cuda.mem_get_info(device)
    if free<1.25*1024**3:
        raise RuntimeError('pending_resources: paired arm requires at least1.25GiB currently free')
    torch.cuda.set_per_process_memory_fraction(.035,device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('paired dynamics compute requires GPU BF16 support')
    config,topology,loaded,sources=open_sources(args.run_dir)
    if sources['train'].manifest['identity']!=plan['train_manifest'] or sources['validation'].manifest['identity']!=plan['validation_manifest']:
        raise ValueError('frozen paired source manifests changed')
    torch.manual_seed(plan['initialization_seed'])
    adjacency=torch.tensor(topology.network.adjacency,dtype=torch.float32)
    static=torch.tensor(topology.network.static_features,dtype=torch.float32)
    codec=FlowProjector(adjacency=adjacency,positive_speed=config.codec.speed_positive).to(device)
    world=loaded.training_state['models']['world_model'];prefix='flow_projector.'
    codec.load_state_dict({key[len(prefix):]:value for key,value in world.items() if key.startswith(prefix)},strict=True)
    codec.requires_grad_(False);codec.eval()
    sigma=float(args.arm)
    dynamics=Dynamics(adjacency=adjacency,static_features=static,low_frequency_rank=config.method.low_frequency_rank,
                       prior_noise_scale=sigma,event_protocol=True).to(device)
    # The fixed spectral coordinates are shared, never a learned dynamics warm start.
    dynamics.graph_prior.basis.copy_(world['dynamics.graph_prior.basis'].to(device))
    dynamics.set_pretraining_execution_profile(space_chunk=64,time_chunk=65536,mlp_chunk=65536,activation_checkpointing=True)
    dynamics.train();codec_hash=module_hash(codec);initial_hash=module_hash(dynamics)
    optimizer=torch.optim.AdamW(dynamics.parameters(),lr=1e-4,weight_decay=1e-2)
    arm_dir=args.output/f'arm_sigma{sigma:g}';arm_dir.mkdir(parents=True,exist_ok=True)
    record=dict(status='training',training_sigma=sigma,codec_hash=codec_hash,initial_dynamics_hash=initial_hash,
        graph_basis_hash=hashlib.sha256(dynamics.graph_prior.basis.cpu().numpy().tobytes()).hexdigest(),
        plan_identity=plan['identity'],source_checkpoint=str(loaded.path),source_schema=10,
        initialization='fresh Dynamics; only fixed graphbasis and frozencodec from complete model',
        optimizer=dict(name='AdamW',lr=1e-4,weight_decay=1e-2,gradient_clip=10),
        precision=dict(codec='fp32',dynamics_training='bf16',loss='fp32',evaluation='fp32'),
        actor_critic_instances=0,mainline_checkpoint_written=False)
    (arm_dir/'progress.json').write_text(json.dumps(record,indent=2)+'\n')
    rng_digest=hashlib.sha256();started=time.perf_counter();losses=[]
    for step,row in enumerate(plan['schedule']):
        batch=sources['train'].batch_from_starts(row['episode_index'],row['starts'],fm_only=True)
        scoped=prepare_transition_targets(batch,capacities=topology.network.capacities)
        with torch.no_grad():
            inputs=codec(batch.flow[:,:3].to(device).reshape(-1,topology.network.edge_count,8)).reshape(4,3,-1,64)
            targets=codec(scoped.transition_target_flow.to(device).reshape(-1,topology.network.edge_count,8)).reshape_as(inputs)
        tau,noise=paired_rng(plan['noise_seed'],step,4,3,dynamics.graph_prior.rank)
        rng_digest.update(tau.numpy().tobytes());rng_digest.update(noise.numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,metrics,_=independent_transition_fm_loss(dynamics,z_inputs=inputs,z_targets=targets,
                action=batch.action.to(device),action_events=batch.action_events.to(device),
                flow_time=tau.to(device),noise=noise.to(device),prior_weight=config.method.prior_loss_weight,
                predict_endpoints=False)
        if not torch.isfinite(loss):
            raise ValueError('paired arm produced nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(dynamics.parameters(),10.)
        if not torch.isfinite(norm):
            raise ValueError('paired arm produced nonfinite gradients')
        optimizer.step()
        losses.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm),
            **{name:float(value) for name,value in metrics.items()}))
        if (step+1)%25==0:
            record.update(updates=step+1,last_loss=losses[-1],elapsed_seconds=time.perf_counter()-started)
            (arm_dir/'progress.json').write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps(dict(arm=sigma,updates=step+1,loss=losses[-1]['loss'])),flush=True)
    train_seconds=time.perf_counter()-started
    (arm_dir/'losses.json').write_text(json.dumps(losses,indent=2)+'\n')
    torch.save(dict(ablation_schema=1,not_mainline_resume=True,dynamics=dynamics.state_dict(),
        codec_hash=codec_hash,plan_identity=plan['identity'],training_sigma=sigma,
        optimizer=optimizer.state_dict(),updates=len(losses)),arm_dir/'dynamics_ablation.pt')
    del optimizer,inputs,targets,loss
    dynamics.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    evaluation=evaluate(codec,dynamics,sources['validation'],plan['heldout_conditions'],config,topology.network,
                        device,plan['evaluation_seed'])
    if module_hash(codec)!=codec_hash:
        raise ValueError('shared frozen codec changed during paired training')
    record.update(status='completed',updates=len(losses),training_seconds=train_seconds,
        total_seconds=time.perf_counter()-started,rng_tau_and_coefficients_sha256=rng_digest.hexdigest(),
        final_dynamics_hash=module_hash(dynamics),codec_unchanged=True,evaluation=evaluation,
        peak_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2,
        peak_reserved_mb=torch.cuda.max_memory_reserved(device)/1024**2)
    (arm_dir/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'outputs/ingolstadt_bf16_pretrain_20261002')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/ingolstadt_noise_training_pair_20261003')
    parser.add_argument('--updates',type=int,default=500)
    parser.add_argument('--arm',choices=('0','0.1'))
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if args.arm is not None:
        run_arm(args,json.loads((args.output/'plan.json').read_text()));return
    if args.updates!=500:
        raise ValueError('this declared matched ablation has a fixed500-update budget')
    config,topology,loaded,sources=open_sources(args.run_dir)
    heldout=json.loads((ROOT/'outputs/ingolstadt_noise_diversity_20261003/results.json').read_text())
    if heldout['validation_manifest']!=sources['validation'].manifest['identity']:
        raise ValueError('paired evaluation must reuse the same frozen held-out conditions')
    schedule=make_schedule(sources['train'].manifest['entries'],args.updates,4,3,20261004)
    train_groups={row['demand_group'] for row in schedule}
    conditions=[row['condition'] for row in heldout['conditions']]
    if train_groups&{row['demand_group'] for row in conditions}:
        raise ValueError('paired schedule includes held-out demand')
    plan=dict(schema=1,updates=500,batch=4,horizon=3,initialization_seed=24680,noise_seed=246810,
        evaluation_seed=20261003,train_manifest=sources['train'].manifest['identity'],
        validation_manifest=sources['validation'].manifest['identity'],source_checkpoint=str(loaded.path),
        source_config_fingerprint=loaded.config_fingerprint,source_input_hash=loaded.input_hash,
        codec_source_updates=6500,training_sigmas=[0,.1],evaluation_sigmas=[0,.1],
        config_snapshot=config.resolved_snapshot(),schedule=schedule,heldout_conditions=conditions,
        interpretation='Small500-update fresh-dynamics matched experiment, not the complete5-epoch production model.',
        spectral_noise=dict(formula='sigma U xi, U columns orthonormal',roads=topology.network.edge_count,
            rank=config.method.low_frequency_rank,theoretical_rms_initial_std=.1*(config.method.low_frequency_rank/topology.network.edge_count)**.5))
    plan['identity']=identity(plan);(args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for sigma in ('0','0.1'):
        arm_dir=args.output/f'arm_sigma{float(sigma):g}'
        result_path=arm_dir/'result.json'
        if result_path.exists():
            raise ValueError('ablation output already contains a result; use a separate output instead of silently overwriting')
        log=args.output/f'arm_sigma{float(sigma):g}.log'
        with log.open('w') as stream:
            completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run-dir',str(args.run_dir),
                '--output',str(args.output),'--arm',sigma],stdout=stream,stderr=subprocess.STDOUT)
        if completed.returncode:
            (args.output/'status.json').write_text(json.dumps(dict(status='pending_or_failed',arm=sigma,
                returncode=completed.returncode,log=str(log)),indent=2)+'\n');return
    arms=[json.loads((args.output/f'arm_sigma{sigma:g}'/'result.json').read_text()) for sigma in (0.,.1)]
    for name in ('codec_hash','initial_dynamics_hash','graph_basis_hash','plan_identity','rng_tau_and_coefficients_sha256'):
        if arms[0][name]!=arms[1][name]:
            raise ValueError(f'paired experiment differs in controlled factor {name}')
    report=dict(status='completed',controlled_factors_verified=True,arms=arms,plan_identity=plan['identity'],
        updates_per_arm=500,sequence_samples_per_arm=2000,window_samples_per_arm=6000,
        interpretation_limits=['Single paired training seed and only500 fresh-dynamics updates, not a completed-pretrain ablation.',
            'Same fixed test sigma0.1 isolates learned responses to perturbations; matched train/testsigma compares separate sampling priors.',
            'Identical frozen completed codec, spectral basis, initialized parameters, training data, tau, Gaussian coefficients and optimizer budget.',
            'Results do not establish statistically significant effects, multimodality or epistemic uncertainty.'])
    (args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cells=[dict(training_sigma=arm['training_sigma'],
        evaluation=arm['evaluation']['aggregate']) for arm in arms])),flush=True)


if __name__=='__main__':
    main()
