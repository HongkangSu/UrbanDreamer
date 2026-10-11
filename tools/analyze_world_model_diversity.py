"""Frozen held-out inference-noise sweep; this is not a noise-training ablation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.models.world_model import UrbanDreamerWorldModel, ConditionEncoding
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.checkpoint import load_checkpoint, CheckpointPhase
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.pipeline import combined_input_hash


def empirical_crps(samples,target):
    values=np.asarray(samples,dtype=np.float64)
    target=np.asarray(target,dtype=np.float64)
    if values.ndim!=2 or values.shape[1:]!=target.shape or not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError('CRPS requires finite (samples,roads) and matching target roads')
    return np.abs(values-target).mean(0)-.5*np.abs(values[:,None]-values[None,:]).mean((0,1))


def summarize_samples(starts,endpoints,decoded,target,speed_target,capacities,limits,cohort_count):
    starts,endpoints,decoded,target=[np.asarray(value,dtype=np.float64) for value in (starts,endpoints,decoded,target)]
    capacities,limits=np.asarray(capacities,dtype=np.float64),np.asarray(limits,dtype=np.float64)
    if not all(np.isfinite(value).all() for value in (starts,endpoints,decoded,target)):
        raise ValueError('nonfinite model samples or labels')
    count=decoded[...,2]*capacities
    speed=decoded[...,1]*limits
    truth=target[...,2]*capacities
    occupied=truth>0
    lower,upper=np.quantile(count,[.05,.95],axis=0)
    totals=count.sum(-1)
    crps=empirical_crps(count,truth)
    result=dict(draws=len(decoded),roads=len(truth),occupied_roads=int(occupied.sum()),finite=True,
        initial_latent_std=float(starts.std(0).mean()),endpoint_latent_std=float(endpoints.std(0).mean()),
        count_std_all=float(count.std(0).mean()),speed_std_all_mps=float(speed.std(0).mean()),
        ensemble_count_mae_all=float(np.abs(count.mean(0)-truth).mean()),
        single_sample_count_mae_all=float(np.abs(count-truth).mean()),
        interval90_coverage_all=float(((truth>=lower)&(truth<=upper)).mean()),
        interval90_width_all=float((upper-lower).mean()),crps_count_all=float(crps.mean()),
        true_total_count=float(truth.sum()),total_count_mean=float(totals.mean()),total_count_std=float(totals.std()),
        total_count_quantiles=np.quantile(totals,[0,.05,.5,.95,1]).tolist(),total_count_samples=totals.tolist(),
        total_count_ensemble_absolute_error=float(abs(totals.mean()-truth.sum())),
        total_count_single_sample_absolute_error=float(np.abs(totals-truth.sum()).mean()),
        previous_active_cohort_including_internal=int(cohort_count),
        total_count_over_previous_cohort_fraction=float((totals>cohort_count+1e-4).mean()),
        total_count_over_previous_cohort_plus_one_vehicle_fraction=float((totals>cohort_count+1.).mean()),
        max_total_count_over_previous_cohort=float(max(0.,totals.max()-cohort_count)),
        negative_count_fraction=float((count<0).mean()))
    result['latent_std_contraction_ratio']=(result['endpoint_latent_std']/result['initial_latent_std']
                                           if result['initial_latent_std']>0 else None)
    fields=dict(count_std_occupied=count.std(0),speed_std_occupied_mps=speed.std(0),
        ensemble_count_mae_occupied=np.abs(count.mean(0)-truth),single_sample_count_mae_occupied=np.abs(count-truth).mean(0),
        interval90_coverage_occupied=((truth>=lower)&(truth<=upper)).astype(float),
        interval90_width_occupied=upper-lower,crps_count_occupied=crps)
    if speed_target is not None:
        speed_truth=np.asarray(speed_target,dtype=np.float64)*limits
        fields['ensemble_speed_mae_occupied_mps']=np.abs(speed.mean(0)-speed_truth)
        fields['single_sample_speed_mae_occupied_mps']=np.abs(speed-speed_truth).mean(0)
        result['ensemble_speed_mae_all_mps']=float(np.abs(speed.mean(0)-speed_truth).mean())
        result['single_sample_speed_mae_all_mps']=float(np.abs(speed-speed_truth).mean())
    for name,values in fields.items():
        result[name]=float(values[occupied].mean()) if occupied.any() else None
    return result


def select_conditions(source,capacities,demand_duration):
    groups=set();selected=[]
    phase_pairs=(('early','peak'),('peak','drain'),('drain','early'))
    for index,entry in enumerate(source.manifest['entries']):
        if entry['demand_group'] in groups:
            continue
        arrays=source._episode(index)
        windows=len(arrays['flow'])-1
        if windows<8:
            continue
        totals=(arrays['flow'][...,2]*capacities).sum(-1)
        valid=np.arange(windows-2)
        valid=valid[totals[valid]>0]
        if not len(valid):
            continue
        early=int(valid[np.argmin(abs(arrays['time'][valid]-120.))])
        peak=int(valid[np.argmax(totals[valid])])
        # A censored episode can remain congested long after departures stop.
        # Require observed decline after the actual peak for the drain label.
        late=valid[(arrays['time'][valid]>=demand_duration)&(valid>peak)
                   &(totals[valid]<=.6*totals[peak])]
        if not len(late):
            continue
        drain=int(late[np.argmin(abs(totals[late]-max(10.,totals[peak]*.25)))])
        positions=dict(early=early,peak=peak,drain=drain)
        for phase in phase_pairs[len(groups)%3]:
            start=positions[phase]
            selected.append(dict(episode_index=index,phase=phase,start=start,
                decision_time=float(arrays['time'][start]),demand_group=entry['demand_group'],
                archive=entry['path'],method=entry['method'],episode=entry['episode'],
                input_road_count=float(totals[start]),peak_road_count=float(totals[peak]),
                peak_time=float(arrays['time'][peak])))
        groups.add(entry['demand_group'])
        if len(groups)==6:
            break
    if len(groups)!=6 or len(selected)!=12:
        raise ValueError('need six disjoint held-out demand groups and twelve early/peak/drain conditions')
    return selected


def parameter_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def select_stress_conditions(source,capacities,demand_duration,used_groups):
    selected=[];groups=set(used_groups)
    for method in ('online_ac','dreamer','mbpo','random'):
        for index,entry in enumerate(source.manifest['entries']):
            if entry['method']!=method or entry['demand_group'] in groups:
                continue
            arrays=source._episode(index)
            totals=(arrays['flow'][...,2]*capacities).sum(-1)
            valid=np.arange(max(0,len(arrays['flow'])-3))
            if not len(valid) or totals[valid].max()<=0:
                continue
            peak=int(valid[np.argmax(totals[valid])]);at=float(arrays['time'][peak])
            selected.append(dict(episode_index=index,phase='congested_late' if at>=demand_duration else 'peak',
                start=peak,decision_time=at,demand_group=entry['demand_group'],archive=entry['path'],
                method=method,episode=entry['episode'],input_road_count=float(totals[peak]),
                peak_road_count=float(totals[peak]),peak_time=at))
            groups.add(entry['demand_group']);break
        else:
            raise ValueError(f'no independent held-out peak condition available for {method}')
    return selected


def aggregate_rows(rows):
    aggregate=[]
    for index,sigma in enumerate((0.,.05,.1,.2)):
        items=[row['sigmas'][index] for row in rows]
        summary=dict(sigma=sigma,conditions=len(items))
        for name,value in items[0].items():
            if isinstance(value,(int,float)) and not isinstance(value,bool) and name!='sigma':
                candidates=[item[name] for item in items if item[name] is not None]
                summary[name]=float(np.mean(candidates)) if candidates else None
        occupied=sum(item['occupied_roads'] for item in items)
        summary['occupied_road_weighted_coverage90']=(float(sum(item['interval90_coverage_occupied']*item['occupied_roads']
            for item in items if item['occupied_roads'])/occupied) if occupied else None)
        aggregate.append(summary)
    return aggregate


def plot_results(output,rows,aggregate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    phases=('early','peak','drain');colors=('#2563eb','#059669','#d97706','#9333ea')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for axis,phase in zip(axes,phases):
        condition=next(row for row in rows if row['condition']['phase']==phase)
        distributions=condition['sigmas']
        all_values=[value for row in distributions for value in row['total_count_samples']]
        bins=np.linspace(min(all_values)-.5,max(all_values)+.5,16)
        for color,row in zip(colors,distributions):
            axis.hist(row['total_count_samples'],bins=bins,histtype='step',linewidth=1.8,
                      label=f"sigma={row['sigma']:g}",color=color)
        axis.axvline(distributions[0]['true_total_count'],color='black',ls='--',label='observed target')
        axis.set_title(f"{phase}, t={condition['condition']['decision_time']:g}s")
        axis.set_xlabel('Predicted road vehicles');axis.set_ylabel('Samples');axis.legend(fontsize=8)
    fig.suptitle('Frozen model: inference-noise sweep, 32 draws per fixed condition')
    fig.tight_layout();fig.savefig(output/'total_count_distributions.png',dpi=170);plt.close(fig)
    sigmas=[row['sigma'] for row in aggregate];fig,axes=plt.subplots(2,2,figsize=(12,8))
    def line(axis,key,label):
        axis.plot(sigmas,[row[key] for row in aggregate],marker='o',label=label)
    line(axes[0,0],'initial_latent_std','initial latent');line(axes[0,0],'endpoint_latent_std','endpoint latent')
    axes[0,0].set_ylabel('Mean latent standard deviation')
    line(axes[0,1],'count_std_occupied','occupied roads');line(axes[0,1],'count_std_all','all roads')
    axes[0,1].set_ylabel('Vehicles: mean standard deviation')
    line(axes[1,0],'ensemble_count_mae_occupied','ensemble mean MAE')
    line(axes[1,0],'single_sample_count_mae_occupied','mean sample MAE')
    line(axes[1,0],'crps_count_occupied','empirical CRPS');axes[1,0].set_ylabel('Vehicles: error/CRPS')
    line(axes[1,1],'occupied_road_weighted_coverage90','road-weighted 90% interval coverage')
    axes[1,1].axhline(.9,color='black',ls='--',label='nominal 90%')
    axes[1,1].set_ylabel('Occupied-road coverage');axes[1,1].set_ylim(0,1)
    right=axes[1,1].twinx();right.plot(sigmas,[row['interval90_width_occupied'] for row in aggregate],
                                    marker='s',color='gray',label='interval width')
    right.set_ylabel('Width in vehicles');right.legend(loc='lower right',fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel('Inference sigma');axis.legend(fontsize=8,
            loc='upper left' if axis is axes[1,1] else 'best');axis.grid(alpha=.2)
    fig.suptitle('Twelve held-out conditions; coverage is descriptive, not a calibrated-uncertainty proof')
    fig.tight_layout();fig.savefig(output/'sigma_diversity_accuracy.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'outputs/ingolstadt_bf16_pretrain_20261002')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/ingolstadt_noise_diversity_20261003')
    parser.add_argument('--device',choices=('cpu','cuda:1'),default='cuda:1')
    parser.add_argument('--samples',type=int,default=32)
    parser.add_argument('--sample-chunk',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261003)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.samples!=32 or args.sample_chunk<1 or args.samples%args.sample_chunk:
        raise ValueError('the declared diagnostic uses exactly32 draws and equal-sized sample chunks')
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.set_float32_matmul_precision('highest')
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
        if free<.035*total+1024**3:
            raise RuntimeError('insufficient GPU headroom for this isolated inference diagnostic')
        torch.cuda.set_per_process_memory_fraction(.035,device)
    config=load_config(args.run_dir/'run_config.toml')
    prepared=validate_prepared_inputs(config);topology=load_traffic_topology(prepared)
    loaded=load_checkpoint(output_dir=args.run_dir,expected_config_fingerprint=config.semantic_fingerprint,
                           expected_input_hash=combined_input_hash(prepared))
    metrics=loaded.training_state['metrics']
    if (loaded.schema_version!=10 or loaded.phase!=CheckpointPhase.ONLINE_TRAINING or loaded.next_unit!=0
            or metrics['world_model_initialization_epochs_completed']!=config.run.world_model_initialization_epochs
            or metrics['online_episodes_completed']!=0):
        raise ValueError('requires the strictly compatible completed-pretraining, pre-online schema10 checkpoint')
    model=UrbanDreamerWorldModel(adjacency=torch.tensor(topology.network.adjacency,dtype=torch.float32),
        static_features=torch.tensor(topology.network.static_features,dtype=torch.float32),
        low_frequency_rank=config.method.low_frequency_rank,prior_noise_scale=config.method.prior_noise_scale,
        event_protocol=True,codec_speed_positive=config.codec.speed_positive).to(device)
    model.load_state_dict(loaded.training_state['models']['world_model'],strict=True)
    model.requires_grad_(False);model.eval()
    model.dynamics.set_pretraining_execution_profile(space_chunk=args.sample_chunk,time_chunk=65536,
        mlp_chunk=16384,activation_checkpointing=False)
    training_sigma=model.dynamics.graph_prior.noise_scale
    if training_sigma!=.1 or any(value.dtype!=torch.float32 for value in model.parameters()):
        raise ValueError('this analysis requires trained sigma0.1 and frozen FP32 model parameters')
    source=CompactPretrainDataset(config,prepared,topology,3,
        cache_manifest_path=config.data.validation_cache_manifest,split='validation')
    capacities=np.maximum(topology.network.capacities,1.)
    limits=topology.network.speed_limits_mps
    conditions=select_conditions(source,capacities,config.simulation.demand_duration_seconds)
    edge_case=dict(conditions[0],phase='zero-start',start=0,decision_time=0.,
                   input_road_count=float((source._episode(conditions[0]['episode_index'])['flow'][0,:,2]*capacities).sum()))
    conditions.append(edge_case)
    conditions.extend(select_stress_conditions(source,capacities,config.simulation.demand_duration_seconds,
                                              {condition['demand_group'] for condition in conditions}))
    (args.output/'conditions.json').write_text(json.dumps(conditions,indent=2)+'\n')
    initial_hash=parameter_hash(model);rows=[];started=time.perf_counter()
    for condition_index,condition in enumerate(conditions):
        batch=source.batch_from_starts(condition['episode_index'],[condition['start']])
        scoped=prepare_transition_targets(batch,capacities=topology.network.capacities)
        target=scoped.transition_target_flow[0,0].numpy()
        speed_target=(batch.flow[0,1,:,1]+batch.flow[0,2,:,1]).numpy()/2.
        cohort_count=int(scoped.transition_cohort_count[0,0])
        if condition['phase']=='zero-start' and cohort_count!=0:
            raise ValueError('the declared zero-start edge case must have no existing active vehicles')
        with torch.no_grad():
            z=model.encode(batch.flow[:,0].to(device)).unsqueeze(1)
            action=batch.action[:,:1].to(device)
            background=model.dynamics.encode_condition(action,action_events=batch.action_events.for_step(0).to(device))
            increment=model.dynamics.graph_prior(z,background)
            coarse=z+increment
            cached=ConditionEncoding(background.edge_tokens,model.dynamics.reference_proj(coarse))
            generator=torch.Generator(device='cpu').manual_seed(args.seed+condition_index)
            coefficients=torch.randn(args.samples,1,model.dynamics.graph_prior.rank,64,generator=generator)
            sigma_rows=[]
            for sigma in (0.,.05,.1,.2):
                model.dynamics.graph_prior.noise_scale=sigma
                starts=[];ends=[];decoded=[]
                for offset in range(0,args.samples,args.sample_chunk):
                    count=args.sample_chunk
                    current=z.repeat(count,1,1,1)
                    start=coarse.repeat(count,1,1,1)+model.dynamics.graph_prior.sample_noise(
                        current,coefficients[offset:offset+count].to(device))
                    context=ConditionEncoding(cached.edge_tokens.repeat(count,1,1,1),
                                              cached.reference_tokens.repeat(count,1,1,1))
                    endpoint=model.dynamics.solve(start,action.repeat(count,1,1),context,
                        steps=config.method.fm_steps,solver=config.method.fm_solver,
                        picard_iterations=config.method.picard_iterations)
                    starts.append(start[:,0].cpu().numpy());ends.append(endpoint[:,0].cpu().numpy())
                    decoded.append(model.decode(endpoint[:,0]).cpu().numpy())
                    if sigma==.1 and offset==0:
                        repeated=model.dynamics.solve(start.clone(),action.repeat(count,1,1),context,
                            steps=config.method.fm_steps,solver=config.method.fm_solver,
                            picard_iterations=config.method.picard_iterations)
                        if not torch.equal(repeated,endpoint):
                            raise ValueError('same noise and parameters did not reproduce the endpoint exactly')
                starts,ends,decoded=[np.concatenate(value) for value in (starts,ends,decoded)]
                if sigma==0 and not np.array_equal(ends,np.repeat(ends[:1],args.samples,axis=0)):
                    raise ValueError('zero-noise predictions must be exactly identical for all32 draws')
                result=summarize_samples(starts,ends,decoded,target,speed_target,capacities,limits,cohort_count)
                result.update(sigma=sigma,zero_noise_exactly_deterministic=(sigma==0),
                              same_noise_repeat_exact=True if sigma==.1 else None)
                sigma_rows.append(result)
        rows.append(dict(condition=condition,sigmas=sigma_rows))
        (args.output/'progress.json').write_text(json.dumps(dict(completed_conditions=len(rows),total_conditions=len(conditions),
            elapsed_seconds=time.perf_counter()-started,last_condition=condition),indent=2)+'\n')
        print(json.dumps(dict(completed=len(rows),phase=condition['phase'],
            training_sigma_stats=sigma_rows[2])),flush=True)
    model.dynamics.graph_prior.noise_scale=training_sigma
    if parameter_hash(model)!=initial_hash:
        raise ValueError('frozen model weights changed during inference-only analysis')
    aggregate=aggregate_rows(rows[:12])
    output=dict(checkpoint=str(loaded.path),checkpoint_schema=loaded.schema_version,
        config_fingerprint=loaded.config_fingerprint,input_hash=loaded.input_hash,
        validation_manifest=source.manifest['identity'],trained_sigma=training_sigma,
        inference_sigmas=[0,.05,.1,.2],samples_per_condition=32,seed=args.seed,
        noise_coefficients='common random numbers across inference sigmas; seed+condition_index',
        conditions=rows[:12],zero_cohort_edge_case=rows[12],multimethod_stress_conditions=rows[13:],
        aggregate=aggregate,multimethod_stress_aggregate=aggregate_rows(rows[13:]),
        parameter_sha256=initial_hash,parameter_unchanged=True,
        optimizer_steps=0,inference_precision='fp32',codec_precision='fp32',
        spectral_noise_scaling=dict(noise_formula='sigma U xi with orthonormal E-by-r basis',
            roads_E=topology.network.edge_count,rank_r=model.dynamics.graph_prior.rank,
            average_coordinate_variance=training_sigma**2*model.dynamics.graph_prior.rank/topology.network.edge_count,
            theoretical_initial_rms_std=training_sigma*(model.dynamics.graph_prior.rank/topology.network.edge_count)**.5,
            interpretation='Sigma scales spectralcoefficients; it is not10%noise perroad. MeanSTD differs from RMSSTD and finite32draws.'),
        scope='previous_state_excluding_current_births',label_speed='endpoint-forward mean using t+30 and t+60 full background',
        interpretation_limits=['Inference sigma sweep on a single noise-trained checkpoint; not a matched noise-training ablation.',
            'Same-model noise variability is conditional stochastic diversity, not a parameter ensemble or an epistemic uncertainty estimate.',
            'Balanced selection requires post-peak observed road count <=60% of peak for a true drain condition.',
            'The eligible balanced selection contains six GNAC archive episodes; it does not establish coverage across every collection method.',
            'The separate four-method peak/congested-late stress set is not mixed into the balanced twelve-condition averages.',
            'The zero-active-vehicle boundary is excluded from balanced aggregates; positive softplus residuals do not denote actual new cars.',
            'One observed endpoint per condition; empirical intervals and CRPS are descriptive diagnostics, not calibrated uncertainty proof.',
            'Initial active cohort includes junction/internal vehicles; decoded road mass must not exceed that upper bound.',
            'Models are frozen; no actor updates, online data, optimizer steps or training checkpoint writes.'],
        elapsed_seconds=time.perf_counter()-started)
    if device.type=='cuda':
        output.update(peak_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2,
                      peak_reserved_mb=torch.cuda.max_memory_reserved(device)/1024**2)
    (args.output/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    plot_results(args.output,rows,aggregate)
    print(json.dumps(dict(aggregate=aggregate,elapsed_seconds=output['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()
