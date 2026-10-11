"""Frozen holdout inference comparison; Euler is a numerical reference, never ground truth."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.models.world_model import UrbanDreamerWorldModel
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config


def tensor_comparison(value, reference):
    value,reference=value.double(),reference.double()
    return dict(finite=bool(torch.isfinite(value).all()),
        relative_l2=float((value-reference).norm()/reference.norm().clamp_min(1e-15)),
        max_abs=float((value-reference).abs().max()))


def physical_errors(decoded,target_count,target_speed,capacities,limits):
    decoded=decoded.double()
    count=decoded[...,2].clamp_min(0)*capacities
    speed=decoded[...,1]*limits
    occupied=target_count>0
    slow=occupied&(target_speed<=2)&(target_count>=5)
    def masked(error,mask):
        return float(error[mask].mean()) if bool(mask.any()) else None
    return dict(count_mae=float((count-target_count).abs().mean()),
        occupied_count_mae=masked((count-target_count).abs(),occupied),
        speed_mae=float((speed-target_speed).abs().mean()),
        occupied_speed_mae=masked((speed-target_speed).abs(),occupied),
        slow_queue_speed_mae=masked((speed-target_speed).abs(),slow),
        slow_queue_road_count=int(slow.sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--cache-manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--memory-fraction',type=float,default=.20)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--noise-counts',type=int,nargs='+',default=[1,3])
    p.add_argument('--iterations',type=int,nargs='+',default=[2,4,6,8])
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--picard-batch-chunk-size',type=int)
    p.add_argument('--case-indices',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--accuracy-only',action='store_true')
    args=p.parse_args()
    if args.accuracy_only:args.repeats=0
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    source=out/'source_checkpoint.pt';cfg=out/'source_config.toml'
    if not source.exists():
        shutil.copy2(args.source_dir/'checkpoints/latest.pt',source)
        shutil.copy2(args.source_dir/'run_config.toml',cfg)
    result=dict(checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_checkpoint=str(source.resolve()),benchmark_only=True,production_modified=False,
        dtype='fp32',steps=100,device=args.device,batch=args.batch,picard_batch_chunk_size=args.picard_batch_chunk_size,
        noise_counts=args.noise_counts,rows=[],timings=[],failures=[],status='running')
    def save():
        tmp=out/'results.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out/'results.json')
    save()
    torch.set_num_threads(4)
    device=torch.device(args.device);torch.cuda.set_device(device)
    free,total=torch.cuda.mem_get_info(device)
    if free<args.memory_fraction*total+819.2*1024**2:
        raise RuntimeError('insufficient free GPU memory for bounded experiment')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
    config=load_config(cfg);prepared=validate_prepared_inputs(config);topology=load_traffic_topology(prepared)
    dataset=CompactPretrainDataset(config,prepared,topology,2,cache_manifest_path=args.cache_manifest,split='validation')
    capacity=torch.as_tensor(topology.network.capacities,dtype=torch.float64)
    limits=torch.as_tensor(topology.network.speed_limits_mps,dtype=torch.float64)
    candidates=[]
    for index,entry in enumerate(dataset.manifest['entries']):
        with np.load(entry['cache_path'],allow_pickle=False) as archive:
            flow=archive['flow']
        for window in range(1,len(flow)-2):
            counts=flow[window,:,2]*capacity.numpy();speed=flow[window,:,1]*limits.numpy()
            active=float(counts.sum());slow=float(counts[(counts>=5)&(speed<=2)].sum())
            if active>0:
                candidates.append((slow,active,index,window))
    ordered=sorted(candidates)
    selected=[ordered[0],ordered[len(ordered)//3],ordered[2*len(ordered)//3],ordered[-1]]
    result['windows']=[dict(slow_queue_vehicles=s,active_vehicles=a,episode_index=i,window=w,
        archive=dataset.manifest['entries'][i]['path']) for s,a,i,w in selected]
    model=UrbanDreamerWorldModel(adjacency=torch.as_tensor(topology.network.adjacency,dtype=torch.float32),
        static_features=torch.as_tensor(topology.network.static_features,dtype=torch.float32),event_protocol=True,
        low_frequency_rank=config.method.low_frequency_rank,prior_noise_scale=config.method.prior_noise_scale,
        codec_speed_positive=config.codec.speed_positive).to(device).eval().requires_grad_(False)
    payload=torch.load(source,map_location='cpu',weights_only=True)
    model.load_state_dict(payload['training_state']['models']['world_model']);del payload
    model.dynamics.set_pretraining_execution_profile(space_chunk=64,time_chunk=65536,mlp_chunk=8192,activation_checkpointing=False)
    chunk_support='picard_batch_chunk_size' in inspect.signature(model.dynamics.solve).parameters
    result['picard_chunk_supported']=chunk_support
    if args.picard_batch_chunk_size is not None and not chunk_support:
        raise RuntimeError('requested Picard batching is unsupported by current source')
    work={'calls':0,'velocity_batch_members':0}
    def hook(module,values,kwargs):
        work['calls']+=1;work['velocity_batch_members']+=int(kwargs['packed_latents'].shape[0])
    handle=model.dynamics.register_forward_pre_hook(hook,with_kwargs=True)
    with torch.no_grad():
        for case,(_,_,index,start) in enumerate(selected):
            if case not in args.case_indices:
                continue
            batch=dataset.batch_from_starts(index,[start]*args.batch,fm_only=True)
            scoped=prepare_transition_targets(batch,capacities=topology.network.capacities)
            z=model.encode(batch.flow[:,0].to(device));action=batch.action[:,0].to(device)
            events=batch.action_events.for_step(0).to(device)
            target_count=scoped.transition_target_flow[:,0,:,2].double()*capacity
            target_speed=.5*(batch.flow[:,1,:,1].double()+batch.flow[:,2,:,1].double())*limits
            for samples in args.noise_counts:
                noise=torch.randn(samples,args.batch,model.dynamics.graph_prior.rank,64,
                    generator=torch.Generator().manual_seed(412000+case)).to(device)
                def run(solver,iterations=None):
                    options=dict(steps=100,solver=solver,action_events=events)
                    if solver=='picard':
                        options['picard_iterations']=iterations
                        if args.picard_batch_chunk_size is not None:
                            options['picard_batch_chunk_size']=args.picard_batch_chunk_size
                    if samples==1:
                        latent=model.transition(z,action,noise=noise[0],**options).unsqueeze(0)
                        variance=torch.zeros_like(latent[0,...,0])
                    else:
                        latent,_,variance=model.latent_disagreement(z,action,noise=noise,n_samples=samples,**options)
                    return latent,variance
                reference,refvar=run('euler');reference_cpu=reference.cpu();refvar=refvar.cpu()
                refdecoded=model.decode(reference).mean(0).cpu();del reference
                methods=[('euler',None)]+[('picard',n) for n in args.iterations]
                valid=[]
                for solver,it in methods:
                    label=f'{solver}_{it}' if it else solver
                    try:
                        for _ in range(1 if args.accuracy_only else 2):
                            latent,variance=run(solver,it)
                        decoded=model.decode(latent).mean(0).cpu();latent=latent.cpu();variance=variance.cpu()
                        row=dict(case=case,samples=samples,method=label,
                            latent_vs_euler=tensor_comparison(latent,reference_cpu),
                            variance_vs_euler=tensor_comparison(variance,refvar),
                            mean_variance=float(variance.mean()),euler_mean_variance=float(refvar.mean()),
                            decoded_count_mae_vs_euler=float(((decoded[...,2]-refdecoded[...,2])*capacity).abs().mean()),
                            decoded_speed_mae_vs_euler=float(((decoded[...,1]-refdecoded[...,1])*limits).abs().mean()),
                            gt=physical_errors(decoded,target_count,target_speed,capacity,limits))
                        result['rows'].append(row);valid.append((solver,it,label));save()
                    except torch.OutOfMemoryError as exc:
                        result['failures'].append(dict(case=case,samples=samples,method=label,error=str(exc)));torch.cuda.empty_cache();save()
                timings={label:[] for _,_,label in valid};peaks={};workload={}
                for repeat in range(args.repeats):
                    for solver,it,label in valid[::1 if repeat%2==0 else -1]:
                        work.update(calls=0,velocity_batch_members=0)
                        torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);started=time.perf_counter()
                        latent,variance=run(solver,it);torch.cuda.synchronize(device)
                        timings[label].append(time.perf_counter()-started)
                        peaks[label]=max(peaks.get(label,0),torch.cuda.max_memory_allocated(device)/1024**2)
                        workload[label]=dict(work);del latent,variance
                for _,_,label in valid:
                    if not timings[label]:continue
                    result['timings'].append(dict(case=case,samples=samples,method=label,seconds=timings[label],
                        median_seconds=statistics.median(timings[label]),peak_allocated_mib=peaks[label],**workload[label]))
                save();print('Completed',case,samples,{k:statistics.median(v) for k,v in timings.items() if v},flush=True)
    handle.remove();result['status']='complete';save()

if __name__=='__main__':main()
