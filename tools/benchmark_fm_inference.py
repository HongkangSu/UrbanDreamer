"""Private Euler100 inference speed/quality proof with fixed weights, controls and noise."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.models.world_model import UrbanDreamerWorldModel
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config
from tools.benchmark_fm_picard import tensor_comparison

PROFILES={'baseline':(16,32768,16384,'fp32',False),
          'wide':(64,65536,65536,'fp32',False),
          'unchunked':(0,0,0,'fp32',False),
          'bf16':(0,0,0,'bf16',False),
          'compiled':(0,0,0,'fp32',True),
          'tf32':(0,0,0,'tf32',False),
          'tf32_compiled':(0,0,0,'tf32',True),
          'compiled_baseline':(16,32768,16384,'fp32',True),
          'cudagraph':(16,32768,16384,'fp32',True)}


def quality(candidate,reference,capacity,limits):
    if not candidate or len(candidate)!=len(reference):
        raise ValueError('candidate and reference must cover the same nonempty physical rollout')
    rows=[]
    for step,(observed,expected) in enumerate(zip(candidate,reference)):
        rows.append(dict(step=step+1,latent=tensor_comparison(observed['latent'],expected['latent']),
            variance=tensor_comparison(observed['variance'],expected['variance']),
            variance_ratio=float(observed['variance'].double().mean()/expected['variance'].double().mean().clamp_min(1e-20)),
            count_mae_vs_reference=float(((observed['decoded'][...,2]-expected['decoded'][...,2])*capacity).abs().mean()),
            speed_mae_vs_reference=float(((observed['decoded'][...,1]-expected['decoded'][...,1])*limits).abs().mean()),
            readout=tensor_comparison(observed['readout'],expected['readout'])))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--cache-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--memory-fraction',type=float,default=.10)
    p.add_argument('--profiles',nargs='+',choices=list(PROFILES),default=list(PROFILES))
    p.add_argument('--batches',nargs='+',type=int,default=[1,8])
    p.add_argument('--cases',nargs='+',type=int,default=[0,2,3])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmups',type=int,default=1)
    p.add_argument('--timing-only',action='store_true')
    p.add_argument('--trainable-parameters',action='store_true')
    p.add_argument('--rollout-steps',type=int,default=8)
    p.add_argument('--production-helper',action='store_true')
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(args.output.parent.resolve()/'inductor_cache'))
    result=dict(status='running',steps=100,samples=3,dtype_accumulation='fp32',production_modified=False,
        production_helper=args.production_helper,memory_fraction=args.memory_fraction,
        checkpoint_sha256=hashlib.sha256((args.source/'source_checkpoint.pt').read_bytes()).hexdigest(),
        timings=[],quality=[],cold=[],failures=[])
    def save():
        temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(args.output)
    torch.set_num_threads(4);device=torch.device(args.device);torch.cuda.set_device(device)
    free,total=torch.cuda.mem_get_info(device)
    if free<args.memory_fraction*total+512*1024**2:raise RuntimeError('Insufficient free memory for isolated inference benchmark')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
    config=load_config(args.source/'source_config.toml');prepared=validate_prepared_inputs(config);topology=load_traffic_topology(prepared)
    dataset=CompactPretrainDataset(config,prepared,topology,args.rollout_steps,cache_manifest_path=args.cache_manifest,split='validation')
    source_result=json.loads((args.source/'results.json').read_text())
    windows=source_result['windows'];result['windows']=windows
    model=UrbanDreamerWorldModel(adjacency=torch.as_tensor(topology.network.adjacency,dtype=torch.float32),
        static_features=torch.as_tensor(topology.network.static_features,dtype=torch.float32),event_protocol=True,
        low_frequency_rank=config.method.low_frequency_rank,
        prior_filter=getattr(config.method, 'prior_filter', 'hard'),
        prior_filter_strength=getattr(config.method, 'prior_filter_strength', 10.0),
        prior_action_conditioned=getattr(config.method, 'prior_action_conditioned', True),
        prior_filter_input=getattr(config.method, 'prior_filter_input', False),prior_noise_scale=config.method.prior_noise_scale,
        prior_usage=getattr(config.method, 'prior_usage', 'source_reference'),
        codec_speed_positive=config.codec.speed_positive).to(device).eval().requires_grad_(False)
    payload=torch.load(args.source/'source_checkpoint.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(payload['training_state']['models']['world_model']);del payload
    if args.trainable_parameters:model.dynamics.requires_grad_(True)
    result['dynamics_parameters_require_grad']=all(p.requires_grad for p in model.dynamics.parameters())
    capacity=torch.as_tensor(topology.network.capacities);limits=torch.as_tensor(topology.network.speed_limits_mps)
    originals=[block.forward for block in model.dynamics.transformer.layers]
    compiled={};original_forward=model.dynamics.forward
    trainer=SimpleNamespace(config=config,world_model=model,device=device)
    def set_profile(name):
        space,temporal,mlp,precision,compile_blocks=PROFILES[name]
        compile_mode='reduce-overhead' if name=='cudagraph' else 'default'
        if args.production_helper:
            from urban_dreamer.learning.fm_inference import configure_imagination_inference
            trainer.config=replace(config,resources=replace(config.resources,
                world_model_imagination_space_chunk=space,world_model_imagination_time_chunk=temporal,
                world_model_imagination_mlp_chunk=mlp,world_model_imagination_precision=precision,
                world_model_imagination_compile=compile_blocks,world_model_imagination_compile_mode=compile_mode))
            configure_imagination_inference(trainer)
            return
        model.dynamics._set_chunk_profile(space_chunk=space,time_chunk=temporal,mlp_chunk=mlp)
        for i,block in enumerate(model.dynamics.transformer.layers):
            key=(i,compile_mode)
            if compile_blocks and key not in compiled:
                compiled[key]=torch.compile(originals[i],backend='inductor',mode=compile_mode,dynamic=False,fullgraph=True)
            if compile_blocks:
                eager,fast=originals[i],compiled[key]
                def dispatch(x,eager=eager,fast=fast):
                    return fast(x) if x.shape[0]==24 and not torch.is_grad_enabled() else eager(x)
                block.forward=dispatch
            else:block.forward=originals[i]
        if precision in ('bf16','tf32'):
            def forward(*values,**kwargs):
                previous=torch.backends.cuda.matmul.allow_tf32
                try:
                    if precision=='tf32':torch.backends.cuda.matmul.allow_tf32=True
                    context=torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
                    with context:output=original_forward(*values,**kwargs)
                finally:
                    torch.backends.cuda.matmul.allow_tf32=previous
                return type(output)(latent_velocity=output.latent_velocity.float())
            model.dynamics.forward=forward
        else:model.dynamics.forward=original_forward
    def make_case(case,batch_size):
        window=windows[case];index=window['episode_index']
        starts=min(window['window'],dataset.manifest['entries'][index]['windows']-args.rollout_steps)
        batch=dataset.batch_from_starts(index,[max(starts,0)]*batch_size,fm_only=True)
        z=model.encode(batch.flow[:,0].to(device))
        controls=[(batch.action[:,i].to(device),batch.action_events.for_step(i).to(device)) for i in range(args.rollout_steps)]
        noise=torch.randn(args.rollout_steps,3,batch_size,model.dynamics.graph_prior.rank,64,
            generator=torch.Generator().manual_seed(99100+case)).to(device)
        return z,controls,noise
    def execute(data,rollout=False):
        z,controls,noise=data;records=[]
        for step in range(args.rollout_steps if rollout else 1):
            action,events=controls[step]
            samples,_,variance=model.latent_disagreement(z,action,n_samples=3,noise=noise[step],steps=100,solver='euler',action_events=events)
            if rollout:
                decoded=model.decode(samples)
                readout=model.active_vehicle_fraction_readout(samples,z_prev=z.unsqueeze(0).expand_as(samples),
                    raw_decoder_fraction=decoded[...,2].mean(-1),transition_seconds=30.)
                records.append(dict(latent=samples.cpu(),variance=variance.cpu(),decoded=decoded.mean(0).cpu(),readout=readout.cpu()))
            z=samples[0]
        return records if rollout else samples
    def tf32_state():
        backend=torch.backends.cuda.matmul
        return backend.fp32_precision if hasattr(backend,'fp32_precision') else backend.allow_tf32
    original_tf32_flag=tf32_state()
    result['initial_tf32_flag']=original_tf32_flag
    with torch.no_grad():
        for batch_size in args.batches:
            data=make_case(3,batch_size);valid=[]
            for name in args.profiles:
                try:
                    set_profile(name);torch.cuda.synchronize();start=time.perf_counter();execute(data);torch.cuda.synchronize()
                    result['cold'].append(dict(profile=name,batch=batch_size,seconds=time.perf_counter()-start))
                    for _ in range(max(args.warmups-1,0)):execute(data)
                    torch.cuda.synchronize();valid.append(name);save()
                except Exception as exc:
                    result['failures'].append(dict(profile=name,batch=batch_size,error=str(exc)));torch.cuda.empty_cache();save()
            measurements={name:[] for name in valid};peaks={}
            for repeat in range(args.repeats):
                for name in valid[::1 if repeat%2==0 else -1]:
                    set_profile(name);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                    samples=execute(data);torch.cuda.synchronize();measurements[name].append(time.perf_counter()-start)
                    peaks[name]=max(peaks.get(name,0),torch.cuda.max_memory_allocated()/1024**2);del samples
                print('Pair',batch_size,repeat+1,{name:values[-1] for name,values in measurements.items()},flush=True)
            for name in valid:
                result['timings'].append(dict(batch=batch_size,profile=name,seconds=measurements[name],median_seconds=statistics.median(measurements[name]),peak_allocated_mib=peaks[name]))
            save();print('Timing',batch_size,{n:statistics.median(v) for n,v in measurements.items()},flush=True)
        if args.timing_only:
            result['tf32_flag_restored']=tf32_state()==original_tf32_flag
            result['status']='complete';save();return
        # Timing all chunk options is cheap; quality uses baseline, best FP32,
        # BF16 and compiled paths, with identical recorded controls and noise.
        b8=[r for r in result['timings'] if r['batch']==max(args.batches) and r['profile'] in ('baseline','wide','unchunked')]
        best=min(b8,key=lambda r:r['median_seconds'])['profile']
        result['best_eager_fp32']=best
        quality_profiles=list(dict.fromkeys([best]+[p for p in ('bf16','compiled','tf32','tf32_compiled','compiled_baseline','cudagraph') if p in args.profiles]))
        for case in args.cases:
            data=make_case(case,1);set_profile('baseline');reference=execute(data,True)
            for name in quality_profiles:
                set_profile(name);observed=reference if name=='baseline' else execute(data,True)
                result['quality'].append(dict(case=case,batch=1,profile=name,steps=quality(observed,reference,capacity,limits)));save()
                print('Quality',case,name,flush=True)
        if max(args.batches)>1:
            data=make_case(3,max(args.batches));set_profile('baseline');reference=execute(data,True)
            for name in quality_profiles:
                set_profile(name);observed=reference if name=='baseline' else execute(data,True)
                result['quality'].append(dict(case=3,batch=max(args.batches),profile=name,steps=quality(observed,reference,capacity,limits)));save()
                print('Quality full batch',name,flush=True)
    result['tf32_flag_restored']=tf32_state()==original_tf32_flag
    result['status']='complete';save()

if __name__=='__main__':main()
