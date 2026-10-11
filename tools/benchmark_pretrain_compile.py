"""Private paired end-to-end FM compilation benchmark; never creates training checkpoints."""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import DataConfig, load_config
from tools.train_codec_ready import SparseGraphCodec


def similarity(reference, observed):
    reference = torch.cat([x.double().flatten() for x in reference])
    observed = torch.cat([x.double().flatten() for x in observed])
    return dict(finite=bool(torch.isfinite(reference).all() and torch.isfinite(observed).all()),
        cosine=float(torch.nn.functional.cosine_similarity(reference, observed, dim=0)),
        relative_l2=float((reference-observed).norm()/reference.norm().clamp_min(1e-20)),
        max_abs=float((reference-observed).abs().max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--cache-manifest',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--repeats',type=int,default=8)
    parser.add_argument('--memory-fraction',type=float,default=.18)
    parser.add_argument('--scope',choices=['block','mlp'],default='block')
    parser.add_argument('--tail-proof', action='store_true')
    parser.add_argument('--tail-batches', type=int, nargs='+', default=[1,7,23,31])
    parser.add_argument('--phase-profile', action='store_true')
    parser.add_argument('--production-static', action='store_true')
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(args.output.parent.resolve()/'inductor_cache'))
    result=dict(benchmark_only=True,codec_readiness_bypass_private=True,production_modified=False,
        batch=args.batch,depth=3,scope=args.scope,device=args.device,precision='bf16',dynamic=True,
        activation_checkpointing=True,optimizer_steps_included=True)
    def save():
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        device=torch.device(args.device)
        torch.cuda.set_device(device)
        free,total=torch.cuda.mem_get_info(device)
        if free < args.memory_fraction*total + 819.2*1024**2:
            raise RuntimeError('insufficient free memory for bounded benchmark')
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
        config=load_config(args.config)
        if args.production_static:
            config=replace(config,method=replace(config.method,fm_training_mode='single_step',fm_steps=100,fm_solver='euler'),
                resources=replace(config.resources,world_model_initialization_sequence_length=1,
                    world_model_initialization_batch_size=args.batch,world_model_pretrain_compile_blocks=True))
            result.update(depth=1,dynamic=False,compile_surface='production compile_pretrain_blocks',
                fm_training_mode='single_step',fm_solver='euler',fm_steps=100,
                solver_executed_during_observed_endpoint_training=False)
        config=replace(config,run=replace(config.run,device=args.device,output_dir=args.output.parent/'private_unused'),
            data=DataConfig(pretrain_cache_manifest=args.cache_manifest,validation_cache_manifest=args.cache_manifest),
            resources=replace(config.resources,world_model_pretrain_precision='bf16',
                world_model_pretrain_space_chunk=64,world_model_pretrain_time_chunk=65536,
                world_model_pretrain_mlp_chunk=65536,world_model_pretrain_activation_checkpointing=True))
        prepared=validate_prepared_inputs(config)
        trainer=MainlineTrainer(config=config,prepared_inputs=prepared,topology=load_traffic_topology(prepared))
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        trainer.world_model.load_state_dict(payload['training_state']['models']['world_model'],strict=True)
        del payload
        trainer.validate_codec_ready=lambda:None
        trainer.world_model.flow_projector.requires_grad_(False)
        trainer.world_model.flow_projector.__class__=SparseGraphCodec
        trainer.world_model.flow_projector.register_buffer('sparse_adjacency',
            trainer._tensor(trainer.network.adjacency).to_sparse().coalesce(),persistent=False)
        trainer.metrics['world_model_updates']=200
        trainer.offline_data.sequence_length=1 if args.production_static else 3
        batch=trainer.offline_data.batch_from_starts(0,list(range(args.batch)),fm_only=True)
        keys=list(trainer.world_model.state_dict())
        ids={n:id(p) for n,p in trainer.world_model.named_parameters()}
        opts={n:trainer.optimizers[n] for n in ('world_model_initial','active_fraction')}
        # Warm Adam first, then freeze a steady-state reference. Restores and CPU
        # diagnostics are excluded from full forward/backward/optimizer timing.
        trainer._update_world(trainer.offline_data,initial=True,batches=[batch])
        reference_weights=copy.deepcopy(trainer.world_model.state_dict())
        reference_opts={n:copy.deepcopy(o.state_dict()) for n,o in opts.items()}
        result['transformer_shape']=[args.batch*(1 if args.production_static else 3),1,trainer.edge_count+4,256]
        modules=[layer if args.scope=='block' else layer.mlp for layer in trainer.world_model.dynamics.transformer.layers]
        if args.production_static:
            from tools.run_complete_pretrain import compile_pretrain_blocks
            eager_forwards=[module.forward for module in modules]
            result['compiled_blocks']=compile_pretrain_blocks(trainer)
            compiled_calls=[module.forward for module in modules]
        else:
            for module in modules:
                module.compile(backend='inductor',mode='default',fullgraph=True,dynamic=True)
            compiled_calls=[module._compiled_call_impl for module in modules]
        result['parameter_ids_preserved']=ids=={n:id(p) for n,p in trainer.world_model.named_parameters()}
        result['state_dict_keys_preserved']=keys==list(trainer.world_model.state_dict())
        result['optimizer_ids_preserved']=all(id(p) in set(ids.values()) for o in opts.values() for group in o.param_groups for p in group['params'])
        def run(compiled,capture=False):
            for index,(module,fn) in enumerate(zip(modules,compiled_calls)):
                if args.production_static:
                    module.forward=fn if compiled else eager_forwards[index]
                else:
                    module._compiled_call_impl=fn if compiled else None
            trainer.world_model.load_state_dict(reference_weights)
            for n,o in opts.items():
                o.load_state_dict(copy.deepcopy(reference_opts[n]))
            torch.manual_seed(20261008)
            torch.cuda.manual_seed_all(20261008)
            trainer.metrics['world_model_updates']=200
            inputs=[]
            hook=None
            if capture:
                def capture_input(module,values):
                    if values[0].requires_grad:
                        values[0].retain_grad()
                        inputs.append(values[0])
                hook=trainer.world_model.dynamics.transformer.register_forward_pre_hook(capture_input)
            torch.cuda.synchronize(device)
            start=time.perf_counter()
            metrics=trainer._update_world(trainer.offline_data,initial=True,batches=[batch])
            torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-start
            grads={}
            input_grads=[]
            if hook is not None:
                hook.remove()
                grads={n:p.grad.detach().cpu().clone() for n,p in trainer.world_model.named_parameters() if p.grad is not None}
                input_grads=[x.grad.detach().cpu().clone() for x in inputs if x.grad is not None]
            return dict(seconds=elapsed,metrics=metrics,grads=grads,input_grads=input_grads)
        eager=run(False,True)
        result.update(status='compiling',eager_first_step_seconds=eager['seconds']);save()
        print('Starting cold compilation',flush=True)
        compiled=run(True,True)
        result['cold_compile_step_seconds']=compiled['seconds']
        result['gradient_names_preserved']=list(eager['grads'])==list(compiled['grads'])
        result['gradient_similarity']=similarity(list(eager['grads'].values()),list(compiled['grads'].values()))
        result['input_gradient_similarity']=similarity(eager['input_grads'][:1],compiled['input_grads'][:1])
        result['losses']={name:dict(eager=eager['metrics'][name],compiled=compiled['metrics'][name],
            relative_error=abs(eager['metrics'][name]-compiled['metrics'][name])/max(abs(eager['metrics'][name]),1e-20))
            for name in ('world_model_loss','active_fraction_loss')}
        result.update(status='warming');save()
        run(True);run(False)
        measurements={'eager':[],'compiled':[]}
        torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(args.repeats):
            order=(False,True) if iteration%2==0 else (True,False)
            for mode in order:
                label='compiled' if mode else 'eager'
                measurements[label].append(run(mode)['seconds'])
            print('Pair',iteration+1,measurements['eager'][-1],measurements['compiled'][-1],flush=True)
        result['measurements']=measurements
        result['median_seconds']={n:statistics.median(v) for n,v in measurements.items()}
        result['speedup']=result['median_seconds']['eager']/result['median_seconds']['compiled']
        if args.phase_profile:
            phases={}
            def timed(name, fn):
                def wrapped(*values, **kwargs):
                    torch.cuda.synchronize(device)
                    started=time.perf_counter()
                    value=fn(*values, **kwargs)
                    torch.cuda.synchronize(device)
                    phases[name]=phases.get(name,0.)+time.perf_counter()-started
                    return value
                return wrapped
            original_loss=trainer._world_loss
            original_backward=torch.Tensor.backward
            original_clip=torch.nn.utils.clip_grad_norm_
            original_steps={name:opt.step for name,opt in opts.items()}
            trainer._world_loss=timed('forward_loss',original_loss)
            torch.Tensor.backward=timed('backward',original_backward)
            torch.nn.utils.clip_grad_norm_=timed('gradient_clip',original_clip)
            for name,opt in opts.items():
                opt.step=timed('adam',original_steps[name])
            timings={'eager':[],'compiled':[]}
            try:
                for iteration in range(3):
                    for mode in ((False,True) if iteration%2==0 else (True,False)):
                        phases={}
                        observation=run(mode)
                        phases['whole_step']=observation['seconds']
                        phases['other']=observation['seconds']-sum(v for k,v in phases.items() if k!='whole_step')
                        timings['compiled' if mode else 'eager'].append(dict(phases))
            finally:
                trainer._world_loss=original_loss
                torch.Tensor.backward=original_backward
                torch.nn.utils.clip_grad_norm_=original_clip
                for name,opt in opts.items():
                    opt.step=original_steps[name]
            result['phase_profile']=dict(measurements=timings,
                median_seconds={mode:{key:statistics.median(x[key] for x in rows) for key in rows[0]}
                    for mode,rows in timings.items()},
                note='Synchronization at each phase changes overlap; interpret as instrumented breakdown, not primary throughput.')
        if args.tail_proof:
            from torch._dynamo.utils import counters
            tail_results=[]
            for size in args.tail_batches:
                batch=trainer.offline_data.batch_from_starts(0,list(range(size)),fm_only=True)
                before=int(counters['stats']['unique_graphs'])
                reference=run(False,True)
                observed=run(True,True)
                compiled_graphs=int(counters['stats']['unique_graphs'])
                row=dict(batch=size,first_compiled_seconds=observed['seconds'],
                    new_graphs=compiled_graphs-before,
                    nonfinite_reference_parameters=[n for n,v in reference['grads'].items() if not bool(torch.isfinite(v).all())],
                    nonfinite_compiled_parameters=[n for n,v in observed['grads'].items() if not bool(torch.isfinite(v).all())],
                    gradient_similarity=similarity(list(reference['grads'].values()),list(observed['grads'].values())),
                    input_gradient_similarity=similarity(reference['input_grads'][:1],observed['input_grads'][:1]),
                    losses={name:dict(eager=reference['metrics'][name],compiled=observed['metrics'][name],
                        relative_error=abs(reference['metrics'][name]-observed['metrics'][name])/max(abs(reference['metrics'][name]),1e-20))
                        for name in ('world_model_loss','active_fraction_loss')})
                warmed=[run(True)['seconds'] for _ in range(2)]
                after_warm=int(counters['stats']['unique_graphs'])
                stable=[run(True)['seconds'] for _ in range(5)]
                row.update(warmed_seconds=warmed,stable_seconds=stable,
                    warmup_additional_graphs=after_warm-compiled_graphs,
                    warmed_additional_graphs=int(counters['stats']['unique_graphs'])-after_warm,
                    parameter_ids_preserved=ids=={n:id(p) for n,p in trainer.world_model.named_parameters()},
                    optimizer_finite=all(bool(torch.isfinite(value).all()) for optimizer in opts.values()
                        for state in optimizer.state.values() for value in state.values() if isinstance(value,torch.Tensor)),
                    parameters_finite=all(bool(torch.isfinite(p).all()) for p in trainer.world_model.parameters()))
                tail_results.append(row)
                if row['warmed_additional_graphs'] or not row['gradient_similarity']['finite'] or not row['optimizer_finite']:
                    result['tail_proof']=tail_results
                    raise RuntimeError('tail compilation did not stabilize or produced nonfinite gradients/optimizer')
                result['tail_proof']=tail_results
                result['dynamo_graph_breaks']=dict(counters['graph_break'])
                save()
                print('Tail',size,'new graphs',row['new_graphs'],'warm',warmed,flush=True)
        result['peak_allocated_mib']=torch.cuda.max_memory_allocated(device)/1024**2
        result['peak_reserved_mib']=torch.cuda.max_memory_reserved(device)/1024**2
        result['status']='ok'
    except Exception as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    save()
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
