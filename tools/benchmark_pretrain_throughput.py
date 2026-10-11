"""Private no-update timing probe; never establishes codec readiness or saves a training checkpoint."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import inspect
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from urban_dreamer.learning.training import MainlineTrainer
import urban_dreamer.learning.training as training_module
from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import DataConfig, load_config
from tools.train_codec_ready import SparseGraphCodec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ingolstadt_custom.toml')
    parser.add_argument('--cache-manifest', type=Path, default=ROOT/'outputs/ingolstadt_pretrain_cache_20261002/cache_manifest.json')
    parser.add_argument('--device', choices=('cuda:0','cuda:1'), default='cuda:1')
    parser.add_argument('--fm-only', action='store_true')
    parser.add_argument('--space-chunk', type=int)
    parser.add_argument('--time-chunk', type=int)
    parser.add_argument('--mlp-chunk', type=int)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--profile', choices=('initial', 'online', 'unchunked'), default='online')
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--profiler', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-fraction', type=float, default=.07)
    parser.add_argument('--headroom-mb', type=float,default=819.2)
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--compare-precision', action='store_true',
                        help='compare fixed-RNG FP32 versus BF16 loss, outputs and gradients before timing')
    parser.add_argument('--checkpoint-path', type=Path,
                        default=ROOT/'outputs/ingolstadt_complete_cohort_pretrain_20261002/checkpoints/latest.pt')
    parser.add_argument('--omit-full-space-mask',action='store_true',
                        help='private mathematically equivalent removal of all-true mask, enabling BF16 FlashAttention')
    parser.add_argument('--singleton-time-attention',action='store_true',
                        help='private exact value-projection shortcut for temporal attention length1')
    parser.add_argument('--teacher-readout',action='store_true',
                        help='private observed-target exposure readout; removes detached sampling solves only')
    parser.add_argument('--materialization-horizon',type=int,
                        help='private prefix-only batch materialization probe; no epoch iterator or manifest is changed')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(benchmark_only=True, codec_validation_bypassed_for_timing_only=True,
                  optimizer_steps=0, batch=args.batch, depth=args.depth,
                  chunk_profile=args.profile, activation_checkpointing=not args.no_checkpoint,
                  allocator_memory_fraction=args.memory_fraction,precision=args.precision,
                  codec_precision='fp32',omit_full_space_mask=args.omit_full_space_mask)
    result['singleton_time_attention'] = args.singleton_time_attention
    result['teacher_readout'] = args.teacher_readout
    result['materialization_horizon'] = args.materialization_horizon
    try:
        torch.set_num_threads(4)
        torch.cuda.set_device(args.device)
        free, total = torch.cuda.mem_get_info(torch.device(args.device))
        result.update(device=args.device, gpu=torch.cuda.get_device_name(torch.device(args.device)),
                      free_before_mb=free/1024**2, total_mb=total/1024**2)
        if free < args.memory_fraction*total + args.headroom_mb*1024**2:
            raise RuntimeError('insufficient currently free GPU memory for isolated probe')
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, torch.device(args.device))
        result['bf16_supported'] = torch.cuda.is_bf16_supported()
        if args.precision == 'bf16' and not result['bf16_supported']:
            raise RuntimeError('GPU does not support BF16')
        base = load_config(args.config)
        manifest = args.cache_manifest
        config = replace(base, run=replace(base.run, device=args.device, output_dir=args.output.parent/'private_unused'),
                         data=DataConfig(pretrain_cache_manifest=manifest, validation_cache_manifest=manifest))
        prepared = validate_prepared_inputs(config)
        topology = load_traffic_topology(prepared)
        start = time.perf_counter()
        trainer = MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology)
        result['startup_seconds'] = time.perf_counter()-start
        # Benchmarking old weights does not resume the old, incorrect target semantics.
        old = torch.load(args.checkpoint_path,map_location='cpu', weights_only=True)
        result['weights_source'] = str(args.checkpoint_path.resolve())
        result['weights_source_schema'] = old.get('schema_version')
        trainer.world_model.load_state_dict(old['training_state']['models']['world_model'], strict=True)
        old_optimizer = old['training_state']['optimizers']['world_model_initial']
        result['historical_adam_state_mb'] = sum(v.numel()*v.element_size()
            for state in old_optimizer['state'].values() for v in state.values() if isinstance(v,torch.Tensor))/1024**2
        del old, old_optimizer
        trainer.world_model.flow_projector.requires_grad_(False)
        trainer.world_model.train()
        sparse = SparseGraphCodec(adjacency=trainer._tensor(topology.network.adjacency),
                                  positive_speed=True).to(trainer.device)
        trainer.world_model.flow_projector.__class__ = SparseGraphCodec
        trainer.world_model.flow_projector.register_buffer('sparse_adjacency', sparse.sparse_adjacency,persistent=False)
        del sparse
        # Explicitly private: never export this gate bypass into a production artifact.
        trainer.validate_codec_ready = lambda: None
        # Preserve the accepted codec's exact FP32 arithmetic while testing only
        # the trainable dynamics and exposure readout under CUDA autocast.
        original_encode = trainer.world_model.encode
        original_decode = trainer.world_model.decode
        def encode_fp32(value):
            with torch.autocast('cuda',enabled=False):
                return original_encode(value.float())
        def decode_fp32(value):
            with torch.autocast('cuda',enabled=False):
                return original_decode(value.float())
        trainer.world_model.encode = encode_fp32
        trainer.world_model.decode = decode_fp32
        event_encoder = trainer.world_model.dynamics.event_encoder
        original_event_features = event_encoder.event_features
        event_encoder.event_features = lambda events: original_event_features(events).float()
        result['private_amp_event_feature_sum_cast'] = True
        trainer.metrics['world_model_updates'] = (args.depth-1)*config.method.supervision_increment_updates
        chunks = {'initial': (1,256,4096), 'online': (16,32768,16384),
                  'unchunked': (2**30,2**30,2**30)}[args.profile]
        trainer.world_model.dynamics.set_pretraining_execution_profile(
            space_chunk=args.space_chunk or chunks[0],time_chunk=args.time_chunk or chunks[1],mlp_chunk=args.mlp_chunk or chunks[2],
            activation_checkpointing=not args.no_checkpoint)
        if args.omit_full_space_mask:
            for layer in trainer.world_model.dynamics.transformer.layers:
                if not bool(layer.space.attn_mask.all()):
                    raise RuntimeError('private all-true mask optimization requires every mask entry allowed')
                original_attention = layer.space.attention.forward
                def unmasked_attention(x, *, attn_mask=None, is_causal=False, original=original_attention):
                    return original(x,attn_mask=None,is_causal=is_causal)
                layer.space.attention.forward = unmasked_attention
        if args.singleton_time_attention:
            for layer in trainer.world_model.dynamics.transformer.layers:
                module = layer.time.attention
                original = module.forward
                def singleton_attention(x, *, attn_mask=None, is_causal=False, module=module, original=original):
                    if x.shape[1] == 1 and attn_mask is None:
                        return module.output(module.qkv(x).chunk(3,dim=-1)[2])
                    return original(x,attn_mask=attn_mask,is_causal=is_causal)
                module.forward = singleton_attention
        if args.teacher_readout:
            original_helper = training_module.independent_transition_fm_loss
            def teacher_helper(dynamics,**kwargs):
                if 'predict_endpoints' in inspect.signature(original_helper).parameters:
                    kwargs['predict_endpoints'] = False
                    return original_helper(dynamics,**kwargs)
                original_solve = dynamics.solve
                target = kwargs['z_targets'].detach()
                teacher = target.reshape(-1,1,*target.shape[-2:])
                dynamics.solve = lambda *values,**options:teacher
                try:
                    return original_helper(dynamics,**kwargs)
                finally:
                    dynamics.solve = original_solve
            training_module.independent_transition_fm_loss = teacher_helper
        result['parameters'] = {}
        for name,module in [('world_model',trainer.world_model),('dynamics',trainer.world_model.dynamics),
                            ('codec',trainer.world_model.flow_projector),('actor',trainer.actor),
                            ('macro_critic',trainer.latent_critic),('event_critic',trainer.event_critic)]:
            result['parameters'][name] = dict(total=sum(p.numel() for p in module.parameters()),
                trainable=sum(p.numel() for p in module.parameters() if p.requires_grad),
                weight_mb=sum(p.numel()*p.element_size() for p in module.parameters())/1024**2)
        data_times=[]
        batches=[]
        if args.materialization_horizon is not None:
            if args.materialization_horizon < args.depth:
                raise ValueError('batch materialization must cover the supervision depth')
            trainer.offline_data.sequence_length = args.materialization_horizon
        for _ in range(args.repeats+1):
            start=time.perf_counter()
            batches.append(trainer.offline_data.batch_from_starts(0,list(range(args.batch)),fm_only=args.fm_only))
            data_times.append(time.perf_counter()-start)
        scoped_start=time.perf_counter()
        prepare_transition_targets(batches[-1],capacities=trainer.network.capacities)
        result['cached_target_preparation_seconds']=time.perf_counter()-scoped_start
        def step(batch, precision=None):
            trainer.world_model.zero_grad(set_to_none=True)
            chosen_precision = precision or args.precision
            context = torch.autocast('cuda',dtype=torch.bfloat16) if chosen_precision == 'bf16' else nullcontext()
            with context:
                loss, readout=trainer._world_loss(batch)
            loss.backward()
            readout.backward()
            return float(loss.detach()),float(readout.detach())
        if args.compare_precision:
            comparisons={}
            for precision in ('fp32','bf16'):
                torch.manual_seed(20261002)
                torch.cuda.manual_seed_all(20261002)
                outputs=[]
                hook=trainer.world_model.dynamics.flow_head.register_forward_hook(
                    lambda module,values,output:outputs.append(output.detach().float().cpu()))
                losses=step(batches[0],precision)
                hook.remove()
                gradients={name:param.grad.detach().cpu().clone() for name,param in
                           trainer.world_model.named_parameters() if param.grad is not None}
                comparisons[precision]=dict(losses=losses,outputs=outputs,gradients=gradients)
            reference,test=comparisons['fp32'],comparisons['bf16']
            dot=norm_reference=norm_test=norm_difference=0.
            gradient_finite=True
            gradient_groups={}
            for name,gradient in reference['gradients'].items():
                candidate=test['gradients'][name]
                gradient_finite &= bool(torch.isfinite(gradient).all() and torch.isfinite(candidate).all())
                dot+=float((gradient*candidate).sum())
                norm_reference+=float(gradient.square().sum())
                norm_test+=float(candidate.square().sum())
                norm_difference+=float((gradient-candidate).square().sum())
                group=gradient_groups.setdefault(name.split('.')[0],[0.,0.,0.,0.])
                group[0]+=float((gradient*candidate).sum())
                group[1]+=float(gradient.square().sum())
                group[2]+=float(candidate.square().sum())
                group[3]+=float((gradient-candidate).square().sum())
            ref_output=torch.cat([value.reshape(-1) for value in reference['outputs']])
            test_output=torch.cat([value.reshape(-1) for value in test['outputs']])
            result['precision_comparison']=dict(fp32_losses=reference['losses'],bf16_losses=test['losses'],
                loss_relative_errors=[abs(a-b)/max(abs(a),1e-12) for a,b in zip(reference['losses'],test['losses'])],
                gradient_cosine=dot/max((norm_reference*norm_test)**.5,1e-12),
                gradient_relative_l2=(norm_difference/max(norm_reference,1e-24))**.5,
                gradients_finite=gradient_finite,outputs_finite=bool(torch.isfinite(ref_output).all() and torch.isfinite(test_output).all()),
                flow_head_output_relative_l2=float((ref_output-test_output).norm()/ref_output.norm().clamp_min(1e-12)),
                flow_head_output_max_abs_difference=float((ref_output-test_output).abs().max()),
                flow_head_call_relative_l2=[float((a-b).norm()/a.norm().clamp_min(1e-12))
                                           for a,b in zip(reference['outputs'],test['outputs'])],
                gradient_groups={name:dict(cosine=values[0]/max((values[1]*values[2])**.5,1e-24),
                    relative_l2=(values[3]/max(values[1],1e-24))**.5,
                    fp32_l2=values[1]**.5,bf16_l2=values[2]**.5)
                    for name,values in gradient_groups.items()})
            del comparisons,reference,test,ref_output,test_output,gradients,gradient,candidate
            trainer.world_model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        step(batches[0]);torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        measurements=[]
        for batch in batches[1:]:
            torch.cuda.synchronize();start=time.perf_counter()
            losses=step(batch)
            torch.cuda.synchronize()
            measurements.append(time.perf_counter()-start)
        result.update(status='ok', data_seconds=data_times, step_seconds=measurements,
            median_step_seconds=statistics.median(measurements),
            sequences_per_second=args.batch/statistics.median(measurements),
            independent_windows_per_second=args.batch*args.depth/statistics.median(measurements),
            final_losses=losses, peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
            peak_reserved_mb=torch.cuda.max_memory_reserved()/1024**2,
            allocated_mb=torch.cuda.memory_allocated()/1024**2,reserved_mb=torch.cuda.memory_reserved()/1024**2)
        result['model_buffer_mb'] = sum(b.numel()*b.element_size() for b in trainer.world_model.buffers()
                                      if not b.is_sparse)/1024**2
        query = subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        for line in query.stdout.splitlines():
            pid, used = line.split(',')
            if int(pid) == os.getpid():
                result['nvidia_smi_process_mb'] = float(used)
                result['cuda_context_and_other_overhead_mb'] = float(used)-result['reserved_mb']
        if args.profiler:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                step(batches[-1]);torch.cuda.synchronize()
            args.output.with_suffix('.profiler.txt').write_text(
                prof.key_averages().table(sort_by='self_device_time_total',row_limit=25)+'\n'+
                prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=25))
            result['profiler_cuda_top']=[dict(key=e.key,count=e.count,cpu_us=e.self_cpu_time_total,
                cuda_us=e.self_device_time_total) for e in sorted(prof.key_averages(),
                    key=lambda e:e.self_device_time_total,reverse=True)[:15]]
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        result.update(status='failed',error=str(exc),exception=type(exc).__name__,traceback=traceback.format_exc())
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
