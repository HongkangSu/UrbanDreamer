"""Isolated block compilation proof; CPU results do not predict CUDA speedups."""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Bound existing compiler workers; no additional dependency is installed.
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','2')
if __name__=='__main__':
    preflight=argparse.ArgumentParser(add_help=False)
    preflight.add_argument('--device',default='cpu')
    preliminary,_=preflight.parse_known_args()
    if preliminary.device=='cpu':
        # Dynamo can ask for the current accelerator stream even while tracing
        # CPU tensors. Hide GPUs before importing torch for an actual CPU probe.
        os.environ['CUDA_VISIBLE_DEVICES']=''

import torch
from torch.utils.checkpoint import checkpoint
from urban_dreamer.models.world_model import BlockCausalLayer, WORLD_MODEL_DIM


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda:1'),default='cpu')
    parser.add_argument('--allow-cuda',action='store_true',help='explicitly opt in to a later isolated GPU probe')
    parser.add_argument('--dtype',choices=('fp32','bf16'),default='fp32')
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--time-steps',type=int,default=1)
    parser.add_argument('--tokens',type=int,default=16)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--gpu-memory-fraction',type=float,default=.075)
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(args.output.parent/'inductor_absolute_cache'))
    result=dict(benchmark_only=True,optimizer_steps=0,device=args.device,dtype=args.dtype,
                shape=[args.batch,args.time_steps,args.tokens,WORLD_MODEL_DIM],
                backend='inductor',mode='default',fullgraph=True,dynamic=False,
                cpu_timing_predicts_cuda_acceleration=False)
    result['cpu_gpu_visibility_disabled']=args.device=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    try:
        if args.device.startswith('cuda') and not args.allow_cuda:
            raise RuntimeError('CUDA probe requires --allow-cuda and an available GPU resource window')
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261002)
        device=torch.device(args.device)
        if device.type=='cuda':
            torch.cuda.set_device(device)
            free,total=torch.cuda.mem_get_info(device)
            if free < args.gpu_memory_fraction*total + .5*1024**3:
                raise RuntimeError('insufficient free GPU memory for the explicit isolated compile probe')
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)
            result['gpu']=torch.cuda.get_device_name(device)
        result['torch_version']=torch.__version__
        eager=BlockCausalLayer(args.tokens).to(device)
        compiled=copy.deepcopy(eager)
        # Construct an optimizer first to prove compilation keeps its references.
        optimizer=torch.optim.SGD(compiled.parameters(),lr=.01)
        parameter_ids={name:id(value) for name,value in compiled.named_parameters()}
        state_keys=list(compiled.state_dict())
        compiled.compile(backend='inductor',mode='default',fullgraph=True,dynamic=False)
        result['parameter_object_ids_preserved']=parameter_ids=={name:id(value) for name,value in compiled.named_parameters()}
        result['state_dict_keys_preserved']=state_keys==list(compiled.state_dict())
        result['optimizer_parameter_ids_preserved']=set(parameter_ids.values())=={
            id(value) for group in optimizer.param_groups for value in group['params']}
        values=torch.randn(*result['shape'],device=device)
        target=torch.randn_like(values)
        def synchronize():
            if device.type=='cuda':
                torch.cuda.synchronize(device)
        def run(module,*,checkpointed=False,capture=False):
            module.zero_grad(set_to_none=True)
            inputs=values.detach().clone().requires_grad_(True)
            context=torch.autocast(device.type,dtype=torch.bfloat16) if args.dtype=='bf16' else nullcontext()
            with context:
                output=checkpoint(module,inputs,use_reentrant=False) if checkpointed else module(inputs)
                loss=(output.float()-target.float()).square().mean()
            loss.backward()
            if not capture:
                return float(loss.detach())
            return dict(output=output.detach().float().cpu(),input_gradient=inputs.grad.detach().float().cpu(),
                        parameter_gradients={name:value.grad.detach().float().cpu().clone() for name,value in
                            module.named_parameters() if value.grad is not None},loss=float(loss.detach()))
        reference=run(eager,capture=True)
        synchronize();start=time.perf_counter()
        observed=run(compiled,capture=True)
        synchronize();result['cold_forward_backward_compile_seconds']=time.perf_counter()-start
        checkpoint_start=time.perf_counter()
        checkpointed=run(compiled,checkpointed=True,capture=True)
        synchronize();result['first_checkpointed_forward_backward_seconds']=time.perf_counter()-checkpoint_start
        def compare(reference,observed):
            gradients_reference=reference['parameter_gradients']
            gradients_observed=observed['parameter_gradients']
            checks=dict(output_close=torch.allclose(reference['output'],observed['output'],atol=2e-5,rtol=1e-4),
                        input_gradient_close=torch.allclose(reference['input_gradient'],observed['input_gradient'],atol=2e-5,rtol=1e-4),
                        gradient_parameter_names_preserved=set(gradients_reference)==set(gradients_observed))
            differences=[]
            checks['all_parameter_gradients_close']=checks['gradient_parameter_names_preserved']
            for name,value in gradients_reference.items():
                candidate=gradients_observed[name]
                checks['all_parameter_gradients_close'] &= torch.allclose(value,candidate,atol=2e-5,rtol=1e-4)
                differences.append(float((value-candidate).abs().max()))
            return dict(**checks,passed=all(checks.values()),loss_reference=reference['loss'],loss_observed=observed['loss'],
                        output_max_abs_difference=float((reference['output']-observed['output']).abs().max()),
                        input_gradient_max_abs_difference=float((reference['input_gradient']-observed['input_gradient']).abs().max()),
                        parameter_gradient_max_abs_difference=max(differences,default=0.))
        result['eager_vs_compiled']=compare(reference,observed)
        result['eager_vs_compiled_checkpoint']=compare(reference,checkpointed)
        timings={}
        for label,module,checkpointed_flag in [('eager',eager,False),('eager_checkpointed',eager,True),('compiled',compiled,False),
                                               ('compiled_checkpointed',compiled,True)]:
            run(module,checkpointed=checkpointed_flag);synchronize()
            seconds=[]
            for _ in range(args.repeats):
                synchronize();start=time.perf_counter()
                run(module,checkpointed=checkpointed_flag);synchronize()
                seconds.append(time.perf_counter()-start)
            timings[label]=dict(median_seconds=statistics.median(seconds),measurements_seconds=seconds)
        result['warm_timings']=timings
        result['illustrative_cpu_speedup']=timings['eager']['median_seconds']/timings['compiled']['median_seconds']
        result['illustrative_cpu_checkpointed_speedup']=timings['eager_checkpointed']['median_seconds']/timings['compiled_checkpointed']['median_seconds']
        result['proof_passed']=all([result['parameter_object_ids_preserved'],result['state_dict_keys_preserved'],
            result['optimizer_parameter_ids_preserved'],result['eager_vs_compiled']['passed'],
            result['eager_vs_compiled_checkpoint']['passed']])
        result['status']='ok' if result['proof_passed'] else 'verification_failed'
        if device.type=='cuda':
            result['peak_allocated_mb']=torch.cuda.max_memory_allocated(device)/1024**2
            result['peak_reserved_mb']=torch.cuda.max_memory_reserved(device)/1024**2
    except Exception as exc:
        result.update(status='failed',error=str(exc),exception=type(exc).__name__,traceback=traceback.format_exc())
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
