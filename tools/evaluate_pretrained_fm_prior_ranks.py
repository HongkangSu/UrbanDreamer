"""Matched FM inference interventions on prior means, with native noise intact.

The completed FM body was trained at rank16. These are explicit inference
interventions, never claimed to be fully trained higher-rank world models.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = Path(os.environ.get('FM_PRIOR_RANK_FROZEN_ROOT', ROOT)).resolve()
sys.path.insert(0, str(FROZEN_ROOT))

import numpy as np
import torch
from torch import nn

from tools.component_experiment_data import atomic_json, dataset, load_frozen, sha256
from tools.evaluate_component_world_models import endpoint_target
from tools.evaluate_pretrained_condition_scope import score
from tools.evaluate_pretrained_distribution import CHECKPOINT_SHA256, trajectory_bank
from tools.summarize_component_predictions import demand_bootstrap
from urban_dreamer.learning.cohort_targets import prepare_cohort_batch
from urban_dreamer.learning.offline_data import archive_content_hash
from urban_dreamer.models.world_model import LowFrequencyGraphPrior

ARMS = ('native16', 'original_projection64', 'original_projection128', 'refit_projection64', 'refit_projection128')
SCOPES = ('c0', 'full_factual')
EVENT_FIELDS = ('batch_index', 'time_index', 'phase', 'current_edge', 'next_edge', 'destination_edge', 'event_index')
METRICS = (tuple(f'{feature}_{metric}_{scope}' for feature in ('count', 'speed')
    for metric in ('mae', 'crps', 'crps_u', 'sample_variance') for scope in ('all', 'occupied'))
    + tuple(f'{feature}_{metric}{level}_{scope}' for feature in ('count', 'speed')
    for metric in ('coverage', 'width') for level in (50, 80, 90, 95) for scope in ('all', 'occupied'))
    + ('energy_score', 'energy_score_u', 'total_count_bias', 'total_count_absolute_error',
       'total_count_sample_absolute_error', 'total_count_sample_variance', 'total_count_over_cohort_fraction',
       'total_count_max_cohort_excess', 'congestion_brier_all', 'congestion_brier_occupied'))


class ProjectionRankPrior(LowFrequencyGraphPrior):
    """Opt-in mean projection override; native rank/basis still define noise.

    Copy the loaded native modules directly to avoid reinitializing a graph
    eigensolver or changing checkpoint parameter names. The optional projection
    buffer is nonpersistent so native state-dict parameter shapes stay intact.
    """
    def __init__(self, original, projection_basis=None, coefficients=None):
        nn.Module.__init__(self)
        if getattr(original, 'prior_filter', 'hard') != 'hard':
            raise ValueError('projection-rank interventions require a hard-filter reference')
        self.rank = original.rank
        self.noise_scale = original.noise_scale
        self.prior_filter = 'hard'
        self.prior_filter_strength = getattr(original, 'prior_filter_strength', 10.0)
        self.prior_action_conditioned = getattr(original, 'prior_action_conditioned', True)
        self.filter_input = getattr(original, 'filter_input', False)
        for name, module in original._modules.items():
            self.add_module(name, copy.deepcopy(module))
        for name, value in original._buffers.items():
            self.register_buffer(name, None if value is None else value.detach().clone(),
                persistent=name not in original._non_persistent_buffers_set)
        basis = None if projection_basis is None else torch.as_tensor(projection_basis,
            device=original.basis.device, dtype=original.basis.dtype).detach().clone()
        if basis is not None and (basis.ndim != 2 or basis.shape[0] != len(original.basis)
                or not 1 <= basis.shape[1] < len(original.basis) or not bool(torch.isfinite(basis).all())):
            raise ValueError('finite matching projection basis with rank below edge count required')
        self.register_buffer('projection_basis', basis, persistent=False)
        self.prediction_rank = original.rank if basis is None else basis.shape[1]
        if coefficients is not None:
            self.load_output_coefficients(coefficients)
        self.eval().requires_grad_(False)

    def load_output_coefficients(self, coefficients):
        coefficients = torch.as_tensor(coefficients, device=self.output.weight.device, dtype=self.output.weight.dtype)
        if coefficients.shape != (self.output.in_features+1, self.output.out_features) or not bool(torch.isfinite(coefficients).all()):
            raise ValueError('finite augmented output coefficients must have shape (65,64)')
        with torch.no_grad():
            self.output.weight.copy_(coefficients[:-1].T)
            self.output.bias.copy_(coefficients[-1])

    @property
    def projection_rank(self):
        return self.prediction_rank

    def project(self, values):
        if self.projection_basis is None:
            return super().project(values)
        basis = self.projection_basis.to(values)
        return basis @ (basis.T @ values)


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.shape).encode()+str(value.dtype).encode()+value.tobytes()).hexdigest()


def controls_identical(batch, cohort):
    a, b = batch.action_events.for_step(0), cohort.transition_action_events.for_step(0)
    return bool(torch.equal(batch.action[:,0], cohort.transition_action[:,0]) and a.vehicle_ids == b.vehicle_ids
        and all(torch.equal(getattr(a,k),getattr(b,k)) for k in EVENT_FIELDS))


def paired_summaries(rows, *, seed=2026101302, repetitions=2000):
    indexed = {}
    for row in rows:
        key = (row['condition_index'],row['control_scope'],row['arm'])
        if key in indexed:
            raise ValueError('duplicate condition/control/arm')
        indexed[key] = row
    conditions = sorted({r['condition_index'] for r in rows})
    if not conditions or any((index,scope,arm) not in indexed for index in conditions for scope in SCOPES for arm in ARMS):
        raise ValueError('every condition needs both scopes and every arm')
    result = dict(arms={}, intervention_minus_native={}, full_minus_c0={})
    for scope in SCOPES:
        result['arms'][scope], result['intervention_minus_native'][scope] = {}, {}
        for arm in ARMS:
            selected = [indexed[(index,scope,arm)] for index in conditions]
            result['arms'][scope][arm] = {metric: demand_bootstrap(selected,metric,seed=seed,repetitions=repetitions) for metric in METRICS}
            if arm == 'native16':
                continue
            differences = []
            for index in conditions:
                a, b = indexed[(index,scope,arm)], indexed[(index,scope,'native16')]
                if any(a[k]!=b[k] for k in ('demand_group','episode','condition_seed')):
                    raise ValueError('paired condition identity differs')
                row = {k: a[k] for k in ('demand_group','episode')}
                for metric in METRICS:
                    if (a.get(metric) is None)!=(b.get(metric) is None):
                        raise ValueError('paired metric availability differs')
                    row[metric] = None if a.get(metric) is None else a[metric]-b[metric]
                differences.append(row)
            result['intervention_minus_native'][scope][arm] = {metric: demand_bootstrap(differences,metric,seed=seed,repetitions=repetitions) for metric in METRICS}
    for arm in ARMS:
        differences = []
        for index in conditions:
            a, b = indexed[(index,'full_factual',arm)],indexed[(index,'c0',arm)]
            row = {k:a[k] for k in ('demand_group','episode')}
            for metric in METRICS:
                if (a.get(metric) is None)!=(b.get(metric) is None):
                    raise ValueError('control metric availability differs')
                row[metric] = None if a.get(metric) is None else a[metric]-b[metric]
            differences.append(row)
        result['full_minus_c0'][arm] = {metric:demand_bootstrap(differences,metric,seed=seed,repetitions=repetitions) for metric in METRICS}
    return result


def audit_source(directory, manifest_path):
    entries = json.loads(manifest_path.read_text())
    for relative, expected in entries.get('files',entries).items():
        if sha256(directory/relative)!=expected:
            raise ValueError('frozen source changed: '+relative)
    return sha256(manifest_path)


def model_digest(model, *, exclude_prior=False):
    digest = hashlib.sha256()
    for name,value in model.state_dict().items():
        if exclude_prior and name.startswith('dynamics.graph_prior.'):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_metrics(summary, output):
    import csv
    with (output/'paired_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['control_scope','arm','metric','estimate','ci_low','ci_high','delta_vs_native','delta_ci_low','delta_ci_high'])
        writer.writeheader()
        for scope in SCOPES:
            for arm in ARMS:
                for metric in METRICS:
                    absolute=summary['arms'][scope][arm][metric]
                    delta=summary['intervention_minus_native'][scope].get(arm,{}).get(metric,{})
                    writer.writerow(dict(control_scope=scope,arm=arm,metric=metric,
                        **{k:absolute[k] for k in ('estimate','ci_low','ci_high')},
                        delta_vs_native=delta.get('estimate'),delta_ci_low=delta.get('ci_low'),delta_ci_high=delta.get('ci_high')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    fields=('count_mae_occupied','speed_mae_occupied','energy_score_u')
    for row,scope in enumerate(SCOPES):
        for axis,metric in zip(axes[row],fields):
            for prefix,label in [('original','Original output head'),('refit','Refitted output head')]:
                arms=['native16',prefix+'_projection64',prefix+'_projection128']
                estimates=[summary['arms'][scope][a][metric] for a in arms]
                values=[e['estimate'] for e in estimates]
                axis.errorbar([16,64,128],values,yerr=[[e['estimate']-e['ci_low'] for e in estimates],
                    [e['ci_high']-e['estimate'] for e in estimates]],fmt='o-',capsize=3,label=label)
            axis.set(title=scope+' — '+metric,xlabel='Prior mean projection rank')
            axis.grid(alpha=.2)
    axes[0,0].legend()
    figure.suptitle('Completed rank16 FM with explicit prior-mean interventions; native rank16 noise unchanged')
    for extension in ('png','pdf'):
        figure.savefig(output/('fm_prior_projection.'+extension),dpi=180)
    plt.close(figure)


def run(args):
    started=time.time()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8' or os.environ.get('CUDA_VISIBLE_DEVICES')!='1':
        raise ValueError('prespecified physical GPU1 and deterministic CUBLAS environment required before import')
    if args.output.exists() and (args.output/'states.jsonl').exists():
        raise ValueError('preserve prior output/failures')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'banks').mkdir(exist_ok=True)
    source_hash=audit_source(FROZEN_ROOT,args.runtime_manifest)
    checkpoint=args.experiment/'snapshot/checkpoints/latest.pt'
    if sha256(checkpoint)!=CHECKPOINT_SHA256:
        raise ValueError('completed checkpoint required')
    reference_summary=json.loads((args.reference/'summary.json').read_text())
    reference_protocol=json.loads((args.reference/'protocol.json').read_text())
    if reference_summary['status']!='complete' or not reference_protocol['deterministic_cuda']:
        raise ValueError('completed deterministic v4 reference required')
    frozen_inputs={str(path.resolve()):sha256(path) for path in (
        checkpoint,args.rank_study/'exact_basis.npz',args.rank_study/'output_head_coefficients.npz',
        args.rank_study/'summary.json',args.reference/'summary.json',args.reference/'protocol.json',
        args.reference/'bank_manifest.json',args.experiment/'development_manifest.json')}
    reference_entries=json.loads((args.reference/'bank_manifest.json').read_text())
    if len(reference_entries)!=24 or len({e['condition']['demand_group'] for e in reference_entries})!=8:
        raise ValueError('prespecified24 states and8 groups required')
    for entry in reference_entries:
        path=Path(entry['file'])
        if sha256(path)!=entry['sha256']:
            raise ValueError('reference bank changed')
        frozen_inputs[str(path)]=entry['sha256']
        reproduction=args.reference/'banks'/('reproduction_'+path.stem.split('_')[-1]+'.npz')
        frozen_inputs[str(reproduction.resolve())]=sha256(reproduction)
    atomic_json(args.output/'input_hashes.json',frozen_inputs)
    protocol=dict(status='prespecified_before_scoring',checkpoint_sha256=CHECKPOINT_SHA256,
        source_manifest_sha256=source_hash,code_sha256=sha256(__file__),arms=list(ARMS),control_scopes=list(SCOPES),
        groups=8,conditions=24,k=8,horizon=1,solver='euler',steps=100,sample_chunk=4,
        physical_gpu=1,gpu_memory_fraction=.06,minimum_free_mib=4096,threads=2,
        target='exact savedv4 C0 cohort count/following-boundary speed; same scales and masks for every arm',
        source_noise='same storedrank16basis,noise_scale,coefficients andseeds; onlyprior meanprojection changes',
        model='completed rank16 FM checkpoint; mean interventions are not fully trained higher-rank models',
        output_refit='existingv1 solved65x64 head from768trainingwindows/64groups; no new fitting or gradient updates',
        controls='C0 and full factual separated; full factual within-window events privilegedoffline only',
        gate='all24 nativeC0 bitwise reproduced before full/control/rank interventions; nativefull also bitwise',
        bootstrap='2000 paired demandgroup resamples, equal demand/episode/state weighting, same seed2026101302',
        metrics=list(METRICS),training_performed=False,final_test_used=False,
        numeric_environment=dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,
            deterministic_algorithms=True,matmul_precision='highest',
            environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE',
                'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
    atomic_json(args.output/'protocol.json',protocol)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0')
    torch.cuda.set_device(device)
    free,total=torch.cuda.mem_get_info(device)
    if free<max(4096*1024**2,.06*total+512*1024**2):
        raise RuntimeError('GPU1 admission headroom below prespecifiedminimum')
    torch.cuda.set_per_process_memory_fraction(.06,device)
    config,prepared,topology,model,_=load_frozen(args.experiment,device=device)
    model.eval().requires_grad_(False)
    native=model.dynamics.graph_prior
    if native.rank!=16:
        raise ValueError('fixed original prior rank16 required')
    original_digest=model_digest(model)
    body_digest=model_digest(model,exclude_prior=True)
    with np.load(args.rank_study/'exact_basis.npz') as stored:
        basis=stored['basis'].copy()
    with np.load(args.rank_study/'output_head_coefficients.npz') as stored:
        heads={rank:stored['rank_'+str(rank)].copy() for rank in (64,128)}
    variants={'native16':native}
    for prefix in ('original','refit'):
        for rank in (64,128):
            variants[prefix+'_projection'+str(rank)]=ProjectionRankPrior(native,basis[:,:rank],
                None if prefix=='original' else heads[rank])
    validation=dataset(args.experiment,config,prepared,topology,split='validation',horizon=9)
    capacities=torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1)
    limits=torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
    protocol['numeric_environment'].update(device_name=torch.cuda.get_device_name(device),
        cudnn_version=torch.backends.cudnn.version(),cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,matmul_precision=torch.get_float32_matmul_precision(),
        matmul_fp32_precision=getattr(torch.backends.cuda.matmul,'fp32_precision',None),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    atomic_json(args.output/'protocol.json',protocol)
    paired_inputs=[]
    baseline_proofs=[]
    noise_proofs=[]
    def progress(stage,done,total,**extra):
        state=dict(stage=stage,done=done,total=total,pid=os.getpid(),elapsed_seconds=time.time()-started,
            max_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,**extra)
        atomic_json(args.output/'progress.json',state)
        print(json.dumps(state),flush=True)
    with torch.no_grad():
        # Global gate: nativeC0 for all24 before any intervention prediction.
        for ordinal,entry in enumerate(reference_entries):
            index=int(Path(entry['file']).stem.split('_')[-1])
            item=entry['condition']
            with np.load(entry['file']) as stored:
                old={k:stored[k].copy() for k in stored.files}
            reproduction=args.reference/'banks'/f'reproduction_{index:03d}.npz'
            with np.load(reproduction) as stored:
                fixed={k:stored[k].copy() for k in stored.files}
            batch=validation.batch_from_starts(item['episode_index'],[item['start']],fm_only=False)
            cohort=prepare_cohort_batch(batch,capacities=capacities)
            initial=model.encode(cohort.transition_flow[:,0].to(device))
            np.testing.assert_array_equal(initial.cpu().numpy(),fixed['initial_latent'])
            np.testing.assert_allclose(endpoint_target(cohort.transition_flow[:,1],batch.flow[:,2],capacities,limits),old['targets'],atol=1e-5,rtol=0)
            same=controls_identical(batch,cohort)
            options=dict(capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],
                horizons=(1,),samples=8,chunk=4,steps=100)
            c0=trajectory_bank(model,initial,cohort.transition_action.to(device),cohort.transition_action_events.to(device),**options)[1]
            np.testing.assert_array_equal(c0,old['c0_draws'])
            baseline_proofs.append(dict(condition_index=index,c0_bit_identical=True,controls_identical=same,
                source_coefficients_sha256=array_hash(fixed['source_coefficients']),
                initial_latent_sha256=array_hash(fixed['initial_latent'])))
            for begin in (0,4):
                coefficient=torch.as_tensor(fixed['source_coefficients'][begin:begin+4,None],device=device)
                repeated=initial.repeat(4,1,1)[:,None]
                expected_noise=native.sample_noise(repeated,coefficient)
                for arm,prior in variants.items():
                    if not torch.equal(prior.sample_noise(repeated,coefficient),expected_noise):
                        raise ValueError('mean intervention changed native sample_noise: '+arm)
                noise_proofs.append(dict(condition_index=index,chunk_start=begin,all_arms_noise_bit_identical=True,
                    native_noise_sha256=array_hash(expected_noise.cpu().numpy())))
            paired_inputs.append((index,item,batch,cohort,initial,c0,old,fixed,options,same))
            atomic_json(args.output/'native_reproduction.json',baseline_proofs)
            progress('native_c0_gate',ordinal+1,24)
        rows=[]
        bank_manifest=[]
        control_proofs=[]
        with (args.output/'states.jsonl').open('x') as stream:
            for ordinal,(index,item,batch,cohort,initial,c0,old,fixed,options,same) in enumerate(paired_inputs):
                predictions={'native16':{'c0':c0}}
                model.dynamics.graph_prior=native
                full=trajectory_bank(model,initial,batch.action.to(device),batch.action_events.to(device),**options)[1]
                np.testing.assert_array_equal(full,old['full_factual_draws'])
                predictions['native16']['full_factual']=full
                if same:
                    np.testing.assert_array_equal(full,c0)
                for arm in ARMS[1:]:
                    model.dynamics.graph_prior=variants[arm]
                    predictions[arm]={}
                    for scope in SCOPES:
                        actions=cohort.transition_action if scope=='c0' else batch.action
                        events=cohort.transition_action_events if scope=='c0' else batch.action_events
                        predictions[arm][scope]=trajectory_bank(model,initial,actions.to(device),events.to(device),**options)[1]
                        progress('intervention',ordinal*8+(ARMS.index(arm)-1)*2+SCOPES.index(scope)+1,192,
                            condition_index=index,arm=arm,control_scope=scope)
                    if same:
                        np.testing.assert_array_equal(predictions[arm]['c0'],predictions[arm]['full_factual'])
                model.dynamics.graph_prior=native
                if model_digest(model,exclude_prior=True)!=body_digest or model_digest(model)!=original_digest:
                    raise ValueError('completed model weights changed during intervention')
                stored=dict(targets=old['targets'],scales=old['scales'],capacities=old['capacities'],speed_limits=old['speed_limits'],
                    initial_latent=fixed['initial_latent'],source_coefficients=fixed['source_coefficients'],coefficient_seeds=fixed['coefficient_seeds'],
                    native_noise_basis=native.basis.cpu().numpy(),native_noise_scale=np.asarray(native.noise_scale))
                for arm in ARMS:
                    for scope in SCOPES:
                        draws=predictions[arm][scope]
                        stored[arm+'_'+scope+'_draws']=draws
                        row=score(draws,old['targets'],old['scales'],int(cohort.cohort_count[0]),limits.numpy(),
                            dict(item,condition_index=index,horizon=1,k=8,control_scope=scope,arm=arm,
                                prediction_rank=16 if arm=='native16' else int(arm.split('projection')[-1]),noise_rank=16,
                                model='completed_rank16_fm_with_explicit_prior_mean_intervention'))
                        rows.append(row)
                        stream.write(json.dumps(row,allow_nan=False)+'\n')
                path=args.output/'banks'/f'condition_{index:03d}.npz'
                np.savez_compressed(path,**stored)
                bank_manifest.append(dict(condition_index=index,condition=item,file=str(path.resolve()),sha256=sha256(path)))
                control_proofs.append(dict(condition_index=index,controls_identical=same,
                    native_full_bit_identical=True,all_arm_control_invariance_verified=bool(same),
                    c0_events=len(cohort.transition_action_events.for_step(0)),full_events=len(batch.action_events.for_step(0))))
                atomic_json(args.output/'bank_manifest.json',bank_manifest)
                atomic_json(args.output/'control_invariance.json',control_proofs)
                stream.flush()
                progress('condition_complete',ordinal+1,24)
    # Final audit of every immutable input and actual imported source.
    for path,expected in frozen_inputs.items():
        if sha256(Path(path))!=expected:
            raise ValueError('immutable input changed: '+path)
    audit_source(FROZEN_ROOT,args.runtime_manifest)
    selected_indices={item['episode_index'] for _,item,*_ in paired_inputs}
    for index in selected_indices:
        entry=validation.manifest['entries'][index]
        if archive_content_hash(Path(entry['cache_path']))!=entry['cache_content_hash']:
            raise ValueError('selected compact data changed')
    imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
        if name.startswith(('tools.','urban_dreamer.')) and getattr(module,'__file__',None)}
    if any(not Path(path).is_relative_to(FROZEN_ROOT) for path in imports.values()):
        raise ValueError('live source contaminated frozen inference')
    atomic_json(args.output/'import_audit.json',imports)
    atomic_json(args.output/'source_noise_audit.json',noise_proofs)
    summary=dict(status='complete',checkpoint_sha256=CHECKPOINT_SHA256,conditions=24,groups=8,states=len(rows),
        native_c0_all_bit_identical=True,native_full_all_bit_identical=True,source_noise_all_bit_identical=True,
        original_body_unchanged=True,inputs_unchanged=True,source_unchanged=True,
        elapsed_seconds=time.time()-started,max_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        caveats=['Previously inspected development validation; eight demand groups and one checkpoint.',
            'Every FM/encoder/condition body trained at rank16; these are mean interventions, not fully retrained higher-rank models.',
            'Full factual controls are privileged offline inputs and cannot establish online actor performance.',
            'Intervals reflect demand-group variation, not independent training seeds; multiple comparisons unadjusted.'],
        **paired_summaries(rows))
    write_metrics(summary,args.output)
    atomic_json(args.output/'summary.json',summary)
    progress('complete',24,24)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--rank-study',type=Path,required=True)
    parser.add_argument('--runtime-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output/'failure.json',dict(status='failed',exception_type=type(error).__name__,message=str(error),
            no_fallback_or_warning_only_retry=True,pid=os.getpid()))
        raise
