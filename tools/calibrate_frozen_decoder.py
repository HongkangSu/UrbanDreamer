"""Calibrate a decoder on frozen train data; keep the pretrained latent encoder exact."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.train_codec_ready import FrameSampler, SparseGraphCodec, write_json
from tools.probe_slow_codec import freeze_encoder_for_decoder_probe, frozen_state_hash
from urban_dreamer.learning.codec_training import codec_objective, codec_decoded_metrics, make_decoder_optimizer
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset, compact_transition_targets
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.pipeline import combined_input_hash


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024): digest.update(chunk)
    return digest.hexdigest()


def verify_partition_separation(train_entries, validation_entries):
    for field in ('path','demand_group','trajectory_hash'):
        train={entry[field] for entry in train_entries}
        validation={entry[field] for entry in validation_entries}
        if train & validation:
            raise ValueError(f'calibration train/validation leakage in {field}')


def completed_epochs(sampler):
    return sampler.epoch+int(bool(sampler.order) and sampler.cursor==len(sampler.order))


def sampled_raw_frames(sampler):
    total=sum(entry['frames'] for entry in sampler.entries)
    visited=sum(sampler.entries[index]['frames'] for index in sampler.order[:sampler.cursor])
    return sampler.epoch*total+visited+sampler.frame_cursor


def calibration_pairs(arrays, starts, capacities):
    """All raw starts plus corrected endpoint starts with a real following frame."""
    starts=np.asarray(starts,dtype=np.int64)
    raw=(torch.from_numpy(arrays['flow'][starts]).clone(),torch.from_numpy(arrays['flow'][starts+1]).clone())
    eligible=starts[starts+2<len(arrays['flow'])]
    hybrid=None
    if len(eligible):
        target,_=compact_transition_targets(arrays,eligible.tolist(),1,capacities)
        hybrid=(torch.from_numpy(target[:,0]).clone(),torch.from_numpy(arrays['flow'][eligible+2]).clone())
    return raw,hybrid


class ValidationAccumulator:
    def __init__(self, edges):
        self.sums,self.weights={},{}
        self.slow_error=torch.zeros(edges,dtype=torch.float64)
        self.slow_count=torch.zeros(edges,dtype=torch.float64)
        self.local_error=torch.zeros(edges,dtype=torch.float64)
        self.local_count=torch.zeros(edges,dtype=torch.float64)
        self.local_frames=0

    def add(self,metrics,decoded,flow,following,capacity,limits,codec_config):
        frames,edges=flow.shape[:2]
        mean_slow_mask=((flow[...,2]*capacity>=codec_config.slow_min_vehicles-1e-5)
                        &(.5*(flow[...,1]+following[...,1])*limits<=codec_config.slow_speed_mps))
        persistent_mask=((flow[...,2]*capacity>=codec_config.slow_min_vehicles-1e-5)
                         &(flow[...,1]*limits<=codec_config.slow_speed_mps)
                         &(following[...,1]*limits<=codec_config.slow_speed_mps))
        errors=((decoded[...,1]-.5*(flow[...,1]+following[...,1]))*limits).abs()
        slow_roads_per_frame=mean_slow_mask.sum(-1)
        local_frames=(slow_roads_per_frame>=1)&(slow_roads_per_frame<=5)
        local_mask=mean_slow_mask&local_frames.unsqueeze(-1)
        self.local_frames+=int(local_frames.sum())
        self.local_error+=(errors*local_mask).sum(0).double().cpu()
        self.local_count+=local_mask.sum(0).double().cpu()
        metrics=dict(metrics,codec_persistent_slow_roads=persistent_mask.sum(),
            codec_speed_mae_persistent_slow_mps=(errors*persistent_mask).sum()/persistent_mask.sum().clamp_min(1))
        occupied=float(metrics['codec_count_occupied_roads'])
        empty=float(metrics['codec_count_empty_roads'])
        slow=float(metrics['codec_slow_roads'])
        persistent=float(metrics['codec_persistent_slow_roads'])
        for name,value in metrics.items():
            value=float(value)
            if not np.isfinite(value): raise FloatingPointError('nonfinite heldout codec metric')
            if name in ('codec_count_occupied_roads','codec_count_empty_roads','codec_slow_roads','codec_persistent_slow_roads'):
                self.sums[name]=self.sums.get(name,0.)+value;self.weights[name]=1.;continue
            weight=frames*edges
            if '_occupied' in name: weight=occupied
            if '_empty' in name or name.startswith('codec_empty_'): weight=empty
            if '_slow_' in name: weight=slow
            if '_persistent_slow_' in name: weight=persistent
            if name.startswith('codec_total_') or name in ('codec_predicted_count_total','codec_target_count_total'): weight=frames
            if name=='codec_count_rmse_all': value*=value
            self.sums[name]=self.sums.get(name,0.)+value*weight
            self.weights[name]=self.weights.get(name,0.)+weight
        self.slow_error+=(errors*mean_slow_mask).sum(0).double().cpu()
        self.slow_count+=mean_slow_mask.sum(0).double().cpu()

    def result(self,min_local_samples):
        metrics={key:self.sums[key]/weight if weight else None for key,weight in self.weights.items()}
        if metrics.get('codec_count_rmse_all') is not None: metrics['codec_count_rmse_all']=metrics['codec_count_rmse_all']**.5
        eligible=self.slow_count>=min_local_samples
        means=self.slow_error/self.slow_count.clamp_min(1)
        metrics['localized_slow_road_mae_max_mps']=float(means[eligible].max()) if bool(eligible.any()) else None
        metrics['localized_slow_eligible_roads']=int(eligible.sum())
        metrics['localized_slow_road_rows']=[dict(edge_index=i,frames=int(self.slow_count[i]),mae_mps=float(means[i]))
                                          for i in eligible.nonzero().flatten().tolist()]
        local_eligible=self.local_count>=min_local_samples
        local_means=self.local_error/self.local_count.clamp_min(1)
        local_cells=float(self.local_count.sum())
        metrics.update(local_context_frames=self.local_frames,local_context_slow_cells=int(local_cells),
            local_context_slow_mae_mps=float(self.local_error.sum()/local_cells) if local_cells else None,
            local_context_slow_road_max_mps=float(local_means[local_eligible].max()) if bool(local_eligible.any()) else None,
            local_context_eligible_roads=int(local_eligible.sum()),
            local_context_slow_road_rows=[dict(edge_index=i,frames=int(self.local_count[i]),mae_mps=float(local_means[i]))
                                        for i in local_eligible.nonzero().flatten().tolist()])
        return metrics


def gate(metrics,codec_config,localized_max):
    thresholds=dict(codec_count_mae_occupied=codec_config.occupied_vehicle_mae_max,
        codec_count_mae_all=codec_config.vehicle_mae_max,codec_count_mae_empty=codec_config.empty_vehicle_mae_max,
        codec_total_relative_error=codec_config.total_relative_mae_max,
        codec_speed_mae_mps=codec_config.speed_mae_mps_max,
        codec_speed_mae_occupied_mps=codec_config.occupied_speed_mae_mps_max,
        codec_speed_mae_slow_mps=codec_config.slow_speed_mae_mps_max,
        local_context_slow_mae_mps=codec_config.slow_speed_mae_mps_max,
        local_context_slow_road_max_mps=localized_max,
        localized_slow_road_mae_max_mps=localized_max)
    checks={key:metrics.get(key) is not None and np.isfinite(metrics[key]) and metrics[key]<=limit
            for key,limit in thresholds.items()}
    return dict(passed=all(checks.values()),checks=checks,thresholds=thresholds)


@torch.no_grad()
def validate(codec,dataset,network,codec_config,batch_size,device,localized_min_samples,localized_max):
    capacity=torch.as_tensor(network.capacities,device=device).float().clamp_min(1.)
    limits=torch.as_tensor(network.speed_limits_mps,device=device).float()
    accumulators={kind:ValidationAccumulator(network.edge_count) for kind in ('raw','hybrid')}
    codec.eval()
    for index,entry in enumerate(dataset.manifest['entries']):
        arrays=dataset._episode(index)
        for left in range(0,entry['windows'],batch_size):
            pairs=calibration_pairs(arrays,np.arange(left,min(left+batch_size,entry['windows'])),network.capacities)
            for kind,pair in zip(('raw','hybrid'),pairs):
                if pair is None: continue
                flow,following=[v.to(device) for v in pair]
                decoded=codec.decode(codec(flow))
                metrics=codec_decoded_metrics(decoded,flow,following,network,codec_config)
                accumulators[kind].add(metrics,decoded,flow,following,capacity,limits,codec_config)
    result={kind:acc.result(localized_min_samples) for kind,acc in accumulators.items()}
    result['gates']={kind:gate(result[kind],codec_config,localized_max) for kind in accumulators}
    result['passed']=all(v['passed'] for v in result['gates'].values())
    return result


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])


def restore_calibration(payload,contract,codec,optimizer,sampler):
    if payload.get('schema_version')!=1 or payload.get('contract')!=contract:
        raise ValueError('calibration resume source/data/objective/budget contract differs')
    codec.load_state_dict(payload['codec'])
    if frozen_state_hash(codec)!=contract['frozen_parameters_sha256']:
        raise ValueError('calibration resume changed the latent encoder or global head')
    optimizer.load_state_dict(payload['optimizer']);sampler.load_state_dict(payload['sampler'])
    restore_rng(payload['rng'])
    return payload['updates'],payload['history'],payload['validation']


def save_calibration(path,codec,optimizer,sampler,contract,updates,history,validation,status):
    if frozen_state_hash(codec)!=contract['frozen_parameters_sha256']:
        raise AssertionError('calibration modified frozen encoder/global head')
    payload=dict(schema_version=1,contract=contract,codec={k:v.detach().cpu() for k,v in codec.state_dict().items()},
        optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),rng=rng_state(),updates=updates,
        history=history,validation=validation,status=status)
    temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parent=ROOT/'outputs/ingolstadt_noincident_pretrain_20261008'
    parser.add_argument('--source-dir',type=Path,default=parent/'world_model_single_step')
    parser.add_argument('--output-dir',type=Path,default=parent/'decoder_calibration')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--gpu-memory-fraction',type=float,default=.10)
    parser.add_argument('--batch-size',type=int,default=64);parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--max-updates',type=int,default=20000);parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--validation-every',type=int,default=5000)
    parser.add_argument('--slow-mae-max',type=float,default=.5)
    parser.add_argument('--localized-slow-mae-max',type=float,default=1.)
    parser.add_argument('--localized-min-samples',type=int,default=20)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.batch_size,args.threads,args.max_updates,args.epochs,args.validation_every,args.localized_min_samples)<1:
        parser.error('budgets, batch size and sample counts must be positive')
    if not 0<args.gpu_memory_fraction<=1 or min(args.slow_mae_max,args.localized_slow_mae_max)<=0:
        parser.error('memory fraction must be in (0,1] and gates must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_file=args.output_dir/'calibration_latest.pt'
    if checkpoint_file.exists() and not args.resume: parser.error('output checkpoint exists; use --resume')
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)
    config=load_config(args.source_dir/'run_config.toml')
    calibration_config=replace(config.codec,occupied_speed_weight=1.,slow_speed_weight=1.,
                               slow_speed_mae_mps_max=args.slow_mae_max)
    prepared=validate_prepared_inputs(config);topology=load_traffic_topology(prepared)
    source_path=args.source_dir/'checkpoints/latest.pt'
    source_sha=file_hash(source_path)
    payload=torch.load(source_path,map_location='cpu',weights_only=False)
    if payload.get('input_hash')!=combined_input_hash(prepared):
        raise ValueError('source checkpoint prepared input hash differs from its run config')
    metrics=payload['training_state']['metrics']
    if metrics.get('world_model_initialization_epochs_completed')!=config.run.world_model_initialization_epochs:
        raise ValueError('decoder calibration requires completed pretrained world-model epochs')
    if metrics.get('online_episodes_completed',0)!=0:
        raise ValueError('decoder calibration source must be pretrained before online policy training')
    codec=SparseGraphCodec(adjacency=torch.tensor(topology.network.adjacency),positive_speed=config.codec.speed_positive)
    codec.load_state_dict({k.removeprefix('flow_projector.'):v for k,v in payload['training_state']['models']['world_model'].items()
                           if k.startswith('flow_projector.')})
    del payload
    freeze_encoder_for_decoder_probe(codec);codec.to(device)
    frozen_hash=frozen_state_hash(codec)
    optimizer=make_decoder_optimizer(codec,lr=calibration_config.decoder_pretrain_learning_rate)
    datasets={split:CompactPretrainDataset(config,prepared,topology,1,
        cache_manifest_path=(config.data.validation_cache_manifest or config.data.pretrain_cache_manifest)
            if split=='validation' else config.data.pretrain_cache_manifest,split=split)
        for split in ('train','validation')}
    verify_partition_separation(datasets['train'].manifest['entries'],datasets['validation'].manifest['entries'])
    entries=[dict(entry,frames=entry['windows']) for entry in datasets['train'].manifest['entries']]
    sampler=FrameSampler(entries,args.batch_size,config.run.seed)
    contract=dict(source_checkpoint=str(source_path.resolve()),source_checkpoint_sha256=source_sha,
        calibration_semantics='decoder_only_raw_transition_mean_persistent_local_context_speed_v2',
        slow_definition='current_count >= slow_min_vehicles - 1e-5 and forward_endpoint_mean_speed <= slow_speed_mps',
        local_context_definition='one to five slow roads in each complete road frame; only those slow cells scored',
        train_manifest_identity=datasets['train'].manifest['identity'],validation_manifest_identity=datasets['validation'].manifest['identity'],
        frozen_parameters_sha256=frozen_hash,codec_config=asdict(calibration_config),batch_size=args.batch_size,
        epochs=args.epochs,max_updates=args.max_updates,localized_slow_mae_max=args.localized_slow_mae_max,
        localized_min_samples=args.localized_min_samples,volume_scope='raw_and_previous_state_excluding_current_births')
    write_json(args.output_dir/'contract.json',dict(**contract,train_episodes=len(entries),
        validation_episodes=len(datasets['validation'].manifest['entries']),train_frames=sum(e['windows'] for e in entries),
        batches_per_full_epoch=sum((e['windows']+args.batch_size-1)//args.batch_size for e in entries)))
    updates,history,validation=0,[],[]
    if args.resume:
        updates,history,validation=restore_calibration(torch.load(checkpoint_file,map_location='cpu',weights_only=False),
                                                     contract,codec,optimizer,sampler)
    else:
        random.seed(config.run.seed);np.random.seed(config.run.seed);torch.manual_seed(config.run.seed)
    started=time.monotonic()
    def checkpoint(status):
        save_calibration(checkpoint_file,codec,optimizer,sampler,contract,updates,history,validation,status)
        write_json(args.output_dir/'progress.json',dict(status=status,updates=updates,epochs_completed=completed_epochs(sampler),
            raw_frames_seen=sampled_raw_frames(sampler),
            epoch_equivalent=sampled_raw_frames(sampler)/max(1,sum(e['windows'] for e in entries)),
            elapsed_seconds=time.monotonic()-started,latest_loss=history[-1] if history else None,
            latest_validation=validation[-1] if validation else None,production_modified=False))
    def evaluation():
        measured=validate(codec,datasets['validation'],topology.network,calibration_config,args.batch_size,device,
                          args.localized_min_samples,args.localized_slow_mae_max)
        validation.append(dict(update=updates,**measured))
        print(json.dumps(dict(stage='validation_complete',update=updates,passed=measured['passed'],
             slow_mae={kind:measured[kind]['codec_speed_mae_slow_mps'] for kind in ('raw','hybrid')})),flush=True)
        checkpoint('training')
    if not validation:
        checkpoint('validating_baseline')
        evaluation()
    previous_epoch=completed_epochs(sampler)
    while updates<args.max_updates and completed_epochs(sampler)<args.epochs:
        index,starts=sampler.next()
        arrays=datasets['train']._episode(index)
        pairs=[pair for pair in calibration_pairs(arrays,starts,topology.network.capacities) if pair is not None]
        flow=torch.cat([p[0] for p in pairs]).to(device);following=torch.cat([p[1] for p in pairs]).to(device)
        codec.train();optimizer.zero_grad(set_to_none=True)
        loss,_=codec_objective(codec,flow,following,topology.network,config.method,codec=calibration_config,decoder_only=True)
        if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite decoder loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in codec.parameters() if p.requires_grad],10.,error_if_nonfinite=True)
        optimizer.step();updates+=1
        if updates%100==0:
            history.append(dict(update=updates,loss=float(loss),gradient_norm=float(norm)))
            checkpoint('training');print(json.dumps(history[-1]),flush=True)
        epoch=completed_epochs(sampler)
        if updates%args.validation_every==0 or epoch!=previous_epoch:
            evaluation();previous_epoch=epoch
    if validation[-1]['update']!=updates: evaluation()
    if file_hash(source_path)!=source_sha: raise AssertionError('source checkpoint changed during calibration')
    checkpoint('candidate_ready' if validation[-1]['passed'] else 'candidate_unready')


if __name__=='__main__':main()
