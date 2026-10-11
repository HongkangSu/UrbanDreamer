"""Read-only, paired online world-model accuracy and cohort-input mismatch audit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urban_dreamer.learning.episode_archive import read_episode_archive, archive_to_replay_episode
from urban_dreamer.learning.replay import ReplayBuffer
from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.models.world_model import UrbanDreamerWorldModel
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config


def metrics(pred, target, full, previous, capacity, limits):
    p, y, f, x = [v.detach().cpu().numpy() for v in (pred, target, full, previous)]
    count, truth, fullcount = p[:, 2] * capacity, y[:, 2] * capacity, f[:, 2] * capacity
    occupied = fullcount > 0
    speed_error = (p[:, 1] - y[:, 1]) * limits
    return dict(count_mae_all=float(abs(count-truth).mean()), count_mae_occupied=float(abs(count-truth)[occupied].mean()) if occupied.any() else 0.,
                predicted_total=float(count.sum()), target_cohort_total=float(truth.sum()), full_total=float(fullcount.sum()),
                omitted_current_births=float((fullcount-truth).sum()), count_bias=float((count-truth).sum()),
                speed_mae_occupied_mps=float(abs(speed_error)[occupied].mean()) if occupied.any() else 0.,
                speed_bias_occupied_mps=float(speed_error[occupied].mean()) if occupied.any() else 0.,
                speed_mass_weighted_mae_mps=float((abs(speed_error)*fullcount).sum()/max(fullcount.sum(),1)),
                persistence_count_mae=float(abs(x[:,2]*capacity-truth).mean()),
                persistence_speed_mae_occupied_mps=float(abs((x[:,1]-y[:,1])*limits)[occupied].mean()) if occupied.any() else 0.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--online-checkpoint',type=Path,required=True)
    parser.add_argument('--online-config',type=Path,default=ROOT/'outputs/ingolstadt_online_libsumo_20261004/run_config.toml')
    parser.add_argument('--pretrain-checkpoint',type=Path,default=ROOT/'outputs/ingolstadt_bf16_pretrain_20261002/checkpoints/latest.pt')
    parser.add_argument('--archive-root',type=Path,default=ROOT/'outputs/ingolstadt_online_libsumo_20261004/episodes/online')
    parser.add_argument('--episodes',type=int,nargs='+',default=[50,160,400,452])
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/convergence_diagnosis_20261005/world')
    parser.add_argument('--device',default='cuda:1'); parser.add_argument('--dataset-only',action='store_true'); args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    config=load_config(args.online_config); prepared=validate_prepared_inputs(config); topology=load_traffic_topology(prepared)
    capacity=np.maximum(topology.network.capacities,1.);limits=topology.network.speed_limits_mps
    batches=[]; dataset=[]
    for episode_index in args.episodes:
        path=args.archive_root/f'episode_{episode_index:06d}.npz'
        data=read_episode_archive(path,expected_session_protocol='event')
        episode=archive_to_replay_episode(data)
        replay=ReplayBuffer(8,max_episodes=1)
        replay.append_episode(episode['transitions'],total_travel_minutes=episode['total_travel_minutes'],
                              total_vehicle_count=episode['total_vehicle_count'],arrival_times=episode['arrival_times'])
        starts=[v for v in (4,20,40,70) if v+8 <len(episode['transitions'])]
        batch=replay._batch_from_picks([(0,start) for start in starts]); scoped=prepare_transition_targets(batch,capacities=capacity)
        batches.append((episode_index,starts,batch,scoped))
        omitted=((batch.flow[:,1:,:,2]-scoped.transition_target_flow[...,2])*torch.tensor(capacity)).sum(-1)
        dataset.append(dict(episode=episode_index,path=str(path),windows=len(episode['transitions']),starts=starts,
             population=episode['total_vehicle_count'],completed=episode['completed'],mean_omitted_births=float(omitted.mean()),
             snapshots_seconds=sorted(set(np.diff([s['time'] for s in data['states']]).tolist())),
             all_fields=list(data.keys())))
    (args.output/'dataset.json').write_text(json.dumps(dataset,indent=2)+'\n')
    if args.dataset_only:
        print(json.dumps(dataset,indent=2));return
    device=torch.device(args.device)
    if device.type=='cuda': torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.035,device)
    rows=[];model_meta=[];started=time.perf_counter()
    for model_name,path in [('pretrained',args.pretrain_checkpoint),('online',args.online_checkpoint)]:
        payload=torch.load(path,map_location='cpu',weights_only=False)
        model=UrbanDreamerWorldModel(adjacency=torch.tensor(topology.network.adjacency,dtype=torch.float32),
             static_features=torch.tensor(topology.network.static_features,dtype=torch.float32),
             low_frequency_rank=config.method.low_frequency_rank,prior_noise_scale=config.method.prior_noise_scale,
             event_protocol=True,codec_speed_positive=config.codec.speed_positive).to(device)
        model.load_state_dict(payload['training_state']['models']['world_model']);model.eval().requires_grad_(False)
        model.dynamics.set_pretraining_execution_profile(space_chunk=16,time_chunk=65536,mlp_chunk=16384,activation_checkpointing=False)
        model_meta.append(dict(name=model_name,path=str(path),next_unit=payload['next_unit'],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        del payload
        with torch.no_grad():
            for episode_index,starts,batch,scoped in batches:
                autoregressive=model.encode(batch.flow[:,0].to(device))
                for step in range(8):
                    trueinput=model.encode(batch.flow[:,step].to(device))
                    torch.manual_seed(123400+episode_index*8+step)
                    noise=torch.randn(len(starts),model.dynamics.graph_prior.rank,64,device=device)
                    args_transition=dict(noise=noise,steps=config.method.fm_steps,solver=config.method.fm_solver,
                         picard_iterations=config.method.picard_iterations,action_events=batch.action_events.for_step(step).to(device))
                    action=batch.action[:,step].to(device)
                    teacher=model.transition(trueinput,action,**args_transition)
                    autoregressive=model.transition(autoregressive,action,**args_transition)
                    for kind,latent in [('teacher_forced',teacher),('autoregressive_recorded_actions',autoregressive)]:
                        decoded=model.decode(latent)
                        for index,start in enumerate(starts):
                            rows.append(dict(model=model_name,episode=episode_index,start=start,step=step+1,kind=kind,
                                **metrics(decoded[index],scoped.transition_target_flow[index,step],batch.flow[index,step+1],
                                    batch.flow[index,step],capacity,limits)))
                print(json.dumps(dict(model=model_name,episode=episode_index,elapsed=time.perf_counter()-started)),flush=True)
        del model
    groups=[]
    for name in ('pretrained','online'):
        for kind in ('teacher_forced','autoregressive_recorded_actions'):
            for step in range(1,9):
                selected=[r for r in rows if r['model']==name and r['kind']==kind and r['step']==step]
                groups.append(dict(model=name,kind=kind,step=step,conditions=len(selected),
                    **{key:float(np.mean([r[key] for r in selected])) for key in selected[0] if key not in ('model','kind','step','episode','start')}))
    result=dict(models=model_meta,dataset=dataset,rows=rows,aggregates=groups,elapsed_seconds=time.perf_counter()-started,
         limits=['Existing online archives are retrospective and may have been used in online WM fine tuning; not untouched heldout data.',
                 'Autoregressive diagnostic holds recorded real future actions fixed; it is not a causally valid policy rollout.',
                 'Targets preserve full-background speed and subtract only current-window births from endpoint volume.',
                 'A single common noise sample per paired condition; do not interpret differences as statistically significant.',
                 'Snapshots occur every30seconds; action event timestamps alone cannot reconstruct 5second physical training labels.'])
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(groups,indent=2))
if __name__=='__main__': main()
