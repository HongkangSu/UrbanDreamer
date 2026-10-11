"""Retrospective seed-cohort value calibration; never used as imagination labels."""
from pathlib import Path
import json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.learning.episode_archive import read_episode_archive,observed_arrival_times
from urban_dreamer.models.world_model import UrbanDreamerWorldModel
from urban_dreamer.models.policy import CentralizedLatentValueCritic

def main():
 torch.set_num_threads(2)
 base=ROOT/'outputs/noincident_decline_diagnosis_20261008';run=ROOT/'outputs/ingolstadt_online_meso_noincident_noprior_20261007'
 c=load_config(base/'frozen/run_config.toml');top=load_traffic_topology(validate_prepared_inputs(c));n=top.network
 p=torch.load(base/'frozen/checkpoints/latest.pt',map_location='cpu',weights_only=True)
 wm=UrbanDreamerWorldModel(adjacency=torch.tensor(n.adjacency),static_features=torch.tensor(n.static_features),low_frequency_rank=c.method.low_frequency_rank,prior_noise_scale=c.method.prior_noise_scale,event_protocol=True,codec_speed_positive=c.codec.speed_positive)
 wm.load_state_dict(p['training_state']['models']['world_model']);v=CentralizedLatentValueCritic();v.load_state_dict(p['training_state']['models']['latent_critic']);wm.eval().requires_grad_(False);v.eval().requires_grad_(False)
 rows=[]
 for episode in [74,124,424,484]:
  a=read_episode_archive(run/f'episodes/online/episode_{episode:06d}.npz',expected_session_protocol='event');arrivals=observed_arrival_times(a);N=a['metadata']['total_vehicle_count'];deadline=3810.
  for i in [10,40,70,100]:
   if i>=len(a['states']):continue
   s=a['states'][i];time=s['time'];cohort=[r for r in s['vehicles'] if r['departed'] and not r['arrived']]
   if not cohort:continue
   cost=sum(max(0.,min(arrivals.get(str(r['vehicle_id']),deadline),deadline)-time) for r in cohort)/(N*60)
   with torch.no_grad():pred=float(v(wm.encode(torch.tensor(a['flow'][i])[None]),torch.tensor([[time/deadline,len(cohort)/N]],dtype=torch.float32)).item())
   rows.append(dict(episode=episode+1,time=time,population=N,seed_cohort=len(cohort),cohort_failures=sum(str(r['vehicle_id']) not in arrivals for r in cohort),predicted_value=pred,actual_seed_cohort_return=-cost,error=pred+cost))
 out=base/'value';out.mkdir(exist_ok=True)
 report=dict(frozen_episode=p['next_unit'],rows=rows,mean_absolute_error=float(np.mean([abs(r['error']) for r in rows])),signed_bias=float(np.mean([r['error'] for r in rows])),note='Retrospective current actor V on actual seed traffic; future route choices are archived behavior, not current-policy Q labels. Cohort is only already-departed vehicles at seed. No future births or individual timeout penalties in return. This calibrates factual behavior and cannot prove causal V^pi error.')
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
