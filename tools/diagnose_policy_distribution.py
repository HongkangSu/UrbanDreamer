"""CPU-only fixed-real-state policy distribution and realized revisit audit."""
from pathlib import Path
import sys,json,math
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology,NextHopCandidates
from urban_dreamer.learning.observations import build_actor_observations
from urban_dreamer.models.policy import NextHopActor

def main():
 torch.set_num_threads(2)
 run=ROOT/'outputs/ingolstadt_online_libsumo_20261004'
 out=ROOT/'outputs/convergence_diagnosis_20261005/policy';out.mkdir(parents=True,exist_ok=True)
 cfg=load_config(run/'run_config.toml');top=load_traffic_topology(validate_prepared_inputs(cfg));candidates=NextHopCandidates(top.network)
 fs=[];ms=[];bad=[]; provenance=[];revisits=[]
 for ep in [50,100,160,240,320,400,450]:
  path=run/f'episodes/online/episode_{ep:06d}.npz'
  if not path.exists():continue
  with np.load(path) as a:
   states=json.loads(a['states_json'].tobytes());events=json.loads(a['decisions_json'].tobytes());summary=json.loads(a['summary_json'].tobytes());flow=a['flow']
  seen={};repeated=0;rv=set()
  for event in events:
   vid=event['vehicle_id'];edge=event['current_edge'];prior=seen.setdefault(vid,set())
   if edge in prior:repeated+=1;rv.add(vid)
   prior.add(edge)
  revisits.append(dict(episode=ep,events=len(events),revisit_fraction=repeated/max(1,len(events)),revisiting_vehicle_fraction=len(rv)/max(1,len(seen)),metrics=summary['metrics']))
  for index in range(5,len(states)-1,10):
   snapshot=states[index];selected=[s for s in snapshot['vehicles'] if s['departed'] and not s['arrived'] and not s.get('in_junction',False) and s['current_edge']!=s['destination_edge'] and len(candidates.candidates(s['current_edge'],s['destination_edge']))>1][:100]
   if not selected:continue
   obs=build_actor_observations(policy_edge_features=flow[index][:,[2,1]],vehicle_states=selected,candidates=candidates,episode_progress=snapshot['time']/3810,active_vehicle_fraction=snapshot['active_vehicle_count']/max(1,len(snapshot['vehicles'])))
   nonprogress=np.zeros(obs.mask.shape,bool)
   for i,s in enumerate(selected):
    cost=candidates.costs(s['destination_edge']);nonprogress[i,obs.mask[i]]=cost[obs.edge_indices[i,obs.mask[i]]]>=cost[s['current_edge']]-1e-8
   fs.append(obs.features);ms.append(obs.mask);bad.append(nonprogress);provenance.append(dict(episode=ep,time=snapshot['time'],states=len(selected)))
 features=torch.from_numpy(np.concatenate(fs));mask=torch.from_numpy(np.concatenate(ms));nonprogress=torch.from_numpy(np.concatenate(bad));cost=torch.expm1(features[...,4]*math.log1p(3600)).masked_fill(~mask,0)
 checkpoints={'pretrain':ROOT/'outputs/ingolstadt_bf16_pretrain_20261002/checkpoints/latest.pt','after1':ROOT/'outputs/ingolstadt_online_pilot_20261003/interim_after1/checkpoints/latest.pt','after160':ROOT/'outputs/imagination_40s_20261004/checkpoint_before_parallel.pt','latest':run/'checkpoints/latest.pt'}
 rows=[]
 for name,path in checkpoints.items():
  checkpoint=torch.load(path,map_location='cpu',weights_only=False);actor=NextHopActor();actor.load_state_dict(checkpoint['training_state']['models']['actor']);actor.eval()
  with torch.no_grad():
   probs=torch.cat([actor.forward_logits(f,m).softmax(-1) for f,m in zip(features.split(512),mask.split(512))]);entropy=-(probs*probs.clamp_min(1e-30).log()).sum(-1);top2=probs.topk(2).values;greedy=probs.argmax(-1);best=cost.masked_fill(~mask,torch.inf).amin(-1);isbest=(cost-best[:,None]).abs()<1e-3
   row=dict(name=name,checkpoint=str(path),completed_online=checkpoint['next_unit'],states=len(features),entropy=float(entropy.mean()),normalized_entropy=float((entropy/mask.sum(-1).float().log()).mean()),max_probability=float(top2[:,0].mean()),top2_margin=float((top2[:,0]-top2[:,1]).mean()),shortest_cost_action_mass=float((probs*isbest).sum(-1).mean()),nonprogress_action_mass=float((probs*nonprogress).sum(-1).mean()),argmax_nonprogress_fraction=float(nonprogress.gather(1,greedy[:,None]).float().mean()),expected_excess_freeflow_seconds=float(((probs*cost).sum(-1)-best).mean()),argmax_excess_freeflow_seconds=float((cost.gather(1,greedy[:,None]).squeeze(1)-best).mean()))
  rows.append(row);print(json.dumps(row),flush=True);del checkpoint
 result=dict(protocol='Identical archived real boundary observations for every actor, CPU FP32, no optimizer. Active external-edge branch states; these are not necessarily observed decision instants. Nonprogress means downstream free-flow cost fails to decrease; not a proof of a cycle. No shortest-path prior applied (mainline original behavior).',provenance=provenance,policies=rows,realized_revisits=revisits)
 (out/'results.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'fixed_observations.npz',features=features.numpy(),mask=mask.numpy(),nonprogress=nonprogress.numpy())
if __name__=='__main__':main()
