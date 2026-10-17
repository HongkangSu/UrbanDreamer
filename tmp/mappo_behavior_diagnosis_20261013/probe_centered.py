import os, json, hashlib
from pathlib import Path
import numpy as np,torch
from torch.distributions import Categorical
from urban_dreamer.models.policy import NextHopActor
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology,NextHopCandidates
from urban_dreamer.learning.observations import build_actor_observations
p=Path('outputs/city_baselines_two_city_fresh2500_20261011/coulommiers/mappo')
torch.set_num_threads(1)
s=torch.load(p/'latest.pt',map_location='cpu',weights_only=False)
a=NextHopActor();a.load_state_dict(s['method_state']['actor']);a.eval()
c=load_config(s['args']['config']); t=load_traffic_topology(validate_prepared_inputs(c));cand=NextHopCandidates(t.network,traffic_model=c.simulation.traffic_model)
x=np.load(p/'episodes/train/episode_000533.npz',allow_pickle=False)
d=json.loads(x['decisions_json'].tobytes());states=json.loads(x['states_json'].tobytes());flow=x['flow'];counts={}
fs=[];ms=[];ac=[];ds=[]
# Actual recorded event current/destination/next edge, actual frozen window traffic;
# Remaining distance and live active fraction are not archived per event.
for w in range(127):
 rows=[r for r in d if r['window_index']==w]
 if not rows:continue
 vs=[dict(current_edge=r['current_edge'],destination_edge=r['destination_edge'],remaining_distance=0.) for r in rows]
 o=build_actor_observations(policy_edge_features=flow[w][:,[2,1]],vehicle_states=vs,candidates=cand,episode_progress=float(states[w]['time'])/3810,active_vehicle_fraction=states[w]['active_vehicle_count']/max(states[w]['started_vehicle_count'],1))
 fs.append(o.features);ms.append(o.mask)
 ac.extend([int(np.flatnonzero(o.edge_indices[i]==r['next_edge'])[0]) for i,r in enumerate(rows)]);ds+=rows
features=torch.from_numpy(np.concatenate(fs));mask=torch.from_numpy(np.concatenate(ms));actions=torch.tensor(ac)
# Retain only first 12000 events to bound runtime.
features,mask,actions=features,mask,actions
@torch.no_grad()
def calc(actor,device,chunk,dtype=torch.float32):
 logits=torch.cat([actor.forward_logits(features[j:j+chunk].to(device=device,dtype=dtype),mask[j:j+chunk].to(device)) .cpu() for j in range(0,len(features),chunk)])
 dist=Categorical(logits=logits)
 return logits,dist.log_prob(actions),dist.probs
import types, torch.nn.functional as F
@torch.no_grad()
def centered(self,features,mask):
    hidden=self.candidate_scorer(features)
    first=mask.to(torch.int64).argmax(-1)
    reference=hidden[torch.arange(len(hidden),device=hidden.device),first]
    logits=F.linear(hidden-reference[:,None,:],self.policy_head.weight,None).squeeze(-1)
    return logits.masked_fill(~mask,-torch.inf)
v={}
a.forward_logits=types.MethodType(centered,a)
for name,chunk in [('cpu512',512),('cpu1',1)]:v[name]=calc(a,'cpu',chunk)
ad=NextHopActor().double();ad.load_state_dict(s['method_state']['actor']);ad.eval();ad.forward_logits=types.MethodType(centered,ad);v['cpu64']=calc(ad,'cpu',512,torch.float64)
if torch.cuda.is_available():
 torch.cuda.set_per_process_memory_fraction(.025,0)
 ag=NextHopActor().to('cuda');ag.load_state_dict(s['method_state']['actor']);ag.eval();ag.forward_logits=types.MethodType(centered,ag);v['gpu512']=calc(ag,'cuda',512)
result={'checkpoint_episode':s['episode'],'checkpoint_sha256':hashlib.sha256((p/'latest.pt').read_bytes()).hexdigest(),'n':len(features),'important':'Inputs derived from actual failed archive but missing per-event remaining distance/live active fraction; NOT exact failed rows','comparisons':{}}
for name,other in v.items():
 l,lp,pr=v['cpu1'];ll,lpp,ppr=other;err=(lp-lpp).abs();tol=1e-4+1e-5*lp.abs();bad=err>tol;legal=mask
 result['comparisons'][name]={'logprob_max_abs':err.max().item(),'bad_count':int(bad.sum()),'prob_max_abs':(pr-ppr).abs().max().item(),'legal_logits_range':[ll[legal].min().item(),ll[legal].max().item()],'legal_logits_max_absdiff':(l[legal]-ll[legal]).abs().max().item(),'top':[{'idx':int(i),'error':err[i].item(),'lp_cpu1':lp[i].item(),'lp_other':lpp[i].item(),'logits_cpu1':l[i][mask[i]].tolist(),'logits_other':ll[i][mask[i]].tolist(),'prob_cpu1':pr[i][mask[i]].tolist(),'prob_other':ppr[i][mask[i]].tolist(),'event':ds[int(i)]} for i in err.topk(5).indices]}
Path('tmp/mappo_behavior_diagnosis_20261013/probe_centered.json').write_text(json.dumps(result,indent=2)); print(json.dumps({k:{kk:vv for kk,vv in z.items() if kk!='top'} for k,z in result['comparisons'].items()},indent=2))
