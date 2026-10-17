"""Frozen-policy CPU evaluation, with compact auditable final vehicles and realized turns."""
import argparse,functools,gzip,hashlib,importlib.util,json,sys,time,os
from pathlib import Path
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--city',required=True);p.add_argument('--method',required=True);a=p.parse_args()
meta=json.loads(a.manifest.read_text());e=next(x for x in meta['entries'] if x['city']==a.city and x['method']==a.method)
out=Path(e['evaluation_output']);source=Path(e['source']);sys.path.insert(0,str(source))
route_cache=a.manifest.parent/'route_cache'/a.city
if route_cache.exists():os.environ['URB_ROUTE_CACHE_DIR']=str(route_cache)
import numpy as np
import torch
torch.set_num_threads(1)
from urban_dreamer.learning.baselines.common import EpisodeRunner,evaluation_rng
from urban_dreamer.learning.sumo_sampling import resolve_session_factory
from urban_dreamer.models.policy import NextHopActor,set_shortest_path_prior
import urban_dreamer.learning.episode_archive as archive_module
class CompactAudit:
 def __init__(self,metadata):self.metadata=metadata;self.turns=Counter()
 def start(self,state):self.initial=state['vehicles'];self.dest={str(v['vehicle_id']):int(v['destination_edge']) for v in self.initial};self.final=state
 def append(self,result):
  self.final=result.next_state;turns=getattr(result,'conditioning_decisions',None)
  for d in result.decisions if turns is None else turns:self.turns[(int(d.current_edge),int(d.next_edge),self.dest[str(d.vehicle_id)])]+=1
 def save(self,path,**kw):
  data=dict(schema='compact_evaluation_audit_v1',metadata=self.metadata,initial_vehicles=self.initial,final_vehicles=self.final['vehicles'],end_time=self.final['time'],realized_turn_counts=[dict(current_edge=k[0],next_edge=k[1],destination_edge=k[2],count=v) for k,v in sorted(self.turns.items())],**kw)
  with gzip.open(str(path)+'.json.gz','wt') as f:json.dump(data,f,allow_nan=False)
archive_module.EpisodeArchive=CompactAudit
reference=a.method in ('shortest_ref','dynamic_shortest_ref')
checkpoint=Path(e['checkpoint']) if e.get('checkpoint') else Path(next(x['checkpoint'] for x in meta['entries'] if x['city']==a.city and x['method']=='mappo'))
saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
args=argparse.Namespace(**saved['args']);args.device='cpu';args.threads=1;args.config=Path(e['config']);args.output_dir=out;args.seed=42
factory=resolve_session_factory('event',1.,'libsumo')
method=None
if a.method=='online_ac':
 spec=importlib.util.spec_from_file_location('frozen_ac',source/'tools/train_toy_actor_critic.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 runner=module.RealActorCritic(args);runner.actor.load_state_dict(saved['actor'],strict=True);runner.actor.eval();runner.actor.requires_grad_(False)
 def one(seed):return runner.run_episode(policy='deterministic',seed=seed,archive_path=out/f'audit_{seed}',archive_metadata=dict(phase='evaluation'))[1]
else:
 runner=EpisodeRunner(args,session_factory=factory)
 if reference:selector='shortest' if a.method=='shortest_ref' else 'dynamic_shortest';hook=None
 elif a.method in ('ippo','mappo','mbpo','dreamer'):
  actor=NextHopActor();actor.score_gauge=('pre_head_legal_reference_v1' if a.method in ('mbpo','dreamer') else getattr(args,'score_gauge','legacy'));actor.load_state_dict(saved['method_state']['actor'],strict=True);actor.eval();actor.requires_grad_(False)
  set_shortest_path_prior(actor,float(getattr(args,'sp_prior_weight',0.)),float(getattr(args,'sp_prior_temperature',30.)))
  def selector(vehicles,obs,flow,now):
   with torch.no_grad():sel=actor.sample_action(torch.as_tensor(obs.features),torch.as_tensor(obs.mask),deterministic=True)
   idx,ent=sel.indices_and_entropy();return idx,[dict(entropy=x) for x in ent]
  hook=None
 else:
  from urban_dreamer.learning.baselines import build_method
  method=build_method(a.method,args,runner)
  if a.method in ('gnac','iql'):
   method.actor.load_state_dict(saved['method_state']['actor'],strict=True);method.actor.eval()
  else:method.load_state_dict(saved['method_state'])
  selector=functools.partial(method.select,explore=False);hook=getattr(method,'begin_episode',None)
 def one(seed):return runner.run_episode(selector,seed=seed,label='deterministic',on_demand=hook,archive_path=out/f'audit_{seed}',archive_metadata=dict(phase='evaluation'))[2]
assert runner.prepared.input_hashes==e['input_hashes'], 'network input mismatch'
assert getattr(runner.config.simulation,'routing_mode')=='current_to_dest'
assert runner.deadline==3810
with checkpoint.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
protocol=dict(actor_score_gauge=getattr(locals().get('actor',getattr(method,'actor',None)),'score_gauge',None),entry=e,checkpoint_sha256=None if reference else digest,checkpoint_episode=None if reference else int(saved['episode']),demand_seed=42,seeds=meta['seeds'],policy='argmax',device='cpu',deadline_seconds=runner.deadline,input_hashes=runner.prepared.input_hashes,restoration='Inference-required weights only for feedforward actors; frozen method restore for MAMBA/DTA. No optimizer steps.',source=str(source))
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
rows=[];start=time.monotonic()
if (out/'result.json').exists():
 previous=json.loads((out/'result.json').read_text())
 assert previous['protocol']['checkpoint_sha256']==protocol['checkpoint_sha256']
 assert previous['protocol'].get('actor_score_gauge')==protocol.get('actor_score_gauge')
 rows=previous['evaluations']
for seed in meta['seeds']:
 if any(r['seed']==seed for r in rows):continue
 with evaluation_rng(seed,method):r=one(seed)
 r['seed']=seed;r['demand_sha256']=r.get('demand_sha256',r.get('demand_realization_sha256'))
 assert all(np.isfinite(r[k]) for k in ('all_demand_censored_att_seconds','all_demand_censored_p90_seconds','all_demand_censored_p95_seconds','branch_freeflow_excess_seconds_per_started_vehicle'))
 rows.append(r)
 (out/'result.json').write_text(json.dumps(dict(status='completed' if len(rows)==len(meta['seeds']) else 'running',protocol=protocol,evaluations=rows,elapsed_seconds=time.monotonic()-start),indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(city=a.city,method=a.method,seed=seed,ATT=r['all_demand_censored_att_seconds'],completion=r['completion_rate'],cost=r['branch_freeflow_excess_seconds_per_started_vehicle'])),flush=True)
