"""Independently audit per-vehicle durations and destination-specific turn costs."""
from pathlib import Path
import csv,gzip,json,math,sys,hashlib
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology,NextHopCandidates
P=Path((ROOT/'tmp/full_reevaluation_latest_path.txt').read_text().strip());manifest=json.loads((P/'manifest.json').read_text());table=[];per_seed=[];paired={};cache={};networks={};audit=[]
audit_path=P/'audit_cache.json'
audit_cache=json.loads(audit_path.read_text()) if audit_path.exists() else {}
audit_code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
KEYS=['all_demand_censored_att_seconds','completion_rate','all_demand_censored_p90_seconds','all_demand_censored_p95_seconds','branch_freeflow_excess_seconds_per_started_vehicle']
for e in manifest['entries']:
 f=Path(e['evaluation_output'])/'result.json'
 if not f.exists():continue
 d=json.loads(f.read_text())
 if d['status']!='completed':continue
 rs=d['evaluations'];assert len(rs)==3 and {r['seed'] for r in rs}==set(manifest['seeds'])
 if e['city'] not in networks:
  c=load_config(e['config']);top=load_traffic_topology(validate_prepared_inputs(c));networks[e['city']]=NextHopCandidates(top.network,traffic_model='meso')
 candidate=networks[e['city']]
 for r in rs:
  identity=(e['city'],r['seed']);h=r.get('demand_sha256',r.get('demand_hash'));assert h
  assert paired.setdefault(identity,h)==h,('unpaired demand',e['city'],e['method'],r['seed'])
  cache_id=f"{e['city']}/{e['method']}/{r['seed']}"
  evidence=Path(r['archive']) if e['method']=='urban_dreamer' else Path(e['evaluation_output'])/f"audit_{r['seed']}.json.gz"
  with evidence.open('rb') as stream: evidence_hash=hashlib.file_digest(stream,'sha256').hexdigest()
  fingerprint=hashlib.sha256((audit_code_hash+evidence_hash+json.dumps(r,sort_keys=True)+json.dumps(e['input_hashes'],sort_keys=True)).encode()).hexdigest()
  previous=audit_cache.get(cache_id,{})
  if previous.get('fingerprint')==fingerprint:
   audit.append(previous['audit'])
   per_seed.append(dict(city=e['city'],method=e['method'],seed=r['seed'],**{k:r[k] for k in KEYS}))
   continue
  if e['method']=='urban_dreamer':
   with np.load(r['archive'],allow_pickle=False) as z:
    summary=json.loads(z['summary_json'].tobytes());states=json.loads(z['states_json'].tobytes());events=json.loads(z['decisions_json'].tobytes())
   metadata=summary['metadata'];final=states[-1]['vehicles'];end=states[-1]['time'];turns=[dict(x,count=1) for x in events]
  else:
   with gzip.open(Path(e['evaluation_output'])/f"audit_{r['seed']}.json.gz",'rt') as f2:data=json.load(f2)
   metadata=data['metadata'];final=data['final_vehicles'];end=data['end_time'];turns=data['realized_turn_counts']
  scheduled={str(v['vehicle_id']):float(v.get('departure_second')) for v in metadata.get('realized_demand',metadata.get('demand_realization',[]))}
  assert len(scheduled)==len(final)
  values=[max(0.,min(float(v['arrival_time']),end)-scheduled[str(v['vehicle_id'])]) if v['arrived'] else max(0.,end-scheduled[str(v['vehicle_id'])]) for v in final]
  checks={KEYS[0]:np.mean(values),KEYS[1]:sum(v['arrived'] for v in final)/len(final),KEYS[2]:np.percentile(values,90),KEYS[3]:np.percentile(values,95)}
  total=0.;count=0
  for event in turns:
   source,chosen,dest=(int(event[k]) for k in ('current_edge','next_edge','destination_edge'));key=(e['city'],source,chosen,dest)
   if key not in cache:
    legal=candidate.candidates(source,dest);assert chosen in legal
    def ff(edge):
     length,speed=candidate.junction(source,edge)
     return length/max(speed,.1)+candidate.network.free_flow_travel_seconds[edge]+candidate.costs(dest)[edge]
    cache[key]=max(0.,float(ff(chosen)-min(ff(x) for x in legal)))
   total+=cache[key]*event['count'];count+=event['count']
  started=sum(v['departed'] for v in final);checks[KEYS[4]]=total/max(started,1)
  assert count==r['diversion_decision_count']
  for k,v in checks.items():assert math.isclose(v,r[k],rel_tol=1e-9,abs_tol=1e-6),(e['city'],e['method'],r['seed'],k,v,r[k])
  if e['method']=='shortest_ref':assert checks[KEYS[4]]<1e-6
  audit.append(dict(city=e['city'],method=e['method'],seed=r['seed'],vehicle_count=len(final),unstarted=len(final)-started,turns=count,verified=True))
  audit_cache[cache_id]=dict(fingerprint=fingerprint,audit=audit[-1])
  temporary=audit_path.with_suffix('.tmp');temporary.write_text(json.dumps(audit_cache,indent=2)+'\n');temporary.replace(audit_path)
  per_seed.append(dict(city=e['city'],method=e['method'],seed=r['seed'],**{k:r[k] for k in KEYS}))
 ep=d['protocol'].get('checkpoint_episode')
 row=dict(city=e['city'],method=e['method'],checkpoint_episode=ep)
 for k in KEYS:
  vals=[r[k] for r in rs];row[k]=float(np.mean(vals));row[k+'_std']=float(np.std(vals))
 table.append(row)
for name,rows in [('performance',table),('per_seed',per_seed)]:
 if rows:
  with (P/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report=dict(status='completed' if len(table)==36 else 'partial',completed_runs=len(table),expected_runs=36,verified_episodes=len(audit),audit=audit,table=table)
(P/'verified_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],runs=len(table),episodes=len(audit))))
for r in table: print(r['city'],r['method'],r['checkpoint_episode'],*[round(r[k]*(100 if k=='completion_rate' else 1),2) for k in KEYS])
