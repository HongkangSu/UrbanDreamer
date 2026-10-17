"""Audit nine fresh GA30 episodes and append them to the verified36-run table."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math,sys,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology,NextHopCandidates
parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
P=Path((ROOT/'tmp/ga30_supplement_latest_path.txt').read_text().strip());manifest=json.loads((P/'manifest.json').read_text());previous=Path(manifest['previous_results'])
completion=json.loads((previous/'completion.json').read_text())
assert hashlib.sha256((previous/'verified_results.json').read_bytes()).hexdigest()==completion['artifacts']['verified_results.json']
old=json.loads((previous/'verified_results.json').read_text());assert old['status']=='completed' and old['completed_runs']==36 and old['verified_episodes']==108
physical=json.loads((previous/'paired_protocol_validation.json').read_text())['cities']
keys=['all_demand_censored_att_seconds','completion_rate','all_demand_censored_p90_seconds','all_demand_censored_p95_seconds','branch_freeflow_excess_seconds_per_started_vehicle']
table=list(old['table']);audit=list(old['audit']);new_audits=[];ga_stats=[];per_seed=list(csv.DictReader((previous/'per_seed.csv').open()))
for e in [x for x in manifest['entries'] if x['method']=='ga_routing']:
 result_path=Path(e['evaluation_output'])/'result.json'
 if args.partial and (not result_path.exists() or json.loads(result_path.read_text())['status']!='completed'):continue
 d=json.loads(result_path.read_text());assert d['status']=='completed' and len(d['evaluations'])==3
 p=d['protocol'];assert p['routing_mode']=='current_to_dest' and p['policy_decision_interval_seconds']==30 and p['deadline_seconds']==3810
 assert (p['population_size'],p['generations'],p['crossover_rate'],p['mutation_rate'],p['tournament_size'],p['cache_ttl_seconds'])==(100,50,.8,.1,5,300)
 assert p['input_hashes']==e['input_hashes']
 assert all(p['simulation'].get(k)==v for k,v in physical[e['city']]['physics'].items())
 cfg=load_config(e['config']);prepared=validate_prepared_inputs(cfg);assert prepared.input_hashes==e['input_hashes']
 candidate=NextHopCandidates(load_traffic_topology(prepared).network,traffic_model='meso');gaps={}
 reference=next(x for x in manifest['entries'] if x['city']==e['city'] and x['method']=='online_ac')
 refrows={x['seed']:x for x in json.loads((Path(reference['evaluation_output'])/'result.json').read_text())['evaluations']}
 for r in d['evaluations']:
  assert r['demand_sha256']==refrows[r['seed']]['demand_sha256']
  with np.load(r['archive'],allow_pickle=False) as z:
   summary=json.loads(z['summary_json'].tobytes());states=json.loads(z['states_json'].tobytes());events=json.loads(z['decisions_json'].tobytes());decisions=json.loads(z['policy_decisions_json'].tobytes())
  assert all(x['decision_time']%30==0 for x in decisions)
  assert len(decisions)==r['event_count']
  assert {x['decision_time'] for x in decisions}==set(r['policy_decision_times'])
  for x in decisions:
   route=x['route'];assert len(set(route))==len(route)
   assert route[0]==x['current_edge'] and route[-1]==x['destination_edge'] and route[1]==x['next_edge']
   assert all(b in candidate.network.adjacency_list[a] for a,b in zip(route,route[1:]))
  final=states[-1];end=min(final['time'],p['deadline_seconds']);vehicles=final['vehicles'];scheduled={str(v['vehicle_id']):v['departure_second'] for v in summary['metadata']['realized_demand']}
  assert len(scheduled)==len(vehicles)==r['all_demand_vehicle_count']
  durations=[max(0.,min(v['arrival_time'],end)-scheduled[str(v['vehicle_id'])]) if v['arrived'] else max(0.,end-scheduled[str(v['vehicle_id'])]) for v in vehicles]
  expected={keys[0]:np.mean(durations),keys[1]:sum(v['arrived'] for v in vehicles)/len(vehicles),keys[2]:np.percentile(durations,90),keys[3]:np.percentile(durations,95)}
  total=0.;counts=Counter((int(x['current_edge']),int(x['next_edge']),int(x['destination_edge'])) for x in events)
  for key,count in counts.items():
   if key not in gaps:
    source,chosen,dest=key;legal=candidate.candidates(source,dest);assert chosen in legal
    def cost(edge):
     length,speed=candidate.junction(source,edge)
     return length/max(speed,.1)+candidate.network.free_flow_travel_seconds[edge]+candidate.costs(dest)[edge]
    gaps[key]=max(0.,float(cost(chosen)-min(cost(a) for a in legal)))
   total+=count*gaps[key]
  expected[keys[4]]=total/max(sum(v['departed'] for v in vehicles),1)
  assert len(events)==r['diversion_decision_count']
  for k,v in expected.items():assert math.isclose(v,r[k],rel_tol=1e-9,abs_tol=1e-6),(e['city'],r['seed'],k,v,r[k])
  new_audits.append(dict(city=e['city'],method='ga_routing',seed=r['seed'],vehicle_count=len(vehicles),unstarted=sum(not v['departed'] for v in vehicles),turns=len(events),policy_decisions=len(decisions),all_policy_times_multiple_of30=True,verified=True))
  per_seed.append(dict(city=e['city'],method='ga_routing',seed=r['seed'],**{k:r[k] for k in keys}))
 stats={key:sum(r['policy_stats'].get(key,0) for r in d['evaluations']) for key in ('decision_count','optimize_calls','failed_searches','fallback_events','full_route_fallbacks','cache_hits','loop_erased_routes','loop_edges_removed','fallback_loop_erased_routes')}
 stats.update(city=e['city'],failed_search_fraction=stats['failed_searches']/max(stats['optimize_calls'],1),full_route_fallback_fraction=stats['full_route_fallbacks']/max(stats['decision_count'],1));ga_stats.append(stats)
 row=dict(city=e['city'],method='ga_routing',checkpoint_episode=None)
 for k in keys:
  values=[r[k] for r in d['evaluations']];row[k]=float(np.mean(values));row[k+'_std']=float(np.std(values))
 table.append(row)
 print(e['city'],*[round(row[k]*(100 if k=='completion_rate' else 1),3) for k in keys],flush=True)
if not args.partial:assert len(table)==39 and len(new_audits)==9
status='completed' if len(table)==39 else 'partial'
report=dict(status=status,completed_runs=len(table),expected_runs=39,verified_episodes=len(audit)+len(new_audits),previous_results=str(previous),previous_verified_sha256=completion['artifacts']['verified_results.json'],audit=audit+new_audits,table=table)
(P/'verified_results.json').write_text(json.dumps(report,indent=2)+'\n')
(P/'ga_validation.json').write_text(json.dumps(dict(status=status,audited_episodes=len(new_audits),audit=new_audits,search_stats=ga_stats,clock_seconds=30,route_space='Complete legal paths, with original GA search operators; first-legal plus shortest-suffix fallback on failure; submitted closed loops erased while raw GA caches retained.'),indent=2)+'\n')
for name,rows in [('performance',table),('per_seed',per_seed)]:
 with (P/(name+'.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(dict(status=status,runs=len(table),episodes=len(audit)+len(new_audits),new_ga_episodes=len(new_audits))))
