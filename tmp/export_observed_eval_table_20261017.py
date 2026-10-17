import csv,json,statistics
from pathlib import Path
P=Path('outputs/city_baselines_static_round2_2000_20261015_173045/existing_results_20261017_053123')
s=json.loads((P/'results_summary.json').read_text())
rows=[]
for run in s['runs']:
 ep=int(run['latest_evaluation']['episode'])
 source=P/f"{run['city']}_{run['method']}_metrics.jsonl"
 selected={}
 for line in source.open():
  r=json.loads(line)
  if r['phase']=='evaluation' and r['policy']=='deterministic' and r['episode']==ep and r['seed'] in (10042,10043,10044):selected[r['seed']]=r
 assert len(selected)==3
 out=dict(city=run['city'],method=run['method'],evaluation_episode=ep,seed_count=3)
 for name,key,scale in [('ATT_seconds','observed_mean_seconds',1),('Completion_percent','completion_rate',100),('P90_seconds','p90_travel_seconds',1),('P95_seconds','p95_travel_seconds',1),('observed_ATT_seconds','observed_mean_seconds',1)]:
  values=[r[key]*scale for r in selected.values()]
  out[name]=statistics.mean(values);out[name+'_std']=statistics.pstdev(values)
 out['diversion_cost']=None
 out['never_departed_total_3seeds']=sum(r['never_departed'] for r in selected.values())
 out['demand_total_3seeds']=sum(r['vehicle_count'] for r in selected.values())
 out['started_total_3seeds']=sum(r['started_vehicle_count'] for r in selected.values())
 for r in selected.values():
  assert abs(r['observed_mean_seconds']-r['cumulative_vehicle_seconds']/r['started_vehicle_count'])<1e-6
 rows.append(out)
rows.sort(key=lambda r:(['ingolstadt_custom','coulommiers','montereau_fault_yonne'].index(r['city']),['ippo','mappo','mamba','mbpo','dreamer'].index(r['method'])))
with (P/'deterministic_observed_metrics_table.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(P/'deterministic_observed_metrics_table.json').write_text(json.dumps({'ATT_definition':'Cumulative observed vehicle-seconds divided by actually started vehicles: includes completed and in-progress vehicles at cutoff; excludes never-departed vehicles. NOT all generated demand or final uncensored ATT.','percentiles_definition':'Within-seed completed-vehicle percentiles, averaged across 3 demand seeds; not pooled vehicle quantiles.','diversion_cost':'Historical definition located in outputs/ingo_latest_argmax_20261010/routing_cost.py: summed chosen-vs-minimum legal candidate free-flow cost increments divided by started vehicles. Current evaluation metrics lack destination-specific decisions, so cannot reconstruct exact cost. Historical numbers must not be transplanted.','rows':rows},indent=2)+'\n')
for r in rows:
 print(r['city'],r['method'],r['evaluation_episode'],*[f"{r[k]:.2f}" for k in ('ATT_seconds','Completion_percent','P90_seconds','P95_seconds','observed_ATT_seconds')])
