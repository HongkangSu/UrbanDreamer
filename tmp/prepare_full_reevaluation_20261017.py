from pathlib import Path
import json, shutil, os, datetime, hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/('all_baselines_fixed_metrics_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S'))
OUT.mkdir(); entries=json.loads((ROOT/'tmp/reeval_checkpoint_discovery_20261017.json').read_text())['entries']
patches=[
("            state = sampler.state\n            archive = None", "            state = sampler.state\n            destinations = {str(v['vehicle_id']): int(v['destination_edge']) for v in state['vehicles']} if diversion_cost is not None else {}\n            archive = None"),
('from ..observations import build_actor_observations, selected_decisions', 'from ..routing_evaluation_metrics import (DiversionCost, all_demand_censored_metrics,\n    DIVERSION_COST_DEFINITION, ALL_DEMAND_CENSORED_DEFINITION)\nfrom ..observations import build_actor_observations, selected_decisions'),
("label = label or getattr(select, 'label', 'method')\n        rows = []", "label = label or getattr(select, 'label', 'method')\n        evaluation_metrics = (not collect and label != 'stochastic'\n                              and (archive_metadata or {}).get('phase') != 'train')\n        diversion_cost = DiversionCost(self.candidates) if evaluation_metrics else None\n        rows = []"),
('result = sample.result\n                if archive is not None:', "result = sample.result\n                if diversion_cost is not None:\n                    realized = getattr(result, 'conditioning_decisions', None)\n                    diversion_cost.add(result.decisions if realized is None else realized, destinations)\n                if archive is not None:"),
('            if metrics_transform is not None:\n', "            if diversion_cost is not None:\n                metrics.update(diversion_cost.metrics(int(state['started_vehicle_count'])))\n                metrics.update(all_demand_censored_metrics(vehicles, state['vehicles'],\n                    min(float(state['time']), self.deadline)))\n                metrics.update(diversion_cost_definition=DIVERSION_COST_DEFINITION,\n                    all_demand_censored_definition=ALL_DEMAND_CENSORED_DEFINITION)\n            if metrics_transform is not None:\n")]
sources={}
for e in entries:
 src=e['source']
 if src not in sources:
  dst=OUT/'sources'/str(len(sources));shutil.copytree(src,dst,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
  shutil.copy2(ROOT/'urban_dreamer/learning/routing_evaluation_metrics.py',dst/'urban_dreamer/learning/routing_evaluation_metrics.py')
  p=dst/'urban_dreamer/learning/baselines/common.py';s=p.read_text()
  for old,new in patches:
   assert s.count(old)==1,(str(p),old,s.count(old));s=s.replace(old,new,1)
  p.write_text(s);sources[src]=str(dst)
 e['original_source']=src;e['source']=sources[src]
 out=OUT/e['city']/e['method'];out.mkdir(parents=True);e['evaluation_output']=str(out)
 if e.get('checkpoint'):
  old=Path(e['checkpoint']);new=out/'checkpoint.pt';os.link(old,new);e['original_checkpoint']=str(old);e['checkpoint']=str(new)
 # Config remains original immutable source configuration; input hashes verified before evaluation.
(OUT/'manifest.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),demand_seed=42,seeds=[10042,10043,10044],excluded='PPO v7',entries=entries),indent=2)+'\n')
(ROOT/'tmp/full_reevaluation_latest_path.txt').write_text(str(OUT)+'\n')
print(OUT)
