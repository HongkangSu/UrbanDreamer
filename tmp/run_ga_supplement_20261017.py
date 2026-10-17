"""Re-run the historical author GA unchanged, then append comparable metrics."""
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures, hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=Path((ROOT/'tmp/full_reevaluation_latest_path.txt').read_text().strip())
HISTORICAL=ROOT/'outputs/three_city_ga_routing_20261010'
SOURCE=HISTORICAL/'source'
AUTHOR=ROOT/'outputs/ingo_causal_routing_20261010/references/author_repo/src/baselines/genetic_algorithm.py'
OUT=ROOT/'outputs'/('all_baselines_with_ga_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
OUT.mkdir();manifest=json.loads((PREVIOUS/'manifest.json').read_text())
manifest.update(previous_results=str(PREVIOUS),created_utc=datetime.now(timezone.utc).isoformat(),ga_decision_protocol='Historical author GA: event-driven next-hop; other methods retain current-to-destination30s protocol. Physical inputs and reported metric definitions matched.')
for city in ('ingolstadt_custom','coulommiers','montereau_fault_yonne'):
 old=json.loads((HISTORICAL/city/'ga_routing/protocol.json').read_text())
 assert hashlib.sha256(AUTHOR.read_bytes()).hexdigest()==old['ga']['author_sha256']
 assert hashlib.sha256((SOURCE/'tools/causal_routing_policies.py').read_bytes()).hexdigest()==old['policy_source_sha256']
 expected=next(e['input_hashes'] for e in manifest['entries'] if e['city']==city)
 assert old['input_hashes']==expected
 output=OUT/city/'ga_routing';output.mkdir(parents=True)
 manifest['entries'].append(dict(city=city,method='ga_routing',source=str(SOURCE),config=str(SOURCE/'configs'/f'{city}.toml'),checkpoint=None,episode=None,input_hashes=expected,evaluation_output=str(output),routing_mode='next_hop',decision_protocol='event_driven_next_hop',raw_output=str(OUT/'raw'/city/'ga_routing')))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'tmp/ga_supplement_latest_path.txt').write_text(str(OUT)+'\n')
print(json.dumps(dict(output=str(OUT),runs=39,episodes=117,new_ga_runs=3,new_ga_episodes=9)),flush=True)
def run(city):
 output=OUT/city/'ga_routing'
 cmd=[sys.executable,'-u',str(SOURCE/'tools/evaluate_causal_routing.py'),'--city',city,'--method','ga_routing','--source',str(SOURCE),'--ga-source',str(AUTHOR),'--reference',str(HISTORICAL/'references'/f'{city}.json'),'--output',str(OUT/'raw'/city),'--seeds','10042','10043','10044']
 with (output/'evaluation.log').open('w') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
  (output/'evaluation.pid').write_text(str(p.pid));code=p.wait()
 return dict(city=city,method='ga_routing',exit_code=code)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
 for future in concurrent.futures.as_completed([executor.submit(run,c) for c in ('ingolstadt_custom','coulommiers','montereau_fault_yonne')]):
  result=future.result();results.append(result);print(json.dumps(result),flush=True)
  (OUT/'ga_queue_status.json').write_text(json.dumps(dict(results=results,total=3),indent=2)+'\n')
if any(r['exit_code'] for r in results):raise SystemExit(1)
