"""Freeze GA30, evaluate three cities, and extend the existing comparison."""
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures, hashlib, json, os, shutil, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=Path((ROOT/'tmp/full_reevaluation_latest_path.txt').read_text().strip())
manifest=json.loads((PREVIOUS/'manifest.json').read_text())
manifest['entries']=[e for e in manifest['entries'] if e['method']!='ga_routing']
OUT=ROOT/'outputs'/('all_baselines_with_ga30_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
OUT.mkdir();SOURCE=OUT/'ga_source'
base=Path(next(e['source'] for e in manifest['entries'] if e['method']=='ippo'))
shutil.copytree(base,SOURCE,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
for name in ('evaluate_ga_routes.py','causal_routing_policies.py'):
 shutil.copy2(ROOT/'tools'/name,SOURCE/'tools'/name)
AUTHOR=SOURCE/'references/genetic_algorithm.py';AUTHOR.parent.mkdir(exist_ok=True)
old_author=ROOT/'outputs/ingo_causal_routing_20261010/references/author_repo/src/baselines/genetic_algorithm.py'
shutil.copy2(old_author,AUTHOR)
manifest.update(previous_results=str(PREVIOUS),created_utc=datetime.now(timezone.utc).isoformat(),ga_protocol='30-second complete routes; original author GA operators and parameters; full legal-path search; original first-legal fallback completed with shortest suffix; closed loops erased for unambiguous SUMO full-route installation.')
for city in ('ingolstadt_custom','coulommiers','montereau_fault_yonne'):
 expected=next(e['input_hashes'] for e in manifest['entries'] if e['city']==city)
 manifest['entries'].append(dict(city=city,method='ga_routing',source=str(SOURCE),config=str(SOURCE/'configs'/f'{city}.toml'),checkpoint=None,episode=None,input_hashes=expected,evaluation_output=str(OUT/city/'ga_routing'),routing_mode='current_to_dest',decision_protocol='30s_complete_route'))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'tmp/ga30_supplement_latest_path.txt').write_text(str(OUT)+'\n')
files=[SOURCE/'tools/evaluate_ga_routes.py',SOURCE/'tools/causal_routing_policies.py',AUTHOR,SOURCE/'urban_dreamer/learning/routing_evaluation_metrics.py',SOURCE/'urban_dreamer/learning/baselines/common.py',SOURCE/'urban_dreamer/traffic/event_session.py']
(OUT/'ga_source_hashes.json').write_text(json.dumps({str(f.relative_to(SOURCE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(dict(output=str(OUT),total_runs=len(manifest['entries']),new_episodes=9)),flush=True)
def run(e):
 output=Path(e['evaluation_output']);output.parent.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,'-u',str(SOURCE/'tools/evaluate_ga_routes.py'),'--config',e['config'],'--output',str(output),'--author-source',str(AUTHOR),'--seeds','10042','10043','10044']
 with (output.parent/'ga30_evaluation.log').open('w') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
  (output.parent/'ga30_evaluation.pid').write_text(str(p.pid));code=p.wait()
 return dict(city=e['city'],method='ga_routing',exit_code=code)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
 for future in concurrent.futures.as_completed([executor.submit(run,e) for e in manifest['entries'] if e['method']=='ga_routing']):
  result=future.result();results.append(result);print(json.dumps(result),flush=True)
  (OUT/'ga_queue_status.json').write_text(json.dumps(dict(results=results,total=3),indent=2)+'\n')
if any(r['exit_code'] for r in results):raise SystemExit(1)
