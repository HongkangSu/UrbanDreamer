"""Continue proof-gated cleanup using parallel checks and current protections."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();base=root/'outputs/pretrain_storage_migration_20261016';source=base/'retirement_source_v2';corpus=base/'corpus';census=root/'outputs/pretrain_data_audit_20261016/total_census';retention=base/'retention';protection=retention/'after_ingo_retirement'
python='/home/wangwenrui/opt/miniconda3/envs/URB/bin/python';env=dict(os.environ,PYTHONPATH=str(source),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def read(p):return json.loads(Path(p).read_text())
def write(value):
 p=base/'pipeline_progress.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,p)
def gate():
 for name,h in read(source/'source_manifest.json').items():
  assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h,name
results={}
try:
 evidence=read(base/'ingo_cleanup_dependency_review/evaluation_completion.json')
 assert evidence['status']=='passed' and evidence['all_21_comparisons_exact']
 for city in ('ingolstadt_custom','coulommiers','montereau_fault_yonne'):
  report=corpus/city/'verification.json'
  while not report.exists():
   p=corpus/city/'progress.json';progress=read(p) if p.exists() else {'status':'queued'}
   write(dict(stage='waiting_for_verified_city_cache',city=city,conversion=progress,completed_cities=results,parallel_city_conversion=True))
   if progress.get('status')=='failed':raise RuntimeError(city+' conversion failed')
   time.sleep(30)
  assert read(report)['status']=='passed';gate()
  out=retention/'deletion_plan'/(city+'_full_verified_v2');out.mkdir(parents=True,exist_ok=True)
  cmd=[python,'-u',str(source/'tools/retire_cached_trajectories.py')]
  plan=cmd+['--plan-only','--workers','4','--manifest',str(corpus/city/'cache_manifest.json'),'--verification-report',str(report),'--canonical',str(census/(city+'_canonical.jsonl')),'--inventory',str(root/'outputs/pretrain_data_audit_20261016/all_raw_inventory.jsonl'),'--targets',str(census/'target_contracts.json'),'--protection',str(protection/'active_protection.json'),'--protected-sources',str(protection/'protected_sources.jsonl'),'--output',str(out)]
  write(dict(stage='planning_verified_raw_retirement',city=city,workers=4,planning_progress=str(out/'planning_progress.json'),completed_cities=results,ingo_obsolete_dependency_removed=True))
  with (out/'planning.log').open('w') as log:subprocess.run(plan,env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
  planned=read(out/'deletion_plan.json');eligible=sum(r['status']=='retire' for r in planned['rows'])
  write(dict(stage='retiring_verified_unprotected_raw',city=city,workers=4,eligible_files=eligible,completed_cities=results))
  gate()
  if eligible:
   with (out/'apply.log').open('w') as log:subprocess.run(cmd+['--apply','--workers','4','--plan',str(out/'deletion_plan.json')],env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
  ledger=out/'deletion_ledger.jsonl';events=[json.loads(l) for l in ledger.open()] if ledger.exists() else []
  results[city]=dict(unlinked=sum(r.get('status')=='unlinked' for r in events),released_allocated_bytes=sum(r.get('released_allocated_bytes',0) for r in events),protected_or_skipped=sum(r['status']!='retire' for r in planned['rows']),manifest=str(corpus/city/'cache_manifest.json'))
 write(dict(stage='eligible_cleanup_complete',cities=results,active_dependencies_preserved=True))
except Exception as exc:
 write(dict(stage='stopped_safely',error=str(exc),completed_cities=results));raise
