"""Retire only verified, unprotected raw archives after each complete city cache."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();migration=root/'outputs/pretrain_storage_migration_20261016'
source=migration/'retirement_source';corpus=migration/'corpus';census=root/'outputs/pretrain_data_audit_20261016/total_census';retention=migration/'retention'
python='/home/wangwenrui/opt/miniconda3/envs/URB/bin/python'
read=lambda p:json.loads(p.read_text())
def save(value):
 p=migration/'pipeline_progress.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,p)
def source_gate():
 for name,digest in read(source/'source_manifest.json').items():
  assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
try:
 results={}
 for city in ('ingolstadt_custom','coulommiers','montereau_fault_yonne'):
  report=corpus/city/'verification.json'
  while not report.exists():
   progress=corpus/city/'progress.json'
   state=read(progress) if progress.exists() else {'status':'queued'}
   save(dict(stage='waiting_for_verified_city_cache',city=city,conversion=state,completed_cities=results,raw_deletion_requires_complete_city=True))
   if state.get('status')=='failed':raise RuntimeError(city+' cache conversion failed; no raw retirement')
   launch=read(corpus/'launch.json');stat=Path('/proc')/str(launch['pid'])/'stat'
   if not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0]=='Z':
    if not report.exists():raise RuntimeError('converter stopped before '+city+' verification')
   time.sleep(30)
  assert read(report)['status']=='passed'
  source_gate()
  out=retention/'deletion_plan'/city;out.mkdir(parents=True,exist_ok=True)
  env=dict(os.environ,PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  command=[python,str(source/'tools/retire_cached_trajectories.py')]
  planning=command+['--plan-only','--manifest',str(corpus/city/'cache_manifest.json'),'--verification-report',str(report),
   '--canonical',str(census/(city+'_canonical.jsonl')),'--inventory',str(root/'outputs/pretrain_data_audit_20261016/all_raw_inventory.jsonl'),
   '--targets',str(census/'target_contracts.json'),'--protection',str(retention/'active_protection.json'),
   '--protected-sources',str(retention/'active_protected_sources.jsonl'),'--output',str(out)]
  save(dict(stage='planning_verified_raw_retirement',city=city,completed_cities=results))
  with (out/'planning.log').open('w') as log:subprocess.run(planning,env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
  plan=read(out/'deletion_plan.json');eligible=sum(r['status']=='retire' for r in plan['rows'])
  save(dict(stage='retiring_verified_unprotected_raw',city=city,eligible=eligible,protected_or_skipped=len(plan['rows'])-eligible,completed_cities=results))
  source_gate()
  if eligible:
   with (out/'apply.log').open('w') as log:subprocess.run(command+['--apply','--plan',str(out/'deletion_plan.json')],env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
  ledger=out/'deletion_ledger.jsonl';events=[json.loads(l) for l in ledger.open()] if ledger.exists() else []
  results[city]=dict(manifest=str(corpus/city/'cache_manifest.json'),eligible=eligible,unlinked=sum(r.get('status')=='unlinked' for r in events),
   released_allocated_bytes=sum(r.get('released_allocated_bytes',0) for r in events),protected_or_skipped=sum(r['status']!='retire' for r in plan['rows']))
 save(dict(stage='verified_conversion_and_eligible_retirement_complete',cities=results,
  remaining_protected_raw=True,note='Protected dependencies and excluded/unconverted archives remain; this is not a claim that every raw file was deleted.'))
except Exception as exc:
 save(dict(stage='stopped_safely',error=str(exc),completed_cities=locals().get('results',{})))
 raise
