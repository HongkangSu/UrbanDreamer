from pathlib import Path
import subprocess,os,json,hashlib
root=Path.cwd();base=root/'outputs/pretrain_storage_migration_20261016';out=base/'incremental/batch002';out.mkdir(exist_ok=True)
python='/home/wangwenrui/opt/miniconda3/envs/URB/bin/python';planner=root/'tools/retire_completed_cache_batch.py';retire=root/'tools/retire_cached_trajectories.py'
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [planner,retire]}
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def status(**value):(out/'controller_status.json').write_text(json.dumps(value,indent=2)+'\n')
try:
 status(stage='validating_completed_batch',limit=512,raw_deleted=False)
 cmd=[python,'-u',str(planner),'--parent',str(base/'corpus/ingolstadt_custom'),'--output',str(out),'--protection',str(base/'retention/active_protection.json'),'--protected-sources',str(base/'retention/active_protected_sources.jsonl'),'--inventory',str(root/'outputs/pretrain_data_audit_20261016/all_raw_inventory.jsonl'),'--targets',str(root/'outputs/pretrain_data_audit_20261016/total_census/target_contracts.json'),'--limit','512']
 with (out/'planning.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 status(stage='retiring_verified_raw',limit=512)
 with (out/'apply.log').open('w') as log:subprocess.run([python,'-u',str(retire),'--apply','--plan',str(out/'retirement/deletion_plan.json')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 ledger=[json.loads(l) for l in (out/'retirement/deletion_ledger.jsonl').open()]
 status(stage='complete',unlinked=sum(r['status']=='unlinked' for r in ledger),released_allocated_bytes=sum(r.get('released_allocated_bytes',0) for r in ledger),refused=sum(r['status']=='refused' for r in ledger))
except Exception as exc:
 status(stage='stopped_safely',error=str(exc));raise
