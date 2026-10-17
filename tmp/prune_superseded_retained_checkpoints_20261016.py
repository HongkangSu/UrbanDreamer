"""Remove only our superseded best-protection copies; keep all metadata and current best."""
from pathlib import Path
import json,hashlib,os,time
root=Path.cwd();base=root/'outputs/pretrain_storage_migration_20261016/retention/available_best'
output=root/'outputs/pretrain_storage_migration_20261016/incremental';output.mkdir(exist_ok=True)
plan=[]
for city in ['ingolstadt_custom','coulommiers','montereau_fault_yonne']:
 folder=base/city;current=(folder/'current').resolve();meta=json.loads((current/'metadata.json').read_text());cp=current/'checkpoint.pt'
 with cp.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==meta['checkpoint_sha256']
 for old in sorted((folder/'generations').glob('episode_*')):
  if old.resolve()==current:continue
  p=old/'checkpoint.pt'
  if not p.exists():continue
  assert not p.is_symlink() and p.is_file()
  m=json.loads((old/'metadata.json').read_text())
  assert meta['mean_selection_score']>=m['mean_selection_score']
  for k in ['config_fingerprint','input_hash','evaluation_protocol','evaluation_demand_seed']:assert meta[k]==m[k],k
  assert [e['demand_hash'] for e in meta['evaluations']]==[e['demand_hash'] for e in m['evaluations']]
  st=p.stat();plan.append(dict(city=city,path=str(p),current_checkpoint=str(cp),current_hash=meta['checkpoint_sha256'],size=st.st_size,allocated=st.st_blocks*512,dev=st.st_dev,ino=st.st_ino,mtime_ns=st.st_mtime_ns,old_score=m['mean_selection_score'],new_score=meta['mean_selection_score']))
(output/'superseded_checkpoint_plan.json').write_text(json.dumps(dict(reason='User authorized keep best and logs; these are older copies made by our protection watcher, not original training checkpoints',files=plan),indent=2)+'\n')
ledger=output/'superseded_checkpoint_deletions.jsonl'
for r in plan:
 p=Path(r['path']);folder=base/r['city'];active=(folder/'current').resolve()
 assert p.parent.resolve()!=active
 m=json.loads((active/'metadata.json').read_text());assert m['mean_selection_score']>=r['new_score']
 st=p.stat();assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(r['dev'],r['ino'],r['size'],r['mtime_ns'])
 with p.open('rb') as f:
  assert os.fstat(f.fileno()).st_ino==r['ino'];p.unlink();after=os.fstat(f.fileno())
  r.update(status='unlinked',time=time.time(),released_allocated_bytes=r['allocated'] if after.st_nlink==0 else 0)
  with ledger.open('a') as log:log.write(json.dumps(r)+'\n');log.flush();os.fsync(log.fileno())
 assert (p.parent/'metadata.json').exists()
print(json.dumps(dict(files=len(plan),released_bytes=sum(r['released_allocated_bytes'] for r in plan),current_best_and_all_logs_retained=True)))
