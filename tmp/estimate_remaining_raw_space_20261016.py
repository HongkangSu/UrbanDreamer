from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter,defaultdict
import json,time
base=Path('outputs/pretrain_data_audit_20261016');start=time.time();wanted={};counts=Counter()
for city in ['ingolstadt_custom','coulommiers','montereau_fault_yonne']:
 for line in (base/'total_census'/f'{city}_canonical.jsonl').open():
  r=json.loads(line)
  for s in r['equivalent_sources']:
   if s['kind']=='raw':
    wanted[s['path']]=city
    for p in s.get('aliases',[]):wanted[p]=city
items=[]
for line in (base/'all_raw_inventory.jsonl').open():
 r=json.loads(line);paths=[r['path']]+[a['path'] for a in r.get('aliases',[])];cities={wanted[p] for p in paths if p in wanted}
 if cities:
  assert len(cities)==1;items.append((next(iter(cities)),paths))
def inspect(item):
 city,paths=item
 for name in paths:
  try:
   s=Path(name).stat()
   return city,dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,allocated=s.st_blocks*512,nlink=s.st_nlink,path=name)
  except FileNotFoundError:continue
  except OSError as e:return city,dict(error=str(e),path=name)
 return city,dict(missing=True)
result={c:dict(files=0,logical_bytes=0,allocated_bytes=0,local_bytes=0,private_bytes=0,missing=0,errors=0,hardlinked_files=0) for c in ['ingolstadt_custom','coulommiers','montereau_fault_yonne']};seen=set();localdev=Path.cwd().stat().st_dev
with ThreadPoolExecutor(max_workers=16) as pool:
 for city,r in pool.map(inspect,items):
  o=result[city]
  if r.get('missing'):o['missing']+=1;continue
  if 'error' in r:o['errors']+=1;continue
  key=(r['dev'],r['ino'])
  if key in seen:continue
  seen.add(key);o['files']+=1;o['logical_bytes']+=r['size'];o['allocated_bytes']+=r['allocated'];o['local_bytes' if r['dev']==localdev else 'private_bytes']+=r['allocated'];o['hardlinked_files']+=r['nlink']>1
out=dict(created_unix=time.time(),elapsed_seconds=time.time()-start,cities=result,scope='Existing original files corresponding to compatible census trajectories; target-inode dedup; includes protected files; upper bound on reclamation, not an approved deletion set')
Path('outputs/pretrain_storage_migration_20261016/remaining_raw_space_estimate.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
