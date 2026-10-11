import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/home/wangwenrui/shk/UrbanDreamer/urban_dreamer/outputs/ingolstadt_full_retrain_20261009')
output = root/'online_bounded_dynamic_fresh_2000_20261011'
change = json.loads((output/'latest_profile_change.json').read_text())
assert change['after'] == {'rollouts':8,'evaluation_prior':0.0,'hold':200,'off':1000}
old = json.loads((output/'launch.json').read_text())
if Path(f"/proc/{old['pid']}").exists():
    raise RuntimeError('previous run still has a live process')
source = Path(old['source_root'])
command = ['/home/wangwenrui/opt/miniconda3/envs/URB/bin/python','-u',
           str(source/'tools/run_bounded_online_continuation.py'),'--output-dir',str(output),
           '--threads','4','--gpu-memory-fraction','.25','--evaluation-every','20']
environment = dict(os.environ,URB_IMAGINATION_WORKERS='8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='2',
                   TORCHINDUCTOR_CACHE_DIR=str(root/'inductor_cache'))
environment.pop('URB_ACTOR_ENTROPY_MODE',None)
with (output/'online.log').open('ab',buffering=0) as log:
    child = subprocess.Popen(command,cwd=source,env=environment,stdout=log,
                             stderr=subprocess.STDOUT,start_new_session=True)
record = dict(old,pid=child.pid,command=command,checkpoint_episode=change['boundary'],
    rollouts_per_pass=8,evaluation_prior_weight=0.,prior_hold_episode=200,prior_off_episode=1000,
    started_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
    profile_change_backup=change['backup'])
record['process_identity']=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
(output/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
s=dict(record,phase='rollout8_annealed_prior_online_starting')
t=root/'status.tmp';t.write_text(json.dumps(s,indent=2)+'\n');t.replace(root/'status.json')
print(json.dumps(record),flush=True)
