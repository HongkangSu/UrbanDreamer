import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/home/wangwenrui/shk/UrbanDreamer/urban_dreamer/outputs/ingolstadt_full_retrain_20261009')
source = root / 'source_bounded_dynamic_fresh_20261011'
output = root / 'online_bounded_dynamic_fresh_2000_20261011'
if (output/'launch.json').exists():
    raise FileExistsError('refusing duplicate fresh-online launch')
verification = json.loads((output/'fresh_start_verification.json').read_text())
if verification['status'] != 'passed' or verification['online_completed'] != 0:
    raise ValueError('fresh initialization verification required')
command = ['/home/wangwenrui/opt/miniconda3/envs/URB/bin/python', '-u',
           str(source/'tools/run_bounded_online_continuation.py'), '--output-dir', str(output),
           '--threads', '4', '--gpu-memory-fraction', '.25', '--evaluation-every', '20']
environment = dict(os.environ, URB_IMAGINATION_WORKERS='8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='2',
                   TORCHINDUCTOR_CACHE_DIR=str(root/'inductor_cache'))
environment.pop('URB_ACTOR_ENTROPY_MODE', None)
with (output/'online.log').open('ab', buffering=0) as log:
    child = subprocess.Popen(command, cwd=source, env=environment, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
record = dict(pid=child.pid, command=command, source_root=str(source), output_dir=str(output),
              checkpoint_episode=0, fresh_online_from_pretrained=True,
              user_authorized_coupled_training=True, actor_unfrozen=True,
              gpu='cuda:1', imagination_workers=8,
              started_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
record['process_identity'] = (Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
(output/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
status = dict(record, phase='fresh_bounded_dynamic_online_starting',
              old_mainline_stopped_at=368, cancelled_continuation_pid=502737)
temporary = root/'status.tmp'
temporary.write_text(json.dumps(status, indent=2)+'\n'); temporary.replace(root/'status.json')
print(json.dumps(record), flush=True)
