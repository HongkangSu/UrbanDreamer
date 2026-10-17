"""Launch an identity-recorded FP16 online handoff for one existing city run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--city', choices=('coulommiers', 'montereau_fault_yonne'), required=True)
parser.add_argument('--execution-source', type=Path)
args = parser.parse_args()
ROOT = Path('/home/wangwenrui/shk/UrbanDreamer/urban_dreamer')
RUN = ROOT / 'outputs/two_city_urbandreamer_pretrain_supplement_20261012'
SOURCE = args.execution_source.resolve() if args.execution_source else RUN / 'source_online_fp16_20261013'
CITY = RUN / args.city
queue = CITY / 'online_queue_fp16_20261013'
if (queue / 'status.json').exists():
    raise RuntimeError('FP16 queue already has state; inspect before duplicate launch')
queue.mkdir(exist_ok=True)
output = CITY / 'online_mainline_2000'
if args.city == 'coulommiers':
    stopped = json.loads((output / 'runtime_status.json').read_text())
    if stopped['phase'] != 'stopped_at_checkpoint_boundary':
        raise RuntimeError('Coulommiers has not stopped at its checkpoint boundary')
    change = json.loads((output / 'latest_profile_change.json').read_text())
    if change['after']['imagination_precision'] != 'fp16':
        raise RuntimeError('Coulommiers checkpoint has not been migrated to FP16')
    state = json.loads((CITY / 'online_queue/status.json').read_text())
    state.update(binding=dict(pretrain_dir=str(CITY / 'world_model'),
        execution_source=str(SOURCE), seed_dir=str(CITY / 'online_seed'), output_dir=str(output)),
        execution_manifest_sha256=hashlib.sha256((SOURCE / 'source_online_manifest.json').read_bytes()).hexdigest(),
        child_pid=None, child_identity=None, phase='online_retry_pending',
        watcher_stopped=False, child_left_running=False,
        parent_queue=str(CITY / 'online_queue'), profile_change=str(output / 'latest_profile_change.json'))
    (queue / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
device = 'cuda:1' if args.city == 'coulommiers' else 'cuda:0'
command = ['/home/wangwenrui/opt/miniconda3/envs/URB/bin/python', '-u',
    str(SOURCE / 'tools/queue_pretrained_online.py'), '--pretrain-dir', str(CITY / 'world_model'),
    '--execution-source', str(SOURCE), '--queue-dir', str(queue),
    '--seed-dir', str(CITY / 'online_seed'), '--output-dir', str(output),
    '--python', '/home/wangwenrui/opt/miniconda3/envs/URB/bin/python', '--device', device,
    '--minimum-free-gib', '6', '--poll-seconds', '5']
if args.city == 'montereau_fault_yonne':
    command.extend(['--pretrain-pid', '2318377'])
log_path = queue / 'watcher.log'
with log_path.open('ab', buffering=0) as log:
    watcher = subprocess.Popen(command, cwd=SOURCE,
        env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1'),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(city=args.city, device=device, watcher_pid=watcher.pid, command=command,
              execution_source=str(SOURCE), queue_dir=str(queue), output_dir=str(output),
              imagination_precision='fp16',
              utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), log=str(log_path))
(queue / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
(CITY / 'online_fp16_handoff_20261013.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
