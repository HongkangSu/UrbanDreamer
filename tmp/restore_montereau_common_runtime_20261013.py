"""Restore Montereau with the GPU execution settings validated by the private probe."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/home/wangwenrui/shk/UrbanDreamer/urban_dreamer')
RUN = ROOT / 'outputs/two_city_urbandreamer_pretrain_supplement_20261012'
CITY = RUN / 'montereau_fault_yonne'
SOURCE = RUN / 'source_recoveryfix_20261012'
QUEUE = CITY / 'online_queue_fp16_20261013'

def process(pid):
    try:
        directory = Path('/proc') / str(pid)
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        command = (directory / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return dict(state=fields[0], identity=fields[19], command=command)
    except FileNotFoundError:
        return None

stop = json.loads((CITY / 'runtime_handoff_20261013/latest_common_runtime_stop.json').read_text())
probe = json.loads((CITY / 'benchmark_common_runtime_20261013/result.json').read_text())
if (probe['status'] != 'complete' or probe['profile']['activation_offload']
        or probe['allocator_fraction'] != 1.0 or len(probe['rows']) < 6):
    raise RuntimeError('common runtime has not passed its consecutive batch64/tail63 probe')
if any(not row['gradients_finite'] or not row['parameters_finite'] or not row['encoder_unchanged']
       or not row['encoder_frozen'] or not row['changed_dynamics_tensors']
       or not row['changed_decoder_parameters'] for row in probe['rows']):
    raise RuntimeError('private probe did not preserve the encoder and update finite dynamics/decoder')
checkpoint = CITY / 'world_model/checkpoints/latest.pt'
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if digest != stop['checkpoint_sha256'] or digest != probe['snapshot_sha256']:
    raise RuntimeError('production checkpoint ancestry differs from the private probe')
for key in ('trainer_pid', 'supervisor_pid', 'parent_pid'):
    old = process(stop[key])
    if old is not None and old['state'] != 'Z':
        raise RuntimeError('previous pretraining process is still alive')
queue_state = json.loads((QUEUE / 'status.json').read_text())
if queue_state.get('child_pid'):
    raise RuntimeError('online training has already started')
watcher = queue_state['watcher_pid']
info = process(watcher)
if info and info['state'] != 'Z':
    if not any(argument.endswith('queue_pretrained_online.py') for argument in info['command']):
        raise RuntimeError('online watcher identity differs')
    os.kill(watcher, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        current = process(watcher)
        if current is None or current['state'] == 'Z':
            break
        time.sleep(.2)
    else:
        raise RuntimeError('online watcher did not stop')
command = ['/home/wangwenrui/opt/miniconda3/envs/URB/bin/python', '-u',
    str(RUN / 'run_montereau_recovery_chain.py'), '--city', 'montereau_fault_yonne',
    '--source-dir', str(SOURCE), '--device', 'cuda:0', '--gpu-admission-min-free-mib', '8192',
    '--pretrain-gpu-memory-fraction', '1.0', '--poll-seconds', '5']
environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
with (CITY / 'recovery_chain_console_common_runtime_20261013.log').open('ab', buffering=0) as log:
    chain = subprocess.Popen(command, cwd=SOURCE, env=environment,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
queue_state.update(pretrain_identity=process(chain.pid)['identity'], watcher_stopped=False,
                   child_left_running=False, phase='waiting_for_pretrain')
(QUEUE / 'status.json').write_text(json.dumps(queue_state, indent=2) + '\n')
queue_command = json.loads((QUEUE / 'launch.json').read_text())['command']
index = queue_command.index('--pretrain-pid')
queue_command[index + 1] = str(chain.pid)
online_source = Path(queue_state['binding']['execution_source'])
with (QUEUE / 'watcher.log').open('ab', buffering=0) as log:
    online = subprocess.Popen(queue_command, cwd=online_source, env=environment,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    kind='restore_common_gpu_execution_after_private_full_tail_probe', command=command,
    chain_pid=chain.pid, online_watcher_pid=online.pid, allocator_fraction=1.0,
    activation_offload=False, checkpoint_sha256_preserved=digest,
    retained_optimizer_updates=stop['latest_status']['optimizer_updates'],
    retained_sampled_sequences=stop['latest_status']['sampled_sequences'],
    probe_result=str(CITY / 'benchmark_common_runtime_20261013/result.json'))
(CITY / 'runtime_handoff_20261013/common_runtime_restored.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
