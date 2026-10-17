"""Retain Montereau's committed update and enable its existing activation offload."""
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

def stop(pid, *, group=False, frozen=False):
    info = process(pid)
    if info is None or info['state'] == 'Z':
        return
    send = os.killpg if group else os.kill
    send(pid, signal.SIGTERM)
    if frozen:
        send(pid, signal.SIGCONT)
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        current = process(pid)
        if current is None or current['state'] == 'Z' or current['identity'] != info['identity']:
            return
        time.sleep(.2)
    raise RuntimeError(f'process did not stop: {pid}')

state = json.loads((CITY / 'supervisor_status.json').read_text())
chain = json.loads((CITY / 'decoder_recovery_chain_status.json').read_text())
queue_state = json.loads((QUEUE / 'status.json').read_text())
if queue_state.get('child_pid'):
    raise RuntimeError('online training has already started; inspect before replacing its watcher')
parent, supervisor = chain['pid'], state['pid']
trainer = state['child_pids']['world_model']
watcher = queue_state['watcher_pid']
for pid, script in ((parent, 'run_montereau_recovery_chain.py'),
                    (supervisor, 'launch_montereau_parallel.py'),
                    (trainer, 'run_complete_pretrain.py'),
                    (watcher, 'queue_pretrained_online.py')):
    info = process(pid)
    if not info or not any(argument.endswith(script) for argument in info['command']):
        raise RuntimeError(f'process identity differs: {pid}')
checkpoint = CITY / 'world_model/checkpoints/latest.pt'
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
archive = CITY / 'runtime_handoff_20261013'
stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
(archive / f'before_offload_{stamp}.json').write_text(json.dumps(
    dict(supervisor=state, chain=chain, queue=queue_state, checkpoint_sha256=digest), indent=2) + '\n')
os.kill(parent, signal.SIGSTOP)
os.kill(supervisor, signal.SIGSTOP)
stop(trainer, group=True)
stop(supervisor, group=True, frozen=True)
stop(parent, group=True, frozen=True)
stop(watcher)
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
    raise RuntimeError('checkpoint changed during handoff; inspect its latest committed cursor')
command = ['/home/wangwenrui/opt/miniconda3/envs/URB/bin/python', '-u',
    str(RUN / 'run_montereau_recovery_chain.py'), '--city', 'montereau_fault_yonne',
    '--source-dir', str(SOURCE), '--device', 'cuda:0', '--gpu-admission-min-free-mib', '8192',
    '--pretrain-gpu-memory-fraction', '.40', '--activation-offload', '--poll-seconds', '5']
environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
with (CITY / 'recovery_chain_console_gpu0_offload_20261013.log').open('ab', buffering=0) as log:
    new_chain = subprocess.Popen(command, cwd=SOURCE, env=environment,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
new_info = process(new_chain.pid)
queue_state.update(pretrain_identity=new_info['identity'], watcher_stopped=False,
                   child_left_running=False, phase='waiting_for_pretrain')
(QUEUE / 'status.json').write_text(json.dumps(queue_state, indent=2) + '\n')
queue_command = json.loads((QUEUE / 'launch.json').read_text())['command']
index = queue_command.index('--pretrain-pid')
queue_command[index + 1] = str(new_chain.pid)
online_source = Path(queue_state['binding']['execution_source'])
with (QUEUE / 'watcher.log').open('ab', buffering=0) as log:
    new_watcher = subprocess.Popen(queue_command, cwd=online_source, env=environment,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    kind='enable_existing_activation_offload_after_allocator_oom', previous_trainer_pid=trainer,
    chain_pid=new_chain.pid, online_watcher_pid=new_watcher.pid, command=command,
    allocator_fraction=.40, activation_offload=True, batch_size=64,
    checkpoint_sha256_preserved=digest, previous_committed_progress=state.get('world_model_status'))
(archive / 'activation_offload_handoff.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
