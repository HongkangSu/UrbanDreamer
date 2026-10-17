"""Resume the accepted Montereau checkpoint with sufficient allocator headroom."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path('/home/wangwenrui/shk/UrbanDreamer/urban_dreamer')
RUN = ROOT / 'outputs/two_city_urbandreamer_pretrain_supplement_20261012'
CITY = RUN / 'montereau_fault_yonne'


def process(pid):
    try:
        directory = Path('/proc') / str(pid)
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        command = (directory / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return dict(state=fields[0], identity=fields[19], command=command)
    except FileNotFoundError:
        return None


state = json.loads((CITY / 'supervisor_status.json').read_text())
if state.get('child_pids'):
    raise RuntimeError('pretraining child became active; inspect before replacing the waiting supervisor')
archive = CITY / 'runtime_handoff_20261013'
archive.mkdir(exist_ok=True)
for name in ('supervisor_status.json', 'decoder_recovery_chain_status.json'):
    backup = archive / ('before_' + name)
    if not backup.exists():
        backup.write_bytes((CITY / name).read_bytes())
previous = {}
for pid, script in ((1879467, 'run_montereau_recovery_chain.py'),
                    (1880700, 'queue_pretrained_online.py')):
    info = process(pid)
    if info is None or not any(argument.endswith(script) for argument in info['command']):
        raise RuntimeError(f'expected process identity missing: {pid}')
    if script == 'queue_pretrained_online.py' and str(CITY / 'online_mainline_2000') not in info['command']:
        raise RuntimeError('queue output identity differs')
    previous[str(pid)] = info
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        current = process(pid)
        if current is None or current['state'] == 'Z' or current['identity'] != info['identity']:
            break
        time.sleep(.2)
    else:
        raise RuntimeError(f'process did not stop: {pid}')
child = process(1993013)
if child is not None and child['state'] != 'Z':
    raise RuntimeError('the previous pretraining supervisor is still alive')
source = RUN / 'source_recoveryfix_20261012'
command = ['/home/wangwenrui/opt/miniconda3/envs/URB/bin/python', '-u',
           str(RUN / 'run_montereau_recovery_chain.py'), '--city', 'montereau_fault_yonne',
           '--source-dir', str(source), '--device', 'cuda:0',
           '--gpu-admission-min-free-mib', '8192', '--pretrain-gpu-memory-fraction', '.35',
           '--poll-seconds', '5']
log_path = CITY / 'recovery_chain_console_gpu0_20261013.log'
with log_path.open('ab', buffering=0) as log:
    child = subprocess.Popen(command, cwd=source,
        env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1'),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    kind='raise_pretrain_allocator_and_reset_oom_admission_backoff', previous_processes=previous,
    new_chain_pid=child.pid, command=command, allocator_fraction_before=.22,
    allocator_fraction_after=.35, initial_admission_mib=8192,
    checkpoint_preserved=str(CITY / 'world_model/checkpoints/latest.pt'),
    online_handoff='Previous watcher stopped; install the validated FP16 execution source next.',
    log=str(log_path))
(archive / 'gpu0_start_handoff.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
