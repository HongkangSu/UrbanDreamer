"""Stop only Coulommiers after its next committed online episode."""
import json
import os
from pathlib import Path
import signal
import time


CITY = Path('/home/wangwenrui/shk/UrbanDreamer/urban_dreamer/outputs/'
            'two_city_urbandreamer_pretrain_supplement_20261012/coulommiers')
OUTPUT = CITY / 'online_mainline_2000'
QUEUE = CITY / 'online_queue'


def process(pid):
    try:
        directory = Path('/proc') / str(pid)
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        command = (directory / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return dict(state=fields[0], identity=fields[19], command=command)
    except FileNotFoundError:
        return None


status = json.loads((QUEUE / 'status.json').read_text())
pid, watcher = status['child_pid'], status['watcher_pid']
child = process(pid)
if (not child or child['identity'] != status['child_identity']
        or str(OUTPUT) not in child['command'] or os.getpgid(pid) != pid):
    raise RuntimeError('online runner identity/output/process group differs')
owner = process(watcher)
if (not owner or not any(argument.endswith('/queue_pretrained_online.py') for argument in owner['command'])
        or str(OUTPUT) not in owner['command']):
    raise RuntimeError('queue identity/output differs')
initial = len(json.loads((OUTPUT / 'pilot_progress.json').read_text())['online'])
archive = OUTPUT / 'runtime_handoff_20261013'
archive.mkdir(exist_ok=True)
record = dict(pid=pid, identity=child['identity'], watcher_pid=watcher,
              initial_completed=initial, requested_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
(archive / 'request.json').write_text(json.dumps(record, indent=2) + '\n')
(archive / 'original_queue_status.json').write_text(json.dumps(status, indent=2) + '\n')
os.kill(watcher, signal.SIGTERM)
deadline = time.monotonic() + 20
while time.monotonic() < deadline:
    owner = process(watcher)
    if owner is None or owner['state'] == 'Z':
        break
    time.sleep(.2)
else:
    raise RuntimeError('queue watcher did not stop; runner remains untouched')
print(json.dumps(dict(phase='waiting_for_next_checkpoint', **record)), flush=True)
deadline = time.monotonic() + 1200
while time.monotonic() < deadline:
    current = process(pid)
    if not current or current['identity'] != child['identity'] or current['state'] == 'Z':
        raise RuntimeError('online runner exited before checkpoint boundary')
    progress = json.loads((OUTPUT / 'pilot_progress.json').read_text())
    if len(progress['online']) > initial:
        break
    time.sleep(.2)
else:
    raise TimeoutError('online checkpoint was not committed; runner remains untouched')
os.killpg(pid, signal.SIGSTOP)
time.sleep(.2)
current = process(pid)
if not current or current['identity'] != child['identity'] or current['state'] not in ('T', 't'):
    raise RuntimeError('could not verify frozen runner')
progress = json.loads((OUTPUT / 'pilot_progress.json').read_text())
completed = len(progress['online'])
import torch
checkpoint = torch.load(OUTPUT / 'checkpoints/latest.pt', map_location='cpu', weights_only=True)
durable = checkpoint['training_state']['metrics']['online_pilot_progress']
if checkpoint['next_unit'] != completed or len(durable['online']) != completed:
    os.killpg(pid, signal.SIGCONT)
    raise RuntimeError('checkpoint and progress cursor differ; runner resumed')
os.killpg(pid, signal.SIGTERM)
os.killpg(pid, signal.SIGCONT)
deadline = time.monotonic() + 15
while time.monotonic() < deadline:
    current = process(pid)
    if current is None or current['state'] == 'Z':
        break
    time.sleep(.2)
else:
    raise RuntimeError('online runner did not exit after saving its checkpoint')
record.update(phase='stopped_at_checkpoint_boundary', completed_episodes=completed,
              config_fingerprint=checkpoint['config_fingerprint'], input_hash=checkpoint['input_hash'],
              stopped_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
(archive / 'stop_record.json').write_text(json.dumps(record, indent=2) + '\n')
(OUTPUT / 'runtime_status.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
