"""Release Montereau's GPU after a committed checkpoint for a runtime comparison."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time

RUN = Path('/home/wangwenrui/shk/UrbanDreamer/urban_dreamer/outputs/'
           'two_city_urbandreamer_pretrain_supplement_20261012')
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
chain = json.loads((CITY / 'decoder_recovery_chain_status.json').read_text())
parent, supervisor, trainer = chain['pid'], state['pid'], state['child_pids']['world_model']
identities = {}
for pid, script in ((parent, 'run_montereau_recovery_chain.py'),
                    (supervisor, 'launch_montereau_parallel.py'),
                    (trainer, 'run_complete_pretrain.py')):
    info = process(pid)
    if not info or not any(argument.endswith(script) for argument in info['command']):
        raise RuntimeError(f'process identity differs: {pid}')
    identities[str(pid)] = info
checkpoint = CITY / 'world_model/checkpoints/latest.pt'
original_stat = checkpoint.stat()
stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
archive = CITY / 'runtime_handoff_20261013' / ('common_runtime_' + stamp)
archive.mkdir()
record = dict(phase='waiting_for_checkpoint', requested_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
              parent_pid=parent, supervisor_pid=supervisor, trainer_pid=trainer,
              process_identities=identities, previous_runtime=state)
(archive / 'request.json').write_text(json.dumps(record, indent=2) + '\n')
os.kill(parent, signal.SIGSTOP)
os.kill(supervisor, signal.SIGSTOP)
print(json.dumps(dict(phase='waiting_for_checkpoint', trainer_pid=trainer, record_dir=str(archive))), flush=True)
deadline = time.monotonic() + 650
try:
    while time.monotonic() < deadline:
        current = process(trainer)
        if current is None or current['state'] == 'Z' or current['identity'] != identities[str(trainer)]['identity']:
            raise RuntimeError('trainer exited before the requested checkpoint')
        latest = checkpoint.stat()
        if latest.st_ino != original_stat.st_ino or latest.st_mtime_ns > original_stat.st_mtime_ns:
            break
        time.sleep(.2)
    else:
        raise TimeoutError('no new committed checkpoint; supervisor resumed')
    os.killpg(trainer, signal.SIGSTOP)
    time.sleep(.2)
    frozen = process(trainer)
    if not frozen or frozen['state'] not in ('T', 't'):
        raise RuntimeError('trainer did not freeze')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for pid in (trainer, supervisor, parent):
        os.killpg(pid, signal.SIGTERM)
        os.killpg(pid, signal.SIGCONT)
        until = time.monotonic() + 25
        while time.monotonic() < until:
            current = process(pid)
            if current is None or current['state'] == 'Z':
                break
            time.sleep(.2)
        else:
            raise RuntimeError(f'process did not stop: {pid}')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError('committed checkpoint changed while stopping')
    record.update(phase='stopped_at_committed_checkpoint', checkpoint_sha256=digest,
        stopped_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        checkpoint=str(checkpoint), latest_status=json.loads((CITY / 'world_model.status.json').read_text()))
    (archive / 'stop_record.json').write_text(json.dumps(record, indent=2) + '\n')
    (CITY / 'runtime_handoff_20261013/latest_common_runtime_stop.json').write_text(
        json.dumps(dict(record_dir=str(archive), **record), indent=2) + '\n')
    print(json.dumps(dict(phase=record['phase'], checkpoint_sha256=digest,
        record_dir=str(archive), latest_status=record['latest_status']), indent=2), flush=True)
except BaseException:
    for pid in (trainer, supervisor, parent):
        current = process(pid)
        if current and current['state'] in ('T', 't'):
            os.killpg(pid, signal.SIGCONT)
    raise
