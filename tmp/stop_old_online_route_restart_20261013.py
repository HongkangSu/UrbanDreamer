"""Preserve and stop the two old UrbanDreamer online runs for a fresh route run."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'outputs/online_route_restart_stop_20261013'
RECORD.mkdir(exist_ok=False)
RUNS = [ROOT / 'outputs/ingolstadt_full_retrain_20261009/online_bounded_noprior_fresh_t2_2000_20261011',
        ROOT / 'outputs/two_city_urbandreamer_pretrain_supplement_20261012/coulommiers/online_mainline_2000']


def proc(pid):
    try:
        root = Path('/proc') / str(pid)
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(identity=fields[19], state=fields[0],
                    argv=[s.decode() for s in (root / 'cmdline').read_bytes().split(b'\0') if s])
    except FileNotFoundError:
        return None


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


pending = []
for output in RUNS:
    status = json.loads((output / 'runtime_status.json').read_text())
    pid = status['pid']
    current = proc(pid)
    if current is None:
        continue
    assert str(output) in current['argv'] and os.getpgid(pid) == pid
    initial = len(json.loads((output / 'pilot_progress.json').read_text())['online'])
    pending.append(dict(output=str(output), pid=pid, identity=current['identity'],
                        initial_completed=initial, cooperative='ingolstadt' in str(output)))
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        owner = proc(int(path.name))
        if owner and str(output) in owner['argv'] and any(
                a.endswith('/queue_pretrained_online.py') for a in owner['argv']):
            os.kill(int(path.name), signal.SIGTERM)
    if pending[-1]['cooperative']:
        os.kill(pid, signal.SIGTERM)
write(RECORD / 'request.json', pending)
print(json.dumps(dict(phase='stop_requested', runs=pending)), flush=True)
deadline = time.monotonic() + 900
while pending:
    for row in pending[:]:
        output, pid = Path(row['output']), row['pid']
        current = proc(pid)
        if current is not None:
            assert current['identity'] == row['identity']
        completed = len(json.loads((output / 'pilot_progress.json').read_text())['online'])
        if row['cooperative'] and current is not None:
            continue
        if not row['cooperative'] and current is not None:
            if completed <= row['initial_completed']:
                continue
            os.killpg(pid, signal.SIGSTOP)
            time.sleep(.2)
            assert proc(pid)['state'] in ('T', 't')
        archive = RECORD / ('ingolstadt' if row['cooperative'] else 'coulommiers')
        archive.mkdir()
        os.link(output / 'checkpoints/latest.pt', archive / 'checkpoint.pt')
        for name in ('run_config.toml', 'pilot_protocol.json', 'pilot_progress.json', 'runtime_status.json'):
            (archive / name).write_bytes((output / name).read_bytes())
        if current is not None:
            os.killpg(pid, signal.SIGTERM)
            os.killpg(pid, signal.SIGCONT)
            end = time.monotonic() + 15
            while time.monotonic() < end and proc(pid):
                time.sleep(.2)
            if proc(pid):
                os.killpg(pid, signal.SIGKILL)
        row.update(phase='stopped_at_checkpoint_boundary', completed_episodes=completed,
                   stopped_utc=datetime.now(timezone.utc).isoformat())
        write(archive / 'stop_record.json', row)
        status = json.loads((output / 'runtime_status.json').read_text())
        status.update(phase='stopped_at_checkpoint_boundary', stopped_utc=row['stopped_utc'],
                      stop_reason='user requested fresh periodic route online training')
        write(output / 'runtime_status.json', status)
        print(json.dumps(row), flush=True)
        pending.remove(row)
    if time.monotonic() > deadline:
        raise TimeoutError(f'old online stop not completed: {pending}')
    time.sleep(.5)
write(RECORD / 'completed.json', dict(phase='old_online_stopped', utc=datetime.now(timezone.utc).isoformat()))
