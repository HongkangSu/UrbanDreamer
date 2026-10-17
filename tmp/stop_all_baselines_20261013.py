"""Stop the baseline tasks requested by the user and record the operation."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/baseline_stop_20261013'
OUT.mkdir(exist_ok=True)


def snapshot():
    rows = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().split(') ', 1)[1].split()
            argv = [s.decode(errors='replace') for s in
                    (path / 'cmdline').read_bytes().split(b'\0') if s]
            if fields[0] != 'Z':
                rows[int(path.name)] = dict(pid=int(path.name), ppid=int(fields[1]),
                    identity=fields[19], state=fields[0], argv=argv)
        except (OSError, IndexError):
            pass
    return rows


def baseline(row):
    scripts = [a for a in row['argv'] if a.endswith('.py')]
    return any(str(ROOT / 'outputs/city_baselines_') in a or
               str(ROOT / 'outputs/mappo_score_gauge_recovery_20261013') in a
               for a in scripts)


def alive(row):
    current = snapshot().get(row['pid'])
    return current is not None and current['identity'] == row['identity']


def send(row, sig):
    if alive(row):
        try:
            os.kill(row['pid'], sig)
        except ProcessLookupError:
            pass


def write(path, value):
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


rows = snapshot()
owners = {pid: row for pid, row in rows.items() if baseline(row)}
controllers = {pid: row for pid, row in owners.items() if any(
    a.endswith(('/queued_city_baselines.py', '/long_supervisor.py')) for a in row['argv'])}
targets = dict(owners)
while True:
    children = {pid: row for pid, row in rows.items()
                if row['ppid'] in targets and pid not in targets}
    if not children:
        break
    targets.update(children)
roots = {Path(row['argv'][row['argv'].index('--output-root') + 1])
         for row in controllers.values() if '--output-root' in row['argv']}
supervisor_roots = {Path(a).parent for row in controllers.values()
                    for a in row['argv'] if a.endswith('/long_supervisor.py')}
write(OUT / 'before.json', dict(processes=list(targets.values()),
                              preserved_other_tasks=[r for r in rows.values()
                                  if r['pid'] not in targets and any(
                                      a.endswith('.py') and str(ROOT) in a for a in r['argv'])]))
for row in controllers.values():
    send(row, signal.SIGSTOP)
for row in controllers.values():
    send(row, signal.SIGKILL)
workers = {pid: row for pid, row in targets.items() if pid not in controllers}
for row in workers.values():
    send(row, signal.SIGTERM)
    if row['state'] == 'T':
        send(row, signal.SIGCONT)
deadline = time.monotonic() + 10
while time.monotonic() < deadline and any(alive(row) for row in workers.values()):
    time.sleep(.25)
forced = []
for row in workers.values():
    if alive(row):
        send(row, signal.SIGKILL)
        forced.append(row['pid'])
time.sleep(.5)
remaining = [row['pid'] for row in targets.values() if alive(row)]
assert not remaining, remaining
utc = datetime.now(timezone.utc).isoformat()
reason = 'user requested all currently running baselines stopped; do not auto-resume'
for root in roots:
    for name in ('manifest.json', 'queue_state.json', 'status.json'):
        path = root / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        write(OUT / f'{root.name}_{name}', data)
        data.update(stopped_at=utc, stop_reason=reason)
        if name == 'manifest.json':
            data.update(launch_status='stopped')
        else:
            data.update(controller_pid=None)
            jobs = list(data.get('jobs', {}).values()) if name == 'queue_state.json' else data.get('runs', [])
            for state in jobs:
                if state.get('status') != 'completed':
                    state.update(status='stopped', stopped_pid=state.get('pid'),
                                 pid=None, stopped_at=utc, last_error=reason)
            if name == 'status.json':
                data.update(timestamp=utc, counts=dict(Counter(s['status'] for s in jobs)))
        write(path, data)
for root in supervisor_roots:
    path = root / 'long_status.json'
    if path.exists():
        data = json.loads(path.read_text())
        write(OUT / f'{root.name}_long_status.json', data)
        data.update(status='stopped', stopped_at=utc, stop_reason=reason,
                    stopped_pid=data.get('pid'), pid=None, supervisor_pid=None)
        write(path, data)
report = dict(utc=utc, baseline_processes=len(owners),
              controllers_stopped=len(controllers), all_target_processes=len(targets),
              forced_worker_pids=forced, remaining_baseline_processes=[
                  r for r in snapshot().values() if baseline(r)],
              stopped_queue_roots=sorted(map(str, roots)),
              stopped_mappo_roots=sorted(map(str, supervisor_roots)),
              checkpoints_and_training_data_preserved=True)
write(OUT / 'report.json', report)
print(json.dumps(report, indent=2))
