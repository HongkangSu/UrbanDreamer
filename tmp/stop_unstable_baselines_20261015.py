"""Stop only the five requested methods, retaining process and queue evidence."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'baseline_stability_20261015'
METHODS = {'ippo', 'mappo', 'mamba', 'mbpo', 'dreamer'}


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


def method(row):
    argv = row['argv']
    if not any(a.endswith('/train_toy_baseline.py') and str(ROOT) in a for a in argv):
        return None
    return argv[argv.index('--method') + 1] if '--method' in argv else None


def send(row, sig):
    current = snapshot().get(row['pid'])
    if current and current['identity'] == row['identity']:
        os.kill(row['pid'], sig)


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.stopping.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


OUT.mkdir(parents=True, exist_ok=True)
rows = snapshot()
targets = {p: r for p, r in rows.items() if method(r) in METHODS}
controllers = {p: r for p, r in rows.items() if any(
    a.endswith('/queued_city_baselines.py') and str(ROOT) in a for a in r['argv'])
    and 'run' in r['argv']}
write(OUT / 'processes_before.json', dict(targets=list(targets.values()),
    controllers=list(controllers.values()), preserved=[r for r in rows.values()
    if method(r) and method(r) not in METHODS]))
for row in controllers.values():
    send(row, signal.SIGSTOP)
for row in controllers.values():
    send(row, signal.SIGTERM)
    send(row, signal.SIGCONT)
while True:
    children = {p: r for p, r in rows.items() if r['ppid'] in targets and p not in targets}
    if not children:
        break
    targets.update(children)
for row in targets.values():
    send(row, signal.SIGTERM)
    if row['state'] == 'T':
        send(row, signal.SIGCONT)
deadline = time.monotonic() + 10
while time.monotonic() < deadline and set(targets).intersection(snapshot()):
    time.sleep(.25)
forced = []
for row in targets.values():
    current = snapshot().get(row['pid'])
    if current and current['identity'] == row['identity']:
        send(row, signal.SIGKILL)
        forced.append(row['pid'])
utc = datetime.now(timezone.utc).isoformat()
reason = 'User stopped unstable IPPO/MAPPO/MAMBA/MBPO/Dreamer; preserve artifacts; no auto-resume'
roots = set()
for row in controllers.values():
    argv = row['argv']
    root = Path(argv[argv.index('--output-root') + 1])
    roots.add(str(root))
    manifest = json.loads((root / 'manifest.json').read_text())
    write(OUT / (root.name + '_manifest_before.json'), manifest)
    manifest.update(launch_status='stopped', stopped_at=utc, stop_reason=reason)
    write(root / 'manifest.json', manifest)
    ids = {r['id'] for r in manifest['runs'] if r['method'] in METHODS}
    for name in ('queue_state.json', 'status.json'):
        path = root / name
        data = json.loads(path.read_text())
        write(OUT / (root.name + '_' + name), data)
        data.update(controller_pid=None, stopped_at=utc, stop_reason=reason)
        states = data['jobs'].items() if name == 'queue_state.json' else ((r['id'], r) for r in data['runs'])
        for run_id, state in states:
            if run_id in ids and state['status'] != 'completed':
                state.update(status='stopped', stopped_pid=state.get('pid'), pid=None, stopped_at=utc)
        if name == 'status.json':
            data['counts'] = dict(Counter(r['status'] for r in data['runs']))
        write(path, data)
time.sleep(.5)
remaining = [r for r in snapshot().values() if method(r) in METHODS]
report = dict(utc=utc, stopped_training_count=sum(method(r) in METHODS for r in targets.values()),
    stopped_by_method=dict(Counter(method(r) for r in targets.values() if method(r))),
    stopped_controller_count=len(controllers), forced_pids=forced, remaining=remaining,
    queue_roots=sorted(roots), checkpoints_and_archives_preserved=True)
write(OUT / 'stop_report.json', report)
print(json.dumps(report, indent=2))
assert not remaining, remaining
