"""Stop only the user-requested city batch and its queue owners."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.queued_city_baselines import owned_alive, process_argv, process_identity, write_json, durable_episode
BATCH=ROOT/'outputs/city_baselines_noincident_prior_penalty_5000ep_20261007'
manifest=json.loads((BATCH/'manifest.json').read_text())
ledger=json.loads((BATCH/'queue_state.json').read_text())
entries={e['id']:e for e in manifest['runs']}
workers={k:s.copy() for k,s in ledger['jobs'].items() if s.get('pid') and owned_alive(s,entries[k])}
controllers={}
for path in Path('/proc').iterdir():
    if not path.name.isdigit(): continue
    pid=int(path.name); argv=process_argv(pid)
    if str(BATCH) in argv and any(a.endswith(('/queued_city_baselines.py','/compact_episode_archives.py')) for a in argv):
        identity=process_identity(pid)
        if identity: controllers[pid]=identity
# Stop all owners first, including paused previous owners, preventing admissions.
for pid,identity in controllers.items():
    if process_identity(pid)==identity: os.kill(pid,signal.SIGKILL)
for key,state in workers.items():
    if owned_alive(state,entries[key]): os.kill(state['pid'],signal.SIGTERM)
deadline=time.monotonic()+15
while time.monotonic()<deadline and any(owned_alive(s,entries[k]) for k,s in workers.items()):
    time.sleep(.25)
forced=[]
for key,state in workers.items():
    if owned_alive(state,entries[key]):
        os.kill(state['pid'],signal.SIGKILL); forced.append(key)
time.sleep(.3)
alive=[k for k,s in workers.items() if owned_alive(s,entries[k])]
assert not alive,alive
timestamp=datetime.now(timezone.utc).isoformat()
for key,state in ledger['jobs'].items():
    if state['status']!='completed':
        state.update(status='stopped',stopped_pid=state.get('pid'),pid=None,
                     stopped_at=timestamp,last_error='user requested stop; do not auto-resume')
ledger.update(controller_pid=None,stopped_at=timestamp)
manifest.update(launch_status='stopped',stopped_at=timestamp,stop_reason='User requested stop and disable all shortest-path priors before throughput diagnosis.')
write_json(BATCH/'queue_state.json',ledger)
write_json(BATCH/'manifest.json',manifest)
status=json.loads((BATCH/'status.json').read_text())
for row in status['runs']:
    row.update(ledger['jobs'][row['id']])
    row['committed_episode']=durable_episode(entries[row['id']])
    write_json(entries[row['id']]['state'],row)
status.update(timestamp=timestamp,controller_pid=None,counts=dict(Counter(r['status'] for r in status['runs'])))
write_json(BATCH/'status.json',status)
report=dict(timestamp=timestamp,controllers_stopped=list(controllers),workers_stopped=len(workers),forced_workers=forced,
            preserved_checkpoints=True,remaining_owned_workers=alive,counts=status['counts'])
write_json(BATCH/'stop_report_20261008.json',report)
print(json.dumps(report,indent=2))
