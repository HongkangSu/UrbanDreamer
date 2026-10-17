"""Finish the first-episode audit when all three asynchronous startups are ready."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'outputs/urbandreamer_current_to_dest_1000_fp16_w8_r8_p2_20261013'
PYTHON = '/home/wangwenrui/opt/miniconda3/envs/URB/bin/python'
status = dict(pid=os.getpid(), phase='waiting_for_first_episodes')
while True:
    cities = []
    for city in ('ingolstadt_custom', 'coulommiers', 'montereau_fault_yonne'):
        row = json.loads((BATCH / city / 'status.json').read_text())
        cities.append(dict(city=city, status=row['status'], phase=row.get('phase'),
                           completed_episodes=row.get('completed_episodes', 0)))
    status.update(cities=cities, updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
    if any(row['status'] == 'failed' for row in cities):
        status['phase'] = 'training_start_failed'
    elif all(row['completed_episodes'] > 0 for row in cities):
        result = subprocess.run([PYTHON, str(ROOT / 'tmp/audit_route_online_start_20261013.py')],
                                cwd=ROOT, capture_output=True, text=True)
        print(result.stdout, flush=True)
        print(result.stderr, flush=True)
        status.update(phase='first_episodes_verified' if result.returncode == 0 else 'audit_failed',
                      audit_exit_code=result.returncode)
    temporary = BATCH / '.startup_monitor.tmp'
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(BATCH / 'startup_monitor.json')
    if status['phase'] != 'waiting_for_first_episodes':
        break
    time.sleep(15)
