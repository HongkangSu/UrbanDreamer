"""Run a frozen experiment queue without interfering with existing jobs.

Each entry carries an explicit command and output log. Failed jobs remain failed;
independent later entries can run, while dependent entries are marked blocked.
This is a launcher, never an assertion that experiments have been completed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.component_experiment_data import atomic_json, sha256


def utc():
    return datetime.now(timezone.utc).isoformat()


def free_gpu_mib(index):
    result = subprocess.run(['nvidia-smi', f'--id={index}', '--query-gpu=memory.free',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True)
    return int(result.stdout.strip())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    args = p.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    directory = plan_path.parent
    lock = (directory/'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = directory/'queue_status.json'
    if state_path.exists():
        raise ValueError('existing queue state requires a separately audited continuation plan')
    state = dict(phase='running', pid=os.getpid(), started_utc=utc(), plan_sha256=sha256(plan_path), jobs={})
    atomic_json(state_path, state)
    for job in plan['jobs']:
        if sha256(plan_path) != state['plan_sha256']:
            raise ValueError('queue plan changed while running')
        missing = [name for name in job.get('depends_on', [])
                   if state['jobs'].get(name, {}).get('phase') != 'completed']
        if missing:
            state['jobs'][job['name']] = dict(phase='blocked', dependencies=missing)
            atomic_json(state_path, state)
            continue
        state['current_job'] = job['name']
        row = state['jobs'][job['name']] = dict(phase='waiting_resources', command=job['command'])
        while free_gpu_mib(job.get('gpu', 1)) < job.get('minimum_free_mib', 2816):
            row['heartbeat_utc'] = utc()
            atomic_json(state_path, state)
            time.sleep(30)
        log_path = directory / f"{job['name']}.log"
        with log_path.open('x') as stream:
            child = subprocess.Popen(job['command'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            row.update(phase='running', pid=child.pid, started_utc=utc(), log=str(log_path))
            atomic_json(state_path, state)
            while child.poll() is None:
                row['heartbeat_utc'] = utc()
                atomic_json(state_path, state)
                time.sleep(10)
            row.update(phase='completed' if child.returncode == 0 else 'failed',
                       returncode=child.returncode, finished_utc=utc())
            atomic_json(state_path, state)
    state['phase'] = ('completed' if all(row['phase']=='completed' for row in state['jobs'].values())
                      else 'finished_with_failures')
    state['finished_utc'] = utc()
    atomic_json(state_path, state)


if __name__ == '__main__':
    main()
