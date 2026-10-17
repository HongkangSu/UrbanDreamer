"""Resume Ingolstadt after its cooperative stop, with eight rollout workers."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/ingolstadt_full_retrain_20261009/online_bounded_noprior_fresh_t2_2000_20261011'
RECORD = OUTPUT / 'parallel_rollout_handoff_20261013.json'


def utc():
    return time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())


def persist(record):
    record['updated_utc'] = utc()
    temporary = RECORD.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(RECORD)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    record = json.loads(RECORD.read_text())
    if record.get('new_pid'):
        raise RuntimeError('this handoff has already launched a continuation')
    old_pid = record['old_pid']
    proc = Path(f'/proc/{old_pid}')
    if proc.exists():
        command = proc.joinpath('cmdline').read_bytes().decode().split('\0')[:-1]
        if command != record['old_command']:
            raise RuntimeError('old training process identity changed')
        environment = dict(item.split('=', 1) for item in
                           proc.joinpath('environ').read_bytes().decode().split('\0') if '=' in item)
        identity = proc.joinpath('stat').read_text().split(') ', 1)[1].split()[19]
    else:
        raise RuntimeError('cannot capture original runtime environment after process exit')
    record['old_identity'] = identity
    record['controller_pid'] = os.getpid()
    persist(record)
    deadline = time.monotonic() + 1800
    while proc.exists():
        stat = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
        if stat[19] != identity or stat[0] == 'Z':
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('cooperative checkpoint stop did not finish within 30 minutes')
        time.sleep(5)
    runtime = json.loads((OUTPUT / 'runtime_status.json').read_text())
    if runtime['pid'] != old_pid or runtime['phase'] != 'stopped_at_checkpoint_boundary':
        raise RuntimeError(f'old training did not stop at a checkpoint boundary: {runtime}')
    checkpoint = OUTPUT / 'checkpoints/latest.pt'
    backup = OUTPUT / 'checkpoints/before_parallel_20261013.pt'
    os.link(checkpoint, backup)
    progress = json.loads((OUTPUT / 'pilot_progress.json').read_text())
    record.update(phase='checkpoint_preserved', resume_episode=len(progress['online']),
                  preserved_checkpoint=str(backup), checkpoint_sha256=digest(backup),
                  config_sha256=digest(OUTPUT / 'run_config.toml'),
                  protocol_sha256=digest(OUTPUT / 'pilot_protocol.json'))
    persist(record)
    environment['URB_IMAGINATION_WORKERS'] = '8'
    log_path = OUTPUT / 'online_parallel_20261013.log'
    with log_path.open('ab', buffering=0) as log:
        child = subprocess.Popen(record['old_command'], cwd=ROOT, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record.update(phase='continuation_launched', new_pid=child.pid, launch_utc=utc(),
                  log=str(log_path), command=record['old_command'])
    persist(record)
    while child.poll() is None:
        current = json.loads((OUTPUT / 'pilot_progress.json').read_text())
        if len(current['online']) > record['resume_episode']:
            record.update(phase='first_parallel_episode_completed',
                          completed_episodes=len(current['online']),
                          first_episode_seconds=current['online'][record['resume_episode']]['seconds'])
            persist(record)
            return
        time.sleep(10)
    raise RuntimeError(f'parallel continuation exited with code {child.returncode}')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        record = json.loads(RECORD.read_text())
        record.update(phase='handoff_failed', error=repr(error))
        persist(record)
        raise
