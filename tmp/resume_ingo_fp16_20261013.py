"""Migrate Ingolstadt imagination precision at its saved episode boundary."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/ingolstadt_full_retrain_20261009/online_bounded_noprior_fresh_t2_2000_20261011'
SOURCE = OUTPUT.parent / 'source_bounded_noprior_argmax_fp16_20261013'
RECORD = OUTPUT / 'fp16_rollout_handoff_20261013.json'
PYTHON = '/home/wangwenrui/opt/miniconda3/envs/URB/bin/python'


def persist(record):
    record['updated_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    temporary = RECORD.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(RECORD)


def main():
    record = json.loads(RECORD.read_text())
    if record.get('new_pid'):
        raise RuntimeError('this precision handoff has already launched')
    proc = Path('/proc', str(record['old_pid']))
    deadline = time.monotonic() + 1800
    record['controller_pid'] = os.getpid()
    persist(record)
    while proc.exists():
        fields = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
        if fields[19] != record['old_identity'] or fields[0] == 'Z':
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('checkpoint boundary stop exceeded 30 minutes')
        time.sleep(5)
    runtime = json.loads((OUTPUT / 'runtime_status.json').read_text())
    if runtime['pid'] != record['old_pid'] or runtime['phase'] != 'stopped_at_checkpoint_boundary':
        raise RuntimeError(f'run did not stop at checkpoint boundary: {runtime}')
    manifest = json.loads((SOURCE / 'source_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'frozen source changed: {name}')
    record.update(phase='migrating_precision', source_root=str(SOURCE))
    persist(record)
    environment = dict(os.environ, **record['runtime_environment'])
    environment['URB_IMAGINATION_WORKERS'] = '8'
    with (OUTPUT / 'fp16_profile_migration_20261013.log').open('ab', buffering=0) as log:
        subprocess.run([PYTHON, '-u', str(SOURCE / 'tools/change_online_training_profile.py'),
                        '--run-dir', str(OUTPUT), '--imagination-precision', 'fp16'],
                       cwd=SOURCE, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    change_path = OUTPUT / 'latest_profile_change.json'
    change = json.loads(change_path.read_text())
    if change['before']['imagination_precision'] != 'bf16' or change['after']['imagination_precision'] != 'fp16':
        raise RuntimeError('unexpected precision migration')
    for name in ('rollouts', 'evaluation_prior', 'hold', 'off'):
        if change['before'][name] != change['after'][name]:
            raise RuntimeError(f'precision migration also changed {name}')
    protocol_path = OUTPUT / 'pilot_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    old_source = protocol['source_root']
    if Path(old_source) / 'tools/run_bounded_online_continuation.py' != Path(record['old_command'][2]):
        raise RuntimeError('saved execution source differs from stopped process')
    protocol['source_root'] = str(SOURCE)
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    change['execution_source_change'] = dict(before=old_source, after=str(SOURCE),
        source_manifest_sha256=hashlib.sha256((SOURCE / 'source_manifest.json').read_bytes()).hexdigest())
    change_path.write_text(json.dumps(change, indent=2) + '\n')
    (Path(change['backup']) / 'change.json').write_text(json.dumps(change, indent=2) + '\n')
    record.update(phase='precision_migrated', resume_episode=change['boundary'], profile_change=change)
    persist(record)
    command = list(record['old_command'])
    command[2] = str(SOURCE / 'tools/run_bounded_online_continuation.py')
    log_path = OUTPUT / 'online_parallel_fp16_20261013.log'
    with log_path.open('ab', buffering=0) as log:
        child = subprocess.Popen(command, cwd=SOURCE, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record.update(phase='fp16_continuation_launched', new_pid=child.pid,
                  command=command, log=str(log_path))
    persist(record)
    while child.poll() is None:
        progress = json.loads((OUTPUT / 'pilot_progress.json').read_text())
        if len(progress['online']) > record['resume_episode']:
            row = progress['online'][record['resume_episode']]
            record.update(phase='first_fp16_episode_completed', completed_episodes=len(progress['online']),
                          first_episode_seconds=row['seconds'],
                          first_imagination_seconds=row['metrics']['imagination_seconds'])
            persist(record)
            return
        time.sleep(10)
    raise RuntimeError(f'FP16 continuation exited with code {child.returncode}')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        record = json.loads(RECORD.read_text())
        record.update(phase='handoff_failed', error=repr(error))
        persist(record)
        raise
