"""Extend Ingolstadt's completed run; preserve its 1000-episode learning clock."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/wangwenrui/shk/UrbanDreamer/urban_dreamer')
BATCH = ROOT / 'outputs/urbandreamer_current_to_dest_1000_fp16_w8_r8_p2_20261013'
CITY = BATCH / 'ingolstadt_custom'
OUTPUT = CITY / 'online'
PYTHON = '/home/wangwenrui/opt/miniconda3/envs/URB/bin/python'


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare():
    progress = json.loads((OUTPUT / 'pilot_progress.json').read_text())
    status = json.loads((CITY / 'status.json').read_text())
    assert progress['phase'] == 'completed' and len(progress['online']) == 1000
    assert status['status'] == 'completed' and status['exit_code'] == 0
    if status.get('pid') and Path(f'/proc/{status["pid"]}').exists():
        fields = Path(f'/proc/{status["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
        assert fields[0] == 'Z'
    root = CITY / 'budget_extensions' / ('2000_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    root.mkdir(parents=True)
    os.link(OUTPUT / 'checkpoints/latest.pt', root / 'completed_1000_checkpoint.pt')
    for name in ('run_config.toml', 'pilot_protocol.json', 'pilot_progress.json', 'pilot_result.json',
                 'effective_run_budget.json'):
        shutil.copy2(OUTPUT / name, root / name)
    shutil.copy2(CITY / 'status.json', root / 'old_status.json')
    source = Path(status['source'])
    frozen = json.loads((source / 'source_manifest.json').read_text())
    assert all(digest(source / name) == value for name, value in frozen['files'].items())
    command = list(status['command'])
    command[command.index('--online-episodes') + 1] = '2000'
    command.append('--extend-run')
    record = dict(kind='completed_online_run_budget_extension', city='ingolstadt_custom',
        from_episodes=1000, target_episodes=2000, entropy_schedule_episodes=1000,
        checkpoint_sha256=digest(root / 'completed_1000_checkpoint.pt'),
        config_sha256=digest(OUTPUT / 'run_config.toml'),
        source=str(source), source_manifest_sha256=digest(source / 'source_manifest.json'),
        cache_directory=status['cache_directory'], command=command, created_utc=utc(),
        preserved=['models', 'optimizers', 'replay', 'RNG', 'training_history', '1000_episode_evaluations'],
        settings=dict(precision='fp16', workers=8, rollouts=8, imagination_passes=2,
                      routing_mode='current_to_dest', shared_cache_capacity=200000),
        admission=dict(device='cuda:1', minimum_free_mib=4096, stable_seconds=20),
        record_dir=str(root))
    write(root / 'extension.json', record)
    write(CITY / 'budget_extension_2000.json', record)
    shutil.copy2(__file__, root / 'controller.py')
    with (root / 'controller.log').open('ab', buffering=0) as log:
        controller = subprocess.Popen([PYTHON, str(root / 'controller.py'), 'supervise', str(root)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record.update(controller_pid=controller.pid, status='queued_for_gpu')
    write(CITY / 'budget_extension_2000.json', record)
    manifest = json.loads((BATCH / 'manifest.json').read_text())
    manifest['online_episodes_by_city'] = dict(ingolstadt_custom=2000, coulommiers=1000, montereau_fault_yonne=1000)
    for entry in manifest['entries']:
        if entry['city'] == 'ingolstadt_custom':
            entry.update(command=command, online_episodes=2000, budget_extension=str(root / 'extension.json'))
    write(BATCH / 'manifest.json', manifest)
    print(json.dumps(record, indent=2), flush=True)


def supervise(root):
    record = json.loads((root / 'extension.json').read_text())
    minimum = record['admission']['minimum_free_mib']
    stable_since = None
    attempts = 0
    previous_status = json.loads((root / 'old_status.json').read_text())
    status = dict(previous_status, pid=None, supervisor_pid=os.getpid(), status='waiting_for_gpu',
        phase='waiting_to_extend_completed_run', target_episodes=2000, restored_episode=1000,
        completed_episodes=1000, extension_record=str(root / 'extension.json'), command=record['command'])
    while True:
        free = int(subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.free',
                                           '--format=csv,noheader,nounits'], text=True).strip())
        if free >= minimum:
            stable_since = time.monotonic() if stable_since is None else stable_since
        else:
            stable_since = None
        status.update(updated_utc=utc(), free_gpu_mib=free, required_free_mib=minimum, launch_attempts=attempts)
        write(CITY / 'status.json', status)
        if stable_since is None or time.monotonic()-stable_since < 20:
            time.sleep(5)
            continue
        source = Path(record['source'])
        frozen = json.loads((source / 'source_manifest.json').read_text())
        assert digest(source / 'source_manifest.json') == record['source_manifest_sha256']
        assert all(digest(source / name) == value for name, value in frozen['files'].items())
        assert digest(OUTPUT / 'run_config.toml') == record['config_sha256']
        if attempts == 0:
            assert digest(OUTPUT / 'checkpoints/latest.pt') == record['checkpoint_sha256']
        environment = dict(os.environ, URB_IMAGINATION_WORKERS='8',
            URB_ROUTE_CACHE_DIR=record['cache_directory'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='2',
            TORCHINDUCTOR_CACHE_DIR=str(CITY / 'inductor_cache'))
        environment.pop('URB_ACTOR_ENTROPY_MODE', None)
        log_path = CITY / 'online_extend_2000_20261014.log'
        offset = log_path.stat().st_size if log_path.exists() else 0
        with log_path.open('ab', buffering=0) as log:
            child = subprocess.Popen(record['command'], cwd=source, env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        attempts += 1
        status.update(pid=child.pid, status='running', phase='initializing_extension',
                      started_utc=utc(), exit_code=None, launch_attempts=attempts)
        while True:
            code = child.poll()
            progress = json.loads((OUTPUT / 'pilot_progress.json').read_text())
            count = len(progress['online'])
            status.update(completed_episodes=count, updated_utc=utc(), exit_code=code,
                phase=progress['phase'] if count > 1000 or progress['phase'] != 'completed' else 'initializing_extension',
                status='running' if code is None else 'completed' if code == 0 else 'failed')
            write(CITY / 'status.json', status)
            if code is not None:
                break
            time.sleep(5)
        if code == 0:
            return
        with log_path.open('rb') as log:
            log.seek(max(offset, log_path.stat().st_size - 24000))
            tail = log.read().decode(errors='replace')
        if attempts >= 3 or not ('CUDA out of memory' in tail or 'less than2GiB' in tail):
            return
        minimum += 2048
        stable_since = None
        status.update(pid=None, status='waiting_for_gpu', phase='retry_after_gpu_memory_failure')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'supervise':
        supervise(Path(sys.argv[2]))
    else:
        prepare()
