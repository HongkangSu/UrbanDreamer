"""Resume the two OOM-stopped runs from their unchanged committed checkpoints."""
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
PYTHON = '/home/wangwenrui/opt/miniconda3/envs/URB/bin/python'
CITIES = ('coulommiers', 'montereau_fault_yonne')


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def live_runners(output):
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [s.decode() for s in (proc / 'cmdline').read_bytes().split(b'\0') if s]
            state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
            if state != 'Z' and str(output) in argv and any(a.endswith('/run_online_pilot.py') for a in argv):
                found.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return found


def start():
    root = BATCH / 'recoveries' / ('two_city_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, root / 'controller.py')
    entries = []
    for i, city in enumerate(CITIES):
        folder, output = BATCH / city, BATCH / city / 'online'
        previous = json.loads((folder / 'status.json').read_text())
        assert previous['status'] == 'failed' and not live_runners(output)
        handoff = json.loads(Path(previous['handoff']).read_text())
        progress = json.loads((output / 'pilot_progress.json').read_text())
        completed = len(progress['online'])
        assert completed == handoff['completed_episodes']
        assert digest(output / 'checkpoints/latest.pt') == handoff['checkpoint_sha256']
        assert digest(output / 'run_config.toml') == handoff['config_sha256']
        source = Path(previous['source'])
        frozen = json.loads((source / 'source_manifest.json').read_text())
        assert all(digest(source / p) == value for p, value in frozen['files'].items())
        backup = root / city
        backup.mkdir()
        os.link(output / 'checkpoints/latest.pt', backup / 'checkpoint.pt')
        for name in ('run_config.toml', 'pilot_protocol.json', 'pilot_progress.json'):
            shutil.copy2(output / name, backup / name)
        shutil.copy2(folder / 'status.json', backup / 'old_status.json')
        failure = output.with_suffix('.failure.json')
        if failure.exists():
            shutil.copy2(failure, backup / 'prior_failure.json')
        entry = dict(city=city, output=str(output), source=str(source), command=previous['command'],
            restored_episode=completed, target_episodes=1000, checkpoint_sha256=handoff['checkpoint_sha256'],
            config_sha256=handoff['config_sha256'], source_manifest_sha256=digest(source / 'source_manifest.json'),
            cache_directory=previous['cache_directory'], shared_route_cache_capacity=200000,
            workers=8, rollouts=8, imagination_passes=2, precision='fp16',
            created_utc=utc(), minimum_free_mib=7680 if i == 0 else 6144,
            not_before_unix=time.time()+60*i, stable_seconds=15)
        assert entry['command'][entry['command'].index('--online-episodes')+1] == '1000'
        write(backup / 'request.json', entry)
        entries.append(entry)
    write(root / 'requests.json', entries)
    for entry in entries:
        with (root / entry['city'] / 'controller.log').open('ab', buffering=0) as log:
            controller = subprocess.Popen([PYTHON, str(root / 'controller.py'), 'supervise', str(root), entry['city']],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        entry['supervisor_pid'] = controller.pid
    record = dict(requested_utc=utc(), record_dir=str(root), entries=entries,
        preserved=['model_weights','optimizers','replay','RNG','episode_clock','configuration','shared_route_cache'])
    write(BATCH / 'two_city_recovery_20261015.json', record)
    print(json.dumps(record, indent=2), flush=True)


def supervise(root, city):
    entry = json.loads((root / city / 'request.json').read_text())
    folder, output, source = BATCH / city, Path(entry['output']), Path(entry['source'])
    minimum = entry['minimum_free_mib']
    attempts, stable_since = 0, None
    status = dict(entry, pid=None, supervisor_pid=os.getpid(), status='waiting_for_gpu',
                  completed_episodes=entry['restored_episode'], phase='waiting_for_gpu',
                  recovery_record=str(root / city / 'request.json'))
    while True:
        free = int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free',
                                           '--format=csv,noheader,nounits'], text=True).strip())
        admitted = free >= minimum and time.time() >= entry['not_before_unix']
        stable_since = (time.monotonic() if stable_since is None else stable_since) if admitted else None
        status.update(updated_utc=utc(), free_gpu_mib=free, required_free_mib=minimum, launch_attempts=attempts)
        write(folder / 'status.json', status)
        if stable_since is None or time.monotonic()-stable_since < entry['stable_seconds']:
            time.sleep(5)
            continue
        assert not live_runners(output)
        assert digest(source / 'source_manifest.json') == entry['source_manifest_sha256']
        assert digest(output / 'run_config.toml') == entry['config_sha256']
        if attempts == 0:
            assert digest(output / 'checkpoints/latest.pt') == entry['checkpoint_sha256']
        environment = dict(os.environ, URB_IMAGINATION_WORKERS='8', URB_ROUTE_CACHE_DIR=entry['cache_directory'],
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
            TORCHINDUCTOR_COMPILE_THREADS='2', TORCHINDUCTOR_CACHE_DIR=str(folder / 'inductor_cache'))
        environment.pop('URB_ACTOR_ENTROPY_MODE', None)
        log_path = folder / 'online_recovery_20261015.log'
        offset = log_path.stat().st_size if log_path.exists() else 0
        with log_path.open('ab', buffering=0) as log:
            child = subprocess.Popen(entry['command'], cwd=source, env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        attempts += 1
        identity = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        status.update(pid=child.pid, identity=identity, status='running', phase='initializing',
                      started_utc=utc(), exit_code=None, launch_attempts=attempts)
        while True:
            code = child.poll()
            progress = json.loads((output / 'pilot_progress.json').read_text())
            count = len(progress['online'])
            with log_path.open('rb') as log:
                log.seek(max(offset, log_path.stat().st_size-12000))
                tail = log.read().decode(errors='replace')
            begun = count > entry['restored_episode'] or 'real_episode_started' in tail or 'actor_critic_pass' in tail
            status.update(completed_episodes=count, updated_utc=utc(), exit_code=code,
                phase=progress['phase'] if begun else 'initializing',
                status='running' if code is None else 'completed' if code == 0 else 'failed')
            if count > entry['restored_episode']:
                first = progress['online'][entry['restored_episode']]
                assert first['optimizer_steps'] == dict(actor=2,event_critic=2,latent_critic=2)
                status['resumed_episode_verified'] = True
                status['first_resumed_episode_seconds'] = first['seconds']
            write(folder / 'status.json', status)
            if code is not None:
                break
            time.sleep(5)
        write(root / city / f'attempt_{attempts}_exit.json', dict(status=status, log_tail=tail))
        if code == 0 or attempts >= 3 or not ('CUDA out of memory' in tail or 'less than2GiB' in tail):
            return
        minimum += 2048
        stable_since = None
        status.update(pid=None, status='waiting_for_gpu', phase='retry_after_oom')


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='supervise':
        supervise(Path(sys.argv[2]), sys.argv[3])
    else:
        start()
