"""Freeze and run city baselines, admitting GPU workers only when memory fits."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CITIES = ('ingolstadt_custom', 'coulommiers', 'montereau_fault_yonne')
LEARNED = ('online_ac', 'gnac', 'dreamer', 'mbpo', 'ippo', 'mappo', 'iql', 'mamba')
METHODS = LEARNED
TERMINAL = {'completed', 'failed', 'blocked_gpu_capacity'}
GPU_REQUIREMENTS = {
    'gnac': (4096, 8192), 'dreamer': (3072, 5120), 'mbpo': (4096, 8192),
    'ippo': (1536, 2048), 'mappo': (2048, 3072), 'iql': (1536, 2048),
    'mamba': (4096, 6144),
}
STEADY_FLOORS = {'gnac':2048, 'dreamer':2048, 'mbpo':2048,
                 'ippo':1280, 'mappo':1024, 'iql':1024, 'mamba':2048}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def effective_gpu_free_mib(gpu, running, allocation_by_pid, headroom_mib=1024):
    pending = sum(max(0., row['required_gpu_mib'] - allocation_by_pid.get(row['pid'], 0.))
                  for row in running if row.get('gpu_index') == gpu['index'])
    return max(0., gpu['free_mib'] - headroom_mib - pending)


def choose_gpu(required_mib, gpus, running, allocation_by_pid, headroom_mib=1024):
    candidates = [(effective_gpu_free_mib(gpu, running, allocation_by_pid, headroom_mib), gpu['index'])
                  for gpu in gpus]
    fits = [(free, index) for free, index in candidates if free >= required_mib]
    return max(fits)[1] if fits else None


def failure_kind(log_text, exit_code):
    lower = log_text.lower()
    patterns = ('cuda out of memory', 'cuda error: out of memory',
                'cublas_status_alloc_failed', 'torch.outofmemoryerror')
    return 'oom' if any(pattern in lower for pattern in patterns) else 'failed'


def bump_gpu_requirement(required_mib, observed_peak_mib, max_usable_mib):
    return min(max_usable_mib, math.ceil(max(required_mib * 1.5, observed_peak_mib * 1.25 + 512)))


def steady_gpu_requirement(method, peak_mib, current_requirement, committed_episode,
                           oom_retries=0, warmup_complete=True):
    """Replace startup guesses only after real updates cover steady operation."""
    if peak_mib <= 0 or committed_episode < 50 or not warmup_complete:
        return current_requirement
    floor = STEADY_FLOORS.get(method, 1024)
    estimate = 256 * math.ceil(max(floor, peak_mib * 1.15 + 256) / 256)
    return max(estimate, current_requirement) if oom_retries else estimate


def calibrate_memory_budgets(entries, jobs, gpus, allocations, gpu_headroom_mib=1024):
    max_usable = max((g['total_mib'] for g in gpus), default=24576) - gpu_headroom_mib
    for run_id, state in jobs.items():
        entry = entries[run_id]
        if state['status'] != 'running' or not state['required_gpu_mib']:
            continue
        try:
            episode = durable_episode(entry)
        except (OSError, ValueError, StopIteration):
            continue
        warmup = True
        if entry['method'] == 'mamba':
            path = Path(entry['output_dir'])/'metrics.jsonl'
            if path.exists():
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size-2097152))
                    tail = stream.read()
                warmup = bool(re.search(rb'"model_warmup_complete"\s*:\s*1', tail))
            else:
                warmup = False
        estimate = steady_gpu_requirement(entry['method'], state['observed_peak_mib'],
            state['required_gpu_mib'], episode, state['oom_retries'], warmup)
        estimate = min(max_usable, estimate)
        if estimate != state['required_gpu_mib']:
            state.update(required_gpu_mib=estimate, memory_calibrated_at=now(),
                         memory_calibration_episode=episode)
    # IQL's learner is minibatch512 and its replay stays on CPU. IPPO's
    # initial budget can use the same-size city's observed footprint, scaling
    # by population; OOM retries are never downgraded by this estimate.
    for run_id, state in jobs.items():
        entry = entries[run_id]
        if state['attempt'] or state['oom_retries'] or entry['city']!='montereau_fault_yonne':
            continue
        peer = jobs.get('coulommiers_'+entry['method'])
        if not peer or not peer.get('memory_calibration_episode', 0)>=50:
            continue
        if entry['method']=='iql':
            estimate = max(1280, 256*math.ceil((peer['observed_peak_mib']*1.15+256)/256))
        elif entry['method']=='ippo':
            estimate = max(1536, 256*math.ceil((peer['observed_peak_mib']*1.5*1.15+256)/256))
        else:
            continue
        state.update(required_gpu_mib=min(state['required_gpu_mib'], estimate),
                     peer_memory_calibration='coulommiers_'+entry['method'])


def apply_admission_overrides(root, manifest, entries, jobs, gpus=None):
    """Reload resource-only controls; never change learner commands or OOM floors."""
    path = root/'admission_overrides.json'
    if not path.exists():
        return False
    policy = json.loads(path.read_text())
    budgets = policy.get('required_gpu_mib', {})
    headroom = policy.get('gpu_headroom_mib', manifest['gpu_headroom_mib'])
    poll = policy.get('poll_seconds', manifest['poll_seconds'])
    margin = policy.get('peak_margin_fraction', .10)
    padding = policy.get('peak_padding_mib', 128)
    if (not isinstance(budgets, dict) or set(budgets)-set(entries)
            or not isinstance(headroom, (int,float)) or not math.isfinite(headroom) or headroom < 0
            or not isinstance(poll, (int,float)) or not math.isfinite(poll) or poll <= 0
            or not isinstance(margin, (int,float)) or not math.isfinite(margin) or margin < 0
            or not isinstance(padding, (int,float)) or not math.isfinite(padding) or padding < 0):
        raise ValueError('invalid admission resource overrides')
    if any(not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0
           or not entries[key].get('required_gpu_mib') for key,value in budgets.items()):
        raise ValueError('GPU override must name a GPU job with a positive budget')
    changed = (manifest['gpu_headroom_mib'] != headroom or manifest['poll_seconds'] != poll
               or manifest.get('active_admission_policy') != policy)
    maximum = max((gpu['total_mib'] for gpu in (gpus or [])), default=24576)-headroom
    manifest.update(gpu_headroom_mib=headroom, poll_seconds=poll)
    for key, minimum in budgets.items():
        state = jobs[key]
        peak = state.get('observed_peak_mib', 0)
        estimate = 256 * math.ceil((peak*(1+margin)+padding)/256) if peak else 0
        needed = max(minimum, estimate)
        if state.get('oom_retries', 0):
            needed = max(needed, state['required_gpu_mib'])
        needed = min(maximum, needed)
        state.update(required_gpu_mib=needed, admission_override=True)
        if entries[key]['required_gpu_mib'] != needed:
            entries[key]['required_gpu_mib'] = needed
            changed = True
    manifest['active_admission_policy'] = policy
    if changed:
        write_json(root/'manifest.json', manifest)
    return True


def query_gpus():
    fields = 'index,uuid,memory.total,memory.free'
    text = subprocess.check_output(['nvidia-smi', f'--query-gpu={fields}',
                                    '--format=csv,noheader,nounits'], text=True, timeout=20)
    gpus = [dict(index=int(row[0]), uuid=row[1].strip(), total_mib=float(row[2]), free_mib=float(row[3]))
            for row in csv.reader(io.StringIO(text)) if row]
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'], text=True, timeout=20)
    allocations = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) == 2 and row[0].strip().isdigit():
            try:
                allocations[int(row[0])] = allocations.get(int(row[0]), 0.) + float(row[1])
            except ValueError:
                pass
    return gpus, allocations


def process_identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def process_argv(pid):
    try:
        return [s.decode(errors='replace') for s in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0') if s]
    except OSError:
        return []


def adopted_exit_code(state):
    """Read a retained child's wait status after a controller handoff on Linux."""
    try:
        fields = (Path('/proc') / str(state['pid']) / 'stat').read_text().split(') ', 1)[1].split()
        if fields[0] == 'Z' and fields[19] == state['process_starttime']:
            return os.waitstatus_to_exitcode(int(fields[49]))
    except (OSError, IndexError, ValueError):
        pass
    return None


def owned_alive(state, entry):
    pid = state.get('pid')
    argv = process_argv(pid) if pid else []
    return bool(pid and process_identity(pid) == state.get('process_starttime')
                and entry['output_dir'] in argv and entry['command'][2] in argv)


def rss_mib(pid):
    try:
        for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return float(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.


def host_available_mib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return float(line.split()[1]) / 1024
    raise RuntimeError('MemAvailable is missing')


def durable_episode(entry):
    if entry['method'] == 'shortest_ref':
        return 6 if (Path(entry['output_dir']) / 'results.json').exists() else 0
    from tools.compact_episode_archives import committed_episode
    return committed_episode(entry)


def prepare(args):
    from tools.run_city_baseline_dataset import build_runs, snapshot
    from urban_dreamer.workflow.config import load_config
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('new batch requires an empty output root')
    if shutil.disk_usage(root).free < args.min_free_gb * 1024 ** 3:
        raise ValueError('insufficient disk space for launch')
    for name in ('logs', 'processes', 'attempts'):
        (root / name).mkdir()
    source = snapshot(root)
    (source / 'networks').unlink()  # replace this newly created link with frozen input copies
    (source / 'networks').mkdir()
    inputs = {}
    for city in CITIES:
        destination = source / 'networks' / city
        destination.mkdir()
        for filename in (f'{city}.net.xml', f'od_{city}.txt', 'agents.csv', 'edge_registry.json'):
            shutil.copy2(ROOT / 'networks' / city / filename, destination / filename)
        cfg = load_config(source / 'configs' / f'{city}.toml')
        assert cfg.simulation.traffic_model == 'meso'
        deadline = 30 * math.ceil((cfg.simulation.world_model_intervals_per_episode * 30
            + cfg.simulation.training_completion_buffer_seconds) / 30)
        assert cfg.simulation.incident_start_seconds > max(deadline, cfg.simulation.evaluation_horizon_seconds)
        prepared = validate_prepared_inputs(cfg)
        assert prepared.vehicle_count == {'ingolstadt_custom':1035, 'coulommiers':1811, 'montereau_fault_yonne':1500}[city]
        inputs[city] = prepared.input_hashes
    settings = argparse.Namespace(cities=CITIES, methods=METHODS, devices=[], episodes=args.episodes,
        seed=42, checkpoint_every=1, eval_every=100, eval_seeds=3, forbid_revisit=False,
        global_reward_only=True, reward_objective='observed_mean', sp_prior_weight=0., sp_prior_temperature=30.,
        sp_prior_hold_episodes=200, sp_prior_off_episode=1000)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    entries = build_runs(root, source, settings, stamp)
    for entry in entries:
        entry['episode_budget'] = args.episodes
        large = entry['city'] != 'ingolstadt_custom'
        entry['required_gpu_mib'] = GPU_REQUIREMENTS.get(entry['method'], (0, 0))[int(large)]
        entry['required_host_mib'] = (16384 if entry['method'] in {'mbpo', 'mamba'} else
            12288 if entry['method'] == 'dreamer' else 6144 if entry['method'] in GPU_REQUIREMENTS else 3072)
        entry['device'] = 'cuda:0' if entry['required_gpu_mib'] else 'cpu'
        entry['command'][entry['command'].index('--device') + 1] = entry['device']
        if entry['method'] == 'dta':
            entry['command'][entry['command'].index('--sp-prior-weight') + 1] = '0'
        entry['effective_prior'] = 0.
        entry['launched'] = False
    for city in CITIES:
        name = f'{city}_shortest_ref'
        entries.append(dict(id=name, city=city, method='shortest_ref', device='cpu',
            episode_budget=6, required_gpu_mib=0, required_host_mib=3072,
            effective_prior=0., output_dir=str(root / city / 'shortest_ref'),
            log=str(root / 'logs' / f'{name}.log'), state=str(root / 'processes' / f'{name}.json'),
            command=[sys.executable, '-u', str(source / 'tools/queued_shortest_reference.py'),
                     '--config', str(source / 'configs' / f'{city}.toml'),
                     '--output-dir', str(root / city / 'shortest_ref')], launched=False))
    manifest = dict(created_at=now(), source_snapshot=str(source), input_hashes_by_city=inputs,
        episode_budget=args.episodes, total_training_episodes=args.episodes * len(CITIES) * len(METHODS),
        seed=42, eval_every=100, eval_seeds=3, checkpoint_every=1,
        reward_mode='global_only', global_reward_only=True, reward_objective='observed_mean',
        individual_timeout_penalty=0., forbid_revisit=False,
        prior=dict(initial_weight=0., temperature_seconds=30., hold_episodes=200,
                   off_episode=1000, decay='cosine', reference_methods_prior=0.),
        poll_seconds=args.poll_seconds, gpu_headroom_mib=args.gpu_headroom_mib,
        host_headroom_mib=16384, max_cpu_jobs=12, min_free_gb=args.min_free_gb,
        max_oom_retries=12, runs=entries, controller_session=f'ud_queue_{stamp}',
        compactor_session=f'ud_queue_compact_{stamp}', launch_status='prepared',
        memory_estimates='Conservative initial peak budgets; increase on CUDA OOM. Never count these as measured peaks.')
    write_json(root / 'manifest.json', manifest)
    jobs = {e['id']: dict(status='queued', attempt=0, required_gpu_mib=e['required_gpu_mib'],
                         observed_peak_mib=0., oom_retries=0, pid=None, next_eligible=0.) for e in entries}
    write_json(root / 'queue_state.json', dict(created_at=now(), jobs=jobs))
    print(json.dumps(dict(prepared=str(root), training_jobs=len(CITIES) * len(METHODS), references=len(CITIES))), flush=True)


def recover_output(root, entry, state):
    output = Path(entry['output_dir'])
    if not output.exists() or not any(output.iterdir()):
        return False
    if entry['method'] == 'shortest_ref':
        return False  # reference evaluator resumes its row ledger itself
    boundary = durable_episode(entry)
    checkpoint = output / ('progress.json' if entry['method'] == 'random' else 'latest.pt')
    preserved = root / 'attempts' / entry['id'] / f'pre_attempt_{state["attempt"]}'
    preserved.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        shutil.move(str(output), str(preserved / 'uncommitted_output'))
        return False
    metrics = output / 'metrics.jsonl'
    if metrics.exists():
        shutil.copy2(metrics, preserved / 'metrics_before.jsonl')
        unique = {}
        with metrics.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(row.get('episode', 0)) <= boundary:
                    key = (row.get('phase'), row.get('episode'), row.get('policy'), row.get('seed'))
                    unique[key] = line
        temporary = metrics.with_suffix('.recovered.tmp')
        temporary.write_text(''.join(unique.values()))
        os.replace(temporary, metrics)
    for path in (output / 'episodes' / 'train').glob('episode_*.npz'):
        if int(path.stem.rsplit('_', 1)[1]) > boundary:
            destination = preserved / 'uncommitted_archives'
            destination.mkdir(exist_ok=True)
            shutil.move(str(path), str(destination / path.name))
    return True


def status(root, manifest, ledger, gpus=None, allocations=None, gpu_error=None):
    rows = []
    for entry in manifest['runs']:
        state = ledger['jobs'][entry['id']]
        try:
            progress = durable_episode(entry)
        except (OSError, ValueError, StopIteration):
            progress = None
        row = dict(id=entry['id'], city=entry['city'], method=entry['method'],
                   device=entry['device'], episode_budget=entry['episode_budget'],
                   committed_episode=progress, **state)
        rows.append(row)
        write_json(Path(entry['state']), row)
    payload = dict(timestamp=now(), controller_pid=ledger.get('controller_pid'),
        reward_mode=manifest['reward_mode'], counts=dict(Counter(r['status'] for r in rows)),
        free_disk_gb=shutil.disk_usage(root).free / 1024 ** 3, gpus=gpus or [],
        gpu_query_error=gpu_error, runs=rows)
    write_json(root / 'status.json', payload)
    write_json(root / 'queue_state.json', ledger)
    return payload


def run(root):
    root = root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('launch_status') == 'stopped':
        raise ValueError('batch stopped by user; prepare a new batch rather than automatically resuming it')
    lock = (root / 'queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = json.loads((root / 'queue_state.json').read_text())
    ledger['controller_pid'] = os.getpid()
    children, logs = {}, {}
    entries = {e['id']:e for e in manifest['runs']}
    hashes = json.loads((root / 'source_hashes.json').read_text())
    source = Path(manifest['source_snapshot'])
    if not all(hashlib.sha256((source / p).read_bytes()).hexdigest() == digest for p,digest in hashes.items()):
        raise ValueError('frozen source snapshot changed')
    while True:
        gpu_error = None
        try:
            gpus, allocations = query_gpus()
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            gpus, allocations = [], {}
            gpu_error = str(error)
        for run_id, state in ledger['jobs'].items():
            entry = entries[run_id]
            if state['status'] != 'running':
                continue
            process = children.get(run_id)
            alive = process.poll() is None if process is not None else owned_alive(state, entry)
            if alive:
                state['observed_peak_mib'] = max(state['observed_peak_mib'], allocations.get(state['pid'], 0.))
                continue
            code = process.returncode if process is not None else adopted_exit_code(state)
            if run_id in logs:
                logs.pop(run_id).close()
            children.pop(run_id, None)
            boundary = durable_episode(entry)
            archives = len(list((Path(entry['output_dir']) / 'episodes/train').glob('episode_*.npz')))
            complete = boundary >= entry['episode_budget'] and (
                entry['method'] == 'shortest_ref' or archives == entry['episode_budget'])
            # A live controller also requires successful final evaluation/exit.
            interrupted_for_disk = state.pop('interrupted_for_disk', False)
            if complete and code == 0:
                state.update(status='completed', completed_at=now(), exit_code=code, pid=None)
            elif interrupted_for_disk or code is None:
                state.update(status='waiting_disk' if interrupted_for_disk else 'queued_recovery',
                             exit_code=code, pid=None, next_eligible=time.time()+10,
                             last_error='disk guard pause' if interrupted_for_disk else 'controller recovery')
            else:
                text = Path(state['attempt_log']).read_text(errors='replace')[-262144:]
                kind = failure_kind(text, code)
                if kind == 'oom' and state['required_gpu_mib'] > 0 and state['oom_retries'] < manifest['max_oom_retries']:
                    maximum = max((g['total_mib'] for g in gpus), default=24576) - manifest['gpu_headroom_mib']
                    previous = state['required_gpu_mib']
                    needed = bump_gpu_requirement(previous, state['observed_peak_mib'], maximum)
                    requested = re.findall(r'Tried to allocate ([\d.]+) (GiB|MiB)', text, flags=re.I)
                    if requested:
                        value, unit = requested[-1]
                        allocation = float(value) * (1024 if unit.lower() == 'gib' else 1)
                        needed = min(maximum, max(needed, math.ceil(state['observed_peak_mib'] + allocation + 512)))
                    state['oom_retries'] += 1
                    state.update(required_gpu_mib=needed, last_error='CUDA OOM', exit_code=code,
                                 pid=None, next_eligible=time.time()+30,
                                 status='queued_oom' if needed > previous else 'blocked_gpu_capacity')
                else:
                    state.update(status='failed', exit_code=code, pid=None, failed_at=now(),
                                 last_error=text[-4000:])
            print(json.dumps(dict(id=run_id, status=state['status'], exit_code=code)), flush=True)
        running = [s for s in ledger['jobs'].values() if s['status']=='running']
        calibrate_memory_budgets(entries, ledger['jobs'], gpus, allocations,
                                 manifest['gpu_headroom_mib'])
        try:
            apply_admission_overrides(root, manifest, entries, ledger['jobs'], gpus)
            ledger.pop('admission_override_error', None)
        except (OSError, ValueError) as error:
            ledger['admission_override_error'] = str(error)
        free_disk = shutil.disk_usage(root).free / 1024 ** 3
        if free_disk < manifest['min_free_gb']:
            # Pause only this batch's workers; the independent compactor can
            # finish committed archives and free space before automatic resume.
            for run_id, state in ledger['jobs'].items():
                if state['status']=='running' and owned_alive(state, entries[run_id]):
                    state['interrupted_for_disk'] = True
                    os.kill(state['pid'], signal.SIGTERM)
        host_free = host_available_mib() - manifest['host_headroom_mib']
        host_free -= sum(max(0, entries[k]['required_host_mib'] - rss_mib(s['pid']))
                         for k,s in ledger['jobs'].items() if s['status']=='running')
        cpu_count = sum(s.get('gpu_index') is None for s in running)
        for entry in manifest['runs']:
            state = ledger['jobs'][entry['id']]
            if state['status'] not in {'queued', 'queued_oom', 'queued_recovery', 'waiting_gpu', 'waiting_host', 'waiting_disk'}:
                continue
            if time.time() < state['next_eligible']:
                continue
            if free_disk < manifest['min_free_gb']:
                state['status']='waiting_disk';continue
            if host_free < entry['required_host_mib']:
                state['status']='waiting_host';continue
            gpu_index = None
            if state['required_gpu_mib']:
                gpu_index = choose_gpu(state['required_gpu_mib'], gpus, running, allocations,
                                       manifest['gpu_headroom_mib'])
                if gpu_index is None:
                    state['status']='waiting_gpu';continue
            elif cpu_count >= manifest['max_cpu_jobs']:
                state['status']='waiting_host';continue
            state['attempt'] += 1
            resume = recover_output(root, entry, state)
            command = list(entry['command']) + (['--resume'] if resume else [])
            environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            environment['PATH'] = str(Path(sys.executable).parent)+os.pathsep+environment.get('PATH','')
            environment['CUDA_VISIBLE_DEVICES'] = '' if gpu_index is None else next(g['uuid'] for g in gpus if g['index']==gpu_index)
            attempt_log = root/'logs'/f'{entry["id"]}.attempt{state["attempt"]}.log'
            log = attempt_log.open('a')
            log.write(f'Launch {now()} resume={resume} physical_gpu={gpu_index}\n');log.flush()
            link = Path(entry['log'])
            if link.is_symlink():
                link.unlink()
            if not link.exists():
                link.symlink_to(attempt_log)
            process = subprocess.Popen(command, cwd=source, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', pid=process.pid, process_starttime=process_identity(process.pid),
                         gpu_index=gpu_index, started_at=now(), attempt_log=str(attempt_log), resume=resume)
            children[entry['id']] = process;logs[entry['id']] = log
            running.append(state)
            host_free -= entry['required_host_mib']
            cpu_count += int(gpu_index is None)
            status(root, manifest, ledger, gpus, allocations, gpu_error)
            print(json.dumps(dict(started=entry['id'], gpu=gpu_index, resume=resume,
                                  required_gpu_mib=state['required_gpu_mib'])), flush=True)
        payload = status(root, manifest, ledger, gpus, allocations, gpu_error)
        if all(s['status'] in TERMINAL for s in ledger['jobs'].values()):
            write_json(root/'final_status.json', payload)
            return
        time.sleep(manifest['poll_seconds'])


def launch(root):
    root = root.resolve();path = root/'manifest.json'
    manifest = json.loads(path.read_text())
    if manifest['launch_status'] != 'prepared':
        raise ValueError('batch already launched; use run to recover its controller')
    source = Path(manifest['source_snapshot'])
    manifest.update(launch_status='launching', launched_at=now());write_json(path, manifest)
    command = [sys.executable, '-u', str(source/'tools/queued_city_baselines.py'),
               'run', '--output-root', str(root)]
    shell = shlex.join(command)+' >> '+shlex.quote(str(root/'controller.log'))+' 2>&1'
    subprocess.run(['/usr/bin/tmux','new-session','-d','-s',manifest['controller_session'],
                    '-c',str(source),shell],check=True)
    compactor = [sys.executable,'-u',str(source/'tools/compact_episode_archives.py'),
                 '--root',str(root),'--workers','2']
    shell = shlex.join(compactor)+' >> '+shlex.quote(str(root/'compaction.log'))+' 2>&1'
    subprocess.run(['/usr/bin/tmux','new-session','-d','-s',manifest['compactor_session'],
                    '-c',str(source),shell],check=True)
    manifest['launch_status']='launched';write_json(path, manifest)
    print(json.dumps(dict(root=str(root),controller=manifest['controller_session'],jobs=len(manifest['runs']))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare','launch','run','status'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--gpu-headroom-mib', type=int, default=1024)
    parser.add_argument('--min-free-gb', type=float, default=20)
    args = parser.parse_args()
    if args.episodes <= 0 or args.poll_seconds <= 0 or args.gpu_headroom_mib < 0 or args.min_free_gb <= 0:
        parser.error('invalid resource or episode budget')
    if args.mode == 'prepare':
        prepare(args)
    elif args.mode == 'launch':
        launch(args.output_root)
    elif args.mode == 'run':
        run(args.output_root)
    else:
        print((args.output_root.resolve()/'status.json').read_text())


if __name__ == '__main__':
    main()
