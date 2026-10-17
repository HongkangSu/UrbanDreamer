"""Replace only the six prior PPO workers with isolated fresh protocol-v7 runs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queued_city_baselines as queue

OLD = ROOT / 'outputs/city_baselines_static_round2_2000_20261015_173045'
AUDIT = ROOT / 'outputs/ppo_repair_20261016/restart'
METHODS = {'ippo', 'mappo'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def start_session(name, source, command, log_path):
    shell = shlex.join(command) + ' >> ' + shlex.quote(str(log_path)) + ' 2>&1'
    subprocess.run(['/usr/bin/tmux', 'new-session', '-d', '-s', name,
                    '-c', str(source), shell], check=True)


def stop_owned(pid, identity, expected_script):
    if queue.process_identity(pid) != identity:
        return
    assert expected_script in queue.process_argv(pid), (pid, expected_script)
    try:
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)  # Also release a deliberately frozen controller.
    except ProcessLookupError:
        pass


def prepare():
    AUDIT.mkdir(parents=True, exist_ok=True)
    record_path = AUDIT / 'restart.json'
    if record_path.exists():
        raise RuntimeError('restart already has a record; inspect rather than overwrite')
    validated = json.loads((ROOT / 'outputs/ppo_repair_20261016/summary.json').read_text())
    assert validated['learning_protocol_version'] == 7
    assert all(digest(ROOT / name) == value for name, value in validated['source_sha256'].items())
    previous = json.loads((OLD / 'manifest.json').read_text())
    source_old = Path(previous['source_snapshot'])
    hashes = json.loads((OLD / 'source_hashes.json').read_text())
    assert all(digest(source_old / name) == value for name, value in hashes.items())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    root = ROOT / 'outputs' / ('city_ppo_v7_2000_' + stamp)
    source = root / 'source'
    for name in ('logs', 'processes', 'attempts', 'source'):
        (root / name).mkdir(parents=True, exist_ok=False)
    # Keep the already-running simulator, features, configs and input corpus.
    # Overlay precisely the three implementation files covered by the v7 gate.
    for name in hashes:
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_old / name, destination)
    for name, value in validated['source_sha256'].items():
        shutil.copy2(ROOT / name, source / name)
        hashes[name] = value
    for city in previous['input_hashes_by_city']:
        destination = source / 'networks' / city
        destination.mkdir(parents=True)
        for filename in (f'{city}.net.xml', f'od_{city}.txt', 'agents.csv', 'edge_registry.json'):
            shutil.copy2(source_old / 'networks' / city / filename, destination / filename)
        config_path = source / 'configs' / f'{city}.toml'
        config_path.write_text(config_path.read_text().replace(
            str(source_old / 'networks' / city), str(destination)))
        hashes[str(config_path.relative_to(source))] = digest(config_path)
    entries = []
    for old in previous['runs']:
        if old['method'] not in METHODS:
            continue
        entry = dict(old)
        for name in ('output_dir', 'log', 'state'):
            entry[name] = entry[name].replace(str(OLD), str(root))
        entry['command'] = [argument.replace(str(OLD), str(root)) for argument in entry['command']]
        assert '--resume' not in entry['command']
        assert entry['command'][entry['command'].index('--episodes') + 1] == '2000'
        entry.update(launched=False, predecessor_output=old['output_dir'], learning_protocol_version=7)
        entries.append(entry)
    assert len(entries) == 6
    storage = Path('/home/wangwenrui/private/UrbanDreamer') / root.name
    storage.mkdir(parents=True, exist_ok=False)
    manifest = dict(previous)
    for key in ('active_admission_policy', 'launched_at', 'stopped_at', 'stop_reason'):
        manifest.pop(key, None)
    manifest.update(created_at=queue.now(), source_snapshot=str(source), runs=entries,
        methods=['ippo', 'mappo'], total_training_episodes=12000,
        evaluation_every_by_method={name: previous['evaluation_every_by_method'][name] for name in METHODS},
        predecessor_batch=str(OLD), validation_gate=str(ROOT / 'outputs/ppo_repair_20261016/summary.json'),
        checkpoint_restart='fresh protocol v7; no protocol-v6 checkpoint migration',
        learning_protocol_version=7, archive_storage_root=str(storage), compaction_workers=2,
        controller_session='ud_ppo7_queue_' + stamp, compactor_session='ud_ppo7_compact_' + stamp,
        launch_status='prepared', pending_reservation_fraction=.5, pending_host_reservation_fraction=.5,
        restart_reason='User requested IPPO and MAPPO restarted on every existing city network')
    jobs = {entry['id']: dict(status='queued', attempt=0, required_gpu_mib=entry['required_gpu_mib'],
        observed_peak_mib=0., oom_retries=0, pid=None, next_eligible=0.) for entry in entries}
    queue.write_json(root / 'manifest.json', manifest)
    queue.write_json(root / 'source_hashes.json', hashes)
    queue.write_json(root / 'queue_state.json', dict(created_at=queue.now(), jobs=jobs))
    assert all(digest(source / name) == value for name, value in hashes.items())
    queue.write_json(record_path, dict(root=str(root), storage=str(storage), status='prepared',
        predecessor=str(OLD), methods=sorted(METHODS), source_verified=True, created_at=queue.now()))
    print(json.dumps(dict(prepared=str(root), jobs=len(entries)), indent=2))


def replace():
    record = json.loads((AUDIT / 'restart.json').read_text())
    assert record['status'] == 'prepared'
    root = Path(record['root'])
    old_manifest = json.loads((OLD / 'manifest.json').read_text())
    source_old = Path(old_manifest['source_snapshot'])
    ledger = json.loads((OLD / 'queue_state.json').read_text())
    controller = ledger['controller_pid']
    controller_identity = queue.process_identity(controller)
    assert controller_identity and str(source_old / 'tools/queued_city_baselines.py') in queue.process_argv(controller)
    os.kill(controller, signal.SIGSTOP)
    try:
        # Freezing the controller prevents old workers being respawned during cutover.
        ledger = json.loads((OLD / 'queue_state.json').read_text())
        for name in ('manifest.json', 'queue_state.json', 'status.json'):
            shutil.copy2(OLD / name, AUDIT / ('before_' + name))
        preserved = {entry['id']: dict(pid=ledger['jobs'][entry['id']]['pid'],
            identity=ledger['jobs'][entry['id']].get('process_starttime'))
            for entry in old_manifest['runs'] if entry['method'] not in METHODS
            and queue.owned_alive(ledger['jobs'][entry['id']], entry)}
        targets = [(entry, ledger['jobs'][entry['id']]) for entry in old_manifest['runs']
                   if entry['method'] in METHODS]
        queue.write_json(AUDIT / 'cutover_before.json', dict(controller_pid=controller,
            controller_identity=controller_identity, preserved=preserved,
            targets=[dict(id=e['id'], pid=s.get('pid'), identity=s.get('process_starttime')) for e, s in targets]))
        stop_owned(controller, controller_identity, str(source_old / 'tools/queued_city_baselines.py'))
        for entry, state in targets:
            if queue.owned_alive(state, entry):
                stop_owned(state['pid'], state['process_starttime'], entry['command'][2])
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and (queue.process_identity(controller) == controller_identity
                or any(queue.owned_alive(s, e) for e, s in targets)):
            time.sleep(.2)
        assert queue.process_identity(controller) != controller_identity
        for entry, state in targets:
            if queue.owned_alive(state, entry):
                os.kill(state['pid'], signal.SIGKILL)
        time.sleep(.5)
        assert not any(queue.owned_alive(s, e) for e, s in targets)
        stamp = queue.now()
        for entry, state in targets:
            state.update(status='stopped', stopped_pid=state.get('pid'), pid=None,
                stopped_at=stamp, stop_reason='superseded by fresh protocol-v7 experiment',
                replacement_output=str(root / entry['city'] / entry['method']))
            state.pop('interrupted_for_disk', None)
        ledger['controller_pid'] = None
        old_manifest['ppo_v7_replacement'] = dict(root=str(root), methods=sorted(METHODS), replaced_at=stamp)
        queue.write_json(OLD / 'manifest.json', old_manifest)
        queue.status(OLD, old_manifest, ledger)
        assert all(queue.process_identity(row['pid']) == row['identity'] for row in preserved.values())
        session = 'ud_nonppo_continue_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        command = [sys.executable, '-u', str(source_old / 'tools/queued_city_baselines.py'),
                   'run', '--output-root', str(OLD)]
        start_session(session, source_old, command, OLD / 'controller.log')
        record.update(status='old_ppo_stopped', preserved_other_workers=preserved,
            old_controller_replacement_session=session, cutover_at=stamp)
        queue.write_json(AUDIT / 'restart.json', record)
    except Exception:
        if queue.process_identity(controller) == controller_identity:
            os.kill(controller, signal.SIGCONT)
        raise
    queue.launch(root)
    source = root / 'source'
    session = 'ud_ppo7_offload_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    command = [sys.executable, '-u', str(source / 'tools/offload_episode_archives.py'),
        '--root', str(root), '--storage-root', record['storage'], '--workers', '2', '--interval', '15']
    start_session(session, source, command, root / 'offload.log')
    record.update(status='launched', offload_session=session, launched_at=queue.now())
    queue.write_json(AUDIT / 'restart.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    {'prepare': prepare, 'replace': replace}[sys.argv[1]]()
