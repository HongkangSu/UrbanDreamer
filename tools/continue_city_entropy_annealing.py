"""Transfer a baseline queue to an entropy-only revision at durable episodes.

Existing output directories and archive compactors are retained. Preparation is
read-only with respect to the active batch; application stops only its owned
unfinished learners and archives checkpoints before explicitly migrating their
configuration identities. Completed jobs and training budgets are preserved.
"""
from __future__ import annotations

import argparse
import copyreg
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queued_city_baselines as queue
from urban_dreamer.workflow.config import load_config

ENTROPY_SETTINGS = {
    'actor_entropy_mode': 'annealed',
    'actor_entropy_initial_coefficient': .002,
    'actor_entropy_final_coefficient': .0001,
    'actor_entropy_hold_fraction': .1,
    'actor_entropy_end_fraction': .8,
}
DEFAULT_OLD_ROOT = ROOT / 'outputs/city_baselines_v5_1500_noprior_penalty_20261009'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_hashes(source):
    return {str(p.relative_to(source)): sha(p)
            for directory in ('urban_dreamer', 'tools', 'configs')
            for p in (source / directory).rglob('*')
            if p.is_file() and p.suffix in {'.py', '.toml'} and '__pycache__' not in p.parts}


def verify_sources(source, hashes):
    require(all(sha(source / p) == h for p, h in hashes.items()), 'source snapshot changed')


def input_hashes(source, city):
    directory = source / 'networks' / city
    return {key: sha(directory / name) for key, name in {
        'network_xml': f'{city}.net.xml', 'network_agents': 'agents.csv',
        'od_file': f'od_{city}.txt', 'edge_registry': 'edge_registry.json'}.items()}


def install_schedule(path):
    """Replace only the entropy fields of the existing method TOML section."""
    path = Path(path)
    text = path.read_text()
    match = re.search(r'(?m)^\[method\]\s*$', text)
    require(match is not None, f'{path}: missing method section')
    end = re.search(r'(?m)^\[', text[match.end():])
    stop = match.end() + end.start() if end else len(text)
    section = text[match.end():stop]
    for key, value in ENTROPY_SETTINGS.items():
        line = f'{key} = {json.dumps(value)}'
        pattern = rf'(?m)^{key}\s*=.*$'
        if re.search(pattern, section):
            section = re.sub(pattern, line, section)
        else:
            section = section.rstrip() + '\n' + line + '\n'
    path.write_text(text[:match.end()] + section.rstrip() + '\n\n' + text[stop:])


def controller_identity(root, ledger):
    pid = ledger.get('controller_pid')
    argv = queue.process_argv(pid)
    identity = queue.process_identity(pid)
    require(identity and '--output-root' in argv and
            Path(argv[argv.index('--output-root') + 1]).resolve() == root.resolve() and
            any(Path(arg).name == 'queued_city_baselines.py' for arg in argv),
            'active controller identity mismatch')
    return {'pid': pid, 'process_starttime': identity}


def prepared_entry(entry, new_source, new_root):
    result = deepcopy(entry)
    old_source = Path(entry['command'][2]).resolve().parents[1]
    command = []
    for value in entry['command']:
        try:
            relative = Path(value).relative_to(old_source)
        except ValueError:
            command.append(value)
        else:
            command.append(str(new_source / relative))
    result.update(command=command,
                  log=str(new_root / 'logs' / f'{entry["id"]}.log'),
                  state=str(new_root / 'processes' / f'{entry["id"]}.json'),
                  continuation_from_source=str(old_source),
                  continuation_reason='fixed entropy cosine annealing; weights and episode cursor retained')
    return result


def prepare(args):
    old_root, new_root = args.old_root, args.new_root
    old = read(old_root / 'manifest.json')
    ledger = read(old_root / 'queue_state.json')
    identity = controller_identity(old_root, ledger)
    require(not new_root.exists() or not any(new_root.iterdir()), 'new root must be empty')
    require(args.overlay_files, 'explicitly audited entropy-only overlay file list is required')
    checkpoint_bytes = sum(path.stat().st_size for entry in old['runs']
                           if ledger['jobs'][entry['id']]['status'] != 'completed'
                           for path in (Path(entry['output_dir']) / 'latest.pt',
                                        Path(entry['output_dir']) / 'best.pt') if path.exists())
    require(shutil.disk_usage(old_root).free >= checkpoint_bytes + old['min_free_gb'] * 1024 ** 3,
            'insufficient disk space to preserve original checkpoints and migrate copies')
    new_root.mkdir(parents=True, exist_ok=True)
    for name in ('logs', 'processes', 'attempts', 'handoff_original'):
        (new_root / name).mkdir()
    old_sources = {city: Path(old.get('source_snapshots_by_city', {}).get(city, old['source_snapshot']))
                   for city in queue.CITIES}
    unique_sources = {}
    sources, fingerprints, overlays = {}, {}, {}
    for city, old_source in old_sources.items():
        if old_source not in unique_sources:
            destination = (new_root / 'source' if old_source == Path(old['source_snapshot']) else
                           new_root / 'sources' / city)
            shutil.copytree(old_source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            original_hashes = source_hashes(destination)
            for relative in args.overlay_files:
                require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                        'overlay path must be relative to source')
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.overlay_root / relative, target)
            for config in (destination / 'configs').glob('*.toml'):
                if config.stem in queue.CITIES or config.stem.removesuffix('_offline') in queue.CITIES:
                    install_schedule(config)
            unique_sources[old_source] = destination
            overlays[str(destination)] = {
                'original_source': str(old_source), 'original_hashes': original_hashes,
                'new_hashes': source_hashes(destination), 'allowed_overlay_files': args.overlay_files,
            }
        destination = unique_sources[old_source]
        sources[city] = str(destination)
        require(input_hashes(destination, city) == old['input_hashes_by_city'][city],
                f'{city}: network or demand changed')
        old_cfg = load_config(old_source / 'configs' / f'{city}.toml')
        new_cfg = load_config(destination / 'configs' / f'{city}.toml')
        fingerprints[city] = {'old': old_cfg.semantic_fingerprint, 'new': new_cfg.semantic_fingerprint}
    stamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    manifest = deepcopy(old)
    manifest.update(created_at=queue.now(), source_snapshot=str(new_root / 'source'),
                    source_snapshots_by_city=sources, launch_status='prepared',
                    controller_session=f'ud_entropy_queue_{stamp}',
                    continuation_from=str(old_root), entropy_schedule=ENTROPY_SETTINGS,
                    archive_policy='Existing output paths and compactors retained; historical archives unmodified.')
    manifest['runs'] = [deepcopy(e) if ledger['jobs'][e['id']]['status'] == 'completed'
                        else prepared_entry(e, Path(sources[e['city']]), new_root)
                        for e in old['runs'] if e['method'] not in {'dta', 'random'}]
    queue.write_json(new_root / 'manifest.json', manifest)
    queue.write_json(new_root / 'source_hashes.json', source_hashes(new_root / 'source'))
    queue.write_json(new_root / 'handoff_protocol.json', {
        'phase': 'prepared', 'prepared_at': queue.now(), 'old_root': str(old_root),
        'new_root': str(new_root), 'controller': identity, 'fingerprints': fingerprints,
        'source_revisions': overlays, 'entropy_schedule': ENTROPY_SETTINGS,
        'policy': 'Continue unfinished jobs at durable checkpoints; preserve completed jobs and budgets.',
    })


def preserved_state_digest(state):
    """Stream the whole Python graph in C, with logical array/tensor reducers.

    Replay dictionaries contain tens of millions of scalar fields. Pickle's C
    dispatcher handles those without a Python callback for every primitive.
    Arrays and tensors reduce to dtype, shape and a hash of every content byte;
    device/storage addresses and serialization container names are irrelevant.
    """
    import numpy as np
    import torch
    digest = hashlib.sha256()

    class HashWriter:
        def write(self, data):
            digest.update(data)
            return len(data)

    def tensor_reduce(value):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy()
        content = hashlib.sha256(memoryview(raw)).hexdigest()
        return tuple, (('logical_tensor_v1', str(tensor.dtype), tuple(tensor.shape), content),)

    def array_reduce(value):
        if value.dtype.hasobject:
            marker = ('logical_object_array_v1', repr(value.dtype), value.shape, value.tolist())
        else:
            array = np.ascontiguousarray(value)
            # Flatten first: memoryview.cast rejects zero-sized multidimensional
            # arrays, which legitimately occur in empty replay/event buffers.
            raw = array.reshape(-1).view(np.uint8)
            content = hashlib.sha256(memoryview(raw)).hexdigest()
            marker = ('logical_array_v1', repr(value.dtype), value.shape, content)
        return tuple, (marker,)

    pickler = pickle.Pickler(HashWriter(), protocol=5)
    pickler.dispatch_table = {**copyreg.dispatch_table, torch.Tensor: tensor_reduce,
                              torch.nn.Parameter: tensor_reduce, np.ndarray: array_reduce}
    pickler.dump({k: v for k, v in state.items()
                  if k not in {'config_fingerprint', 'entropy_migration'}})
    return digest.hexdigest()


def migrate_checkpoint(path, backup, *, old_fingerprint, new_fingerprint, expected_inputs,
                       schedule=ENTROPY_SETTINGS):
    """Explicit metadata migration; no optimizer, RNG, replay, or weight edits."""
    import torch
    path, backup = Path(path), Path(backup)
    state = torch.load(path, map_location='cpu', weights_only=False)
    require(state['input_hashes'] == expected_inputs, 'checkpoint inputs mismatch')
    if state['config_fingerprint'] == new_fingerprint:
        migration = state.get('entropy_migration', {})
        require(migration.get('old_fingerprint') == old_fingerprint and
                migration.get('new_fingerprint') == new_fingerprint and
                migration.get('schedule') == dict(schedule),
                'checkpoint already changed without migration provenance')
        require(backup.exists(), 'migrated checkpoint lacks original backup')
        original_sha = sha(backup)
        require(original_sha == migration.get('old_checkpoint_sha256'), 'checkpoint backup changed')
        before = preserved_state_digest(state)
        episode = state['episode']
        del state
        original = torch.load(backup, map_location='cpu', weights_only=False)
        require(original['config_fingerprint'] == old_fingerprint and
                original['input_hashes'] == expected_inputs, 'original backup identity mismatch')
        require(preserved_state_digest(original) == before, 'migrated learning state differs from backup')
        del original
        return {'episode': episode, 'status': 'already_migrated', 'backup': str(backup),
                'original_sha256': original_sha, 'preserved_learning_state_sha256': before,
                'digest_algorithm': 'streaming_pickle_logical_tensors_v1',
                'migrated_sha256': sha(path)}
    require(state['config_fingerprint'] == old_fingerprint, 'checkpoint old configuration mismatch')
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        require(sha(path) == sha(backup), 'existing original backup differs from unmigrated checkpoint')
    else:
        try:
            os.link(path, backup)
        except OSError:
            shutil.copy2(path, backup)
    before = preserved_state_digest(state)
    old_sha = sha(backup)
    state['config_fingerprint'] = new_fingerprint
    state['entropy_migration'] = {
        'old_fingerprint': old_fingerprint, 'new_fingerprint': new_fingerprint,
        'at_episode': state['episode'], 'old_checkpoint_sha256': old_sha,
        'schedule': dict(schedule), 'reason': 'User authorized fixed entropy annealing continuation',
        'preserved_learning_state_sha256': before,
        'digest_algorithm': 'streaming_pickle_logical_tensors_v1',
    }
    temporary = path.with_name(f'.{path.name}.entropy.{os.getpid()}.tmp')
    try:
        torch.save(state, temporary)
        del state
        loaded = torch.load(temporary, map_location='cpu', weights_only=False)
        require(preserved_state_digest(loaded) == before, 'learning state changed during migration')
        require(sha(backup) == old_sha, 'original checkpoint changed')
        episode = loaded['episode']
        del loaded
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'episode': episode, 'status': 'migrated', 'backup': str(backup),
            'original_sha256': old_sha, 'preserved_learning_state_sha256': before,
            'digest_algorithm': 'streaming_pickle_logical_tensors_v1',
            'migrated_sha256': sha(path)}


def owned_workers(entries, jobs):
    """Find exact owned PIDs, including a launch racing the final ledger write."""
    found = {}
    for entry in entries:
        if jobs[entry['id']]['status'] == 'completed':
            continue
        state = jobs[entry['id']]
        if queue.owned_alive(state, entry):
            found[state['pid']] = (state, entry)
        for process in Path('/proc').iterdir():
            if process.name.isdigit():
                candidate = {'pid': int(process.name),
                             'process_starttime': queue.process_identity(int(process.name))}
                if queue.owned_alive(candidate, entry):
                    found[candidate['pid']] = (candidate, entry)
    return found


def copy_once(source, destination):
    if Path(source).exists() and not Path(destination).exists():
        shutil.copy2(source, destination)


def controller_stopped(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] in {'T', 't'}
    except (OSError, IndexError):
        return False


def verify_finished_migrations(entry, protocol):
    """Reuse recorded semantic proofs only while both proven files are intact."""
    for name, proof in protocol['checkpoint_migrations'].get(entry['id'], {}).items():
        require(sha(Path(entry['output_dir']) / name) == proof.get('migrated_sha256'),
                'previously migrated checkpoint changed')
        require(sha(proof['backup']) == proof.get('original_sha256'),
                'previously proven original checkpoint changed')


def apply(args):
    old_root, new_root = args.old_root, args.new_root
    protocol = read(new_root / 'handoff_protocol.json')
    resume_handoff = getattr(args, 'resume_handoff', False)
    require(protocol['phase'] == ('migrating' if resume_handoff else 'prepared'),
            'resume-handoff is only valid during migration; fresh apply requires prepared phase')
    require(protocol['old_root'] == str(old_root), 'old root changed')
    for source, revision in protocol['source_revisions'].items():
        verify_sources(Path(source), revision['new_hashes'])
        verify_sources(Path(revision['original_source']), revision['original_hashes'])
    old = read(old_root / 'manifest.json')
    ledger = read(old_root / 'queue_state.json')
    require(controller_identity(old_root, ledger) == protocol['controller'], 'controller changed')
    manifest = read(new_root / 'manifest.json')
    require(subprocess.run(['/usr/bin/tmux', 'has-session', '-t', manifest['controller_session']],
                           capture_output=True).returncode != 0, 'new controller session already exists')
    pid = protocol['controller']['pid']
    if resume_handoff:
        require(controller_stopped(pid), 'recovery requires the original controller already stopped')
        require(not owned_workers(old['runs'], ledger['jobs']), 'recovery requires all old workers stopped')
        require((new_root / 'handoff_original/queue_state.json').exists(), 'frozen ledger backup missing')
        ledger = read(new_root / 'handoff_original/queue_state.json')
    else:
        os.kill(pid, signal.SIGSTOP)
        for _ in range(50):
            if controller_stopped(pid):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('controller did not freeze')
        protocol.update(phase='controller_frozen', frozen_at=queue.now(), stopped_pids=[])
        queue.write_json(new_root / 'handoff_protocol.json', protocol)
        ledger = read(old_root / 'queue_state.json')
        for name in ('manifest.json', 'queue_state.json', 'status.json', 'admission_overrides.json'):
            copy_once(old_root / name, new_root / 'handoff_original' / name)
        victims = owned_workers(old['runs'], ledger['jobs'])
        for worker, (state, entry) in victims.items():
            if queue.owned_alive(state, entry):
                os.kill(worker, signal.SIGTERM)
                protocol['stopped_pids'].append(worker)
                queue.write_json(new_root / 'handoff_protocol.json', protocol)
        deadline = time.monotonic() + 45
        while any(queue.owned_alive(state, entry) for state, entry in victims.values()):
            require(time.monotonic() < deadline, 'worker remains alive; controller stays frozen')
            time.sleep(.2)
        protocol.update(phase='migrating', checkpoint_migrations={}, continuation_episodes={})
        queue.write_json(new_root / 'handoff_protocol.json', protocol)
    old_entries = {e['id']: e for e in old['runs']}
    fresh_jobs = {}
    for index, entry in enumerate(manifest['runs']):
        state = deepcopy(ledger['jobs'][entry['id']])
        old_entry = old_entries[entry['id']]
        output = Path(entry['output_dir'])
        boundary = queue.durable_episode(old_entry)
        # A job can finish between preparation and the controller freeze.
        if state['status'] == 'completed' or boundary >= entry['episode_budget']:
            require(state['status'] == 'completed',
                    f'{entry["id"]}: final checkpoint pending final evaluation; inspect before migration')
            manifest['runs'][index] = old_entry
            fresh_jobs[entry['id']] = state
            continue
        if entry['id'] in protocol['continuation_episodes']:
            require(boundary == protocol['continuation_episodes'][entry['id']],
                    'previously migrated episode cursor changed')
            verify_finished_migrations(entry, protocol)
            state.update(status='queued_recovery' if boundary else 'queued', pid=None,
                         process_starttime=None, next_eligible=0., continuation_episode=boundary)
            fresh_jobs[entry['id']] = state
            entry['required_gpu_mib'] = state['required_gpu_mib']
            continue
        archive = new_root / 'handoff_original' / entry['id']
        archive.mkdir(exist_ok=True)
        if state.get('attempt_log') and Path(state['attempt_log']).exists():
            copy_once(state['attempt_log'], archive / 'attempt.log')
        if output.exists():
            for name in ('protocol.json', 'metrics.jsonl'):
                if (output / name).exists():
                    copy_once(output / name, archive / name)
            queue.recover_output(new_root, old_entry, state)
            output.mkdir(parents=True, exist_ok=True)
            fingerprints = protocol['fingerprints'][entry['city']]
            migrations = {}
            for name in ('latest.pt', 'best.pt'):
                if (output / name).exists():
                    migrations[name] = migrate_checkpoint(output / name, archive / name,
                        old_fingerprint=fingerprints['old'], new_fingerprint=fingerprints['new'],
                        expected_inputs=old['input_hashes_by_city'][entry['city']])
            protocol['checkpoint_migrations'][entry['id']] = migrations
            if (output / 'protocol.json').exists():
                run_protocol = read(output / 'protocol.json')
                run_protocol['entropy_schedule'] = ENTROPY_SETTINGS
                run_protocol['config'] = load_config(Path(manifest['source_snapshots_by_city'][entry['city']]) /
                    'configs' / f'{entry["city"]}.toml').resolved_snapshot()
                run_protocol['entropy_continuation'] = {
                    'from_episode': boundary, 'at': queue.now(),
                    'old_protocol_backup': str(archive / 'protocol.json'),
                }
                queue.write_json(output / 'protocol.json', run_protocol)
            with (output / 'entropy_schedule_changes.jsonl').open('a') as stream:
                stream.write(json.dumps({'at_episode': boundary, 'schedule': ENTROPY_SETTINGS,
                    'at': queue.now(), 'handoff_root': str(new_root)}) + '\n')
        protocol['continuation_episodes'][entry['id']] = boundary
        state.update(status='queued_recovery' if boundary else 'queued', pid=None,
                     process_starttime=None, next_eligible=0.,
                     continuation_episode=boundary)
        fresh_jobs[entry['id']] = state
        entry['required_gpu_mib'] = state['required_gpu_mib']
        queue.write_json(new_root / 'handoff_protocol.json', protocol)
    fresh = {k: v for k, v in ledger.items() if k not in {'jobs', 'controller_pid'}}
    fresh.update(jobs=fresh_jobs, transferred_at=queue.now())
    manifest.update(launch_status='launching', launched_at=queue.now())
    queue.write_json(new_root / 'manifest.json', manifest)
    queue.write_json(new_root / 'queue_state.json', fresh)
    if (old_root / 'admission_overrides.json').exists():
        shutil.copy2(old_root / 'admission_overrides.json', new_root / 'admission_overrides.json')
    command = [sys.executable, '-u', str(new_root / 'source/tools/queued_city_baselines.py'),
               'run', '--output-root', str(new_root)]
    shell = shlex.join(command) + ' >> ' + shlex.quote(str(new_root / 'controller.log')) + ' 2>&1'
    subprocess.run(['/usr/bin/tmux', 'new-session', '-d', '-s', manifest['controller_session'],
                    '-c', str(new_root / 'source'), shell], check=True)
    manifest['launch_status'] = 'launched'
    queue.write_json(new_root / 'manifest.json', manifest)
    protocol.update(phase='launched', completed_at=queue.now(),
                    keeper_note='Never SIGCONT the old controller after handoff.')
    queue.write_json(new_root / 'handoff_protocol.json', protocol)
    queue.write_json(old_root / 'entropy_handoff.json', protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'apply'))
    parser.add_argument('--old-root', type=lambda value: Path(value).resolve(), default=DEFAULT_OLD_ROOT)
    parser.add_argument('--new-root', required=True, type=lambda value: Path(value).resolve())
    parser.add_argument('--overlay-root', type=lambda value: Path(value).resolve(), default=ROOT)
    parser.add_argument('--overlay-files', nargs='+')
    parser.add_argument('--resume-handoff', action='store_true',
                        help='Recover interrupted metadata migration while old controller/workers stay stopped')
    args = parser.parse_args()
    try:
        globals()[args.mode](args)
    except Exception as error:
        record = args.new_root / 'handoff_protocol.json'
        if record.exists():
            queue.write_json(record, {**read(record), 'last_error': repr(error), 'error_at': queue.now()})
        raise
    print(json.dumps({'mode': args.mode, 'new_root': str(args.new_root),
                      'phase': read(args.new_root / 'handoff_protocol.json')['phase']}))


if __name__ == '__main__':
    main()
