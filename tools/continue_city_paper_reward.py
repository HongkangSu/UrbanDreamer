"""Continue owned city learners with causal observed-mean rewards.

Preparation preserves immutable execution sources. Application archives original
checkpoints, keeps the durable episode/RNG/entropy clock, zero-extends context
layers and Adam moments, clears stale reward replay, and resets best selection.
It never signals world-model pretraining or completed baseline jobs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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
from tools import continue_city_entropy_annealing as previous
from tools import queued_city_baselines as queue
from tools.continue_city_entropy_annealing import (
    read, sha, require, source_hashes, verify_sources, controller_identity,
    owned_workers, copy_once, controller_stopped, verify_finished_migrations,
)
from urban_dreamer.workflow.config import load_config

REWARD = dict(objective='observed_mean', formula='-delta(C/max(B,1))/60',
              trip_start='actual_departure_before_deadline', context_dim=4,
              global_reward_only=True, individual_timeout_penalty=0.)


def source_fingerprint(source, city):
    """Resolve implicit config defaults with the corresponding frozen loader."""
    source = Path(source).resolve()
    script = ('import sys; from pathlib import Path; '
              'sys.path.insert(0, str(Path(sys.argv[1]).resolve())); '
              'from urban_dreamer.workflow.config import load_config; '
              'print(load_config(Path(sys.argv[1])/"configs"/(sys.argv[2]+".toml")).semantic_fingerprint)')
    result = subprocess.run([sys.executable, '-c', script, str(source), city], cwd=source,
                            check=True, text=True, capture_output=True)
    return result.stdout.strip()


def atomic_checkpoint_copy(source, destination, *, preserve_existing=False):
    source, destination = Path(source), Path(destination)
    if preserve_existing and destination.exists():
        return
    temporary = destination.with_name(f'.{destination.name}.paper-copy.{os.getpid()}.tmp')
    try:
        shutil.copy2(source, temporary)
        require(sha(source) == sha(temporary), 'checkpoint copy changed')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(args):
    # Older city snapshots may predate the entropy continuation utilities.
    # Every template subprocess imports these tools from its own frozen source.
    required_helpers = (
        'tools/continue_city_paper_reward.py',
        'tools/continue_city_entropy_annealing.py',
        'tools/queued_city_baselines.py',
    )
    args.overlay_files = list(dict.fromkeys([*(args.overlay_files or ()), *required_helpers]))
    previous.prepare(args)
    manifest = read(args.new_root / 'manifest.json')
    ledger = read(args.old_root / 'queue_state.json')
    for entry in manifest['runs']:
        if ledger['jobs'][entry['id']]['status'] == 'completed':
            continue
        entry['command'] = [value for value in entry['command']
                            if value not in {'--global-reward-only', '--no-global-reward-only'}]
        entry['command'].extend(['--global-reward-only', '--reward-objective', 'observed_mean'])
        entry['continuation_reason'] = 'causal observed-mean reward; durable cursor and entropy clock retained'
    manifest.update(reward_mode='global_only', global_reward_only=True,
                    individual_timeout_penalty=0., reward_objective='observed_mean', paper_reward=REWARD,
                    controller_session='ud_paper_reward_' + time.strftime('%Y%m%d_%H%M%S', time.gmtime()))
    protocol = read(args.new_root / 'handoff_protocol.json')
    old_manifest = read(args.old_root / 'manifest.json')
    for city in queue.CITIES:
        old_source = old_manifest.get('source_snapshots_by_city', {}).get(city, old_manifest['source_snapshot'])
        new_source = manifest['source_snapshots_by_city'][city]
        protocol['fingerprints'][city] = dict(old=source_fingerprint(old_source, city),
                                             new=source_fingerprint(new_source, city))
    protocol.update(kind='observed_mean_reward_handoff', reward=REWARD,
                    policy='Retain weights/optimizers/RNG/episode budget; expand causal context; '
                           'purge legacy reward replay; reset best comparisons; preserve completed jobs.')
    queue.write_json(args.new_root / 'manifest.json', manifest)
    queue.write_json(args.new_root / 'handoff_protocol.json', protocol)


def _make_runtime(request):
    """Build the new schema in the frozen source, without collecting SUMO steps."""
    import torch
    from argparse import Namespace
    from urban_dreamer.learning.baselines import common, build_method
    args = Namespace(**request['args'])
    args.config = Path(request['config'])
    args.output_dir = Path(request['output'])
    args.global_reward_only = True
    args.reward_objective = 'observed_mean'
    args.device = 'cpu'
    args.threads = 1
    torch.set_num_threads(1)
    if request['method'] == 'online_ac':
        from tools.train_toy_actor_critic import RealActorCritic
        runtime = RealActorCritic(args)
        state = {key: getattr(runtime, key).state_dict() for key in
                 ('actor', 'global_critic', 'actor_optimizer', 'global_optimizer')}
    else:
        runtime = build_method(request['method'], args, common.EpisodeRunner(args))
        state = runtime.state_dict()
    modules = {key: value for key, value in vars(runtime).items() if isinstance(value, torch.nn.Module)}
    parameter_names = {}
    for module_name, module in modules.items():
        for name, parameter in module.named_parameters():
            parameter_names.setdefault(id(parameter), (module_name, name))
    optimizer_parameters = {}
    for name, optimizer in vars(runtime).items():
        if isinstance(optimizer, torch.optim.Optimizer) and name in state:
            optimizer_parameters[name] = [parameter_names[id(parameter)]
                for group in optimizer.param_groups for parameter in group['params']]
    template = dict(shapes={key: {name: list(value.shape) for name, value in values.items()}
                    for key, values in state.items() if isinstance(values, dict)
                    and values and all(isinstance(value, torch.Tensor) for value in values.values())},
                    optimizer_parameters=optimizer_parameters,
                    metadata={key: state[key] for key in
                              ('learning_protocol', 'protocol', 'schema', 'cfg', 'reward_mode') if key in state},
                    empty_buffers={key: state[key] for key in
                                   ('replay', 'sequence_replay', 'seed_replay', 'model_rows') if key in state},
                    config_fingerprint=runtime.config.semantic_fingerprint if request['method'] == 'online_ac'
                        else runtime.runner.config.semantic_fingerprint,
                    advantage_normalization=getattr(runtime, 'advantage_normalization', None))
    return runtime, template


def template_worker(args):
    request = read(args.request)
    runtime, template = _make_runtime(request)
    if args.validate_checkpoint:
        import torch
        state = torch.load(args.validate_checkpoint, map_location='cpu', weights_only=False)
        method_state = state.get('method_state', state)
        if request['method'] == 'online_ac':
            for key in template['shapes']:
                getattr(runtime, key).load_state_dict(method_state[key])
            for key in template['optimizer_parameters']:
                getattr(runtime, key).load_state_dict(method_state[key])
        else:
            runtime.load_state_dict(method_state)
        for name in template['optimizer_parameters']:
            optimizer = getattr(runtime, name)
            for parameter, values in optimizer.state.items():
                for key, value in values.items():
                    if key in {'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'}:
                        require(value.shape == parameter.shape, f'{name}: Adam moment shape mismatch')
        require(state['args']['reward_objective'] == 'observed_mean' and state['args']['global_reward_only'],
                'checkpoint reward identity is not migrated')
        require(state['config_fingerprint'] == template['config_fingerprint'], 'new configuration mismatch')
    queue.write_json(args.template_output, template)


def _pad(value, shape):
    import torch
    require(len(shape) == value.ndim and all(b >= a for a, b in zip(value.shape, shape)),
            'migration would shrink/restructure a parameter')
    differences = [b-a for a, b in zip(value.shape, shape) if b != a]
    require(len(differences) == 1 and differences[0] in {2, 4},
            'only audited causal context expansion by 2 or 4 is allowed')
    expanded = torch.zeros(shape, dtype=value.dtype, device=value.device)
    expanded[tuple(slice(0, size) for size in value.shape)] = value
    return expanded


def migrate_state(state, template):
    """Modify only named reward metadata, empty caches and context-shaped tensors."""
    method_state = state.get('method_state', state)
    expansions = []
    for module_name, shapes in template['shapes'].items():
        require(module_name in method_state and set(shapes) == set(method_state[module_name]),
                f'{module_name}: unsupported parameter schema change')
        for name, shape in shapes.items():
            original = method_state[module_name][name]
            if list(original.shape) == shape:
                continue
            require(module_name not in {'actor', 'target', 'world_model'} or
                    (module_name == 'world_model' and name in
                     {'action_embed.weight', 'reward_head.0.weight'}),
                    f'{module_name}.{name}: actor or unrelated world parameters changed')
            old_shape = list(original.shape)
            method_state[module_name][name] = _pad(original, shape)
            expansions.append(dict(module=module_name, parameter=name, old_shape=old_shape, new_shape=shape))
    for optimizer_name, names in template['optimizer_parameters'].items():
        optimizer = method_state[optimizer_name]
        ids = [parameter for group in optimizer['param_groups'] for parameter in group['params']]
        require(len(ids) == len(names), f'{optimizer_name}: optimizer parameter count changed')
        for parameter_id, (module_name, name) in zip(ids, names):
            target_shape = template['shapes'][module_name][name]
            for moment_name, moment in optimizer['state'].get(parameter_id, {}).items():
                if moment_name in {'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'} and list(moment.shape) != target_shape:
                    optimizer['state'][parameter_id][moment_name] = _pad(moment, target_shape)
    cleared = {key: len(method_state.get(key, [])) for key in template['empty_buffers']}
    for key, value in template['empty_buffers'].items():
        method_state[key] = deepcopy(value)
    for key, value in template['metadata'].items():
        method_state[key] = deepcopy(value)
    state['args'].update(global_reward_only=True, reward_objective='observed_mean')
    state['best'] = float('inf')
    if 'advantage_normalization' in state:
        state['advantage_normalization'] = 'global_scale_only_std_floor_1_clip_2_5'
    return dict(expansions=expansions, cleared_buffers=cleared)


def preserved_learning_digest(state, template, expansions=()):
    """Hash retained learning state after projecting new context columns away."""
    projected = dict(state)
    for key in ('best', 'advantage_normalization', 'reward_migration'):
        projected.pop(key, None)
    projected['args'] = {key: value for key, value in state['args'].items()
                         if key not in {'global_reward_only', 'reward_objective'}}
    method_state = dict(state.get('method_state', projected))
    for key in (*template['metadata'], *template['empty_buffers']):
        method_state.pop(key, None)
    # Module state_dicts are OrderedDicts with an auxiliary _metadata attribute.
    # Compare logical parameter maps consistently before and after projection.
    for module_name in template['shapes']:
        if module_name in method_state:
            method_state[module_name] = dict(method_state[module_name])
    old_shapes = {(entry['module'], entry['parameter']): entry['old_shape'] for entry in expansions}
    for (module_name, name), old_shape in old_shapes.items():
        module = dict(method_state[module_name])
        module[name] = module[name][tuple(slice(0, size) for size in old_shape)]
        method_state[module_name] = module
    for optimizer_name, names in template['optimizer_parameters'].items():
        optimizer = dict(method_state[optimizer_name])
        optimizer['state'] = dict(optimizer['state'])
        ids = [parameter for group in optimizer['param_groups'] for parameter in group['params']]
        for parameter_id, names_pair in zip(ids, names):
            old_shape = old_shapes.get(tuple(names_pair))
            if old_shape is None or parameter_id not in optimizer['state']:
                continue
            values = dict(optimizer['state'][parameter_id])
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in values:
                    values[key] = values[key][tuple(slice(0, size) for size in old_shape)]
            optimizer['state'][parameter_id] = values
        method_state[optimizer_name] = optimizer
    if 'method_state' in state:
        projected['method_state'] = method_state
    else:
        projected = method_state
    return previous.preserved_state_digest(projected)


def migrate_checkpoint(path, backup, *, old_fingerprint, new_fingerprint, expected_inputs,
                       source, city, method, schedule=None):
    import torch
    path, backup = Path(path), Path(backup)
    state = torch.load(path, map_location='cpu', weights_only=False)
    require(state['input_hashes'] == expected_inputs, 'checkpoint inputs mismatch')
    if state['args'].get('reward_objective') == 'observed_mean':
        marker = state.get('reward_migration', {})
        require(state['config_fingerprint'] == new_fingerprint and
                marker.get('old_fingerprint') == old_fingerprint and backup.exists() and
                sha(backup) == marker.get('original_checkpoint_sha256'),
                'already migrated checkpoint lacks matching provenance')
        template = read(backup.with_suffix('.template.json'))
        require(preserved_learning_digest(state, template, marker['expansions']) ==
                marker['preserved_learning_state_sha256'], 'migrated learning state changed')
        result = dict(episode=state['episode'], status='already_migrated', backup=str(backup),
            original_sha256=sha(backup), migrated_sha256=sha(path),
            preserved_learning_state_sha256=marker['preserved_learning_state_sha256'],
            expansions=marker['expansions'], cleared_buffers=marker['cleared_buffers'])
        del state
        original = torch.load(backup, map_location='cpu', weights_only=False)
        require(original['config_fingerprint'] == old_fingerprint and
                preserved_learning_digest(original, template) == result['preserved_learning_state_sha256'],
                'original backup learning state mismatch')
        return result
    require(state['config_fingerprint'] == old_fingerprint, 'checkpoint old configuration mismatch')
    require(state['args'].get('reward_objective', 'legacy_exposure') == 'legacy_exposure',
            'checkpoint already uses a different reward; fresh migration refused')
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        require(sha(path) == sha(backup), 'existing immutable backup differs')
    else:
        try:
            os.link(path, backup)
        except OSError:
            shutil.copy2(path, backup)
    request_path = backup.with_suffix('.request.json')
    template_path = backup.with_suffix('.template.json')
    queue.write_json(request_path, dict(args=state['args'], source=str(source), city=city, method=method,
        config=str(source / 'configs' / f'{city}.toml'), output=str(path.parent)))
    command = [sys.executable, str(source / 'tools/continue_city_paper_reward.py'), 'template',
               '--request', str(request_path), '--template-output', str(template_path)]
    subprocess.run(command, cwd=source, check=True)
    template = read(template_path)
    require(template['config_fingerprint'] == new_fingerprint, 'template configuration changed')
    before = preserved_learning_digest(state, template)
    result = migrate_state(state, template)
    state['config_fingerprint'] = new_fingerprint
    require(preserved_learning_digest(state, template, result['expansions']) == before,
            'retained weights, Adam state, counters or RNG changed')
    original_sha = sha(backup)
    state['reward_migration'] = dict(from_reward='legacy_exposure', to_reward=REWARD,
        at_episode=state['episode'], old_fingerprint=old_fingerprint, new_fingerprint=new_fingerprint,
        original_checkpoint_sha256=original_sha, preserved_learning_state_sha256=before, **result)
    temporary = path.with_name(f'.{path.name}.paper.{os.getpid()}.tmp')
    try:
        torch.save(state, temporary)
        episode = state['episode']
        del state
        loaded = torch.load(temporary, map_location='cpu', weights_only=False)
        require(preserved_learning_digest(loaded, template, result['expansions']) == before,
                'serialization changed retained learning state')
        del loaded
        subprocess.run(command + ['--validate-checkpoint', str(temporary)], cwd=source, check=True)
        require(sha(backup) == original_sha, 'original checkpoint changed')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(episode=episode, status='migrated', backup=str(backup), original_sha256=original_sha,
                migrated_sha256=sha(path), preserved_learning_state_sha256=before, **result)


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
    # Resume the interrupted learners before admitting previously waiting jobs.
    # Their measured resource budgets were already running together at handoff.
    active_ids = {key for key, state in ledger['jobs'].items() if state['status'] == 'running'}
    manifest['runs'].sort(key=lambda entry: entry['id'] not in active_ids)
    manifest['continuation_priority'] = [entry['id'] for entry in manifest['runs'] if entry['id'] in active_ids]
    protocol['continuation_priority'] = manifest['continuation_priority']
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
            for name in ('latest.pt',):
                if (output / name).exists():
                    migrations[name] = migrate_checkpoint(output / name, archive / name,
                        old_fingerprint=fingerprints['old'], new_fingerprint=fingerprints['new'],
                        expected_inputs=old['input_hashes_by_city'][entry['city']],
                        source=Path(manifest['source_snapshots_by_city'][entry['city']]),
                        city=entry['city'], method=entry['method'])
            if 'latest.pt' in migrations:
                if (output / 'best.pt').exists():
                    atomic_checkpoint_copy(output / 'best.pt', archive / 'best.pt', preserve_existing=True)
                atomic_checkpoint_copy(output / 'latest.pt', output / 'best.pt')
                migrations['best.pt'] = dict(migrations['latest.pt'], migrated_sha256=sha(output / 'best.pt'),
                    role='new-objective fallback; historical best archived')
            protocol['checkpoint_migrations'][entry['id']] = migrations
            if (output / 'protocol.json').exists():
                run_protocol = read(output / 'protocol.json')
                run_protocol.update(reward_mode='global_only', reward_objective='observed_mean',
                    global_reward=REWARD['formula'], reward=REWARD['formula'], timeout_penalty='disabled',
                    reward_contract=REWARD, global_critic_observation='physical flow plus t/T, active/B, log1p(C/60), log1p(B)',
                    reward_note='shared observed-mean delta; selection_score equals global_reward_sum',
                    advantage_normalization='global_scale_only_std_floor_1_clip_2_5')
                run_protocol.setdefault('args', {}).update(global_reward_only=True, reward_objective='observed_mean')
                run_protocol.setdefault('demand', {})['normalization'] = 'causal cumulative actual starts B(t)'
                run_protocol['reward_continuation'] = dict(from_episode=boundary, at=queue.now(),
                    original_protocol=str(archive / 'protocol.json'), best_score_reset=True,
                    replay_policy='legacy reward caches purged; model and optimizer progress retained')
                run_protocol['config'] = load_config(Path(manifest['source_snapshots_by_city'][entry['city']]) /
                    'configs' / f'{entry["city"]}.toml').resolved_snapshot()
                run_protocol['paper_reward_continuation'] = {
                    'from_episode': boundary, 'at': queue.now(),
                    'old_protocol_backup': str(archive / 'protocol.json'),
                }
                queue.write_json(output / 'protocol.json', run_protocol)
            with (output / 'reward_objective_changes.jsonl').open('a') as stream:
                stream.write(json.dumps({'at_episode': boundary, 'reward': REWARD,
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
    queue.write_json(old_root / 'paper_reward_handoff.json', protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'apply', 'template'))
    parser.add_argument('--old-root', type=lambda value: Path(value).resolve(),
                        default=ROOT / 'outputs/city_baselines_entropy_annealed_20261010')
    parser.add_argument('--new-root', type=lambda value: Path(value).resolve())
    parser.add_argument('--overlay-root', type=lambda value: Path(value).resolve(), default=ROOT)
    parser.add_argument('--overlay-files', nargs='+')
    parser.add_argument('--resume-handoff', action='store_true')
    parser.add_argument('--request', type=Path)
    parser.add_argument('--template-output', type=Path)
    parser.add_argument('--validate-checkpoint', type=Path)
    args = parser.parse_args()
    if args.mode == 'template':
        template_worker(args)
        return
    require(args.new_root is not None, 'new-root is required')
    try:
        globals()[args.mode](args)
    except Exception as error:
        record = args.new_root / 'handoff_protocol.json'
        if record.exists():
            queue.write_json(record, {**read(record), 'last_error': repr(error), 'error_at': queue.now()})
        raise
    print(json.dumps(dict(mode=args.mode, new_root=str(args.new_root),
                          phase=read(args.new_root / 'handoff_protocol.json')['phase'])))


if __name__ == '__main__':
    main()

