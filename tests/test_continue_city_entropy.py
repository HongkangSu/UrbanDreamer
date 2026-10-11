"""Entropy continuation preserves learning state and existing archive paths."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import continue_city_entropy_annealing as continuation


def checkpoint():
    return {
        'config_fingerprint': 'old', 'input_hashes': {'network_xml': 'unchanged'},
        'episode': 731, 'best': -4.2, 'args': {'episodes': 5000, 'seed': 42},
        'method_state': {
            'actor': {'weight': torch.arange(12.).reshape(3, 4)},
            'optimizer': {0: {'step': torch.tensor(28.), 'exp_avg': torch.randn(3, 4)}},
            'replay': [{'flow': np.arange(18).reshape(3, 6), 'ids': [4, 2, 9]}],
            'learning_protocol': {'entropy_coef': .002, 'version': 3},
            'protocol': {'entropy_coef': .002, 'physical_cohort': 'v2'},
            'rng': {'state': 123}, 'model_updates': 999,
        },
        'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
    }


def test_checkpoint_migration_preserves_every_learning_component(tmp_path):
    path, backup = tmp_path / 'latest.pt', tmp_path / 'original/latest.pt'
    state = checkpoint()
    before = continuation.preserved_state_digest(state)
    torch.save(state, path)
    old_file_hash = continuation.sha(path)
    result = continuation.migrate_checkpoint(path, backup, old_fingerprint='old',
        new_fingerprint='new', expected_inputs=state['input_hashes'])
    after = torch.load(path, weights_only=False)
    assert after['config_fingerprint'] == 'new'
    assert after['episode'] == 731
    assert continuation.preserved_state_digest(after) == before
    assert continuation.sha(backup) == old_file_hash
    assert result['preserved_learning_state_sha256'] == before
    assert after['method_state']['learning_protocol'] == state['method_state']['learning_protocol']
    assert after['entropy_migration']['schedule']['actor_entropy_final_coefficient'] == .0001


def test_repeat_migration_is_idempotent_with_provenance(tmp_path):
    path, backup = tmp_path / 'latest.pt', tmp_path / 'original.pt'
    state = checkpoint()
    torch.save(state, path)
    kwargs = dict(old_fingerprint='old', new_fingerprint='new', expected_inputs=state['input_hashes'])
    continuation.migrate_checkpoint(path, backup, **kwargs)
    digest = continuation.sha(path)
    assert continuation.migrate_checkpoint(path, backup, **kwargs)['status'] == 'already_migrated'
    assert continuation.sha(path) == digest


@pytest.mark.parametrize('which', ['input', 'configuration'])
def test_rejects_unrelated_checkpoint_change_without_writing(tmp_path, which):
    path, backup = tmp_path / 'latest.pt', tmp_path / 'original.pt'
    state = checkpoint()
    torch.save(state, path)
    digest = continuation.sha(path)
    with pytest.raises(ValueError):
        continuation.migrate_checkpoint(path, backup,
            old_fingerprint='unrelated' if which == 'configuration' else 'old',
            new_fingerprint='new',
            expected_inputs={'network_xml': 'different'} if which == 'input' else state['input_hashes'])
    assert continuation.sha(path) == digest
    assert not backup.exists()


def test_state_digest_detects_optimizer_replay_and_rng_changes():
    state = checkpoint()
    original = continuation.preserved_state_digest(state)
    for component in ('optimizer', 'replay', 'rng'):
        changed = deepcopy(state)
        changed['method_state'][component] = 'accidentally reset'
        assert continuation.preserved_state_digest(changed) != original
    state['config_fingerprint'] = 'permitted new identity'
    state['entropy_migration'] = {'at_episode': 731}
    assert continuation.preserved_state_digest(state) == original


def test_bfloat16_scalar_and_empty_tensor_state_digest():
    state = {'weights': torch.arange(6, dtype=torch.bfloat16),
             'step': torch.tensor(3.), 'empty': torch.empty(0)}
    assert continuation.preserved_state_digest(state) == continuation.preserved_state_digest(deepcopy(state))


def test_schedule_edit_is_confined_to_method_entropy_fields(tmp_path):
    path = tmp_path / 'city.toml'
    path.write_text('[run]\nepisodes = 5000\n[method]\nactor_entropy_mode = "population"\n'
                    'sp_prior_weight = 0.0\n[simulation]\ndemand_duration_seconds = 1800\n')
    continuation.install_schedule(path)
    import tomllib
    data = tomllib.loads(path.read_text())
    assert data['run'] == {'episodes': 5000}
    assert data['simulation'] == {'demand_duration_seconds': 1800}
    assert data['method']['sp_prior_weight'] == 0.
    for key, value in continuation.ENTROPY_SETTINGS.items():
        assert data['method'][key] == value
    first = path.read_text()
    continuation.install_schedule(path)
    assert path.read_text() == first


def test_prepared_entry_retains_output_budget_and_only_replaces_source_paths(tmp_path):
    old_source, new_source = tmp_path / 'original/source', tmp_path / 'continuation/source'
    output = tmp_path / 'original/city/online_ac'
    entry = {'id': 'city_online_ac', 'city': 'city', 'method': 'online_ac',
        'episode_budget': 5000, 'output_dir': str(output),
        'command': ['python', '-u', str(old_source / 'tools/train_toy_actor_critic.py'),
                    '--config', str(old_source / 'configs/city.toml'), '--episodes', '5000',
                    '--output-dir', str(output), '--sp-prior-weight', '0'],
        'log': 'old_log', 'state': 'old_state'}
    original = deepcopy(entry)
    result = continuation.prepared_entry(entry, new_source, tmp_path / 'continuation')
    assert entry == original
    assert result['output_dir'] == str(output)
    assert result['episode_budget'] == 5000
    assert result['command'][2] == str(new_source / 'tools/train_toy_actor_critic.py')
    assert result['command'][4] == str(new_source / 'configs/city.toml')
    assert result['command'][6] == '5000'
    assert result['command'][8] == str(output)


def test_owned_workers_excludes_completed_and_unrelated(monkeypatch, tmp_path):
    proc = tmp_path / 'proc'
    proc.mkdir()
    for pid in (101, 102, 103):
        (proc / str(pid)).mkdir()
    entries = [dict(id='unfinished'), dict(id='completed')]
    jobs = {'unfinished': dict(status='running', pid=101, process_starttime='one'),
            'completed': dict(status='completed', pid=102, process_starttime='two')}
    real_path = continuation.Path
    monkeypatch.setattr(continuation, 'Path', lambda value: proc if value == '/proc' else real_path(value))
    monkeypatch.setattr(continuation.queue, 'process_identity', lambda pid: str(pid))
    monkeypatch.setattr(continuation.queue, 'owned_alive',
                        lambda state, entry: state.get('pid') == 101 and entry['id'] == 'unfinished')
    victims = continuation.owned_workers(entries, jobs)
    assert set(victims) == {101}


def test_prepare_keeps_completed_jobs_and_exact_city_sources(monkeypatch, tmp_path):
    from argparse import Namespace
    old_root, new_root = tmp_path / 'old_batch', tmp_path / 'new_batch'
    old_root.mkdir()
    original_source = old_root / 'source'
    ingo_source = tmp_path / 'older_ingo/source'
    overlay = tmp_path / 'entropy_overlay'
    (overlay / 'tools').mkdir(parents=True)
    (overlay / 'tools/train.py').write_text('# only entropy integration\n')
    cities = continuation.queue.CITIES
    old_sources = {c: ingo_source if c == cities[0] else original_source for c in cities}
    for source in set(old_sources.values()):
        for directory in ('configs', 'tools', 'urban_dreamer', 'networks'):
            (source / directory).mkdir(parents=True, exist_ok=True)
        (source / 'tools/train.py').write_text(f'# frozen original {source.name}\n')
        for city in cities:
            (source / f'configs/{city}.toml').write_text(
                '[method]\nactor_entropy_mode = "population"\nsp_prior_weight = 0.0\n')
            (source / 'networks' / city).mkdir()
            for name in (f'{city}.net.xml', f'od_{city}.txt', 'agents.csv', 'edge_registry.json'):
                (source / 'networks' / city / name).write_text(f'{source}:{city}:{name}')
    runs, jobs = [], {}
    for city in cities:
        run = {'id': city + '_online_ac', 'city': city, 'method': 'online_ac',
               'output_dir': str(old_root / city / 'online_ac'), 'episode_budget': 5000,
               'log': 'original.log', 'state': 'original_state.json',
               'command': ['python', '-u', str(old_sources[city] / 'tools/train.py'),
                           '--config', str(old_sources[city] / f'configs/{city}.toml')]}
        runs.append(run)
        jobs[run['id']] = {'status': 'completed' if city == cities[0] else 'running'}
    manifest = {'runs': runs, 'source_snapshot': str(original_source),
                'source_snapshots_by_city': {c: str(s) for c, s in old_sources.items()},
                'input_hashes_by_city': {c: continuation.input_hashes(s, c) for c, s in old_sources.items()},
                'min_free_gb': 0}
    (old_root / 'manifest.json').write_text(json.dumps(manifest))
    (old_root / 'queue_state.json').write_text(json.dumps({'jobs': jobs, 'controller_pid': 77}))
    original_config_hash = continuation.sha(original_source / f'configs/{cities[1]}.toml')
    monkeypatch.setattr(continuation, 'controller_identity',
                        lambda root, ledger: {'pid': 77, 'process_starttime': 'identity'})
    monkeypatch.setattr(continuation, 'load_config',
                        lambda path: Namespace(semantic_fingerprint=continuation.sha(path)))
    continuation.prepare(Namespace(old_root=old_root, new_root=new_root, overlay_root=overlay,
                                   overlay_files=['tools/train.py']))
    prepared = continuation.read(new_root / 'manifest.json')
    assert prepared['runs'][0] == runs[0]
    for run in prepared['runs'][1:]:
        assert run['output_dir'] == manifest['runs'][cities.index(run['city'])]['output_dir']
        assert run['episode_budget'] == 5000
        assert run['command'][2] == str(new_root / 'source/tools/train.py')
    assert continuation.sha(original_source / f'configs/{cities[1]}.toml') == original_config_hash
    assert prepared['source_snapshots_by_city'][cities[0]] == str(new_root / 'sources' / cities[0])
    assert continuation.input_hashes(new_root / 'source', cities[1]) == manifest['input_hashes_by_city'][cities[1]]


def test_resume_unmigrated_checkpoint_with_existing_original_backup(tmp_path):
    path, backup = tmp_path / 'latest.pt', tmp_path / 'original.pt'
    state = checkpoint()
    torch.save(state, path)
    import shutil
    shutil.copy2(path, backup)
    old_sha = continuation.sha(backup)
    result = continuation.migrate_checkpoint(path, backup, old_fingerprint='old',
        new_fingerprint='new', expected_inputs=state['input_hashes'])
    assert result['status'] == 'migrated'
    assert continuation.sha(backup) == old_sha


def test_resume_rejects_mismatched_existing_original_backup(tmp_path):
    path, backup = tmp_path / 'latest.pt', tmp_path / 'original.pt'
    state = checkpoint()
    torch.save(state, path)
    changed = deepcopy(state)
    changed['episode'] = 732
    torch.save(changed, backup)
    with pytest.raises(ValueError, match='existing original backup differs'):
        continuation.migrate_checkpoint(path, backup, old_fingerprint='old',
            new_fingerprint='new', expected_inputs=state['input_hashes'])


def test_already_migrated_checkpoint_requires_full_state_equal_to_backup(tmp_path):
    path, backup = tmp_path / 'latest.pt', tmp_path / 'original.pt'
    state = checkpoint()
    torch.save(state, path)
    kwargs = dict(old_fingerprint='old', new_fingerprint='new', expected_inputs=state['input_hashes'])
    continuation.migrate_checkpoint(path, backup, **kwargs)
    migrated = torch.load(path, weights_only=False)
    migrated['method_state']['replay'][0]['flow'][1, 2] += 1
    torch.save(migrated, path)
    with pytest.raises(ValueError, match='learning state differs from backup'):
        continuation.migrate_checkpoint(path, backup, **kwargs)


def test_fast_digest_covers_numpy_content_dtype_shape_and_object_arrays():
    state = {'array': np.array([1, 2, 3]), 'objects': np.array(['a', {'b': 2}], dtype=object)}
    before = continuation.preserved_state_digest(state)
    assert continuation.preserved_state_digest(deepcopy(state)) == before
    for array in (np.array([1, 2, 4]), np.array([1., 2., 3.]), np.array([[1, 2, 3]])):
        changed = deepcopy(state)
        changed['array'] = array
        assert continuation.preserved_state_digest(changed) != before
    changed = deepcopy(state)
    changed['objects'][1]['b'] = 3
    assert continuation.preserved_state_digest(changed) != before


def test_fast_digest_handles_empty_and_noncontiguous_numpy_buffers():
    state = {'empty': np.empty((0, 3)), 'strided': np.arange(12).reshape(3, 4)[:, ::2],
             'scalar': np.array(2.)}
    assert continuation.preserved_state_digest(state) == continuation.preserved_state_digest(deepcopy(state))


def test_resume_handoff_skips_proven_jobs_and_sends_no_signals(monkeypatch, tmp_path):
    from argparse import Namespace
    old_root, new_root = tmp_path / 'old', tmp_path / 'new'
    old_root.mkdir()
    (new_root / 'handoff_original').mkdir(parents=True)
    output = old_root / 'city/online_ac'
    output.mkdir(parents=True)
    (output / 'latest.pt').write_bytes(b'fully verified migrated checkpoint')
    backup = new_root / 'handoff_original/original.pt'
    backup.write_bytes(b'original checkpoint')
    entry = {'id': 'city_online_ac', 'city': 'city', 'method': 'online_ac',
             'output_dir': str(output), 'episode_budget': 5000, 'required_gpu_mib': 0,
             'command': ['python', '-u', 'train.py']}
    state = {'status': 'running', 'pid': 111, 'required_gpu_mib': 0,
             'oom_retries': 3, 'observed_peak_mib': 567}
    old_manifest = {'runs': [entry], 'input_hashes_by_city': {}}
    ledger = {'controller_pid': 77, 'jobs': {entry['id']: state}}
    manifest = {**old_manifest, 'controller_session': 'new_queue'}
    proof = {'backup': str(backup), 'original_sha256': continuation.sha(backup),
             'migrated_sha256': continuation.sha(output / 'latest.pt'),
             'preserved_learning_state_sha256': 'prior verified semantic proof'}
    protocol = {'phase': 'migrating', 'old_root': str(old_root), 'source_revisions': {},
                'controller': {'pid': 77, 'process_starttime': 'identity'},
                'checkpoint_migrations': {entry['id']: {'latest.pt': proof}},
                'continuation_episodes': {entry['id']: 731}}
    for path, value in ((old_root / 'manifest.json', old_manifest),
                        (old_root / 'queue_state.json', ledger),
                        (new_root / 'manifest.json', manifest),
                        (new_root / 'handoff_protocol.json', protocol),
                        (new_root / 'handoff_original/queue_state.json', ledger)):
        path.write_text(json.dumps(value))
    monkeypatch.setattr(continuation, 'controller_identity', lambda *args: protocol['controller'])
    monkeypatch.setattr(continuation, 'controller_stopped', lambda pid: True)
    monkeypatch.setattr(continuation, 'owned_workers', lambda *args: {})
    monkeypatch.setattr(continuation.queue, 'durable_episode', lambda entry: 731)
    def forbidden(*args, **kwargs):
        pytest.fail('recovery must skip completed migrations and signals')
    monkeypatch.setattr(continuation.os, 'kill', forbidden)
    monkeypatch.setattr(continuation, 'migrate_checkpoint', forbidden)
    monkeypatch.setattr(continuation.queue, 'recover_output', forbidden)
    monkeypatch.setattr(continuation.subprocess, 'run',
                        lambda args, **kwargs: Namespace(returncode=1 if 'has-session' in args else 0))
    continuation.apply(Namespace(old_root=old_root, new_root=new_root, resume_handoff=True))
    after = continuation.read(new_root / 'handoff_protocol.json')
    assert after['phase'] == 'launched'
    assert after['checkpoint_migrations'][entry['id']]['latest.pt'] == proof
    resumed = continuation.read(new_root / 'queue_state.json')['jobs'][entry['id']]
    assert resumed['continuation_episode'] == 731
    assert resumed['observed_peak_mib'] == 567
    assert resumed['oom_retries'] == 3


def test_resume_handoff_rejects_other_phases_before_actions(tmp_path):
    from argparse import Namespace
    root = tmp_path / 'new'
    root.mkdir()
    (root / 'handoff_protocol.json').write_text(json.dumps({'phase': 'launched'}))
    with pytest.raises(ValueError, match='only valid during migration'):
        continuation.apply(Namespace(old_root=tmp_path / 'old', new_root=root, resume_handoff=True))
