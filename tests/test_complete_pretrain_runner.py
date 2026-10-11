"""Acceptance artifacts must prove codec readiness before the full FM runner starts."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from types import SimpleNamespace

import pytest
import torch

from tools import run_complete_pretrain as runner
from tools.train_codec_ready import FrameSampler, GATES, gate_result
from urban_dreamer.learning.offline_data import _identity


def write_json(path, value):
    path.write_text(json.dumps(value))


def test_migrated_resume_revalidates_codec_and_saves_proof_before_training(tmp_path, monkeypatch):
    (tmp_path / 'single_step_migration.json').write_text('{}')
    trainer = object.__new__(runner.MonitoredTrainer)
    trainer.config = SimpleNamespace(run=SimpleNamespace(output_dir=tmp_path), semantic_fingerprint='target')
    trainer.monitor_path = tmp_path / 'status.json'
    trainer._runtime_profile = {}
    trainer.prepared_inputs = object()
    trainer.codec_validation_data = SimpleNamespace(summary={'imported_episodes': 2})
    trainer.metrics = {'codec_ready': 0, 'world_model_initialization_epochs_completed': 0,
                       'single_step_migration_codec_revalidation_required': True}
    codec = torch.nn.Linear(2, 2)
    trainer.world_model = SimpleNamespace(flow_projector=codec)
    calls = []
    monkeypatch.setattr(runner.MainlineTrainer, 'load_state_dict', lambda self, state: calls.append('load'))
    monkeypatch.setattr(trainer, 'codec_ready', lambda: bool(trainer.metrics['codec_ready']))
    def validate():
        calls.append('validate')
        trainer.metrics['codec_ready'] = 1
        return {'codec_ready': True}
    monkeypatch.setattr(trainer, '_validate_codec', validate)
    monkeypatch.setattr(trainer, 'validate_codec_ready', lambda: calls.append('gate'))
    monkeypatch.setattr(trainer, 'state_dict', lambda: dict(metrics=dict(trainer.metrics)))
    monkeypatch.setattr(runner, 'combined_input_hash', lambda prepared: 'net')
    monkeypatch.setattr(runner, 'save_checkpoint', lambda **kwargs: calls.append(('save', kwargs)))
    trainer.load_state_dict({})
    assert calls[:3] == ['load', 'validate', 'gate']
    saved = calls[3][1]
    assert saved['phase'] is runner.CheckpointPhase.PRETRAIN
    assert saved['next_unit'] == 0
    assert not saved['training_state']['metrics']['single_step_migration_codec_revalidation_required']
    assert not any(parameter.requires_grad for parameter in codec.parameters())
    # A valid resumed proof is reused; no redundant full-partition validation.
    trainer.load_state_dict({})
    assert calls[-1] == 'load'


def fixture_artifacts(tmp_path):
    codec_dir = tmp_path / 'codec'
    codec_dir.mkdir()
    physical = {'vehicle_max_speed_mps': 40., 'demand_duration_seconds': 1800.}
    entries = {'train': [], 'validation': []}
    cache_entries = []
    for split in entries:
        source = tmp_path / f'{split}_source.npz'
        source.write_bytes(b'fixture source membership')
        compact = tmp_path / f'{split}_compact.npz'
        compact.write_bytes(b'fixture compact artifact existence')
        entries[split].append(dict(path=str(source), frames=2, demand_group=split + '-demand'))
        cache_entries.append(dict(path=str(source), cache_path=str(compact),
            cache_content_hash=split + '-artifact-hash', split=split, windows=2,
            demand_group=split + '-demand'))
    manifest = dict(entries=entries, physical=physical, input_hashes={'net': 'frozen-net'},
                    edge_ids=['a', 'b'], scope='frozen pilot')
    manifest['identity'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    write_json(codec_dir / 'codec_manifest.json', manifest)
    metrics = {key: threshold / 2 for key, threshold in GATES.items()}
    result = dict(status='raw_gate_passed', update=10, manifest_identity=manifest['identity'],
                  validation=metrics, gate=gate_result(metrics),
                  hybrid_validation=metrics.copy(), hybrid_gate=gate_result(metrics))
    write_json(codec_dir / 'result.json', result)
    parameter = torch.nn.Parameter(torch.ones(1))
    checkpoint = dict(schema_version=1, codec={'weight': torch.ones(1)},
        optimizer=torch.optim.AdamW([parameter]).state_dict(),
        sampler=FrameSampler(entries['train'], 2, 42).state_dict(),
        manifest_identity=manifest['identity'], update=10, batch_size=2, codec_speed_positive=True)
    torch.save(checkpoint, codec_dir / 'codec_latest.pt')
    cache = dict(schema_version=1, status='complete',
        contract=dict(city='toy', input_hashes={'net': 'frozen-net'}, physical=physical),
        entries=cache_entries, sources=copy.deepcopy(cache_entries), duplicate_files=0,
        train_episodes=1, validation_episodes=1)
    cache['identity'] = _identity(cache)
    cache_path = tmp_path / 'cache_manifest.json'
    write_json(cache_path, cache)
    return codec_dir, cache_path, result, checkpoint, cache


def readiness(codec_dir, cache_path):
    return runner.prerequisite_state(codec_dir, cache_path, expected_train=1, expected_validation=1)


def test_acceptance_binds_real_gate_metrics_source_sampler_and_frozen_parameter_objects(tmp_path):
    codec_dir, cache_path, _, _, cache = fixture_artifacts(tmp_path)
    accepted = readiness(codec_dir, cache_path)
    assert accepted['ready'] and accepted['codec_ready'] and accepted['cache_ready']
    assert accepted['proof']['cache_manifest_identity'] == cache['identity']
    assert accepted['proof']['codec_update'] == 10
    # The runner reuses these checked objects, rather than reopening a mutable
    # producer checkpoint after waiting for the cache to finish.
    torch.save({'codec': {'weight': torch.full((1,), 99.)}}, codec_dir / 'codec_latest.pt')
    torch.testing.assert_close(accepted['checkpoint']['codec']['weight'], torch.ones(1))


def test_forged_ready_flag_cannot_override_failed_physical_metrics(tmp_path):
    codec_dir, cache_path, result, _, _ = fixture_artifacts(tmp_path)
    result['validation']['codec_count_mae_occupied'] = 2.
    write_json(codec_dir / 'result.json', result)
    with pytest.raises(ValueError, match='reported codec gate'):
        readiness(codec_dir, cache_path)


def test_terminal_hybrid_failure_reports_recovery_instead_of_waiting_forever(tmp_path):
    codec_dir, cache_path, result, _, _ = fixture_artifacts(tmp_path)
    result['hybrid_validation']['codec_count_mae_occupied'] = 2.
    result['hybrid_gate'] = gate_result(result['hybrid_validation'])
    write_json(codec_dir / 'result.json', result)
    with pytest.raises(RuntimeError, match='recovery must run'):
        readiness(codec_dir, cache_path)
    write_json(codec_dir / 'progress.json', {'status': 'hybrid_training_active'})
    pending = readiness(codec_dir, cache_path)
    assert not pending['ready'] and pending['recovery'] == 'hybrid_training_active'


def test_active_recovery_uses_fresh_progress_heartbeat_and_rejects_stale_producer(tmp_path):
    codec_dir, cache_path, result, _, _ = fixture_artifacts(tmp_path)
    result['status'] = 'hybrid_training_active'
    write_json(codec_dir / 'result.json', result)
    stale = time.time() - 301
    os.utime(codec_dir / 'result.json', (stale, stale))
    # A recovery marker is authoritative while training, even if prior pilot
    # metrics happen to be passing: new cohort weights have not been validated.
    result['hybrid_gate']['passed'] = False
    write_json(codec_dir / 'result.json', result)
    os.utime(codec_dir / 'result.json', (stale, stale))
    with pytest.raises(RuntimeError, match='heartbeat is stale'):
        readiness(codec_dir, cache_path)
    write_json(codec_dir / 'progress.json', {'status': 'hybrid_training_active'})
    assert not readiness(codec_dir, cache_path)['ready']


def test_recovery_marker_can_precede_any_final_result(tmp_path):
    codec_dir = tmp_path / 'codec'
    codec_dir.mkdir()
    write_json(codec_dir / 'progress.json', {'status': 'hybrid_training_active'})
    assert not readiness(codec_dir, tmp_path / 'not-yet-built.json')['ready']


def test_final_passing_result_overrides_old_recovery_marker(tmp_path):
    codec_dir, cache_path, _, _, _ = fixture_artifacts(tmp_path)
    write_json(codec_dir / 'progress.json', {'status': 'hybrid_training_active'})
    assert readiness(codec_dir, cache_path)['ready']


@pytest.mark.parametrize('mutate,message', [
    (lambda state: state.__setitem__('update', 9), 'source/update'),
    (lambda state: state.__setitem__('manifest_identity', 'other-source'), 'source/update'),
    (lambda state: state.__setitem__('codec_speed_positive', False), 'positive physical'),
    (lambda state: state.__setitem__('schema_version', 2), 'invalid schema'),
    (lambda state: state['sampler'].__setitem__('cursor', 99), 'sampler is invalid'),
    (lambda state: state['codec']['weight'].fill_(float('nan')), 'finite tensors'),
])
def test_unverified_checkpoint_cannot_be_promoted_to_fm(tmp_path, mutate, message):
    codec_dir, cache_path, _, checkpoint, _ = fixture_artifacts(tmp_path)
    mutate(checkpoint)
    torch.save(checkpoint, codec_dir / 'codec_latest.pt')
    with pytest.raises(ValueError, match=message):
        readiness(codec_dir, cache_path)


def test_full_runner_requires_2920_training_and_520_held_out_episodes(tmp_path):
    _, _, _, _, cache = fixture_artifacts(tmp_path)
    with pytest.raises(ValueError, match='2920 train/520 held-out'):
        runner.checked_cache_manifest(cache)
    cache['entries'] = []
    cache.pop('identity')
    cache['identity'] = _identity(cache)
    with pytest.raises(ValueError, match='empty episodes'):
        runner.checked_cache_manifest(cache)


def test_cache_held_out_group_cannot_overlap_codec_training(tmp_path):
    codec_dir, cache_path, _, _, cache = fixture_artifacts(tmp_path)
    cache['entries'][1]['demand_group'] = 'train-demand'
    cache['sources'][1]['demand_group'] = 'train-demand'
    cache.pop('identity')
    cache['identity'] = _identity(cache)
    write_json(cache_path, cache)
    with pytest.raises(ValueError, match='share demand identities'):
        readiness(codec_dir, cache_path)


def test_unfinished_cache_waits_without_declaring_fm_ready(tmp_path):
    codec_dir, _, _, _, _ = fixture_artifacts(tmp_path)
    state = readiness(codec_dir, tmp_path / 'building.json')
    assert state == dict(ready=False, codec_ready=True, cache_ready=False)


def test_progress_checkpoint_saves_first_complete_group_then_time_or_update_intervals(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint
    trainer = runner.MonitoredTrainer.__new__(runner.MonitoredTrainer)
    trainer.config = SimpleNamespace(run=SimpleNamespace(output_dir=tmp_path / 'run'),
                                     semantic_fingerprint='frozen-full-fm')
    trainer.prepared_inputs = SimpleNamespace(input_hashes={'network': 'frozen-source'})
    trainer.monitor_path = tmp_path / 'monitor.json'
    trainer.metrics = {'world_model_updates': 1, 'world_model_initialization_samples_seen': 8}
    trainer._last_checkpoint_time = None
    trainer._last_checkpoint_update = None
    clock = {'now': 100.}
    monkeypatch.setattr(runner, 'time', SimpleNamespace(monotonic=lambda: clock['now'],
        strftime=time.strftime, gmtime=time.gmtime))
    def state():
        update = int(trainer.metrics['world_model_updates'])
        return dict(reward_contract='event_q_counterfactual_global_individual_timeout_v2',
            models={'world_model': {'weight': torch.tensor([float(update)])}},
            optimizers={'world_model_initial': {'step': update}},
            replay={'offline': {'sampler_cursor': update}, 'online': {}},
            rng={'torch_cpu': torch.tensor([update % 256], dtype=torch.uint8)},
            metrics=dict(trainer.metrics), next_environment_episode=0)
    trainer.state_dict = state
    def restored():
        return load_checkpoint(output_dir=trainer.config.run.output_dir,
            expected_config_fingerprint='frozen-full-fm',
            expected_input_hash=runner.combined_input_hash(trainer.prepared_inputs))
    trainer._save_pretrain_progress(3)
    first = restored()
    assert first.phase is CheckpointPhase.PRETRAIN and first.next_unit == 3
    assert first.training_state['models']['world_model']['weight'].item() == 1
    assert first.training_state['replay']['offline']['sampler_cursor'] == 1
    assert first.training_state['optimizers']['world_model_initial']['step'] == 1
    assert first.training_state['metrics']['pretrain_progress_checkpoint_updates'] == 1
    assert first.training_state['metrics']['pretrain_progress_checkpoint_utc']
    original_bytes = first.path.read_bytes()
    clock['now'] = 200.
    trainer.metrics['world_model_updates'] = 400
    trainer._save_pretrain_progress(3)
    assert first.path.read_bytes() == original_bytes
    clock['now'] = 401.
    trainer._save_pretrain_progress(3)
    timed = restored()
    assert timed.training_state['models']['world_model']['weight'].item() == 400
    assert timed.training_state['replay']['offline']['sampler_cursor'] == 400
    clock['now'] = 402.
    trainer.metrics['world_model_updates'] = 900
    trainer._save_pretrain_progress(3)
    counted = restored()
    assert counted.next_unit == 3
    assert counted.training_state['optimizers']['world_model_initial']['step'] == 900
    assert counted.training_state['metrics']['pretrain_progress_checkpoint_updates'] == 900
    assert json.loads(trainer.monitor_path.read_text())['phase'] == 'fm_pretrain_checkpoint'


@pytest.mark.parametrize('tool', ['pretrain_offline', 'verify_event_mainline'])
def test_historical_tools_fail_before_data_reads_outputs_or_codec_gate_bypass(tmp_path, capsys, tool):
    from tools import pretrain_offline, verify_event_mainline
    output = tmp_path / 'must-not-be-created'
    if tool == 'pretrain_offline':
        invoke = pretrain_offline.main
        args = ['--config', str(tmp_path / 'not-read.toml'), '--census-dir', str(tmp_path / 'not-read-census'),
                '--output-dir', str(output)]
    else:
        invoke = verify_event_mainline.main
        args = ['--archive', str(tmp_path / 'not-read.npz'), '--output-dir', str(output)]
    with pytest.raises(SystemExit) as error:
        invoke(args)
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert 'Historical' in message
    assert 'independent' in message
    assert 'tools/run_complete_pretrain.py' in message
    assert 'No' in message and 'training has started' in message
    assert not output.exists()


def warm_start_fixture(tmp_path):
    from types import SimpleNamespace
    class World(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.flow_projector = torch.nn.Linear(2, 2)
            self.dynamics = torch.nn.Linear(2, 2)
    trainer = SimpleNamespace(world_model=World(), actor=torch.nn.Linear(2, 1),
        latent_critic=torch.nn.Linear(2, 1), event_critic=torch.nn.Linear(2, 1),
        prepared_inputs=SimpleNamespace(input_hashes={'network': 'frozen-net'}),
        metrics=dict(codec_updates=0., codec_ready=0., world_model_updates=0.,
                     world_model_initialization_epochs_completed=0.))
    trainer.optimizers = {'codec': torch.optim.AdamW(trainer.world_model.flow_projector.parameters()),
        'world_model_initial': torch.optim.AdamW(trainer.world_model.dynamics.parameters())}
    codec = torch.nn.Linear(2, 2)
    with torch.no_grad():
        codec.weight.fill_(.25)
        codec.bias.fill_(.5)
    optimizer = torch.optim.AdamW(codec.parameters())
    codec(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    world = {'flow_projector.' + name: value for name, value in codec.state_dict().items()}
    world['dynamics.weight'] = torch.full_like(trainer.world_model.dynamics.weight, 99.)
    payload = dict(schema_version=8, input_hash=runner.combined_input_hash(trainer.prepared_inputs),
        training_state=dict(models=dict(world_model=world, actor={'weight': torch.full((1, 2), 99.)},
                                        latent_critic={}, event_critic={}),
            optimizers={'codec': optimizer.state_dict(), 'world_model_initial': {'legacy_bad_dynamics': True}},
            metrics=dict(codec_updates=6500., codec_ready=True, world_model_updates=500.,
                         world_model_initialization_epochs_completed=2., codec_validation_updates=6500)))
    path = tmp_path / 'wrong-label-world-model.pt'
    torch.save(payload, path)
    return trainer, path, payload, codec


def test_schema8_import_only_warms_codec_and_never_reuses_wrong_label_dynamics_or_gate(tmp_path):
    trainer, path, _, codec = warm_start_fixture(tmp_path)
    untouched = {name: copy.deepcopy(getattr(trainer, name).state_dict())
                 for name in ('actor', 'latent_critic', 'event_critic')}
    dynamics = copy.deepcopy(trainer.world_model.dynamics.state_dict())
    optimizer = copy.deepcopy(trainer.optimizers['world_model_initial'].state_dict())
    proof = runner.import_codec_only(trainer, path)
    for key, expected in codec.state_dict().items():
        torch.testing.assert_close(trainer.world_model.flow_projector.state_dict()[key], expected)
    for key, expected in dynamics.items():
        torch.testing.assert_close(trainer.world_model.dynamics.state_dict()[key], expected)
    for name, previous in untouched.items():
        for key, expected in previous.items():
            torch.testing.assert_close(getattr(trainer, name).state_dict()[key], expected)
    assert trainer.optimizers['world_model_initial'].state_dict() == optimizer
    assert trainer.metrics['codec_updates'] == 6500
    assert trainer.metrics['codec_ready'] == 0
    assert trainer.metrics['world_model_updates'] == 0
    assert trainer.metrics['world_model_initialization_epochs_completed'] == 0
    assert 'codec_validation_updates' not in trainer.metrics
    assert proof['imported_components'] == ['flow_projector', 'codec_optimizer']
    assert proof['source_schema'] == 8
    assert {int(state['step']) for state in trainer.optimizers['codec'].state.values()} == {1}


@pytest.mark.parametrize('mutate,message', [
    (lambda payload: payload.__setitem__('schema_version', 7), 'historical schema8'),
    (lambda payload: payload.__setitem__('input_hash', 'different-network'), 'different prepared network'),
    (lambda payload: payload['training_state']['metrics'].__setitem__('codec_updates', 1.5), 'invalid codec update'),
    (lambda payload: payload['training_state']['optimizers'].pop('codec'), 'lacks codec optimizer'),
    (lambda payload: payload['training_state']['models']['world_model']['flow_projector.weight'].fill_(float('nan')),
     'finite flow_projector'),
])
def test_codec_only_warm_import_rejects_unbound_or_invalid_sources(tmp_path, mutate, message):
    trainer, path, payload, _ = warm_start_fixture(tmp_path)
    mutate(payload)
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        runner.import_codec_only(trainer, path)


def test_runtime_monitor_reports_per_transition_birth_exclusion_scope():
    assert runner.VOLUME_SCOPE == 'previous_state_excluding_current_births'


@pytest.mark.parametrize('flag', ['--threads', '--batch-size', '--accumulation-steps',
                                 '--space-chunk', '--time-chunk', '--mlp-chunk'])
def test_runner_rejects_invalid_runtime_profile_before_reading_data(tmp_path, capsys, flag):
    args = ['--config', str(tmp_path / 'not-read.toml'), '--codec-dir', str(tmp_path / 'not-read-codec'),
            '--cache-manifest', str(tmp_path / 'not-read.json'), '--output-dir', str(tmp_path / 'no-output'),
            flag, '0']
    with pytest.raises(SystemExit) as error:
        runner.main(args)
    assert error.value.code == 2
    assert 'must be positive' in capsys.readouterr().err
    assert not (tmp_path / 'no-output').exists()


@pytest.mark.parametrize('value', ['0', '-.1', '1.1', 'nan', 'inf'])
def test_runner_gpu_budget_is_finite_positive_and_at_most_one(tmp_path, capsys, value):
    args = ['--config', str(tmp_path / 'not-read.toml'), '--codec-dir', str(tmp_path / 'not-read-codec'),
            '--cache-manifest', str(tmp_path / 'not-read.json'), '--output-dir', str(tmp_path / 'no-output'),
            '--gpu-memory-fraction', value]
    with pytest.raises(SystemExit) as error:
        runner.main(args)
    assert error.value.code == 2
    assert 'gpu memory fraction must be finite' in capsys.readouterr().err
    assert not (tmp_path / 'no-output').exists()


def test_exact_run_config_export_roundtrips_newscope_profiles_and_optional_cache_paths(tmp_path):
    from dataclasses import replace
    from test_urban_dreamer_config import _write_config
    from urban_dreamer.workflow.config import load_config, _semantic_fingerprint
    config = load_config(_write_config(tmp_path / 'source.toml'))
    config = replace(config, run=replace(config.run, output_dir=tmp_path / 'run', device='cuda:1'),
        resources=replace(config.resources, world_model_initialization_batch_size=4,
            world_model_gradient_accumulation_steps=2, world_model_pretrain_space_chunk=16,
            world_model_pretrain_time_chunk=32768, world_model_pretrain_mlp_chunk=16384),
        data=replace(config.data, pretrain_cache_manifest=tmp_path / 'cache/manifest.json'))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(inputs=config.inputs,
        run=config.run, simulation=config.simulation, resources=config.resources,
        method=config.method, data=config.data, codec=config.codec))
    destination = tmp_path / 'run/run_config.toml'
    runner.export_run_config(config, destination)
    restored = load_config(destination)
    assert restored.semantic_fingerprint == config.semantic_fingerprint
    assert restored.resolved_snapshot() == config.resolved_snapshot()
    assert 'null' not in destination.read_text()
    assert restored.data.validation_cache_manifest is None
    profile = runner.runtime_profile(restored, gpu_memory_fraction=.08, threads=4)
    assert profile['effective_batch_size'] == 8
    assert profile['activation_checkpointing'] and profile['space_chunk'] == 16
    assert profile['gpu_memory_fraction'] == .08


def migration_fixture(tmp_path):
    from types import SimpleNamespace
    from dataclasses import replace
    from test_urban_dreamer_config import _write_config
    from urban_dreamer.workflow.config import load_config
    source_config = load_config(_write_config(tmp_path / 'source.toml'))
    target_config = replace(source_config, resources=replace(source_config.resources,
        world_model_initialization_sequence_length=3, world_model_pretrain_precision='bf16',
        world_model_initialization_batch_size=16, world_model_gradient_accumulation_steps=1))
    class Dataset:
        def __init__(self, split):
            self.manifest = dict(schema_version=1, contract=dict(source_manifest='same-frozen-full-cache',
                split=split, sequence_length=3, target_deadline_seconds=3800), entries=[{'path': split}], files=[])
            self.manifest['identity'] = _identity(self.manifest)
        def state_dict(self):
            return dict(schema_version=1, manifest_identity=self.manifest['identity'], epoch_order=[],
                        episode_cursor=0, start_order=[], start_cursor=0, epochs_completed=0)
    trainer = SimpleNamespace(config=target_config,
        prepared_inputs=SimpleNamespace(input_hashes={'net': 'unchanged-net'}),
        offline_data=Dataset('train'), codec_validation_data=Dataset('validation'))
    trainer.load_state_dict = lambda state: setattr(trainer, 'loaded', state)
    def old_sampler(dataset):
        manifest = copy.deepcopy(dataset.manifest)
        manifest['contract']['sequence_length'] = 16
        state = dataset.state_dict()
        state['manifest_identity'] = _identity({key: value for key, value in manifest.items() if key != 'identity'})
        state.update(epoch_order=[0], start_order=[0, 2, 1], start_cursor=2)
        return state
    original = dict(reward_contract='event_q_counterfactual_global_individual_timeout_v2',
        models={'world_model': {'weight': torch.tensor([[.2, .3]])},
                'actor': {'weight': torch.tensor([[.4, .5]])},
                'latent_critic': {'weight': torch.tensor([[.6, .7]])},
                'event_critic': {'weight': torch.tensor([[.8, .9]])}},
        optimizers={'world_model_initial': {'state': {0: {'step': torch.tensor(1234)}}},
                    'codec': {'state': {0: {'step': torch.tensor(6500)}}}},
        replay={'offline': old_sampler(trainer.offline_data),
                'codec_validation': old_sampler(trainer.codec_validation_data), 'online': {'episodes': []}},
        rng={'torch_cpu': torch.tensor([2, 3, 4], dtype=torch.uint8)},
        metrics={'world_model_updates': 1234., 'codec_updates': 6500., 'codec_ready': True,
            'codec_validation_manifest_identity': 'legacy-proof', 'codec_hybrid_validation_codec_count_mae_all': .01,
            'world_model_initialization_epochs_completed': 0., 'world_model_initialization_samples_seen': 9872.,
            'world_model_initialization_updates': 1234., 'world_model_initialization_batches_completed': 2468.,
            'world_model_epoch_samples': 9872., 'world_model_epoch_batches': 2468., 'world_model_epoch_updates': 1234.,
            'world_model_epoch_loss_sum': 12., 'world_model_epoch_readout_loss_sum': 13.,
            'codec_epochs_completed': 0., 'online_episodes_completed': 0., 'evaluation_episodes_completed': 0.},
        next_environment_episode=0)
    directory = tmp_path / 'trusted-source'
    (directory / 'checkpoints').mkdir(parents=True)
    source_snapshot = source_config.resolved_snapshot()
    source_snapshot['method'].pop('exposure_readout_supervision')  # Schema9 predates observed-endpoint default.
    source_snapshot['resources'].pop('world_model_pretrain_precision')
    write_json(directory / 'resolved_config.json', source_snapshot)
    checkpoint = dict(schema_version=9, phase='pretrain', next_unit=0,
        input_hash=runner.combined_input_hash(trainer.prepared_inputs), training_state=original)
    path = directory / 'checkpoints/latest.pt'
    torch.save(checkpoint, path)
    return trainer, path, checkpoint, source_snapshot


def test_correct_label_migration_preserves_all_models_adam_rng_but_restarts_full_l3_coverage(tmp_path):
    trainer, path, checkpoint, _ = migration_fixture(tmp_path)
    proof = runner.migrate_correct_scope_checkpoint(trainer, path)
    loaded = trainer.loaded
    source = checkpoint['training_state']
    for model, weights in source['models'].items():
        for name, value in weights.items():
            torch.testing.assert_close(loaded['models'][model][name], value)
    assert loaded['optimizers']['world_model_initial']['state'][0]['step'] == 1234
    assert loaded['optimizers']['codec']['state'][0]['step'] == 6500
    torch.testing.assert_close(loaded['rng']['torch_cpu'], source['rng']['torch_cpu'])
    assert loaded['metrics']['world_model_updates'] == 1234
    assert loaded['metrics']['codec_updates'] == 6500
    for name in source['metrics']:
        if name.startswith('world_model_initialization_') or name.startswith('world_model_epoch_'):
            assert loaded['metrics'][name] == 0
    assert loaded['replay']['offline']['start_cursor'] == 0
    assert loaded['replay']['offline']['epoch_order'] == []
    assert loaded['replay']['codec_validation']['start_cursor'] == 0
    assert loaded['metrics']['codec_ready'] == 0
    assert 'codec_validation_manifest_identity' not in loaded['metrics']
    assert 'codec_hybrid_validation_codec_count_mae_all' not in loaded['metrics']
    assert loaded['metrics']['migration_source_initialization_counters']['world_model_initialization_samples_seen'] == 9872
    assert proof['old_sequence_length'] == 16 and proof['new_sequence_length'] == 3
    assert proof['sampler_reset'] and proof['codec_revalidation_required']
    assert loaded['metrics']['migration_source_schema'] == 9


@pytest.mark.parametrize('mutate,message', [
    (lambda payload: payload.__setitem__('schema_version', 8), 'correct-label schema9'),
    (lambda payload: payload.__setitem__('input_hash', 'wrong-net'), 'different prepared inputs'),
    (lambda payload: payload['training_state'].__setitem__('next_environment_episode', 1), 'online or evaluated'),
    (lambda payload: payload['training_state']['metrics'].__setitem__('online_episodes_completed', 1), 'online or evaluated'),
    (lambda payload: payload['training_state']['replay']['online']['episodes'].append({}), 'online or evaluated'),
    (lambda payload: payload['training_state']['replay']['offline'].__setitem__('manifest_identity', 'other-data'), 'manifest differs'),
])
def test_migration_rejects_wrong_labels_unbound_data_and_any_online_training(tmp_path, mutate, message):
    trainer, path, checkpoint, _ = migration_fixture(tmp_path)
    mutate(checkpoint)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match=message):
        runner.migrate_correct_scope_checkpoint(trainer, path)
    assert not hasattr(trainer, 'loaded')


@pytest.mark.parametrize('section,key,value,message', [
    ('method', 'transition_volume_scope', 'initial_cohort', 'source labels'),
    ('method', 'fm_steps', 99, 'method.fm_steps'),
    ('simulation', 'vehicle_max_speed_mps', 20., 'simulation contract'),
    ('codec', 'occupied_vehicle_mae_max', 9., 'codec contract'),
])
def test_migration_source_configuration_allows_only_reviewed_runtime_or_readout_changes(tmp_path, section, key, value, message):
    trainer, path, _, snapshot = migration_fixture(tmp_path)
    snapshot[section][key] = value
    write_json(path.parent.parent / 'resolved_config.json', snapshot)
    with pytest.raises(ValueError, match=message):
        runner.migrate_correct_scope_checkpoint(trainer, path)


def test_calibrated_import_only_changes_codec_and_truthful_metrics():
    codec = torch.nn.Linear(2, 2)
    initial_actor = torch.nn.Linear(2, 2)
    actor_before = copy.deepcopy(initial_actor.state_dict())
    optimizer = torch.optim.Adam(codec.parameters(), lr=.001)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    trainer = SimpleNamespace(world_model=SimpleNamespace(flow_projector=codec),
        actor=initial_actor, optimizers={'codec': optimizer}, metrics={'codec_ready': 1., 'world_model_updates': 0.},
        _set_dynamics_codec_gradients=lambda: None)
    target = {name: torch.ones_like(value) for name, value in codec.state_dict().items()}
    report = runner.import_calibrated_codec(trainer, {'codec': target, 'updates': 20682})
    assert all(torch.equal(codec.state_dict()[key], value) for key, value in target.items())
    assert all(torch.equal(initial_actor.state_dict()[key], value) for key, value in actor_before.items())
    assert optimizer.state_dict() == optimizer_before
    assert report['all_optimizers_fresh'] and report['imported_components'] == ['flow_projector']
    assert trainer.metrics['codec_ready'] == 0 and trainer.metrics['codec_updates'] == 0
    assert trainer.metrics['world_model_updates'] == 0


def test_calibrated_warm_start_skips_encoder_phase_without_faking_readiness():
    trainer = object.__new__(runner.MonitoredTrainer)
    trainer.metrics = {'calibrated_codec_only_warm_start': True, 'codec_ready': 0.}
    calls = []
    trainer.validate_codec_pretraining_ready = lambda: calls.append('admission')
    trainer._set_dynamics_codec_gradients = lambda: calls.append('decoder_only')
    trainer.codec_ready = lambda: False
    result = trainer.initialize_codec_epoch(0)
    assert calls == ['admission', 'decoder_only']
    assert result['skipped'] and result['codec_ready'] is False
    assert trainer.metrics['codec_ready'] == 0.


def test_calibrated_import_requires_explicit_continuous_admission(tmp_path):
    config = SimpleNamespace(codec=SimpleNamespace(decoder_allow_unready=False))
    with pytest.raises(ValueError, match='explicit unready admission'):
        runner.calibrated_codec_prerequisites(tmp_path/'candidate.pt', tmp_path/'cache.json', config)


def archived_calibration_fixture(tmp_path, monkeypatch):
    from tools.probe_slow_codec import freeze_encoder_for_decoder_probe, frozen_state_hash
    from tools.calibrate_frozen_decoder import file_hash
    _, cache_path, _, _, cache = fixture_artifacts(tmp_path)
    adjacency = torch.eye(2)
    codec = runner.SparseGraphCodec(adjacency=adjacency)
    freeze_encoder_for_decoder_probe(codec)
    frozen_hash = frozen_state_hash(codec)
    missing = tmp_path / 'removed_source.pt'
    checkpoint = dict(schema_version=1, status='candidate_unready', updates=20682,
        codec=codec.state_dict(), contract=dict(source_checkpoint=str(missing),
            source_checkpoint_sha256='recorded-source-hash', frozen_parameters_sha256=frozen_hash))
    checkpoint_path = tmp_path / 'calibration.pt'
    torch.save(checkpoint, checkpoint_path)
    binding = dict(schema_version=1, source_original_unavailable=True,
        calibration_checkpoint_sha256=file_hash(checkpoint_path),
        recorded_source_checkpoint_sha256='recorded-source-hash',
        frozen_parameters_sha256=frozen_hash, input_hashes=cache['contract']['input_hashes'],
        physical=cache['contract']['physical'], physics_identity={'traffic_model': 'micro'},
        original_demand_groups={'train-demand': 'train', 'validation-demand': 'validation'})
    binding['identity'] = _identity(binding)
    binding_path = tmp_path / 'calibration_binding.json'
    write_json(binding_path, binding)
    config = SimpleNamespace(codec=SimpleNamespace(decoder_allow_unready=True,
        decoder_pretrain_enabled=True, decoder_online_enabled=True, speed_positive=True), simulation={})
    prepared = SimpleNamespace(input_hashes=cache['contract']['input_hashes'])
    monkeypatch.setattr(runner, 'validate_prepared_inputs', lambda config: prepared)
    monkeypatch.setattr(runner, 'load_traffic_topology', lambda prepared:
        SimpleNamespace(network=SimpleNamespace(adjacency=adjacency.numpy())))
    return checkpoint_path, cache_path, binding_path, config, checkpoint, binding, cache


def test_archived_calibration_records_limited_proof_without_original_source(tmp_path, monkeypatch):
    from tools.calibrate_frozen_decoder import file_hash
    path, cache_path, binding_path, config, _, _, _ = archived_calibration_fixture(tmp_path, monkeypatch)
    accepted = runner.calibrated_codec_prerequisites(path, cache_path, config,
        binding_path=binding_path, expected_train=1, expected_validation=1)
    assert accepted['ready'] and not accepted['codec_ready']
    assert accepted['proof']['archived_calibration_binding'] is True
    assert accepted['proof']['source_checkpoint_revalidated'] is False
    assert accepted['proof']['calibration_binding_sha256'] == file_hash(binding_path)
    assert accepted['proof']['calibration_checkpoint_sha256'] == file_hash(path)
    # No implicit fallback: normal callers still require the original source.
    with pytest.raises(FileNotFoundError):
        runner.calibrated_codec_prerequisites(path, cache_path, config,
            expected_train=1, expected_validation=1)


@pytest.mark.parametrize('field,value,message', [
    ('calibration_checkpoint_sha256', 'changed', 'checkpoint hash'),
    ('recorded_source_checkpoint_sha256', 'changed', 'recorded_source_checkpoint'),
    ('frozen_parameters_sha256', 'changed', 'frozen_parameters'),
    ('input_hashes', {'net': 'changed'}, 'network inputs'),
    ('physical', {'changed': True}, 'physical contract'),
    ('physics_identity', {'traffic_model': 'meso'}, 'traffic physics'),
    ('original_demand_groups', {'train-demand': 'validation', 'validation-demand': 'train'}, 'partition changed'),
    ('source_original_unavailable', False, 'declaration'),
])
def test_archived_calibration_rejects_changed_binding_contract(tmp_path, monkeypatch, field, value, message):
    path, cache_path, binding_path, config, _, binding, _ = archived_calibration_fixture(tmp_path, monkeypatch)
    binding[field] = value
    binding.pop('identity')
    binding['identity'] = _identity(binding)
    write_json(binding_path, binding)
    with pytest.raises(ValueError, match=message):
        runner.calibrated_codec_prerequisites(path, cache_path, config,
            binding_path=binding_path, expected_train=1, expected_validation=1)


def test_archived_calibration_rejects_binding_edit_without_identity_update(tmp_path, monkeypatch):
    path, cache_path, binding_path, config, _, binding, _ = archived_calibration_fixture(tmp_path, monkeypatch)
    binding['physical'] = {'edited': True}
    write_json(binding_path, binding)
    with pytest.raises(ValueError, match='identity is invalid'):
        runner.calibrated_codec_prerequisites(path, cache_path, config,
            binding_path=binding_path, expected_train=1, expected_validation=1)


@pytest.mark.parametrize('change,expected', [('decoder', 'checkpoint hash'), ('encoder', 'frozen encoder'),
                                           ('nonfinite', 'finite'), ('shape', 'size mismatch')])
def test_archived_calibration_rejects_corrupted_codec(tmp_path, monkeypatch, change, expected):
    from tools.calibrate_frozen_decoder import file_hash
    path, cache_path, binding_path, config, checkpoint, binding, _ = archived_calibration_fixture(tmp_path, monkeypatch)
    state = checkpoint['codec']
    if change == 'decoder':
        state['decoder_speed.bias'].add_(1)
    elif change == 'encoder':
        state['input_proj.bias'].add_(1)
    elif change == 'nonfinite':
        state['decoder_speed.bias'].fill_(float('nan'))
    else:
        state['decoder_speed.bias'] = torch.zeros(2)
    torch.save(checkpoint, path)
    if change != 'decoder':
        # Rebinding a malformed payload cannot bypass tensor and frozen-state checks.
        binding['calibration_checkpoint_sha256'] = file_hash(path)
        binding.pop('identity')
        binding['identity'] = _identity(binding)
        write_json(binding_path, binding)
    with pytest.raises((ValueError, RuntimeError), match=expected):
        runner.calibrated_codec_prerequisites(path, cache_path, config,
            binding_path=binding_path, expected_train=1, expected_validation=1)
