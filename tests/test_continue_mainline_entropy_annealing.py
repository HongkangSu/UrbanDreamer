import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import signal
import sys
from types import SimpleNamespace

import pytest
import torch

from tools import continue_mainline_entropy_annealing as continuation
from tools.run_complete_pretrain import export_run_config
from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


def fixture_run(tmp_path, *, episodes=507, budget=2000, phase=CheckpointPhase.ONLINE_TRAINING):
    source = tmp_path / 'source'
    source.mkdir()
    config = load_config(Path(__file__).parents[1] / 'configs/ingolstadt_custom.toml')
    config = replace(config, run=replace(config.run, output_dir=source, online_training_episodes=budget),
                     method=replace(config.method, actor_entropy_mode='population'))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    export_run_config(config, source / 'run_config.toml')
    protocol = dict(source={'sha256': 'pretrained'}, runtime_config_fingerprint=config.semantic_fingerprint,
                    online_episode_budget=budget, evaluation_every=20, gpu_memory_fraction=.25, threads=4)
    (source / 'pilot_protocol.json').write_text(json.dumps(protocol))
    progress = dict(online=[{'episode_index': index} for index in range(episodes)], before=[], after=[],
                    milestones=[], phase='online_training')
    state = dict(reward_contract='event_q_counterfactual_global_individual_timeout_v2',
        models={'actor': {'weight': torch.arange(6.).reshape(2, 3)}},
        optimizers={'actor': {'state': {0: {'exp_avg': torch.ones(2, 3), 'step': torch.tensor(80)}}}},
        rng={'torch': torch.get_rng_state()}, replay={'online': {'episodes': [{'events': [1, 2, 3]}]}},
        metrics={'online_pilot_progress': progress, 'online_episodes_completed': episodes,
                 'actor_entropy_mode': 'population'}, next_environment_episode=episodes)
    save_checkpoint(output_dir=source, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=phase, next_unit=episodes, training_state=state)
    # Contradicting JSON must not override the durable checkpoint cursor.
    (source / 'pilot_progress.json').write_text(json.dumps({'online': []}))
    return source, config, state


def test_lossless_annealing_migration_uses_global_cursor_and_keeps_budget(tmp_path):
    source, original_config, state = fixture_run(tmp_path)
    old_hash = continuation.sha256_file(source / 'checkpoints/latest.pt')
    target = tmp_path / 'target'
    manifest = continuation.migrate(source, target)
    config = load_config(target / 'run_config.toml')
    restored = load_checkpoint(output_dir=target, expected_config_fingerprint=config.semantic_fingerprint,
                               expected_input_hash='input')
    continuation._assert_equal(restored.training_state, state)
    assert restored.next_unit == 507
    assert config.run.online_training_episodes == 2000
    assert config.method.actor_entropy_mode == 'annealed'
    assert config.method.actor_entropy_final_coefficient == .0001
    assert .0001 < manifest['next_entropy']['coefficient'] < .002
    assert config.semantic_fingerprint != original_config.semantic_fingerprint
    assert continuation.sha256_file(source / 'checkpoints/latest.pt') == old_hash
    assert len(json.loads((target / 'pilot_progress.json').read_text())['online']) == 507
    assert continuation.validate_continuation(target, config, 'input', 'pretrained') == manifest


@pytest.mark.parametrize('phase,episodes', [(CheckpointPhase.COMPLETED, 2000),
                                         (CheckpointPhase.ONLINE_TRAINING, 2000)])
def test_completed_budgets_never_restart(tmp_path, phase, episodes):
    source, _, _ = fixture_run(tmp_path, phase=phase, episodes=episodes)
    target = tmp_path / 'target'
    with pytest.raises(ValueError, match='must not restart'):
        continuation.migrate(source, target)
    assert not target.exists()


@pytest.mark.parametrize('section,field,value', [
    ('run', 'online_training_episodes', 3000), ('method', 'fm_steps', 99),
    ('method', 'sp_prior_weight', 1.), ('resources', 'imagination_passes_per_episode', 99)])
def test_only_entropy_changes_are_permitted(tmp_path, section, field, value):
    _, config, _ = fixture_run(tmp_path)
    target = continuation.annealed_config(config, tmp_path / 'target')
    target = replace(target, **{section: replace(getattr(target, section), **{field: value})})
    with pytest.raises(ValueError, match='only entropy'):
        continuation.validate_transfer(config, target)


def test_tampering_and_backwards_progress_are_rejected(tmp_path):
    source, _, _ = fixture_run(tmp_path)
    target = tmp_path / 'target'
    continuation.migrate(source, target)
    config = load_config(target / 'run_config.toml')
    with (target / 'source_checkpoint.pt').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='hash differs'):
        continuation.validate_continuation(target, config, 'input', 'pretrained')


def test_advanced_checkpoint_preserves_immutable_initial_state(tmp_path):
    source, _, _ = fixture_run(tmp_path)
    target = tmp_path / 'target'
    manifest = continuation.migrate(source, target)
    config = load_config(target / 'run_config.toml')
    current = load_checkpoint(output_dir=target, expected_config_fingerprint=config.semantic_fingerprint,
                              expected_input_hash='input')
    state = copy.deepcopy(current.training_state)
    state['metrics']['online_pilot_progress']['online'].append({'episode_index': 507})
    state['metrics']['online_episodes_completed'] = 508
    state['next_environment_episode'] = 508
    save_checkpoint(output_dir=target, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.ONLINE_TRAINING, next_unit=508, training_state=state)
    assert continuation.validate_continuation(target, config, 'input', 'pretrained') == manifest


@pytest.mark.parametrize('anneal_pretrained_online', [False, True])
def test_entropy_overlay_preserves_unrelated_frozen_code(tmp_path, anneal_pretrained_online):
    frozen = Path(__file__).parents[1] / 'outputs/ingolstadt_full_retrain_20261009/source_memory_safe_20261010'
    if not frozen.is_dir():
        pytest.skip('local production frozen revision unavailable')
    revision = tmp_path / 'revision'
    manifest = continuation.stage_source(frozen, revision, anneal_pretrained_online=anneal_pretrained_online)
    assert continuation.sha256_file(frozen / 'urban_dreamer/learning/training.py') == manifest['frozen_source_hashes']['urban_dreamer/learning/training.py']
    for name, digest in manifest['frozen_source_hashes'].items():
        if name not in manifest['changed_files'] and name != 'source_online_manifest.json':
            assert continuation.sha256_file(revision / name) == digest
    code = "from tools.queue_pretrained_online import verify_execution_source; verify_execution_source('.')"
    subprocess.run([sys.executable, '-c', code], cwd=revision, check=True, capture_output=True, text=True)
    frozen_config = Path(__file__).parents[1] / 'outputs/ingolstadt_full_retrain_20261009/pretrain_config.toml'
    code = f"from urban_dreamer.workflow.config import load_config; print(load_config({str(frozen_config)!r}).semantic_fingerprint)"
    result = subprocess.run([sys.executable, '-c', code], cwd=revision, check=True, capture_output=True, text=True)
    from urban_dreamer.workflow.config import legacy_semantic_fingerprint
    assert result.stdout.strip() == legacy_semantic_fingerprint(load_config(frozen_config))
    if anneal_pretrained_online:
        code = (f"from urban_dreamer.workflow.config import load_config; "
                f"from tools.queue_pretrained_online import target_config, validate_profile_transfer; "
                f"s=load_config({str(frozen_config)!r}); "
                "t=target_config(s, 'seed'); validate_profile_transfer(s,t); "
                "assert s.method.actor_entropy_mode == 'population'; "
                "assert t.method.actor_entropy_mode == 'annealed'; "
                "assert t.method.actor_entropy_final_coefficient == .0001; "
                "assert s.run.online_training_episodes == t.run.online_training_episodes == 2000")
        subprocess.run([sys.executable, '-c', code], cwd=revision, check=True, capture_output=True, text=True)


def bridge_fixture(tmp_path, monkeypatch):
    revision = tmp_path / 'revision'
    revision.mkdir()
    controller = dict(pid=10, identity='controller-birth', command=['controller'], state='S')
    pretrainer = dict(pid=11, identity='pretrainer-birth', command=['pretrainer'], state='R')
    plan = dict(run_dir=str(tmp_path), revision_root=str(revision),
                controller=controller, pretrainer=pretrainer, online_command=['/fake/python', 'queue'])
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    (tmp_path / 'status.json').write_text(json.dumps(dict(phase='pretraining', controller_pid=10, child_pid=11)))
    records = {10: controller.copy(), 11: pretrainer.copy()}
    signals = []
    def send_signal(pid, value):
        signals.append((pid, value))
        records[pid]['state'] = 'T' if value == signal.SIGSTOP else 'S'
    monkeypatch.setattr(continuation, '_validate_future_plan', lambda plan: None)
    monkeypatch.setattr(continuation, 'process_record', lambda pid: records.get(pid))
    monkeypatch.setattr(continuation.os, 'kill', send_signal)
    return path, plan, records, signals


def test_future_handoff_signals_controller_only_and_preserves_pretrainer(tmp_path, monkeypatch):
    path, plan, records, signals = bridge_fixture(tmp_path, monkeypatch)
    calls = []
    def launch(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(pid=12, poll=lambda: None)
    monkeypatch.setattr(continuation.subprocess, 'Popen', launch)
    record = continuation.apply_future(path)
    assert signals == [(10, signal.SIGSTOP)]
    assert records[11] == plan['pretrainer']
    assert record['same_pretrainer']['pid'] == 11
    assert record['bridge_pid'] == 12
    assert calls[0][calls[0].index('run-future') + 1:] == ['--plan-path', str(path)]
    with pytest.raises(ValueError, match='already applied'):
        continuation.apply_future(path)


def test_reused_controller_pid_is_rejected_before_any_signal(tmp_path, monkeypatch):
    path, _, records, signals = bridge_fixture(tmp_path, monkeypatch)
    records[10]['identity'] = 'another-controller'
    with pytest.raises(ValueError, match='identity'):
        continuation.apply_future(path)
    assert signals == []


def test_bridge_launch_failure_resumes_original_controller_without_worker_signal(tmp_path, monkeypatch):
    path, plan, records, signals = bridge_fixture(tmp_path, monkeypatch)
    def launch(*args, **kwargs):
        raise OSError('launch failure')
    monkeypatch.setattr(continuation.subprocess, 'Popen', launch)
    with pytest.raises(OSError, match='launch failure'):
        continuation.apply_future(path)
    assert signals == [(10, signal.SIGSTOP), (10, signal.SIGCONT)]
    assert records[11] == plan['pretrainer']


def test_future_bridge_waits_for_pretrain_success_then_uses_revised_queue(tmp_path, monkeypatch):
    path, plan, records, signals = bridge_fixture(tmp_path, monkeypatch)
    records[10]['state'] = 'T'
    records[11]['state'] = 'Z'
    (tmp_path / 'world_model').mkdir()
    (tmp_path / 'world_model/pretrain_summary.json').write_text(json.dumps({'status': 'pretrained'}))
    (tmp_path / 'online_supervisor').mkdir()
    (tmp_path / 'online_supervisor/status.json').write_text(json.dumps({'phase': 'completed'}))
    commands = []
    def launch(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(pid=12, poll=lambda: 0, returncode=0)
    monkeypatch.setattr(continuation.subprocess, 'Popen', launch)
    continuation.run_future(path)
    assert commands == [plan['online_command']]
    assert signals == []
    assert json.loads((tmp_path / 'status.json').read_text())['phase'] == 'completed'
