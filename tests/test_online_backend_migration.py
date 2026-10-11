from dataclasses import replace
import copy
import json

import pytest
import torch

from tools import run_online_pilot as pilot
from tools.migrate_online_backend import backend_config, migrate, validate_backend_only_change
from tools.run_complete_pretrain import export_run_config
from urban_dreamer.workflow.checkpoint import load_checkpoint
from urban_dreamer.workflow.config import load_config
from test_online_pilot import FakeTrainer


def make_run(tmp_path):
    trainer = FakeTrainer(tmp_path)
    source = trainer.config.run.output_dir
    protocol = dict(source={'input_hash': 'inputs', 'config_fingerprint': trainer.config.semantic_fingerprint,
        'sha256': 'pretrain-sha', 'path': 'pretrain.pt'}, backend='traci', device='cpu')
    runner = pilot.OnlinePilot(trainer, output=source, input_hash='inputs', protocol=protocol,
        online_episodes=2000, evaluation_every=20)
    runner.resume()
    runner.evaluate('before')
    for index in range(2):
        trainer.train_online_episode(index)
        runner.progress['online'].append({'episode_index': index})
    runner.progress['milestones'] = [{'online_episodes': 1, 'phase': 'completed', 'evaluations': []}]
    runner.save()
    return trainer, source, runner


def test_backend_migration_preserves_training_and_references_history(tmp_path):
    trainer, source, old = make_run(tmp_path)
    source_hash = pilot.sha256_file(source / 'checkpoints/latest.pt')
    previous = copy.deepcopy(trainer.state_dict())
    target = tmp_path / 'libsumo'
    result = migrate(source, target, source_engine_version='1.15', target_engine_version='1.27.1')
    config = load_config(target / 'run_config.toml')
    checkpoint = load_checkpoint(output_dir=target, expected_config_fingerprint=config.semantic_fingerprint,
        expected_input_hash='inputs')
    assert checkpoint.next_unit == 2
    assert checkpoint.training_state['next_environment_episode'] == 2
    assert config.semantic_fingerprint != trainer.config.semantic_fingerprint
    assert result['online_episode_budget'] == 2000
    assert result['source_checkpoint_sha256'] == source_hash
    assert pilot.sha256_file(source / 'checkpoints/latest.pt') == source_hash
    for name, values in previous['models'].items():
        for key, value in values.items():
            assert torch.equal(checkpoint.training_state['models'][name][key], value)
    for key in ('optimizers', 'rng', 'replay'):
        assert checkpoint.training_state[key] == previous[key]
    progress = checkpoint.training_state['metrics']['online_pilot_progress']
    assert progress['online'] == old.progress['online']
    assert progress['before'] == progress['after'] == progress['milestones'] == []
    assert progress['backend_history'][0]['before'] == old.progress['before']
    assert progress['backend_history'][0]['milestones'] == old.progress['milestones']
    assert progress['evaluation_reference_online_episodes'] == 2
    trainer.config = config
    protocol = json.loads((target / 'pilot_protocol.json').read_text())
    continued = pilot.OnlinePilot(trainer, output=target, input_hash='inputs', protocol=protocol,
        online_episodes=2000, evaluation_every=20)
    continued.resume()
    assert len(continued.progress['online']) == 2
    trainer.eval_calls.clear()
    continued.run(baseline_only=True)
    assert trainer.eval_calls == [0, 1, 2]
    assert trainer._next_environment_episode == 2
    assert trainer.train_calls == [0, 1]


def test_backend_migration_rejects_other_configuration_changes(tmp_path):
    trainer = FakeTrainer(tmp_path)
    target = backend_config(trainer.config, tmp_path / 'target')
    validate_backend_only_change(trainer.config, target)
    with pytest.raises(ValueError, match='only'):
        validate_backend_only_change(trainer.config,
            replace(target, method=replace(target.method, prior_noise_scale=.2)))


def test_backend_migration_refuses_reusing_outputs(tmp_path):
    trainer, source, _ = make_run(tmp_path)
    with pytest.raises(ValueError, match='new, absent'):
        migrate(source, source, source_engine_version='1', target_engine_version='2')


def test_backend_migration_rejects_inconsistent_episode_boundary(tmp_path):
    trainer, source, runner = make_run(tmp_path)
    trainer._next_environment_episode += 1
    runner.save()
    with pytest.raises(ValueError, match='boundary'):
        migrate(source, tmp_path / 'new', source_engine_version='1', target_engine_version='2')


def test_cross_engine_evaluation_cannot_be_paired(tmp_path):
    trainer, source, runner = make_run(tmp_path)
    before = runner.progress['before']
    after = copy.deepcopy(before)
    after[0]['engine_provenance'] = dict(backend='libsumo', version='1.27.1')
    with pytest.raises(ValueError, match='engine provenance'):
        pilot.compare_paired(before, after)
