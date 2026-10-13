import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from test_urban_dreamer_config import _write_config
from tools import change_online_training_profile as profile
from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import learning_budgets, validate_profile_continuation
from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


@pytest.fixture
def precision_continuation(tmp_path, monkeypatch):
    output = tmp_path / 'run'
    output.mkdir()
    old = load_config(_write_config(tmp_path / 'source.toml'))
    old = replace(old, run=replace(old.run, output_dir=output),
                  resources=replace(old.resources, world_model_imagination_precision='fp32'))
    old = replace(old, semantic_fingerprint=_semantic_fingerprint(**{
        name: getattr(old, name) for name in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    export_run_config(old, output / 'run_config.toml')
    progress = dict(online=[dict(episode_index=0)], before=[], after=[], milestones=[], phase='online_training')
    state = dict(reward_contract='causal_cumulative_mean_global_v1', models={'weight': torch.ones(2)},
                 optimizers={}, replay={}, rng={}, next_environment_episode=1,
                 metrics={'online_pilot_progress': copy.deepcopy(progress)})
    save_checkpoint(output_dir=output, config_fingerprint=old.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.ONLINE_TRAINING, next_unit=1, training_state=state)
    args = SimpleNamespace(online_episodes=2000, evaluation_every=20, threads=4,
                           gpu_memory_fraction=.25, device=old.run.device,
                           evaluation_policy='argmax', evaluation_temperature=.2)
    protocol = dict(source={'sha256': 'pretrained'}, runtime_config_fingerprint=old.semantic_fingerprint,
                    budgets=learning_budgets(old), online_episode_budget=args.online_episodes,
                    evaluation_every=args.evaluation_every, threads=args.threads,
                    gpu_memory_fraction=args.gpu_memory_fraction, device=args.device)
    profile.atomic_json(output / 'pilot_protocol.json', protocol)
    profile.atomic_json(output / 'pilot_progress.json', progress)
    profile.atomic_json(output / 'runtime_status.json', dict(phase='stopped_at_checkpoint_boundary', pid=99999999))
    monkeypatch.setattr(sys, 'argv', ['change_online_training_profile.py', '--run-dir', str(output)])
    profile.main()
    return output, load_config(output / 'run_config.toml'), args


def test_recorded_fp16_continuation_preserves_seed_and_accepts_matching_history(precision_continuation):
    output, config, args = precision_continuation
    protocol = validate_profile_continuation(output, config, 'input', 'pretrained', args)
    assert protocol == json.loads((output / 'pilot_protocol.json').read_text())
    assert protocol['source'] == {'sha256': 'pretrained'}
    assert protocol['runtime_config_fingerprint'] == config.semantic_fingerprint
    assert 'evaluation' not in protocol and 'evaluation_policy' not in protocol


def test_precision_continuation_rejects_unrecorded_legacy_evaluation_policy(precision_continuation):
    output, config, args = precision_continuation
    args.evaluation_policy = 'sampled'
    with pytest.raises(ValueError, match='frozen argmax evaluation policy'):
        validate_profile_continuation(output, config, 'input', 'pretrained', args)


def test_precision_continuation_rejects_another_pretraining_seed(precision_continuation):
    output, config, args = precision_continuation
    with pytest.raises(ValueError, match='seed ancestry'):
        validate_profile_continuation(output, config, 'input', 'another-seed', args)


def test_precision_continuation_rejects_unrecorded_runner_setting(precision_continuation):
    output, config, args = precision_continuation
    args.evaluation_every = 100
    with pytest.raises(ValueError, match='runner setting differs: evaluation_every'):
        validate_profile_continuation(output, config, 'input', 'pretrained', args)


def test_precision_continuation_rejects_tampered_source_checkpoint(precision_continuation):
    output, config, args = precision_continuation
    record = json.loads((output / 'latest_profile_change.json').read_text())
    with (Path(record['backup']) / 'checkpoint.pt').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='source ancestry'):
        validate_profile_continuation(output, config, 'input', 'pretrained', args)


def test_precision_continuation_rejects_history_cursor_mismatch(precision_continuation):
    output, config, args = precision_continuation
    checkpoint = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                                 expected_input_hash='input')
    save_checkpoint(output_dir=output, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.ONLINE_TRAINING, next_unit=2, training_state=checkpoint.training_state)
    with pytest.raises(ValueError, match='history cursor'):
        validate_profile_continuation(output, config, 'input', 'pretrained', args)


def test_precision_continuation_accepts_later_committed_episode(precision_continuation):
    output, config, args = precision_continuation
    checkpoint = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                                 expected_input_hash='input')
    checkpoint.training_state['metrics']['online_pilot_progress']['online'].append(dict(episode_index=1))
    save_checkpoint(output_dir=output, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.ONLINE_TRAINING, next_unit=2, training_state=checkpoint.training_state)
    assert validate_profile_continuation(output, config, 'input', 'pretrained', args)['source']['sha256'] == 'pretrained'
