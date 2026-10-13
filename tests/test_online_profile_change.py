import copy
from dataclasses import replace
import json
import sys

import pytest
import torch

from tools import change_online_training_profile as profile
from tools.change_online_training_profile import changed_config, revised_protocol, rebase_evaluation, archive_prior_evaluations
from tools.migrate_mainline_prior import _assert_equal
from tools.run_complete_pretrain import export_run_config
from test_urban_dreamer_config import _write_config
from urban_dreamer.workflow.config import load_config, _semantic_fingerprint
from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint, save_checkpoint
from urban_dreamer.models.policy import shortest_path_prior_weight


def test_profile_changes_only_selected_settings_and_keeps_training_clock(tmp_path):
    config = load_config(_write_config(tmp_path/'config.toml'))
    target = changed_config(config, rollouts=8, evaluation_weight=0., hold=200, off=1000)
    assert target.run == config.run
    assert target.method.actor_entropy_mode == config.method.actor_entropy_mode
    assert target.method.sp_prior_cost_mode == config.method.sp_prior_cost_mode
    assert target.resources.imagination_passes_per_episode == config.resources.imagination_passes_per_episode
    assert shortest_path_prior_weight(99, initial_weight=1, hold_episodes=200, off_episode=1000) == 1.
    assert shortest_path_prior_weight(600, initial_weight=1, hold_episodes=200, off_episode=1000) == .5
    assert shortest_path_prior_weight(1000, initial_weight=1, hold_episodes=200, off_episode=1000) == 0.
    protocol = {'source':{'sha256':'source'},'prior':{'weight':1.,'evaluation_weight':1.,'hold_episodes':2000,'off_episode':2001}}
    before = copy.deepcopy(protocol)
    revised = revised_protocol(protocol,target)
    assert protocol == before
    assert revised['prior']['weight'] == 1.
    assert revised['prior']['evaluation_weight'] == 0.
    assert revised['budgets']['rollouts_per_pass'] == 8


def test_zero_prior_eval_rebase_preserves_online_history():
    old = dict(online=[{'episode_index':98}],before=[{'prior':1}],after=[],
               milestones=[{'online_episodes':80}],phase='online_training')
    before = copy.deepcopy(old)
    new = rebase_evaluation(old,boundary=99)
    assert old == before and new['online'] == old['online']
    assert new['before'] == new['after'] == new['milestones'] == []
    assert new['evaluation_reference_online_episodes'] == 99
    assert new['evaluation_policy'] == 'zero_prior_argmax'


def test_changed_eval_policy_archives_existing_episode_names(tmp_path):
    output=tmp_path/'run';backup=tmp_path/'backup';output.mkdir();backup.mkdir()
    for name in ['before_evaluation','milestones']:
        (output/name).mkdir();(output/name/'old.json').write_text('prior=1')
    (output/'episodes').mkdir();(output/'episodes'/'training.json').write_text('keep')
    moved=archive_prior_evaluations(output,backup)
    assert len(moved)==2 and not (output/'before_evaluation').exists()
    assert (backup/'before_evaluation'/'old.json').read_text()=='prior=1'
    assert (output/'episodes'/'training.json').read_text()=='keep'


def test_fp16_default_profile_preserves_every_other_setting(tmp_path):
    config = load_config(_write_config(tmp_path/'config.toml'))
    config = replace(config, resources=replace(config.resources,
        world_model_imagination_precision='fp32', imagination_rollouts_per_pass=16))
    target = changed_config(config)
    assert target.resources.world_model_imagination_precision == 'fp16'
    assert replace(target.resources, world_model_imagination_precision='fp32') == config.resources
    assert target.method == config.method


def test_precision_change_supports_protocol_without_prior_and_preserves_source_history(tmp_path):
    config = load_config(_write_config(tmp_path/'config.toml'))
    protocol = dict(config_fingerprint='old', runtime_config_fingerprint='old', source={'sha256': 'pretrained'},
        pretrain_handoff={'sha256': 'handoff'}, actor_entropy={'mode': 'annealed'}, global_reward='causal')
    before = copy.deepcopy(protocol)
    result = revised_protocol(protocol, changed_config(config))
    assert protocol == before
    assert 'prior' not in result
    assert result['runtime_config_fingerprint'] == config.semantic_fingerprint
    for field in ('source', 'pretrain_handoff', 'actor_entropy', 'global_reward'):
        assert result[field] == before[field]


def test_invalid_precision_profile_is_rejected(tmp_path):
    config = load_config(_write_config(tmp_path/'config.toml'))
    with pytest.raises(ValueError, match='imagination precision'):
        changed_config(config, precision='float16')


def test_fp16_in_place_migration_preserves_exact_checkpoint_and_evaluation_history(tmp_path, monkeypatch):
    output = tmp_path / 'run'
    output.mkdir()
    config = load_config(_write_config(tmp_path/'config.toml'))
    config = replace(config, run=replace(config.run, output_dir=output), resources=replace(
        config.resources, world_model_imagination_precision='fp32'))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    export_run_config(config, output / 'run_config.toml')
    progress = dict(online=[{'episode': 0}], before=[{'score': 1}], after=[],
                    milestones=[{'online_episodes': 1}], phase='online_training')
    state = dict(models={'world_model': {'weight': torch.arange(4.)}},
        reward_contract='causal_cumulative_mean_global_v1', next_environment_episode=1,
        optimizers={'actor': {'state': {0: {'exp_avg': torch.ones(4), 'step': torch.tensor(1.)}}}},
        rng={'torch': torch.get_rng_state()}, replay={'episodes': [1]},
        metrics={'online_pilot_progress': copy.deepcopy(progress)})
    save_checkpoint(output_dir=output, config_fingerprint=config.semantic_fingerprint, input_hash='input',
        phase=CheckpointPhase.ONLINE_TRAINING, next_unit=1, training_state=state)
    protocol = dict(config_fingerprint=config.semantic_fingerprint,
        prior=dict(weight=config.method.sp_prior_weight, evaluation_weight=config.method.sp_prior_evaluation_weight,
                   hold_episodes=config.method.sp_prior_hold_episodes, off_episode=config.method.sp_prior_off_episode),
        source={'sha256': 'pretrained'})
    profile.atomic_json(output/'pilot_protocol.json', protocol)
    profile.atomic_json(output/'pilot_progress.json', progress)
    profile.atomic_json(output/'runtime_status.json', dict(phase='stopped_at_checkpoint_boundary', pid=99999999))
    (output/'milestones').mkdir()
    (output/'milestones/history.json').write_text('retained evaluation')
    monkeypatch.setattr(sys, 'argv', ['change_online_training_profile.py', '--run-dir', str(output)])
    profile.main()
    target = load_config(output / 'run_config.toml')
    restored = load_checkpoint(output_dir=output, expected_config_fingerprint=target.semantic_fingerprint,
                               expected_input_hash='input')
    _assert_equal(state, restored.training_state)
    assert restored.next_unit == 1
    assert target.resources.world_model_imagination_precision == 'fp16'
    assert replace(target.resources, world_model_imagination_precision='fp32') == config.resources
    assert target.method == config.method
    assert json.loads((output/'pilot_progress.json').read_text()) == progress
    assert (output/'milestones/history.json').read_text() == 'retained evaluation'
    record = json.loads((output/'latest_profile_change.json').read_text())
    assert not record['evaluation_rebased']
    assert record['before']['imagination_precision'] == 'fp32'
    assert record['after']['imagination_precision'] == 'fp16'
