import copy
from tools.change_online_training_profile import changed_config, revised_protocol, rebase_evaluation, archive_prior_evaluations
from test_urban_dreamer_config import _write_config
from urban_dreamer.workflow.config import load_config
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
