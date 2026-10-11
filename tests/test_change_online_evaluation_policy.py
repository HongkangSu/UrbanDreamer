import copy
import pytest
from tools.change_online_evaluation_policy import revised_progress
from urban_dreamer.learning.training import evaluation_protocol


def test_rebase_preserves_training_and_separates_evaluation():
    old = dict(online=[{'episode_index': 0, 'metrics': {'loss': 3.}}],
               before=[{'temperature': 2.}], after=[{}], milestones=[{}],
               phase='online_training')
    saved = copy.deepcopy(old)
    new = revised_progress(old, 1, evaluation_protocol(None))
    assert old == saved
    assert new['online'] == old['online']
    assert new['before'] == new['after'] == new['milestones'] == []
    assert new['evaluation_reference_online_episodes'] == 1
    assert new['evaluation_policy'] == 'zero_prior_argmax'
    assert new['phase'] == 'before_evaluation'


def test_rebase_rejects_a_history_ahead_of_checkpoint():
    with pytest.raises(ValueError, match='history'):
        revised_progress({'online': [{}]}, 0, evaluation_protocol(None))
