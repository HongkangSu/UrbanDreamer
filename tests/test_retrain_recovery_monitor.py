import pytest

from tools.monitor_retrain_recovery import bind_expanded_groups
from urban_dreamer.learning.offline_data import _identity


def binding():
    value = dict(original_demand_groups={'old': 'train'})
    value['identity'] = _identity(value)
    return value


def test_expansion_preserves_existing_demand_assignments():
    value = binding()
    manifest = dict(sources=[dict(demand_group='old', split='train')],
        expansion={'revision': 'new'}, identity='frozen')
    result = bind_expanded_groups(value, manifest)
    assert result['original_demand_groups'] == value['original_demand_groups']
    assert result['authorized_expansion']['added_demand_groups'] == 0
    assert result['identity'] == _identity({k: v for k, v in result.items() if k != 'identity'})
    manifest['sources'][0]['split'] = 'validation'
    with pytest.raises(ValueError, match='preserved'):
        bind_expanded_groups(value, manifest)


def test_changed_initial_binding_is_rejected():
    value = binding()
    value['original_demand_groups']['old'] = 'validation'
    with pytest.raises(ValueError, match='binding changed'):
        bind_expanded_groups(value, {})
