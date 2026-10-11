from tools.component_experiment_data import select_manifest
from urban_dreamer.learning.offline_data import _identity
import pytest


def fixture():
    entries = [dict(path=f'{s}{g}{method}', split=s, demand_group=f'{s}{g}', windows=10)
               for s in ('train', 'validation') for g in range(5) for method in range(2)]
    obj = dict(status='complete', entries=entries,
               sources=[dict(path=e['path']) for e in entries], duplicate_files=0)
    obj['identity'] = _identity(obj)
    return obj


def test_group_selection_is_reproducible_and_preserves_paired_methods():
    source = fixture()
    subset = select_manifest(source, train_groups=2, validation_groups=1, seed=42)
    assert subset == select_manifest(source, train_groups=2, validation_groups=1, seed=42)
    assert len(subset['entries']) == 6
    assert len(subset['sources']) == 6
    assert subset['component_selection']['role'] == 'development_only'
    assert subset['identity'] == _identity({k: v for k, v in subset.items() if k != 'identity'})
    assert len(source['entries']) == 20


def test_reject_corrupt_manifest_and_overlap():
    source = fixture()
    source['entries'][0]['demand_group'] = 'validation0'
    with pytest.raises(ValueError, match='valid identity'):
        select_manifest(source, train_groups=2, validation_groups=1, seed=42)
    source['identity'] = _identity({k: v for k, v in source.items() if k != 'identity'})
    with pytest.raises(ValueError, match='overlap'):
        select_manifest(source, train_groups=2, validation_groups=1, seed=42)
