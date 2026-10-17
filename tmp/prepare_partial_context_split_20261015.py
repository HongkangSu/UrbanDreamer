"""Keep previous training/validation, reserve new test groups before fitting."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

root = Path.cwd()
parent = root / 'outputs/nonlinear_prior_fm_20261014/v1'
output = root / 'outputs/partial_context_fm_20261015/v1'
output.mkdir(parents=True, exist_ok=True)
assert not (output / 'split.json').exists()
sys.path.insert(0, str(parent / 'source'))
from urban_dreamer.learning.offline_data import _identity

old = json.loads((parent / 'split.json').read_text())
old_audit = json.loads((parent / 'data_split_audit.json').read_text())
full_path = root / 'outputs/ingolstadt_full_retrain_20261009/data/cache_manifest.json'
full = json.loads(full_path.read_text())
used = set()

def collect(value):
    if isinstance(value, dict):
        if isinstance(value.get('demand_group'), str):
            used.add(value['demand_group'])
        for item in value.values():
            collect(item)
    elif isinstance(value, list):
        for item in value:
            collect(item)

for item in old_audit['previous_studies_scanned']:
    path = Path(item['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    collect(json.loads(path.read_text()))
collect(old)
collect(json.loads((root/'outputs/direct_context_fm_20261014/v2_cpu4/split.json').read_text()))
groups = {}
for entry in full['entries']:
    if entry['split'] == 'validation' and entry['demand_group'] not in used:
        groups.setdefault(entry['demand_group'], []).append(entry)
ordered = sorted(groups, key=lambda key: hashlib.sha256(('partial_context_fresh_test_v1:' + key).encode()).hexdigest())
assert len(ordered) >= 16
test = []
for group in ordered[:16]:
    entries = sorted(groups[group], key=lambda e: hashlib.sha256(e['cache_path'].encode()).hexdigest())
    entry = dict(entries[0])
    with np.load(entry['cache_path'], allow_pickle=False) as arrays:
        frames = len(arrays['flow'])
    assert frames >= 18
    entry['starts'] = np.unique(np.linspace(0, frames - 6, 12).round().astype(int)).tolist()
    entry['evaluation_starts'] = sorted({int(round((frames - 6) * .25)), int(round((frames - 6) * .65))})
    test.append(entry)

train, validation = old['train'], old['validation']
selected = train + validation + test
selected_cache = {e['cache_path'] for e in selected}
selected_raw = {e['path'] for e in selected}
subset = {k: v for k, v in full.items() if k != 'identity'}
subset.update(entries=[e for e in full['entries'] if e['cache_path'] in selected_cache],
              sources=[e for e in full['sources'] if e['path'] in selected_raw],
              holdout_groups=sorted({e['demand_group'] for e in validation + test}),
              train_episodes=64, validation_episodes=32, duplicate_files=0,
              cache_bytes=sum(e['cache_bytes'] for e in selected),
              direct_context_selection_role='partial FM: same train/validation, fresh test beyond direct heads')
assert len(subset['entries']) == len(subset['sources']) == 96
subset['identity'] = _identity(subset)
(output / 'cache_manifest.json').write_text(json.dumps(subset, indent=2) + '\n')
indices = {e['cache_path']: i for i, e in enumerate([e for e in subset['entries'] if e['split'] == 'validation'])}
val_conditions = [dict(r, episode_index=indices[r['cache_path']]) for r in old['validation_conditions']]
conditions = []
for entry in test:
    for start in entry['evaluation_starts']:
        seed = int(hashlib.sha256(('partial_context_test:' + entry['demand_group'] + ':' + str(start)).encode()).hexdigest()[:8], 16)
        conditions.append(dict(condition_index=len(conditions), episode_index=indices[entry['cache_path']],
            source_path=entry['original_source_path'], cache_path=entry['cache_path'], cache_content_hash=entry['cache_content_hash'],
            episode=entry['episode'], demand_group=entry['demand_group'], start=start,
            condition_seed=seed, requested_horizon=1, traffic_phase='prespecified_temporal_quarter_or_65percent', role='test'))
assert len(conditions) == 32 and len(val_conditions) == 8
assert not used & {e['demand_group'] for e in test}
split = dict(old, test=test, test_conditions=conditions, validation_conditions=val_conditions,
             cache_manifest_path=str((output / 'cache_manifest.json').resolve()),
             status='prespecified_before_new_context_cache_or_scores',
             parent_feature_cache=str((parent / 'feature_manifest.json').resolve()),
             parent_feature_seed=20261021,
             final_test_policy='New groups beyond parent test; no scoring before validation selects direct heads')
(output / 'split.json').write_text(json.dumps(split, indent=2) + '\n')
audit = dict(full_manifest_sha256=hashlib.sha256(full_path.read_bytes()).hexdigest(),
             full_manifest_identity=full['identity'], parent_split_sha256=hashlib.sha256((parent / 'split.json').read_bytes()).hexdigest(),
             parent_features_sha256=hashlib.sha256((parent / 'feature_manifest.json').read_bytes()).hexdigest(),
             previously_used_groups=len(used), fresh_test_pool=len(groups), test_groups=16,
             parent_train_and_validation_unchanged=True, parent_test_not_reused=True,
             train_validation_test_disjoint=True, score_independent_selection=True,
             split_sha256=hashlib.sha256((output / 'split.json').read_bytes()).hexdigest(),
             subset_manifest_sha256=hashlib.sha256((output / 'cache_manifest.json').read_bytes()).hexdigest())
(output / 'data_split_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(dict(status='complete', fresh_pool=len(groups), test_groups=16, test_states=32, reused_train=64, reused_validation=16,
                     split_sha256=audit['split_sha256'])))
