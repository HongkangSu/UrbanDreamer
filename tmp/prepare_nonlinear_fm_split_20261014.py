"""Choose fresh adapter-validation/test groups before any feature fit or score."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

root = Path.cwd()
source = root / 'outputs/prior_source_reference_20261014/v1/source'
sys.path.insert(0, str(source))
from urban_dreamer.learning.offline_data import _identity
from tools.analyze_traffic_spectrum import select_starts

output = root / 'outputs/nonlinear_prior_fm_20261014/v1'
output.mkdir(parents=True, exist_ok=True)
assert not (output / 'split.json').exists()
full_path = root / 'outputs/ingolstadt_full_retrain_20261009/data/cache_manifest.json'
full = json.loads(full_path.read_text())
old_path = root / 'outputs/pretrained_prior_rank_20261013/v1/selection.json'
training = json.loads(old_path.read_text())['train']
used = set()
audited = []

def collect(value):
    if isinstance(value, dict):
        if isinstance(value.get('demand_group'), str):
            used.add(value['demand_group'])
        for item in value.values():
            collect(item)
    elif isinstance(value, list):
        for item in value:
            collect(item)

for name in ('pretrained_evaluation_20261012', 'pretrained_prior_rank_20261013',
             'traffic_increment_spectrum_20261013', 'soft_background_20261014',
             'soft_prior_fm_20261014', 'hard_rank_fm_20261014',
             'fm_head_adaptation_20261014', 'prior_source_reference_20261014'):
    directory = root / 'outputs' / name
    for pattern in ('selection.json', 'conditions.json', 'data_audit.json', 'pairs.json'):
        for path in sorted(directory.rglob(pattern)):
            if 'source' in path.relative_to(directory).parts:
                continue
            collect(json.loads(path.read_text()))
            audited.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

groups = {}
for entry in full['entries']:
    if entry['split'] == 'validation' and entry['demand_group'] not in used:
        groups.setdefault(entry['demand_group'], []).append(entry)
ordered = sorted(groups, key=lambda key: hashlib.sha256(('nonlinear_fm_fresh_v1:' + key).encode()).hexdigest())
assert len(ordered) >= 32
validation, testing = [], []
for number, group in enumerate(ordered[:32]):
    entries = sorted(groups[group], key=lambda entry: hashlib.sha256(entry['cache_path'].encode()).hexdigest())
    entry = dict(entries[0])
    with np.load(entry['cache_path'], allow_pickle=False) as arrays:
        frames = len(arrays['flow'])
    assert frames >= 18
    entry['starts'] = np.unique(np.linspace(0, frames - 6, 12).round().astype(int)).tolist()
    entry['evaluation_starts'] = sorted({int(round((frames - 6) * .25)), int(round((frames - 6) * .65))})
    assert len(entry['starts']) == 12 and len(entry['evaluation_starts']) == 2
    (validation if number < 16 else testing).append(entry)

for entry in training:
    with np.load(entry['cache_path'], allow_pickle=False) as arrays:
        entry['starts'] = select_starts(arrays['flow'], 12).tolist()
    assert len(entry['starts']) == 12

train_groups = {e['demand_group'] for e in training}
val_groups = {e['demand_group'] for e in validation}
test_groups = {e['demand_group'] for e in testing}
assert len(train_groups) == 64 and len(val_groups) == len(test_groups) == 16
assert not (train_groups & val_groups or train_groups & test_groups or val_groups & test_groups)
assert not (used & (val_groups | test_groups))
selected = training + validation + testing
selected_paths = {e['cache_path'] for e in selected}
selected_raw = {e['path'] for e in selected}
subset = {k: v for k, v in full.items() if k != 'identity'}
subset.update(entries=[e for e in full['entries'] if e['cache_path'] in selected_paths],
              sources=[s for s in full['sources'] if s['path'] in selected_raw],
              holdout_groups=sorted(val_groups | test_groups),
              train_episodes=64, validation_episodes=32, duplicate_files=0,
              cache_bytes=sum(e['cache_bytes'] for e in selected),
              nonlinear_adapter_selection_role='fresh validation and final test, training reused from prior fit')
assert len(subset['entries']) == 96 and len(subset['sources']) == 96
subset['identity'] = _identity(subset)
(output / 'cache_manifest.json').write_text(json.dumps(subset, indent=2) + '\n')
split = dict(status='prespecified_before_features_or_scores', train=training,
             validation=validation, test=testing,
             rollout_validation=validation[:4],
             feature_times='four independent stratified uniform times, one in each quarter, paired across arms',
             counts=dict(train_groups=64, train_windows=768, validation_groups=16,
                         validation_feature_windows=192, rollout_validation_groups=4,
                         rollout_validation_states=8, test_groups=16, test_states=32),
             target='training-aligned HYBRID-Ct; full current state and full factual controls',
             final_test_policy='No test feature cache or predictions until validation selects final checkpoints')
(output / 'split.json').write_text(json.dumps(split, indent=2) + '\n')
audit = dict(full_manifest=str(full_path), full_manifest_sha256=hashlib.sha256(full_path.read_bytes()).hexdigest(),
             full_manifest_identity=full['identity'], previous_training_selection=str(old_path),
             previous_selection_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
             previous_studies_scanned=audited, previously_used_groups=len(used),
             eligible_fresh_validation_groups=len(groups), selection_score_independent=True,
             validation_and_test_not_previously_selected=True, splits_disjoint=True,
             subset_manifest_identity=subset['identity'],
             selection_sha256=hashlib.sha256((output / 'split.json').read_bytes()).hexdigest())
(output / 'data_split_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(dict(status='complete', previously_used_groups=len(used), fresh_pool=len(groups),
                     subset_sources=len(subset['sources']), counts=split['counts'])))
