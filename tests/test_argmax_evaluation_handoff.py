from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import continue_city_evaluation_temperature as handoff


def test_argmax_command_removes_sampling_and_temperature(tmp_path):
    source = tmp_path / 'old'
    entry = dict(id='city_ippo', command=['python', '-u', str(source / 'tools/train.py'),
        '--config', str(source / 'configs/city.toml'), '--episodes', '5000',
        '--evaluation-sampled', '--evaluation-temperature', '.2', '--evaluation-argmax'])
    original = deepcopy(entry)
    result = handoff.continuation_entry(entry, tmp_path / 'new', tmp_path, None)
    assert result['command'] == ['python', '-u', str(tmp_path / 'new/tools/train.py'),
        '--config', str(tmp_path / 'new/configs/city.toml'), '--episodes', '5000', '--evaluation-argmax']
    assert result['evaluation_policy'] == 'deterministic'
    assert result['evaluation_temperature'] is None
    assert entry == original


def test_argmax_protocol_retains_training_and_metrics(tmp_path):
    original = dict(config={'training': 'untouched'}, args={'actor_lr': .0002},
                    shortest_path_prior={'best_selection': 'sampled_temperature_0.2'})
    handoff.queue.write_json(tmp_path / 'protocol.json', original)
    metrics = tmp_path / 'metrics.jsonl'
    metrics.write_text('{"policy":"sampled_temperature_0.2"}\n')
    before = metrics.read_bytes()
    handoff.update_protocol(tmp_path, 300, tmp_path / 'archive', [10042], None)
    actual = handoff.read(tmp_path / 'protocol.json')
    assert actual['evaluation_policy'] == 'argmax'
    assert actual['evaluation_policies'] == ['deterministic']
    assert actual['evaluation_temperature'] is None
    assert actual['args']['evaluation_argmax'] is True
    assert actual['args']['evaluation_sampling'] is False
    assert actual['args']['actor_lr'] == original['args']['actor_lr']
    assert actual['config'] == original['config']
    assert actual['shortest_path_prior']['best_selection'] == 'deterministic'
    assert metrics.read_bytes() == before


@pytest.mark.parametrize('label', ['deterministic', 'sampled_temperature_0.2'])
def test_repeated_protocol_switch_preserves_both_historical_bests(tmp_path, label):
    output = tmp_path / 'output'
    output.mkdir()
    previous = output / f'best_{label}.pt'
    previous.write_bytes(b'earlier best under same protocol')
    backup = tmp_path / 'backup.pt'
    backup.write_bytes(b'newer best under same protocol')
    archived = handoff.archive_previous_best(output, backup, label)
    assert archived != previous
    assert archived.read_bytes() == backup.read_bytes()
    assert previous.read_bytes() == b'earlier best under same protocol'
    assert handoff.archive_previous_best(output, backup, label) == archived


def test_argmax_prepare_preserves_models_and_terminal_jobs(tmp_path, monkeypatch):
    old_root, new_root, source = (tmp_path / name for name in ('old', 'new', 'frozen'))
    source.mkdir()
    (source / 'legacy_model.py').write_text('original critic architecture\n')
    overlay_root = tmp_path / 'overlay'
    overlays = ['tools/train_toy_actor_critic.py', 'tools/train_toy_baseline.py',
                'urban_dreamer/learning/baselines/common.py']
    for relative in overlays:
        for root, content in ((source, 'old evaluation entrypoint\n'),
                              (overlay_root, 'argmax evaluation entrypoint\n')):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    old_root.mkdir()
    runs = [dict(id=status, city='city', command=['python', '-u', str(source / 'tools/train.py'),
            '--evaluation-temperature', '.2']) for status in ('running', 'failed', 'completed')]
    manifest = dict(source_snapshot=str(source), input_hashes_by_city={'city': {'input': 'same'}},
                    seed=42, eval_seeds=2, runs=runs)
    ledger = dict(jobs={entry['id']: dict(status=entry['id']) for entry in runs})
    handoff.queue.write_json(old_root / 'manifest.json', manifest)
    handoff.queue.write_json(old_root / 'queue_state.json', ledger)
    monkeypatch.setattr(handoff.queue, 'CITIES', ['city'])
    monkeypatch.setattr(handoff, 'controller_identity', lambda *args: {'pid': 99})
    monkeypatch.setattr(handoff, 'input_hashes', lambda *args: {'input': 'same'})
    monkeypatch.setattr(handoff, 'load_config', lambda *args: SimpleNamespace(semantic_fingerprint='same'))
    monkeypatch.setattr(handoff, 'source_hashes', lambda path: {
        str(file.relative_to(path)): handoff.sha(file) for file in path.rglob('*.py')})
    handoff.prepare(SimpleNamespace(old_root=old_root, new_root=new_root, argmax=True,
                                    temperature=.2, overlay_root=overlay_root))
    actual = handoff.read(new_root / 'manifest.json')
    protocol = handoff.read(new_root / 'handoff_protocol.json')
    assert actual['evaluation_temperature'] is None
    assert actual['evaluation_policy'] == 'deterministic'
    assert actual['runs'][0]['command'][-1] == '--evaluation-argmax'
    assert actual['runs'][1:] == manifest['runs'][1:]
    assert (source / 'legacy_model.py').read_bytes() == (new_root / 'source/legacy_model.py').read_bytes()
    revision = protocol['source_revisions'][str(new_root / 'source')]
    assert revision['allowed_overlay_files'] == overlays
    changed = {relative for relative, digest in revision['new_hashes'].items()
               if digest != revision['original_hashes'][relative]}
    assert changed == set(overlays)
    for relative in overlays:
        assert (new_root / 'source' / relative).read_bytes() == (overlay_root / relative).read_bytes()
