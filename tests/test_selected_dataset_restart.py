import json

import pytest

from tools.restart_selected_city_baselines import argument, committed_metrics, deployment_command, preserve_file


def test_retarget_only_script_and_explicit_device_keeps_resume_contract(tmp_path):
    original = ['python', '-u', '/old/tools/train_toy_baseline.py', '--config',
                '/old/configs/city.toml', '--output-dir', '/data/city/mbpo',
                '--device', 'cuda:0', '--episodes', '5000', '--sp-prior-weight', '1']
    actual = deployment_command({'command': original}, tmp_path, 'cuda:1')
    assert actual[2] == str(tmp_path / 'tools/train_toy_baseline.py')
    assert actual[actual.index('--config') + 1] == '/old/configs/city.toml'
    assert actual[actual.index('--device') + 1] == 'cuda:1'
    assert actual[actual.index('--episodes') + 1] == '5000'
    assert original[8] == 'cuda:0'
    assert argument([b'--run-id', b'city_mbpo'], '--run-id') == 'city_mbpo'
    assert argument([b'--run-id'], '--run-id') is None


def test_committed_metrics_drop_uncommitted_progress_and_partial_tail():
    text = '\n'.join(json.dumps(dict(phase=phase, episode=episode))
                     for phase, episode in [('train', 10), ('evaluation', 10), ('train', 11)])
    result = committed_metrics(text + '\n{"phase":', 10)
    assert [json.loads(line) for line in result.splitlines()] == [
        {'phase': 'train', 'episode': 10}, {'phase': 'evaluation', 'episode': 10}]
    with pytest.raises(json.JSONDecodeError):
        committed_metrics('{bad}\n' + text, 10)


def test_atomic_replacement_leaves_hardlinked_backup_unchanged(tmp_path):
    source, backup = tmp_path / 'episode.npz', tmp_path / 'backup/episode.npz'
    source.write_bytes(b'old episode')
    preserve_file(source, backup, link=True)
    replacement = tmp_path / 'replacement.npz'
    replacement.write_bytes(b'replayed episode')
    replacement.replace(source)
    assert backup.read_bytes() == b'old episode'
    assert source.read_bytes() == b'replayed episode'


def test_collection_device_override_leaves_learner_reward_and_prior_unchanged(tmp_path):
    original = ['python', '-u', '/old/tools/train_toy_baseline.py', '--device', 'cuda:0',
                '--episodes', '5000', '--global-reward-only', '--sp-prior-weight', '1']
    actual = deployment_command({'command': original}, tmp_path, collection_device='cpu')
    assert actual[actual.index('--device') + 1] == 'cuda:0'
    assert actual[-2:] == ['--collection-device', 'cpu']
    assert actual[3:-2] == original[3:]
    updated = deployment_command({'command': actual}, tmp_path, collection_device='cuda:0')
    assert updated.count('--collection-device') == 1 and updated[-1] == 'cuda:0'


def test_exited_zombie_has_no_running_command(monkeypatch):
    from types import SimpleNamespace
    from tools import stop_city_dataset as stop
    monkeypatch.setattr(stop, 'Path', lambda _: SimpleNamespace(read_bytes=lambda: b''))
    assert stop.command(123) == []
    monkeypatch.setattr(stop, 'Path', lambda _: SimpleNamespace(read_bytes=lambda: b'python\0script.py\0'))
    assert stop.command(123) == [b'python', b'script.py']
