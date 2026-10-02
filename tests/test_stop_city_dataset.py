from pathlib import Path
import signal

from tools import stop_city_dataset as stop


def test_stop_signals_only_exact_manifest_root_or_descendant_paths(monkeypatch):
    commands = {1: [b'python', b'/data/run-other/source/runner.py'],
                2: [b'python', b'/data/run/source/runner.py'],
                3: [b'python', b'--root', b'/data/run']}
    monkeypatch.setattr(stop, 'command', lambda pid: commands[pid])
    calls = []
    monkeypatch.setattr(stop.os, 'kill', lambda pid, sig: calls.append((pid, sig)))
    assert not stop.signal_owned(1, signal.SIGTERM, Path('/data/run'))
    assert stop.signal_owned(2, signal.SIGTERM, Path('/data/run'))
    assert stop.signal_owned(3, signal.SIGTERM, Path('/data/run'))
    assert [pid for pid, _ in calls] == [2, 3]


def test_disappearing_process_is_not_reported_as_signaled(monkeypatch):
    monkeypatch.setattr(stop, 'command', lambda pid: [b'--root', b'/data/run'])
    def disappeared(pid, sig):
        raise ProcessLookupError()
    monkeypatch.setattr(stop.os, 'kill', disappeared)
    assert not stop.signal_owned(4, signal.SIGTERM, Path('/data/run'))
