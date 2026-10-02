from __future__ import annotations

from pathlib import Path
import json
import pickle
import zipfile

import numpy as np
import pytest

from tools import compact_episode_archives as compactor


@pytest.mark.parametrize('protocol', [2, 4, 5])
def test_reads_checkpoint_cursor_without_deserializing_models(tmp_path, protocol):
    with zipfile.ZipFile(tmp_path / 'latest.pt', 'w') as archive:
        archive.writestr('checkpoint/data.pkl', pickle.dumps(
            {'format': 'baseline', 'episode': 257, 'model': b'x' * 100000}, protocol=protocol))
    assert compactor.committed_episode({'output_dir': str(tmp_path), 'method': 'gnac'}) == 257


def test_watch_leaves_uncheckpointed_episode_untouched(tmp_path):
    output = tmp_path / 'random'
    directory = output / 'episodes' / 'train'
    directory.mkdir(parents=True)
    first, second = directory / 'episode_000001.npz', directory / 'episode_000002.npz'
    fixture_archive(first)
    fixture_archive(second)
    original_second = second.read_bytes()
    (output / 'progress.json').write_text('{"last_episode":1}')
    state = tmp_path / 'process.json'
    state.write_text('{"status":"failed"}')
    (tmp_path / 'manifest.json').write_text(json.dumps({'runs': [
        {'id': 'random', 'method': 'random', 'output_dir': str(output), 'state': str(state)}]}))
    compactor.watch(tmp_path, workers=1, interval=.01)
    assert second.read_bytes() == original_second
    with zipfile.ZipFile(first) as archive:
        assert all(info.compress_type == zipfile.ZIP_LZMA for info in archive.infolist())
    status = json.loads((tmp_path / 'compaction_status.json').read_text())
    assert status['counts']['compacted'] == 1 and not status['errors']


def fixture_archive(path):
    rng = np.random.default_rng(123)
    np.savez_compressed(path, flow=rng.normal(size=(3, 4, 8)).astype(np.float32),
                        states_json=np.frombuffer(b'{"repeated":true},' * 10000, dtype=np.uint8))


def members(path):
    with zipfile.ZipFile(path) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def test_lossless_compaction_and_idempotence(tmp_path):
    path = tmp_path / "episode_000001.npz"
    fixture_archive(path)
    before = members(path)
    report = compactor.repack_npz(path, committed_episode=1)
    assert report["status"] == "compacted"
    assert report["after_bytes"] < report["before_bytes"]
    assert members(path) == before
    with np.load(path, allow_pickle=False) as data:
        assert data["flow"].shape == (3, 4, 8)
        assert data["states_json"].dtype == np.uint8
    assert compactor.repack_npz(path, committed_episode=1)["status"] == "already_lzma"


def test_newer_episode_never_touched(tmp_path):
    path = tmp_path / "episode_000002.npz"
    fixture_archive(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="newer"):
        compactor.repack_npz(path, committed_episode=1)
    assert path.read_bytes() == before


def test_failed_verification_keeps_original_and_removes_temporary(tmp_path, monkeypatch):
    path = tmp_path / "episode_000001.npz"
    fixture_archive(path)
    before = path.read_bytes()
    def reject(*args):
        raise ValueError("injected corruption")
    monkeypatch.setattr(compactor, "_verify_members", reject)
    with pytest.raises(ValueError, match="corruption"):
        compactor.repack_npz(path, committed_episode=1)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_intervening_source_write_is_preserved(tmp_path, monkeypatch):
    path = tmp_path / "episode_000001.npz"
    fixture_archive(path)
    original_verify = compactor._verify_members
    def replace_source(source, destination):
        original_verify(source, destination)
        other = tmp_path / "replacement.npz"
        np.savez_compressed(other, new=np.array([123]))
        other.replace(source)
    monkeypatch.setattr(compactor, "_verify_members", replace_source)
    with pytest.raises(RuntimeError, match="source changed"):
        compactor.repack_npz(path, committed_episode=1)
    with np.load(path, allow_pickle=False) as data:
        assert data["new"].tolist() == [123]
    assert not list(tmp_path.glob("*.tmp"))
