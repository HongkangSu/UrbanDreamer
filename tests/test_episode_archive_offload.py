"""Archive storage relocation preserves every byte and committed local names."""
import hashlib
import io
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from tools import offload_episode_archives as offloader


@pytest.fixture(autouse=True)
def ample_storage(monkeypatch):
    monkeypatch.setattr(offloader.shutil, 'disk_usage', lambda path:
        SimpleNamespace(free=100 * 1024 ** 3))


def archive(path, *, compression=zipfile.ZIP_LZMA):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(flow=np.arange(96, dtype=np.float32).reshape(3, 4, 8),
        policy_decisions_json=np.frombuffer(b'[{"route":[1,2]}]', dtype=np.uint8))
    with zipfile.ZipFile(path, 'w', compression=compression) as output:
        for name, array in arrays.items():
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            output.writestr(name + '.npy', stream.getvalue())
    return arrays


def paths(tmp_path):
    return (tmp_path / 'local/episode_000001.npz', tmp_path / 'remote/episode_000001.npz')


def test_lossless_hash_arrays_symlink_and_idempotence(tmp_path):
    source, destination = paths(tmp_path)
    arrays = archive(source)
    raw = source.read_bytes()
    report = offloader.offload_npz(source, destination, 1)
    assert report['status'] == 'offloaded'
    assert report['sha256'] == report['remote_sha256'] == hashlib.sha256(raw).hexdigest()
    assert report['verified_bytes'] == report['local_bytes_reclaimed'] == len(raw)
    assert source.is_symlink() and source.resolve() == destination
    assert destination.read_bytes() == source.read_bytes() == raw
    with np.load(source, allow_pickle=False) as actual:
        for name, value in arrays.items():
            np.testing.assert_array_equal(actual[name], value)
    inode = source.lstat().st_ino
    assert offloader.offload_npz(source, destination, 1)['status'] == 'already_offloaded'
    assert source.lstat().st_ino == inode
    assert not list(destination.parent.glob('*.part'))


def test_uncommitted_episode_rejected_without_copy_or_link(tmp_path):
    source, destination = paths(tmp_path)
    archive(source)
    raw = source.read_bytes()
    with pytest.raises(ValueError, match='newer'):
        offloader.offload_npz(source, destination, 0)
    assert source.read_bytes() == raw and not source.is_symlink()
    assert not destination.exists()


@pytest.mark.parametrize('compression', [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_uncompressed_or_deflated_archive_stays_local(tmp_path, compression):
    source, destination = paths(tmp_path)
    archive(source, compression=compression)
    raw = source.read_bytes()
    assert offloader.offload_npz(source, destination, 1)['status'] == 'not_lzma'
    assert source.read_bytes() == raw and not source.is_symlink()
    assert not destination.exists()


def test_existing_identical_destination_reused_without_rewrite(tmp_path):
    source, destination = paths(tmp_path)
    archive(source)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())
    original = destination.stat()
    report = offloader.offload_npz(source, destination, 1)
    assert report['status'] == 'offloaded' and report['reused_destination']
    assert destination.stat().st_ino == original.st_ino
    assert destination.stat().st_mtime_ns == original.st_mtime_ns


def test_conflicting_remote_file_never_overwritten(tmp_path):
    source, destination = paths(tmp_path)
    archive(source)
    raw = source.read_bytes()
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b'other committed content')
    with pytest.raises(ValueError, match='overwrite refused'):
        offloader.offload_npz(source, destination, 1)
    assert source.read_bytes() == raw and not source.is_symlink()
    assert destination.read_bytes() == b'other committed content'


def test_remote_publication_race_never_overwrites(tmp_path, monkeypatch):
    source, destination = paths(tmp_path)
    archive(source)
    raw = source.read_bytes()
    original = offloader._rename_new

    def conflict(temporary, final):
        final.write_bytes(b'concurrent final')
        original(temporary, final)

    monkeypatch.setattr(offloader, '_rename_new', conflict)
    with pytest.raises(FileExistsError):
        offloader.offload_npz(source, destination, 1)
    assert destination.read_bytes() == b'concurrent final'
    assert source.read_bytes() == raw and not source.is_symlink()
    assert not list(destination.parent.glob('*.part'))


def test_readback_corruption_retains_local_archive(tmp_path, monkeypatch):
    source, destination = paths(tmp_path)
    archive(source)
    raw = source.read_bytes()
    original = offloader._readback

    def corrupt(path):
        digest, size, identity = original(path)
        return '0' * 64, size, identity

    monkeypatch.setattr(offloader, '_readback', corrupt)
    with pytest.raises(ValueError, match='SHA256 readback'):
        offloader.offload_npz(source, destination, 1)
    assert source.read_bytes() == raw and not source.is_symlink()
    assert not destination.exists() and not list(destination.parent.glob('*.part'))


def test_source_changed_immediately_before_link_is_preserved(tmp_path, monkeypatch):
    source, destination = paths(tmp_path)
    archive(source)
    raw = source.read_bytes()
    original = offloader._check_source

    def replace_before_link(path, stream, identity):
        if list(path.parent.glob(f'.{path.name}.*.link')):
            replacement = path.with_name('replacement.npz')
            replacement.write_bytes(b'new producer content')
            replacement.replace(path)
        original(path, stream, identity)

    monkeypatch.setattr(offloader, '_check_source', replace_before_link)
    with pytest.raises(RuntimeError, match='source changed'):
        offloader.offload_npz(source, destination, 1)
    assert source.read_bytes() == b'new producer content' and not source.is_symlink()
    assert destination.read_bytes() == raw
    assert not list(source.parent.glob('*.link'))


def test_low_storage_prevents_new_copy_and_releases_reservation(tmp_path, monkeypatch):
    source, destination = paths(tmp_path)
    archive(source)
    raw = source.read_bytes()
    monkeypatch.setattr(offloader.shutil, 'disk_usage', lambda path:
        SimpleNamespace(free=offloader.MIN_STORAGE_FREE_BYTES))
    with pytest.raises(offloader.StorageFullError):
        offloader.offload_npz(source, destination, 1)
    assert source.read_bytes() == raw and not source.is_symlink()
    assert not destination.exists()
    assert offloader._RESERVED_BYTES == 0


def run(root, city='toy', method='gnac', boundary=1):
    output = root / city / method
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output / 'latest.pt', 'w') as checkpoint:
        checkpoint.writestr('checkpoint/data.pkl', pickle.dumps(dict(episode=boundary)))
    return dict(id=city+'_'+method, city=city, method=method, output_dir=str(output))


def test_once_mirrors_known_paths_skips_references_and_continues_file_errors(tmp_path):
    root, storage = tmp_path / 'local', tmp_path / 'storage'
    entry = run(root)
    reference = run(root, method='shortest_ref')
    directory = Path(entry['output_dir']) / 'episodes/train'
    first, future = directory / 'episode_000001.npz', directory / 'episode_000002.npz'
    archive(first)
    archive(future)
    ref = Path(reference['output_dir']) / 'episodes/train/episode_000001.npz'
    archive(ref)
    (root / 'manifest.json').write_text(json.dumps(dict(runs=[entry, reference])))
    offloader.main(['--root', str(root), '--storage-root', str(storage), '--workers', '2', '--once'])
    assert first.is_symlink() and first.resolve() == storage / first.relative_to(root)
    assert not future.is_symlink() and not ref.is_symlink()
    status = json.loads((root / 'offload_status.json').read_text())
    assert status['counts']['offloaded'] == 1 and not status['errors']
    lines = [json.loads(line) for line in (root / 'offload_log.jsonl').read_text().splitlines()]
    assert lines[0]['status'] == 'offloaded'
    assert lines[0]['sha256'] == lines[0]['remote_sha256']


def test_watch_reports_file_error_and_continues_other_archives(tmp_path):
    root, storage = tmp_path / 'local', tmp_path / 'storage'
    entry = run(root, boundary=2)
    directory = Path(entry['output_dir']) / 'episodes/train'
    bad, good = directory / 'episode_000001.npz', directory / 'episode_000002.npz'
    archive(bad)
    archive(good)
    destination = storage / bad.relative_to(root)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b'conflicting remote')
    (root / 'manifest.json').write_text(json.dumps(dict(runs=[entry])))
    offloader.watch(root, storage, workers=2, once=True)
    assert not bad.is_symlink() and good.is_symlink()
    status = json.loads((root / 'offload_status.json').read_text())
    assert status['counts']['error'] == status['counts']['offloaded'] == 1
    assert str(bad) in status['errors']


def test_watch_reloads_manifest_and_checkpoint_between_passes(tmp_path, monkeypatch):
    root, storage = tmp_path / 'local', tmp_path / 'storage'
    first = run(root, boundary=1)
    second = run(root, method='ippo')
    original = Path(first['output_dir']) / 'episodes/train/episode_000001.npz'
    future = original.with_name('episode_000002.npz')
    added = Path(second['output_dir']) / 'episodes/train/episode_000001.npz'
    for path in (original, future, added):
        archive(path)
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps(dict(runs=[first])))
    passes = 0

    class StopAfterTwoPasses(Exception):
        pass

    def advance(interval):
        nonlocal passes
        passes += 1
        if passes == 2:
            raise StopAfterTwoPasses
        run(root, boundary=2)
        manifest.write_text(json.dumps(dict(runs=[first, second])))

    monkeypatch.setattr(offloader.time, 'sleep', advance)
    with pytest.raises(StopAfterTwoPasses):
        offloader.watch(root, storage, workers=2)
    assert all(path.is_symlink() for path in (original, future, added))


def test_manifest_output_outside_root_is_rejected(tmp_path):
    root, storage = tmp_path / 'local', tmp_path / 'storage'
    root.mkdir()
    outside = run(tmp_path / 'outside')
    source = Path(outside['output_dir']) / 'episodes/train/episode_000001.npz'
    archive(source)
    (root / 'manifest.json').write_text(json.dumps(dict(runs=[outside])))
    offloader.watch(root, storage, once=True)
    assert not source.is_symlink()
    assert json.loads((root / 'offload_status.json').read_text())['errors']
