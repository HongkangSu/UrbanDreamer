"""Standalone compact archives preserve training data after raw-source retirement."""
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from test_offline_data import setup
from test_pretrain_cache import make_cache
from urban_dreamer.learning.offline_data import _identity
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset, LazyBoundaries, member_signature


def publish(path, manifest):
    manifest.pop('identity', None)
    manifest['identity'] = _identity(manifest)
    path.write_text(json.dumps(manifest))


def standalone(compact):
    path = compact.cache_manifest_path
    manifest = json.loads(path.read_text())
    manifest['source_storage'] = 'compact_only_v1'
    for row in manifest['entries'] + manifest['sources']:
        with zipfile.ZipFile(row['cache_path']) as archive:
            row['cache_member_signature'] = member_signature(archive.infolist())
    publish(path, manifest)
    return manifest


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, LazyBoundaries):
        assert tuple(left) == tuple(right)
    elif is_dataclass(left):
        for field in fields(left):
            assert_same(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_deleted_raw_preserves_every_batch_field_and_codec_stream(setup, tmp_path, monkeypatch):
    compact, entries = make_cache(setup, tmp_path, monkeypatch)
    expected = [compact.batch_from_starts(i, [0, 1]) for i in range(len(entries))]
    expected_codec = list(compact.iter_codec_batches(2))
    standalone(compact)
    for entry in entries:
        Path(entry['path']).unlink()
    restored = CompactPretrainDataset(*setup, 2, cache_manifest_path=compact.cache_manifest_path)
    for i, batch in enumerate(expected):
        assert_same(batch, restored.batch_from_starts(i, [0, 1]))
    assert_same(expected_codec, list(restored.iter_codec_batches(2)))
    restored.verify_integrity()


def test_legacy_still_requires_raw(setup, tmp_path, monkeypatch):
    compact, entries = make_cache(setup, tmp_path, monkeypatch)
    Path(entries[0]['path']).unlink()
    with pytest.raises(FileNotFoundError):
        CompactPretrainDataset(*setup, 2, cache_manifest_path=compact.cache_manifest_path)


@pytest.mark.parametrize('kind', ['missing', 'changed'])
def test_cache_retirement_does_not_disable_verification(setup, tmp_path, monkeypatch, kind):
    compact, entries = make_cache(setup, tmp_path, monkeypatch)
    standalone(compact)
    loaded = CompactPretrainDataset(*setup, 2, cache_manifest_path=compact.cache_manifest_path)
    loaded.batch_from_starts(0, [0])
    path = Path(entries[0]['cache_path'])
    if kind == 'missing':
        path.unlink()
        error = FileNotFoundError
    else:
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays['flow'][0, 0, 0] += 1
        np.savez_compressed(path, **arrays)
        error = ValueError
    with pytest.raises(error):
        loaded.batch_from_starts(0, [0])
    with pytest.raises(error):
        CompactPretrainDataset(*setup, 2, cache_manifest_path=compact.cache_manifest_path)


def test_unknown_storage_rejected(setup, tmp_path, monkeypatch):
    compact, _ = make_cache(setup, tmp_path, monkeypatch)
    manifest = standalone(compact)
    manifest['source_storage'] = 'skip_all_checks'
    publish(compact.cache_manifest_path, manifest)
    with pytest.raises(ValueError, match='source_storage'):
        CompactPretrainDataset(*setup, 2, cache_manifest_path=compact.cache_manifest_path)


def test_duplicates_bound_to_canonical_cache_without_raw(setup, tmp_path, monkeypatch):
    compact, entries = make_cache(setup, tmp_path, monkeypatch)
    manifest = standalone(compact)
    duplicate = dict(manifest['sources'][0], path=str(tmp_path / 'gone_duplicate.npz'))
    manifest['sources'].append(duplicate)
    manifest['duplicate_files'] += 1
    publish(compact.cache_manifest_path, manifest)
    for entry in entries:
        Path(entry['path']).unlink()
    CompactPretrainDataset(*setup, 2, cache_manifest_path=compact.cache_manifest_path)
    duplicate['cache_path'] = manifest['entries'][1]['cache_path']
    publish(compact.cache_manifest_path, manifest)
    with pytest.raises(ValueError, match='binding mismatch'):
        CompactPretrainDataset(*setup, 2, cache_manifest_path=compact.cache_manifest_path)


def test_source_provenance_missing_rejected(setup, tmp_path, monkeypatch):
    compact, _ = make_cache(setup, tmp_path, monkeypatch)
    manifest = standalone(compact)
    manifest['sources'].pop()
    publish(compact.cache_manifest_path, manifest)
    with pytest.raises(ValueError, match='missing matching source'):
        CompactPretrainDataset(*setup, 2, cache_manifest_path=compact.cache_manifest_path)


def test_updated_zip_signature_cannot_bypass_content_hash(setup, tmp_path, monkeypatch):
    compact, entries = make_cache(setup, tmp_path, monkeypatch)
    manifest = standalone(compact)
    path = Path(entries[0]['cache_path'])
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['flow'][0, 0, 0] += 1
    np.savez_compressed(path, **arrays)
    with zipfile.ZipFile(path) as archive:
        signature = member_signature(archive.infolist())
    manifest['entries'][0]['cache_member_signature'] = signature
    manifest['sources'][0]['cache_member_signature'] = signature
    publish(compact.cache_manifest_path, manifest)
    with pytest.raises(ValueError, match='content changed'):
        CompactPretrainDataset(*setup, 2, cache_manifest_path=compact.cache_manifest_path)


def test_changed_storage_requires_new_manifest_identity(setup, tmp_path, monkeypatch):
    compact, _ = make_cache(setup, tmp_path, monkeypatch)
    manifest = json.loads(compact.cache_manifest_path.read_text())
    manifest['source_storage'] = 'compact_only_v1'
    compact.cache_manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest identity'):
        CompactPretrainDataset(*setup, 2, cache_manifest_path=compact.cache_manifest_path)


def test_content_validation_reused_only_for_unchanged_cache_identity(setup, tmp_path, monkeypatch):
    from urban_dreamer.learning import pretrain_cache
    compact, entries = make_cache(setup, tmp_path, monkeypatch)
    standalone(compact)
    original = pretrain_cache.archive_content_hash
    calls = []
    def counting(path):
        calls.append(str(path))
        return original(path)
    monkeypatch.setattr(pretrain_cache, 'archive_content_hash', counting)
    first = CompactPretrainDataset(*setup, 2, cache_manifest_path=compact.cache_manifest_path)
    assert len(calls) == len(entries)
    first.batch_from_starts(0, [0])
    CompactPretrainDataset(*setup, 2, cache_manifest_path=compact.cache_manifest_path)
    assert len(calls) == len(entries)
    CompactPretrainDataset(*setup, 2, cache_manifest_path=compact.cache_manifest_path, verify_sources='always')
    assert len(calls) == 2 * len(entries)
