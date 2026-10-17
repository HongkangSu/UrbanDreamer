import gzip
import io
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

import pytest
import torch

from test_offline_data import setup
from test_pretrain_cache import make_cache
from tools import export_portable_corpus as tool
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.learning.pretrain_cache import member_signature


def fixture_inventory(setup, tmp_path, monkeypatch):
    dataset, entries = make_cache(setup, tmp_path, monkeypatch)
    manifest_path = dataset.cache_manifest_path
    manifest = json.loads(manifest_path.read_text())
    manifest['source_storage'] = 'compact_only_v1'
    for category in ('entries', 'sources'):
        for row in manifest[category]:
            with zipfile.ZipFile(row['cache_path']) as archive:
                row['cache_member_signature'] = member_signature(archive.infolist())
            log = tmp_path / f"{row['trajectory_hash']}.json.gz"
            log.write_bytes(gzip.compress(b'{"complete":true}', mtime=0))
            row['summary_log_path'] = str(log)
    manifest.pop('identity')
    manifest['identity'] = tool.identity(manifest)
    manifest_path.write_text(json.dumps(manifest))
    city = dataset.config.inputs.network_name
    network = tmp_path / 'networks' / city
    network.mkdir(parents=True)
    (network / 'network.xml').write_text('<net/>')
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps({'cities': {city: {'manifest': str(manifest_path), 'network_dir': str(network)}}}))
    return inventory, dataset, manifest


def test_roundtrip_real_batch_without_sources(setup, tmp_path, monkeypatch):
    inventory, original, manifest = fixture_inventory(setup, tmp_path, monkeypatch)
    batch = original.batch_from_starts(0, [0])
    bundle = tmp_path / 'bundle'
    index = tool.export(inventory, bundle, shard_bytes=2000, reserve_bytes=0)
    assert index['status'] == 'complete'
    assert len(index['shards']) > 1
    city = original.config.inputs.network_name
    portable = json.loads(gzip.decompress((bundle / index['manifests'][city]['path']).read_bytes()))
    assert not Path(portable['entries'][0]['cache_path']).is_absolute()
    stamps = {row['path']: (bundle / row['path']).stat().st_mtime_ns for row in index['shards']}
    assert tool.export(inventory, bundle, shard_bytes=2000, reserve_bytes=0) == index
    assert stamps == {name: (bundle / name).stat().st_mtime_ns for name in stamps}
    for row in manifest['entries']:
        Path(row['path']).unlink()
        Path(row['cache_path']).unlink()
        Path(row['summary_log_path']).unlink()
    destination = tmp_path / 'restored'
    result = tool.restore(bundle, destination)
    restored_path = Path(result[city]['path'])
    restored_manifest = json.loads(restored_path.read_text())
    assert restored_manifest['entries'][0]['path'] == manifest['entries'][0]['path']
    assert restored_manifest['entries'][0]['source_stamp'] == manifest['entries'][0]['source_stamp']
    assert restored_manifest['identity'] != manifest['identity']
    restored = CompactPretrainDataset(original.config, original.prepared, original.topology, 2,
                                     cache_manifest_path=restored_path)
    actual = restored.batch_from_starts(0, [0])
    for name in ('flow', 'action', 'departure', 'context', 'time', 'terminal', 'total_vehicle_count'):
        torch.testing.assert_close(getattr(actual, name), getattr(batch, name))


def test_corrupt_shard_refused_before_extract(setup, tmp_path, monkeypatch):
    inventory, _, _ = fixture_inventory(setup, tmp_path, monkeypatch)
    bundle = tmp_path / 'bundle'
    index = tool.export(inventory, bundle, reserve_bytes=0)
    shard = bundle / index['shards'][0]['path']
    with shard.open('r+b') as stream:
        stream.write(b'bad')
    destination = tmp_path / 'restored'
    with pytest.raises(ValueError, match='checksum'):
        tool.restore(bundle, destination)
    assert not list(destination.iterdir())
    with pytest.raises(ValueError, match='checksum'):
        tool.export(inventory, bundle, reserve_bytes=0)


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'a/../../outside', 'a\\b', 'a//b', './a'])
def test_unsafe_paths_refused(name):
    with pytest.raises(ValueError, match='unsafe'):
        tool.safe_relative(name)


def test_malicious_archive_with_valid_checksum_refused(setup, tmp_path, monkeypatch):
    inventory, _, _ = fixture_inventory(setup, tmp_path, monkeypatch)
    bundle = tmp_path / 'bundle'
    index = tool.export(inventory, bundle, reserve_bytes=0)
    record = index['shards'][0]
    shard = bundle / record['path']
    name = '../outside'
    with tarfile.open(shard, 'w') as archive:
        info = tarfile.TarInfo(name)
        info.size = 4
        archive.addfile(info, io.BytesIO(b'evil'))
    record.update(bytes=shard.stat().st_size, sha256=tool.sha256(shard), members=[dict(path=name, bytes=4)])
    (bundle / 'index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError, match='unsafe'):
        tool.restore(bundle, tmp_path / 'restored')
    assert not (tmp_path / 'outside').exists()


def test_interrupted_shard_can_resume(setup, tmp_path, monkeypatch):
    inventory, _, _ = fixture_inventory(setup, tmp_path, monkeypatch)
    bundle = tmp_path / 'bundle'
    real_write = tool.write_shard
    calls = []
    def interrupted(output, name, records, reserve):
        calls.append(name)
        if len(calls) == 2:
            raise RuntimeError('interrupted')
        return real_write(output, name, records, reserve)
    monkeypatch.setattr(tool, 'write_shard', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        tool.export(inventory, bundle, shard_bytes=2000, reserve_bytes=0)
    partial = json.loads((bundle / 'index.json').read_text())
    assert partial['status'] == 'building'
    assert len(partial['shards']) == 1
    first = bundle / partial['shards'][0]['path']
    before = first.stat().st_mtime_ns
    monkeypatch.setattr(tool, 'write_shard', real_write)
    final = tool.export(inventory, bundle, shard_bytes=2000, reserve_bytes=0)
    assert final['status'] == 'complete'
    assert first.stat().st_mtime_ns == before


def test_space_reserve_prevents_writes(setup, tmp_path, monkeypatch):
    inventory, _, _ = fixture_inventory(setup, tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match='insufficient disk'):
        tool.export(inventory, tmp_path / 'bundle', reserve_bytes=10**30)
    assert not (tmp_path / 'bundle/index.json').exists()
