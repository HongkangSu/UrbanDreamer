import json
import os
from pathlib import Path
import zipfile
import numpy as np
import pytest
from tools import repack_training_cache as tool
from urban_dreamer.learning.offline_data import archive_content_hash


def setup(tmp_path):
    target=tmp_path/'cache.npz'
    np.savez_compressed(target, flow=np.tile(np.arange(500,dtype=np.float32),(200,1)), metadata_json=np.frombuffer(b'{}',dtype=np.uint8))
    output=tmp_path/'receipts_root'
    (output/'receipts').mkdir(parents=True)
    (output/'locks').mkdir()
    row=dict(path=str(target), aliases=[str(target)], city='test', expected_hash=archive_content_hash(target))
    return target, output, row


def test_repack_hash_symlink_and_idempotence(tmp_path):
    target, output, row=setup(tmp_path)
    alias=tmp_path/'alias.npz'
    alias.symlink_to(target.name)
    row['aliases'].append(str(alias))
    before=target.stat().st_size
    result=tool.repack_one(row,output,min_free_bytes=0)
    assert result['status']=='compacted'
    assert result['after_bytes']<before
    assert alias.is_symlink() and alias.samefile(target)
    assert archive_content_hash(alias)==row['expected_hash']
    with np.load(alias,allow_pickle=False) as archive:
        assert archive['flow'].shape==(200,500)
    repeated=tool.repack_one(row,output,min_free_bytes=0)
    assert repeated['status']=='already_lzma'
    assert repeated['original_compaction']['saved_bytes']==result['saved_bytes']


def test_corruption_refuses_replacement(tmp_path):
    target, output, row=setup(tmp_path)
    old=target.read_bytes()
    row['expected_hash']='wrong'
    with pytest.raises(ValueError,match='hash mismatch'):
        tool.repack_one(row,output,min_free_bytes=0)
    assert target.read_bytes()==old


def test_member_verification_failure_preserves_source(tmp_path, monkeypatch):
    target, output, row=setup(tmp_path)
    old=target.read_bytes()
    def fail(*args):
        raise ValueError('simulated member mismatch')
    monkeypatch.setattr(tool,'_verify_members',fail)
    with pytest.raises(ValueError,match='member mismatch'):
        tool.repack_one(row,output,min_free_bytes=0)
    assert target.read_bytes()==old
    assert not list(tmp_path.glob('*.repack.tmp'))


def test_concurrent_source_change_refuses_replacement(tmp_path,monkeypatch):
    target, output, row=setup(tmp_path)
    verify=tool._verify_members
    def mutate(a,b):
        verify(a,b)
        st=a.stat()
        os.utime(a,ns=(st.st_atime_ns,st.st_mtime_ns+1_000_000))
    monkeypatch.setattr(tool,'_verify_members',mutate)
    old=target.read_bytes()
    with pytest.raises(RuntimeError,match='source changed'):
        tool.repack_one(row,output,min_free_bytes=0)
    assert target.read_bytes()==old


def test_hardlinks_skip_without_breaking_references(tmp_path):
    target, output, row=setup(tmp_path)
    alias=tmp_path/'hard.npz'
    os.link(target,alias)
    row['aliases'].append(str(alias))
    original=target.stat().st_ino
    result=tool.repack_one(row,output,min_free_bytes=0)
    assert result['status']=='skipped_hardlinks'
    assert target.stat().st_ino==alias.stat().st_ino==original


def test_free_space_gate_preserves_source(tmp_path):
    target, output, row=setup(tmp_path)
    original=target.read_bytes()
    with pytest.raises(RuntimeError,match='free space'):
        tool.repack_one(row,output,min_free_bytes=2**70)
    assert target.read_bytes()==original


def test_plan_deduplicates_target_inode_and_keeps_manifest(tmp_path):
    target, output, row=setup(tmp_path)
    alias=tmp_path/'alias.npz'
    alias.symlink_to(target.name)
    manifest=tmp_path/'manifest.json'
    data=dict(status='complete',contract=dict(city='test'),entries=[dict(cache_path=str(p),cache_content_hash=row['expected_hash']) for p in (target,alias)])
    manifest.write_text(json.dumps(data))
    old=manifest.read_bytes()
    plan=tool.manifest_plan([manifest])
    assert len(plan['entries'])==1
    assert len(plan['entries'][0]['aliases'])==2
    assert manifest.read_bytes()==old
