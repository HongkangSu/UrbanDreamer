from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from tools.recover_full_retrain_data import inventory, validate_receipt
from urban_dreamer.learning.offline_data import _identity, archive_content_hash
from urban_dreamer.learning.pretrain_cache import file_stamp, member_signature
import zipfile


def fixture(tmp_path):
    data = tmp_path/'data';data.mkdir();(data/'receipts').mkdir();(data/'episodes').mkdir()
    source = tmp_path/'original.npz';np.savez(source, value=np.array([1.,2.]))
    with zipfile.ZipFile(source) as z: signature=member_signature(z.infolist())
    row=dict(path=str(source), split='train', demand_group='d',episode=1,method='ac',root=str(tmp_path),
             crc_trajectory_candidate='crc',source_stamp=file_stamp(source))
    (data/'frozen_candidates.json').write_text(json.dumps({'paths':[str(source)],'frozen_at':'date'}))
    (data/'census.json').write_text(json.dumps({'rows':[row],'groups':{'d':'train'},'frozen_at':'date'}))
    cache=data/'episodes'/'trajectory.npz'
    metadata=dict(population=2,completed=True,city='ingo',schema_version=1)
    np.savez(cache, metadata_json=np.frombuffer(json.dumps(metadata).encode(),dtype=np.uint8),
        flow=np.zeros((2,3,4)),time=np.array([0.,30.]),event_vehicle=np.array([0]),
        vehicle_ids=np.array(['a','b']),boundary_offsets=np.array([0,2,4]))
    receipt=dict(**row,content_hash=archive_content_hash(source),trajectory_hash='trajectory',
                 source_member_signature=signature,census_trajectory_signature='crc',
                 population=2,completed=True,windows=1,events=1,cache_path=str(cache),
                 cache_content_hash=archive_content_hash(cache),cache_bytes=cache.stat().st_size)
    rp=data/'receipts'/f"{_identity({'path':str(source)})}.json";rp.write_text(json.dumps(receipt))
    return data,row,dict(row=row,receipt=str(rp),raw_available=True),cache


def test_inventory_preserves_selection_and_partition(tmp_path):
    data,row,item,cache=fixture(tmp_path)
    appended=tmp_path/'appended.npz';np.savez(appended,value=np.array([9]))
    census,audit,eligible=inventory(data)
    assert audit['eligible_files']==1
    assert [e['row']['path'] for e in eligible]==[row['path']]
    assert eligible[0]['row']['split']=='train'


def test_missing_original_explicitly_excluded(tmp_path):
    data,row,item,cache=fixture(tmp_path);Path(row['path']).unlink()
    census,audit,eligible=inventory(data)
    assert eligible==[] and audit['excluded_files']==1
    assert audit['exclusions'][0]['crc_trajectory_candidate']=='crc'
    assert audit['exclusions'][0]['source_stamp']==row['source_stamp']
    with pytest.raises(ValueError,match='disappeared'):
        validate_receipt(item)


def test_cache_content_mutation_rejected(tmp_path):
    data,row,item,cache=fixture(tmp_path)
    validate_receipt(item)
    np.savez(cache,value=np.array([42]))
    with pytest.raises(ValueError,match='identity mismatch'):
        validate_receipt(item)


def test_lossless_recompression_original_is_valid(tmp_path):
    data,row,item,cache=fixture(tmp_path);source=Path(row['path'])
    np.savez_compressed(source,value=np.array([1.,2.]))
    receipt,audit=validate_receipt(item)
    assert receipt['source_stamp']==file_stamp(source)
    assert len(audit['sha256'])==64


def test_original_source_mutation_rejected(tmp_path):
    data,row,item,cache=fixture(tmp_path)
    np.savez(Path(row['path']),value=np.array([1.,3.]))
    with pytest.raises(ValueError,match='source content changed'):
        validate_receipt(item)


def test_census_cannot_reassign_original_group(tmp_path):
    data,row,item,cache=fixture(tmp_path)
    c=json.loads((data/'census.json').read_text());c['rows'][0]['split']='validation'
    (data/'census.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='partition mismatch'):
        inventory(data)


def test_hardlink_snapshot_survives_original_removal_and_detects_changes(tmp_path):
    from tools.recover_full_retrain_data import seal_source_archives
    data,row,item,cache=fixture(tmp_path)
    receipt,_=validate_receipt(item)
    audit=dict(exclusions=[],receipt_identities=[]);audit['selection_identity']=_identity(audit)
    (data/'recovery_audit.json').write_text(json.dumps(audit))
    manifest=dict(sources=[receipt],entries=[receipt],recovery={},status='complete')
    manifest['identity']=_identity(manifest)
    (data/'cache_manifest.json').write_text(json.dumps(manifest))
    sealed=seal_source_archives(data)
    assert sealed['sources'][0]['original_source_path']==row['path']
    assert sealed['sources'][0]['path']!=row['path']
    pinned=Path(sealed['sources'][0]['path'])
    assert pinned.stat().st_ino==Path(row['path']).stat().st_ino
    Path(row['path']).unlink()
    assert pinned.is_file() and archive_content_hash(pinned)==receipt['content_hash']
    np.savez(pinned,value=np.array([99.]))
    assert archive_content_hash(pinned)!=receipt['content_hash']
    assert sealed['identity']==_identity({k:v for k,v in sealed.items() if k!='identity'})
    assert sealed['raw_source_snapshot']['additional_copied_bytes']==0


def test_revision_adds_frozen_sources_preserving_base_groups_and_duplicate_aliases(tmp_path,monkeypatch):
    import tools.recover_full_retrain_data as recovery
    data,row,item,cache=fixture(tmp_path)
    receipt,_=validate_receipt(item)
    audit=dict(exclusions=[],receipt_identities=[]);audit['selection_identity']=_identity(audit)
    (data/'recovery_audit.json').write_text(json.dumps(audit))
    manifest=dict(sources=[receipt],entries=[receipt],recovery={'excluded_unrecoverable_files':0},
                  status='complete',train_episodes=1,validation_episodes=0)
    manifest['identity']=_identity(manifest)
    (data/'cache_manifest.json').write_text(json.dumps(manifest))
    recovery.seal_source_archives(data)
    newsource=tmp_path/'new.npz';np.savez(newsource,value=np.array([3.,4.]))
    newrow={**row,'path':str(newsource),'crc_trajectory_candidate':'newcrc'}
    with zipfile.ZipFile(newsource) as z: signature=member_signature(z.infolist())
    newreceipt={**receipt,**newrow,'content_hash':archive_content_hash(newsource),
                'source_member_signature':signature,'source_stamp':file_stamp(newsource),
                'census_trajectory_signature':'newcrc'}
    rp=data/'receipts'/f"{_identity({'path':str(newsource)})}.json";rp.write_text(json.dumps(newreceipt))
    (data/'recovery_revision_frozen_candidates.json').write_text(json.dumps(dict(paths=[str(newsource)],frozen_at='newdate')))
    class ImmediatePool:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def map(self,func,items,**kwargs):return map(func,items)
    monkeypatch.setattr(recovery,'ProcessPoolExecutor',ImmediatePool)
    monkeypatch.setattr(recovery,'scan_revision_path',lambda path:dict(newrow))
    result=recovery.extend_revision(data,tmp_path/'unused.toml',workers=1)
    assert len(result['sources'])==2 and len(result['entries'])==1
    assert result['duplicate_files']==1
    assert result['expansion']['compatible_added_sources']==1
    assert result['expansion']['new_unique_trajectories']==0
    assert result['recovery']['raw_source_snapshot']=='hardlink'
    assert len({s['path'] for s in result['sources']})==2
    assert all(s['split']=='train' for s in result['sources'])
    assert json.loads((data/'base_recovery_manifest.json').read_text())['train_episodes']==1
