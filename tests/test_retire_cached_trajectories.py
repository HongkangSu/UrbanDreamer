import gzip
import json
import os
import fcntl
from pathlib import Path
import zipfile

import numpy as np
import pytest

from tools import retire_cached_trajectories as retire
from tools.prepare_city_pretrain import PHYSICAL
from urban_dreamer.learning.offline_data import _identity, archive_content_hash
from urban_dreamer.learning.pretrain_cache import member_signature


def write(path,data):path.write_text(json.dumps(data))


def live():return dict(roots=[],paths=[],inodes=[],consumers=[],manifests=[],errors=[])


def fixture(tmp_path,*,alias=None,raw_folder=None,role='raw_archive_candidate'):
    root=tmp_path/'outputs';root.mkdir();private=tmp_path/'private';private.mkdir()
    raw_parent=root/raw_folder if raw_folder else root
    raw_parent.mkdir(parents=True,exist_ok=True)
    raw=raw_parent/'episode.npz';cache=root/'cached.npz';corpus=root/'corpus';corpus.mkdir();(corpus/'receipts').mkdir()
    targets={'testcity':dict(input_hashes={'net':'abc'},edge_ids=['edge'],physics_identity={'traffic_model':'micro'},physical={k:None for k in PHYSICAL})}
    summary=dict(source='real_sumo',metadata=dict(city='testcity',phase='train',input_hashes={'net':'abc'},edge_ids=['edge'],session_protocol='event',action_semantics='branch-only',requested_step_seconds=1.,macro_step_seconds=30.,config={'simulation':{k:None for k in PHYSICAL}},demand_hash='demand',demand_law='independent_od_poisson'))
    np.savez(raw,summary_json=np.frombuffer(json.dumps(summary).encode(),dtype=np.uint8),flow=np.arange(4))
    np.savez(cache,metadata_json=np.frombuffer(b'{"city":"testcity"}',dtype=np.uint8),flow=np.arange(4))
    additional=None
    if alias:
        additional=private/'alias.npz'
        if alias=='symlink':additional.symlink_to(raw)
        if alias=='hardlink':os.link(raw,additional)
        if alias=='equivalent':
            alternate=json.loads(json.dumps(summary));alternate['metrics']={'reward':-9}
            np.savez(additional,summary_json=np.frombuffer(json.dumps(alternate).encode(),dtype=np.uint8),flow=np.arange(4))
    canonical=dict(path=str(raw),city='testcity',kind='raw',trajectory_hash=archive_content_hash(raw,trajectory_only=True),demand_group='demand',split='train',aliases=[str(additional)] if additional and alias!='equivalent' else [],equivalent_sources=[dict(path=str(additional),kind='raw',aliases=[])] if alias=='equivalent' else [])
    canonical_path=root/'canonical.jsonl';canonical_path.write_text(json.dumps(canonical)+'\n')
    contract=dict(targets['testcity'],city='testcity');contract.pop('edge_ids')
    plan=dict(city='testcity',limit=0,census_sha256=retire.sha(canonical_path),rows=[canonical],contract=contract);plan['identity']=_identity(plan);write(corpus/'plan.json',plan)
    log=retire.retain_summary(summary,corpus/'logs')
    with zipfile.ZipFile(raw) as z:raw_sig=member_signature(z.infolist())
    with zipfile.ZipFile(cache) as z:cache_sig=member_signature(z.infolist())
    entry=dict(path=str(raw),census_row=canonical,trajectory_hash=canonical['trajectory_hash'],content_hash=archive_content_hash(raw),source_stamp=retire.identity(raw.stat()),source_member_signature=raw_sig,cache_path=str(cache),cache_content_hash=archive_content_hash(cache),cache_member_signature=cache_sig,plan_identity=plan['identity'],demand_group='demand',split='train',**log)
    receipt=corpus/'receipts'/f"{_identity({'census_path':str(raw)})}.json";write(receipt,entry)
    manifest=dict(status='complete',source_storage='compact_only_v1',contract=contract,entries=[entry],sources=[entry],plan_identity=plan['identity']);manifest['identity']=_identity(manifest);mp=corpus/'cache_manifest.json';write(mp,manifest)
    proof={k:entry[k] for k in ('trajectory_hash','cache_path','cache_content_hash','cache_member_signature','summary_log_path','summary_json_sha256')}
    report=dict(status='passed',city='testcity',manifest_identity=manifest['identity'],manifest_sha256=retire.sha(mp),receipts=[dict(path=str(receipt),sha256=retire.sha(receipt),**proof)],**{k:True for k in retire.REPORT_FLAGS});rp=corpus/'verification.json';write(rp,report)
    inventory=[]
    for path in [raw,*([additional] if additional else [])]:
        st=path.stat();inventory.append(dict(path=str(path),dev=st.st_dev,ino=st.st_ino,size=st.st_size,mtime_ns=st.st_mtime_ns,train_eval_hint='train',role=role))
    ip=root/'inventory.jsonl';ip.write_text(''.join(json.dumps(x)+'\n' for x in inventory))
    tp=root/'targets.json';write(tp,targets);pp=root/'protection.json';write(pp,{'roots':[]});ps=root/'protected.jsonl';ps.write_text('')
    out=root/'retention';args=(mp,rp,canonical_path,ip,tp,pp,ps,out)
    return dict(args=args,raw=raw,cache=cache,output=out,roots=(root,private),alias=additional,report=rp,entry=entry)


def make_plan(f):return retire.create_plan(*f['args'],allowed_roots=f['roots'],live_scan=live)


def apply(f,scanner=live):return retire.apply_plan(f['output']/'deletion_plan.json',allowed_roots=f['roots'],live_scan=scanner)


def test_plan_only_then_authorized_apply_preserves_cache_and_logs(tmp_path):
    f=fixture(tmp_path);plan=make_plan(f)
    assert f['raw'].exists() and plan['rows'][0]['status']=='retire'
    old=f['cache'].read_bytes();result=apply(f)
    assert not f['raw'].exists() and result[0]['status']=='unlinked'
    assert f['cache'].read_bytes()==old
    assert gzip.decompress(Path(plan['rows'][0]['summary_log_path']).read_bytes())
    assert result[0]['last_hardlink_removed'] and result[0]['released_logical_bytes']>0
    records=retire.rows(f['output']/'deletion_ledger.jsonl')
    assert [x['status'] for x in records]==['unlink_intent','unlinked']


def test_wrong_cache_content_never_deletes(tmp_path):
    f=fixture(tmp_path);make_plan(f);np.savez(f['cache'],flow=np.arange(5))
    result=apply(f)
    assert f['raw'].exists() and result[0]['status']=='refused'
    assert 'cache_' in result[0]['reason']


def test_raw_changed_after_plan_never_deletes(tmp_path):
    f=fixture(tmp_path);make_plan(f);f['raw'].write_bytes(f['raw'].read_bytes()+b'changed')
    result=apply(f);assert f['raw'].exists() and result[0]['reason']=='raw_changed_since_plan'


def test_bad_trajectory_hash_refused(tmp_path,monkeypatch):
    f=fixture(tmp_path)
    original=retire.inspect_raw
    def invalid(handle,path,entry,targets):return original(handle,path,dict(entry,trajectory_hash='wrong'),targets)
    monkeypatch.setattr(retire,'inspect_raw',invalid)
    plan=make_plan(f)
    assert plan['rows'][0]['reason']=='raw_trajectory_hash_mismatch'
    assert apply(f)==[] and f['raw'].exists()


def test_active_path_protected(tmp_path):
    f=fixture(tmp_path);write(f['args'][5],{'roots':[str(f['raw'].parent)]})
    plan=make_plan(f);assert plan['rows'][0]['reason']=='active_or_protected_root'
    assert apply(f)==[] and f['raw'].exists()


def test_new_active_consumer_blocks_all_apply(tmp_path):
    f=fixture(tmp_path);make_plan(f)
    def changed():return dict(live(),roots=[str(f['raw'].parent)],consumers=['new trainer'])
    with pytest.raises(ValueError,match='active_dependency'):apply(f,changed)
    assert f['raw'].exists()


def test_outside_root_and_unknown_symlink_target_refused(tmp_path):
    f=fixture(tmp_path);outside=tmp_path/'outside.npz';outside.write_bytes(f['raw'].read_bytes())
    with pytest.raises(ValueError,match='outside_authorized_roots'):retire.safe_path(outside,f['roots'])
    link=f['raw'].parent/'bad.npz';link.symlink_to(outside)
    with pytest.raises(ValueError,match='outside_authorized_roots'):retire.safe_path(link,f['roots'])
    assert outside.exists()


@pytest.mark.parametrize('alias',['symlink','hardlink'])
def test_aliases_retired_without_double_counting(tmp_path,alias):
    f=fixture(tmp_path,alias=alias);plan=make_plan(f)
    assert all(x['status']=='retire' for x in plan['rows'])
    if alias=='symlink':assert plan['rows'][0]['is_symlink']
    result=apply(f)
    assert len(result)==2 and all(x['status']=='unlinked' for x in result)
    assert sum(x['last_hardlink_removed'] for x in result)==1
    assert not f['raw'].exists() and not f['alias'].exists()
    assert f['cache'].exists()


def test_missing_batch_verification_blocks_plan(tmp_path):
    f=fixture(tmp_path);report=retire.read(f['report']);report['real_batch_verified']=False;write(f['report'],report)
    with pytest.raises(ValueError,match='verification_missing'):make_plan(f)
    assert f['raw'].exists()


def test_retained_summary_changed_blocks_apply(tmp_path):
    f=fixture(tmp_path);plan=make_plan(f);Path(plan['rows'][0]['summary_log_path']).write_bytes(gzip.compress(b'{}'))
    result=apply(f);assert f['raw'].exists() and result[0]['status']=='refused'


def test_file_lock_contention_blocks_apply(tmp_path):
    f=fixture(tmp_path);make_plan(f)
    with f['raw'].open('rb') as locked:
        fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=apply(f)
    assert f['raw'].exists() and result[0]['status']=='refused'


def test_replacement_between_check_and_unlink_is_restored_not_deleted(tmp_path,monkeypatch):
    f=fixture(tmp_path);make_plan(f);old_rename=os.rename;replacement=f['raw'].parent/'replacement.npz'
    np.savez(replacement,flow=np.arange(100));replacement_bytes=replacement.read_bytes()
    def racing_rename(src,dst,**kwargs):
        if src==f['raw'].name and str(dst).startswith('.retirement-'):
            os.replace(replacement,f['raw'])
        return old_rename(src,dst,**kwargs)
    monkeypatch.setattr(os,'rename',racing_rename)
    result=apply(f)
    assert result[0]['status']=='refused'
    assert result[0]['reason']=='raw_replaced_at_unlink_boundary'
    assert f['raw'].read_bytes()==replacement_bytes


def test_evaluation_raw_explicitly_excluded(tmp_path):
    f=fixture(tmp_path);inventory=retire.rows(f['args'][3]);inventory[0]['train_eval_hint']='evaluation'
    f['args'][3].write_text(json.dumps(inventory[0])+'\n')
    plan=make_plan(f)
    assert plan['rows'][0]['reason']=='not_training_raw_inventory'
    assert f['raw'].exists()


def test_equivalent_trajectory_keeps_each_original_summary(tmp_path):
    f=fixture(tmp_path,alias='equivalent');plan=make_plan(f)
    assert all(x['status']=='retire' for x in plan['rows'])
    assert len({x['summary_json_sha256'] for x in plan['rows']})==2
    result=apply(f)
    assert all(x['status']=='unlinked' for x in result)
    summaries=[json.loads(gzip.decompress(Path(x['summary_log_path']).read_bytes())) for x in plan['rows']]
    assert any(x.get('metrics',{}).get('reward')==-9 for x in summaries)


def test_new_consumer_during_hash_is_rechecked_before_unlink(tmp_path):
    f=fixture(tmp_path);make_plan(f);calls=[]
    def scanner():
        calls.append(1)
        return live() if len(calls)==1 else dict(live(),consumers=['newly_started'])
    result=apply(f,scanner)
    assert f['raw'].exists() and result[0]['status']=='refused'
    assert 'active_dependency' in result[0]['reason']


def fast_for(manifest,*,consumers=()):
    return dict(roots=[],consumers=list(consumers),configs=[],manifests=[str(manifest)],
                dependency_stamps={str(manifest):retire.dependency_stamp(manifest)},errors=[])


def test_fast_guard_same_consumer_changed_manifest_refuses(tmp_path):
    f=fixture(tmp_path);dependency=f['raw'].parent/'active_manifest.json';write(dependency,{'sources':[]})
    probe=lambda:fast_for(dependency,consumers=['same pid and argv'])
    retire.create_plan(*f['args'],allowed_roots=f['roots'],live_scan=live,fast_probe=probe)
    write(dependency,{'sources':[],'changed':True})
    with pytest.raises(ValueError,match='manifest_changed'):
        retire.apply_plan(f['output']/'deletion_plan.json',allowed_roots=f['roots'],live_scan=live,fast_probe=probe)
    assert f['raw'].exists()


def test_fast_guard_new_consumer_refuses(tmp_path):
    f=fixture(tmp_path);dependency=f['raw'].parent/'active_manifest.json';write(dependency,{'sources':[]})
    retire.create_plan(*f['args'],allowed_roots=f['roots'],live_scan=live,fast_probe=lambda:fast_for(dependency))
    with pytest.raises(ValueError,match='active_dependency'):
        retire.apply_plan(f['output']/'deletion_plan.json',allowed_roots=f['roots'],live_scan=live,fast_probe=lambda:fast_for(dependency,consumers=['new']))
    assert f['raw'].exists()


def test_static_membership_and_full_snapshot_compiled_once(tmp_path,monkeypatch):
    f=fixture(tmp_path,alias='hardlink');dependency=f['raw'].parent/'active_manifest.json';write(dependency,{'sources':[]})
    probe=lambda:fast_for(dependency)
    retire.create_plan(*f['args'],allowed_roots=f['roots'],live_scan=live,fast_probe=probe)
    full_calls=[];compile_calls=[];protection_calls=[];set_ids=[]
    original_compile=retire.compile_protection;original_snapshot=retire.protection_snapshot;original_reason=retire.protected_reason
    def full():full_calls.append(1);return live()
    def compile_once(value):compile_calls.append(1);return original_compile(value)
    def snapshot_once(*args):protection_calls.append(1);return original_snapshot(*args)
    def reason(path,compiled,*args):
        assert isinstance(compiled['paths'],frozenset) and isinstance(compiled['inodes'],frozenset)
        set_ids.append((id(compiled['paths']),id(compiled['inodes'])))
        return original_reason(path,compiled,*args)
    monkeypatch.setattr(retire,'compile_protection',compile_once);monkeypatch.setattr(retire,'protection_snapshot',snapshot_once);monkeypatch.setattr(retire,'protected_reason',reason)
    result=retire.apply_plan(f['output']/'deletion_plan.json',allowed_roots=f['roots'],live_scan=full,fast_probe=probe)
    assert all(x['status']=='unlinked' for x in result)
    assert len(full_calls)==len(compile_calls)==len(protection_calls)==1
    assert len(set(set_ids))==1 and len(set_ids)==2


@pytest.mark.parametrize('folder,role,eligible',[
    ('source_archives','source_archive_fallback',True),
    ('source_archives','raw_archive_candidate',False),
    ('source_snapshot','source_archive_fallback',False),
    ('source_archives_backup','source_archive_fallback',False),
])
def test_source_archives_narrow_exception(tmp_path,folder,role,eligible):
    f=fixture(tmp_path,raw_folder=folder,role=role)
    plan=retire.create_plan(*f['args'],allowed_roots=f['roots'],live_scan=live,workers=4)
    assert (plan['rows'][0]['status']=='retire') is eligible
    result=retire.apply_plan(f['output']/'deletion_plan.json',allowed_roots=f['roots'],live_scan=live,workers=4)
    assert (not f['raw'].exists()) is eligible
    assert f['cache'].exists()
    if eligible:assert result[0]['status']=='unlinked'


def test_parallel_independent_inodes_and_atomic_ledger(tmp_path,monkeypatch):
    import threading
    f=fixture(tmp_path,alias='equivalent')
    retire.create_plan(*f['args'],allowed_roots=f['roots'],live_scan=live,workers=4)
    original=retire.inspect_raw;barrier=threading.Barrier(2);threads=set()
    def synchronized(*args):
        threads.add(threading.get_ident());barrier.wait(timeout=10)
        return original(*args)
    monkeypatch.setattr(retire,'inspect_raw',synchronized)
    result=retire.apply_plan(f['output']/'deletion_plan.json',allowed_roots=f['roots'],live_scan=live,workers=4)
    assert len(threads)==2 and len(result)==2
    assert all(x['status']=='unlinked' and x['last_hardlink_removed'] for x in result)
    records=retire.rows(f['output']/'deletion_ledger.jsonl')
    assert len(records)==4 and sum(x['status']=='unlinked' for x in records)==2


def test_parallel_same_inode_aliases_are_serial_and_not_double_counted(tmp_path,monkeypatch):
    import threading
    f=fixture(tmp_path,alias='hardlink');make_plan(f)
    original=retire.inspect_raw;threads=set()
    def observed(*args):threads.add(threading.get_ident());return original(*args)
    monkeypatch.setattr(retire,'inspect_raw',observed)
    result=retire.apply_plan(f['output']/'deletion_plan.json',allowed_roots=f['roots'],live_scan=live,workers=4)
    assert len(threads)==1 and len(result)==2
    assert all(x['status']=='unlinked' for x in result)
    assert sum(x['last_hardlink_removed'] for x in result)==1


def test_bounded_scheduler_stops_submitting_after_guard_failure():
    import threading
    stop=threading.Event();started=[]
    def guarded(index):
        started.append(index)
        try:retire.assert_fast_unchanged(dict(roots=[],consumers=['new'],configs=[],manifests=[],dependency_stamps={},errors=[]),dict(roots=[],consumers=[],configs=[],manifests=[],dependency_stamps={}))
        except ValueError:stop.set();raise
    with pytest.raises(ValueError,match='active_dependency'):
        list(retire.run_bounded(guarded,range(20),4,stop))
    assert len(started)<=4


def test_parallel_guard_failure_prevents_unlink(tmp_path):
    f=fixture(tmp_path,alias='equivalent');dependency=f['raw'].parent/'active_manifest.json';write(dependency,{'sources':[]})
    retire.create_plan(*f['args'],allowed_roots=f['roots'],live_scan=live,fast_probe=lambda:fast_for(dependency),workers=4)
    with pytest.raises(ValueError,match='active_dependency|guard_failure'):
        retire.apply_plan(f['output']/'deletion_plan.json',allowed_roots=f['roots'],live_scan=live,fast_probe=lambda:fast_for(dependency,consumers=['new']),workers=4)
    assert f['raw'].exists() and f['alias'].exists()


def test_bounded_pool_runs_distinct_jobs_with_maximum_four_workers():
    import threading
    barrier=threading.Barrier(4);lock=threading.Lock();active=0;peak=0
    def job(index):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        barrier.wait(timeout=10)
        with lock:active-=1
        return index
    assert sorted(retire.run_bounded(job,range(8),4))==list(range(8))
    assert peak==4


def fast_plan(f):
    return retire.create_plan(*f['args'],allowed_roots=f['roots'],live_scan=live,reuse_verified_proofs=True)


def fast_apply(f):
    return retire.apply_plan(f['output']/'deletion_plan.json',allowed_roots=f['roots'],live_scan=live,reuse_verified_proofs=True)


def test_fast_proofs_hash_cache_once_and_reuse_receipted_raw(tmp_path,monkeypatch):
    f=fixture(tmp_path,alias='symlink');calls=[];original=retire.archive_content_hash
    def counted(path,*args,**kwargs):calls.append(str(path));return original(path,*args,**kwargs)
    monkeypatch.setattr(retire,'archive_content_hash',counted)
    def forbidden(*args,**kwargs):raise AssertionError('unchanged receipt must not rehash raw')
    monkeypatch.setattr(retire,'inspect_raw',forbidden)
    plan=fast_plan(f)
    assert all(row['raw_receipt_reused'] and len(row['raw_proof_stamp'])==5 for row in plan['rows'])
    assert all(len(row['cache_proof']['stamp'])==5 for row in plan['rows'])
    assert all(row['status']=='unlinked' for row in fast_apply(f))
    assert calls==[str(f['cache'])]


@pytest.mark.parametrize('change',['inode','mtime','ctime'])
def test_fast_cache_version_change_requires_full_hash(tmp_path,monkeypatch,change):
    f=fixture(tmp_path);fast_plan(f)
    if change=='inode':
        replacement=f['cache'].with_suffix('.tmp');replacement.write_bytes(f['cache'].read_bytes());os.replace(replacement,f['cache'])
    elif change=='mtime':
        st=f['cache'].stat();os.utime(f['cache'],ns=(st.st_atime_ns,st.st_mtime_ns+1))
    else:os.chmod(f['cache'],0o600 if f['cache'].stat().st_mode&0o777!=0o600 else 0o644)
    calls=[];original=retire.archive_content_hash
    def counted(path,*args,**kwargs):calls.append(str(path));return original(path,*args,**kwargs)
    monkeypatch.setattr(retire,'archive_content_hash',counted)
    assert fast_apply(f)[0]['status']=='unlinked'
    assert calls==[str(f['cache'])]


def test_fast_cache_payload_corruption_with_restored_mtime_refused(tmp_path):
    f=fixture(tmp_path);fast_plan(f);st=f['cache'].stat()
    # Change stored payload without updating the ZIP directory; CRC catches it on fallback.
    payload=f['cache'].read_bytes();position=payload.index(np.arange(4).tobytes())
    changed=bytearray(payload);changed[position]^=1;f['cache'].write_bytes(changed)
    os.utime(f['cache'],ns=(st.st_atime_ns,st.st_mtime_ns))
    result=fast_apply(f)
    assert result[0]['status']=='refused' and f['raw'].exists()


def test_fast_raw_payload_corruption_with_restored_mtime_refused(tmp_path):
    f=fixture(tmp_path);fast_plan(f);st=f['raw'].stat()
    payload=f['raw'].read_bytes();position=payload.index(np.arange(4).tobytes())
    changed=bytearray(payload);changed[position]^=1;f['raw'].write_bytes(changed)
    os.utime(f['raw'],ns=(st.st_atime_ns,st.st_mtime_ns))
    assert fast_apply(f)[0]['status']=='refused' and f['raw'].exists()


def test_fast_equivalent_inode_uses_full_raw_verification(tmp_path,monkeypatch):
    f=fixture(tmp_path,alias='equivalent');calls=[];original=retire.inspect_raw
    def counted(handle,path,*args):calls.append(str(path));return original(handle,path,*args)
    monkeypatch.setattr(retire,'inspect_raw',counted)
    plan=fast_plan(f)
    assert calls==[str(f['alias'])]
    assert len({r['summary_json_sha256'] for r in plan['rows']})==2
    assert all(row['status']=='unlinked' for row in fast_apply(f))
    assert calls==[str(f['alias'])]


def test_fast_hardlink_ctime_after_first_unlink_falls_back(tmp_path,monkeypatch):
    f=fixture(tmp_path,alias='hardlink');fast_plan(f);calls=[];original=retire.inspect_raw
    def counted(handle,path,*args):calls.append(str(path));return original(handle,path,*args)
    monkeypatch.setattr(retire,'inspect_raw',counted)
    assert all(row['status']=='unlinked' for row in fast_apply(f))
    assert len(calls)==1


def test_fast_receipt_mismatch_falls_back_to_raw_hash(tmp_path,monkeypatch):
    f=fixture(tmp_path);calls=[];original=retire.inspect_raw
    def counted(handle,path,*args):calls.append(str(path));return original(handle,path,*args)
    monkeypatch.setattr(retire,'inspect_raw',counted)
    entry=dict(f['entry'],source_stamp=[0,0,0,0])
    with f['raw'].open('rb') as stream:
        (_,summary),reused=retire.inspect_receipted_raw(stream,f['raw'],entry,retire.read(f['args'][4]))
    assert not reused and calls==[str(f['raw'])] and summary['source']=='real_sumo'
