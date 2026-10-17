"""Plan raw-trajectory retirement; unlink only with explicit --apply and all proofs.

Never recursively deletes directories. Logs, checkpoints, frozen source, compact
arrays and evaluation archives are not retirement candidates. Plan-only is default.
"""
from __future__ import annotations
import argparse
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import tomllib
import zipfile
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from urban_dreamer.learning.offline_data import _identity, archive_content_hash
from urban_dreamer.learning.pretrain_cache import member_signature
from tools.census_pretrain_total import compatible_summary

ALLOWED_ROOTS=((ROOT/'outputs').resolve(),Path('/home/wangwenrui/private/UrbanDreamer').resolve())
REPORT_FLAGS=('all_entries_content_hash_verified','all_entries_member_signature_verified',
              'complete_summary_json_preserved','real_batch_verified','raw_unavailable_verified')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        tmp=Path(stream.name);stream.write((json.dumps(value,indent=2,sort_keys=True)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def identity(st):return [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns]


def within(path,roots):
    return any(path==root or root in path.parents for root in roots)


def safe_path(path,roots,*,inventory_role=None):
    p=Path(os.path.abspath(path));roots=tuple(Path(os.path.abspath(r)) for r in roots)
    if not within(p,roots) or not within(p.resolve(),roots):raise ValueError('outside_authorized_roots')
    if p.parent.resolve()!=p.parent:raise ValueError('symlink_ancestor_not_allowed')
    if p.suffix!='.npz':raise ValueError('not_raw_npz')
    for part in (*p.parts,*p.resolve().parts):
        if part=='source_archives' and inventory_role=='source_archive_fallback':continue
        if part in {'source','source_snapshot','checkpoints','logs','banks','frozen_inputs'} or part.startswith('source_'):
            raise ValueError('reserved_artifact_path')
    return p


def scan_consumers():
    """Cheap explicit consumer/config declarations; never read manifest sources."""
    roots=set();consumers=set();configs=set();manifests=set();errors=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if proc.stat().st_uid!=os.getuid():continue
            args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if not args or 'python' not in Path(args[0]).name or int(proc.name)==os.getpid():continue
            if not any('UrbanDreamer' in a or 'urban_dreamer' in a for a in args):continue
            if any('retire_cached_trajectories' in a for a in args):continue
            consumers.add(json.dumps(args))
            cwd=Path(os.readlink(proc/'cwd'))
            for flag in ('--output-dir','--output-root','--root','--storage-root','--run-dir','--source-dir','--source-root','--output'):
                if flag in args:
                    p=Path(args[args.index(flag)+1]);p=p if p.is_absolute() else cwd/p
                    roots.add(str(p.absolute()))
                    if (p/'run_config.toml').is_file():configs.add(str(p/'run_config.toml'))
            for flag in ('--config','--cache-manifest'):
                if flag in args:
                    p=Path(args[args.index(flag)+1]);p=p if p.is_absolute() else cwd/p
                    (configs if flag=='--config' else manifests).add(str(p))
        except (FileNotFoundError,ProcessLookupError):continue
        except (OSError,UnicodeError,IndexError) as exc:errors.append(f'{proc.name}:{exc}')
    return dict(roots=sorted(roots),consumers=sorted(consumers),configs=sorted(configs),
                manifests=sorted(manifests),errors=errors)


def dependency_stamp(path):
    st=Path(path).stat()
    return [*identity(st),st.st_ctime_ns]


def snapshot_fast(dependencies):
    probe=scan_consumers()
    probe['dependency_stamps']={}
    for path in dependencies:
        try:probe['dependency_stamps'][path]=dependency_stamp(path)
        except OSError as exc:probe['errors'].append(f'{path}:{exc}')
    return probe


def snapshot_live():
    """Resolve full dependencies once, binding config/manifest stat identities."""
    declarations=scan_consumers()
    roots=set(declarations['roots']);consumers=set(declarations['consumers'])
    configs=set(declarations['configs']);manifests=set(declarations['manifests'])
    errors=list(declarations['errors']);files=set();stamps={}
    for path in configs:
        try:
            before=dependency_stamp(path)
            data=tomllib.loads(Path(path).read_text()).get('data',{})
            if dependency_stamp(path)!=before:raise ValueError('config_changed_during_snapshot')
            stamps[path]=before
            manifests.update(str(v) for k,v in data.items() if k.endswith('cache_manifest') and v)
            roots.update(str(v) for k,v in data.items() if k.endswith('archive_roots') for v in (v if isinstance(v,list) else []))
        except (OSError,ValueError) as exc:errors.append(f'{path}:{exc}')
    inodes=set()
    for path in manifests:
        try:
            before=dependency_stamp(path)
            data=read(path)
            for row in data.get('sources',[]):
                source=Path(row['path']);files.add(str(source.absolute()));files.add(str(source.resolve()))
                if source.exists():st=source.stat();inodes.add((st.st_dev,st.st_ino))
            if dependency_stamp(path)!=before:raise ValueError('manifest_changed_during_snapshot')
            stamps[path]=before
        except (OSError,ValueError,KeyError) as exc:errors.append(f'{path}:{exc}')
    return dict(roots=sorted(roots),paths=sorted(files),inodes=[list(x) for x in sorted(inodes)],
                consumers=sorted(consumers),manifests=sorted(manifests),configs=sorted(configs),
                dependency_stamps=stamps,errors=errors)


def protection_snapshot(protection,protected_sources,live):
    if live.get('errors'):raise ValueError('live_dependency_scan_incomplete:'+str(live['errors']))
    previous=read(protection);source_rows=rows(protected_sources)
    return dict(roots=sorted(set(previous.get('roots',[]))|set(live['roots'])),
                paths=sorted(set(live['paths'])|{p for r in source_rows for p in (r['path'],r.get('resolved_path',r['path']))}),
                inodes=[list(x) for x in sorted({tuple(x) for x in live['inodes']}|{(r['dev'],r['ino']) for r in source_rows if 'dev' in r})])


def compile_protection(protection):
    return dict(roots=tuple(Path(r) for r in protection['roots']),
                paths=frozenset(protection['paths']),
                inodes=frozenset(tuple(x) for x in protection['inodes']))


def protected_reason(path,protection,cache_inodes=()):
    real=path.resolve();st=path.stat()
    if any(within(p,protection['roots']) for p in (path,real)):return 'active_or_protected_root'
    if str(path) in protection['paths'] or str(real) in protection['paths']:return 'active_raw_source_path'
    key=(st.st_dev,st.st_ino)
    if key in protection['inodes']:return 'active_raw_source_inode'
    if key in cache_inodes:return 'compact_or_retained_log_inode'
    return None


def verified_bundle(manifest_path,report_path,canonical_path):
    manifest=read(manifest_path);report=read(report_path)
    if manifest.get('status')!='complete' or manifest.get('source_storage')!='compact_only_v1':raise ValueError('standalone_manifest_incomplete')
    if manifest.get('identity')!=_identity({k:v for k,v in manifest.items() if k!='identity'}):raise ValueError('manifest_identity_mismatch')
    city=manifest['contract']['city']
    if report.get('status')!='passed' or report.get('city')!=city or report.get('manifest_identity')!=manifest['identity'] or report.get('manifest_sha256')!=sha(manifest_path):raise ValueError('verification_report_binding_mismatch')
    if not all(report.get(k) is True for k in REPORT_FLAGS):raise ValueError('real_batch_or_cache_verification_missing')
    plan=read(Path(manifest_path).parent/'plan.json')
    if plan.get('identity')!=manifest['plan_identity'] or plan.get('limit',0)!=0 or plan.get('census_sha256')!=sha(canonical_path):raise ValueError('builder_plan_or_census_mismatch')
    if plan['identity']!=_identity({k:v for k,v in plan.items() if k!='identity'}):raise ValueError('builder_plan_identity_mismatch')
    canonical={r['path']:r for r in rows(canonical_path)}
    receipts={str(Path(r['path']).absolute()):r for r in report['receipts']}
    provenance={r['path']:r for r in manifest['sources']}
    bundles=[]
    for entry in manifest['entries']:
        row=entry['census_row'];key=_identity({'census_path':row['path']})
        receipt=Path(manifest_path).parent/'receipts'/f'{key}.json'
        if canonical.get(row['path'])!=row or row.get('city')!=city:raise ValueError('census_entry_mismatch')
        if provenance.get(entry['path'])!=entry or read(receipt)!=entry or entry['plan_identity']!=manifest['plan_identity']:raise ValueError('receipt_or_provenance_mismatch')
        proof=receipts.get(str(receipt.absolute()))
        if proof is None or proof['sha256']!=sha(receipt):raise ValueError('receipt_hash_mismatch')
        for field in ('trajectory_hash','cache_path','cache_content_hash','cache_member_signature','summary_log_path','summary_json_sha256'):
            if proof.get(field)!=entry.get(field):raise ValueError('receipt_verification_field_mismatch:'+field)
        bundles.append((entry,row))
    return manifest,bundles


def verify_cache_and_log(entry):
    p=Path(entry['cache_path']);before=identity(p.stat())
    with zipfile.ZipFile(p) as archive:
        if member_signature(archive.infolist())!=entry['cache_member_signature']:raise ValueError('cache_signature_mismatch')
    if archive_content_hash(p)!=entry['cache_content_hash']:raise ValueError('cache_hash_mismatch')
    if identity(p.stat())!=before:raise ValueError('cache_changed_during_verification')
    if hashlib.sha256(gzip.decompress(Path(entry['summary_log_path']).read_bytes())).hexdigest()!=entry['summary_json_sha256']:raise ValueError('canonical_summary_log_mismatch')


def cache_proof_binding(entry):
    return {key:entry[key] for key in ('cache_path','cache_content_hash','cache_member_signature')}


def verify_cache_proof(entry,proof=None):
    """Reuse only a full hash tied to this exact file version, including ctime."""
    binding=cache_proof_binding(entry)
    before=dependency_stamp(entry['cache_path'])
    if not proof or any(proof.get(k)!=v for k,v in binding.items()) or proof.get('stamp')!=before:
        verify_cache_and_log(entry)
        if dependency_stamp(entry['cache_path'])!=before:raise ValueError('cache_changed_during_verification')
    else:
        payload=gzip.decompress(Path(entry['summary_log_path']).read_bytes())
        if hashlib.sha256(payload).hexdigest()!=entry['summary_json_sha256']:raise ValueError('canonical_summary_log_mismatch')
    return dict(binding,stamp=before)


def inspect_receipted_raw(handle,path,entry,targets):
    """Fast path for the original receipt inode only; unknown versions fall back.

    Historical receipts have a four-field stamp (no ctime). Directory signatures
    and the retained summary are rechecked; this does not claim to detect hostile
    payload edits that deliberately preserve both mtime and the ZIP directory.
    Newly recorded plan proofs include ctime and never reuse after such edits.
    """
    if identity(os.fstat(handle.fileno()))!=entry.get('source_stamp'):
        return inspect_raw(handle,path,entry,targets),False
    with zipfile.ZipFile(handle) as archive:
        members=archive.infolist();names=[m.filename for m in members]
        if len(names)!=len(set(names)) or 'summary_json.npy' not in names or any(not n.endswith('.npy') for n in names):
            raise ValueError('not_raw_archive_schema')
        signature=member_signature(members)
        if signature!=entry['source_member_signature']:
            handle.seek(0)
            return inspect_raw(handle,path,entry,targets),False
        a=np.load(io.BytesIO(archive.read('summary_json.npy')),allow_pickle=False)
        summary=json.loads(a.tobytes() if a.ndim else a.item())
    retained=gzip.decompress(Path(entry['summary_log_path']).read_bytes())
    if hashlib.sha256(retained).hexdigest()!=entry['summary_json_sha256'] or json.loads(retained)!=summary:
        raise ValueError('canonical_summary_log_mismatch')
    compatible=compatible_summary(summary,path,targets)
    if compatible['city']!=entry['census_row']['city'] or compatible['demand_group']!=entry['demand_group']:
        raise ValueError('raw_city_or_demand_mismatch')
    return (dict(content_hash=entry['content_hash'],trajectory_hash=entry['trajectory_hash'],source_member_signature=signature),summary),True


def inspect_raw(handle,path,entry,targets):
    content=hashlib.sha256();trajectory=hashlib.sha256();summary=None
    with zipfile.ZipFile(handle) as archive:
        members=sorted(archive.infolist(),key=lambda x:x.filename)
        names=[x.filename for x in members]
        if len(names)!=len(set(names)) or 'summary_json.npy' not in names or any(not n.endswith('.npy') for n in names):raise ValueError('not_raw_archive_schema')
        for member in members:
            digests=(content,) if member.filename=='summary_json.npy' else (content,trajectory)
            for digest in digests:digest.update(member.filename.encode()+b'\0')
            payload=bytearray() if member.filename=='summary_json.npy' else None
            with archive.open(member) as stream:
                while block:=stream.read(1024*1024):
                    for digest in digests:digest.update(block)
                    if payload is not None:payload.extend(block)
            for digest in digests:digest.update(b'\0')
            if payload is not None:
                a=np.load(io.BytesIO(payload),allow_pickle=False);summary=json.loads(a.tobytes() if a.ndim else a.item())
        signature=member_signature(members)
    if trajectory.hexdigest()!=entry['trajectory_hash']:raise ValueError('raw_trajectory_hash_mismatch')
    compatible=compatible_summary(summary,path,targets)
    if compatible['city']!=entry['census_row']['city'] or compatible['demand_group']!=entry['demand_group']:raise ValueError('raw_city_or_demand_mismatch')
    if str(path)==entry['path'] and content.hexdigest()!=entry['content_hash']:raise ValueError('canonical_raw_content_mismatch')
    return dict(content_hash=content.hexdigest(),trajectory_hash=trajectory.hexdigest(),source_member_signature=signature),summary


def retain_summary(summary,directory):
    payload=json.dumps(summary,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode();digest=hashlib.sha256(payload).hexdigest()
    directory.mkdir(parents=True,exist_ok=True);path=directory/f'{digest}.json.gz'
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=directory,delete=False) as stream:
            tmp=Path(stream.name);stream.write(gzip.compress(payload,mtime=0));stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    if gzip.decompress(path.read_bytes())!=payload:raise ValueError('raw_summary_retention_mismatch')
    return dict(summary_log_path=str(path.absolute()),summary_json_sha256=digest)


def candidate_paths(entry,row):
    paths={entry['path']}
    if row.get('kind')=='raw':paths.add(row['path']);paths.update(row.get('aliases',[]))
    for source in row.get('equivalent_sources',[]):
        if source['kind']=='raw':paths.add(source['path']);paths.update(source.get('aliases',[]))
    return sorted(paths)


def create_plan(manifest_path,report_path,canonical_path,inventory_path,targets_path,protection_path,protected_sources_path,output,*,allowed_roots=ALLOWED_ROOTS,live_scan=snapshot_live,fast_probe=None,workers=1,reuse_verified_proofs=False):
    validate_workers(workers)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    atomic_json(output/'planning_progress.json',dict(stage='loading_verified_manifest',workers=workers))
    manifest,bundles=verified_bundle(manifest_path,report_path,canonical_path)
    targets=read(targets_path);city=manifest['contract']['city']
    target=dict(targets[city],city=city);target.pop('edge_ids',None)
    if target!=manifest['contract']:raise ValueError('target_contract_mismatch')
    inventory={}
    for row in rows(inventory_path):
        for item in [row,*row.get('aliases',[])]:inventory[item['path']]=item
    live=live_scan();protection=protection_snapshot(protection_path,protected_sources_path,live)
    compiled=compile_protection(protection)
    if fast_probe is None and live_scan is snapshot_live:
        fast_probe=lambda:snapshot_fast(live['dependency_stamps'])
    guard=fast_probe() if fast_probe is not None else None
    if guard is not None:
        if guard.get('errors'):raise ValueError('fast_guard_incomplete')
        if 'dependency_stamps' in live and guard['dependency_stamps']!=live['dependency_stamps']:
            raise ValueError('dependencies_changed_after_full_snapshot')
    cache_inodes=set()
    for entry,_ in bundles:
        for k in ('cache_path','summary_log_path'):
            st=Path(entry[k]).stat();cache_inodes.add((st.st_dev,st.st_ino))
    def plan_bundle(bundle):
        entry,row=bundle
        results=[];checked=set();cache_proof=None;raw_proofs={}
        for raw in candidate_paths(entry,row):
            item=dict(path=raw,cache_path=entry['cache_path'],trajectory_hash=entry['trajectory_hash'],entry_path=entry['path'],status='skip')
            try:
                inv=inventory.get(str(Path(raw).absolute()))
                path=safe_path(raw,allowed_roots,inventory_role=(inv or {}).get('role'))
                if inv is None:raise ValueError('not_in_frozen_inventory')
                if inv.get('train_eval_hint')=='evaluation' or inv.get('role') not in {'raw_archive_candidate','source_archive_fallback'}:raise ValueError('not_training_raw_inventory')
                reason=protected_reason(path,compiled,cache_inodes)
                if reason:raise ValueError(reason)
                current=identity(path.stat())
                if current!=[inv[k] for k in ('dev','ino','size','mtime_ns')]:raise ValueError('changed_since_frozen_inventory')
                if reuse_verified_proofs:
                    cache_proof=verify_cache_proof(entry,cache_proof)
                elif entry['cache_path'] not in checked:verify_cache_and_log(entry);checked.add(entry['cache_path'])
                lstat=identity(path.lstat())
                with path.open('rb') as stream:
                    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if identity(os.fstat(stream.fileno()))!=current:raise ValueError('raw_open_identity_changed')
                    proof_stamp=dependency_stamp(path)
                    if reuse_verified_proofs:
                        known=raw_proofs.get(tuple(proof_stamp))
                        if known is None:
                            (hashes,summary),receipt_reused=inspect_receipted_raw(stream,path,entry,targets)
                            raw_proofs[tuple(proof_stamp)]=(hashes,summary,receipt_reused)
                        else:hashes,summary,receipt_reused=known
                        if dependency_stamp(path)!=proof_stamp:raise ValueError('raw_changed_during_plan')
                    else:hashes,summary=inspect_raw(stream,path,entry,targets)
                    if identity(os.fstat(stream.fileno()))!=current or identity(path.lstat())!=lstat:raise ValueError('raw_changed_during_plan')
                item.update(inventory_role=inv['role'],status='retire',source_stamp=current,lstat=lstat,is_symlink=path.is_symlink(),resolved_path=str(path.resolve()),**hashes,**retain_summary(summary,output/'raw_summary_logs'))
                if reuse_verified_proofs:item.update(cache_proof=cache_proof,raw_proof_stamp=proof_stamp,raw_receipt_reused=receipt_reused)
            except (ValueError,OSError,KeyError,zipfile.BadZipFile) as exc:item['reason']=str(exc)
            results.append(item)
        return results
    results=[]
    processed=eligible=skipped=0
    for result in run_bounded(plan_bundle,bundles,workers):
        results.extend(result);processed+=1
        eligible+=sum(row['status']=='retire' for row in result)
        skipped+=sum(row['status']=='skip' for row in result)
        if processed%50==0 or processed==len(bundles):
            atomic_json(output/'planning_progress.json',dict(stage='checking_raw_replacements',
                processed_trajectories=processed,total_trajectories=len(bundles),eligible_files=eligible,
                skipped_files=skipped,workers=workers,elapsed_seconds=time.monotonic()-started))
    # Remove symlinks first, then regular aliases; never follow unlink to a target.
    results.sort(key=lambda x:(not x.get('is_symlink',False),x['path']))
    inputs={k:dict(path=str(Path(v).absolute()),sha256=sha(v)) for k,v in dict(manifest=manifest_path,verification_report=report_path,canonical=canonical_path,inventory=inventory_path,targets=targets_path,protection=protection_path,protected_sources=protected_sources_path).items()}
    plan=dict(schema_version=1,status='planned',created_unix=time.time(),manifest_identity=manifest['identity'],city=city,inputs=inputs,allowed_roots=[str(Path(r).absolute()) for r in allowed_roots],live_snapshot=live,fast_guard=guard,protection=protection,rows=results)
    if reuse_verified_proofs:plan['reuse_verified_proofs']=True
    plan['identity']=_identity(plan);atomic_json(output/'deletion_plan.json',plan)
    atomic_json(output/'planning_progress.json',dict(stage='plan_complete',
        processed_trajectories=processed,total_trajectories=len(bundles),eligible_files=eligible,
        skipped_files=skipped,workers=workers,elapsed_seconds=time.monotonic()-started))
    return plan


_LEDGER_LOCK=threading.Lock()


def validate_workers(workers):
    if isinstance(workers,bool) or workers not in (1,2,3,4):raise ValueError('workers must be 1..4')


def run_bounded(function,items,workers,stop=None):
    validate_workers(workers)
    if workers==1:
        for item in items:
            if stop is not None and stop.is_set():break
            yield function(item)
        return
    iterator=iter(items);failure=None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=set()
        def submit():
            if stop is not None and stop.is_set():return False
            try:item=next(iterator)
            except StopIteration:return False
            pending.add(pool.submit(function,item));return True
        for _ in range(workers):
            if not submit():break
        while pending:
            completed,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in completed:
                try:yield future.result()
                except Exception as exc:
                    failure=failure or exc
                    if stop is not None:stop.set()
            if failure is None:
                for _ in completed:
                    if not submit():break
        if failure is not None:raise failure


def append_ledger(path,row):
    with _LEDGER_LOCK:
        with path.open('a') as stream:stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())


def assert_live_unchanged(live, baseline):
    if live.get('errors') or any(set(map(str,live[k]))-set(map(str,baseline[k])) for k in ('roots','paths','inodes','consumers','manifests')):
        raise ValueError('new_or_unreadable_active_dependency; regenerate plan')


def assert_fast_unchanged(probe,baseline):
    keys=('roots','consumers','configs','manifests')
    if probe.get('errors') or any(set(probe[k])-set(baseline[k]) for k in keys):
        raise ValueError('new_or_unreadable_active_dependency; regenerate plan')
    if probe.get('dependency_stamps')!=baseline.get('dependency_stamps'):
        raise ValueError('active_config_or_manifest_changed; regenerate plan')


def apply_plan(plan_path,*,allowed_roots=ALLOWED_ROOTS,live_scan=snapshot_live,fast_probe=None,workers=1,reuse_verified_proofs=False):
    validate_workers(workers)
    plan_path=Path(plan_path);plan=read(plan_path)
    if plan.get('identity')!=_identity({k:v for k,v in plan.items() if k!='identity'}):raise ValueError('deletion_plan_identity_mismatch')
    if plan['allowed_roots']!=[str(Path(p).absolute()) for p in allowed_roots]:raise ValueError('authorized_roots_changed')
    for item in plan['inputs'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('plan_input_changed')
    files={k:v['path'] for k,v in plan['inputs'].items()}
    manifest,bundles=verified_bundle(files['manifest'],files['verification_report'],files['canonical'])
    entries={e['path']:e for e,_ in bundles};targets=read(files['targets'])
    admitted={e['path']:set(candidate_paths(e,r)) for e,r in bundles}
    inventory={}
    for source in rows(files['inventory']):
        for item in [source,*source.get('aliases',[])]:inventory[item['path']]=item
    ledger=plan_path.parent/'deletion_ledger.jsonl';cache_inodes=set()
    for entry,_ in bundles:
        for key in ('cache_path','summary_log_path'):
            st=Path(entry[key]).stat();cache_inodes.add((st.st_dev,st.st_ino))
    lock=(plan_path.parent/'retirement.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    outcomes=[]
    initial_live=live_scan()
    try:
        assert_live_unchanged(initial_live,plan['live_snapshot'])
        protection=protection_snapshot(files['protection'],files['protected_sources'],initial_live)
        # Never drop protections just because a process exited since planning.
        protection['roots']=list(set(protection['roots'])|set(plan['protection']['roots']))
        protection['paths']=list(set(protection['paths'])|set(plan['protection']['paths']))
        protection['inodes']=list(set(map(tuple,protection['inodes']))|set(map(tuple,plan['protection']['inodes'])))
        compiled=compile_protection(protection)
        if fast_probe is None and live_scan is snapshot_live:
            if plan.get('fast_guard') is None:raise ValueError('plan_missing_fast_guard; regenerate plan')
            fast_probe=lambda:snapshot_fast(plan['fast_guard']['dependency_stamps'])
        first_fallback=[initial_live];guard_lock=threading.Lock();stop=threading.Event()
        def guard_check():
            if stop.is_set():raise ValueError('retirement_stopped_after_guard_failure')
            try:
                if fast_probe is not None:
                    if plan.get('fast_guard') is None:raise ValueError('plan_missing_fast_guard; regenerate plan')
                    assert_fast_unchanged(fast_probe(),plan['fast_guard'])
                else:
                    with guard_lock:
                        snapshot=first_fallback.pop() if first_fallback else live_scan()
                    assert_live_unchanged(snapshot,plan['live_snapshot'])
            except Exception:
                stop.set()
                raise
        def retire_row(row):
            # Conservative stop if a new consumer or dependency appeared.
            try:guard_check()
            except ValueError as exc:
                append_ledger(ledger,dict(status='aborted',unix=time.time(),plan_identity=plan['identity'],reason=str(exc)))
                raise
            event=dict(path=row['path'],unix=time.time(),plan_identity=plan['identity'],status='refused')
            try:
                inv=inventory.get(row['path'])
                if inv is None:raise ValueError('not_in_frozen_inventory')
                path=safe_path(row['path'],allowed_roots,inventory_role=inv.get('role'))
                if inv.get('train_eval_hint')=='evaluation' or inv.get('role') not in {'raw_archive_candidate','source_archive_fallback'}:raise ValueError('not_training_raw_inventory')
                if row['path'] not in admitted.get(row['entry_path'],set()):raise ValueError('raw_not_in_census_candidates')
                if not path.exists():raise ValueError('raw_missing_or_already_retired')
                reason=protected_reason(path,compiled,cache_inodes)
                if reason:raise ValueError(reason)
                if identity(path.stat())!=row['source_stamp'] or identity(path.lstat())!=row['lstat'] or str(path.resolve())!=row['resolved_path']:raise ValueError('raw_changed_since_plan')
                entry=entries[row['entry_path']]
                reuse=bool(reuse_verified_proofs and plan.get('reuse_verified_proofs'))
                cache_proof=verify_cache_proof(entry,row.get('cache_proof')) if reuse else None
                if not reuse:verify_cache_and_log(entry)
                retained=gzip.decompress(Path(row['summary_log_path']).read_bytes())
                if hashlib.sha256(retained).hexdigest()!=row['summary_json_sha256']:raise ValueError('retained_raw_log_changed')
                with path.open('rb') as stream:
                    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if identity(os.fstat(stream.fileno()))!=row['source_stamp']:raise ValueError('raw_open_identity_changed')
                    proof_stamp=dependency_stamp(path)
                    if reuse and proof_stamp==row.get('raw_proof_stamp'):
                        hashes={k:row[k] for k in ('content_hash','trajectory_hash','source_member_signature')}
                        summary=json.loads(retained)
                    else:hashes,summary=inspect_raw(stream,path,entry,targets)
                    if reuse and dependency_stamp(path)!=proof_stamp:raise ValueError('raw_changed_during_apply')
                    if any(hashes[k]!=row[k] for k in hashes):raise ValueError('raw_hash_changed_since_plan')
                    if json.loads(retained)!=summary:raise ValueError('raw_summary_changed')
                    # Hashing may take minutes; refresh dependencies at unlink time too.
                    guard_check()
                    if reuse:
                        cache_proof=verify_cache_proof(entry,cache_proof)
                        if dependency_stamp(path)!=proof_stamp:raise ValueError('raw_changed_before_unlink')
                    parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
                    try:
                        # Bind unlink to this directory/name; do not delete realpath.
                        if identity(os.stat(path.name,dir_fd=parent,follow_symlinks=False))!=row['lstat'] or identity(os.fstat(stream.fileno()))!=row['source_stamp']:raise ValueError('raw_changed_before_unlink')
                        quarantine=f'.retirement-{uuid.uuid4().hex}.npz'
                        event['quarantine_path']=str(path.parent/quarantine)
                        append_ledger(ledger,dict(event,status='unlink_intent',source_stamp=row['source_stamp'],cache_path=entry['cache_path'],summary_log_path=row['summary_log_path']))
                        # Atomic rename pins the directory entry before the final
                        # check; a replacement cannot be mistaken for inspected raw.
                        if stop.is_set():raise ValueError('retirement_stopped_after_guard_failure')
                        os.rename(path.name,quarantine,src_dir_fd=parent,dst_dir_fd=parent)
                        try:
                            if identity(os.stat(quarantine,dir_fd=parent,follow_symlinks=False))!=row['lstat'] or identity(os.fstat(stream.fileno()))!=row['source_stamp']:
                                raise ValueError('raw_replaced_at_unlink_boundary')
                            if stop.is_set():raise ValueError('retirement_stopped_after_guard_failure')
                            os.unlink(quarantine,dir_fd=parent)
                            remaining=os.fstat(stream.fileno())
                            released=(not row['is_symlink']) and remaining.st_nlink==0
                            event.update(status='unlinked',last_hardlink_removed=released,remaining_hardlinks=remaining.st_nlink,
                                         released_logical_bytes=remaining.st_size if released else 0,
                                         released_allocated_bytes=remaining.st_blocks*512 if released else 0)
                            os.fsync(parent)
                        except Exception:
                            # Restore without overwriting a newly created original
                            # path. If occupied, keep quarantine and report its path.
                            try:
                                os.link(quarantine,path.name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
                                os.unlink(quarantine,dir_fd=parent);os.fsync(parent)
                                event['restored_original_name']=True
                            except OSError:pass
                            raise
                    finally:os.close(parent)
            except (ValueError,OSError,KeyError,zipfile.BadZipFile) as exc:event['reason']=str(exc)
            append_ledger(ledger,event);return event
        groups={}
        for row in plan['rows']:
            if row['status']=='retire':groups.setdefault(tuple(row['source_stamp'][:2]),[]).append(row)
        def retire_group(group):
            result=[]
            for row in sorted(group,key=lambda x:(not x['is_symlink'],x['path'])):
                if stop.is_set():break
                result.append(retire_row(row))
            return result
        for result in run_bounded(retire_group,groups.values(),workers,stop):outcomes.extend(result)
        if stop.is_set():append_ledger(ledger,dict(status='aborted',unix=time.time(),plan_identity=plan['identity'],reason='guard_failure_stopped_further_group_submission'))
    finally:lock.close()
    return outcomes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path);p.add_argument('--verification-report',type=Path)
    p.add_argument('--canonical',type=Path);p.add_argument('--inventory',type=Path)
    p.add_argument('--targets',type=Path);p.add_argument('--protection',type=Path);p.add_argument('--protected-sources',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--plan',type=Path)
    p.add_argument('--workers',type=int,choices=(1,2,3,4),default=1)
    p.add_argument('--reuse-verified-proofs',action='store_true',help='Reuse receipt-bound raw proofs and unchanged full cache checks; defaults to full verification')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--plan-only',action='store_true');mode.add_argument('--apply',action='store_true')
    a=p.parse_args()
    if a.apply:
        if a.plan is None:p.error('--apply requires --plan')
        result=apply_plan(a.plan,workers=a.workers,reuse_verified_proofs=a.reuse_verified_proofs)
        print(json.dumps(dict(unlinked=sum(x['status']=='unlinked' for x in result),refused=sum(x['status']!='unlinked' for x in result),released_allocated_bytes=sum(x.get('released_allocated_bytes',0) for x in result))))
    else:
        names=('manifest','verification_report','canonical','inventory','targets','protection','protected_sources','output')
        if any(getattr(a,k) is None for k in names):p.error('plan-only requires all manifest/report/census/inventory/protection/output inputs')
        result=create_plan(*(getattr(a,k) for k in names),workers=a.workers,reuse_verified_proofs=a.reuse_verified_proofs)
        print(json.dumps(dict(plan=str(a.output/'deletion_plan.json'),eligible=sum(x['status']=='retire' for x in result['rows']),skipped=sum(x['status']=='skip' for x in result['rows']))))

if __name__=='__main__':main()
