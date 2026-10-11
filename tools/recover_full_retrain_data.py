"""Recover a frozen retraining selection after deletion of source/cache files.

Base recovery preserves original census paths and exclusions. An explicitly
authorized expansion uses its own frozen selection and retains base partitions.
No compact archive is represented as a reconstructed original SUMO NPZ.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_pretrain_cache import atomic_json, build_one, initialize_worker
from urban_dreamer.learning.offline_data import _identity, archive_content_hash
from urban_dreamer.learning.pretrain_cache import file_stamp, member_signature


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(data_dir):
    census = json.loads((data_dir / 'census.json').read_text())
    frozen_path = data_dir / 'frozen_candidates.json'
    frozen = json.loads(frozen_path.read_text())
    frozen_set = set(frozen['paths'])
    eligible, excluded = [], []
    for row in census['rows']:
        if row['path'] not in frozen_set:
            raise ValueError('census path outside original frozen selection')
        if row['split'] != census['groups'][row['demand_group']]:
            raise ValueError('original demand group partition mismatch')
        receipt = data_dir / 'receipts' / f"{_identity({'path':row['path']})}.json"
        raw_available = Path(row['path']).is_file()
        saved = json.loads(receipt.read_text()) if receipt.exists() else None
        cache_available = bool(saved and Path(saved.get('cache_path', '')).is_file())
        if raw_available:
            eligible.append(dict(row=row, receipt=str(receipt) if cache_available else None,
                                 raw_available=raw_available))
        else:
            excluded.append(dict(path=row['path'], split=row['split'],
                demand_group=row['demand_group'], root=row['root'],
                reason='missing_original_and_no_validated_compact_receipt',
                crc_trajectory_candidate=row['crc_trajectory_candidate'],
                source_stamp=row['source_stamp'], episode=row['episode'],method=row['method']))
    return census, dict(schema_version=1, original_frozen_sha256=file_sha256(frozen_path),
        original_census_sha256=file_sha256(data_dir / 'census.json'),
        frozen_files=len(frozen['paths']), compatible_files=len(census['rows']),
        eligible_files=len(eligible), excluded_files=len(excluded),
        pending_raw_files=sum(e['receipt'] is None for e in eligible),
        missing_raw_recovered_files=sum(not e['raw_available'] for e in eligible),
        excluded_roots=dict(Counter(e['root'] for e in excluded)), exclusions=excluded), eligible


def validate_receipt(item):
    row, receipt_path = item['row'], Path(item['receipt'])
    receipt = json.loads(receipt_path.read_text())
    for key in ('path', 'demand_group', 'split', 'episode', 'method', 'root'):
        if receipt[key] != row[key]:
            raise ValueError(f'receipt/census {key} mismatch: {receipt_path}')
    if receipt['census_trajectory_signature'] != row['crc_trajectory_candidate']:
        raise ValueError(f'receipt/census trajectory mismatch: {receipt_path}')
    cache = Path(receipt['cache_path'])
    if cache.stem != receipt['trajectory_hash'] or archive_content_hash(cache) != receipt['cache_content_hash']:
        raise ValueError(f'compact cache identity mismatch: {cache}')
    with np.load(cache, allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata_json'].tobytes())
        windows = receipt['windows']
        if metadata['population'] != receipt['population'] or metadata['completed'] != receipt['completed']:
            raise ValueError(f'compact metadata mismatch: {cache}')
        if archive['flow'].shape[0] != windows + 1 or len(archive['time']) != windows + 1:
            raise ValueError(f'compact flow/time length mismatch: {cache}')
        if len(archive['event_vehicle']) != receipt['events']:
            raise ValueError(f'compact event count mismatch: {cache}')
        if len(archive['vehicle_ids']) != receipt['population']:
            raise ValueError(f'compact population mismatch: {cache}')
        if len(archive['boundary_offsets']) != windows + 2:
            raise ValueError(f'compact boundary count mismatch: {cache}')
        if not np.isfinite(archive['flow']).all():
            raise ValueError(f'nonfinite compact observations: {cache}')
    source = Path(row['path'])
    if source.exists():
        with zipfile.ZipFile(source) as archive:
            if member_signature(archive.infolist()) != receipt['source_member_signature']:
                raise ValueError(f'original source content changed: {source}')
        stamp = file_stamp(source)
        if stamp != receipt['source_stamp']:
            if archive_content_hash(source) != receipt['content_hash']:
                raise ValueError(f'original source changed: {source}')
            receipt['source_stamp'] = stamp
    else:
        raise ValueError(f'original source disappeared during recovery: {source}')
    receipt['cache_bytes'] = cache.stat().st_size
    return receipt, dict(path=str(receipt_path.resolve()), sha256=file_sha256(receipt_path))


def build_pending(arguments):
    row, data_dir = arguments
    result = build_one(row, data_dir, True, None)
    path = Path(data_dir) / 'receipts' / f"{_identity({'path':row['path']})}.json"
    atomic_json(path, result)
    return dict(row=row, receipt=str(path), raw_available=True)


def contract(config_path):
    import tools.build_pretrain_cache as builder
    initialize_worker(str(config_path))
    validator = builder._VALIDATOR
    config = validator.config
    physical = {key: config.resolved_snapshot()['simulation'][key] for key in (
        'vehicle_max_speed_mps', 'incident_edge', 'incident_start_seconds',
        'incident_end_seconds', 'incident_speed_mps', 'sumo_extra_args', 'demand_duration_seconds')}
    from urban_dreamer.traffic.simulation_settings import physics_identity
    return dict(city=config.inputs.network_name, input_hashes=validator.prepared.input_hashes,
                physical=physical, physics_identity=physics_identity(config.simulation))


def recover(data_dir, config_path, *, workers=16, inventory_only=False):
    started = time.monotonic()
    census, audit, eligible = inventory(data_dir)
    atomic_json(data_dir / 'recovery_inventory.json', audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in {'exclusions'}}), flush=True)
    if inventory_only:
        return audit
    dataset_contract = contract(config_path)
    pending = [(e['row'], str(data_dir)) for e in eligible if e['receipt'] is None]
    reusable = [e for e in eligible if e['receipt'] is not None]
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(workers, mp_context=context, initializer=initialize_worker,
                             initargs=(str(config_path),)) as pool:
        for i, item in enumerate(pool.map(build_pending, pending, chunksize=1), 1):
            reusable.append(item)
            if i % 50 == 0 or i == len(pending):
                progress = dict(stage='rebuilding_available_originals', completed=i,
                    total=len(pending), elapsed_seconds=time.monotonic()-started)
                atomic_json(data_dir / 'recovery_progress.json', progress)
                print(json.dumps(progress), flush=True)
    sources, receipts = [], []
    with ProcessPoolExecutor(workers, mp_context=context) as pool:
        for i, (source, receipt) in enumerate(pool.map(validate_receipt, reusable, chunksize=1), 1):
            sources.append(source); receipts.append(receipt)
            if i % 250 == 0 or i == len(reusable):
                progress = dict(stage='validating_compact_and_original_identities',
                    completed=i, total=len(reusable), elapsed_seconds=time.monotonic()-started)
                atomic_json(data_dir / 'recovery_progress.json', progress)
                print(json.dumps(progress), flush=True)
    sources.sort(key=lambda s:s['path']); receipts.sort(key=lambda r:r['path'])
    grouped = {}
    for source in sources:
        prior = grouped.setdefault(source['trajectory_hash'], source)
        if prior['split'] != source['split']:
            raise ValueError('same trajectory spans train and validation demand groups')
        if prior['cache_content_hash'] != source['cache_content_hash']:
            raise ValueError('same original trajectory has inconsistent compact content')
    entries = sorted(grouped.values(), key=lambda s:s['path'])
    audit.update(status='complete', receipt_identities=receipts,
        recovered_source_files=len(sources), unique_trajectories=len(entries),
        config_sha256=file_sha256(config_path))
    audit['selection_identity'] = _identity(audit)
    atomic_json(data_dir / 'recovery_audit.json', audit)
    manifest = dict(schema_version=1, status='complete', contract=dataset_contract,
        plan_identity=audit['selection_identity'], census_frozen_at=census['frozen_at'],
        holdout_groups=sorted(g for g,split in census['groups'].items() if split=='validation'),
        sources=sources, entries=entries, duplicate_files=len(sources)-len(entries),
        train_episodes=sum(e['split']=='train' for e in entries),
        validation_episodes=sum(e['split']=='validation' for e in entries),
        cache_bytes=sum(e['cache_bytes'] for e in entries),
        elapsed_seconds=time.monotonic()-started,
        recovery=dict(selection_identity=audit['selection_identity'],
                      audit_path=str((data_dir/'recovery_audit.json').resolve()),
                      audit_sha256=file_sha256(data_dir/'recovery_audit.json'),
                      excluded_unrecoverable_files=len(audit['exclusions'])))
    manifest['identity'] = _identity(manifest)
    atomic_json(data_dir / 'cache_manifest.json', manifest)
    totals = {split:sum(e['windows'] for e in entries if e['split']==split) for split in ('train','validation')}
    summary = dict(train_episodes=manifest['train_episodes'],validation_episodes=manifest['validation_episodes'],
        unique_trajectories=len(entries), duplicate_files=manifest['duplicate_files'], windows=totals,
        excluded_unrecoverable_files=len(audit['exclusions']),
        methods={m:dict(episodes=sum(e['method']==m for e in entries),
            train_windows=sum(e['windows'] for e in entries if e['method']==m and e['split']=='train'))
            for m in sorted({e['method'] for e in entries})},
        microbatches={str(b):sum((e['windows']+b-1)//b for e in entries if e['split']=='train')
                     for b in (32,64,128)})
    atomic_json(data_dir / 'dataset_summary.json', summary)
    atomic_json(data_dir / 'recovery_progress.json', dict(stage='complete',**summary))
    print(json.dumps(summary), flush=True)
    return manifest



def seal_source_archives(data_dir):
    """Pin verified originals with hard links before publishing the final identity."""
    manifest_path = data_dir / 'cache_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['identity'] != _identity({k:v for k,v in manifest.items() if k!='identity'}):
        raise ValueError('manifest identity changed before sealing originals')
    if manifest.get('raw_source_snapshot', {}).get('method') == 'hardlink':
        return manifest
    destination = data_dir / 'source_archives'
    destination.mkdir(exist_ok=True)
    rewritten, mapping, snapshots = [], {}, {}
    for source in manifest['sources']:
        original = Path(source['path'])
        original_label = source.get('original_source_path',str(original))
        path_identity = hashlib.sha256(original_label.encode()).hexdigest()[:16]
        pinned = destination / f"{path_identity}_{source['content_hash']}.npz"
        if not pinned.exists():
            os.link(original, pinned)
        original_stat, pinned_stat = original.stat(), pinned.stat()
        if (original_stat.st_dev, original_stat.st_ino) != (pinned_stat.st_dev, pinned_stat.st_ino):
            raise ValueError(f'snapshot is not an original hard link: {pinned}')
        stamp = file_stamp(pinned)
        if stamp != source['source_stamp']:
            if archive_content_hash(pinned) != source['content_hash']:
                raise ValueError(f'original changed while pinning snapshot: {original}')
        with zipfile.ZipFile(pinned) as archive:
            if member_signature(archive.infolist()) != source['source_member_signature']:
                raise ValueError(f'original snapshot signature changed: {original}')
        result = {**source, 'path':str(pinned.resolve()), 'source_stamp':stamp,
                  'original_source_path':original_label, 'census_original_path':source.get('census_original_path',original_label)}
        rewritten.append(result); mapping[source['path']] = result
        snapshots[str(pinned.resolve())] = dict(content_hash=source['content_hash'],source_stamp=stamp)
    manifest['sources'] = rewritten
    manifest['entries'] = [mapping[e['path']] for e in manifest['entries']]
    manifest['raw_source_snapshot'] = dict(method='hardlink',directory=str(destination.resolve()),
        unique_archives=len(snapshots),additional_copied_bytes=0,identity=_identity(snapshots))
    audit_path = data_dir / 'recovery_audit.json'
    audit = json.loads(audit_path.read_text())
    audit.pop('selection_identity',None)
    audit['raw_source_snapshot'] = {**manifest['raw_source_snapshot'],'archives':snapshots}
    audit['selection_identity'] = _identity(audit)
    atomic_json(audit_path,audit)
    manifest['plan_identity'] = audit['selection_identity']
    manifest['recovery'].update(selection_identity=audit['selection_identity'],audit_sha256=file_sha256(audit_path),
                                raw_source_snapshot='hardlink')
    manifest.pop('identity',None);manifest['identity'] = _identity(manifest)
    atomic_json(manifest_path,manifest)
    atomic_json(data_dir/'recovery_sealed.json',dict(status='complete',manifest_identity=manifest['identity'],
                selection_identity=audit['selection_identity'],**manifest['raw_source_snapshot']))
    print(json.dumps(dict(stage='source_snapshot_sealed',**manifest['raw_source_snapshot'])),flush=True)
    return manifest

_SCAN = None


def initialize_scan(config_path, prepare_path):
    import importlib.util
    import tools.build_pretrain_cache as builder
    initialize_worker(config_path)
    spec = importlib.util.spec_from_file_location('_frozen_retrain_prepare',prepare_path)
    global _SCAN
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.V = builder._VALIDATOR
    _SCAN = module.scan


def scan_revision_path(path):
    return _SCAN(path)


def extend_revision(data_dir, config_path, *, workers=16):
    """Add the explicitly authorized new frozen collection to the sealed base."""
    started = time.monotonic()
    base_path = data_dir/'base_recovery_manifest.json'
    base = json.loads((data_dir/'cache_manifest.json').read_text())
    if base.get('recovery',{}).get('raw_source_snapshot') != 'hardlink':
        raise ValueError('base recovery must be sealed before expansion')
    if base['identity'] != _identity({k:v for k,v in base.items() if k!='identity'}):
        raise ValueError('sealed base manifest identity mismatch')
    if base_path.exists():
        base = json.loads(base_path.read_text())
    else:
        atomic_json(base_path,base)
        atomic_json(data_dir/'base_recovery_audit.json',json.loads((data_dir/'recovery_audit.json').read_text()))
    frozen_path = data_dir/'recovery_revision_frozen_candidates.json'
    frozen = json.loads(frozen_path.read_text())
    original = json.loads((data_dir/'frozen_candidates.json').read_text())
    if set(frozen['paths']) & set(original['paths']):
        raise ValueError('expansion contains original frozen candidates')
    context = multiprocessing.get_context('spawn')
    compatible, excluded = [], []
    with ProcessPoolExecutor(workers,mp_context=context,initializer=initialize_scan,
                             initargs=(str(config_path),str(data_dir/'prepare.py'))) as pool:
        for i,result in enumerate(pool.map(scan_revision_path,frozen['paths'],chunksize=8),1):
            (excluded if 'excluded' in result else compatible).append(result)
            if i % 250 == 0 or i == len(frozen['paths']):
                progress=dict(stage='scanning_new_frozen_sources',completed=i,total=len(frozen['paths']),
                    compatible=len(compatible),excluded=len(excluded),elapsed_seconds=time.monotonic()-started)
                atomic_json(data_dir/'recovery_revision_progress.json',progress)
                print(json.dumps(progress),flush=True)
    groups=dict(json.loads((data_dir/'census.json').read_text())['groups'])
    for row in compatible:
        g=row['demand_group']
        groups.setdefault(g,'validation' if int(hashlib.sha256(('full-retrain-v1:'+g).encode()).hexdigest()[:8],16)/2**32<.15 else 'train')
        row['split']=groups[g]
    census=dict(schema_version=1,frozen_at=frozen['frozen_at'],rows=compatible,excluded=excluded,
        groups=groups,excluded_counts=dict(Counter(e['excluded'] for e in excluded)),
        frozen_candidates_sha256=file_sha256(frozen_path))
    atomic_json(data_dir/'recovery_revision_census.json',census)
    reusable, pending = [], []
    for row in compatible:
        receipt=data_dir/'receipts'/f"{_identity({'path':row['path']})}.json"
        if receipt.exists():
            reusable.append(dict(row=row,receipt=str(receipt),raw_available=True))
        else:pending.append((row,str(data_dir)))
    with ProcessPoolExecutor(workers,mp_context=context,initializer=initialize_worker,
                             initargs=(str(config_path),)) as pool:
        for i,item in enumerate(pool.map(build_pending,pending,chunksize=1),1):
            reusable.append(item)
            if i % 50 == 0 or i == len(pending):
                progress=dict(stage='building_new_frozen_sources',completed=i,total=len(pending),
                    elapsed_seconds=time.monotonic()-started)
                atomic_json(data_dir/'recovery_revision_progress.json',progress)
                print(json.dumps(progress),flush=True)
    new_sources, receipts = [], []
    with ProcessPoolExecutor(workers,mp_context=context) as pool:
        for i,(source,receipt) in enumerate(pool.map(validate_receipt,reusable,chunksize=1),1):
            new_sources.append(source);receipts.append(receipt)
            if i % 250 == 0 or i == len(reusable):
                progress=dict(stage='validating_new_frozen_sources',completed=i,total=len(reusable),
                    elapsed_seconds=time.monotonic()-started)
                atomic_json(data_dir/'recovery_revision_progress.json',progress)
                print(json.dumps(progress),flush=True)
    sources=sorted(base['sources']+new_sources,key=lambda s:s['path'])
    grouped={}
    for source in sources:
        prior=grouped.setdefault(source['trajectory_hash'],source)
        if prior['split'] != source['split']:
            raise ValueError('new collection causes trajectory train/validation leakage')
        if prior['cache_content_hash'] != source['cache_content_hash']:
            raise ValueError('duplicate original trajectory has inconsistent compact content')
    entries=sorted(grouped.values(),key=lambda s:s['path'])
    expansion=dict(status='complete',revision='20261010_new_collection',
        frozen_candidates_sha256=file_sha256(frozen_path),frozen_at=frozen['frozen_at'],
        candidate_files=len(frozen['paths']),compatible_added_sources=len(new_sources),
        excluded_files=len(excluded),new_unique_trajectories=len(entries)-len(base['entries']),
        base_manifest_identity=base['identity'],revision_census_sha256=file_sha256(data_dir/'recovery_revision_census.json'))
    audit=json.loads((data_dir/'base_recovery_audit.json').read_text())
    audit.pop('selection_identity',None)
    audit['expansion']={**expansion,'receipt_identities':sorted(receipts,key=lambda r:r['path'])}
    audit['selection_identity']=_identity(audit)
    atomic_json(data_dir/'recovery_audit.json',audit)
    manifest={**base,'sources':sources,'entries':entries,'duplicate_files':len(sources)-len(entries),
        'train_episodes':sum(e['split']=='train' for e in entries),
        'validation_episodes':sum(e['split']=='validation' for e in entries),
        'cache_bytes':sum(e['cache_bytes'] for e in entries),'expansion':expansion,
        'holdout_groups':sorted(g for g,s in groups.items() if s=='validation'),
        'elapsed_seconds':time.monotonic()-started,'plan_identity':audit['selection_identity']}
    manifest.pop('raw_source_snapshot',None)
    manifest['recovery']={**manifest['recovery'],'selection_identity':audit['selection_identity'],
                         'audit_sha256':file_sha256(data_dir/'recovery_audit.json')}
    manifest['recovery'].pop('raw_source_snapshot',None)
    manifest.pop('identity',None);manifest['identity']=_identity(manifest)
    atomic_json(data_dir/'cache_manifest.json',manifest)
    manifest=seal_source_archives(data_dir)
    summary=dict(train_episodes=manifest['train_episodes'],validation_episodes=manifest['validation_episodes'],
        unique_trajectories=len(entries),duplicate_files=manifest['duplicate_files'],
        windows={s:sum(e['windows'] for e in entries if e['split']==s) for s in ('train','validation')},
        excluded_unrecoverable_files=manifest['recovery']['excluded_unrecoverable_files'],
        expansion=expansion,
        methods={m:dict(episodes=sum(e['method']==m for e in entries),
            train_windows=sum(e['windows'] for e in entries if e['method']==m and e['split']=='train'))
            for m in sorted({e['method'] for e in entries})},
        microbatches={str(b):sum((e['windows']+b-1)//b for e in entries if e['split']=='train') for b in (32,64,128)})
    atomic_json(data_dir/'dataset_summary.json',summary)
    atomic_json(data_dir/'recovery_revision_progress.json',dict(stage='complete',**summary))
    print(json.dumps(summary),flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--inventory-only',action='store_true')
    parser.add_argument('--seal-only',action='store_true')
    parser.add_argument('--extend-revision',action='store_true')
    args = parser.parse_args()
    if args.workers < 1: parser.error('workers must be positive')
    if args.extend_revision:
        extend_revision(args.data_dir.resolve(),args.config.resolve(),workers=args.workers)
    elif args.seal_only:
        seal_source_archives(args.data_dir.resolve())
    else:
        recover(args.data_dir.resolve(),args.config.resolve(),workers=args.workers,inventory_only=args.inventory_only)
        if not args.inventory_only:
            seal_source_archives(args.data_dir.resolve())

if __name__=='__main__': main()
