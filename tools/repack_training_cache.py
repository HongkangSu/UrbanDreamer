"""Atomically repack immutable manifest-backed compact caches, never raw archives."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compact_episode_archives import _verify_members, _identity
from urban_dreamer.learning.offline_data import archive_content_hash


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def manifest_plan(manifests):
    rows, manifest_records = {}, []
    for manifest in manifests:
        blob = manifest.read_bytes()
        data = json.loads(blob)
        if data.get('status') != 'complete':
            raise ValueError(f'incomplete manifest: {manifest}')
        manifest_records.append(dict(path=str(manifest.resolve()), sha256=hashlib.sha256(blob).hexdigest(), entries=len(data['entries'])))
        for entry in data['entries']:
            alias = Path(entry['cache_path']).absolute()
            path = alias.resolve(strict=True)
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or path.suffix != '.npz':
                raise ValueError(f'not regular compact cache: {path}')
            key = (info.st_dev, info.st_ino)
            if key not in rows:
                rows[key] = dict(path=str(path), aliases=[], expected_hash=entry['cache_content_hash'], city=data['contract']['city'])
            row = rows[key]
            if row['expected_hash'] != entry['cache_content_hash']:
                raise ValueError('manifests disagree on same inode content')
            if str(alias) not in row['aliases']:
                row['aliases'].append(str(alias))
    return dict(manifests=manifest_records, entries=list(rows.values()), created_at=now())


def repack_one(row, output, *, min_free_bytes=1024**3):
    path = Path(row['path'])
    key = hashlib.sha256(str(path).encode()).hexdigest()
    receipt = output/'receipts'/f'{key}.json'
    with (output/'locks'/f'{key}.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.monotonic()
        original = path.lstat()
        report = dict(path=str(path), aliases=row['aliases'], city=row['city'], timestamp=now(),
                      before_bytes=original.st_size, expected_hash=row['expected_hash'])
        if stat.S_ISLNK(original.st_mode) or not stat.S_ISREG(original.st_mode):
            raise ValueError('resolved target is no longer a regular file')
        for alias in row['aliases']:
            if not Path(alias).samefile(path):
                raise ValueError('cache alias changed since plan')
        actual = archive_content_hash(path)
        if actual != row['expected_hash']:
            raise ValueError('manifest cache content hash mismatch')
        if _identity(path.lstat()) != _identity(original):
            raise RuntimeError('source changed during original hash verification')
        report['verified_content_hash'] = actual
        temporary = None
        try:
            with zipfile.ZipFile(path) as source:
                members = source.infolist()
                if not members or len({m.filename for m in members}) != len(members) or any(not m.filename.endswith('.npy') for m in members):
                    raise ValueError('expected unique NPY members')
                if all(m.compress_type == zipfile.ZIP_LZMA for m in members):
                    report.update(status='already_lzma', after_bytes=original.st_size)
                elif original.st_nlink != 1:
                    report.update(status='skipped_hardlinks', after_bytes=original.st_size, hardlinks=original.st_nlink)
                elif shutil.disk_usage(path.parent).free < min_free_bytes + max(original.st_size*2, 16*1024**2):
                    raise RuntimeError('less than required free space reserve')
                else:
                    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', suffix='.repack.tmp', delete=False) as stream:
                        temporary = Path(stream.name)
                        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_LZMA) as destination:
                            destination.comment = source.comment
                            for member in members:
                                info = zipfile.ZipInfo(member.filename, date_time=member.date_time)
                                info.compress_type = zipfile.ZIP_LZMA
                                info.external_attr = member.external_attr
                                info.comment = member.comment
                                with source.open(member) as old, destination.open(info, 'w', force_zip64=True) as new:
                                    shutil.copyfileobj(old, new, 1024*1024)
                        stream.flush()
                        os.fsync(stream.fileno())
                    _verify_members(path, temporary)
                    # Exact per-member byte equality proves archive_content_hash is unchanged.
                    report['new_hash_proof'] = 'original_sha256_verified_then_all_member_bytes_equal'
                    compressed = temporary.stat().st_size
                    if compressed >= original.st_size:
                        report.update(status='not_smaller', after_bytes=original.st_size)
                    else:
                        for alias in row['aliases']:
                            if not Path(alias).samefile(path):
                                raise RuntimeError('cache alias changed during compression')
                        current = path.lstat()
                        if _identity(current) != _identity(original) or current.st_nlink != 1:
                            raise RuntimeError('source changed during compression; refusing replacement')
                        os.chmod(temporary, stat.S_IMODE(original.st_mode))
                        os.replace(temporary, path)
                        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                        try:
                            os.fsync(directory)
                        finally:
                            os.close(directory)
                        report.update(status='compacted', after_bytes=compressed)
            report.update(saved_bytes=report['before_bytes']-report['after_bytes'], seconds=time.monotonic()-started)
            if receipt.exists():
                previous = json.loads(receipt.read_text())
                if report['status'] == 'already_lzma' and previous.get('status') == 'compacted' and previous['expected_hash'] == actual:
                    report.update(original_compaction=previous.get('original_compaction', previous))
            atomic_json(receipt, report)
            return report
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/pretrain_storage_migration_20261016/repack')
    parser.add_argument('--workers', type=int, default=4, choices=(1,2,3,4))
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for directory in ('receipts','locks'):
        (args.output/directory).mkdir(exist_ok=True)
    # All invocations share one global lock regardless of output destination.
    lock_path = ROOT/'outputs/pretrain_storage_migration_20261016/repack/global.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+b') as global_lock:
        fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.nice(10)
        corpus = json.loads((ROOT/'outputs/pretrain_data_audit_20261016/cached_corpus.json').read_text())
        plan = manifest_plan([Path(info['manifest']) for info in corpus.values()])
        atomic_json(args.output/('smoke_plan.json' if args.smoke else 'plan.json'), plan)
        entries = plan['entries']
        if args.smoke:
            cities = sorted({row['city'] for row in entries})
            groups = {city: [row for row in entries if row['city']==city] for city in cities}
            entries = [groups[city][i] for i in range(3) for city in cities if i<len(groups[city])][:8]
            for manifest in plan['manifests']:
                data = json.loads(Path(manifest['path']).read_text())
                by_path = {entry['cache_path']: entry for entry in data['entries']}
                for row in entries:
                    for alias in row['aliases']:
                        if alias in by_path:
                            row['smoke_entry'] = by_path[alias]
                            row['smoke_physics'] = data['contract'].get('physics_identity', {'traffic_model':'micro'})
        elif args.limit:
            entries = entries[:args.limit]
        progress_path = args.output/('smoke_progress.json' if args.smoke else 'progress.json')
        progress = dict(pid=os.getpid(), status='running', started_at=now(), total=len(entries), completed=0,
                        counts={}, bytes_saved=0, workers=args.workers, errors=[])
        atomic_json(progress_path, progress)
        def run(row):
            try:
                result = repack_one(row, args.output)
                if args.smoke:
                    import numpy as np
                    with np.load(row['path'], allow_pickle=False) as archive:
                        for name in archive.files:
                            array = archive[name]
                            if array.dtype.hasobject:
                                raise ValueError('unexpected object array')
                    if archive_content_hash(Path(row['path'])) != row['expected_hash']:
                        raise ValueError('post-replacement smoke hash mismatch')
                    from collections import OrderedDict
                    from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
                    dataset = object.__new__(CompactPretrainDataset)
                    dataset.manifest = {'entries': [row['smoke_entry']]}
                    dataset._source_stamps = {}
                    dataset.verify_sources = 'always'
                    dataset._cache = OrderedDict()
                    dataset.cache_limit = 1
                    dataset._physics_identity = row['smoke_physics']
                    loaded = dataset._episode(0)
                    assert loaded['flow'].ndim == 3 and 'metadata' in loaded
                    result['smoke_numpy_all_arrays_read'] = True
                    result['smoke_real_compact_dataset_read'] = True
                    atomic_json(args.output/'receipts'/(hashlib.sha256(row['path'].encode()).hexdigest()+'.json'), result)
                return result
            except Exception as error:
                result = dict(path=row['path'], expected_hash=row['expected_hash'], status='error', error=repr(error), timestamp=now())
                key = hashlib.sha256(row['path'].encode()).hexdigest()
                atomic_json(args.output/'receipts'/f'{key}.error.json', result)
                return result
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run, row): row for row in entries}
            for future in as_completed(futures):
                result = future.result()
                status = result['status']
                progress['counts'][status] = progress['counts'].get(status,0)+1
                progress['bytes_saved'] += result.get('saved_bytes',0)
                progress['completed'] += 1
                progress['updated_at'] = now()
                if status == 'error':
                    progress['errors'].append(result)
                atomic_json(progress_path, progress)
        unchanged = all(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'] for row in plan['manifests'])
        progress.update(status='complete' if not progress['errors'] and unchanged else 'completed_with_errors',
                        manifests_byte_identical=unchanged, completed_at=now())
        atomic_json(progress_path, progress)
        print(json.dumps(progress), flush=True)
        if progress['status'] != 'complete':
            raise SystemExit(1)

if __name__ == '__main__':
    main()
