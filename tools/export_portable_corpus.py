"""Export exact compact caches in uncompressed, resumable tar shards; restore locally.

python tools/export_portable_corpus.py --inventory inventory.json --output bundle export
python tools/export_portable_corpus.py --bundle bundle --destination restored restore

Original raw provenance is preserved. Restored cache manifests have local absolute
cache/log paths and recomputed identities; original caches are never modified.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode())


def safe_relative(name):
    value = PurePosixPath(name)
    if not isinstance(name, str) or not name or '\\' in name or value.is_absolute() or any(
            part in {'..', '.'} for part in name.split('/')) or '' in name.split('/'):
        raise ValueError(f'unsafe archive path: {name!r}')
    return value


def local_path(root, name):
    relative = safe_relative(name)
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'path escapes destination: {name}')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'symlink in destination: {name}')
    return target


def file_stamp(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def check_space(output, needed, reserve):
    if shutil.disk_usage(output).free < needed + reserve:
        raise RuntimeError(f'insufficient disk: require {needed + reserve} free bytes including reserve')


def verified_file(root, record):
    path = local_path(root, record['path'])
    if not path.is_file() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
        raise ValueError(f'checksum/size mismatch: {path}')
    return path


def translated_manifest(manifest, city, destination=None):
    value = dict(manifest)
    def translate(entry):
        row = dict(entry)
        trajectory = row['trajectory_hash']
        if len(trajectory) != 64 or any(c not in '0123456789abcdef' for c in trajectory):
            raise ValueError('invalid trajectory hash')
        cache = f'cities/{city}/episodes/{trajectory}.npz'
        log = f'cities/{city}/logs/{trajectory}.json.gz'
        row['cache_path'] = str(local_path(destination, cache)) if destination else cache
        if row.get('summary_log_path'):
            row['summary_log_path'] = str(local_path(destination, log)) if destination else log
        return row
    for key in ('entries', 'sources'):
        value[key] = [translate(entry) for entry in manifest[key]]
    value.pop('identity', None)
    value['identity'] = identity(value)
    return value


def source_records(manifest, city):
    records = []
    seen = set()
    for entry in manifest['entries']:
        for source_key, relative in (
                ('cache_path', f"cities/{city}/episodes/{entry['trajectory_hash']}.npz"),
                ('summary_log_path', f"cities/{city}/logs/{entry['trajectory_hash']}.json.gz")):
            if not entry.get(source_key):
                continue
            safe_relative(relative)
            if relative in seen:
                raise ValueError(f'duplicate canonical artifact: {relative}')
            seen.add(relative)
            source = Path(entry[source_key]).resolve(strict=True)
            if not source.is_file():
                raise ValueError(f'not a file: {source}')
            stamp = file_stamp(source)
            records.append(dict(source=str(source), path=relative, bytes=stamp[2], stamp=stamp))
    return records


def shard_groups(records, target):
    batch, size = [], 0
    for record in records:
        cost = ((record['bytes'] + 511) // 512) * 512 + 1536
        if batch and size + cost > target:
            yield batch
            batch, size = [], 0
        batch.append(record)
        size += cost
    if batch:
        yield batch


def write_shard(output, name, records, reserve):
    path = local_path(output, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    # A crashed, unpublished shard is regenerated, never mistaken for completion.
    temporary.unlink(missing_ok=True)
    check_space(output, sum(r['bytes'] + 2048 for r in records) + 10240, reserve)
    with temporary.open('wb') as stream:
        with tarfile.open(fileobj=stream, mode='w', format=tarfile.PAX_FORMAT) as archive:
            for row in records:
                source = Path(row['source'])
                if file_stamp(source) != row['stamp']:
                    raise ValueError(f'source changed before export: {source}')
                info = tarfile.TarInfo(row['path'])
                info.size, info.mode, info.mtime = row['bytes'], 0o644, 0
                with source.open('rb') as data:
                    archive.addfile(info, data)
                if file_stamp(source) != row['stamp']:
                    raise ValueError(f'source changed during export: {source}')
        stream.flush()
        os.fsync(stream.fileno())
    record = dict(path=name, bytes=temporary.stat().st_size, sha256=sha256(temporary),
                  members=[dict(path=r['path'], bytes=r['bytes']) for r in records],
                  source_recipe_identity=identity(records))
    temporary.replace(path)
    return record


def export(inventory_path, output, *, shard_bytes=GIB, reserve_bytes=10 * GIB):
    inventory_path, output = Path(inventory_path), Path(output).resolve()
    if shard_bytes <= 0 or reserve_bytes < 0:
        raise ValueError('invalid shard/reserve size')
    output.mkdir(parents=True, exist_ok=True)
    # One writer owns all publication state. Advisory flock releases after crashes.
    import fcntl
    with (output / '.export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_space(output, 0, reserve_bytes)
        inventory = json.loads(inventory_path.read_text())
        contract = dict(schema_version=1, shard_bytes=shard_bytes, cities={})
        for city, details in sorted(inventory['cities'].items()):
            safe_relative(city)
            if '/' in city:
                raise ValueError('invalid city')
            contract['cities'][city] = dict(manifest_sha256=sha256(details['manifest']))
        contract['identity'] = identity(contract)
        index_path = output / 'index.json'
        if index_path.exists():
            index = json.loads(index_path.read_text())
            if index['contract'] != contract:
                raise ValueError('export input manifest or shard plan changed')
        else:
            index = dict(schema_version=1, status='building', contract=contract, shards=[], manifests={})
            atomic_json(index_path, index)
        readme = (
            '# UrbanDreamer portable compact corpus\n\n'
            'Download all indexed files and run:\n\n'
            '    python restore_corpus.py --bundle . --destination /absolute/data/root restore\n\n'
            'The restore tool uses only the Python standard library, verifies every shard SHA256, '
            'rejects unsafe tar members, and writes manifests/<city>.json with local absolute paths. '
            'Use these manifests with CompactPretrainDataset (compact_only_v1). '
            'Raw provenance paths are historical identifiers and need not exist. '
            'Original raw trajectories are not included. Network inputs are under networks/<city>.\n\n'
            'Retarget training config network/output/cache paths to your machine. '
            'Training code, its environment, and accepted codec/calibration prerequisites are '
            'separate requirements. Changed manifest identities are appropriate for fresh training; '
            'do not bypass checkpoint replay-identity checks to resume an old run. '
            'Ingolstadt uses the corrected 42926/7637 train/validation split.\n'
        )
        atomic_bytes(output / 'README.md', readme.encode())
        atomic_bytes(output / 'restore_corpus.py', Path(__file__).read_bytes())
        index['artifacts'] = [dict(path=name, bytes=(output / name).stat().st_size,
                                   sha256=sha256(output / name))
                              for name in ('README.md', 'restore_corpus.py')]
        atomic_json(index_path, index)
        old = {r['path']: r for r in index['shards']}
        expected_shards = set()
        def publish(name, records):
            expected_shards.add(name)
            if name in old:
                if old[name]['source_recipe_identity'] != identity(records):
                    raise ValueError(f'source recipe changed: {name}')
                verified_file(output, old[name])
            else:
                record = write_shard(output, name, records, reserve_bytes)
                index['shards'].append(record)
                old[name] = record
                atomic_json(index_path, index)
            atomic_json(output / 'progress.json', dict(status='building', last_shard=name,
                completed_shards=len(index['shards']), completed_bytes=sum(r['bytes'] for r in index['shards'])))
        networks = []
        for city, details in sorted(inventory['cities'].items()):
            manifest = json.loads(Path(details['manifest']).read_text())
            if (manifest.get('source_storage') != 'compact_only_v1' or manifest.get('status') != 'complete'
                    or manifest.get('identity') != identity({k: v for k, v in manifest.items() if k != 'identity'})):
                raise ValueError('invalid standalone manifest')
            records = source_records(manifest, city)
            portable = translated_manifest(manifest, city)
            portable_name = f'manifests/{city}.json.gz'
            portable_path = local_path(output, portable_name)
            data = gzip.compress(json.dumps(portable, sort_keys=True, separators=(',', ':')).encode(), mtime=0)
            atomic_bytes(portable_path, data)
            index['manifests'][city] = dict(path=portable_name, bytes=len(data), sha256=sha256(portable_path),
                original_identity=manifest['identity'], portable_identity=portable['identity'])
            atomic_json(index_path, index)
            del manifest, portable, data
            for number, group in enumerate(shard_groups(records, shard_bytes)):
                publish(f'shards/{city}-{number:05d}.tar', group)
            network_dir = Path(details.get('network_dir', ROOT / 'networks' / city))
            for path in sorted(network_dir.rglob('*')):
                if path.is_file():
                    stamp = file_stamp(path)
                    networks.append(dict(source=str(path.resolve()), path=f'networks/{city}/{path.relative_to(network_dir).as_posix()}',
                                         bytes=stamp[2], stamp=stamp))
            if not any(r['path'].startswith(f'networks/{city}/') for r in networks):
                raise ValueError(f'no network files for {city}')
        publish('shards/networks.tar', networks)
        if set(old) != expected_shards:
            raise ValueError('unexpected shards from different source plan')
        index['status'] = 'complete'
        atomic_json(index_path, index)
        atomic_json(output / 'progress.json', dict(status='complete', completed_shards=len(index['shards']),
                    completed_bytes=sum(r['bytes'] for r in index['shards'])))
        return index


def restore(bundle, destination):
    bundle, destination = Path(bundle).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    index = json.loads((bundle / 'index.json').read_text())
    if index.get('schema_version') != 1 or index.get('status') != 'complete':
        raise ValueError('bundle is incomplete')
    # Verify every checksum before any extraction, including all manifest files.
    for record in index['shards'] + list(index['manifests'].values()):
        verified_file(bundle, record)
    seen = set()
    for record in index['shards']:
        expected = {row['path']: row['bytes'] for row in record['members']}
        if len(expected) != len(record['members']):
            raise ValueError('duplicate shard index member')
        extracted = set()
        with tarfile.open(local_path(bundle, record['path']), 'r:') as archive:
            for member in archive:
                name = member.name
                if not member.isfile() or name not in expected or member.size != expected[name] or name in seen:
                    raise ValueError(f'unsafe/unexpected tar member: {name}')
                path = local_path(destination, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + '.restore.tmp')
                if temporary.is_symlink():
                    raise ValueError('unsafe restore temporary path')
                with archive.extractfile(member) as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target, 8 * 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                temporary.replace(path)
                seen.add(name)
                extracted.add(name)
        if extracted != set(expected):
            raise ValueError('missing indexed tar member')
    restored = {}
    for city, record in index['manifests'].items():
        safe_relative(city)
        if '/' in city:
            raise ValueError('invalid city')
        manifest = json.loads(gzip.decompress(local_path(bundle, record['path']).read_bytes()))
        if manifest['identity'] != identity({k: v for k, v in manifest.items() if k != 'identity'}):
            raise ValueError('invalid portable manifest identity')
        for row in manifest['entries'] + manifest['sources']:
            for key in ('cache_path', 'summary_log_path'):
                if row.get(key) and (row[key] not in seen or not local_path(destination, row[key]).is_file()):
                    raise ValueError(f'unrestored manifest artifact: {row.get(key)}')
        relocated = translated_manifest(manifest, city, destination)
        path = local_path(destination, f'manifests/{city}.json')
        atomic_json(path, relocated)
        restored[city] = dict(path=str(path), identity=relocated['identity'], original_identity=record['original_identity'])
    atomic_json(destination / 'restore_result.json', dict(status='complete', manifests=restored))
    return restored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--shard-bytes', type=int, default=GIB)
    parser.add_argument('--reserve-bytes', type=int, default=10 * GIB)
    parser.add_argument('command', choices=('export', 'restore'))
    args = parser.parse_args(argv)
    if args.command == 'export':
        if args.inventory is None or args.output is None:
            parser.error('export requires --inventory and --output')
        try:
            export(args.inventory, args.output, shard_bytes=args.shard_bytes, reserve_bytes=args.reserve_bytes)
        except Exception as exc:
            atomic_json(args.output / 'progress.json', dict(status='failed', error=str(exc)))
            raise
    else:
        if args.bundle is None or args.destination is None:
            parser.error('restore requires --bundle and --destination')
        restore(args.bundle, args.destination)


if __name__ == '__main__':
    main()
