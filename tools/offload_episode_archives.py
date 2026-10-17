"""Move durable LZMA episode bytes to immutable storage, retaining local symlinks."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import threading
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compact_episode_archives import committed_episode


MIN_STORAGE_FREE_BYTES = 20 * 1024 ** 3
_SPACE_LOCK = threading.Lock()
_RESERVED_BYTES = 0


class StorageFullError(RuntimeError):
    pass


def _identity(value):
    return dict(device=value.st_dev, inode=value.st_ino, bytes=value.st_size,
        mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def _stream_hash(stream, output=None):
    digest = hashlib.sha256()
    size = 0
    while block := stream.read(1024 * 1024):
        digest.update(block)
        size += len(block)
        if output is not None:
            output.write(block)
    return digest.hexdigest(), size


def _readback(path):
    with path.open('rb') as stream:
        before = _identity(os.fstat(stream.fileno()))
        digest, size = _stream_hash(stream)
        if before != _identity(os.fstat(stream.fileno())) or before != _identity(path.stat()):
            raise RuntimeError('remote archive changed during readback')
    return digest, size, before


def _check_source(source, stream, original):
    if (original != _identity(source.lstat())
            or original != _identity(os.fstat(stream.fileno()))):
        raise RuntimeError('source changed during offload; local replacement refused')


def _rename_new(source, destination):
    # Linux/CIFS rename with NOREPLACE: a competing final path is never clobbered.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


@contextmanager
def _storage_reservation(directory, size):
    global _RESERVED_BYTES
    with _SPACE_LOCK:
        free = shutil.disk_usage(directory).free
        if free - _RESERVED_BYTES - size < MIN_STORAGE_FREE_BYTES:
            raise StorageFullError('private storage must retain at least 20 GiB free after upload reservations')
        _RESERVED_BYTES += size
    try:
        yield
    finally:
        with _SPACE_LOCK:
            _RESERVED_BYTES -= size


def offload_npz(source, destination, committed_episode):
    """Verify an immutable raw-byte copy before atomically linking its local name."""
    source, destination = Path(source).absolute(), Path(destination).absolute()
    match = re.fullmatch(r'episode_(\d+)\.npz', source.name)
    if match is None:
        raise ValueError('expected an episode_N.npz archive')
    episode = int(match[1])
    if (isinstance(committed_episode, bool) or not isinstance(committed_episode, Integral)
            or episode > committed_episode):
        raise ValueError('episode is newer than the committed checkpoint')
    if source == destination:
        raise ValueError('source and storage destination must differ')
    started = time.monotonic()
    with source.with_suffix('.compact.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if source.is_symlink():
            return dict(path=str(source), destination=str(destination), episode=episode,
                status='already_offloaded', link_target=os.readlink(source),
                local_identity=_identity(source.lstat()), seconds=time.monotonic()-started)
        original = source.lstat()
        if not stat.S_ISREG(original.st_mode):
            raise ValueError('expected a regular episode archive')
        original = _identity(original)
        report = dict(path=str(source), destination=str(destination), episode=episode,
            before_bytes=original['bytes'], source_identity=original)
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as stream:
            _check_source(source, stream, original)
            with zipfile.ZipFile(stream) as archive:
                members = archive.infolist()
                if not members or any(info.compress_type != zipfile.ZIP_LZMA for info in members):
                    return dict(report, status='not_lzma', seconds=time.monotonic()-started)
                if (len({info.filename for info in members}) != len(members)
                        or any(not info.filename.endswith('.npy') for info in members)):
                    raise ValueError('expected unique NPY members in a LZMA NPZ archive')
            stream.seek(0)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            reused = destination.exists()
            try:
                if reused:
                    digest, size = _stream_hash(stream)
                    remote_digest, remote_size, remote_identity = _readback(destination)
                else:
                    with _storage_reservation(destination.parent, original['bytes']):
                        temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.part')
                        with temporary.open('xb') as upload:
                            digest, size = _stream_hash(stream, upload)
                            upload.flush()
                            os.fsync(upload.fileno())
                        remote_digest, remote_size, remote_identity = _readback(temporary)
                        if digest != remote_digest or size != remote_size:
                            raise ValueError('remote archive SHA256 readback differs from source copy stream')
                        _check_source(source, stream, original)
                        _rename_new(temporary, destination)
                        temporary = None
                if digest != remote_digest or size != remote_size:
                    raise ValueError('existing remote archive differs from source; overwrite refused')
                _check_source(source, stream, original)
                link = source.with_name(f'.{source.name}.{uuid.uuid4().hex}.link')
                try:
                    link.symlink_to(destination)
                    _check_source(source, stream, original)
                    os.replace(link, source)
                    directory = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                finally:
                    link.unlink(missing_ok=True)
                return dict(report, status='offloaded', sha256=digest, remote_sha256=remote_digest,
                    verified_bytes=size, remote_identity=remote_identity, reused_destination=reused,
                    local_bytes_reclaimed=original['bytes'], seconds=time.monotonic()-started)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def watch(root, storage_root, *, workers=4, interval=30., once=False):
    root, storage_root = Path(root).resolve(), Path(storage_root).resolve()
    if root == storage_root or root in storage_root.parents or storage_root in root.parents:
        raise ValueError('local archive root and storage root must be separate trees')
    storage_root.mkdir(parents=True, exist_ok=True)
    counts = dict(offloaded=0, already_offloaded=0, not_lzma=0, error=0, blocked_storage=0)
    errors, seen = {}, {}
    reclaimed = 0

    def record(result=None):
        value = dict(pid=os.getpid(), storage_root=str(storage_root), counts=counts,
            local_bytes_reclaimed=reclaimed, min_storage_free_bytes=MIN_STORAGE_FREE_BYTES,
            storage_free_bytes=shutil.disk_usage(storage_root).free,
            last_check_unix=time.time(), errors=errors, last_result=result)
        path = root / 'offload_status.json'
        temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        os.replace(temporary, path)
        if result is not None:
            line = json.dumps(result, sort_keys=True)
            with (root / 'offload_log.jsonl').open('a') as log:
                log.write(line + '\n')
            print(line, flush=True)

    def pending():
        manifest = json.loads((root / 'manifest.json').read_text())
        for run in manifest['runs']:
            if run['method'].endswith('_ref'):
                continue
            try:
                output = Path(run['output_dir']).resolve()
                relative = output.relative_to(root)
                if (len(relative.parts) != 2 or relative.parts[1] != run['method']
                        or relative.parts[0] != run['city']):
                    raise ValueError('run output must match the manifest city/method under archive root')
                boundary = committed_episode(run)
                errors.pop(run['id'], None)
                for source in sorted((output / 'episodes/train').glob('episode_*.npz')):
                    if source.is_symlink():
                        continue
                    episode = re.fullmatch(r'episode_(\d+)\.npz', source.name)
                    if episode is None or int(episode[1]) > boundary:
                        continue
                    identity = _identity(source.lstat())
                    if seen.get(str(source)) != identity:
                        yield source, storage_root / source.relative_to(root), boundary
            except Exception as error:
                counts['error'] += 1
                errors[run['id']] = repr(error)
                record(dict(path=run['output_dir'], status='error', error=repr(error)))

    def run(item):
        source, destination, boundary = item
        try:
            return offload_npz(source, destination, boundary)
        except Exception as error:
            return dict(path=str(source), destination=str(destination),
                status='blocked_storage' if isinstance(error, StorageFullError) else 'error', error=repr(error))

    while True:
        record()
        items = iter(pending())
        with ThreadPoolExecutor(max_workers=workers) as executor:
            active = set()
            for _ in range(workers):
                item = next(items, None)
                if item is not None:
                    active.add(executor.submit(run, item))
            while active:
                completed, active = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    result = future.result()
                    counts[result['status']] += 1
                    if result['status'] in {'error', 'blocked_storage'}:
                        errors[result['path']] = result['error']
                    else:
                        errors.pop(result['path'], None)
                        if result['status'] == 'offloaded':
                            reclaimed += result['local_bytes_reclaimed']
                        elif result['status'] == 'not_lzma':
                            seen[result['path']] = result['source_identity']
                    record(result)
                    item = next(items, None)
                    if item is not None:
                        active.add(executor.submit(run, item))
        record()
        if once:
            return
        time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--storage-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--interval', type=float, default=30.)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1 or not 0 < args.interval < float('inf'):
        parser.error('workers and finite interval must be positive')
    watch(args.root, args.storage_root, workers=args.workers, interval=args.interval, once=args.once)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
