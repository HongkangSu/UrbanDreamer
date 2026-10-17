"""Losslessly repack finalized NPZ archives with ZIP LZMA.

The archive producer must have committed its checkpoint beyond an episode
before this helper is called. This is essential: producers do not acquire the
compactor lock. A file-stat recheck additionally detects intervening changes.
No arrays are deserialized, quantized, reordered, or omitted.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
import pickletools
from pathlib import Path
import shutil
import tempfile
import time
from typing import Iterable
import zipfile


def _identity(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _verify_members(source: Path, destination: Path) -> None:
    """Compare decompressed bytes, not merely CRCs or decoded array values."""
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(destination) as new:
        expected = [(info.filename, info.file_size, info.CRC) for info in old.infolist()]
        actual = [(info.filename, info.file_size, info.CRC) for info in new.infolist()]
        if expected != actual:
            raise ValueError("repacked member names, sizes, or CRCs differ")
        for info in old.infolist():
            with old.open(info) as left, new.open(info.filename) as right:
                while True:
                    a, b = left.read(1024 * 1024), right.read(1024 * 1024)
                    if a != b:
                        raise ValueError(f"repacked bytes differ for {info.filename}")
                    if not a:
                        break


def repack_npz(path: Path, *, committed_episode: int) -> dict:
    """Atomically compress a checkpoint-committed episode, preserving all bytes.

    ``committed_episode`` is a durable training/progress checkpoint, not the
    latest filename or a log line. Episodes newer than this boundary are never
    touched. Repeated invocations skip already-LZMA files. A persistent advisory
    lock serializes compactors; producers are protected by the checkpoint gate.
    """
    path = Path(path)
    number = int(path.stem.removeprefix("episode_"))
    if isinstance(committed_episode, bool) or committed_episode < number:
        raise ValueError("episode is newer than the committed checkpoint")
    if path.is_symlink() or path.suffix != ".npz" or not path.stem.startswith("episode_"):
        raise ValueError("expected a regular episode_N.npz archive")
    started = time.monotonic()
    with path.with_suffix(".compact.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = path.stat()
        report = {"path": str(path), "episode": number, "before_bytes": original.st_size}
        with zipfile.ZipFile(path) as archive:
            if archive.infolist() and all(info.compress_type == zipfile.ZIP_LZMA for info in archive.infolist()):
                return {**report, "after_bytes": original.st_size, "status": "already_lzma", "seconds": time.monotonic() - started}
            members = archive.infolist()
            if (not members or len({info.filename for info in members}) != len(members)
                    or any(not info.filename.endswith(".npy") for info in members)):
                raise ValueError("expected unique NPY members in NPZ archive")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".compact.tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                    with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_LZMA) as output:
                        output.comment = archive.comment
                        for info in members:
                            replacement = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                            replacement.compress_type = zipfile.ZIP_LZMA
                            replacement.external_attr = info.external_attr
                            replacement.comment = info.comment
                            with archive.open(info) as source, output.open(replacement, "w", force_zip64=True) as dest:
                                shutil.copyfileobj(source, dest, length=1024 * 1024)
                    handle.flush()
                    os.fsync(handle.fileno())
                _verify_members(path, temporary)
                if _identity(path.stat()) != _identity(original):
                    raise RuntimeError("source changed during compression; replacement refused")
                compressed_size = temporary.stat().st_size
                if compressed_size >= original.st_size:
                    return {**report, "after_bytes": original.st_size, "status": "not_smaller", "seconds": time.monotonic() - started}
                os.chmod(temporary, original.st_mode & 0o777)
                os.replace(temporary, path)
                directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
                return {**report, "after_bytes": compressed_size, "status": "compacted", "seconds": time.monotonic() - started}
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def compact_committed(paths: Iterable[tuple[Path, int]], *, workers: int = 2) -> list[dict]:
    """Bound concurrent LZMA encoders; callers supply checkpoint-eligible paths."""
    def run(item):
        path, committed = item
        try:
            return repack_npz(path, committed_episode=committed)
        except Exception as error:
            return {"path": str(path), "status": "error", "error": repr(error)}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run, paths))


def committed_episode(run: dict) -> int:
    """Read only the durable episode cursor, without unpickling model/replay data."""
    output = Path(run['output_dir'])
    if run['method'] == 'random':
        path = output / 'progress.json'
        return int(json.loads(path.read_text())['last_episode']) if path.exists() else 0
    path = output / 'latest.pt'
    if not path.exists():
        return 0
    with zipfile.ZipFile(path) as checkpoint:
        name = next(name for name in checkpoint.namelist() if name.endswith('/data.pkl'))
        with checkpoint.open(name) as stream:
            found = False
            for opcode, value, position in pickletools.genops(stream):
                if position > 65536:
                    break
                if opcode.name in {'BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE'} and value == 'episode':
                    found = True
                    continue
                if found and opcode.name in {'BINPUT', 'LONG_BINPUT', 'MEMOIZE', 'PUT'}:
                    continue
                if found:
                    if opcode.name in {'INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4'} and isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        return value
                    raise ValueError('invalid checkpoint episode cursor')
    raise ValueError('checkpoint has no leading episode cursor')


def watch(root: Path, *, workers: int, interval: float) -> None:
    root = root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    seen, failures = {}, {}
    counts = {'compacted': 0, 'already_lzma': 0, 'already_offloaded': 0, 'not_smaller': 0, 'error': 0}
    saved = 0
    status_path = root / 'compaction_status.json'

    def record():
        temporary = status_path.with_name(f'.{status_path.name}.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(dict(pid=os.getpid(), counts=counts, bytes_saved=saved,
            last_check_unix=time.time(), errors=failures), indent=2) + '\n')
        os.replace(temporary, status_path)

    def offloaded(path):
        storage = manifest.get('archive_storage_root')
        if not storage or not path.is_symlink():
            return False
        target = Path(os.path.abspath(path.parent / os.readlink(path)))
        return target.is_relative_to(Path(os.path.abspath(storage)))

    def compact(item):
        path, boundary = item
        skipped = dict(path=str(path), episode=int(path.stem.removeprefix('episode_')),
                       status='already_offloaded', before_bytes=0, after_bytes=0, seconds=0.)
        try:
            if offloaded(path):
                return skipped
            return repack_npz(path, committed_episode=boundary)
        except Exception as error:
            if offloaded(path):
                return skipped  # The uploader may finish after the initial scan.
            return {'path': str(path), 'status': 'error', 'error': repr(error)}

    while True:
        pending = []
        all_done = True
        for run in manifest['runs']:
            state_path = Path(run['state'])
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            all_done &= state.get('status') in {'completed', 'failed', 'blocked_disk', 'incomplete_archives', 'blocked_gpu_capacity'}
            try:
                boundary = committed_episode(run)
            except (OSError, ValueError, StopIteration, zipfile.BadZipFile) as error:
                failures[run['id']] = repr(error)
                continue
            for path in sorted((Path(run['output_dir']) / 'episodes' / 'train').glob('episode_*.npz')):
                if int(path.stem.removeprefix('episode_')) > boundary:
                    continue
                if path.is_symlink():
                    if offloaded(path):
                        failures.pop(str(path), None)
                        continue  # Immutable verified LZMA files already offloaded.
                    failures[str(path)] = 'archive symlink outside configured storage'
                    continue
                identity = _identity(path.stat())
                if seen.get(str(path)) != identity:
                    pending.append((path, boundary))
        record()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(compact, pending):
                counts[result['status']] += 1
                if result['status'] == 'error':
                    failures[result['path']] = result['error']
                else:
                    saved += result['before_bytes'] - result['after_bytes']
                    seen[result['path']] = _identity(Path(result['path']).stat())
                    failures.pop(result['path'], None)
                print(json.dumps(result, sort_keys=True), flush=True)
                record()
        if all_done:
            return
        time.sleep(interval)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument('--root', type=Path, help='Watch a dataset manifest and compact only durable episodes')
    parser.add_argument('--interval', type=float, default=60.)
    parser.add_argument("--committed-episode", type=int,
                        help="verified durable checkpoint boundary for all supplied paths")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.interval <= 0:
        parser.error('workers and interval must be positive')
    if args.root:
        watch(args.root, workers=args.workers, interval=args.interval)
    else:
        if not args.paths or args.committed_episode is None:
            parser.error('provide --root or paths with --committed-episode')
        for result in compact_committed(((path, args.committed_episode) for path in args.paths), workers=args.workers):
            print(json.dumps(result, sort_keys=True), flush=True)
