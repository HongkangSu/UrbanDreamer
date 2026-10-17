"""Upload sealed corpus shards to a private HF dataset while export continues.

Uses the existing huggingface_hub installation. Local caches are never removed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def file_hash(path, git=False):
    h = hashlib.sha1() if git else hashlib.sha256()
    if git:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def verify_remote(api, repo, name, path, expected):
    files = api.get_paths_info(repo, [name], repo_type='dataset')
    if len(files) != 1 or files[0].size != expected['bytes']:
        raise ValueError('remote_size_mismatch')
    remote = files[0]
    if remote.lfs:
        sha = remote.lfs.get('sha256') if isinstance(remote.lfs, dict) else remote.lfs.sha256
        if sha != expected['sha256']:
            raise ValueError('remote_lfs_hash_mismatch')
    elif remote.blob_id != file_hash(path, git=True):
        raise ValueError('remote_git_hash_mismatch')


def upload_one(bundle, receipts, repo, record):
    name = record['path']
    path = bundle / name
    if path.is_symlink() or not path.resolve().is_relative_to(bundle.resolve()):
        raise ValueError('unsafe_upload_path')
    if path.stat().st_size != record['bytes'] or file_hash(path) != record['sha256']:
        raise ValueError('local_artifact_hash_mismatch')
    receipt_path = receipts / (hashlib.sha256(name.encode()).hexdigest() + '.json')
    api = HfApi(endpoint='https://huggingface.co')
    for attempt in range(5):
        try:
            previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            if previous and all(previous.get(k) == record[k] for k in ('path', 'bytes', 'sha256')) and previous.get('repo') == repo:
                verify_remote(api, repo, name, path, record)
                return previous
            start = time.monotonic()
            commit = api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo,
                repo_type='dataset', commit_message=f'Upload verified corpus artifact: {name}')
            verify_remote(api, repo, name, path, record)
            result = {k: record[k] for k in ('path', 'bytes', 'sha256')}
            result.update(repo=repo, commit=commit.oid, seconds=time.monotonic()-start,
                          verified_at=time.time(), remote_hash_verified=True)
            atomic_json(receipt_path, result)
            return result
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(30, 2**attempt))


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    receipts = args.output / 'receipts'
    receipts.mkdir(exist_ok=True)
    lock = (args.output / 'upload.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    api = HfApi(endpoint='https://huggingface.co')
    if api.repo_info(args.repo, repo_type='dataset').private is not True:
        raise ValueError('upload_requires_existing_private_dataset')
    status = dict(status='waiting_for_export', pid=os.getpid(), repo=args.repo,
                  started_at=time.time(), completed_files=0, verified_bytes=0)
    pending, submitted, results = {}, set(), {}
    def publish(phase):
        status.update(status=phase, heartbeat_at=time.time(), completed_files=len(results),
                      verified_bytes=sum(r['bytes'] for r in results.values()),
                      uploading=[r['path'] for r in pending.values()])
        atomic_json(args.output / 'status.json', status)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while True:
                index_path = args.bundle / 'index.json'
                index = json.loads(index_path.read_text()) if index_path.exists() else {}
                producer = args.bundle / 'progress.json'
                progress = json.loads(producer.read_text()) if producer.exists() else {}
                if progress.get('status') == 'failed':
                    raise RuntimeError('export_failed')
                records = index.get('shards', []) + list(index.get('manifests', {}).values())
                for record in records:
                    if len(pending) >= args.workers:
                        break
                    if record['path'] not in submitted:
                        future = pool.submit(upload_one, args.bundle, receipts, args.repo, record)
                        pending[future] = record
                        submitted.add(record['path'])
                publish('uploading' if pending else 'waiting_for_export')
                if pending:
                    done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                    for future in done:
                        record = pending.pop(future)
                        results[record['path']] = future.result()
                elif index.get('status') == 'complete' and len(results) == len(records):
                    break
                else:
                    time.sleep(10)
            # Publish the complete index last, after all listed artifacts verify.
            for name in ('restore_corpus.py', 'README.md', 'index.json'):
                path = args.bundle / name
                record = dict(path=name, bytes=path.stat().st_size, sha256=file_hash(path))
                results[name] = upload_one(args.bundle, receipts, args.repo, record)
            atomic_json(args.output / 'verification.json', dict(status='passed', repo=args.repo,
                private=api.repo_info(args.repo, repo_type='dataset').private,
                files=list(results.values()), completed_at=time.time()))
            publish('complete')
    except Exception as error:
        status['error_type'] = type(error).__name__
        response = getattr(error, 'response', None)
        if response is not None:
            status['http_status'] = response.status_code
        publish('failed')
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
