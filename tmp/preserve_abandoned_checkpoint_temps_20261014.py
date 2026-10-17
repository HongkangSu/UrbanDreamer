"""Losslessly retain abandoned checkpoint writes while releasing their SSD copies."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queued_city_baselines as queue

BATCH = ROOT/'outputs/city_baselines_current_to_dest_1000_20261013_180607'
manifest = json.loads((BATCH/'manifest.json').read_text())
ledger = json.loads((BATCH/'queue_state.json').read_text())
assert not queue.process_identity(ledger['controller_pid'])
assert not any(queue.owned_alive(ledger['jobs'][e['id']], e) for e in manifest['runs'])
archive = BATCH/'disk_recovery_20261014/orphan_checkpoints'
archive.mkdir(parents=True, exist_ok=False)
items = []
for entry in manifest['runs']:
    if entry['method'].endswith('_ref'):
        continue
    output = Path(entry['output_dir'])
    for path in output.glob('.latest.pt.*.tmp'):
        if path.stat().st_size:
            assert queue.durable_episode(entry) > 0
            items.append((entry,path))


def preserve(item):
    entry,source = item
    started = time.monotonic()
    original = source.stat()
    target = archive/entry['city']/entry['method']/(source.name+'.gz')
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = target.with_suffix('.writing')
    digest = hashlib.sha256()
    with source.open('rb') as inp, temporary.open('xb') as out:
        with gzip.GzipFile(fileobj=out, mode='wb',compresslevel=1,mtime=0) as compressed:
            while chunk := inp.read(4*1024*1024):
                digest.update(chunk)
                compressed.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    verified = hashlib.sha256()
    with gzip.open(temporary,'rb') as inp:
        while chunk := inp.read(4*1024*1024):
            verified.update(chunk)
    assert digest.digest() == verified.digest()
    current = source.stat()
    assert (current.st_ino,current.st_size,current.st_mtime_ns)==(original.st_ino,original.st_size,original.st_mtime_ns)
    assert not queue.owned_alive(ledger['jobs'][entry['id']],entry)
    assert queue.durable_episode(entry)>0
    os.replace(temporary,target)
    source.unlink()
    result=dict(source=str(source),compressed_copy=str(target),sha256_uncompressed=digest.hexdigest(),
        before_bytes=original.st_size,after_bytes=target.stat().st_size,
        checkpoint_cursor=queue.durable_episode(entry),seconds=time.monotonic()-started,
        all_uncompressed_bytes_verified=True)
    queue.write_json(target.with_suffix('.json'),result)
    print(json.dumps(result),flush=True)
    return result


with ThreadPoolExecutor(max_workers=3) as pool:
    results=list(pool.map(preserve,items))
queue.write_json(BATCH/'disk_recovery_20261014/orphan_checkpoint_preservation.json',dict(
    updated_at=queue.now(),results=results,
    bytes_reclaimed=sum(r['before_bytes']-r['after_bytes'] for r in results),
    committed_checkpoints_unchanged=True))
