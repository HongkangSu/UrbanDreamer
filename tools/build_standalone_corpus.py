"""Build verified compact-only corpora; never remove or alter original archives."""
from __future__ import annotations
import argparse
import dataclasses
from contextlib import contextmanager, nullcontext
from unittest.mock import patch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, CancelledError, FIRST_COMPLETED, wait, as_completed
from functools import lru_cache
import fcntl
import gzip
import hashlib
import io
import json
import os
import multiprocessing
import uuid
from pathlib import Path
import shutil
import sys
import tempfile
import time
import threading
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from urban_dreamer.learning.offline_data import OfflineEpisodeDataset, _identity, archive_content_hash
from urban_dreamer.learning.pretrain_cache import compact_arrays, file_stamp, member_signature, read_hash_source
from urban_dreamer.learning.episode_archive import _decode, observed_arrival_times
from tools.probe_cache_compression import check_raw
from tools.census_pretrain_total import CITIES, target_contracts
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(value, sort_keys=True, indent=2)+'\n').encode())
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_dir(path.parent)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def process_identity(pid):
    """Linux PID start ticks and boot ID distinguish reused process IDs."""
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        if fields[0]=='Z': return None
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+fields[19]
    except FileNotFoundError:
        return None


def live_reservations(path):
    records=json.loads(path.read_text()) if path.exists() else {}
    return {token:row for token,row in records.items()
            if process_identity(row['pid'])==row['process_identity']}


@contextmanager
def reserve_disk(output, amount, *, min_free_bytes=1024**3, ledger_dir=None, timeout=120., free_bytes=None):
    """Short, durable admission lock; compression never runs under this lock."""
    ledger_dir=Path(ledger_dir or ROOT/'outputs/pretrain_storage_migration_20261016/corpus').resolve()
    ledger_dir.mkdir(parents=True,exist_ok=True)
    ledger=ledger_dir/'disk_reservations.json'
    lockpath=ledger_dir/'disk_admission.lock'
    token=uuid.uuid4().hex
    identity=process_identity(os.getpid())
    if identity is None or amount<0: raise ValueError('invalid disk reservation owner/size')
    deadline=time.monotonic()+timeout
    admitted=False
    while not admitted:
        with lockpath.open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            records=live_reservations(ledger)
            free=shutil.disk_usage(output).free if free_bytes is None else free_bytes
            outstanding=sum(r['bytes'] for r in records.values())
            if free>=min_free_bytes+amount+outstanding:
                records[token]=dict(pid=os.getpid(),process_identity=identity,bytes=amount)
                admitted=True
            atomic_json(ledger,records)
            if not admitted and (free<min_free_bytes+amount or time.monotonic()>=deadline):
                raise RuntimeError('insufficient disk reserve including in-flight reservations')
        if not admitted: time.sleep(.1)
    try:
        yield token
    finally:
        with lockpath.open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            records=live_reservations(ledger)
            own=records.get(token)
            if own and own['pid']==os.getpid() and own['process_identity']==identity:
                del records[token]
            atomic_json(ledger,records)


def initialize_process():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


@lru_cache(maxsize=3)
def validator(city):
    config = load_config(ROOT/f'configs/{city}.toml')
    obj = object.__new__(OfflineEpisodeDataset)
    obj.config, obj.prepared = config, validate_prepared_inputs(config)
    obj.topology = load_traffic_topology(obj.prepared)
    obj._explicit = True
    return obj


@lru_cache(maxsize=8)
def old_entries(manifest):
    data = json.loads(Path(manifest).read_text())
    if data['status'] != 'complete' or data['identity'] != _identity({k:v for k,v in data.items() if k!='identity'}):
        raise ValueError('invalid previous cache manifest')
    return {str(Path(e['cache_path']).resolve()): e for e in data['entries']}


def signature(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names): raise ValueError('duplicate archive members')
        return member_signature(archive.infolist())


def verify_cache(entry):
    path = Path(entry['cache_path'])
    stamp = file_stamp(path)
    if signature(path) != entry['cache_member_signature'] or archive_content_hash(path) != entry['cache_content_hash']:
        raise ValueError(f'compact content verification failed: {path}')
    if file_stamp(path) != stamp:
        # A concurrent lossless repack is allowed only after the replacement is checked too.
        if signature(path) != entry['cache_member_signature'] or archive_content_hash(path) != entry['cache_content_hash']:
            raise ValueError('compact changed during verification')


def check_preservation(data, arrays):
    result = check_raw(data, arrays)
    table = data['states'][0]['vehicles']
    if arrays['vehicle_ids'].tolist() != [v['vehicle_id'] for v in table]:
        raise ValueError('vehicle registry order changed')
    for column, key in (('vehicle_destination','destination_edge'), ('vehicle_origin','current_edge'),
                        ('vehicle_departure','departure_time')):
        if arrays[column].tolist() != [v[key] for v in table]:
            raise ValueError(f'vehicle registry field changed: {key}')
    restored = {str(arrays['vehicle_ids'][i]): float(at) for i,at in zip(arrays['arrival_vehicle'],arrays['arrival_time'])}
    if restored != observed_arrival_times(data): raise ValueError('arrival observations changed')
    result.update(full_initial_registry_equal=True, all_arrival_observations_equal=True)
    return result


def write_lzma(path, arrays):
    """Write each NPY directly to ZIP_LZMA and verify exact dtype/shape/bytes."""
    with path.open('wb') as stream:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_LZMA) as archive:
            for name, array in arrays.items():
                if array.dtype.hasobject: raise ValueError('object arrays are forbidden')
                with archive.open(name+'.npy', 'w', force_zip64=True) as member:
                    np.lib.format.write_array(member, array, allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(arrays): raise ValueError('compact member set changed')
        for name, expected in arrays.items():
            actual = loaded[name]
            if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes()!=expected.tobytes():
                raise ValueError(f'compact array bytes changed: {name}')


def save_summary(source, output, expected_signature=None):
    """Retain the entire original summary JSON, including resolved demand and metrics."""
    with zipfile.ZipFile(source) as archive:
        if expected_signature and member_signature(archive.infolist()) != expected_signature:
            raise ValueError('source changed before summary retention')
        value = _decode(np.load(io.BytesIO(archive.read('summary_json.npy')), allow_pickle=False))
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(payload).hexdigest()
    destination = output/'logs'/f'{digest}.json.gz'
    if not destination.exists():
        with reserve_disk(output,len(payload)+16*1024**2):
            temporary=None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(gzip.compress(payload, mtime=0)); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, destination); fsync_dir(destination.parent)
            finally:
                if temporary is not None: temporary.unlink(missing_ok=True)
    if gzip.decompress(destination.read_bytes()) != payload: raise ValueError('summary preservation mismatch')
    return dict(summary_log_path=str(destination.resolve()), summary_json_sha256=digest)


def convert_one(row, output, plan_identity, *, min_free_bytes=1024**3):
    output = Path(output)
    key = _identity({'census_path':row['path']})
    receipt = output/'receipts'/f'{key}.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['plan_identity'] != plan_identity: raise ValueError('receipt plan mismatch')
        verify_cache(saved)
        if saved.get('summary_log_path'):
            payload = gzip.decompress(Path(saved['summary_log_path']).read_bytes())
            if hashlib.sha256(payload).hexdigest()!=saved['summary_json_sha256']: raise ValueError('retained summary changed')
        return saved
    started = time.monotonic()
    if row.get('reusable_cache'):
        cache = str(Path(row['reusable_cache']).resolve())
        old = dict(old_entries(row['cache_manifest'])[cache])
        if old['trajectory_hash'] != row['trajectory_hash'] or old['split'] != row['split'] or old['demand_group']!=row['demand_group']:
            raise ValueError('reused canonical identity/split mismatch')
        saved = dict(old, cache_member_signature=signature(Path(cache)), reuse_existing_cache=True)
        verify_cache(saved)
        source = Path(old['path'])
        if source.exists():
            saved.update(save_summary(source, output, old['source_member_signature']))
        else:
            raise ValueError(f'original summary unavailable; preserve logs before raw retirement: {source}')
    else:
        source = Path(row['path'])
        if signature(source) != row['source_member_signature']: raise ValueError('census source signature mismatch')
        data, identity = read_hash_source(source)
        if identity['source_member_signature']!=row['source_member_signature'] or identity['census_trajectory_signature']!=row['crc_trajectory_candidate']:
            raise ValueError('census source/trajectory signature mismatch')
        for name in ('trajectory_hash','content_hash'):
            if row.get(name) and row[name]!=identity[name]: raise ValueError(f'census {name} mismatch')
        demand = next((data['metadata'][k] for k in ('demand_sha256','demand_hash','demand_realization_sha256','demand_realization_hash') if data['metadata'].get(k)),None)
        if demand != row['demand_group']: raise ValueError('census demand group mismatch')
        validator(row['city'])._validate(source, data)
        arrays = compact_arrays(data)
        preservation = check_preservation(data, arrays)
        destination = output/'episodes'/f"{identity['trajectory_hash']}.npz"
        estimate = sum(a.nbytes for a in arrays.values())+len(arrays)*2048+16*1024**2
        with reserve_disk(output,estimate,min_free_bytes=min_free_bytes):
            temporary = destination.with_name(f'.{destination.stem}.{os.getpid()}.{key[:12]}.tmp')
            try:
                write_lzma(temporary, arrays)
                content_hash = archive_content_hash(temporary)
                if destination.exists():
                    if archive_content_hash(destination)!=content_hash: raise ValueError('content-addressed destination conflict')
                else:
                    os.replace(temporary, destination); fsync_dir(destination.parent)
            finally: temporary.unlink(missing_ok=True)
        logs = save_summary(source, output, identity['source_member_signature'])
        if file_stamp(source)!=identity['source_stamp']:
            if archive_content_hash(source)!=identity['content_hash']: raise ValueError('raw changed during conversion')
        saved = dict(row, **identity, cache_path=str(destination.resolve()), cache_content_hash=content_hash,
            cache_member_signature=signature(destination), cache_bytes=destination.stat().st_size,
            cache_source_deadline_seconds=data['deadline_seconds'], events=len(data['decisions']),
            episode=data['metadata']['episode'], population=data['metadata']['total_vehicle_count'],
            windows=len(data['action']), completed=bool(data['completed']),
            canonical=True, reuse_existing_cache=False, preservation=preservation,
            compact_array_bytes_verified=True, **logs)
    saved.update(plan_identity=plan_identity, census_row=row, conversion_seconds=time.monotonic()-started)
    atomic_json(receipt, saved)
    return saved



def assert_batch_equal(left, right):
    import torch
    if torch.is_tensor(left):
        if not torch.equal(left, right): raise ValueError('batch tensor mismatch')
    elif isinstance(left, np.ndarray):
        if left.dtype!=right.dtype or left.shape!=right.shape or left.tobytes()!=right.tobytes(): raise ValueError('batch array mismatch')
    elif dataclasses.is_dataclass(left):
        for field in dataclasses.fields(left): assert_batch_equal(getattr(left,field.name),getattr(right,field.name))
    elif isinstance(left, dict):
        if left.keys()!=right.keys(): raise ValueError('batch mapping mismatch')
        for key in left: assert_batch_equal(left[key],right[key])
    elif isinstance(left,(tuple,list)) or type(left).__name__=='LazyBoundaries':
        if len(left)!=len(right): raise ValueError('batch sequence length mismatch')
        for a,b in zip(left,right): assert_batch_equal(a,b)
    elif left!=right: raise ValueError('batch scalar mismatch')


@contextmanager
def forbid_raw_access(paths):
    forbidden={str(Path(p).absolute()) for p in paths}
    original_open,original_stat=Path.open,Path.stat
    def guarded_open(path,*args,**kwargs):
        if str(path.absolute()) in forbidden: raise AssertionError(f'raw access attempted: {path}')
        return original_open(path,*args,**kwargs)
    def guarded_stat(path,*args,**kwargs):
        if str(path.absolute()) in forbidden: raise AssertionError(f'raw stat attempted: {path}')
        return original_stat(path,*args,**kwargs)
    with patch.object(Path,'open',guarded_open),patch.object(Path,'stat',guarded_stat): yield


def verify_batches(manifest, output, city):
    from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
    valid=validator(city)
    selected=[]
    for reused in (False,True):
        candidate=next((e for e in manifest['entries'] if e['reuse_existing_cache']==reused),None)
        if candidate: selected.append(candidate)
    reports=[]
    directory=output/'batch_checks';directory.mkdir(exist_ok=True)
    raw_paths=[e['path'] for e in manifest['entries']]
    for index,entry in enumerate(selected):
        baseline=None
        starts=sorted({0,max(0,entry['windows']-8)})
        for mode in ('raw_required','compact_only_v1'):
            test=dict(schema_version=1,status='complete',contract=manifest['contract'],
                source_storage=mode,sources=[entry],entries=[entry],duplicate_files=0)
            test['identity']=_identity(test)
            path=directory/f'{index}_{mode}.json';atomic_json(path,test)
            guard=forbid_raw_access(raw_paths) if mode=='compact_only_v1' else nullcontext()
            with guard:
                dataset=CompactPretrainDataset(valid.config,valid.prepared,valid.topology,8,
                    cache_manifest_path=path,split=entry['split'])
                batch=dataset.batch_from_starts(0,starts)
                if baseline is None: baseline=batch
                else: assert_batch_equal(baseline,batch)
        reports.append(dict(trajectory_hash=entry['trajectory_hash'],reused=entry['reuse_existing_cache'],
            starts=starts,sequence_length=8,raw_required_equals_compact_only=True,
            raw_open_and_stat_forbidden_during_init_and_batch=True))
    return reports


def publish_verification(manifest, output, city):
    reports=verify_batches(manifest,output,city)
    receipts=[]
    for entry in manifest['entries']:
        key=_identity({'census_path':entry['census_row']['path']})
        path=output/'receipts'/f'{key}.json'
        receipts.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            trajectory_hash=entry['trajectory_hash'],cache_path=entry['cache_path'],
            cache_content_hash=entry['cache_content_hash'],cache_member_signature=entry['cache_member_signature'],
            summary_log_path=entry['summary_log_path'],summary_json_sha256=entry['summary_json_sha256']))
    verification=dict(status='passed',city=city,manifest_identity=manifest['identity'],
        manifest_sha256=hashlib.sha256((output/'cache_manifest.json').read_bytes()).hexdigest(),
        all_entries_content_hash_verified=True,all_entries_member_signature_verified=True,
        complete_summary_json_preserved=True,real_batch_verified=bool(reports),raw_unavailable_verified=bool(reports),
        raw_unavailable_method='Path.open and Path.stat for every raw provenance path raise during compact-only Dataset init and full batch materialization; real raw archives untouched',
        batch_checks=reports,receipts=receipts)
    atomic_json(output/'verification.json',verification)
    return verification


def verify_manifest_file(output, city):
    publish_verification(json.loads((output/'cache_manifest.json').read_text()),output,city)


def build_city(census, output, city, *, workers=4, limit=0, stop_event=None):
    if not 1<=workers<=16: raise ValueError('workers must be between one and sixteen')
    if stop_event is not None and stop_event.is_set(): raise CancelledError('another city failed')
    rows=[json.loads(line) for line in (census/f'{city}_canonical.jsonl').open()]
    if limit:
        # Pilot should include one new raw and one reusable cache whenever available.
        rows=([next(r for r in rows if not r.get('reusable_cache'))]+[next(r for r in rows if r.get('reusable_cache'))])[:limit] if limit<=2 else rows[:limit]
    output.mkdir(parents=True,exist_ok=True)
    for name in ('episodes','receipts','logs'): (output/name).mkdir(exist_ok=True)
    contract = dict(json.loads((census/'target_contracts.json').read_text())[city], city=city)
    contract.pop('edge_ids',None)
    plan=dict(city=city, census_sha256=hashlib.sha256((census/f'{city}_canonical.jsonl').read_bytes()).hexdigest(), rows=rows, limit=limit,contract=contract)
    plan['identity']=_identity(plan)
    path=output/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan: raise ValueError('frozen conversion plan changed')
    atomic_json(path,plan)
    results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_process) as pool:
        iterator=iter(rows)
        futures={}
        def submit():
            row=next(iterator,None)
            if row is not None:
                futures[pool.submit(convert_one,row,output,plan['identity'])]=row
        for _ in range(workers): submit()
        while futures:
            if stop_event is not None and stop_event.is_set():
                for pending in futures: pending.cancel()
                raise CancelledError('another city failed; completed receipts retained')
            done,_=wait(futures,return_when=FIRST_COMPLETED,timeout=1.)
            for future in done:
                row=futures.pop(future)
                try: results.append(future.result())
                except Exception as exc:
                    for pending in futures: pending.cancel()
                    atomic_json(output/'progress.json',dict(status='failed',completed=len(results),total=len(rows),error=str(exc),path=row['path']))
                    raise
                submit()
            atomic_json(output/'progress.json',dict(status='building',completed=len(results),total=len(rows)))
    if stop_event is not None and stop_event.is_set(): raise CancelledError('another city failed')
    results.sort(key=lambda r:r['path'])
    if len({r['trajectory_hash'] for r in results})!=len(results): raise ValueError('canonical trajectory duplicated')
    manifest=dict(schema_version=1,status='complete',source_storage='compact_only_v1',contract=contract,
        entries=results,sources=results,duplicate_files=0,plan_identity=plan['identity'],
        train_episodes=sum(r['split']=='train' for r in results),validation_episodes=sum(r['split']=='validation' for r in results))
    manifest['identity']=_identity(manifest)
    atomic_json(output/'cache_manifest.json',manifest)
    # Batch verification patches Path access guards: isolate it from other city threads.
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_process) as verifier:
        verifier.submit(verify_manifest_file,output,city).result()
    atomic_json(output/'progress.json',dict(status='complete',completed=len(results),total=len(rows),manifest_identity=manifest['identity']))
    return manifest


def corpus_summary(output, selected_cities):
    """Merge verified completed cities without changing their existing artifacts."""
    summaries={}
    for city in CITIES:
        manifest_path=output/city/'cache_manifest.json'
        verification_path=output/city/'verification.json'
        if not manifest_path.exists() or not verification_path.exists():
            if city in selected_cities: raise ValueError(f'selected city lacks completed verification: {city}')
            continue
        payload=manifest_path.read_bytes()
        manifest=json.loads(payload)
        verification=json.loads(verification_path.read_text())
        identity=_identity({k:v for k,v in manifest.items() if k!='identity'})
        if (manifest.get('status')!='complete' or manifest.get('identity')!=identity
                or manifest.get('source_storage')!='compact_only_v1'
                or manifest.get('contract',{}).get('city')!=city):
            raise ValueError(f'invalid completed city manifest: {city}')
        if (verification.get('status')!='passed' or verification.get('city')!=city
                or verification.get('manifest_identity')!=identity
                or verification.get('manifest_sha256')!=hashlib.sha256(payload).hexdigest()
                or verification.get('real_batch_verified') is not True
                or verification.get('raw_unavailable_verified') is not True):
            raise ValueError(f'invalid completed city verification: {city}')
        entries=manifest['entries']
        summaries[city]=dict(entries=len(entries),newly_compacted=sum(not e['reuse_existing_cache'] for e in entries),
            reused=sum(e['reuse_existing_cache'] for e in entries),manifest_identity=identity)
    return dict(status='complete' if len(summaries)==len(CITIES) else 'selected_cities_complete',
                cities=summaries,raw_sources_deleted=False)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census-dir',type=Path,default=ROOT/'outputs/pretrain_data_audit_20261016/total_census')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/pretrain_storage_migration_20261016/corpus')
    parser.add_argument('--workers',type=int,choices=range(1,17),default=4)
    parser.add_argument('--cities',nargs='+',choices=CITIES,default=list(CITIES))
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--parallel-cities',action='store_true',help='run selected cities concurrently; workers are per city')
    args=parser.parse_args(argv)
    if args.limit<0: parser.error('limit must be nonnegative')
    if len(set(args.cities))!=len(args.cities): parser.error('cities must not repeat')
    if sys.flags.optimize: parser.error('optimized Python disables preservation assertions')
    return args


def run_selected_cities(args):
    # Environment changes happen once, before any coordinator threads/spawns.
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    stop_event=threading.Event()
    atomic_json(args.output/'summary.json',dict(status='running',selected_cities=args.cities,raw_sources_deleted=False))
    def run(city):
        result=build_city(args.census_dir,args.output/city,city,workers=args.workers,limit=args.limit,stop_event=stop_event)
        print(json.dumps(dict(city=city,entries=len(result['entries']),status='complete')),flush=True)
    try:
        if args.parallel_cities and len(args.cities)>1:
            with ThreadPoolExecutor(max_workers=len(args.cities)) as coordinator:
                futures={coordinator.submit(run,city):city for city in args.cities}
                try:
                    for future in as_completed(futures): future.result()
                except BaseException:
                    stop_event.set()
                    for future in futures: future.cancel()
                    raise
        else:
            for city in args.cities: run(city)
        summary=corpus_summary(args.output,args.cities)
        atomic_json(args.output/'summary.json',summary)
        return summary
    except BaseException as exc:
        stop_event.set()
        atomic_json(args.output/'summary.json',dict(status='failed',selected_cities=args.cities,
            error=f'{type(exc).__name__}: {exc}',raw_sources_deleted=False))
        raise


def main(argv=None):
    args=parse_args(argv)
    if json.loads((args.census_dir/'summary.json').read_text()).get('status')!='complete': raise ValueError('census incomplete')
    os.nice(10)
    import torch
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'build.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_selected_cities(args)


if __name__=='__main__': main()
