import gzip
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from test_offline_data import setup, write_episode
from tools import build_standalone_corpus as builder
from urban_dreamer.learning.offline_data import OfflineEpisodeDataset, _identity
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset, read_hash_source


def test_exact_lzma_roundtrip_and_tamper(tmp_path):
    path=tmp_path/'x.npz'
    arrays={'a':np.array([0.,-0.,np.nan],dtype=np.float64),'s':np.array(['车','x']),'empty':np.zeros((0,3),np.int32)}
    builder.write_lzma(path,arrays)
    with zipfile.ZipFile(path) as z:
        assert all(m.compress_type==zipfile.ZIP_LZMA for m in z.infolist())
    entry={'cache_path':str(path),'cache_content_hash':builder.archive_content_hash(path),'cache_member_signature':builder.signature(path)}
    builder.verify_cache(entry)
    builder.write_lzma(path,dict(arrays,a=np.ones(3)))
    with pytest.raises(ValueError,match='verification failed'): builder.verify_cache(entry)


def test_source_retirement_batch_still_works_and_resume_checks_cache(setup,tmp_path,monkeypatch):
    config,prepared,topology=setup
    topology.network.capacities=np.ones(3,dtype=np.float32)
    source=write_episode(setup,windows=3)
    data,identity=read_hash_source(source)
    valid=object.__new__(OfflineEpisodeDataset)
    valid.config,valid.prepared,valid.topology,valid._explicit=config,prepared,topology,True
    monkeypatch.setattr(builder,'validator',lambda _:valid)
    output=tmp_path/'out';output.mkdir()
    for part in ('episodes','receipts','logs'): (output/part).mkdir()
    row=dict(path=str(source),city=config.inputs.network_name,split='train',demand_group=None,
        crc_trajectory_candidate=identity['census_trajectory_signature'],source_member_signature=identity['source_member_signature'])
    result=builder.convert_one(row,output,'test-plan',min_free_bytes=0)
    assert result['compact_array_bytes_verified']
    summary=json.loads(gzip.decompress(Path(result['summary_log_path']).read_bytes()))
    assert summary['metadata']==data['metadata']
    physical={k:config.resolved_snapshot()['simulation'][k] for k in ('vehicle_max_speed_mps','incident_edge','incident_start_seconds','incident_end_seconds','incident_speed_mps','sumo_extra_args','demand_duration_seconds')}
    manifest=dict(schema_version=1,status='complete',source_storage='compact_only_v1',contract=dict(city=config.inputs.network_name,input_hashes=prepared.input_hashes,physical=physical),entries=[result],sources=[result],duplicate_files=0)
    manifest['identity']=_identity(manifest)
    path=output/'cache_manifest.json';path.write_text(json.dumps(manifest))
    dataset=CompactPretrainDataset(config,prepared,topology,2,cache_manifest_path=path)
    before=dataset.batch_from_starts(0,[0])
    source.unlink() # Synthetic fixture only; production tool never deletes raw sources.
    restored=CompactPretrainDataset(config,prepared,topology,2,cache_manifest_path=path)
    after=restored.batch_from_starts(0,[0])
    assert np.array_equal(before.flow,after.flow)
    assert builder.convert_one(row,output,'test-plan',min_free_bytes=0)==result
    with pytest.raises(ValueError,match='plan mismatch'): builder.convert_one(row,output,'different',min_free_bytes=0)
    Path(result['cache_path']).write_bytes(b'corrupted cache')
    with pytest.raises((ValueError,zipfile.BadZipFile)):
        builder.convert_one(row,output,'test-plan',min_free_bytes=0)


def test_census_change_refused(setup,tmp_path):
    source=write_episode(setup)
    output=tmp_path/'out';(output/'receipts').mkdir(parents=True)
    row=dict(path=str(source),source_member_signature='0'*64)
    with pytest.raises(ValueError,match='census source signature mismatch'): builder.convert_one(row,output,'plan',min_free_bytes=0)


def test_guard_proves_source_open_and_stat_are_forbidden(tmp_path):
    path=tmp_path/'raw.npz';path.write_bytes(b'raw')
    with builder.forbid_raw_access([str(path)]):
        with pytest.raises(AssertionError,match='raw access'): path.open('rb')
        with pytest.raises(AssertionError,match='raw stat'): path.stat()
    assert path.read_bytes()==b'raw'


def test_reused_cache_restores_original_raw_provenance(setup,tmp_path,monkeypatch):
    from test_pretrain_cache import make_cache
    dataset,entries=make_cache(setup,tmp_path,monkeypatch,indexes=(0,))
    original=entries[0]
    row=dict(original,city=setup[0].inputs.network_name,reusable_cache=original['cache_path'],
        path=original['cache_path'],source_stamp=[1,2,3,4],cache_manifest=str(dataset.cache_manifest_path))
    output=tmp_path/'reused';output.mkdir()
    for part in ('episodes','receipts','logs'): (output/part).mkdir()
    result=builder.convert_one(row,output,'reuse-plan',min_free_bytes=0)
    assert result['path']==original['path']
    assert result['source_stamp']==original['source_stamp']
    assert result['cache_path']==original['cache_path']
    assert not list((output/'episodes').iterdir())
    assert result['census_row']['path']==original['cache_path']


def _reservation_child(directory, ready, release):
    with builder.reserve_disk(directory,60,min_free_bytes=100,ledger_dir=directory,free_bytes=200):
        ready.set()
        release.wait(10)


def test_multiprocess_reservation_denies_overcommit_and_releases(tmp_path):
    import multiprocessing
    context=multiprocessing.get_context('spawn')
    ready,release=context.Event(),context.Event()
    process=context.Process(target=_reservation_child,args=(str(tmp_path),ready,release))
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(RuntimeError,match='in-flight'):
            with builder.reserve_disk(tmp_path,60,min_free_bytes=100,ledger_dir=tmp_path,free_bytes=200,timeout=0): pass
        with builder.reserve_disk(tmp_path,40,min_free_bytes=100,ledger_dir=tmp_path,free_bytes=200,timeout=0):
            assert sum(r['bytes'] for r in json.loads((tmp_path/'disk_reservations.json').read_text()).values())==100
    finally:
        release.set();process.join(15)
        if process.is_alive(): process.terminate();process.join()
    assert process.exitcode==0
    assert json.loads((tmp_path/'disk_reservations.json').read_text())=={}


def test_reservation_exception_releases_and_stale_identity_recovers(tmp_path):
    import os
    ledger=tmp_path/'disk_reservations.json'
    ledger.write_text(json.dumps({'stale':dict(pid=os.getpid(),process_identity='wrong-start',bytes=10**12)}))
    with pytest.raises(ValueError,match='simulated'):
        with builder.reserve_disk(tmp_path,80,min_free_bytes=100,ledger_dir=tmp_path,free_bytes=200):
            assert 'stale' not in json.loads(ledger.read_text())
            raise ValueError('simulated compression failure')
    assert json.loads(ledger.read_text())=={}
    assert not list(tmp_path.glob('*.npz'))


def test_cli_workers_and_selected_cities():
    defaults=builder.parse_args([])
    assert defaults.cities==list(builder.CITIES) and defaults.workers==4
    args=builder.parse_args(['--workers','16','--cities','coulommiers','montereau_fault_yonne'])
    assert args.workers==16 and args.cities==list(builder.CITIES[1:])
    for argv in (['--workers','0'],['--workers','17'],['--cities','unknown'],['--cities','coulommiers','coulommiers']):
        with pytest.raises(SystemExit): builder.parse_args(argv)
    with pytest.raises(ValueError,match='sixteen'): builder.build_city(None,None,'coulommiers',workers=17)


def _completed_city(output,city):
    import hashlib
    folder=output/city;folder.mkdir()
    manifest=dict(status='complete',source_storage='compact_only_v1',contract={'city':city},
                  entries=[{'reuse_existing_cache':True},{'reuse_existing_cache':False}])
    manifest['identity']=_identity(manifest)
    path=folder/'cache_manifest.json';path.write_text(json.dumps(manifest))
    proof=dict(status='passed',city=city,manifest_identity=manifest['identity'],
               manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               real_batch_verified=True,raw_unavailable_verified=True)
    (folder/'verification.json').write_text(json.dumps(proof))
    return path


def test_summary_merges_prior_city_without_mutating_its_proof(tmp_path):
    old=_completed_city(tmp_path,builder.CITIES[0])
    before={p:p.read_bytes() for p in old.parent.iterdir()}
    for city in builder.CITIES[1:]: _completed_city(tmp_path,city)
    summary=builder.corpus_summary(tmp_path,builder.CITIES[1:])
    assert summary['status']=='complete' and set(summary['cities'])==set(builder.CITIES)
    assert summary['cities'][builder.CITIES[0]]['entries']==2
    assert all(p.read_bytes()==data for p,data in before.items())
    old.write_text(old.read_text()+' ')
    with pytest.raises(ValueError,match='verification'): builder.corpus_summary(tmp_path,builder.CITIES[1:])


def test_partial_summary_and_missing_selected_verification(tmp_path):
    _completed_city(tmp_path,'coulommiers')
    assert builder.corpus_summary(tmp_path,['coulommiers'])['status']=='selected_cities_complete'
    with pytest.raises(ValueError,match='lacks completed verification'):
        builder.corpus_summary(tmp_path,list(builder.CITIES))
    path=tmp_path/'coulommiers/cache_manifest.json'
    manifest=json.loads(path.read_text());manifest['entries']=[];path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='manifest'): builder.corpus_summary(tmp_path,['coulommiers'])


def test_parallel_cities_start_together_and_merge_existing_city(tmp_path,monkeypatch):
    import threading
    import os
    _completed_city(tmp_path,builder.CITIES[0])
    barrier=threading.Barrier(2)
    started=[]
    def fake_build(census,output,city,*,workers,limit,stop_event):
        assert workers==12 and not stop_event.is_set()
        assert os.environ['OPENBLAS_NUM_THREADS']=='1'
        started.append(city)
        barrier.wait(timeout=5)
        path=_completed_city(tmp_path,city)
        return json.loads(path.read_text())
    monkeypatch.setattr(builder,'build_city',fake_build)
    args=builder.parse_args(['--output',str(tmp_path),'--workers','12','--cities',*builder.CITIES[1:],'--parallel-cities'])
    summary=builder.run_selected_cities(args)
    assert set(started)==set(builder.CITIES[1:]) and summary['status']=='complete'
    assert set(summary['cities'])==set(builder.CITIES)


def test_parallel_failure_signals_other_city_and_never_publishes_complete(tmp_path,monkeypatch):
    import threading
    barrier=threading.Barrier(2)
    cancelled=threading.Event()
    (tmp_path/'summary.json').write_text('{"status":"complete"}')
    def fake_build(census,output,city,*,workers,limit,stop_event):
        barrier.wait(timeout=5)
        if city=='coulommiers': raise RuntimeError('simulated city failure')
        assert stop_event.wait(5)
        cancelled.set()
        raise builder.CancelledError('stopped before new submissions')
    monkeypatch.setattr(builder,'build_city',fake_build)
    monkeypatch.setattr(builder,'corpus_summary',lambda *args: pytest.fail('summary must not aggregate failed cities'))
    args=builder.parse_args(['--output',str(tmp_path),'--cities',*builder.CITIES[1:],'--parallel-cities'])
    with pytest.raises(RuntimeError,match='simulated city failure'): builder.run_selected_cities(args)
    assert cancelled.is_set()
    assert json.loads((tmp_path/'summary.json').read_text())['status']=='failed'
