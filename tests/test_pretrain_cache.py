from dataclasses import replace
from pathlib import Path
import json
import zipfile

import numpy as np
import pytest
import torch

from test_offline_data import setup, write_episode
from tools import build_pretrain_cache
from urban_dreamer.learning.cohort_targets import prepare_cohort_batch,prepare_transition_targets
from urban_dreamer.learning.episode_archive import to_replay_episode
from urban_dreamer.learning.offline_data import OfflineEpisodeDataset, _identity, archive_content_hash
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset, boundary_vehicles, compact_arrays, compact_transition_targets, read_hash_source
from urban_dreamer.learning.replay import ReplayBuffer
from urban_dreamer.learning.replay import SequenceBatch
from urban_dreamer.learning.action_events import ActionEvent,pack_action_events


def make_cache(setup, tmp_path, monkeypatch, *, indexes=(0, 1),windows=3):
    config, prepared, topology = setup
    topology.network.capacities = np.ones(3, dtype=np.float32)
    validator = object.__new__(OfflineEpisodeDataset)
    validator.config, validator.prepared, validator.topology, validator._explicit = config, prepared, topology, True
    monkeypatch.setattr(build_pretrain_cache, '_VALIDATOR', validator)
    output = tmp_path / 'compact'
    (output / 'episodes').mkdir(parents=True)
    results = []
    for index in indexes:
        path = write_episode(setup, index,windows=windows)
        data, identity = read_hash_source(path)
        row = dict(path=str(path.resolve()), crc_trajectory_candidate=identity['census_trajectory_signature'],
                   demand_hash=None, demand_group=f'demand{index}', seed=index, episode=index,
                   method='random', root=str(config.data.archive_roots[0]), split='train')
        results.append(build_pretrain_cache.build_one(row, output, True, identity['trajectory_hash']))
    physical = {key: config.resolved_snapshot()['simulation'][key] for key in (
        'vehicle_max_speed_mps', 'incident_edge', 'incident_start_seconds', 'incident_end_seconds',
        'incident_speed_mps', 'sumo_extra_args', 'demand_duration_seconds')}
    manifest = dict(schema_version=1, status='complete', contract=dict(city=config.inputs.network_name,
        input_hashes=prepared.input_hashes, physical=physical), entries=results, sources=results,
        duplicate_files=0)
    manifest['identity'] = _identity(manifest)
    path = output / 'cache_manifest.json'
    path.write_text(json.dumps(manifest))
    return CompactPretrainDataset(config, prepared, topology, 2, cache_manifest_path=path), results


def test_single_pass_source_hashes_equal_canonical_hashes(setup):
    path = write_episode(setup)
    data, identity = read_hash_source(path)
    assert identity['content_hash'] == archive_content_hash(path)
    assert identity['trajectory_hash'] == archive_content_hash(path, trajectory_only=True)
    assert len(data['states']) == 4
    assert data['decisions'][0]['destination_edge'] == 2
    assert data['decisions'][0]['event_index'] == 0
    packed = compact_arrays(data)
    assert packed['boundary_offsets'].tolist() == [0, 0, 1, 2, 3]
    assert packed['active_vehicle'].tolist() == [0, 0, 0]
    assert packed['event_vehicle'].shape == (6,)
    assert all(array.dtype != object for array in packed.values())


def test_decoder_following_labels_preserve_single_window_coverage_and_mask_last_boundary(setup, tmp_path, monkeypatch):
    compact, _ = make_cache(setup, tmp_path, monkeypatch, indexes=(0,), windows=3)
    compact.sequence_length = 1
    compact.config = replace(compact.config,
        codec=replace(compact.config.codec, decoder_pretrain_enabled=True))
    batch = compact.batch_from_starts(0, [0, 1, 2], fm_only=True)
    assert batch.flow.shape[1] == 2
    assert batch.action.shape[1] == batch.action_events.time_steps == 1
    assert batch.decoder_following_valid.tolist() == [[True], [True], [False]]
    following = torch.tensor(compact._episode(0)['flow'])
    torch.testing.assert_close(batch.decoder_following_flow[:2, 0], following[[2, 3]])
    assert batch.decoder_following_flow[2].eq(0).all()  # Placeholder excluded by validity mask.
    compact.config = replace(compact.config,
        codec=replace(compact.config.codec, decoder_pretrain_enabled=False))
    legacy = compact.batch_from_starts(0, [0, 1, 2], fm_only=True)
    assert legacy.decoder_following_flow is legacy.decoder_following_valid is None
    for name in ('flow', 'action', 'transition_target_flow', 'transition_cohort_count'):
        torch.testing.assert_close(getattr(batch, name), getattr(legacy, name))


def test_recursive_epoch_keeps_arrival_and_vehicle_metadata(setup, tmp_path, monkeypatch):
    compact, _ = make_cache(setup, tmp_path, monkeypatch, indexes=(0,), windows=4)
    compact.config = replace(compact.config,
        method=replace(compact.config.method, fm_training_mode='recursive_3step', supervision_horizon=3))
    compact.sequence_length = 3
    batch = next(compact.iter_epoch(1, np.random.default_rng(3)))
    assert batch.flow.shape[1] == 4
    assert batch.arrival_times is not None
    assert batch.boundary_vehicles is not None
    cohort = prepare_cohort_batch(batch, capacities=torch.ones(3))
    assert cohort.transition_flow.shape[1] == 4
    assert torch.isfinite(cohort.transition_flow).all()


def test_compact_batch_equals_raw_replay_and_refreshed_window_targets(setup, tmp_path, monkeypatch):
    compact, entries = make_cache(setup, tmp_path, monkeypatch, indexes=(0,))
    episode = to_replay_episode(Path(entries[0]['path']), expected_session_protocol='event',
                               target_deadline_seconds=compact.target_deadline_seconds)
    replay = ReplayBuffer(2)
    replay.append_episode(episode['transitions'], total_travel_minutes=episode['total_travel_minutes'],
                          total_vehicle_count=episode['total_vehicle_count'], arrival_times=episode['arrival_times'])
    raw = replay._batch_from_picks([(0, 1)])
    cached = compact.batch_from_starts(0, [1])
    for name in ('flow', 'action', 'departure', 'context', 'active_vehicle_fraction', 'time',
                 'terminal', 'episode_complete', 'total_vehicle_count', 'total_travel_minutes'):
        assert torch.equal(getattr(raw, name), getattr(cached, name)), name
    assert raw.action_events.vehicle_ids == cached.action_events.vehicle_ids
    assert torch.equal(raw.action_events.phase, cached.action_events.phase)
    assert raw.arrival_times == cached.arrival_times
    original_cohort = prepare_transition_targets(raw, capacities=torch.ones(3))
    cached_cohort = prepare_transition_targets(cached, capacities=torch.ones(3))
    reference_cached = prepare_transition_targets(replace(cached,transition_volume_scope=None),capacities=torch.ones(3))
    assert cached_cohort is cached
    assert torch.equal(original_cohort.transition_target_flow, cached_cohort.transition_target_flow)
    for name in ('transition_target_flow','transition_cohort_count'):
        assert torch.allclose(getattr(reference_cached,name),getattr(cached,name)),name
    assert cached.transition_flow is None
    assert cached.transition_action is None
    assert cached.transition_action_events is None
    assert cached.transition_active_vehicle_fraction is None
    assert cached.cohort_count is None
    assert cached.transition_volume_scope=='previous_state_excluding_current_births'


def test_codec_stream_visits_real_frames_once_without_vehicle_decode_or_fm_sampler(setup, tmp_path, monkeypatch):
    compact, _ = make_cache(setup, tmp_path, monkeypatch)
    before = compact.state_dict()
    batches = list(compact.iter_codec_batches(4))
    assert [len(current) for current, _ in batches] == [4, 2]
    assert torch.cat([current[:, 0, 0] for current, _ in batches]).tolist() == [0, 30, 60, 0, 30, 60]
    assert all(torch.equal(following[:, 0, 0]-current[:, 0, 0], torch.full((len(current),), 30.))
               for current, following in batches)
    assert compact.state_dict() == before
    assert all(set(arrays) == {'flow', '_cache_stamp'} for arrays in compact._cache.values())
    # FM batches can subsequently request CSR/event members from the same cache.
    assert compact.batch_from_starts(0, [0]).boundary_vehicles is not None


def test_cache_epoch_restore_recompression_and_mutation_detection(setup, tmp_path, monkeypatch):
    compact, entries = make_cache(setup, tmp_path, monkeypatch)
    rng = np.random.default_rng(4)
    stream = compact.iter_epoch(1, rng)
    next(stream)
    state, rng_state = compact.state_dict(), rng.bit_generator.state
    expected = [(b.episode.tolist(), b.start_index.tolist()) for b in stream]
    original = Path(entries[0]['path'])
    with zipfile.ZipFile(original) as archive:
        members = [(name, archive.read(name)) for name in archive.namelist()]
    with zipfile.ZipFile(original, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, content in reversed(members):
            archive.writestr(name, content)
    compact.verify_integrity()
    restored = CompactPretrainDataset(*setup, 2, cache_manifest_path=compact.cache_manifest_path)
    restored.load_state_dict(state)
    rng.bit_generator.state = rng_state
    assert [(b.episode.tolist(), b.start_index.tolist()) for b in restored.iter_epoch(1, rng)] == expected
    cache = Path(entries[0]['cache_path'])
    with np.load(cache, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['flow'][0, 0, 0] = 99.
    np.savez_compressed(cache, **arrays)
    with pytest.raises(ValueError, match='compact episode content changed'):
        restored.batch_from_starts(0, [0])


def test_compact_membership_excludes_internal_roads_and_preserves_observed_arrival(setup):
    data, _ = read_hash_source(write_episode(setup))
    data['states'][2]['vehicles'][0].update(in_junction=True, junction_connection=[0, 1],
        junction_remaining_distance=3., junction_speed=4.)
    data['states'][-1]['vehicles'][0].update(arrived=True, arrival_time=74.)
    data['states'][-1]['done'] = True
    data['completed'] = True
    data['travel_times'] = {'v0': 74.}
    data['terminated'][-1], data['truncated'][-1] = True, False
    arrays = compact_arrays(data)
    assert arrays['active_external_edge'].tolist() == [0, -1]
    assert boundary_vehicles(arrays, 2)[0]['in_junction']
    assert boundary_vehicles(arrays, 2)[0]['junction_remaining_distance'] == 3.
    assert boundary_vehicles(arrays, 3) == ()
    assert arrays['arrival_time'].tolist() == [74.]  # Never round up to90s boundary.


def test_cache_plan_reserves_all_heldout_demand_groups_across_methods(tmp_path):
    census, pilot = tmp_path / 'census', tmp_path / 'pilot'
    census.mkdir();pilot.mkdir()
    paths = [str(tmp_path / f'episode{i}.npz') for i in range(4)]
    rows = [dict(path=path, metadata_compatible=True, demand_hash='g0' if i < 2 else 'g1',
                 demand_group='g0' if i < 2 else 'g1') for i, path in enumerate(paths)]
    (census/'file_rows.json').write_text(json.dumps({'rows':rows}))
    (census/'archive_paths.json').write_text(json.dumps(paths))
    (census/'census.json').write_text(json.dumps(dict(frozen_at='frozen',unique_trajectories=3,
        verified_duplicate_groups=[dict(files=paths[:2],trajectory_hash='same')])) )
    (pilot/'split.json').write_text(json.dumps({'holdout_groups':['g1']}))
    config = Path(__file__).parents[1] / 'configs/toy_cross_bottleneck.toml'
    plan = build_pretrain_cache.make_plan(census,pilot,config)
    assert [row['split'] for row in plan['sources']] == ['train','train','validation','validation']
    assert plan['canonical'][paths[1]] == paths[0]
    assert plan['expected_unique_trajectories'] == 3


def test_cohort_codec_stream_samples_valid_forward_anchors_without_fm_progress(setup,tmp_path,monkeypatch):
    compact,_=make_cache(setup,tmp_path,monkeypatch,indexes=(0,))
    before=compact.state_dict()
    batches=list(compact.iter_cohort_codec_batches(3))
    # Shortfixture T2, two legalstart anchors. Each contributes current0/1 and
    # following1/2, never the source's lastboundary as a current codec input.
    assert [len(current) for current,_ in batches]==[3,1]
    current=torch.cat([pair[0] for pair in batches]);following=torch.cat([pair[1] for pair in batches])
    assert current[:,0,0].tolist()==[0.,30.,30.,60.]
    assert torch.equal(following[:,0,0]-current[:,0,0],torch.full((4,),30.))
    # C0 empty atstart0, but containsv0 atstart1: same observed futuretraffic
    # receives distinct valid labels depending on its fixed start cohort.
    assert current[:2,:,2].sum()==0
    assert current[2:,:,2].sum()==2
    assert compact.state_dict()==before
    assert compact.summary['cached_episodes']<=2


def test_completed_cache_rebuild_is_idempotent_and_keeps_manifest_identity(setup,tmp_path,monkeypatch):
    compact,entries=make_cache(setup,tmp_path,monkeypatch,indexes=(0,))
    plan={'identity':'fixed-plan'}
    path=compact.cache_manifest_path
    manifest=json.loads(path.read_text())
    manifest.update(plan_identity=plan['identity'],train_episodes=1,validation_episodes=0)
    manifest['identity']=_identity({key:value for key,value in manifest.items() if key!='identity'})
    path.write_text(json.dumps(manifest))
    before=path.read_bytes()
    restored=build_pretrain_cache.build_cache(plan,path.parent,workers=1)
    assert restored['identity']==manifest['identity']
    assert path.read_bytes()==before


def test_cached_newborn_enters_next_window_input_and_membership(setup,tmp_path,monkeypatch):
    compact,_=make_cache(setup,tmp_path,monkeypatch,indexes=(0,))
    batch=compact.batch_from_starts(0,[0])
    # v0 is unborn at0, active at30 and60. Only the first endpoint excludes it.
    assert batch.flow[0,1,:,2].sum()>0
    assert batch.transition_target_flow[0,0,:,2].sum()==0
    assert batch.transition_target_flow[0,1,:,2].sum()==1
    assert batch.transition_cohort_count.tolist()==[[0,1]]
    assert batch.action.sum()==4  # Bothwindow's raw birth/action events remain.
    assert len(batch.action_events)==4
    assert batch.active_vehicle_fraction.sum()==2
    assert prepare_transition_targets(batch,capacities=torch.ones(3)) is batch


def test_transition_codec_stream_uses_endpoint_and_real_following_speed(setup,tmp_path,monkeypatch):
    compact,_=make_cache(setup,tmp_path,monkeypatch,indexes=(0,))
    before=compact.state_dict()
    batches=list(compact.iter_transition_codec_batches(2))
    assert [len(current) for current,_ in batches]==[2,1]
    current=torch.cat([current for current,_ in batches])
    following=torch.cat([following for _,following in batches])
    assert current[:,0,0].tolist()==[30.,60.,60.]
    assert following[:,0,0].tolist()==[60.,90.,90.]
    assert current[:, :,2].sum(-1).tolist()==[0.,1.,1.]
    assert compact.state_dict()==before


def test_transition_target_contract_rejects_prior_fixed_sequence_sampler_identity(setup,tmp_path,monkeypatch):
    compact,_=make_cache(setup,tmp_path,monkeypatch,indexes=(0,))
    old={**compact.manifest,'contract':dict(compact.manifest['contract'])}
    old['contract'].pop('transition_target_scope')
    old.pop('identity')
    state=compact.state_dict()
    state['manifest_identity']=_identity(old)
    with pytest.raises(ValueError,match='state/manifest mismatch'):
        compact.load_state_dict(state)


def test_vector_transition_targets_refresh_births_and_handle_arrival_internal_edge():
    capacity=np.asarray([10.,20.,30.],dtype=np.float32)
    def vehicle(name,edge,internal=False):
        return dict(vehicle_id=name,current_edge=edge,departed=True,arrived=False,in_junction=internal)
    boundaries=((vehicle('A',0),),(vehicle('A',1),vehicle('B',0)),
                (vehicle('B',2),vehicle('C',1)),(vehicle('B',2,True),vehicle('C',1)))
    flow=np.zeros((4,3,8),dtype=np.float32)
    flow[...,1]=np.arange(4)[:,None]*.1+.5
    flow[...,4:]=1.
    for frame,snapshots in enumerate(boundaries):
        for row in snapshots:
            if not row['in_junction']:
                flow[frame,row['current_edge'],2]+=1/capacity[row['current_edge']]
    before=flow.copy()
    arrays=dict(flow=flow,metadata={'population':3},time=np.arange(4,dtype=np.float64)*30,
        boundary_offsets=np.asarray([0,1,3,5,7]),active_vehicle=np.asarray([0,0,1,1,2,1,2]),
        active_current_edge=np.asarray([0,1,0,2,1,2,1]),
        active_in_junction=np.asarray([False,False,False,False,False,True,False]),
        arrival_vehicle=np.asarray([0]),arrival_time=np.asarray([45.]))
    target,counts=compact_transition_targets(arrays,[0],3,capacity)
    assert np.array_equal(arrays['flow'],before)
    np.testing.assert_array_equal(counts,[[1,2,2]])
    np.testing.assert_allclose(target[0,:, :,2]*capacity,[[0,1,0],[0,0,1],[0,1,0]])
    np.testing.assert_array_equal(target[..., [0,1,3,4,5,6,7]],flow[None,1:,...][..., [0,1,3,4,5,6,7]])
    events=pack_action_events([[(ActionEvent('B',0,10.,0,1,2),),(),()]])
    action=torch.zeros(1,3,3);action[0,0,1]=1/3
    raw=SequenceBatch(flow=torch.from_numpy(flow[None]),action=action,departure=torch.zeros(1,3,3),
        context=torch.zeros(1,3,2),active_vehicle_fraction=torch.tensor([[.5,.6,.6]]),
        vehicles=(boundaries[0],),time=torch.tensor([0.],dtype=torch.float64),terminal=torch.zeros(1,3,dtype=torch.bool),
        total_travel_minutes=torch.zeros(1),episode_complete=torch.zeros(1,dtype=torch.bool),episode=torch.zeros(1,dtype=torch.long),
        start_index=torch.zeros(1,dtype=torch.long),total_vehicle_count=torch.tensor([3]),
        action_events=events,boundary_vehicles=(boundaries,),arrival_times=({'A':45.},))
    reference=prepare_transition_targets(raw,capacities=capacity)
    torch.testing.assert_close(reference.transition_target_flow,torch.from_numpy(target))
    assert torch.equal(reference.transition_cohort_count,torch.from_numpy(counts))
    assert reference.flow is raw.flow
    assert reference.action_events is raw.action_events
    assert reference.active_vehicle_fraction is raw.active_vehicle_fraction


def test_fm_prefix_is_exact_and_does_not_decode_snapshots(setup,tmp_path,monkeypatch):
    from urban_dreamer.learning import pretrain_cache as module
    base,_=make_cache(setup,tmp_path,monkeypatch,indexes=(0,),windows=20)
    data=CompactPretrainDataset(*setup,16,cache_manifest_path=base.cache_manifest_path)
    full=data.batch_from_starts(0,[0,2])
    calls=[];original=module.boundary_vehicles
    def tracked(arrays,frame):
        calls.append(frame)
        return original(arrays,frame)
    monkeypatch.setattr(module,'boundary_vehicles',tracked)
    prefix=data.batch_from_starts(0,[0,2],materialization_horizon=3,fm_only=True)
    assert calls==[]
    assert prefix.vehicles==((),())
    assert prefix.arrival_times is None
    assert all(view.seed is None for view in prefix.boundary_vehicles)
    for name in ('action','departure','context','active_vehicle_fraction','terminal',
                 'transition_target_flow','transition_cohort_count'):
        assert torch.equal(getattr(prefix,name),getattr(full,name)[:,:3]),name
    assert torch.equal(prefix.flow,full.flow[:,:4])
    expected=full.action_events.select(full.action_events.time_index<3)
    for name in ('batch_index','time_index','phase','current_edge','next_edge','destination_edge','event_index'):
        assert torch.equal(getattr(prefix.action_events,name),getattr(expected,name)),name
    assert prefix.action_events.vehicle_ids==expected.vehicle_ids
    assert prepare_transition_targets(prefix,capacities=torch.ones(3)) is prefix
    assert calls==[]
    assert prefix.boundary_vehicles[0][0]==()
    assert calls==[0]


def test_prefix_epoch_keeps_full_sequence_starts_order_rng_and_contract(setup,tmp_path,monkeypatch):
    base,_=make_cache(setup,tmp_path,monkeypatch,windows=20)
    data=CompactPretrainDataset(*setup,16,cache_manifest_path=base.cache_manifest_path)
    # Prefix-only metadata elision belongs to independent training; recursion
    # intentionally keeps vehicle membership and observed arrival metadata.
    data.config=replace(data.config,method=replace(data.config.method,fm_training_mode='legacy_independent'))
    identity=data.manifest['identity'];expected=[];reference_rng=np.random.default_rng(5)
    for episode in reference_rng.permutation(2):
        expected.extend((int(episode),int(start)) for start in reference_rng.permutation(5))
    actual=[];rng=np.random.default_rng(5)
    for batch in data.iter_epoch(2,rng):
        assert batch.flow.shape[1]==4
        assert batch.action.shape[1]==3
        assert batch.vehicles==tuple(() for _ in batch.episode)
        actual.extend((int(e),int(s)) for e,s in zip(batch.episode,batch.start_index))
    assert actual==expected
    assert data.sequence_length==16
    assert data.manifest['identity']==identity
    assert data.epochs_completed==1
    assert rng.bit_generator.state==reference_rng.bit_generator.state


def test_resident_cache_hashes_only_on_load_or_changed_stamp(setup,tmp_path,monkeypatch):
    from urban_dreamer.learning import pretrain_cache as module
    data,entries=make_cache(setup,tmp_path,monkeypatch,indexes=(0,))
    cache=Path(entries[0]['cache_path']);calls=[];original=module.archive_content_hash
    def tracked(path,**kwargs):
        calls.append(Path(path))
        return original(path,**kwargs)
    monkeypatch.setattr(module,'archive_content_hash',tracked)
    data.batch_from_starts(0,[0],materialization_horizon=1,fm_only=True)
    data.batch_from_starts(0,[1],materialization_horizon=1,fm_only=True)
    assert calls.count(cache)==1
    with zipfile.ZipFile(cache) as archive:
        members=[(name,archive.read(name)) for name in archive.namelist()]
    with zipfile.ZipFile(cache,'w',compression=zipfile.ZIP_STORED) as archive:
        for name,content in reversed(members):
            archive.writestr(name,content)
    data.batch_from_starts(0,[0],fm_only=True)
    data.batch_from_starts(0,[1],fm_only=True)
    assert calls.count(cache)==2
    with np.load(cache,allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    arrays['flow'][0,0,0]=999.
    np.savez_compressed(cache,**arrays)
    with pytest.raises(ValueError,match='compact episode content changed'):
        data.batch_from_starts(0,[0],fm_only=True)
