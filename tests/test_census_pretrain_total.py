import hashlib
import json
import zipfile
import numpy as np
import pytest
from tools.census_pretrain_total import train_phase, stable_holdout, streaming_trajectory_hash
from urban_dreamer.learning.pretrain_cache import member_signature


def test_training_phase_does_not_admit_evaluation():
    assert train_phase({'collection_phase':'online'})
    assert train_phase({'phase':'train'})
    assert not train_phase({'phase':'evaluation','collection_phase':'online'})
    assert not train_phase({'phase':'train','collection_phase':'evaluation'})
    assert not train_phase({})


def test_split_preserves_existing_groups_and_append_stability():
    groups={str(i) for i in range(100)}
    forced={'1','3'}; trained={'0','2','4'}
    old=stable_holdout(groups,forced,trained)
    assert forced <= old
    assert not old & trained
    assert stable_holdout(groups|{'new'},forced,trained)&groups == old
    with pytest.raises(ValueError,match='conflict'):
        stable_holdout(groups,{'1'},{'1'})


def test_streaming_sha_matches_member_bytes_and_ignores_summary(tmp_path):
    a=tmp_path/'a.npz';b=tmp_path/'b.npz'
    np.savez(a, summary_json=np.frombuffer(b'{"first":true}',dtype=np.uint8), action=np.arange(12).reshape(3,4))
    np.savez_compressed(b, summary_json=np.frombuffer(b'{"second":true}',dtype=np.uint8), action=np.arange(12).reshape(3,4))
    with zipfile.ZipFile(a) as z:
        signature=member_signature(z.infolist(),trajectory_only=True)
        expected=hashlib.sha256()
        for m in sorted(z.infolist(),key=lambda m:m.filename):
            if m.filename!='summary_json.npy':
                expected.update(m.filename.encode()+b'\0');expected.update(z.read(m));expected.update(b'\0')
    assert streaming_trajectory_hash(a,signature)==expected.hexdigest()
    assert streaming_trajectory_hash(b,signature)==expected.hexdigest()
    with pytest.raises(ValueError,match='changed'):
        streaming_trajectory_hash(a,'wrong-signature')


def test_compatibility_uses_metadata_and_physics_not_routing_label():
    from tools.census_pretrain_total import compatible_summary, PHYSICAL
    from urban_dreamer.traffic.simulation_settings import physics_identity
    simulation={k:0 for k in PHYSICAL}
    simulation.update(sumo_extra_args=[],incident_edge='road',traffic_model='meso',routing_mode='next_hop')
    metadata=dict(city='city',phase='train',input_hashes={'net':'hash'},edge_ids=['a','b'],
        session_protocol='event',action_semantics='branch-only',step_seconds=1.,
        macro_step_seconds=30.,config={'simulation':simulation},demand_hash='demand',
        demand_law='independent_od_poisson')
    targets={'city':dict(input_hashes={'net':'hash'},edge_ids=['a','b'],
        physics_identity=physics_identity(simulation),physical={k:simulation[k] for k in PHYSICAL})}
    summary=dict(source='real_sumo',metadata=metadata)
    assert compatible_summary(summary,'unrelated/directory/archive.npz',targets)['city']=='city'
    simulation['routing_mode']='current_to_dest'
    assert compatible_summary(summary,'archive.npz',targets)['routing_mode']=='current_to_dest'
    simulation['incident_start_seconds']=5
    with pytest.raises(ValueError,match='physical_parameters_mismatch'):
        compatible_summary(summary,'archive.npz',targets)
