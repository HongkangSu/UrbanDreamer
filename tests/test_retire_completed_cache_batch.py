from pathlib import Path
import pytest
from tools import retire_completed_cache_batch as batch
from tools import retire_cached_trajectories as retire
from tests.test_retire_cached_trajectories import fixture, write


def fake_real_batch_report(manifest, output, city):
    proofs=[]
    for entry in manifest['entries']:
        path=output/'receipts'/f"{batch._identity({'census_path':entry['census_row']['path']})}.json"
        fields={k:entry[k] for k in ('trajectory_hash','cache_path','cache_content_hash','cache_member_signature','summary_log_path','summary_json_sha256')}
        proofs.append(dict(path=str(path),sha256=retire.sha(path),**fields))
    retire.atomic_json(output/'verification.json',dict(status='passed',city=city,manifest_identity=manifest['identity'],manifest_sha256=retire.sha(output/'cache_manifest.json'),receipts=proofs,**{k:True for k in retire.REPORT_FLAGS}))


def test_batch_scope_parent_trace_and_independent_binding(tmp_path):
    f=fixture(tmp_path);parent=f['args'][0].parent;original=(parent/'plan.json').read_bytes()
    output=tmp_path/'batch'
    manifest=batch.publish_batch(parent,[f['entry']['census_row']],output,verify=fake_real_batch_report)
    assert manifest['scope']=='completed_receipt_batch'
    assert (parent/'plan.json').read_bytes()==original
    assert f['raw'].exists()
    entry=manifest['entries'][0]
    assert entry['parent_receipt']['plan_identity']==f['entry']['plan_identity']
    assert entry['plan_identity']!=entry['parent_receipt']['plan_identity']
    retire.verified_bundle(output/'cache_manifest.json',output/'verification.json',output/'canonical.jsonl')


def test_bad_parent_receipt_binding_rejected(tmp_path):
    f=fixture(tmp_path);parent=f['args'][0].parent
    receipt=next((parent/'receipts').glob('*.json'));entry=retire.read(receipt);entry['plan_identity']='wrong';write(receipt,entry)
    with pytest.raises(ValueError,match='parent_receipt_binding'):
        batch.publish_batch(parent,[f['entry']['census_row']],tmp_path/'batch',verify=fake_real_batch_report)
    assert f['raw'].exists()


def test_changed_cache_rejected_before_publication(tmp_path):
    f=fixture(tmp_path);f['cache'].write_bytes(f['raw'].read_bytes())
    with pytest.raises(ValueError,match='cache_'):
        batch.publish_batch(f['args'][0].parent,[f['entry']['census_row']],tmp_path/'batch',verify=fake_real_batch_report)
    assert f['raw'].exists()


def test_final_validation_anchors_follow_entry_path_sort_not_plan_order(tmp_path,monkeypatch):
    c1=str(tmp_path/'cache1.npz');c2=str(tmp_path/'cache2.npz')
    monkeypatch.setattr(batch,'old_entries',lambda _: {c1:{'path':'/raw/z.npz'},c2:{'path':'/raw/a.npz'}})
    plan={'rows':[{'path':'/raw/y.npz'},{'path':'/canonical/a','reusable_cache':c1,'cache_manifest':'old'},
                  {'path':'/raw/b.npz'},{'path':'/canonical/z','reusable_cache':c2,'cache_manifest':'old'}]}
    assert batch.validation_anchors(plan)==['/raw/a.npz','/raw/b.npz']
