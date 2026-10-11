from copy import deepcopy
import json

import numpy as np
import pytest

from tools import build_city_supplement_cache
from tools.build_city_supplement_cache import seed_receipts
from urban_dreamer.learning.offline_data import _identity, archive_content_hash
from urban_dreamer.learning.pretrain_cache import file_stamp


def fixture(tmp_path):
    source, compact = tmp_path / 'source.npz', tmp_path / 'compact.npz'
    np.savez(source, values=np.arange(5))
    np.savez_compressed(compact, values=np.arange(5))
    path = str(source)
    contract = dict(city='test', input_hashes={'network': 'hash'}, physical={'grid': 30})
    row = dict(path=path, demand_group='demand', demand_hash='demand', episode=1,
               method='actor', root='/old', split='train', crc_trajectory_candidate='crc')
    old = {key: row[key] for key in ('path', 'demand_group', 'episode', 'method', 'root', 'split')}
    old.update(canonical=True, cache_path=str(compact), source_stamp=file_stamp(source),
               content_hash=archive_content_hash(source), cache_content_hash=archive_content_hash(compact),
               census_trajectory_signature='crc', trajectory_hash='sha', cache_bytes=compact.stat().st_size)
    manifest = dict(status='complete', contract=contract, sources=[old])
    manifest['identity'] = _identity(manifest)
    plan = dict(identity='new-plan', contract=contract, sources=[row], canonical={}, expected_trajectory={})
    return source, compact, plan, manifest


def signed(manifest):
    manifest['identity'] = _identity({key: value for key, value in manifest.items() if key != 'identity'})
    return manifest


def test_reuses_old_immutable_cache_with_new_receipt_identity(tmp_path):
    source, compact, plan, manifest = fixture(tmp_path)
    before = compact.read_bytes()
    output = tmp_path / 'supplement'
    stats = seed_receipts(plan, manifest, output)
    receipt = json.loads(next((output / 'receipts').glob('*.json')).read_text())
    assert receipt['plan_identity'] == 'new-plan'
    assert receipt['cache_path'] == str(compact)
    assert compact.read_bytes() == before
    assert stats['reused_sources'] == 1 and stats['sources_to_build'] == 0
    assert seed_receipts(plan, manifest, output)['existing_receipts'] == 1


def test_lossless_repacking_refreshes_stamp_but_content_changes_fail(tmp_path):
    source, compact, plan, manifest = fixture(tmp_path)
    np.savez_compressed(source, values=np.arange(5))
    assert file_stamp(source) != manifest['sources'][0]['source_stamp']
    output = tmp_path / 'repacked'
    assert seed_receipts(plan, manifest, output)['reused_sources'] == 1
    saved = json.loads(next((output / 'receipts').glob('*.json')).read_text())
    assert saved['source_stamp'] == file_stamp(source)
    np.savez_compressed(source, values=np.arange(9))
    with pytest.raises(ValueError, match='source content changed'):
        seed_receipts(plan, manifest, tmp_path / 'changed')


def test_rejects_compact_corruption_and_contract_changes(tmp_path):
    source, compact, plan, manifest = fixture(tmp_path)
    changed = deepcopy(plan)
    changed['contract']['physical']['grid'] = 15
    with pytest.raises(ValueError, match='contracts differ'):
        seed_receipts(changed, manifest, tmp_path / 'contract')
    np.savez_compressed(compact, values=np.arange(9))
    with pytest.raises(ValueError, match='compact episode changed'):
        seed_receipts(plan, manifest, tmp_path / 'corrupt')


@pytest.mark.parametrize('key', ['path', 'demand_group', 'episode', 'method', 'root', 'split'])
def test_rejects_old_receipt_identity_changes(tmp_path, key):
    source, compact, plan, manifest = fixture(tmp_path)
    if key == 'path':
        # A prior source absent from the new plan is simply not reusable.
        manifest['sources'][0][key] = 'different'
        assert seed_receipts(plan, signed(manifest), tmp_path / 'other')['reused_sources'] == 0
    else:
        manifest['sources'][0][key] = 'different'
        with pytest.raises(ValueError, match='identity differs'):
            seed_receipts(plan, signed(manifest), tmp_path / 'other')


@pytest.mark.parametrize('key', ['census_trajectory_signature', 'trajectory_hash'])
def test_rejects_trajectory_signature_and_sha_mismatch(tmp_path, key):
    source, compact, plan, manifest = fixture(tmp_path)
    plan['expected_trajectory'][str(source)] = 'sha'
    manifest['sources'][0][key] = 'changed'
    with pytest.raises(ValueError, match='identity differs|SHA256 differs'):
        seed_receipts(plan, signed(manifest), tmp_path / 'other')


def test_noncanonical_and_newly_noncanonical_sources_are_rebuilt_not_reused(tmp_path):
    source, compact, plan, manifest = fixture(tmp_path)
    manifest['sources'][0]['canonical'] = False
    stats = seed_receipts(plan, signed(manifest), tmp_path / 'noncanonical')
    assert stats['old_noncanonical_skipped'] == 1 and stats['sources_to_build'] == 1
    manifest['sources'][0]['canonical'] = True
    plan['canonical'][str(source)] = 'new-canonical-source'
    stats = seed_receipts(plan, signed(manifest), tmp_path / 'new-canonical')
    assert stats['changed_canonical_skipped'] == 1 and stats['sources_to_build'] == 1


def test_old_manifest_and_existing_plan_identity_are_verified(tmp_path):
    source, compact, plan, manifest = fixture(tmp_path)
    invalid = deepcopy(manifest)
    invalid['sources'][0]['split'] = 'changed'
    with pytest.raises(ValueError, match='manifest is incomplete or changed'):
        seed_receipts(plan, invalid, tmp_path / 'invalid')
    output = tmp_path / 'different-plan'
    seed_receipts(plan, manifest, output)
    plan['identity'] = 'other-plan'
    with pytest.raises(ValueError, match='different frozen plan'):
        seed_receipts(plan, manifest, output)


def test_producer_failure_is_recorded_for_waiting_supervisor(tmp_path, monkeypatch):
    city = tmp_path / 'city'
    progress_path = city / 'cache/cache_progress.json'
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text(json.dumps(dict(status='seeding', completed=25, total=50)))
    def fail(*args, **kwargs):
        raise ValueError('source changed')
    monkeypatch.setattr(build_city_supplement_cache, 'prepare_and_build', fail)
    with pytest.raises(ValueError, match='source changed'):
        build_city_supplement_cache.main([
            '--city-dir', str(city), '--previous-city-dir', str(tmp_path / 'old')])
    progress = json.loads(progress_path.read_text())
    assert progress == dict(status='failed', completed=25, total=50,
                            error='ValueError: source changed')


def test_seeding_has_durable_progress(tmp_path):
    source, compact, plan, manifest = fixture(tmp_path)
    output = tmp_path / 'supplement'
    seed_receipts(plan, manifest, output)
    progress = json.loads((output / 'cache_progress.json').read_text())
    assert progress['status'] == 'seeding'
    assert progress['completed'] == progress['total'] == 1
    assert progress['reused_sources'] == 1
