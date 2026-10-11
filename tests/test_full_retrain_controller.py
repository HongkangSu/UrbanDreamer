import json
from pathlib import Path

import pytest

from tools.run_full_retrain import (completed_input, estimate_epoch, immutable_plan,
                                    make_plan, preparation_verified, run_child, tail_seconds)


def inputs(windows):
    cache = dict(status='complete', identity='frozen', entries=[
        dict(split='train', windows=windows, method='random'),
        dict(split='validation', windows=20)])
    profile = dict(batch_size=32, full_update_seconds_median=.1,
                   tail_update_seconds={'1': .2, '16': .3, '31': .4},
                   sample_loading_seconds=0)
    return cache, dict(status='complete', profiles=[profile])


def test_epochs_scale_by_full_dataset_exposure():
    small, benchmark = inputs(1000000)
    benchmark['profiles'][0]['full_update_seconds_median'] = .05
    large, _ = inputs(3000000)
    a, b = make_plan(small, benchmark), make_plan(large, benchmark)
    assert a['epochs'] == 6
    assert b['epochs'] == 2
    assert a['actual_sample_exposures'] == b['actual_sample_exposures'] == 6000000


def test_full_batches_and_eager_tails_are_separate():
    _, benchmark = inputs(10)
    profile = benchmark['profiles'][0]
    result = estimate_epoch([dict(windows=48), dict(windows=16)], profile)
    assert result['full_batches'] == 1
    assert result['tail_batches'] == 2
    assert result['training_seconds_per_epoch'] == pytest.approx(.7)
    assert tail_seconds(profile, 8) == pytest.approx(.2 + .1 * 7 / 15)


def test_cold_loading_once_per_episode():
    _, benchmark = inputs(10)
    profile = benchmark['profiles'][0]
    profile['sample_loading_seconds'] = {
        '16': {'cold': .5, 'warm_median': .1},
        '32': {'cold': .8, 'warm_median': .2}}
    result = estimate_epoch([dict(windows=80)], profile)
    assert result['training_seconds_per_epoch'] == pytest.approx(.2 + .3 + .8 + .2 + .1)


def test_unfinished_manifest_waits_and_failure_is_explicit(tmp_path):
    path = tmp_path / 'cache.json'
    assert completed_input(path) is None
    path.write_text(json.dumps(dict(status='building')))
    assert completed_input(path) is None
    path.write_text(json.dumps(dict(status='failed', error='source changed')))
    with pytest.raises(ValueError, match='source changed'):
        completed_input(path)


def test_plan_identity_immutable(tmp_path):
    path = tmp_path / 'plan.json'
    immutable_plan(path, dict(identity='a'))
    immutable_plan(path, dict(identity='a'))
    with pytest.raises(ValueError, match='changed'):
        immutable_plan(path, dict(identity='b'))


def test_completed_cache_waits_for_postbuild_provenance_validation(tmp_path):
    manifest = tmp_path / 'cache_manifest.json'
    manifest.write_text(json.dumps(dict(status='complete')))
    status = tmp_path / 'status.json'
    status.write_text(json.dumps(dict(status='verifying')))
    assert not preparation_verified(manifest)
    status.write_text(json.dumps(dict(status='complete')))
    assert preparation_verified(manifest)
    status.write_text(json.dumps(dict(status='failed', error='split overlap')))
    with pytest.raises(ValueError, match='split overlap'):
        preparation_verified(manifest)


def test_single_epoch_over_budget_refuses_subsetting():
    cache, benchmark = inputs(10000000)
    benchmark['profiles'][0]['full_update_seconds_median'] = 1.
    with pytest.raises(ValueError, match='one complete epoch'):
        make_plan(cache, benchmark)


def test_reduced_corpus_retains_two_epochs_when_three_exceed_eight_hours():
    cache, benchmark = inputs(2000000)
    benchmark['profiles'][0]['full_update_seconds_median'] = .15
    plan = make_plan(cache, benchmark)
    assert plan['exposure_target_epochs'] == 3
    assert plan['epochs'] == 2
    assert plan['estimated_total_seconds'] <= 8 * 3600


def test_gpu_headroom_rejects_faster_but_unusable_batch():
    cache, benchmark = inputs(1000000)
    small = benchmark['profiles'][0]
    small['peak_reserved_mib'] = 2100
    large = dict(small, batch_size=128, full_update_seconds_median=.1,
                 peak_reserved_mib=3600, tail_update_seconds={'96': .1})
    benchmark['profiles'].append(large)
    plan = make_plan(cache, benchmark, available_memory_mib=3100)
    assert plan['batch_size'] == 32
    assert plan['memory_excluded_profiles'][0]['batch_size'] == 128
    with pytest.raises(ValueError, match='memory headroom'):
        make_plan(cache, benchmark, available_memory_mib=1000)


def test_failed_child_propagates(tmp_path):
    class Child:
        pid = 42
        def wait(self):
            return 3
    captured = []
    def launch(command, **kwargs):
        captured.append((command, kwargs))
        return Child()
    with pytest.raises(RuntimeError, match='exited 3'):
        run_child(['python', '-u', 'train.py'], tmp_path / 'run.log', cwd=tmp_path, popen=launch)
    assert captured[0][1]['start_new_session'] is True


def test_reference_coverage_retained_when_above_six_but_under_eight_hours():
    cache, benchmark = inputs(3000000)
    # Two epochs including margin take roughly 7.35 hours.
    benchmark['profiles'][0]['full_update_seconds_median'] = .12
    plan = make_plan(cache, benchmark)
    assert plan['epochs'] == 2
    assert 6 * 3600 < plan['estimated_total_seconds'] < 8 * 3600


def test_eager_tail_timings_shared_across_profiles():
    cache, benchmark = inputs(48)
    benchmark['profiles'].append(dict(batch_size=64, full_update_seconds_median=.15,
                                     tail_update_seconds={'32': .1}))
    plan = make_plan(cache, benchmark)
    assert plan['eager_tail_seconds']['32'] == .1
    # B64's 48-row tail interpolates globally measured 31/32 then extrapolates,
    # rather than using the B64 compiled full timing.
    estimate = next(row for row in plan['candidate_profiles'] if row['batch_size'] == 64)
    assert estimate['training_seconds_per_epoch'] == pytest.approx(.15)
