import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.run_full_retrain import (completed_input, estimate_epoch, immutable_plan,
                                    make_plan, preparation_verified, run_child, tail_seconds,
                                    commands, validate_benchmark_profile, write_config)


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


def test_recursive_epoch_counts_valid_sequence_starts_and_loading():
    _, benchmark = inputs(10)
    profile = benchmark['profiles'][0]
    profile.update(sequence_length=3, sample_loading_seconds={
        '16': {'cold': .5, 'warm_median': .1},
        '32': {'cold': .8, 'warm_median': .2}})
    result = estimate_epoch([dict(windows=82), dict(windows=2)], profile)
    assert result['train_windows'] == 81  # 80 starts plus one padded short episode
    assert result['full_batches'] == 2
    assert result['tail_batches'] == 2
    assert result['training_seconds_per_epoch'] == pytest.approx(2.3)


@pytest.mark.parametrize('metadata', [
    {},
    {'fm_training_mode': 'single_step', 'sequence_length': 3},
    {'fm_training_mode': 'recursive_3step', 'sequence_length': 1},
])
def test_recursive_execution_rejects_incompatible_benchmark(metadata):
    config = SimpleNamespace(method=SimpleNamespace(fm_training_mode='recursive_3step'),
        resources=SimpleNamespace(world_model_initialization_sequence_length=3))
    with pytest.raises(ValueError, match='matching benchmark'):
        validate_benchmark_profile(config, dict(profiles=[metadata]))


def test_recursive_execution_accepts_matching_successful_benchmarks():
    config = SimpleNamespace(method=SimpleNamespace(fm_training_mode='recursive_3step'),
        resources=SimpleNamespace(world_model_initialization_sequence_length=3))
    validate_benchmark_profile(config, dict(profiles=[
        dict(fm_training_mode='recursive_3step', sequence_length=3),
        dict(status='failed', fm_training_mode='single_step', sequence_length=1)]))


def test_explicit_historical_single_step_accepts_benchmark_without_metadata():
    config = SimpleNamespace(method=SimpleNamespace(fm_training_mode='single_step'))
    validate_benchmark_profile(config, dict(profiles=[{}]))


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


@pytest.mark.parametrize('mode,sequence,supervision,risk', [
    ('recursive_3step', 3, 'observed_endpoint', .1),
    ('single_step', 1, 'sampled_endpoint', 0.),
])
def test_execution_preserves_configured_supervision_and_risk(
        tmp_path, mode, sequence, supervision, risk):
    from test_urban_dreamer_config import _write_config
    from urban_dreamer.workflow.config import load_config

    source = _write_config(tmp_path / 'source.toml', extra=f'''
[method]
fm_training_mode = "{mode}"
exposure_readout_supervision = "{supervision}"
disagreement_weight = {risk}
''')
    source.write_text(source.read_text().replace(
        'world_model_initialization_sequence_length = 16',
        f'world_model_initialization_sequence_length = {sequence}'))
    args = SimpleNamespace(config=source, run_dir=tmp_path / 'run', device='cpu',
        cache_manifest=tmp_path / 'cache_manifest.json', source_root=tmp_path,
        calibrated_codec=tmp_path / 'codec.pt', gpu_memory_fraction=.5)
    plan = dict(epochs=2, batch_size=32, train_episodes=10, validation_episodes=2)
    destination = write_config(args, plan)
    frozen = load_config(destination)
    assert frozen.method.fm_training_mode == mode
    assert frozen.resources.world_model_initialization_sequence_length == sequence
    assert frozen.method.exposure_readout_supervision == supervision
    assert frozen.method.disagreement_weight == risk
    assert frozen.resources.world_model_initialization_batch_size == 32
    assert frozen.resources.imagination_rollouts_per_pass == 16

    pretrain, _ = commands(args, plan, destination)
    assert pretrain[pretrain.index('--fm-training-mode') + 1] == mode
    assert pretrain[pretrain.index('--sequence-length') + 1] == str(sequence)
    assert pretrain[pretrain.index('--readout-supervision') + 1] == supervision
