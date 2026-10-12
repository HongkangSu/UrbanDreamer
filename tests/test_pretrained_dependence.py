"""Synthetic validation of dependence-only references and immutable bank scoring."""
import json
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_pretrained_dependence as analysis
from tools.component_experiment_data import sha256


def test_spatial_permutations_preserve_marginal_and_complete_local_trajectories():
    rng = np.random.default_rng(12)
    draws = rng.normal(size=(4, 8, 5, 2)).astype(np.float32)
    targets = rng.normal(size=(4, 5, 2))
    spatial, orders = analysis.spatial_shuffle(draws, seed=43)
    temporal, _ = analysis.horizon_shuffle(draws, seed=71)
    for road in range(5):
        np.testing.assert_array_equal(spatial[:, :, road, :], draws[:, orders[:, road], road, :])
    assert len({tuple(orders[:, road]) for road in range(5)}) > 1
    result = analysis.verify_invariance(draws, spatial, temporal, targets, np.array([2., 3.]))
    assert all(value is True for key, value in result.items() if key != 'arithmetic')
    again, again_orders = analysis.spatial_shuffle(draws, seed=43)
    np.testing.assert_array_equal(again, spatial)
    np.testing.assert_array_equal(again_orders, orders)


def test_private_shuffles_do_not_modify_global_numpy_rng():
    np.random.seed(17)
    before = np.random.get_state()
    draws = np.ones((4, 8, 5, 2))
    analysis.spatial_shuffle(draws, seed=43)
    analysis.horizon_shuffle(draws, seed=71)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_horizon_permutations_preserve_full_joint_states_and_frame_scores():
    draws = np.arange(4 * 8 * 5 * 2).reshape(4, 8, 5, 2).astype(float)
    changed, orders = analysis.horizon_shuffle(draws, seed=11)
    for horizon in range(4):
        np.testing.assert_array_equal(changed[horizon], draws[horizon, orders[horizon]])
        y = np.zeros((5, 2))
        assert analysis.energy_score_u(changed[horizon], y, [1., 2.]) == analysis.energy_score_u(draws[horizon], y, [1., 2.])
    assert len({tuple(order) for order in orders}) > 1


def test_synthetic_spatial_dependence_changes_joint_scores_without_marginal_changes():
    original = np.array([[[0., 0.], [0., 0.]], [[1., 0.], [1., 0.]]])
    shuffled = np.array([[[0., 0.], [1., 0.]], [[1., 0.], [0., 0.]]])
    target = np.zeros((2, 2))
    assert analysis.energy_score_u(original, target, [1., 1.])['score'] == pytest.approx(0.)
    assert analysis.energy_score_u(shuffled, target, [1., 1.])['score'] == pytest.approx(1. - np.sqrt(2) / 2)
    a = analysis.variogram_scores(original, target, [1., 1.], [[0, 1]])
    b = analysis.variogram_scores(shuffled, target, [1., 1.], [[0, 1]])
    assert a['count']['plugin'] == 0. and b['count']['plugin'] == 1.
    for key, value in analysis.marginal_statistics(original[None], target[None]).items():
        np.testing.assert_array_equal(value, analysis.marginal_statistics(shuffled[None], target[None])[key])


def test_synthetic_temporal_dependence_changes_trajectory_but_not_frame_energy():
    original = np.array([[[[0., 0.]], [[1., 0.]]], [[[0., 0.]], [[1., 0.]]]])
    changed = original.copy()
    changed[1] = changed[1, ::-1]
    target = np.zeros((2, 1, 2))
    for horizon in range(2):
        assert analysis.energy_score_u(original[horizon], target[horizon], [1., 1.]) == analysis.energy_score_u(changed[horizon], target[horizon], [1., 1.])
    left = analysis.energy_score_u(original.transpose(1, 0, 2, 3), target, [1., 1.])
    right = analysis.energy_score_u(changed.transpose(1, 0, 2, 3), target, [1., 1.])
    assert left['score'] == pytest.approx(0.)
    assert right['score'] == pytest.approx(1. - np.sqrt(2) / 2)


def test_variogram_analytic_formula_scale_and_optional_u_correction():
    samples = np.array([[[0., 0.], [1., 0.]], [[0., 0.], [9., 0.]]])
    target = np.array([[0., 0.], [4., 0.]])
    score = analysis.variogram_scores(samples, target, [1., 1.], [[0, 1]])
    assert score['count']['plugin'] == 0.
    assert score['count']['u_corrected'] == -1.
    assert score['joint']['u_corrected'] == -.5
    scaled = analysis.variogram_scores(samples, target, [4., 1.], [[0, 1]])
    assert scaled['count']['u_corrected'] == -.25


def test_global_mass_decomposition_and_normalized_spatial_tv():
    result = analysis.mass_diagnosis([3., 5., 0., 2.], [4., 0., 3., 0.])
    assert result['occupied_signed_bias'] == -4.
    assert result['ghost_mass_true_empty'] == 7.
    assert result['missing_occupied_mass'] == 4.
    assert result['global_signed_mass'] == 3.
    assert result['decomposition_residual'] == 0.
    assert result['normalized_spatial_tv'] == pytest.approx(.7)
    negative = analysis.mass_diagnosis([-1., 2.], [0., 1.])
    assert negative['status'] == 'negative_predicted_mass'
    assert negative['normalized_spatial_tv'] is None
    assert negative['negative_empty_mass'] == 1.
    assert negative['occupied_signed_bias'] + negative['ghost_mass_true_empty'] == negative['global_signed_mass']


@pytest.mark.parametrize('predicted,target,status', [([1., 2.], [0., 0.], 'no_positive_target_mass'),
    ([0., 0.], [1., 0.], 'no_positive_predicted_mass'), ([1., 0.], [np.nan, 1.], 'missing_count_labels')])
def test_mass_undefined_strata_are_retained(predicted, target, status):
    result = analysis.mass_diagnosis(predicted, target)
    assert result['status'] == status and result['normalized_spatial_tv'] is None


def test_fixed_pairs_are_undirected_disjoint_reproducible_and_validate():
    graph = np.zeros((8, 8))
    graph[0, 1] = graph[2, 1] = graph[4, 5] = 1.
    graph[0, 0] = 1.
    adjacent, nonneighbor = analysis.select_pairs(graph, seed=77, nonneighbors=5)
    assert adjacent.tolist() == [[0, 1], [1, 2], [4, 5]]
    assert not set(map(tuple, adjacent)) & set(map(tuple, nonneighbor))
    np.testing.assert_array_equal(nonneighbor, analysis.select_pairs(graph, seed=77, nonneighbors=5)[1])
    for invalid in ([[1, 0]], [[0, 0]], [[0, 8]], [[0, 1], [0, 1]], [[0., 1.]]):
        with pytest.raises(ValueError):
            analysis.validate_pairs(invalid, 8)
    with pytest.raises(ValueError, match='fewer'):
        analysis.select_pairs(graph, seed=77, nonneighbors=99)


def test_target_masks_are_identical_and_missing_pair_features_are_explicit():
    samples = np.ones((4, 3, 2))
    target = np.array([[1., np.nan], [2., 3.], [0., 5.]])
    scores = analysis.variogram_scores(samples, target, [1., 2.], [[0, 1], [1, 2]])
    assert scores['count']['valid_pairs'] == 2
    assert scores['speed']['valid_pairs'] == 1 and scores['speed']['omitted_pairs'] == 1
    assert analysis.energy_score_u(samples, target, [1., 2.])['dimensions'] == 5
    with pytest.raises(ValueError, match='target_mask'):
        analysis.energy_score_u(samples, target, [1., 2.], target_mask=np.ones((3, 2), dtype=bool))
    empty = analysis.energy_score_u(samples, target, [1., 2.], target_mask=np.zeros((3, 2), dtype=bool))
    assert empty == {'score': None, 'dimensions': 0}
    with pytest.raises(ValueError, match='finite'):
        analysis.energy_score_u(samples * np.inf, target, [1., 2.])


def synthetic_archive(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    (source / 'banks').mkdir(parents=True)
    prespec = dict(nonneighbor_pair_seed=3, nonneighbor_pairs=1, spatial_shuffle_seed=1,
        horizon_shuffle_seed=2, bootstrap_seed=4, bootstrap_repetitions=20)
    spec_path, pairs_path = tmp_path / 'prespec.json', tmp_path / 'pairs.json'
    spec_path.write_text(json.dumps(prespec))
    pairs = dict(edge_ids=['a', 'b', 'c', 'd'], adjacent=[[0, 1], [1, 2], [2, 3]], nonneighbor=[[0, 3]])
    pairs_path.write_text(json.dumps(pairs))
    protocol = dict(role='fully_pretrained_development_distribution', checkpoint_sha256=analysis.CHECKPOINT_SHA,
        training_performed=False, conditions=2, scales={'scales': [2., 3.]})
    (source / 'protocol.json').write_text(json.dumps(protocol))
    (source / 'summary.json').write_text(json.dumps(dict(status='complete', conditions=2)))
    conditions, manifest = [], []
    rng = np.random.default_rng(22)
    for index in range(2):
        condition = dict(demand_group=str(index), episode='episode' + str(index), start=0,
            traffic_phase='empty_stress' if index else 'early', condition_seed=100 + index)
        conditions.append(condition)
        draws = rng.uniform(size=(4, 32, 1, 4, 2)).astype(np.float32)
        targets = np.zeros((4, 1, 4, 2))
        if not index:
            targets[:, :, 0, 0] = 1.
        targets[..., 1] = 2.
        path = source / 'banks' / f'condition_{index:03d}.npz'
        np.savez_compressed(path, horizons=[1, 2, 4, 8], draws=draws, targets=targets, scales=[2., 3.])
        manifest.append(dict(condition=condition, file=str(path), sha256=sha256(path)))
    (source / 'conditions.json').write_text(json.dumps(conditions))
    (source / 'bank_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(analysis, '_source_files', lambda directory, *_: {
        str(directory / name): sha256(directory / name) for name in ('protocol.json', 'summary.json', 'conditions.json', 'bank_manifest.json')})
    return source, spec_path, pairs_path


def test_complete_analysis_preserves_banks_and_keeps_empty_pilot_descriptive(tmp_path, monkeypatch):
    source, spec, pairs = synthetic_archive(tmp_path, monkeypatch)
    before = {str(path): sha256(path) for path in source.rglob('*') if path.is_file()}
    result = analysis.analyze(source, tmp_path / 'output', experiment_dir=tmp_path,
        prespecification_path=spec, pairs_path=pairs)
    assert result['conditions'] == result['invariance_conditions_passed'] == 2
    assert result['scored_rows'] == 30 and result['pilot_descriptive_only']
    assert result['no_positive_target_mass_rows'] == 12
    assert {row['stratum'] for row in result['aggregates']} == {'empty_stress', 'traffic_phases'}
    assert all(row['ci_low'] is None and row['ci_high'] is None for row in result['aggregates'])
    assert before == {str(path): sha256(path) for path in source.rglob('*') if path.is_file()}
    unchanged = [r for r in result['aggregates'] if r['metric'] == 'energy_u' and r['contrast'] == 'original_minus_horizon_shuffle']
    assert all(row['estimate'] == 0. for row in unchanged)


def test_missing_condition_or_tampered_bank_is_not_silently_dropped(tmp_path, monkeypatch):
    source, spec, pairs = synthetic_archive(tmp_path, monkeypatch)
    manifest_path = source / 'bank_manifest.json'
    original = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps(original[:1]))
    with pytest.raises(ValueError, match='full declared condition'):
        analysis.analyze(source, tmp_path / 'output', experiment_dir=tmp_path, prespecification_path=spec, pairs_path=pairs)
    manifest_path.write_text(json.dumps(original))
    with Path(original[0]['file']).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='immutable hash'):
        analysis.analyze(source, tmp_path / 'output', experiment_dir=tmp_path, prespecification_path=spec, pairs_path=pairs)
