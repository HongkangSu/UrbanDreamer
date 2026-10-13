import numpy as np
import pytest
import json

import tools.analyze_multicity_increment_spectrum as spectrum

from tools.analyze_multicity_increment_spectrum import (
    complete_eigenspace_cut, group_aggregates, mode_energy, prespecified_cuts,
    raw_pairs, summarize, training_scale,
)
from tools.analyze_traffic_spectrum import normalized_laplacian


def test_complete_projection_parseval_and_nested_cumulative_energy():
    rng = np.random.default_rng(42)
    basis = np.linalg.qr(rng.normal(size=(12, 12)))[0]
    delta = rng.normal(size=(7, 12, 2))
    energy, total = mode_energy(delta, basis)
    np.testing.assert_allclose(energy.sum(axis=-1), total)
    assert np.all(np.diff(np.cumsum(energy, axis=-1), axis=-1) >= 0)
    np.testing.assert_allclose(energy[:, 2], energy[:, 0] + energy[:, 1])


def test_known_low_and_high_modes_are_separated():
    adjacency = np.diag(np.ones(4), 1) + np.diag(np.ones(4), -1)
    eigenvalues, basis = np.linalg.eigh(normalized_laplacian(adjacency))
    delta = np.stack((basis[:, 0], basis[:, -1]), axis=-1)[None]
    energy, _ = mode_energy(delta, basis)
    assert energy[0, 0, 0] == pytest.approx(1)
    assert energy[0, 1, 0] == pytest.approx(0, abs=1e-20)
    assert eigenvalues[0] == pytest.approx(0, abs=1e-12)


def test_tied_boundary_expands_to_rotation_invariant_complete_space():
    eigenvalues = np.array([0., 0., 0., .2, .2, .8])
    assert complete_eigenspace_cut(eigenvalues, 1) == 3
    assert complete_eigenspace_cut(eigenvalues, 4) == 5
    assert complete_eigenspace_cut(eigenvalues, 0) == 0
    cuts = {requested: rank for kind, requested, rank in prespecified_cuts(eigenvalues) if kind == 'lambda_cutoff'}
    assert cuts == {'0.1': 3, '0.25': 5, '0.5': 5}


def test_actual_observed_pair_retains_birth_and_last_endpoint():
    flow = np.zeros((3, 2, 8))
    flow[..., 1] = 1
    flow[2, 0, 2] = .5
    arrays = dict(flow=flow, time=np.array([0., 30., 60.]))
    _, _, delta = raw_pairs(arrays, [1], np.array([10., 20.]), np.array([15., 25.]))
    assert delta[0, 0, 0] == 5
    with pytest.raises(ValueError, match='padding'):
        raw_pairs(arrays, [2], np.ones(2), np.ones(2))
    arrays['time'][2] = 59.
    with pytest.raises(ValueError, match='interval'):
        raw_pairs(arrays, [1], np.ones(2), np.ones(2))


def test_training_rms_is_not_centered_and_validation_does_not_fit_scale():
    train = [np.full((2, 3, 2), [2., 4.])]
    scale, raw, squares, elements = training_scale(train)
    np.testing.assert_allclose(scale, [2., 4.])
    np.testing.assert_allclose(raw, [2., 4.])
    assert elements == 6
    validation = np.full((1, 3, 2), [100., 100.])
    np.testing.assert_allclose(validation / scale, np.full((1, 3, 2), [50., 25.]))
    np.testing.assert_allclose(squares, [24., 96.])


def test_group_aggregation_bootstraps_pooled_energy_and_retains_zero_groups():
    energy = np.array([[[1., 9.], [2., 8.], [3., 17.]], [[90., 10.], [80., 20.], [170., 30.]]])
    total = energy.sum(axis=-1)
    num, den, windows = group_aggregates(energy, total, np.array([0, 2]))
    np.testing.assert_allclose(num[0], energy.sum(axis=0))
    assert windows.tolist() == [2, 1, 0, 1]
    zero_num, zero_den, zero_windows = group_aggregates(np.zeros_like(energy), np.zeros_like(total), np.array([0, 2]))
    rows = summarize('test', np.array([0., 1.]), np.stack((num, zero_num)), np.stack((den, zero_den)),
                     np.stack((windows, zero_windows)), 100, 42)
    row = next(r for r in rows if r['space'] == 'raw_count' and r['stratum'] == 'all'
               and r['cut_kind'] == 'mode_fraction' and r['requested'] == '0.1')
    assert row['energy_fraction'] == pytest.approx(91 / 110)
    assert row['groups'] == 2 and row['zero_energy_groups'] == 1
    assert any(r['energy_fraction'] is None for r in rows if r['stratum'] == 'medium')


def test_active_graph_supplement_keeps_training_mask_when_validation_uses_other_roads(tmp_path, monkeypatch):
    monkeypatch.setattr(spectrum, 'SOURCES', {'city': ('unused', 'unused')})
    monkeypatch.setattr(spectrum, 'plot_results', lambda rows, output: None)
    (tmp_path / 'prespec.json').write_text(json.dumps(dict(bootstrap=100, seed=42)))
    (tmp_path / 'result.json').write_text(json.dumps(dict(status='complete')))
    city = tmp_path / 'city'
    city.mkdir()
    mask = np.array([True, True, False])
    adjacency = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    np.savez(city / 'graph.npz', adjacency=adjacency, train_positive_roads=mask, edge_ids=np.array(['a', 'b', 'c']))
    delta = np.ones((2, 2, 3, 2))
    delta[:, :, 2] = 100
    np.savez(city / 'raw_observed_increments.npz', delta=delta, rms_scale=np.ones(2), demand_groups=np.array(['g1', 'g2']))
    spectrum.active_road_supplement(tmp_path)
    result = json.loads((tmp_path / 'training_active_graph_supplement/result.json').read_text())
    audit = result['graph_audits']['city']
    assert audit['induced_roads'] == 2
    assert audit['omitted_validation_nonzero_count_roads'] == 1
    assert audit['omitted_validation_energy_fraction']['raw_joint'] == pytest.approx(10000 / 10002)
    saved = np.load(tmp_path / 'training_active_graph_supplement/city_group_energy.npz')
    np.testing.assert_array_equal(saved['train_positive_mask'], mask)
    np.testing.assert_allclose(saved['denominators'], np.array([[4., 4., 8.], [4., 4., 8.]]))
