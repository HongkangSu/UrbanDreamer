import numpy as np
import pytest
import json

from tools.analyze_traffic_spectrum import (
    normalized_laplacian, projection_energy, select_grouped, bootstrap_ratio, read_episode,
)
from urban_dreamer.learning.offline_data import archive_content_hash


def test_reads_compact_archive_without_original_source_summary(tmp_path):
    path = tmp_path / 'episode.npz'
    metadata = np.frombuffer(json.dumps({'schema_version': 1, 'population': 2}).encode(), dtype=np.uint8)
    np.savez(path, metadata_json=metadata, flow=np.zeros((2, 3, 8)))
    entry = dict(cache_path=str(path), cache_content_hash=archive_content_hash(path))
    episode = read_episode(entry)
    assert episode['metadata']['population'] == 2
    assert episode['flow'].shape == (2, 3, 8)
    entry['cache_content_hash'] = 'corrupt'
    with pytest.raises(ValueError, match='content mismatch'):
        read_episode(entry)


def test_low_graph_mode_captures_constant_signal_on_regular_graph():
    adjacency = np.ones((4, 4)) - np.eye(4)
    values, vectors = np.linalg.eigh(normalized_laplacian(adjacency))
    numerator, denominator = projection_energy(np.ones((2, 4, 1)), vectors, [1, 4])
    assert values[0] == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(numerator / denominator[:, None], 1)


def test_high_mode_is_not_reported_as_low_frequency():
    adjacency = np.ones((4, 4)) - np.eye(4)
    _, vectors = np.linalg.eigh(normalized_laplacian(adjacency))
    numerator, denominator = projection_energy(vectors[:, -1][None, :, None], vectors, [1, 4])
    assert numerator[0, 0] / denominator[0] == pytest.approx(0, abs=1e-12)
    assert numerator[0, 1] / denominator[0] == pytest.approx(1)


def test_directed_normalization_and_isolated_node_are_well_defined():
    adjacency = np.array([[0, 2, 0], [0, 0, 0], [0, 0, 0]])
    expected = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]])
    np.testing.assert_allclose(normalized_laplacian(adjacency), expected)


def test_group_selection_is_unique_deterministic_and_split_specific():
    entries = [dict(split=split, demand_group=str(group), cache_path=str(episode))
               for split, group in [('train', 1), ('train', 2), ('validation', 3)] for episode in range(4)]
    selected = select_grouped(entries, 'train', 20, 4)
    assert len(selected) == 2
    assert len({x['demand_group'] for x in selected}) == 2
    assert all(x['split'] == 'train' for x in selected)
    assert selected == select_grouped(entries, 'train', 20, 4)


def test_bootstrap_uses_energy_ratio_not_mean_of_per_episode_ratios():
    estimate, low, high = bootstrap_ratio([1, 90], [10, 100], repetitions=1000, seed=1)
    assert estimate == pytest.approx(91 / 110)
    assert low <= estimate <= high
    assert bootstrap_ratio([0, 0], [0, 0]) == (None, None, None)


def test_nonfinite_increment_rejected():
    with pytest.raises(ValueError, match='finite'):
        projection_energy(np.full((1, 2, 1), np.nan), np.eye(2), [1])
