"""Pure helper checks; deliberately avoid starting a SUMO session."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from tools import run_ac_counterfactual as experiment


def test_keyed_uniform_is_reproducible_and_separates_decision_identity():
    values = [experiment.keyed_uniform(seed, vehicle, decision)
              for seed in (42, 43) for vehicle in ("a", "b") for decision in range(5)]
    assert all(0 < value < 1 for value in values)
    assert len(set(values)) == len(values)
    assert experiment.keyed_uniform(42, "a", 3) == experiment.keyed_uniform(42, "a", 3)


@pytest.mark.parametrize("digest_byte", [b"\x00", b"\xff"])
def test_keyed_uniform_remains_open_at_digest_extremes(monkeypatch, digest_byte):
    monkeypatch.setattr(experiment.hashlib, "sha256",
                        lambda _: SimpleNamespace(digest=lambda: digest_byte * 32))
    assert 0 < experiment.keyed_uniform(42, "vehicle", 0) < 1


def test_vehicle_sampling_is_independent_of_batch_order():
    vehicles = [{"vehicle_id": name} for name in ("a", "b", "c")]
    probabilities = np.array([[.1, .2, .7], [.4, .6, 0.], [.3, 0., .7]])
    counts = Counter({"a": 3, "b": 1, "c": 8})
    reordered_counts = counts.copy()
    chosen = experiment.sample_indices(probabilities, vehicles, counts, 321)
    permutation = [2, 0, 1]
    reordered = experiment.sample_indices(probabilities[permutation],
        [vehicles[index] for index in permutation], reordered_counts, 321)
    assert dict(zip((v["vehicle_id"] for v in vehicles), chosen)) == {
        vehicles[index]["vehicle_id"]: action
        for index, action in zip(permutation, reordered)
    }
    assert counts == reordered_counts == Counter({"a": 4, "b": 2, "c": 9})


def test_masked_actions_never_receive_a_sample():
    probabilities = np.array([[0., .3, 0., .7, 0.]])
    vehicles = [{"vehicle_id": "a"}]
    for seed in range(1_000):
        assert experiment.sample_indices(probabilities, vehicles, Counter(), seed)[0] in (1, 3)


@pytest.mark.parametrize("uniform, expected", [(1e-15, 1), (.3, 3),
                                               (np.nextafter(1., 0.), 3)])
def test_mask_and_roundoff_are_respected_at_cdf_boundaries(monkeypatch, uniform, expected):
    monkeypatch.setattr(experiment, "keyed_uniform", lambda *args: uniform)
    probabilities = np.array([[0., .3, 0., .6999999999999998, 0.]])
    actual = experiment.sample_indices(probabilities, [{"vehicle_id": "a"}], Counter(), 42)
    assert actual == [expected]


def test_ranked_subsets_have_exact_nested_counts_and_ignore_input_order_and_duplicates():
    ids = [f"vehicle_{i}" for i in range(100)]
    fractions = [0., .05, .10, .20, 1.]
    subsets = [experiment.ranked_subset(ids, fraction) for fraction in fractions]
    assert [len(subset) for subset in subsets] == [0, 5, 10, 20, 100]
    assert all(left <= right for left, right in zip(subsets, subsets[1:]))
    duplicate_ids = list(reversed(ids)) + ids[:20]
    assert subsets == [experiment.ranked_subset(duplicate_ids, fraction) for fraction in fractions]


@pytest.mark.parametrize("fraction", [-.1, 1.1, np.nan])
def test_ranked_subset_rejects_invalid_fractions(fraction):
    with pytest.raises(ValueError, match="fraction"):
        experiment.ranked_subset(["a"], fraction)


@pytest.fixture
def fake_runner():
    class Candidates:
        def __init__(self):
            self.route_calls = []

        def costs(self, destination):
            assert destination == 7
            return np.array([0., 90., 100., 100., 80., 261., 260., 0.])

        def full_route(self, current, edge, destination):
            self.route_calls.append((current, edge, destination))
            assert current == 0 and destination == 7
            return [0, 2, 1, 7] if edge == 2 else [0, edge, 7]

    return SimpleNamespace(candidates=Candidates(),
                           network=SimpleNamespace(free_flow_travel_seconds=
                                                   np.array([0., 10., 10., 20., 5., 20., 20., 0.])))


def test_alternate_rejects_suffix_returning_through_bottleneck(fake_runner):
    vehicle = {"current_edge": 0, "destination_edge": 7}
    assert experiment.alternate(fake_runner, vehicle, chosen=1, legal=[1, 2], bottleneck=1) is None
    assert fake_runner.candidates.route_calls == [(0, 2, 7)]


def test_alternate_rejects_more_than_180_seconds_of_added_planned_free_flow(fake_runner):
    vehicle = {"current_edge": 0, "destination_edge": 7}
    assert experiment.alternate(fake_runner, vehicle, chosen=1, legal=[1, 5], bottleneck=1) is None
    assert fake_runner.candidates.route_calls == []
    # The cap is inclusive: 280 - 100 = 180 seconds is admissible.
    assert experiment.alternate(fake_runner, vehicle, chosen=1, legal=[1, 6], bottleneck=1) == 6


def test_alternate_accepts_cheapest_legal_bypass_and_excludes_illegal_edges(fake_runner):
    vehicle = {"current_edge": 0, "destination_edge": 7}
    # Edge 2 is cheaper but returns through the bottleneck; edge 4 is cheaper
    # than edge 3 but cannot be chosen until it appears in the legal list.
    assert experiment.alternate(fake_runner, vehicle, chosen=1, legal=[3, 1, 2], bottleneck=1) == 3
    assert experiment.alternate(fake_runner, vehicle, chosen=1, legal=[3, 4, 1, 2], bottleneck=1) == 4
