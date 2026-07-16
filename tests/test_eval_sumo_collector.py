import numpy as np

from scripts.eval_sumo_collector import _uniform_unique_route_probs


def test_uniform_unique_route_probs_equalizes_physical_paths_with_duplicate_slots():
    routes = [
        ["start", "left", "dest"],
        ["start", "right", "dest"],
        ["start", "left", "dest"],
        ["start", "left", "dest"],
    ]

    probs = _uniform_unique_route_probs(routes, n_routes=4)

    np.testing.assert_allclose(probs, [1.0 / 6.0, 0.5, 1.0 / 6.0, 1.0 / 6.0])
    assert np.isclose(probs[[0, 2, 3]].sum(), probs[1])


def test_uniform_unique_route_probs_remains_normalized_for_one_physical_path():
    routes = [["same"] for _ in range(4)]

    probs = _uniform_unique_route_probs(routes, n_routes=4)

    np.testing.assert_allclose(probs, np.full(4, 0.25, dtype=np.float32))
    assert np.isclose(probs.sum(), 1.0)
