"""Static routes remain identical while independent processes reuse cold searches."""
from dataclasses import replace
import multiprocessing
import pickle
import sqlite3
import time
import warnings

import numpy as np
import pytest

from urban_dreamer.traffic.route_candidates import RouteCandidates
from urban_dreamer.traffic.shared_route_cache import SharedRouteCache, network_identity
from urban_dreamer.traffic.topology import TopologyError


def _network(*args):
    # Import lazily so spawn cache-only workers do not load the learning package.
    from test_route_candidates import network
    return network(*args)


def _paths(current, destination):
    return ((current, destination),) * 4


def _unexpected_compute(*args):
    raise AssertionError('a persisted shared hit must avoid route computation')


def _cache_worker(cache, keys, ready, counter, results):
    """Module-level spawn target; the cache itself must survive pickling."""
    if not ready.wait(30.):
        raise TimeoutError('parent did not release cache workers')

    def compute(current, destination):
        with counter.get_lock():
            counter.value += 1
        time.sleep(.05)
        return _paths(current, destination)

    values = [cache.get_or_compute(*key, compute) for key in keys]
    results.put((values, cache.statistics()))
    cache.close()


def _run_workers(cache, batches):
    context = multiprocessing.get_context('spawn')
    ready, counter, results = context.Event(), context.Value('i', 0), context.Queue()
    processes = [context.Process(target=_cache_worker,
        args=(cache, keys, ready, counter, results)) for keys in batches]
    try:
        for process in processes:
            process.start()
        ready.set()
        for process in processes:
            process.join(40.)
            assert not process.is_alive(), 'shared-cache worker deadlocked'
            assert process.exitcode == 0
        return counter.value, [results.get(timeout=5.) for _ in processes]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5.)
        results.close()
        results.join_thread()


def test_independent_trainer_and_session_candidates_reuse_dijkstra(tmp_path, monkeypatch):
    monkeypatch.setenv('URB_ROUTE_CACHE_DIR', str(tmp_path))
    trainer = RouteCandidates(_network(), traffic_model='meso')
    expected = trainer.generate_paths(0, 4)
    # SUMO sessions create their own candidates instance on each episode.
    session = RouteCandidates(_network(), traffic_model='micro')
    monkeypatch.setattr(session, '_shortest_path', _unexpected_compute)
    assert session.generate_paths(0, 4) == expected
    assert session._shared_route_cache.path == trainer._shared_route_cache.path
    assert session._shared_route_cache.statistics()['hits'] == 1
    trainer._shared_route_cache.close()
    session._shared_route_cache.close()


def test_spawn_workers_compute_one_cold_key_once_then_reuse_after_restart(tmp_path):
    cache = SharedRouteCache(tmp_path, 'same-network')
    count, values = _run_workers(cache, [[(1, 4)]] * 4)
    assert count == 1
    assert all(value == [_paths(1, 4)] for value, _ in values)
    assert all(statistics['entries'] == 1 for _, statistics in values)
    count, values = _run_workers(SharedRouteCache(tmp_path, 'same-network'), [[(1, 4)]] * 2)
    assert count == 0
    assert all(statistics['hits'] == 1 for _, statistics in values)
    cache.close()


def test_concurrent_distinct_keys_keep_strict_capacity_and_metadata(tmp_path):
    cache = SharedRouteCache(tmp_path, 'small-store', capacity=3)
    batches = [[(index, index + 100) for index in range(start, start + 4)]
               for start in (0, 4, 8, 12)]
    count, values = _run_workers(cache, batches)
    assert count == 16
    assert all(statistics['entries'] <= 3 for _, statistics in values)
    assert cache.statistics()['entries'] == 3
    with sqlite3.connect(cache.path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM routes').fetchone()[0] == 3
        assert connection.execute('SELECT entries FROM size WHERE id=1').fetchone()[0] == 3
    cache.close()


def test_shared_fifo_hits_do_not_change_eviction_order(tmp_path):
    cache = SharedRouteCache(tmp_path, 'fifo', capacity=2)
    cache.get_or_compute(0, 4, _paths)
    cache.get_or_compute(1, 4, _paths)
    cache.get_or_compute(0, 4, _unexpected_compute)
    cache.get_or_compute(2, 4, _paths)
    with sqlite3.connect(cache.path) as connection:
        assert connection.execute('SELECT current,destination FROM routes ORDER BY sequence').fetchall() == [
            (1, 4), (2, 4)]
    cache.close()


@pytest.mark.parametrize('field', ['adjacency_order', 'edge_ids', 'passenger_allowed', 'costs'])
def test_network_identity_separates_every_static_routing_input(field):
    graph = _network()
    if field == 'adjacency_order':
        altered = replace(graph, adjacency_list=((1,), (3, 2), (4,), (4,), ()))
    elif field == 'edge_ids':
        altered = replace(graph, edge_ids=('another-edge', *graph.edge_ids[1:]))
    elif field == 'passenger_allowed':
        allowed = graph.passenger_allowed.copy()
        allowed[2] = False
        altered = replace(graph, passenger_allowed=allowed)
    else:
        costs = graph.free_flow_travel_seconds.copy()
        costs[2] = np.nextafter(costs[2], np.inf)
        altered = replace(graph, free_flow_travel_seconds=costs)
    assert network_identity(graph) != network_identity(altered)
    assert network_identity(graph) == network_identity(_network())


def test_adjacency_order_isolated_preserves_tie_breaking(tmp_path, monkeypatch):
    monkeypatch.setenv('URB_ROUTE_CACHE_DIR', str(tmp_path))
    graphs = [_network(((2, 1), (3,), (3,), ())), _network(((1, 2), (3,), (3,), ()))]
    candidates = []
    for graph in graphs:
        graph.free_flow_travel_seconds[:] = 1.
        candidates.append(RouteCandidates(graph))
    assert candidates[0].generate_paths(0, 3)[0] == (0, 2, 3)
    assert candidates[1].generate_paths(0, 3)[0] == (0, 1, 3)
    assert candidates[0]._shared_route_cache.path != candidates[1]._shared_route_cache.path
    for item in candidates:
        item._shared_route_cache.close()


def test_algorithm_version_invalidates_persisted_network_identity(monkeypatch):
    import urban_dreamer.traffic.shared_route_cache as shared
    before = network_identity(_network())
    monkeypatch.setattr(shared, 'ALGORITHM_VERSION', shared.ALGORITHM_VERSION + '_changed')
    assert before != network_identity(_network())


def test_shared_cold_and_warm_routes_preserve_history_and_rng(tmp_path, monkeypatch):
    from test_route_candidates import historical_paths
    monkeypatch.setenv('URB_ROUTE_CACHE_DIR', str(tmp_path))
    graph = _network()
    candidates = RouteCandidates(graph)
    before = np.random.get_state()
    for source, destination in [(0, 4), (1, 4), (2, 4), (4, 4)]:
        expected = historical_paths(graph, source, destination)
        assert candidates.generate_paths(source, destination) == expected
        assert candidates.generate_paths(source, destination) == expected
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    statistics = candidates._shared_route_cache.statistics()
    assert statistics['computations'] == statistics['hits'] == statistics['entries'] == 4
    assert not candidates._route_candidate_cache
    candidates._shared_route_cache.close()


def test_unreachable_topology_error_remains_visible_without_cache_warning(tmp_path, monkeypatch):
    monkeypatch.setenv('URB_ROUTE_CACHE_DIR', str(tmp_path))
    candidates = RouteCandidates(_network())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(TopologyError, match='no destination-reachable route'):
            candidates.generate_paths(4, 0)
    assert not caught
    assert candidates._shared_route_cache.statistics()['entries'] == 0
    candidates._shared_route_cache.close()


def test_pickle_reopens_connection_without_serializing_handles(tmp_path):
    cache = SharedRouteCache(tmp_path, 'pickle')
    cache.get_or_compute(1, 4, _paths)
    restored = pickle.loads(pickle.dumps(cache))
    assert restored.path == cache.path
    assert restored.get_or_compute(1, 4, _unexpected_compute) == _paths(1, 4)
    restored.close()
    assert cache.get_or_compute(1, 4, _unexpected_compute) == _paths(1, 4)
    cache.close()


def test_unwritable_cache_warns_and_preserves_computed_result(tmp_path, monkeypatch):
    cache = SharedRouteCache(tmp_path, 'permission-denied')

    def denied():
        raise PermissionError('cache directory is not writable')

    monkeypatch.setattr(cache, '_open', denied)
    with pytest.warns(RuntimeWarning, match='Shared route cache unavailable'):
        assert cache.get_or_compute(1, 4, _paths) == _paths(1, 4)
    assert cache.get_or_compute(2, 4, _paths) == _paths(2, 4)
    cache.close()


def test_corruption_detected_after_key_lock_warns_and_uses_explicit_fallback(tmp_path, monkeypatch):
    cache = SharedRouteCache(tmp_path, 'second-read-corruption')
    original_get = cache._get
    reads = []

    def corrupt_second_read(current, destination):
        reads.append((current, destination))
        if len(reads) == 2:
            raise ValueError('payload became invalid while waiting for the key lock')
        return original_get(current, destination)

    monkeypatch.setattr(cache, '_get', corrupt_second_read)
    with pytest.warns(RuntimeWarning, match='Shared route cache unavailable'):
        assert cache.get_or_compute(1, 4, _unexpected_compute, fallback=_paths) == _paths(1, 4)
    assert reads == [(1, 4), (1, 4)]
    cache.close()


@pytest.mark.parametrize('failure', ['unavailable', 'corrupt'])
def test_persistent_shared_failure_reuses_local_routes_after_one_compute(tmp_path, monkeypatch, failure):
    monkeypatch.setenv('URB_ROUTE_CACHE_DIR', str(tmp_path))
    candidates = RouteCandidates(_network())
    expected = candidates.generate_paths(0, 4)
    assert not candidates._route_candidate_cache
    cache = candidates._shared_route_cache
    if failure == 'corrupt':
        with sqlite3.connect(cache.path) as connection:
            connection.execute('UPDATE routes SET paths=? WHERE current=0 AND destination=4', ('invalid',))
    else:
        def denied():
            raise PermissionError('shared database remains inaccessible')
        monkeypatch.setattr(cache, '_open', denied)
    original_compute = candidates._compute_paths
    computations = []

    def record_compute(current, destination):
        computations.append((current, destination))
        return original_compute(current, destination)

    monkeypatch.setattr(candidates, '_compute_paths', record_compute)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        first = candidates.generate_paths(0, 4)
        assert first == expected
        for _ in range(3):
            assert candidates.generate_paths(0, 4) is first
    assert computations == [(0, 4)]
    assert tuple(candidates._route_candidate_cache) == ((0, 4),)
    assert len(caught) == 1
    assert caught[0].category is RuntimeWarning
    assert 'Shared route cache unavailable' in str(caught[0].message)
    cache.close()


@pytest.mark.parametrize('payload', ['not-json', '[[1,3]]', '[[1,true,4],[1,4],[1,4],[1,4]]'])
def test_corrupt_payload_warns_and_recomputes_correct_routes(tmp_path, monkeypatch, payload):
    monkeypatch.setenv('URB_ROUTE_CACHE_DIR', str(tmp_path))
    candidates = RouteCandidates(_network())
    expected = candidates.generate_paths(0, 4)
    with sqlite3.connect(candidates._shared_route_cache.path) as connection:
        connection.execute('UPDATE routes SET paths=? WHERE current=0 AND destination=4', (payload,))
    with pytest.warns(RuntimeWarning, match='Shared route cache unavailable'):
        assert candidates.generate_paths(0, 4) == expected
    candidates._shared_route_cache.close()


def test_default_local_cache_preserves_historical_routes_rng_and_lru(monkeypatch):
    from test_route_candidates import historical_paths
    monkeypatch.delenv('URB_ROUTE_CACHE_DIR', raising=False)
    candidates = RouteCandidates(_network())
    assert candidates._shared_route_cache is None
    assert candidates._route_candidate_cache_limit == 200000
    candidates._route_candidate_cache_limit = 2
    rng_before = np.random.get_state()
    first = candidates.generate_paths(0, 4)
    assert first == historical_paths(candidates.network, 0, 4)
    candidates.generate_paths(1, 4)
    assert candidates.generate_paths(0, 4) is first
    candidates.generate_paths(2, 4)
    assert tuple(candidates._route_candidate_cache) == ((0, 4), (2, 4))
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]


def test_shared_routes_preserve_parallel_training_actions_losses_gradients_and_rng(tmp_path, monkeypatch):
    import torch
    from test_parallel_imagination import collect_result
    from test_route_training import route_trainer
    from urban_dreamer.learning.parallel_imagination import close_parallel_imagination

    monkeypatch.setattr('urban_dreamer.learning.training.IMAGINED_HORIZON', 3)
    trainers = []
    try:
        monkeypatch.delenv('URB_ROUTE_CACHE_DIR', raising=False)
        reference = route_trainer(rollouts=2)
        trainers.append(reference)
        expected = collect_result(reference, workers=1)
        monkeypatch.setenv('URB_ROUTE_CACHE_DIR', str(tmp_path))
        shared = route_trainer(rollouts=2)
        trainers.append(shared)
        actual = collect_result(shared, workers=2)
        assert expected[2] == actual[2]
        for key, value in expected[0].items():
            torch.testing.assert_close(value, actual[0][key], rtol=0, atol=0)
        assert len(expected[1]) == len(actual[1])
        for before, after in zip(expected[1], actual[1]):
            assert torch.isfinite(after).all()
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)
        torch.testing.assert_close(expected[4], actual[4], rtol=0, atol=0)
        # Worker writes must be visible to the parent without an in-memory copy.
        assert shared.candidates._shared_route_cache.statistics()['entries'] > 0
        assert not shared.candidates._route_candidate_cache
    finally:
        for trainer in trainers:
            close_parallel_imagination(trainer)
            if trainer.candidates._shared_route_cache is not None:
                trainer.candidates._shared_route_cache.close()
