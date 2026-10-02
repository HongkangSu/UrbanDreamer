import argparse
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from urban_dreamer.learning.baselines import METHODS, build_method
from urban_dreamer.learning.baselines import common
from urban_dreamer.learning.baselines.dta import DtaMethod
from urban_dreamer.learning.observations import build_actor_observations

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/train_toy_baseline.py'


def module():
    spec = importlib.util.spec_from_file_location('toy_baseline_tool', TOOL)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def make_args(tmp_path, **overrides):
    values = dict(method='dta', config=ROOT / 'configs/toy_cross_bottleneck.toml',
        output_dir=tmp_path, episodes=1, eval_every=1, eval_seeds=1, seed=42, device='cpu',
        threads=1, actor_lr=2e-4, critic_lr=3e-4, critic_updates=1, vehicle_limit=8, resume=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def make_method(tmp_path, **overrides):
    args = make_args(tmp_path, **overrides)
    runner = common.EpisodeRunner(args)
    return build_method('dta', args, runner), runner


def sumo_available():
    return shutil.which('sumo') is not None or (Path(sys.executable).parent / 'sumo').exists()


def fake_windows(network, congested, *, occupancy=.4, speed_ratio=.5, count=3):
    windows = []
    for _ in range(count):
        next_flow = np.zeros((network.edge_count, 8), dtype=np.float32)
        for edge in congested:
            next_flow[edge, 0] = occupancy
            next_flow[edge, 1] = speed_ratio
            next_flow[edge, 2] = occupancy
        windows.append(dict(next_flow=next_flow))
    return windows


def test_registry_and_placeholder_contract():
    assert METHODS['dta'] is DtaMethod
    with pytest.raises(NotImplementedError, match='not implemented yet'):
        build_method('dta', None, None)
    with pytest.raises(NotImplementedError, match='not implemented yet'):
        DtaMethod(argparse.Namespace(seed=42), None)


def test_shortest_paths_match_free_flow_cost_tables(tmp_path):
    method, runner = make_method(tmp_path)
    network = runner.network
    free_flow = network.free_flow_travel_seconds.astype(np.float64)
    assert method.od_groups, 'expected at least one OD group'
    for origin, destination in method.od_groups:
        path, dist = method._shortest_path(origin, destination, free_flow)
        assert path[0] == origin and path[-1] == destination
        for source, target in zip(path, path[1:]):
            assert target in network.adjacency_list[source]
        reference = runner.candidates.costs(destination)
        reachable = np.isfinite(reference)
        assert np.isfinite(dist[reachable]).all()
        assert np.allclose(dist[reachable], reference[reachable], rtol=1e-9, atol=1e-9)
        assert dist[origin] == pytest.approx(reference[origin], rel=1e-9)
        assert method._path_cost(path, free_flow) == pytest.approx(reference[origin], rel=1e-9)


def test_initial_path_sets_and_vehicle_assignment(tmp_path):
    method, runner = make_method(tmp_path, vehicle_limit=0)
    network = runner.network
    for (origin, destination), group in method.od_groups.items():
        assert sum(group['path_fractions'].values()) == pytest.approx(1.)
        assert len(group['path_fractions']) == 1
        path = next(iter(group['path_fractions']))
        assert path[0] == origin and path[-1] == destination
        for source, target in zip(path, path[1:]):
            assert target in network.adjacency_list[source]
    assert set(method.assignment) == {v.vehicle_id for v in runner.topology.vehicles}
    for vehicle in runner.topology.vehicles:
        origin = network.edge_to_index[runner.topology.origins[vehicle.origin_index]]
        destination = network.edge_to_index[runner.topology.destinations[vehicle.destination_index]]
        path = method.assignment[vehicle.vehicle_id]
        assert path[0] == origin and path[-1] == destination


def test_select_follows_assignment_and_returns_legal_candidates(tmp_path):
    method, runner = make_method(tmp_path)
    network = runner.network
    vehicles = []
    for vehicle in runner.topology.vehicles[:6]:
        origin = network.edge_to_index[runner.topology.origins[vehicle.origin_index]]
        destination = network.edge_to_index[runner.topology.destinations[vehicle.destination_index]]
        assert origin != destination
        vehicles.append(dict(vehicle_id=vehicle.vehicle_id, current_edge=origin,
            destination_edge=destination, remaining_distance=float(network.lengths_m[origin])))
    obs = build_actor_observations(policy_edge_features=np.zeros((network.edge_count, 2), dtype=np.float32),
        vehicle_states=vehicles, candidates=runner.candidates,
        episode_progress=0., active_vehicle_fraction=0.5)
    indices = method.select(vehicles, obs, np.zeros((network.edge_count, 8), dtype=np.float32), 0., explore=False)
    assert len(indices) == len(vehicles)
    for row, (vehicle, index) in enumerate(zip(vehicles, indices)):
        assert obs.mask[row, index]
        assigned = method.assignment[vehicle['vehicle_id']]
        assert int(obs.edge_indices[row, index]) == assigned[1]
    detoured = dict(vehicles[0])
    assigned = method.assignment[detoured['vehicle_id']]
    off_path = next(edge for edge in range(network.edge_count)
                    if edge not in assigned and edge != detoured['destination_edge']
                    and runner.candidates.candidates(edge, detoured['destination_edge']))
    detoured['current_edge'] = off_path
    detoured['remaining_distance'] = float(network.lengths_m[off_path])
    obs = build_actor_observations(policy_edge_features=np.zeros((network.edge_count, 2), dtype=np.float32),
        vehicle_states=[detoured], candidates=runner.candidates,
        episode_progress=0., active_vehicle_fraction=0.5)
    (index,) = method.select([detoured], obs, np.zeros((network.edge_count, 8), dtype=np.float32), 0., explore=True)
    assert obs.mask[0, index]


def test_msa_updates_marginal_costs_and_converges(tmp_path):
    method, runner = make_method(tmp_path)
    network = runner.network
    free_flow = network.free_flow_travel_seconds.astype(np.float64)
    origin, destination = next(iter(method.od_groups))
    congested = sorted(set(next(iter(method.od_groups[(origin, destination)]['path_fractions']))[1:-1]))
    assert congested
    windows = fake_windows(network, congested)
    first = method.on_episode([], windows, episode=1)
    assert first['dta_iteration'] == 1
    assert math.isfinite(first['mc_gap']) and first['mc_gap'] >= 0.
    assert first['assignment_changes'] >= 0
    assert math.isfinite(first['mean_estimated_edge_seconds'])
    assert first['active_paths'] >= 1
    for edge in congested:
        realized = float(network.lengths_m[edge]) / max(.5 * float(network.speed_limits_mps[edge]), .5)
        assert method.realized_seconds[edge] == pytest.approx(realized)
        assert method.edge_cost[edge] > free_flow[edge]
        assert method.edge_cost[edge] >= method.realized_seconds[edge] - 1e-9
    untouched = next(edge for edge in range(network.edge_count) if edge not in congested)
    assert method.realized_seconds[untouched] == pytest.approx(free_flow[untouched])
    for group in method.od_groups.values():
        assert sum(group['path_fractions'].values()) == pytest.approx(1.)
    for episode in range(2, 41):
        metrics = method.on_episode([], windows, episode=episode)
        for group in method.od_groups.values():
            assert sum(group['path_fractions'].values()) == pytest.approx(1.)
            assert all(fraction >= 0. for fraction in group['path_fractions'].values())
        if method.converged:
            break
    assert method.converged
    assert method.stable_count >= 10
    snapshot = dict(method.assignment)
    metrics = method.on_episode([], windows, episode=41)
    assert metrics['assignment_changes'] == 0
    assert method.assignment == snapshot


def test_dta_checkpoint_roundtrip(tmp_path):
    method, runner = make_method(tmp_path)
    windows = fake_windows(runner.network, sorted({0, 1}))
    method.on_episode([], windows, episode=1)
    original = method.state_dict()
    path = tmp_path / 'ckpt.pt'
    common.save_checkpoint(path, method_name='dta', episode=1, best=2.5,
        method_state=method.state_dict(), args=runner.args, prepared=runner.prepared, config=runner.config)
    method.on_episode([], windows, episode=2)
    assert method.state_dict()['iteration'] == 2
    state = common.load_checkpoint(path, method_name='dta',
        prepared=runner.prepared, config=runner.config, device=runner.device)
    method.load_state_dict(state['method_state'])
    restored = method.state_dict()
    assert restored['iteration'] == original['iteration']
    assert restored['assignment'] == original['assignment']
    assert restored['path_sets'] == original['path_sets']
    assert np.allclose(restored['edge_cost'], original['edge_cost'])
    assert np.allclose(restored['volume_ema'], original['volume_ema'])
    assert method.rng.bit_generator.state == original['rng']


def split_od(method, runner):
    key, group = next(iter(method.od_groups.items()))
    original = next(iter(group['path_fractions']))
    cost = method.edge_cost.copy()
    cost[list(original[1:-1])] *= 1000.
    alternate, _ = method._shortest_path(*key, cost)
    assert alternate != original
    group['path_fractions'] = {original: .25, alternate: .75}
    return key, group


def test_poisson_assignments_replace_ids_and_respect_realized_ods(tmp_path):
    method, runner = make_method(tmp_path, vehicle_limit=0)
    split_od(method, runner)
    vehicles = tuple(replace(v, vehicle_id=f'fresh_{i}')
                     for i, v in enumerate(runner.topology.vehicles[::3]))
    topology = replace(runner.topology, vehicles=vehicles)
    method.begin_episode(topology, seed=13)
    assert set(method.assignment) == {v.vehicle_id for v in vehicles}
    for vehicle in vehicles:
        path = method.assignment[vehicle.vehicle_id]
        assert path[0] == runner.network.edge_to_index[topology.origins[vehicle.origin_index]]
        assert path[-1] == runner.network.edge_to_index[topology.destinations[vehicle.destination_index]]
    assignment = dict(method.assignment)
    method.begin_episode(topology, seed=13)
    assert method.assignment == assignment
    method.begin_episode(replace(topology, vehicles=()), seed=14)
    assert method.assignment == {}
    assert method.on_episode([], [], episode=1)['mc_gap'] >= 0.


def test_single_vehicle_poisson_od_samples_learned_fractions(tmp_path):
    method, runner = make_method(tmp_path, vehicle_limit=0)
    _, group = split_od(method, runner)
    vehicle = replace(group['vehicles'][0], vehicle_id='new_vehicle')
    topology = replace(runner.topology, vehicles=(vehicle,))
    target = next(path for path, fraction in group['path_fractions'].items() if fraction == .25)
    count = 0
    for seed in range(800):
        method.begin_episode(topology, seed=seed)
        count += method.assignment[vehicle.vehicle_id] == target
    assert count / 800 == pytest.approx(.25, abs=.04)


def test_poisson_existing_vehicle_choice_is_independent_of_future_demand(tmp_path):
    method, runner = make_method(tmp_path, vehicle_limit=0)
    _, group = split_od(method, runner)
    early = replace(group['vehicles'][0], vehicle_id='early', departure_second=0.)
    future = tuple(replace(early, vehicle_id=f'future_{i}', departure_second=100. + i)
                   for i in range(20))
    alone = replace(runner.topology, vehicles=(early,))
    expanded = replace(runner.topology, vehicles=(early, *future))
    reordered = replace(runner.topology, vehicles=tuple(reversed(expanded.vehicles)))
    for seed in range(50):
        method.begin_episode(alone, seed=seed)
        selected = method.assignment['early']
        method.begin_episode(expanded, seed=seed)
        assignment = dict(method.assignment)
        assert assignment['early'] == selected
        method.begin_episode(reordered, seed=seed)
        assert method.assignment == assignment
        # Even a different departure schedule is not an assignment input.
        method.begin_episode(replace(alone, vehicles=(replace(early, departure_second=300.),)), seed=seed)
        assert method.assignment['early'] == selected


def test_eval_assignment_does_not_change_learning_or_next_training(tmp_path):
    method, runner = make_method(tmp_path, vehicle_limit=0)
    split_od(method, runner)
    topology = replace(runner.topology, vehicles=runner.topology.vehicles[::2])
    method.begin_episode(topology, seed=100)
    windows = fake_windows(runner.network, [0, 1])
    method.on_episode([], windows, episode=1)
    state = method.state_dict()
    control, _ = make_method(tmp_path / 'control', vehicle_limit=0)
    control.load_state_dict(state)
    method.begin_episode(replace(topology, vehicles=topology.vehicles[:1]), seed=900)
    assert method.state_dict()['path_sets'] == state['path_sets']
    assert np.array_equal(method.edge_cost, state['edge_cost'])
    assert method.rng.bit_generator.state == state['rng']
    for obj in (method, control):
        obj.begin_episode(topology, seed=101)
    assert method.assignment == control.assignment
    assert method.on_episode([], windows, episode=2) == control.on_episode([], windows, episode=2)
    assert method.state_dict() == control.state_dict()


def test_poisson_gap_uses_reference_weights_and_never_stops_on_ids(tmp_path):
    method, runner = make_method(tmp_path, vehicle_limit=0)
    split_od(method, runner)
    method.iteration = 10
    empty = replace(runner.topology, vehicles=())
    method.begin_episode(empty, seed=100)
    metrics = method.on_episode([], [], episode=11)
    weighted = sum(len(group['vehicles']) * (
        sum(fraction * method._path_cost(path, method.edge_cost)
            for path, fraction in group['path_fractions'].items())
        - min(method._path_cost(path, method.edge_cost) for path in group['path_fractions']))
        for group in method.od_groups.values())
    assert metrics['mc_gap'] == pytest.approx(weighted / len(runner.topology.vehicles))
    assert metrics['path_fraction_change'] > 0.
    for episode in range(12, 30):
        method.begin_episode(empty, seed=episode)
        metrics = method.on_episode([], [], episode=episode)
        assert metrics['assignment_changes'] == 0
        assert not method.converged
        assert method.stable_count == 0
    state = method.state_dict()
    restored, _ = make_method(tmp_path / 'restore', vehicle_limit=0)
    restored.load_state_dict(state)
    assert restored.poisson_mode
    assert restored.state_dict() == state


def test_dta_cli_smoke(tmp_path):
    if not sumo_available():
        pytest.skip('SUMO unavailable')
    tool = module()
    output = tmp_path / 'run'
    status = tool.main(['--method', 'dta', '--output-dir', str(output), '--episodes', '3',
                        '--eval-every', '2', '--eval-seeds', '1', '--vehicle-limit', '8'])
    assert status == 0
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    train = [r for r in records if r['phase'] == 'train']
    assert 1 <= len(train) <= 3
    for record in train:
        assert math.isfinite(record['global_reward_sum'])
        assert math.isfinite(record['mc_gap']) and record['mc_gap'] >= 0.
        assert math.isfinite(record['mean_estimated_edge_seconds'])
        assert record['dta_iteration'] >= 1
        assert record['assignment_changes'] >= 0
        assert record['active_paths'] >= 1
    assert {r['policy'] for r in records if r['phase'] == 'baseline'} == {'random', 'shortest', 'deterministic'}
    assert any(r['phase'] == 'evaluation' for r in records)
    assert any(r['phase'] == 'holdout_final' for r in records)
    assert any(r['phase'] == 'holdout_best' for r in records)
    for record in records:
        assert math.isfinite(record['global_reward_sum'])
    protocol = json.loads((output / 'protocol.json').read_text())
    assert protocol['method'] == 'dta'
    for name in ('best.pt', 'initial.pt', 'latest.pt'):
        assert (output / name).is_file()
