import json

import numpy as np

from tools import compare_revisit_failures as module


def test_stall_excludes_late_departure_and_motion_hidden_by_revisit(monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'capacities', lambda city: np.ones(3))
    def vehicle(identity, departed=True, edge=0):
        return dict(vehicle_id=identity, departed=departed, arrived=False,
                    current_edge=edge, remaining_distance=10., in_junction=False,
                    actual_departure_time=0. if identity == 'early' else 590., departure_time=0.)
    states = [dict(time=t, vehicles=[vehicle('early'), vehicle('late', t==600, 1)]) for t in (0., 300., 600.)]
    events = [dict(vehicle_id='early', current_edge=0, decision_time=t, window_index=0) for t in (10., 40.)]
    flow = np.zeros((3, 3, 8), dtype=np.float32)
    flow[-1, 0, 2:4] = 6.
    summary = dict(metadata=dict(city='fake', episode=1, session_protocol='event', demand_sha256='hash'),
                   metrics=dict(seed=43, completion_rate=0., global_reward_sum=-5., microstep_seconds=1.),
                   deadline_seconds=600.)
    encode = lambda x: np.frombuffer(json.dumps(x).encode(), dtype=np.uint8)
    path = tmp_path/'episode.npz'
    np.savez_compressed(path, summary_json=encode(summary), states_json=encode(states),
                        decisions_json=encode(events), flow=flow)
    result = module.analyze(('old', 'fake', 'random', path, {'early': [('road0', 20.)]}))
    c = result['counts']
    assert c['departed_failed'] == 2
    assert c['failed_revisit'] == c['failed_stall300'] == c['failed_both'] == 1
    assert c['failed_stall600'] == 0
    assert c['failed_motionless_window_seconds'] == 300.
    assert c['failed_queue'] == c['stall300_preceding_connection_warning'] == 1
    assert result['failed_active_seconds'] == 610.


def test_warning_parser_filters_requested_demand_seeds(tmp_path):
    path = tmp_path/'log.txt'
    path.write_text("Warning: Vehicle 'ep43_v3' performs emergency stop at the end of lane 'road_x_2' because there is no connection to the next edge (decel=-12), time=40.00.\n"
                    "Warning: Vehicle 'ep44_v3' performs emergency stop at the end of lane 'road_0' because there is no connection to the next edge, time=50.00.\n")
    assert module.route_warnings(path, {43}) == {43: {'ep43_v3': [('road_x', 40.)]}}


def test_internal_junction_position_uses_internal_distance_and_lane():
    vehicle = dict(current_edge=4, in_junction=True, remaining_distance=0.,
                   junction_remaining_distance=8., junction_lane=':junction_1_0')
    assert module.physical_position(vehicle) == (4, True, ':junction_1_0', 8.)
    vehicle['junction_lane'] = ':junction_2_0'
    assert module.physical_position(vehicle)[:3] == (4, True, ':junction_2_0')
