from types import SimpleNamespace

import numpy as np

from tools.diagnose_episode_stalls import describe


def test_stationary_creeping_segment_progress_and_recent_entry_are_distinct():
    times = np.arange(2610., 4411.)
    data = np.zeros((len(times), 4, 6), dtype=float)
    data[:, :, 0] = 1
    data[:, :, 3] = 5
    data[:, :, 4] = 100
    # Vehicle1 moves only0.1m each second:600s of cumulative motion must
    # not be mistaken for a stall based on a per-step1m tolerance.
    data[:, 1, 3] += .1*(times-2610)
    data[:, 1, 4] += .1*(times-2610)
    data[:, 1, 5] = .1
    # Vehicle2 physically changes segment, independently of interpolation.
    data[times >= 3500, 2, 2] = 1
    data[times >= 3500, 2, 4] = 200
    # Vehicle3 entered only510s before the original deadline.
    data[times < 3300, 3] = np.nan
    data[times < 3300, 3, 0] = 0
    vehicles = [dict(vehicle_id=f'v{i}', current_edge=0, departed=True,
        arrived=False, actual_departure_time=3300 if i == 3 else 1000)
        for i in range(4)]
    baseline = dict(vehicles=vehicles, unfinished=4, never_departed=0,
        last_decisions={}, reference_checks={}, edge_traffic={
            'edge':dict(mean_speed_mps=0,halting=4,vehicles=4,lanes=1)})
    probe = SimpleNamespace(samples=list(data), times=times, original_deadline=3810.,
        baseline=baseline, final=dict(time=4410.,travel_times={},unfinished=4,never_departed=0),
        ids=[f'v{i}' for i in range(4)], id_index={f'v{i}':i for i in range(4)},
        network=SimpleNamespace(edge_ids=['edge'],edge_to_index={'edge':0},lengths_m=[1000]),
        series=[])
    summary, rows, _, _ = describe(probe)
    assert rows[0]['stationary_600']
    assert not rows[1]['stationary_600']
    assert rows[1]['interpolated_movement_same_segment_600']
    assert rows[1]['odometer_delta_600_m'] == 60
    assert rows[2]['structural_progress_600']
    assert not rows[2]['stationary_600']
    assert rows[3]['entered_last600']
    assert not rows[3]['active_continuously_600']
    assert summary['counts']['stationary_600'] == 1
    assert summary['counts']['structural_progress_600'] == 1
    assert summary['counts']['entered_last600'] == 1
