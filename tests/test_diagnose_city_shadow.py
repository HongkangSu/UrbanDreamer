from copy import deepcopy

import numpy as np

from test_event_imagination_protocol import network, seed
from tools.diagnose_city_shadow import comparison_rows, run_shadow, summarize
from urban_dreamer.learning import imagination as physical_engine
from urban_dreamer.traffic.topology import NextHopCandidates


class Controls:
    def state(self, movement, tick):
        return 'G'


def test_shadow_preserves_route_failures_and_restores_entry_hook():
    graph = network()
    states = [seed('one', current_edge=1, remaining_distance=0., committed_next_edge=3)]
    original = physical_engine._record_physical_entry
    result = run_shadow(states, implementation='legacy', candidates=NextHopCandidates(graph, traffic_model='meso'),
        controls=Controls(), speeds=np.full(4, 10.), flow=np.zeros((4, 8)), timestamp=0,
        population=1, decisions=[], routes={'one': (1, 2, 3)}, max_speed=40., seed=610004)
    assert result['error']['type'] == 'RouteMismatch'
    assert physical_engine._record_physical_entry is original
    assert states[0]['current_edge'] == 1


def test_comparison_rows_match_ids_not_input_order_and_retain_failures():
    before = [seed('a'), seed('b')]
    truth = {s['vehicle_id']: deepcopy(s) for s in before}
    predicted = deepcopy(list(reversed(before)))
    predicted[0]['remaining_distance'] += 3.
    shadow = dict(states=predicted, error=None)
    rows = comparison_rows(before, truth, shadow, timestamp=30, speed_arm='boundary_causal', implementation='city')
    assert [row['vehicle_id'] for row in rows] == ['a', 'b']
    assert [row['remaining_error_m'] for row in rows] == [0., 3.]
    shadow['error'] = dict(type='ValueError', message='tied positions')
    rows = comparison_rows(before, truth, shadow, timestamp=30, speed_arm='boundary_causal', implementation='city')
    assert len(rows) == 2 and all(row['status'] == 'error' for row in rows)
    assert all('predicted_road' not in row for row in rows)


def test_common_metrics_do_not_drop_failed_coverage_or_mix_position_denominators():
    windows, rows = [], []
    for timestamp in [0, 30]:
        for implementation in ['legacy', 'city']:
            failed = timestamp == 30 and implementation == 'city'
            error = dict(type='ValueError', message='ambiguous FIFO') if failed else None
            windows.append(dict(timestamp=timestamp, speed_arm='boundary_causal', implementation=implementation,
                status='error' if failed else 'ok', error=error, cohort=1, counters={}, active_seconds_error=0.))
            rows.append(dict(timestamp=timestamp, vehicle_id='v', speed_arm='boundary_causal', implementation=implementation,
                status='error' if failed else 'ok', road_match=implementation == 'legacy', arrival_match=True,
                remaining_error_m=2. if implementation == 'legacy' else None))
    report = summarize(windows, rows)
    assert report['arms']['boundary_causal/city']['failed_windows'] == 1
    assert report['arms']['boundary_causal/city']['attempted_vehicle_rows'] == 2
    assert report['arms']['boundary_causal/city']['error_types'] == {'ValueError': 1}
    common = report['common_rows']['boundary_causal']
    assert common['common_success_rows'] == 1
    assert common['common_position_rows'] == 0
    assert common['implementations']['legacy']['common_position_mae_m'] is None
