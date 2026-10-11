import json

import numpy as np
import pytest

from test_event_imagination_protocol import network, seed
from tools.diagnose_component_executor import ExactStepRoute, RouteMismatch, replay_constraints, run_audit, shadow_step
from urban_dreamer.traffic.environment import NextHopDecision
from urban_dreamer.traffic.topology import NextHopCandidates
import urban_dreamer.learning.imagination as physical_engine


def test_exact_control_refuses_fallback_and_audits_hidden_route_entries():
    control = ExactStepRoute([NextHopDecision('v', 1, 3, 4.)], {'v': (1, 3)}, 4.)
    pending = [seed(current_edge=1)]
    assert control(pending, None, 4.)[0].next_edge == 3
    with pytest.raises(RouteMismatch, match='unmatched'):
        control(pending, None, 4.)
    with pytest.raises(RouteMismatch, match='physical entry'):
        control.entry(pending[0], 2)


def test_shadow_route_failure_restores_production_hook_and_does_not_mutate_seed():
    states = [seed(remaining_distance=0.)]
    original = physical_engine._record_physical_entry
    with pytest.raises(RouteMismatch, match='physical entry'):
        shadow_step(states, candidates=NextHopCandidates(network(), traffic_model='meso'),
            speeds=np.full(4, 10.), flow=np.zeros((4, 8)), timestamp=0., population=1,
            decisions=[], routes={'v': (0, 3)}, max_speed=10.)
    assert physical_engine._record_physical_entry is original
    assert states[0]['current_edge'] == 0


def test_real_meso_oracle_road_speed_does_not_replay_red_queue(tmp_path):
    pytest.importorskip('libsumo')
    output = tmp_path/'audit'
    report = run_audit(output, seconds=45, release_at=40)
    arm = report['arms']['realized_step_mean']
    assert arm['route_match_rate'] == 1.
    assert arm['road_mismatches'] > 0
    assert arm['position_valid_vehicle_steps'] > 0
    assert arm['active_seconds_max_abs'] == 0.
    witnesses = report['stopped_front_moving_behind_witnesses']
    assert witnesses
    witness = witnesses[0]
    assert witness['front']['road_before'] == witness['front']['true_road'] == 'A_hold'
    assert witness['front']['remaining_before'] == 0.
    assert witness['front']['predicted_road'] == 'A_to_R'
    assert witness['behind']['true_progress_same_road'] > 0.
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['physics']['traffic_model'] == 'meso'
    assert protocol['physics']['meso_junction_control'] is True
    assert protocol['physics']['meso_lane_queue'] is False
    assert len(protocol['network_sha256']) == 64
    assert len(protocol['sumo_version']) == 2
    repaired = tmp_path/'paired_constraint'
    comparison = replay_constraints(output, repaired)
    paired_rows = [json.loads(line) for line in (repaired/'rows.jsonl').read_text().splitlines()]
    original_rows = {(row['arm'], row['timestamp'], row['vehicle_id']): row
        for row in map(json.loads, (output/'rows.jsonl').read_text().splitlines())}
    # Default production path exactly reproduces the original unmodified audit.
    for row in paired_rows:
        if row['implementation'] != 'baseline':
            continue
        original = original_rows[(row['speed_arm'], row['timestamp'], row['vehicle_id'])]
        for key in ('predicted_road', 'true_road', 'predicted_remaining', 'true_remaining', 'road_match', 'arrival_match'):
            assert row[key] == original[key]
    for arm in ('boundary_last_step', 'realized_step_mean'):
        variants = {row['implementation']: row for row in comparison['evaluations'] if row['speed_arm'] == arm}
        assert variants['constrained']['red_hold_violations'] == 0
        assert variants['constrained']['road_mismatches'] < variants['baseline']['road_mismatches']
        assert variants['constrained']['route_matched_steps'] == variants['constrained']['steps']
        assert variants['baseline']['paired_position_steps'] == variants['constrained']['paired_position_steps']
    with pytest.raises(ValueError, match='preserve'):
        run_audit(output)
