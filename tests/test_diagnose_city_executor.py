import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_city_imagination import table
from test_event_imagination_protocol import network, seed
from tools import diagnose_city_executor as diagnostic
from tools.component_experiment_data import sha256
from urban_dreamer.learning import imagination as runtime
from urban_dreamer.traffic.movement_controls import UnsupportedControl
from urban_dreamer.traffic.topology import NextHopCandidates


def xml_fixture(path):
    path.write_text('<net><tlLogic id="A2" type="static" offset="0" programID="0">'
                    '<phase duration="3600" state="rGG"/></tlLogic></net>')


def test_declared_override_is_materialized_without_vehicle_truth(tmp_path):
    source, resolved = tmp_path / 'source.xml', tmp_path / 'resolved.xml'
    xml_fixture(source)
    before = source.read_bytes()
    declaration = diagnostic.resolve_declared_override(source, resolved, release_at=40, horizon=70)
    from urban_dreamer.learning.meso_constraints import StaticSignalSchedule
    program = StaticSignalSchedule.from_network(resolved, [], no_external_overrides=True)
    program.links[(0, 1)] = {('A2', 0)}
    assert not program.permits(0, 1, 39.999)
    assert program.permits(0, 1, 40.)
    assert source.read_bytes() == before
    assert declaration['valid_through'] == 70.
    assert declaration['authority'].startswith('immutable source protocol')


def test_city_shadow_checks_every_entry_and_restores_runtime_hook_on_error():
    topology = network()
    original = runtime._record_physical_entry
    with pytest.raises(diagnostic.RouteMismatch, match='differs from commanded route'):
        diagnostic.city_shadow_step([seed()], candidates=NextHopCandidates(topology, traffic_model='meso'),
            speeds=np.full(4, 10.), timestamp=0., population=1, decisions=[], routes={'v': [0, 2]},
            max_speed=40., controls=table(topology), seed=1)
    assert runtime._record_physical_entry is original


def archive(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    xml_fixture(source / 'combined_queue.net.xml')
    topology = network()
    records = []
    for tick in (1, 2):
        records.append(dict(timestamp=tick, states=[seed(current_edge=3, remaining_distance=30. - tick * 10.,
            committed_next_edge=None)], controls=[], commanded_route_suffixes={'v': [3]}, edge_mean_speed_mps=[10.] * 4))
    (source / 'real_seconds.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    protocol = dict(network_sha256=sha256(source / 'combined_queue.net.xml'),
        registry_sha256=sha256(diagnostic.ROOT / 'networks/toy_cross_bottleneck/edge_registry.json'),
        edge_ids=list(topology.edge_ids), seconds=2, release_at=1,
        simulation_parameters={'vehicle_max_speed_mps': 40.}, physics={}, sumo_version=[])
    (source / 'protocol.json').write_text(json.dumps(protocol))
    monkeypatch.setattr(diagnostic, 'load_traffic_topology', lambda prepared: SimpleNamespace(network=topology, vehicles=(1,)))
    controls = table(topology)
    controls.preflight = lambda: {'scope': 'synthetic test controls'}
    monkeypatch.setattr(diagnostic.CityControlTable, 'from_network', lambda *args, **kwargs: controls)
    return source


def test_paired_replay_preserves_inputs_and_uses_common_position_subset(tmp_path, monkeypatch):
    source = archive(tmp_path, monkeypatch)
    hashes = {p.name: sha256(p) for p in source.iterdir()}
    output = tmp_path / 'audit'
    result = diagnostic.replay_audit(source, output)
    assert result['status'] == 'complete'
    assert result['source_pairs'] == 1
    assert result['attempted_windows'] == result['attempted_vehicle_rows'] == 6
    for row in result['evaluations']:
        assert row['route_matched_steps'] == row['common_position_steps'] == 1
        assert row['common_position_mae_m'] == 0.
    assert hashes == {p.name: sha256(p) for p in source.iterdir()}
    with pytest.raises(ValueError, match='preserve previous audits'):
        diagnostic.replay_audit(source, output)


def test_unsupported_attempts_and_vehicle_rows_are_not_dropped(tmp_path, monkeypatch):
    source = archive(tmp_path, monkeypatch)
    def unsupported(*args, **kwargs):
        raise UnsupportedControl('unsupported synthetic control')
    monkeypatch.setattr(diagnostic, 'city_shadow_step', unsupported)
    output = tmp_path / 'audit'
    result = diagnostic.replay_audit(source, output)
    assert result['status'] == 'complete_with_failures'
    assert result['attempted_windows'] == result['attempted_vehicle_rows'] == 6
    assert len(result['errors']) == 2
    rows = [json.loads(line) for line in (output / 'rows.jsonl').read_text().splitlines()]
    assert sum(row['status'] == 'unsupported' for row in rows) == 2
    for row in result['evaluations']:
        assert row['common_position_steps'] == 0
        if row['implementation'] == 'chronological_city':
            assert row['unsupported_steps'] == 1 and row['evaluated_vehicle_steps'] == 0


def test_unexpected_execution_errors_remain_explicit(tmp_path, monkeypatch):
    source = archive(tmp_path, monkeypatch)
    def broken(*args, **kwargs):
        raise RuntimeError('test clock failure')
    monkeypatch.setattr(diagnostic, 'city_shadow_step', broken)
    result = diagnostic.replay_audit(source, tmp_path / 'audit')
    assert len(result['errors']) == 2
    assert all(row['status'] == 'error' and row['error_type'] == 'RuntimeError' for row in result['errors'])


def test_archive_hash_mismatch_fails_before_creating_output(tmp_path, monkeypatch):
    source = archive(tmp_path, monkeypatch)
    with (source / 'combined_queue.net.xml').open('a') as stream:
        stream.write('\n')
    output = tmp_path / 'audit'
    with pytest.raises(ValueError, match='network hash mismatch'):
        diagnostic.replay_audit(source, output)
    assert not output.exists()
