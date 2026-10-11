"""Keep candidate release duration aligned with Ingolstadt."""

import argparse
import csv
import json
import math
from pathlib import Path
import tomllib
from types import SimpleNamespace

import pytest

from tools import prepare_ingo_reference_candidates as preparer
from urban_dreamer.workflow.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ('coulommiers', 'montereau_fault_yonne')


def check_candidate(config):
    simulation = config['simulation']
    assert simulation['demand_duration_seconds'] == 1800.
    assert simulation['world_model_intervals_per_episode'] == 60
    assert simulation['training_completion_buffer_seconds'] == 2000
    assert simulation['evaluation_horizon_seconds'] == 3800
    deadline = 30 * math.ceil((simulation['world_model_intervals_per_episode'] * 30
        + simulation['training_completion_buffer_seconds']) / 30)
    assert deadline == 3810
    assert simulation['incident_start_seconds'] > deadline
    assert config['method']['sp_prior_weight'] == 0.


@pytest.mark.parametrize('city', CANDIDATES)
@pytest.mark.parametrize('suffix', ['', '_offline'])
def test_candidate_profiles_cover_spread_demand_and_completion_buffer(city, suffix):
    path = ROOT / 'configs' / f'{city}{suffix}.toml'
    load_config(path)  # Exercise normal typed configuration validation too.
    check_candidate(tomllib.loads(path.read_text()))


@pytest.mark.parametrize('city', CANDIDATES)
@pytest.mark.parametrize('suffix', ['', '_offline'])
def test_preparer_generated_configs_match_new_duration_without_regenerating_inputs(city, suffix):
    template = (ROOT / 'configs' / f'ingolstadt_custom{suffix}.toml').read_text()
    config = tomllib.loads(preparer.candidate_config(template, city, 'placeholder'))
    check_candidate(config)
    assert config['inputs']['network_name'] == city
    assert config['simulation']['incident_edge'] == 'placeholder'


def test_preparer_cli_defaults_to_1800_without_copying_networks(monkeypatch):
    class ParsedDefaults(Exception):
        pass

    def inspect_parser(parser, *args, **kwargs):
        assert parser.get_default('demand_duration_seconds') == 1800
        assert parser.get_default('manifest') is None
        raise ParsedDefaults

    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', inspect_parser)
    with pytest.raises(ParsedDefaults):
        preparer.main([])


def test_ingo_profile_keeps_original_release_window():
    config = load_config(ROOT / 'configs' / 'ingolstadt_custom.toml')
    assert config.simulation.demand_duration_seconds == 1800.
    assert config.simulation.world_model_intervals_per_episode == 60


def test_historical_import_cannot_overwrite_either_installed_default(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    installed = project / 'networks' / 'montereau_fault_yonne'
    installed.mkdir(parents=True)
    demand = installed / 'agents.csv'
    demand.write_text('accepted occupancy1500 demand')
    monkeypatch.setattr(preparer, 'ROOT', project)
    with pytest.raises(SystemExit) as error:
        preparer.main([])
    assert error.value.code == 2
    assert demand.read_text() == 'accepted occupancy1500 demand'
    assert not (project / 'networks' / 'coulommiers').exists()


def test_preparer_filters_unreachable_demand_without_reallocating(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    (project / 'configs').mkdir(parents=True)
    for suffix in ('', '_offline'):
        template = ROOT / 'configs' / f'ingolstadt_custom{suffix}.toml'
        (project / 'configs' / template.name).write_text(template.read_text())
    for city in CANDIDATES:
        source = tmp_path / 'networks' / city
        source.mkdir(parents=True)
        (source / f'{city}.net.xml').write_text('source network')
        (source / f'od_{city}.txt').write_text("{'origins': ['placeholder'], 'destinations': ['a', 'b']}")
        (source / 'agents.csv').write_text('id,origin,destination,start_time\n0,0,0,5\n1,0,1,10\n')

    def build_patch(original, destination, city, work, netconvert):
        assert work.is_dir()
        destination.write_text('patched network')
        return dict(original_sha256=preparer.sha(original), patched_sha256=preparer.sha(destination))

    def drop_unreachable(directory, city):
        assert (directory / 'edge_registry.json').exists()
        path = directory / 'agents.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        kept = [row for row in rows if row['destination'] == '0']
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(kept)
        return dict(before_expected_vehicles=len(rows), remaining_expected_vehicles=len(kept),
                    removed_expected_vehicles=len(rows)-len(kept), remaining_od_pairs=1,
                    removed_od_pairs=[dict(origin_index=0, destination_index=1)])

    def validate(config):
        path = project / 'networks' / config.city / 'agents.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert {row['destination'] for row in rows} == {'0'}
        assert all(0 <= int(row['start_time']) < 1800 for row in rows)
        return SimpleNamespace(vehicle_count=len(rows), input_hashes={})

    monkeypatch.setattr(preparer, 'ROOT', project)
    monkeypatch.setattr(preparer, 'build_patch', build_patch)
    monkeypatch.setattr(preparer, 'drop_unreachable_od', drop_unreachable)
    monkeypatch.setattr(preparer, 'registry', lambda *args: {})
    monkeypatch.setattr(preparer, 'load_config', lambda path: SimpleNamespace(
        city=path.stem.removesuffix('_offline'), simulation=SimpleNamespace(
            incident_start_seconds=1e6, evaluation_horizon_seconds=3800, incident_edge='placeholder')))
    monkeypatch.setattr(preparer, 'validate_prepared_inputs', validate)
    monkeypatch.setattr(preparer, 'load_traffic_topology', lambda prepared: SimpleNamespace(network=SimpleNamespace(
        edge_to_index={'placeholder': 0}, edge_count=1, passenger_allowed=SimpleNamespace(sum=lambda: 1))))
    manifest_path = project / 'output' / 'manifest.json'
    preparer.main(['--manifest', str(manifest_path)])
    manifest = json.loads(manifest_path.read_text())
    for city, initial in zip(CANDIDATES, (2000, 3000)):
        network = manifest['networks'][city]
        assert network['initial_target_vehicle_count'] == initial
        assert network['expected_vehicle_count'] == initial // 2
        assert network['unique_ods'] == 1
        assert network['od_allocations'] == [dict(origin=0, destination=0, source_count=1, prepared_count=initial // 2)]
