from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import InputValidationError, validate_prepared_inputs

ROOT = Path(__file__).resolve().parents[1]


def _write_prepared_inputs(tmp_path: Path) -> Path:
    network_dir = tmp_path / 'network'
    network_dir.mkdir()
    (network_dir / 'agents.csv').write_text('id,origin,destination,start_time\n1,0,0,0.125\n2,0,0,30\n')
    (network_dir / 'toy_network.net.xml').write_text('<net/>')
    (network_dir / 'od_toy_network.txt').write_text("{'origins': ['a'], 'destinations': ['b']}")
    (network_dir / 'edge_registry.json').write_text(json.dumps({'network': 'toy_network', 'edge_ids': ['a', 'b']}))
    config_text = (ROOT / 'configs/toy_cross_bottleneck.toml').read_text()
    config_text = config_text.replace('toy_cross_bottleneck', 'toy_network').replace('../networks/toy_network', 'network')
    config_path = tmp_path / 'mainline.toml'
    config_path.write_text(config_text)
    return config_path


def test_validate_prepared_inputs_derives_total_count_and_only_input_hashes(tmp_path: Path) -> None:
    prepared = validate_prepared_inputs(load_config(_write_prepared_inputs(tmp_path)))
    assert prepared.vehicle_count == 2
    assert set(prepared.input_hashes) == {'network_xml', 'network_agents', 'od_file', 'edge_registry'}
    assert all(len(digest) == 64 for digest in prepared.input_hashes.values())
    with prepared.network_agents_csv.open() as stream:
        assert list(csv.DictReader(stream))[0]['start_time'] == '0.125'


@pytest.mark.parametrize(('text', 'message'), [
    ('id,origin,destination,start_time\n1,0,0,0\n1,0,0,1\n', 'duplicate'),
    ('id,origin,destination,start_time\n1,0,0,nan\n', 'start_time'),
    ('id,origin,destination,start_time\n1,0,0,-1\n', 'start_time'),
    ('id,origin,destination,start_time\n1,3,0,1\n', 'origin'),
    ('id,origin,destination,start_time,kind\n1,0,0,0,Human\n', 'columns'),
    ('id,origin,destination,start_time\n', 'non-empty'),
])
def test_rejects_invalid_demand(tmp_path: Path, text: str, message: str) -> None:
    path = _write_prepared_inputs(tmp_path)
    (tmp_path / 'network/agents.csv').write_text(text)
    with pytest.raises(InputValidationError, match=message):
        validate_prepared_inputs(load_config(path))


def test_missing_input_is_explicit(tmp_path: Path) -> None:
    path = _write_prepared_inputs(tmp_path)
    (tmp_path / 'network/edge_registry.json').unlink()
    with pytest.raises(InputValidationError, match='incomplete.*edge_registry'):
        validate_prepared_inputs(load_config(path))


def test_demand_change_changes_input_digest(tmp_path: Path) -> None:
    config = load_config(_write_prepared_inputs(tmp_path))
    before = validate_prepared_inputs(config)
    before.network_agents_csv.write_text(before.network_agents_csv.read_text().replace('0.125', '0.25'))
    after = validate_prepared_inputs(config)
    assert before.input_hashes['network_agents'] != after.input_hashes['network_agents']


@pytest.mark.parametrize('city', ['toy_cross_bottleneck', 'gretz_armainvilliers', 'fontenay_en_parisis', 'ingolstadt_custom'])
def test_bundled_city_uses_all_controlled_demand(city: str) -> None:
    prepared = validate_prepared_inputs(load_config(ROOT / 'configs' / f'{city}.toml'))
    assert prepared.vehicle_count > 0
